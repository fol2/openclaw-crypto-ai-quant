use crate::config::HubConfig;
use crate::error::HubError;
use crate::live_hyperliquid::{
    extract_exchange_order_id, normalise_limit_px_for_wire, response_has_embedded_error,
    HyperliquidClient, HyperliquidFill, LiveOrderType, OrderRequest,
};
use crate::live_secrets::load_live_secrets;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use sha3::Digest;
use std::path::Path;
use std::process;
use std::time::{Duration, Instant};

const DEFAULT_SLIPPAGE_PCT: f64 = 0.01;
const HL_TAKER_FEE_RATE: f64 = 0.00035;
const DEFAULT_MAX_NOTIONAL_USD: f64 = 5_000.0;
const MIN_NOTIONAL_USD: f64 = 10.0;
const FILL_POLL_TIMEOUT: Duration = Duration::from_secs(8);
const FILL_POLL_INTERVAL: Duration = Duration::from_millis(800);
const MANUAL_CONFIRM_TTL_MS: i64 = 60_000;
const MANUAL_CONFIRM_RETENTION_MS: i64 = 86_400_000;
const MANUAL_RECONCILE_MISSING_GRACE_MS: i64 = 30_000;
const MANUAL_CLOID_PREFIX: &[u8] = b"man_";
const MANUAL_RUNTIME_STREAM: &str = "manual_trade";
const MANUAL_RUNTIME_MODE: &str = "live";
const MANUAL_CONFIRM_ACTION_OPEN: &str = "OPEN";
const MANUAL_CONFIRM_ACTION_CLOSE: &str = "CLOSE";

#[derive(Debug, Clone)]
pub struct ManualTradeOpenRequest {
    pub symbol: String,
    pub side: String,
    pub notional_usd: f64,
    pub leverage: u32,
    pub order_type: String,
    pub limit_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ManualTradeCloseRequest {
    pub symbol: String,
    pub close_pct: f64,
    pub order_type: String,
    pub limit_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ManualTradeCancelRequest {
    pub symbol: String,
    pub oid: Option<String>,
    pub intent_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedOrderType {
    Market,
    LimitIoc,
    LimitGtc,
}

struct PreparedOpen {
    symbol: String,
    side: String,
    direction: String,
    is_buy: bool,
    mid_price: f64,
    est_size: f64,
    est_notional_usd: f64,
    est_margin_usd: f64,
    est_fee_usd: f64,
    leverage: u32,
    max_leverage: u32,
    sz_decimals: u32,
    order_type: ParsedOrderType,
    limit_price: Option<f64>,
    account_value_usd: f64,
}

struct PreparedClose {
    symbol: String,
    pos_type: String,
    is_buy: bool,
    current_size: f64,
    close_size: f64,
    leverage: f64,
    account_value_usd: f64,
    mid_price: f64,
    order_type: ParsedOrderType,
    limit_price: Option<f64>,
}

struct ManualIntentRecord<'a> {
    intent_id: &'a str,
    created_ts_ms: i64,
    symbol: &'a str,
    action: &'a str,
    side: &'a str,
    requested_size: f64,
    requested_notional: f64,
    leverage: Option<f64>,
    status: &'a str,
    dedupe_key: Option<&'a str>,
    client_order_id: &'a str,
    reason: &'a str,
    confidence: &'a str,
    meta_json: String,
}

struct ManualOrderSubmission<'a> {
    intent_id: &'a str,
    sent_ts_ms: i64,
    symbol: &'a str,
    side: &'a str,
    order_type: &'a str,
    requested_size: f64,
    reduce_only: bool,
    client_order_id: &'a str,
    exchange_order_id: Option<&'a str>,
    status: &'a str,
    last_error: &'a str,
    raw_json: String,
}

struct ManualCancelAudit<'a> {
    intent_id: Option<&'a str>,
    symbol: &'a str,
    side: Option<&'a str>,
    order_type: &'a str,
    requested_size: Option<f64>,
    reduce_only: Option<bool>,
    client_order_id: Option<&'a str>,
    exchange_order_id: Option<&'a str>,
    status: &'a str,
    raw_json: String,
}

struct FillWriteSummary {
    observed_fills: usize,
    inserted_oms_fills: usize,
    parsed_trade_rows: usize,
    parse_failures: usize,
    filled_size: f64,
}

struct ManualOpenOrderSnapshot<'a> {
    last_seen_ts_ms: i64,
    symbol: &'a str,
    side: &'a str,
    price: Option<f64>,
    remaining_size: Option<f64>,
    reduce_only: bool,
    client_order_id: &'a str,
    exchange_order_id: Option<&'a str>,
    intent_id: &'a str,
    raw_json: String,
}

#[derive(Clone)]
struct ManualCancelContext {
    intent_exists: bool,
    intent_id: Option<String>,
    symbol: String,
    side: Option<String>,
    current_status: Option<String>,
    requested_size: Option<f64>,
    reduce_only: Option<bool>,
    client_order_id: Option<String>,
    exchange_order_id: Option<String>,
}

#[derive(Clone)]
struct ExistingManualIntent {
    intent_id: String,
    created_ts_ms: i64,
    symbol: String,
    action: String,
    side: String,
    status: String,
    requested_size: Option<f64>,
    leverage: Option<f64>,
    dedupe_key: Option<String>,
    client_order_id: Option<String>,
    exchange_order_id: Option<String>,
    last_error: Option<String>,
    sent_ts_ms: Option<i64>,
    meta_json: Option<String>,
}

enum ManualExecutionClaim {
    Submit {
        intent_id: String,
        cloid: String,
        resumed_existing: bool,
    },
    Existing(Box<ExistingManualIntent>),
}

pub fn preview_open(cfg: &HubConfig, request: &ManualTradeOpenRequest) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, false)?;
    let client = build_client(cfg)?;
    let prepared = prepare_open(&client, cfg, request)?;
    Ok(json!({
        "ok": true,
        "action": "preview",
        "symbol": prepared.symbol,
        "side": prepared.side,
        "direction": prepared.direction,
        "order_type": request.order_type,
        "mid_price": prepared.mid_price,
        "est_size": prepared.est_size,
        "est_notional_usd": prepared.est_notional_usd,
        "est_margin_usd": prepared.est_margin_usd,
        "est_fee_usd": prepared.est_fee_usd,
        "leverage": prepared.leverage,
        "max_leverage": prepared.max_leverage,
        "account_value_usd": prepared.account_value_usd,
        "sz_decimals": prepared.sz_decimals,
        "limit_price": prepared.limit_price,
    }))
}

pub fn issue_confirm_token(
    cfg: &HubConfig,
    action: &str,
    param_hash: &str,
    symbol: &str,
    preview: &Value,
) -> Result<String, HubError> {
    enforce_manual_trade_ready(cfg, false)?;
    let conn = open_manual_trade_db(&cfg.live_db)?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    purge_stale_manual_confirmations(&conn, now_ms)?;
    if let Some(token) = find_reusable_manual_confirmation(&conn, action, param_hash, now_ms)? {
        write_manual_audit_event(
            &conn,
            Some(symbol),
            "MANUAL_PREVIEW_REUSED",
            "INFO",
            json!({
                "action": action,
                "confirm_token": token.clone(),
                "param_hash": param_hash,
            }),
        )?;
        return Ok(token);
    }

    let token = uuid::Uuid::new_v4().to_string();
    conn.execute(
        "INSERT INTO manual_trade_confirmations (
            confirm_token, created_ts_ms, expires_ts_ms, action, symbol, param_hash,
            status, preview_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'PREVIEWED', ?7)",
        params![
            &token,
            now_ms,
            now_ms + MANUAL_CONFIRM_TTL_MS,
            action,
            symbol.trim().to_ascii_uppercase(),
            param_hash,
            preview.to_string(),
        ],
    )?;
    write_manual_audit_event(
        &conn,
        Some(symbol),
        "MANUAL_PREVIEW_ISSUED",
        "INFO",
        json!({
            "action": action,
            "confirm_token": token.clone(),
            "param_hash": param_hash,
            "preview": preview,
        }),
    )?;
    Ok(token)
}

pub fn execute_open(
    cfg: &HubConfig,
    request: &ManualTradeOpenRequest,
    confirm_token: &str,
    param_hash: &str,
) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, true)?;
    let mut conn = open_manual_trade_db(&cfg.live_db)?;
    let mut early_retry_claim: Option<(String, String)> = None;
    if let Some(existing) = load_existing_manual_intent_by_dedupe_key(&conn, confirm_token)? {
        if should_resume_existing_manual_submission(&existing) {
            validate_resumed_open_request(&existing, request)?;
            if let Some(recovered) =
                recover_existing_new_manual_execution(cfg, &mut conn, &existing)?
            {
                bind_manual_confirmation_to_intent(&conn, confirm_token, &recovered.intent_id)?;
                return Ok(existing_manual_execution_payload("open", &recovered));
            }
            early_retry_claim =
                Some((existing.intent_id.clone(), existing_intent_cloid(&existing)));
        } else {
            bind_manual_confirmation_to_intent(&conn, confirm_token, &existing.intent_id)?;
            return Ok(existing_manual_execution_payload("open", &existing));
        }
    }
    if early_retry_claim.is_none() {
        validate_manual_confirmation(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_OPEN,
            param_hash,
            chrono::Utc::now().timestamp_millis(),
        )?;
    }

    let client = build_client(cfg)?;
    let prepared = prepare_open(&client, cfg, request)?;
    let start_ms = chrono::Utc::now().timestamp_millis() - 5_000;
    let created_ts_ms = chrono::Utc::now().timestamp_millis();
    let intent_id = manual_intent_id_from_confirm_token(confirm_token);
    let cloid = manual_cloid_from_intent_id(&intent_id);

    let (intent_id, cloid, resumed_existing) = if let Some((intent_id, cloid)) = early_retry_claim {
        (intent_id, cloid, true)
    } else {
        let claim = initialise_manual_execution(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_OPEN,
            param_hash,
            ManualIntentRecord {
                intent_id: &intent_id,
                created_ts_ms,
                symbol: &prepared.symbol,
                action: "OPEN",
                side: &prepared.side,
                requested_size: prepared.est_size,
                requested_notional: prepared.est_notional_usd,
                leverage: Some(f64::from(prepared.leverage)),
                status: "NEW",
                dedupe_key: Some(confirm_token),
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: json!({
                    "source": "manual_trade",
                    "request": {
                        "order_type": request.order_type,
                        "side": request.side,
                        "notional_usd": request.notional_usd,
                        "leverage": request.leverage,
                        "limit_price": request.limit_price,
                    },
                    "prepared": {
                        "est_size": prepared.est_size,
                        "est_notional_usd": prepared.est_notional_usd,
                        "mid_price": prepared.mid_price,
                        "direction": prepared.direction,
                    }
                })
                .to_string(),
            },
        )?;
        match claim {
            ManualExecutionClaim::Submit {
                intent_id,
                cloid,
                resumed_existing,
            } => (intent_id, cloid, resumed_existing),
            ManualExecutionClaim::Existing(existing) => {
                return Ok(existing_manual_execution_payload("open", &existing));
            }
        }
    };
    write_manual_runtime_log(
        &conn,
        "INFO",
        &format!(
            "manual_trade {} intent_id={} symbol={} action=OPEN side={} order_type={} requested_size={} requested_notional={}",
            if resumed_existing {
                "intent_resumed"
            } else {
                "intent_created"
            },
            intent_id,
            prepared.symbol,
            prepared.side,
            request.order_type,
            prepared.est_size,
            prepared.est_notional_usd
        ),
    )?;
    write_manual_audit_event(
        &conn,
        Some(&prepared.symbol),
        if resumed_existing {
            "MANUAL_INTENT_RESUMED"
        } else {
            "MANUAL_INTENT_CREATED"
        },
        "INFO",
        json!({
            "intent_id": &intent_id,
            "action": "OPEN",
            "side": &prepared.side,
            "order_type": request.order_type,
            "requested_size": prepared.est_size,
            "requested_notional_usd": prepared.est_notional_usd,
            "client_order_id": &cloid,
        }),
    )?;

    let response = match submit_open_order(&client, &prepared, &cloid) {
        Ok(response) => response,
        Err(error) => {
            let error_text = error.to_string();
            update_manual_intent_status(
                &conn,
                &intent_id,
                failure_status_for_error(&error_text),
                &error_text,
            )?;
            write_manual_runtime_log(
                &conn,
                "ERROR",
                &format!(
                    "manual_trade submit_failed intent_id={} symbol={} action=OPEN error={}",
                    intent_id, prepared.symbol, error_text
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_ORDER_SUBMIT_FAILED",
                "ERROR",
                json!({
                    "intent_id": &intent_id,
                    "action": "OPEN",
                    "error": &error_text,
                }),
            )?;
            return Err(error);
        }
    };
    let exchange_order_id = extract_exchange_order_id(&response);
    let embedded_reject = response_has_embedded_error(&response);
    let response_text = response.to_string();
    record_manual_order_submission(
        &conn,
        ManualOrderSubmission {
            intent_id: &intent_id,
            sent_ts_ms: chrono::Utc::now().timestamp_millis(),
            symbol: &prepared.symbol,
            side: &prepared.side,
            order_type: submission_order_type(prepared.order_type, false),
            requested_size: prepared.est_size,
            reduce_only: false,
            client_order_id: &cloid,
            exchange_order_id: exchange_order_id.as_deref(),
            status: if embedded_reject { "REJECTED" } else { "SENT" },
            last_error: if embedded_reject { &response_text } else { "" },
            raw_json: response_text.clone(),
        },
    )?;
    if embedded_reject {
        write_manual_runtime_log(
            &conn,
            "ERROR",
            &format!(
                "manual_trade submit_rejected intent_id={} symbol={} action=OPEN exchange_order_id={} response={}",
                intent_id,
                prepared.symbol,
                exchange_order_id.as_deref().unwrap_or("unknown"),
                response_text
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&prepared.symbol),
            "MANUAL_ORDER_REJECTED",
            "ERROR",
            json!({
                "intent_id": &intent_id,
                "action": "OPEN",
                "exchange_order_id": &exchange_order_id,
                "response": &response,
            }),
        )?;
        return Err(HubError::BadRequest(response_text));
    }
    write_manual_runtime_log(
        &conn,
        "INFO",
        &format!(
            "manual_trade submitted intent_id={} symbol={} action=OPEN exchange_order_id={}",
            intent_id,
            prepared.symbol,
            exchange_order_id.as_deref().unwrap_or("unknown")
        ),
    )?;
    write_manual_audit_event(
        &conn,
        Some(&prepared.symbol),
        "MANUAL_ORDER_SUBMITTED",
        "INFO",
        json!({
            "intent_id": &intent_id,
            "action": "OPEN",
            "exchange_order_id": &exchange_order_id,
            "client_order_id": &cloid,
            "order_type": submission_order_type(prepared.order_type, false),
        }),
    )?;
    let fills = if prepared.order_type == ParsedOrderType::LimitGtc {
        Vec::new()
    } else {
        match poll_fill(
            &client,
            &prepared.symbol,
            start_ms,
            &cloid,
            exchange_order_id.as_deref(),
        ) {
            Ok(fills) => fills,
            Err(error) => {
                let error_text = format!("submitted_but_fill_poll_failed: {error}");
                update_manual_intent_status(&conn, &intent_id, "UNKNOWN", &error_text)?;
                write_manual_runtime_log(
                    &conn,
                    "ERROR",
                    &format!(
                        "manual_trade fill_poll_failed intent_id={} symbol={} action=OPEN error={}",
                        intent_id, prepared.symbol, error_text
                    ),
                )?;
                write_manual_audit_event(
                    &conn,
                    Some(&prepared.symbol),
                    "MANUAL_FILL_POLL_FAILED",
                    "ERROR",
                    json!({
                        "intent_id": &intent_id,
                        "action": "OPEN",
                        "error": &error_text,
                    }),
                )?;
                return Err(error);
            }
        }
    };

    let fill_summary = record_manual_fill_batch(
        &mut conn,
        &fills,
        &prepared.symbol,
        &intent_id,
        &cloid,
        prepared.account_value_usd,
        f64::from(prepared.leverage),
    )?;
    if fills.is_empty() {
        if prepared.order_type == ParsedOrderType::LimitGtc {
            write_manual_runtime_log(
                &conn,
                "INFO",
                &format!(
                    "manual_trade resting intent_id={} symbol={} action=OPEN exchange_order_id={}",
                    intent_id,
                    prepared.symbol,
                    exchange_order_id.as_deref().unwrap_or("unknown")
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_ORDER_RESTING",
                "INFO",
                json!({
                    "intent_id": &intent_id,
                    "action": "OPEN",
                    "exchange_order_id": &exchange_order_id,
                }),
            )?;
        } else {
            update_manual_intent_status(
                &conn,
                &intent_id,
                "UNKNOWN",
                "submitted_but_no_fills_observed_during_poll",
            )?;
            write_manual_runtime_log(
                &conn,
                "WARN",
                &format!(
                    "manual_trade no_fills_observed intent_id={} symbol={} action=OPEN exchange_order_id={}",
                    intent_id,
                    prepared.symbol,
                    exchange_order_id.as_deref().unwrap_or("unknown")
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_NO_FILLS_OBSERVED",
                "WARN",
                json!({
                    "intent_id": &intent_id,
                    "action": "OPEN",
                    "exchange_order_id": &exchange_order_id,
                }),
            )?;
        }
    } else {
        let status = if fill_summary.parse_failures > 0 {
            "UNKNOWN"
        } else if fill_summary.filled_size + 1e-9 >= prepared.est_size {
            "FILLED"
        } else {
            "PARTIAL"
        };
        let last_error = if fill_summary.parse_failures > 0 {
            "one_or_more_fills_failed_to_parse_into_trades"
        } else {
            ""
        };
        update_manual_intent_status(&conn, &intent_id, status, last_error)?;
        write_manual_runtime_log(
            &conn,
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            &format!(
                "manual_trade fills_recorded intent_id={} symbol={} action=OPEN observed_fills={} parsed_trade_rows={} filled_size={} status={}",
                intent_id,
                prepared.symbol,
                fill_summary.observed_fills,
                fill_summary.parsed_trade_rows,
                fill_summary.filled_size,
                status
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&prepared.symbol),
            "MANUAL_FILLS_RECORDED",
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            json!({
                "intent_id": &intent_id,
                "action": "OPEN",
                "status": status,
                "observed_fills": fill_summary.observed_fills,
                "parsed_trade_rows": fill_summary.parsed_trade_rows,
                "filled_size": fill_summary.filled_size,
            }),
        )?;
    }

    Ok(json!({
        "ok": true,
        "status": if prepared.order_type == ParsedOrderType::LimitGtc { "resting" } else { "submitted" },
        "intent_id": intent_id,
        "symbol": prepared.symbol,
        "side": prepared.side,
        "direction": prepared.direction,
        "order_type": request.order_type,
        "requested_size": prepared.est_size,
        "exchange_order_id": exchange_order_id,
        "fills": fills.into_iter().map(|item| item.raw).collect::<Vec<_>>(),
    }))
}

pub fn preview_close(
    cfg: &HubConfig,
    request: &ManualTradeCloseRequest,
) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, false)?;
    let client = build_client(cfg)?;
    let prepared = prepare_close(&client, request)?;
    Ok(json!({
        "ok": true,
        "status": "ready",
        "symbol": prepared.symbol,
        "pos_type": prepared.pos_type,
        "order_type": request.order_type,
        "close_size": prepared.close_size,
        "current_size": prepared.current_size,
        "limit_price": prepared.limit_price,
    }))
}

pub fn execute_close(
    cfg: &HubConfig,
    request: &ManualTradeCloseRequest,
    confirm_token: &str,
    param_hash: &str,
) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, true)?;
    let mut conn = open_manual_trade_db(&cfg.live_db)?;
    let mut early_retry_claim: Option<(String, String)> = None;
    if let Some(existing) = load_existing_manual_intent_by_dedupe_key(&conn, confirm_token)? {
        if should_resume_existing_manual_submission(&existing) {
            validate_resumed_close_request(&existing, request)?;
            if let Some(recovered) =
                recover_existing_new_manual_execution(cfg, &mut conn, &existing)?
            {
                bind_manual_confirmation_to_intent(&conn, confirm_token, &recovered.intent_id)?;
                return Ok(existing_manual_execution_payload("close", &recovered));
            }
            early_retry_claim =
                Some((existing.intent_id.clone(), existing_intent_cloid(&existing)));
        } else {
            bind_manual_confirmation_to_intent(&conn, confirm_token, &existing.intent_id)?;
            return Ok(existing_manual_execution_payload("close", &existing));
        }
    }
    if early_retry_claim.is_none() {
        validate_manual_confirmation(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_CLOSE,
            param_hash,
            chrono::Utc::now().timestamp_millis(),
        )?;
    }

    let client = build_client(cfg)?;
    let prepared = prepare_close(&client, request)?;
    let start_ms = chrono::Utc::now().timestamp_millis() - 5_000;
    let created_ts_ms = chrono::Utc::now().timestamp_millis();
    let side = if prepared.is_buy { "BUY" } else { "SELL" };
    let close_action = if prepared.close_size + 1e-9 >= prepared.current_size {
        "CLOSE"
    } else {
        "REDUCE"
    };
    let intent_id = manual_intent_id_from_confirm_token(confirm_token);
    let cloid = manual_cloid_from_intent_id(&intent_id);

    let (intent_id, cloid, resumed_existing) = if let Some((intent_id, cloid)) = early_retry_claim {
        (intent_id, cloid, true)
    } else {
        let claim = initialise_manual_execution(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_CLOSE,
            param_hash,
            ManualIntentRecord {
                intent_id: &intent_id,
                created_ts_ms,
                symbol: &prepared.symbol,
                action: close_action,
                side,
                requested_size: prepared.close_size,
                requested_notional: prepared.close_size * prepared.mid_price,
                leverage: Some(prepared.leverage),
                status: "NEW",
                dedupe_key: Some(confirm_token),
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: json!({
                    "source": "manual_trade",
                    "request": {
                        "order_type": request.order_type,
                        "close_pct": request.close_pct,
                        "limit_price": request.limit_price,
                    },
                    "prepared": {
                        "close_size": prepared.close_size,
                        "current_size": prepared.current_size,
                        "mid_price": prepared.mid_price,
                        "pos_type": prepared.pos_type,
                    }
                })
                .to_string(),
            },
        )?;
        match claim {
            ManualExecutionClaim::Submit {
                intent_id,
                cloid,
                resumed_existing,
            } => (intent_id, cloid, resumed_existing),
            ManualExecutionClaim::Existing(existing) => {
                return Ok(existing_manual_execution_payload("close", &existing));
            }
        }
    };
    write_manual_runtime_log(
        &conn,
        "INFO",
        &format!(
            "manual_trade {} intent_id={} symbol={} action={} side={} order_type={} requested_size={} current_size={}",
            if resumed_existing {
                "intent_resumed"
            } else {
                "intent_created"
            },
            intent_id,
            prepared.symbol,
            close_action,
            side,
            request.order_type,
            prepared.close_size,
            prepared.current_size
        ),
    )?;
    write_manual_audit_event(
        &conn,
        Some(&prepared.symbol),
        if resumed_existing {
            "MANUAL_INTENT_RESUMED"
        } else {
            "MANUAL_INTENT_CREATED"
        },
        "INFO",
        json!({
            "intent_id": &intent_id,
            "action": close_action,
            "side": side,
            "order_type": request.order_type,
            "close_size": prepared.close_size,
            "current_size": prepared.current_size,
            "client_order_id": &cloid,
        }),
    )?;

    let response = match submit_close_order(&client, &prepared, &cloid) {
        Ok(response) => response,
        Err(error) => {
            let error_text = error.to_string();
            update_manual_intent_status(
                &conn,
                &intent_id,
                failure_status_for_error(&error_text),
                &error_text,
            )?;
            write_manual_runtime_log(
                &conn,
                "ERROR",
                &format!(
                    "manual_trade submit_failed intent_id={} symbol={} action={} error={}",
                    intent_id, prepared.symbol, close_action, error_text
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_ORDER_SUBMIT_FAILED",
                "ERROR",
                json!({
                    "intent_id": &intent_id,
                    "action": close_action,
                    "error": &error_text,
                }),
            )?;
            return Err(error);
        }
    };
    let exchange_order_id = extract_exchange_order_id(&response);
    let embedded_reject = response_has_embedded_error(&response);
    let response_text = response.to_string();
    record_manual_order_submission(
        &conn,
        ManualOrderSubmission {
            intent_id: &intent_id,
            sent_ts_ms: chrono::Utc::now().timestamp_millis(),
            symbol: &prepared.symbol,
            side,
            order_type: submission_order_type(prepared.order_type, true),
            requested_size: prepared.close_size,
            reduce_only: true,
            client_order_id: &cloid,
            exchange_order_id: exchange_order_id.as_deref(),
            status: if embedded_reject { "REJECTED" } else { "SENT" },
            last_error: if embedded_reject { &response_text } else { "" },
            raw_json: response_text.clone(),
        },
    )?;
    if embedded_reject {
        write_manual_runtime_log(
            &conn,
            "ERROR",
            &format!(
                "manual_trade submit_rejected intent_id={} symbol={} action={} exchange_order_id={} response={}",
                intent_id,
                prepared.symbol,
                close_action,
                exchange_order_id.as_deref().unwrap_or("unknown"),
                response_text
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&prepared.symbol),
            "MANUAL_ORDER_REJECTED",
            "ERROR",
            json!({
                "intent_id": &intent_id,
                "action": close_action,
                "exchange_order_id": &exchange_order_id,
                "response": &response,
            }),
        )?;
        return Err(HubError::BadRequest(response_text));
    }
    write_manual_runtime_log(
        &conn,
        "INFO",
        &format!(
            "manual_trade submitted intent_id={} symbol={} action={} exchange_order_id={}",
            intent_id,
            prepared.symbol,
            close_action,
            exchange_order_id.as_deref().unwrap_or("unknown")
        ),
    )?;
    write_manual_audit_event(
        &conn,
        Some(&prepared.symbol),
        "MANUAL_ORDER_SUBMITTED",
        "INFO",
        json!({
            "intent_id": &intent_id,
            "action": close_action,
            "exchange_order_id": &exchange_order_id,
            "client_order_id": &cloid,
            "order_type": submission_order_type(prepared.order_type, true),
        }),
    )?;
    let fills = if prepared.order_type == ParsedOrderType::LimitGtc {
        Vec::new()
    } else {
        match poll_fill(
            &client,
            &prepared.symbol,
            start_ms,
            &cloid,
            exchange_order_id.as_deref(),
        ) {
            Ok(fills) => fills,
            Err(error) => {
                let error_text = format!("submitted_but_fill_poll_failed: {error}");
                update_manual_intent_status(&conn, &intent_id, "UNKNOWN", &error_text)?;
                write_manual_runtime_log(
                    &conn,
                    "ERROR",
                    &format!(
                        "manual_trade fill_poll_failed intent_id={} symbol={} action={} error={}",
                        intent_id, prepared.symbol, close_action, error_text
                    ),
                )?;
                write_manual_audit_event(
                    &conn,
                    Some(&prepared.symbol),
                    "MANUAL_FILL_POLL_FAILED",
                    "ERROR",
                    json!({
                        "intent_id": &intent_id,
                        "action": close_action,
                        "error": &error_text,
                    }),
                )?;
                return Err(error);
            }
        }
    };

    let fill_summary = record_manual_fill_batch(
        &mut conn,
        &fills,
        &prepared.symbol,
        &intent_id,
        &cloid,
        prepared.account_value_usd,
        prepared.leverage.max(1.0),
    )?;
    if fills.is_empty() {
        if prepared.order_type == ParsedOrderType::LimitGtc {
            write_manual_runtime_log(
                &conn,
                "INFO",
                &format!(
                    "manual_trade resting intent_id={} symbol={} action={} exchange_order_id={}",
                    intent_id,
                    prepared.symbol,
                    close_action,
                    exchange_order_id.as_deref().unwrap_or("unknown")
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_ORDER_RESTING",
                "INFO",
                json!({
                    "intent_id": &intent_id,
                    "action": close_action,
                    "exchange_order_id": &exchange_order_id,
                }),
            )?;
        } else {
            update_manual_intent_status(
                &conn,
                &intent_id,
                "UNKNOWN",
                "submitted_but_no_fills_observed_during_poll",
            )?;
            write_manual_runtime_log(
                &conn,
                "WARN",
                &format!(
                    "manual_trade no_fills_observed intent_id={} symbol={} action={} exchange_order_id={}",
                    intent_id,
                    prepared.symbol,
                    close_action,
                    exchange_order_id.as_deref().unwrap_or("unknown")
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&prepared.symbol),
                "MANUAL_NO_FILLS_OBSERVED",
                "WARN",
                json!({
                    "intent_id": &intent_id,
                    "action": close_action,
                    "exchange_order_id": &exchange_order_id,
                }),
            )?;
        }
    } else {
        let status = if fill_summary.parse_failures > 0 {
            "UNKNOWN"
        } else if fill_summary.filled_size + 1e-9 >= prepared.close_size {
            "FILLED"
        } else {
            "PARTIAL"
        };
        let last_error = if fill_summary.parse_failures > 0 {
            "one_or_more_fills_failed_to_parse_into_trades"
        } else {
            ""
        };
        update_manual_intent_status(&conn, &intent_id, status, last_error)?;
        write_manual_runtime_log(
            &conn,
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            &format!(
                "manual_trade fills_recorded intent_id={} symbol={} action={} observed_fills={} parsed_trade_rows={} filled_size={} status={}",
                intent_id,
                prepared.symbol,
                close_action,
                fill_summary.observed_fills,
                fill_summary.parsed_trade_rows,
                fill_summary.filled_size,
                status
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&prepared.symbol),
            "MANUAL_FILLS_RECORDED",
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            json!({
                "intent_id": &intent_id,
                "action": close_action,
                "status": status,
                "observed_fills": fill_summary.observed_fills,
                "parsed_trade_rows": fill_summary.parsed_trade_rows,
                "filled_size": fill_summary.filled_size,
            }),
        )?;
    }

    Ok(json!({
        "ok": true,
        "status": if prepared.order_type == ParsedOrderType::LimitGtc { "resting" } else { "submitted" },
        "symbol": prepared.symbol,
        "pos_type": prepared.pos_type,
        "close_size": prepared.close_size,
        "order_type": request.order_type,
        "intent_id": intent_id,
        "exchange_order_id": exchange_order_id,
        "fills": fills.into_iter().map(|item| item.raw).collect::<Vec<_>>(),
    }))
}

pub fn open_orders(cfg: &HubConfig, symbol: &str) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, false)?;
    let client = build_client(cfg)?;
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let raw_orders = client
        .open_orders()
        .map_err(|error| HubError::Internal(error.to_string()))?;
    let orders = raw_orders
        .into_iter()
        .filter_map(|order| {
            let coin = order
                .get("coin")
                .and_then(|value| value.as_str())
                .map(|value| value.trim().to_ascii_uppercase())?;
            if coin != symbol_upper {
                return None;
            }
            let oid = order.get("oid")?.as_i64()?;
            let side = order
                .get("side")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let price = order
                .get("limitPx")
                .or_else(|| order.get("price"))
                .and_then(parse_json_number)?;
            let size = order
                .get("sz")
                .or_else(|| order.get("origSz"))
                .and_then(parse_json_number)?;
            Some(json!({
                "oid": oid,
                "side": if side.eq_ignore_ascii_case("B") { "BUY" } else { "SELL" },
                "price": price,
                "size": size,
            }))
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "ok": true,
        "symbol": symbol_upper,
        "orders": orders,
    }))
}

pub fn cancel_order(
    cfg: &HubConfig,
    request: &ManualTradeCancelRequest,
) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, true)?;
    let symbol = request.symbol.trim().to_ascii_uppercase();
    let conn = open_manual_trade_db(&cfg.live_db)?;

    if let Some(oid) = request
        .oid
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        let order_id_text = oid.trim().to_string();
        let parsed_oid = oid
            .trim()
            .parse::<u64>()
            .map_err(|_| HubError::BadRequest("oid must be a positive integer".into()))?;
        let cancel_context =
            load_manual_cancel_context_by_exchange_order_id(&conn, &symbol, &order_id_text)?;
        write_manual_runtime_log(
            &conn,
            "INFO",
            &format!(
                "manual_trade cancel_requested symbol={} exchange_order_id={} intent_id={}",
                symbol,
                order_id_text,
                cancel_context
                    .as_ref()
                    .and_then(|item| item.intent_id.as_deref())
                    .unwrap_or("unknown")
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&symbol),
            "MANUAL_CANCEL_REQUESTED",
            "INFO",
            json!({
                "exchange_order_id": &order_id_text,
                "intent_id": cancel_context
                    .as_ref()
                    .and_then(|item| item.intent_id.as_deref()),
            }),
        )?;
        let client = build_client(cfg)?;
        let response = match client.cancel_order(&symbol, parsed_oid) {
            Ok(response) => response,
            Err(error) => {
                let error_text = error.to_string();
                record_manual_cancel_audit(
                    &conn,
                    cancel_context.as_ref(),
                    ManualCancelAudit {
                        intent_id: cancel_context
                            .as_ref()
                            .and_then(|item| item.intent_id.as_deref()),
                        symbol: &symbol,
                        side: cancel_context
                            .as_ref()
                            .and_then(|item| item.side.as_deref()),
                        order_type: "cancel_by_oid",
                        requested_size: cancel_context
                            .as_ref()
                            .and_then(|item| item.requested_size),
                        reduce_only: cancel_context.as_ref().and_then(|item| item.reduce_only),
                        client_order_id: cancel_context
                            .as_ref()
                            .and_then(|item| item.client_order_id.as_deref()),
                        exchange_order_id: Some(order_id_text.as_str()),
                        status: "ERROR",
                        raw_json: error_text.clone(),
                    },
                )?;
                write_manual_runtime_log(
                    &conn,
                    "ERROR",
                    &format!(
                        "manual_trade cancel_failed symbol={} exchange_order_id={} error={}",
                        symbol, order_id_text, error_text
                    ),
                )?;
                write_manual_audit_event(
                    &conn,
                    Some(&symbol),
                    "MANUAL_CANCEL_FAILED",
                    "ERROR",
                    json!({
                        "exchange_order_id": &order_id_text,
                        "intent_id": cancel_context
                            .as_ref()
                            .and_then(|item| item.intent_id.as_deref()),
                        "error": &error_text,
                    }),
                )?;
                return Err(HubError::BadRequest(error_text));
            }
        };
        let response_text = response.to_string();
        if response_has_embedded_error(&response) {
            record_manual_cancel_audit(
                &conn,
                cancel_context.as_ref(),
                ManualCancelAudit {
                    intent_id: cancel_context
                        .as_ref()
                        .and_then(|item| item.intent_id.as_deref()),
                    symbol: &symbol,
                    side: cancel_context
                        .as_ref()
                        .and_then(|item| item.side.as_deref()),
                    order_type: "cancel_by_oid",
                    requested_size: cancel_context.as_ref().and_then(|item| item.requested_size),
                    reduce_only: cancel_context.as_ref().and_then(|item| item.reduce_only),
                    client_order_id: cancel_context
                        .as_ref()
                        .and_then(|item| item.client_order_id.as_deref()),
                    exchange_order_id: Some(order_id_text.as_str()),
                    status: "REJECTED",
                    raw_json: response_text.clone(),
                },
            )?;
            write_manual_runtime_log(
                &conn,
                "ERROR",
                &format!(
                    "manual_trade cancel_rejected symbol={} exchange_order_id={} response={}",
                    symbol, order_id_text, response_text
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&symbol),
                "MANUAL_CANCEL_REJECTED",
                "ERROR",
                json!({
                    "exchange_order_id": &order_id_text,
                    "intent_id": cancel_context
                        .as_ref()
                        .and_then(|item| item.intent_id.as_deref()),
                    "response": &response,
                }),
            )?;
            return Err(HubError::BadRequest(response_text));
        }
        record_manual_cancel_audit(
            &conn,
            cancel_context.as_ref(),
            ManualCancelAudit {
                intent_id: cancel_context
                    .as_ref()
                    .and_then(|item| item.intent_id.as_deref()),
                symbol: &symbol,
                side: cancel_context
                    .as_ref()
                    .and_then(|item| item.side.as_deref()),
                order_type: "cancel_by_oid",
                requested_size: cancel_context.as_ref().and_then(|item| item.requested_size),
                reduce_only: cancel_context.as_ref().and_then(|item| item.reduce_only),
                client_order_id: cancel_context
                    .as_ref()
                    .and_then(|item| item.client_order_id.as_deref()),
                exchange_order_id: Some(order_id_text.as_str()),
                status: "CANCELLED",
                raw_json: response_text.clone(),
            },
        )?;
        write_manual_runtime_log(
            &conn,
            "INFO",
            &format!(
                "manual_trade cancelled symbol={} exchange_order_id={} intent_id={}",
                symbol,
                order_id_text,
                cancel_context
                    .as_ref()
                    .and_then(|item| item.intent_id.as_deref())
                    .unwrap_or("unknown")
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&symbol),
            "MANUAL_CANCELLED",
            "INFO",
            json!({
                "exchange_order_id": &order_id_text,
                "intent_id": cancel_context
                    .as_ref()
                    .and_then(|item| item.intent_id.as_deref()),
                "response": &response,
            }),
        )?;
        return Ok(json!({
            "ok": true,
            "symbol": symbol,
            "oid": parsed_oid,
            "intent_id": cancel_context.and_then(|item| item.intent_id),
            "response": response,
        }));
    }

    if let Some(intent_id) = request
        .intent_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        let cloid = manual_cloid_from_intent_id(intent_id);
        let cancel_context = load_manual_cancel_context_by_intent_id(&conn, intent_id)?
            .unwrap_or_else(|| ManualCancelContext {
                intent_exists: false,
                intent_id: Some(intent_id.to_string()),
                symbol: symbol.clone(),
                side: None,
                current_status: None,
                requested_size: None,
                reduce_only: None,
                client_order_id: Some(cloid.clone()),
                exchange_order_id: None,
            });
        let effective_symbol = cancel_context.symbol.clone();
        write_manual_runtime_log(
            &conn,
            "INFO",
            &format!(
                "manual_trade cancel_requested symbol={} intent_id={} client_order_id={}",
                effective_symbol, intent_id, cloid
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&effective_symbol),
            "MANUAL_CANCEL_REQUESTED",
            "INFO",
            json!({
                "intent_id": intent_id,
                "client_order_id": &cloid,
            }),
        )?;
        let client = build_client(cfg)?;
        let response = match client.cancel_order_by_cloid(&effective_symbol, &cloid) {
            Ok(response) => response,
            Err(error) => {
                let error_text = error.to_string();
                record_manual_cancel_audit(
                    &conn,
                    Some(&cancel_context),
                    ManualCancelAudit {
                        intent_id: cancel_context.intent_id.as_deref(),
                        symbol: &effective_symbol,
                        side: cancel_context.side.as_deref(),
                        order_type: "cancel_by_cloid",
                        requested_size: cancel_context.requested_size,
                        reduce_only: cancel_context.reduce_only,
                        client_order_id: Some(cloid.as_str()),
                        exchange_order_id: cancel_context.exchange_order_id.as_deref(),
                        status: "ERROR",
                        raw_json: error_text.clone(),
                    },
                )?;
                write_manual_runtime_log(
                    &conn,
                    "ERROR",
                    &format!(
                        "manual_trade cancel_failed symbol={} intent_id={} error={}",
                        effective_symbol, intent_id, error_text
                    ),
                )?;
                write_manual_audit_event(
                    &conn,
                    Some(&effective_symbol),
                    "MANUAL_CANCEL_FAILED",
                    "ERROR",
                    json!({
                        "intent_id": intent_id,
                        "client_order_id": &cloid,
                        "error": &error_text,
                    }),
                )?;
                return Err(HubError::BadRequest(error_text));
            }
        };
        let response_text = response.to_string();
        if response_has_embedded_error(&response) {
            record_manual_cancel_audit(
                &conn,
                Some(&cancel_context),
                ManualCancelAudit {
                    intent_id: cancel_context.intent_id.as_deref(),
                    symbol: &effective_symbol,
                    side: cancel_context.side.as_deref(),
                    order_type: "cancel_by_cloid",
                    requested_size: cancel_context.requested_size,
                    reduce_only: cancel_context.reduce_only,
                    client_order_id: Some(cloid.as_str()),
                    exchange_order_id: cancel_context.exchange_order_id.as_deref(),
                    status: "REJECTED",
                    raw_json: response_text.clone(),
                },
            )?;
            write_manual_runtime_log(
                &conn,
                "ERROR",
                &format!(
                    "manual_trade cancel_rejected symbol={} intent_id={} response={}",
                    effective_symbol, intent_id, response_text
                ),
            )?;
            write_manual_audit_event(
                &conn,
                Some(&effective_symbol),
                "MANUAL_CANCEL_REJECTED",
                "ERROR",
                json!({
                    "intent_id": intent_id,
                    "client_order_id": &cloid,
                    "response": &response,
                }),
            )?;
            return Err(HubError::BadRequest(response_text));
        }
        record_manual_cancel_audit(
            &conn,
            Some(&cancel_context),
            ManualCancelAudit {
                intent_id: cancel_context.intent_id.as_deref(),
                symbol: &effective_symbol,
                side: cancel_context.side.as_deref(),
                order_type: "cancel_by_cloid",
                requested_size: cancel_context.requested_size,
                reduce_only: cancel_context.reduce_only,
                client_order_id: Some(cloid.as_str()),
                exchange_order_id: cancel_context.exchange_order_id.as_deref(),
                status: "CANCELLED",
                raw_json: response_text.clone(),
            },
        )?;
        write_manual_runtime_log(
            &conn,
            "INFO",
            &format!(
                "manual_trade cancelled symbol={} intent_id={} client_order_id={}",
                effective_symbol, intent_id, cloid
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&effective_symbol),
            "MANUAL_CANCELLED",
            "INFO",
            json!({
                "intent_id": intent_id,
                "client_order_id": &cloid,
                "response": &response,
            }),
        )?;
        return Ok(json!({
            "ok": true,
            "symbol": effective_symbol,
            "intent_id": intent_id,
            "cloid": cloid,
            "response": response,
        }));
    }

    Err(HubError::BadRequest(
        "oid or intent_id required for cancel".into(),
    ))
}

pub fn reconcile_manual_intents(cfg: &HubConfig, limit: usize) -> Result<Value, HubError> {
    if limit == 0 {
        return Ok(json!({
            "ok": true,
            "scanned": 0,
            "recovered_fills": 0,
            "recovered_open_orders": 0,
            "tracked_open_orders": 0,
            "cleared_open_orders": 0,
            "status_updates": 0,
        }));
    }
    let mut conn = open_manual_trade_db(&cfg.live_db)?;
    let intents = load_reconcilable_manual_intents(&conn, limit)?;
    if intents.is_empty() {
        return Ok(json!({
            "ok": true,
            "scanned": 0,
            "recovered_fills": 0,
            "recovered_open_orders": 0,
            "tracked_open_orders": 0,
            "cleared_open_orders": 0,
            "status_updates": 0,
        }));
    }

    let client = build_client(cfg)?;
    let open_orders = client
        .open_orders()
        .map_err(|error| HubError::Internal(format!("failed to inspect open orders: {error}")))?;
    let mut recovered_fills = 0usize;
    let mut recovered_open_orders = 0usize;
    let mut tracked_open_orders = 0usize;
    let mut cleared_open_orders = 0usize;
    let mut status_updates = 0usize;
    let mut scanned = 0usize;
    let mut account_value_usd_cache: Option<f64> = None;

    for intent in intents {
        scanned += 1;
        let cloid = existing_intent_cloid(&intent);
        let now_ms = chrono::Utc::now().timestamp_millis();
        let fills = collect_matching_fills_once(
            &client,
            &intent.symbol,
            intent.created_ts_ms.saturating_sub(5_000),
            now_ms,
            &cloid,
            intent.exchange_order_id.as_deref(),
        )?;
        let fill_summary = if fills.is_empty() {
            FillWriteSummary {
                observed_fills: 0,
                inserted_oms_fills: 0,
                parsed_trade_rows: 0,
                parse_failures: 0,
                filled_size: 0.0,
            }
        } else {
            let account_value_usd = if let Some(cached) = account_value_usd_cache {
                cached
            } else {
                let fetched = client
                    .account_snapshot()
                    .map_err(|error| HubError::Internal(error.to_string()))?
                    .account_value_usd;
                account_value_usd_cache = Some(fetched);
                fetched
            };
            record_manual_fill_batch(
                &mut conn,
                &fills,
                &intent.symbol,
                &intent.intent_id,
                &cloid,
                account_value_usd,
                intent.leverage.unwrap_or(1.0).max(1.0),
            )?
        };

        let matching_open_order = find_matching_open_order(&open_orders, &intent, &cloid);
        let recovered_exchange_order_id = matching_open_order
            .and_then(|order| order.get("oid").and_then(parse_json_integer))
            .map(|value| value.to_string());
        let open_order_present = matching_open_order.is_some();
        let open_order_snapshot_changed = if let Some(order) = matching_open_order {
            upsert_manual_open_order_snapshot(
                &conn,
                ManualOpenOrderSnapshot {
                    last_seen_ts_ms: now_ms,
                    symbol: &intent.symbol,
                    side: open_order_side(order),
                    price: order
                        .get("limitPx")
                        .or_else(|| order.get("price"))
                        .and_then(parse_json_number),
                    remaining_size: order
                        .get("sz")
                        .or_else(|| order.get("origSz"))
                        .or_else(|| order.get("remainingSz"))
                        .and_then(parse_json_number),
                    reduce_only: open_order_reduce_only(order),
                    client_order_id: &cloid,
                    exchange_order_id: recovered_exchange_order_id.as_deref(),
                    intent_id: &intent.intent_id,
                    raw_json: order.to_string(),
                },
            )?
        } else {
            clear_manual_open_order_snapshot(&conn, &intent.intent_id, &cloid)?
        };
        let mut recovered_open_order_for_intent = false;
        let mut tracked_open_order_for_intent = false;
        let mut cleared_open_order_for_intent = false;
        if open_order_present {
            if intent.sent_ts_ms.is_none() || intent.status.eq_ignore_ascii_case("UNKNOWN") {
                let order = matching_open_order.expect("open order presence already checked");
                record_manual_order_submission(
                    &conn,
                    ManualOrderSubmission {
                        intent_id: &intent.intent_id,
                        sent_ts_ms: now_ms,
                        symbol: &intent.symbol,
                        side: open_order_side(order),
                        order_type: "recovered_active_open_order",
                        requested_size: intent.requested_size.unwrap_or(0.0),
                        reduce_only: matches!(intent.action.as_str(), "CLOSE" | "REDUCE"),
                        client_order_id: &cloid,
                        exchange_order_id: recovered_exchange_order_id.as_deref(),
                        status: "SENT",
                        last_error: "",
                        raw_json: order.to_string(),
                    },
                )?;
                recovered_open_orders += 1;
                recovered_open_order_for_intent = true;
            } else if open_order_snapshot_changed {
                tracked_open_orders += 1;
                tracked_open_order_for_intent = true;
            }
        } else if open_order_snapshot_changed {
            cleared_open_orders += 1;
            cleared_open_order_for_intent = true;
        }

        let (next_status, next_last_error) =
            desired_manual_reconcile_state(&intent, &fill_summary, open_order_present, now_ms);
        let current_last_error = intent.last_error.as_deref().unwrap_or_default();
        let effective_last_error =
            if intent.status.eq_ignore_ascii_case(next_status) && next_last_error.is_empty() {
                current_last_error
            } else {
                next_last_error
            };
        let status_changed = !intent.status.eq_ignore_ascii_case(next_status)
            || current_last_error != effective_last_error;
        if status_changed {
            update_manual_intent_status(
                &conn,
                &intent.intent_id,
                next_status,
                effective_last_error,
            )?;
            status_updates += 1;
        }
        if fill_summary.inserted_oms_fills > 0 {
            recovered_fills += 1;
        }

        if fill_summary.inserted_oms_fills == 0
            && !recovered_open_order_for_intent
            && !tracked_open_order_for_intent
            && !cleared_open_order_for_intent
            && !status_changed
        {
            continue;
        }

        let level = if matches!(next_status, "UNKNOWN" | "UNKNOWN_RECONCILED") {
            "WARN"
        } else {
            "INFO"
        };
        let result = if fill_summary.inserted_oms_fills > 0 {
            if fill_summary.parse_failures > 0 {
                "fills_parse_failed"
            } else {
                "fills_recovered"
            }
        } else if recovered_open_order_for_intent {
            "open_order_recovered"
        } else if tracked_open_order_for_intent {
            "open_order_tracked"
        } else if cleared_open_order_for_intent {
            "open_order_cleared"
        } else if next_status == "UNKNOWN" {
            "intent_missing"
        } else {
            "status_updated"
        };
        write_manual_runtime_log(
            &conn,
            level,
            &format!(
                "manual_trade reconciled intent_id={} symbol={} previous_status={} status={} observed_fills={} new_oms_fills={} parsed_trade_rows={} filled_size={} open_order_present={}",
                intent.intent_id,
                intent.symbol,
                intent.status,
                next_status,
                fill_summary.observed_fills,
                fill_summary.inserted_oms_fills,
                fill_summary.parsed_trade_rows,
                fill_summary.filled_size,
                open_order_present
            ),
        )?;
        write_manual_audit_event(
            &conn,
            Some(&intent.symbol),
            "MANUAL_RECONCILED",
            level,
            json!({
                "intent_id": &intent.intent_id,
                "action": &intent.action,
                "previous_status": &intent.status,
                "status": next_status,
                "observed_fills": fill_summary.observed_fills,
                "new_oms_fills": fill_summary.inserted_oms_fills,
                "parsed_trade_rows": fill_summary.parsed_trade_rows,
                "filled_size": fill_summary.filled_size,
                "open_order_present": open_order_present,
                "exchange_order_id": &recovered_exchange_order_id,
                "source": "active_recovery",
            }),
        )?;
        write_manual_reconcile_event(
            &conn,
            "manual_active_reconcile",
            Some(&intent.symbol),
            result,
            json!({
                "intent_id": &intent.intent_id,
                "previous_status": &intent.status,
                "status": next_status,
                "observed_fills": fill_summary.observed_fills,
                "new_oms_fills": fill_summary.inserted_oms_fills,
                "filled_size": fill_summary.filled_size,
                "open_order_present": open_order_present,
                "exchange_order_id": &recovered_exchange_order_id,
            }),
        )?;
    }

    Ok(json!({
        "ok": true,
        "scanned": scanned,
        "recovered_fills": recovered_fills,
        "recovered_open_orders": recovered_open_orders,
        "tracked_open_orders": tracked_open_orders,
        "cleared_open_orders": cleared_open_orders,
        "status_updates": status_updates,
    }))
}

#[allow(dead_code)]
pub fn recover_unknown_manual_intents(cfg: &HubConfig, limit: usize) -> Result<Value, HubError> {
    reconcile_manual_intents(cfg, limit)
}

fn build_client(cfg: &HubConfig) -> Result<HyperliquidClient, HubError> {
    let secrets_path = cfg
        .secrets_path
        .as_deref()
        .ok_or_else(|| HubError::Forbidden("AI_QUANT_SECRETS_PATH is not configured".into()))?;
    let secrets = load_live_secrets(secrets_path)
        .map_err(|error| HubError::Internal(format!("failed to load live secrets: {error}")))?;
    HyperliquidClient::new(&secrets, Some(4.0))
        .map_err(|error| HubError::Internal(format!("failed to build Hyperliquid client: {error}")))
}

fn enforce_manual_trade_ready(cfg: &HubConfig, is_write: bool) -> Result<(), HubError> {
    if !cfg.manual_trade_enabled {
        return Err(HubError::Forbidden("manual trading is disabled".into()));
    }
    if is_write {
        let hard_kill = std::env::var("AI_QUANT_HARD_KILL_SWITCH")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        if hard_kill {
            return Err(HubError::Forbidden(
                "live trading halted: AI_QUANT_HARD_KILL_SWITCH is active".into(),
            ));
        }
    }
    Ok(())
}

pub fn record_manual_guardrail_block(
    cfg: &HubConfig,
    symbol: &str,
    action: &str,
    reason: &str,
) -> Result<(), HubError> {
    let conn = open_manual_trade_db(&cfg.live_db)?;
    let symbol = symbol.trim().to_ascii_uppercase();
    let action = action.trim().to_ascii_uppercase();
    let reason = reason.trim().to_string();
    write_manual_runtime_log(
        &conn,
        "WARN",
        &format!(
            "manual_trade blocked symbol={} action={} reason={}",
            symbol, action, reason
        ),
    )?;
    write_manual_audit_event(
        &conn,
        Some(&symbol),
        "MANUAL_GUARDRAIL_BLOCKED",
        "WARN",
        json!({
            "action": action,
            "reason": reason,
        }),
    )?;
    Ok(())
}

fn open_manual_trade_db(path: &Path) -> Result<Connection, HubError> {
    let mut conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    ensure_manual_trade_tables(&mut conn)?;
    Ok(conn)
}

fn ensure_manual_trade_tables(conn: &mut Connection) -> Result<(), HubError> {
    let tx = conn.transaction()?;
    tx.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS oms_intents (
            intent_id TEXT PRIMARY KEY,
            created_ts_ms INTEGER NOT NULL,
            sent_ts_ms INTEGER,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            side TEXT NOT NULL,
            requested_size REAL,
            requested_notional REAL,
            entry_atr REAL,
            leverage REAL,
            decision_ts_ms INTEGER,
            strategy_version TEXT,
            strategy_sha1 TEXT,
            reason TEXT,
            confidence TEXT,
            status TEXT,
            dedupe_key TEXT,
            client_order_id TEXT,
            exchange_order_id TEXT,
            last_error TEXT,
            meta_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_oms_intents_dedupe ON oms_intents(dedupe_key);
        CREATE INDEX IF NOT EXISTS idx_oms_intents_symbol_status ON oms_intents(symbol, status, sent_ts_ms);
        CREATE INDEX IF NOT EXISTS idx_oms_intents_client_order_id ON oms_intents(client_order_id);

        CREATE TABLE IF NOT EXISTS manual_trade_confirmations (
            confirm_token TEXT PRIMARY KEY,
            created_ts_ms INTEGER NOT NULL,
            expires_ts_ms INTEGER NOT NULL,
            action TEXT NOT NULL,
            symbol TEXT,
            param_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            intent_id TEXT,
            preview_json TEXT,
            last_error TEXT,
            consumed_ts_ms INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_manual_trade_confirmations_hash
            ON manual_trade_confirmations(action, param_hash, created_ts_ms);
        CREATE INDEX IF NOT EXISTS idx_manual_trade_confirmations_status_expiry
            ON manual_trade_confirmations(status, expires_ts_ms);
        CREATE INDEX IF NOT EXISTS idx_manual_trade_confirmations_intent
            ON manual_trade_confirmations(intent_id);

        CREATE TABLE IF NOT EXISTS oms_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id TEXT,
            created_ts_ms INTEGER NOT NULL,
            symbol TEXT,
            side TEXT,
            order_type TEXT,
            requested_size REAL,
            reduce_only INTEGER,
            client_order_id TEXT,
            exchange_order_id TEXT,
            status TEXT,
            raw_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_oms_orders_intent ON oms_orders(intent_id);
        CREATE INDEX IF NOT EXISTS idx_oms_orders_symbol ON oms_orders(symbol, created_ts_ms);

        CREATE TABLE IF NOT EXISTS oms_open_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_seen_ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            price REAL,
            remaining_size REAL,
            reduce_only INTEGER,
            client_order_id TEXT,
            exchange_order_id TEXT,
            intent_id TEXT,
            raw_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_oms_open_orders_client_order_id
            ON oms_open_orders(client_order_id);
        CREATE INDEX IF NOT EXISTS idx_oms_open_orders_symbol_last_seen
            ON oms_open_orders(symbol, last_seen_ts_ms);

        CREATE TABLE IF NOT EXISTS oms_fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER,
            symbol TEXT,
            intent_id TEXT,
            order_id INTEGER,
            action TEXT,
            side TEXT,
            pos_type TEXT,
            price REAL,
            size REAL,
            notional REAL,
            fee_usd REAL,
            fee_token TEXT,
            fee_rate REAL,
            pnl_usd REAL,
            fill_hash TEXT,
            fill_tid INTEGER,
            matched_via TEXT,
            raw_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_oms_fills_hash_tid ON oms_fills(fill_hash, fill_tid);
        CREATE INDEX IF NOT EXISTS idx_oms_fills_intent ON oms_fills(intent_id);
        CREATE INDEX IF NOT EXISTS idx_oms_fills_symbol_ts ON oms_fills(symbol, ts_ms);

        CREATE TABLE IF NOT EXISTS runtime_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts TEXT NOT NULL,
            pid INTEGER,
            mode TEXT,
            stream TEXT,
            level TEXT,
            message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_logs_ts_ms ON runtime_logs(ts_ms);
        CREATE INDEX IF NOT EXISTS idx_runtime_logs_mode_ts_ms ON runtime_logs(mode, ts_ms);

        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT,
            event TEXT NOT NULL,
            level TEXT,
            data_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_events_event_timestamp
            ON audit_events(event, timestamp);

        CREATE TABLE IF NOT EXISTS oms_reconcile_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            kind TEXT NOT NULL,
            symbol TEXT,
            intent_id TEXT,
            client_order_id TEXT,
            exchange_order_id TEXT,
            result TEXT,
            detail_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_oms_reconcile_events_ts_ms ON oms_reconcile_events(ts_ms);
        CREATE INDEX IF NOT EXISTS idx_oms_reconcile_events_kind_ts_ms
            ON oms_reconcile_events(kind, ts_ms);
        ",
    )?;
    tx.commit()?;
    Ok(())
}

fn purge_stale_manual_confirmations(conn: &Connection, now_ms: i64) -> Result<(), HubError> {
    conn.execute(
        "DELETE FROM manual_trade_confirmations
         WHERE created_ts_ms < ?1",
        [now_ms - MANUAL_CONFIRM_RETENTION_MS],
    )?;
    conn.execute(
        "UPDATE manual_trade_confirmations
         SET status = 'EXPIRED', last_error = 'confirm token expired'
         WHERE status = 'PREVIEWED' AND expires_ts_ms < ?1",
        [now_ms],
    )?;
    Ok(())
}

fn find_reusable_manual_confirmation(
    conn: &Connection,
    action: &str,
    param_hash: &str,
    now_ms: i64,
) -> Result<Option<String>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT confirm_token
         FROM manual_trade_confirmations
         WHERE action = ?1 AND param_hash = ?2 AND status = 'PREVIEWED' AND expires_ts_ms >= ?3
         ORDER BY created_ts_ms DESC
         LIMIT 1",
    )?;
    let token = stmt.query_row(params![action, param_hash, now_ms], |row| row.get(0));
    match token {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(HubError::Db(error.to_string())),
    }
}

fn validate_manual_confirmation(
    conn: &Connection,
    confirm_token: &str,
    action: &str,
    expected_hash: &str,
    now_ms: i64,
) -> Result<(), HubError> {
    let mut stmt = conn.prepare(
        "SELECT action, param_hash, expires_ts_ms
         FROM manual_trade_confirmations
         WHERE confirm_token = ?1",
    )?;
    let row = stmt.query_row([confirm_token], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    });
    let (stored_action, stored_hash, expires_ts_ms) = match row {
        Ok(value) => value,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(HubError::BadRequest(
                "invalid or expired confirm token".into(),
            ))
        }
        Err(error) => return Err(HubError::Db(error.to_string())),
    };
    if expires_ts_ms < now_ms {
        return Err(HubError::BadRequest("confirm token expired".into()));
    }
    if stored_action != action || stored_hash != expected_hash {
        return Err(HubError::BadRequest(
            "confirm token does not match parameters".into(),
        ));
    }
    Ok(())
}

fn bind_manual_confirmation_to_intent(
    conn: &Connection,
    confirm_token: &str,
    intent_id: &str,
) -> Result<(), HubError> {
    conn.execute(
        "UPDATE manual_trade_confirmations
         SET status = 'BOUND', intent_id = ?1, consumed_ts_ms = COALESCE(consumed_ts_ms, ?2)
         WHERE confirm_token = ?3",
        params![
            intent_id,
            chrono::Utc::now().timestamp_millis(),
            confirm_token
        ],
    )?;
    Ok(())
}

fn load_existing_manual_intent_by_dedupe_key(
    conn: &Connection,
    dedupe_key: &str,
) -> Result<Option<ExistingManualIntent>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT intent_id, created_ts_ms, symbol, action, side, status, requested_size, leverage,
                dedupe_key, client_order_id, exchange_order_id, last_error, sent_ts_ms, meta_json
         FROM oms_intents
         WHERE dedupe_key = ?1
         LIMIT 1",
    )?;
    let row = stmt.query_row([dedupe_key], |row| {
        Ok(ExistingManualIntent {
            intent_id: row.get(0)?,
            created_ts_ms: row.get(1)?,
            symbol: row.get(2)?,
            action: row.get(3)?,
            side: row.get(4)?,
            status: row.get(5)?,
            requested_size: row.get(6)?,
            leverage: row.get(7)?,
            dedupe_key: row.get(8)?,
            client_order_id: row.get(9)?,
            exchange_order_id: row.get(10)?,
            last_error: row.get(11)?,
            sent_ts_ms: row.get(12)?,
            meta_json: row.get(13)?,
        })
    });
    match row {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(HubError::Db(error.to_string())),
    }
}

fn load_reconcilable_manual_intents(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<ExistingManualIntent>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT intent_id, created_ts_ms, symbol, action, side, status, requested_size, leverage,
                dedupe_key, client_order_id, exchange_order_id, last_error, sent_ts_ms, meta_json
         FROM oms_intents
         WHERE reason = 'manual_trade' AND status IN ('NEW', 'UNKNOWN', 'SENT', 'PARTIAL')
         ORDER BY COALESCE(sent_ts_ms, created_ts_ms) DESC
         LIMIT ?1",
    )?;
    let rows = stmt
        .query_map([limit as i64], |row| {
            Ok(ExistingManualIntent {
                intent_id: row.get(0)?,
                created_ts_ms: row.get(1)?,
                symbol: row.get(2)?,
                action: row.get(3)?,
                side: row.get(4)?,
                status: row.get(5)?,
                requested_size: row.get(6)?,
                leverage: row.get(7)?,
                dedupe_key: row.get(8)?,
                client_order_id: row.get(9)?,
                exchange_order_id: row.get(10)?,
                last_error: row.get(11)?,
                sent_ts_ms: row.get(12)?,
                meta_json: row.get(13)?,
            })
        })?
        .filter_map(|row| row.ok())
        .collect();
    Ok(rows)
}

fn load_manual_cancel_context_by_intent_id(
    conn: &Connection,
    intent_id: &str,
) -> Result<Option<ManualCancelContext>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT symbol, side, status, requested_size, action, client_order_id, exchange_order_id
         FROM oms_intents
         WHERE intent_id = ?1
         LIMIT 1",
    )?;
    let row = stmt.query_row([intent_id], |row| {
        let action = row.get::<_, Option<String>>(4)?;
        Ok(ManualCancelContext {
            intent_exists: true,
            intent_id: Some(intent_id.to_string()),
            symbol: row.get::<_, String>(0)?.trim().to_ascii_uppercase(),
            side: row.get(1)?,
            current_status: row.get(2)?,
            requested_size: row.get(3)?,
            reduce_only: action
                .as_deref()
                .map(|value| matches!(value, "CLOSE" | "REDUCE")),
            client_order_id: row.get(5)?,
            exchange_order_id: row.get(6)?,
        })
    });
    match row {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(HubError::Db(error.to_string())),
    }
}

fn load_manual_cancel_context_by_exchange_order_id(
    conn: &Connection,
    symbol: &str,
    exchange_order_id: &str,
) -> Result<Option<ManualCancelContext>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT intent_id, side, status, requested_size, action, client_order_id, exchange_order_id
         FROM oms_intents
         WHERE symbol = ?1 AND exchange_order_id = ?2
         ORDER BY sent_ts_ms DESC
         LIMIT 1",
    )?;
    let row = stmt.query_row(params![symbol, exchange_order_id], |row| {
        let action = row.get::<_, Option<String>>(4)?;
        Ok(ManualCancelContext {
            intent_exists: true,
            intent_id: row.get(0)?,
            symbol: symbol.to_string(),
            side: row.get(1)?,
            current_status: row.get(2)?,
            requested_size: row.get(3)?,
            reduce_only: action
                .as_deref()
                .map(|value| matches!(value, "CLOSE" | "REDUCE")),
            client_order_id: row.get(5)?,
            exchange_order_id: row.get(6)?,
        })
    });
    match row {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(HubError::Db(error.to_string())),
    }
}

fn cancelled_intent_status(current_status: Option<&str>) -> Option<&'static str> {
    let status = current_status?.trim().to_ascii_uppercase();
    match status.as_str() {
        "PARTIAL" => Some("PARTIAL_CANCELLED"),
        "NEW" | "SENT" | "UNKNOWN" => Some("CANCELLED"),
        _ => None,
    }
}

fn existing_intent_cloid(existing: &ExistingManualIntent) -> String {
    existing
        .client_order_id
        .clone()
        .unwrap_or_else(|| manual_cloid_from_intent_id(&existing.intent_id))
}

fn should_resume_existing_manual_submission(existing: &ExistingManualIntent) -> bool {
    existing.sent_ts_ms.is_none() && existing.status.eq_ignore_ascii_case("NEW")
}

fn update_manual_intent_last_error(
    conn: &Connection,
    intent_id: &str,
    last_error: &str,
) -> Result<(), HubError> {
    let changed = conn.execute(
        "UPDATE oms_intents SET last_error = ?1 WHERE intent_id = ?2",
        params![last_error, intent_id],
    )?;
    if changed == 0 {
        return Err(HubError::Internal(format!(
            "failed to update manual OMS intent error for {intent_id}"
        )));
    }
    Ok(())
}

fn existing_manual_execution_payload(kind: &str, existing: &ExistingManualIntent) -> Value {
    json!({
        "ok": true,
        "status": if should_resume_existing_manual_submission(existing) {
            "pending"
        } else if existing.status.eq_ignore_ascii_case("REJECTED") {
            "rejected"
        } else {
            "submitted"
        },
        "deduped": true,
        "kind": kind,
        "intent_id": existing.intent_id,
        "symbol": existing.symbol,
        "action": existing.action,
        "side": existing.side,
        "intent_status": existing.status,
        "client_order_id": existing.client_order_id,
        "exchange_order_id": existing.exchange_order_id,
        "last_error": existing.last_error,
        "sent_ts_ms": existing.sent_ts_ms,
    })
}

fn same_optional_price(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs() < 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn parse_resume_request_meta(existing: &ExistingManualIntent) -> Result<Value, HubError> {
    existing
        .meta_json
        .as_deref()
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing meta_json for retry validation",
                existing.intent_id
            ))
        })
        .and_then(|raw| serde_json::from_str(raw).map_err(HubError::from))
}

fn validate_resumed_open_request(
    existing: &ExistingManualIntent,
    request: &ManualTradeOpenRequest,
) -> Result<(), HubError> {
    let meta = parse_resume_request_meta(existing)?;
    let stored = meta
        .get("request")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing request metadata",
                existing.intent_id
            ))
        })?;
    let stored_side = stored
        .get("side")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase();
    let stored_notional = stored
        .get("notional_usd")
        .and_then(parse_json_number)
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing stored notional",
                existing.intent_id
            ))
        })?;
    let stored_leverage = stored
        .get("leverage")
        .and_then(parse_json_integer)
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing stored leverage",
                existing.intent_id
            ))
        })? as u32;
    let stored_order_type = stored
        .get("order_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let stored_limit_price = stored.get("limit_price").and_then(parse_json_number);

    if !existing
        .symbol
        .trim()
        .eq_ignore_ascii_case(request.symbol.trim())
        || stored_side != request.side.trim().to_ascii_uppercase()
        || (stored_notional - request.notional_usd).abs() >= 1e-9
        || stored_leverage != request.leverage
        || stored_order_type != request.order_type.trim().to_ascii_lowercase()
        || !same_optional_price(stored_limit_price, request.limit_price)
    {
        return Err(HubError::BadRequest(
            "confirm token does not match parameters".into(),
        ));
    }
    Ok(())
}

fn validate_resumed_close_request(
    existing: &ExistingManualIntent,
    request: &ManualTradeCloseRequest,
) -> Result<(), HubError> {
    let meta = parse_resume_request_meta(existing)?;
    let stored = meta
        .get("request")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing request metadata",
                existing.intent_id
            ))
        })?;
    let stored_close_pct = stored
        .get("close_pct")
        .and_then(parse_json_number)
        .ok_or_else(|| {
            HubError::Internal(format!(
                "manual intent {} is missing stored close_pct",
                existing.intent_id
            ))
        })?;
    let stored_order_type = stored
        .get("order_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let stored_limit_price = stored.get("limit_price").and_then(parse_json_number);

    if !existing
        .symbol
        .trim()
        .eq_ignore_ascii_case(request.symbol.trim())
        || (stored_close_pct - request.close_pct).abs() >= 1e-9
        || stored_order_type != request.order_type.trim().to_ascii_lowercase()
        || !same_optional_price(stored_limit_price, request.limit_price)
    {
        return Err(HubError::BadRequest(
            "confirm token does not match parameters".into(),
        ));
    }
    Ok(())
}

fn find_matching_open_order<'a>(
    orders: &'a [Value],
    existing: &ExistingManualIntent,
    cloid: &str,
) -> Option<&'a Value> {
    let symbol_upper = existing.symbol.trim().to_ascii_uppercase();
    orders.iter().find(|order| {
        let order_symbol = order
            .get("coin")
            .or_else(|| order.get("symbol"))
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_ascii_uppercase());
        if order_symbol.as_deref() != Some(symbol_upper.as_str()) {
            return false;
        }
        let order_cloid = order
            .get("cloid")
            .or_else(|| order.get("clientOrderId"))
            .or_else(|| order.get("client_order_id"))
            .and_then(|value| value.as_str())
            .map(str::trim);
        if order_cloid == Some(cloid) {
            return true;
        }
        let order_exchange_id = order
            .get("oid")
            .and_then(parse_json_integer)
            .map(|value| value.to_string());
        order_exchange_id.as_deref() == existing.exchange_order_id.as_deref()
    })
}

fn recover_existing_new_manual_execution(
    cfg: &HubConfig,
    conn: &mut Connection,
    existing: &ExistingManualIntent,
) -> Result<Option<ExistingManualIntent>, HubError> {
    let client = build_client(cfg)?;
    let cloid = existing_intent_cloid(existing);
    let start_ms = existing.created_ts_ms.saturating_sub(5_000);
    let fills = poll_fill(
        &client,
        &existing.symbol,
        start_ms,
        &cloid,
        existing.exchange_order_id.as_deref(),
    )?;
    if !fills.is_empty() {
        let account_value_usd = client
            .account_snapshot()
            .map_err(|error| HubError::Internal(error.to_string()))?
            .account_value_usd;
        let fill_summary = record_manual_fill_batch(
            conn,
            &fills,
            &existing.symbol,
            &existing.intent_id,
            &cloid,
            account_value_usd,
            existing.leverage.unwrap_or(1.0).max(1.0),
        )?;
        let requested_size = existing.requested_size.unwrap_or(fill_summary.filled_size);
        let status = if fill_summary.parse_failures > 0 {
            "UNKNOWN"
        } else if fill_summary.filled_size + 1e-9 >= requested_size {
            "FILLED"
        } else {
            "PARTIAL"
        };
        let last_error = if fill_summary.parse_failures > 0 {
            "one_or_more_fills_failed_to_parse_into_trades"
        } else {
            ""
        };
        update_manual_intent_status(conn, &existing.intent_id, status, last_error)?;
        write_manual_runtime_log(
            conn,
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            &format!(
                "manual_trade recovered_pending_intent intent_id={} symbol={} observed_fills={} parsed_trade_rows={} filled_size={} status={}",
                existing.intent_id,
                existing.symbol,
                fill_summary.observed_fills,
                fill_summary.parsed_trade_rows,
                fill_summary.filled_size,
                status
            ),
        )?;
        write_manual_audit_event(
            conn,
            Some(&existing.symbol),
            "MANUAL_FILLS_RECORDED",
            if fill_summary.parse_failures > 0 {
                "WARN"
            } else {
                "INFO"
            },
            json!({
                "intent_id": &existing.intent_id,
                "action": &existing.action,
                "status": status,
                "observed_fills": fill_summary.observed_fills,
                "parsed_trade_rows": fill_summary.parsed_trade_rows,
                "filled_size": fill_summary.filled_size,
                "source": "recovery",
            }),
        )?;
        return load_existing_manual_intent_by_dedupe_key(
            conn,
            existing.dedupe_key.as_deref().ok_or_else(|| {
                HubError::Internal(format!(
                    "manual intent {} is missing its dedupe key",
                    existing.intent_id
                ))
            })?,
        );
    }

    let open_orders = client
        .open_orders()
        .map_err(|error| HubError::Internal(format!("failed to inspect open orders: {error}")))?;
    if let Some(order) = find_matching_open_order(&open_orders, existing, &cloid) {
        let exchange_order_id = order
            .get("oid")
            .and_then(parse_json_integer)
            .map(|value| value.to_string());
        record_manual_order_submission(
            conn,
            ManualOrderSubmission {
                intent_id: &existing.intent_id,
                sent_ts_ms: chrono::Utc::now().timestamp_millis(),
                symbol: &existing.symbol,
                side: &existing.side,
                order_type: "recovered_open_order",
                requested_size: existing.requested_size.unwrap_or(0.0),
                reduce_only: matches!(existing.action.as_str(), "CLOSE" | "REDUCE"),
                client_order_id: &cloid,
                exchange_order_id: exchange_order_id.as_deref(),
                status: "SENT",
                last_error: "",
                raw_json: order.to_string(),
            },
        )?;
        write_manual_runtime_log(
            conn,
            "INFO",
            &format!(
                "manual_trade recovered_pending_open_order intent_id={} symbol={} exchange_order_id={}",
                existing.intent_id,
                existing.symbol,
                exchange_order_id.as_deref().unwrap_or("unknown")
            ),
        )?;
        write_manual_audit_event(
            conn,
            Some(&existing.symbol),
            "MANUAL_ORDER_SUBMITTED",
            "INFO",
            json!({
                "intent_id": &existing.intent_id,
                "action": &existing.action,
                "exchange_order_id": &exchange_order_id,
                "client_order_id": &cloid,
                "order_type": "recovered_open_order",
                "source": "recovery",
            }),
        )?;
        return load_existing_manual_intent_by_dedupe_key(
            conn,
            existing.dedupe_key.as_deref().ok_or_else(|| {
                HubError::Internal(format!(
                    "manual intent {} is missing its dedupe key",
                    existing.intent_id
                ))
            })?,
        );
    }

    Ok(None)
}

fn initialise_manual_execution(
    conn: &Connection,
    confirm_token: &str,
    action: &str,
    expected_hash: &str,
    record: ManualIntentRecord<'_>,
) -> Result<ManualExecutionClaim, HubError> {
    let new_intent_id = record.intent_id.to_string();
    let new_cloid = record.client_order_id.to_string();
    let now_ms = chrono::Utc::now().timestamp_millis();
    purge_stale_manual_confirmations(conn, now_ms)?;
    if let Some(existing) = load_existing_manual_intent_by_dedupe_key(conn, confirm_token)? {
        bind_manual_confirmation_to_intent(conn, confirm_token, &existing.intent_id)?;
        return Ok(ManualExecutionClaim::Existing(Box::new(existing)));
    }

    validate_manual_confirmation(conn, confirm_token, action, expected_hash, now_ms)?;
    let inserted = insert_manual_intent(conn, record)?;
    if !inserted {
        if let Some(existing) = load_existing_manual_intent_by_dedupe_key(conn, confirm_token)? {
            bind_manual_confirmation_to_intent(conn, confirm_token, &existing.intent_id)?;
            return Ok(ManualExecutionClaim::Existing(Box::new(existing)));
        }
        return Err(HubError::Internal(format!(
            "manual intent insert dedupe race without existing intent for {}",
            new_intent_id
        )));
    }
    bind_manual_confirmation_to_intent(conn, confirm_token, &new_intent_id)?;
    Ok(ManualExecutionClaim::Submit {
        intent_id: new_intent_id,
        cloid: new_cloid,
        resumed_existing: false,
    })
}

fn insert_manual_intent(
    conn: &Connection,
    record: ManualIntentRecord<'_>,
) -> Result<bool, HubError> {
    let changed = conn.execute(
        "INSERT OR IGNORE INTO oms_intents (
            intent_id, created_ts_ms, symbol, action, side, requested_size,
            requested_notional, leverage, reason, confidence, status,
            dedupe_key, client_order_id, meta_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            record.intent_id,
            record.created_ts_ms,
            record.symbol,
            record.action,
            record.side,
            record.requested_size,
            record.requested_notional,
            record.leverage,
            record.reason,
            record.confidence,
            record.status,
            record.dedupe_key,
            record.client_order_id,
            record.meta_json,
        ],
    )?;
    Ok(changed > 0)
}

fn update_manual_intent_status(
    conn: &Connection,
    intent_id: &str,
    status: &str,
    last_error: &str,
) -> Result<(), HubError> {
    let changed = conn.execute(
        "UPDATE oms_intents SET status = ?1, last_error = ?2 WHERE intent_id = ?3",
        params![status, last_error, intent_id],
    )?;
    if changed == 0 {
        return Err(HubError::Internal(format!(
            "failed to update manual OMS intent status for {intent_id}"
        )));
    }
    Ok(())
}

fn record_manual_order_submission(
    conn: &Connection,
    submission: ManualOrderSubmission<'_>,
) -> Result<(), HubError> {
    let changed = conn.execute(
        "UPDATE oms_intents
         SET status = ?1, sent_ts_ms = ?2, exchange_order_id = ?3, last_error = ?4
         WHERE intent_id = ?5",
        params![
            submission.status,
            submission.sent_ts_ms,
            submission.exchange_order_id,
            submission.last_error,
            submission.intent_id,
        ],
    )?;
    if changed == 0 {
        return Err(HubError::Internal(format!(
            "failed to record manual order submission for {}",
            submission.intent_id
        )));
    }

    conn.execute(
        "INSERT INTO oms_orders (
            intent_id, created_ts_ms, symbol, side, order_type, requested_size,
            reduce_only, client_order_id, exchange_order_id, status, raw_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            submission.intent_id,
            submission.sent_ts_ms,
            submission.symbol,
            submission.side,
            submission.order_type,
            submission.requested_size,
            if submission.reduce_only { 1 } else { 0 },
            submission.client_order_id,
            submission.exchange_order_id,
            submission.status,
            submission.raw_json,
        ],
    )?;
    Ok(())
}

fn record_manual_cancel_audit(
    conn: &Connection,
    context: Option<&ManualCancelContext>,
    audit: ManualCancelAudit<'_>,
) -> Result<(), HubError> {
    let ts_ms = chrono::Utc::now().timestamp_millis();
    let can_update_intent = context.map(|item| item.intent_exists).unwrap_or(false);
    if can_update_intent {
        let intent_id = audit.intent_id.ok_or_else(|| {
            HubError::Internal("cancel audit expected an intent_id for a persisted context".into())
        })?;
        match audit.status {
            "CANCELLED" => {
                if let Some(next_status) =
                    cancelled_intent_status(context.and_then(|item| item.current_status.as_deref()))
                {
                    update_manual_intent_status(conn, intent_id, next_status, "")?;
                }
            }
            "REJECTED" | "ERROR" => {
                update_manual_intent_last_error(conn, intent_id, &audit.raw_json)?;
            }
            _ => {}
        }
    }

    conn.execute(
        "INSERT INTO oms_orders (
            intent_id, created_ts_ms, symbol, side, order_type, requested_size,
            reduce_only, client_order_id, exchange_order_id, status, raw_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            audit.intent_id,
            ts_ms,
            audit.symbol,
            audit.side,
            audit.order_type,
            audit.requested_size,
            audit.reduce_only.map(|value| if value { 1 } else { 0 }),
            audit.client_order_id,
            audit.exchange_order_id,
            audit.status,
            audit.raw_json,
        ],
    )?;
    Ok(())
}

fn write_manual_runtime_log(conn: &Connection, level: &str, message: &str) -> Result<(), HubError> {
    let ts_ms = chrono::Utc::now().timestamp_millis();
    let ts = chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
    conn.execute(
        "INSERT INTO runtime_logs (ts_ms, ts, pid, mode, stream, level, message)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            ts_ms,
            ts,
            i64::from(process::id()),
            MANUAL_RUNTIME_MODE,
            MANUAL_RUNTIME_STREAM,
            level.trim().to_ascii_uppercase(),
            message,
        ],
    )?;
    Ok(())
}

fn write_manual_audit_event(
    conn: &Connection,
    symbol: Option<&str>,
    event: &str,
    level: &str,
    data: Value,
) -> Result<(), HubError> {
    let ts_ms = chrono::Utc::now().timestamp_millis();
    let ts = chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
    let symbol = symbol.map(|value| value.trim().to_ascii_uppercase());
    let event = event.trim().to_ascii_uppercase();
    let level = level.trim().to_ascii_uppercase();
    let payload = data.to_string();
    if table_has_column(conn, "audit_events", "ts_ms")? {
        conn.execute(
            "INSERT INTO audit_events (ts_ms, timestamp, symbol, event, level, data_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![ts_ms, ts, symbol, event, level, payload],
        )?;
    } else {
        conn.execute(
            "INSERT INTO audit_events (timestamp, symbol, event, level, data_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ts, symbol, event, level, payload],
        )?;
    }
    Ok(())
}

fn write_manual_reconcile_event(
    conn: &Connection,
    kind: &str,
    symbol: Option<&str>,
    result: &str,
    data: Value,
) -> Result<(), HubError> {
    let ts_ms = chrono::Utc::now().timestamp_millis();
    let kind = kind.trim().to_ascii_lowercase();
    let symbol = symbol.map(|value| value.trim().to_ascii_uppercase());
    let payload = data.to_string();
    if table_has_column(conn, "oms_reconcile_events", "detail_json")? {
        conn.execute(
            "INSERT INTO oms_reconcile_events (ts_ms, kind, symbol, result, detail_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ts_ms, kind, symbol, result, payload],
        )?;
    } else {
        conn.execute(
            "INSERT INTO oms_reconcile_events (ts_ms, kind, symbol, result, data_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ts_ms, kind, symbol, result, payload],
        )?;
    }
    Ok(())
}

fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, HubError> {
    let pragma = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&pragma)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|row| row.ok())
        .collect::<Vec<_>>();
    Ok(rows.into_iter().any(|name| name == column))
}

fn upsert_manual_open_order_snapshot(
    conn: &Connection,
    snapshot: ManualOpenOrderSnapshot<'_>,
) -> Result<bool, HubError> {
    let mut stmt = conn.prepare(
        "SELECT symbol, side, price, remaining_size, reduce_only, exchange_order_id, intent_id, raw_json
         FROM oms_open_orders
         WHERE client_order_id = ?1
         LIMIT 1",
    )?;
    let existing = stmt.query_row([snapshot.client_order_id], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<f64>>(2)?,
            row.get::<_, Option<f64>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, Option<String>>(6)?,
        ))
    });
    let materially_changed = match existing {
        Ok(existing) => {
            existing.0.as_deref() != Some(snapshot.symbol)
                || existing.1.as_deref() != Some(snapshot.side)
                || existing.2 != snapshot.price
                || existing.3 != snapshot.remaining_size
                || existing.4.unwrap_or_default() != if snapshot.reduce_only { 1 } else { 0 }
                || existing.5.as_deref() != snapshot.exchange_order_id
                || existing.6.as_deref() != Some(snapshot.intent_id)
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => true,
        Err(error) => return Err(HubError::Db(error.to_string())),
    };
    conn.execute(
        "INSERT INTO oms_open_orders (
            last_seen_ts_ms, symbol, side, price, remaining_size, reduce_only,
            client_order_id, exchange_order_id, intent_id, raw_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ON CONFLICT(client_order_id) DO UPDATE SET
            last_seen_ts_ms = excluded.last_seen_ts_ms,
            symbol = excluded.symbol,
            side = excluded.side,
            price = excluded.price,
            remaining_size = excluded.remaining_size,
            reduce_only = excluded.reduce_only,
            exchange_order_id = excluded.exchange_order_id,
            intent_id = excluded.intent_id,
            raw_json = excluded.raw_json",
        params![
            snapshot.last_seen_ts_ms,
            snapshot.symbol,
            snapshot.side,
            snapshot.price,
            snapshot.remaining_size,
            if snapshot.reduce_only { 1 } else { 0 },
            snapshot.client_order_id,
            snapshot.exchange_order_id,
            snapshot.intent_id,
            snapshot.raw_json,
        ],
    )?;
    Ok(materially_changed)
}

fn clear_manual_open_order_snapshot(
    conn: &Connection,
    intent_id: &str,
    client_order_id: &str,
) -> Result<bool, HubError> {
    let changed = conn.execute(
        "DELETE FROM oms_open_orders WHERE intent_id = ?1 OR client_order_id = ?2",
        params![intent_id, client_order_id],
    )?;
    Ok(changed > 0)
}

fn open_order_side(order: &Value) -> &'static str {
    match order
        .get("side")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
    {
        value if value.eq_ignore_ascii_case("B") || value.eq_ignore_ascii_case("BUY") => "BUY",
        _ => "SELL",
    }
}

fn open_order_reduce_only(order: &Value) -> bool {
    match order.get("reduceOnly") {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_i64().unwrap_or_default() != 0,
        Some(Value::String(value)) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        _ => false,
    }
}

fn desired_manual_reconcile_state(
    existing: &ExistingManualIntent,
    fill_summary: &FillWriteSummary,
    has_open_order: bool,
    now_ms: i64,
) -> (&'static str, &'static str) {
    if fill_summary.parse_failures > 0 {
        return (
            "UNKNOWN_RECONCILED",
            "one_or_more_fills_failed_to_parse_into_trades",
        );
    }

    let requested_size = existing.requested_size.unwrap_or(fill_summary.filled_size);
    if fill_summary.filled_size > 0.0 {
        if fill_summary.filled_size + 1e-9 >= requested_size {
            return ("FILLED", "");
        }
        if has_open_order {
            return ("PARTIAL", "");
        }
        return ("PARTIAL_CANCELLED", "");
    }

    if has_open_order {
        return ("SENT", "");
    }

    if now_ms.saturating_sub(existing.created_ts_ms) < MANUAL_RECONCILE_MISSING_GRACE_MS {
        return match existing.status.trim().to_ascii_uppercase().as_str() {
            "NEW" => ("NEW", ""),
            "SENT" => ("SENT", ""),
            "PARTIAL" => ("PARTIAL", ""),
            "UNKNOWN" => ("UNKNOWN", "manual intent still awaiting reconcile evidence"),
            _ => ("UNKNOWN", "manual intent still awaiting reconcile evidence"),
        };
    }

    (
        "UNKNOWN",
        "manual order missing from exchange during reconcile",
    )
}

fn submission_order_type(order_type: ParsedOrderType, reduce_only: bool) -> &'static str {
    match (order_type, reduce_only) {
        (ParsedOrderType::Market, false) => "market_open",
        (ParsedOrderType::Market, true) => "market_close",
        (ParsedOrderType::LimitIoc, _) => "limit_ioc",
        (ParsedOrderType::LimitGtc, _) => "limit_gtc",
    }
}

fn failure_status_for_error(error_text: &str) -> &'static str {
    let lower = error_text.trim().to_ascii_lowercase();
    if lower.contains("timed out")
        || lower.contains("request failed")
        || lower.contains("connection")
        || lower.contains("dns")
        || lower.contains("transport")
    {
        "UNKNOWN"
    } else {
        "REJECTED"
    }
}

fn record_manual_fill_batch(
    conn: &mut Connection,
    fills: &[HyperliquidFill],
    symbol_hint: &str,
    intent_id: &str,
    cloid: &str,
    account_value_usd: f64,
    leverage: f64,
) -> Result<FillWriteSummary, HubError> {
    let tx = conn.transaction()?;
    let mut summary = FillWriteSummary {
        observed_fills: fills.len(),
        inserted_oms_fills: 0,
        parsed_trade_rows: 0,
        parse_failures: 0,
        filled_size: 0.0,
    };
    for fill in fills {
        let parsed = parse_fill(&fill.raw);
        if insert_manual_oms_fill(&tx, fill, parsed.as_ref(), symbol_hint, intent_id)? {
            summary.inserted_oms_fills += 1;
        }
        if let Some(parsed) = parsed.as_ref() {
            if insert_manual_trade_row(
                &tx,
                parsed,
                &fill.raw,
                intent_id,
                cloid,
                account_value_usd,
                leverage,
            )? {
                summary.parsed_trade_rows += 1;
            }
            summary.filled_size += parsed.size;
        } else {
            summary.parse_failures += 1;
        }
    }
    tx.commit()?;
    Ok(summary)
}

fn insert_manual_oms_fill(
    conn: &Connection,
    fill: &HyperliquidFill,
    parsed: Option<&ParsedFill>,
    symbol_hint: &str,
    intent_id: &str,
) -> Result<bool, HubError> {
    let fill_ts_ms = fill_timestamp_ms(&fill.raw);
    let fill_hash = fill.raw.get("hash").and_then(|value| value.as_str());
    let fill_tid = fill.raw.get("tid").and_then(|value| value.as_i64());
    let changed = conn.execute(
        "INSERT OR IGNORE INTO oms_fills (
            ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
            fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        params![
            fill_ts_ms,
            parsed
                .as_ref()
                .map(|item| item.symbol.as_str())
                .unwrap_or(symbol_hint),
            intent_id,
            fill.raw.get("oid").and_then(parse_json_integer),
            parsed.as_ref().map(|item| item.action.as_str()),
            parsed.as_ref().map(|item| item.side.as_str()),
            parsed.as_ref().map(|item| item.pos_type.as_str()),
            parsed.as_ref().map(|item| item.price),
            parsed.as_ref().map(|item| item.size),
            parsed.as_ref().map(|item| item.notional_usd),
            parsed.as_ref().map(|item| item.fee_usd),
            parsed
                .as_ref()
                .and_then(|item| item.fee_token.as_deref())
                .or_else(|| fill.raw.get("feeToken").and_then(|value| value.as_str())),
            parsed.as_ref().map(|item| item.fee_rate),
            parsed.as_ref().map(|item| item.pnl_usd),
            fill_hash,
            fill_tid,
            "manual_trade",
            fill.raw.to_string(),
        ],
    )?;
    Ok(changed > 0)
}

fn insert_manual_trade_row(
    conn: &Connection,
    parsed: &ParsedFill,
    raw_fill: &Value,
    intent_id: &str,
    cloid: &str,
    account_value_usd: f64,
    leverage: f64,
) -> Result<bool, HubError> {
    let meta_json = json!({
        "source": "manual_trade",
        "intent_id": intent_id,
        "cloid": cloid,
        "fill": raw_fill,
        "oms": {
            "intent_id": intent_id,
            "client_order_id": cloid,
            "exchange_order_id": parsed.exchange_order_id,
            "matched_via": "manual_trade",
        }
    })
    .to_string();

    let fill_hash = parsed.fill_hash.as_deref();
    let fill_tid = parsed.fill_tid;
    let changed = conn.execute(
        "INSERT OR IGNORE INTO trades (
            timestamp, symbol, type, action, price, size, notional,
            reason, reason_code, confidence, pnl, fee_usd, fee_token, fee_rate,
            balance, entry_atr, leverage, margin_used, meta_json, fill_hash, fill_tid
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        params![
            parsed.timestamp,
            parsed.symbol,
            parsed.pos_type,
            parsed.action,
            parsed.price,
            parsed.size,
            parsed.notional_usd,
            parsed.reason,
            "manual_trade",
            "MANUAL",
            parsed.pnl_usd,
            parsed.fee_usd,
            parsed.fee_token,
            parsed.fee_rate,
            account_value_usd,
            Option::<f64>::None,
            leverage,
            if leverage > 0.0 {
                Some(parsed.notional_usd / leverage)
            } else {
                None
            },
            meta_json,
            fill_hash,
            fill_tid,
        ],
    )?;
    Ok(changed > 0)
}

fn prepare_open(
    client: &HyperliquidClient,
    cfg: &HubConfig,
    request: &ManualTradeOpenRequest,
) -> Result<PreparedOpen, HubError> {
    let symbol = request.symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return Err(HubError::BadRequest("symbol is required".into()));
    }
    let side = request.side.trim().to_ascii_uppercase();
    if side != "BUY" && side != "SELL" {
        return Err(HubError::BadRequest("side must be BUY or SELL".into()));
    }
    if !request.notional_usd.is_finite() || request.notional_usd < MIN_NOTIONAL_USD {
        return Err(HubError::BadRequest(format!(
            "minimum notional is ${MIN_NOTIONAL_USD:.0}"
        )));
    }
    let max_notional = std::env::var("AI_QUANT_MANUAL_MAX_NOTIONAL_USD")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(DEFAULT_MAX_NOTIONAL_USD);
    if request.notional_usd > max_notional {
        return Err(HubError::BadRequest(format!(
            "notional ${:.2} exceeds max ${max_notional:.2}",
            request.notional_usd
        )));
    }

    let meta = client
        .asset_meta(&symbol)
        .map_err(|error| HubError::BadRequest(error.to_string()))?;
    let leverage = request.leverage.clamp(1, meta.max_leverage.max(1));
    if request.leverage > meta.max_leverage.max(1) {
        return Err(HubError::BadRequest(format!(
            "leverage {}x exceeds max {}x for {}",
            request.leverage,
            meta.max_leverage.max(1),
            symbol
        )));
    }

    let mid_price = client
        .all_mids()
        .map_err(|error| HubError::Internal(error.to_string()))?
        .get(&symbol)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| HubError::BadRequest(format!("could not fetch mid price for {symbol}")))?;

    let raw_size = request.notional_usd / mid_price;
    let mut est_size = round_size_down(raw_size, meta.sz_decimals);
    if est_size <= 0.0 {
        est_size = round_size_up(raw_size, meta.sz_decimals);
    }
    if est_size <= 0.0 {
        return Err(HubError::BadRequest(format!(
            "computed size is zero for {symbol}"
        )));
    }

    let est_notional_usd = est_size * mid_price;
    let est_margin_usd = est_notional_usd / f64::from(leverage.max(1));
    let est_fee_usd = est_notional_usd * cfg.fee_rate.max(HL_TAKER_FEE_RATE);
    let account_value_usd = client
        .account_snapshot()
        .map_err(|error| HubError::Internal(error.to_string()))?
        .account_value_usd;

    let order_type = parse_order_type(&request.order_type)?;
    let limit_price = match order_type {
        ParsedOrderType::Market => None,
        ParsedOrderType::LimitIoc | ParsedOrderType::LimitGtc => Some(
            normalise_limit_px_for_wire(
                request
                    .limit_price
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| {
                        HubError::BadRequest("limit_price is required for limit orders".into())
                    })?,
                side == "BUY",
            )
            .ok_or_else(|| HubError::BadRequest("invalid limit_price".into()))?,
        ),
    };

    Ok(PreparedOpen {
        symbol,
        side: side.clone(),
        direction: if side == "BUY" {
            "LONG".to_string()
        } else {
            "SHORT".to_string()
        },
        is_buy: side == "BUY",
        mid_price,
        est_size,
        est_notional_usd,
        est_margin_usd,
        est_fee_usd,
        leverage,
        max_leverage: meta.max_leverage.max(1),
        sz_decimals: meta.sz_decimals,
        order_type,
        limit_price,
        account_value_usd,
    })
}

fn prepare_close(
    client: &HyperliquidClient,
    request: &ManualTradeCloseRequest,
) -> Result<PreparedClose, HubError> {
    let symbol = request.symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return Err(HubError::BadRequest("symbol is required".into()));
    }
    if !request.close_pct.is_finite() || request.close_pct <= 0.0 || request.close_pct > 100.0 {
        return Err(HubError::BadRequest(
            "close_pct must be between 0 and 100".into(),
        ));
    }
    let order_type = parse_order_type(&request.order_type)?;
    let positions = client
        .positions()
        .map_err(|error| HubError::Internal(error.to_string()))?;
    let position = positions
        .into_iter()
        .find(|position| position.symbol == symbol)
        .ok_or_else(|| HubError::BadRequest(format!("no open position for {symbol}")))?;
    let meta = client
        .asset_meta(&symbol)
        .map_err(|error| HubError::BadRequest(error.to_string()))?;

    let mut close_size = round_size_down(
        position.size * (request.close_pct / 100.0),
        meta.sz_decimals,
    );
    if close_size <= 0.0 {
        close_size = round_size_up(
            position.size * (request.close_pct / 100.0),
            meta.sz_decimals,
        );
    }
    close_size = close_size.min(position.size);
    if close_size <= 0.0 {
        return Err(HubError::BadRequest("close size rounded to zero".into()));
    }

    let mid_price = client
        .all_mids()
        .map_err(|error| HubError::Internal(error.to_string()))?
        .get(&symbol)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| HubError::BadRequest(format!("could not fetch mid price for {symbol}")))?;

    let is_buy = position.pos_type.eq_ignore_ascii_case("SHORT");
    let limit_price = match order_type {
        ParsedOrderType::Market => None,
        ParsedOrderType::LimitIoc | ParsedOrderType::LimitGtc => Some(
            normalise_limit_px_for_wire(
                request
                    .limit_price
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| {
                        HubError::BadRequest("limit_price is required for limit orders".into())
                    })?,
                is_buy,
            )
            .ok_or_else(|| HubError::BadRequest("invalid limit_price".into()))?,
        ),
    };

    let account_value_usd = client
        .account_snapshot()
        .map_err(|error| HubError::Internal(error.to_string()))?
        .account_value_usd;

    Ok(PreparedClose {
        symbol,
        pos_type: position.pos_type,
        is_buy,
        current_size: position.size,
        close_size,
        leverage: position.leverage.max(1.0),
        account_value_usd,
        mid_price,
        order_type,
        limit_price,
    })
}

fn submit_open_order(
    client: &HyperliquidClient,
    prepared: &PreparedOpen,
    cloid: &str,
) -> Result<Value, HubError> {
    let leverage_response = client
        .update_leverage(&prepared.symbol, prepared.leverage, true)
        .map_err(|error| HubError::BadRequest(format!("failed to set leverage: {error}")))?;
    if response_has_embedded_error(&leverage_response) {
        return Err(HubError::BadRequest(leverage_response.to_string()));
    }

    let response = match prepared.order_type {
        ParsedOrderType::Market => client.market_open(
            &prepared.symbol,
            prepared.is_buy,
            prepared.est_size,
            Some(prepared.mid_price),
            DEFAULT_SLIPPAGE_PCT,
            Some(cloid),
        ),
        ParsedOrderType::LimitIoc => client.order(OrderRequest {
            symbol: prepared.symbol.clone(),
            is_buy: prepared.is_buy,
            size: prepared.est_size,
            limit_px: prepared
                .limit_price
                .ok_or_else(|| HubError::BadRequest("limit_price is required".into()))?,
            reduce_only: false,
            cloid: Some(cloid.to_string()),
            order_type: LiveOrderType::LimitIoc,
        }),
        ParsedOrderType::LimitGtc => client.order(OrderRequest {
            symbol: prepared.symbol.clone(),
            is_buy: prepared.is_buy,
            size: prepared.est_size,
            limit_px: prepared
                .limit_price
                .ok_or_else(|| HubError::BadRequest("limit_price is required".into()))?,
            reduce_only: false,
            cloid: Some(cloid.to_string()),
            order_type: LiveOrderType::LimitGtc,
        }),
    }
    .map_err(|error| HubError::BadRequest(error.to_string()))?;

    Ok(response)
}

fn submit_close_order(
    client: &HyperliquidClient,
    prepared: &PreparedClose,
    cloid: &str,
) -> Result<Value, HubError> {
    let response = match prepared.order_type {
        ParsedOrderType::Market => client.market_close(
            &prepared.symbol,
            prepared.is_buy,
            prepared.close_size,
            Some(prepared.mid_price),
            DEFAULT_SLIPPAGE_PCT,
            Some(cloid),
        ),
        ParsedOrderType::LimitIoc => client.order(OrderRequest {
            symbol: prepared.symbol.clone(),
            is_buy: prepared.is_buy,
            size: prepared.close_size,
            limit_px: prepared
                .limit_price
                .ok_or_else(|| HubError::BadRequest("limit_price is required".into()))?,
            reduce_only: true,
            cloid: Some(cloid.to_string()),
            order_type: LiveOrderType::LimitIoc,
        }),
        ParsedOrderType::LimitGtc => client.order(OrderRequest {
            symbol: prepared.symbol.clone(),
            is_buy: prepared.is_buy,
            size: prepared.close_size,
            limit_px: prepared
                .limit_price
                .ok_or_else(|| HubError::BadRequest("limit_price is required".into()))?,
            reduce_only: true,
            cloid: Some(cloid.to_string()),
            order_type: LiveOrderType::LimitGtc,
        }),
    }
    .map_err(|error| HubError::BadRequest(error.to_string()))?;

    Ok(response)
}

fn fill_matches_manual_intent(
    fill: &HyperliquidFill,
    symbol_upper: &str,
    cloid: &str,
    exchange_order_id: Option<&str>,
) -> bool {
    let raw_symbol = fill
        .raw
        .get("coin")
        .or_else(|| fill.raw.get("symbol"))
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_ascii_uppercase());
    if raw_symbol.as_deref() != Some(symbol_upper) {
        return false;
    }
    let fill_cloid = fill
        .raw
        .get("cloid")
        .and_then(|value| value.as_str())
        .map(str::trim);
    if fill_cloid == Some(cloid) {
        return true;
    }
    if let Some(expected_oid) = exchange_order_id {
        let fill_oid = fill
            .raw
            .get("oid")
            .map(|value| value.to_string().trim_matches('"').to_string());
        return fill_oid.as_deref() == Some(expected_oid);
    }
    false
}

fn collect_matching_fills_once(
    client: &HyperliquidClient,
    symbol: &str,
    start_ms: i64,
    end_ms: i64,
    cloid: &str,
    exchange_order_id: Option<&str>,
) -> Result<Vec<HyperliquidFill>, HubError> {
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let fills = client
        .user_fills_by_time(start_ms, end_ms)
        .map_err(|error| HubError::Internal(format!("failed to poll fills: {error}")))?;
    let mut matched = Vec::<HyperliquidFill>::new();
    for fill in fills {
        if fill_matches_manual_intent(&fill, &symbol_upper, cloid, exchange_order_id) {
            push_unique_fill(&mut matched, fill);
        }
    }
    Ok(matched)
}

fn poll_fill(
    client: &HyperliquidClient,
    symbol: &str,
    start_ms: i64,
    cloid: &str,
    exchange_order_id: Option<&str>,
) -> Result<Vec<HyperliquidFill>, HubError> {
    let deadline = Instant::now() + FILL_POLL_TIMEOUT;
    let mut grace_deadline: Option<Instant> = None;
    let mut collected = Vec::<HyperliquidFill>::new();
    while Instant::now() < deadline {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let fills = collect_matching_fills_once(
            client,
            symbol,
            start_ms,
            now_ms,
            cloid,
            exchange_order_id,
        )?;
        let before = collected.len();
        for fill in fills {
            push_unique_fill(&mut collected, fill);
        }
        if !collected.is_empty() {
            if collected.len() > before {
                grace_deadline = Some(Instant::now() + Duration::from_millis(1500));
            }
            if let Some(grace_deadline) = grace_deadline {
                if Instant::now() >= grace_deadline {
                    break;
                }
            }
        }
        std::thread::sleep(FILL_POLL_INTERVAL);
    }
    Ok(collected)
}

struct ParsedFill {
    timestamp: String,
    symbol: String,
    pos_type: String,
    action: String,
    side: String,
    price: f64,
    size: f64,
    notional_usd: f64,
    pnl_usd: f64,
    fee_usd: f64,
    fee_token: Option<String>,
    fee_rate: f64,
    fill_hash: Option<String>,
    fill_tid: Option<i64>,
    exchange_order_id: Option<String>,
    reason: String,
}

fn parse_fill(raw: &Value) -> Option<ParsedFill> {
    let symbol = raw
        .get("coin")
        .or_else(|| raw.get("symbol"))
        .and_then(|value| value.as_str())?
        .trim()
        .to_ascii_uppercase();
    let price = raw.get("px").and_then(parse_json_number)?;
    let size = raw.get("sz").and_then(parse_json_number)?;
    if price <= 0.0 || size <= 0.0 {
        return None;
    }
    let ts_ms = raw
        .get("time")
        .or_else(|| raw.get("timestamp"))
        .and_then(|value| value.as_i64())
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    let timestamp = chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
    let dir = raw
        .get("dir")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let start_position = raw
        .get("startPosition")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let (action, pos_type) = classify_fill_direction(&dir, start_position, size)?;
    let side = raw
        .get("side")
        .and_then(|value| value.as_str())
        .map(|value| {
            if value.eq_ignore_ascii_case("B") {
                "BUY".to_string()
            } else {
                "SELL".to_string()
            }
        })
        .unwrap_or_else(|| {
            if pos_type == "LONG" {
                if action == "OPEN" || action == "ADD" {
                    "BUY".to_string()
                } else {
                    "SELL".to_string()
                }
            } else if action == "OPEN" || action == "ADD" {
                "SELL".to_string()
            } else {
                "BUY".to_string()
            }
        });
    let fee_usd = raw.get("fee").and_then(parse_json_number).unwrap_or(0.0);
    let notional_usd = price * size;
    Some(ParsedFill {
        timestamp,
        symbol,
        pos_type: pos_type.to_string(),
        action: action.to_string(),
        side,
        price,
        size,
        notional_usd,
        pnl_usd: raw
            .get("closedPnl")
            .and_then(parse_json_number)
            .unwrap_or(0.0),
        fee_usd,
        fee_token: raw
            .get("feeToken")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        fee_rate: if notional_usd > 0.0 {
            fee_usd / notional_usd
        } else {
            0.0
        },
        fill_hash: raw
            .get("hash")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        fill_tid: raw.get("tid").and_then(|value| value.as_i64()),
        exchange_order_id: raw
            .get("oid")
            .map(|value| value.to_string().trim_matches('"').to_string())
            .filter(|value| !value.is_empty()),
        reason: "manual_trade".to_string(),
    })
}

fn classify_fill_direction(
    dir: &str,
    start_position: f64,
    fill_size: f64,
) -> Option<(&'static str, &'static str)> {
    let pos_type = if dir.contains("long") {
        "LONG"
    } else if dir.contains("short") {
        "SHORT"
    } else {
        return None;
    };
    if dir.starts_with("open") {
        let action = if start_position.abs() < 1e-9 {
            "OPEN"
        } else {
            "ADD"
        };
        return Some((action, pos_type));
    }
    if dir.starts_with("close") {
        let ending_position = if pos_type == "LONG" {
            start_position - fill_size
        } else {
            start_position + fill_size
        };
        let action = if ending_position.abs() < 1e-9 {
            "CLOSE"
        } else {
            "REDUCE"
        };
        return Some((action, pos_type));
    }
    None
}

fn parse_order_type(raw: &str) -> Result<ParsedOrderType, HubError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "market" => Ok(ParsedOrderType::Market),
        "limit_ioc" => Ok(ParsedOrderType::LimitIoc),
        "limit_gtc" => Ok(ParsedOrderType::LimitGtc),
        _ => Err(HubError::BadRequest(format!(
            "unsupported order_type: {raw}"
        ))),
    }
}

fn parse_json_number(value: &Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    value.as_str()?.parse::<f64>().ok()
}

fn parse_json_integer(value: &Value) -> Option<i64> {
    if let Some(number) = value.as_i64() {
        return Some(number);
    }
    value.as_str()?.trim().parse::<i64>().ok()
}

fn fill_timestamp_ms(raw: &Value) -> i64 {
    raw.get("time")
        .or_else(|| raw.get("timestamp"))
        .and_then(parse_json_integer)
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis())
}

fn round_size_down(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).floor() / factor
}

fn round_size_up(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).ceil() / factor
}

fn push_unique_fill(collected: &mut Vec<HyperliquidFill>, fill: HyperliquidFill) {
    let fill_hash = fill
        .raw
        .get("hash")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let fill_tid = fill.raw.get("tid").and_then(|value| value.as_i64());
    let already_present = collected.iter().any(|existing| {
        existing
            .raw
            .get("hash")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned)
            == fill_hash
            && existing.raw.get("tid").and_then(|value| value.as_i64()) == fill_tid
    });
    if !already_present {
        collected.push(fill);
    }
}

fn manual_cloid_from_intent_id(intent_id: &str) -> String {
    let digest = sha3::Sha3_256::digest(intent_id.as_bytes());
    let mut bytes = MANUAL_CLOID_PREFIX.to_vec();
    bytes.extend_from_slice(&digest[..(16 - MANUAL_CLOID_PREFIX.len())]);
    format!("0x{}", hex::encode(bytes))
}

fn manual_intent_id_from_confirm_token(confirm_token: &str) -> String {
    let digest = sha3::Sha3_256::digest(confirm_token.as_bytes());
    format!("manual_{}", hex::encode(&digest[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_trades_table(conn: &Connection) {
        conn.execute_batch(
            "
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                type TEXT,
                action TEXT,
                price REAL,
                size REAL,
                notional REAL,
                reason TEXT,
                confidence TEXT,
                pnl REAL,
                fee_usd REAL,
                fee_token TEXT,
                fee_rate REAL,
                balance REAL,
                entry_atr REAL,
                leverage REAL,
                margin_used REAL,
                meta_json TEXT,
                fill_hash TEXT,
                fill_tid INTEGER,
                run_fingerprint TEXT,
                reason_code TEXT
            );
            CREATE UNIQUE INDEX idx_trades_fill_hash_tid ON trades(fill_hash, fill_tid);
            ",
        )
        .unwrap();
    }

    #[test]
    fn manual_db_hardening_persists_intent_submission_fill_and_log() {
        let db = NamedTempFile::new().unwrap();
        let bootstrap = Connection::open(db.path()).unwrap();
        create_trades_table(&bootstrap);
        drop(bootstrap);

        let mut conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_test_intent";
        let cloid = manual_cloid_from_intent_id(intent_id);

        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_417_718_266,
                symbol: "HYPE",
                action: "REDUCE",
                side: "BUY",
                requested_size: 2.82,
                requested_notional: 102.39138,
                leverage: Some(4.0),
                status: "NEW",
                dedupe_key: None,
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        record_manual_order_submission(
            &conn,
            ManualOrderSubmission {
                intent_id,
                sent_ts_ms: 1_773_417_718_266,
                symbol: "HYPE",
                side: "BUY",
                order_type: "market_close",
                requested_size: 2.82,
                reduce_only: true,
                client_order_id: &cloid,
                exchange_order_id: Some("347951877775"),
                status: "SENT",
                last_error: "",
                raw_json: json!({"status":"ok"}).to_string(),
            },
        )
        .unwrap();
        let fills = vec![HyperliquidFill {
            raw: json!({
                "coin": "HYPE",
                "px": "36.309",
                "sz": "2.82",
                "side": "B",
                "time": 1773417718266_i64,
                "startPosition": "-8.31",
                "dir": "Close Short",
                "closedPnl": "-1.606554",
                "hash": "0xdef7b1fb28bbd467e0710436fd3db102036600e0c3bef33982c05d4de7bfae52",
                "oid": 347951877775_i64,
                "crossed": true,
                "fee": "0.044233",
                "tid": 1104752028053608_i64,
                "feeToken": "USDC"
            }),
        }];
        let summary =
            record_manual_fill_batch(&mut conn, &fills, "HYPE", intent_id, &cloid, 223.84, 4.0)
                .unwrap();
        assert_eq!(summary.observed_fills, 1);
        assert_eq!(summary.parsed_trade_rows, 1);
        assert_eq!(summary.parse_failures, 0);
        update_manual_intent_status(&conn, intent_id, "FILLED", "").unwrap();
        write_manual_runtime_log(&conn, "INFO", "manual_trade fills_recorded").unwrap();

        let intent_status: String = conn
            .query_row(
                "SELECT status FROM oms_intents WHERE intent_id = ?1",
                [intent_id],
                |row| row.get(0),
            )
            .unwrap();
        let oms_fills_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_fills", [], |row| row.get(0))
            .unwrap();
        let trades_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        let orders_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_orders", [], |row| row.get(0))
            .unwrap();
        let runtime_logs_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_logs", [], |row| row.get(0))
            .unwrap();

        assert_eq!(intent_status, "FILLED");
        assert_eq!(orders_count, 1);
        assert_eq!(oms_fills_count, 1);
        assert_eq!(trades_count, 1);
        assert_eq!(runtime_logs_count, 1);
    }

    #[test]
    fn manual_db_hardening_keeps_raw_oms_fill_when_trade_parse_fails() {
        let db = NamedTempFile::new().unwrap();
        let bootstrap = Connection::open(db.path()).unwrap();
        create_trades_table(&bootstrap);
        drop(bootstrap);

        let mut conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_test_parse_fail";
        let cloid = manual_cloid_from_intent_id(intent_id);
        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_439_518_520,
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: 0.0515,
                requested_notional: 107.5629,
                leverage: Some(4.0),
                status: "NEW",
                dedupe_key: None,
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        let fills = vec![HyperliquidFill {
            raw: json!({
                "coin": "ETH",
                "px": "2088.6",
                "sz": "0.0515",
                "side": "A",
                "time": 1773439518520_i64,
                "hash": "0x174eaf57fd1197c218c80437013d980206d9003d9814b694bb175aaabc1571ac",
                "oid": 348262211344_i64,
                "fee": "0.046467",
                "tid": 186240033668508_i64,
                "feeToken": "USDC"
            }),
        }];
        let summary =
            record_manual_fill_batch(&mut conn, &fills, "ETH", intent_id, &cloid, 214.97, 4.0)
                .unwrap();
        assert_eq!(summary.parsed_trade_rows, 0);
        assert_eq!(summary.parse_failures, 1);
        write_manual_runtime_log(&conn, "WARN", "manual_trade fill_parse_failed").unwrap();

        let oms_fills_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_fills", [], |row| row.get(0))
            .unwrap();
        let trades_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        let runtime_logs_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_logs", [], |row| row.get(0))
            .unwrap();

        assert_eq!(oms_fills_count, 1);
        assert_eq!(trades_count, 0);
        assert_eq!(runtime_logs_count, 1);
    }

    #[test]
    fn manual_db_hardening_keeps_rejected_order_raw_json() {
        let db = NamedTempFile::new().unwrap();
        let bootstrap = Connection::open(db.path()).unwrap();
        create_trades_table(&bootstrap);
        drop(bootstrap);

        let conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_test_reject";
        let cloid = manual_cloid_from_intent_id(intent_id);
        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_439_600_000,
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: 0.0515,
                requested_notional: 107.5629,
                leverage: Some(10.0),
                status: "NEW",
                dedupe_key: None,
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        let rejected_raw = json!({
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        { "error": "Order has invalid price." }
                    ]
                }
            }
        })
        .to_string();
        record_manual_order_submission(
            &conn,
            ManualOrderSubmission {
                intent_id,
                sent_ts_ms: 1_773_439_600_123,
                symbol: "ETH",
                side: "SELL",
                order_type: "market_open",
                requested_size: 0.0515,
                reduce_only: false,
                client_order_id: &cloid,
                exchange_order_id: None,
                status: "REJECTED",
                last_error: &rejected_raw,
                raw_json: rejected_raw.clone(),
            },
        )
        .unwrap();

        let status: String = conn
            .query_row(
                "SELECT status FROM oms_intents WHERE intent_id = ?1",
                [intent_id],
                |row| row.get(0),
            )
            .unwrap();
        let raw_json: String = conn
            .query_row(
                "SELECT raw_json FROM oms_orders WHERE intent_id = ?1",
                [intent_id],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(status, "REJECTED");
        assert!(raw_json.contains("Order has invalid price."));
    }

    #[test]
    fn manual_confirmations_reuse_preview_token_until_bound() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let now_ms = chrono::Utc::now().timestamp_millis();
        conn.execute(
            "INSERT INTO manual_trade_confirmations (
                confirm_token, created_ts_ms, expires_ts_ms, action, symbol, param_hash, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'PREVIEWED')",
            params![
                "confirm-preview-token",
                now_ms,
                now_ms + MANUAL_CONFIRM_TTL_MS,
                MANUAL_CONFIRM_ACTION_OPEN,
                "ETH",
                "hash_open",
            ],
        )
        .unwrap();

        let token = find_reusable_manual_confirmation(
            &conn,
            MANUAL_CONFIRM_ACTION_OPEN,
            "hash_open",
            now_ms,
        )
        .unwrap();
        assert_eq!(token.as_deref(), Some("confirm-preview-token"));

        bind_manual_confirmation_to_intent(&conn, "confirm-preview-token", "manual_bound").unwrap();

        let token_after_bind = find_reusable_manual_confirmation(
            &conn,
            MANUAL_CONFIRM_ACTION_OPEN,
            "hash_open",
            now_ms,
        )
        .unwrap();
        assert!(token_after_bind.is_none());
    }

    #[test]
    fn manual_execution_claim_reuses_existing_intent_for_same_confirm_token() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let now_ms = chrono::Utc::now().timestamp_millis();
        let confirm_token = "confirm-execute-token";
        let param_hash = "open_hash";
        conn.execute(
            "INSERT INTO manual_trade_confirmations (
                confirm_token, created_ts_ms, expires_ts_ms, action, symbol, param_hash, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'PREVIEWED')",
            params![
                confirm_token,
                now_ms,
                now_ms + MANUAL_CONFIRM_TTL_MS,
                MANUAL_CONFIRM_ACTION_OPEN,
                "ETH",
                param_hash,
            ],
        )
        .unwrap();

        let intent_id = manual_intent_id_from_confirm_token(confirm_token);
        let cloid = manual_cloid_from_intent_id(&intent_id);
        let record = ManualIntentRecord {
            intent_id: &intent_id,
            created_ts_ms: now_ms,
            symbol: "ETH",
            action: "OPEN",
            side: "SELL",
            requested_size: 0.0515,
            requested_notional: 107.5629,
            leverage: Some(10.0),
            status: "NEW",
            dedupe_key: Some(confirm_token),
            client_order_id: &cloid,
            reason: "manual_trade",
            confidence: "MANUAL",
            meta_json: "{}".to_string(),
        };
        let first = initialise_manual_execution(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_OPEN,
            param_hash,
            record,
        )
        .unwrap();
        match first {
            ManualExecutionClaim::Submit {
                intent_id: first_intent_id,
                resumed_existing,
                ..
            } => {
                assert_eq!(first_intent_id, intent_id);
                assert!(!resumed_existing);
            }
            ManualExecutionClaim::Existing(_) => panic!("expected new intent claim"),
        }

        let duplicate = initialise_manual_execution(
            &conn,
            confirm_token,
            MANUAL_CONFIRM_ACTION_OPEN,
            param_hash,
            ManualIntentRecord {
                intent_id: &intent_id,
                created_ts_ms: now_ms,
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: 0.0515,
                requested_notional: 107.5629,
                leverage: Some(10.0),
                status: "NEW",
                dedupe_key: Some(confirm_token),
                client_order_id: &cloid,
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        match duplicate {
            ManualExecutionClaim::Existing(existing) => {
                assert_eq!(existing.intent_id, intent_id);
                assert_eq!(existing.status, "NEW");
            }
            ManualExecutionClaim::Submit { .. } => panic!("expected duplicate to reuse intent"),
        }

        let intent_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_intents", [], |row| row.get(0))
            .unwrap();
        let confirmation_status: String = conn
            .query_row(
                "SELECT status FROM manual_trade_confirmations WHERE confirm_token = ?1",
                [confirm_token],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(intent_count, 1);
        assert_eq!(confirmation_status, "BOUND");
    }

    #[test]
    fn resumed_open_retry_rejects_changed_request_body() {
        let existing = ExistingManualIntent {
            intent_id: "manual_resume".to_string(),
            created_ts_ms: 1_773_500_000_000,
            symbol: "ETH".to_string(),
            action: "OPEN".to_string(),
            side: "SELL".to_string(),
            status: "NEW".to_string(),
            requested_size: Some(0.0515),
            leverage: Some(10.0),
            dedupe_key: Some("confirm-open".to_string()),
            client_order_id: Some("0x6d616e5f1234567890abcdef12345678".to_string()),
            exchange_order_id: None,
            last_error: None,
            sent_ts_ms: None,
            meta_json: Some(
                json!({
                    "request": {
                        "order_type": "market",
                        "side": "SELL",
                        "notional_usd": 500.0,
                        "leverage": 10,
                        "limit_price": Value::Null
                    }
                })
                .to_string(),
            ),
        };
        let changed_request = ManualTradeOpenRequest {
            symbol: "ETH".to_string(),
            side: "SELL".to_string(),
            notional_usd: 750.0,
            leverage: 10,
            order_type: "market".to_string(),
            limit_price: None,
        };

        let error = validate_resumed_open_request(&existing, &changed_request).unwrap_err();
        assert!(error
            .to_string()
            .contains("confirm token does not match parameters"));
    }

    #[test]
    fn manual_cancel_audit_updates_intent_and_persists_order_row() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_cancel_ok";
        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_500_100_000,
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: 0.0515,
                requested_notional: 107.5629,
                leverage: Some(10.0),
                status: "SENT",
                dedupe_key: None,
                client_order_id: "0x6d616e5f1234567890abcdef12345678",
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        let context = load_manual_cancel_context_by_intent_id(&conn, intent_id)
            .unwrap()
            .expect("cancel context should exist");
        record_manual_cancel_audit(
            &conn,
            Some(&context),
            ManualCancelAudit {
                intent_id: Some(intent_id),
                symbol: "ETH",
                side: Some("SELL"),
                order_type: "cancel_by_cloid",
                requested_size: Some(0.0515),
                reduce_only: Some(false),
                client_order_id: Some("0x6d616e5f1234567890abcdef12345678"),
                exchange_order_id: Some("348262211344"),
                status: "CANCELLED",
                raw_json: json!({"status":"ok"}).to_string(),
            },
        )
        .unwrap();

        let status: String = conn
            .query_row(
                "SELECT status FROM oms_intents WHERE intent_id = ?1",
                [intent_id],
                |row| row.get(0),
            )
            .unwrap();
        let order_status: String = conn
            .query_row(
                "SELECT status FROM oms_orders WHERE intent_id = ?1 ORDER BY id DESC LIMIT 1",
                [intent_id],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(status, "CANCELLED");
        assert_eq!(order_status, "CANCELLED");
    }

    #[test]
    fn manual_cancel_audit_preserves_status_on_reject() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_cancel_reject";
        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_500_200_000,
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: 0.0515,
                requested_notional: 107.5629,
                leverage: Some(10.0),
                status: "SENT",
                dedupe_key: None,
                client_order_id: "0x6d616e5f1234567890abcdef12345678",
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();
        let context = load_manual_cancel_context_by_intent_id(&conn, intent_id)
            .unwrap()
            .expect("cancel context should exist");
        record_manual_cancel_audit(
            &conn,
            Some(&context),
            ManualCancelAudit {
                intent_id: Some(intent_id),
                symbol: "ETH",
                side: Some("SELL"),
                order_type: "cancel_by_cloid",
                requested_size: Some(0.0515),
                reduce_only: Some(false),
                client_order_id: Some("0x6d616e5f1234567890abcdef12345678"),
                exchange_order_id: Some("348262211344"),
                status: "REJECTED",
                raw_json: "already gone".to_string(),
            },
        )
        .unwrap();

        let row: (String, String) = conn
            .query_row(
                "SELECT status, last_error FROM oms_intents WHERE intent_id = ?1",
                [intent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(row.0, "SENT");
        assert!(row.1.contains("already gone"));
    }

    #[test]
    fn cancel_context_by_intent_id_uses_persisted_symbol() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_cancel_symbol";
        insert_manual_intent(
            &conn,
            ManualIntentRecord {
                intent_id,
                created_ts_ms: 1_773_500_300_000,
                symbol: "HYPE",
                action: "OPEN",
                side: "BUY",
                requested_size: 2.82,
                requested_notional: 102.39138,
                leverage: Some(4.0),
                status: "SENT",
                dedupe_key: None,
                client_order_id: "0x6d616e5f1234567890abcdef12345678",
                reason: "manual_trade",
                confidence: "MANUAL",
                meta_json: "{}".to_string(),
            },
        )
        .unwrap();

        let context = load_manual_cancel_context_by_intent_id(&conn, intent_id)
            .unwrap()
            .expect("cancel context should exist");
        assert_eq!(context.symbol, "HYPE");
    }

    #[test]
    fn manual_cancel_audit_allows_missing_intent_fallback_on_reject() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let fallback = ManualCancelContext {
            intent_exists: false,
            intent_id: Some("missing_intent".to_string()),
            symbol: "ETH".to_string(),
            side: None,
            current_status: None,
            requested_size: None,
            reduce_only: None,
            client_order_id: Some("0x6d616e5f1234567890abcdef12345678".to_string()),
            exchange_order_id: None,
        };

        record_manual_cancel_audit(
            &conn,
            Some(&fallback),
            ManualCancelAudit {
                intent_id: fallback.intent_id.as_deref(),
                symbol: "ETH",
                side: None,
                order_type: "cancel_by_cloid",
                requested_size: None,
                reduce_only: None,
                client_order_id: fallback.client_order_id.as_deref(),
                exchange_order_id: None,
                status: "REJECTED",
                raw_json: "missing order".to_string(),
            },
        )
        .unwrap();

        let order_status: String = conn
            .query_row(
                "SELECT status FROM oms_orders ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(order_status, "REJECTED");
    }

    #[test]
    fn manual_audit_event_helper_persists_structured_row() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        write_manual_audit_event(
            &conn,
            Some("ETH"),
            "manual_test_event",
            "info",
            json!({ "intent_id": "manual_test", "ok": true }),
        )
        .unwrap();

        let row: (String, String, String) = conn
            .query_row(
                "SELECT symbol, event, level FROM audit_events ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row.0, "ETH");
        assert_eq!(row.1, "MANUAL_TEST_EVENT");
        assert_eq!(row.2, "INFO");
    }

    #[test]
    fn manual_reconcile_event_helper_persists_structured_row() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        write_manual_reconcile_event(
            &conn,
            "manual_unknown_recovery",
            Some("ETH"),
            "fills_recovered",
            json!({ "intent_id": "manual_test", "status": "FILLED" }),
        )
        .unwrap();

        let row: (String, String, String) = conn
            .query_row(
                "SELECT kind, symbol, result FROM oms_reconcile_events ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row.0, "manual_unknown_recovery");
        assert_eq!(row.1, "ETH");
        assert_eq!(row.2, "fills_recovered");
    }

    #[test]
    fn manual_guardrail_block_persists_runtime_and_audit_rows() {
        let db = NamedTempFile::new().unwrap();
        let cfg = HubConfig {
            live_db: db.path().to_path_buf(),
            ..HubConfig::from_env()
        };
        record_manual_guardrail_block(&cfg, "ETH", "OPEN", "close_only").unwrap();

        let conn = open_manual_trade_db(db.path()).unwrap();
        let runtime_row: (String, String) = conn
            .query_row(
                "SELECT level, message FROM runtime_logs ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let audit_row: (String, String) = conn
            .query_row(
                "SELECT event, symbol FROM audit_events ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(runtime_row.0, "WARN");
        assert!(runtime_row.1.contains("close_only"));
        assert_eq!(audit_row.0, "MANUAL_GUARDRAIL_BLOCKED");
        assert_eq!(audit_row.1, "ETH");
    }

    #[test]
    fn manual_fill_batch_counts_only_new_rows_for_duplicates() {
        let db = NamedTempFile::new().unwrap();
        let bootstrap = Connection::open(db.path()).unwrap();
        create_trades_table(&bootstrap);
        drop(bootstrap);

        let mut conn = open_manual_trade_db(db.path()).unwrap();
        let intent_id = "manual_duplicate_fill";
        let cloid = manual_cloid_from_intent_id(intent_id);
        let fill = HyperliquidFill {
            raw: json!({
                "coin": "HYPE",
                "px": "36.309",
                "sz": "2.82",
                "side": "B",
                "time": 1773417718266_i64,
                "startPosition": "-8.31",
                "dir": "Close Short",
                "closedPnl": "-1.606554",
                "hash": "0xdupfill",
                "oid": 347951877775_i64,
                "crossed": true,
                "fee": "0.044233",
                "tid": 1104752028053608_i64,
                "feeToken": "USDC"
            }),
        };

        let first = record_manual_fill_batch(
            &mut conn,
            &[fill.clone()],
            "HYPE",
            intent_id,
            &cloid,
            223.84,
            4.0,
        )
        .unwrap();
        let second =
            record_manual_fill_batch(&mut conn, &[fill], "HYPE", intent_id, &cloid, 223.84, 4.0)
                .unwrap();

        assert_eq!(first.inserted_oms_fills, 1);
        assert_eq!(first.parsed_trade_rows, 1);
        assert_eq!(second.inserted_oms_fills, 0);
        assert_eq!(second.parsed_trade_rows, 0);
        assert_eq!(second.filled_size, 2.82);
    }

    #[test]
    fn find_matching_open_order_matches_exchange_order_id_without_cloid() {
        let existing = ExistingManualIntent {
            intent_id: "manual_existing".to_string(),
            created_ts_ms: 1_773_500_000_000,
            symbol: "ETH".to_string(),
            action: "OPEN".to_string(),
            side: "SELL".to_string(),
            status: "SENT".to_string(),
            requested_size: Some(0.0515),
            leverage: Some(10.0),
            dedupe_key: Some("confirm-open".to_string()),
            client_order_id: Some("0x6d616e5f1234567890abcdef12345678".to_string()),
            exchange_order_id: Some("348262211344".to_string()),
            last_error: None,
            sent_ts_ms: Some(1_773_500_000_100),
            meta_json: Some("{}".to_string()),
        };
        let orders = vec![json!({
            "coin": "ETH",
            "oid": 348262211344_i64,
            "side": "A",
            "limitPx": "2100.5",
            "sz": "0.0515"
        })];

        let matched = find_matching_open_order(&orders, &existing, "0xdeadbeef");
        assert!(matched.is_some());
    }

    #[test]
    fn manual_open_order_snapshot_upserts_and_clears() {
        let db = NamedTempFile::new().unwrap();
        let conn = open_manual_trade_db(db.path()).unwrap();
        let changed = upsert_manual_open_order_snapshot(
            &conn,
            ManualOpenOrderSnapshot {
                last_seen_ts_ms: 1_773_500_000_000,
                symbol: "ETH",
                side: "SELL",
                price: Some(2100.5),
                remaining_size: Some(0.0515),
                reduce_only: false,
                client_order_id: "0x6d616e5f1234567890abcdef12345678",
                exchange_order_id: Some("348262211344"),
                intent_id: "manual_snapshot",
                raw_json: json!({"coin":"ETH","oid":348262211344_i64}).to_string(),
            },
        )
        .unwrap();
        assert!(changed);

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_open_orders", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let cleared = clear_manual_open_order_snapshot(
            &conn,
            "manual_snapshot",
            "0x6d616e5f1234567890abcdef12345678",
        )
        .unwrap();
        assert!(cleared);

        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_open_orders", [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 0);
    }

    #[test]
    fn desired_manual_reconcile_state_marks_partial_cancelled_without_open_order() {
        let existing = ExistingManualIntent {
            intent_id: "manual_partial".to_string(),
            created_ts_ms: 1_773_500_000_000,
            symbol: "ETH".to_string(),
            action: "OPEN".to_string(),
            side: "SELL".to_string(),
            status: "SENT".to_string(),
            requested_size: Some(2.0),
            leverage: Some(4.0),
            dedupe_key: None,
            client_order_id: None,
            exchange_order_id: None,
            last_error: None,
            sent_ts_ms: Some(1_773_500_000_100),
            meta_json: Some("{}".to_string()),
        };
        let summary = FillWriteSummary {
            observed_fills: 1,
            inserted_oms_fills: 1,
            parsed_trade_rows: 1,
            parse_failures: 0,
            filled_size: 0.75,
        };

        let state = desired_manual_reconcile_state(
            &existing,
            &summary,
            false,
            existing.created_ts_ms + MANUAL_RECONCILE_MISSING_GRACE_MS + 1,
        );
        assert_eq!(state.0, "PARTIAL_CANCELLED");
    }

    #[test]
    fn desired_manual_reconcile_state_marks_missing_sent_intent_unknown_after_grace() {
        let existing = ExistingManualIntent {
            intent_id: "manual_missing".to_string(),
            created_ts_ms: 1_773_500_000_000,
            symbol: "ETH".to_string(),
            action: "OPEN".to_string(),
            side: "SELL".to_string(),
            status: "SENT".to_string(),
            requested_size: Some(2.0),
            leverage: Some(4.0),
            dedupe_key: None,
            client_order_id: None,
            exchange_order_id: None,
            last_error: None,
            sent_ts_ms: Some(1_773_500_000_100),
            meta_json: Some("{}".to_string()),
        };
        let summary = FillWriteSummary {
            observed_fills: 0,
            inserted_oms_fills: 0,
            parsed_trade_rows: 0,
            parse_failures: 0,
            filled_size: 0.0,
        };

        let state = desired_manual_reconcile_state(
            &existing,
            &summary,
            false,
            existing.created_ts_ms + MANUAL_RECONCILE_MISSING_GRACE_MS + 1,
        );
        assert_eq!(state.0, "UNKNOWN");
        assert!(state.1.contains("missing from exchange"));
    }

    #[test]
    fn manual_audit_event_helper_supports_old_and_new_schemas() {
        let old_db = NamedTempFile::new().unwrap();
        let old_conn = Connection::open(old_db.path()).unwrap();
        old_conn
            .execute_batch(
                "
                CREATE TABLE audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    event TEXT NOT NULL,
                    level TEXT,
                    data_json TEXT
                );
                ",
            )
            .unwrap();
        write_manual_audit_event(
            &old_conn,
            Some("ETH"),
            "manual_test_event",
            "info",
            json!({ "ok": true }),
        )
        .unwrap();
        let old_count: i64 = old_conn
            .query_row("SELECT COUNT(*) FROM audit_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(old_count, 1);

        let new_db = NamedTempFile::new().unwrap();
        let new_conn = Connection::open(new_db.path()).unwrap();
        new_conn
            .execute_batch(
                "
                CREATE TABLE audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    event TEXT NOT NULL,
                    level TEXT,
                    data_json TEXT
                );
                ",
            )
            .unwrap();
        write_manual_audit_event(
            &new_conn,
            Some("ETH"),
            "manual_test_event",
            "info",
            json!({ "ok": true }),
        )
        .unwrap();
        let new_count: i64 = new_conn
            .query_row("SELECT COUNT(*) FROM audit_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(new_count, 1);
    }

    #[test]
    fn manual_reconcile_event_helper_supports_old_and_new_schemas() {
        let old_db = NamedTempFile::new().unwrap();
        let old_conn = Connection::open(old_db.path()).unwrap();
        old_conn
            .execute_batch(
                "
                CREATE TABLE oms_reconcile_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER,
                    kind TEXT,
                    symbol TEXT,
                    intent_id TEXT,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    result TEXT,
                    detail_json TEXT
                );
                ",
            )
            .unwrap();
        write_manual_reconcile_event(
            &old_conn,
            "manual_unknown_recovery",
            Some("ETH"),
            "fills_recovered",
            json!({ "ok": true }),
        )
        .unwrap();
        let old_count: i64 = old_conn
            .query_row("SELECT COUNT(*) FROM oms_reconcile_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(old_count, 1);

        let new_db = NamedTempFile::new().unwrap();
        let new_conn = Connection::open(new_db.path()).unwrap();
        new_conn
            .execute_batch(
                "
                CREATE TABLE oms_reconcile_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    symbol TEXT,
                    result TEXT,
                    data_json TEXT
                );
                ",
            )
            .unwrap();
        write_manual_reconcile_event(
            &new_conn,
            "manual_unknown_recovery",
            Some("ETH"),
            "fills_recovered",
            json!({ "ok": true }),
        )
        .unwrap();
        let new_count: i64 = new_conn
            .query_row("SELECT COUNT(*) FROM oms_reconcile_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(new_count, 1);
    }
}
