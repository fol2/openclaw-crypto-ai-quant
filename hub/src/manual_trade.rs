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
use std::time::{Duration, Instant};

const DEFAULT_SLIPPAGE_PCT: f64 = 0.01;
const HL_TAKER_FEE_RATE: f64 = 0.00035;
const DEFAULT_MAX_NOTIONAL_USD: f64 = 5_000.0;
const MIN_NOTIONAL_USD: f64 = 10.0;
const FILL_POLL_TIMEOUT: Duration = Duration::from_secs(8);
const FILL_POLL_INTERVAL: Duration = Duration::from_millis(800);
const MANUAL_CLOID_PREFIX: &[u8] = b"man_";

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

pub fn execute_open(cfg: &HubConfig, request: &ManualTradeOpenRequest) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, true)?;
    let client = build_client(cfg)?;
    let prepared = prepare_open(&client, cfg, request)?;
    let intent_id = format!("manual_{}", uuid::Uuid::new_v4().simple());
    let cloid = manual_cloid_from_intent_id(&intent_id);
    let start_ms = chrono::Utc::now().timestamp_millis() - 5_000;

    let response = submit_open_order(&client, &prepared, &cloid)?;
    let exchange_order_id = extract_exchange_order_id(&response);
    let fills = if prepared.order_type == ParsedOrderType::LimitGtc {
        Vec::new()
    } else {
        poll_fill(
            &client,
            &prepared.symbol,
            start_ms,
            &cloid,
            exchange_order_id.as_deref(),
        )?
    };

    for fill in &fills {
        record_fill_trade(
            &cfg.live_db,
            fill,
            &intent_id,
            &cloid,
            prepared.account_value_usd,
            f64::from(prepared.leverage),
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
) -> Result<Value, HubError> {
    enforce_manual_trade_ready(cfg, true)?;
    let client = build_client(cfg)?;
    let prepared = prepare_close(&client, request)?;
    let intent_id = format!("manual_{}", uuid::Uuid::new_v4().simple());
    let cloid = manual_cloid_from_intent_id(&intent_id);
    let start_ms = chrono::Utc::now().timestamp_millis() - 5_000;

    let response = submit_close_order(&client, &prepared, &cloid)?;
    let exchange_order_id = extract_exchange_order_id(&response);
    let fills = if prepared.order_type == ParsedOrderType::LimitGtc {
        Vec::new()
    } else {
        poll_fill(
            &client,
            &prepared.symbol,
            start_ms,
            &cloid,
            exchange_order_id.as_deref(),
        )?
    };

    for fill in &fills {
        record_fill_trade(
            &cfg.live_db,
            fill,
            &intent_id,
            &cloid,
            prepared.account_value_usd,
            prepared.leverage.max(1.0),
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
    let client = build_client(cfg)?;
    let symbol = request.symbol.trim().to_ascii_uppercase();

    if let Some(oid) = request
        .oid
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        let parsed_oid = oid
            .trim()
            .parse::<u64>()
            .map_err(|_| HubError::BadRequest("oid must be a positive integer".into()))?;
        let response = client
            .cancel_order(&symbol, parsed_oid)
            .map_err(|error| HubError::BadRequest(error.to_string()))?;
        if response_has_embedded_error(&response) {
            return Err(HubError::BadRequest(response.to_string()));
        }
        return Ok(json!({
            "ok": true,
            "symbol": symbol,
            "oid": parsed_oid,
            "response": response,
        }));
    }

    if let Some(intent_id) = request
        .intent_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        let cloid = manual_cloid_from_intent_id(intent_id);
        let response = client
            .cancel_order_by_cloid(&symbol, &cloid)
            .map_err(|error| HubError::BadRequest(error.to_string()))?;
        if response_has_embedded_error(&response) {
            return Err(HubError::BadRequest(response.to_string()));
        }
        return Ok(json!({
            "ok": true,
            "symbol": symbol,
            "intent_id": intent_id,
            "cloid": cloid,
            "response": response,
        }));
    }

    Err(HubError::BadRequest(
        "oid or intent_id required for cancel".into(),
    ))
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

    if response_has_embedded_error(&response) {
        return Err(HubError::BadRequest(response.to_string()));
    }
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

    if response_has_embedded_error(&response) {
        return Err(HubError::BadRequest(response.to_string()));
    }
    Ok(response)
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
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let mut collected = Vec::<HyperliquidFill>::new();
    while Instant::now() < deadline {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let fills = client
            .user_fills_by_time(start_ms, now_ms)
            .map_err(|error| HubError::Internal(format!("failed to poll fills: {error}")))?;
        let before = collected.len();
        for fill in fills {
            let raw_symbol = fill
                .raw
                .get("coin")
                .or_else(|| fill.raw.get("symbol"))
                .and_then(|value| value.as_str())
                .map(|value| value.trim().to_ascii_uppercase());
            if raw_symbol.as_deref() != Some(symbol_upper.as_str()) {
                continue;
            }
            let fill_cloid = fill
                .raw
                .get("cloid")
                .and_then(|value| value.as_str())
                .map(str::trim);
            if fill_cloid == Some(cloid) {
                push_unique_fill(&mut collected, fill);
                continue;
            }
            if let Some(expected_oid) = exchange_order_id {
                let fill_oid = fill
                    .raw
                    .get("oid")
                    .map(|value| value.to_string().trim_matches('"').to_string());
                if fill_oid.as_deref() == Some(expected_oid) {
                    push_unique_fill(&mut collected, fill);
                }
            }
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

fn record_fill_trade(
    live_db: &Path,
    fill: &HyperliquidFill,
    intent_id: &str,
    cloid: &str,
    account_value_usd: f64,
    leverage: f64,
) -> Result<(), HubError> {
    let parsed = parse_fill(&fill.raw)
        .ok_or_else(|| HubError::Internal("failed to parse exchange fill".into()))?;
    let meta_json = json!({
        "source": "manual_trade",
        "intent_id": intent_id,
        "cloid": cloid,
        "fill": fill.raw,
        "oms": {
            "intent_id": intent_id,
            "client_order_id": cloid,
            "exchange_order_id": parsed.exchange_order_id,
            "matched_via": "manual_trade",
        }
    })
    .to_string();

    let conn = Connection::open(live_db)?;
    let fill_hash = parsed.fill_hash.as_deref();
    let fill_tid = parsed.fill_tid;
    conn.execute(
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
    Ok(())
}

struct ParsedFill {
    timestamp: String,
    symbol: String,
    pos_type: String,
    action: String,
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
    let fee_usd = raw.get("fee").and_then(parse_json_number).unwrap_or(0.0);
    let notional_usd = price * size;
    Some(ParsedFill {
        timestamp,
        symbol,
        pos_type: pos_type.to_string(),
        action: action.to_string(),
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
