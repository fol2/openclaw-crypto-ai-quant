use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::error::HubError;
use crate::live_risk::{CancelCheck, OrderAction, OrderCheck, OrderRecord};
use crate::live_safety;
use crate::manual_trade;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct TradeOpenBody {
    pub symbol: String,
    pub side: String,
    pub notional_usd: f64,
    pub leverage: u32,
    pub order_type: String,
    pub limit_price: Option<f64>,
    pub confirm_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TradeCloseBody {
    pub symbol: String,
    pub close_pct: f64,
    pub order_type: String,
    pub limit_price: Option<f64>,
    pub confirm_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TradeCancelBody {
    pub symbol: String,
    pub oid: Option<String>,
    pub intent_id: Option<String>,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/trade/enabled", get(trade_enabled))
        .route("/api/trade/preview", post(trade_preview))
        .route("/api/trade/execute", post(trade_execute))
        .route("/api/trade/close", post(trade_close))
        .route("/api/trade/cancel", post(trade_cancel))
        .route("/api/trade/open-orders/{symbol}", get(trade_open_orders))
        .route("/api/trade/{id}/result", get(trade_result))
}

fn require_trade_enabled(state: &AppState) -> Result<(), HubError> {
    if !state.config.manual_trade_enabled {
        return Err(HubError::Forbidden("manual trading is disabled".into()));
    }
    Ok(())
}

fn open_param_fingerprint(body: &TradeOpenBody) -> String {
    manual_trade::open_request_hash(&manual_trade::ManualTradeOpenRequest {
        symbol: body.symbol.clone(),
        side: body.side.clone(),
        notional_usd: body.notional_usd,
        leverage: body.leverage,
        order_type: body.order_type.clone(),
        limit_price: body.limit_price,
    })
}

fn close_param_fingerprint(body: &TradeCloseBody) -> String {
    manual_trade::close_request_hash(&manual_trade::ManualTradeCloseRequest {
        symbol: body.symbol.clone(),
        close_pct: body.close_pct,
        order_type: body.order_type.clone(),
        limit_price: body.limit_price,
    })
}

async fn check_rate_limit(
    state: &AppState,
    symbol: &str,
    confirm_token: Option<&str>,
) -> Result<(), HubError> {
    let symbol = symbol.trim().to_ascii_uppercase();
    let mut limits = state.trade_rate_limits.lock().await;
    limits.retain(|_, (_, created)| created.elapsed().as_secs() < 60);
    if let Some((last_token, last_created)) = limits.get(&symbol) {
        let same_token = confirm_token.is_some() && confirm_token == Some(last_token.as_str());
        if last_created.elapsed().as_secs() < 5 && !same_token {
            return Err(HubError::BadRequest(format!(
                "rate limited: wait before trading {symbol} again"
            )));
        }
    }
    if let Some(token) = confirm_token {
        limits.insert(symbol, (token.to_string(), Instant::now()));
    }
    Ok(())
}

fn manual_risk_block_error(reason: &str) -> HubError {
    match reason {
        "close_only" | "halt_all" => HubError::Forbidden(format!("manual trade blocked: {reason}")),
        _ => HubError::BadRequest(format!("manual trade blocked: {reason}")),
    }
}

async fn record_guardrail_block(state: &AppState, symbol: &str, action: &str, reason: &str) {
    let config = state.config.clone();
    let symbol = symbol.to_string();
    let action = action.to_string();
    let reason = reason.to_string();
    let _ = tokio::task::spawn_blocking(move || {
        manual_trade::record_manual_guardrail_block(&config, &symbol, &action, &reason)
    })
    .await;
}

async fn ensure_live_orders_enabled(
    state: &AppState,
    symbol: &str,
    action: &str,
) -> Result<(), HubError> {
    if live_safety::live_orders_enabled() {
        return Ok(());
    }
    record_guardrail_block(state, symbol, action, "live_orders_disabled").await;
    Err(HubError::Forbidden(
        "manual trade blocked: live orders are disabled".into(),
    ))
}

async fn check_manual_order_risk(
    state: &AppState,
    symbol: &str,
    action: OrderAction,
    reduce_risk: bool,
    audit_action: &str,
) -> Result<(), HubError> {
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let now_ms = chrono::Utc::now().timestamp_millis();
    let decision = {
        let mut risk = state.manual_trade_risk.lock().await;
        risk.refresh(now_ms, None);
        risk.allow_order(OrderCheck {
            now_ms,
            symbol: &symbol_upper,
            action,
            reduce_risk,
        })
    };
    if decision.allowed {
        return Ok(());
    }
    record_guardrail_block(state, &symbol_upper, audit_action, &decision.reason).await;
    Err(manual_risk_block_error(&decision.reason))
}

async fn note_manual_order_sent(
    state: &AppState,
    symbol: &str,
    action: OrderAction,
    reduce_risk: bool,
) {
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let now_ms = chrono::Utc::now().timestamp_millis();
    let mut risk = state.manual_trade_risk.lock().await;
    risk.note_order_sent(OrderRecord {
        now_ms,
        symbol: &symbol_upper,
        action,
        reduce_risk,
    });
}

async fn check_manual_cancel_risk(state: &AppState, symbol: &str) -> Result<(), HubError> {
    let symbol_upper = symbol.trim().to_ascii_uppercase();
    let now_ms = chrono::Utc::now().timestamp_millis();
    let decision = {
        let mut risk = state.manual_trade_risk.lock().await;
        risk.refresh(now_ms, None);
        risk.allow_cancel(CancelCheck {
            now_ms,
            symbol: &symbol_upper,
            exchange_order_id: None,
        })
    };
    if decision.allowed {
        return Ok(());
    }
    record_guardrail_block(state, &symbol_upper, "CANCEL", &decision.reason).await;
    Err(manual_risk_block_error(&decision.reason))
}

async fn trade_enabled(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "enabled": state.config.manual_trade_enabled,
        "mode": if state.config.manual_trade_enabled { "live" } else { "disabled" },
    }))
}

async fn trade_preview(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeOpenBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let request = manual_trade::ManualTradeOpenRequest {
        symbol: body.symbol,
        side: body.side,
        notional_usd: body.notional_usd,
        leverage: body.leverage,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let param_hash = manual_trade::open_request_hash(&request);
    let config = state.config.clone();
    let symbol = request.symbol.clone();
    let (preview, confirm_token) = tokio::task::spawn_blocking(move || {
        let preview = manual_trade::preview_open(&config, &request)?;
        let confirm_token =
            manual_trade::issue_confirm_token(&config, "OPEN", &param_hash, &symbol, &preview)?;
        Ok::<_, HubError>((preview, confirm_token))
    })
    .await
    .map_err(|error| HubError::Internal(format!("preview task failed: {error}")))??;
    let mut payload = preview;
    if let Some(object) = payload.as_object_mut() {
        object.insert("confirm_token".to_string(), Value::String(confirm_token));
    }
    Ok(Json(payload))
}

async fn trade_execute(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeOpenBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let request = manual_trade::ManualTradeOpenRequest {
        symbol: body.symbol,
        side: body.side,
        notional_usd: body.notional_usd,
        leverage: body.leverage,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let token = body
        .confirm_token
        .as_deref()
        .ok_or_else(|| HubError::BadRequest("confirm_token required".into()))?;
    let param_hash = manual_trade::open_request_hash(&request);
    let config = state.config.clone();
    let preflight_request = request.clone();
    let preflight_token = token.to_string();
    let preflight_hash = param_hash.clone();
    let preflight = tokio::task::spawn_blocking(move || {
        manual_trade::preflight_open_execution(
            &config,
            &preflight_request,
            &preflight_token,
            &preflight_hash,
        )
    })
    .await
    .map_err(|error| HubError::Internal(format!("execute preflight task failed: {error}")))??;
    if matches!(
        preflight,
        manual_trade::ManualExecutionPreflight::NewSubmission
    ) {
        ensure_live_orders_enabled(&state, &request.symbol, "OPEN").await?;
        check_manual_order_risk(&state, &request.symbol, OrderAction::Open, false, "OPEN").await?;
        check_rate_limit(&state, &request.symbol, Some(token)).await?;
    }
    let symbol = request.symbol.clone();
    let config = state.config.clone();
    let token = token.to_string();
    let result = tokio::task::spawn_blocking(move || {
        manual_trade::execute_open(&config, &request, &token, &param_hash)
    })
    .await
    .map_err(|error| HubError::Internal(format!("execute task failed: {error}")))??;
    if result.get("deduped").and_then(|value| value.as_bool()) != Some(true) {
        note_manual_order_sent(&state, &symbol, OrderAction::Open, false).await;
    }
    Ok(Json(result))
}

async fn trade_close(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeCloseBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    if body.confirm_token.is_none() {
        let request = manual_trade::ManualTradeCloseRequest {
            symbol: body.symbol,
            close_pct: body.close_pct,
            order_type: body.order_type,
            limit_price: body.limit_price,
        };
        let hash_for_preview = manual_trade::close_request_hash(&request);
        let config = state.config.clone();
        let symbol = request.symbol.clone();
        let (preview, confirm_token) = tokio::task::spawn_blocking(move || {
            let preview = manual_trade::preview_close(&config, &request)?;
            let confirm_token = manual_trade::issue_confirm_token(
                &config,
                "CLOSE",
                &hash_for_preview,
                &symbol,
                &preview,
            )?;
            Ok::<_, HubError>((preview, confirm_token))
        })
        .await
        .map_err(|error| HubError::Internal(format!("close preview task failed: {error}")))??;
        let mut payload = preview;
        if let Some(object) = payload.as_object_mut() {
            object.insert("confirm_token".to_string(), Value::String(confirm_token));
        }
        return Ok(Json(payload));
    }

    let request = manual_trade::ManualTradeCloseRequest {
        symbol: body.symbol,
        close_pct: body.close_pct,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let hash = manual_trade::close_request_hash(&request);
    let config = state.config.clone();
    let token = body
        .confirm_token
        .as_deref()
        .ok_or_else(|| HubError::BadRequest("confirm_token required".into()))?
        .to_string();
    let preflight_request = request.clone();
    let preflight_token = token.clone();
    let preflight_hash = hash.clone();
    let preflight_config = config.clone();
    let preflight = tokio::task::spawn_blocking(move || {
        manual_trade::preflight_close_execution(
            &preflight_config,
            &preflight_request,
            &preflight_token,
            &preflight_hash,
        )
    })
    .await
    .map_err(|error| HubError::Internal(format!("close preflight task failed: {error}")))??;
    if matches!(
        preflight,
        manual_trade::ManualExecutionPreflight::NewSubmission
    ) {
        ensure_live_orders_enabled(&state, &request.symbol, "CLOSE").await?;
        check_manual_order_risk(&state, &request.symbol, OrderAction::Close, true, "CLOSE").await?;
    }
    let symbol = request.symbol.clone();
    let result = tokio::task::spawn_blocking(move || {
        manual_trade::execute_close(&config, &request, &token, &hash)
    })
    .await
    .map_err(|error| HubError::Internal(format!("close task failed: {error}")))??;
    if result.get("deduped").and_then(|value| value.as_bool()) != Some(true) {
        note_manual_order_sent(&state, &symbol, OrderAction::Close, true).await;
    }
    Ok(Json(result))
}

async fn trade_cancel(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeCancelBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let config = state.config.clone();
    let request = manual_trade::ManualTradeCancelRequest {
        symbol: body.symbol,
        oid: body.oid,
        intent_id: body.intent_id,
    };
    let preflight_config = config.clone();
    let preflight_request = request.clone();
    let preflight = tokio::task::spawn_blocking(move || {
        manual_trade::preflight_cancel_retry(&preflight_config, &preflight_request)
    })
    .await
    .map_err(|error| HubError::Internal(format!("cancel preflight task failed: {error}")))??;
    if let Some(existing) = preflight {
        return Ok(Json(existing));
    }
    ensure_live_orders_enabled(&state, &request.symbol, "CANCEL").await?;
    check_manual_cancel_risk(&state, &request.symbol).await?;
    let result = tokio::task::spawn_blocking(move || manual_trade::cancel_order(&config, &request))
        .await
        .map_err(|error| HubError::Internal(format!("cancel task failed: {error}")))??;
    Ok(Json(result))
}

async fn trade_open_orders(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let config = state.config.clone();
    let result = tokio::task::spawn_blocking(move || manual_trade::open_orders(&config, &symbol))
        .await
        .map_err(|error| HubError::Internal(format!("open-orders task failed: {error}")))??;
    Ok(Json(result))
}

async fn trade_result(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, HubError> {
    let config = state.config.clone();
    let result = tokio::task::spawn_blocking(move || manual_trade::execution_result(&config, &id))
        .await
        .map_err(|error| HubError::Internal(format!("trade-result task failed: {error}")))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_param_fingerprint_normalises_case_and_spacing() {
        let a = TradeOpenBody {
            symbol: " eth ".to_string(),
            side: " sell ".to_string(),
            notional_usd: 500.0,
            leverage: 10,
            order_type: " MARKET ".to_string(),
            limit_price: None,
            confirm_token: None,
        };
        let b = TradeOpenBody {
            symbol: "ETH".to_string(),
            side: "SELL".to_string(),
            notional_usd: 500.0,
            leverage: 10,
            order_type: "market".to_string(),
            limit_price: None,
            confirm_token: None,
        };

        assert_eq!(open_param_fingerprint(&a), open_param_fingerprint(&b));
    }

    #[test]
    fn open_param_fingerprint_changes_when_limit_price_changes() {
        let base = TradeOpenBody {
            symbol: "ETH".to_string(),
            side: "SELL".to_string(),
            notional_usd: 500.0,
            leverage: 10,
            order_type: "limit_gtc".to_string(),
            limit_price: Some(2100.5),
            confirm_token: None,
        };
        let changed = TradeOpenBody {
            limit_price: Some(2100.6),
            ..base
        };

        assert_ne!(
            open_param_fingerprint(&TradeOpenBody {
                symbol: "ETH".to_string(),
                side: "SELL".to_string(),
                notional_usd: 500.0,
                leverage: 10,
                order_type: "limit_gtc".to_string(),
                limit_price: Some(2100.5),
                confirm_token: None,
            }),
            open_param_fingerprint(&changed)
        );
    }

    #[test]
    fn close_param_fingerprint_normalises_case_and_spacing() {
        let a = TradeCloseBody {
            symbol: " hype ".to_string(),
            close_pct: 25.0,
            order_type: " LIMIT_IOC ".to_string(),
            limit_price: Some(35.5),
            confirm_token: None,
        };
        let b = TradeCloseBody {
            symbol: "HYPE".to_string(),
            close_pct: 25.0,
            order_type: "limit_ioc".to_string(),
            limit_price: Some(35.5),
            confirm_token: None,
        };

        assert_eq!(close_param_fingerprint(&a), close_param_fingerprint(&b));
    }
}
