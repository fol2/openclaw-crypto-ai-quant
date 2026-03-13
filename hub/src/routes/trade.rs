use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use crate::error::HubError;
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

fn open_param_hash(body: &TradeOpenBody) -> String {
    let mut hasher = DefaultHasher::new();
    body.symbol.hash(&mut hasher);
    body.side.hash(&mut hasher);
    body.notional_usd.to_bits().hash(&mut hasher);
    body.leverage.hash(&mut hasher);
    body.order_type.hash(&mut hasher);
    body.limit_price
        .map(|price| price.to_bits())
        .hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn close_param_hash(body: &TradeCloseBody) -> String {
    let mut hasher = DefaultHasher::new();
    "close".hash(&mut hasher);
    body.symbol.hash(&mut hasher);
    body.close_pct.to_bits().hash(&mut hasher);
    body.order_type.hash(&mut hasher);
    body.limit_price
        .map(|price| price.to_bits())
        .hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

async fn generate_confirm_token(state: &AppState, param_hash: String) -> String {
    let token = uuid::Uuid::new_v4().to_string();
    let mut tokens = state.trade_confirm_tokens.lock().await;
    tokens.retain(|_, (_, created)| created.elapsed().as_secs() < 300);
    tokens.insert(token.clone(), (param_hash, Instant::now()));
    token
}

async fn validate_confirm_token(
    state: &AppState,
    token: &str,
    expected_hash: &str,
) -> Result<(), HubError> {
    let mut tokens = state.trade_confirm_tokens.lock().await;
    let (hash, created) = tokens
        .remove(token)
        .ok_or_else(|| HubError::BadRequest("invalid or expired confirm token".into()))?;
    if created.elapsed().as_secs() > 60 {
        return Err(HubError::BadRequest("confirm token expired".into()));
    }
    if hash != expected_hash {
        return Err(HubError::BadRequest(
            "confirm token does not match parameters".into(),
        ));
    }
    Ok(())
}

async fn check_rate_limit(state: &AppState, symbol: &str) -> Result<(), HubError> {
    let symbol = symbol.trim().to_ascii_uppercase();
    let mut limits = state.trade_rate_limits.lock().await;
    limits.retain(|_, created| created.elapsed().as_secs() < 60);
    if let Some(last) = limits.get(&symbol) {
        if last.elapsed().as_secs() < 5 {
            return Err(HubError::BadRequest(format!(
                "rate limited: wait before trading {symbol} again"
            )));
        }
    }
    limits.insert(symbol, Instant::now());
    Ok(())
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
    let param_hash = open_param_hash(&body);
    let config = state.config.clone();
    let request = manual_trade::ManualTradeOpenRequest {
        symbol: body.symbol,
        side: body.side,
        notional_usd: body.notional_usd,
        leverage: body.leverage,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let preview =
        tokio::task::spawn_blocking(move || manual_trade::preview_open(&config, &request))
            .await
            .map_err(|error| HubError::Internal(format!("preview task failed: {error}")))??;
    let mut payload = preview;
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "confirm_token".to_string(),
            Value::String(generate_confirm_token(&state, param_hash).await),
        );
    }
    Ok(Json(payload))
}

async fn trade_execute(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeOpenBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let token = body
        .confirm_token
        .as_deref()
        .ok_or_else(|| HubError::BadRequest("confirm_token required".into()))?;
    check_rate_limit(&state, &body.symbol).await?;
    validate_confirm_token(&state, token, &open_param_hash(&body)).await?;

    let config = state.config.clone();
    let request = manual_trade::ManualTradeOpenRequest {
        symbol: body.symbol,
        side: body.side,
        notional_usd: body.notional_usd,
        leverage: body.leverage,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let result = tokio::task::spawn_blocking(move || manual_trade::execute_open(&config, &request))
        .await
        .map_err(|error| HubError::Internal(format!("execute task failed: {error}")))??;
    Ok(Json(result))
}

async fn trade_close(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TradeCloseBody>,
) -> Result<Json<Value>, HubError> {
    require_trade_enabled(&state)?;
    let hash = close_param_hash(&body);

    if body.confirm_token.is_none() {
        let config = state.config.clone();
        let request = manual_trade::ManualTradeCloseRequest {
            symbol: body.symbol,
            close_pct: body.close_pct,
            order_type: body.order_type,
            limit_price: body.limit_price,
        };
        let preview =
            tokio::task::spawn_blocking(move || manual_trade::preview_close(&config, &request))
                .await
                .map_err(|error| {
                    HubError::Internal(format!("close preview task failed: {error}"))
                })??;
        let mut payload = preview;
        if let Some(object) = payload.as_object_mut() {
            object.insert(
                "confirm_token".to_string(),
                Value::String(generate_confirm_token(&state, hash).await),
            );
        }
        return Ok(Json(payload));
    }

    validate_confirm_token(
        &state,
        body.confirm_token
            .as_deref()
            .ok_or_else(|| HubError::BadRequest("confirm_token required".into()))?,
        &hash,
    )
    .await?;

    let config = state.config.clone();
    let request = manual_trade::ManualTradeCloseRequest {
        symbol: body.symbol,
        close_pct: body.close_pct,
        order_type: body.order_type,
        limit_price: body.limit_price,
    };
    let result =
        tokio::task::spawn_blocking(move || manual_trade::execute_close(&config, &request))
            .await
            .map_err(|error| HubError::Internal(format!("close task failed: {error}")))??;
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

async fn trade_result(Path(id): Path<String>) -> Result<Json<Value>, HubError> {
    Err(HubError::NotFound(format!(
        "job {id} not found (rust-native manual trade executes synchronously)"
    )))
}
