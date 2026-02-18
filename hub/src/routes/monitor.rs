use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::{candles, runtime, trading};
use crate::db::pool::open_ro_pool;
use crate::error::HubError;
use crate::state::AppState;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ── Query params ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ModeQuery {
    #[serde(default = "default_mode")]
    mode: String,
}

fn default_mode() -> String {
    "paper".to_string()
}

#[derive(Debug, Deserialize)]
pub struct SparklineQuery {
    symbol: String,
    #[serde(default = "default_window_s")]
    window_s: u32,
}

fn default_window_s() -> u32 {
    600
}

#[derive(Debug, Deserialize)]
pub struct CandleQuery {
    #[serde(default = "default_mode")]
    mode: String,
    symbol: String,
    #[serde(default)]
    interval: Option<String>,
    #[serde(default = "default_limit")]
    limit: u32,
}

fn default_limit() -> u32 {
    200
}

#[derive(Debug, Deserialize)]
pub struct MarksQuery {
    #[serde(default = "default_mode")]
    mode: String,
    symbol: String,
}

// ── Route definitions ────────────────────────────────────────────────────

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/health", get(api_health))
        .route("/api/snapshot", get(api_snapshot))
        .route("/api/mids", get(api_mids))
        .route("/api/candles", get(api_candles))
        .route("/api/metrics", get(api_metrics))
}

// ── Handlers ─────────────────────────────────────────────────────────────

async fn api_health(
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    let sidecar_ok = state.sidecar.health().await.is_ok();
    Json(json!({
        "ok": true,
        "now_ts_ms": now_ms(),
        "sidecar_connected": sidecar_ok,
    }))
}

async fn api_mids(
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    // Best-effort: query sidecar for BTC at minimum.
    match state.sidecar.get_mids(&["BTC".to_string()]).await {
        Ok(snap) => Json(json!({
            "ok": true,
            "mids": snap.mids,
            "mids_age_s": snap.mids_age_s,
        })),
        Err(e) => Json(json!({
            "ok": false,
            "error": e,
        })),
    }
}

async fn api_snapshot(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let ts = now_ms();
    let (db_path, _log_path) = state.config.mode_paths(&mode);

    // Try to get a DB pool.
    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db(format!("db not available for mode={mode}")))?;

    let conn = pool.get()?;

    // Heartbeat
    let heartbeat = runtime::fetch_heartbeat_from_db(&conn)?.unwrap_or_else(|| {
        crate::heartbeat::Heartbeat {
            ok: false,
            error: Some("heartbeat_missing".to_string()),
            ..Default::default()
        }
    });

    // Open positions
    let positions = trading::compute_open_positions(&conn)?;

    // Recent symbols
    let interval = &state.config.monitor_interval;

    // Try opening candle DB
    let candle_db_path = state.candle_db_path(interval);
    let candle_pool = open_ro_pool(&candle_db_path, 2);
    let candle_conn = candle_pool.as_ref().and_then(|p| p.get().ok());

    let symbols = trading::list_recent_symbols(
        &conn,
        candle_conn.as_deref(),
        interval,
        ts,
        200,
    )?;

    // Merge: open positions first, then other symbols
    let mut merged: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for pos in &positions {
        if seen.insert(pos.symbol.clone()) {
            merged.push(pos.symbol.clone());
        }
    }
    for s in &symbols {
        if seen.insert(s.clone()) {
            merged.push(s.clone());
        }
    }
    merged.truncate(200);

    // Last signals and trades per symbol
    let last_signals = trading::last_signals_by_symbol(&conn, &merged)?;
    let last_trades = trading::last_trades_by_symbol(&conn, &merged)?;

    // Get mids from sidecar
    let mids_snap = state.sidecar.get_mids(&merged).await.ok();
    let mids: HashMap<String, f64> = mids_snap
        .as_ref()
        .map(|s| s.mids.clone())
        .unwrap_or_default();

    // Build symbols output with mid prices and positions
    let fee_rate = state.config.fee_rate;
    let mut unreal_total = 0.0_f64;
    let mut close_fee_total = 0.0_f64;
    let pos_map: HashMap<String, &trading::OpenPosition> =
        positions.iter().map(|p| (p.symbol.clone(), p)).collect();

    let mut symbols_out: Vec<Value> = Vec::new();
    for sym in &merged {
        let mid = mids.get(sym).copied();
        let pos = pos_map.get(sym).copied();
        let mut unreal_pnl: Option<f64> = None;

        if let (Some(p), Some(m)) = (pos, mid) {
            let u = if p.pos_type == "LONG" {
                (m - p.entry_price) * p.size
            } else {
                (p.entry_price - m) * p.size
            };
            unreal_pnl = Some(u);
            unreal_total += u;
            close_fee_total += p.size.abs() * m * fee_rate;
        }

        symbols_out.push(json!({
            "symbol": sym,
            "mid": mid,
            "last_signal": last_signals.get(sym),
            "last_trade": last_trades.get(sym),
            "position": pos.map(|p| json!({
                "symbol": p.symbol,
                "type": p.pos_type,
                "entry_price": p.entry_price,
                "size": p.size,
                "leverage": p.leverage,
                "margin_used": p.margin_used,
                "unreal_pnl_est": unreal_pnl,
            })),
        }));
    }

    // Balance
    let bal = trading::latest_balance(&conn)?;
    let realised_usd = bal.as_ref().map(|(b, _)| *b);
    let equity_est = realised_usd.map(|r| r + unreal_total - close_fee_total);

    // Recent trades/signals
    let recent_trades = trading::recent_trades(&conn, 60)?;
    let recent_signals = trading::recent_signals(&conn, 60)?;

    // Available candle intervals
    let intervals = candles::list_available_intervals(&state.config.candles_db_dir);

    Ok(Json(json!({
        "now_ts_ms": ts,
        "mode": mode,
        "db_path": db_path.to_string_lossy(),
        "health": heartbeat,
        "config": {
            "trader_interval": state.config.trader_interval,
            "candle_intervals": intervals,
        },
        "balances": {
            "realised_usd": realised_usd,
            "equity_est_usd": equity_est,
            "unreal_pnl_est_usd": unreal_total,
            "est_close_fees_usd": close_fee_total,
            "fee_rate": fee_rate,
        },
        "symbols": symbols_out,
        "open_positions": positions,
        "recent": {
            "trades": recent_trades,
            "signals": recent_signals,
        },
        "ws": {
            "ok": mids_snap.is_some(),
            "mids": mids,
            "mids_age_s": mids_snap.as_ref().and_then(|s| s.mids_age_s),
        },
    })))
}

async fn api_candles(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CandleQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let sym = q.symbol.to_uppercase();
    let interval = q.interval.as_deref().unwrap_or(&state.config.trader_interval);
    let limit = q.limit.clamp(2, 2000);

    // Try candle-specific DB first, fall back to trading DB.
    let candle_path = state.candle_db_path(interval);
    let candle_pool = open_ro_pool(&candle_path, 2);

    if let Some(pool) = &candle_pool {
        if let Ok(conn) = pool.get() {
            let data = candles::fetch_candles(&conn, &sym, interval, limit)?;
            if !data.is_empty() {
                return Ok(Json(json!({
                    "ok": true,
                    "symbol": sym,
                    "interval": interval,
                    "count": data.len(),
                    "candles": data,
                })));
            }
        }
    }

    // Fall back to trading DB.
    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;
    let data = candles::fetch_candles(&conn, &sym, interval, limit)?;
    Ok(Json(json!({
        "ok": true,
        "symbol": sym,
        "interval": interval,
        "count": data.len(),
        "candles": data,
    })))
}

async fn api_metrics(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let ts = now_ms();

    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;

    let heartbeat = runtime::fetch_heartbeat_from_db(&conn)?.unwrap_or_default();

    let mut gauges: HashMap<String, Value> = HashMap::new();
    gauges.insert("engine_up".into(), json!(if heartbeat.ok { 1 } else { 0 }));
    if let Some(ts_hb) = heartbeat.ts_ms {
        let age = ((ts - ts_hb) as f64 / 1000.0).max(0.0);
        gauges.insert("heartbeat_age_s".into(), json!(age));
    }
    if let Some(n) = heartbeat.open_pos {
        gauges.insert("open_pos".into(), json!(n));
    }

    Ok(Json(json!({
        "ok": true,
        "now_ts_ms": ts,
        "mode": mode,
        "health": heartbeat,
        "gauges": gauges,
    })))
}

fn normalize_mode(mode: &str) -> String {
    let m = mode.trim().to_lowercase();
    match m.as_str() {
        "live" | "paper1" | "paper2" | "paper3" => m,
        "paper" => "paper1".to_string(),
        _ => "paper1".to_string(),
    }
}
