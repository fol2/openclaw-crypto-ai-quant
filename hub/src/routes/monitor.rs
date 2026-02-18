use axum::extract::{Query, State};
use axum::http::header;
use axum::response::IntoResponse;
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
        .route("/api/marks", get(api_marks))
        .route("/api/sparkline", get(api_sparkline))
        .route("/api/metrics", get(api_metrics))
        .route("/metrics", get(api_prometheus))
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

    // Last signals, trades, and intents per symbol
    let last_signals = trading::last_signals_by_symbol(&conn, &merged)?;
    let last_trades = trading::last_trades_by_symbol(&conn, &merged)?;
    let last_intents = trading::last_intents_by_symbol(&conn, &merged)?;

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
    let mut margin_used_total = 0.0_f64;
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
            let lev = p.leverage.max(0.01);
            margin_used_total += p.size.abs() * m / lev;
        }

        symbols_out.push(json!({
            "symbol": sym,
            "mid": mid,
            "last_signal": last_signals.get(sym),
            "last_trade": last_trades.get(sym),
            "last_intent": last_intents.get(sym),
            "position": pos.map(|p| json!({
                "symbol": p.symbol,
                "type": p.pos_type,
                "entry_price": p.entry_price,
                "size": p.size,
                "leverage": p.leverage,
                "margin_used": p.margin_used,
                "open_trade_id": p.open_trade_id,
                "open_timestamp": p.open_timestamp,
                "confidence": p.confidence,
                "entry_atr": p.entry_atr,
                "unreal_pnl_est": unreal_pnl,
            })),
        }));
    }

    // Balance
    let bal = trading::latest_balance(&conn)?;
    let realised_usd = bal.as_ref().map(|(b, _)| *b);
    let realised_asof = bal.as_ref().and_then(|(_, ts)| ts.clone());
    let equity_est = realised_usd.map(|r| r + unreal_total - close_fee_total);

    // Recent data
    let recent_trades = trading::recent_trades(&conn, 60)?;
    let recent_signals = trading::recent_signals(&conn, 60)?;
    let audit_events = trading::recent_audit_events(&conn, 60)?;
    let oms_intents = trading::recent_oms_intents(&conn, 80)?;
    let oms_fills = trading::recent_oms_fills(&conn, 80)?;
    let oms_open_orders = trading::recent_oms_open_orders(&conn, 80)?;
    let oms_reconcile = trading::recent_oms_reconcile(&conn, 80)?;

    // Daily metrics
    let daily = trading::daily_metrics(&conn, ts)?;

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
            "balance_source": if mode == "live" { "db_snapshot" } else { "paper_estimate" },
            "realised_usd": realised_usd,
            "realised_asof": realised_asof,
            "equity_est_usd": equity_est,
            "unreal_pnl_est_usd": unreal_total,
            "est_close_fees_usd": close_fee_total,
            "margin_used_est_usd": if mode == "live" { Some(margin_used_total) } else { None::<f64> },
            "fee_rate": fee_rate,
        },
        "daily": daily,
        "symbols": symbols_out,
        "open_positions": positions,
        "recent": {
            "trades": recent_trades,
            "signals": recent_signals,
            "audit_events": audit_events,
            "oms_intents": oms_intents,
            "oms_fills": oms_fills,
            "oms_open_orders": oms_open_orders,
            "oms_reconcile_events": oms_reconcile,
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

async fn api_marks(
    State(state): State<Arc<AppState>>,
    Query(q): Query<MarksQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let sym = q.symbol.to_uppercase();

    if sym.is_empty() {
        return Err(HubError::BadRequest("missing symbol".to_string()));
    }

    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;

    let pos = trading::open_position_for_symbol(&conn, &sym)?;
    let entries = if let Some(ref p) = pos {
        trading::position_entries(&conn, &sym, p.open_trade_id)?
    } else {
        Vec::new()
    };

    Ok(Json(json!({
        "ok": true,
        "now_ts_ms": now_ms(),
        "mode": mode,
        "symbol": sym,
        "position": pos,
        "entries": entries,
    })))
}

async fn api_sparkline(
    State(_state): State<Arc<AppState>>,
    Query(q): Query<SparklineQuery>,
) -> Json<Value> {
    // Sparkline requires in-memory mid-price history.
    // In Phase 2, return empty — the frontend will use WebSocket mids instead.
    Json(json!({
        "symbol": q.symbol.to_uppercase(),
        "window_s": q.window_s,
        "points": [],
    }))
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
    if let Some(ws) = heartbeat.ws_connected {
        gauges.insert("ws_connected".into(), json!(if ws { 1 } else { 0 }));
    }

    // Kill mode gauge
    let kill_mode = heartbeat.kill_mode.as_deref().unwrap_or("off").to_lowercase();
    let kill_val = match kill_mode.as_str() {
        "halt_all" => 2,
        "close_only" => 1,
        _ => 0,
    };
    gauges.insert("kill_mode".into(), json!(kill_val));

    // Slippage gauges from heartbeat
    if let Some(v) = heartbeat.slip_enabled {
        gauges.insert("slip_enabled".into(), json!(if v { 1 } else { 0 }));
    }
    if let Some(v) = heartbeat.slip_n {
        gauges.insert("slip_n".into(), json!(v));
    }
    if let Some(v) = heartbeat.slip_thr_bps {
        gauges.insert("slip_thr_bps".into(), json!(v));
    }
    if let Some(v) = heartbeat.slip_last_bps {
        gauges.insert("slip_last_bps".into(), json!(v));
    }
    if let Some(v) = heartbeat.slip_median_bps {
        gauges.insert("slip_median_bps".into(), json!(v));
    }

    // Daily PnL gauges
    let daily = trading::daily_metrics(&conn, ts)?;
    gauges.insert("pnl_today_usd".into(), json!(daily.pnl_usd));
    gauges.insert("fees_today_usd".into(), json!(daily.fees_usd));
    gauges.insert("net_pnl_today_usd".into(), json!(daily.net_pnl_usd));
    gauges.insert("drawdown_today_pct".into(), json!(daily.drawdown_pct));

    // Counters
    let counters = trading::metrics_counters(&conn)?;

    Ok(Json(json!({
        "ok": true,
        "now_ts_ms": ts,
        "mode": mode,
        "health": heartbeat,
        "gauges": gauges,
        "counters": counters,
    })))
}

async fn api_prometheus(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ModeQuery>,
) -> impl IntoResponse {
    let mode = normalize_mode(&q.mode);
    let ts = now_ms();

    let pool = match state.db_pool(&mode) {
        Some(p) => p,
        None => {
            return (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                "# no db\n".to_string(),
            );
        }
    };

    let conn = match pool.get() {
        Ok(c) => c,
        Err(_) => {
            return (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                "# db error\n".to_string(),
            );
        }
    };

    let heartbeat = runtime::fetch_heartbeat_from_db(&conn)
        .ok()
        .flatten()
        .unwrap_or_default();

    let mut lines: Vec<String> = Vec::new();

    // Engine up
    lines.push(prom_line("aiq_engine_up", if heartbeat.ok { 1.0 } else { 0.0 }, &mode));

    if let Some(ts_hb) = heartbeat.ts_ms {
        let age = ((ts - ts_hb) as f64 / 1000.0).max(0.0);
        lines.push(prom_line("aiq_heartbeat_age_s", age, &mode));
    }
    if let Some(n) = heartbeat.open_pos {
        lines.push(prom_line("aiq_open_pos", n as f64, &mode));
    }

    // Kill mode
    let kill_mode = heartbeat.kill_mode.as_deref().unwrap_or("off").to_lowercase();
    let kill_val = match kill_mode.as_str() {
        "halt_all" => 2.0,
        "close_only" => 1.0,
        _ => 0.0,
    };
    lines.push(prom_line("aiq_kill_mode", kill_val, &mode));

    // Daily PnL
    if let Ok(daily) = trading::daily_metrics(&conn, ts) {
        lines.push(prom_line("aiq_pnl_today_usd", daily.pnl_usd, &mode));
        lines.push(prom_line("aiq_fees_today_usd", daily.fees_usd, &mode));
        lines.push(prom_line("aiq_net_pnl_today_usd", daily.net_pnl_usd, &mode));
        lines.push(prom_line("aiq_drawdown_today_pct", daily.drawdown_pct, &mode));
    }

    // Counters
    if let Ok(counters) = trading::metrics_counters(&conn) {
        if let Some(obj) = counters.as_object() {
            for (k, v) in obj {
                if let Some(f) = v.as_f64() {
                    lines.push(prom_line(&format!("aiq_{k}"), f, &mode));
                }
            }
        }
    }

    let body = lines.join("") + "\n";
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
}

fn prom_line(name: &str, value: f64, mode: &str) -> String {
    format!("{name}{{mode=\"{mode}\"}} {value}\n")
}

pub fn normalize_mode(mode: &str) -> String {
    let m = mode.trim().to_lowercase();
    match m.as_str() {
        "live" | "paper1" | "paper2" | "paper3" => m,
        "paper" => "paper1".to_string(),
        _ => "paper1".to_string(),
    }
}
