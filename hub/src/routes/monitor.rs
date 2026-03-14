use axum::extract::{Query, State};
use axum::http::header;
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::pool::open_ro_pool;
use crate::db::{candles, runtime, trading, tunnel};
use crate::error::HubError;
use crate::heartbeat::Heartbeat;
use crate::hyperliquid::HlPositionSnapshot;
use crate::state::AppState;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn deployed_config_boundary(
    artifacts_dir: &std::path::Path,
    lane: &str,
    conn: &rusqlite::Connection,
    heartbeat: &Heartbeat,
) -> Result<Option<(String, String)>, HubError> {
    let Some(config_id) = heartbeat.config_id.clone() else {
        return Ok(None);
    };
    let ledger_from_ts = crate::config_audit::latest_successful_event_for_config_id(
        artifacts_dir,
        lane,
        &config_id,
    )?
    .and_then(|event| {
        chrono::DateTime::from_timestamp_millis(event.ts_ms)
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
    });
    let runtime_from_ts =
        runtime::first_seen_config_id_ts_ms(conn, &config_id)?.and_then(|ts_ms| {
            chrono::DateTime::from_timestamp_millis(ts_ms)
                .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
        });
    let Some(from_ts) = ledger_from_ts.or(runtime_from_ts) else {
        return Ok(None);
    };
    Ok(Some((config_id, from_ts)))
}

fn redacted_heartbeat(heartbeat: &crate::heartbeat::Heartbeat) -> Value {
    let mut value = serde_json::to_value(heartbeat).unwrap_or_else(|_| json!({}));
    if let Some(obj) = value.as_object_mut() {
        obj.remove("line");
        obj.insert("line_redacted".to_string(), Value::Bool(true));
    }
    value
}

fn redacted_recent_audit_events(events: Vec<Value>) -> Vec<Value> {
    events
        .into_iter()
        .map(|event| {
            let mut obj = event.as_object().cloned().unwrap_or_default();
            let had_data = obj.remove("data_json").is_some();
            obj.insert("data_redacted".to_string(), Value::Bool(had_data));
            Value::Object(obj)
        })
        .collect()
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

#[derive(Debug, Deserialize)]
pub struct JourneyQuery {
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_journey_limit")]
    limit: u32,
    #[serde(default)]
    offset: u32,
    #[serde(default)]
    symbol: Option<String>,
}

fn default_journey_limit() -> u32 {
    50
}

#[derive(Debug, Deserialize)]
pub struct TradesQuery {
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_trades_limit")]
    limit: u32,
    #[serde(default)]
    offset: u32,
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    from_ts: Option<String>,
    #[serde(default)]
    to_ts: Option<String>,
}
fn default_trades_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct CandleRangeQuery {
    #[serde(default = "default_mode")]
    mode: String,
    symbol: String,
    #[serde(default)]
    interval: Option<String>,
    #[serde(default)]
    from_ts: Option<i64>,
    #[serde(default)]
    to_ts: Option<i64>,
    #[serde(default = "default_limit")]
    limit: u32,
}

#[derive(Debug, Deserialize)]
pub struct FlashDebugEvent {
    symbol: String,
    prev: f64,
    mid: f64,
    direction: String,
    phase: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    tone: Option<String>,
    at_ms: i64,
}

#[derive(Debug, Deserialize)]
pub struct FlashDebugBatch {
    events: Vec<FlashDebugEvent>,
}

#[derive(Debug, Deserialize)]
pub struct TrendClosesQuery {
    #[serde(default = "default_trend_interval")]
    interval: String,
    #[serde(default = "default_trend_limit")]
    limit: u32,
}
fn default_trend_interval() -> String {
    "5m".to_string()
}
fn default_trend_limit() -> u32 {
    60
}

#[derive(Debug, Deserialize)]
pub struct TunnelQuery {
    #[serde(default = "default_mode")]
    mode: String,
    symbol: String,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    #[serde(default = "default_tunnel_limit")]
    limit: u32,
}
fn default_tunnel_limit() -> u32 {
    2000
}

// ── Route definitions ────────────────────────────────────────────────────

pub fn routes() -> Router<Arc<AppState>> {
    let read_routes = Router::new()
        .route("/api/health", get(api_health))
        .route("/api/snapshot", get(api_snapshot))
        .route("/api/mids", get(api_mids))
        .route("/api/candles", get(api_candles))
        .route("/api/marks", get(api_marks))
        .route("/api/sparkline", get(api_sparkline))
        .route("/api/journeys", get(api_journeys))
        .route("/api/candles/range", get(api_candles_range))
        .route("/api/trend-closes", get(api_trend_closes))
        .route("/api/trend-candles", get(api_trend_candles))
        .route("/api/volumes", get(api_volumes))
        .route("/api/metrics", get(api_metrics))
        .route("/api/tunnel", get(api_tunnel))
        .route("/api/trades", get(api_trades))
        .route("/metrics", get(api_prometheus));
    let mutation_routes = Router::new()
        .route("/api/flash-debug", post(api_flash_debug))
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));

    read_routes.merge(mutation_routes)
}

fn live_position_overrides(
    mode: &str,
    hl_snap: Option<&crate::hyperliquid::HlAccountSnapshot>,
) -> HashMap<String, HlPositionSnapshot> {
    if mode != "live" {
        return HashMap::new();
    }

    hl_snap
        .map(|snap| {
            snap.positions
                .iter()
                .map(|position| (position.symbol.clone(), position.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn live_position_overrides_from_db(
    mode: &str,
    db_positions: &[trading::RuntimeExchangePositionSnapshot],
) -> HashMap<String, HlPositionSnapshot> {
    if mode != "live" {
        return HashMap::new();
    }

    db_positions
        .iter()
        .map(|position| {
            (
                position.symbol.clone(),
                HlPositionSnapshot {
                    symbol: position.symbol.clone(),
                    pos_type: position.pos_type.clone(),
                    size: position.size,
                    entry_price: position.entry_price,
                    leverage: position.leverage,
                    margin_used: position.margin_used,
                },
            )
        })
        .collect()
}

fn live_position_override_for<'a>(
    db_position: &trading::OpenPosition,
    live_positions: &'a HashMap<String, HlPositionSnapshot>,
) -> Option<&'a HlPositionSnapshot> {
    live_positions
        .get(&db_position.symbol)
        .filter(|live_position| {
            live_position
                .pos_type
                .eq_ignore_ascii_case(&db_position.pos_type)
        })
}

fn synthetic_open_position_from_live_snapshot(
    live_position: &HlPositionSnapshot,
) -> trading::OpenPosition {
    trading::OpenPosition {
        symbol: live_position.symbol.clone(),
        pos_type: live_position.pos_type.clone(),
        open_trade_id: 0,
        open_timestamp: None,
        entry_price: live_position.entry_price,
        size: live_position.size,
        confidence: None,
        entry_atr: 0.0,
        leverage: live_position.leverage.max(0.01),
        margin_used: live_position.margin_used.max(0.0),
    }
}

fn merge_positions_with_live_snapshot(
    ledger_positions: &[trading::OpenPosition],
    live_positions: &HashMap<String, HlPositionSnapshot>,
    live_snapshot_authoritative: bool,
) -> Vec<trading::OpenPosition> {
    if !live_snapshot_authoritative {
        return ledger_positions.to_vec();
    }

    let mut merged = Vec::new();
    let mut symbols = live_positions.keys().cloned().collect::<Vec<_>>();
    symbols.sort();
    for symbol in symbols {
        let Some(live_position) = live_positions.get(&symbol) else {
            continue;
        };
        if let Some(ledger_position) = ledger_positions.iter().find(|position| {
            position.symbol == symbol
                && position
                    .pos_type
                    .eq_ignore_ascii_case(&live_position.pos_type)
        }) {
            let mut merged_position = ledger_position.clone();
            merged_position.entry_price = live_position.entry_price;
            merged_position.size = live_position.size;
            merged_position.leverage = live_position.leverage.max(0.01);
            merged_position.margin_used = live_position.margin_used.max(0.0);
            merged.push(merged_position);
        } else {
            merged.push(synthetic_open_position_from_live_snapshot(live_position));
        }
    }
    merged
}

fn position_json(
    position: &trading::OpenPosition,
    live_override: Option<&HlPositionSnapshot>,
    unreal_pnl: Option<f64>,
) -> Value {
    json!({
        "symbol": position.symbol,
        "type": position.pos_type,
        "entry_price": live_override.map(|live| live.entry_price).unwrap_or(position.entry_price),
        "size": live_override.map(|live| live.size).unwrap_or(position.size),
        "leverage": live_override.map(|live| live.leverage).unwrap_or(position.leverage),
        "margin_used": live_override.map(|live| live.margin_used).unwrap_or(position.margin_used),
        "open_trade_id": position.open_trade_id,
        "open_timestamp": position.open_timestamp,
        "confidence": position.confidence,
        "entry_atr": position.entry_atr,
        "unreal_pnl_est": unreal_pnl,
    })
}

// ── Handlers ─────────────────────────────────────────────────────────────

async fn api_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let sidecar_ok = state.sidecar.health().await.is_ok();
    Json(json!({
        "ok": true,
        "now_ts_ms": now_ms(),
        "sidecar_connected": sidecar_ok,
    }))
}

async fn api_mids(State(state): State<Arc<AppState>>) -> Json<Value> {
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

async fn api_flash_debug(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FlashDebugBatch>,
) -> Json<Value> {
    if !state.config.flash_debug_log {
        return Json(json!({
            "ok": false,
            "disabled": true,
            "reason": "AIQ_MONITOR_FLASH_DEBUG_LOG is disabled",
        }));
    }

    // Keep log volume bounded even under very high tick throughput.
    let max_logged = payload.events.len().min(400);
    for ev in payload.events.iter().take(max_logged) {
        tracing::info!(
            symbol = %ev.symbol,
            prev = ev.prev,
            mid = ev.mid,
            direction = %ev.direction,
            phase = %ev.phase,
            source = ?ev.source,
            tone = ?ev.tone,
            at_ms = ev.at_ms,
            "mid flash trigger"
        );
    }
    if payload.events.len() > max_logged {
        tracing::info!(
            total = payload.events.len(),
            logged = max_logged,
            dropped = payload.events.len() - max_logged,
            "mid flash trigger batch truncated"
        );
    }

    Json(json!({
        "ok": true,
        "received": payload.events.len(),
        "logged": max_logged,
    }))
}

async fn api_snapshot(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let ts = now_ms();
    let (_db_path, _log_path) = state.config.mode_paths(&mode);

    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db(format!("db not available for mode={mode}")))?;
    let conn = pool.get()?;

    // Heartbeat
    let heartbeat =
        runtime::fetch_heartbeat_from_db(&conn)?.unwrap_or_else(|| crate::heartbeat::Heartbeat {
            ok: false,
            error: Some("heartbeat_missing".to_string()),
            ..Default::default()
        });

    // Open positions
    let ledger_positions = trading::compute_open_positions(&conn)?;

    // Recent symbols
    let interval = &state.config.monitor_interval;
    let candle_db_path = state.candle_db_path(interval);
    let candle_pool = open_ro_pool(&candle_db_path, 2);
    let candle_conn = candle_pool.as_ref().and_then(|p| p.get().ok());

    let symbols = trading::list_recent_symbols(&conn, candle_conn.as_deref(), interval, ts, 200)?;

    // Balance — prefer HL API snapshot for live mode, fall back to DB.
    let bal = trading::latest_balance(&conn)?;
    let db_realised_usd = bal.as_ref().map(|(b, _)| *b);
    let realised_asof = bal.as_ref().and_then(|(_, ts)| ts.clone());

    // Staleness threshold: if HL snapshot is older than 60s, consider it stale.
    const HL_STALE_SECS: u64 = 60;

    let hl_snap = if mode == "live" {
        let guard = state.hl_snapshot.read().await;
        guard.as_ref().and_then(|cached| {
            if cached.fetched_at.elapsed().as_secs() <= HL_STALE_SECS {
                Some(cached.snapshot.clone())
            } else {
                None
            }
        })
    } else {
        None
    };
    let db_live_snapshot = if mode == "live" {
        trading::latest_runtime_account_snapshot(&conn)?
    } else {
        None
    };
    let db_live_positions = if mode == "live" {
        trading::runtime_exchange_positions(&conn)?
    } else {
        Vec::new()
    };
    const DB_SYNC_STALE_MS: i64 = 2 * 60 * 60 * 1000 + 5 * 60 * 1000;
    let db_live_snapshot_fresh = db_live_snapshot
        .as_ref()
        .map(|snapshot| ts.saturating_sub(snapshot.ts_ms) <= DB_SYNC_STALE_MS)
        .unwrap_or(false);
    let live_positions = if let Some(ref hl) = hl_snap {
        live_position_overrides(mode.as_str(), Some(hl))
    } else if db_live_snapshot_fresh {
        live_position_overrides_from_db(mode.as_str(), &db_live_positions)
    } else {
        HashMap::new()
    };
    let live_snapshot_authoritative = hl_snap.is_some() || db_live_snapshot_fresh;
    let positions = merge_positions_with_live_snapshot(
        &ledger_positions,
        &live_positions,
        live_snapshot_authoritative,
    );

    // Merge: open positions first, then other symbols
    let mut merged: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for pos in &positions {
        if seen.insert(pos.symbol.clone()) {
            merged.push(pos.symbol.clone());
        }
    }
    for symbol in live_positions.keys() {
        if seen.insert(symbol.clone()) {
            merged.push(symbol.clone());
        }
    }
    for s in &symbols {
        if seen.insert(s.clone()) {
            merged.push(s.clone());
        }
    }
    merged.truncate(200);

    // Update the shared symbol list so the background mids poller broadcasts real prices.
    {
        let mut tracked = state.tracked_symbols.write().await;
        *tracked = merged.clone();
    }

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
            let live_override = live_position_override_for(p, &live_positions);
            let entry_price = live_override
                .map(|live| live.entry_price)
                .unwrap_or(p.entry_price);
            let size = live_override.map(|live| live.size).unwrap_or(p.size);
            let u = if p.pos_type == "LONG" {
                (m - entry_price) * size
            } else {
                (entry_price - m) * size
            };
            unreal_pnl = Some(u);
            unreal_total += u;
            close_fee_total += size.abs() * m * fee_rate;
            let lev = live_override
                .map(|live| live.leverage)
                .unwrap_or(p.leverage)
                .max(0.01);
            margin_used_total += size.abs() * m / lev;
        }

        let position_out = pos.map(|p| {
            let live_override = live_position_override_for(p, &live_positions);
            position_json(p, live_override, unreal_pnl)
        });

        symbols_out.push(json!({
            "symbol": sym,
            "mid": mid,
            "last_signal": last_signals.get(sym),
            "last_trade": last_trades.get(sym),
            "last_intent": last_intents.get(sym),
            "position": position_out,
        }));
    }

    let open_positions_out = positions
        .iter()
        .map(|position| {
            let live_override = live_position_override_for(position, &live_positions);
            position_json(position, live_override, None)
        })
        .collect::<Vec<_>>();

    let (realised_usd, equity_est, unreal_pnl_out, margin_used_out, balance_source) =
        if let Some(ref hl) = hl_snap {
            // HL account_value is equity (includes unrealised PnL).
            let unreal = hl.account_value - hl.withdrawable;
            (
                Some(hl.account_value),
                Some(hl.account_value),
                unreal,
                hl.total_margin_used,
                "hyperliquid",
            )
        } else if let Some(snapshot) = db_live_snapshot.filter(|_| db_live_snapshot_fresh) {
            let unreal = snapshot.account_value_usd - snapshot.withdrawable_usd;
            (
                Some(snapshot.account_value_usd),
                Some(snapshot.account_value_usd),
                unreal,
                snapshot.total_margin_used_usd,
                "db_exchange_snapshot",
            )
        } else {
            (
                db_realised_usd,
                db_realised_usd.map(|r| r + unreal_total - close_fee_total),
                unreal_total,
                margin_used_total,
                if mode == "live" {
                    "db_snapshot"
                } else {
                    "paper_estimate"
                },
            )
        };

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

    // Range metrics: since-config and all-time
    let deployed_config =
        deployed_config_boundary(&state.config.artifacts_dir, &mode, &conn, &heartbeat)?;

    let since_config = deployed_config
        .as_ref()
        .map(|(_, from_ts)| from_ts.as_str())
        .map(|ts| trading::range_metrics(&conn, Some(ts)))
        .transpose()?;
    let all_time = trading::range_metrics(&conn, None)?;

    // Available candle intervals
    let intervals = candles::list_available_intervals(&state.config.candles_db_dir);

    Ok(Json(json!({
        "now_ts_ms": ts,
        "mode": mode,
        "db_path_redacted": true,
        "health": redacted_heartbeat(&heartbeat),
        "config": {
            "trader_interval": state.config.trader_interval,
            "candle_intervals": intervals,
            "admin_actions_enabled": state.config.admin_actions_enabled,
            "live_service": state.config.live_service,
        },
        "balances": {
            "balance_source": balance_source,
            "realised_usd": realised_usd,
            "realised_asof": realised_asof,
            "equity_est_usd": equity_est,
            "unreal_pnl_est_usd": unreal_pnl_out,
            "est_close_fees_usd": close_fee_total,
            "margin_used_est_usd": if mode == "live" { Some(margin_used_out) } else { None::<f64> },
            "db_realised_usd": db_realised_usd,
            "fee_rate": fee_rate,
        },
        "daily": daily,
        "since_config": since_config.as_ref().map(|m| json!({
            "from_ts": m.from_ts,
            "config_id": deployed_config.as_ref().map(|(config_id, _)| config_id.clone()),
            "trades": m.trades,
            "start_balance": m.start_balance,
            "end_balance": m.end_balance,
            "pnl_usd": m.pnl_usd,
            "fees_usd": m.fees_usd,
            "net_pnl_usd": m.net_pnl_usd,
            "peak_balance": m.peak_balance,
            "drawdown_pct": m.drawdown_pct,
            "label": deployed_config.as_ref().map(|(config_id, _)| {
                format!("Since cfg {}", &config_id[..config_id.len().min(8)])
            }),
        })),
        "all_time": json!({
            "from_ts": all_time.from_ts,
            "trades": all_time.trades,
            "start_balance": all_time.start_balance,
            "end_balance": all_time.end_balance,
            "pnl_usd": all_time.pnl_usd,
            "fees_usd": all_time.fees_usd,
            "net_pnl_usd": all_time.net_pnl_usd,
            "peak_balance": all_time.peak_balance,
            "drawdown_pct": all_time.drawdown_pct,
        }),
        "symbols": symbols_out,
        "open_positions": open_positions_out,
        "recent": {
            "trades": recent_trades,
            "signals": recent_signals,
            "audit_events": redacted_recent_audit_events(audit_events),
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
    let interval = q
        .interval
        .as_deref()
        .unwrap_or(&state.config.trader_interval);
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

    let ledger_pos = trading::open_position_for_symbol(&conn, &sym)?;
    let entries = if let Some(ref p) = ledger_pos {
        if p.open_trade_id > 0 {
            trading::position_entries(&conn, &sym, p.open_trade_id)?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Fetch mid price and compute unrealised PnL
    let mid = state
        .sidecar
        .get_mids(std::slice::from_ref(&sym))
        .await
        .ok()
        .and_then(|s| s.mids.get(&sym).copied());

    let hl_snap = if mode == "live" {
        let guard = state.hl_snapshot.read().await;
        guard.as_ref().and_then(|cached| {
            if cached.fetched_at.elapsed().as_secs() <= 60 {
                Some(cached.snapshot.clone())
            } else {
                None
            }
        })
    } else {
        None
    };
    let db_live_snapshot = if mode == "live" {
        trading::latest_runtime_account_snapshot(&conn)?
    } else {
        None
    };
    let db_live_positions = if mode == "live" {
        trading::runtime_exchange_positions(&conn)?
    } else {
        Vec::new()
    };
    const DB_SYNC_STALE_MS: i64 = 2 * 60 * 60 * 1000 + 5 * 60 * 1000;
    let db_live_snapshot_fresh = db_live_snapshot
        .as_ref()
        .map(|snapshot| now_ms().saturating_sub(snapshot.ts_ms) <= DB_SYNC_STALE_MS)
        .unwrap_or(false);
    let live_positions = if let Some(ref hl) = hl_snap {
        live_position_overrides(mode.as_str(), Some(hl))
    } else if db_live_snapshot_fresh {
        live_position_overrides_from_db(mode.as_str(), &db_live_positions)
    } else {
        HashMap::new()
    };
    let live_snapshot_authoritative = hl_snap.is_some() || db_live_snapshot_fresh;
    let pos = if live_snapshot_authoritative {
        match (ledger_pos, live_positions.get(&sym)) {
            (Some(mut ledger), Some(live)) if live.pos_type.eq_ignore_ascii_case(&ledger.pos_type) => {
                ledger.entry_price = live.entry_price;
                ledger.size = live.size;
                ledger.leverage = live.leverage.max(0.01);
                ledger.margin_used = live.margin_used.max(0.0);
                Some(ledger)
            }
            (_, Some(live)) => Some(synthetic_open_position_from_live_snapshot(live)),
            _ => None,
        }
    } else {
        ledger_pos.or_else(|| {
            live_positions
                .get(&sym)
                .map(synthetic_open_position_from_live_snapshot)
        })
    };

    let position_json = pos.as_ref().map(|p| {
        let live_override = live_position_override_for(p, &live_positions);
        let entry_price = live_override
            .map(|live| live.entry_price)
            .unwrap_or(p.entry_price);
        let size = live_override.map(|live| live.size).unwrap_or(p.size);
        let unreal_pnl = mid.map(|m| {
            if p.pos_type == "LONG" {
                (m - entry_price) * size
            } else {
                (entry_price - m) * size
            }
        });
        position_json(p, live_override, unreal_pnl)
    });

    Ok(Json(json!({
        "ok": true,
        "now_ts_ms": now_ms(),
        "mode": mode,
        "symbol": sym,
        "mid": mid,
        "position": position_json,
        "entries": entries,
    })))
}

async fn api_journeys(
    State(state): State<Arc<AppState>>,
    Query(q): Query<JourneyQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let limit = q.limit.clamp(1, 200);
    let offset = q.offset;

    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;

    let sym_filter = q.symbol.as_deref().map(|s| s.to_uppercase());
    let journeys = trading::trade_journeys(&conn, limit, offset, sym_filter.as_deref())?;

    Ok(Json(json!({
        "ok": true,
        "mode": mode,
        "count": journeys.len(),
        "offset": offset,
        "journeys": journeys,
    })))
}

async fn api_trades(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TradesQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let limit = q.limit.clamp(1, 500);
    let offset = q.offset;

    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;

    let result = trading::paginated_trades(
        &conn,
        limit,
        offset,
        q.symbol.as_deref(),
        q.action.as_deref(),
        q.from_ts.as_deref(),
        q.to_ts.as_deref(),
    )?;

    Ok(Json(json!({
        "ok": true,
        "mode": mode,
        "total": result.total,
        "offset": offset,
        "summary_pnl": result.summary_pnl,
        "summary_fees": result.summary_fees,
        "trades": result.trades,
    })))
}

async fn api_candles_range(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CandleRangeQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let sym = q.symbol.to_uppercase();
    let interval = q
        .interval
        .as_deref()
        .unwrap_or(&state.config.trader_interval);
    let limit = q.limit.clamp(2, 2000);

    let candle_path = state.candle_db_path(interval);
    let candle_pool = open_ro_pool(&candle_path, 2);

    if let Some(pool) = &candle_pool {
        if let Ok(conn) = pool.get() {
            let data =
                candles::fetch_candles_range(&conn, &sym, interval, q.from_ts, q.to_ts, limit)?;
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

    // Fallback to trading DB
    let pool = state
        .db_pool(&mode)
        .ok_or_else(|| HubError::Db("db not available".to_string()))?;
    let conn = pool.get()?;
    let data = candles::fetch_candles_range(&conn, &sym, interval, q.from_ts, q.to_ts, limit)?;
    Ok(Json(json!({
        "ok": true,
        "symbol": sym,
        "interval": interval,
        "count": data.len(),
        "candles": data,
    })))
}

async fn api_trend_closes(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TrendClosesQuery>,
) -> Result<Json<Value>, HubError> {
    let interval = q.interval;
    let limit = q.limit.clamp(2, 200);

    let tracked = state.tracked_symbols.read().await;
    let symbols: Vec<String> = tracked.clone();
    drop(tracked);

    let candle_path = state.candle_db_path(&interval);
    let pool = open_ro_pool(&candle_path, 2)
        .ok_or_else(|| HubError::Db(format!("candle db not available for {interval}")))?;
    let conn = pool.get()?;

    let data = candles::fetch_recent_closes_batch(&conn, &symbols, &interval, limit)?;
    Ok(Json(
        json!({ "interval": interval, "limit": limit, "closes": data }),
    ))
}

async fn api_trend_candles(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TrendClosesQuery>,
) -> Result<Json<Value>, HubError> {
    let interval = q.interval;
    let limit = q.limit.clamp(2, 200);

    let tracked = state.tracked_symbols.read().await;
    let symbols: Vec<String> = tracked.clone();
    drop(tracked);

    let candle_path = state.candle_db_path(&interval);
    let pool = open_ro_pool(&candle_path, 2)
        .ok_or_else(|| HubError::Db(format!("candle db not available for {interval}")))?;
    let conn = pool.get()?;

    let data = candles::fetch_recent_candles_batch(&conn, &symbols, &interval, limit)?;
    Ok(Json(
        json!({ "interval": interval, "limit": limit, "candles": data }),
    ))
}

async fn api_volumes(State(state): State<Arc<AppState>>) -> Result<Json<Value>, HubError> {
    let interval = "1h";
    let cutoff_ms = now_ms() - 24 * 60 * 60 * 1000;

    let candle_path = state.candle_db_path(interval);
    let pool = open_ro_pool(&candle_path, 2)
        .ok_or_else(|| HubError::Db(format!("candle db not available for {interval}")))?;
    let conn = pool.get()?;

    let volumes = candles::fetch_24h_volumes(&conn, interval, cutoff_ms)?;
    Ok(Json(json!({ "volumes": volumes })))
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
    let kill_mode = heartbeat
        .kill_mode
        .as_deref()
        .unwrap_or("off")
        .to_lowercase();
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
    lines.push(prom_line(
        "aiq_engine_up",
        if heartbeat.ok { 1.0 } else { 0.0 },
        &mode,
    ));

    if let Some(ts_hb) = heartbeat.ts_ms {
        let age = ((ts - ts_hb) as f64 / 1000.0).max(0.0);
        lines.push(prom_line("aiq_heartbeat_age_s", age, &mode));
    }
    if let Some(n) = heartbeat.open_pos {
        lines.push(prom_line("aiq_open_pos", n as f64, &mode));
    }

    // Kill mode
    let kill_mode = heartbeat
        .kill_mode
        .as_deref()
        .unwrap_or("off")
        .to_lowercase();
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
        lines.push(prom_line(
            "aiq_drawdown_today_pct",
            daily.drawdown_pct,
            &mode,
        ));
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
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body)
}

fn prom_line(name: &str, value: f64, mode: &str) -> String {
    format!("{name}{{mode=\"{mode}\"}} {value}\n")
}

async fn api_tunnel(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TunnelQuery>,
) -> Result<Json<Value>, HubError> {
    let mode = normalize_mode(&q.mode);
    let pool = match state.db_pool(&mode) {
        Some(p) => p,
        None => return Ok(Json(json!({ "tunnel": [] }))),
    };
    let conn = pool.get()?;
    let symbol = q.symbol.trim().to_uppercase();
    let limit = q.limit.min(10_000);
    let points = tunnel::fetch_tunnel(&conn, &symbol, q.from_ts, q.to_ts, limit)?;
    Ok(Json(json!({ "tunnel": points })))
}

pub fn normalize_mode(mode: &str) -> String {
    let m = mode.trim().to_lowercase();
    match m.as_str() {
        "live" | "paper1" | "paper2" | "paper3" => m,
        "paper" => "paper1".to_string(),
        _ => "paper1".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_audit::{
        append_event, ConfigAuditActor, ConfigAuditEvent, ConfigAuditIdentity,
    };
    use rusqlite::Connection;
    use tempfile::tempdir;

    #[test]
    fn deployed_config_boundary_uses_heartbeat_config_id() {
        let dir = tempdir().unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE runtime_logs (
                ts_ms INTEGER NOT NULL,
                message TEXT NOT NULL
            )",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_logs (ts_ms, message) VALUES (?1, ?2)",
            rusqlite::params![
                1_710_000_000_000_i64,
                "engine ok config_id=abcdef0123456789"
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_logs (ts_ms, message) VALUES (?1, ?2)",
            rusqlite::params![
                1_710_000_600_000_i64,
                "engine ok config_id=abcdef0123456789"
            ],
        )
        .unwrap();

        let heartbeat = Heartbeat {
            config_id: Some("abcdef0123456789".to_string()),
            ..Default::default()
        };

        let (config_id, from_ts) = deployed_config_boundary(dir.path(), "live", &conn, &heartbeat)
            .unwrap()
            .unwrap();

        assert_eq!(config_id, "abcdef0123456789");
        assert_eq!(from_ts, "2024-03-09T16:00:00");
    }

    #[test]
    fn deployed_config_boundary_prefers_latest_matching_audit_event() {
        let dir = tempdir().unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE runtime_logs (
                ts_ms INTEGER NOT NULL,
                message TEXT NOT NULL
            )",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_logs (ts_ms, message) VALUES (?1, ?2)",
            rusqlite::params![
                1_710_000_000_000_i64,
                "engine ok config_id=abcdef0123456789"
            ],
        )
        .unwrap();

        append_event(
            dir.path(),
            &ConfigAuditEvent {
                version: "config_audit_event_v1".to_string(),
                ts_ms: 1_710_005_000_000_i64,
                ts_utc: "2024-03-09T17:23:20".to_string(),
                lane: "live".to_string(),
                file_variant: "live".to_string(),
                action: "apply_live".to_string(),
                actor: ConfigAuditActor {
                    auth_scope: "admin_token".to_string(),
                    label: "operator".to_string(),
                    source_ip: None,
                    user_agent: None,
                },
                reason: Some("deploy".to_string()),
                validation: json!({ "ok": true }),
                before: ConfigAuditIdentity {
                    lock_id: None,
                    config_id: Some("old".to_string()),
                },
                after: ConfigAuditIdentity {
                    lock_id: None,
                    config_id: Some("abcdef0123456789".to_string()),
                },
                result: json!({ "ok": true }),
                artifact_path: None,
            },
        )
        .unwrap();

        let heartbeat = Heartbeat {
            config_id: Some("abcdef0123456789".to_string()),
            ..Default::default()
        };

        let (_config_id, from_ts) = deployed_config_boundary(dir.path(), "live", &conn, &heartbeat)
            .unwrap()
            .unwrap();

        assert_eq!(from_ts, "2024-03-09T17:23:20");
    }

    #[test]
    fn merge_positions_with_live_snapshot_clears_stale_ledger_when_snapshot_is_authoritative() {
        let ledger_positions = vec![trading::OpenPosition {
            symbol: "AXS".to_string(),
            pos_type: "SHORT".to_string(),
            open_trade_id: 42,
            open_timestamp: Some("2026-03-03T22:59:03.822+00:00".to_string()),
            entry_price: 1.2149,
            size: 109.6,
            confidence: Some("MANUAL".to_string()),
            entry_atr: 0.0,
            leverage: 4.0,
            margin_used: 33.3,
        }];
        let live_positions = HashMap::<String, HlPositionSnapshot>::new();

        let merged = merge_positions_with_live_snapshot(&ledger_positions, &live_positions, true);
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_positions_with_live_snapshot_uses_live_values_for_matching_symbol() {
        let ledger_positions = vec![trading::OpenPosition {
            symbol: "DOGE".to_string(),
            pos_type: "LONG".to_string(),
            open_trade_id: 11904,
            open_timestamp: Some("2026-03-11T17:31:59.945000+00:00".to_string()),
            entry_price: 0.093828,
            size: 1036.0,
            confidence: Some("MANUAL".to_string()),
            entry_atr: 0.0,
            leverage: 4.0,
            margin_used: 24.0,
        }];
        let live_positions = HashMap::from([(
            "DOGE".to_string(),
            HlPositionSnapshot {
                symbol: "DOGE".to_string(),
                pos_type: "LONG".to_string(),
                size: 1692.0,
                entry_price: 0.094602,
                leverage: 4.0,
                margin_used: 40.112244,
            },
        )]);

        let merged = merge_positions_with_live_snapshot(&ledger_positions, &live_positions, true);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].open_trade_id, 11904);
        assert!((merged[0].size - 1692.0).abs() < 1e-9);
        assert!((merged[0].entry_price - 0.094602).abs() < 1e-9);
    }
}
