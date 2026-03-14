mod auth;
mod config;
mod db;
mod error;
mod factory_capability;
mod heartbeat;
mod hyperliquid;
#[path = "../../runtime/aiq-runtime/src/live_hyperliquid.rs"]
mod live_hyperliquid;
#[allow(dead_code)]
#[path = "../../runtime/aiq-runtime/src/live_risk.rs"]
mod live_risk;
#[allow(dead_code)]
#[path = "../../runtime/aiq-runtime/src/live_safety.rs"]
mod live_safety;
#[path = "../../runtime/aiq-runtime/src/live_secrets.rs"]
mod live_secrets;
mod manual_trade;
mod routes;
mod sidecar;
mod state;
mod subprocess;
#[cfg(test)]
#[path = "../../runtime/aiq-runtime/src/test_support.rs"]
mod test_support;
mod ws;

use axum::http::{header, HeaderValue, Method};
use axum::middleware;
use axum::Router;
use chrono::Utc;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::EnvFilter;

use auth::HubAuthConfig;
use config::HubConfig;
use state::AppState;

#[tokio::main]
async fn main() {
    // Initialise tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cfg = HubConfig::from_env();
    let bind = cfg.bind.clone();
    let port = cfg.port;
    let auth_config = HubAuthConfig::from_hub_config(&cfg);
    auth_config
        .validate_startup()
        .expect("invalid Hub auth configuration");
    let cors = build_cors_layer(&cfg).expect("invalid Hub CORS configuration");

    let state = AppState::new(cfg);

    // Start background mids poller.
    spawn_mids_poller(Arc::clone(&state));

    // Start background HL balance poller (live mode only).
    spawn_hl_poller(Arc::clone(&state));

    // Start background manual-trade reconcile poller.
    spawn_manual_trade_reconcile_poller(Arc::clone(&state));

    // Build router.
    let api = routes::api_router();

    // Static file serving: serve frontend/dist/ at root.
    // During development, you can run `npm run dev` in frontend/ instead.
    let static_dir = std::env::current_dir()
        .unwrap_or_default()
        .join("hub/frontend/dist");

    let app = Router::new()
        .merge(api)
        .route("/ws", axum::routing::get(ws::ws_handler))
        .route("/health", axum::routing::get(health))
        .fallback_service(ServeDir::new(&static_dir).append_index_html_on_directories(true))
        .layer(middleware::from_fn(auth::require_read_auth))
        .layer(axum::Extension(auth_config))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{bind}:{port}")
        .parse()
        .expect("invalid bind address");

    tracing::info!("AQC Hub listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

fn build_cors_layer(config: &HubConfig) -> Result<CorsLayer, String> {
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]);

    if config.cors_allowed_origins.is_empty() {
        return Ok(layer);
    }

    let allowed_origins = config
        .cors_allowed_origins
        .iter()
        .map(|origin| {
            HeaderValue::from_str(origin)
                .map_err(|err| format!("invalid CORS origin {origin:?}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(layer.allow_origin(allowed_origins))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::util::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn cors_denies_cross_origin_requests_by_default() {
        let mut config = HubConfig::from_env();
        config.cors_allowed_origins = Vec::new();
        let app = Router::new()
            .route("/test", axum::routing::get(ok_handler))
            .layer(build_cors_layer(&config).unwrap());

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(header::ORIGIN, "https://example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none());
    }

    #[tokio::test]
    async fn cors_allows_configured_origins_only() {
        let mut config = HubConfig::from_env();
        config.cors_allowed_origins = vec!["https://console.example".to_string()];
        let app = Router::new()
            .route("/test", axum::routing::get(ok_handler))
            .layer(build_cors_layer(&config).unwrap());

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(header::ORIGIN, "https://console.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            "https://console.example"
        );
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("Shutdown signal received, gracefully stopping…");
}

/// Background task: bridge sidecar mids to frontend WS topic.
///
/// Preferred mode uses sidecar `wait_mids` so updates are push-driven instead of
/// fixed-cycle polling. If connected to an older sidecar build, it gracefully
/// falls back to `get_mids` with `mids_poll_ms` cadence.
fn spawn_mids_poller(state: Arc<AppState>) {
    let poll_ms = state.config.mids_poll_ms.max(20);
    let wait_timeout_ms = state.config.mids_wait_timeout_ms.max(100);
    let mids_debug_log = state.config.mids_debug_log;
    tokio::spawn(async move {
        let mut after_seq: Option<u64> = None;
        let mut after_bbo_seq: Option<u64> = None;
        let mut last_published_mids: HashMap<String, f64> = HashMap::new();
        let mut last_published_bbo: HashMap<String, sidecar::BboQuote> = HashMap::new();
        loop {
            // Read the symbols last seen in a snapshot query.
            // Skips until the first REST snapshot has been fetched.
            let symbols: Vec<String> = {
                let tracked = state.tracked_symbols.read().await;
                if tracked.is_empty() {
                    tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
                    continue;
                }
                tracked.clone()
            };

            match state
                .sidecar
                .wait_mids(&symbols, after_seq, after_bbo_seq, wait_timeout_ms)
                .await
            {
                Ok(snap) => {
                    if let Some(seq) = snap.mids_seq {
                        after_seq = Some(seq);
                    }
                    if let Some(seq) = snap.bbo_seq {
                        after_bbo_seq = Some(seq);
                    }
                    // Timeout heartbeats do not carry new price data.
                    if !snap.changed {
                        continue;
                    }
                    if snap.seq_reset {
                        tracing::debug!("sidecar mids sequence reset detected; re-synchronised");
                    }
                    let mids_changed = snap.seq_reset || snap.mids != last_published_mids;
                    let bbo_changed = snap.seq_reset || snap.bbo != last_published_bbo;
                    let mut changed_symbols: Vec<String> = Vec::new();
                    let mut changed_bbo_symbols: Vec<String> = Vec::new();
                    if mids_debug_log {
                        for (sym, mid) in &snap.mids {
                            let is_changed = match last_published_mids.get(sym) {
                                Some(prev) => *prev != *mid,
                                None => true,
                            };
                            if is_changed {
                                changed_symbols.push(sym.clone());
                            }
                        }
                        for (sym, quote) in &snap.bbo {
                            let is_changed = match last_published_bbo.get(sym) {
                                Some(prev) => *prev != *quote,
                                None => true,
                            };
                            if is_changed {
                                changed_bbo_symbols.push(sym.clone());
                            }
                        }
                        changed_symbols.sort_unstable();
                        changed_bbo_symbols.sort_unstable();
                    }
                    last_published_mids = snap.mids.clone();
                    last_published_bbo = snap.bbo.clone();
                    let mut ws_mids_receivers = 0usize;
                    let mut ws_bbo_receivers = 0usize;
                    let mut payload_data =
                        serde_json::to_value(&snap).unwrap_or_else(|_| serde_json::json!({}));
                    if let Some(obj) = payload_data.as_object_mut() {
                        obj.insert(
                            "server_ts_ms".to_string(),
                            serde_json::json!(Utc::now().timestamp_millis()),
                        );
                    }
                    if mids_changed {
                        if let Ok(json) = serde_json::to_string(&serde_json::json!({
                            "type": "mids",
                            "data": payload_data.clone(),
                        })) {
                            ws_mids_receivers =
                                state.broadcast.publish(ws::topics::TOPIC_MIDS, json);
                        }
                    }
                    if bbo_changed {
                        if let Ok(json) = serde_json::to_string(&serde_json::json!({
                            "type": "bbo",
                            "data": payload_data,
                        })) {
                            ws_bbo_receivers = state.broadcast.publish(ws::topics::TOPIC_BBO, json);
                        }
                    }
                    if mids_debug_log {
                        let sample: Vec<String> =
                            changed_symbols.iter().take(20).cloned().collect();
                        let bbo_sample: Vec<String> =
                            changed_bbo_symbols.iter().take(20).cloned().collect();
                        tracing::info!(
                            mids_seq = ?snap.mids_seq,
                            bbo_seq = ?snap.bbo_seq,
                            changed_symbols = changed_symbols.len(),
                            changed_bbo_symbols = changed_bbo_symbols.len(),
                            tracked_symbols = snap.mids.len(),
                            tracked_bbo_symbols = snap.bbo.len(),
                            ws_mids_receivers,
                            ws_bbo_receivers,
                            timed_out = snap.timed_out,
                            seq_reset = snap.seq_reset,
                            sample = ?sample,
                            bbo_sample = ?bbo_sample,
                            "mids publish"
                        );
                    }
                    // Old sidecar fallback path has no sequence support; throttle to poll cadence.
                    if after_seq.is_none() && after_bbo_seq.is_none() {
                        tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
                    }
                }
                Err(e) => {
                    tracing::debug!("Mids poll failed: {e}");
                    // Recover cleanly after sidecar reconnects/restarts.
                    after_seq = None;
                    after_bbo_seq = None;
                    last_published_mids.clear();
                    last_published_bbo.clear();
                    tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
                }
            }
        }
    });
}

/// Background task: poll Hyperliquid clearinghouse API for live account balance.
///
/// Only runs when `hl_balance_enable` is true and `hl_main_address` is configured.
/// Caches the result in `AppState::hl_snapshot` for `/api/snapshot?mode=live`.
fn spawn_hl_poller(state: Arc<AppState>) {
    let enabled = state.config.hl_balance_enable;
    let address = match &state.config.hl_main_address {
        Some(addr) if enabled => addr.clone(),
        _ => {
            if enabled {
                tracing::warn!(
                    "HL balance polling enabled but no address configured \
                     (set AIQ_MONITOR_HL_MAIN_ADDRESS or main_address in secrets.json)"
                );
            }
            return;
        }
    };

    tracing::info!(
        "HL balance poller starting (address={}…)",
        &address[..8.min(address.len())]
    );

    tokio::spawn(async move {
        let poll_interval = std::time::Duration::from_secs(10);
        loop {
            match hyperliquid::fetch_account_snapshot(&state.hl_client, &address).await {
                Some(snap) => {
                    tracing::debug!(
                        account_value = snap.account_value,
                        withdrawable = snap.withdrawable,
                        margin_used = snap.total_margin_used,
                        "HL snapshot updated"
                    );
                    let mut guard = state.hl_snapshot.write().await;
                    *guard = Some(state::CachedHlSnapshot {
                        snapshot: snap,
                        fetched_at: std::time::Instant::now(),
                    });
                }
                None => {
                    // Keep stale data rather than clearing — better to show slightly
                    // old values than nothing.
                    tracing::debug!("HL snapshot fetch failed, keeping previous value");
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    });
}

fn spawn_manual_trade_reconcile_poller(state: Arc<AppState>) {
    if !state.config.manual_trade_enabled {
        return;
    }
    let cfg = state.config.clone();
    tokio::spawn(async move {
        let poll_interval = std::time::Duration::from_secs(30);
        loop {
            match tokio::task::spawn_blocking({
                let cfg = cfg.clone();
                move || manual_trade::reconcile_manual_intents(&cfg, 20)
            })
            .await
            {
                Ok(Ok(summary)) => {
                    let recovered = summary
                        .get("recovered_fills")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0)
                        + summary
                            .get("recovered_open_orders")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0)
                        + summary
                            .get("tracked_open_orders")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0)
                        + summary
                            .get("cleared_open_orders")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0)
                        + summary
                            .get("status_updates")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0);
                    if recovered > 0 {
                        tracing::info!(summary = %summary, "manual reconcile applied");
                    }
                }
                Ok(Err(error)) => {
                    tracing::warn!(error = %error, "manual reconcile failed");
                }
                Err(error) => {
                    tracing::warn!(error = %error, "manual reconcile task failed");
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    });
}
