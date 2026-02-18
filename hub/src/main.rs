mod auth;
mod config;
mod db;
mod error;
mod heartbeat;
mod routes;
mod sidecar;
mod state;
mod subprocess;
mod ws;

use axum::middleware;
use axum::Router;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::EnvFilter;

use auth::AuthToken;
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
    let token = cfg.token.clone();

    let state = AppState::new(cfg);

    // Start background mids poller.
    spawn_mids_poller(Arc::clone(&state));

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
        .layer(middleware::from_fn(auth::require_auth))
        .layer(axum::Extension(AuthToken(token)))
        .layer(CorsLayer::permissive())
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
        let mut last_published_mids: HashMap<String, f64> = HashMap::new();
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
                .wait_mids(&symbols, after_seq, wait_timeout_ms)
                .await
            {
                Ok(snap) => {
                    if let Some(seq) = snap.mids_seq {
                        after_seq = Some(seq);
                    }
                    // Timeout heartbeats do not carry new price data.
                    if !snap.changed {
                        continue;
                    }
                    if snap.seq_reset {
                        tracing::debug!("sidecar mids sequence reset detected; re-synchronised");
                    }
                    if mids_debug_log {
                        let mut changed_symbols: Vec<String> = Vec::new();
                        for (sym, mid) in &snap.mids {
                            let is_changed = match last_published_mids.get(sym) {
                                Some(prev) => *prev != *mid,
                                None => true,
                            };
                            if is_changed {
                                changed_symbols.push(sym.clone());
                            }
                        }
                        changed_symbols.sort_unstable();
                        let sample: Vec<String> = changed_symbols.iter().take(20).cloned().collect();
                        tracing::info!(
                            mids_seq = ?snap.mids_seq,
                            changed_symbols = changed_symbols.len(),
                            tracked_symbols = snap.mids.len(),
                            timed_out = snap.timed_out,
                            seq_reset = snap.seq_reset,
                            sample = ?sample,
                            "mids publish"
                        );
                    }
                    last_published_mids.clear();
                    last_published_mids.extend(snap.mids.iter().map(|(sym, mid)| (sym.clone(), *mid)));
                    if let Ok(json) = serde_json::to_string(&serde_json::json!({
                        "type": "mids",
                        "data": snap,
                    })) {
                        state.broadcast.publish(ws::topics::TOPIC_MIDS, json);
                    }
                    // Old sidecar fallback path has no mids sequence; throttle to poll cadence.
                    if after_seq.is_none() {
                        tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
                    }
                }
                Err(e) => {
                    tracing::debug!("Mids poll failed: {e}");
                    // Recover cleanly after sidecar reconnects/restarts.
                    after_seq = None;
                    last_published_mids.clear();
                    tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
                }
            }
        }
    });
}
