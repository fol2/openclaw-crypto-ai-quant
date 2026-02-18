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

/// Background task: poll sidecar for mid prices and broadcast to WS clients.
fn spawn_mids_poller(state: Arc<AppState>) {
    let poll_ms = state.config.mids_poll_ms;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(poll_ms));
        loop {
            interval.tick().await;

            // Gather tracked symbols from recent DB queries (simplified: just use BTC for now).
            let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];

            match state.sidecar.get_mids(&symbols).await {
                Ok(snap) => {
                    if let Ok(json) = serde_json::to_string(&serde_json::json!({
                        "type": "mids",
                        "data": snap,
                    })) {
                        state.broadcast.publish(ws::topics::TOPIC_MIDS, json);
                    }
                }
                Err(e) => {
                    tracing::debug!("Mids poll failed: {e}");
                }
            }
        }
    });
}
