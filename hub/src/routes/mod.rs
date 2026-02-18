pub mod monitor;
pub mod decisions;
pub mod config;
pub mod backtest;
pub mod sweep;

use axum::Router;
use std::sync::Arc;

use crate::state::AppState;

/// Assemble the API router.
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(monitor::routes())
        .merge(decisions::routes())
        .merge(config::routes())
        .merge(backtest::routes())
        .merge(sweep::routes())
}
