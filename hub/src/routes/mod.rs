pub mod monitor;

use axum::Router;
use std::sync::Arc;

use crate::state::AppState;

/// Assemble the API router.
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(monitor::routes())
}
