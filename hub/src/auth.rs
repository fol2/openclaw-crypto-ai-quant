use anyhow::{bail, Result};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::config::HubConfig;

#[derive(Clone, Debug, Default)]
pub struct HubAuthConfig {
    pub read_token: String,
    pub admin_token: String,
    pub dev_mode: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthScope {
    Read,
    Admin,
}

impl HubAuthConfig {
    pub fn from_hub_config(config: &HubConfig) -> Self {
        Self {
            read_token: config.token.clone(),
            admin_token: config.admin_token.clone(),
            dev_mode: config.dev_mode,
        }
    }

    pub fn validate_startup(&self) -> Result<()> {
        if !self.dev_mode && self.read_token.is_empty() {
            bail!("AIQ_MONITOR_TOKEN is required unless AIQ_MONITOR_DEV_MODE=1 is set explicitly");
        }
        if !self.read_token.is_empty()
            && !self.admin_token.is_empty()
            && self.read_token == self.admin_token
        {
            bail!("AIQ_MONITOR_ADMIN_TOKEN must differ from AIQ_MONITOR_TOKEN");
        }
        Ok(())
    }
}

pub async fn require_read_auth(request: Request, next: Next) -> Response {
    require_scope(request, next, AuthScope::Read).await
}

pub async fn require_admin_auth(request: Request, next: Next) -> Response {
    require_scope(request, next, AuthScope::Admin).await
}

async fn require_scope(request: Request, next: Next, scope: AuthScope) -> Response {
    let config = request
        .extensions()
        .get::<HubAuthConfig>()
        .cloned()
        .unwrap_or_default();

    let expected_token = match scope {
        AuthScope::Read => {
            if config.read_token.is_empty() {
                if config.dev_mode {
                    return next.run(request).await;
                }
                return auth_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "read auth is not configured",
                );
            }
            config.read_token.as_str()
        }
        AuthScope::Admin => {
            if config.admin_token.is_empty() {
                return auth_error(StatusCode::FORBIDDEN, "admin auth is not configured");
            }
            config.admin_token.as_str()
        }
    };

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    match scope {
        AuthScope::Read => {
            if matches_query_token(&request, expected_token)
                || matches_bearer_token(auth_header, expected_token)
                || (!config.admin_token.is_empty()
                    && (matches_query_token(&request, &config.admin_token)
                        || matches_bearer_token(auth_header, &config.admin_token)))
            {
                return next.run(request).await;
            }
        }
        AuthScope::Admin => {
            if matches_bearer_token(auth_header, expected_token) {
                return next.run(request).await;
            }
        }
    }

    auth_error(StatusCode::UNAUTHORIZED, "unauthorized")
}

fn auth_error(status: StatusCode, message: &str) -> Response {
    let body = json!({ "error": message });
    (status, axum::Json(body)).into_response()
}

fn matches_bearer_token(auth_header: &str, token: &str) -> bool {
    let expected_header = format!("Bearer {token}");
    constant_time_eq(auth_header.as_bytes(), expected_header.as_bytes())
}

fn matches_query_token(request: &Request, token: &str) -> bool {
    if request.uri().path() != "/ws" || token.is_empty() {
        return false;
    }
    request
        .uri()
        .query()
        .and_then(extract_query_token)
        .map(|query_token| constant_time_eq(query_token.as_bytes(), token.as_bytes()))
        .unwrap_or(false)
}

fn extract_query_token(query: &str) -> Option<&str> {
    query.split('&').find_map(|segment| {
        let (key, value) = segment.split_once('=')?;
        (key == "token" && !value.is_empty()).then_some(value)
    })
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::middleware;
    use axum::routing::get;
    use axum::Router;
    use tower::util::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    fn read_app(config: HubAuthConfig) -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(config))
    }

    fn admin_app(config: HubAuthConfig) -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .route_layer(middleware::from_fn(require_admin_auth))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(config))
    }

    #[tokio::test]
    async fn startup_validation_requires_read_token_outside_dev_mode() {
        let config = HubAuthConfig {
            read_token: String::new(),
            admin_token: String::new(),
            dev_mode: false,
        };

        let err = config.validate_startup().unwrap_err();
        assert!(err.to_string().contains("AIQ_MONITOR_TOKEN"));
    }

    #[tokio::test]
    async fn read_auth_allows_dev_mode_without_token() {
        let response = read_app(HubAuthConfig {
            read_token: String::new(),
            admin_token: String::new(),
            dev_mode: true,
        })
        .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn startup_validation_rejects_shared_read_and_admin_tokens() {
        let config = HubAuthConfig {
            read_token: "shared-secret".to_string(),
            admin_token: "shared-secret".to_string(),
            dev_mode: false,
        };

        let err = config.validate_startup().unwrap_err();
        assert!(err.to_string().contains("AIQ_MONITOR_ADMIN_TOKEN"));
    }

    #[tokio::test]
    async fn admin_auth_requires_dedicated_admin_token() {
        let response = admin_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("authorization", "Bearer viewer-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_token_can_pass_the_global_read_gate() {
        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("authorization", "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn admin_auth_rejects_when_admin_token_is_not_configured() {
        let response = admin_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: String::new(),
            dev_mode: true,
        })
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("authorization", "Bearer viewer-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn websocket_query_token_can_pass_the_read_gate() {
        let app = Router::new()
            .route("/ws", get(ok_handler))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(HubAuthConfig {
                read_token: "viewer-secret".to_string(),
                admin_token: "admin-secret".to_string(),
                dev_mode: false,
            }));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ws?token=viewer-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
