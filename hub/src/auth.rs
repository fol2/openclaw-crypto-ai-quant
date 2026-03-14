use anyhow::{bail, Result};
use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use crate::config::HubConfig;

#[derive(Clone, Debug, Default)]
pub struct HubAuthConfig {
    pub read_token: String,
    pub admin_token: String,
    pub editor_token: String,
    pub approver_token: String,
    pub dev_mode: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthScope {
    Read,
    Admin,
    Editor,
    Approver,
}

impl HubAuthConfig {
    pub fn from_hub_config(config: &HubConfig) -> Self {
        Self {
            read_token: config.token.clone(),
            admin_token: config.admin_token.clone(),
            editor_token: config.editor_token.clone(),
            approver_token: config.approver_token.clone(),
            dev_mode: config.dev_mode,
        }
    }

    pub fn validate_startup(&self) -> Result<()> {
        if !self.dev_mode && self.read_token.is_empty() {
            bail!("AIQ_MONITOR_TOKEN is required unless AIQ_MONITOR_DEV_MODE=1 is set explicitly");
        }
        let configured = [
            ("AIQ_MONITOR_TOKEN", self.read_token.as_str()),
            ("AIQ_MONITOR_ADMIN_TOKEN", self.admin_token.as_str()),
            ("AIQ_MONITOR_EDITOR_TOKEN", self.editor_token.as_str()),
            ("AIQ_MONITOR_APPROVER_TOKEN", self.approver_token.as_str()),
        ];
        for (index, (lhs_name, lhs_value)) in configured.iter().enumerate() {
            if lhs_value.is_empty() {
                continue;
            }
            for (rhs_name, rhs_value) in configured.iter().skip(index + 1) {
                if !rhs_value.is_empty() && lhs_value == rhs_value {
                    bail!("{lhs_name} must differ from {rhs_name}");
                }
            }
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

pub async fn require_editor_auth(request: Request, next: Next) -> Response {
    require_scope(request, next, AuthScope::Editor).await
}

pub async fn require_approver_auth(request: Request, next: Next) -> Response {
    require_scope(request, next, AuthScope::Approver).await
}

pub async fn require_editor_or_approver_auth(request: Request, next: Next) -> Response {
    require_any_scope(request, next, &[AuthScope::Editor, AuthScope::Approver]).await
}

async fn require_scope(request: Request, next: Next, scope: AuthScope) -> Response {
    require_any_scope(request, next, &[scope]).await
}

async fn require_any_scope(request: Request, next: Next, scopes: &[AuthScope]) -> Response {
    let config = request
        .extensions()
        .get::<HubAuthConfig>()
        .cloned()
        .unwrap_or_default();

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");

    if scopes == [AuthScope::Read] && config.read_token.is_empty() && config.dev_mode {
        return next.run(request).await;
    }

    if scopes == [AuthScope::Read] && read_auth_bypassed(&request) {
        return next.run(request).await;
    }

    if scopes
        .iter()
        .filter_map(|scope| expected_token_for_scope(&config, *scope))
        .any(|token| {
            matches_bearer_token(auth_header, token)
                || (scopes == [AuthScope::Read] && matches_query_token(&request, token))
        })
    {
        return next.run(request).await;
    }

    if scopes == [AuthScope::Read]
        && [
            config.admin_token.as_str(),
            config.editor_token.as_str(),
            config.approver_token.as_str(),
        ]
        .into_iter()
        .filter(|token| !token.is_empty())
        .any(|token| {
            matches_bearer_token(auth_header, token) || matches_query_token(&request, token)
        })
    {
        return next.run(request).await;
    }

    if scopes == [AuthScope::Read] && config.read_token.is_empty() {
        return auth_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "read auth is not configured",
        );
    }
    if scopes.contains(&AuthScope::Admin) && config.admin_token.is_empty() {
        return auth_error(StatusCode::FORBIDDEN, "admin auth is not configured");
    }
    if scopes.contains(&AuthScope::Editor) && config.editor_token.is_empty() {
        return auth_error(StatusCode::FORBIDDEN, "editor auth is not configured");
    }
    if scopes.contains(&AuthScope::Approver) && config.approver_token.is_empty() {
        return auth_error(StatusCode::FORBIDDEN, "approver auth is not configured");
    }

    auth_error(StatusCode::UNAUTHORIZED, "unauthorized")
}

fn expected_token_for_scope<'a>(config: &'a HubAuthConfig, scope: AuthScope) -> Option<&'a str> {
    let token = match scope {
        AuthScope::Read => config.read_token.as_str(),
        AuthScope::Admin => config.admin_token.as_str(),
        AuthScope::Editor => config.editor_token.as_str(),
        AuthScope::Approver => config.approver_token.as_str(),
    };
    (!token.is_empty()).then_some(token)
}

fn auth_error(status: StatusCode, message: &str) -> Response {
    let body = json!({ "error": message });
    (status, axum::Json(body)).into_response()
}

fn read_auth_bypassed(request: &Request) -> bool {
    trusted_tailscale_identity(request)
        || request_client_ip(request)
            .map(is_trusted_read_ip)
            .unwrap_or(false)
}

fn request_client_ip(request: &Request) -> Option<IpAddr> {
    let connect_ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|value| value.0.ip());
    match connect_ip {
        Some(ip) if ip.is_loopback() => forwarded_client_ip(request),
        Some(ip) => Some(ip),
        None => None,
    }
}

fn trusted_tailscale_identity(request: &Request) -> bool {
    if !direct_peer_is_loopback(request) {
        return false;
    }
    [
        "tailscale-user-login",
        "tailscale-user-name",
        "tailscale-user-profile-pic",
        "tailscale-headers-info",
    ]
    .into_iter()
    .any(|header| {
        request
            .headers()
            .get(header)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_some()
    })
}

fn direct_peer_is_loopback(request: &Request) -> bool {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|value| value.0.ip().is_loopback())
        .unwrap_or(false)
}

fn forwarded_client_ip(request: &Request) -> Option<IpAddr> {
    request
        .headers()
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .and_then(parse_ip_like)
        .or_else(|| {
            request
                .headers()
                .get("x-real-ip")
                .and_then(|value| value.to_str().ok())
                .and_then(parse_ip_like)
        })
}

fn parse_ip_like(value: &str) -> Option<IpAddr> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value
        .parse::<IpAddr>()
        .ok()
        .or_else(|| value.parse::<SocketAddr>().ok().map(|addr| addr.ip()))
}

fn is_trusted_read_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => is_trusted_read_ipv4(ipv4),
        IpAddr::V6(ipv6) => is_trusted_read_ipv6(ipv6),
    }
}

fn is_trusted_read_ipv4(ip: Ipv4Addr) -> bool {
    if ip.is_private() || ip.is_link_local() {
        return true;
    }
    let octets = ip.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

fn is_trusted_read_ipv6(ip: Ipv6Addr) -> bool {
    if ip.is_unique_local() || ip.is_unicast_link_local() {
        return true;
    }
    ip.segments()[0] == 0xfd7a && ip.segments()[1] == 0x115c && ip.segments()[2] == 0xa1e0
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

fn extract_query_token(query: &str) -> Option<String> {
    query.split('&').find_map(|segment| {
        let (key, value) = segment.split_once('=')?;
        (key == "token" && !value.is_empty())
            .then(|| percent_decode(value))
            .flatten()
    })
}

fn percent_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut idx = 0;

    while idx < bytes.len() {
        match bytes[idx] {
            b'%' => {
                let hi = bytes.get(idx + 1).copied()?;
                let lo = bytes.get(idx + 2).copied()?;
                decoded.push((hex_value(hi)? << 4) | hex_value(lo)?);
                idx += 3;
            }
            b'+' => {
                decoded.push(b' ');
                idx += 1;
            }
            other => {
                decoded.push(other);
                idx += 1;
            }
        }
    }

    String::from_utf8(decoded).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
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

    fn editor_app(config: HubAuthConfig) -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .route_layer(middleware::from_fn(require_editor_auth))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(config))
    }

    fn approver_app(config: HubAuthConfig) -> Router {
        Router::new()
            .route("/test", get(ok_handler))
            .route_layer(middleware::from_fn(require_approver_auth))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(config))
    }

    #[tokio::test]
    async fn startup_validation_requires_read_token_outside_dev_mode() {
        let config = HubAuthConfig {
            read_token: String::new(),
            admin_token: String::new(),
            editor_token: String::new(),
            approver_token: String::new(),
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
            editor_token: String::new(),
            approver_token: String::new(),
            dev_mode: true,
        })
        .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn trusted_read_ip_accepts_rfc1918_and_tailscale_ranges() {
        assert!(is_trusted_read_ip(IpAddr::V4(Ipv4Addr::new(
            192, 168, 1, 20
        ))));
        assert!(is_trusted_read_ip(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 42))));
        assert!(is_trusted_read_ip(IpAddr::V4(Ipv4Addr::new(
            100, 101, 102, 103
        ))));
        assert!(is_trusted_read_ip(
            "fd7a:115c:a1e0::1234".parse::<IpAddr>().unwrap()
        ));
        assert!(!is_trusted_read_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(!is_trusted_read_ip(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
    }

    #[test]
    fn request_client_ip_prefers_forwarded_header_only_from_loopback_proxy() {
        let mut proxied = Request::builder().uri("/test").body(Body::empty()).unwrap();
        proxied
            .headers_mut()
            .insert("x-forwarded-for", "100.88.1.9".parse().unwrap());
        proxied
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 12345))));
        assert_eq!(
            request_client_ip(&proxied),
            Some(IpAddr::V4(Ipv4Addr::new(100, 88, 1, 9)))
        );

        let mut spoofed = Request::builder().uri("/test").body(Body::empty()).unwrap();
        spoofed
            .headers_mut()
            .insert("x-forwarded-for", "192.168.1.9".parse().unwrap());
        spoofed
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([8, 8, 8, 8], 12345))));
        assert_eq!(
            request_client_ip(&spoofed),
            Some(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)))
        );
    }

    #[test]
    fn request_client_ip_does_not_trust_bare_loopback_proxy() {
        let mut proxied = Request::builder().uri("/test").body(Body::empty()).unwrap();
        proxied
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 12345))));

        assert_eq!(request_client_ip(&proxied), None);
    }

    #[test]
    fn request_client_ip_uses_x_real_ip_from_loopback_proxy() {
        let mut proxied = Request::builder().uri("/test").body(Body::empty()).unwrap();
        proxied
            .headers_mut()
            .insert("x-real-ip", "100.77.1.9".parse().unwrap());
        proxied
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 12345))));

        assert_eq!(
            request_client_ip(&proxied),
            Some(IpAddr::V4(Ipv4Addr::new(100, 77, 1, 9)))
        );
    }

    #[test]
    fn trusted_tailscale_identity_requires_loopback_peer() {
        let mut proxied = Request::builder().uri("/test").body(Body::empty()).unwrap();
        proxied
            .headers_mut()
            .insert("tailscale-user-login", "me@example.com".parse().unwrap());
        proxied
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 12345))));
        assert!(trusted_tailscale_identity(&proxied));

        let mut spoofed = Request::builder().uri("/test").body(Body::empty()).unwrap();
        spoofed
            .headers_mut()
            .insert("tailscale-user-login", "me@example.com".parse().unwrap());
        spoofed
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([8, 8, 8, 8], 12345))));
        assert!(!trusted_tailscale_identity(&spoofed));
    }

    #[tokio::test]
    async fn startup_validation_rejects_shared_read_and_admin_tokens() {
        let config = HubAuthConfig {
            read_token: "shared-secret".to_string(),
            admin_token: "shared-secret".to_string(),
            editor_token: String::new(),
            approver_token: String::new(),
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
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
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
    async fn read_auth_allows_trusted_lan_without_token() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 50], 23456))));

        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn read_auth_allows_loopback_tailscale_identity_without_token() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request
            .headers_mut()
            .insert("tailscale-user-login", "me@example.com".parse().unwrap());
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 23456))));

        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn read_auth_keeps_token_for_loopback_without_forwarded_ip() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 23456))));

        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn read_auth_allows_trusted_tailscale_ipv6_without_token() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            "fd7a:115c:a1e0::42".parse::<IpAddr>().unwrap(),
            23456,
        )));

        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn admin_auth_still_requires_token_from_trusted_lan() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([100, 90, 12, 34], 23456))));

        let response = admin_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_auth_still_requires_token_with_tailscale_identity_header() {
        let mut request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        request
            .headers_mut()
            .insert("tailscale-user-login", "me@example.com".parse().unwrap());
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 23456))));

        let response = admin_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(request)
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_token_can_pass_the_global_read_gate() {
        let response = read_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
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
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
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
    async fn editor_auth_requires_editor_token() {
        let response = editor_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("authorization", "Bearer editor-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn approver_auth_rejects_editor_token() {
        let response = approver_app(HubAuthConfig {
            read_token: "viewer-secret".to_string(),
            admin_token: "admin-secret".to_string(),
            editor_token: "editor-secret".to_string(),
            approver_token: "approver-secret".to_string(),
            dev_mode: false,
        })
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("authorization", "Bearer editor-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn websocket_query_token_can_pass_the_read_gate() {
        let app = Router::new()
            .route("/ws", get(ok_handler))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(HubAuthConfig {
                read_token: "viewer-secret".to_string(),
                admin_token: "admin-secret".to_string(),
                editor_token: "editor-secret".to_string(),
                approver_token: "approver-secret".to_string(),
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

    #[tokio::test]
    async fn websocket_query_token_is_percent_decoded_before_validation() {
        let app = Router::new()
            .route("/ws", get(ok_handler))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(HubAuthConfig {
                read_token: "abc+def=".to_string(),
                admin_token: "admin-secret".to_string(),
                editor_token: "editor-secret".to_string(),
                approver_token: "approver-secret".to_string(),
                dev_mode: false,
            }));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ws?token=abc%2Bdef%3D")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn websocket_query_editor_token_can_pass_the_read_gate() {
        let app = Router::new()
            .route("/ws", get(ok_handler))
            .layer(middleware::from_fn(require_read_auth))
            .layer(axum::Extension(HubAuthConfig {
                read_token: "viewer-secret".to_string(),
                admin_token: "admin-secret".to_string(),
                editor_token: "editor-secret".to_string(),
                approver_token: "approver-secret".to_string(),
                dev_mode: false,
            }));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ws?token=editor-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
