use axum::http::{header, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::HubError;

const CONFIG_AUDIT_DIR: &str = "config_audit";
const CONFIG_AUDIT_LEDGER: &str = "config_events.jsonl";
const MAX_RECENT_EVENTS: usize = 500;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigAuditActor {
    pub auth_scope: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_ip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigAuditIdentity {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigAuditEvent {
    pub version: String,
    pub ts_ms: i64,
    pub ts_utc: String,
    pub lane: String,
    pub file_variant: String,
    pub action: String,
    pub actor: ConfigAuditActor,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub validation: Value,
    pub before: ConfigAuditIdentity,
    pub after: ConfigAuditIdentity,
    pub result: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfigAuditQuery {
    pub file: Option<String>,
    pub limit: Option<usize>,
}

pub fn weak_actor_from_headers(headers: &HeaderMap, auth_scope: &str) -> ConfigAuditActor {
    let label = headers
        .get("x-aiq-actor")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| auth_scope.to_string());

    let source_ip = headers
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            headers
                .get("x-real-ip")
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        });

    let user_agent = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    ConfigAuditActor {
        auth_scope: auth_scope.to_string(),
        label,
        source_ip,
        user_agent,
    }
}

pub fn append_event(artifacts_dir: &Path, event: &ConfigAuditEvent) -> Result<PathBuf, HubError> {
    let ledger_path = ledger_path(artifacts_dir);
    if let Some(parent) = ledger_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string(event)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ledger_path)?;
    writeln!(file, "{payload}")?;
    Ok(ledger_path)
}

pub fn read_recent_events(
    artifacts_dir: &Path,
    file_variant: Option<&str>,
    limit: usize,
) -> Result<Vec<ConfigAuditEvent>, HubError> {
    let ledger_path = ledger_path(artifacts_dir);
    if !ledger_path.exists() {
        return Ok(Vec::new());
    }

    let payload = fs::read_to_string(&ledger_path)?;
    let filtered = payload
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<ConfigAuditEvent>(line).ok())
        .filter(|event| {
            file_variant
                .map(|variant| event.file_variant == variant)
                .unwrap_or(true)
        })
        .take(limit.min(MAX_RECENT_EVENTS))
        .collect::<Vec<_>>();

    Ok(filtered)
}

fn ledger_path(artifacts_dir: &Path) -> PathBuf {
    artifacts_dir
        .join(CONFIG_AUDIT_DIR)
        .join(CONFIG_AUDIT_LEDGER)
}
