use axum::http::{header, HeaderMap};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
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
        .read(true)
        .open(&ledger_path)?;
    file.lock_exclusive()?;
    writeln!(file, "{payload}")?;
    file.sync_data()?;
    file.unlock()?;
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

    let mut file = OpenOptions::new().read(true).open(&ledger_path)?;
    file.lock_shared()?;
    file.seek(SeekFrom::Start(0))?;

    let mut events = Vec::new();
    for (index, line_result) in BufReader::new(&file).lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<ConfigAuditEvent>(&line).map_err(|err| {
            HubError::Internal(format!(
                "failed to parse config audit ledger line {}: {err}",
                index + 1
            ))
        })?;
        events.push(event);
    }
    file.unlock()?;

    let filtered = events
        .into_iter()
        .rev()
        .filter(|event| {
            file_variant
                .map(|variant| event.file_variant == variant)
                .unwrap_or(true)
        })
        .take(limit.min(MAX_RECENT_EVENTS))
        .collect::<Vec<_>>();

    Ok(filtered)
}

pub fn latest_successful_event_for_config_id(
    artifacts_dir: &Path,
    lane: &str,
    config_id: &str,
) -> Result<Option<ConfigAuditEvent>, HubError> {
    let events = read_recent_events(artifacts_dir, Some(lane), MAX_RECENT_EVENTS)?;
    Ok(events.into_iter().find(|event| {
        matches!(event.action.as_str(), "apply_live" | "rollback_live")
            && event.after.config_id.as_deref() == Some(config_id)
            && event
                .result
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
    }))
}

fn ledger_path(artifacts_dir: &Path) -> PathBuf {
    artifacts_dir
        .join(CONFIG_AUDIT_DIR)
        .join(CONFIG_AUDIT_LEDGER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_event(config_id: &str, ts_ms: i64) -> ConfigAuditEvent {
        ConfigAuditEvent {
            version: "config_audit_event_v1".to_string(),
            ts_ms,
            ts_utc: "2026-03-14T00:00:00Z".to_string(),
            lane: "live".to_string(),
            file_variant: "live".to_string(),
            action: "apply_live".to_string(),
            actor: ConfigAuditActor {
                auth_scope: "admin_token".to_string(),
                label: "operator".to_string(),
                source_ip: None,
                user_agent: None,
            },
            reason: Some("test".to_string()),
            validation: serde_json::json!({ "ok": true }),
            before: ConfigAuditIdentity {
                lock_id: Some("before".to_string()),
                config_id: Some("before-cfg".to_string()),
            },
            after: ConfigAuditIdentity {
                lock_id: Some("after".to_string()),
                config_id: Some(config_id.to_string()),
            },
            result: serde_json::json!({ "ok": true }),
            artifact_path: None,
        }
    }

    #[test]
    fn latest_successful_event_prefers_most_recent_matching_identity() {
        let dir = tempdir().unwrap();
        append_event(dir.path(), &sample_event("cfg-a", 10)).unwrap();
        append_event(dir.path(), &sample_event("cfg-b", 20)).unwrap();
        append_event(dir.path(), &sample_event("cfg-a", 30)).unwrap();

        let latest = latest_successful_event_for_config_id(dir.path(), "live", "cfg-a")
            .unwrap()
            .expect("latest matching event");

        assert_eq!(latest.after.config_id.as_deref(), Some("cfg-a"));
        assert_eq!(latest.ts_ms, 30);
    }

    #[test]
    fn latest_successful_event_ignores_save_only_entries() {
        let dir = tempdir().unwrap();
        let mut save_event = sample_event("cfg-a", 40);
        save_event.action = "save_config".to_string();
        append_event(dir.path(), &save_event).unwrap();
        append_event(dir.path(), &sample_event("cfg-a", 30)).unwrap();

        let latest = latest_successful_event_for_config_id(dir.path(), "live", "cfg-a")
            .unwrap()
            .expect("latest matching deploy event");

        assert_eq!(latest.action, "apply_live");
        assert_eq!(latest.ts_ms, 30);
    }

    #[test]
    fn read_recent_events_fails_on_malformed_line() {
        let dir = tempdir().unwrap();
        let ledger = ledger_path(dir.path());
        fs::create_dir_all(ledger.parent().unwrap()).unwrap();
        fs::write(&ledger, "{\"bad\":\n").unwrap();

        let err = read_recent_events(dir.path(), None, 10).unwrap_err();
        match err {
            HubError::Internal(message) => {
                assert!(message.contains("failed to parse config audit ledger line"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
