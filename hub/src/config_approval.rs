use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::config_audit::ConfigAuditActor;
use crate::error::HubError;

const APPROVALS_DIR: &str = "config_approvals";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigApprovalAction {
    ApplyLive,
    RollbackLive,
}

impl ConfigApprovalAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ApplyLive => "apply_live",
            Self::RollbackLive => "rollback_live",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigApprovalRequest {
    pub version: String,
    pub request_id: String,
    pub action: ConfigApprovalAction,
    pub lane: String,
    pub created_at_ms: i64,
    pub created_at_utc: String,
    pub requester: ConfigAuditActor,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approver: Option<ConfigAuditActor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approved_at_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approved_at_utc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_at_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_at_utc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub summary: Value,
    pub execute: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigApprovalSummary {
    pub request_id: String,
    pub action: String,
    pub lane: String,
    pub created_at_ms: i64,
    pub created_at_utc: String,
    pub requester: ConfigAuditActor,
    pub reason: Option<String>,
    pub summary: Value,
}

pub fn create_request(
    artifacts_dir: &Path,
    action: ConfigApprovalAction,
    requester: ConfigAuditActor,
    reason: Option<String>,
    summary: Value,
    execute: Value,
) -> Result<ConfigApprovalRequest, HubError> {
    let request = ConfigApprovalRequest {
        version: "config_approval_request_v1".to_string(),
        request_id: Uuid::new_v4().simple().to_string(),
        action,
        lane: "live".to_string(),
        created_at_ms: chrono::Utc::now().timestamp_millis(),
        created_at_utc: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        requester,
        approver: None,
        approved_at_ms: None,
        approved_at_utc: None,
        rejected_at_ms: None,
        rejected_at_utc: None,
        rejection_reason: None,
        execution_result: None,
        reason,
        summary,
        execute,
    };
    write_request(&pending_path(artifacts_dir, &request.request_id), &request)?;
    Ok(request)
}

pub fn list_pending_requests(artifacts_dir: &Path) -> Result<Vec<ConfigApprovalSummary>, HubError> {
    let dir = pending_dir(artifacts_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut requests = fs::read_dir(&dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
        })
        .filter_map(|entry| load_request(&entry.path()).ok())
        .map(|request| ConfigApprovalSummary {
            request_id: request.request_id,
            action: request.action.as_str().to_string(),
            lane: request.lane,
            created_at_ms: request.created_at_ms,
            created_at_utc: request.created_at_utc,
            requester: request.requester,
            reason: request.reason,
            summary: request.summary,
        })
        .collect::<Vec<_>>();
    requests.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    Ok(requests)
}

pub fn load_for_processing(
    artifacts_dir: &Path,
    request_id: &str,
) -> Result<ConfigApprovalRequest, HubError> {
    let pending = pending_path(artifacts_dir, request_id);
    if !pending.exists() {
        return Err(HubError::NotFound(format!(
            "pending config approval request not found: {request_id}"
        )));
    }
    let processing = processing_path(artifacts_dir, request_id);
    fs::create_dir_all(processing.parent().unwrap())?;
    fs::rename(&pending, &processing)?;
    load_request(&processing)
}

pub fn mark_approved(
    artifacts_dir: &Path,
    mut request: ConfigApprovalRequest,
    approver: ConfigAuditActor,
    execution_result: Value,
) -> Result<ConfigApprovalRequest, HubError> {
    request.approver = Some(approver);
    request.approved_at_ms = Some(chrono::Utc::now().timestamp_millis());
    request.approved_at_utc = Some(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());
    request.execution_result = Some(execution_result);
    write_request(&approved_path(artifacts_dir, &request.request_id), &request)?;
    let _ = fs::remove_file(processing_path(artifacts_dir, &request.request_id));
    Ok(request)
}

pub fn mark_rejected(
    artifacts_dir: &Path,
    mut request: ConfigApprovalRequest,
    approver: ConfigAuditActor,
    rejection_reason: Option<String>,
) -> Result<ConfigApprovalRequest, HubError> {
    request.approver = Some(approver);
    request.rejected_at_ms = Some(chrono::Utc::now().timestamp_millis());
    request.rejected_at_utc = Some(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());
    request.rejection_reason = rejection_reason;
    write_request(&rejected_path(artifacts_dir, &request.request_id), &request)?;
    let _ = fs::remove_file(processing_path(artifacts_dir, &request.request_id));
    Ok(request)
}

pub fn mark_failed(
    artifacts_dir: &Path,
    mut request: ConfigApprovalRequest,
    approver: ConfigAuditActor,
    execution_result: Value,
) -> Result<ConfigApprovalRequest, HubError> {
    request.approver = Some(approver);
    request.approved_at_ms = Some(chrono::Utc::now().timestamp_millis());
    request.approved_at_utc = Some(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());
    request.execution_result = Some(execution_result);
    write_request(&failed_path(artifacts_dir, &request.request_id), &request)?;
    let _ = fs::remove_file(processing_path(artifacts_dir, &request.request_id));
    Ok(request)
}

fn load_request(path: &Path) -> Result<ConfigApprovalRequest, HubError> {
    let payload = fs::read_to_string(path)
        .with_context(|| format!("failed to read config approval request {}", path.display()))
        .map_err(|err| HubError::Internal(err.to_string()))?;
    serde_json::from_str(&payload).map_err(|err| {
        HubError::Internal(format!(
            "failed to parse config approval request {}: {err}",
            path.display()
        ))
    })
}

fn write_request(path: &Path, request: &ConfigApprovalRequest) -> Result<(), HubError> {
    let payload = serde_json::to_string_pretty(request)? + "\n";
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("{}.tmp", Uuid::new_v4().simple()));
    fs::write(&tmp, payload)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn pending_dir(artifacts_dir: &Path) -> PathBuf {
    artifacts_dir.join(APPROVALS_DIR).join("pending")
}

fn pending_path(artifacts_dir: &Path, request_id: &str) -> PathBuf {
    pending_dir(artifacts_dir).join(format!("{request_id}.json"))
}

fn processing_path(artifacts_dir: &Path, request_id: &str) -> PathBuf {
    artifacts_dir
        .join(APPROVALS_DIR)
        .join("processing")
        .join(format!("{request_id}.json"))
}

fn approved_path(artifacts_dir: &Path, request_id: &str) -> PathBuf {
    artifacts_dir
        .join(APPROVALS_DIR)
        .join("approved")
        .join(format!("{request_id}.json"))
}

fn rejected_path(artifacts_dir: &Path, request_id: &str) -> PathBuf {
    artifacts_dir
        .join(APPROVALS_DIR)
        .join("rejected")
        .join(format!("{request_id}.json"))
}

fn failed_path(artifacts_dir: &Path, request_id: &str) -> PathBuf {
    artifacts_dir
        .join(APPROVALS_DIR)
        .join("failed")
        .join(format!("{request_id}.json"))
}
