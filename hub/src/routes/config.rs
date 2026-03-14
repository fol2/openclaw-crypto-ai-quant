use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use uuid::Uuid;

use crate::config_approval::{
    create_request as create_config_approval_request, list_pending_requests, load_for_processing,
    mark_approved, mark_failed, mark_rejected, ConfigApprovalAction,
};
use crate::config_audit::{
    append_event as append_config_audit_ledger_event, read_recent_events, weak_actor_from_headers,
    ConfigAuditEvent, ConfigAuditIdentity, ConfigAuditQuery,
};
use crate::diagnostic_audit::{append_read_event, DiagnosticReadEvent};
use crate::error::HubError;
use crate::paper_config::PaperEffectiveConfig;
use crate::paper_lane::PaperLane;
use crate::state::AppState;

/// Query parameter for config endpoints — selects which YAML file to operate on.
#[derive(Deserialize)]
pub struct ConfigQuery {
    /// Config file variant: "main", "live", "paper1", "paper2", "paper3",
    /// "promoted_primary", "promoted_fallback". Default: "main".
    pub file: Option<String>,
}

#[derive(Deserialize)]
pub struct PrivilegedConfigQuery {
    pub file: Option<String>,
}

/// Body for PUT /api/config (raw YAML string).
#[derive(Deserialize)]
pub struct ConfigWriteBody {
    pub yaml: String,
    pub expected_config_id: Option<String>,
    pub reason: Option<String>,
}

/// Diff query parameters.
#[derive(Deserialize)]
pub struct DiffQuery {
    /// Backup timestamp A (filename stem).
    pub a: String,
    /// Backup timestamp B (or "current" for live version).
    pub b: String,
    /// Config file variant (default "main").
    pub file: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RollbackLiveBody {
    pub steps: Option<u32>,
    pub reason: Option<String>,
    pub restart: Option<String>,
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ApplyLiveBody {
    pub yaml: String,
    pub expected_config_id: Option<String>,
    pub reason: Option<String>,
    pub restart: Option<String>,
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ConfigApprovalDecisionBody {
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ConfigApprovalListQuery {
    pub status: Option<String>,
}

/// A single backup entry.
#[derive(Serialize)]
pub struct BackupEntry {
    pub filename: String,
    pub modified: String,
    pub size: u64,
}

struct ResolvedConfigState {
    path: PathBuf,
    raw_text: String,
    lock_id: String,
    runtime_config_id: Option<String>,
}

struct ValidatedConfigWrite {
    payload: String,
    lock_id: String,
    config_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveApplyRestartMode {
    Auto,
    Always,
    Never,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum LiveManifestLaunchStateProof {
    Blocked,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum LiveServiceStateProof {
    Blocked,
    Ready,
    Running,
    RestartRequired,
    StatusStale,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum LiveSupervisorActionProof {
    Hold,
    Start,
    Restart,
    Monitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum LiveServiceAppliedActionProof {
    Noop,
    Start,
    Restart,
    Stop,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveServiceApplyProof {
    ok: bool,
    applied_action: LiveServiceAppliedActionProof,
    action_reason: String,
    preview: LiveServiceProof,
    final_service: LiveServiceProof,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveServiceProof {
    ok: bool,
    desired_action: LiveSupervisorActionProof,
    action_reason: String,
    status: LiveStatusProof,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveStatusProof {
    ok: bool,
    service_state: LiveServiceStateProof,
    contract_matches_status: bool,
    #[serde(default)]
    mismatch_reasons: Vec<String>,
    manifest: LiveManifestProof,
    daemon_status: Option<LiveDaemonStatusProof>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveManifestProof {
    launch_state: LiveManifestLaunchStateProof,
    config_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveDaemonStatusProof {
    ok: bool,
    running: bool,
    #[serde(default)]
    errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LiveConfigTransactionResult {
    ok: bool,
    action_dir: String,
    backup: Option<String>,
    restart_required: bool,
    runtime_result: Value,
    restore_result: Option<Value>,
    error: Option<String>,
}

struct LiveTransactionEventInput<'a> {
    version: &'a str,
    action: &'a str,
    reason: &'a str,
    dry_run: bool,
    current: &'a ResolvedConfigState,
    current_config_id: &'a str,
    candidate_label: &'a str,
    candidate: &'a ValidatedConfigWrite,
    runtime_result: &'a Value,
    restore_result: Option<&'a Value>,
    extra: Value,
}

struct LiveConfigTransactionInput<'a> {
    action_dir: &'a Path,
    event_version: &'a str,
    action_name: &'a str,
    current: &'a ResolvedConfigState,
    current_config_id: &'a str,
    candidate_label: &'a str,
    candidate: &'a ValidatedConfigWrite,
    reason: &'a str,
    runtime_action: &'a str,
    dry_run: bool,
    extra_event_details: Value,
}

struct PreparedLiveApply {
    current: ResolvedConfigState,
    incumbent: ValidatedConfigWrite,
    candidate: ValidatedConfigWrite,
    reason: String,
    restart_mode: LiveApplyRestartMode,
    preview_restart_required: bool,
}

struct PreparedLiveRollback {
    steps: usize,
    src_dir: PathBuf,
    restored_from: PathBuf,
    current: ResolvedConfigState,
    incumbent: ValidatedConfigWrite,
    restored: ValidatedConfigWrite,
    reason: String,
    restart_mode: LiveApplyRestartMode,
}

struct StagedConfig {
    root: PathBuf,
    path: PathBuf,
}

impl Drop for StagedConfig {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl LiveApplyRestartMode {
    fn parse(raw: Option<&str>) -> Result<Self, HubError> {
        match raw
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("auto")
            .to_ascii_lowercase()
            .as_str()
        {
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            _ => Err(HubError::BadRequest(
                "invalid restart mode (expected auto/always/never)".to_string(),
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Always => "always",
            Self::Never => "never",
        }
    }

    fn runtime_action(self, dry_run: bool) -> Result<&'static str, HubError> {
        match self {
            Self::Auto => Ok("auto"),
            Self::Always => Ok("restart"),
            Self::Never if dry_run => Ok("auto"),
            Self::Never => Err(HubError::BadRequest(
                "restart=never is not allowed for live apply actions; use dry_run to preview without mutating runtime state".to_string(),
            )),
        }
    }
}

/// Build the config sub-router.
pub fn routes() -> Router<Arc<AppState>> {
    let read_routes = Router::new()
        .route("/api/config", get(get_config))
        .route("/api/config/raw", get(get_config_raw))
        .route("/api/config/history", get(get_config_history))
        .route("/api/config/audit", get(get_config_audit))
        .route("/api/config/diff", get(get_config_diff))
        .route("/api/config/files", get(get_config_files));
    let admin_routes = Router::new()
        .route("/api/config/raw/privileged", get(get_config_raw_privileged))
        .route(
            "/api/config/audit/privileged",
            get(get_config_audit_privileged),
        )
        .route(
            "/api/config/diff/privileged",
            get(get_config_diff_privileged),
        )
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));
    let editor_routes = Router::new()
        .route("/api/config", put(put_config))
        .route("/api/config/reload", post(post_config_reload))
        .route(
            "/api/config/actions/apply-live",
            post(post_preview_apply_live),
        )
        .route(
            "/api/config/actions/apply-live/request",
            post(post_request_apply_live),
        )
        .route(
            "/api/config/actions/rollback-live",
            post(post_preview_rollback_live),
        )
        .route(
            "/api/config/actions/rollback-live/request",
            post(post_request_rollback_live),
        )
        .route_layer(middleware::from_fn(crate::auth::require_editor_auth));
    let approval_read_routes = Router::new()
        .route("/api/config/approvals", get(get_config_approvals))
        .route_layer(middleware::from_fn(
            crate::auth::require_editor_or_approver_auth,
        ));
    let approver_routes = Router::new()
        .route(
            "/api/config/approvals/{request_id}/approve",
            post(post_approve_config_approval),
        )
        .route(
            "/api/config/approvals/{request_id}/reject",
            post(post_reject_config_approval),
        )
        .route("/api/config/actions/promote-live", post(post_promote_live))
        .route_layer(middleware::from_fn(crate::auth::require_approver_auth));

    read_routes
        .merge(admin_routes)
        .merge(editor_routes)
        .merge(approval_read_routes)
        .merge(approver_routes)
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Resolve the YAML file path for a given variant name.
/// Returns an error if the variant is not in the allow-list (path-traversal protection).
fn resolve_config_path(config_dir: &Path, file: &str) -> Result<PathBuf, HubError> {
    let filename = match file {
        "main" => "strategy_overrides.yaml",
        "live" => "strategy_overrides.live.yaml",
        "livepaper" => "strategy_overrides.livepaper.yaml",
        "paper1" => "strategy_overrides.paper1.yaml",
        "paper2" => "strategy_overrides.paper2.yaml",
        "paper3" => "strategy_overrides.paper3.yaml",
        "promoted_primary" => "strategy_overrides._promoted_primary.yaml",
        "promoted_fallback" => "strategy_overrides._promoted_fallback.yaml",
        _ => {
            return Err(HubError::BadRequest(format!(
                "unknown config variant: {file}"
            )))
        }
    };
    Ok(config_dir.join(filename))
}

fn resolve_config_path_for_state(state: &AppState, file: &str) -> Result<PathBuf, HubError> {
    if file == "live" {
        Ok(state.config.live_yaml_path.clone())
    } else {
        resolve_config_path(&state.config.config_dir, file)
    }
}

/// Variant name from ConfigQuery, defaulting to "main".
fn variant(q: &ConfigQuery) -> &str {
    q.file.as_deref().unwrap_or("main")
}

fn validation_lane(file_variant: &str) -> Option<PaperLane> {
    match file_variant {
        "paper1" => Some(PaperLane::Paper1),
        "paper2" => Some(PaperLane::Paper2),
        "paper3" => Some(PaperLane::Paper3),
        "livepaper" => Some(PaperLane::Livepaper),
        _ => None,
    }
}

fn normalise_yaml_payload(text: &str) -> String {
    format!("{}\n", text.trim_end())
}

fn raw_config_id(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Ensure the backups directory exists and return its path.
fn backups_dir(config_dir: &Path) -> Result<PathBuf, HubError> {
    let dir = config_dir.join("backups");
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

fn ensure_admin_actions_enabled(state: &AppState) -> Result<(), HubError> {
    if state.config.admin_actions_enabled {
        Ok(())
    } else {
        Err(HubError::Forbidden(
            "admin actions disabled (set AIQ_MONITOR_ADMIN_ACTIONS_ENABLE=1)".to_string(),
        ))
    }
}

fn parse_json_object_from_output(text: &str) -> Option<Value> {
    let raw = text.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        if v.is_object() {
            return Some(v);
        }
    }
    for line in raw.lines().rev() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(s) {
            if v.is_object() {
                return Some(v);
            }
        }
    }
    None
}

async fn run_action_command(command: Vec<String>, cwd: &Path, timeout_s: u64) -> Value {
    let started_ms = chrono::Utc::now().timestamp_millis();
    if command.is_empty() {
        return json!({
            "ok": false,
            "error": "empty_command",
            "command": [],
            "exit_code": Value::Null,
            "stdout": "",
            "stderr": "",
            "timeout": false,
        });
    }

    let mut cmd = Command::new(&command[0]);
    for arg in command.iter().skip(1) {
        cmd.arg(arg);
    }
    cmd.current_dir(cwd);

    match tokio::time::timeout(Duration::from_secs(timeout_s.max(1)), cmd.output()).await {
        Err(_) => json!({
            "ok": false,
            "error": "command_timeout",
            "command": command,
            "exit_code": Value::Null,
            "stdout": "",
            "stderr": "",
            "timeout": true,
            "elapsed_ms": chrono::Utc::now().timestamp_millis() - started_ms,
            "timeout_s": timeout_s,
        }),
        Ok(Err(e)) => json!({
            "ok": false,
            "error": format!("command_failed:{e}"),
            "command": command,
            "exit_code": Value::Null,
            "stdout": "",
            "stderr": "",
            "timeout": false,
            "elapsed_ms": chrono::Utc::now().timestamp_millis() - started_ms,
            "timeout_s": timeout_s,
        }),
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let parsed = parse_json_object_from_output(&stdout);
            let parsed_ok = parsed
                .as_ref()
                .and_then(|v| v.get("ok"))
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            json!({
                "ok": output.status.success() && parsed_ok,
                "command": command,
                "exit_code": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
                "timeout": false,
                "elapsed_ms": chrono::Utc::now().timestamp_millis() - started_ms,
                "timeout_s": timeout_s,
                "parsed": parsed,
            })
        }
    }
}

fn sorted_subdirs_desc(root: &Path) -> Result<Vec<PathBuf>, HubError> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    Ok(dirs)
}

fn atomic_write_text(path: &Path, text: &str) -> Result<(), HubError> {
    let parent = path
        .parent()
        .ok_or_else(|| HubError::Internal("invalid target path".to_string()))?;
    fs::create_dir_all(parent)?;
    let stem = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("config");
    let tmp = parent.join(format!(".{stem}.tmp.{}", Uuid::new_v4().simple()));
    fs::write(&tmp, text)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn stage_config_payload(target_path: &Path, text: &str) -> Result<StagedConfig, HubError> {
    let file_name = target_path
        .file_name()
        .ok_or_else(|| HubError::Internal("invalid target file name".to_string()))?;
    let root =
        std::env::temp_dir().join(format!("aiq-hub-config-stage-{}", Uuid::new_v4().simple()));
    let config_dir = root.join("config");
    fs::create_dir_all(&config_dir)?;
    let staged_path = config_dir.join(file_name);
    fs::write(&staged_path, text)?;
    Ok(StagedConfig {
        root,
        path: staged_path,
    })
}

fn resolve_effective_config_for_variant(
    state: &AppState,
    file_variant: &str,
    path: &Path,
) -> Result<PaperEffectiveConfig, HubError> {
    let project_dir = Some(state.config.aiq_root.as_path());
    let effective_config = if file_variant == "live" {
        PaperEffectiveConfig::resolve_live(Some(path), project_dir)
    } else {
        PaperEffectiveConfig::resolve(Some(path), validation_lane(file_variant), project_dir)
    };
    effective_config.map_err(|err| {
        HubError::BadRequest(format!(
            "runtime validation failed for {file_variant}: {err}"
        ))
    })
}

fn resolve_current_config_state(
    state: &AppState,
    file_variant: &str,
) -> Result<ResolvedConfigState, HubError> {
    let path = resolve_config_path_for_state(state, file_variant)?;
    if !path.exists() {
        return Err(HubError::NotFound(format!(
            "config file not found: {}",
            path.display()
        )));
    }
    let raw_text = fs::read_to_string(&path)?;
    let lock_id = raw_config_id(&raw_text);
    let runtime_config_id = resolve_effective_config_for_variant(state, file_variant, &path)
        .ok()
        .map(|effective_config| effective_config.config_id().to_string());

    Ok(ResolvedConfigState {
        path,
        raw_text,
        lock_id,
        runtime_config_id,
    })
}

fn validate_candidate_config_write(
    state: &AppState,
    file_variant: &str,
    target_path: &Path,
    yaml_text: &str,
) -> Result<ValidatedConfigWrite, HubError> {
    let payload = normalise_yaml_payload(yaml_text);
    let lock_id = raw_config_id(&payload);
    let staged = stage_config_payload(target_path, &payload)?;
    let effective_config = resolve_effective_config_for_variant(state, file_variant, &staged.path)?;

    Ok(ValidatedConfigWrite {
        payload,
        lock_id,
        config_id: effective_config.config_id().to_string(),
    })
}

fn extract_expected_config_id(
    headers: &HeaderMap,
    body: &ConfigWriteBody,
) -> Result<String, HubError> {
    extract_expected_config_id_value(headers, body.expected_config_id.as_deref())
}

fn apply_config_identity_headers(
    headers: &mut HeaderMap,
    state: &ResolvedConfigState,
) -> Result<(), HubError> {
    let etag = HeaderValue::from_str(&format!("\"{}\"", state.lock_id))
        .map_err(|err| HubError::Internal(format!("invalid ETag header: {err}")))?;
    headers.insert(header::ETAG, etag);
    headers.insert(
        "x-aiq-config-lock-id",
        HeaderValue::from_str(&state.lock_id)
            .map_err(|err| HubError::Internal(format!("invalid config-lock-id header: {err}")))?,
    );
    if let Some(runtime_config_id) = state.runtime_config_id.as_deref() {
        headers.insert(
            "x-aiq-config-id",
            HeaderValue::from_str(runtime_config_id)
                .map_err(|err| HubError::Internal(format!("invalid config-id header: {err}")))?,
        );
        headers.insert(
            "x-aiq-config-id-source",
            HeaderValue::from_static("runtime"),
        );
    }
    Ok(())
}

fn write_config_backup(
    config_dir: &Path,
    target_path: &Path,
    raw_text: &str,
) -> Result<String, HubError> {
    let bk_dir = backups_dir(config_dir)?;
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let stem = target_path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("config");
    let mut bk_name = format!("{stem}.{ts}.bak");
    let mut bk_path = bk_dir.join(&bk_name);
    if bk_path.exists() {
        bk_name = format!("{stem}.{ts}.{}.bak", Uuid::new_v4().simple());
        bk_path = bk_dir.join(&bk_name);
    }
    fs::write(&bk_path, raw_text)?;
    Ok(bk_name)
}

fn extract_expected_config_id_value(
    headers: &HeaderMap,
    body_expected_config_id: Option<&str>,
) -> Result<String, HubError> {
    body_expected_config_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            headers
                .get(header::IF_MATCH)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.trim_matches('"').to_string())
        })
        .ok_or_else(|| {
            HubError::BadRequest(
                "expected_config_id or If-Match header is required for config writes".to_string(),
            )
        })
}

fn compact_utc_now() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

fn iso_utc_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn now_ts_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn append_diagnostic_read_audit(
    state: &AppState,
    headers: &HeaderMap,
    route: &str,
    target: &str,
) -> Result<(), HubError> {
    let event = DiagnosticReadEvent {
        version: "diagnostic_read_event_v1".to_string(),
        ts_ms: now_ts_ms(),
        ts_utc: iso_utc_now(),
        route: route.to_string(),
        target: target.to_string(),
        actor: weak_actor_from_headers(headers, "admin_token"),
    };
    append_read_event(&state.config.artifacts_dir, &event)?;
    Ok(())
}

fn actor_from_headers(
    headers: &HeaderMap,
    auth_scope: &str,
) -> crate::config_audit::ConfigAuditActor {
    weak_actor_from_headers(headers, auth_scope)
}

fn summarise_runtime_apply_result(result: &Value) -> Value {
    let parsed = result.get("parsed");
    json!({
        "ok": result.get("ok").and_then(|value| value.as_bool()).unwrap_or(false),
        "exit_code": result.get("exit_code").cloned().unwrap_or(Value::Null),
        "timeout": result.get("timeout").cloned().unwrap_or(Value::Bool(false)),
        "elapsed_ms": result.get("elapsed_ms").cloned().unwrap_or(Value::Null),
        "applied_action": parsed.and_then(|value| value.get("applied_action")).cloned().unwrap_or(Value::Null),
        "final_service_state": parsed
            .and_then(|value| value.get("final_service"))
            .and_then(|value| value.get("status"))
            .and_then(|value| value.get("service_state"))
            .cloned()
            .unwrap_or(Value::Null),
        "final_config_id": parsed
            .and_then(|value| value.get("final_service"))
            .and_then(|value| value.get("status"))
            .and_then(|value| value.get("manifest"))
            .and_then(|value| value.get("config_id"))
            .cloned()
            .unwrap_or(Value::Null),
        "error": result.get("error").cloned().unwrap_or(Value::Null),
    })
}

fn redacted_config_audit_event(event: &ConfigAuditEvent) -> Value {
    json!({
        "version": event.version,
        "ts_ms": event.ts_ms,
        "ts_utc": event.ts_utc,
        "lane": event.lane,
        "file_variant": event.file_variant,
        "action": event.action,
        "actor": {
            "auth_scope": event.actor.auth_scope,
            "label": event.actor.label,
        },
        "reason": event.reason,
        "validation": event.validation,
        "before": event.before,
        "after": event.after,
        "result": {
            "ok": event.result.get("ok").and_then(|value| value.as_bool()).unwrap_or(false),
            "restart_required": event.result.get("restart_required").cloned().unwrap_or(Value::Null),
        },
        "artifact_path_redacted": event.artifact_path.is_some(),
    })
}

fn normalise_reason(reason: Option<&str>) -> Option<String> {
    reason
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn append_config_audit_event(
    state: &AppState,
    actor: crate::config_audit::ConfigAuditActor,
    file_variant: &str,
    action: &str,
    reason: Option<&str>,
    before_lock_id: Option<&str>,
    before_config_id: Option<&str>,
    after_lock_id: Option<&str>,
    after_config_id: Option<&str>,
    validation: Value,
    result: Value,
    artifact_path: Option<&str>,
    request_id: Option<&str>,
    requester: Option<crate::config_audit::ConfigAuditActor>,
    approver: Option<crate::config_audit::ConfigAuditActor>,
) -> Result<(), HubError> {
    let event = ConfigAuditEvent {
        version: "config_audit_event_v1".to_string(),
        ts_ms: now_ts_ms(),
        ts_utc: iso_utc_now(),
        lane: file_variant.to_string(),
        file_variant: file_variant.to_string(),
        action: action.to_string(),
        actor,
        request_id: request_id.map(ToOwned::to_owned),
        requester,
        approver,
        reason: normalise_reason(reason),
        validation,
        before: ConfigAuditIdentity {
            lock_id: before_lock_id.map(ToOwned::to_owned),
            config_id: before_config_id.map(ToOwned::to_owned),
        },
        after: ConfigAuditIdentity {
            lock_id: after_lock_id.map(ToOwned::to_owned),
            config_id: after_config_id.map(ToOwned::to_owned),
        },
        result,
        artifact_path: artifact_path.map(ToOwned::to_owned),
    };
    append_config_audit_ledger_event(&state.config.artifacts_dir, &event)?;
    Ok(())
}

fn build_action_dir(root: &Path) -> Result<PathBuf, HubError> {
    fs::create_dir_all(root)?;
    let ts_compact = compact_utc_now();
    let mut action_dir = root.join(&ts_compact);
    if action_dir.exists() {
        action_dir = root.join(format!("{ts_compact}-{}", Uuid::new_v4().simple()));
    }
    fs::create_dir_all(&action_dir)?;
    Ok(action_dir)
}

fn command_failure_value(error: impl Into<String>) -> Value {
    json!({
        "ok": false,
        "error": error.into(),
    })
}

fn parse_live_service_apply_proof(result: &Value) -> Result<LiveServiceApplyProof, String> {
    let parsed = result
        .get("parsed")
        .cloned()
        .ok_or_else(|| "runtime apply command did not emit a JSON report".to_string())?;
    serde_json::from_value(parsed)
        .map_err(|err| format!("failed to parse runtime live service apply report: {err}"))
}

fn verify_live_service_apply_proof(
    proof: &LiveServiceApplyProof,
    expected_config_id: &str,
) -> Result<(), String> {
    let mut reasons = Vec::new();
    if !proof.ok {
        reasons.push("runtime service apply reported ok=false".to_string());
    }
    if !proof.final_service.ok {
        reasons.push("final live service report is unhealthy".to_string());
    }
    if !proof.final_service.status.ok {
        reasons.push("final live status report is unhealthy".to_string());
    }
    if proof.final_service.status.service_state != LiveServiceStateProof::Running {
        reasons.push(format!(
            "final live service state is {:?} instead of running",
            proof.final_service.status.service_state
        ));
    }
    if proof.final_service.status.manifest.launch_state != LiveManifestLaunchStateProof::Ready {
        reasons.push("final live launch contract is not ready".to_string());
    }
    if !proof.final_service.status.contract_matches_status {
        let mismatch = proof.final_service.status.mismatch_reasons.join("; ");
        reasons.push(if mismatch.is_empty() {
            "final live status does not match the launch contract".to_string()
        } else {
            format!("final live status does not match the launch contract: {mismatch}")
        });
    }
    if proof.final_service.status.manifest.config_id != expected_config_id {
        reasons.push(format!(
            "final live config_id {} did not match expected {}",
            proof.final_service.status.manifest.config_id, expected_config_id
        ));
    }
    match proof.final_service.status.daemon_status.as_ref() {
        Some(status) => {
            if !status.ok {
                reasons.push("final live daemon status reported ok=false".to_string());
            }
            if !status.running {
                reasons.push("final live daemon status reported running=false".to_string());
            }
            if !status.errors.is_empty() {
                reasons.push(format!(
                    "final live daemon status reported errors: {}",
                    status.errors.join("; ")
                ));
            }
        }
        None => reasons.push("final live daemon status proof is missing".to_string()),
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(reasons.join("; "))
    }
}

fn restart_required_from_proof(proof: &LiveServiceApplyProof) -> bool {
    matches!(
        proof.preview.desired_action,
        LiveSupervisorActionProof::Restart
    ) || matches!(
        proof.final_service.status.service_state,
        LiveServiceStateProof::RestartRequired | LiveServiceStateProof::StatusStale
    )
}

async fn run_live_service_apply_command(
    state: &AppState,
    config_path: &Path,
    expected_config_id: &str,
    runtime_action: &str,
) -> Value {
    run_action_command(
        vec![
            state.config.runtime_bin.display().to_string(),
            "live".to_string(),
            "service".to_string(),
            "apply".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--project-dir".to_string(),
            state.config.aiq_root.display().to_string(),
            "--expected-config-id".to_string(),
            expected_config_id.to_string(),
            "--action".to_string(),
            runtime_action.to_string(),
            "--json".to_string(),
        ],
        &state.config.aiq_root,
        state.config.admin_action_timeout_s,
    )
    .await
}

fn build_live_transaction_event(state: &AppState, input: LiveTransactionEventInput<'_>) -> Value {
    json!({
        "version": input.version,
        "ts_utc": iso_utc_now(),
        "ts_compact_utc": compact_utc_now(),
        "who": {
            "user": env::var("USER").unwrap_or_default(),
            "hostname": env::var("HOSTNAME").unwrap_or_default(),
        },
        "what": {
            "action": input.action,
            "mode": "live",
            "yaml_path": state.config.live_yaml_path.to_string_lossy().to_string(),
            "current_lock_id": input.current.lock_id,
            "current_config_id": input.current_config_id,
            "candidate_lock_id": input.candidate.lock_id,
            "candidate_config_id": input.candidate.config_id,
            "candidate_snapshot": input.candidate_label,
        },
        "why": { "reason": input.reason },
        "dry_run": input.dry_run,
        "runtime_apply": input.runtime_result,
        "recovery": input.restore_result,
        "details": input.extra,
    })
}

async fn execute_live_config_transaction(
    state: &AppState,
    input: LiveConfigTransactionInput<'_>,
) -> Result<LiveConfigTransactionResult, HubError> {
    atomic_write_text(
        &input.action_dir.join("previous_config.yaml"),
        &input.current.raw_text,
    )?;
    atomic_write_text(
        &input.action_dir.join(input.candidate_label),
        &input.candidate.payload,
    )?;

    let mut backup = None;
    let mut runtime_result = Value::Null;
    let mut restore_result = None;
    let mut ok = true;
    let mut error = None;
    let mut restart_required = false;

    if !input.dry_run {
        backup = Some(write_config_backup(
            &state.config.config_dir,
            &state.config.live_yaml_path,
            &input.current.raw_text,
        )?);
        atomic_write_text(&state.config.live_yaml_path, &input.candidate.payload)?;

        runtime_result = run_live_service_apply_command(
            state,
            &state.config.live_yaml_path,
            &input.candidate.config_id,
            input.runtime_action,
        )
        .await;
        match parse_live_service_apply_proof(&runtime_result).and_then(|proof| {
            restart_required = restart_required_from_proof(&proof);
            verify_live_service_apply_proof(&proof, &input.candidate.config_id)
        }) {
            Ok(()) => {}
            Err(reason_text) => {
                ok = false;
                error = Some(reason_text);

                let recovery_result = match atomic_write_text(
                    &state.config.live_yaml_path,
                    &input.current.raw_text,
                ) {
                    Ok(()) => {
                        let raw = run_live_service_apply_command(
                            state,
                            &state.config.live_yaml_path,
                            input.current_config_id,
                            "auto",
                        )
                        .await;
                        let recovery_error = parse_live_service_apply_proof(&raw)
                            .and_then(|proof| {
                                verify_live_service_apply_proof(&proof, input.current_config_id)
                            })
                            .err();
                        if let Some(recovery_error) = recovery_error {
                            json!({
                                "ok": false,
                                "error": format!("runtime recovery verification failed: {recovery_error}"),
                                "result": raw,
                            })
                        } else {
                            raw
                        }
                    }
                    Err(err) => command_failure_value(format!(
                        "failed to restore the previous live YAML before recovery: {err}"
                    )),
                };
                restore_result = Some(recovery_result);
            }
        }
    }

    let event = build_live_transaction_event(
        state,
        LiveTransactionEventInput {
            version: input.event_version,
            action: input.action_name,
            reason: input.reason,
            dry_run: input.dry_run,
            current: input.current,
            current_config_id: input.current_config_id,
            candidate_label: input.candidate_label,
            candidate: input.candidate,
            runtime_result: &runtime_result,
            restore_result: restore_result.as_ref(),
            extra: input.extra_event_details,
        },
    );
    let event_text = serde_json::to_string_pretty(&event)? + "\n";
    atomic_write_text(&input.action_dir.join("event.json"), &event_text)?;

    Ok(LiveConfigTransactionResult {
        ok,
        action_dir: input.action_dir.to_string_lossy().to_string(),
        backup,
        restart_required,
        runtime_result,
        restore_result,
        error,
    })
}

fn prepare_live_apply(
    state: &AppState,
    headers: &HeaderMap,
    body: &ApplyLiveBody,
) -> Result<PreparedLiveApply, HubError> {
    let _: Value = serde_yaml::from_str(&body.yaml)
        .map_err(|err| HubError::BadRequest(format!("invalid YAML: {err}")))?;
    let restart_mode = LiveApplyRestartMode::parse(body.restart.as_deref())?;
    let reason = body.reason.clone().unwrap_or_default().trim().to_string();
    let current = resolve_current_config_state(state, "live")?;
    let expected_config_id =
        extract_expected_config_id_value(headers, body.expected_config_id.as_deref())?;
    if expected_config_id != current.lock_id {
        return Err(HubError::Conflict(format!(
            "stale config write for live: expected {expected_config_id} but current config is {}",
            current.lock_id
        )));
    }
    let incumbent = validate_candidate_config_write(
        state,
        "live",
        &state.config.live_yaml_path,
        &current.raw_text,
    )?;
    let candidate =
        validate_candidate_config_write(state, "live", &state.config.live_yaml_path, &body.yaml)?;
    let preview_restart_required = candidate.config_id != incumbent.config_id;
    Ok(PreparedLiveApply {
        current,
        incumbent,
        candidate,
        reason,
        restart_mode,
        preview_restart_required,
    })
}

fn prepare_live_rollback(
    state: &AppState,
    body: &RollbackLiveBody,
) -> Result<PreparedLiveRollback, HubError> {
    let steps = body.steps.unwrap_or(1).max(1) as usize;
    let restart_mode = LiveApplyRestartMode::parse(body.restart.as_deref())?;
    let reason = body.reason.clone().unwrap_or_default().trim().to_string();

    let deploy_root = state.config.artifacts_dir.join("deployments").join("live");
    let deploy_dirs = sorted_subdirs_desc(&deploy_root)?;
    if deploy_dirs.len() < steps {
        return Err(HubError::BadRequest(format!(
            "insufficient live deployments for rollback: requested {steps}, available {}",
            deploy_dirs.len()
        )));
    }
    let src_dir = deploy_dirs[steps - 1].clone();
    let mut restored_from = src_dir.join("prev_config.yaml");
    let mut restored_text = fs::read_to_string(&restored_from).unwrap_or_default();

    if restored_text.trim().is_empty() {
        if let Some(alt_dir) = deploy_dirs.get(steps) {
            for name in ["promoted_config.yaml", "prev_config.yaml"] {
                let candidate = alt_dir.join(name);
                let txt = fs::read_to_string(&candidate).unwrap_or_default();
                if !txt.trim().is_empty() {
                    restored_text = txt;
                    restored_from = candidate;
                    break;
                }
            }
        }
    }
    if restored_text.trim().is_empty() {
        return Err(HubError::BadRequest("missing rollback config".to_string()));
    }
    serde_yaml::from_str::<serde_yaml::Value>(&restored_text)
        .map_err(|err| HubError::BadRequest(format!("rollback config invalid YAML: {err}")))?;

    let current = resolve_current_config_state(state, "live")?;
    let incumbent = validate_candidate_config_write(
        state,
        "live",
        &state.config.live_yaml_path,
        &current.raw_text,
    )?;
    let restored = validate_candidate_config_write(
        state,
        "live",
        &state.config.live_yaml_path,
        &restored_text,
    )?;

    Ok(PreparedLiveRollback {
        steps,
        src_dir,
        restored_from,
        current,
        incumbent,
        restored,
        reason,
        restart_mode,
    })
}

async fn execute_live_apply(
    state: &AppState,
    headers: &HeaderMap,
    prepared: PreparedLiveApply,
    request_id: Option<&str>,
    requester: Option<crate::config_audit::ConfigAuditActor>,
    approver: Option<crate::config_audit::ConfigAuditActor>,
) -> Result<Json<Value>, HubError> {
    let runtime_action = prepared.restart_mode.runtime_action(false)?;
    let apply_dir = build_action_dir(&state.config.artifacts_dir.join("applies").join("live"))?;
    let transaction = execute_live_config_transaction(
        state,
        LiveConfigTransactionInput {
            action_dir: &apply_dir,
            event_version: "apply_event_v1",
            action_name: "apply_live",
            current: &prepared.current,
            current_config_id: &prepared.incumbent.config_id,
            candidate_label: "candidate_config.yaml",
            candidate: &prepared.candidate,
            reason: &prepared.reason,
            runtime_action,
            dry_run: false,
            extra_event_details: json!({
                "restart_mode": prepared.restart_mode.as_str(),
            }),
        },
    )
    .await?;

    let approver_actor = approver
        .clone()
        .unwrap_or_else(|| actor_from_headers(headers, "approver_token"));
    append_config_audit_event(
        state,
        approver_actor.clone(),
        "live",
        "apply_live",
        Some(prepared.reason.as_str()),
        Some(&prepared.current.lock_id),
        Some(&prepared.incumbent.config_id),
        Some(&prepared.candidate.lock_id),
        Some(&prepared.candidate.config_id),
        json!({
            "kind": "runtime_validation",
            "ok": true,
            "config_id": prepared.candidate.config_id,
        }),
        json!({
            "ok": transaction.ok,
            "restart_required": transaction.restart_required,
            "runtime_apply": transaction.runtime_result,
            "recovery": transaction.restore_result,
        }),
        Some(&transaction.action_dir),
        request_id,
        requester,
        Some(approver_actor),
    )?;

    Ok(Json(json!({
        "ok": transaction.ok,
        "action": "apply_live",
        "backup": transaction.backup,
        "lock_id": prepared.candidate.lock_id,
        "config_id": prepared.candidate.config_id,
        "previous_lock_id": prepared.current.lock_id,
        "previous_config_id": prepared.incumbent.config_id,
        "restart_required": transaction.restart_required,
        "service": state.config.live_service,
        "artifact_path_redacted": true,
        "request_id": request_id,
        "restart": {
            "mode": prepared.restart_mode.as_str(),
            "result": summarise_runtime_apply_result(&transaction.runtime_result),
            "recovery": transaction.restore_result.as_ref().map(summarise_runtime_apply_result),
        },
        "error": transaction.error,
    })))
}

async fn execute_live_rollback(
    state: &AppState,
    headers: &HeaderMap,
    prepared: PreparedLiveRollback,
    request_id: Option<&str>,
    requester: Option<crate::config_audit::ConfigAuditActor>,
    approver: Option<crate::config_audit::ConfigAuditActor>,
) -> Result<Json<Value>, HubError> {
    let runtime_action = prepared.restart_mode.runtime_action(false)?;
    let rollback_dir =
        build_action_dir(&state.config.artifacts_dir.join("rollbacks").join("live"))?;
    let transaction = execute_live_config_transaction(
        state,
        LiveConfigTransactionInput {
            action_dir: &rollback_dir,
            event_version: "rollback_event_v2",
            action_name: "rollback_live",
            current: &prepared.current,
            current_config_id: &prepared.incumbent.config_id,
            candidate_label: "restored_config.yaml",
            candidate: &prepared.restored,
            reason: &prepared.reason,
            runtime_action,
            dry_run: false,
            extra_event_details: json!({
                "restart_mode": prepared.restart_mode.as_str(),
                "steps": prepared.steps,
                "source_deploy_dir": prepared.src_dir.to_string_lossy().to_string(),
                "restored_from": prepared.restored_from.to_string_lossy().to_string(),
            }),
        },
    )
    .await?;

    let approver_actor = approver
        .clone()
        .unwrap_or_else(|| actor_from_headers(headers, "approver_token"));
    append_config_audit_event(
        state,
        approver_actor.clone(),
        "live",
        "rollback_live",
        Some(prepared.reason.as_str()),
        Some(&prepared.current.lock_id),
        Some(&prepared.incumbent.config_id),
        Some(&prepared.restored.lock_id),
        Some(&prepared.restored.config_id),
        json!({
            "kind": "runtime_validation",
            "ok": true,
            "config_id": prepared.restored.config_id,
        }),
        json!({
            "ok": transaction.ok,
            "restart_required": transaction.restart_required,
            "runtime_apply": transaction.runtime_result,
            "recovery": transaction.restore_result,
            "steps": prepared.steps,
        }),
        Some(&transaction.action_dir),
        request_id,
        requester,
        Some(approver_actor),
    )?;

    Ok(Json(json!({
        "ok": transaction.ok,
        "action": "rollback_live",
        "backup": transaction.backup,
        "steps": prepared.steps,
        "previous_lock_id": prepared.current.lock_id,
        "previous_config_id": prepared.incumbent.config_id,
        "restored_lock_id": prepared.restored.lock_id,
        "restored_config_id": prepared.restored.config_id,
        "restart_required": transaction.restart_required,
        "artifact_path_redacted": true,
        "request_id": request_id,
        "restart": {
            "mode": prepared.restart_mode.as_str(),
            "service": state.config.live_service,
            "result": summarise_runtime_apply_result(&transaction.runtime_result),
            "recovery": transaction.restore_result.as_ref().map(summarise_runtime_apply_result),
        },
        "error": transaction.error,
    })))
}

// ── Handlers ────────────────────────────────────────────────────────

/// GET /api/config — Read YAML file, return as JSON.
async fn get_config(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Response, HubError> {
    let file_variant = variant(&q);
    let resolved = resolve_current_config_state(&state, file_variant)?;
    let val: Value = serde_yaml::from_str(&resolved.raw_text)?;
    let mut response = Json(val).into_response();
    apply_config_identity_headers(response.headers_mut(), &resolved)?;
    Ok(response)
}

/// GET /api/config/raw — Read YAML file, return raw text.
async fn get_config_raw(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Response, HubError> {
    let resolved = resolve_current_config_state(&state, variant(&q))?;
    let message = format!(
        "# redacted\n# raw config access is privileged; use /api/config/raw/privileged?file={} with admin auth\n",
        variant(&q)
    );
    let mut response = message.into_response();
    apply_config_identity_headers(response.headers_mut(), &resolved)?;
    Ok(response)
}

/// GET /api/config/raw/privileged — Read raw YAML through an explicit privileged route.
async fn get_config_raw_privileged(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PrivilegedConfigQuery>,
    headers: HeaderMap,
) -> Result<Response, HubError> {
    let file_variant = q.file.as_deref().unwrap_or("main");
    let resolved = resolve_current_config_state(&state, file_variant)?;
    append_diagnostic_read_audit(
        &state,
        &headers,
        "config_raw_privileged",
        &format!("file={file_variant}"),
    )?;
    let mut response = resolved.raw_text.clone().into_response();
    apply_config_identity_headers(response.headers_mut(), &resolved)?;
    Ok(response)
}

/// PUT /api/config — Write YAML with atomic backup.
async fn put_config(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
    headers: HeaderMap,
    Json(body): Json<ConfigWriteBody>,
) -> Result<Json<Value>, HubError> {
    // Validate YAML parses.
    let _: Value = serde_yaml::from_str(&body.yaml)
        .map_err(|e| HubError::BadRequest(format!("invalid YAML: {e}")))?;

    let file_variant = variant(&q);
    let current = resolve_current_config_state(&state, file_variant)?;
    let expected_config_id = extract_expected_config_id(&headers, &body)?;
    if expected_config_id != current.lock_id {
        return Err(HubError::Conflict(format!(
            "stale config write for {file_variant}: expected {expected_config_id} but current config is {}",
            current.lock_id
        )));
    }
    let validated =
        validate_candidate_config_write(&state, file_variant, &current.path, &body.yaml)?;

    let backup_path_str =
        write_config_backup(&state.config.config_dir, &current.path, &current.raw_text)?;

    atomic_write_text(&current.path, &validated.payload)?;

    append_config_audit_event(
        &state,
        actor_from_headers(&headers, "editor_token"),
        file_variant,
        "save_config",
        body.reason.as_deref(),
        Some(&current.lock_id),
        current.runtime_config_id.as_deref(),
        Some(&validated.lock_id),
        Some(&validated.config_id),
        json!({
            "kind": "runtime_validation",
            "ok": true,
            "config_id": validated.config_id,
        }),
        json!({
            "ok": true,
            "backup": backup_path_str,
        }),
        None,
        None,
        None,
        None,
    )?;

    Ok(Json(serde_json::json!({
        "ok": true,
        "file": file_variant,
        "backup": backup_path_str,
        "lock_id": validated.lock_id,
        "config_id": validated.config_id,
    })))
}

/// POST /api/config/reload — Retired false-reload surface kept only to fail closed.
async fn post_config_reload(
    State(_state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Json<Value>, HubError> {
    let file_variant = variant(&q);
    Err(HubError::BadRequest(format!(
        "reload semantics were retired for {file_variant}; save non-live YAML directly, use /api/config/actions/apply-live for live changes, or restart the affected service explicitly"
    )))
}

/// POST /api/config/actions/apply-live — Preview a live apply without mutating runtime state.
async fn post_preview_apply_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ApplyLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    if !body.dry_run.unwrap_or(false) {
        return Err(HubError::BadRequest(
            "direct live apply execution retired; create a pending live apply request instead"
                .to_string(),
        ));
    }
    let prepared = prepare_live_apply(&state, &headers, &body)?;
    Ok(Json(json!({
        "ok": true,
        "action": "apply_live_preview",
        "lock_id": prepared.candidate.lock_id,
        "config_id": prepared.candidate.config_id,
        "previous_lock_id": prepared.current.lock_id,
        "previous_config_id": prepared.incumbent.config_id,
        "restart_required": prepared.preview_restart_required,
        "service": state.config.live_service,
        "dry_run": true,
    })))
}

async fn post_apply_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ApplyLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let prepared = prepare_live_apply(&state, &headers, &body)?;
    execute_live_apply(&state, &headers, prepared, None, None, None).await
}

/// POST /api/config/actions/apply-live/request — Create a pending live apply request.
async fn post_request_apply_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ApplyLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let prepared = prepare_live_apply(&state, &headers, &body)?;
    let requester = actor_from_headers(&headers, "editor_token");
    let request = create_config_approval_request(
        &state.config.artifacts_dir,
        ConfigApprovalAction::ApplyLive,
        requester.clone(),
        normalise_reason(Some(prepared.reason.as_str())),
        json!({
            "previous_lock_id": prepared.current.lock_id,
            "previous_config_id": prepared.incumbent.config_id,
            "lock_id": prepared.candidate.lock_id,
            "config_id": prepared.candidate.config_id,
            "restart_required": prepared.preview_restart_required,
            "service": state.config.live_service,
        }),
        json!({
            "yaml": body.yaml,
            "expected_config_id": prepared.current.lock_id,
            "reason": prepared.reason,
            "restart": prepared.restart_mode.as_str(),
            "dry_run": false,
        }),
    )?;
    append_config_audit_event(
        &state,
        requester.clone(),
        "live",
        "apply_live_requested",
        request.reason.as_deref(),
        Some(&prepared.current.lock_id),
        Some(&prepared.incumbent.config_id),
        Some(&prepared.candidate.lock_id),
        Some(&prepared.candidate.config_id),
        json!({ "kind": "request_created", "ok": true }),
        json!({ "ok": true, "status": "pending" }),
        None,
        Some(&request.request_id),
        Some(requester),
        None,
    )?;
    Ok(Json(json!({
        "ok": true,
        "request_id": request.request_id,
        "action": "apply_live_request",
        "previous_config_id": prepared.incumbent.config_id,
        "config_id": prepared.candidate.config_id,
        "restart_required": prepared.preview_restart_required,
        "service": state.config.live_service,
        "status": "pending",
    })))
}

/// GET /api/config/audit — Read recent append-only config mutation events.
async fn get_config_audit(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigAuditQuery>,
) -> Result<Json<Vec<Value>>, HubError> {
    let events = read_recent_events(
        &state.config.artifacts_dir,
        q.file.as_deref(),
        q.limit.unwrap_or(50),
    )?;
    Ok(Json(
        events
            .iter()
            .map(redacted_config_audit_event)
            .collect::<Vec<_>>(),
    ))
}

/// GET /api/config/approvals — List pending live approval requests.
async fn get_config_approvals(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigApprovalListQuery>,
) -> Result<Json<Value>, HubError> {
    if let Some(status) = q.status.as_deref().filter(|value| !value.trim().is_empty()) {
        if !status.eq_ignore_ascii_case("pending") {
            return Ok(Json(json!({ "requests": [] })));
        }
    }
    let requests = list_pending_requests(&state.config.artifacts_dir)?;
    Ok(Json(json!({ "requests": requests })))
}

/// GET /api/config/audit/privileged — Read full config audit events.
async fn get_config_audit_privileged(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigAuditQuery>,
    headers: HeaderMap,
) -> Result<Json<Vec<ConfigAuditEvent>>, HubError> {
    append_diagnostic_read_audit(
        &state,
        &headers,
        "config_audit_privileged",
        &format!("file={}", q.file.as_deref().unwrap_or("all")),
    )?;
    let events = read_recent_events(
        &state.config.artifacts_dir,
        q.file.as_deref(),
        q.limit.unwrap_or(50),
    )?;
    Ok(Json(events))
}

/// GET /api/config/history — List backup files for a config variant.
async fn get_config_history(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Json<Vec<BackupEntry>>, HubError> {
    let file_variant = variant(&q);
    let path = resolve_config_path_for_state(&state, file_variant)?;
    let stem = path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("strategy_overrides.yaml");

    let bk_dir = backups_dir(&state.config.config_dir)?;
    let mut entries = Vec::new();

    if bk_dir.exists() {
        for entry in fs::read_dir(&bk_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(stem) && name.ends_with(".bak") {
                let meta = entry.metadata()?;
                let modified = meta
                    .modified()
                    .ok()
                    .map(|t| {
                        let dt: chrono::DateTime<chrono::Utc> = t.into();
                        dt.to_rfc3339()
                    })
                    .unwrap_or_default();
                entries.push(BackupEntry {
                    filename: name,
                    modified,
                    size: meta.len(),
                });
            }
        }
    }

    // Sort newest first.
    entries.sort_by(|a, b| b.filename.cmp(&a.filename));

    Ok(Json(entries))
}

/// GET /api/config/diff — Simple line-by-line diff between two versions.
async fn get_config_diff(
    State(_state): State<Arc<AppState>>,
    Query(q): Query<DiffQuery>,
) -> Result<Json<Value>, HubError> {
    Ok(Json(serde_json::json!({
        "a": q.a,
        "b": q.b,
        "file": q.file.as_deref().unwrap_or("main"),
        "redacted": true,
        "message": "raw config diffs are privileged; use /api/config/diff/privileged with admin auth",
    })))
}

/// GET /api/config/diff/privileged — Full raw diff through an explicit privileged route.
async fn get_config_diff_privileged(
    State(state): State<Arc<AppState>>,
    Query(q): Query<DiffQuery>,
    headers: HeaderMap,
) -> Result<Json<Value>, HubError> {
    let file_variant = q.file.as_deref().unwrap_or("main");
    append_diagnostic_read_audit(
        &state,
        &headers,
        "config_diff_privileged",
        &format!("file={file_variant}"),
    )?;
    let config_path = resolve_config_path_for_state(&state, file_variant)?;
    let bk_dir = backups_dir(&state.config.config_dir)?;

    let read_version = |version: &str| -> Result<String, HubError> {
        if version == "current" {
            if !config_path.exists() {
                return Err(HubError::NotFound("current config not found".into()));
            }
            Ok(fs::read_to_string(&config_path)?)
        } else {
            // version is a backup filename
            let bk_path = bk_dir.join(version);
            // Safety: ensure the resolved path is inside backups dir
            let canonical = bk_path
                .canonicalize()
                .map_err(|_| HubError::NotFound(format!("backup not found: {version}")))?;
            let bk_dir_canon = bk_dir.canonicalize().unwrap_or_else(|_| bk_dir.clone());
            if !canonical.starts_with(&bk_dir_canon) {
                return Err(HubError::BadRequest("invalid backup path".into()));
            }
            Ok(fs::read_to_string(&canonical)?)
        }
    };

    let text_a = read_version(&q.a)?;
    let text_b = read_version(&q.b)?;

    // Simple unified-style diff.
    let lines_a: Vec<&str> = text_a.lines().collect();
    let lines_b: Vec<&str> = text_b.lines().collect();

    let mut diff_lines = Vec::new();
    let max_len = lines_a.len().max(lines_b.len());
    for i in 0..max_len {
        let la = lines_a.get(i).copied().unwrap_or("");
        let lb = lines_b.get(i).copied().unwrap_or("");
        if la != lb {
            if !la.is_empty() {
                diff_lines.push(format!("-{la}"));
            }
            if !lb.is_empty() {
                diff_lines.push(format!("+{lb}"));
            }
        } else {
            diff_lines.push(format!(" {la}"));
        }
    }

    Ok(Json(serde_json::json!({
        "a": q.a,
        "b": q.b,
        "file": file_variant,
        "diff": diff_lines,
    })))
}

/// GET /api/config/files — List available config files with metadata.
async fn get_config_files(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Value>>, HubError> {
    let variants = [
        "main",
        "live",
        "livepaper",
        "paper1",
        "paper2",
        "paper3",
        "promoted_primary",
        "promoted_fallback",
    ];

    let mut files = Vec::new();
    for v in &variants {
        if let Ok(path) = resolve_config_path_for_state(&state, v) {
            let exists = path.exists();
            let (modified, size) = if exists {
                let meta = fs::metadata(&path).ok();
                let modified = meta
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .map(|t| {
                        let dt: chrono::DateTime<chrono::Utc> = t.into();
                        dt.to_rfc3339()
                    })
                    .unwrap_or_default();
                let size = meta.map(|m| m.len()).unwrap_or(0);
                (modified, size)
            } else {
                (String::new(), 0)
            };
            files.push(serde_json::json!({
                "variant": v,
                "filename": path.file_name().unwrap_or_default().to_str().unwrap_or(""),
                "exists": exists,
                "modified": modified,
                "size": size,
            }));
        }
    }

    Ok(Json(files))
}

/// POST /api/config/actions/promote-live — Promote a selected paper config to live.
async fn post_promote_live(
    State(_state): State<Arc<AppState>>,
    Json(_body): Json<Value>,
) -> Result<Json<Value>, HubError> {
    Err(HubError::Forbidden(
        "live promotion tooling was retired with the zero-Python repository cutover".into(),
    ))
}

/// POST /api/config/actions/rollback-live — Direct rollback execution is retired; use a request route.
async fn post_preview_rollback_live(
    State(_state): State<Arc<AppState>>,
    Json(_body): Json<RollbackLiveBody>,
) -> Result<Json<Value>, HubError> {
    Err(HubError::BadRequest(
        "direct live rollback execution retired; create a pending rollback request instead"
            .to_string(),
    ))
}

async fn post_rollback_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RollbackLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let prepared = prepare_live_rollback(&state, &body)?;
    execute_live_rollback(&state, &headers, prepared, None, None, None).await
}

/// POST /api/config/actions/rollback-live/request — Create a pending live rollback request.
async fn post_request_rollback_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RollbackLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    if body.dry_run.unwrap_or(false) {
        return Err(HubError::BadRequest(
            "rollback request creation does not support dry_run; create a pending request instead"
                .to_string(),
        ));
    }
    let prepared = prepare_live_rollback(&state, &body)?;
    let requester = actor_from_headers(&headers, "editor_token");
    let request = create_config_approval_request(
        &state.config.artifacts_dir,
        ConfigApprovalAction::RollbackLive,
        requester.clone(),
        normalise_reason(Some(prepared.reason.as_str())),
        json!({
            "steps": prepared.steps,
            "previous_lock_id": prepared.current.lock_id,
            "previous_config_id": prepared.incumbent.config_id,
            "restored_lock_id": prepared.restored.lock_id,
            "restored_config_id": prepared.restored.config_id,
            "restart_required": prepared.restored.config_id != prepared.incumbent.config_id,
            "service": state.config.live_service,
        }),
        json!({
            "steps": prepared.steps,
            "reason": prepared.reason,
            "restart": prepared.restart_mode.as_str(),
            "dry_run": false,
        }),
    )?;
    append_config_audit_event(
        &state,
        requester.clone(),
        "live",
        "rollback_live_requested",
        request.reason.as_deref(),
        Some(&prepared.current.lock_id),
        Some(&prepared.incumbent.config_id),
        Some(&prepared.restored.lock_id),
        Some(&prepared.restored.config_id),
        json!({ "kind": "request_created", "ok": true }),
        json!({ "ok": true, "status": "pending", "steps": prepared.steps }),
        None,
        Some(&request.request_id),
        Some(requester),
        None,
    )?;
    Ok(Json(json!({
        "ok": true,
        "request_id": request.request_id,
        "action": "rollback_live_request",
        "steps": prepared.steps,
        "previous_config_id": prepared.incumbent.config_id,
        "restored_config_id": prepared.restored.config_id,
        "restart_required": prepared.restored.config_id != prepared.incumbent.config_id,
        "service": state.config.live_service,
        "status": "pending",
    })))
}

/// POST /api/config/approvals/{request_id}/approve — Approve and execute a pending live request.
async fn post_approve_config_approval(
    State(state): State<Arc<AppState>>,
    AxumPath(request_id): AxumPath<String>,
    headers: HeaderMap,
    Json(_body): Json<ConfigApprovalDecisionBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let request = load_for_processing(&state.config.artifacts_dir, &request_id)?;
    let requester = request.requester.clone();
    let approver = actor_from_headers(&headers, "approver_token");

    let result = match request.action {
        ConfigApprovalAction::ApplyLive => {
            let body: ApplyLiveBody = serde_json::from_value(request.execute.clone())?;
            let prepared = prepare_live_apply(&state, &HeaderMap::new(), &body)?;
            execute_live_apply(
                &state,
                &headers,
                prepared,
                Some(&request.request_id),
                Some(requester),
                Some(approver.clone()),
            )
            .await
            .map(|json| json.0)
        }
        ConfigApprovalAction::RollbackLive => {
            let body: RollbackLiveBody = serde_json::from_value(request.execute.clone())?;
            let prepared = prepare_live_rollback(&state, &body)?;
            execute_live_rollback(
                &state,
                &headers,
                prepared,
                Some(&request.request_id),
                Some(requester),
                Some(approver.clone()),
            )
            .await
            .map(|json| json.0)
        }
    };

    match result {
        Ok(payload) => {
            let request = mark_approved(
                &state.config.artifacts_dir,
                request,
                approver,
                payload.clone(),
            )?;
            Ok(Json(json!({
                "ok": payload.get("ok").and_then(|value| value.as_bool()).unwrap_or(false),
                "request_id": request.request_id,
                "status": "approved",
                "result": payload,
            })))
        }
        Err(err) => {
            let failed = mark_failed(
                &state.config.artifacts_dir,
                request,
                approver,
                json!({ "ok": false, "error": err.to_string() }),
            )?;
            Ok(Json(json!({
                "ok": false,
                "request_id": failed.request_id,
                "status": "failed",
                "error": err.to_string(),
            })))
        }
    }
}

/// POST /api/config/approvals/{request_id}/reject — Reject a pending live request.
async fn post_reject_config_approval(
    State(state): State<Arc<AppState>>,
    AxumPath(request_id): AxumPath<String>,
    headers: HeaderMap,
    Json(body): Json<ConfigApprovalDecisionBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let request = load_for_processing(&state.config.artifacts_dir, &request_id)?;
    let approver = actor_from_headers(&headers, "approver_token");
    let rejected = mark_rejected(
        &state.config.artifacts_dir,
        request.clone(),
        approver.clone(),
        normalise_reason(body.reason.as_deref()),
    )?;
    append_config_audit_event(
        &state,
        approver.clone(),
        "live",
        &format!("{}_rejected", request.action.as_str()),
        rejected.reason.as_deref(),
        None,
        None,
        None,
        None,
        json!({ "kind": "request_rejected", "ok": true }),
        json!({ "ok": true, "status": "rejected" }),
        None,
        Some(&rejected.request_id),
        Some(rejected.requester.clone()),
        Some(approver),
    )?;
    Ok(Json(json!({
        "ok": true,
        "request_id": rejected.request_id,
        "status": "rejected",
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{env_lock, EnvGuard};
    use axum::extract::State;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    fn write_main_config(root: &Path, yaml: &str) -> PathBuf {
        let config_dir = root.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("strategy_overrides.yaml");
        fs::write(&config_path, yaml).unwrap();
        config_path
    }

    fn write_live_config(root: &Path, yaml: &str) -> PathBuf {
        let config_dir = root.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("strategy_overrides.live.yaml");
        fs::write(&config_path, yaml).unwrap();
        config_path
    }

    fn test_state(root: &Path) -> Arc<AppState> {
        let mut config = crate::config::HubConfig::from_env();
        config.aiq_root = root.to_path_buf();
        config.config_dir = root.join("config");
        config.live_yaml_path = config.config_dir.join("strategy_overrides.live.yaml");
        config.artifacts_dir = root.join("artifacts");
        AppState::new(config)
    }

    fn admin_test_state(root: &Path, runtime_bin: &Path) -> Arc<AppState> {
        let mut config = crate::config::HubConfig::from_env();
        config.aiq_root = root.to_path_buf();
        config.config_dir = root.join("config");
        config.live_yaml_path = config.config_dir.join("strategy_overrides.live.yaml");
        config.artifacts_dir = root.join("artifacts");
        config.admin_actions_enabled = true;
        config.runtime_bin = runtime_bin.to_path_buf();
        AppState::new(config)
    }

    fn admin_test_state_with_live_yaml(
        root: &Path,
        runtime_bin: &Path,
        live_yaml_path: &Path,
    ) -> Arc<AppState> {
        let mut config = crate::config::HubConfig::from_env();
        config.aiq_root = root.to_path_buf();
        config.config_dir = root.join("config");
        config.live_yaml_path = live_yaml_path.to_path_buf();
        config.artifacts_dir = root.join("artifacts");
        config.admin_actions_enabled = true;
        config.runtime_bin = runtime_bin.to_path_buf();
        AppState::new(config)
    }

    fn if_match_headers(config_id: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::IF_MATCH, HeaderValue::from_str(config_id).unwrap());
        headers
    }

    fn fake_live_service_apply_report(
        config_id: &str,
        final_state: &str,
        contract_matches_status: bool,
    ) -> Value {
        let preview_action = if final_state == "running" {
            "restart"
        } else {
            "hold"
        };
        json!({
            "ok": final_state == "running" && contract_matches_status,
            "applied_action": "restart",
            "action_reason": "test live apply",
            "preview": {
                "ok": true,
                "desired_action": preview_action,
                "action_reason": "preview",
                "status": {
                    "ok": true,
                    "service_state": "restart_required",
                    "contract_matches_status": false,
                    "mismatch_reasons": ["config drift"],
                    "manifest": {
                        "launch_state": "ready",
                        "config_id": config_id,
                    },
                    "daemon_status": {
                        "ok": true,
                        "running": true,
                        "errors": [],
                    },
                },
            },
            "final_service": {
                "ok": final_state == "running" && contract_matches_status,
                "desired_action": "monitor",
                "action_reason": "final",
                "status": {
                    "ok": final_state == "running" && contract_matches_status,
                    "service_state": final_state,
                    "contract_matches_status": contract_matches_status,
                    "mismatch_reasons": if contract_matches_status { json!([]) } else { json!(["config drift"]) },
                    "manifest": {
                        "launch_state": "ready",
                        "config_id": config_id,
                    },
                    "daemon_status": {
                        "ok": final_state == "running",
                        "running": final_state == "running",
                        "errors": if final_state == "running" { json!([]) } else { json!(["daemon unhealthy"]) },
                    },
                },
            },
        })
    }

    fn write_fake_runtime_script(
        root: &Path,
        first: &Value,
        second: &Value,
    ) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let responses_dir = root.join("runtime-fixtures");
        fs::create_dir_all(&responses_dir).unwrap();
        let script_path = responses_dir.join("fake-aiq-runtime.sh");
        let count_path = responses_dir.join("count.txt");
        let first_path = responses_dir.join("first.json");
        let second_path = responses_dir.join("second.json");
        fs::write(&first_path, serde_json::to_string(first).unwrap()).unwrap();
        fs::write(&second_path, serde_json::to_string(second).unwrap()).unwrap();
        fs::write(
            &script_path,
            r#"#!/bin/sh
count_file="${AIQ_TEST_RUNTIME_COUNT_FILE:?}"
first_path="${AIQ_TEST_RUNTIME_RESPONSE_1:?}"
second_path="${AIQ_TEST_RUNTIME_RESPONSE_2:?}"
count=0
if [ -f "$count_file" ]; then
  count=$(cat "$count_file")
fi
count=$((count + 1))
printf "%s" "$count" > "$count_file"
if [ "$count" -eq 1 ]; then
  cat "$first_path"
else
  cat "$second_path"
fi
"#,
        )
        .unwrap();
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
        (script_path, count_path, first_path, second_path)
    }

    #[tokio::test]
    async fn get_config_raw_returns_current_config_identity_headers() {
        let dir = tempdir().unwrap();
        write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = test_state(dir.path());

        let response = get_config_raw(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
        )
        .await
        .unwrap();

        assert!(response.headers().contains_key(header::ETAG));
        assert!(response.headers().contains_key("x-aiq-config-lock-id"));
        assert!(response.headers().contains_key("x-aiq-config-id"));
        assert_eq!(
            response.headers()["x-aiq-config-id-source"],
            HeaderValue::from_static("runtime")
        );
    }

    #[tokio::test]
    async fn get_config_raw_privileged_returns_raw_yaml_and_audits_access() {
        let dir = tempdir().unwrap();
        write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = admin_test_state(dir.path(), Path::new("/bin/true"));
        let mut headers = HeaderMap::new();
        headers.insert("x-aiq-actor", HeaderValue::from_static("diag-user"));

        let response = get_config_raw_privileged(
            State(Arc::clone(&state)),
            Query(PrivilegedConfigQuery {
                file: Some("main".to_string()),
            }),
            headers,
        )
        .await
        .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("interval: 30m"));
        assert!(state
            .config
            .artifacts_dir
            .join("diagnostic_audit")
            .join("read_events.jsonl")
            .exists());
    }

    #[tokio::test]
    async fn post_config_reload_retires_false_reload_semantics() {
        let dir = tempdir().unwrap();
        write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = test_state(dir.path());

        let err = post_config_reload(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
        )
        .await
        .unwrap_err();

        match err {
            HubError::BadRequest(message) => {
                assert!(message.contains("reload semantics were retired"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_config_rejects_runtime_invalid_yaml_before_write() {
        let dir = tempdir().unwrap();
        let config_path = write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let original = fs::read_to_string(&config_path).unwrap();
        let state = test_state(dir.path());
        let current = resolve_current_config_state(&state, "main").unwrap();

        let err = put_config(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
            if_match_headers(&current.lock_id),
            Json(ConfigWriteBody {
                yaml: "global:\n  trade:\n    leverage: nope\n".to_string(),
                expected_config_id: None,
                reason: None,
            }),
        )
        .await
        .unwrap_err();

        match err {
            HubError::BadRequest(message) => {
                assert!(message.contains("runtime validation failed"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(fs::read_to_string(&config_path).unwrap(), original);
    }

    #[tokio::test]
    async fn put_config_rejects_stale_expected_config_id() {
        let dir = tempdir().unwrap();
        let config_path = write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = test_state(dir.path());
        let current = resolve_current_config_state(&state, "main").unwrap();

        fs::write(
            &config_path,
            "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        )
        .unwrap();

        let err = put_config(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
            if_match_headers(&current.lock_id),
            Json(ConfigWriteBody {
                yaml: "global:\n  engine:\n    interval: 5m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n".to_string(),
                expected_config_id: None,
                reason: None,
            }),
        )
        .await
        .unwrap_err();

        match err {
            HubError::Conflict(message) => {
                assert!(message.contains("stale config write"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(fs::read_to_string(&config_path)
            .unwrap()
            .contains("interval: 15m"));
    }

    #[tokio::test]
    async fn put_config_rejects_stale_raw_text_edits_even_when_runtime_config_is_unchanged() {
        let dir = tempdir().unwrap();
        let config_path = write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = test_state(dir.path());
        let current = resolve_current_config_state(&state, "main").unwrap();

        fs::write(
            &config_path,
            "# operator note\nglobal:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        )
        .unwrap();

        let err = put_config(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
            if_match_headers(&current.lock_id),
            Json(ConfigWriteBody {
                yaml: "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 3.0\n".to_string(),
                expected_config_id: None,
                reason: None,
            }),
        )
        .await
        .unwrap_err();

        match err {
            HubError::Conflict(message) => {
                assert!(message.contains("stale config write"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(fs::read_to_string(&config_path)
            .unwrap()
            .contains("operator note"));
    }

    #[tokio::test]
    async fn put_config_appends_config_audit_event() {
        let dir = tempdir().unwrap();
        let config_path = write_main_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let state = test_state(dir.path());
        let current = resolve_current_config_state(&state, "main").unwrap();
        let mut headers = if_match_headers(&current.lock_id);
        headers.insert("x-aiq-actor", HeaderValue::from_static("operator-jt"));
        headers.insert(header::USER_AGENT, HeaderValue::from_static("config-test"));

        let Json(response) = put_config(
            State(Arc::clone(&state)),
            Query(ConfigQuery {
                file: Some("main".to_string()),
            }),
            headers,
            Json(ConfigWriteBody {
                yaml: "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.5\n".to_string(),
                expected_config_id: None,
                reason: Some("operator save".to_string()),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(true));
        assert!(fs::read_to_string(&config_path)
            .unwrap()
            .contains("interval: 15m"));

        let events = read_recent_events(&state.config.artifacts_dir, Some("main"), 10).unwrap();
        let latest = events.first().expect("config audit event");
        assert_eq!(latest.action, "save_config");
        assert_eq!(latest.actor.label, "operator-jt");
        assert_eq!(latest.reason.as_deref(), Some("operator save"));
        assert_eq!(
            latest.before.lock_id.as_deref(),
            Some(current.lock_id.as_str())
        );
        assert_eq!(
            latest.after.config_id.as_deref(),
            response["config_id"].as_str()
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn apply_live_writes_candidate_only_after_runtime_proof_succeeds() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let live_path = write_live_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let candidate_yaml =
            "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 3.0\n";

        let baseline_state = test_state(dir.path());
        let current = resolve_current_config_state(&baseline_state, "live").unwrap();
        let candidate =
            validate_candidate_config_write(&baseline_state, "live", &live_path, candidate_yaml)
                .unwrap();

        let success_report = fake_live_service_apply_report(&candidate.config_id, "running", true);
        let (runtime_bin, count_path, first_path, second_path) =
            write_fake_runtime_script(dir.path(), &success_report, &success_report);
        let _env = EnvGuard::set(&[
            (
                "AIQ_TEST_RUNTIME_COUNT_FILE",
                Some(count_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_1",
                Some(first_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_2",
                Some(second_path.to_str().unwrap()),
            ),
        ]);
        let state = admin_test_state(dir.path(), &runtime_bin);

        let Json(response) = post_apply_live(
            State(Arc::clone(&state)),
            if_match_headers(&current.lock_id),
            Json(ApplyLiveBody {
                yaml: candidate_yaml.to_string(),
                expected_config_id: None,
                reason: Some("test apply".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(true));
        assert_eq!(response["artifact_path_redacted"], Value::Bool(true));
        assert_eq!(
            response["config_id"],
            Value::String(candidate.config_id.clone())
        );
        assert_eq!(
            fs::read_to_string(&live_path).unwrap(),
            normalise_yaml_payload(candidate_yaml)
        );

        let events = read_recent_events(&state.config.artifacts_dir, Some("live"), 10).unwrap();
        let latest = events
            .iter()
            .find(|event| event.action == "apply_live")
            .unwrap();
        let expected_current_config_id =
            validate_candidate_config_write(&baseline_state, "live", &live_path, &current.raw_text)
                .unwrap()
                .config_id;
        assert_eq!(
            latest.before.config_id.as_deref(),
            Some(expected_current_config_id.as_str())
        );
        assert_eq!(
            latest.after.config_id.as_deref(),
            Some(candidate.config_id.as_str())
        );
    }

    #[tokio::test]
    async fn apply_live_dry_run_reports_restart_requirement_without_mutating_file() {
        let dir = tempdir().unwrap();
        let live_yaml =
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n";
        let live_path = write_live_config(dir.path(), live_yaml);
        let candidate_yaml =
            "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 3.0\n";
        let state = admin_test_state(dir.path(), Path::new("/bin/true"));
        let current = resolve_current_config_state(&state, "live").unwrap();

        let Json(response) = post_preview_apply_live(
            State(Arc::clone(&state)),
            if_match_headers(&current.lock_id),
            Json(ApplyLiveBody {
                yaml: candidate_yaml.to_string(),
                expected_config_id: None,
                reason: Some("preview".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(true),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(true));
        assert_eq!(response["restart_required"], Value::Bool(true));
        assert_eq!(fs::read_to_string(&live_path).unwrap(), live_yaml);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn apply_live_uses_live_yaml_override_for_read_and_write_boundaries() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let config_dir_live = write_live_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let override_live = dir.path().join("runtime").join("live-override.yaml");
        fs::create_dir_all(override_live.parent().unwrap()).unwrap();
        fs::write(
            &override_live,
            "global:\n  engine:\n    interval: 1h\nsymbols:\n  ETH:\n    trade:\n      leverage: 1.5\n",
        )
        .unwrap();
        let candidate_yaml =
            "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 3.0\n";

        let bootstrap_state =
            admin_test_state_with_live_yaml(dir.path(), Path::new("/bin/true"), &override_live);
        let current = resolve_current_config_state(&bootstrap_state, "live").unwrap();
        let candidate = validate_candidate_config_write(
            &bootstrap_state,
            "live",
            &override_live,
            candidate_yaml,
        )
        .unwrap();
        let success_report = fake_live_service_apply_report(&candidate.config_id, "running", true);
        let (runtime_bin, count_path, first_path, second_path) =
            write_fake_runtime_script(dir.path(), &success_report, &success_report);
        let _env = EnvGuard::set(&[
            (
                "AIQ_TEST_RUNTIME_COUNT_FILE",
                Some(count_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_1",
                Some(first_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_2",
                Some(second_path.to_str().unwrap()),
            ),
        ]);
        let state = admin_test_state_with_live_yaml(dir.path(), &runtime_bin, &override_live);

        let Json(response) = post_apply_live(
            State(Arc::clone(&state)),
            if_match_headers(&current.lock_id),
            Json(ApplyLiveBody {
                yaml: candidate_yaml.to_string(),
                expected_config_id: None,
                reason: Some("test override".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(true));
        assert_eq!(
            response["previous_lock_id"],
            Value::String(current.lock_id.clone())
        );
        assert_eq!(
            fs::read_to_string(&override_live).unwrap(),
            normalise_yaml_payload(candidate_yaml)
        );
        assert!(fs::read_to_string(&config_dir_live)
            .unwrap()
            .contains("interval: 30m"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn apply_live_restores_previous_config_when_runtime_proof_fails() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let live_yaml =
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n";
        let live_path = write_live_config(dir.path(), live_yaml);
        let candidate_yaml =
            "global:\n  engine:\n    interval: 5m\nsymbols:\n  ETH:\n    trade:\n      leverage: 4.0\n";

        let baseline_state = test_state(dir.path());
        let current = resolve_current_config_state(&baseline_state, "live").unwrap();
        let incumbent =
            validate_candidate_config_write(&baseline_state, "live", &live_path, live_yaml)
                .unwrap();
        let candidate =
            validate_candidate_config_write(&baseline_state, "live", &live_path, candidate_yaml)
                .unwrap();
        let failed_report = fake_live_service_apply_report("wrong-config-id", "running", true);
        let recovery_report = fake_live_service_apply_report(&incumbent.config_id, "running", true);
        let (runtime_bin, count_path, first_path, second_path) =
            write_fake_runtime_script(dir.path(), &failed_report, &recovery_report);
        let _env = EnvGuard::set(&[
            (
                "AIQ_TEST_RUNTIME_COUNT_FILE",
                Some(count_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_1",
                Some(first_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_2",
                Some(second_path.to_str().unwrap()),
            ),
        ]);
        let state = admin_test_state(dir.path(), &runtime_bin);

        let Json(response) = post_apply_live(
            State(Arc::clone(&state)),
            if_match_headers(&current.lock_id),
            Json(ApplyLiveBody {
                yaml: candidate_yaml.to_string(),
                expected_config_id: None,
                reason: Some("test failure".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(false));
        assert!(response["error"]
            .as_str()
            .unwrap()
            .contains("did not match expected"));
        assert_eq!(fs::read_to_string(&live_path).unwrap(), live_yaml);
        assert!(response["restart"]["recovery"].is_object());
        assert_eq!(response["config_id"], Value::String(candidate.config_id),);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn rollback_live_records_previous_and_restored_config_ids() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let live_yaml =
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n";
        let restored_yaml =
            "global:\n  engine:\n    interval: 1h\nsymbols:\n  ETH:\n    trade:\n      leverage: 1.0\n";
        let live_path = write_live_config(dir.path(), live_yaml);
        let deploy_dir = dir
            .path()
            .join("artifacts")
            .join("deployments")
            .join("live")
            .join("20260314T120000Z");
        fs::create_dir_all(&deploy_dir).unwrap();
        fs::write(deploy_dir.join("prev_config.yaml"), restored_yaml).unwrap();

        let baseline_state = test_state(dir.path());
        let incumbent =
            validate_candidate_config_write(&baseline_state, "live", &live_path, live_yaml)
                .unwrap();
        let restored =
            validate_candidate_config_write(&baseline_state, "live", &live_path, restored_yaml)
                .unwrap();
        let success_report = fake_live_service_apply_report(&restored.config_id, "running", true);
        let (runtime_bin, count_path, first_path, second_path) =
            write_fake_runtime_script(dir.path(), &success_report, &success_report);
        let _env = EnvGuard::set(&[
            (
                "AIQ_TEST_RUNTIME_COUNT_FILE",
                Some(count_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_1",
                Some(first_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_2",
                Some(second_path.to_str().unwrap()),
            ),
        ]);
        let state = admin_test_state(dir.path(), &runtime_bin);

        let Json(response) = post_rollback_live(
            State(Arc::clone(&state)),
            HeaderMap::new(),
            Json(RollbackLiveBody {
                steps: Some(1),
                reason: Some("test rollback".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], Value::Bool(true));
        assert_eq!(
            response["previous_config_id"],
            Value::String(incumbent.config_id.clone())
        );
        assert_eq!(
            response["restored_config_id"],
            Value::String(restored.config_id.clone())
        );
        assert_eq!(
            fs::read_to_string(&live_path).unwrap(),
            normalise_yaml_payload(restored_yaml)
        );
        assert_eq!(response["artifact_path_redacted"], Value::Bool(true));

        let events = read_recent_events(&state.config.artifacts_dir, Some("live"), 10).unwrap();
        let latest = events
            .iter()
            .find(|event| event.action == "rollback_live")
            .unwrap();
        assert_eq!(
            latest.before.config_id.as_deref(),
            Some(incumbent.config_id.as_str())
        );
        assert_eq!(
            latest.after.config_id.as_deref(),
            Some(restored.config_id.as_str())
        );
    }

    #[tokio::test]
    async fn apply_live_request_can_be_approved_and_executes_once() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let live_path = write_live_config(
            dir.path(),
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n",
        );
        let candidate_yaml =
            "global:\n  engine:\n    interval: 15m\nsymbols:\n  ETH:\n    trade:\n      leverage: 3.0\n";
        let baseline_state = test_state(dir.path());
        let current = resolve_current_config_state(&baseline_state, "live").unwrap();
        let candidate =
            validate_candidate_config_write(&baseline_state, "live", &live_path, candidate_yaml)
                .unwrap();
        let success_report = fake_live_service_apply_report(&candidate.config_id, "running", true);
        let (runtime_bin, count_path, first_path, second_path) =
            write_fake_runtime_script(dir.path(), &success_report, &success_report);
        let _env = EnvGuard::set(&[
            (
                "AIQ_TEST_RUNTIME_COUNT_FILE",
                Some(count_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_1",
                Some(first_path.to_str().unwrap()),
            ),
            (
                "AIQ_TEST_RUNTIME_RESPONSE_2",
                Some(second_path.to_str().unwrap()),
            ),
        ]);
        let state = admin_test_state(dir.path(), &runtime_bin);
        let mut request_headers = if_match_headers(&current.lock_id);
        request_headers.insert("x-aiq-actor", HeaderValue::from_static("editor-jt"));

        let Json(request_response) = post_request_apply_live(
            State(Arc::clone(&state)),
            request_headers,
            Json(ApplyLiveBody {
                yaml: candidate_yaml.to_string(),
                expected_config_id: None,
                reason: Some("maker request".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        let request_id = request_response["request_id"].as_str().unwrap().to_string();
        let Json(pending) = get_config_approvals(
            State(Arc::clone(&state)),
            Query(ConfigApprovalListQuery {
                status: Some("pending".to_string()),
            }),
        )
        .await
        .unwrap();
        assert_eq!(pending["requests"].as_array().unwrap().len(), 1);

        let mut approver_headers = HeaderMap::new();
        approver_headers.insert("x-aiq-actor", HeaderValue::from_static("approver-jm"));
        let Json(approved) = post_approve_config_approval(
            State(Arc::clone(&state)),
            AxumPath(request_id.clone()),
            approver_headers,
            Json(ConfigApprovalDecisionBody { reason: None }),
        )
        .await
        .unwrap();

        assert_eq!(approved["ok"], Value::Bool(true));
        assert_eq!(approved["status"], Value::String("approved".to_string()));
        assert!(fs::read_to_string(&live_path)
            .unwrap()
            .contains("interval: 15m"));

        let Json(pending_after) = get_config_approvals(
            State(Arc::clone(&state)),
            Query(ConfigApprovalListQuery {
                status: Some("pending".to_string()),
            }),
        )
        .await
        .unwrap();
        assert_eq!(pending_after["requests"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn rollback_request_can_be_rejected_without_mutating_live_yaml() {
        let dir = tempdir().unwrap();
        let live_yaml =
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  ETH:\n    trade:\n      leverage: 2.0\n";
        let live_path = write_live_config(dir.path(), live_yaml);
        let deploy_dir = dir
            .path()
            .join("artifacts")
            .join("deployments")
            .join("live")
            .join("20260314T120000Z");
        fs::create_dir_all(&deploy_dir).unwrap();
        fs::write(
            deploy_dir.join("prev_config.yaml"),
            "global:\n  engine:\n    interval: 1h\nsymbols:\n  ETH:\n    trade:\n      leverage: 1.0\n",
        )
        .unwrap();
        let state = admin_test_state(dir.path(), Path::new("/bin/true"));
        let mut request_headers = HeaderMap::new();
        request_headers.insert("x-aiq-actor", HeaderValue::from_static("editor-jt"));

        let Json(request_response) = post_request_rollback_live(
            State(Arc::clone(&state)),
            request_headers,
            Json(RollbackLiveBody {
                steps: Some(1),
                reason: Some("need rollback".to_string()),
                restart: Some("auto".to_string()),
                dry_run: Some(false),
            }),
        )
        .await
        .unwrap();

        let request_id = request_response["request_id"].as_str().unwrap().to_string();
        let mut approver_headers = HeaderMap::new();
        approver_headers.insert("x-aiq-actor", HeaderValue::from_static("approver-jm"));
        let Json(rejected) = post_reject_config_approval(
            State(Arc::clone(&state)),
            AxumPath(request_id),
            approver_headers,
            Json(ConfigApprovalDecisionBody {
                reason: Some("hold".to_string()),
            }),
        )
        .await
        .unwrap();

        assert_eq!(rejected["ok"], Value::Bool(true));
        assert_eq!(rejected["status"], Value::String("rejected".to_string()));
        assert_eq!(fs::read_to_string(&live_path).unwrap(), live_yaml);
    }
}
