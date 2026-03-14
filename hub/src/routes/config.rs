use axum::{
    extract::{Query, State},
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

/// Body for PUT /api/config (raw YAML string).
#[derive(Deserialize)]
pub struct ConfigWriteBody {
    pub yaml: String,
    pub expected_config_id: Option<String>,
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
        .route("/api/config/diff", get(get_config_diff))
        .route("/api/config/files", get(get_config_files));
    let mutation_routes = Router::new()
        .route("/api/config", put(put_config))
        .route("/api/config/reload", post(post_config_reload))
        .route("/api/config/actions/apply-live", post(post_apply_live))
        .route("/api/config/actions/promote-live", post(post_promote_live))
        .route(
            "/api/config/actions/rollback-live",
            post(post_rollback_live),
        )
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));

    read_routes.merge(mutation_routes)
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

/// POST /api/config/actions/apply-live — Transactionally apply a live YAML change and prove runtime health.
async fn post_apply_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ApplyLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;
    let _: Value = serde_yaml::from_str(&body.yaml)
        .map_err(|err| HubError::BadRequest(format!("invalid YAML: {err}")))?;

    let dry_run = body.dry_run.unwrap_or(false);
    let restart_mode = LiveApplyRestartMode::parse(body.restart.as_deref())?;
    let runtime_action = restart_mode.runtime_action(dry_run)?;
    let reason = body.reason.unwrap_or_default().trim().to_string();

    let current = resolve_current_config_state(&state, "live")?;
    let expected_config_id =
        extract_expected_config_id_value(&headers, body.expected_config_id.as_deref())?;
    if expected_config_id != current.lock_id {
        return Err(HubError::Conflict(format!(
            "stale config write for live: expected {expected_config_id} but current config is {}",
            current.lock_id
        )));
    }

    let incumbent = validate_candidate_config_write(
        &state,
        "live",
        &state.config.live_yaml_path,
        &current.raw_text,
    )?;
    let candidate =
        validate_candidate_config_write(&state, "live", &state.config.live_yaml_path, &body.yaml)?;
    let preview_restart_required = candidate.config_id != incumbent.config_id;
    let apply_dir = build_action_dir(&state.config.artifacts_dir.join("applies").join("live"))?;
    let transaction = execute_live_config_transaction(
        &state,
        LiveConfigTransactionInput {
            action_dir: &apply_dir,
            event_version: "apply_event_v1",
            action_name: "apply_live",
            current: &current,
            current_config_id: &incumbent.config_id,
            candidate_label: "candidate_config.yaml",
            candidate: &candidate,
            reason: &reason,
            runtime_action,
            dry_run,
            extra_event_details: json!({
                "restart_mode": restart_mode.as_str(),
            }),
        },
    )
    .await?;

    Ok(Json(json!({
        "ok": transaction.ok,
        "action": "apply_live",
        "apply_dir": transaction.action_dir,
        "backup": transaction.backup,
        "lock_id": candidate.lock_id,
        "config_id": candidate.config_id,
        "previous_lock_id": current.lock_id,
        "previous_config_id": incumbent.config_id,
        "restart_required": if dry_run { preview_restart_required } else { transaction.restart_required },
        "service": state.config.live_service,
        "restart": {
            "mode": restart_mode.as_str(),
            "result": transaction.runtime_result,
            "recovery": transaction.restore_result,
        },
        "dry_run": dry_run,
        "error": transaction.error,
    })))
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
    State(state): State<Arc<AppState>>,
    Query(q): Query<DiffQuery>,
) -> Result<Json<Value>, HubError> {
    let file_variant = q.file.as_deref().unwrap_or("main");
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

/// POST /api/config/actions/rollback-live — Roll back live config to previous deployment version.
async fn post_rollback_live(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RollbackLiveBody>,
) -> Result<Json<Value>, HubError> {
    ensure_admin_actions_enabled(&state)?;

    let steps = body.steps.unwrap_or(1).max(1) as usize;
    let dry_run = body.dry_run.unwrap_or(false);
    let restart_mode = LiveApplyRestartMode::parse(body.restart.as_deref())?;
    let runtime_action = restart_mode.runtime_action(dry_run)?;
    let reason = body.reason.unwrap_or_default().trim().to_string();

    let deploy_root = state.config.artifacts_dir.join("deployments").join("live");
    let deploy_dirs = sorted_subdirs_desc(&deploy_root)?;
    if deploy_dirs.len() < steps {
        return Ok(Json(json!({
            "ok": false,
            "action": "rollback_live",
            "error": "insufficient_live_deployments",
            "steps": steps,
            "available": deploy_dirs.len(),
            "deploy_root": deploy_root.to_string_lossy().to_string(),
        })));
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
        return Ok(Json(json!({
            "ok": false,
            "action": "rollback_live",
            "error": "missing_rollback_config",
            "source_deploy_dir": src_dir.to_string_lossy().to_string(),
        })));
    }

    if let Err(err) = serde_yaml::from_str::<serde_yaml::Value>(&restored_text) {
        return Ok(Json(json!({
            "ok": false,
            "action": "rollback_live",
            "error": "rollback_config_invalid_yaml",
            "detail": err.to_string(),
            "restored_from": restored_from.to_string_lossy().to_string(),
        })));
    }

    let current = resolve_current_config_state(&state, "live")?;
    let incumbent = validate_candidate_config_write(
        &state,
        "live",
        &state.config.live_yaml_path,
        &current.raw_text,
    )?;
    let restored = validate_candidate_config_write(
        &state,
        "live",
        &state.config.live_yaml_path,
        &restored_text,
    )?;
    let rollback_dir =
        build_action_dir(&state.config.artifacts_dir.join("rollbacks").join("live"))?;
    let transaction = execute_live_config_transaction(
        &state,
        LiveConfigTransactionInput {
            action_dir: &rollback_dir,
            event_version: "rollback_event_v2",
            action_name: "rollback_live",
            current: &current,
            current_config_id: &incumbent.config_id,
            candidate_label: "restored_config.yaml",
            candidate: &restored,
            reason: &reason,
            runtime_action,
            dry_run,
            extra_event_details: json!({
                "restart_mode": restart_mode.as_str(),
                "steps": steps,
                "source_deploy_dir": src_dir.to_string_lossy().to_string(),
                "restored_from": restored_from.to_string_lossy().to_string(),
            }),
        },
    )
    .await?;

    Ok(Json(json!({
        "ok": transaction.ok,
        "action": "rollback_live",
        "rollback_dir": transaction.action_dir,
        "backup": transaction.backup,
        "source_deploy_dir": src_dir.to_string_lossy().to_string(),
        "restored_from": restored_from.to_string_lossy().to_string(),
        "steps": steps,
        "previous_lock_id": current.lock_id,
        "previous_config_id": incumbent.config_id,
        "restored_lock_id": restored.lock_id,
        "restored_config_id": restored.config_id,
        "restart_required": transaction.restart_required,
        "restart": {
            "mode": restart_mode.as_str(),
            "service": state.config.live_service,
            "result": transaction.runtime_result,
            "recovery": transaction.restore_result,
        },
        "dry_run": dry_run,
        "error": transaction.error,
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
        assert_eq!(
            response["config_id"],
            Value::String(candidate.config_id.clone())
        );
        assert_eq!(
            fs::read_to_string(&live_path).unwrap(),
            normalise_yaml_payload(candidate_yaml)
        );

        let apply_dir = PathBuf::from(response["apply_dir"].as_str().unwrap());
        let event: Value =
            serde_json::from_str(&fs::read_to_string(apply_dir.join("event.json")).unwrap())
                .unwrap();
        assert_eq!(
            event["what"]["current_config_id"],
            Value::String(
                validate_candidate_config_write(
                    &baseline_state,
                    "live",
                    &live_path,
                    &current.raw_text
                )
                .unwrap()
                .config_id,
            )
        );
        assert_eq!(
            event["what"]["candidate_config_id"],
            Value::String(candidate.config_id)
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

        let Json(response) = post_apply_live(
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

        let rollback_dir = PathBuf::from(response["rollback_dir"].as_str().unwrap());
        let event: Value =
            serde_json::from_str(&fs::read_to_string(rollback_dir.join("event.json")).unwrap())
                .unwrap();
        assert_eq!(
            event["what"]["current_config_id"],
            Value::String(incumbent.config_id)
        );
        assert_eq!(
            event["what"]["candidate_config_id"],
            Value::String(restored.config_id)
        );
    }
}
