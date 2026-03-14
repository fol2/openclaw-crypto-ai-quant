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
    config_id: String,
    config_id_source: &'static str,
}

struct ValidatedConfigWrite {
    payload: String,
    config_id: String,
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
    let path = resolve_config_path(&state.config.config_dir, file_variant)?;
    if !path.exists() {
        return Err(HubError::NotFound(format!(
            "config file not found: {}",
            path.display()
        )));
    }
    let raw_text = fs::read_to_string(&path)?;
    let (config_id, config_id_source) =
        match resolve_effective_config_for_variant(state, file_variant, &path) {
            Ok(effective_config) => (effective_config.config_id().to_string(), "runtime"),
            Err(_) => (raw_config_id(&raw_text), "raw_sha256"),
        };

    Ok(ResolvedConfigState {
        path,
        raw_text,
        config_id,
        config_id_source,
    })
}

fn validate_candidate_config_write(
    state: &AppState,
    file_variant: &str,
    target_path: &Path,
    yaml_text: &str,
) -> Result<ValidatedConfigWrite, HubError> {
    let payload = normalise_yaml_payload(yaml_text);
    let staged = stage_config_payload(target_path, &payload)?;
    let effective_config = resolve_effective_config_for_variant(state, file_variant, &staged.path)?;

    Ok(ValidatedConfigWrite {
        payload,
        config_id: effective_config.config_id().to_string(),
    })
}

fn extract_expected_config_id(
    headers: &HeaderMap,
    body: &ConfigWriteBody,
) -> Result<String, HubError> {
    body.expected_config_id
        .as_deref()
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

fn apply_config_identity_headers(
    headers: &mut HeaderMap,
    config_id: &str,
    config_id_source: &str,
) -> Result<(), HubError> {
    let etag = HeaderValue::from_str(&format!("\"{config_id}\""))
        .map_err(|err| HubError::Internal(format!("invalid ETag header: {err}")))?;
    let config_id_header = HeaderValue::from_str(config_id)
        .map_err(|err| HubError::Internal(format!("invalid config-id header: {err}")))?;
    let config_id_source_header = HeaderValue::from_str(config_id_source)
        .map_err(|err| HubError::Internal(format!("invalid config-id source header: {err}")))?;
    headers.insert(header::ETAG, etag);
    headers.insert("x-aiq-config-id", config_id_header);
    headers.insert("x-aiq-config-id-source", config_id_source_header);
    Ok(())
}

fn load_yaml_engine_interval(yaml_text: &str) -> String {
    let parsed: serde_yaml::Value = match serde_yaml::from_str(yaml_text) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    parsed
        .get("global")
        .and_then(|v| v.get("engine"))
        .and_then(|v| v.get("interval"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn compact_utc_now() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

fn iso_utc_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn service_unit_name(service: &str) -> String {
    let s = service.trim();
    if s.ends_with(".service") {
        s.to_string()
    } else {
        format!("{s}.service")
    }
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
    apply_config_identity_headers(
        response.headers_mut(),
        &resolved.config_id,
        resolved.config_id_source,
    )?;
    Ok(response)
}

/// GET /api/config/raw — Read YAML file, return raw text.
async fn get_config_raw(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Response, HubError> {
    let resolved = resolve_current_config_state(&state, variant(&q))?;
    let mut response = resolved.raw_text.into_response();
    apply_config_identity_headers(
        response.headers_mut(),
        &resolved.config_id,
        resolved.config_id_source,
    )?;
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
    if expected_config_id != current.config_id {
        return Err(HubError::Conflict(format!(
            "stale config write for {file_variant}: expected {expected_config_id} but current config is {}",
            current.config_id
        )));
    }
    let validated =
        validate_candidate_config_write(&state, file_variant, &current.path, &body.yaml)?;

    // Backup current file (if it exists).
    let bk_dir = backups_dir(&state.config.config_dir)?;
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let stem = current
        .path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("config");
    let bk_name = format!("{stem}.{ts}.bak");
    let bk_path = bk_dir.join(&bk_name);
    fs::write(&bk_path, &current.raw_text)?;
    let backup_path_str = bk_name;

    atomic_write_text(&current.path, &validated.payload)?;

    Ok(Json(serde_json::json!({
        "ok": true,
        "file": file_variant,
        "backup": backup_path_str,
        "config_id": validated.config_id,
    })))
}

/// POST /api/config/reload — Touch the config file mtime to trigger hot-reload.
async fn post_config_reload(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Json<Value>, HubError> {
    let path = resolve_config_path(&state.config.config_dir, variant(&q))?;
    if !path.exists() {
        return Err(HubError::NotFound(format!(
            "config file not found: {}",
            path.display()
        )));
    }

    // Touch mtime by opening and syncing.
    let file = fs::OpenOptions::new().write(true).open(&path)?;
    file.sync_all()?;

    drop(file);

    // Set mtime via libc utimensat for reliability.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|e| HubError::Internal(e.to_string()))?;
        unsafe {
            libc::utimensat(
                libc::AT_FDCWD,
                c_path.as_ptr(),
                std::ptr::null(), // null = set to current time
                0,
            );
        }
    }

    Ok(Json(
        serde_json::json!({ "ok": true, "reloaded": variant(&q) }),
    ))
}

/// GET /api/config/history — List backup files for a config variant.
async fn get_config_history(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ConfigQuery>,
) -> Result<Json<Vec<BackupEntry>>, HubError> {
    let file_variant = variant(&q);
    let path = resolve_config_path(&state.config.config_dir, file_variant)?;
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
    let config_path = resolve_config_path(&state.config.config_dir, file_variant)?;
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
        if let Ok(path) = resolve_config_path(&state.config.config_dir, v) {
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
    let restart_mode = body
        .restart
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_lowercase();
    if !matches!(restart_mode.as_str(), "auto" | "always" | "never") {
        return Err(HubError::BadRequest(
            "invalid restart mode (expected auto/always/never)".to_string(),
        ));
    }
    let reason = body.reason.unwrap_or_default().trim().to_string();
    let dry_run = body.dry_run.unwrap_or(false);

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

    if let Err(e) = serde_yaml::from_str::<serde_yaml::Value>(&restored_text) {
        return Ok(Json(json!({
            "ok": false,
            "action": "rollback_live",
            "error": "rollback_config_invalid_yaml",
            "detail": e.to_string(),
            "restored_from": restored_from.to_string_lossy().to_string(),
        })));
    }

    let current_text = fs::read_to_string(&state.config.live_yaml_path).unwrap_or_default();
    let current_interval = load_yaml_engine_interval(&current_text);
    let restored_interval = load_yaml_engine_interval(&restored_text);
    let restart_required = !current_interval.is_empty()
        && !restored_interval.is_empty()
        && current_interval != restored_interval;

    let ts_compact = compact_utc_now();
    let rollback_root = state.config.artifacts_dir.join("rollbacks").join("live");
    fs::create_dir_all(&rollback_root)?;
    let mut rollback_dir = rollback_root.join(&ts_compact);
    if rollback_dir.exists() {
        rollback_dir = rollback_root.join(format!("{ts_compact}-{}", Uuid::new_v4().simple()));
    }
    fs::create_dir_all(&rollback_dir)?;

    let restored_payload = format!("{}\n", restored_text.trim_end());
    atomic_write_text(
        &rollback_dir.join("restored_config.yaml"),
        &restored_payload,
    )?;
    if !dry_run {
        atomic_write_text(&state.config.live_yaml_path, &restored_payload)?;
    }

    let do_restart = restart_mode == "always" || (restart_mode == "auto" && restart_required);
    let mut restart_result: Value = Value::Null;
    if do_restart && !dry_run {
        let unit = service_unit_name(&state.config.live_service);
        restart_result = run_action_command(
            vec![
                "systemctl".to_string(),
                "--user".to_string(),
                "restart".to_string(),
                unit,
            ],
            &state.config.aiq_root,
            state.config.admin_action_timeout_s,
        )
        .await;
    }

    let event = json!({
        "version": "rollback_event_v1",
        "ts_utc": iso_utc_now(),
        "ts_compact_utc": ts_compact,
        "who": {
            "user": env::var("USER").unwrap_or_default(),
            "hostname": env::var("HOSTNAME").unwrap_or_default(),
        },
        "what": {
            "mode": "live",
            "yaml_path": state.config.live_yaml_path.to_string_lossy().to_string(),
            "source_deploy_dir": src_dir.to_string_lossy().to_string(),
            "restored_from": restored_from.to_string_lossy().to_string(),
            "current_engine_interval": current_interval,
            "restored_engine_interval": restored_interval,
            "restart_required": restart_required,
            "steps": steps,
        },
        "why": { "reason": reason },
        "dry_run": dry_run,
        "restart": {
            "mode": restart_mode,
            "service": state.config.live_service,
            "result": restart_result,
        }
    });

    let event_text = serde_json::to_string_pretty(&event)? + "\n";
    atomic_write_text(&rollback_dir.join("rollback_event.json"), &event_text)?;

    let restart_failed = event
        .get("restart")
        .and_then(|v| v.get("result"))
        .and_then(|v| v.get("ok"))
        .and_then(|v| v.as_bool())
        .map(|ok| !ok)
        .unwrap_or(false);

    if restart_failed {
        return Ok(Json(json!({
            "ok": false,
            "action": "rollback_live",
            "error": "service_restart_failed",
            "rollback_dir": rollback_dir.to_string_lossy().to_string(),
            "source_deploy_dir": src_dir.to_string_lossy().to_string(),
            "restored_from": restored_from.to_string_lossy().to_string(),
            "restart_required": restart_required,
            "restart": event.get("restart"),
        })));
    }

    Ok(Json(json!({
        "ok": true,
        "action": "rollback_live",
        "rollback_dir": rollback_dir.to_string_lossy().to_string(),
        "source_deploy_dir": src_dir.to_string_lossy().to_string(),
        "restored_from": restored_from.to_string_lossy().to_string(),
        "steps": steps,
        "restart_required": restart_required,
        "restart": event.get("restart"),
        "dry_run": dry_run,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use tempfile::tempdir;

    fn write_main_config(root: &Path, yaml: &str) -> PathBuf {
        let config_dir = root.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("strategy_overrides.yaml");
        fs::write(&config_path, yaml).unwrap();
        config_path
    }

    fn test_state(root: &Path) -> Arc<AppState> {
        let mut config = crate::config::HubConfig::from_env();
        config.aiq_root = root.to_path_buf();
        config.config_dir = root.join("config");
        config.live_yaml_path = config.config_dir.join("strategy_overrides.live.yaml");
        AppState::new(config)
    }

    fn if_match_headers(config_id: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::IF_MATCH, HeaderValue::from_str(config_id).unwrap());
        headers
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

        assert!(response.headers().contains_key("x-aiq-config-id"));
        assert_eq!(
            response.headers()["x-aiq-config-id-source"],
            HeaderValue::from_static("runtime")
        );
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
            if_match_headers(&current.config_id),
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
            if_match_headers(&current.config_id),
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
}
