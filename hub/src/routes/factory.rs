use axum::{
    extract::{Path, State},
    middleware,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;
use tokio::process::Command;

use crate::error::HubError;
use crate::factory_capability::{
    disabled_response, FactoryCapability, FACTORY_SERVICE_UNITS, FACTORY_SETTINGS_PATH,
};
use crate::state::AppState;
use crate::subprocess::factory as factory_subprocess;
use crate::subprocess::JobStatus;

/// Build factory sub-router.
pub fn routes() -> Router<Arc<AppState>> {
    let read_routes = Router::new()
        .route("/api/factory/capability", get(get_capability))
        // Existing read-only artifact routes
        .route("/api/factory/runs", get(list_runs))
        .route("/api/factory/runs/{date}/{run_id}", get(run_detail))
        .route("/api/factory/runs/{date}/{run_id}/report", get(run_report))
        .route(
            "/api/factory/runs/{date}/{run_id}/candidates",
            get(run_candidates),
        )
        // Job control routes
        .route("/api/factory/jobs", get(list_jobs))
        .route("/api/factory/jobs/{id}/status", get(job_status))
        // Settings routes
        .route("/api/factory/settings", get(get_settings))
        // Timer routes
        .route("/api/factory/timer", get(get_timer));
    let mutation_routes = Router::new()
        .route("/api/factory/run", post(run_factory))
        .route("/api/factory/jobs/{id}", delete(cancel_job))
        .route("/api/factory/settings", axum::routing::put(put_settings))
        .route("/api/factory/timer/{action}", post(timer_action))
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));

    read_routes.merge(mutation_routes)
}

fn capability(state: &AppState) -> FactoryCapability {
    FactoryCapability::current(&state.config)
}

/// GET /api/factory/capability — current dormant/active contract state.
async fn get_capability(State(state): State<Arc<AppState>>) -> Json<FactoryCapability> {
    Json(capability(&state))
}

// ── Artifact routes (existing) ─────────────────────────────────────

/// Resolve the artifacts directory.
fn artifacts_dir(state: &AppState) -> PathBuf {
    state.config.aiq_root.join("artifacts")
}

/// Validate a date string is YYYY-MM-DD (path-traversal protection).
fn validate_date(date: &str) -> Result<(), HubError> {
    if date.len() == 10
        && date.chars().all(|c| c.is_ascii_digit() || c == '-')
        && date.chars().filter(|c| *c == '-').count() == 2
    {
        Ok(())
    } else {
        Err(HubError::BadRequest(format!("invalid date: {date}")))
    }
}

/// Validate a run_id (alphanumeric + underscores + dashes + T/Z).
fn validate_run_id(id: &str) -> Result<(), HubError> {
    if !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == 'T' || c == 'Z')
    {
        Ok(())
    } else {
        Err(HubError::BadRequest(format!("invalid run_id: {id}")))
    }
}

fn canonical_run_id(raw: &str) -> Result<String, HubError> {
    let run_id = raw.strip_prefix("run_").unwrap_or(raw);
    validate_run_id(run_id)?;
    Ok(run_id.to_string())
}

fn resolve_run_dir(
    state: &AppState,
    date: &str,
    raw_run_id: &str,
) -> Result<(String, PathBuf), HubError> {
    validate_date(date)?;
    let run_id = canonical_run_id(raw_run_id)?;
    let date_dir = artifacts_dir(state).join(date);

    for candidate in [format!("run_{run_id}"), run_id.clone()] {
        let run_dir = date_dir.join(&candidate);
        if run_dir.is_dir() {
            return Ok((run_id, run_dir));
        }
    }

    Err(HubError::NotFound(format!(
        "run not found: {date}/{run_id}"
    )))
}

fn load_optional_json(path: &StdPath) -> Result<Option<Value>, HubError> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str::<Value>(&raw)?))
}

fn load_optional_json_lossy(path: &StdPath) -> Option<Value> {
    if !path.exists() {
        return None;
    }

    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
}

fn candidate_count_from_sources(report: Option<&Value>, metadata: &Value) -> Option<usize> {
    report
        .and_then(|value| value["candidate_count"].as_u64())
        .map(|value| value as usize)
        .or_else(|| metadata["items"].as_array().map(|items| items.len()))
}

fn run_directory_preference(dir_name: &str, canonical_run_id: &str) -> u8 {
    let prefixed = format!("run_{canonical_run_id}");
    if dir_name == prefixed {
        0
    } else if dir_name == canonical_run_id {
        1
    } else {
        2
    }
}

fn should_replace_run_summary(existing: &Value, candidate: &Value) -> bool {
    let run_id = candidate["run_id"].as_str().unwrap_or("");
    let existing_dir = existing["directory_name"].as_str().unwrap_or("");
    let candidate_dir = candidate["directory_name"].as_str().unwrap_or("");

    let existing_pref = run_directory_preference(existing_dir, run_id);
    let candidate_pref = run_directory_preference(candidate_dir, run_id);
    if candidate_pref != existing_pref {
        return candidate_pref < existing_pref;
    }

    let existing_generated_at = existing["generated_at_ms"].as_i64().unwrap_or(i64::MIN);
    let candidate_generated_at = candidate["generated_at_ms"].as_i64().unwrap_or(i64::MIN);
    candidate_generated_at > existing_generated_at
        || (candidate_generated_at == existing_generated_at && candidate_dir > existing_dir)
}

fn selection_summary(selection: &Value) -> Value {
    let role_candidates_by_role = selection["role_candidates_by_role"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let selected_candidates_by_role = selection["selected_candidates_by_role"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let selected_targets = selection["selected_targets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let challenges = selection["challenges"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let deployments = selection["deployments"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    json!({
        "selection_stage": selection["selection_stage"],
        "deploy_stage": selection["deploy_stage"],
        "promotion_stage": selection["promotion_stage"],
        "step5_gate_status": selection["step5_gate_status"],
        "deployed": selection["deployed"],
        "role_candidate_count": role_candidates_by_role.len(),
        "role_candidates_by_role": role_candidates_by_role,
        "selected_count": selected_candidates_by_role.len(),
        "selected_candidates_by_role": selected_candidates_by_role,
        "selected_targets": selected_targets,
        "challenge_count": challenges.len(),
        "deployment_count": deployments.len(),
        "live_promotion": selection.get("live_promotion").cloned().unwrap_or(Value::Null),
    })
}

fn candidate_rows_from_sources(
    run_dir: &StdPath,
    report: Option<&Value>,
    metadata: &Value,
) -> Result<Vec<Value>, HubError> {
    if let Some(candidates) = report.and_then(|value| value["candidates"].as_array()) {
        return Ok(candidates.clone());
    }

    if let Some(items) = metadata["items"].as_array() {
        return Ok(items.clone());
    }

    let configs_dir = run_dir.join("configs");
    if !configs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut candidates = Vec::new();
    for entry in fs::read_dir(&configs_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".yaml") || name.ends_with(".yml") {
            let size = entry.metadata()?.len();
            candidates.push(json!({
                "filename": name,
                "size": size,
            }));
        }
    }
    candidates.sort_by(|a, b| {
        a["filename"]
            .as_str()
            .unwrap_or("")
            .cmp(b["filename"].as_str().unwrap_or(""))
    });

    Ok(candidates)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct TimerStatus {
    active: String,
    next_trigger: String,
    load: String,
    unit_file_state: String,
}

fn parse_timer_status(output: &str) -> TimerStatus {
    let mut status = TimerStatus::default();

    for line in output.lines() {
        if let Some(value) = line.strip_prefix("ActiveState=") {
            status.active = value.to_string();
        } else if let Some(value) = line.strip_prefix("NextElapseUSecRealtime=") {
            status.next_trigger = value.to_string();
        } else if let Some(value) = line.strip_prefix("LoadState=") {
            status.load = value.to_string();
        } else if let Some(value) = line.strip_prefix("UnitFileState=") {
            status.unit_file_state = value.to_string();
        }
    }

    status
}

/// GET /api/factory/runs — List all factory runs across all dates.
async fn list_runs(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Value>>, HubError> {
    let base = artifacts_dir(&state);
    let mut runs = Vec::new();

    if !base.exists() {
        return Ok(Json(runs));
    }

    // Iterate date directories.
    let mut dates: Vec<String> = Vec::new();
    for entry in fs::read_dir(&base)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type()?.is_dir() && validate_date(&name).is_ok() {
            dates.push(name);
        }
    }
    dates.sort();
    dates.reverse(); // newest first

    for date in &dates {
        let date_dir = base.join(date);
        if !date_dir.is_dir() {
            continue;
        }
        let mut date_runs: HashMap<String, Value> = HashMap::new();
        for entry in fs::read_dir(&date_dir)? {
            let entry = entry?;
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let run_id = match canonical_run_id(&dir_name) {
                Ok(run_id) => run_id,
                Err(_) => continue,
            };

            // Read metadata if available.
            let meta_path = entry.path().join("run_metadata.json");
            let metadata = if meta_path.exists() {
                fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            } else {
                None
            };

            let report_path = entry.path().join("reports/report.json");
            let selection_path = entry.path().join("reports/selection.json");
            let report = load_optional_json_lossy(&report_path);
            let selection = load_optional_json_lossy(&selection_path);
            let has_report = report_path.exists();
            let metadata = metadata.unwrap_or_else(|| json!({}));

            let row = json!({
                "date": date,
                "run_id": run_id.clone(),
                "directory_name": dir_name,
                "has_report": has_report,
                "profile": metadata["args"]["profile"]
                    .as_str()
                    .unwrap_or("unknown"),
                "candidate_count": candidate_count_from_sources(report.as_ref(), &metadata),
                "role_candidate_count": selection.as_ref()
                    .and_then(|value| value["role_candidates_by_role"].as_array())
                    .map(|items| items.len()),
                "selected_count": selection.as_ref()
                    .and_then(|value| value["selected_candidates_by_role"].as_array())
                    .map(|items| items.len()),
                "selection_stage": selection.as_ref().and_then(|value| value["selection_stage"].as_str()),
                "deploy_stage": selection.as_ref().and_then(|value| value["deploy_stage"].as_str()),
                "generated_at_ms": metadata["generated_at_ms"].as_i64(),
            });

            match date_runs.get(&run_id) {
                Some(existing) if !should_replace_run_summary(existing, &row) => {}
                _ => {
                    date_runs.insert(run_id, row);
                }
            }
        }
        let mut date_runs: Vec<Value> = date_runs.into_values().collect();
        date_runs.sort_by(|a, b| {
            let ta = a["generated_at_ms"].as_i64();
            let tb = b["generated_at_ms"].as_i64();
            tb.cmp(&ta)
                .then_with(|| {
                    b["run_id"]
                        .as_str()
                        .unwrap_or("")
                        .cmp(a["run_id"].as_str().unwrap_or(""))
                })
                .then_with(|| {
                    b["directory_name"]
                        .as_str()
                        .unwrap_or("")
                        .cmp(a["directory_name"].as_str().unwrap_or(""))
                })
        });
        runs.extend(date_runs);
    }

    Ok(Json(runs))
}

/// GET /api/factory/runs/{date}/{run_id} — Run metadata.
async fn run_detail(
    State(state): State<Arc<AppState>>,
    Path((date, run_id)): Path<(String, String)>,
) -> Result<Json<Value>, HubError> {
    let (run_id, run_dir) = resolve_run_dir(&state, &date, &run_id)?;

    let meta_path = run_dir.join("run_metadata.json");
    let metadata = load_optional_json(&meta_path)?.unwrap_or_else(|| json!({}));
    let report = load_optional_json(&run_dir.join("reports/report.json"))?;
    let selection = load_optional_json(&run_dir.join("reports/selection.json"))?;

    // List subdirectories to show available data.
    let subdirs: Vec<String> = fs::read_dir(&run_dir)?
        .filter_map(|e| {
            let e = e.ok()?;
            if e.file_type().ok()?.is_dir() {
                Some(e.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();

    Ok(Json(json!({
        "date": date,
        "run_id": run_id,
        "directory_name": run_dir.file_name().map(|value| value.to_string_lossy().to_string()),
        "metadata": metadata,
        "subdirs": subdirs,
        "report_available": report.is_some(),
        "candidate_count": candidate_count_from_sources(report.as_ref(), &metadata),
        "selection_summary": selection.as_ref().map(selection_summary),
    })))
}

/// GET /api/factory/runs/{date}/{run_id}/report — Factory report.json content.
async fn run_report(
    State(state): State<Arc<AppState>>,
    Path((date, run_id)): Path<(String, String)>,
) -> Result<Json<Value>, HubError> {
    let (_, run_dir) = resolve_run_dir(&state, &date, &run_id)?;
    let report_path = run_dir.join("reports/report.json");

    if !report_path.exists() {
        return Err(HubError::NotFound("report.json not found".into()));
    }

    let raw = fs::read_to_string(&report_path)?;
    let val: Value = serde_json::from_str(&raw)?;
    Ok(Json(val))
}

/// GET /api/factory/runs/{date}/{run_id}/candidates — List candidate configs.
async fn run_candidates(
    State(state): State<Arc<AppState>>,
    Path((date, run_id)): Path<(String, String)>,
) -> Result<Json<Vec<Value>>, HubError> {
    let (_, run_dir) = resolve_run_dir(&state, &date, &run_id)?;
    let report = load_optional_json(&run_dir.join("reports/report.json"))?;
    let metadata =
        load_optional_json(&run_dir.join("run_metadata.json"))?.unwrap_or_else(|| json!({}));
    let candidates = candidate_rows_from_sources(&run_dir, report.as_ref(), &metadata)?;
    Ok(Json(candidates))
}

// ── Job control routes ─────────────────────────────────────────────

/// Resolve settings file path.
fn settings_path(state: &AppState) -> PathBuf {
    state.config.aiq_root.join(FACTORY_SETTINGS_PATH)
}

/// Load saved factory defaults (returns empty Value if file doesn't exist).
fn load_settings(state: &AppState) -> Value {
    let path = settings_path(state);
    if path.exists() {
        fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_yaml::from_str::<Value>(&s).ok())
            .unwrap_or(json!({}))
    } else {
        json!({})
    }
}

#[derive(Debug, Deserialize)]
struct RunFactoryBody {
    profile: Option<String>,
    config: Option<String>,
    settings_path: Option<String>,
}

fn resolve_requested_path(
    project_root: &StdPath,
    default_path: &StdPath,
    raw: Option<&str>,
) -> PathBuf {
    match raw {
        Some(value) if PathBuf::from(value).is_absolute() => PathBuf::from(value),
        Some(value) => project_root.join(value),
        None => default_path.to_path_buf(),
    }
}

/// POST /api/factory/run — launch a new factory cycle.
async fn run_factory(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RunFactoryBody>,
) -> Response {
    let cap = capability(&state);
    if !cap.execution_enabled {
        return disabled_response(&cap, "run");
    }
    {
        let jobs = state.jobs.jobs.lock().await;
        if jobs
            .values()
            .any(|job| job.kind == "factory" && job.status == JobStatus::Running)
        {
            return HubError::BadRequest(
                "a factory job is already running; wait for it to finish or cancel it first".into(),
            )
            .into_response();
        }
    }
    let profile = body
        .profile
        .unwrap_or_else(|| "daily".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(profile.as_str(), "daily" | "deep") {
        return HubError::BadRequest(format!(
            "invalid factory profile: {profile} (valid: daily, deep)"
        ))
        .into_response();
    }
    let config_path = resolve_requested_path(
        &state.config.aiq_root,
        &state.config.factory_config_path,
        body.config.as_deref(),
    );
    let settings_path = resolve_requested_path(
        &state.config.aiq_root,
        &state.config.aiq_root.join(FACTORY_SETTINGS_PATH),
        body.settings_path.as_deref(),
    );
    let job_id = uuid::Uuid::new_v4().to_string();
    let args = factory_subprocess::FactoryArgs {
        executor_bin: state.config.factory_bin.display().to_string(),
        config_path: config_path.display().to_string(),
        settings_path: settings_path.display().to_string(),
        profile: profile.clone(),
    };
    factory_subprocess::spawn_factory(
        job_id.clone(),
        args,
        state.config.aiq_root.display().to_string(),
        Arc::clone(&state.jobs),
        state.broadcast.clone(),
    )
    .await;
    Json(json!({
        "job_id": job_id,
        "profile": profile,
        "status": "running",
    }))
    .into_response()
}

/// GET /api/factory/jobs — list all factory jobs.
async fn list_jobs(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Value>>, HubError> {
    let jobs = state.jobs.jobs.lock().await;
    let mut result: Vec<Value> = jobs
        .values()
        .filter(|j| j.kind == "factory")
        .map(|j| {
            json!({
                "id": j.id,
                "status": j.status,
                "created_at": j.created_at,
                "finished_at": j.finished_at,
                "error": j.error,
            })
        })
        .collect();
    result.sort_by(|a, b| {
        let ta = a["created_at"].as_str().unwrap_or("");
        let tb = b["created_at"].as_str().unwrap_or("");
        tb.cmp(ta)
    });
    Ok(Json(result))
}

/// GET /api/factory/jobs/{id}/status — job status + stderr tail.
async fn job_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, HubError> {
    let jobs = state.jobs.jobs.lock().await;
    let job = jobs
        .get(&id)
        .ok_or_else(|| HubError::NotFound(format!("job {id} not found")))?;
    Ok(Json(json!({
        "id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "stderr_tail": job.stderr_tail,
        "error": job.error,
    })))
}

/// DELETE /api/factory/jobs/{id} — cancel a running factory job.
async fn cancel_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let cap = capability(&state);
    if !cap.execution_enabled {
        return disabled_response(&cap, "cancel");
    }

    let mut jobs = state.jobs.jobs.lock().await;
    let job = match jobs.get_mut(&id) {
        Some(job) => job,
        None => return HubError::NotFound(format!("job {id} not found")).into_response(),
    };
    if job.status != JobStatus::Running {
        return HubError::BadRequest("job is not running".into()).into_response();
    }
    job.status = JobStatus::Cancelled;
    job.finished_at = Some(chrono::Utc::now().to_rfc3339());

    let mut handles = state.jobs.handles.lock().await;
    if let Some(pid) = handles.remove(&id) {
        let rc = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                return HubError::Internal(format!("failed to signal pid {pid}: {err}"))
                    .into_response();
            }
        }
    }

    Json(json!({ "ok": true, "cancelled": id })).into_response()
}

// ── Settings routes ────────────────────────────────────────────────

/// GET /api/factory/settings — read saved factory defaults.
async fn get_settings(State(state): State<Arc<AppState>>) -> Result<Json<Value>, HubError> {
    Ok(Json(load_settings(&state)))
}

/// PUT /api/factory/settings — save factory defaults.
async fn put_settings(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let cap = capability(&state);
    if !cap.execution_enabled {
        return disabled_response(&cap, "settings.put");
    }

    let path = settings_path(&state);
    let yaml = match serde_yaml::to_string(&body) {
        Ok(yaml) => yaml,
        Err(err) => return HubError::from(err).into_response(),
    };
    if let Err(err) = fs::write(&path, &yaml) {
        return HubError::from(err).into_response();
    }
    Json(json!({ "ok": true })).into_response()
}

// ── Timer routes ───────────────────────────────────────────────────

/// Factory timer unit names.
/// GET /api/factory/timer — timer status.
async fn get_timer(State(state): State<Arc<AppState>>) -> Result<Json<Value>, HubError> {
    let capability = capability(&state);
    let mut timers = Vec::new();
    for timer in FACTORY_SERVICE_UNITS.map(|name| format!("{name}.timer")) {
        let output = Command::new("systemctl")
            .arg("--user")
            .arg("show")
            .arg(&timer)
            .arg("--property=ActiveState,NextElapseUSecRealtime,LoadState,UnitFileState")
            .output()
            .await
            .map_err(|e| HubError::Internal(format!("systemctl failed: {e}")))?;

        let text = String::from_utf8_lossy(&output.stdout);
        let status = parse_timer_status(&text);
        let enabled = matches!(
            status.unit_file_state.as_str(),
            "enabled" | "enabled-runtime"
        );
        timers.push(json!({
            "unit": timer,
            "active": status.active,
            "load": status.load,
            "enabled": enabled,
            "available": status.load != "not-found",
            "unit_file_state": status.unit_file_state,
            "next_trigger": status.next_trigger,
            "mode": capability.mode,
        }));
    }

    Ok(Json(json!({ "capability": capability, "timers": timers })))
}

/// POST /api/factory/timer/{action} — enable or disable factory timer.
async fn timer_action(State(state): State<Arc<AppState>>, Path(action): Path<String>) -> Response {
    let cap = capability(&state);
    if !cap.execution_enabled {
        return disabled_response(&cap, "timer");
    }

    match action.as_str() {
        "enable" | "disable" => {}
        _ => {
            return HubError::BadRequest(format!(
                "invalid action: {action} (valid: enable, disable)"
            ))
            .into_response();
        }
    }

    let mut results = Vec::new();
    for timer in FACTORY_SERVICE_UNITS.map(|name| format!("{name}.timer")) {
        // enable/disable controls auto-start
        let output = Command::new("systemctl")
            .arg("--user")
            .arg(&action)
            .arg(&timer)
            .output()
            .await
            .map_err(|e| HubError::Internal(format!("systemctl failed: {e}")));
        let output = match output {
            Ok(output) => output,
            Err(err) => return err.into_response(),
        };

        let success = output.status.success();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // For enable, also start the timer; for disable, also stop it.
        if success {
            let extra_action = if action == "enable" { "start" } else { "stop" };
            let _ = Command::new("systemctl")
                .arg("--user")
                .arg(extra_action)
                .arg(&timer)
                .output()
                .await;
        }

        results.push(json!({
            "unit": timer,
            "ok": success,
            "action": action,
            "stderr": stderr,
        }));
    }

    let ok = results
        .iter()
        .all(|item| item["ok"].as_bool().unwrap_or(false));
    Json(json!({ "ok": ok, "results": results })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::Path;
    use tempfile::tempdir;

    fn test_state(root: &StdPath) -> Arc<AppState> {
        let mut config = crate::config::HubConfig::from_env();
        config.aiq_root = root.to_path_buf();
        config.candles_db_dir = root.join("candles_dbs");
        config.live_db = root.join("live.db");
        config.paper1_db = root.join("paper1.db");
        config.paper2_db = root.join("paper2.db");
        config.paper3_db = root.join("paper3.db");
        AppState::new(config)
    }

    #[test]
    fn candidate_rows_prefer_report_candidates_before_legacy_fallbacks() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("configs")).unwrap();
        std::fs::write(dir.path().join("configs/candidate.yaml"), b"foo: bar\n").unwrap();

        let report = json!({
            "candidates": [
                { "config_id": "cfg-primary", "validation_gate": "candidate->validated_holdout" }
            ]
        });
        let metadata = json!({
            "items": [
                { "config_id": "legacy-item" }
            ]
        });

        let rows = candidate_rows_from_sources(dir.path(), Some(&report), &metadata).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["config_id"], "cfg-primary");
    }

    #[test]
    fn parse_timer_status_reads_unit_file_state_separately_from_active_state() {
        let status = parse_timer_status(
            "ActiveState=inactive\nNextElapseUSecRealtime=Mon 2026-03-16 00:00:00 UTC\nLoadState=loaded\nUnitFileState=enabled\n",
        );

        assert_eq!(
            status,
            TimerStatus {
                active: "inactive".to_string(),
                next_trigger: "Mon 2026-03-16 00:00:00 UTC".to_string(),
                load: "loaded".to_string(),
                unit_file_state: "enabled".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn list_runs_surfaces_rust_candidate_and_selection_summary_fields() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path().join("artifacts/2026-03-15/run_20260315T010203Z");
        std::fs::create_dir_all(run_dir.join("reports")).unwrap();
        std::fs::write(
            run_dir.join("run_metadata.json"),
            serde_json::to_vec(&json!({
                "generated_at_ms": 1234567890i64,
                "args": { "profile": "daily" },
                "items": [{ "config_id": "cfg-from-metadata" }],
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("reports/report.json"),
            serde_json::to_vec(&json!({
                "candidate_count": 3,
                "candidates": [{ "config_id": "cfg-a" }, { "config_id": "cfg-b" }, { "config_id": "cfg-c" }],
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("reports/selection.json"),
            serde_json::to_vec(&json!({
                "selection_stage": "selected_partial",
                "deploy_stage": "paper_partial",
                "selected_candidates_by_role": [{ "role": "primary", "config_id": "cfg-a" }],
            }))
            .unwrap(),
        )
        .unwrap();

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state)).await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["run_id"], "20260315T010203Z");
        assert_eq!(runs[0]["candidate_count"], 3);
        assert_eq!(runs[0]["selected_count"], 1);
        assert_eq!(runs[0]["selection_stage"], "selected_partial");
        assert_eq!(runs[0]["deploy_stage"], "paper_partial");
    }

    #[tokio::test]
    async fn list_runs_ignores_malformed_optional_reports_per_row() {
        let dir = tempdir().unwrap();
        let run_dir = dir
            .path()
            .join("artifacts/2026-03-15/run_daily_20260315T010203Z_001");
        std::fs::create_dir_all(run_dir.join("reports")).unwrap();
        std::fs::write(
            run_dir.join("run_metadata.json"),
            serde_json::to_vec(&json!({
                "generated_at_ms": 1234567890i64,
                "args": { "profile": "daily" },
                "items": [{ "config_id": "cfg-from-metadata" }],
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("reports/report.json"), b"{not-json").unwrap();

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state)).await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["run_id"], "daily_20260315T010203Z_001");
        assert_eq!(runs[0]["has_report"], Value::Bool(true));
        assert_eq!(runs[0]["candidate_count"], json!(1));
    }

    fn write_run_metadata(
        root: &StdPath,
        date: &str,
        directory_name: &str,
        generated_at_ms: i64,
        profile: &str,
    ) {
        let run_dir = root.join("artifacts").join(date).join(directory_name);
        fs::create_dir_all(run_dir.join("reports")).unwrap();
        fs::write(
            run_dir.join("run_metadata.json"),
            serde_json::to_vec(&json!({
                "generated_at_ms": generated_at_ms,
                "args": {
                    "profile": profile,
                },
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(run_dir.join("reports/report.json"), b"{}").unwrap();
    }

    #[tokio::test]
    async fn list_runs_canonicalises_rust_run_directories_and_sorts_newest_first() {
        let dir = tempdir().unwrap();
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_daily_20260315T010000Z_001",
            10,
            "daily",
        );
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_deep_20260315T020000Z_002",
            20,
            "deep",
        );

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state)).await.unwrap();

        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0]["run_id"], "deep_20260315T020000Z_002");
        assert_eq!(runs[0]["directory_name"], "run_deep_20260315T020000Z_002");
        assert_eq!(runs[1]["run_id"], "daily_20260315T010000Z_001");
    }

    #[tokio::test]
    async fn list_runs_deduplicates_prefixed_and_unprefixed_directory_forms() {
        let dir = tempdir().unwrap();
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_daily_20260315T010000Z_001",
            10,
            "daily",
        );
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "daily_20260315T010000Z_001",
            10,
            "daily-recovered",
        );

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state)).await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["run_id"], "daily_20260315T010000Z_001");
        assert_eq!(runs[0]["directory_name"], "run_daily_20260315T010000Z_001");
    }

    #[tokio::test]
    async fn run_detail_resolves_canonical_run_id_against_prefixed_directory() {
        let dir = tempdir().unwrap();
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_daily_20260315T010000Z_001",
            10,
            "daily",
        );

        let state = test_state(dir.path());
        let Json(detail) = run_detail(
            State(state),
            Path((
                "2026-03-15".to_string(),
                "daily_20260315T010000Z_001".to_string(),
            )),
        )
        .await
        .unwrap();

        assert_eq!(detail["run_id"], "daily_20260315T010000Z_001");
        assert_eq!(detail["directory_name"], "run_daily_20260315T010000Z_001");
    }

    #[tokio::test]
    async fn list_runs_keeps_listing_when_historical_report_or_selection_is_malformed() {
        let dir = tempdir().unwrap();
        let broken_run_dir = dir.path().join("artifacts/2026-03-15/run_20260315T010203Z");
        fs::create_dir_all(broken_run_dir.join("reports")).unwrap();
        fs::write(
            broken_run_dir.join("run_metadata.json"),
            serde_json::to_vec(&json!({
                "generated_at_ms": 10i64,
                "args": { "profile": "broken-history" },
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            broken_run_dir.join("reports/report.json"),
            b"{not valid json",
        )
        .unwrap();
        fs::write(
            broken_run_dir.join("reports/selection.json"),
            b"{still not valid json",
        )
        .unwrap();

        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_20260315T020304Z",
            20,
            "healthy-history",
        );

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state)).await.unwrap();

        assert_eq!(runs.len(), 2);
        let broken = runs
            .iter()
            .find(|run| run["run_id"] == "20260315T010203Z")
            .unwrap();
        assert_eq!(broken["profile"], "broken-history");
        assert_eq!(broken["has_report"], true);
        assert_eq!(broken["candidate_count"], Value::Null);
        assert_eq!(broken["selected_count"], Value::Null);
        assert_eq!(broken["selection_stage"], Value::Null);
        assert_eq!(broken["deploy_stage"], Value::Null);
    }

    #[tokio::test]
    async fn list_runs_deduplicates_canonical_run_ids_using_detail_resolution_preference() {
        let dir = tempdir().unwrap();
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "run_daily_20260315T010000Z_001",
            10,
            "prefixed",
        );
        write_run_metadata(
            dir.path(),
            "2026-03-15",
            "daily_20260315T010000Z_001",
            20,
            "unprefixed",
        );

        let state = test_state(dir.path());
        let Json(runs) = list_runs(State(state.clone())).await.unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["run_id"], "daily_20260315T010000Z_001");
        assert_eq!(runs[0]["directory_name"], "run_daily_20260315T010000Z_001");
        assert_eq!(runs[0]["profile"], "prefixed");

        let Json(detail) = run_detail(
            State(state),
            Path((
                "2026-03-15".to_string(),
                "daily_20260315T010000Z_001".to_string(),
            )),
        )
        .await
        .unwrap();

        assert_eq!(detail["directory_name"], "run_daily_20260315T010000Z_001");
        assert_eq!(detail["metadata"]["args"]["profile"], "prefixed");
    }
}
