use axum::{
    extract::{Path, State},
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;

use crate::error::HubError;
use crate::state::AppState;
use crate::subprocess::factory;
use crate::subprocess::JobStatus;

/// Build factory sub-router.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Existing read-only artifact routes
        .route("/api/factory/runs", get(list_runs))
        .route("/api/factory/runs/{date}/{run_id}", get(run_detail))
        .route("/api/factory/runs/{date}/{run_id}/report", get(run_report))
        .route(
            "/api/factory/runs/{date}/{run_id}/candidates",
            get(run_candidates),
        )
        // Job control routes
        .route("/api/factory/run", post(run_factory))
        .route("/api/factory/jobs", get(list_jobs))
        .route("/api/factory/jobs/{id}/status", get(job_status))
        .route("/api/factory/jobs/{id}", delete(cancel_job))
        // Settings routes
        .route("/api/factory/settings", get(get_settings).put(put_settings))
        // Timer routes
        .route("/api/factory/timer", get(get_timer))
        .route("/api/factory/timer/{action}", post(timer_action))
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
        for entry in fs::read_dir(&date_dir)? {
            let entry = entry?;
            let run_id = entry.file_name().to_string_lossy().to_string();
            if !entry.file_type()?.is_dir() {
                continue;
            }

            // Read metadata if available.
            let meta_path = entry.path().join("run_metadata.json");
            let metadata = if meta_path.exists() {
                fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            } else {
                None
            };

            let has_report = entry.path().join("reports/report.json").exists();

            runs.push(json!({
                "date": date,
                "run_id": run_id,
                "has_report": has_report,
                "profile": metadata.as_ref()
                    .and_then(|m| m["args"]["profile"].as_str())
                    .unwrap_or("unknown"),
                "num_candidates": metadata.as_ref()
                    .and_then(|m| m["args"]["num_candidates"].as_i64()),
            }));
        }
    }

    Ok(Json(runs))
}

/// GET /api/factory/runs/{date}/{run_id} — Run metadata.
async fn run_detail(
    State(state): State<Arc<AppState>>,
    Path((date, run_id)): Path<(String, String)>,
) -> Result<Json<Value>, HubError> {
    validate_date(&date)?;
    validate_run_id(&run_id)?;

    let run_dir = artifacts_dir(&state).join(&date).join(&run_id);
    if !run_dir.is_dir() {
        return Err(HubError::NotFound(format!(
            "run not found: {date}/{run_id}"
        )));
    }

    let meta_path = run_dir.join("run_metadata.json");
    let metadata = if meta_path.exists() {
        let raw = fs::read_to_string(&meta_path)?;
        serde_json::from_str::<Value>(&raw)?
    } else {
        json!({})
    };

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
        "metadata": metadata,
        "subdirs": subdirs,
    })))
}

/// GET /api/factory/runs/{date}/{run_id}/report — Factory report.json content.
async fn run_report(
    State(state): State<Arc<AppState>>,
    Path((date, run_id)): Path<(String, String)>,
) -> Result<Json<Value>, HubError> {
    validate_date(&date)?;
    validate_run_id(&run_id)?;

    let report_path = artifacts_dir(&state)
        .join(&date)
        .join(&run_id)
        .join("reports/report.json");

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
    validate_date(&date)?;
    validate_run_id(&run_id)?;

    let configs_dir = artifacts_dir(&state)
        .join(&date)
        .join(&run_id)
        .join("configs");

    if !configs_dir.exists() {
        return Ok(Json(Vec::new()));
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

    Ok(Json(candidates))
}

// ── Job control routes ─────────────────────────────────────────────

/// Valid factory profiles.
const VALID_PROFILES: &[&str] = &["smoke", "daily", "deep", "weekly"];

/// Body for POST /api/factory/run.
#[derive(Deserialize)]
struct RunFactoryBody {
    profile: Option<String>,
    strategy_mode: Option<String>,
    gpu: Option<bool>,
    tpe: Option<bool>,
    walk_forward: Option<bool>,
    slippage_stress: Option<bool>,
    concentration_checks: Option<bool>,
    sensitivity_checks: Option<bool>,
    candidate_count: Option<u32>,
    max_age_fail_hours: Option<f64>,
    funding_max_stale_symbols: Option<u32>,
    dry_run: Option<bool>,
    no_deploy: Option<bool>,
    enable_livepaper_promotion: Option<bool>,
}

/// Resolve settings file path.
fn settings_path(state: &AppState) -> PathBuf {
    state.config.aiq_root.join("config/factory_defaults.yaml")
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

/// POST /api/factory/run — launch a new factory cycle.
async fn run_factory(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RunFactoryBody>,
) -> Result<Json<Value>, HubError> {
    // Load saved settings as defaults, then override with request body.
    let defaults = load_settings(&state);

    let profile = body
        .profile
        .or_else(|| defaults["profile"].as_str().map(String::from))
        .unwrap_or_else(|| "daily".to_string());

    if !VALID_PROFILES.contains(&profile.as_str()) {
        return Err(HubError::BadRequest(format!(
            "invalid profile: {profile} (valid: {})",
            VALID_PROFILES.join(", ")
        )));
    }

    let candidate_count = body
        .candidate_count
        .or_else(|| defaults["candidate_count"].as_u64().map(|n| n as u32))
        .unwrap_or(3);

    if !(1..=10).contains(&candidate_count) {
        return Err(HubError::BadRequest(format!(
            "candidate_count must be 1-10, got {candidate_count}"
        )));
    }

    let args = factory::FactoryArgs {
        profile,
        strategy_mode: body
            .strategy_mode
            .or_else(|| defaults["strategy_mode"].as_str().map(String::from)),
        gpu: body
            .gpu
            .or_else(|| defaults["gpu"].as_bool())
            .unwrap_or(true),
        tpe: body
            .tpe
            .or_else(|| defaults["tpe"].as_bool())
            .unwrap_or(true),
        walk_forward: body
            .walk_forward
            .or_else(|| defaults["walk_forward"].as_bool())
            .unwrap_or(true),
        slippage_stress: body
            .slippage_stress
            .or_else(|| defaults["slippage_stress"].as_bool())
            .unwrap_or(true),
        concentration_checks: body
            .concentration_checks
            .or_else(|| defaults["concentration_checks"].as_bool())
            .unwrap_or(true),
        sensitivity_checks: body
            .sensitivity_checks
            .or_else(|| defaults["sensitivity_checks"].as_bool())
            .unwrap_or(true),
        candidate_count,
        max_age_fail_hours: body
            .max_age_fail_hours
            .or_else(|| defaults["max_age_fail_hours"].as_f64()),
        funding_max_stale_symbols: body
            .funding_max_stale_symbols
            .or_else(|| defaults["funding_max_stale_symbols"].as_u64().map(|n| n as u32)),
        dry_run: body
            .dry_run
            .or_else(|| defaults["dry_run"].as_bool())
            .unwrap_or(false),
        no_deploy: body
            .no_deploy
            .or_else(|| defaults["no_deploy"].as_bool())
            .unwrap_or(false),
        enable_livepaper_promotion: body
            .enable_livepaper_promotion
            .or_else(|| defaults["enable_livepaper_promotion"].as_bool())
            .unwrap_or(true),
    };

    let job_id = uuid::Uuid::new_v4().to_string();

    factory::spawn_factory(
        job_id.clone(),
        args,
        state.config.aiq_root.display().to_string(),
        Arc::clone(&state.jobs),
        state.broadcast.clone(),
    )
    .await;

    Ok(Json(json!({
        "job_id": job_id,
        "status": "running",
    })))
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
async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, HubError> {
    let mut jobs = state.jobs.jobs.lock().await;
    let job = jobs
        .get_mut(&id)
        .ok_or_else(|| HubError::NotFound(format!("job {id} not found")))?;
    if job.status != JobStatus::Running {
        return Err(HubError::BadRequest("job is not running".into()));
    }
    job.status = JobStatus::Cancelled;
    job.finished_at = Some(chrono::Utc::now().to_rfc3339());

    let mut handles = state.jobs.handles.lock().await;
    if let Some(mut child) = handles.remove(&id) {
        let _ = child.kill().await;
    }

    Ok(Json(json!({ "ok": true, "cancelled": id })))
}

// ── Settings routes ────────────────────────────────────────────────

/// GET /api/factory/settings — read saved factory defaults.
async fn get_settings(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, HubError> {
    Ok(Json(load_settings(&state)))
}

/// PUT /api/factory/settings — save factory defaults.
async fn put_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, HubError> {
    let path = settings_path(&state);
    let yaml = serde_yaml::to_string(&body)?;
    fs::write(&path, &yaml)?;
    Ok(Json(json!({ "ok": true })))
}

// ── Timer routes ───────────────────────────────────────────────────

/// Factory timer unit names.
const FACTORY_TIMER: &str = "openclaw-ai-quant-factory-v8.timer";
const FACTORY_DEEP_TIMER: &str = "openclaw-ai-quant-factory-v8-deep.timer";

/// GET /api/factory/timer — timer status.
async fn get_timer(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<Value>, HubError> {
    let mut timers = Vec::new();
    for timer in [FACTORY_TIMER, FACTORY_DEEP_TIMER] {
        let output = Command::new("systemctl")
            .arg("--user")
            .arg("show")
            .arg(timer)
            .arg("--property=ActiveState,NextElapseUSecRealtime,LoadState")
            .output()
            .await
            .map_err(|e| HubError::Internal(format!("systemctl failed: {e}")))?;

        let text = String::from_utf8_lossy(&output.stdout);
        let mut active = String::new();
        let mut next_elapse = String::new();
        let mut load = String::new();
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("ActiveState=") {
                active = v.to_string();
            } else if let Some(v) = line.strip_prefix("NextElapseUSecRealtime=") {
                next_elapse = v.to_string();
            } else if let Some(v) = line.strip_prefix("LoadState=") {
                load = v.to_string();
            }
        }

        let enabled = active == "active" || active == "waiting";
        timers.push(json!({
            "unit": timer,
            "active": active,
            "load": load,
            "enabled": enabled,
            "next_trigger": next_elapse,
        }));
    }

    Ok(Json(json!({ "timers": timers })))
}

/// POST /api/factory/timer/{action} — enable or disable factory timer.
async fn timer_action(
    Path(action): Path<String>,
) -> Result<Json<Value>, HubError> {
    match action.as_str() {
        "enable" | "disable" => {}
        _ => {
            return Err(HubError::BadRequest(format!(
                "invalid action: {action} (valid: enable, disable)"
            )));
        }
    }

    let mut results = Vec::new();
    for timer in [FACTORY_TIMER, FACTORY_DEEP_TIMER] {
        // enable/disable controls auto-start
        let output = Command::new("systemctl")
            .arg("--user")
            .arg(&action)
            .arg(timer)
            .output()
            .await
            .map_err(|e| HubError::Internal(format!("systemctl failed: {e}")))?;

        let success = output.status.success();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // For enable, also start the timer; for disable, also stop it.
        if success {
            let extra_action = if action == "enable" { "start" } else { "stop" };
            let _ = Command::new("systemctl")
                .arg("--user")
                .arg(extra_action)
                .arg(timer)
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

    Ok(Json(json!({ "ok": true, "results": results })))
}
