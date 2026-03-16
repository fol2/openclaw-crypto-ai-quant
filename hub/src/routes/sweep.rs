use axum::{
    extract::{Path, State},
    middleware,
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path as FsPath, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::error::HubError;
use crate::state::AppState;
use crate::subprocess::{JobInfo, JobStatus, JobStore};

/// Body for POST /api/sweep/run.
#[derive(Deserialize)]
pub struct RunSweepBody {
    /// Config file variant (default "main").
    pub config: Option<String>,
    /// Path to sweep spec YAML file.
    pub sweep_spec: String,
    /// Initial balance.
    pub initial_balance: Option<f64>,
}

fn resolve_config_path(aiq_root: &FsPath, config_variant: &str) -> Result<String, HubError> {
    let config_name = match config_variant {
        "main" => "strategy_overrides.yaml",
        "live" => "strategy_overrides.live.yaml",
        "paper1" => "strategy_overrides.paper1.yaml",
        "paper2" => "strategy_overrides.paper2.yaml",
        "paper3" => "strategy_overrides.paper3.yaml",
        _ => {
            return Err(HubError::BadRequest(format!(
                "unknown config: {config_variant}"
            )))
        }
    };

    Ok(aiq_root
        .join("config")
        .join(config_name)
        .display()
        .to_string())
}

fn resolve_sweep_spec_path(aiq_root: &FsPath, sweep_spec: &str) -> String {
    let spec_path = FsPath::new(sweep_spec);
    if spec_path.is_absolute() {
        spec_path.display().to_string()
    } else {
        aiq_root.join(spec_path).display().to_string()
    }
}

fn prepare_sweep_output_path(artifacts_dir: &FsPath, job_id: &str) -> Result<PathBuf, HubError> {
    let dir = artifacts_dir.join("sweeps");
    std::fs::create_dir_all(&dir).map_err(|e| {
        HubError::Internal(format!(
            "failed to create sweep artifact directory {}: {e}",
            dir.display()
        ))
    })?;
    Ok(dir.join(format!("{job_id}.jsonl")))
}

fn parse_sweep_output_file(path: &FsPath) -> Result<Value, HubError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        HubError::Internal(format!(
            "failed to read sweep output {}: {e}",
            path.display()
        ))
    })?;

    let mut rows = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value = serde_json::from_str::<Value>(trimmed).map_err(|e| {
            HubError::Internal(format!(
                "failed to parse sweep output {} line {}: {e}",
                path.display(),
                index + 1
            ))
        })?;
        rows.push(value);
    }

    Ok(Value::Array(rows))
}

fn parse_sweep_results(stdout_output: &str, output_file: &FsPath) -> Result<Value, HubError> {
    let trimmed = stdout_output.trim();
    if !trimmed.is_empty() {
        serde_json::from_str::<Value>(trimmed).map_err(HubError::from)
    } else {
        parse_sweep_output_file(output_file)
    }
}

async fn spawn_sweep(
    job_id: String,
    config_path: String,
    sweep_spec_path: String,
    initial_balance: f64,
    output_file: PathBuf,
    aiq_root: PathBuf,
    store: Arc<JobStore>,
    broadcast: crate::ws::broadcast::BroadcastHub,
) {
    tokio::spawn(async move {
        let now = chrono::Utc::now().to_rfc3339();
        let info = JobInfo {
            id: job_id.clone(),
            kind: "sweep".to_string(),
            status: JobStatus::Running,
            created_at: now,
            finished_at: None,
            stderr_tail: Vec::new(),
            result_json: None,
            error: None,
        };

        {
            let mut jobs = store.jobs.lock().await;
            jobs.insert(job_id.clone(), info);
        }

        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--release")
            .arg("--manifest-path")
            .arg("backtester/Cargo.toml")
            .arg("--")
            .arg("sweep")
            .arg("--config")
            .arg(&config_path)
            .arg("--sweep-spec")
            .arg(&sweep_spec_path)
            .arg("--initial-balance")
            .arg(initial_balance.to_string())
            .arg("--output")
            .arg(&output_file)
            .current_dir(&aiq_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(e) => {
                let mut jobs = store.jobs.lock().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed;
                    job.error = Some(format!("spawn failed: {e}"));
                    job.finished_at = Some(chrono::Utc::now().to_rfc3339());
                }
                return;
            }
        };

        if let Some(child_id) = child.id() {
            let mut handles = store.handles.lock().await;
            handles.insert(job_id.clone(), child_id);
        }

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let stdout_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stdout) = stdout {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    output.push_str(&line);
                    output.push('\n');
                }
            }
            output
        });

        let store_for_stderr = Arc::clone(&store);
        let job_id_for_stderr = job_id.clone();
        let broadcast_for_stderr = broadcast.clone();
        let stderr_handle = tokio::spawn(async move {
            let mut tail: Vec<String> = Vec::new();
            if let Some(stderr) = stderr {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if let Ok(message) = serde_json::to_string(&json!({
                        "type": "job_progress",
                        "job_id": job_id_for_stderr,
                        "line": line,
                    })) {
                        broadcast_for_stderr.publish(&format!("job:{job_id_for_stderr}"), message);
                    }

                    tail.push(line);
                    if tail.len() > 100 {
                        tail.remove(0);
                    }
                }
            }

            let mut jobs = store_for_stderr.jobs.lock().await;
            if let Some(job) = jobs.get_mut(&job_id_for_stderr) {
                job.stderr_tail = tail;
            }
        });

        let exit_status = child.wait().await;

        let _ = stderr_handle.await;
        let stdout_output = stdout_handle.await.unwrap_or_default();

        {
            let mut handles = store.handles.lock().await;
            handles.remove(&job_id);
        }

        let mut jobs = store.jobs.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            if job.finished_at.is_none() {
                job.finished_at = Some(chrono::Utc::now().to_rfc3339());
            }

            if job.status != JobStatus::Cancelled {
                match exit_status {
                    Ok(status) if status.success() => {
                        job.status = JobStatus::Done;
                        match parse_sweep_results(&stdout_output, &output_file) {
                            Ok(value) => job.result_json = Some(value),
                            Err(e) => job.error = Some(e.to_string()),
                        }
                    }
                    Ok(status) => {
                        job.status = JobStatus::Failed;
                        job.error =
                            Some(format!("exited with code {}", status.code().unwrap_or(-1)));
                        if let Ok(value) = parse_sweep_results(&stdout_output, &output_file) {
                            job.result_json = Some(value);
                        }
                    }
                    Err(e) => {
                        job.status = JobStatus::Failed;
                        job.error = Some(format!("wait error: {e}"));
                        if let Ok(value) = parse_sweep_results(&stdout_output, &output_file) {
                            job.result_json = Some(value);
                        }
                    }
                }
            }

            if let Ok(message) = serde_json::to_string(&json!({
                "type": "job_done",
                "job_id": job_id,
                "status": job.status,
            })) {
                broadcast.publish(&format!("job:{job_id}"), message);
            }
        }
    });
}

/// Build sweep sub-router.
pub fn routes() -> Router<Arc<AppState>> {
    let read_routes = Router::new()
        .route("/api/sweep/jobs", get(list_jobs))
        .route("/api/sweep/{id}/status", get(job_status))
        .route("/api/sweep/{id}/results", get(job_results));
    let mutation_routes = Router::new()
        .route("/api/sweep/run", post(run_sweep))
        .route("/api/sweep/{id}", delete(cancel_job))
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));

    read_routes.merge(mutation_routes)
}

/// POST /api/sweep/run — launch a new sweep.
async fn run_sweep(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RunSweepBody>,
) -> Result<Json<Value>, HubError> {
    let job_id = uuid::Uuid::new_v4().to_string();
    let config_variant = body.config.as_deref().unwrap_or("main");
    let config_path = resolve_config_path(&state.config.aiq_root, config_variant)?;
    let sweep_spec = resolve_sweep_spec_path(&state.config.aiq_root, &body.sweep_spec);
    let balance = body.initial_balance.unwrap_or(10000.0);
    let output_file = prepare_sweep_output_path(&state.config.artifacts_dir, &job_id)?;

    spawn_sweep(
        job_id.clone(),
        config_path,
        sweep_spec,
        balance,
        output_file,
        state.config.aiq_root.clone(),
        Arc::clone(&state.jobs),
        state.broadcast.clone(),
    )
    .await;

    Ok(Json(json!({
        "job_id": job_id,
        "status": "running",
    })))
}

/// GET /api/sweep/jobs — list all sweep jobs.
async fn list_jobs(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Value>>, HubError> {
    let jobs = state.jobs.jobs.lock().await;
    let mut result: Vec<Value> = jobs
        .values()
        .filter(|j| j.kind == "sweep")
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

/// GET /api/sweep/{id}/status — job status + stderr tail.
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

/// GET /api/sweep/{id}/results — full result JSON.
async fn job_results(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, HubError> {
    let jobs = state.jobs.jobs.lock().await;
    let job = jobs
        .get(&id)
        .ok_or_else(|| HubError::NotFound(format!("job {id} not found")))?;
    if job.status == JobStatus::Running {
        return Err(HubError::BadRequest("job still running".into()));
    }
    match &job.result_json {
        Some(result) => Ok(Json(result.clone())),
        None => Err(HubError::NotFound("no result available".into())),
    }
}

/// DELETE /api/sweep/{id} — cancel a running sweep.
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
    if let Some(pid) = handles.remove(&id) {
        let rc = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                return Err(HubError::Internal(format!(
                    "failed to signal pid {pid}: {err}"
                )));
            }
        }
    }

    Ok(Json(json!({ "ok": true, "cancelled": id })))
}

#[cfg(test)]
mod tests {
    use super::{parse_sweep_output_file, prepare_sweep_output_path};
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn parse_sweep_output_file_reads_jsonl_rows() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("result.jsonl");
        std::fs::write(
            &path,
            [
                json!({ "config_id": "cfg_1", "total_pnl": 12.5 }).to_string(),
                String::new(),
                json!({ "config_id": "cfg_2", "total_pnl": 7.0 }).to_string(),
            ]
            .join("\n"),
        )
        .unwrap();

        let parsed = parse_sweep_output_file(&path).unwrap();
        let rows = parsed.as_array().unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["config_id"], "cfg_1");
        assert_eq!(rows[1]["config_id"], "cfg_2");
    }

    #[test]
    fn prepare_sweep_output_path_creates_sweeps_directory() {
        let dir = tempdir().unwrap();
        let path = prepare_sweep_output_path(dir.path(), "job-123").unwrap();

        assert_eq!(path, dir.path().join("sweeps").join("job-123.jsonl"));
        assert!(dir.path().join("sweeps").is_dir());
    }
}
