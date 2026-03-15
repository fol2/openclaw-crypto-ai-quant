use axum::{
    extract::{Path, State},
    middleware,
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::HubError;
use crate::state::AppState;
use crate::subprocess::backtester;
use crate::subprocess::JobStatus;

/// Body for POST /api/backtest/run.
#[derive(Deserialize)]
pub struct RunBacktestBody {
    /// Config file variant (default "main").
    pub config: Option<String>,
    /// Initial balance for the replay.
    pub initial_balance: Option<f64>,
    /// Optional single-symbol filter.
    pub symbol: Option<String>,
}

/// Build backtest sub-router.
pub fn routes() -> Router<Arc<AppState>> {
    let read_routes = Router::new()
        .route("/api/backtest/jobs", get(list_jobs))
        .route("/api/backtest/{id}/status", get(job_status))
        .route("/api/backtest/{id}/result", get(job_result));
    let mutation_routes = Router::new()
        .route("/api/backtest/run", post(run_backtest))
        .route("/api/backtest/{id}", delete(cancel_job))
        .route_layer(middleware::from_fn(crate::auth::require_admin_auth));

    read_routes.merge(mutation_routes)
}

/// POST /api/backtest/run — launch a new backtest.
async fn run_backtest(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RunBacktestBody>,
) -> Result<Json<Value>, HubError> {
    let job_id = uuid::Uuid::new_v4().to_string();

    let config_variant = body.config.as_deref().unwrap_or("main");
    let config_path = format!(
        "{}/config/{}",
        state.config.aiq_root.display(),
        match config_variant {
            "main" => "strategy_overrides.yaml",
            "live" => "strategy_overrides.live.yaml",
            "paper1" => "strategy_overrides.paper1.yaml",
            "paper2" => "strategy_overrides.paper2.yaml",
            "paper3" => "strategy_overrides.paper3.yaml",
            _ =>
                return Err(HubError::BadRequest(format!(
                    "unknown config: {config_variant}"
                ))),
        }
    );

    let balance = body.initial_balance.unwrap_or(10000.0);
    let output_file = prepare_backtest_output_path(&state.config.artifacts_dir, &job_id)?;

    let args = backtester::ReplayArgs {
        config_path,
        initial_balance: balance,
        symbol: body.symbol,
        output_file: Some(output_file.display().to_string()),
        include_equity_curve: true,
    };

    backtester::spawn_replay(
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

fn prepare_backtest_output_path(artifacts_dir: &PathBuf, job_id: &str) -> Result<PathBuf, HubError> {
    let dir = artifacts_dir.join("backtests");
    std::fs::create_dir_all(&dir).map_err(|e| {
        HubError::Internal(format!(
            "failed to create backtest artifact directory {}: {e}",
            dir.display()
        ))
    })?;
    Ok(dir.join(format!("{job_id}.json")))
}

fn normalise_backtest_result(result: &Value) -> Value {
    let mut result = result.clone();
    let Some(obj) = result.as_object_mut() else {
        return result;
    };

    if let Some(total_pnl) = obj.get("total_pnl").cloned() {
        obj.entry("net_pnl".to_string()).or_insert(total_pnl);
    }

    if let Some(max_drawdown_pct) = obj.get("max_drawdown_pct").and_then(Value::as_f64) {
        obj.entry("max_drawdown_percent".to_string())
            .or_insert_with(|| json!(max_drawdown_pct * 100.0));
    }

    if !obj.contains_key("duration_days") {
        if let Some(duration_days) = extract_duration_days(obj.get("equity_curve")) {
            obj.insert("duration_days".to_string(), json!(duration_days));
        }
    }

    if let Some(per_symbol) = obj.get_mut("per_symbol").and_then(Value::as_object_mut) {
        for stats in per_symbol.values_mut() {
            let Some(stats_obj) = stats.as_object_mut() else {
                continue;
            };

            if stats_obj.contains_key("pnl") {
                continue;
            }

            if let Some(net_pnl) = stats_obj
                .get("net_pnl_usd")
                .cloned()
                .or_else(|| stats_obj.get("realised_pnl_usd").cloned())
            {
                stats_obj.insert("pnl".to_string(), net_pnl);
            }
        }
    }

    result
}

fn extract_duration_days(equity_curve: Option<&Value>) -> Option<f64> {
    let points = equity_curve?.as_array()?;
    let first = points.first().and_then(extract_equity_point_ts)?;
    let last = points.last().and_then(extract_equity_point_ts)?;
    let duration_ms = last.saturating_sub(first).max(0) as f64;
    Some(duration_ms / 86_400_000.0)
}

fn extract_equity_point_ts(point: &Value) -> Option<i64> {
    point
        .as_array()
        .and_then(|values| values.first())
        .and_then(Value::as_i64)
        .or_else(|| point.get("timestamp_ms").and_then(Value::as_i64))
        .or_else(|| point.get("timestamp").and_then(Value::as_i64))
}

/// GET /api/backtest/jobs — list all backtest jobs.
async fn list_jobs(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Value>>, HubError> {
    let jobs = state.jobs.jobs.lock().await;
    let mut result: Vec<Value> = jobs
        .values()
        .filter(|j| j.kind == "backtest")
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
    // Sort newest first.
    result.sort_by(|a, b| {
        let ta = a["created_at"].as_str().unwrap_or("");
        let tb = b["created_at"].as_str().unwrap_or("");
        tb.cmp(ta)
    });
    Ok(Json(result))
}

/// GET /api/backtest/{id}/status — job status + stderr tail.
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

/// GET /api/backtest/{id}/result — full result JSON.
async fn job_result(
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
        Some(result) => Ok(Json(normalise_backtest_result(result))),
        None => Err(HubError::NotFound("no result available".into())),
    }
}

/// DELETE /api/backtest/{id} — cancel a running job.
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

    // Kill the subprocess via handles.
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
    use super::{normalise_backtest_result, prepare_backtest_output_path};
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn normalise_backtest_result_adds_page_aliases() {
        let result = json!({
            "total_pnl": 123.45,
            "max_drawdown_pct": 0.125,
            "equity_curve": [
                [1_700_000_000_000_i64, 10_000.0],
                [1_700_086_400_000_i64, 10_123.45]
            ],
            "per_symbol": {
                "BTC": {
                    "trades": 3,
                    "net_pnl_usd": 45.67
                }
            }
        });

        let normalised = normalise_backtest_result(&result);

        assert_eq!(normalised["total_pnl"], json!(123.45));
        assert_eq!(normalised["net_pnl"], json!(123.45));
        assert_eq!(normalised["max_drawdown_pct"], json!(0.125));
        assert_eq!(normalised["max_drawdown_percent"], json!(12.5));
        assert_eq!(normalised["duration_days"], json!(1.0));
        assert_eq!(normalised["per_symbol"]["BTC"]["pnl"], json!(45.67));
    }

    #[test]
    fn prepare_backtest_output_path_creates_backtests_directory() {
        let dir = tempdir().unwrap();
        let path = prepare_backtest_output_path(&dir.path().to_path_buf(), "job-123").unwrap();

        assert_eq!(path, dir.path().join("backtests").join("job-123.json"));
        assert!(dir.path().join("backtests").is_dir());
    }
}
