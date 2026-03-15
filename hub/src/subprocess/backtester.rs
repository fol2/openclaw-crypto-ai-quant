use serde_json::Value;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::ws::broadcast::BroadcastHub;

use super::{run_subprocess, JobId, JobInfo, JobStatus, JobStore};

/// Arguments for spawning a backtest replay.
pub struct ReplayArgs {
    pub config_path: String,
    pub initial_balance: f64,
    pub symbol: Option<String>,
    pub output_file: Option<String>,
    pub include_equity_curve: bool,
}

/// Arguments for spawning a parameter sweep.
pub struct SweepArgs {
    pub config_path: String,
    pub sweep_spec_path: String,
    pub initial_balance: f64,
    pub output_dir: Option<String>,
}

/// Spawn a backtest replay as a subprocess.
///
/// Runs `cargo run --release --manifest-path backtester/Cargo.toml -- replay ...`
/// in the project root directory. Stdout captures JSON report, stderr streams
/// progress lines.
pub async fn spawn_replay(
    job_id: JobId,
    args: ReplayArgs,
    aiq_root: String,
    store: Arc<JobStore>,
    broadcast: BroadcastHub,
) {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--release")
        .arg("--manifest-path")
        .arg("backtester/Cargo.toml")
        .arg("--")
        .arg("replay")
        .arg("--config")
        .arg(&args.config_path)
        .arg("--initial-balance")
        .arg(args.initial_balance.to_string());

    if let Some(ref sym) = args.symbol {
        cmd.arg("--symbol").arg(sym);
    }

    let output_file = args.output_file.clone();
    if let Some(ref out) = output_file {
        cmd.arg("--output").arg(out);
    }

    if args.include_equity_curve {
        cmd.arg("--equity-curve");
    }

    cmd.current_dir(&aiq_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    tokio::spawn(async move {
        let now = chrono::Utc::now().to_rfc3339();
        let info = JobInfo {
            id: job_id.clone(),
            kind: "backtest".to_string(),
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

        let child = cmd.spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let mut jobs = store.jobs.lock().await;
                if let Some(j) = jobs.get_mut(&job_id) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("spawn failed: {e}"));
                    j.finished_at = Some(chrono::Utc::now().to_rfc3339());
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

        let store_clone = Arc::clone(&store);
        let jid_clone = job_id.clone();
        let bc = broadcast.clone();
        let stderr_handle = tokio::spawn(async move {
            let mut tail: Vec<String> = Vec::new();
            if let Some(stderr) = stderr {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                        "type": "job_progress",
                        "job_id": jid_clone,
                        "line": line,
                    })) {
                        bc.publish(&format!("job:{jid_clone}"), msg);
                    }

                    tail.push(line);
                    if tail.len() > 100 {
                        tail.remove(0);
                    }
                }
            }

            let mut jobs = store_clone.jobs.lock().await;
            if let Some(j) = jobs.get_mut(&jid_clone) {
                j.stderr_tail = tail;
            }
        });

        let exit_status = child.wait().await;

        let _ = stderr_handle.await;
        let stdout_output = stdout_handle.await.unwrap_or_default();

        {
            let mut handles = store.handles.lock().await;
            handles.remove(&job_id);
        }

        let parsed_output = parse_replay_result(output_file.as_deref().map(Path::new), &stdout_output)
            .map(limit_backtest_result_payload);

        let mut jobs = store.jobs.lock().await;
        if let Some(j) = jobs.get_mut(&job_id) {
            j.finished_at = Some(chrono::Utc::now().to_rfc3339());

            match exit_status {
                Ok(status) if status.success() => {
                    j.status = JobStatus::Done;
                    match parsed_output {
                        Ok(val) => j.result_json = Some(val),
                        Err(e) => {
                            j.error = Some(e);
                        }
                    }
                }
                Ok(status) => {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("exited with code {}", status.code().unwrap_or(-1)));
                    if let Ok(val) = parsed_output {
                        j.result_json = Some(val);
                    }
                }
                Err(e) => {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("wait error: {e}"));
                }
            }

            if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                "type": "job_done",
                "job_id": job_id,
                "status": j.status,
            })) {
                broadcast.publish(&format!("job:{job_id}"), msg);
            }
        }
    });
}

const MAX_BACKTEST_EQUITY_POINTS: usize = 240;

fn parse_replay_result(output_file: Option<&Path>, stdout_output: &str) -> Result<Value, String> {
    let trimmed = stdout_output.trim();
    if !trimmed.is_empty() {
        return serde_json::from_str::<Value>(trimmed)
            .map_err(|e| format!("failed to parse result JSON: {e}"));
    }

    let Some(output_file) = output_file else {
        return Err("no replay result payload was emitted".to_string());
    };

    let raw = std::fs::read_to_string(output_file).map_err(|e| {
        format!(
            "failed to read replay output {}: {e}",
            output_file.display()
        )
    })?;
    serde_json::from_str::<Value>(&raw).map_err(|e| {
        format!(
            "failed to parse replay output {}: {e}",
            output_file.display()
        )
    })
}

fn limit_backtest_result_payload(mut result: Value) -> Value {
    let Some(obj) = result.as_object_mut() else {
        return result;
    };

    let Some(points) = obj.get_mut("equity_curve").and_then(Value::as_array_mut) else {
        return result;
    };

    if points.len() <= MAX_BACKTEST_EQUITY_POINTS {
        return result;
    }

    let len = points.len();
    let target = MAX_BACKTEST_EQUITY_POINTS.max(2);
    let last_index = len - 1;
    let interior = target - 2;

    let mut sampled = Vec::with_capacity(target);
    sampled.push(points[0].clone());
    for slot in 1..=interior {
        let index = slot * last_index / (interior + 1);
        sampled.push(points[index].clone());
    }
    sampled.push(points[last_index].clone());
    *points = sampled;

    result
}

#[cfg(test)]
mod tests {
    use super::{limit_backtest_result_payload, MAX_BACKTEST_EQUITY_POINTS};
    use serde_json::json;

    #[test]
    fn limit_backtest_result_payload_downsamples_equity_curve() {
        let curve: Vec<_> = (0..1_000)
            .map(|i| json!([1_700_000_000_000_i64 + i as i64, 10_000.0 + i as f64]))
            .collect();
        let result = json!({
            "equity_curve": curve,
        });

        let limited = limit_backtest_result_payload(result);
        let points = limited["equity_curve"].as_array().unwrap();

        assert_eq!(points.len(), MAX_BACKTEST_EQUITY_POINTS);
        assert_eq!(points.first().unwrap(), &json!([1_700_000_000_000_i64, 10_000.0]));
        assert_eq!(points.last().unwrap(), &json!([1_700_000_000_999_i64, 10_999.0]));
    }
}

/// Spawn a parameter sweep as a subprocess.
pub async fn spawn_sweep(
    job_id: JobId,
    args: SweepArgs,
    aiq_root: String,
    store: Arc<JobStore>,
    broadcast: BroadcastHub,
) {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--release")
        .arg("--manifest-path")
        .arg("backtester/Cargo.toml")
        .arg("--")
        .arg("sweep")
        .arg("--config")
        .arg(&args.config_path)
        .arg("--sweep-spec")
        .arg(&args.sweep_spec_path)
        .arg("--initial-balance")
        .arg(args.initial_balance.to_string());

    if let Some(ref out_dir) = args.output_dir {
        cmd.arg("--output-dir").arg(out_dir);
    }

    cmd.current_dir(&aiq_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let jid = job_id.clone();
    tokio::spawn(async move {
        run_subprocess(jid, "sweep", cmd, store, broadcast).await;
    });
}
