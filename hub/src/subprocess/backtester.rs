use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::ws::broadcast::BroadcastHub;

use super::{JobId, JobInfo, JobStatus, JobStore};

/// Arguments for spawning a backtest replay.
pub struct ReplayArgs {
    pub config_path: String,
    pub initial_balance: f64,
    pub symbol: Option<String>,
    pub output_file: Option<String>,
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

    if let Some(ref out) = args.output_file {
        cmd.arg("--output").arg(out);
    }

    cmd.current_dir(&aiq_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let jid = job_id.clone();
    tokio::spawn(async move {
        run_subprocess(jid, "backtest", cmd, store, broadcast).await;
    });
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

/// Generic subprocess runner: captures stdout + stderr, updates job store.
async fn run_subprocess(
    job_id: JobId,
    kind: &str,
    mut cmd: Command,
    store: Arc<JobStore>,
    broadcast: BroadcastHub,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let info = JobInfo {
        id: job_id.clone(),
        kind: kind.to_string(),
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

    // Store child handle for cancellation.
    let child_id = child.id();
    {
        // We can't store Child directly since we need to take stdout/stderr.
        // Instead, we'll handle cancellation via the PID.
        let _ = child_id;
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Collect stdout (JSON result).
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

    // Stream stderr (progress lines) and broadcast via WebSocket.
    let store_clone = Arc::clone(&store);
    let jid_clone = job_id.clone();
    let bc = broadcast.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut tail: Vec<String> = Vec::new();
        if let Some(stderr) = stderr {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                // Broadcast progress line.
                if let Ok(msg) = serde_json::to_string(&serde_json::json!({
                    "type": "job_progress",
                    "job_id": jid_clone,
                    "line": line,
                })) {
                    bc.publish(&format!("job:{jid_clone}"), msg);
                }

                // Keep last 100 lines.
                tail.push(line);
                if tail.len() > 100 {
                    tail.remove(0);
                }
            }
        }

        // Update stderr tail.
        let mut jobs = store_clone.jobs.lock().await;
        if let Some(j) = jobs.get_mut(&jid_clone) {
            j.stderr_tail = tail;
        }
    });

    // Wait for process to exit.
    let exit_status = child.wait().await;

    let _ = stderr_handle.await;
    let stdout_output = stdout_handle.await.unwrap_or_default();

    // Update job status.
    let mut jobs = store.jobs.lock().await;
    if let Some(j) = jobs.get_mut(&job_id) {
        j.finished_at = Some(chrono::Utc::now().to_rfc3339());

        match exit_status {
            Ok(status) if status.success() => {
                j.status = JobStatus::Done;
                // Parse stdout as JSON.
                match serde_json::from_str::<serde_json::Value>(&stdout_output) {
                    Ok(val) => j.result_json = Some(val),
                    Err(e) => {
                        j.error = Some(format!("failed to parse result JSON: {e}"));
                        // Still mark as done — the output file may have been used.
                    }
                }
            }
            Ok(status) => {
                j.status = JobStatus::Failed;
                j.error = Some(format!("exited with code {}", status.code().unwrap_or(-1)));
            }
            Err(e) => {
                j.status = JobStatus::Failed;
                j.error = Some(format!("wait error: {e}"));
            }
        }

        // Broadcast completion.
        if let Ok(msg) = serde_json::to_string(&serde_json::json!({
            "type": "job_done",
            "job_id": job_id,
            "status": j.status,
        })) {
            broadcast.publish(&format!("job:{job_id}"), msg);
        }
    }
}
