use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::decision_events;
use crate::paper_daemon;
use crate::paper_lane::PaperLane;
use crate::paper_status::{self, PaperServiceState, PaperStatusInput, PaperStatusReport};

#[derive(Clone, Copy)]
pub struct PaperServiceInput<'a> {
    pub config: Option<&'a Path>,
    pub lane: Option<PaperLane>,
    pub project_dir: Option<&'a Path>,
    pub live: bool,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub candles_db: Option<&'a Path>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub watch_symbols_file: bool,
    pub btc_symbol: &'a str,
    pub lookback_bars: Option<usize>,
    pub start_step_close_ts_ms: Option<i64>,
    pub bootstrap_from_latest_common_close: bool,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub stale_after_ms: Option<i64>,
}

#[derive(Clone, Copy)]
pub struct PaperServiceApplyInput<'a> {
    pub service: PaperServiceInput<'a>,
    pub expected_config_id: Option<&'a str>,
    pub requested_action: PaperServiceApplyRequestedAction,
    pub start_wait_ms: u64,
    pub stop_wait_ms: u64,
    pub poll_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperSupervisorAction {
    Hold,
    Start,
    Restart,
    Monitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperServiceApplyRequestedAction {
    Auto,
    Start,
    Restart,
    Stop,
    Resume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperServiceApplyExecutedAction {
    Noop,
    Start,
    Restart,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperServiceReport {
    pub ok: bool,
    pub desired_action: PaperSupervisorAction,
    pub action_reason: String,
    pub daemon_command: Vec<String>,
    pub lock_path: String,
    pub status_path: String,
    pub status: PaperStatusReport,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperServiceApplyReport {
    pub ok: bool,
    pub requested_action: PaperServiceApplyRequestedAction,
    pub applied_action: PaperServiceApplyExecutedAction,
    pub action_reason: String,
    pub lock_owner_pid: Option<u32>,
    pub previous_pid: Option<u32>,
    pub spawned_pid: Option<u32>,
    pub preview: PaperServiceReport,
    pub final_service: PaperServiceReport,
}

#[derive(Debug, Clone)]
struct PaperServiceApplyPlan {
    applied_action: PaperServiceApplyExecutedAction,
    action_reason: String,
    stop_pid: Option<u32>,
}

pub fn build_service(input: PaperServiceInput<'_>) -> Result<PaperServiceReport> {
    let status = paper_status::build_status(PaperStatusInput {
        config: input.config,
        lane: input.lane,
        project_dir: input.project_dir,
        live: input.live,
        profile: input.profile,
        db: input.db,
        candles_db: input.candles_db,
        symbols: input.symbols,
        symbols_file: input.symbols_file,
        watch_symbols_file: input.watch_symbols_file,
        btc_symbol: input.btc_symbol,
        lookback_bars: input.lookback_bars,
        start_step_close_ts_ms: input.start_step_close_ts_ms,
        bootstrap_from_latest_common_close: input.bootstrap_from_latest_common_close,
        lock_path: input.lock_path,
        status_path: input.status_path,
        stale_after_ms: input.stale_after_ms,
    })?;
    let (desired_action, action_reason) = derive_action(&status);
    let warnings = status.warnings.clone();

    Ok(PaperServiceReport {
        ok: status.ok,
        desired_action,
        action_reason,
        daemon_command: status.manifest.daemon_command.clone(),
        lock_path: status.manifest.lock_path.clone(),
        status_path: status.manifest.status_path.clone(),
        status,
        warnings,
    })
}

pub fn apply_service(input: PaperServiceApplyInput<'_>) -> Result<PaperServiceApplyReport> {
    let preview = build_service(input.service)?;
    verify_expected_config_id(input.expected_config_id, &preview.status.manifest.config_id)?;
    let lock_path = PathBuf::from(&preview.lock_path);
    let status_path = PathBuf::from(&preview.status_path);
    let lock_owner_pid = paper_daemon::probe_lock_owner(&lock_path)?;
    let plan = build_apply_plan(&preview, input.requested_action, lock_owner_pid)?;
    let previous_pid = running_status_pid(&preview);
    let poll_interval = std::time::Duration::from_millis(input.poll_ms.max(10));
    let mut spawned_pid = None;

    match plan.applied_action {
        PaperServiceApplyExecutedAction::Noop => {}
        PaperServiceApplyExecutedAction::Start => {
            reconcile_decision_events_preflight(&preview)?;
            spawned_pid = Some(start_daemon(
                &preview.daemon_command,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.start_wait_ms.max(100)),
                poll_interval,
            )?);
        }
        PaperServiceApplyExecutedAction::Restart => {
            let stop_pid = plan
                .stop_pid
                .context("restart plan requires a supervised stop pid")?;
            stop_daemon(
                stop_pid,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.stop_wait_ms.max(100)),
                poll_interval,
            )?;
            reconcile_decision_events_preflight(&preview)?;
            spawned_pid = Some(start_daemon(
                &preview.daemon_command,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.start_wait_ms.max(100)),
                poll_interval,
            )?);
        }
        PaperServiceApplyExecutedAction::Stop => {
            let stop_pid = plan
                .stop_pid
                .context("stop plan requires a supervised stop pid")?;
            stop_daemon(
                stop_pid,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.stop_wait_ms.max(100)),
                poll_interval,
            )?;
        }
    }

    let final_service = build_service(input.service)?;
    verify_expected_config_id(
        input.expected_config_id,
        &final_service.status.manifest.config_id,
    )?;

    Ok(PaperServiceApplyReport {
        ok: final_service.ok,
        requested_action: input.requested_action,
        applied_action: plan.applied_action,
        action_reason: plan.action_reason,
        lock_owner_pid,
        previous_pid,
        spawned_pid,
        preview,
        final_service,
    })
}

fn reconcile_decision_events_preflight(preview: &PaperServiceReport) -> Result<()> {
    let paper_db = Path::new(&preview.status.manifest.paper_db);
    decision_events::reconcile_db_schema(paper_db).with_context(|| {
        format!(
            "paper service apply refused to start because decision_events schema preflight failed for {}",
            paper_db.display()
        )
    })
}

fn verify_expected_config_id(
    expected_config_id: Option<&str>,
    actual_config_id: &str,
) -> Result<()> {
    if let Some(expected_config_id) = expected_config_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if expected_config_id != actual_config_id {
            anyhow::bail!(
                "paper service apply refused to continue because expected config_id {} did not match resolved {}",
                expected_config_id,
                actual_config_id
            );
        }
    }
    Ok(())
}

fn derive_action(status: &PaperStatusReport) -> (PaperSupervisorAction, String) {
    match status.service_state {
        PaperServiceState::Blocked => (
            PaperSupervisorAction::Hold,
            "current launch contract is blocked and should not be supervised yet".to_string(),
        ),
        PaperServiceState::IdleNoSymbols => {
            if status.manifest.resume.launch_ready {
                (
                    PaperSupervisorAction::Start,
                    "launch contract is launch-ready but currently has no active symbols; the Rust daemon may start and wait for a watched symbols file or future open positions"
                        .to_string(),
                )
            } else {
                (
                    PaperSupervisorAction::Hold,
                    "no active symbols or open paper positions are currently available for this lane"
                        .to_string(),
                )
            }
        }
        PaperServiceState::BootstrapRequired => (
            PaperSupervisorAction::Hold,
            "first launch still requires --start-step-close-ts-ms or AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS"
                .to_string(),
        ),
        PaperServiceState::BootstrapReady => (
            PaperSupervisorAction::Start,
            "launch contract is bootstrap-ready and no Rust daemon is currently supervising the lane"
                .to_string(),
        ),
        PaperServiceState::ResumeReady => (
            PaperSupervisorAction::Start,
            "launch contract is resumable and the Rust daemon should be started with the current lane contract"
                .to_string(),
        ),
        PaperServiceState::CaughtUpIdle => (
            PaperSupervisorAction::Start,
            "launch contract is caught up and the Rust daemon may start in idle follow mode"
                .to_string(),
        ),
        PaperServiceState::Running => (
            PaperSupervisorAction::Monitor,
            "daemon is running and matches the current launch contract".to_string(),
        ),
        PaperServiceState::RestartRequired => (
            PaperSupervisorAction::Restart,
            format!(
                "running daemon no longer matches the current launch contract: {}",
                status.mismatch_reasons.join("; ")
            ),
        ),
        PaperServiceState::StatusStale => (
            PaperSupervisorAction::Restart,
            match status.status_age_ms.zip(status.stale_after_ms) {
                Some((status_age_ms, stale_after_ms)) => format!(
                    "daemon status is stale: age {}ms exceeds the configured threshold {}ms",
                    status_age_ms, stale_after_ms
                ),
                None => "daemon status is stale and should be refreshed via a supervised restart"
                    .to_string(),
            },
        ),
        PaperServiceState::Stopped => {
            if status.manifest.resume.launch_ready {
                (
                    PaperSupervisorAction::Start,
                    "persisted daemon status is stopped; start the lane with the current launch contract"
                        .to_string(),
                )
            } else {
                (
                    PaperSupervisorAction::Hold,
                    "persisted daemon status is stopped and the current launch contract is not ready to restart"
                        .to_string(),
                )
            }
        }
    }
}

fn build_apply_plan(
    preview: &PaperServiceReport,
    requested_action: PaperServiceApplyRequestedAction,
    lock_owner_pid: Option<u32>,
) -> Result<PaperServiceApplyPlan> {
    let rust_owned_running_pid = rust_owned_running_pid(preview, lock_owner_pid)?;

    match requested_action {
        PaperServiceApplyRequestedAction::Auto => match preview.desired_action {
            PaperSupervisorAction::Hold => Ok(PaperServiceApplyPlan {
                applied_action: PaperServiceApplyExecutedAction::Noop,
                action_reason: format!(
                    "paper service apply kept the lane on hold: {}",
                    preview.action_reason
                ),
                stop_pid: None,
            }),
            PaperSupervisorAction::Monitor => {
                if rust_owned_running_pid.is_some() {
                    Ok(PaperServiceApplyPlan {
                        applied_action: PaperServiceApplyExecutedAction::Noop,
                        action_reason:
                            "paper service apply confirmed that the Rust daemon already owns this healthy lane"
                                .to_string(),
                        stop_pid: None,
                    })
                } else {
                    build_start_plan(
                        preview,
                        lock_owner_pid,
                        "paper service apply detected an orphaned Rust status contract and will start a fresh daemon".to_string(),
                    )
                }
            }
            PaperSupervisorAction::Start => build_start_plan(
                preview,
                lock_owner_pid,
                format!(
                    "paper service apply is enacting the read-only start recommendation: {}",
                    preview.action_reason
                ),
            ),
            PaperSupervisorAction::Restart => {
                if let Some(pid) = rust_owned_running_pid {
                    Ok(PaperServiceApplyPlan {
                        applied_action: PaperServiceApplyExecutedAction::Restart,
                        action_reason: format!(
                            "paper service apply will restart the current Rust daemon owner (pid {}) before relaunching the lane",
                            pid
                        ),
                        stop_pid: Some(pid),
                    })
                } else {
                    build_start_plan(
                        preview,
                        lock_owner_pid,
                        "paper service apply collapsed restart into start because the lane lock is currently free"
                            .to_string(),
                    )
                }
            }
        },
        PaperServiceApplyRequestedAction::Start => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(PaperServiceApplyPlan {
                    applied_action: PaperServiceApplyExecutedAction::Noop,
                    action_reason: format!(
                        "paper service apply skipped a duplicate start because the Rust daemon already owns the lane (pid {})",
                        pid
                    ),
                    stop_pid: None,
                })
            } else {
                build_start_plan(
                    preview,
                    lock_owner_pid,
                    "paper service apply will start the lane with the current Rust daemon contract"
                        .to_string(),
                )
            }
        }
        PaperServiceApplyRequestedAction::Resume => {
            if preview.status.service_state != PaperServiceState::Stopped
                || !preview.status.manifest.resume.launch_ready
            {
                anyhow::bail!(
                    "paper service apply --action resume requires a stopped, launch-ready Rust lane"
                );
            }
            build_start_plan(
                preview,
                lock_owner_pid,
                "paper service apply will resume the stopped Rust lane with the current launch contract"
                    .to_string(),
            )
        }
        PaperServiceApplyRequestedAction::Restart => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(PaperServiceApplyPlan {
                    applied_action: PaperServiceApplyExecutedAction::Restart,
                    action_reason: format!(
                        "paper service apply will restart the current Rust daemon owner (pid {})",
                        pid
                    ),
                    stop_pid: Some(pid),
                })
            } else {
                build_start_plan(
                    preview,
                    lock_owner_pid,
                    "paper service apply collapsed the requested restart into start because the lane lock is free"
                        .to_string(),
                )
            }
        }
        PaperServiceApplyRequestedAction::Stop => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(PaperServiceApplyPlan {
                    applied_action: PaperServiceApplyExecutedAction::Stop,
                    action_reason: format!(
                        "paper service apply will stop the current Rust daemon owner (pid {})",
                        pid
                    ),
                    stop_pid: Some(pid),
                })
            } else if lock_owner_pid.is_none() {
                Ok(PaperServiceApplyPlan {
                    applied_action: PaperServiceApplyExecutedAction::Noop,
                    action_reason: "paper service apply found no active Rust daemon owner to stop"
                        .to_string(),
                    stop_pid: None,
                })
            } else {
                anyhow::bail!(
                    "paper service apply refused to stop the lane because the lock owner could not be proven to be the current Rust daemon status owner"
                );
            }
        }
    }
}

fn build_start_plan(
    preview: &PaperServiceReport,
    lock_owner_pid: Option<u32>,
    action_reason: String,
) -> Result<PaperServiceApplyPlan> {
    if !preview.status.manifest.resume.launch_ready {
        anyhow::bail!(
            "paper service apply cannot start the lane because the current launch contract is not launch-ready"
        );
    }
    if let Some(lock_owner_pid) = lock_owner_pid {
        anyhow::bail!(
            "paper service apply refused to start the lane because the lock is already held by pid {}",
            lock_owner_pid
        );
    }
    Ok(PaperServiceApplyPlan {
        applied_action: PaperServiceApplyExecutedAction::Start,
        action_reason,
        stop_pid: None,
    })
}

fn rust_owned_running_pid(
    preview: &PaperServiceReport,
    lock_owner_pid: Option<u32>,
) -> Result<Option<u32>> {
    let Some(status) = preview.status.daemon_status.as_ref() else {
        if let Some(lock_owner_pid) = lock_owner_pid {
            anyhow::bail!(
                "paper service apply found lock owner pid {} but no Rust daemon status file at {}",
                lock_owner_pid,
                preview.status_path
            );
        }
        return Ok(None);
    };

    if status.lock_path != preview.lock_path || status.status_path != preview.status_path {
        anyhow::bail!(
            "paper service apply refused to supervise the lane because the persisted Rust status paths no longer match the current launch contract"
        );
    }

    if !status.running {
        if let Some(lock_owner_pid) = lock_owner_pid {
            anyhow::bail!(
                "paper service apply found lock owner pid {} but the Rust status contract reports running=false",
                lock_owner_pid
            );
        }
        return Ok(None);
    }

    match lock_owner_pid {
        Some(lock_owner_pid) if lock_owner_pid == status.pid => Ok(Some(status.pid)),
        Some(lock_owner_pid) => anyhow::bail!(
            "paper service apply found a lock owner pid mismatch (status={} lock={})",
            status.pid,
            lock_owner_pid
        ),
        None => {
            if status_pid_is_alive(status.pid)? {
                anyhow::bail!(
                    "paper service apply found a running Rust daemon pid {} but could not prove lock ownership through {}",
                    status.pid,
                    preview.lock_path
                );
            }
            Ok(None)
        }
    }
}

fn running_status_pid(preview: &PaperServiceReport) -> Option<u32> {
    preview
        .status
        .daemon_status
        .as_ref()
        .and_then(|status| status.running.then_some(status.pid))
}

#[cfg(unix)]
fn start_daemon(
    daemon_command: &[String],
    lock_path: &Path,
    status_path: &Path,
    start_wait: std::time::Duration,
    poll_interval: std::time::Duration,
) -> Result<u32> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Instant;

    let program = resolve_runtime_program(daemon_command)?;
    let mut command = Command::new(&program);
    command.args(&daemon_command[1..]);
    command.stdin(Stdio::null());
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    // Safety: setsid(2) is invoked in the short-lived forked child before exec so the supervised
    // daemon can detach from the caller's process group without touching shared Rust state.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().with_context(|| {
        format!(
            "paper service apply failed to spawn the Rust paper daemon: {}",
            program.display()
        )
    })?;
    let child_pid = child.id();
    let deadline = Instant::now() + start_wait;
    loop {
        if let Some(exit_status) = child
            .try_wait()
            .context("failed to read the spawned paper daemon exit status")?
        {
            anyhow::bail!(
                "paper service apply spawned the Rust paper daemon (pid {}) but it exited before publishing a running status: {}",
                child_pid,
                exit_status
            );
        }

        if let Some(status) = paper_daemon::load_status_file(status_path)? {
            let lock_owner_pid = paper_daemon::probe_lock_owner(lock_path)?;
            if status.running && status.pid == child_pid && lock_owner_pid == Some(child_pid) {
                if child
                    .try_wait()
                    .context("failed to read the spawned paper daemon exit status")?
                    .is_some()
                {
                    anyhow::bail!(
                        "paper service apply observed a transient running status for pid {}, but the daemon exited before supervision completed",
                        child_pid
                    );
                }
                return Ok(child_pid);
            }
        }

        if Instant::now() >= deadline {
            let _ = send_sigterm(child_pid);
            let _ =
                wait_for_process_exit(child_pid, poll_interval, std::time::Duration::from_secs(2));
            anyhow::bail!(
                "paper service apply timed out waiting for the Rust paper daemon to publish a running status at {}",
                status_path.display()
            );
        }

        thread::sleep(poll_interval);
    }
}

#[cfg(not(unix))]
fn start_daemon(
    _daemon_command: &[String],
    _lock_path: &Path,
    _status_path: &Path,
    _start_wait: std::time::Duration,
    _poll_interval: std::time::Duration,
) -> Result<u32> {
    anyhow::bail!("paper service apply is currently supported on Unix-like hosts only")
}

#[cfg(unix)]
fn stop_daemon(
    pid: u32,
    lock_path: &Path,
    status_path: &Path,
    stop_wait: std::time::Duration,
    poll_interval: std::time::Duration,
) -> Result<()> {
    use std::thread;
    use std::time::Instant;

    if process_exists(pid)? {
        send_sigterm(pid)?;
    }

    let deadline = Instant::now() + stop_wait;
    loop {
        let lock_owner_pid = paper_daemon::probe_lock_owner(lock_path)?;
        let status_written = paper_daemon::load_status_file(status_path)?.is_some_and(|status| {
            !status.running && status.pid == pid && status.stopped_at_ms.is_some()
        });
        let pid_exited = !process_exists(pid)?;

        if lock_owner_pid.is_none() && (status_written || pid_exited) {
            return Ok(());
        }

        if Instant::now() >= deadline {
            anyhow::bail!(
                "paper service apply timed out waiting for pid {} to stop and publish a stopped status contract",
                pid
            );
        }

        thread::sleep(poll_interval);
    }
}

#[cfg(not(unix))]
fn stop_daemon(
    _pid: u32,
    _lock_path: &Path,
    _status_path: &Path,
    _stop_wait: std::time::Duration,
    _poll_interval: std::time::Duration,
) -> Result<()> {
    anyhow::bail!("paper service apply is currently supported on Unix-like hosts only")
}

#[cfg(unix)]
fn resolve_runtime_program(daemon_command: &[String]) -> Result<PathBuf> {
    let Some(program) = daemon_command.first() else {
        anyhow::bail!("paper service apply requires a non-empty daemon command");
    };
    if Path::new(program).components().count() > 1 {
        return Ok(PathBuf::from(program));
    }
    if let Some(runtime_bin) = std::env::var_os("AI_QUANT_RUNTIME_BIN") {
        return Ok(PathBuf::from(runtime_bin));
    }
    if program == "aiq-runtime" {
        return std::env::current_exe()
            .context("failed to resolve the current aiq-runtime executable for supervision");
    }
    Ok(PathBuf::from(program))
}

#[cfg(unix)]
fn process_exists(pid: u32) -> Result<bool> {
    let rc = unsafe { libc::kill(pid as i32, 0) };
    if rc == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(code) if code == libc::ESRCH => Ok(false),
        Some(code) if code == libc::EPERM => Ok(true),
        _ => Err(err).with_context(|| format!("failed to probe pid {}", pid)),
    }
}

#[cfg(unix)]
fn status_pid_is_alive(pid: u32) -> Result<bool> {
    process_exists(pid)
}

#[cfg(not(unix))]
fn status_pid_is_alive(_pid: u32) -> Result<bool> {
    Ok(false)
}

#[cfg(unix)]
fn send_sigterm(pid: u32) -> Result<()> {
    let rc = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(code) if code == libc::ESRCH => Ok(()),
        _ => Err(err).with_context(|| format!("failed to send SIGTERM to pid {}", pid)),
    }
}

#[cfg(unix)]
fn wait_for_process_exit(
    pid: u32,
    poll_interval: std::time::Duration,
    timeout: std::time::Duration,
) -> Result<()> {
    use std::thread;
    use std::time::Instant;

    let deadline = Instant::now() + timeout;
    loop {
        if !process_exists(pid)? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for pid {} to exit", pid);
        }
        thread::sleep(poll_interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use tempfile::tempdir;

    use crate::paper_daemon::{
        PaperDaemonStatusRuntimeBootstrap, PaperDaemonStatusRuntimePipeline,
        PaperDaemonStatusSnapshot,
    };
    use crate::paper_manifest::{
        PaperManifestLaunchState, PaperManifestReport, PaperManifestResumeState,
    };

    fn build_manifest_report(
        service_state: PaperServiceState,
        launch_ready: bool,
    ) -> PaperStatusReport {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        std::fs::write(&config_path, "global:\n  engine:\n    interval: 30m\n").unwrap();
        let config =
            bt_core::config::load_config_checked(config_path.to_str().unwrap(), None, false)
                .unwrap();
        let runtime_bootstrap =
            build_bootstrap(&config, RuntimeMode::Paper, Some("production")).unwrap();

        let launch_state = match service_state {
            PaperServiceState::IdleNoSymbols => PaperManifestLaunchState::IdleNoSymbols,
            PaperServiceState::BootstrapReady => PaperManifestLaunchState::BootstrapReady,
            PaperServiceState::ResumeReady => PaperManifestLaunchState::ResumeReady,
            PaperServiceState::CaughtUpIdle => PaperManifestLaunchState::CaughtUpIdle,
            _ if launch_ready => PaperManifestLaunchState::ResumeReady,
            _ => PaperManifestLaunchState::BootstrapRequired,
        };

        let manifest = PaperManifestReport {
            ok: true,
            runtime_bootstrap,
            base_config_path: config_path.display().to_string(),
            config_path: config_path.display().to_string(),
            active_yaml_path: config_path.display().to_string(),
            effective_yaml_path: config_path.display().to_string(),
            paper_db: dir.path().join("paper.db").display().to_string(),
            paper_db_exists: true,
            candles_db: dir.path().join("candles.db").display().to_string(),
            candles_db_exists: true,
            interval: "30m".to_string(),
            lookback_bars: 200,
            symbols: vec!["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC".to_string(),
            start_step_close_ts_ms: None,
            lock_path: dir.path().join("paper.lock").display().to_string(),
            status_path: dir.path().join("paper.status.json").display().to_string(),
            instance_tag: None,
            lane: None,
            service_name: None,
            promoted_role: None,
            strategy_mode: None,
            promoted_config_path: None,
            strategy_mode_source: None,
            strategy_overrides_sha1: "a".repeat(64),
            config_id: "a".repeat(64),
            resume: PaperManifestResumeState {
                launch_ready,
                launch_state,
                active_symbols: vec!["ETH".to_string()],
                last_applied_step_close_ts_ms: Some(1_773_424_200_000),
                latest_common_close_ts_ms: Some(1_773_426_000_000),
                next_due_step_close_ts_ms: Some(1_773_426_000_000),
            },
            warnings: Vec::new(),
            daemon_command: vec![
                "aiq-runtime".to_string(),
                "paper".to_string(),
                "daemon".to_string(),
            ],
        };

        PaperStatusReport {
            ok: true,
            service_state,
            status_file_present: false,
            status_age_ms: Some(120_000),
            stale_after_ms: Some(60_000),
            contract_matches_status: true,
            mismatch_reasons: Vec::new(),
            manifest,
            daemon_status: None,
            warnings: Vec::new(),
        }
    }

    fn running_status_report(service_state: PaperServiceState, pid: u32) -> PaperServiceReport {
        let mut status = build_manifest_report(service_state, true);
        status.status_file_present = true;
        status.daemon_status = Some(PaperDaemonStatusSnapshot {
            ok: true,
            running: true,
            pid,
            config_path: status.manifest.config_path.clone(),
            config_id: status.manifest.config_id.clone(),
            paper_db: status.manifest.paper_db.clone(),
            candles_db: status.manifest.candles_db.clone(),
            lock_path: status.manifest.lock_path.clone(),
            status_path: status.manifest.status_path.clone(),
            started_at_ms: 1_773_424_200_000,
            updated_at_ms: 1_773_424_260_000,
            stopped_at_ms: None,
            stop_requested: false,
            dry_run: false,
            runtime_bootstrap: PaperDaemonStatusRuntimeBootstrap {
                config_fingerprint: status.manifest.runtime_bootstrap.config_fingerprint.clone(),
                pipeline: PaperDaemonStatusRuntimePipeline {
                    profile: status.manifest.runtime_bootstrap.pipeline.profile.clone(),
                },
            },
            btc_symbol: "BTC".to_string(),
            lookback_bars: status.manifest.lookback_bars,
            explicit_symbols: status.manifest.symbols.clone(),
            watch_symbols_file: status.manifest.watch_symbols_file,
            symbols_file: status.manifest.symbols_file.clone(),
            start_step_close_ts_ms: status.manifest.start_step_close_ts_ms,
            manifest_symbols: status.manifest.resume.active_symbols.clone(),
            last_active_symbols: status.manifest.resume.active_symbols.clone(),
            manifest_reload_count: 0,
            manifest_reload_failure_count: 0,
            initial_last_applied_step_close_ts_ms: status
                .manifest
                .resume
                .last_applied_step_close_ts_ms,
            latest_common_close_ts_ms: status.manifest.resume.latest_common_close_ts_ms,
            next_due_step_close_ts_ms: status.manifest.resume.next_due_step_close_ts_ms,
            executed_steps: 0,
            idle_polls: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
        });
        let (desired_action, action_reason) = derive_action(&status);
        PaperServiceReport {
            ok: true,
            desired_action,
            action_reason,
            daemon_command: status.manifest.daemon_command.clone(),
            lock_path: status.manifest.lock_path.clone(),
            status_path: status.manifest.status_path.clone(),
            status,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn service_reports_monitor_for_running_lane() {
        let report = build_manifest_report(PaperServiceState::Running, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Monitor);
        assert!(reason.contains("matches the current launch contract"));
    }

    #[test]
    fn service_reports_restart_for_restart_required_lane() {
        let mut report = build_manifest_report(PaperServiceState::RestartRequired, true);
        report.mismatch_reasons = vec!["config fingerprint mismatch".to_string()];
        report.contract_matches_status = false;

        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Restart);
        assert!(reason.contains("config fingerprint mismatch"));
    }

    #[test]
    fn service_reports_restart_for_stale_lane() {
        let report = build_manifest_report(PaperServiceState::StatusStale, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Restart);
        assert!(reason.contains("stale"));
    }

    #[test]
    fn service_reports_hold_for_bootstrap_required_lane() {
        let report = build_manifest_report(PaperServiceState::BootstrapRequired, false);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Hold);
        assert!(reason.contains("start-step-close-ts-ms"));
    }

    #[test]
    fn service_reports_start_for_launch_ready_idle_watch_lane() {
        let report = build_manifest_report(PaperServiceState::IdleNoSymbols, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("wait for a watched symbols file"));
    }

    #[test]
    fn service_reports_hold_for_non_ready_idle_lane() {
        let report = build_manifest_report(PaperServiceState::IdleNoSymbols, false);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Hold);
        assert!(reason.contains("no active symbols"));
    }

    #[test]
    fn service_reports_start_for_stopped_launch_ready_lane() {
        let report = build_manifest_report(PaperServiceState::Stopped, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("stopped"));
    }

    #[test]
    fn service_reports_start_for_bootstrap_ready_lane() {
        let report = build_manifest_report(PaperServiceState::BootstrapReady, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("bootstrap-ready"));
    }

    #[test]
    fn service_reports_start_for_resume_ready_lane() {
        let report = build_manifest_report(PaperServiceState::ResumeReady, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("resumable"));
    }

    #[test]
    fn service_reports_start_for_caught_up_idle_lane() {
        let report = build_manifest_report(PaperServiceState::CaughtUpIdle, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("caught up"));
    }

    #[test]
    fn service_reports_hold_for_stopped_non_ready_lane() {
        let report = build_manifest_report(PaperServiceState::Stopped, false);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Hold);
        assert!(reason.contains("not ready to restart"));
    }

    #[test]
    fn apply_auto_collapses_restart_to_start_when_lock_is_free() {
        let mut preview = running_status_report(PaperServiceState::RestartRequired, 2_000_000_000);
        preview.desired_action = PaperSupervisorAction::Restart;
        preview.status.mismatch_reasons = vec!["config fingerprint mismatch".to_string()];
        preview.status.contract_matches_status = false;

        let plan =
            build_apply_plan(&preview, PaperServiceApplyRequestedAction::Auto, None).unwrap();

        assert_eq!(plan.applied_action, PaperServiceApplyExecutedAction::Start);
        assert!(plan.action_reason.contains("collapsed restart into start"));
    }

    #[test]
    fn apply_resume_requires_stopped_launch_ready_lane() {
        let preview = PaperServiceReport {
            ok: true,
            desired_action: PaperSupervisorAction::Start,
            action_reason: "launch-ready".to_string(),
            daemon_command: vec![
                "aiq-runtime".to_string(),
                "paper".to_string(),
                "daemon".to_string(),
            ],
            lock_path: "/tmp/paper.lock".to_string(),
            status_path: "/tmp/paper.status.json".to_string(),
            status: build_manifest_report(PaperServiceState::ResumeReady, true),
            warnings: Vec::new(),
        };

        let err =
            build_apply_plan(&preview, PaperServiceApplyRequestedAction::Resume, None).unwrap_err();

        assert!(err.to_string().contains("stopped, launch-ready Rust lane"));
    }

    #[test]
    fn apply_stop_requires_rust_lock_ownership_proof() {
        let preview = running_status_report(PaperServiceState::Running, 2_000_000_000);
        let err = build_apply_plan(
            &preview,
            PaperServiceApplyRequestedAction::Stop,
            Some(2_000_000_001),
        )
        .unwrap_err();

        assert!(err.to_string().contains("lock owner pid mismatch"));
    }

    #[test]
    fn expected_config_id_guard_rejects_mismatch() {
        let err = verify_expected_config_id(Some("cfg-wrong"), "cfg-actual").unwrap_err();
        assert!(err
            .to_string()
            .contains("expected config_id cfg-wrong did not match resolved"));
    }
}
