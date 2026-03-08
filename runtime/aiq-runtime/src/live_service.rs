use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::live_daemon;
use crate::live_status::{self, LiveServiceState, LiveStatusInput, LiveStatusReport};

#[derive(Clone, Copy)]
pub struct LiveServiceInput<'a> {
    pub config: Option<&'a Path>,
    pub project_dir: Option<&'a Path>,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub market_db: Option<&'a Path>,
    pub candles_db: Option<&'a Path>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub secrets_path: Option<&'a Path>,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub lookback_bars: Option<usize>,
    pub stale_after_ms: Option<i64>,
}

#[derive(Clone, Copy)]
pub struct LiveServiceApplyInput<'a> {
    pub service: LiveServiceInput<'a>,
    pub requested_action: LiveServiceApplyRequestedAction,
    pub start_wait_ms: u64,
    pub stop_wait_ms: u64,
    pub poll_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveSupervisorAction {
    Hold,
    Start,
    Restart,
    Monitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveServiceApplyRequestedAction {
    Auto,
    Start,
    Restart,
    Stop,
    Resume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveServiceApplyExecutedAction {
    Noop,
    Start,
    Restart,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveServiceReport {
    pub ok: bool,
    pub desired_action: LiveSupervisorAction,
    pub action_reason: String,
    pub daemon_command: Vec<String>,
    pub lock_path: String,
    pub status_path: String,
    pub status: LiveStatusReport,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveServiceApplyReport {
    pub ok: bool,
    pub requested_action: LiveServiceApplyRequestedAction,
    pub applied_action: LiveServiceApplyExecutedAction,
    pub action_reason: String,
    pub lock_owner_pid: Option<u32>,
    pub previous_pid: Option<u32>,
    pub spawned_pid: Option<u32>,
    pub preview: LiveServiceReport,
    pub final_service: LiveServiceReport,
}

#[derive(Debug, Clone)]
struct LiveServiceApplyPlan {
    applied_action: LiveServiceApplyExecutedAction,
    action_reason: String,
    stop_pid: Option<u32>,
}

pub fn build_service(input: LiveServiceInput<'_>) -> Result<LiveServiceReport> {
    let status = live_status::build_status(LiveStatusInput {
        config: input.config,
        project_dir: input.project_dir,
        profile: input.profile,
        db: input.db,
        market_db: input.market_db,
        candles_db: input.candles_db,
        symbols: input.symbols,
        symbols_file: input.symbols_file,
        btc_symbol: input.btc_symbol,
        secrets_path: input.secrets_path,
        lock_path: input.lock_path,
        status_path: input.status_path,
        lookback_bars: input.lookback_bars,
        stale_after_ms: input.stale_after_ms,
    })?;
    let (desired_action, action_reason) = derive_action(&status);
    let warnings = status.warnings.clone();

    Ok(LiveServiceReport {
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

pub fn apply_service(input: LiveServiceApplyInput<'_>) -> Result<LiveServiceApplyReport> {
    let preview = build_service(input.service)?;
    let lock_path = PathBuf::from(&preview.lock_path);
    let status_path = PathBuf::from(&preview.status_path);
    let lock_owner_pid = live_daemon::probe_lock_owner(&lock_path)?;
    let plan = build_apply_plan(&preview, input.requested_action, lock_owner_pid)?;
    let previous_pid = running_status_pid(&preview);
    let poll_interval = std::time::Duration::from_millis(input.poll_ms.max(10));
    let mut spawned_pid = None;

    match plan.applied_action {
        LiveServiceApplyExecutedAction::Noop => {}
        LiveServiceApplyExecutedAction::Start => {
            spawned_pid = Some(start_daemon(
                &preview.daemon_command,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.start_wait_ms.max(100)),
                poll_interval,
            )?);
        }
        LiveServiceApplyExecutedAction::Restart => {
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
            spawned_pid = Some(start_daemon(
                &preview.daemon_command,
                &lock_path,
                &status_path,
                std::time::Duration::from_millis(input.start_wait_ms.max(100)),
                poll_interval,
            )?);
        }
        LiveServiceApplyExecutedAction::Stop => {
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

    Ok(LiveServiceApplyReport {
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

fn derive_action(status: &LiveStatusReport) -> (LiveSupervisorAction, String) {
    match status.service_state {
        LiveServiceState::Blocked => (
            LiveSupervisorAction::Hold,
            "current launch contract is blocked and should not be supervised yet".to_string(),
        ),
        LiveServiceState::Ready => (
            LiveSupervisorAction::Start,
            "launch contract is ready and no Rust live daemon is currently supervising the lane"
                .to_string(),
        ),
        LiveServiceState::Running => (
            LiveSupervisorAction::Monitor,
            "daemon is running and matches the current launch contract".to_string(),
        ),
        LiveServiceState::RestartRequired => (
            LiveSupervisorAction::Restart,
            format!(
                "running daemon no longer matches the current launch contract: {}",
                status.mismatch_reasons.join("; ")
            ),
        ),
        LiveServiceState::StatusStale => (
            LiveSupervisorAction::Restart,
            match status.status_age_ms.zip(status.stale_after_ms) {
                Some((status_age_ms, stale_after_ms)) => format!(
                    "daemon status is stale: age {}ms exceeds the configured threshold {}ms",
                    status_age_ms, stale_after_ms
                ),
                None => "daemon status is stale and should be refreshed via a supervised restart"
                    .to_string(),
            },
        ),
        LiveServiceState::Stopped => {
            if status.manifest.launch_state == crate::live_manifest::LiveManifestLaunchState::Ready
            {
                (
                    LiveSupervisorAction::Start,
                    "persisted daemon status is stopped; start the lane with the current launch contract"
                        .to_string(),
                )
            } else {
                (
                    LiveSupervisorAction::Hold,
                    "persisted daemon status is stopped and the current launch contract is not ready to restart"
                        .to_string(),
                )
            }
        }
    }
}

fn build_apply_plan(
    preview: &LiveServiceReport,
    requested_action: LiveServiceApplyRequestedAction,
    lock_owner_pid: Option<u32>,
) -> Result<LiveServiceApplyPlan> {
    let rust_owned_running_pid = rust_owned_running_pid(preview, lock_owner_pid)?;

    match requested_action {
        LiveServiceApplyRequestedAction::Auto => match preview.desired_action {
            LiveSupervisorAction::Hold => Ok(LiveServiceApplyPlan {
                applied_action: LiveServiceApplyExecutedAction::Noop,
                action_reason: format!(
                    "live service apply kept the lane on hold: {}",
                    preview.action_reason
                ),
                stop_pid: None,
            }),
            LiveSupervisorAction::Monitor => {
                if rust_owned_running_pid.is_some() {
                    Ok(LiveServiceApplyPlan {
                        applied_action: LiveServiceApplyExecutedAction::Noop,
                        action_reason:
                            "live service apply confirmed that the Rust daemon already owns this healthy lane"
                                .to_string(),
                        stop_pid: None,
                    })
                } else {
                    build_start_plan(
                        preview,
                        lock_owner_pid,
                        "live service apply detected an orphaned Rust status contract and will start a fresh daemon".to_string(),
                    )
                }
            }
            LiveSupervisorAction::Start => build_start_plan(
                preview,
                lock_owner_pid,
                format!(
                    "live service apply is enacting the read-only start recommendation: {}",
                    preview.action_reason
                ),
            ),
            LiveSupervisorAction::Restart => {
                if let Some(pid) = rust_owned_running_pid {
                    Ok(LiveServiceApplyPlan {
                        applied_action: LiveServiceApplyExecutedAction::Restart,
                        action_reason: format!(
                            "live service apply will restart the current Rust daemon owner (pid {}) before relaunching the lane",
                            pid
                        ),
                        stop_pid: Some(pid),
                    })
                } else {
                    build_start_plan(
                        preview,
                        lock_owner_pid,
                        "live service apply collapsed restart into start because the lane lock is currently free"
                            .to_string(),
                    )
                }
            }
        },
        LiveServiceApplyRequestedAction::Start => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(LiveServiceApplyPlan {
                    applied_action: LiveServiceApplyExecutedAction::Noop,
                    action_reason: format!(
                        "live service apply skipped a duplicate start because the Rust daemon already owns the lane (pid {})",
                        pid
                    ),
                    stop_pid: None,
                })
            } else {
                build_start_plan(
                    preview,
                    lock_owner_pid,
                    "live service apply will start the lane with the current Rust daemon contract"
                        .to_string(),
                )
            }
        }
        LiveServiceApplyRequestedAction::Resume => {
            if preview.status.service_state != LiveServiceState::Stopped
                || preview.status.manifest.launch_state
                    != crate::live_manifest::LiveManifestLaunchState::Ready
            {
                anyhow::bail!(
                    "live service apply --action resume requires a stopped, launch-ready Rust lane"
                );
            }
            build_start_plan(
                preview,
                lock_owner_pid,
                "live service apply will resume the stopped Rust lane with the current launch contract"
                    .to_string(),
            )
        }
        LiveServiceApplyRequestedAction::Restart => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(LiveServiceApplyPlan {
                    applied_action: LiveServiceApplyExecutedAction::Restart,
                    action_reason: format!(
                        "live service apply will restart the current Rust daemon owner (pid {})",
                        pid
                    ),
                    stop_pid: Some(pid),
                })
            } else {
                build_start_plan(
                    preview,
                    lock_owner_pid,
                    "live service apply collapsed the requested restart into start because the lane lock is free"
                        .to_string(),
                )
            }
        }
        LiveServiceApplyRequestedAction::Stop => {
            if let Some(pid) = rust_owned_running_pid {
                Ok(LiveServiceApplyPlan {
                    applied_action: LiveServiceApplyExecutedAction::Stop,
                    action_reason: format!(
                        "live service apply will stop the current Rust daemon owner (pid {})",
                        pid
                    ),
                    stop_pid: Some(pid),
                })
            } else if lock_owner_pid.is_none() {
                Ok(LiveServiceApplyPlan {
                    applied_action: LiveServiceApplyExecutedAction::Noop,
                    action_reason: "live service apply found no active Rust daemon owner to stop"
                        .to_string(),
                    stop_pid: None,
                })
            } else {
                anyhow::bail!(
                    "live service apply refused to stop the lane because the lock owner could not be proven to be the current Rust daemon status owner"
                );
            }
        }
    }
}

fn build_start_plan(
    preview: &LiveServiceReport,
    lock_owner_pid: Option<u32>,
    action_reason: String,
) -> Result<LiveServiceApplyPlan> {
    if preview.status.manifest.launch_state != crate::live_manifest::LiveManifestLaunchState::Ready
    {
        anyhow::bail!(
            "live service apply cannot start the lane because the current launch contract is not launch-ready"
        );
    }
    if let Some(lock_owner_pid) = lock_owner_pid {
        anyhow::bail!(
            "live service apply refused to start the lane because the lock is already held by pid {}",
            lock_owner_pid
        );
    }
    Ok(LiveServiceApplyPlan {
        applied_action: LiveServiceApplyExecutedAction::Start,
        action_reason,
        stop_pid: None,
    })
}

fn rust_owned_running_pid(
    preview: &LiveServiceReport,
    lock_owner_pid: Option<u32>,
) -> Result<Option<u32>> {
    let Some(status) = preview.status.daemon_status.as_ref() else {
        if let Some(lock_owner_pid) = lock_owner_pid {
            anyhow::bail!(
                "live service apply found lock owner pid {} but no Rust daemon status file at {}",
                lock_owner_pid,
                preview.status_path
            );
        }
        return Ok(None);
    };

    if status.lock_path != preview.lock_path || status.status_path != preview.status_path {
        anyhow::bail!(
            "live service apply refused to supervise the lane because the persisted Rust status paths no longer match the current launch contract"
        );
    }

    if !status.running {
        if let Some(lock_owner_pid) = lock_owner_pid {
            anyhow::bail!(
                "live service apply found lock owner pid {} but the Rust status contract reports running=false",
                lock_owner_pid
            );
        }
        return Ok(None);
    }

    match lock_owner_pid {
        Some(lock_owner_pid) if lock_owner_pid == status.pid => Ok(Some(status.pid)),
        Some(lock_owner_pid) => anyhow::bail!(
            "live service apply found a lock owner pid mismatch (status={} lock={})",
            status.pid,
            lock_owner_pid
        ),
        None => {
            if status_pid_is_alive(status.pid)? {
                anyhow::bail!(
                    "live service apply found a running Rust daemon pid {} but could not prove lock ownership through {}",
                    status.pid,
                    preview.lock_path
                );
            }
            Ok(None)
        }
    }
}

fn running_status_pid(preview: &LiveServiceReport) -> Option<u32> {
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
            "live service apply failed to spawn the Rust live daemon: {}",
            program.display()
        )
    })?;
    let child_pid = child.id();
    let deadline = Instant::now() + start_wait;
    loop {
        if let Some(exit_status) = child
            .try_wait()
            .context("failed to read the spawned live daemon exit status")?
        {
            anyhow::bail!(
                "live service apply spawned the Rust live daemon (pid {}) but it exited before publishing a running status: {}",
                child_pid,
                exit_status
            );
        }

        if let Some(status) = live_daemon::load_status_file(status_path)? {
            let lock_owner_pid = live_daemon::probe_lock_owner(lock_path)?;
            if status.running && status.pid == child_pid && lock_owner_pid == Some(child_pid) {
                if child
                    .try_wait()
                    .context("failed to read the spawned live daemon exit status")?
                    .is_some()
                {
                    anyhow::bail!(
                        "live service apply observed a transient running status for pid {}, but the daemon exited before supervision completed",
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
                "live service apply timed out waiting for the Rust live daemon to publish a running status at {}",
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
    anyhow::bail!("live service apply is currently supported on Unix-like hosts only")
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
        let lock_owner_pid = live_daemon::probe_lock_owner(lock_path)?;
        let status_written = live_daemon::load_status_file(status_path)?.is_some_and(|status| {
            !status.running && status.pid == pid && status.stopped_at_ms.is_some()
        });
        let pid_exited = !process_exists(pid)?;

        if lock_owner_pid.is_none() && (status_written || pid_exited) {
            return Ok(());
        }

        if Instant::now() >= deadline {
            anyhow::bail!(
                "live service apply timed out waiting for pid {} to stop and publish a stopped status contract",
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
    anyhow::bail!("live service apply is currently supported on Unix-like hosts only")
}

#[cfg(unix)]
fn resolve_runtime_program(daemon_command: &[String]) -> Result<PathBuf> {
    let Some(program) = daemon_command.first() else {
        anyhow::bail!("live service apply requires a non-empty daemon command");
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
    use crate::test_support::{env_lock, EnvGuard};
    use std::fs;
    use tempfile::tempdir;

    fn write_config(path: &Path, interval: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            format!("global:\n  engine:\n    interval: {}\n", interval),
        )
        .unwrap();
    }

    #[test]
    fn build_service_recommends_start_for_ready_unsupervised_lane() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let config_path = dir
            .path()
            .join("config")
            .join("strategy_overrides.live.yaml");
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            ("AI_QUANT_STRATEGY_YAML", None),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_MARKET_DB_PATH", None),
            ("AI_QUANT_CANDLES_DB_DIR", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_EVENT_LOG_DIR", None),
            ("AI_QUANT_INSTANCE_TAG", None),
            ("AI_QUANT_LIVE_SERVICE_NAME", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_SECRETS_PATH", None),
            ("AI_QUANT_LIVE_ENABLE", Some("1")),
            (
                "AI_QUANT_LIVE_CONFIRM",
                Some("I_UNDERSTAND_THIS_CAN_LOSE_MONEY"),
            ),
        ]);

        let report = build_service(LiveServiceInput {
            config: None,
            project_dir: Some(dir.path()),
            profile: None,
            db: None,
            market_db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            btc_symbol: "BTC",
            secrets_path: None,
            lock_path: None,
            status_path: None,
            lookback_bars: Some(200),
            stale_after_ms: Some(30_000),
        })
        .unwrap();

        assert_eq!(report.status.service_state, LiveServiceState::Ready);
        assert_eq!(report.desired_action, LiveSupervisorAction::Start);
    }
}
