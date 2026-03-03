use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;

use crate::ws::broadcast::BroadcastHub;

use super::{run_subprocess, JobId, JobStore};

/// Arguments for spawning a factory cycle.
pub struct FactoryArgs {
    pub profile: String,
    pub strategy_mode: Option<String>,
    pub gpu: bool,
    pub tpe: bool,
    pub walk_forward: bool,
    pub slippage_stress: bool,
    pub concentration_checks: bool,
    pub sensitivity_checks: bool,
    pub candidate_count: u32,
    pub max_age_fail_hours: Option<f64>,
    pub funding_max_stale_symbols: Option<u32>,
    pub dry_run: bool,
    pub no_deploy: bool,
    pub enable_livepaper_promotion: bool,
}

/// Spawn a factory cycle as a subprocess.
///
/// Wraps the command with `flock -n` to prevent overlap with the systemd timer.
/// Runs `uv run tools/factory_cycle.py` in the project root directory.
pub async fn spawn_factory(
    job_id: JobId,
    args: FactoryArgs,
    aiq_root: String,
    store: Arc<JobStore>,
    broadcast: BroadcastHub,
) {
    // Use XDG_RUNTIME_DIR for lock file, fallback to /tmp.
    let runtime_dir =
        std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    let lock_path = format!("{runtime_dir}/openclaw-ai-quant-factory-hub.lock");

    // flock -n <lockfile> <command> [args...] — direct exec, no shell needed.
    let mut cmd = Command::new("flock");
    cmd.arg("-n")
        .arg(&lock_path)
        .arg("uv")
        .arg("run")
        .arg("tools/factory_cycle.py")
        .arg("--profile")
        .arg(&args.profile)
        .arg("--candidate-count")
        .arg(args.candidate_count.to_string());

    if let Some(ref mode) = args.strategy_mode {
        if !mode.is_empty() {
            cmd.arg("--strategy-mode").arg(mode);
        }
    }

    if args.gpu {
        cmd.arg("--gpu");
    }
    if args.tpe {
        cmd.arg("--tpe");
    }
    if args.walk_forward {
        cmd.arg("--walk-forward");
    }
    if args.slippage_stress {
        cmd.arg("--slippage-stress");
    }
    if args.concentration_checks {
        cmd.arg("--concentration-checks");
    }
    if args.sensitivity_checks {
        cmd.arg("--sensitivity-checks");
    }
    if args.dry_run {
        cmd.arg("--dry-run");
    }
    if args.no_deploy {
        cmd.arg("--no-deploy");
    }
    if args.enable_livepaper_promotion {
        cmd.arg("--enable-livepaper-promotion");
    }
    if let Some(hours) = args.max_age_fail_hours {
        cmd.arg("--max-age-fail-hours")
            .arg(hours.to_string());
    }
    if let Some(stale) = args.funding_max_stale_symbols {
        cmd.arg("--funding-max-stale-symbols")
            .arg(stale.to_string());
    }

    cmd.current_dir(&aiq_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let jid = job_id.clone();
    tokio::spawn(async move {
        run_subprocess(jid, "factory", cmd, store, broadcast).await;
    });
}
