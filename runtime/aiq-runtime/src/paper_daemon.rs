use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{Context, Result};
use chrono::Utc;
use fs2::FileExt;
use serde::Serialize;
use signal_hook::{consts::signal::SIGINT, consts::signal::SIGTERM, flag, SigId};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::paper_loop::{self, PaperLoopInput, PaperLoopReport};

pub struct PaperDaemonInput<'a> {
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: &'a Path,
    pub live: bool,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub start_step_close_ts_ms: Option<i64>,
    pub idle_sleep_ms: u64,
    pub max_idle_polls: usize,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
    pub lock_path: Option<&'a Path>,
    pub emit_progress: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperDaemonReport {
    pub ok: bool,
    pub pid: u32,
    pub lock_path: String,
    pub started_at_ms: i64,
    pub stopped_at_ms: i64,
    pub stop_requested: bool,
    pub dry_run: bool,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub loop_report: PaperLoopReport,
}

pub fn run_daemon(input: PaperDaemonInput<'_>) -> Result<PaperDaemonReport> {
    let lock_path = resolve_lock_path(input.lock_path, input.live);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;
    if input.emit_progress {
        eprintln!(
            "paper daemon started: pid={} lock={} dry_run={} profile={}",
            std::process::id(),
            lock_path.display(),
            input.dry_run,
            input.runtime_bootstrap.pipeline.profile,
        );
    }

    let loop_report = paper_loop::run_loop(PaperLoopInput {
        runtime_bootstrap: input.runtime_bootstrap.clone(),
        config_path: input.config_path,
        live: input.live,
        paper_db: input.paper_db,
        candles_db: input.candles_db,
        explicit_symbols: input.explicit_symbols,
        btc_symbol: input.btc_symbol,
        lookback_bars: input.lookback_bars,
        start_step_close_ts_ms: input.start_step_close_ts_ms,
        max_steps: usize::MAX,
        follow: true,
        idle_sleep_ms: input.idle_sleep_ms,
        max_idle_polls: input.max_idle_polls,
        exported_at_ms: input.exported_at_ms,
        dry_run: input.dry_run,
        stop_flag: Some(stop_flag.as_ref()),
    })?;

    let stopped_at_ms = Utc::now().timestamp_millis();
    let stop_requested = loop_report.stop_requested || stop_flag.load(Ordering::SeqCst);
    if input.emit_progress {
        eprintln!(
            "paper daemon stopped: pid={} steps={} stop_requested={} latest_common={:?} next_due={:?}",
            std::process::id(),
            loop_report.executed_steps,
            stop_requested,
            loop_report.latest_common_close_ts_ms,
            loop_report.next_due_step_close_ts_ms,
        );
    }
    Ok(PaperDaemonReport {
        ok: loop_report.ok,
        pid: std::process::id(),
        lock_path: lock_path.display().to_string(),
        started_at_ms,
        stopped_at_ms,
        stop_requested,
        dry_run: input.dry_run,
        runtime_bootstrap: input.runtime_bootstrap,
        loop_report,
    })
}

fn resolve_lock_path(lock_path: Option<&Path>, live: bool) -> PathBuf {
    if let Some(lock_path) = lock_path {
        return lock_path.to_path_buf();
    }
    if let Some(env_lock_path) = std::env::var_os("AI_QUANT_LOCK_PATH") {
        return PathBuf::from(env_lock_path);
    }

    project_root().join(if live {
        "ai_quant_live.lock"
    } else {
        "ai_quant_paper.lock"
    })
}

fn project_root() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.canonicalize().unwrap_or(path)
}

fn acquire_lock(lock_path: &Path) -> Result<File> {
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create daemon lock parent directory: {}",
                parent.display()
            )
        })?;
    }

    let mut lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("failed to open daemon lock file: {}", lock_path.display()))?;
    if let Err(err) = lock_file.try_lock_exclusive() {
        if err.kind() == std::io::ErrorKind::WouldBlock {
            anyhow::bail!("paper daemon lock already held: {}", lock_path.display());
        }
        return Err(err).with_context(|| {
            format!(
                "failed to acquire paper daemon lock: {}",
                lock_path.display()
            )
        });
    }
    write_pid(&mut lock_file)?;
    Ok(lock_file)
}

fn write_pid(lock_file: &mut File) -> Result<()> {
    lock_file
        .set_len(0)
        .context("failed to truncate daemon lock file")?;
    lock_file
        .seek(SeekFrom::Start(0))
        .context("failed to seek daemon lock file")?;
    writeln!(lock_file, "{}", std::process::id()).context("failed to write daemon lock pid")?;
    lock_file
        .flush()
        .context("failed to flush daemon lock pid")?;
    Ok(())
}

struct SignalHandlerGuard {
    ids: Vec<SigId>,
}

impl Drop for SignalHandlerGuard {
    fn drop(&mut self) {
        for id in self.ids.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}

fn install_signal_handlers(stop_flag: &Arc<AtomicBool>) -> Result<SignalHandlerGuard> {
    let mut ids = Vec::new();
    ids.push(
        flag::register(SIGINT, Arc::clone(stop_flag))
            .context("failed to install paper daemon SIGINT handler")?,
    );
    ids.push(
        flag::register(SIGTERM, Arc::clone(stop_flag))
            .context("failed to install paper daemon SIGTERM handler")?,
    );
    Ok(SignalHandlerGuard { ids })
}
