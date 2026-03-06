use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{Context, Result};
use chrono::Utc;
use fs2::FileExt;
use serde::Serialize;
use signal_hook::{consts::signal::SIGINT, consts::signal::SIGTERM, flag, SigId};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use crate::paper_loop::{self, PaperLoopInput, PaperLoopReport};

pub struct PaperDaemonInput<'a> {
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: &'a Path,
    pub live: bool,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PaperDaemonManifestReport {
    pub symbols_file: Option<String>,
    pub explicit_symbols: Vec<String>,
    pub file_symbols: Vec<String>,
    pub manifest_symbols: Vec<String>,
    pub reload_count: usize,
    pub reload_failures: usize,
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
    pub manifest: PaperDaemonManifestReport,
    pub active_symbols: Vec<String>,
    pub loop_report: PaperLoopReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileVersion {
    exists: bool,
    modified: Option<SystemTime>,
    len: Option<u64>,
}

struct SymbolsManifestState {
    explicit_symbols: Vec<String>,
    symbols_file: Option<PathBuf>,
    last_good_file_symbols: Vec<String>,
    last_loaded_version: Option<FileVersion>,
    last_failed_version: Option<FileVersion>,
    reload_count: usize,
    reload_failures: usize,
}

impl SymbolsManifestState {
    fn new(explicit_symbols: &[String], symbols_file: Option<&Path>) -> Result<Self> {
        let explicit_symbols = paper_loop::normalise_symbols(explicit_symbols);
        let mut state = Self {
            explicit_symbols,
            symbols_file: symbols_file.map(Path::to_path_buf),
            last_good_file_symbols: Vec::new(),
            last_loaded_version: None,
            last_failed_version: None,
            reload_count: 0,
            reload_failures: 0,
        };

        if let Some(symbols_file) = symbols_file {
            let version = capture_file_version(symbols_file);
            let file_symbols = read_symbols_file(symbols_file)?;
            state.last_good_file_symbols = file_symbols;
            state.last_loaded_version = Some(version);
        }

        Ok(state)
    }

    fn manifest_symbols(&self) -> Vec<String> {
        let mut merged = self.explicit_symbols.clone();
        merged.extend(self.last_good_file_symbols.iter().cloned());
        paper_loop::normalise_symbols(&merged)
    }

    fn report(&self) -> PaperDaemonManifestReport {
        PaperDaemonManifestReport {
            symbols_file: self
                .symbols_file
                .as_ref()
                .map(|path| path.display().to_string()),
            explicit_symbols: self.explicit_symbols.clone(),
            file_symbols: self.last_good_file_symbols.clone(),
            manifest_symbols: self.manifest_symbols(),
            reload_count: self.reload_count,
            reload_failures: self.reload_failures,
        }
    }

    fn refresh_if_changed(&mut self) -> Result<Option<String>> {
        let Some(symbols_file) = self.symbols_file.as_deref() else {
            return Ok(None);
        };

        let version = capture_file_version(symbols_file);
        if self.last_loaded_version.as_ref() == Some(&version)
            || self.last_failed_version.as_ref() == Some(&version)
        {
            return Ok(None);
        }

        match read_symbols_file(symbols_file) {
            Ok(file_symbols) => {
                let prior_manifest = self.manifest_symbols();
                self.last_good_file_symbols = file_symbols;
                self.last_loaded_version = Some(version);
                self.last_failed_version = None;
                self.reload_count = self.reload_count.saturating_add(1);
                let manifest = self.manifest_symbols();
                if manifest == prior_manifest {
                    return Ok(None);
                }
                Ok(Some(format!(
                    "paper loop reloaded symbols file {}: explicit symbols now [{}]",
                    symbols_file.display(),
                    manifest.join(", ")
                )))
            }
            Err(err) => {
                self.last_failed_version = Some(version);
                self.reload_failures = self.reload_failures.saturating_add(1);
                Ok(Some(format!(
                    "paper daemon retained last good symbols manifest after reload failure ({}): {err:#}",
                    symbols_file.display()
                )))
            }
        }
    }
}

pub fn run_daemon(input: PaperDaemonInput<'_>) -> Result<PaperDaemonReport> {
    let lock_path = resolve_lock_path(input.lock_path, input.live);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;
    let mut manifest = SymbolsManifestState::new(input.explicit_symbols, input.symbols_file)?;
    if input.emit_progress {
        eprintln!(
            "paper daemon started: pid={} lock={} dry_run={} profile={}",
            std::process::id(),
            lock_path.display(),
            input.dry_run,
            input.runtime_bootstrap.pipeline.profile,
        );
    }

    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    let mut stop_requested = false;
    let mut initial_last_applied_step_close_ts_ms = None;
    let mut interval = None;
    let mut latest_common_close_ts_ms = None;
    let mut next_due_step_close_ts_ms = None;
    let mut idle_polls = 0usize;
    let mut active_symbols = Vec::new();
    let initial_manifest_symbols = manifest.manifest_symbols();

    loop {
        if paper_loop::is_stop_requested(Some(stop_flag.as_ref())) {
            stop_requested = true;
            push_unique(
                &mut warnings,
                "paper daemon stop requested".to_string(),
            );
            break;
        }

        if let Some(message) = manifest.refresh_if_changed()? {
            if input.emit_progress {
                eprintln!("{message}");
            }
            push_unique(&mut warnings, message);
        }

        let manifest_symbols = manifest.manifest_symbols();
        let maybe_context = paper_loop::inspect_loop_context(
            &input.runtime_bootstrap,
            input.config_path,
            input.live,
            input.paper_db,
            input.candles_db,
            &manifest_symbols,
            input.btc_symbol,
        )?;
        let Some(context) = maybe_context else {
            active_symbols.clear();
            if manifest_symbols.is_empty() && input.symbols_file.is_none() && steps.is_empty() {
                anyhow::bail!(
                    "paper daemon requires explicit symbols, symbols-file, or open paper positions"
                );
            }
            push_unique(
                &mut warnings,
                "paper loop idle: no active symbols available yet".to_string(),
            );
            idle_polls = idle_polls.saturating_add(1);
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                push_unique(
                    &mut warnings,
                    format!(
                        "paper daemon exhausted after {} idle poll(s)",
                        input.max_idle_polls
                    ),
                );
                break;
            }
            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        };

        active_symbols = context.active_symbols.clone();
        if initial_last_applied_step_close_ts_ms.is_none() {
            initial_last_applied_step_close_ts_ms = context.last_applied_step_close_ts_ms;
        }
        interval = Some(context.interval.clone());
        latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);

        let interval_ms = paper_loop::interval_to_ms(&context.interval).with_context(|| {
            format!(
                "unsupported interval for paper daemon: {}",
                context.interval
            )
        })?;
        let candidate_next_due = match context.last_applied_step_close_ts_ms {
            Some(last_applied_step_close_ts_ms) => last_applied_step_close_ts_ms
                .checked_add(interval_ms)
                .context("paper daemon interval overflow")?,
            None => input.start_step_close_ts_ms.context(
                "paper daemon requires --start-step-close-ts-ms when no prior runtime_cycle_steps exist",
            )?,
        };
        next_due_step_close_ts_ms = Some(candidate_next_due);
        if candidate_next_due > context.latest_common_close_ts_ms {
            push_unique(
                &mut warnings,
                format!(
                    "paper loop idle: next due step {} is newer than latest common close {}",
                    candidate_next_due, context.latest_common_close_ts_ms
                ),
            );
            idle_polls = idle_polls.saturating_add(1);
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                push_unique(
                    &mut warnings,
                    format!(
                        "paper daemon exhausted after {} idle poll(s)",
                        input.max_idle_polls
                    ),
                );
                break;
            }
            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        }

        let step_report = paper_loop::run_loop(PaperLoopInput {
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            config_path: input.config_path,
            live: input.live,
            paper_db: input.paper_db,
            candles_db: input.candles_db,
            explicit_symbols: manifest_symbols.as_slice(),
            symbols_file: None,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            start_step_close_ts_ms: context
                .last_applied_step_close_ts_ms
                .is_none()
                .then_some(candidate_next_due),
            max_steps: 1,
            follow: false,
            idle_sleep_ms: input.idle_sleep_ms,
            max_idle_polls: 0,
            exported_at_ms: input.exported_at_ms,
            dry_run: input.dry_run,
            stop_flag: Some(stop_flag.as_ref()),
        })?;

        interval = step_report.interval.clone().or(interval);
        latest_common_close_ts_ms = step_report
            .latest_common_close_ts_ms
            .or(latest_common_close_ts_ms);
        next_due_step_close_ts_ms = step_report.next_due_step_close_ts_ms.or_else(|| {
            step_report
                .latest_common_close_ts_ms
                .and_then(|_| candidate_next_due.checked_add(interval_ms))
        });
        if initial_last_applied_step_close_ts_ms.is_none() {
            initial_last_applied_step_close_ts_ms =
                step_report.initial_last_applied_step_close_ts_ms;
        }
        append_unique(&mut warnings, step_report.warnings);
        errors.extend(step_report.errors);
        steps.extend(step_report.steps);
        idle_polls = 0;
    }

    let final_manifest = manifest.manifest_symbols();
    let final_context = paper_loop::inspect_loop_context(
        &input.runtime_bootstrap,
        input.config_path,
        input.live,
        input.paper_db,
        input.candles_db,
        &final_manifest,
        input.btc_symbol,
    )?;
    if let Some(context) = final_context {
        active_symbols = context.active_symbols;
        interval = Some(context.interval.clone());
        latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);
        let interval_ms = paper_loop::interval_to_ms(&context.interval).with_context(|| {
            format!(
                "unsupported interval for paper daemon: {}",
                context.interval
            )
        })?;
        next_due_step_close_ts_ms = context
            .last_applied_step_close_ts_ms
            .and_then(|last_applied_step_close_ts_ms| {
                last_applied_step_close_ts_ms.checked_add(interval_ms)
            })
            .or(input.start_step_close_ts_ms);
    }

    let stopped_at_ms = Utc::now().timestamp_millis();
    let stop_requested = stop_requested || stop_flag.load(Ordering::SeqCst);
    let manifest_report = manifest.report();
    let loop_report = PaperLoopReport {
        ok: errors.is_empty(),
        dry_run: input.dry_run,
        interval,
        explicit_symbols: initial_manifest_symbols,
        latest_explicit_symbols: manifest_report.manifest_symbols.clone(),
        symbols_file: manifest_report.symbols_file.clone(),
        symbols_file_reload_count: manifest_report.reload_count,
        initial_last_applied_step_close_ts_ms,
        latest_common_close_ts_ms,
        next_due_step_close_ts_ms,
        executed_steps: steps.len(),
        follow: true,
        idle_polls,
        runtime_bootstrap: input.runtime_bootstrap.clone(),
        steps,
        stop_requested,
        warnings,
        errors,
    };
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
        manifest: manifest_report,
        active_symbols,
        loop_report,
    })
}

fn capture_file_version(path: &Path) -> FileVersion {
    match fs::metadata(path) {
        Ok(metadata) => FileVersion {
            exists: true,
            modified: metadata.modified().ok(),
            len: Some(metadata.len()),
        },
        Err(_) => FileVersion {
            exists: false,
            modified: None,
            len: None,
        },
    }
}

fn read_symbols_file(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read(path).with_context(|| format!("failed to read symbols file: {}", path.display()))?;
    let raw = std::str::from_utf8(&raw)
        .with_context(|| format!("symbols file must be valid UTF-8: {}", path.display()))?;
    let symbols = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    Ok(paper_loop::normalise_symbols(&symbols))
}

fn append_unique(target: &mut Vec<String>, items: Vec<String>) {
    for item in items {
        push_unique(target, item);
    }
}

fn push_unique(target: &mut Vec<String>, item: String) {
    if !target.iter().any(|existing| existing == &item) {
        target.push(item);
    }
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
