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

use crate::paper_cycle::{self, PaperCycleInput};
use crate::paper_loop::{self, LoopContext, PaperLoopReport, PaperLoopStepReport};

pub struct PaperDaemonInput<'a> {
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: &'a Path,
    pub live: bool,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub watch_symbols_file: bool,
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
    pub watch_symbols_file: bool,
    pub watched_symbols_file: Option<String>,
    pub manifest_symbols: Vec<String>,
    pub active_symbols: Vec<String>,
    pub manifest_reload_count: usize,
    pub manifest_reload_failure_count: usize,
    pub loop_report: PaperLoopReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SymbolsFileSnapshot {
    Missing,
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
struct SymbolsManifest {
    cli_symbols: Vec<String>,
    symbols_file: Option<PathBuf>,
    watch_symbols_file: bool,
    last_good_file_symbols: Vec<String>,
    last_observed_snapshot: Option<SymbolsFileSnapshot>,
    reload_count: usize,
    reload_failure_count: usize,
}

pub fn run_daemon(input: PaperDaemonInput<'_>) -> Result<PaperDaemonReport> {
    if input.watch_symbols_file && input.symbols_file.is_none() {
        anyhow::bail!("paper daemon --watch-symbols-file requires --symbols-file");
    }
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }

    let lock_path = resolve_lock_path(input.lock_path, input.live);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;
    let mut manifest = SymbolsManifest::load(
        input.explicit_symbols,
        input.symbols_file,
        input.watch_symbols_file,
    )?;

    if input.emit_progress {
        eprintln!(
            "paper daemon started: pid={} lock={} dry_run={} profile={} watch_symbols_file={}",
            std::process::id(),
            lock_path.display(),
            input.dry_run,
            input.runtime_bootstrap.pipeline.profile,
            input.watch_symbols_file,
        );
    }

    let working_paper_db = paper_loop::prepare_working_paper_db(input.paper_db, input.dry_run)?;
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    let mut stop_requested = false;
    let mut initial_last_applied_step_close_ts_ms = None;
    let mut interval = None;
    let mut latest_common_close_ts_ms = None;
    let mut next_due_step_close_ts_ms = None;
    let mut idle_polls = 0usize;
    let mut last_active_symbols = manifest.explicit_symbols();

    loop {
        if paper_loop::is_stop_requested(Some(stop_flag.as_ref())) {
            stop_requested = true;
            push_unique_warning(&mut warnings, "paper daemon stop requested".to_string());
            break;
        }

        for warning in manifest.reload_if_changed()? {
            push_unique_warning(&mut warnings, warning);
        }
        let manifest_symbols = manifest.explicit_symbols();

        let maybe_context = paper_loop::inspect_loop_context(
            &input.runtime_bootstrap,
            input.config_path,
            input.live,
            working_paper_db.path(),
            input.candles_db,
            &manifest_symbols,
            input.btc_symbol,
        )?;

        let Some(context) = maybe_context else {
            last_active_symbols = manifest_symbols.clone();
            interval = None;
            latest_common_close_ts_ms = None;
            next_due_step_close_ts_ms = None;

            if !input.watch_symbols_file {
                if steps.is_empty() {
                    anyhow::bail!("paper daemon requires explicit symbols or open paper positions");
                }
                push_unique_warning(
                    &mut warnings,
                    "paper daemon stopped: no active symbols remain".to_string(),
                );
                break;
            }

            idle_polls = idle_polls.saturating_add(1);
            push_unique_warning(
                &mut warnings,
                "paper daemon idle: no active symbols remain".to_string(),
            );
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                push_unique_warning(
                    &mut warnings,
                    format!(
                        "paper daemon follow exhausted after {} idle poll(s)",
                        input.max_idle_polls
                    ),
                );
                break;
            }

            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        };

        update_context_state(
            &context,
            &mut last_active_symbols,
            &mut interval,
            &mut latest_common_close_ts_ms,
            &mut initial_last_applied_step_close_ts_ms,
        );

        let interval_ms = paper_loop::interval_to_ms(&context.interval).with_context(|| {
            format!(
                "unsupported interval for paper daemon: {}",
                context.interval
            )
        })?;
        let candidate_next_due = next_due_step(
            &context,
            steps.is_empty(),
            input.start_step_close_ts_ms,
            interval_ms,
        )?;
        next_due_step_close_ts_ms = Some(candidate_next_due);

        if candidate_next_due > context.latest_common_close_ts_ms {
            idle_polls = idle_polls.saturating_add(1);
            push_unique_warning(
                &mut warnings,
                format!(
                    "paper daemon idle: next due step {} is newer than latest common close {}",
                    candidate_next_due, context.latest_common_close_ts_ms
                ),
            );
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                push_unique_warning(
                    &mut warnings,
                    format!(
                        "paper daemon follow exhausted after {} idle poll(s)",
                        input.max_idle_polls
                    ),
                );
                break;
            }

            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        }

        paper_loop::ensure_exact_step_candle_coverage(
            input.candles_db,
            &context.interval,
            &context.active_symbols,
            input.btc_symbol,
            candidate_next_due,
        )?;

        let cycle_report = paper_cycle::run_cycle(PaperCycleInput {
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            config_path: input.config_path,
            live: input.live,
            paper_db: working_paper_db.path(),
            candles_db: input.candles_db,
            explicit_symbols: &manifest_symbols,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            step_close_ts_ms: candidate_next_due,
            exported_at_ms: Some(input.exported_at_ms.unwrap_or(candidate_next_due)),
            dry_run: false,
        })?;
        warnings.extend(cycle_report.warnings.iter().cloned());
        errors.extend(cycle_report.errors.iter().cloned());
        steps.push(PaperLoopStepReport::from_cycle_report(
            cycle_report,
            input.dry_run,
        ));
        idle_polls = 0;
    }

    let final_manifest_symbols = manifest.explicit_symbols();
    let final_context = paper_loop::inspect_loop_context(
        &input.runtime_bootstrap,
        input.config_path,
        input.live,
        working_paper_db.path(),
        input.candles_db,
        &final_manifest_symbols,
        input.btc_symbol,
    )?;
    if let Some(context) = final_context.as_ref() {
        update_context_state(
            context,
            &mut last_active_symbols,
            &mut interval,
            &mut latest_common_close_ts_ms,
            &mut initial_last_applied_step_close_ts_ms,
        );
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

    let loop_report = PaperLoopReport {
        ok: errors.is_empty(),
        dry_run: input.dry_run,
        interval,
        explicit_symbols: final_manifest_symbols.clone(),
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

    let stopped_at_ms = Utc::now().timestamp_millis();
    let stop_requested = loop_report.stop_requested || stop_flag.load(Ordering::SeqCst);
    if input.emit_progress {
        eprintln!(
            "paper daemon stopped: pid={} steps={} reloads={} reload_failures={} stop_requested={} latest_common={:?} next_due={:?}",
            std::process::id(),
            loop_report.executed_steps,
            manifest.reload_count,
            manifest.reload_failure_count,
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
        watch_symbols_file: input.watch_symbols_file,
        watched_symbols_file: input.symbols_file.map(|path| path.display().to_string()),
        manifest_symbols: final_manifest_symbols,
        active_symbols: last_active_symbols,
        manifest_reload_count: manifest.reload_count,
        manifest_reload_failure_count: manifest.reload_failure_count,
        loop_report,
    })
}

impl SymbolsManifest {
    fn load(
        explicit_symbols: &[String],
        symbols_file: Option<&Path>,
        watch_symbols_file: bool,
    ) -> Result<Self> {
        let cli_symbols = paper_loop::normalise_symbols(explicit_symbols);
        let mut manifest = Self {
            cli_symbols,
            symbols_file: symbols_file.map(Path::to_path_buf),
            watch_symbols_file,
            last_good_file_symbols: Vec::new(),
            last_observed_snapshot: None,
            reload_count: 0,
            reload_failure_count: 0,
        };

        if manifest.symbols_file.is_some() {
            manifest.initial_load()?;
        }

        Ok(manifest)
    }

    fn explicit_symbols(&self) -> Vec<String> {
        let mut merged = self.cli_symbols.clone();
        merged.extend(self.last_good_file_symbols.iter().cloned());
        paper_loop::normalise_symbols(&merged)
    }

    fn reload_if_changed(&mut self) -> Result<Vec<String>> {
        if !self.watch_symbols_file {
            return Ok(Vec::new());
        }

        let Some(symbols_file) = self.symbols_file.as_ref() else {
            return Ok(Vec::new());
        };
        let snapshot = read_symbols_file_snapshot(symbols_file)?;
        if self.last_observed_snapshot.as_ref() == Some(&snapshot) {
            return Ok(Vec::new());
        }
        self.last_observed_snapshot = Some(snapshot.clone());

        let warning = match parse_symbols_snapshot(symbols_file, snapshot) {
            Ok(symbols) => {
                self.last_good_file_symbols = symbols;
                self.reload_count = self.reload_count.saturating_add(1);
                None
            }
            Err(err) => {
                self.reload_failure_count = self.reload_failure_count.saturating_add(1);
                Some(format!(
                    "paper daemon symbols-file reload failed for {}; keeping last good manifest: {err}",
                    symbols_file.display()
                ))
            }
        };

        Ok(warning.into_iter().collect())
    }

    fn initial_load(&mut self) -> Result<()> {
        let symbols_file = self
            .symbols_file
            .as_ref()
            .context("symbols file should be present for initial load")?;
        let snapshot = read_symbols_file_snapshot(symbols_file)?;
        self.last_good_file_symbols = parse_symbols_snapshot(symbols_file, snapshot.clone())?;
        self.last_observed_snapshot = Some(snapshot);
        Ok(())
    }
}

fn read_symbols_file_snapshot(path: &Path) -> Result<SymbolsFileSnapshot> {
    match fs::read(path) {
        Ok(bytes) => Ok(SymbolsFileSnapshot::Bytes(bytes)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(SymbolsFileSnapshot::Missing),
        Err(err) => Err(err)
            .with_context(|| format!("failed to read symbols file snapshot: {}", path.display())),
    }
}

fn parse_symbols_snapshot(path: &Path, snapshot: SymbolsFileSnapshot) -> Result<Vec<String>> {
    match snapshot {
        SymbolsFileSnapshot::Missing => {
            anyhow::bail!("symbols file missing: {}", path.display());
        }
        SymbolsFileSnapshot::Bytes(bytes) => {
            let text = String::from_utf8(bytes)
                .with_context(|| format!("symbols file is not valid UTF-8: {}", path.display()))?;
            let symbols = text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            Ok(paper_loop::normalise_symbols(&symbols))
        }
    }
}

fn next_due_step(
    context: &LoopContext,
    first_iteration: bool,
    start_step_close_ts_ms: Option<i64>,
    interval_ms: i64,
) -> Result<i64> {
    match context.last_applied_step_close_ts_ms {
        Some(last_applied_step_close_ts_ms) => {
            let expected_next = last_applied_step_close_ts_ms
                .checked_add(interval_ms)
                .context("paper daemon interval overflow")?;
            if first_iteration {
                if let Some(start_step_close_ts_ms) = start_step_close_ts_ms {
                    if start_step_close_ts_ms != expected_next {
                        anyhow::bail!(
                            "paper daemon start_step_close_ts_ms {} does not match next unapplied step {}",
                            start_step_close_ts_ms,
                            expected_next
                        );
                    }
                }
            }
            Ok(expected_next)
        }
        None => start_step_close_ts_ms.context(
            "paper daemon requires --start-step-close-ts-ms when no prior runtime_cycle_steps exist",
        ),
    }
}

fn update_context_state(
    context: &LoopContext,
    active_symbols: &mut Vec<String>,
    interval: &mut Option<String>,
    latest_common_close_ts_ms: &mut Option<i64>,
    initial_last_applied_step_close_ts_ms: &mut Option<i64>,
) {
    *active_symbols = context.active_symbols.clone();
    *interval = Some(context.interval.clone());
    *latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);
    if initial_last_applied_step_close_ts_ms.is_none() {
        *initial_last_applied_step_close_ts_ms = context.last_applied_step_close_ts_ms;
    }
}

fn push_unique_warning(warnings: &mut Vec<String>, warning: String) {
    if !warnings.iter().any(|existing| existing == &warning) {
        warnings.push(warning);
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
