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
use std::time::UNIX_EPOCH;

use crate::paper_cycle::{self, PaperCycleInput};
use crate::paper_loop::{self, PaperLoopReport, PaperLoopStepReport};

pub struct PaperDaemonInput<'a> {
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: &'a Path,
    pub strategy_mode: Option<&'a str>,
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
    pub status_path: Option<&'a Path>,
    pub watch_symbols_file: bool,
    pub emit_progress: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperDaemonReport {
    pub ok: bool,
    pub pid: u32,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub stopped_at_ms: i64,
    pub stop_requested: bool,
    pub dry_run: bool,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub watch_symbols_file: bool,
    pub symbols_file: Option<String>,
    pub manifest_symbols: Vec<String>,
    pub last_active_symbols: Vec<String>,
    pub manifest_reload_count: usize,
    pub manifest_reload_failure_count: usize,
    pub loop_report: PaperLoopReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct PaperDaemonStatus {
    pub ok: bool,
    pub running: bool,
    pub pid: u32,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub updated_at_ms: i64,
    pub stopped_at_ms: Option<i64>,
    pub stop_requested: bool,
    pub dry_run: bool,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub watch_symbols_file: bool,
    pub symbols_file: Option<String>,
    pub manifest_symbols: Vec<String>,
    pub last_active_symbols: Vec<String>,
    pub manifest_reload_count: usize,
    pub manifest_reload_failure_count: usize,
    pub latest_common_close_ts_ms: Option<i64>,
    pub next_due_step_close_ts_ms: Option<i64>,
    pub executed_steps: usize,
    pub idle_polls: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    modified_ms: Option<u128>,
    len: u64,
}

#[derive(Debug, Clone)]
struct SymbolManifestState {
    base_symbols: Vec<String>,
    symbols_file: Option<PathBuf>,
    watch_symbols_file: bool,
    file_symbols: Vec<String>,
    last_seen_stamp: Option<FileStamp>,
    reload_count: usize,
    reload_failure_count: usize,
}

enum ManifestRefresh {
    NoChange,
    Candidate(Vec<String>),
    Warning(String),
}

impl SymbolManifestState {
    fn new(
        explicit_symbols: &[String],
        symbols_file: Option<&Path>,
        watch_symbols_file: bool,
    ) -> Result<Self> {
        if watch_symbols_file && symbols_file.is_none() {
            anyhow::bail!("paper daemon --watch-symbols-file requires --symbols-file");
        }

        let base_symbols = paper_loop::normalise_symbols(explicit_symbols);
        let mut state = Self {
            base_symbols,
            symbols_file: symbols_file.map(Path::to_path_buf),
            watch_symbols_file,
            file_symbols: Vec::new(),
            last_seen_stamp: None,
            reload_count: 0,
            reload_failure_count: 0,
        };
        if let Some(path) = state.symbols_file.clone() {
            let stamp = file_stamp(&path)?;
            let file_symbols = load_symbols_file(&path)?;
            state.file_symbols = file_symbols;
            state.last_seen_stamp = Some(stamp);
        }
        Ok(state)
    }

    fn current_symbols(&self) -> Vec<String> {
        let mut merged = self.base_symbols.clone();
        merged.extend(self.file_symbols.iter().cloned());
        paper_loop::normalise_symbols(&merged)
    }

    fn symbols_file_display(&self) -> Option<String> {
        self.symbols_file
            .as_ref()
            .map(|path| path.display().to_string())
    }

    fn refresh_if_needed(&mut self) -> Result<ManifestRefresh> {
        if !self.watch_symbols_file {
            return Ok(ManifestRefresh::NoChange);
        }
        let Some(symbols_file) = self.symbols_file.clone() else {
            return Ok(ManifestRefresh::NoChange);
        };

        let current_stamp = file_stamp(&symbols_file)?;
        if self.last_seen_stamp.as_ref() == Some(&current_stamp) {
            return Ok(ManifestRefresh::NoChange);
        }
        self.last_seen_stamp = Some(current_stamp);

        match load_symbols_file(&symbols_file) {
            Ok(file_symbols) => {
                if file_symbols.is_empty() && !self.file_symbols.is_empty() {
                    self.reload_failure_count = self.reload_failure_count.saturating_add(1);
                    return Ok(ManifestRefresh::Warning(format!(
                        "paper daemon ignored empty symbols file reload; retaining last good manifest: {}",
                        self.file_symbols.join(",")
                    )));
                }
                if file_symbols != self.file_symbols {
                    return Ok(ManifestRefresh::Candidate(file_symbols));
                }
                Ok(ManifestRefresh::NoChange)
            }
            Err(err) => {
                self.reload_failure_count = self.reload_failure_count.saturating_add(1);
                Ok(ManifestRefresh::Warning(format!(
                    "paper daemon ignored symbols file reload; retaining last good manifest: {err}"
                )))
            }
        }
    }

    fn candidate_symbols(&self, file_symbols: &[String]) -> Vec<String> {
        let mut merged = self.base_symbols.clone();
        merged.extend(file_symbols.iter().cloned());
        paper_loop::normalise_symbols(&merged)
    }

    fn accept_candidate(&mut self, file_symbols: Vec<String>) -> Option<String> {
        let prior_manifest = self.current_symbols();
        self.file_symbols = file_symbols;
        let next_manifest = self.current_symbols();
        if next_manifest == prior_manifest {
            return None;
        }
        self.reload_count = self.reload_count.saturating_add(1);
        Some(format!(
            "paper daemon reloaded symbols: {}",
            next_manifest.join(",")
        ))
    }

    fn reject_candidate(&mut self, err: &anyhow::Error) -> String {
        self.reload_failure_count = self.reload_failure_count.saturating_add(1);
        format!("paper daemon ignored symbols file reload; retaining last good manifest: {err:#}")
    }
}

#[derive(Debug, Clone)]
struct StatusSnapshot<'a> {
    status_path: &'a Path,
    started_at_ms: i64,
    stopped_at_ms: Option<i64>,
    stop_requested: bool,
    last_active_symbols: &'a [String],
    latest_common_close_ts_ms: Option<i64>,
    next_due_step_close_ts_ms: Option<i64>,
    idle_polls: usize,
    executed_steps: usize,
    warnings: &'a [String],
    errors: &'a [String],
}

fn write_status_snapshot(
    input: &PaperDaemonInput<'_>,
    lock_path: &Path,
    manifest_state: &SymbolManifestState,
    snapshot: StatusSnapshot<'_>,
) -> Result<()> {
    let status = PaperDaemonStatus {
        ok: snapshot.errors.is_empty(),
        running: snapshot.stopped_at_ms.is_none(),
        pid: std::process::id(),
        lock_path: lock_path.display().to_string(),
        status_path: snapshot.status_path.display().to_string(),
        started_at_ms: snapshot.started_at_ms,
        updated_at_ms: Utc::now().timestamp_millis(),
        stopped_at_ms: snapshot.stopped_at_ms,
        stop_requested: snapshot.stop_requested,
        dry_run: input.dry_run,
        runtime_bootstrap: input.runtime_bootstrap.clone(),
        watch_symbols_file: input.watch_symbols_file,
        symbols_file: manifest_state.symbols_file_display(),
        manifest_symbols: manifest_state.current_symbols(),
        last_active_symbols: snapshot.last_active_symbols.to_vec(),
        manifest_reload_count: manifest_state.reload_count,
        manifest_reload_failure_count: manifest_state.reload_failure_count,
        latest_common_close_ts_ms: snapshot.latest_common_close_ts_ms,
        next_due_step_close_ts_ms: snapshot.next_due_step_close_ts_ms,
        executed_steps: snapshot.executed_steps,
        idle_polls: snapshot.idle_polls,
        warnings: snapshot.warnings.to_vec(),
        errors: snapshot.errors.to_vec(),
    };
    write_status_file(snapshot.status_path, &status)
}

fn write_status_file(status_path: &Path, status: &PaperDaemonStatus) -> Result<()> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create daemon status parent directory: {}",
                parent.display()
            )
        })?;
    }

    let file_name = status_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("paper-daemon.status.json");
    let tmp_name = format!("{file_name}.tmp-{}", std::process::id());
    let tmp_path = status_path.with_file_name(tmp_name);
    let payload =
        serde_json::to_vec_pretty(status).context("failed to serialise paper daemon status")?;
    std::fs::write(&tmp_path, payload).with_context(|| {
        format!(
            "failed to write paper daemon status tmp file: {}",
            tmp_path.display()
        )
    })?;
    std::fs::rename(&tmp_path, status_path).with_context(|| {
        format!(
            "failed to promote paper daemon status file into place: {}",
            status_path.display()
        )
    })?;
    Ok(())
}

pub fn run_daemon(input: PaperDaemonInput<'_>) -> Result<PaperDaemonReport> {
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }

    let lock_path = resolve_lock_path(input.lock_path, input.live);
    let status_path = resolve_status_path(input.status_path, &lock_path);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;
    let working_paper_db = paper_loop::prepare_working_paper_db(input.paper_db, input.dry_run)?;
    let mut manifest_state = SymbolManifestState::new(
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

    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    let mut stop_requested = false;
    let mut initial_last_applied_step_close_ts_ms = None;
    let mut interval = None;
    let mut latest_common_close_ts_ms = None;
    let mut next_due_step_close_ts_ms = None;
    let mut idle_polls = 0usize;
    let mut last_active_symbols = Vec::new();
    let mut emitted_empty_idle_warning = false;
    let mut emitted_due_idle_warning = false;

    write_status_snapshot(
        &input,
        &lock_path,
        &manifest_state,
        StatusSnapshot {
            status_path: &status_path,
            started_at_ms,
            stopped_at_ms: None,
            stop_requested: false,
            last_active_symbols: &last_active_symbols,
            latest_common_close_ts_ms,
            next_due_step_close_ts_ms,
            idle_polls,
            executed_steps: steps.len(),
            warnings: &warnings,
            errors: &errors,
        },
    )?;

    loop {
        if paper_loop::is_stop_requested(Some(stop_flag.as_ref())) {
            stop_requested = true;
            warnings.push("paper daemon stop requested".to_string());
            break;
        }

        match manifest_state.refresh_if_needed()? {
            ManifestRefresh::NoChange => {}
            ManifestRefresh::Warning(warning) => warnings.push(warning),
            ManifestRefresh::Candidate(file_symbols) => {
                let candidate_symbols = manifest_state.candidate_symbols(&file_symbols);
                match paper_loop::inspect_loop_context(
                    &input.runtime_bootstrap,
                    input.config_path,
                    input.strategy_mode,
                    input.live,
                    working_paper_db.path(),
                    input.candles_db,
                    &candidate_symbols,
                    input.btc_symbol,
                ) {
                    Ok(_) => {
                        if let Some(message) = manifest_state.accept_candidate(file_symbols) {
                            warnings.push(message);
                        }
                    }
                    Err(err) => warnings.push(manifest_state.reject_candidate(&err)),
                }
            }
        }
        let manifest_symbols = manifest_state.current_symbols();
        let maybe_context = paper_loop::inspect_loop_context(
            &input.runtime_bootstrap,
            input.config_path,
            input.strategy_mode,
            input.live,
            working_paper_db.path(),
            input.candles_db,
            &manifest_symbols,
            input.btc_symbol,
        )?;
        let Some(context) = maybe_context else {
            last_active_symbols.clear();
            if !input.watch_symbols_file {
                if steps.is_empty() {
                    anyhow::bail!("paper daemon requires explicit symbols or open paper positions");
                }
                warnings.push("paper daemon stopped: no active symbols remain".to_string());
                break;
            }
            if !emitted_empty_idle_warning {
                warnings.push("paper daemon idle: no active symbols available yet".to_string());
                emitted_empty_idle_warning = true;
            }
            idle_polls = idle_polls.saturating_add(1);
            write_status_snapshot(
                &input,
                &lock_path,
                &manifest_state,
                StatusSnapshot {
                    status_path: &status_path,
                    started_at_ms,
                    stopped_at_ms: None,
                    stop_requested: false,
                    last_active_symbols: &last_active_symbols,
                    latest_common_close_ts_ms,
                    next_due_step_close_ts_ms,
                    idle_polls,
                    executed_steps: steps.len(),
                    warnings: &warnings,
                    errors: &errors,
                },
            )?;
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                warnings.push(format!(
                    "paper daemon follow exhausted after {} idle poll(s)",
                    input.max_idle_polls
                ));
                write_status_snapshot(
                    &input,
                    &lock_path,
                    &manifest_state,
                    StatusSnapshot {
                        status_path: &status_path,
                        started_at_ms,
                        stopped_at_ms: None,
                        stop_requested: false,
                        last_active_symbols: &last_active_symbols,
                        latest_common_close_ts_ms,
                        next_due_step_close_ts_ms,
                        idle_polls,
                        executed_steps: steps.len(),
                        warnings: &warnings,
                        errors: &errors,
                    },
                )?;
                break;
            }
            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        };

        emitted_empty_idle_warning = false;
        last_active_symbols = context.active_symbols.clone();
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
            Some(last_applied_step_close_ts_ms) => {
                let expected_next = last_applied_step_close_ts_ms
                    .checked_add(interval_ms)
                    .context("paper daemon interval overflow")?;
                if steps.is_empty() {
                    if let Some(start_step_close_ts_ms) = input.start_step_close_ts_ms {
                        if start_step_close_ts_ms != expected_next {
                            anyhow::bail!(
                                "paper daemon start_step_close_ts_ms {} does not match next unapplied step {}",
                                start_step_close_ts_ms,
                                expected_next
                            );
                        }
                    }
                }
                expected_next
            }
            None => input.start_step_close_ts_ms.context(
                "paper daemon requires --start-step-close-ts-ms when no prior runtime_cycle_steps exist",
            )?,
        };
        next_due_step_close_ts_ms = Some(candidate_next_due);

        if candidate_next_due > context.latest_common_close_ts_ms {
            if !emitted_due_idle_warning {
                warnings.push(format!(
                    "paper daemon idle: next due step {} is newer than latest common close {}",
                    candidate_next_due, context.latest_common_close_ts_ms
                ));
                emitted_due_idle_warning = true;
            }
            idle_polls = idle_polls.saturating_add(1);
            write_status_snapshot(
                &input,
                &lock_path,
                &manifest_state,
                StatusSnapshot {
                    status_path: &status_path,
                    started_at_ms,
                    stopped_at_ms: None,
                    stop_requested: false,
                    last_active_symbols: &last_active_symbols,
                    latest_common_close_ts_ms,
                    next_due_step_close_ts_ms,
                    idle_polls,
                    executed_steps: steps.len(),
                    warnings: &warnings,
                    errors: &errors,
                },
            )?;
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                warnings.push(format!(
                    "paper daemon follow exhausted after {} idle poll(s)",
                    input.max_idle_polls
                ));
                write_status_snapshot(
                    &input,
                    &lock_path,
                    &manifest_state,
                    StatusSnapshot {
                        status_path: &status_path,
                        started_at_ms,
                        stopped_at_ms: None,
                        stop_requested: false,
                        last_active_symbols: &last_active_symbols,
                        latest_common_close_ts_ms,
                        next_due_step_close_ts_ms,
                        idle_polls,
                        executed_steps: steps.len(),
                        warnings: &warnings,
                        errors: &errors,
                    },
                )?;
                break;
            }
            paper_loop::sleep_with_stop_flag(input.idle_sleep_ms, Some(stop_flag.as_ref()));
            continue;
        }

        let cycle_report = paper_cycle::run_cycle(PaperCycleInput {
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            config_path: input.config_path,
            strategy_mode: input.strategy_mode,
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
        emitted_due_idle_warning = false;
        write_status_snapshot(
            &input,
            &lock_path,
            &manifest_state,
            StatusSnapshot {
                status_path: &status_path,
                started_at_ms,
                stopped_at_ms: None,
                stop_requested: false,
                last_active_symbols: &last_active_symbols,
                latest_common_close_ts_ms,
                next_due_step_close_ts_ms,
                idle_polls,
                executed_steps: steps.len(),
                warnings: &warnings,
                errors: &errors,
            },
        )?;
    }

    let manifest_symbols = manifest_state.current_symbols();
    let final_context = paper_loop::inspect_loop_context(
        &input.runtime_bootstrap,
        input.config_path,
        input.strategy_mode,
        input.live,
        working_paper_db.path(),
        input.candles_db,
        &manifest_symbols,
        input.btc_symbol,
    )?;
    if let Some(context) = final_context.as_ref() {
        interval = Some(context.interval.clone());
        latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);
        last_active_symbols = context.active_symbols.clone();
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
        explicit_symbols: manifest_symbols.clone(),
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
    write_status_snapshot(
        &input,
        &lock_path,
        &manifest_state,
        StatusSnapshot {
            status_path: &status_path,
            started_at_ms,
            stopped_at_ms: Some(stopped_at_ms),
            stop_requested,
            last_active_symbols: &last_active_symbols,
            latest_common_close_ts_ms: loop_report.latest_common_close_ts_ms,
            next_due_step_close_ts_ms: loop_report.next_due_step_close_ts_ms,
            idle_polls: loop_report.idle_polls,
            executed_steps: loop_report.executed_steps,
            warnings: &loop_report.warnings,
            errors: &loop_report.errors,
        },
    )?;
    if input.emit_progress {
        eprintln!(
            "paper daemon stopped: pid={} steps={} stop_requested={} latest_common={:?} next_due={:?} reloads={} reload_failures={}",
            std::process::id(),
            loop_report.executed_steps,
            stop_requested,
            loop_report.latest_common_close_ts_ms,
            loop_report.next_due_step_close_ts_ms,
            manifest_state.reload_count,
            manifest_state.reload_failure_count,
        );
    }
    Ok(PaperDaemonReport {
        ok: loop_report.ok,
        pid: std::process::id(),
        lock_path: lock_path.display().to_string(),
        status_path: status_path.display().to_string(),
        started_at_ms,
        stopped_at_ms,
        stop_requested,
        dry_run: input.dry_run,
        runtime_bootstrap: input.runtime_bootstrap,
        watch_symbols_file: input.watch_symbols_file,
        symbols_file: manifest_state.symbols_file_display(),
        manifest_symbols,
        last_active_symbols,
        manifest_reload_count: manifest_state.reload_count,
        manifest_reload_failure_count: manifest_state.reload_failure_count,
        loop_report,
    })
}

fn load_symbols_file(symbols_file: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(symbols_file)
        .with_context(|| format!("failed to read symbols file: {}", symbols_file.display()))?;
    let raw = String::from_utf8(bytes).with_context(|| {
        format!(
            "symbols file must be valid UTF-8: {}",
            symbols_file.display()
        )
    })?;
    let symbols = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    Ok(paper_loop::normalise_symbols(&symbols))
}

fn file_stamp(path: &Path) -> Result<FileStamp> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to stat symbols file: {}", path.display()))?;
    let modified_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    Ok(FileStamp {
        modified_ms,
        len: metadata.len(),
    })
}

pub(crate) fn resolve_lock_path(lock_path: Option<&Path>, live: bool) -> PathBuf {
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

pub(crate) fn resolve_status_path(status_path: Option<&Path>, lock_path: &Path) -> PathBuf {
    if let Some(status_path) = status_path {
        return status_path.to_path_buf();
    }
    if let Some(env_status_path) = std::env::var_os("AI_QUANT_STATUS_PATH") {
        return PathBuf::from(env_status_path);
    }

    let file_name = lock_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ai_quant_paper.lock");
    let status_name = file_name
        .strip_suffix(".lock")
        .map(|stem| format!("{stem}.status.json"))
        .unwrap_or_else(|| format!("{file_name}.status.json"));
    match lock_path.parent() {
        Some(parent) => parent.join(status_name),
        None => PathBuf::from(status_name),
    }
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
