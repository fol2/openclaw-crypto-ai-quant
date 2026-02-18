//! Axis-by-axis CPU/GPU parity ledger with failure trace artifacts.
//!
//! This binary is intended as a regression gate:
//! - compares CPU SSOT vs GPU runtime one axis/value at a time
//! - classifies each mismatch as `STATE_MACHINE` or `REDUCTION_ORDER`
//! - emits JSONL rows plus GPU trace artifacts for failed samples
//! - exits non-zero if any sample fails

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use bt_core::candle::CandleData;
use bt_core::config::StrategyConfig;
use bt_core::sweep::{apply_one_pub, SweepSpec};
use bt_gpu::buffers::{GpuComboState, GPU_MAX_SYMBOLS, GPU_TRACE_CAP, GPU_TRACE_SYMBOL_ALL};
use bt_gpu::{run_gpu_sweep, run_gpu_sweep_with_states};
use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Strategy YAML path.
    #[arg(long, default_value = "config/strategy_overrides.yaml")]
    config: String,

    /// Sweep axis spec YAML path.
    #[arg(long)]
    sweep_spec: String,

    /// Candle DB path.
    #[arg(long, default_value = "candles_dbs/candles_1h.db")]
    candles_db: String,

    /// Candle interval (for example 1h, 30m, 5m).
    #[arg(long, default_value = "1h")]
    interval: String,

    /// Optional start timestamp (epoch milliseconds).
    #[arg(long)]
    from_ts: Option<i64>,

    /// Optional end timestamp (epoch milliseconds).
    #[arg(long)]
    to_ts: Option<i64>,

    /// Output JSONL parity ledger.
    #[arg(long, default_value = "artifacts/axis_parity_smoke_current.jsonl")]
    output: PathBuf,

    /// Directory for failed-case trace artifacts.
    #[arg(long, default_value = "artifacts/axis_trace")]
    trace_dir: PathBuf,

    /// Limit number of axes evaluated (smoke mode).
    #[arg(long)]
    max_axes: Option<usize>,

    /// Limit values sampled per axis.
    #[arg(long)]
    max_values_per_axis: Option<usize>,

    /// Apply the first value of each selected axis to the base config before baseline parity.
    ///
    /// Useful for full-combo parity checks when the sweep spec contains single-value axes.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    apply_all_axes_as_base: bool,

    /// Absolute tolerance for final balance.
    #[arg(long, default_value_t = 1e-6)]
    balance_eps: f64,

    /// Absolute tolerance for total PnL.
    #[arg(long, default_value_t = 1e-6)]
    pnl_eps: f64,

    /// Capture GPU trace artifact on every failed sample.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    trace_on_failure: bool,

    /// Restrict trace to one symbol (for example BTC). Defaults to all symbols.
    #[arg(long)]
    trace_symbol: Option<String>,

    /// Maximum number of CPU trade events embedded in each trace artifact.
    #[arg(long, default_value_t = 200)]
    cpu_trace_limit: usize,

    /// Use the tail window of CPU trades (best aligned with GPU ring trace).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cpu_trace_from_tail: bool,

    /// Fail closed when baseline (no overrides) CPU/GPU parity already fails.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    require_baseline_pass: bool,

    /// Fail if trace event parity reports any mismatch in failed samples.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fail_on_event_parity_mismatch: bool,

    /// Include event parity summary in JSONL ledger rows.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    ledger_include_event_parity: bool,

    /// Include first mismatch field names in event parity payloads.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    trace_include_mismatch_fields: bool,

    /// Maximum prefix offset scan window when aligning CPU/GPU event streams.
    #[arg(long, default_value_t = 8)]
    event_parity_max_offset_scan: usize,

    /// Allow absolute 2D offset scan (head/tail skew) instead of baseline-relative scan.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    event_parity_allow_absolute_scan: bool,

    /// Absolute tolerance for numeric event fields (price/size/pnl) in event parity.
    #[arg(long, default_value_t = 1e-4)]
    event_numeric_abs_tol: f64,

    /// Relative tolerance for numeric event fields (price/size/pnl) in event parity.
    #[arg(long, default_value_t = 1e-5)]
    event_numeric_rel_tol: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CompactResult {
    total_trades: u32,
    final_balance: f64,
    total_pnl: f64,
}

#[derive(Debug, Serialize)]
struct AxisLedgerRow {
    axis_index: usize,
    axis_path: String,
    sample_index: usize,
    sample_value: f64,
    status: String,
    cause: Option<String>,
    cpu: CompactResult,
    gpu: CompactResult,
    trade_delta: i64,
    balance_delta_abs: f64,
    pnl_delta_abs: f64,
    trace_artifact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    event_parity: Option<LedgerEventParitySummary>,
}

#[derive(Debug, Serialize)]
struct LedgerEventParitySummary {
    status: String,
    aligned_len: usize,
    cpu_len: usize,
    gpu_len: usize,
    cpu_tail_offset: usize,
    gpu_tail_offset: usize,
    first_mismatch_at: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_mismatch_fields: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct TraceArtifact {
    axis_path: String,
    sample_index: usize,
    sample_value: f64,
    cause: String,
    trace_symbol_requested: Option<String>,
    trace_symbol_index: u32,
    trace_symbol_name: Option<String>,
    cpu: CompactResult,
    gpu: CompactResult,
    trace_count: u32,
    trace_head: u32,
    events: Vec<TraceEventRow>,
    cpu_event_total: usize,
    cpu_event_start: usize,
    cpu_event_end: usize,
    cpu_events: Vec<CpuTradeEventRow>,
    event_parity: EventParitySummary,
}

#[derive(Debug, Serialize)]
struct TraceEventRow {
    idx: usize,
    t_sec: u32,
    sym_idx: u32,
    symbol: Option<String>,
    kind: String,
    side: String,
    reason: String,
    price: f32,
    size: f32,
    pnl: f64,
}

#[derive(Debug, Serialize)]
struct CpuTradeEventRow {
    idx: usize,
    global_idx: usize,
    t_sec: u32,
    symbol: String,
    action: String,
    action_kind: String,
    action_side: String,
    intent_signal: String,
    action_taken: String,
    event_type: String,
    status: String,
    decision_phase: String,
    triggered_by: String,
    reason: String,
    price: f64,
    size: f64,
    pnl: f64,
    balance: f64,
}

#[derive(Debug)]
struct CpuTraceWindow {
    total: usize,
    start: usize,
    end: usize,
    events: Vec<CpuTradeEventRow>,
}

#[derive(Debug, Serialize)]
struct EventParitySummary {
    status: String,
    aligned_len: usize,
    cpu_len: usize,
    gpu_len: usize,
    cpu_tail_offset: usize,
    gpu_tail_offset: usize,
    first_mismatch_at: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_mismatch_fields: Option<Vec<String>>,
    cpu_event: Option<CanonicalEventRow>,
    gpu_event: Option<CanonicalEventRow>,
}

#[derive(Debug, Clone, Serialize)]
struct CanonicalEventRow {
    source_idx: usize,
    global_idx: Option<usize>,
    t_sec: u32,
    symbol: String,
    action: String,
    action_kind: String,
    action_side: String,
    intent_signal: String,
    action_taken: String,
    event_type: String,
    status: String,
    decision_phase: String,
    triggered_by: String,
    reason: String,
    reason_code: String,
    price: f64,
    size: f64,
    pnl: f64,
}

#[derive(Debug, Clone)]
struct DecisionActionCanonical {
    action_kind: String,
    action_side: String,
    intent_signal: String,
    action_taken: String,
}

#[derive(Debug, Clone)]
struct DecisionEventEnvelope {
    event_type: String,
    status: String,
    decision_phase: String,
    triggered_by: String,
}

#[derive(Debug, Clone, Copy)]
struct EventNumericTolerance {
    abs: f64,
    rel: f64,
}

#[cfg(target_os = "linux")]
fn ensure_wsl_cuda_path() {
    const WSL_LIB: &str = "/usr/lib/wsl/lib";
    const MARKER: &str = "__AQC_WSL_CUDA_REEXEC";

    if std::env::var_os(MARKER).is_some() {
        return;
    }
    if !Path::new("/usr/lib/wsl/lib/libcuda.so.1").exists() {
        return;
    }

    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if current.split(':').any(|seg| seg == WSL_LIB) {
        return;
    }

    let next = if current.is_empty() {
        WSL_LIB.to_string()
    } else {
        format!("{WSL_LIB}:{current}")
    };

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("[axis-parity] WSL2 CUDA env fix skipped (current_exe failed: {err})");
            return;
        }
    };

    eprintln!("[axis-parity] WSL2 detected — re-exec with LD_LIBRARY_PATH={next}");
    match Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", &next)
        .env(MARKER, "1")
        .status()
    {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            eprintln!("[axis-parity] WSL2 re-exec failed ({err}), continuing without env fix");
        }
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    ensure_wsl_cuda_path();

    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("[axis-parity] ERROR: {err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if !args.event_numeric_abs_tol.is_finite() || !args.event_numeric_rel_tol.is_finite() {
        return Err(
            "--event-numeric-abs-tol and --event-numeric-rel-tol must be finite values"
                .to_string()
                .into(),
        );
    }

    if args.fail_on_event_parity_mismatch && !args.trace_on_failure {
        return Err(
            "--fail-on-event-parity-mismatch requires --trace-on-failure=true"
                .to_string()
                .into(),
        );
    }

    let mut base_cfg = bt_core::config::load_config(&args.config, None, false);
    let spec = bt_core::sweep::load_sweep_spec(&args.sweep_spec)
        .map_err(|e| format!("failed to load sweep spec {}: {e}", args.sweep_spec))?;

    let db_paths = vec![args.candles_db.clone()];
    let candles = bt_data::sqlite_loader::load_candles_multi(&db_paths, &args.interval)
        .map_err(|e| format!("failed to load candles from {}: {e}", args.candles_db))?;
    if candles.is_empty() {
        return Err(format!("no candles loaded from {}", args.candles_db).into());
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(&args.trace_dir)?;

    let mut out = BufWriter::new(File::create(&args.output)?);
    let event_tol = EventNumericTolerance {
        abs: args.event_numeric_abs_tol.max(0.0),
        rel: args.event_numeric_rel_tol.max(0.0),
    };
    let sorted_symbols = sorted_symbols_with_cap(&candles);
    let trace_symbol_idx =
        resolve_trace_symbol_index(args.trace_symbol.as_deref(), &sorted_symbols);

    let axes: Vec<_> = match args.max_axes {
        Some(limit) => spec.axes.iter().take(limit).cloned().collect(),
        None => spec.axes.clone(),
    };

    if args.apply_all_axes_as_base {
        let mut applied = 0usize;
        for axis in &spec.axes {
            if let Some(first) = axis.values.first() {
                apply_one_pub(&mut base_cfg, &axis.path, *first);
                applied += 1;
            }
        }
        eprintln!(
            "[axis-parity] Applied first values from {} axes (full sweep spec) onto base config before baseline parity",
            applied
        );
    }

    let single_spec = SweepSpec {
        axes: vec![],
        initial_balance: spec.initial_balance,
        lookback: spec.lookback,
    };

    run_gpu_single(&base_cfg, &single_spec, &candles, args.from_ts, args.to_ts)
        .map_err(|e| format!("gpu preflight failed before axis loop: {e}"))?;

    let baseline_cpu = run_cpu_single(&base_cfg, &single_spec, &candles, args.from_ts, args.to_ts)?;
    let baseline_gpu = run_gpu_single(&base_cfg, &single_spec, &candles, args.from_ts, args.to_ts)?;
    let baseline_trade_delta = baseline_gpu.total_trades as i64 - baseline_cpu.total_trades as i64;
    let baseline_balance_delta_abs =
        (baseline_gpu.final_balance - baseline_cpu.final_balance).abs();
    let baseline_pnl_delta_abs = (baseline_gpu.total_pnl - baseline_cpu.total_pnl).abs();
    let baseline_pass = baseline_trade_delta == 0
        && baseline_balance_delta_abs <= args.balance_eps
        && baseline_pnl_delta_abs <= args.pnl_eps;
    let baseline_cause = if baseline_pass {
        None
    } else if baseline_trade_delta != 0 {
        Some("STATE_MACHINE".to_string())
    } else {
        Some("REDUCTION_ORDER".to_string())
    };

    let mut baseline_trace_artifact = None;
    let mut baseline_event_parity = None;
    let mut event_parity_mismatches = 0usize;
    if !baseline_pass && args.trace_on_failure {
        let (trace_gpu, trace_state, trace_symbols) = run_gpu_single_with_trace(
            &base_cfg,
            &single_spec,
            &candles,
            args.from_ts,
            args.to_ts,
            trace_symbol_idx,
        )?;
        let trace_symbol_name = symbol_name(trace_symbol_idx, &trace_symbols);
        let cpu_trace = run_cpu_single_with_trade_events(
            &base_cfg,
            &single_spec,
            &candles,
            args.from_ts,
            args.to_ts,
            trace_symbol_name.as_deref(),
            args.cpu_trace_limit,
            args.cpu_trace_from_tail,
        )?;
        let gpu_events = collect_trace_events(&trace_state, &trace_symbols);
        let event_parity = compare_event_streams(
            &cpu_trace.events,
            &gpu_events,
            args.trace_include_mismatch_fields,
            args.event_parity_max_offset_scan,
            args.event_parity_allow_absolute_scan,
            event_tol,
        );
        if args.ledger_include_event_parity {
            baseline_event_parity = Some(summarise_event_parity(&event_parity));
        }
        if is_event_parity_mismatch(&event_parity.status) {
            event_parity_mismatches += 1;
        }
        let trace = TraceArtifact {
            axis_path: "__baseline__".to_string(),
            sample_index: 0,
            sample_value: 0.0,
            cause: baseline_cause
                .clone()
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            trace_symbol_requested: args.trace_symbol.clone(),
            trace_symbol_index: trace_symbol_idx,
            trace_symbol_name,
            cpu: baseline_cpu.clone(),
            gpu: CompactResult {
                total_trades: trace_gpu.total_trades,
                final_balance: trace_gpu.final_balance,
                total_pnl: trace_gpu.total_pnl,
            },
            trace_count: trace_state.trace_count,
            trace_head: trace_state.trace_head,
            events: gpu_events,
            cpu_event_total: cpu_trace.total,
            cpu_event_start: cpu_trace.start,
            cpu_event_end: cpu_trace.end,
            cpu_events: cpu_trace.events,
            event_parity,
        };
        let trace_path = build_trace_path(
            &args.trace_dir,
            "__baseline__",
            0,
            0.0,
            baseline_cause.as_deref().unwrap_or("UNKNOWN"),
        );
        write_json_pretty(&trace_path, &trace)?;
        baseline_trace_artifact = Some(trace_path.display().to_string());
    }

    let baseline_row = AxisLedgerRow {
        axis_index: 0,
        axis_path: "__baseline__".to_string(),
        sample_index: 0,
        sample_value: 0.0,
        status: if baseline_pass {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        cause: baseline_cause.clone(),
        cpu: baseline_cpu.clone(),
        gpu: CompactResult {
            total_trades: baseline_gpu.total_trades,
            final_balance: baseline_gpu.final_balance,
            total_pnl: baseline_gpu.total_pnl,
        },
        trade_delta: baseline_trade_delta,
        balance_delta_abs: baseline_balance_delta_abs,
        pnl_delta_abs: baseline_pnl_delta_abs,
        trace_artifact: baseline_trace_artifact,
        event_parity: baseline_event_parity,
    };
    writeln!(out, "{}", serde_json::to_string(&baseline_row)?)?;

    if !baseline_pass {
        eprintln!(
            "[axis-parity] baseline mismatch: trade_delta={} balance_delta_abs={} pnl_delta_abs={}",
            baseline_trade_delta, baseline_balance_delta_abs, baseline_pnl_delta_abs
        );
        if args.require_baseline_pass {
            out.flush()?;
            return Err(
                "baseline CPU/GPU parity failed before axis loop; fix global mismatch first"
                    .to_string()
                    .into(),
            );
        }
    }

    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut failures_by_cause: BTreeMap<String, usize> = BTreeMap::new();

    eprintln!(
        "[axis-parity] axes={} (selected), interval={}, candles_db={}",
        axes.len(),
        args.interval,
        args.candles_db
    );

    for (axis_index, axis) in axes.iter().enumerate() {
        if axis.values.is_empty() {
            continue;
        }
        let sample_indices = select_sample_indices(&axis.values, args.max_values_per_axis);
        for sample_index in sample_indices {
            total += 1;
            let sample_value = axis.values[sample_index];

            let mut cfg = base_cfg.clone();
            apply_one_pub(&mut cfg, &axis.path, sample_value);

            let cpu = run_cpu_single(&cfg, &single_spec, &candles, args.from_ts, args.to_ts)?;
            let gpu = run_gpu_single(&cfg, &single_spec, &candles, args.from_ts, args.to_ts)?;

            let trade_delta = gpu.total_trades as i64 - cpu.total_trades as i64;
            let balance_delta_abs = (gpu.final_balance - cpu.final_balance).abs();
            let pnl_delta_abs = (gpu.total_pnl - cpu.total_pnl).abs();

            let pass = trade_delta == 0
                && balance_delta_abs <= args.balance_eps
                && pnl_delta_abs <= args.pnl_eps;

            let cause = if pass {
                None
            } else if trade_delta != 0 {
                Some("STATE_MACHINE".to_string())
            } else {
                Some("REDUCTION_ORDER".to_string())
            };

            let mut trace_artifact = None;
            let mut event_parity_summary = None;
            if !pass && args.trace_on_failure {
                let (trace_gpu, trace_state, trace_symbols) = run_gpu_single_with_trace(
                    &cfg,
                    &single_spec,
                    &candles,
                    args.from_ts,
                    args.to_ts,
                    trace_symbol_idx,
                )?;
                let trace_symbol_name = symbol_name(trace_symbol_idx, &trace_symbols);
                let cpu_trace = run_cpu_single_with_trade_events(
                    &cfg,
                    &single_spec,
                    &candles,
                    args.from_ts,
                    args.to_ts,
                    trace_symbol_name.as_deref(),
                    args.cpu_trace_limit,
                    args.cpu_trace_from_tail,
                )?;
                let gpu_events = collect_trace_events(&trace_state, &trace_symbols);
                let event_parity = compare_event_streams(
                    &cpu_trace.events,
                    &gpu_events,
                    args.trace_include_mismatch_fields,
                    args.event_parity_max_offset_scan,
                    args.event_parity_allow_absolute_scan,
                    event_tol,
                );
                if args.ledger_include_event_parity {
                    event_parity_summary = Some(summarise_event_parity(&event_parity));
                }
                if is_event_parity_mismatch(&event_parity.status) {
                    event_parity_mismatches += 1;
                }
                let trace = TraceArtifact {
                    axis_path: axis.path.clone(),
                    sample_index,
                    sample_value,
                    cause: cause.clone().unwrap_or_else(|| "UNKNOWN".to_string()),
                    trace_symbol_requested: args.trace_symbol.clone(),
                    trace_symbol_index: trace_symbol_idx,
                    trace_symbol_name,
                    cpu: cpu.clone(),
                    gpu: CompactResult {
                        total_trades: trace_gpu.total_trades,
                        final_balance: trace_gpu.final_balance,
                        total_pnl: trace_gpu.total_pnl,
                    },
                    trace_count: trace_state.trace_count,
                    trace_head: trace_state.trace_head,
                    events: gpu_events,
                    cpu_event_total: cpu_trace.total,
                    cpu_event_start: cpu_trace.start,
                    cpu_event_end: cpu_trace.end,
                    cpu_events: cpu_trace.events,
                    event_parity,
                };
                let trace_path = build_trace_path(
                    &args.trace_dir,
                    &axis.path,
                    sample_index,
                    sample_value,
                    cause.as_deref().unwrap_or("UNKNOWN"),
                );
                write_json_pretty(&trace_path, &trace)?;
                trace_artifact = Some(trace_path.display().to_string());
            }

            if pass {
                passed += 1;
            } else {
                failed += 1;
                let key = cause.clone().unwrap_or_else(|| "UNKNOWN".to_string());
                *failures_by_cause.entry(key).or_insert(0) += 1;
            }

            let row = AxisLedgerRow {
                axis_index,
                axis_path: axis.path.clone(),
                sample_index,
                sample_value,
                status: if pass {
                    "PASS".to_string()
                } else {
                    "FAIL".to_string()
                },
                cause,
                cpu,
                gpu: CompactResult {
                    total_trades: gpu.total_trades,
                    final_balance: gpu.final_balance,
                    total_pnl: gpu.total_pnl,
                },
                trade_delta,
                balance_delta_abs,
                pnl_delta_abs,
                trace_artifact,
                event_parity: event_parity_summary,
            };
            writeln!(out, "{}", serde_json::to_string(&row)?)?;
        }
    }
    out.flush()?;

    eprintln!(
        "[axis-parity] completed={} passed={} failed={} event_parity_mismatches={} failures_by_cause={:?}",
        total, passed, failed, event_parity_mismatches, failures_by_cause
    );
    eprintln!("[axis-parity] ledger={}", args.output.display());

    if args.fail_on_event_parity_mismatch && event_parity_mismatches > 0 {
        std::process::exit(3);
    }
    if failed > 0 {
        std::process::exit(2);
    }
    Ok(())
}

fn run_cpu_single(
    cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Result<CompactResult, Box<dyn std::error::Error>> {
    let rows = bt_core::sweep::run_sweep(cfg, spec, candles, None, None, None, from_ts, to_ts);
    let first = rows
        .first()
        .ok_or("cpu sweep returned no rows for single config")?;
    Ok(CompactResult {
        total_trades: first.report.total_trades,
        final_balance: first.report.final_balance,
        total_pnl: first.report.total_pnl,
    })
}

fn run_cpu_single_with_trade_events(
    cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    symbol_filter: Option<&str>,
    limit: usize,
    from_tail: bool,
) -> Result<CpuTraceWindow, Box<dyn std::error::Error>> {
    if limit == 0 {
        return Ok(CpuTraceWindow {
            total: 0,
            start: 0,
            end: 0,
            events: Vec::new(),
        });
    }
    let sim = bt_core::engine::run_simulation(bt_core::engine::RunSimulationInput {
        candles,
        cfg,
        initial_balance: spec.initial_balance,
        lookback: spec.lookback,
        exit_candles: None,
        entry_candles: None,
        funding_rates: None,
        init_state: None,
        from_ts,
        to_ts,
    });
    let mut filtered: Vec<(usize, bt_core::position::TradeRecord)> = Vec::new();
    for (global_idx, trade) in sim.trades.into_iter().enumerate() {
        if let Some(sym) = symbol_filter {
            if trade.symbol != sym {
                continue;
            }
        }
        filtered.push((global_idx, trade));
    }

    let total = filtered.len();
    if total == 0 {
        return Ok(CpuTraceWindow {
            total: 0,
            start: 0,
            end: 0,
            events: Vec::new(),
        });
    }

    let start = if from_tail {
        total.saturating_sub(limit)
    } else {
        0usize
    };
    let end = if from_tail { total } else { limit.min(total) };
    let mut out = Vec::with_capacity(end - start);
    for (window_idx, (global_idx, trade)) in filtered
        .into_iter()
        .enumerate()
        .skip(start)
        .take(end - start)
    {
        let t_sec = if trade.timestamp_ms <= 0 {
            0u32
        } else {
            (trade.timestamp_ms / 1000) as u32
        };
        let decision = canonical_cpu_decision_action(&trade.action);
        let envelope = derive_decision_event_envelope(&decision.action_kind, &trade.reason);
        out.push(CpuTradeEventRow {
            idx: window_idx - start,
            global_idx,
            t_sec,
            symbol: trade.symbol,
            action: trade.action,
            action_kind: decision.action_kind,
            action_side: decision.action_side,
            intent_signal: decision.intent_signal,
            action_taken: decision.action_taken,
            event_type: envelope.event_type,
            status: envelope.status,
            decision_phase: envelope.decision_phase,
            triggered_by: envelope.triggered_by,
            reason: trade.reason,
            price: trade.price,
            size: trade.size,
            pnl: trade.pnl,
            balance: trade.balance,
        });
    }
    Ok(CpuTraceWindow {
        total,
        start,
        end,
        events: out,
    })
}

fn run_gpu_single(
    cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Result<CompactResult, Box<dyn std::error::Error>> {
    let rows = run_gpu_sweep(candles, cfg, spec, None, None, from_ts, to_ts);
    let first = rows.first().ok_or(
        "gpu sweep returned no rows for single config. \
likely no CUDA-capable device (or driver visibility failure) in this runtime; \
verify with `nvidia-smi` in the same environment",
    )?;
    Ok(CompactResult {
        total_trades: first.total_trades,
        final_balance: first.final_balance,
        total_pnl: first.total_pnl,
    })
}

fn run_gpu_single_with_trace(
    cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    trace_symbol_idx: u32,
) -> Result<(CompactResult, GpuComboState, Vec<String>), Box<dyn std::error::Error>> {
    with_trace_env(0, trace_symbol_idx, || {
        let (rows, states, symbols) =
            run_gpu_sweep_with_states(candles, cfg, spec, None, None, from_ts, to_ts);
        let first = rows.first().ok_or(
            "gpu trace sweep returned no rows for single config. \
likely no CUDA-capable device (or driver visibility failure) in this runtime; \
verify with `nvidia-smi` in the same environment",
        )?;
        let state = states.first().copied().ok_or(
            "gpu trace sweep returned no state for single config; \
gpu runtime likely unavailable in this environment",
        )?;
        Ok((
            CompactResult {
                total_trades: first.total_trades,
                final_balance: first.final_balance,
                total_pnl: first.total_pnl,
            },
            state,
            symbols,
        ))
    })
}

fn with_trace_env<T, E, F>(combo_idx: usize, symbol_idx: u32, f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    const KEYS: [&str; 3] = [
        "AQC_GPU_TRACE",
        "AQC_GPU_TRACE_COMBO",
        "AQC_GPU_TRACE_SYMBOL",
    ];
    let mut prev: Vec<(&str, Option<OsString>)> = Vec::with_capacity(KEYS.len());
    for key in KEYS {
        prev.push((key, std::env::var_os(key)));
    }

    std::env::set_var("AQC_GPU_TRACE", "1");
    std::env::set_var("AQC_GPU_TRACE_COMBO", combo_idx.to_string());
    std::env::set_var("AQC_GPU_TRACE_SYMBOL", symbol_idx.to_string());

    let out = f();

    for (key, val) in prev {
        if let Some(v) = val {
            std::env::set_var(key, v);
        } else {
            std::env::remove_var(key);
        }
    }
    out
}

fn select_sample_indices(values: &[f64], max_values: Option<usize>) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let limit = max_values.unwrap_or(n).max(1).min(n);
    if limit == n {
        return (0..n).collect();
    }

    let mut idxs = BTreeSet::new();
    if limit == 1 {
        idxs.insert(0usize);
    } else {
        for i in 0..limit {
            let pos = (i as f64) * ((n - 1) as f64) / ((limit - 1) as f64);
            idxs.insert(pos.round() as usize);
        }
    }
    idxs.into_iter().collect()
}

fn sorted_symbols_with_cap(candles: &CandleData) -> Vec<String> {
    let mut symbols: Vec<String> = candles.keys().cloned().collect();
    symbols.sort();
    if symbols.len() > GPU_MAX_SYMBOLS {
        symbols.truncate(GPU_MAX_SYMBOLS);
    }
    symbols
}

fn resolve_trace_symbol_index(requested: Option<&str>, symbols: &[String]) -> u32 {
    let Some(name) = requested else {
        return GPU_TRACE_SYMBOL_ALL;
    };
    symbols
        .iter()
        .position(|s| s == name)
        .map(|i| i as u32)
        .unwrap_or(GPU_TRACE_SYMBOL_ALL)
}

fn symbol_name(sym_idx: u32, symbols: &[String]) -> Option<String> {
    if sym_idx == GPU_TRACE_SYMBOL_ALL {
        return None;
    }
    symbols.get(sym_idx as usize).cloned()
}

fn collect_trace_events(state: &GpuComboState, symbols: &[String]) -> Vec<TraceEventRow> {
    let count = (state.trace_count as usize).min(GPU_TRACE_CAP);
    if count == 0 {
        return Vec::new();
    }

    let head = state.trace_head as usize;
    let start = head.saturating_sub(count);
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let ring_idx = (start + idx) % GPU_TRACE_CAP;
        let ev = state.trace_events[ring_idx];
        out.push(TraceEventRow {
            idx,
            t_sec: ev.t_sec,
            sym_idx: ev.sym,
            symbol: symbols.get(ev.sym as usize).cloned(),
            kind: trace_kind_name(ev.kind).to_string(),
            side: trace_side_name(ev.side).to_string(),
            reason: trace_reason_name(ev.reason).to_string(),
            price: ev.price,
            size: ev.size,
            pnl: ev.pnl,
        });
    }
    out
}

fn compare_event_streams(
    cpu_events: &[CpuTradeEventRow],
    gpu_events: &[TraceEventRow],
    include_mismatch_fields: bool,
    max_offset_scan: usize,
    allow_absolute_scan: bool,
    event_tol: EventNumericTolerance,
) -> EventParitySummary {
    let cpu: Vec<CanonicalEventRow> = cpu_events.iter().map(canonicalise_cpu_event).collect();
    let gpu: Vec<CanonicalEventRow> = gpu_events.iter().map(canonicalise_gpu_event).collect();
    let (cpu_tail_offset, gpu_tail_offset, aligned_len) =
        choose_alignment_offsets(&cpu, &gpu, max_offset_scan, allow_absolute_scan, event_tol);

    for rel_idx in 0..aligned_len {
        let cpu_ev = &cpu[cpu_tail_offset + rel_idx];
        let gpu_ev = &gpu[gpu_tail_offset + rel_idx];
        if !canonical_events_equal(cpu_ev, gpu_ev, event_tol) {
            return EventParitySummary {
                status: "MISMATCH".to_string(),
                aligned_len,
                cpu_len: cpu.len(),
                gpu_len: gpu.len(),
                cpu_tail_offset,
                gpu_tail_offset,
                first_mismatch_at: Some(rel_idx),
                first_mismatch_fields: if include_mismatch_fields {
                    Some(diff_canonical_fields(cpu_ev, gpu_ev, event_tol))
                } else {
                    None
                },
                cpu_event: Some(cpu_ev.clone()),
                gpu_event: Some(gpu_ev.clone()),
            };
        }
    }

    if cpu.len() != gpu.len() || cpu_tail_offset != 0 || gpu_tail_offset != 0 {
        return EventParitySummary {
            status: "LENGTH_MISMATCH".to_string(),
            aligned_len,
            cpu_len: cpu.len(),
            gpu_len: gpu.len(),
            cpu_tail_offset,
            gpu_tail_offset,
            first_mismatch_at: None,
            first_mismatch_fields: None,
            cpu_event: None,
            gpu_event: None,
        };
    }

    EventParitySummary {
        status: "MATCH".to_string(),
        aligned_len,
        cpu_len: cpu.len(),
        gpu_len: gpu.len(),
        cpu_tail_offset,
        gpu_tail_offset,
        first_mismatch_at: None,
        first_mismatch_fields: None,
        cpu_event: None,
        gpu_event: None,
    }
}

fn choose_alignment_offsets(
    cpu: &[CanonicalEventRow],
    gpu: &[CanonicalEventRow],
    max_offset_scan: usize,
    allow_absolute_scan: bool,
    event_tol: EventNumericTolerance,
) -> (usize, usize, usize) {
    let base_aligned = cpu.len().min(gpu.len());
    let base_cpu_off = cpu.len().saturating_sub(base_aligned);
    let base_gpu_off = gpu.len().saturating_sub(base_aligned);
    if base_aligned == 0 {
        return (base_cpu_off, base_gpu_off, 0);
    }

    if cpu.len() == gpu.len() {
        return (0, 0, base_aligned);
    }

    if max_offset_scan == 0 {
        return (base_cpu_off, base_gpu_off, base_aligned);
    }

    let mut best_cpu_off = base_cpu_off;
    let mut best_gpu_off = base_gpu_off;
    let mut best_aligned = base_aligned;
    let mut best_prefix = matching_prefix_len(
        cpu,
        base_cpu_off,
        gpu,
        base_gpu_off,
        base_aligned,
        event_tol,
    );

    if allow_absolute_scan {
        let cpu_scan_max = cpu.len().min(max_offset_scan);
        let gpu_scan_max = gpu.len().min(max_offset_scan);
        for cpu_off in 0..=cpu_scan_max {
            for gpu_off in 0..=gpu_scan_max {
                let aligned = cpu
                    .len()
                    .saturating_sub(cpu_off)
                    .min(gpu.len().saturating_sub(gpu_off));
                if aligned == 0 {
                    continue;
                }
                let prefix = matching_prefix_len(cpu, cpu_off, gpu, gpu_off, aligned, event_tol);
                let better_prefix = prefix > best_prefix;
                let better_coverage = prefix == best_prefix && aligned > best_aligned;
                if better_prefix || better_coverage {
                    best_prefix = prefix;
                    best_aligned = aligned;
                    best_cpu_off = cpu_off;
                    best_gpu_off = gpu_off;
                }
            }
        }
    } else {
        let scan_cap = base_aligned.min(max_offset_scan);
        for shared_trim in 1..=scan_cap {
            let cpu_off = base_cpu_off + shared_trim;
            let gpu_off = base_gpu_off + shared_trim;
            let aligned = base_aligned - shared_trim;
            if aligned == 0 {
                break;
            }
            let prefix = matching_prefix_len(cpu, cpu_off, gpu, gpu_off, aligned, event_tol);
            let better_prefix = prefix > best_prefix;
            let better_coverage = prefix == best_prefix && aligned > best_aligned;
            if better_prefix || better_coverage {
                best_prefix = prefix;
                best_aligned = aligned;
                best_cpu_off = cpu_off;
                best_gpu_off = gpu_off;
            }
        }
    }

    (best_cpu_off, best_gpu_off, best_aligned)
}

fn matching_prefix_len(
    cpu: &[CanonicalEventRow],
    cpu_offset: usize,
    gpu: &[CanonicalEventRow],
    gpu_offset: usize,
    aligned_len: usize,
    event_tol: EventNumericTolerance,
) -> usize {
    let mut count = 0usize;
    for idx in 0..aligned_len {
        if canonical_events_equal(&cpu[cpu_offset + idx], &gpu[gpu_offset + idx], event_tol) {
            count += 1;
        } else {
            break;
        }
    }
    count
}

fn summarise_event_parity(summary: &EventParitySummary) -> LedgerEventParitySummary {
    LedgerEventParitySummary {
        status: summary.status.clone(),
        aligned_len: summary.aligned_len,
        cpu_len: summary.cpu_len,
        gpu_len: summary.gpu_len,
        cpu_tail_offset: summary.cpu_tail_offset,
        gpu_tail_offset: summary.gpu_tail_offset,
        first_mismatch_at: summary.first_mismatch_at,
        first_mismatch_fields: summary.first_mismatch_fields.clone(),
    }
}

fn is_event_parity_mismatch(status: &str) -> bool {
    matches!(status, "MISMATCH" | "LENGTH_MISMATCH")
}

fn canonicalise_cpu_event(ev: &CpuTradeEventRow) -> CanonicalEventRow {
    let reason_code = canonical_cpu_reason_code(&ev.action_kind, &ev.reason);
    CanonicalEventRow {
        source_idx: ev.idx,
        global_idx: Some(ev.global_idx),
        t_sec: ev.t_sec,
        symbol: ev.symbol.clone(),
        action: ev.action.clone(),
        action_kind: ev.action_kind.clone(),
        action_side: ev.action_side.clone(),
        intent_signal: ev.intent_signal.clone(),
        action_taken: ev.action_taken.clone(),
        event_type: ev.event_type.clone(),
        status: ev.status.clone(),
        decision_phase: ev.decision_phase.clone(),
        triggered_by: ev.triggered_by.clone(),
        reason: ev.reason.clone(),
        reason_code,
        price: ev.price,
        size: ev.size,
        pnl: ev.pnl,
    }
}

fn canonicalise_gpu_event(ev: &TraceEventRow) -> CanonicalEventRow {
    let decision = canonical_gpu_decision_action(&ev.kind, &ev.side);
    let envelope = derive_decision_event_envelope(&decision.action_kind, &ev.reason);
    let reason_code = canonical_gpu_reason_code(&ev.reason);
    CanonicalEventRow {
        source_idx: ev.idx,
        global_idx: None,
        t_sec: ev.t_sec,
        symbol: ev
            .symbol
            .clone()
            .unwrap_or_else(|| format!("SYM#{}", ev.sym_idx)),
        action: canonical_gpu_action(&ev.kind, &ev.side),
        action_kind: decision.action_kind,
        action_side: decision.action_side,
        intent_signal: decision.intent_signal,
        action_taken: decision.action_taken,
        event_type: envelope.event_type,
        status: envelope.status,
        decision_phase: envelope.decision_phase,
        triggered_by: envelope.triggered_by,
        reason: ev.reason.clone(),
        reason_code,
        price: ev.price as f64,
        size: ev.size as f64,
        pnl: ev.pnl,
    }
}

fn canonical_cpu_decision_action(action: &str) -> DecisionActionCanonical {
    let raw = action.trim().to_uppercase();
    if raw == "FUNDING" {
        return DecisionActionCanonical {
            action_kind: "FUNDING".to_string(),
            action_side: "EMPTY".to_string(),
            intent_signal: "NEUTRAL".to_string(),
            action_taken: "funding".to_string(),
        };
    }

    let (kind, side) = match raw.rsplit_once('_') {
        Some((k, s)) if s == "LONG" || s == "SHORT" => (k.to_string(), s.to_string()),
        _ => (raw.clone(), "EMPTY".to_string()),
    };

    let action_kind = normalise_decision_kind(&kind);
    let intent_signal = decision_signal_for(&action_kind, &side).to_string();
    let action_taken = decision_action_taken(&action_kind, &side);

    DecisionActionCanonical {
        action_kind,
        action_side: side,
        intent_signal,
        action_taken,
    }
}

fn canonical_gpu_decision_action(kind: &str, side: &str) -> DecisionActionCanonical {
    let action_kind = normalise_decision_kind(kind);
    let action_side = side.trim().to_uppercase();
    let intent_signal = decision_signal_for(&action_kind, &action_side).to_string();
    let action_taken = decision_action_taken(&action_kind, &action_side);
    DecisionActionCanonical {
        action_kind,
        action_side,
        intent_signal,
        action_taken,
    }
}

fn normalise_decision_kind(kind: &str) -> String {
    let upper = kind.trim().to_uppercase();
    match upper.as_str() {
        "PARTIAL_CLOSE" => "REDUCE".to_string(),
        _ => upper,
    }
}

fn decision_signal_for(kind: &str, side: &str) -> &'static str {
    match (kind, side) {
        ("OPEN", "LONG") | ("ADD", "LONG") => "BUY",
        ("OPEN", "SHORT") | ("ADD", "SHORT") => "SELL",
        ("CLOSE", "LONG") | ("REDUCE", "LONG") => "SELL",
        ("CLOSE", "SHORT") | ("REDUCE", "SHORT") => "BUY",
        _ => "NEUTRAL",
    }
}

fn decision_action_taken(kind: &str, side: &str) -> String {
    if side == "LONG" || side == "SHORT" {
        return format!("{}_{}", kind.to_lowercase(), side.to_lowercase());
    }
    kind.to_lowercase()
}

fn derive_decision_event_envelope(action_kind: &str, raw_reason: &str) -> DecisionEventEnvelope {
    let reason = raw_reason.trim().to_uppercase();
    let triggered_by = if action_kind == "FUNDING" {
        "schedule"
    } else if reason.contains("SIGNAL_FLIP") || reason.contains("SIGNAL FLIP") {
        "signal_flip"
    } else if reason.contains("STOP LOSS") || reason.contains("TRAILING STOP") {
        "stop_loss"
    } else if action_kind == "OPEN"
        || action_kind == "ADD"
        || reason.contains("ENTRY")
        || reason.contains("PYRAMID")
    {
        "schedule"
    } else {
        "price_update"
    };

    DecisionEventEnvelope {
        event_type: if action_kind == "FUNDING" {
            "funding".to_string()
        } else {
            "fill".to_string()
        },
        status: "executed".to_string(),
        decision_phase: "execution".to_string(),
        triggered_by: triggered_by.to_string(),
    }
}

fn canonical_cpu_reason_code(action_kind: &str, raw_reason: &str) -> String {
    let reason_upper = raw_reason.trim().to_uppercase();
    if reason_upper.contains("SIGNAL FLIP") {
        return "SIGNAL_FLIP".to_string();
    }
    match action_kind {
        "OPEN" => "ENTRY".to_string(),
        "ADD" => "PYRAMID".to_string(),
        "CLOSE" => "EXIT".to_string(),
        "REDUCE" => "PARTIAL".to_string(),
        "FUNDING" => "FUNDING".to_string(),
        _ => "UNKNOWN".to_string(),
    }
}

fn canonical_gpu_reason_code(raw_reason: &str) -> String {
    let reason = raw_reason.trim().to_uppercase();
    if reason.is_empty() {
        return "UNKNOWN".to_string();
    }
    if reason.starts_with("EXIT") {
        return "EXIT".to_string();
    }
    reason
}

fn canonical_gpu_action(kind: &str, side: &str) -> String {
    if side == "EMPTY" {
        return kind.to_string();
    }
    format!("{kind}_{side}")
}

fn canonical_events_equal(
    a: &CanonicalEventRow,
    b: &CanonicalEventRow,
    event_tol: EventNumericTolerance,
) -> bool {
    a.t_sec == b.t_sec
        && a.symbol == b.symbol
        && a.action_kind == b.action_kind
        && a.action_side == b.action_side
        && a.intent_signal == b.intent_signal
        && a.event_type == b.event_type
        && a.status == b.status
        && a.decision_phase == b.decision_phase
        && a.reason_code == b.reason_code
        && numeric_eq(a.price, b.price, event_tol)
        && numeric_eq(a.size, b.size, event_tol)
        && numeric_eq(a.pnl, b.pnl, event_tol)
}

fn diff_canonical_fields(
    a: &CanonicalEventRow,
    b: &CanonicalEventRow,
    event_tol: EventNumericTolerance,
) -> Vec<String> {
    let mut diff = Vec::new();
    if a.t_sec != b.t_sec {
        diff.push("t_sec".to_string());
    }
    if a.symbol != b.symbol {
        diff.push("symbol".to_string());
    }
    if a.action_kind != b.action_kind {
        diff.push("action_kind".to_string());
    }
    if a.action_side != b.action_side {
        diff.push("action_side".to_string());
    }
    if a.intent_signal != b.intent_signal {
        diff.push("intent_signal".to_string());
    }
    if a.event_type != b.event_type {
        diff.push("event_type".to_string());
    }
    if a.status != b.status {
        diff.push("status".to_string());
    }
    if a.decision_phase != b.decision_phase {
        diff.push("decision_phase".to_string());
    }
    if a.reason_code != b.reason_code {
        diff.push("reason_code".to_string());
    }
    if !numeric_eq(a.price, b.price, event_tol) {
        diff.push("price".to_string());
    }
    if !numeric_eq(a.size, b.size, event_tol) {
        diff.push("size".to_string());
    }
    if !numeric_eq(a.pnl, b.pnl, event_tol) {
        diff.push("pnl".to_string());
    }
    diff
}

fn numeric_eq(a: f64, b: f64, tol: EventNumericTolerance) -> bool {
    if !a.is_finite() || !b.is_finite() || !tol.abs.is_finite() || !tol.rel.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    if diff <= tol.abs {
        return true;
    }
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        return true;
    }
    diff <= tol.rel * scale
}

fn trace_kind_name(kind: u32) -> &'static str {
    match kind {
        1 => "OPEN",
        2 => "ADD",
        3 => "CLOSE",
        4 => "PARTIAL_CLOSE",
        _ => "UNKNOWN",
    }
}

fn trace_side_name(side: u32) -> &'static str {
    match side {
        1 => "LONG",
        2 => "SHORT",
        _ => "EMPTY",
    }
}

fn trace_reason_name(reason: u32) -> &'static str {
    match reason {
        1 => "ENTRY",
        2 => "PYRAMID",
        3 => "EXIT_STOP",
        4 => "EXIT_TRAILING",
        5 => "EXIT_TP",
        6 => "EXIT_SMART",
        7 => "SIGNAL_FLIP",
        8 => "PARTIAL",
        9 => "EXIT_EOB",
        _ => "UNKNOWN",
    }
}

fn build_trace_path(
    trace_dir: &Path,
    axis_path: &str,
    sample_index: usize,
    sample_value: f64,
    cause: &str,
) -> PathBuf {
    let axis_tag = axis_path
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '_',
        })
        .collect::<String>();
    let value_tag = format!("{sample_value:.8}")
        .replace('-', "m")
        .replace('.', "_");
    trace_dir.join(format!(
        "{axis_tag}__sample{sample_index}__{value_tag}__{cause}.json"
    ))
}

fn write_json_pretty<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
