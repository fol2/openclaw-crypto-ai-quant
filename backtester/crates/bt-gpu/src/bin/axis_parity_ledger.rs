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
    pnl: f32,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("[axis-parity] ERROR: {err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let base_cfg = bt_core::config::load_config(&args.config, None, false);
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
    let sorted_symbols = sorted_symbols_with_cap(&candles);
    let trace_symbol_idx = resolve_trace_symbol_index(args.trace_symbol.as_deref(), &sorted_symbols);

    let axes: Vec<_> = match args.max_axes {
        Some(limit) => spec.axes.iter().take(limit).cloned().collect(),
        None => spec.axes.clone(),
    };
    let single_spec = SweepSpec {
        axes: vec![],
        initial_balance: spec.initial_balance,
        lookback: spec.lookback,
    };

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
            if !pass && args.trace_on_failure {
                let (trace_gpu, trace_state, trace_symbols) = run_gpu_single_with_trace(
                    &cfg,
                    &single_spec,
                    &candles,
                    args.from_ts,
                    args.to_ts,
                    trace_symbol_idx,
                )?;
                let trace = TraceArtifact {
                    axis_path: axis.path.clone(),
                    sample_index,
                    sample_value,
                    cause: cause.clone().unwrap_or_else(|| "UNKNOWN".to_string()),
                    trace_symbol_requested: args.trace_symbol.clone(),
                    trace_symbol_index: trace_symbol_idx,
                    trace_symbol_name: symbol_name(trace_symbol_idx, &trace_symbols),
                    cpu: cpu.clone(),
                    gpu: CompactResult {
                        total_trades: trace_gpu.total_trades,
                        final_balance: trace_gpu.final_balance,
                        total_pnl: trace_gpu.total_pnl,
                    },
                    trace_count: trace_state.trace_count,
                    trace_head: trace_state.trace_head,
                    events: collect_trace_events(&trace_state, &trace_symbols),
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
                status: if pass { "PASS".to_string() } else { "FAIL".to_string() },
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
            };
            writeln!(out, "{}", serde_json::to_string(&row)?)?;
        }
    }
    out.flush()?;

    eprintln!(
        "[axis-parity] completed={} passed={} failed={} failures_by_cause={:?}",
        total, passed, failed, failures_by_cause
    );
    eprintln!("[axis-parity] ledger={}", args.output.display());

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

fn run_gpu_single(
    cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Result<CompactResult, Box<dyn std::error::Error>> {
    let rows = run_gpu_sweep(candles, cfg, spec, None, None, from_ts, to_ts);
    let first = rows
        .first()
        .ok_or("gpu sweep returned no rows for single config")?;
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
        let (rows, states, symbols) = run_gpu_sweep_with_states(
            candles, cfg, spec, None, None, from_ts, to_ts,
        );
        let first = rows
            .first()
            .ok_or("gpu trace sweep returned no rows for single config")?;
        let state = states
            .first()
            .copied()
            .ok_or("gpu trace sweep returned no state for single config")?;
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
    const KEYS: [&str; 3] = ["AQC_GPU_TRACE", "AQC_GPU_TRACE_COMBO", "AQC_GPU_TRACE_SYMBOL"];
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
        3 => "EXIT",
        4 => "SIGNAL_FLIP",
        5 => "PARTIAL",
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

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
