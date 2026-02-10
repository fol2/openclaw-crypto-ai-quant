//! CLI entry point for the mei-backtester Rust backtesting simulator.
//!
//! Subcommands:
//!   - `replay`          — Run a single backtest against historical candles
//!   - `sweep`           — Parallel parameter sweep (Cartesian product of axes)
//!   - `dump-indicators` — Dump raw indicator values as CSV for debugging

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

const VERSION: &str = "0.1.0";
const DEFAULT_LOOKBACK: usize = 200;

/// Read only the `balance` field from an export_state.py JSON file.
/// Panics with a descriptive message on any I/O or parse error.
fn read_balance_from_json(path: &str) -> f64 {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| { eprintln!("[error] Cannot read {:?}: {}", path, e); std::process::exit(1); });
    let json: serde_json::Value = serde_json::from_str(&data)
        .unwrap_or_else(|e| { eprintln!("[error] Invalid JSON in {:?}: {}", path, e); std::process::exit(1); });
    let balance = json.get("balance")
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| { eprintln!("[error] {:?} has no numeric \"balance\" field", path); std::process::exit(1); });
    eprintln!("[balance-from] Read balance=${:.2} from {:?}", balance, path);
    balance
}

// ---------------------------------------------------------------------------
// CLI argument structs
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "mei-backtester",
    version = VERSION,
    about = "High-performance backtesting simulator for the Mei Alpha strategy",
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single backtest
    Replay(ReplayArgs),
    /// Run a parallel parameter sweep
    Sweep(SweepArgs),
    /// Dump indicator values for debugging/validation
    DumpIndicators(DumpArgs),
}

#[derive(Parser)]
struct ReplayArgs {
    /// Path to the strategy YAML config
    #[arg(long, default_value = "strategy_overrides.yaml")]
    config: String,

    /// Path to the SQLite candle database.
    /// If omitted, auto-resolved from --interval: candles_dbs/candles_{interval}.db
    #[arg(long)]
    candles_db: Option<String>,

    /// Candle interval for indicator computation and entry signals (e.g. "1h", "15m", "5m").
    /// If omitted, reads from YAML engine.interval (default "1h").
    /// The DB path is auto-resolved to candles_dbs/candles_{interval}.db unless --candles-db is set.
    #[arg(long)]
    interval: Option<String>,

    /// Starting account balance in USD
    #[arg(long, default_value_t = 10_000.0)]
    initial_balance: f64,

    /// Indicator warmup bars before trading begins
    #[arg(long, default_value_t = DEFAULT_LOOKBACK)]
    lookback: usize,

    /// Write JSON report to this file instead of stdout
    #[arg(long)]
    output: Option<PathBuf>,

    /// Export trade rows as CSV for analysis (deterministic ordering, includes trade_id).
    ///
    /// The export is one row per exit event (CLOSE_* / REDUCE_*), with entry fields repeated.
    #[arg(long)]
    export_trades: Option<PathBuf>,

    /// Only backtest this single symbol (otherwise all symbols in DB)
    #[arg(long)]
    symbol: Option<String>,

    /// Include the full trade log in the JSON output
    #[arg(long, default_value_t = false)]
    trades: bool,

    /// Include the equity curve in the JSON output
    #[arg(long, default_value_t = false)]
    equity_curve: bool,

    /// Apply the `live` config overlay on top of global + per-symbol
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Path to a secondary SQLite candle database for higher-resolution exit checks.
    /// If omitted but --exit-interval is set, auto-resolved to candles_dbs/candles_{exit_interval}.db.
    /// When active, exit conditions are evaluated on these sub-bars within each indicator bar,
    /// aligning backtest exits with live behavior.
    #[arg(long)]
    exit_candles_db: Option<String>,

    /// Candle interval for exit checks (e.g. "1m", "3m", "5m").
    /// If omitted, reads from YAML engine.exit_interval. Empty string = disabled.
    /// Must be a shorter interval than --interval for meaningful sub-bar precision.
    #[arg(long)]
    exit_interval: Option<String>,

    /// Path to a funding rate SQLite database. When provided, funding payments
    /// are applied at hourly boundaries for open positions.
    #[arg(long)]
    funding_db: Option<String>,

    /// Path to a secondary SQLite candle database for higher-resolution entry checks.
    /// If omitted but --entry-interval is set, auto-resolved to candles_dbs/candles_{entry_interval}.db.
    #[arg(long)]
    entry_candles_db: Option<String>,

    /// Candle interval for entry checks (e.g. "15m", "5m").
    /// If omitted, reads from YAML engine.entry_interval. Empty string = disabled.
    /// Entry signals are evaluated at this resolution using indicator-bar indicators + sub-bar price.
    #[arg(long)]
    entry_interval: Option<String>,

    /// Start timestamp in milliseconds (inclusive). Only trade on bars >= this time.
    #[arg(long)]
    from_ts: Option<i64>,

    /// End timestamp in milliseconds (inclusive). Only trade on bars <= this time.
    #[arg(long)]
    to_ts: Option<i64>,

    /// Disable auto-scoping. By default, replay auto-scopes all candle DBs to the
    /// shortest overlapping time range for apple-to-apple comparison.
    /// Use --no-auto-scope to disable this and rely on explicit --from-ts / --to-ts.
    #[arg(long, default_value_t = false)]
    no_auto_scope: bool,

    /// Path to a JSON file exported by export_state.py. When provided, the simulation
    /// starts with the exported balance and open positions instead of a blank slate.
    /// Overrides --initial-balance.
    #[arg(long)]
    init_state: Option<String>,

    /// Read initial balance from an export_state.py JSON file (ignores positions).
    /// Overrides --initial-balance.
    #[arg(long)]
    balance_from: Option<String>,
}

#[derive(Parser)]
struct SweepArgs {
    /// Path to the strategy YAML config (base config for sweep)
    #[arg(long, default_value = "strategy_overrides.yaml")]
    config: String,

    /// Path to the SQLite candle database.
    /// If omitted, auto-resolved from --interval: candles_dbs/candles_{interval}.db
    #[arg(long)]
    candles_db: Option<String>,

    /// Candle interval (e.g. "1h", "15m"). If omitted, reads from YAML engine.interval.
    #[arg(long)]
    interval: Option<String>,

    /// Path to the YAML file defining sweep axes
    #[arg(long)]
    sweep_spec: String,

    /// Output file for JSONL results (one JSON object per line per combo)
    #[arg(long, default_value = "sweep_results.jsonl")]
    output: PathBuf,

    /// Starting account balance in USD
    #[arg(long, default_value_t = 10_000.0)]
    initial_balance: f64,

    /// Override rayon's thread count (defaults to num CPUs)
    #[arg(long)]
    threads: Option<usize>,

    /// Only print the top N results sorted by total PnL
    #[arg(long)]
    top_n: Option<usize>,

    /// Apply the `live` config overlay on top of global
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Path to exit candle DB. Auto-resolved from --exit-interval if omitted.
    #[arg(long)]
    exit_candles_db: Option<String>,

    /// Exit check interval (e.g. "1m", "5m"). If omitted, reads from YAML engine.exit_interval.
    #[arg(long)]
    exit_interval: Option<String>,

    /// Path to entry candle DB. Auto-resolved from --entry-interval if omitted.
    #[arg(long)]
    entry_candles_db: Option<String>,

    /// Entry check interval (e.g. "15m", "5m"). If omitted, reads from YAML engine.entry_interval.
    #[arg(long)]
    entry_interval: Option<String>,

    /// Path to funding rate SQLite database.
    #[arg(long)]
    funding_db: Option<String>,

    /// Start timestamp in milliseconds (inclusive). Only trade on bars >= this time.
    #[arg(long)]
    from_ts: Option<i64>,

    /// End timestamp in milliseconds (inclusive). Only trade on bars <= this time.
    #[arg(long)]
    to_ts: Option<i64>,

    /// Disable auto-scoping. By default, sweep auto-scopes all candle DBs to the
    /// shortest overlapping time range for apple-to-apple comparison.
    /// Use --no-auto-scope to disable this and rely on explicit --from-ts / --to-ts.
    #[arg(long, default_value_t = false)]
    no_auto_scope: bool,

    /// Read initial balance from an export_state.py JSON file (ignores positions).
    /// Overrides --initial-balance.
    #[arg(long)]
    balance_from: Option<String>,

    /// Use GPU-accelerated sweep (requires building with `--features gpu`)
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Use TPE (Bayesian optimization) instead of grid search.
    /// Requires --gpu. Samples intelligently from the parameter ranges.
    #[arg(long, default_value_t = false)]
    tpe: bool,

    /// Number of TPE trials to evaluate (default: 5000)
    #[arg(long, default_value_t = 5000)]
    tpe_trials: usize,

    /// Number of trials per GPU batch for TPE (default: 256)
    #[arg(long, default_value_t = 256)]
    tpe_batch: usize,

    /// RNG seed for TPE reproducibility (default: 42)
    #[arg(long, default_value_t = 42)]
    tpe_seed: u64,
}

#[derive(Parser)]
struct DumpArgs {
    /// Path to the strategy YAML config
    #[arg(long, default_value = "strategy_overrides.yaml")]
    config: String,

    /// Path to the SQLite candle database.
    /// If omitted, auto-resolved from --interval: candles_dbs/candles_{interval}.db
    #[arg(long)]
    candles_db: Option<String>,

    /// Candle interval (e.g. "1h", "15m"). If omitted, reads from YAML engine.interval.
    #[arg(long)]
    interval: Option<String>,

    /// Symbol to dump indicators for
    #[arg(long)]
    symbol: String,

    /// Write CSV to this file instead of stdout
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Auto-scope: detect the shortest overlapping time range across all candle DBs
// ---------------------------------------------------------------------------

/// Given a list of (db_path, interval) pairs, query each for its time range
/// and return the narrowest overlap: (max of all min_t, min of all max_t).
fn compute_auto_scope(
    dbs: &[(&str, &str)],
    explicit_from: Option<i64>,
    explicit_to: Option<i64>,
) -> (Option<i64>, Option<i64>) {
    let mut global_from: Option<i64> = explicit_from;
    let mut global_to: Option<i64> = explicit_to;

    for (db_path, interval) in dbs {
        match bt_data::sqlite_loader::query_time_range(db_path, interval) {
            Ok(Some((min_t, max_t))) => {
                eprintln!(
                    "[auto-scope] {} ({}): {}..{} ({:.1} days)",
                    db_path, interval, min_t, max_t,
                    (max_t - min_t) as f64 / 86_400_000.0,
                );
                global_from = Some(global_from.map_or(min_t, |f| f.max(min_t)));
                global_to = Some(global_to.map_or(max_t, |t| t.min(max_t)));
            }
            Ok(None) => {
                eprintln!("[auto-scope] {} ({}): empty — skipping", db_path, interval);
            }
            Err(e) => {
                eprintln!("[auto-scope] {} ({}): error querying range: {}", db_path, interval, e);
            }
        }
    }

    if let (Some(f), Some(t)) = (global_from, global_to) {
        eprintln!(
            "[auto-scope] Common range: {}..{} ({:.1} days)",
            f, t, (t - f) as f64 / 86_400_000.0,
        );
    }

    (global_from, global_to)
}

// ---------------------------------------------------------------------------
// Trade export (CSV)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct TradeExportRow {
    trade_id: String,
    position_id: u64,
    entry_ts_ms: i64,
    exit_ts_ms: i64,
    symbol: String,
    side: String,
    entry_price: f64,
    exit_price: f64,
    exit_size: f64,
    pnl_usd: f64,
    fee_usd: f64,
    mae_pct: f64,
    mfe_pct: f64,
    reason_code: String,
    reason: String,
}

#[derive(Debug, Clone, Copy)]
enum Side {
    Long,
    Short,
}

#[derive(Debug, Clone)]
struct ActiveTrade {
    position_id: u64,
    entry_ts_ms: i64,
    entry_price: f64,
    side: Side,
    max_high: f64,
    min_low: f64,
    exit_seq: u32,
    open_count_for_symbol: u32,
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn reason_code_string(code: bt_core::reason_codes::ReasonCode) -> String {
    // ReasonCode serialises to a snake_case string; reuse that canonical representation.
    serde_json::to_string(&code)
        .unwrap_or_else(|_| "\"unknown\"".to_string())
        .trim_matches('"')
        .to_string()
}

fn update_extremes(active: &mut ActiveTrade, bar: &bt_core::candle::OhlcvBar) {
    if bar.h > active.max_high {
        active.max_high = bar.h;
    }
    if bar.l < active.min_low {
        active.min_low = bar.l;
    }
}

fn build_trade_export_rows(
    candles: &bt_core::candle::CandleData,
    trades: &[bt_core::position::TradeRecord],
) -> Vec<TradeExportRow> {
    // Assign a stable position_id per OPEN event in the trade log order.
    let mut next_position_id: u64 = 1;
    let mut open_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut position_ids: BTreeMap<(String, u32), u64> = BTreeMap::new();
    for tr in trades {
        if tr.action.starts_with("OPEN_") {
            let c = open_counts.entry(tr.symbol.clone()).or_insert(0);
            *c += 1;
            position_ids.insert((tr.symbol.clone(), *c), next_position_id);
            next_position_id += 1;
        }
    }

    // Collect relevant events per symbol with stable ordering.
    let mut per_symbol: BTreeMap<String, Vec<(i64, usize)>> = BTreeMap::new();
    for (idx, tr) in trades.iter().enumerate() {
        let action = tr.action.as_str();
        if action.starts_with("OPEN_")
            || action.starts_with("ADD_")
            || action.starts_with("CLOSE_")
            || action.starts_with("REDUCE_")
        {
            per_symbol
                .entry(tr.symbol.clone())
                .or_default()
                .push((tr.timestamp_ms, idx));
        }
    }
    for events in per_symbol.values_mut() {
        events.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    }

    let mut rows: Vec<TradeExportRow> = Vec::new();

    for (symbol, events) in per_symbol {
        let bars = match candles.get(&symbol) {
            Some(b) => b,
            None => continue,
        };

        let mut candle_i: usize = 0;
        let mut active: Option<ActiveTrade> = None;

        for (ts, idx) in events {
            let tr = &trades[idx];

            // Advance candles strictly before this trade timestamp.
            while candle_i < bars.len() && bars[candle_i].t < ts {
                if let Some(ref mut a) = active {
                    update_extremes(a, &bars[candle_i]);
                }
                candle_i += 1;
            }

            // OPEN must start the trade before we include the candle at ts.
            if tr.action.starts_with("OPEN_") {
                let open_count_for_symbol = active
                    .as_ref()
                    .map(|a| a.open_count_for_symbol)
                    .unwrap_or(0)
                    + 1;

                let position_id = position_ids
                    .get(&(symbol.clone(), open_count_for_symbol))
                    .copied()
                    .unwrap_or(0);

                let side = if tr.action.ends_with("_LONG") {
                    Side::Long
                } else {
                    Side::Short
                };

                let mut a = ActiveTrade {
                    position_id,
                    entry_ts_ms: tr.timestamp_ms,
                    entry_price: tr.price,
                    side,
                    max_high: tr.price,
                    min_low: tr.price,
                    exit_seq: 0,
                    open_count_for_symbol,
                };

                // Include the entry candle bar at this timestamp.
                while candle_i < bars.len() && bars[candle_i].t == ts {
                    update_extremes(&mut a, &bars[candle_i]);
                    candle_i += 1;
                }

                active = Some(a);
                continue;
            }

            // For non-OPEN events, include the candle bar at ts before processing the event.
            while candle_i < bars.len() && bars[candle_i].t == ts {
                if let Some(ref mut a) = active {
                    update_extremes(a, &bars[candle_i]);
                }
                candle_i += 1;
            }

            // ADD events update the position but do not emit a trade row.
            if tr.action.starts_with("ADD_") {
                continue;
            }

            // Exit events emit a row.
            if tr.action.starts_with("CLOSE_") || tr.action.starts_with("REDUCE_") {
                let Some(ref mut a) = active else {
                    continue;
                };

                a.exit_seq += 1;

                let (mae_pct, mfe_pct) = if a.entry_price.abs() > 1e-12 {
                    match a.side {
                        Side::Long => (
                            (a.min_low - a.entry_price) / a.entry_price,
                            (a.max_high - a.entry_price) / a.entry_price,
                        ),
                        Side::Short => (
                            (a.entry_price - a.max_high) / a.entry_price,
                            (a.entry_price - a.min_low) / a.entry_price,
                        ),
                    }
                } else {
                    (0.0, 0.0)
                };

                let rc = bt_core::reason_codes::classify_reason_code(&tr.action, &tr.reason);

                rows.push(TradeExportRow {
                    trade_id: format!("{}:{}", a.position_id, a.exit_seq),
                    position_id: a.position_id,
                    entry_ts_ms: a.entry_ts_ms,
                    exit_ts_ms: tr.timestamp_ms,
                    symbol: symbol.clone(),
                    side: match a.side {
                        Side::Long => "LONG".to_string(),
                        Side::Short => "SHORT".to_string(),
                    },
                    entry_price: a.entry_price,
                    exit_price: tr.price,
                    exit_size: tr.size,
                    pnl_usd: tr.pnl,
                    fee_usd: tr.fee_usd,
                    mae_pct,
                    mfe_pct,
                    reason_code: reason_code_string(rc),
                    reason: tr.reason.clone(),
                });

                if tr.action.starts_with("CLOSE_") {
                    active = None;
                }
            }
        }
    }

    // Deterministic output order.
    rows.sort_by(|a, b| {
        a.exit_ts_ms
            .cmp(&b.exit_ts_ms)
            .then_with(|| a.symbol.cmp(&b.symbol))
            .then_with(|| a.trade_id.cmp(&b.trade_id))
    });

    rows
}

fn write_trade_export_csv(
    path: &PathBuf,
    rows: &[TradeExportRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;

    writeln!(
        f,
        "trade_id,position_id,entry_ts_ms,exit_ts_ms,symbol,side,entry_price,exit_price,exit_size,pnl_usd,fee_usd,mae_pct,mfe_pct,reason_code,reason"
    )?;

    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{}",
            csv_escape(&r.trade_id),
            r.position_id,
            r.entry_ts_ms,
            r.exit_ts_ms,
            csv_escape(&r.symbol),
            csv_escape(&r.side),
            r.entry_price,
            r.exit_price,
            r.exit_size,
            r.pnl_usd,
            r.fee_usd,
            r.mae_pct,
            r.mfe_pct,
            csv_escape(&r.reason_code),
            csv_escape(&r.reason),
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_replay(args: ReplayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = bt_core::config::load_config(
        &args.config,
        args.symbol.as_deref(),
        args.live,
    );

    // Resolve intervals: CLI arg > YAML engine section > default "1h"
    let interval = args.interval.unwrap_or_else(|| {
        let yaml_iv = &cfg.engine.interval;
        if yaml_iv.is_empty() { "1h".to_string() } else { yaml_iv.clone() }
    });

    // Auto-resolve candle DB path from interval if not explicitly provided
    let candles_db = args.candles_db.unwrap_or_else(|| {
        format!("candles_dbs/candles_{}.db", interval)
    });

    // Resolve exit_interval: CLI arg > YAML engine.exit_interval > None
    let exit_interval = args.exit_interval.or_else(|| {
        let yaml_eiv = &cfg.engine.exit_interval;
        if yaml_eiv.is_empty() { None } else { Some(yaml_eiv.clone()) }
    });

    // Auto-resolve exit candle DB path from exit_interval if not explicitly provided
    let (exit_candles_db, exit_interval) = match (args.exit_candles_db, exit_interval) {
        (Some(db), iv) => (Some(db), iv),
        (None, Some(iv)) => (Some(format!("candles_dbs/candles_{}.db", iv)), Some(iv)),
        (None, None) => (None, None),
    };

    // Resolve entry_interval: CLI arg > YAML engine.entry_interval > None
    let entry_interval = args.entry_interval.or_else(|| {
        let yaml_niv = &cfg.engine.entry_interval;
        if yaml_niv.is_empty() { None } else { Some(yaml_niv.clone()) }
    });

    // Auto-resolve entry candle DB path from entry_interval if not explicitly provided
    let (entry_candles_db, entry_interval) = match (args.entry_candles_db, entry_interval) {
        (Some(db), iv) => (Some(db), iv),
        (None, Some(iv)) => (Some(format!("candles_dbs/candles_{}.db", iv)), Some(iv)),
        (None, None) => (None, None),
    };

    eprintln!(
        "[replay] Config loaded from {:?} (live={}, symbol={:?})",
        args.config, args.live, args.symbol,
    );
    eprintln!(
        "[replay] Indicator interval: {} (db: {})",
        interval, candles_db,
    );
    if let (Some(ref edb), Some(ref eiv)) = (&exit_candles_db, &exit_interval) {
        eprintln!("[replay] Exit interval: {} (db: {})", eiv, edb);
    }
    if let (Some(ref ndb), Some(ref niv)) = (&entry_candles_db, &entry_interval) {
        eprintln!("[replay] Entry interval: {} (db: {})", niv, ndb);
    }
    eprintln!(
        "[replay] SL={:.1}x ATR, TP={:.1}x ATR, leverage={:.1}, alloc={:.1}%",
        cfg.trade.sl_atr_mult,
        cfg.trade.tp_atr_mult,
        cfg.trade.leverage,
        cfg.trade.allocation_pct * 100.0,
    );

    // Compute time range: explicit --from-ts / --to-ts, or auto-scope (default ON)
    let (from_ts, to_ts) = if !args.no_auto_scope {
        let mut dbs: Vec<(&str, &str)> = Vec::new();
        // Don't scope the indicator DB — it needs full history for warmup.
        // Only scope the entry/exit DBs that determine the trading window.
        if let (Some(ref edb), Some(ref eiv)) = (&exit_candles_db, &exit_interval) {
            dbs.push((edb.as_str(), eiv.as_str()));
        }
        if let (Some(ref ndb), Some(ref niv)) = (&entry_candles_db, &entry_interval) {
            dbs.push((ndb.as_str(), niv.as_str()));
        }
        if dbs.is_empty() {
            // No sub-bar DBs — scope the main indicator DB
            dbs.push((&candles_db, &interval));
        }
        compute_auto_scope(&dbs, args.from_ts, args.to_ts)
    } else {
        (args.from_ts, args.to_ts)
    };

    // Load indicator candles (full history for warmup — no time filter)
    let candles = bt_data::sqlite_loader::load_candles(&candles_db, &interval)?;

    if candles.is_empty() {
        eprintln!("[replay] No candles found. Check --candles-db and --interval.");
        std::process::exit(1);
    }

    // If --symbol specified, verify it exists in the candle data
    if let Some(ref sym) = args.symbol {
        if !candles.contains_key(sym) {
            let mut available: Vec<&String> = candles.keys().collect();
            available.sort();
            eprintln!(
                "[replay] Symbol {:?} not found in candle DB. Available: {:?}",
                sym, available,
            );
            std::process::exit(1);
        }
    }

    let num_symbols = candles.len();
    let total_bars: usize = candles.values().map(|v| v.len()).sum();
    eprintln!("[replay] Loaded {total_bars} bars across {num_symbols} symbols");

    // Optional: load exit candles for two-level simulation (filtered)
    let exit_candles = if let Some(ref exit_db) = exit_candles_db {
        let exit_iv = exit_interval.as_deref().unwrap_or(&interval);
        eprintln!("[replay] Loading exit candles from {:?} (interval={})", exit_db, exit_iv);
        let ec = bt_data::sqlite_loader::load_candles_filtered(exit_db, exit_iv, from_ts, to_ts)?;
        let ec_bars: usize = ec.values().map(|v| v.len()).sum();
        eprintln!("[replay] Exit candles: {} bars across {} symbols", ec_bars, ec.len());
        Some(ec)
    } else {
        None
    };

    // Optional: load entry candles for sub-bar entry evaluation (filtered)
    let entry_candles = if let Some(ref entry_db) = entry_candles_db {
        let entry_iv = entry_interval.as_deref().unwrap_or(&interval);
        eprintln!("[replay] Loading entry candles from {:?} (interval={})", entry_db, entry_iv);
        let nc = bt_data::sqlite_loader::load_candles_filtered(entry_db, entry_iv, from_ts, to_ts)?;
        let nc_bars: usize = nc.values().map(|v| v.len()).sum();
        eprintln!("[replay] Entry candles: {} bars across {} symbols", nc_bars, nc.len());
        Some(nc)
    } else {
        None
    };

    // Optional: load funding rates (filtered)
    let funding_rates = if let Some(ref fdb) = args.funding_db {
        eprintln!("[replay] Loading funding rates from {:?}", fdb);
        let fr = bt_data::sqlite_loader::load_funding_rates_filtered(fdb, from_ts, to_ts)?;
        let fr_count: usize = fr.values().map(|v| v.len()).sum();
        eprintln!("[replay] Funding rates: {} entries across {} symbols", fr_count, fr.len());
        Some(fr)
    } else {
        None
    };

    // Resolve effective initial balance: --balance-from overrides --initial-balance
    let base_balance = if let Some(ref path) = args.balance_from {
        read_balance_from_json(path)
    } else {
        args.initial_balance
    };

    // Load init-state if provided (overrides both --initial-balance and --balance-from)
    let (effective_balance, init_state) = if let Some(ref path) = args.init_state {
        eprintln!("[replay] Loading init-state from {:?}", path);
        let state_file = bt_core::init_state::load(path)
            .unwrap_or_else(|e| { eprintln!("[error] {e}"); std::process::exit(1); });

        // Collect valid symbols from candle data for filtering
        let sym_strs: Vec<&str> = candles.keys().map(|s| s.as_str()).collect();
        let (balance, positions) = bt_core::init_state::into_sim_state(state_file, Some(&sym_strs));
        eprintln!("[replay] Init-state: balance=${:.2}, {} position(s)", balance, positions.len());
        (balance, Some((balance, positions)))
    } else {
        (base_balance, None)
    };

    let start = Instant::now();
    let sim = bt_core::engine::run_simulation(
        &candles,
        &cfg,
        effective_balance,
        args.lookback,
        exit_candles.as_ref(),
        entry_candles.as_ref(),
        funding_rates.as_ref(),
        init_state,
        from_ts,
        to_ts,
    );
    let elapsed = start.elapsed();

    let report = bt_core::report::build_report(
        &sim.trades,
        &sim.signals,
        &sim.equity_curve,
        &sim.gate_stats,
        effective_balance,
        sim.final_balance,
        "replay",
        args.trades,
        args.equity_curve,
    );

    // Print summary to stderr
    print_summary(&report, effective_balance);
    eprintln!("\nCompleted in {:.3}s", elapsed.as_secs_f64());

    // Optional trade-level CSV export (one row per exit event).
    if let Some(ref path) = args.export_trades {
        let rows = build_trade_export_rows(&candles, &sim.trades);
        write_trade_export_csv(path, &rows)?;
        eprintln!("[replay] Trade CSV written to {}", path.display());
    }

    // JSON output
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(ref path) = args.output {
        let mut f = std::fs::File::create(path)?;
        f.write_all(json.as_bytes())?;
        f.write_all(b"\n")?;
        eprintln!("[replay] Report written to {}", path.display());
    } else {
        println!("{json}");
    }

    Ok(())
}

fn cmd_sweep(args: SweepArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Load base config
    let base_cfg = bt_core::config::load_config(&args.config, None, args.live);

    // Resolve intervals: CLI arg > YAML engine section > default "1h"
    let interval = args.interval.unwrap_or_else(|| {
        let yaml_iv = &base_cfg.engine.interval;
        if yaml_iv.is_empty() { "1h".to_string() } else { yaml_iv.clone() }
    });

    // Auto-resolve candle DB path from interval
    let candles_db = args.candles_db.unwrap_or_else(|| {
        format!("candles_dbs/candles_{}.db", interval)
    });

    // Resolve exit_interval: CLI arg > YAML engine.exit_interval > None
    let exit_interval = args.exit_interval.or_else(|| {
        let yaml_eiv = &base_cfg.engine.exit_interval;
        if yaml_eiv.is_empty() { None } else { Some(yaml_eiv.clone()) }
    });

    // Auto-resolve exit candle DB paths
    let (exit_candles_db, exit_interval) = match (args.exit_candles_db, exit_interval) {
        (Some(db), iv) => (Some(db), iv),
        (None, Some(iv)) => (Some(format!("candles_dbs/candles_{}.db", iv)), Some(iv)),
        (None, None) => (None, None),
    };

    // Resolve entry_interval: CLI arg > YAML engine.entry_interval > None
    let entry_interval = args.entry_interval.or_else(|| {
        let yaml_niv = &base_cfg.engine.entry_interval;
        if yaml_niv.is_empty() { None } else { Some(yaml_niv.clone()) }
    });

    let (entry_candles_db, entry_interval) = match (args.entry_candles_db, entry_interval) {
        (Some(db), iv) => (Some(db), iv),
        (None, Some(iv)) => (Some(format!("candles_dbs/candles_{}.db", iv)), Some(iv)),
        (None, None) => (None, None),
    };

    // Resolve effective initial balance: --balance-from overrides --initial-balance
    let initial_balance = if let Some(ref path) = args.balance_from {
        read_balance_from_json(path)
    } else {
        args.initial_balance
    };

    // Load sweep spec
    let mut spec = bt_core::sweep::load_sweep_spec(&args.sweep_spec)?;
    // Override spec balance with CLI-resolved value (--balance-from > --initial-balance > default)
    spec.initial_balance = initial_balance;

    eprintln!(
        "[sweep] {} axes, initial_balance={}, lookback={}, interval={} (db: {})",
        spec.axes.len(),
        initial_balance,
        spec.lookback,
        interval,
        candles_db,
    );
    if let (Some(ref edb), Some(ref eiv)) = (&exit_candles_db, &exit_interval) {
        eprintln!("[sweep] Exit interval: {} (db: {})", eiv, edb);
    }
    if let (Some(ref ndb), Some(ref niv)) = (&entry_candles_db, &entry_interval) {
        eprintln!("[sweep] Entry interval: {} (db: {})", niv, ndb);
    }
    for axis in &spec.axes {
        eprintln!("  - {} ({} values: {:?})", axis.path, axis.values.len(), axis.values);
    }

    // Compute time range: explicit --from-ts / --to-ts, or auto-scope
    let (from_ts, to_ts) = if !args.no_auto_scope {
        // Always include main DB so its range participates in the intersection
        let mut dbs: Vec<(&str, &str)> = vec![(&candles_db, &interval)];
        if let (Some(ref edb), Some(ref eiv)) = (&exit_candles_db, &exit_interval) {
            dbs.push((edb.as_str(), eiv.as_str()));
        }
        if let (Some(ref ndb), Some(ref niv)) = (&entry_candles_db, &entry_interval) {
            dbs.push((ndb.as_str(), niv.as_str()));
        }
        compute_auto_scope(&dbs, args.from_ts, args.to_ts)
    } else {
        (args.from_ts, args.to_ts)
    };

    // Load indicator candles (full history for warmup — no time filter)
    let candles = bt_data::sqlite_loader::load_candles(&candles_db, &interval)?;
    if candles.is_empty() {
        eprintln!("[sweep] No candles found. Check --candles-db.");
        std::process::exit(1);
    }

    let num_symbols = candles.len();
    let total_bars: usize = candles.values().map(|v| v.len()).sum();
    eprintln!("[sweep] Loaded {total_bars} bars across {num_symbols} symbols");

    // Load exit candles (filtered)
    let exit_candles = if let Some(ref exit_db) = exit_candles_db {
        let exit_iv = exit_interval.as_deref().unwrap_or(&interval);
        let ec = bt_data::sqlite_loader::load_candles_filtered(exit_db, exit_iv, from_ts, to_ts)?;
        let ec_bars: usize = ec.values().map(|v| v.len()).sum();
        eprintln!("[sweep] Exit candles: {} bars across {} symbols", ec_bars, ec.len());
        Some(ec)
    } else {
        None
    };

    // Load entry candles (filtered)
    let entry_candles = if let Some(ref entry_db) = entry_candles_db {
        let entry_iv = entry_interval.as_deref().unwrap_or(&interval);
        let nc = bt_data::sqlite_loader::load_candles_filtered(entry_db, entry_iv, from_ts, to_ts)?;
        let nc_bars: usize = nc.values().map(|v| v.len()).sum();
        eprintln!("[sweep] Entry candles: {} bars across {} symbols", nc_bars, nc.len());
        Some(nc)
    } else {
        None
    };

    // Load funding rates (filtered)
    let funding_rates = if let Some(ref fdb) = args.funding_db {
        let fr = bt_data::sqlite_loader::load_funding_rates_filtered(fdb, from_ts, to_ts)?;
        let fr_count: usize = fr.values().map(|v| v.len()).sum();
        eprintln!("[sweep] Funding rates: {} entries across {} symbols", fr_count, fr.len());
        Some(fr)
    } else {
        None
    };

    // Configure rayon thread pool
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
        eprintln!("[sweep] Rayon thread pool: {n} threads");
    } else {
        eprintln!(
            "[sweep] Rayon thread pool: {} threads (auto)",
            rayon::current_num_threads(),
        );
    }

    // ------------------------------------------------------------------
    // GPU-accelerated sweep path (feature-gated)
    // ------------------------------------------------------------------
    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("Error: --gpu requires building with `--features gpu`");
        std::process::exit(1);
    }

    #[cfg(not(feature = "gpu"))]
    if args.tpe {
        eprintln!("Error: --tpe requires building with `--features gpu`");
        std::process::exit(1);
    }

    // GPU sub-bar candles: prefer exit_candles, fallback to entry_candles
    #[cfg(feature = "gpu")]
    let gpu_sub_candles: Option<&bt_core::candle::CandleData> = exit_candles.as_ref()
        .or(entry_candles.as_ref());

    #[cfg(feature = "gpu")]
    if args.tpe {
        if !args.gpu {
            eprintln!("Error: --tpe requires --gpu");
            std::process::exit(1);
        }
        eprintln!("[sweep] Using TPE Bayesian optimization (GPU)");
        let gpu_start = Instant::now();
        let tpe_cfg = bt_gpu::tpe_sweep::TpeConfig {
            trials: args.tpe_trials,
            batch_size: args.tpe_batch,
            seed: args.tpe_seed,
        };
        let results = bt_gpu::tpe_sweep::run_tpe_sweep(
            &candles,
            &base_cfg,
            &spec,
            funding_rates.as_ref(),
            &tpe_cfg,
            gpu_sub_candles,
            from_ts,
            to_ts,
        );
        let gpu_elapsed = gpu_start.elapsed();

        // Write JSONL output
        {
            let mut f = std::fs::File::create(&args.output)?;
            let top_n = args.top_n.unwrap_or(results.len());
            for r in results.iter().take(top_n) {
                let json = serde_json::json!({
                    "config_id": r.config_id,
                    "total_pnl": r.total_pnl,
                    "final_balance": r.final_balance,
                    "total_trades": r.total_trades,
                    "total_wins": r.total_wins,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "overrides": r.overrides,
                });
                writeln!(f, "{}", json)?;
            }
            eprintln!(
                "\n[sweep] TPE results written to {} ({} rows)",
                args.output.display(),
                top_n.min(results.len()),
            );
        }

        eprintln!(
            "\n[sweep] {} TPE trials completed via GPU in {:.2}s ({:.1} trials/s)",
            results.len(),
            gpu_elapsed.as_secs_f64(),
            results.len() as f64 / gpu_elapsed.as_secs_f64(),
        );

        // Print top 5 summary
        eprintln!("\n[TPE] Top 5 results:");
        for (i, r) in results.iter().take(5).enumerate() {
            eprintln!(
                "  #{}: PnL=${:.2}, Trades={}, WR={:.1}%, PF={:.2}, DD={:.1}%",
                i + 1,
                r.total_pnl,
                r.total_trades,
                r.win_rate * 100.0,
                r.profit_factor,
                r.max_drawdown_pct * 100.0,
            );
        }

        return Ok(());
    }

    #[cfg(feature = "gpu")]
    if args.gpu {
        eprintln!("[sweep] Using GPU-accelerated sweep");
        let gpu_start = Instant::now();
        let results = bt_gpu::run_gpu_sweep(
            &candles,
            &base_cfg,
            &spec,
            funding_rates.as_ref(),
            gpu_sub_candles,
            from_ts,
            to_ts,
        );
        let gpu_elapsed = gpu_start.elapsed();

        // Write JSONL output (same format as CPU sweep)
        {
            let mut f = std::fs::File::create(&args.output)?;
            for r in &results {
                let json = serde_json::json!({
                    "config_id": r.config_id,
                    "total_pnl": r.total_pnl,
                    "final_balance": r.final_balance,
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "overrides": r.overrides,
                });
                writeln!(f, "{}", json)?;
            }
            eprintln!(
                "\n[sweep] GPU results written to {} ({} rows)",
                args.output.display(),
                results.len(),
            );
        }

        eprintln!(
            "\n[sweep] {} combos completed via GPU in {:.2}s ({:.1} combos/s)",
            results.len(),
            gpu_elapsed.as_secs_f64(),
            results.len() as f64 / gpu_elapsed.as_secs_f64(),
        );
        return Ok(());
    }

    // ------------------------------------------------------------------
    // CPU sweep path (default)
    // ------------------------------------------------------------------
    let sweep_start = Instant::now();
    let results = bt_core::sweep::run_sweep(
        &base_cfg,
        &spec,
        &candles,
        exit_candles.as_ref(),
        entry_candles.as_ref(),
        funding_rates.as_ref(),
    );
    let sweep_elapsed = sweep_start.elapsed();

    let total_combos = results.len();

    // Write JSONL output (report + overrides flattened)
    {
        let mut f = std::fs::File::create(&args.output)?;
        for r in &results {
            let mut obj = serde_json::to_value(&r.report)?;
            if let serde_json::Value::Object(ref mut map) = obj {
                let mut ov = serde_json::Map::new();
                for (k, v) in &r.overrides {
                    ov.insert(k.clone(), serde_json::Value::from(*v));
                }
                map.insert("overrides".to_string(), serde_json::Value::Object(ov));
            }
            let line = serde_json::to_string(&obj)?;
            writeln!(f, "{line}")?;
        }
        eprintln!(
            "\n[sweep] Results written to {} ({} rows)",
            args.output.display(),
            results.len(),
        );
    }

    // Print top N summary
    let display_n = args.top_n.unwrap_or(10).min(results.len());
    eprintln!("\n--- Top {display_n} results (by total PnL) ---\n");
    eprintln!(
        "{:>5}  {:>10}  {:>12}  {:>8}  {:>8}  {:>8}  {:>10}  {}",
        "Rank", "Config", "PnL", "WR%", "Trades", "Sharpe", "MaxDD%", "Overrides",
    );
    eprintln!("{}", "-".repeat(100));

    for (rank, r) in results.iter().take(display_n).enumerate() {
        let overrides_str: Vec<String> = r
            .overrides
            .iter()
            .map(|(k, v)| format!("{k}={v:.3}"))
            .collect();

        eprintln!(
            "{:>5}  {:>10}  {:>12.2}  {:>7.1}%  {:>8}  {:>8.3}  {:>9.2}%  {}",
            rank + 1,
            r.config_id,
            r.report.total_pnl,
            r.report.win_rate * 100.0,
            r.report.total_trades,
            r.report.sharpe_ratio,
            r.report.max_drawdown_pct * 100.0,
            overrides_str.join(", "),
        );
    }

    eprintln!(
        "\n[sweep] {total_combos} combos completed in {:.2}s ({:.1} combos/s)",
        sweep_elapsed.as_secs_f64(),
        total_combos as f64 / sweep_elapsed.as_secs_f64(),
    );

    Ok(())
}

fn cmd_dump_indicators(args: DumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = bt_core::config::load_config(&args.config, Some(&args.symbol), false);

    // Resolve interval: CLI arg > YAML engine section > default "1h"
    let interval = args.interval.unwrap_or_else(|| {
        let yaml_iv = &cfg.engine.interval;
        if yaml_iv.is_empty() { "1h".to_string() } else { yaml_iv.clone() }
    });

    // Auto-resolve candle DB path from interval
    let candles_db = args.candles_db.unwrap_or_else(|| {
        format!("candles_dbs/candles_{}.db", interval)
    });

    let candles = bt_data::sqlite_loader::load_candles(&candles_db, &interval)?;

    let bars = candles.get(&args.symbol).ok_or_else(|| {
        let mut available: Vec<&String> = candles.keys().collect();
        available.sort();
        format!(
            "Symbol {:?} not found in candle DB. Available: {:?}",
            args.symbol, available,
        )
    })?;

    eprintln!(
        "[dump-indicators] {} bars for {} (config: {:?})",
        bars.len(),
        args.symbol,
        args.config,
    );

    let use_stoch_rsi = cfg.filters.use_stoch_rsi_filter;
    let mut bank = bt_core::indicators::IndicatorBank::new(&cfg.indicators, use_stoch_rsi);

    let mut writer: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(std::io::BufWriter::new(std::fs::File::create(path)?))
    } else {
        Box::new(std::io::BufWriter::new(std::io::stdout().lock()))
    };

    // CSV header
    writeln!(
        writer,
        "t,close,ema_slow,ema_fast,ema_macro,adx,adx_pos,adx_neg,\
         bb_upper,bb_lower,bb_width,atr,rsi,macd_hist,\
         stoch_rsi_k,stoch_rsi_d,vol_sma,vol_trend"
    )?;

    let start = Instant::now();

    for bar in bars {
        let snap = bank.update(bar);

        writeln!(
            writer,
            "{},{:.8},{:.8},{:.8},{:.8},{:.4},{:.4},{:.4},\
             {:.8},{:.8},{:.6},{:.8},{:.4},{:.8},\
             {:.4},{:.4},{:.4},{}",
            snap.t,
            snap.close,
            snap.ema_slow,
            snap.ema_fast,
            snap.ema_macro,
            snap.adx,
            snap.adx_pos,
            snap.adx_neg,
            snap.bb_upper,
            snap.bb_lower,
            snap.bb_width,
            snap.atr,
            snap.rsi,
            snap.macd_hist,
            snap.stoch_rsi_k,
            snap.stoch_rsi_d,
            snap.vol_sma,
            snap.vol_trend as u8,
        )?;
    }

    writer.flush()?;

    let elapsed = start.elapsed();

    if let Some(ref path) = args.output {
        eprintln!(
            "[dump-indicators] Wrote {} rows to {} in {:.2}s",
            bars.len(),
            path.display(),
            elapsed.as_secs_f64(),
        );
    } else {
        eprintln!(
            "[dump-indicators] Dumped {} rows in {:.2}s",
            bars.len(),
            elapsed.as_secs_f64(),
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Summary printer
// ---------------------------------------------------------------------------

fn print_summary(r: &bt_core::report::SimReport, initial_balance: f64) {
    eprintln!("\n=== Backtest Summary ===\n");
    eprintln!("Initial Balance:   ${:.2}", initial_balance);
    eprintln!("Final Balance:     ${:.2}", r.final_balance);
    eprintln!("Total PnL:         ${:.2} ({:+.2}%)", r.total_pnl, r.total_pnl / initial_balance * 100.0);
    eprintln!("Total Trades:      {}", r.total_trades);
    eprintln!("Win Rate:          {:.1}% ({}/{})", r.win_rate * 100.0, r.total_wins, r.total_trades);
    eprintln!("Profit Factor:     {:.2}", r.profit_factor);
    eprintln!("Sharpe Ratio:      {:.3}", r.sharpe_ratio);
    eprintln!("Max Drawdown:      ${:.2} ({:.2}%)", r.max_drawdown_usd, r.max_drawdown_pct * 100.0);
    eprintln!("Avg Win:           ${:.2}", r.avg_win);
    eprintln!("Avg Loss:          ${:.2}", r.avg_loss);
    eprintln!("Total Fees:        ${:.2}", r.total_fees);
    eprintln!("Signals Generated: {}", r.total_signals);

    if !r.by_confidence.is_empty() {
        eprintln!("\n--- By Confidence ---");
        for b in &r.by_confidence {
            eprintln!("  {:>8}: {:>4} trades, PnL ${:>8.2}, WR {:.1}%, Avg ${:.2}",
                b.confidence, b.trades, b.pnl, b.win_rate * 100.0, b.avg_pnl);
        }
    }

    if !r.by_exit_reason.is_empty() {
        eprintln!("\n--- By Exit Reason ---");
        for b in &r.by_exit_reason {
            eprintln!("  {:>20}: {:>4} trades, PnL ${:>8.2}, WR {:.1}%",
                b.reason, b.trades, b.pnl, b.win_rate * 100.0);
        }
    }

    // Gate stats
    let gs = &r.gate_stats;
    eprintln!("\n--- Gate Stats ---");
    eprintln!("  Total checks:         {}", gs.total_checks);
    eprintln!("  Signals generated:    {} ({:.1}%)", gs.signals_generated, gs.signal_pct * 100.0);
    if gs.blocked_by_ranging > 0 { eprintln!("  Blocked by ranging:   {}", gs.blocked_by_ranging); }
    if gs.blocked_by_anomaly > 0 { eprintln!("  Blocked by anomaly:   {}", gs.blocked_by_anomaly); }
    if gs.blocked_by_adx_low > 0 { eprintln!("  Blocked by ADX low:   {}", gs.blocked_by_adx_low); }
    if gs.blocked_by_confidence > 0 { eprintln!("  Blocked by confidence:{}", gs.blocked_by_confidence); }
    if gs.blocked_by_max_positions > 0 { eprintln!("  Blocked by max pos:   {}", gs.blocked_by_max_positions); }
    if gs.blocked_by_pesc > 0 { eprintln!("  Blocked by PESC:      {}", gs.blocked_by_pesc); }
    if gs.blocked_by_ssf > 0 { eprintln!("  Blocked by SSF:       {}", gs.blocked_by_ssf); }
    if gs.blocked_by_reef > 0 { eprintln!("  Blocked by REEF:      {}", gs.blocked_by_reef); }
    if gs.blocked_by_margin > 0 { eprintln!("  Blocked by margin:    {}", gs.blocked_by_margin); }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("mei-backtester v{VERSION}");
    eprintln!();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Replay(args) => cmd_replay(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::DumpIndicators(args) => cmd_dump_indicators(args),
    };

    if let Err(e) = result {
        eprintln!("\n[error] {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use bt_core::candle::{CandleData, OhlcvBar};
    use bt_core::config::Confidence;
    use bt_core::position::TradeRecord;

    fn bar(t: i64, h: f64, l: f64) -> OhlcvBar {
        OhlcvBar {
            t,
            t_close: t,
            o: 0.0,
            h,
            l,
            c: 0.0,
            v: 0.0,
            n: 0,
        }
    }

    fn tr(
        ts: i64,
        symbol: &str,
        action: &str,
        price: f64,
        size: f64,
        notional: f64,
        pnl: f64,
        fee_usd: f64,
        reason: &str,
    ) -> TradeRecord {
        TradeRecord {
            timestamp_ms: ts,
            symbol: symbol.to_string(),
            action: action.to_string(),
            price,
            size,
            notional,
            reason: reason.to_string(),
            confidence: Confidence::Low,
            pnl,
            fee_usd,
            balance: 0.0,
            entry_atr: 0.0,
            leverage: 1.0,
            margin_used: 0.0,
        }
    }

    #[test]
    fn trade_export_is_deterministic_and_has_unique_trade_ids() {
        let mut candles: CandleData = CandleData::default();
        candles.insert(
            "BTC".to_string(),
            vec![
                bar(0, 110.0, 90.0),
                bar(3_600_000, 120.0, 80.0),
            ],
        );

        let trades = vec![
            tr(0, "BTC", "OPEN_LONG", 100.0, 1.0, 100.0, 0.0, 1.0, "High entry"),
            tr(
                3_600_000,
                "BTC",
                "CLOSE_LONG",
                105.0,
                1.0,
                105.0,
                5.0,
                1.0,
                "Take Profit",
            ),
        ];

        let rows1 = build_trade_export_rows(&candles, &trades);
        let rows2 = build_trade_export_rows(&candles, &trades);
        assert_eq!(rows1, rows2);

        assert_eq!(rows1.len(), 1);
        let r = &rows1[0];
        assert_eq!(r.trade_id, "1:1");
        assert_eq!(r.position_id, 1);
        assert_eq!(r.symbol, "BTC");
        assert_eq!(r.side, "LONG");
        assert!((r.mae_pct - (-0.2)).abs() < 1e-9);
        assert!((r.mfe_pct - 0.2).abs() < 1e-9);
        assert_eq!(r.reason_code, "exit_take_profit");

        let mut ids = HashSet::new();
        for row in &rows1 {
            assert!(ids.insert(row.trade_id.clone()), "duplicate trade_id");
        }
    }
}
