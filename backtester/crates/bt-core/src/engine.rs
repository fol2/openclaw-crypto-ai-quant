//! Main simulation loop for the backtesting engine.
//!
//! Processes all symbols bar-by-bar in chronological order, orchestrating the
//! full trading lifecycle: indicator warmup → exit checks → gate evaluation →
//! signal generation → engine-level transforms → entry execution → PnL.

use crate::candle::{CandleData, FundingRateData, OhlcvBar};
use crate::config::{Confidence, Signal, StrategyConfig};
use crate::exits::{self, ExitResult};
use crate::indicators::{IndicatorBank, IndicatorSnapshot};
use crate::position::{Position, PositionType, SignalRecord, TradeRecord};
use crate::signals::{entry, gates};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Fee constant
// ---------------------------------------------------------------------------

const FEE_RATE: f64 = 0.00035; // Hyperliquid taker fee

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Aggregate statistics about gate-level blocking.
#[derive(Debug, Clone, Default)]
pub struct GateStats {
    pub total_checks: u64,
    pub neutral_count: u64,
    pub buy_count: u64,
    pub sell_count: u64,
    pub blocked_by_ranging: u64,
    pub blocked_by_anomaly: u64,
    pub blocked_by_extension: u64,
    pub blocked_by_adx_low: u64,
    pub blocked_by_adx_not_rising: u64,
    pub blocked_by_volume: u64,
    pub blocked_by_btc: u64,
    pub blocked_by_confidence: u64,
    pub blocked_by_max_positions: u64,
    pub blocked_by_pesc: u64,
    pub blocked_by_ssf: u64,
    pub blocked_by_reef: u64,
    pub blocked_by_margin: u64,
}

/// Complete output of a simulation run.
pub struct SimResult {
    pub trades: Vec<TradeRecord>,
    pub signals: Vec<SignalRecord>,
    pub final_balance: f64,
    pub equity_curve: Vec<(i64, f64)>,
    pub gate_stats: GateStats,
}

// ---------------------------------------------------------------------------
// Entry candidate for signal ranking
// ---------------------------------------------------------------------------

struct EntryCandidate {
    symbol: String,
    signal: Signal,
    confidence: Confidence,
    adx: f64,
    atr: f64,
    entry_adx_threshold: f64,
    snap: IndicatorSnapshot,
    ts: i64,
}

// ---------------------------------------------------------------------------
// Internal simulation state
// ---------------------------------------------------------------------------

struct SimState {
    balance: f64,
    positions: FxHashMap<String, Position>,
    indicators: FxHashMap<String, IndicatorBank>,
    /// EMA slow history per symbol (for slow-drift slope computation).
    ema_slow_history: FxHashMap<String, Vec<f64>>,
    /// Per-symbol bar counter (tracks warmup independently from IndicatorBank).
    bar_counts: FxHashMap<String, usize>,
    /// PESC: last close (timestamp_ms, position_type, exit_reason) per symbol.
    last_close: FxHashMap<String, (i64, PositionType, String)>,
    trades: Vec<TradeRecord>,
    signals: Vec<SignalRecord>,
    equity_curve: Vec<(i64, f64)>,
    gate_stats: GateStats,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run a full backtest simulation.
///
/// # Arguments
/// * `candles`         - Bar data keyed by symbol (indicator-resolution, e.g. 1h).
/// * `cfg`             - Strategy configuration.
/// * `initial_balance` - Starting equity in USD.
/// * `lookback`        - Number of warmup bars before trading begins.
/// * `exit_candles`    - Optional higher-resolution bar data for exit checks (e.g. 1m).
///                       When provided, exit conditions are evaluated on sub-bars within
///                       each indicator bar, aligning backtest exits with live behavior.
/// * `entry_candles`   - Optional higher-resolution bar data for entry checks (e.g. 15m).
///                       When provided, entry signals are evaluated at sub-bar resolution
///                       (indicator-bar indicators + sub-bar price) instead of indicator bars.
///                       This matches live engine behavior where signals are checked every ~60s.
/// * `funding_rates`   - Optional per-symbol funding rate data. When provided, funding
///                       payments are applied at hourly boundaries for open positions.
/// * `from_ts`         - Optional start timestamp (ms, inclusive). Bars before this
///                       still update indicators (warmup) but no trading occurs.
/// * `to_ts`           - Optional end timestamp (ms, inclusive). Bars after this are
///                       skipped for trading.
pub fn run_simulation(
    candles: &CandleData,
    cfg: &StrategyConfig,
    initial_balance: f64,
    lookback: usize,
    exit_candles: Option<&CandleData>,
    entry_candles: Option<&CandleData>,
    funding_rates: Option<&FundingRateData>,
    init_state: Option<(f64, FxHashMap<String, Position>)>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> SimResult {
    let use_stoch_rsi = cfg.filters.use_stoch_rsi_filter;

    // -- Build sorted timeline of unique timestamps --
    let timestamps = build_timeline(candles);
    if timestamps.is_empty() {
        return SimResult {
            trades: vec![],
            signals: vec![],
            final_balance: initial_balance,
            equity_curve: vec![],
            gate_stats: GateStats::default(),
        };
    }

    // -- Build per-symbol bar index for O(1) lookup --
    let bar_index = build_bar_index(candles);

    // -- Initialize state (optionally from exported live/paper snapshot) --
    let (init_balance, init_positions) = match init_state {
        Some((b, p)) => (b, p),
        None => (initial_balance, FxHashMap::default()),
    };
    let mut state = SimState {
        balance: init_balance,
        positions: init_positions,
        indicators: FxHashMap::default(),
        ema_slow_history: FxHashMap::default(),
        bar_counts: FxHashMap::default(),
        last_close: FxHashMap::default(),
        trades: Vec::with_capacity(4096),
        signals: Vec::with_capacity(8192),
        equity_curve: Vec::with_capacity(timestamps.len()),
        gate_stats: GateStats::default(),
    };

    // Pre-create indicator banks for every symbol
    for sym in candles.keys() {
        state
            .indicators
            .insert(sym.clone(), IndicatorBank::new(&cfg.indicators, use_stoch_rsi));
    }

    // Build exit candle index: symbol → sorted Vec<(timestamp_ms, index)>
    // Used for binary-search lookups when doing sub-bar exit checks.
    let exit_bar_index: Option<FxHashMap<String, Vec<(i64, usize)>>> = exit_candles.map(|ec| {
        let mut idx: FxHashMap<String, Vec<(i64, usize)>> = FxHashMap::default();
        for (sym, bars) in ec {
            let entries: Vec<(i64, usize)> = bars.iter().enumerate().map(|(i, b)| (b.t, i)).collect();
            idx.insert(sym.clone(), entries);
        }
        idx
    });

    // Build entry candle index (same structure as exit, but for entry sub-bars).
    let entry_bar_index: Option<FxHashMap<String, Vec<(i64, usize)>>> = entry_candles.map(|ec| {
        let mut idx: FxHashMap<String, Vec<(i64, usize)>> = FxHashMap::default();
        for (sym, bars) in ec {
            let entries: Vec<(i64, usize)> = bars.iter().enumerate().map(|(i, b)| (b.t, i)).collect();
            idx.insert(sym.clone(), entries);
        }
        idx
    });

    // Collect symbol names for ordered iteration (deterministic)
    let mut symbols: Vec<&String> = candles.keys().collect();
    symbols.sort();

    // -- Main loop: iterate every timestamp --
    for &ts in &timestamps {
        // 1. BTC context: determine if BTC is bullish
        let btc_bullish = compute_btc_bullish(&state, ts, &bar_index, candles);

        // 2. Compute market breadth (percentage of symbols where EMA_fast > EMA_slow)
        let breadth_pct = compute_market_breadth(&state);

        // Per-symbol EMA-slow slope cache for sub-bar entry evaluation
        let mut sub_bar_slopes: FxHashMap<String, f64> = FxHashMap::default();

        // Per-bar entry counter (limits entries to max_entry_orders_per_loop)
        let mut entries_this_bar: usize = 0;

        // Indicator-bar entry candidates (collected in phase 1, ranked in phase 2)
        let mut indicator_bar_candidates: Vec<EntryCandidate> = Vec::new();

        // 3. Process each symbol at this timestamp
        for sym in &symbols {
            let sym_str: &str = sym.as_str();

            // Check if this symbol has a bar at this timestamp
            let bar = match lookup_bar(&bar_index, sym_str, ts, candles) {
                Some(b) => b,
                None => continue,
            };

            // Feed bar to indicator bank
            let bank = state.indicators.get_mut(sym_str).unwrap();
            let snap = bank.update(bar);

            // Store EMA slow for slope computation (borrow ends immediately)
            state
                .ema_slow_history
                .entry(sym.to_string())
                .or_insert_with(Vec::new)
                .push(snap.ema_slow);

            // Increment bar count (copy value so mutable borrow ends)
            let bar_count = {
                let bc = state.bar_counts.entry(sym.to_string()).or_insert(0);
                *bc += 1;
                *bc
            };

            // ── Trade scope guard ─────────────────────────────────────
            // Skip exits and entries for bars outside [from_ts, to_ts].
            // Indicators are already updated above (needed for warmup).
            if let Some(ft) = from_ts {
                if ts < ft { continue; }
            }
            if let Some(tt) = to_ts {
                if ts > tt { continue; }
            }

            // ── Exit check for existing position ────────────────────────
            // Runs BEFORE warmup guard so that init-state positions are
            // monitored from the very first bar (without init-state,
            // positions are empty during warmup so this is a no-op).
            if let Some(pos) = state.positions.get(sym_str) {
                let exit_result = exits::check_all_exits(pos, &snap, cfg, ts);
                if exit_result.should_exit {
                    apply_exit(
                        &mut state,
                        sym_str,
                        &exit_result,
                        &snap,
                        ts,
                    );
                } else {
                    // Update trailing stop if applicable
                    update_trailing_stop(&mut state, sym_str, &snap, cfg);
                }
            }

            // Skip warmup period (entries only — exits above always run)
            if bar_count < lookback {
                continue;
            }

            // Compute EMA slow slope (re-borrow immutably after exit check)
            let slope_window = cfg.thresholds.entry.slow_drift_slope_window;
            let ema_slow_slope_pct = {
                let hist = state.ema_slow_history.get(sym_str).unwrap();
                compute_ema_slow_slope(hist, slope_window, snap.close)
            };
            sub_bar_slopes.insert(sym.to_string(), ema_slow_slope_pct);

            // ── Entry evaluation (collect candidates) ─────────────────
            let btc_bull_opt = btc_bullish;
            let gate_result =
                gates::check_gates(&snap, cfg, sym_str, btc_bull_opt, ema_slow_slope_pct);
            let (mut signal, confidence, entry_adx_threshold) =
                entry::generate_signal(&snap, &gate_result, cfg, ema_slow_slope_pct);

            // Track gate statistics
            state.gate_stats.total_checks += 1;
            match signal {
                Signal::Neutral => state.gate_stats.neutral_count += 1,
                Signal::Buy => state.gate_stats.buy_count += 1,
                Signal::Sell => state.gate_stats.sell_count += 1,
            }

            // Track gate-level blocks
            if gate_result.is_ranging {
                state.gate_stats.blocked_by_ranging += 1;
            }
            if gate_result.is_anomaly {
                state.gate_stats.blocked_by_anomaly += 1;
            }
            if gate_result.is_extended {
                state.gate_stats.blocked_by_extension += 1;
            }
            if !gate_result.adx_above_min {
                state.gate_stats.blocked_by_adx_low += 1;
            }
            if !gate_result.is_trending_up {
                state.gate_stats.blocked_by_adx_not_rising += 1;
            }
            if !gate_result.vol_confirm {
                state.gate_stats.blocked_by_volume += 1;
            }
            if !gate_result.btc_ok_long || !gate_result.btc_ok_short {
                state.gate_stats.blocked_by_btc += 1;
            }

            // Record raw signal
            if signal != Signal::Neutral {
                state.signals.push(SignalRecord {
                    timestamp_ms: ts,
                    symbol: sym.to_string(),
                    signal,
                    confidence,
                    price: snap.close,
                    adx: snap.adx,
                    rsi: snap.rsi,
                    atr: snap.atr,
                });
            }

            // ── Collect entry candidates (only on indicator bars when no entry_candles) ──
            // When entry_candles is provided, entry signals are deferred to the
            // entry sub-bar block below for higher-resolution timing.
            if entry_candles.is_none() {
                // Engine-level transforms
                let atr = apply_atr_floor(snap.atr, snap.close, cfg.trade.min_atr_pct);

                if signal != Signal::Neutral {
                    signal = apply_reverse(signal, cfg, breadth_pct);
                }
                if signal != Signal::Neutral {
                    signal = apply_regime_filter(signal, cfg, breadth_pct);
                }

                if signal == Signal::Neutral {
                    continue;
                }

                let desired_type = match signal {
                    Signal::Buy => PositionType::Long,
                    Signal::Sell => PositionType::Short,
                    Signal::Neutral => unreachable!(),
                };

                // Handle signal flip: close opposite position first
                if let Some(pos) = state.positions.get(sym_str) {
                    if pos.pos_type != desired_type {
                        let exit = ExitResult::exit("Signal Flip", snap.close);
                        apply_exit(&mut state, sym_str, &exit, &snap, ts);
                    }
                }

                // Handle same-direction pyramiding (immediate, not ranked)
                if let Some(pos) = state.positions.get(sym_str) {
                    if pos.pos_type == desired_type && cfg.trade.enable_pyramiding {
                        try_pyramid(&mut state, sym_str, &snap, cfg, confidence, atr, ts);
                    }
                    continue;
                }

                // Pre-filter gates that don't depend on cross-symbol state
                if !confidence.meets_min(cfg.trade.entry_min_confidence) {
                    state.gate_stats.blocked_by_confidence += 1;
                    continue;
                }
                if is_pesc_blocked(&state, sym_str, desired_type, ts, snap.adx, cfg) {
                    state.gate_stats.blocked_by_pesc += 1;
                    continue;
                }
                if cfg.trade.enable_ssf_filter {
                    let ssf_ok = match signal {
                        Signal::Buy => snap.macd_hist > 0.0,
                        Signal::Sell => snap.macd_hist < 0.0,
                        Signal::Neutral => true,
                    };
                    if !ssf_ok {
                        state.gate_stats.blocked_by_ssf += 1;
                        continue;
                    }
                }
                if cfg.trade.enable_reef_filter {
                    let reef_blocked = match signal {
                        Signal::Buy => {
                            if snap.adx < cfg.trade.reef_adx_threshold {
                                snap.rsi > cfg.trade.reef_long_rsi_block_gt
                            } else {
                                snap.rsi > cfg.trade.reef_long_rsi_extreme_gt
                            }
                        }
                        Signal::Sell => {
                            if snap.adx < cfg.trade.reef_adx_threshold {
                                snap.rsi < cfg.trade.reef_short_rsi_block_lt
                            } else {
                                snap.rsi < cfg.trade.reef_short_rsi_extreme_lt
                            }
                        }
                        Signal::Neutral => false,
                    };
                    if reef_blocked {
                        state.gate_stats.blocked_by_reef += 1;
                        continue;
                    }
                }

                // Collect as candidate for ranking
                indicator_bar_candidates.push(EntryCandidate {
                    symbol: sym.to_string(),
                    signal,
                    confidence,
                    adx: snap.adx,
                    atr,
                    entry_adx_threshold,
                    snap: snap.clone(),
                    ts,
                });
            }
        }

        // ── Phase 2: Rank indicator-bar entry candidates by score ────
        // Score = confidence_rank * 100 + ADX (descending), tiebreak by symbol (ascending).
        if !indicator_bar_candidates.is_empty() {
            indicator_bar_candidates.sort_by(|a, b| {
                let score_a = (a.confidence as i32) * 100 + a.adx as i32;
                let score_b = (b.confidence as i32) * 100 + b.adx as i32;
                score_b.cmp(&score_a)
                    .then_with(|| a.symbol.cmp(&b.symbol))
            });

            let equity = state.balance + unrealized_pnl(
                &state.positions, &state.indicators, ts, candles, &bar_index,
            );

            for cand in &indicator_bar_candidates {
                if entries_this_bar >= cfg.trade.max_entry_orders_per_loop {
                    break;
                }
                if state.positions.len() >= cfg.trade.max_open_positions {
                    state.gate_stats.blocked_by_max_positions += 1;
                    break;
                }
                // Skip if a position was opened for this symbol (e.g. via signal flip above)
                if state.positions.contains_key(&cand.symbol) {
                    continue;
                }

                let total_margin: f64 = state.positions.values().map(|p| p.margin_used).sum();
                let margin_headroom = equity * cfg.trade.max_total_margin_pct - total_margin;
                if margin_headroom <= 0.0 {
                    state.gate_stats.blocked_by_margin += 1;
                    continue;
                }

                let (size, margin_used, leverage) =
                    compute_entry_size(&cand.symbol, equity, cand.snap.close, cand.confidence, cand.atr, &cand.snap, cfg);
                let (mut size, mut margin_used) = if margin_used > margin_headroom {
                    let ratio = margin_headroom / margin_used;
                    (size * ratio, margin_headroom)
                } else {
                    (size, margin_used)
                };

                let mut notional = size * cand.snap.close;
                if notional < cfg.trade.min_notional_usd {
                    if cfg.trade.bump_to_min_notional && cand.snap.close > 0.0 {
                        size = cfg.trade.min_notional_usd / cand.snap.close;
                        margin_used = size * cand.snap.close / leverage;
                        notional = size * cand.snap.close;
                    } else {
                        continue;
                    }
                }

                let desired_type = match cand.signal {
                    Signal::Buy => PositionType::Long,
                    Signal::Sell => PositionType::Short,
                    Signal::Neutral => continue,
                };

                let fill_price = match cand.signal {
                    Signal::Buy => cand.snap.close * (1.0 + cfg.trade.slippage_bps / 10_000.0),
                    Signal::Sell => cand.snap.close * (1.0 - cfg.trade.slippage_bps / 10_000.0),
                    Signal::Neutral => continue,
                };

                let fee_usd = notional * FEE_RATE;
                state.balance -= fee_usd;

                let pos = Position {
                    symbol: cand.symbol.clone(),
                    pos_type: desired_type,
                    entry_price: fill_price,
                    size,
                    confidence: cand.confidence,
                    entry_atr: cand.atr,
                    entry_adx_threshold: cand.entry_adx_threshold,
                    trailing_sl: None,
                    leverage,
                    margin_used,
                    adds_count: 0,
                    tp1_taken: false,
                    open_time_ms: cand.ts,
                    last_add_time_ms: cand.ts,
                    mae_usd: 0.0,
                    mfe_usd: 0.0,
                };
                state.positions.insert(cand.symbol.clone(), pos);
                entries_this_bar += 1;

                let action = match desired_type {
                    PositionType::Long => "OPEN_LONG",
                    PositionType::Short => "OPEN_SHORT",
                };
                state.trades.push(TradeRecord {
                    timestamp_ms: cand.ts,
                    symbol: cand.symbol.clone(),
                    action: action.to_string(),
                    price: fill_price,
                    size,
                    notional,
                    reason: format!("{:?} entry", cand.confidence),
                    confidence: cand.confidence,
                    pnl: 0.0,
                    fee_usd,
                    balance: state.balance,
                    entry_atr: cand.atr,
                    leverage,
                    margin_used,
                    ..Default::default()
                });
            }
        }

        // Pre-compute next indicator-bar timestamp (shared by exit + entry sub-bar blocks)
        let ts_i = timestamps.binary_search(&ts).unwrap_or(0);
        let next_ts = if ts_i + 1 < timestamps.len() {
            timestamps[ts_i + 1]
        } else {
            i64::MAX
        };

        // ── Exit sub-bar block ──────────────────────────────────────────
        // Scan exit candles (e.g. 1m) within this indicator bar's time range
        // for SL/TP exit precision, matching live behavior.
        if let (Some(ec), Some(ref ec_idx)) = (exit_candles, &exit_bar_index) {
            if !state.positions.is_empty() {
                let exit_syms: Vec<String> = state.positions.keys().cloned().collect();

                for sym in &exit_syms {
                    if let Some(sym_exit_idx) = ec_idx.get(sym.as_str()) {
                        let start = match sym_exit_idx.binary_search_by_key(&(ts + 1), |(t, _)| *t) {
                            Ok(i) => i,
                            Err(i) => i,
                        };

                        if let Some(exit_bars) = ec.get(sym.as_str()) {
                            let base_snap = state.indicators.get(sym.as_str())
                                .map(|bank| bank.latest_snap())
                                .unwrap_or_else(|| make_minimal_snap(0.0, ts));

                            for &(sub_ts, bar_i) in &sym_exit_idx[start..] {
                                if sub_ts > next_ts {
                                    break;
                                }
                                if !state.positions.contains_key(sym.as_str()) {
                                    break; // Already exited
                                }
                                if let Some(sub_bar) = exit_bars.get(bar_i) {
                                    let sub_snap = make_exit_snap(&base_snap, sub_bar);
                                    let pos = state.positions.get(sym.as_str()).unwrap();
                                    let exit_result = exits::check_all_exits(pos, &sub_snap, cfg, sub_ts);
                                    if exit_result.should_exit {
                                        apply_exit(&mut state, sym.as_str(), &exit_result, &sub_snap, sub_ts);
                                    } else {
                                        update_trailing_stop(&mut state, sym.as_str(), &sub_snap, cfg);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Entry sub-bar block (with signal ranking) ─────────────────
        // When entry_candles is provided, evaluate entry signals at sub-bar
        // resolution (indicator-bar indicators + sub-bar price).
        // Signals at each sub-bar tick are collected across all symbols,
        // ranked by score, then executed — matching the indicator-bar ranking.
        if let (Some(enc), Some(ref enc_idx)) = (entry_candles, &entry_bar_index) {
            let sub_equity = state.balance + unrealized_pnl(
                &state.positions, &state.indicators, ts, candles, &bar_index,
            );

            // Build a merged timeline of unique sub-bar timestamps across all symbols
            // within this indicator bar's range (ts+1 .. next_ts].
            let mut sub_bar_ticks: Vec<i64> = Vec::new();
            for sym in &symbols {
                if let Some(sym_entry_idx) = enc_idx.get(sym.as_str()) {
                    let start = match sym_entry_idx.binary_search_by_key(&(ts + 1), |(t, _)| *t) {
                        Ok(i) => i,
                        Err(i) => i,
                    };
                    for &(sub_ts, _) in &sym_entry_idx[start..] {
                        if sub_ts > next_ts {
                            break;
                        }
                        sub_bar_ticks.push(sub_ts);
                    }
                }
            }
            sub_bar_ticks.sort_unstable();
            sub_bar_ticks.dedup();

            // Process each sub-bar tick: collect candidates across symbols, rank, execute.
            for sub_ts in &sub_bar_ticks {
                if entries_this_bar >= cfg.trade.max_entry_orders_per_loop {
                    break;
                }

                let mut sub_candidates: Vec<EntryCandidate> = Vec::new();

                for sym in &symbols {
                    // Skip if position already open
                    if state.positions.contains_key(sym.as_str()) {
                        continue;
                    }
                    // Skip warmup
                    let bc = state.bar_counts.get(sym.as_str()).copied().unwrap_or(0);
                    if bc < lookback {
                        continue;
                    }

                    // Look up this symbol's bar at this sub-bar tick
                    if let Some(sym_entry_idx) = enc_idx.get(sym.as_str()) {
                        // Binary search for exact timestamp match
                        if let Ok(idx) = sym_entry_idx.binary_search_by_key(sub_ts, |(t, _)| *t) {
                            let (_, bar_i) = sym_entry_idx[idx];
                            if let Some(entry_bars) = enc.get(sym.as_str()) {
                                if let Some(sub_bar) = entry_bars.get(bar_i) {
                                    let base_snap = state.indicators.get(sym.as_str())
                                        .map(|bank| bank.latest_snap())
                                        .unwrap_or_else(|| make_minimal_snap(0.0, *sub_ts));
                                    let sub_snap = make_exit_snap(&base_snap, sub_bar);
                                    let slope = sub_bar_slopes.get(sym.as_str()).copied().unwrap_or(0.0);

                                    // Evaluate signal (same logic as try_sub_bar_entry but collect instead)
                                    if let Some(cand) = evaluate_sub_bar_candidate(
                                        &state, sym.as_str(), &sub_snap, cfg,
                                        btc_bullish, breadth_pct, slope, *sub_ts,
                                    ) {
                                        sub_candidates.push(cand);
                                    }
                                }
                            }
                        }
                    }
                }

                // Rank sub-bar candidates by score (same formula as indicator-bar)
                if !sub_candidates.is_empty() {
                    sub_candidates.sort_by(|a, b| {
                        let score_a = (a.confidence as i32) * 100 + a.adx as i32;
                        let score_b = (b.confidence as i32) * 100 + b.adx as i32;
                        score_b.cmp(&score_a)
                            .then_with(|| a.symbol.cmp(&b.symbol))
                    });

                    for cand in &sub_candidates {
                        if entries_this_bar >= cfg.trade.max_entry_orders_per_loop {
                            break;
                        }
                        // Re-check: position may have been opened for this symbol by a higher-ranked candidate
                        if state.positions.contains_key(&cand.symbol) {
                            continue;
                        }
                        if state.positions.len() >= cfg.trade.max_open_positions {
                            break;
                        }

                        let opened = execute_sub_bar_entry(
                            &mut state, &cand.symbol, &cand.snap, cfg,
                            cand.confidence, cand.atr, cand.entry_adx_threshold,
                            cand.signal, cand.ts, sub_equity,
                        );
                        if opened {
                            entries_this_bar += 1;
                        }
                    }
                }
            }
        }

        // ── Funding rate payments at hourly boundaries ───────────────────
        // Funding is settled every hour at the :00 boundary on Hyperliquid.
        // Check if this bar crosses an hourly boundary and apply funding to open positions.
        if let Some(fr) = funding_rates {
            if !state.positions.is_empty() {
                let ts_i = timestamps.binary_search(&ts).unwrap_or(0);
                let prev_ts = if ts_i > 0 { timestamps[ts_i - 1] } else { ts };

                // Find hourly boundaries crossed between prev_ts (exclusive) and ts (inclusive).
                // Hourly boundary = timestamp divisible by 3_600_000 ms.
                let hour_ms: i64 = 3_600_000;
                let first_boundary = ((prev_ts / hour_ms) + 1) * hour_ms;

                let mut boundary = first_boundary;
                while boundary <= ts {
                    let open_syms: Vec<String> = state.positions.keys().cloned().collect();
                    for sym in open_syms {
                        if let Some(pos) = state.positions.get(&sym) {
                            if let Some(rates) = fr.get(&sym) {
                                // Find the funding rate at this boundary
                                if let Some(rate) = lookup_funding_rate(rates, boundary) {
                                    // Funding: shorts receive, longs pay (when rate > 0)
                                    // delta = -signed_size * price * rate
                                    let price = state.indicators.get(&sym)
                                        .map(|b| b.prev_close)
                                        .unwrap_or(pos.entry_price);
                                    let signed_size = match pos.pos_type {
                                        PositionType::Long => pos.size,
                                        PositionType::Short => -pos.size,
                                    };
                                    let delta_usd = -signed_size * price * rate;
                                    state.balance += delta_usd;

                                    state.trades.push(TradeRecord {
                                        timestamp_ms: boundary,
                                        symbol: sym.to_string(),
                                        action: "FUNDING".to_string(),
                                        price,
                                        size: pos.size,
                                        notional: pos.size * price,
                                        reason: format!("Funding rate={:.6}", rate),
                                        confidence: pos.confidence,
                                        pnl: delta_usd,
                                        fee_usd: 0.0,
                                        balance: state.balance,
                                        entry_atr: pos.entry_atr,
                                        leverage: pos.leverage,
                                        margin_used: pos.margin_used,
                                        ..Default::default()
                                    });
                                }
                            }
                        }
                    }
                    boundary += hour_ms;
                }
            }
        }

        // Record equity curve point
        let unrealized = unrealized_pnl_simple(&state.positions, &timestamps, ts, candles, &bar_index);
        state
            .equity_curve
            .push((ts, state.balance + unrealized));
    }

    // -- Force-close all remaining positions at last known price --
    let remaining: Vec<String> = state.positions.keys().cloned().collect();
    for sym in remaining {
        if state.positions.contains_key(&sym) {
            let last_price = last_price_for_symbol(candles, &sym);
            let exit = ExitResult::exit("End of Backtest", last_price);
            let snap = make_minimal_snap(last_price, *timestamps.last().unwrap_or(&0));
            apply_exit(&mut state, &sym, &exit, &snap, *timestamps.last().unwrap_or(&0));
        }
    }

    SimResult {
        trades: state.trades,
        signals: state.signals,
        final_balance: state.balance,
        equity_curve: state.equity_curve,
        gate_stats: state.gate_stats,
    }
}

// ---------------------------------------------------------------------------
// Timeline construction
// ---------------------------------------------------------------------------

fn build_timeline(candles: &CandleData) -> Vec<i64> {
    let mut set: Vec<i64> = candles
        .values()
        .flat_map(|bars| bars.iter().map(|b| b.t))
        .collect();
    set.sort_unstable();
    set.dedup();
    set
}

/// Build a per-symbol HashMap<timestamp -> index> for O(1) bar lookup.
fn build_bar_index(candles: &CandleData) -> FxHashMap<String, FxHashMap<i64, usize>> {
    let mut idx: FxHashMap<String, FxHashMap<i64, usize>> = FxHashMap::default();
    for (sym, bars) in candles {
        let mut sym_idx = FxHashMap::default();
        for (i, bar) in bars.iter().enumerate() {
            sym_idx.insert(bar.t, i);
        }
        idx.insert(sym.clone(), sym_idx);
    }
    idx
}

fn lookup_bar<'a>(
    index: &FxHashMap<String, FxHashMap<i64, usize>>,
    symbol: &str,
    ts: i64,
    candles: &'a CandleData,
) -> Option<&'a OhlcvBar> {
    let sym_idx = index.get(symbol)?;
    let &i = sym_idx.get(&ts)?;
    candles.get(symbol).and_then(|bars| bars.get(i))
}

// ---------------------------------------------------------------------------
// BTC context
// ---------------------------------------------------------------------------

fn compute_btc_bullish(
    state: &SimState,
    _ts: i64,
    _bar_index: &FxHashMap<String, FxHashMap<i64, usize>>,
    _candles: &CandleData,
) -> Option<bool> {
    // Check if we have BTC indicators
    if let Some(bank) = state.indicators.get("BTC") {
        if bank.bar_count > 0 {
            // BTC is bullish when price > EMA_slow
            return Some(bank.prev_close > bank.prev_ema_slow);
        }
    }
    // Also check BTCUSDT variant
    if let Some(bank) = state.indicators.get("BTCUSDT") {
        if bank.bar_count > 0 {
            return Some(bank.prev_close > bank.prev_ema_slow);
        }
    }
    None // No BTC data available
}

// ---------------------------------------------------------------------------
// Market breadth
// ---------------------------------------------------------------------------

fn compute_market_breadth(state: &SimState) -> f64 {
    let mut bullish = 0u32;
    let mut total = 0u32;
    for bank in state.indicators.values() {
        if bank.bar_count < 2 {
            continue;
        }
        total += 1;
        // EMA_fast > EMA_slow means bullish
        if bank.prev_ema_fast > bank.prev_ema_slow {
            bullish += 1;
        }
    }
    if total == 0 {
        return 50.0;
    }
    (bullish as f64 / total as f64) * 100.0
}

// ---------------------------------------------------------------------------
// EMA slow slope
// ---------------------------------------------------------------------------

fn compute_ema_slow_slope(history: &[f64], window: usize, current_close: f64) -> f64 {
    if history.len() < window || current_close <= 0.0 {
        return 0.0;
    }
    let current = history[history.len() - 1];
    let past = history[history.len() - window];
    (current - past) / current_close
}

// ---------------------------------------------------------------------------
// Engine-level transforms
// ---------------------------------------------------------------------------

fn apply_atr_floor(atr: f64, price: f64, min_atr_pct: f64) -> f64 {
    if min_atr_pct > 0.0 {
        let floor = price * min_atr_pct;
        if atr < floor {
            return floor;
        }
    }
    atr
}

fn apply_reverse(signal: Signal, cfg: &StrategyConfig, breadth_pct: f64) -> Signal {
    let mut should_reverse = cfg.trade.reverse_entry_signal;

    // Auto-reverse based on market breadth
    if cfg.market_regime.enable_auto_reverse {
        let low = cfg.market_regime.auto_reverse_breadth_low;
        let high = cfg.market_regime.auto_reverse_breadth_high;
        if breadth_pct >= low && breadth_pct <= high {
            should_reverse = true; // Choppy market → fade
        } else {
            should_reverse = false; // Trending → follow
        }
    }

    if should_reverse {
        match signal {
            Signal::Buy => Signal::Sell,
            Signal::Sell => Signal::Buy,
            Signal::Neutral => Signal::Neutral,
        }
    } else {
        signal
    }
}

fn apply_regime_filter(signal: Signal, cfg: &StrategyConfig, breadth_pct: f64) -> Signal {
    if !cfg.market_regime.enable_regime_filter {
        return signal;
    }
    match signal {
        Signal::Sell if breadth_pct > cfg.market_regime.breadth_block_short_above => {
            Signal::Neutral // Block SHORT in strong bull market
        }
        Signal::Buy if breadth_pct < cfg.market_regime.breadth_block_long_below => {
            Signal::Neutral // Block LONG in strong bear market
        }
        _ => signal,
    }
}

// ---------------------------------------------------------------------------
// PESC (Post-Exit Same-Direction Cooldown)
// ---------------------------------------------------------------------------

fn is_pesc_blocked(
    state: &SimState,
    symbol: &str,
    desired_type: PositionType,
    current_ts: i64,
    adx: f64,
    cfg: &StrategyConfig,
) -> bool {
    // Gate: reentry_cooldown_minutes == 0 disables PESC entirely (matches Python engine)
    if cfg.trade.reentry_cooldown_minutes == 0 {
        return false;
    }

    let (close_ts, close_type, ref close_reason) = match state.last_close.get(symbol) {
        Some(v) => v.clone(),
        None => return false,
    };

    // No cooldown after signal flips
    if close_reason == "Signal Flip" {
        return false;
    }

    // Only applies to same direction
    if close_type != desired_type {
        return false;
    }

    // ADX-adaptive cooldown interpolation
    let min_cd = cfg.trade.reentry_cooldown_min_mins as f64;
    let max_cd = cfg.trade.reentry_cooldown_max_mins as f64;

    let cooldown_mins = if adx >= 40.0 {
        min_cd
    } else if adx <= 25.0 {
        max_cd
    } else {
        // Linear interpolation: ADX 25→40 maps to max_cd→min_cd
        let t = (adx - 25.0) / 15.0;
        max_cd + t * (min_cd - max_cd)
    };

    let cooldown_ms = (cooldown_mins * 60_000.0) as i64;
    let elapsed = current_ts - close_ts;

    elapsed < cooldown_ms
}

// ---------------------------------------------------------------------------
// Entry sizing
// ---------------------------------------------------------------------------

fn compute_entry_size(
    _symbol: &str,
    equity: f64,
    price: f64,
    confidence: Confidence,
    atr: f64,
    snap: &IndicatorSnapshot,
    cfg: &StrategyConfig,
) -> (f64, f64, f64) {
    let tc = &cfg.trade;

    // Base margin
    let mut margin = equity * tc.allocation_pct;

    if tc.enable_dynamic_sizing {
        // Confidence multiplier
        let conf_mult = match confidence {
            Confidence::High => tc.confidence_mult_high,
            Confidence::Medium => tc.confidence_mult_medium,
            Confidence::Low => tc.confidence_mult_low,
        };

        // ADX sizing multiplier: linear interpolation
        let adx_mult = (snap.adx / tc.adx_sizing_full_adx).clamp(tc.adx_sizing_min_mult, 1.0);

        // Volatility scalar: ATR/price normalized by vol_baseline_pct, then inverted
        let vol_ratio = if tc.vol_baseline_pct > 0.0 && price > 0.0 {
            (atr / price) / tc.vol_baseline_pct
        } else {
            1.0
        };
        let vol_scalar_raw = if vol_ratio > 0.0 { 1.0 / vol_ratio } else { 1.0 };
        let vol_scalar = vol_scalar_raw.clamp(tc.vol_scalar_min, tc.vol_scalar_max);

        margin *= conf_mult * adx_mult * vol_scalar;
    }

    // Leverage
    let leverage = if tc.enable_dynamic_leverage {
        let base_lev = match confidence {
            Confidence::High => tc.leverage_high,
            Confidence::Medium => tc.leverage_medium,
            Confidence::Low => tc.leverage_low,
        };
        if tc.leverage_max_cap > 0.0 {
            base_lev.min(tc.leverage_max_cap)
        } else {
            base_lev
        }
    } else {
        tc.leverage
    };

    let notional = margin * leverage;
    let size = if price > 0.0 { notional / price } else { 0.0 };

    (size, margin, leverage)
}

// ---------------------------------------------------------------------------
// Exit application
// ---------------------------------------------------------------------------

fn apply_exit(
    state: &mut SimState,
    symbol: &str,
    exit: &ExitResult,
    snap: &IndicatorSnapshot,
    ts: i64,
) {
    let pos = match state.positions.get(symbol) {
        Some(p) => p.clone(),
        None => return,
    };

    let exit_price = if exit.exit_price > 0.0 {
        exit.exit_price
    } else {
        snap.close
    };

    // Apply slippage to exit
    let fill_price = match pos.pos_type {
        PositionType::Long => exit_price * (1.0 - 0.5 / 10_000.0), // Conservative: half a bps
        PositionType::Short => exit_price * (1.0 + 0.5 / 10_000.0),
    };

    if let Some(partial_pct) = exit.partial_pct {
        // Partial exit
        let exit_size = pos.size * partial_pct;
        let exit_notional = exit_size * fill_price;
        let fee_usd = exit_notional * FEE_RATE;

        let pnl = match pos.pos_type {
            PositionType::Long => (fill_price - pos.entry_price) * exit_size,
            PositionType::Short => (pos.entry_price - fill_price) * exit_size,
        };

        state.balance += pnl - fee_usd;

        // Record PESC info (partial exits also count)
        state.last_close.insert(
            symbol.to_string(),
            (ts, pos.pos_type, exit.reason.clone()),
        );

        let action = match pos.pos_type {
            PositionType::Long => "REDUCE_LONG",
            PositionType::Short => "REDUCE_SHORT",
        };
        state.trades.push(TradeRecord {
            timestamp_ms: ts,
            symbol: symbol.to_string(),
            action: action.to_string(),
            price: fill_price,
            size: exit_size,
            notional: exit_notional,
            reason: exit.reason.clone(),
            confidence: pos.confidence,
            pnl,
            fee_usd,
            balance: state.balance,
            entry_atr: pos.entry_atr,
            leverage: pos.leverage,
            margin_used: pos.margin_used * partial_pct,
            ..Default::default()
        });

        // Reduce position in place
        if let Some(p) = state.positions.get_mut(symbol) {
            p.reduce_by_fraction(partial_pct);
            // Partial exits represent TP1 in the current engine contract.
            // Do not couple this state transition to human-readable reason text.
            p.tp1_taken = true;
        }
    } else {
        // Full exit
        let exit_notional = pos.size * fill_price;
        let fee_usd = exit_notional * FEE_RATE;

        let pnl = match pos.pos_type {
            PositionType::Long => (fill_price - pos.entry_price) * pos.size,
            PositionType::Short => (pos.entry_price - fill_price) * pos.size,
        };

        state.balance += pnl - fee_usd;

        // Record PESC info
        state.last_close.insert(
            symbol.to_string(),
            (ts, pos.pos_type, exit.reason.clone()),
        );

        let action = match pos.pos_type {
            PositionType::Long => "CLOSE_LONG",
            PositionType::Short => "CLOSE_SHORT",
        };
        state.trades.push(TradeRecord {
            timestamp_ms: ts,
            symbol: symbol.to_string(),
            action: action.to_string(),
            price: fill_price,
            size: pos.size,
            notional: exit_notional,
            reason: exit.reason.clone(),
            confidence: pos.confidence,
            pnl,
            fee_usd,
            balance: state.balance,
            entry_atr: pos.entry_atr,
            leverage: pos.leverage,
            margin_used: pos.margin_used,
            ..Default::default()
        });

        state.positions.remove(symbol);
    }
}

// ---------------------------------------------------------------------------
// Trailing stop update
// ---------------------------------------------------------------------------

fn update_trailing_stop(
    state: &mut SimState,
    symbol: &str,
    snap: &IndicatorSnapshot,
    cfg: &StrategyConfig,
) {
    let pos = match state.positions.get_mut(symbol) {
        Some(p) => p,
        None => return,
    };

    let profit_atr = pos.profit_atr(snap.close);
    let trailing_start = cfg.trade.trailing_start_atr;
    let trailing_dist = cfg.trade.trailing_distance_atr;

    if profit_atr < trailing_start {
        return;
    }

    let new_sl = match pos.pos_type {
        PositionType::Long => snap.close - trailing_dist * pos.entry_atr,
        PositionType::Short => snap.close + trailing_dist * pos.entry_atr,
    };

    match pos.trailing_sl {
        Some(current_sl) => {
            match pos.pos_type {
                PositionType::Long => {
                    if new_sl > current_sl {
                        pos.trailing_sl = Some(new_sl);
                    }
                }
                PositionType::Short => {
                    if new_sl < current_sl {
                        pos.trailing_sl = Some(new_sl);
                    }
                }
            }
        }
        None => {
            pos.trailing_sl = Some(new_sl);
        }
    }

    // Breakeven stop
    if cfg.trade.enable_breakeven_stop && profit_atr >= cfg.trade.breakeven_start_atr {
        let be_price = match pos.pos_type {
            PositionType::Long => pos.entry_price + cfg.trade.breakeven_buffer_atr * pos.entry_atr,
            PositionType::Short => {
                pos.entry_price - cfg.trade.breakeven_buffer_atr * pos.entry_atr
            }
        };
        match pos.pos_type {
            PositionType::Long => {
                if let Some(ref mut sl) = pos.trailing_sl {
                    if be_price > *sl {
                        *sl = be_price;
                    }
                } else {
                    pos.trailing_sl = Some(be_price);
                }
            }
            PositionType::Short => {
                if let Some(ref mut sl) = pos.trailing_sl {
                    if be_price < *sl {
                        *sl = be_price;
                    }
                } else {
                    pos.trailing_sl = Some(be_price);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pyramiding
// ---------------------------------------------------------------------------

fn try_pyramid(
    state: &mut SimState,
    symbol: &str,
    snap: &IndicatorSnapshot,
    cfg: &StrategyConfig,
    confidence: Confidence,
    atr: f64,
    ts: i64,
) {
    let tc = &cfg.trade;
    let pos = match state.positions.get(symbol) {
        Some(p) => p,
        None => return,
    };

    // Check add limits
    if pos.adds_count >= tc.max_adds_per_symbol as u32 {
        return;
    }

    // Confidence gate for adds
    if !confidence.meets_min(tc.add_min_confidence) {
        return;
    }

    // Cooldown since last add
    let elapsed_mins = (ts - pos.last_add_time_ms) as f64 / 60_000.0;
    if elapsed_mins < tc.add_cooldown_minutes as f64 {
        return;
    }

    // Must be in profit by min ATR
    let profit_atr = pos.profit_atr(snap.close);
    if profit_atr < tc.add_min_profit_atr {
        return;
    }

    // Compute add size
    let equity = state.balance + unrealized_pnl_for_positions(&state.positions, snap.close);
    let base_margin = equity * tc.allocation_pct;
    let add_margin = base_margin * tc.add_fraction_of_base_margin;

    let leverage = pos.leverage;
    let mut add_notional = add_margin * leverage;
    let mut add_size = if snap.close > 0.0 {
        add_notional / snap.close
    } else {
        return;
    };

    if add_notional < tc.min_notional_usd {
        if tc.bump_to_min_notional && snap.close > 0.0 {
            add_notional = tc.min_notional_usd;
            add_size = add_notional / snap.close;
        } else {
            return;
        }
    }

    // Margin cap check
    let total_margin: f64 = state.positions.values().map(|p| p.margin_used).sum();
    if total_margin + add_margin > equity * tc.max_total_margin_pct {
        return;
    }

    let fill_price = match pos.pos_type {
        PositionType::Long => snap.close * (1.0 + tc.slippage_bps / 10_000.0),
        PositionType::Short => snap.close * (1.0 - tc.slippage_bps / 10_000.0),
    };

    let fee_usd = add_notional * FEE_RATE;
    state.balance -= fee_usd;

    let action = match pos.pos_type {
        PositionType::Long => "ADD_LONG",
        PositionType::Short => "ADD_SHORT",
    };
    state.trades.push(TradeRecord {
        timestamp_ms: ts,
        symbol: symbol.to_string(),
        action: action.to_string(),
        price: fill_price,
        size: add_size,
        notional: add_notional,
        reason: format!("Pyramid #{}", pos.adds_count + 1),
        confidence,
        pnl: 0.0,
        fee_usd,
        balance: state.balance,
        entry_atr: atr,
        leverage,
        margin_used: add_margin,
        ..Default::default()
    });

    // Update position
    if let Some(p) = state.positions.get_mut(symbol) {
        p.add_to_position(fill_price, add_size, add_margin, ts);
    }
}

// ---------------------------------------------------------------------------
// Unrealized PnL helpers
// ---------------------------------------------------------------------------

/// Compute total unrealized PnL using last known price from indicator banks.
fn unrealized_pnl(
    positions: &FxHashMap<String, Position>,
    indicators: &FxHashMap<String, IndicatorBank>,
    _ts: i64,
    _candles: &CandleData,
    _bar_index: &FxHashMap<String, FxHashMap<i64, usize>>,
) -> f64 {
    positions
        .values()
        .map(|pos| {
            let price = indicators
                .get(&pos.symbol)
                .map(|b| b.prev_close)
                .unwrap_or(pos.entry_price);
            pos.profit_usd(price)
        })
        .sum()
}

/// Simple unrealized PnL for equity curve (uses last indicator close).
fn unrealized_pnl_simple(
    positions: &FxHashMap<String, Position>,
    _timestamps: &[i64],
    _ts: i64,
    _candles: &CandleData,
    _bar_index: &FxHashMap<String, FxHashMap<i64, usize>>,
) -> f64 {
    positions
        .values()
        .map(|pos| {
            // Use entry price as fallback (will be updated by indicator banks)
            pos.profit_usd(pos.entry_price) // This is 0 for just-opened positions
        })
        .sum()
}

/// Quick unrealized PnL using a single snapshot price (for pyramid sizing).
fn unrealized_pnl_for_positions(
    positions: &FxHashMap<String, Position>,
    _current_price: f64,
) -> f64 {
    // Approximate: each position at its entry (gives 0 unrealized)
    // In practice during pyramid check we just use balance as equity
    positions
        .values()
        .map(|pos| pos.profit_usd(pos.entry_price))
        .sum()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn last_price_for_symbol(candles: &CandleData, symbol: &str) -> f64 {
    candles
        .get(symbol)
        .and_then(|bars| bars.last())
        .map(|b| b.c)
        .unwrap_or(0.0)
}

/// Evaluate whether a sub-bar entry candidate should be collected for ranking.
/// Returns Some(EntryCandidate) if the signal passes gates, None otherwise.
fn evaluate_sub_bar_candidate(
    state: &SimState,
    sym: &str,
    snap: &IndicatorSnapshot,
    cfg: &StrategyConfig,
    btc_bullish: Option<bool>,
    breadth_pct: f64,
    ema_slow_slope_pct: f64,
    ts: i64,
) -> Option<EntryCandidate> {
    if state.positions.contains_key(sym) {
        return None;
    }

    let gate_result = gates::check_gates(snap, cfg, sym, btc_bullish, ema_slow_slope_pct);
    let (mut signal, confidence, entry_adx_threshold) =
        entry::generate_signal(snap, &gate_result, cfg, ema_slow_slope_pct);

    if signal == Signal::Neutral {
        return None;
    }

    let atr = apply_atr_floor(snap.atr, snap.close, cfg.trade.min_atr_pct);
    signal = apply_reverse(signal, cfg, breadth_pct);
    if signal == Signal::Neutral {
        return None;
    }
    signal = apply_regime_filter(signal, cfg, breadth_pct);
    if signal == Signal::Neutral {
        return None;
    }

    let desired_type = match signal {
        Signal::Buy => PositionType::Long,
        Signal::Sell => PositionType::Short,
        Signal::Neutral => return None,
    };

    if !confidence.meets_min(cfg.trade.entry_min_confidence) {
        return None;
    }
    if is_pesc_blocked(state, sym, desired_type, ts, snap.adx, cfg) {
        return None;
    }
    if cfg.trade.enable_ssf_filter {
        let ssf_ok = match signal {
            Signal::Buy => snap.macd_hist > 0.0,
            Signal::Sell => snap.macd_hist < 0.0,
            Signal::Neutral => true,
        };
        if !ssf_ok {
            return None;
        }
    }
    if cfg.trade.enable_reef_filter {
        let reef_blocked = match signal {
            Signal::Buy => {
                if snap.adx < cfg.trade.reef_adx_threshold {
                    snap.rsi > cfg.trade.reef_long_rsi_block_gt
                } else {
                    snap.rsi > cfg.trade.reef_long_rsi_extreme_gt
                }
            }
            Signal::Sell => {
                if snap.adx < cfg.trade.reef_adx_threshold {
                    snap.rsi < cfg.trade.reef_short_rsi_block_lt
                } else {
                    snap.rsi < cfg.trade.reef_short_rsi_extreme_lt
                }
            }
            Signal::Neutral => false,
        };
        if reef_blocked {
            return None;
        }
    }

    Some(EntryCandidate {
        symbol: sym.to_string(),
        signal,
        confidence,
        adx: snap.adx,
        atr,
        entry_adx_threshold,
        snap: snap.clone(),
        ts,
    })
}

/// Execute a ranked sub-bar entry candidate. Opens a new position.
/// Returns true if a new position was opened.
fn execute_sub_bar_entry(
    state: &mut SimState,
    sym: &str,
    snap: &IndicatorSnapshot,
    cfg: &StrategyConfig,
    confidence: Confidence,
    atr: f64,
    entry_adx_threshold: f64,
    signal: Signal,
    ts: i64,
    equity: f64,
) -> bool {
    if state.positions.contains_key(sym) {
        return false;
    }

    let desired_type = match signal {
        Signal::Buy => PositionType::Long,
        Signal::Sell => PositionType::Short,
        Signal::Neutral => return false,
    };

    // Margin cap check
    let total_margin: f64 = state.positions.values().map(|p| p.margin_used).sum();
    let margin_headroom = equity * cfg.trade.max_total_margin_pct - total_margin;
    if margin_headroom <= 0.0 {
        return false;
    }

    // Sizing
    let (size, margin_used, leverage) =
        compute_entry_size(sym, equity, snap.close, confidence, atr, snap, cfg);
    let (mut size, mut margin_used) = if margin_used > margin_headroom {
        let ratio = margin_headroom / margin_used;
        (size * ratio, margin_headroom)
    } else {
        (size, margin_used)
    };

    let mut notional = size * snap.close;
    if notional < cfg.trade.min_notional_usd {
        if cfg.trade.bump_to_min_notional && snap.close > 0.0 {
            size = cfg.trade.min_notional_usd / snap.close;
            margin_used = size * snap.close / leverage;
            notional = size * snap.close;
        } else {
            return false;
        }
    }

    let fill_price = match signal {
        Signal::Buy => snap.close * (1.0 + cfg.trade.slippage_bps / 10_000.0),
        Signal::Sell => snap.close * (1.0 - cfg.trade.slippage_bps / 10_000.0),
        Signal::Neutral => return false,
    };
    let fee_usd = notional * FEE_RATE;
    state.balance -= fee_usd;

    let pos = Position {
        symbol: sym.to_string(),
        pos_type: desired_type,
        entry_price: fill_price,
        size,
        confidence,
        entry_atr: atr,
        entry_adx_threshold,
        trailing_sl: None,
        leverage,
        margin_used,
        adds_count: 0,
        tp1_taken: false,
        open_time_ms: ts,
        last_add_time_ms: ts,
        mae_usd: 0.0,
        mfe_usd: 0.0,
    };
    state.positions.insert(sym.to_string(), pos);

    let action = match desired_type {
        PositionType::Long => "OPEN_LONG",
        PositionType::Short => "OPEN_SHORT",
    };
    state.trades.push(TradeRecord {
        timestamp_ms: ts,
        symbol: sym.to_string(),
        action: action.to_string(),
        price: fill_price,
        size,
        notional,
        reason: format!("{:?} entry (sub-bar)", confidence),
        confidence,
        pnl: 0.0,
        fee_usd,
        balance: state.balance,
        entry_atr: atr,
        leverage,
        margin_used,
        ..Default::default()
    });

    true
}

/// Create a lightweight IndicatorSnapshot for sub-bar exit checks.
/// Copies all indicator values from the 1h snap, but overrides OHLC + time
/// with the sub-bar (e.g. 1m) values. This way exit conditions see the correct
/// price while indicator thresholds (ADX, RSI, etc.) stay at hourly resolution.
fn make_exit_snap(indicator_snap: &IndicatorSnapshot, exit_bar: &OhlcvBar) -> IndicatorSnapshot {
    let mut s = indicator_snap.clone();
    s.close = exit_bar.c;
    s.high = exit_bar.h;
    s.low = exit_bar.l;
    s.open = exit_bar.o;
    s.t = exit_bar.t;
    s
}

/// Binary-search for the funding rate at or just before `target_ts`.
fn lookup_funding_rate(rates: &[(i64, f64)], target_ts: i64) -> Option<f64> {
    match rates.binary_search_by_key(&target_ts, |(t, _)| *t) {
        Ok(i) => Some(rates[i].1),
        Err(i) => {
            // Use the rate from the most recent funding settlement before target_ts
            if i > 0 {
                Some(rates[i - 1].1)
            } else {
                None
            }
        }
    }
}

/// Create a minimal IndicatorSnapshot for force-close at end of backtest.
fn make_minimal_snap(price: f64, ts: i64) -> IndicatorSnapshot {
    IndicatorSnapshot {
        close: price,
        high: price,
        low: price,
        open: price,
        volume: 0.0,
        t: ts,
        ema_slow: price,
        ema_fast: price,
        ema_macro: price,
        adx: 0.0,
        adx_pos: 0.0,
        adx_neg: 0.0,
        adx_slope: 0.0,
        bb_upper: price,
        bb_lower: price,
        bb_width: 0.0,
        bb_width_avg: 0.0,
        bb_width_ratio: 1.0,
        atr: 0.0,
        atr_slope: 0.0,
        avg_atr: 0.0,
        rsi: 50.0,
        stoch_rsi_k: 0.5,
        stoch_rsi_d: 0.5,
        macd_hist: 0.0,
        prev_macd_hist: 0.0,
        prev2_macd_hist: 0.0,
        prev3_macd_hist: 0.0,
        vol_sma: 0.0,
        vol_trend: false,
        prev_close: price,
        prev_ema_fast: price,
        prev_ema_slow: price,
        bar_count: 0,
        funding_rate: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_floor() {
        assert!((apply_atr_floor(0.5, 100.0, 0.01) - 1.0).abs() < 1e-9);
        assert!((apply_atr_floor(2.0, 100.0, 0.01) - 2.0).abs() < 1e-9);
        assert!((apply_atr_floor(0.5, 100.0, 0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reverse_signal() {
        let mut cfg = StrategyConfig::default();
        cfg.trade.reverse_entry_signal = true;
        assert_eq!(apply_reverse(Signal::Buy, &cfg, 50.0), Signal::Sell);
        assert_eq!(apply_reverse(Signal::Sell, &cfg, 50.0), Signal::Buy);
        assert_eq!(apply_reverse(Signal::Neutral, &cfg, 50.0), Signal::Neutral);
    }

    #[test]
    fn test_regime_filter() {
        let mut cfg = StrategyConfig::default();
        cfg.market_regime.enable_regime_filter = true;
        cfg.market_regime.breadth_block_short_above = 80.0;
        cfg.market_regime.breadth_block_long_below = 20.0;
        assert_eq!(apply_regime_filter(Signal::Sell, &cfg, 85.0), Signal::Neutral);
        assert_eq!(apply_regime_filter(Signal::Buy, &cfg, 15.0), Signal::Neutral);
        assert_eq!(apply_regime_filter(Signal::Buy, &cfg, 50.0), Signal::Buy);
        assert_eq!(apply_regime_filter(Signal::Sell, &cfg, 50.0), Signal::Sell);
    }

    #[test]
    fn test_ema_slow_slope() {
        let history = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let slope = compute_ema_slow_slope(&history, 3, 104.0);
        // (104 - 102) / 104 ≈ 0.01923
        assert!((slope - 2.0 / 104.0).abs() < 1e-9);
    }

    #[test]
    fn test_ema_slow_slope_too_short() {
        let history = vec![100.0, 101.0];
        let slope = compute_ema_slow_slope(&history, 5, 101.0);
        assert!((slope - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_candles() {
        let candles = FxHashMap::default();
        let cfg = StrategyConfig::default();
        let result = run_simulation(&candles, &cfg, 1000.0, 50, None, None, None, None, None, None);
        assert_eq!(result.trades.len(), 0);
        assert!((result.final_balance - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_entry_size_basic() {
        let cfg = StrategyConfig::default();
        let snap = make_minimal_snap(100.0, 0);
        let (size, margin, leverage) =
            compute_entry_size("ETH", 10000.0, 100.0, Confidence::High, 1.0, &snap, &cfg);
        // allocation_pct=0.03, dynamic sizing enabled, confidence_mult_high=1.0
        // adx=0 → adx_mult=min(0/40, 1.0).clamp(0.6, 1.0) = 0.6
        // vol_ratio = (1.0/100.0) / 0.01 = 1.0 → vol_scalar = 1.0
        // margin = 10000 * 0.03 * 1.0 * 0.6 * 1.0 = 180
        // leverage_high = 5.0
        // notional = 180 * 5 = 900
        // size = 900 / 100 = 9
        assert!(size > 0.0);
        assert!(margin > 0.0);
        assert!(leverage > 0.0);
    }

    #[test]
    fn test_market_breadth() {
        let state = SimState {
            balance: 1000.0,
            positions: FxHashMap::default(),
            indicators: FxHashMap::default(),
            ema_slow_history: FxHashMap::default(),
            bar_counts: FxHashMap::default(),
            last_close: FxHashMap::default(),
            trades: vec![],
            signals: vec![],
            equity_curve: vec![],
            gate_stats: GateStats::default(),
        };
        // No indicators → 50%
        assert!((compute_market_breadth(&state) - 50.0).abs() < 1e-9);
    }

    fn make_state_with_open_long(symbol: &str) -> SimState {
        let mut positions = FxHashMap::default();
        positions.insert(
            symbol.to_string(),
            Position {
                symbol: symbol.to_string(),
                pos_type: PositionType::Long,
                entry_price: 100.0,
                size: 1.0,
                confidence: Confidence::High,
                entry_atr: 1.0,
                entry_adx_threshold: 20.0,
                trailing_sl: None,
                leverage: 3.0,
                margin_used: 33.333_333,
                adds_count: 0,
                tp1_taken: false,
                open_time_ms: 0,
                last_add_time_ms: 0,
                mae_usd: 0.0,
                mfe_usd: 0.0,
            },
        );

        SimState {
            balance: 1_000.0,
            positions,
            indicators: FxHashMap::default(),
            ema_slow_history: FxHashMap::default(),
            bar_counts: FxHashMap::default(),
            last_close: FxHashMap::default(),
            trades: vec![],
            signals: vec![],
            equity_curve: vec![],
            gate_stats: GateStats::default(),
        }
    }

    #[test]
    fn test_apply_exit_partial_take_profit_marks_tp1_taken() {
        let symbol = "BTC";
        let mut state = make_state_with_open_long(symbol);
        let snap = make_minimal_snap(105.0, 1_700_000_000_000);
        let exit = ExitResult::partial_exit("Take Profit (Partial)", 105.0, 0.5);

        apply_exit(&mut state, symbol, &exit, &snap, snap.t);

        let pos = state
            .positions
            .get(symbol)
            .expect("position should remain after partial exit");
        assert!(pos.tp1_taken);
        assert!((pos.size - 0.5).abs() < 1e-12);
        assert_eq!(state.trades.len(), 1);
        assert_eq!(state.trades[0].action, "REDUCE_LONG");
    }

    #[test]
    fn test_apply_exit_partial_marks_tp1_taken_without_reason_match() {
        let symbol = "ETH";
        let mut state = make_state_with_open_long(symbol);
        let snap = make_minimal_snap(103.0, 1_700_000_000_001);
        let exit = ExitResult::partial_exit("Risk Trim", 103.0, 0.25);

        apply_exit(&mut state, symbol, &exit, &snap, snap.t);

        let pos = state
            .positions
            .get(symbol)
            .expect("position should remain after partial exit");
        assert!(pos.tp1_taken);
        assert!((pos.size - 0.75).abs() < 1e-12);
        assert_eq!(state.trades.len(), 1);
        assert_eq!(state.trades[0].action, "REDUCE_LONG");
    }
}
