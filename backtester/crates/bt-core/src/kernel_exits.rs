//! Kernel-level exit evaluation (SL, trailing, TP with partial support).
//!
//! Mirrors the priority logic from `exits/mod.rs::check_all_exits()` but operates
//! on the kernel's own `Position` type.  Glitch guard (AQC-712) blocks exits during
//! extreme price deviations.  Smart exits (AQC-713) will be added in a later ticket.

use crate::behaviour::ResolvedBehaviourPlan;
use crate::decision_kernel::{Position, PositionSide};
use crate::indicators::IndicatorSnapshot;
use bt_signals::behaviour::{BehaviourGroupPlan, BehaviourTrace};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// ExitParams — extracted from StrategyConfig trade fields
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExitParams {
    pub sl_atr_mult: f64,
    pub tp_atr_mult: f64,
    pub trailing_start_atr: f64,
    pub trailing_distance_atr: f64,
    pub enable_partial_tp: bool,
    pub tp_partial_pct: f64,
    pub tp_partial_atr_mult: f64,
    pub tp_partial_min_notional_usd: f64,
    pub enable_breakeven_stop: bool,
    pub breakeven_start_atr: f64,
    pub breakeven_buffer_atr: f64,
    pub enable_vol_buffered_trailing: bool,
    // Glitch guard (AQC-712)
    pub block_exits_on_extreme_dev: bool,
    pub glitch_price_dev_pct: f64,
    pub glitch_atr_mult: f64,
    // Smart exits (AQC-713 — not evaluated yet)
    pub smart_exit_adx_exhaustion_lt: f64,
    pub tsme_min_profit_atr: f64,
    pub tsme_require_adx_slope_negative: bool,
    pub enable_rsi_overextension_exit: bool,
    pub rsi_exit_profit_atr_switch: f64,
    pub rsi_exit_ub_lo_profit: f64,
    pub rsi_exit_ub_hi_profit: f64,
    pub rsi_exit_lb_lo_profit: f64,
    pub rsi_exit_lb_hi_profit: f64,
    // Low-confidence RSI overrides
    pub rsi_exit_ub_lo_profit_low_conf: f64,
    pub rsi_exit_lb_lo_profit_low_conf: f64,
    pub rsi_exit_ub_hi_profit_low_conf: f64,
    pub rsi_exit_lb_hi_profit_low_conf: f64,
    // Low-confidence ADX exhaustion override
    pub smart_exit_adx_exhaustion_lt_low_conf: f64,
    // Macro alignment (from filters config)
    pub require_macro_alignment: bool,
    // Per-confidence trailing overrides
    pub trailing_start_atr_low_conf: f64,
    pub trailing_distance_atr_low_conf: f64,
}

impl Default for ExitParams {
    fn default() -> Self {
        Self {
            sl_atr_mult: 2.0,
            tp_atr_mult: 4.0,
            trailing_start_atr: 1.0,
            trailing_distance_atr: 0.8,
            enable_partial_tp: true,
            tp_partial_pct: 0.5,
            tp_partial_atr_mult: 0.0,
            tp_partial_min_notional_usd: 10.0,
            enable_breakeven_stop: true,
            breakeven_start_atr: 0.7,
            breakeven_buffer_atr: 0.05,
            enable_vol_buffered_trailing: true,
            block_exits_on_extreme_dev: false,
            glitch_price_dev_pct: 0.40,
            glitch_atr_mult: 12.0,
            smart_exit_adx_exhaustion_lt: 18.0,
            tsme_min_profit_atr: 1.0,
            tsme_require_adx_slope_negative: true,
            enable_rsi_overextension_exit: true,
            rsi_exit_profit_atr_switch: 1.5,
            rsi_exit_ub_lo_profit: 80.0,
            rsi_exit_ub_hi_profit: 70.0,
            rsi_exit_lb_lo_profit: 20.0,
            rsi_exit_lb_hi_profit: 30.0,
            rsi_exit_ub_lo_profit_low_conf: 0.0,
            rsi_exit_lb_lo_profit_low_conf: 0.0,
            rsi_exit_ub_hi_profit_low_conf: 0.0,
            rsi_exit_lb_hi_profit_low_conf: 0.0,
            smart_exit_adx_exhaustion_lt_low_conf: 0.0,
            require_macro_alignment: false,
            trailing_start_atr_low_conf: 0.0,
            trailing_distance_atr_low_conf: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KernelExitResult
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum KernelExitResult {
    Hold,
    FullClose {
        reason: String,
        exit_price: f64,
    },
    PartialClose {
        reason: String,
        exit_price: f64,
        fraction: f64,
    },
}

/// Diagnostic exit-tunnel boundaries: the price levels at which the engine
/// would exit if breached.  Purely informational — no effect on decision logic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExitBounds {
    /// Composite take-profit level (full TP).
    pub upper_full: f64,
    /// TP1 partial level (None if partial TP disabled or already taken).
    pub upper_partial: Option<f64>,
    /// Composite stop-loss / trailing / breakeven level.
    pub lower_full: f64,
}

/// Bundled exit evaluation result with diagnostic threshold records and context.
#[derive(Debug, Clone)]
pub struct ExitEvaluation {
    pub result: KernelExitResult,
    /// Threshold records captured during exit evaluation.
    pub threshold_records: Vec<crate::decision_kernel::ThresholdRecord>,
    /// Exit context (populated when an exit fires).
    pub exit_context: Option<crate::decision_kernel::ExitContext>,
    /// Exit tunnel boundaries (always Some when a position exists).
    pub exit_bounds: Option<ExitBounds>,
    /// Behaviour-level trace for parity debugging.
    pub behaviour_trace: Vec<BehaviourTrace>,
}

#[derive(Debug, Clone, Copy)]
struct StopLossState {
    sl_mult: f64,
    sl_price: Option<f64>,
    breakeven_active: bool,
}

#[derive(Debug, Clone, Copy)]
struct TrailingState {
    start_atr: f64,
    distance_atr: f64,
    vol_buffer_active: bool,
    base_seen: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute TP1 partial price (None if disabled or already taken).
fn compute_partial_tp_price(
    pos: &Position,
    entry: f64,
    atr: f64,
    params: &ExitParams,
) -> Option<f64> {
    if !params.enable_partial_tp || pos.tp1_taken {
        return None;
    }
    let mult = if params.tp_partial_atr_mult > 0.0 {
        params.tp_partial_atr_mult
    } else {
        params.tp_atr_mult
    };
    Some(match pos.side {
        PositionSide::Long => entry + (atr * mult),
        PositionSide::Short => entry - (atr * mult),
    })
}

fn effective_atr(entry: f64, entry_atr: Option<f64>) -> f64 {
    match entry_atr {
        Some(a) if a > 0.0 => a,
        _ => entry * 0.005,
    }
}

fn profit_atr(side: PositionSide, entry: f64, close: f64, atr: f64) -> f64 {
    if atr <= 0.0 {
        return 0.0;
    }
    let diff = match side {
        PositionSide::Long => close - entry,
        PositionSide::Short => entry - close,
    };
    diff / atr
}

fn trace(id: &str, enabled: bool, status: &str, detail: impl Into<String>) -> BehaviourTrace {
    BehaviourTrace {
        group: "exits".to_string(),
        id: id.to_string(),
        enabled,
        status: status.to_string(),
        detail: detail.into(),
    }
}

fn default_exit_behaviour_plan() -> BehaviourGroupPlan {
    ResolvedBehaviourPlan::production().exits
}

fn compute_full_tp_price(side: PositionSide, entry: f64, atr: f64, params: &ExitParams) -> f64 {
    match side {
        PositionSide::Long => entry + (atr * params.tp_atr_mult),
        PositionSide::Short => entry - (atr * params.tp_atr_mult),
    }
}

fn recompute_sl_price(
    side: PositionSide,
    entry: f64,
    atr: f64,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    state: &StopLossState,
) -> f64 {
    let atr = atr.max(1e-12);
    let mut sl_price = match side {
        PositionSide::Long => entry - (atr * state.sl_mult),
        PositionSide::Short => entry + (atr * state.sl_mult),
    };

    if state.breakeven_active && params.enable_breakeven_stop && params.breakeven_start_atr > 0.0 {
        let be_start = atr * params.breakeven_start_atr;
        let be_buffer = atr * params.breakeven_buffer_atr;
        match side {
            PositionSide::Long => {
                if (snap.close - entry) >= be_start {
                    sl_price = sl_price.max(entry + be_buffer);
                }
            }
            PositionSide::Short => {
                if (entry - snap.close) >= be_start {
                    sl_price = sl_price.min(entry - be_buffer);
                }
            }
        }
    }

    sl_price
}

fn build_exit_bounds(
    pos: &Position,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    entry: f64,
    atr: f64,
    sl_price: Option<f64>,
) -> Option<ExitBounds> {
    let trailing_sl = if behaviour_plan.is_enabled("exit.trailing.base") {
        pos.trailing_sl
    } else {
        None
    };
    let lower_full = match (sl_price, trailing_sl) {
        (Some(sl), Some(tsl)) => match pos.side {
            PositionSide::Long => sl.max(tsl),
            PositionSide::Short => sl.min(tsl),
        },
        (Some(sl), None) => sl,
        (None, Some(tsl)) => tsl,
        (None, None) => 0.0,
    };
    let upper_full = if behaviour_plan.is_enabled("exit.take_profit.full") {
        compute_full_tp_price(pos.side, entry, atr, params)
    } else {
        0.0
    };
    let upper_partial = if behaviour_plan.is_enabled("exit.take_profit.partial") {
        compute_partial_tp_price(pos, entry, atr, params)
    } else {
        None
    };

    Some(ExitBounds {
        upper_full,
        upper_partial,
        lower_full,
    })
}

fn maybe_debug_stop_snapshot(
    pos: &Position,
    snap: &IndicatorSnapshot,
    current_time_ms: i64,
    entry: f64,
    atr: f64,
    sl_price: f64,
    sl_hit: bool,
) {
    let Ok(symbol_filter) = std::env::var("AQC_DEBUG_EXIT_SYMBOL") else {
        return;
    };
    if pos.symbol != symbol_filter {
        return;
    }

    let Ok(ts_filter_raw) = std::env::var("AQC_DEBUG_EXIT_TS_MS") else {
        return;
    };
    let Ok(ts_filter) = ts_filter_raw.parse::<i64>() else {
        return;
    };
    if current_time_ms != ts_filter {
        return;
    }

    eprintln!(
        "[exit-debug] symbol={} ts_ms={} side={:?} close={:.10} entry={:.10} atr={:.10} adx={:.10} adx_slope={:.10} trailing_sl={:?} sl_price={:.10} sl_hit={}",
        pos.symbol,
        current_time_ms,
        pos.side,
        snap.close,
        entry,
        atr,
        snap.adx,
        snap.adx_slope,
        pos.trailing_sl,
        sl_price,
        sl_hit
    );
}

/// Compute trailing stop price (or None if not yet active).
/// Updates are ratcheted — the trailing SL can only improve.
/// Mirrors `exits/trailing.rs::compute_trailing`.
struct ComputeTrailingInput<'a> {
    side: PositionSide,
    atr: f64,
    current_trailing: Option<f64>,
    snap: &'a IndicatorSnapshot,
    trailing_state: TrailingState,
    profit_atr_val: f64,
}

fn compute_trailing(input: ComputeTrailingInput<'_>) -> Option<f64> {
    let ComputeTrailingInput {
        side,
        atr,
        current_trailing,
        snap,
        trailing_state,
        profit_atr_val,
    } = input;
    let trailing_start = trailing_state.start_atr;
    let trailing_dist = trailing_state.distance_atr;

    // RSI Trend-Guard floor
    let min_trailing_dist = match side {
        PositionSide::Long if snap.rsi > 60.0 => 0.7,
        PositionSide::Short if snap.rsi < 40.0 => 0.7,
        _ => 0.5,
    };

    let mut effective_dist = trailing_dist;

    // Vol-Buffered Trailing Stop (VBTS)
    if trailing_state.vol_buffer_active && snap.bb_width_ratio > 1.2 {
        effective_dist *= 1.25;
    }

    // High-profit tightening with TATP / TSPV overrides
    if profit_atr_val > 2.0 {
        let tighten_mult = if snap.adx > 35.0 && snap.adx_slope > 0.0 {
            1.0 // TATP: trend accelerating → don't tighten
        } else if snap.atr_slope > 0.0 {
            0.75 // TSPV: vol expanding → partial tighten
        } else {
            0.5
        };
        effective_dist = trailing_dist * tighten_mult;
    } else if snap.adx < 25.0 {
        effective_dist = trailing_dist * 0.7;
    }

    // Clamp to RSI Trend-Guard floor
    effective_dist = effective_dist.max(min_trailing_dist);

    // Activation gate
    if profit_atr_val < trailing_start {
        return current_trailing;
    }

    // Candidate trailing stop
    let candidate = match side {
        PositionSide::Long => snap.close - (atr * effective_dist),
        PositionSide::Short => snap.close + (atr * effective_dist),
    };

    // Ratchet: only improve
    let ratcheted = match side {
        PositionSide::Long => match current_trailing {
            Some(prev) => candidate.max(prev),
            None => candidate,
        },
        PositionSide::Short => match current_trailing {
            Some(prev) => candidate.min(prev),
            None => candidate,
        },
    };

    Some(ratcheted)
}

fn check_take_profit_partial(
    side: PositionSide,
    entry: f64,
    atr: f64,
    quantity: f64,
    tp1_taken: bool,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
) -> Option<KernelExitResult> {
    if !params.enable_partial_tp || tp1_taken {
        return None;
    }

    let tp_mult = params.tp_atr_mult;
    let partial_mult = if params.tp_partial_atr_mult > 0.0 {
        params.tp_partial_atr_mult
    } else {
        tp_mult
    };
    let partial_tp_price = match side {
        PositionSide::Long => entry + (atr * partial_mult),
        PositionSide::Short => entry - (atr * partial_mult),
    };
    let partial_hit = match side {
        PositionSide::Long => snap.close >= partial_tp_price,
        PositionSide::Short => snap.close <= partial_tp_price,
    };

    if !partial_hit {
        return None;
    }

    let pct = params.tp_partial_pct.clamp(0.0, 1.0);
    if pct > 0.0 && pct < 1.0 {
        let remaining_notional = quantity * (1.0 - pct) * snap.close;
        if remaining_notional < params.tp_partial_min_notional_usd {
            return None;
        }
        return Some(KernelExitResult::PartialClose {
            reason: "Take Profit (Partial)".to_string(),
            exit_price: snap.close,
            fraction: pct,
        });
    }

    if pct >= 1.0 {
        return Some(KernelExitResult::FullClose {
            reason: "Take Profit".to_string(),
            exit_price: snap.close,
        });
    }

    None
}

fn check_take_profit_full(
    side: PositionSide,
    entry: f64,
    atr: f64,
    tp1_taken: bool,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
) -> Option<KernelExitResult> {
    if params.enable_partial_tp && tp1_taken && params.tp_partial_atr_mult <= 0.0 {
        return None;
    }
    let tp_price = compute_full_tp_price(side, entry, atr, params);
    let tp_hit = match side {
        PositionSide::Long => snap.close >= tp_price,
        PositionSide::Short => snap.close <= tp_price,
    };
    if tp_hit {
        return Some(KernelExitResult::FullClose {
            reason: "Take Profit".to_string(),
            exit_price: snap.close,
        });
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Smart exit evaluation (mirrors exits/smart_exits.rs)
// ═══════════════════════════════════════════════════════════════════════════

fn check_smart_exit(
    behaviour_id: &str,
    pos: &Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    profit_atr_val: f64,
    duration_hours: f64,
) -> Option<KernelExitResult> {
    let entry = pos.avg_entry_price;
    let atr = effective_atr(entry, pos.entry_atr);

    let is_long = pos.side == PositionSide::Long;
    let is_low_conf = pos.confidence == Some(0);

    // ADX exhaustion threshold: prefer entry's threshold, then low-conf override, then config.
    let entry_adx_thr = pos.entry_adx_threshold.unwrap_or(0.0);
    let adx_exhaustion_lt = if entry_adx_thr > 0.0 {
        entry_adx_thr
    } else if is_low_conf && params.smart_exit_adx_exhaustion_lt_low_conf > 0.0 {
        params.smart_exit_adx_exhaustion_lt_low_conf
    } else {
        params.smart_exit_adx_exhaustion_lt
    }
    .max(0.0);

    let ema_dev = if snap.ema_slow > 0.0 {
        (snap.ema_fast - snap.ema_slow).abs() / snap.ema_slow
    } else {
        0.0
    };
    let is_weak_cross = ema_dev < 0.001 && snap.adx > 25.0;

    let ema_cross_exit = if is_long {
        snap.ema_fast < snap.ema_slow && !is_weak_cross
    } else {
        snap.ema_fast > snap.ema_slow && !is_weak_cross
    };

    let exhausted = adx_exhaustion_lt > 0.0 && snap.adx < adx_exhaustion_lt;

    match behaviour_id {
        "exit.smart.trend_breakdown" => {
            if ema_cross_exit {
                Some(KernelExitResult::FullClose {
                    reason: "Trend Breakdown (EMA Cross)".to_string(),
                    exit_price: snap.close,
                })
            } else {
                None
            }
        }
        "exit.smart.trend_exhaustion" => {
            if exhausted {
                Some(KernelExitResult::FullClose {
                    reason: format!("Trend Exhaustion (ADX < {adx_exhaustion_lt})"),
                    exit_price: snap.close,
                })
            } else {
                None
            }
        }
        "exit.smart.ema_macro_breakdown" => {
            if !params.require_macro_alignment || snap.ema_macro <= 0.0 {
                return None;
            }
            if is_long && snap.close < snap.ema_macro {
                return Some(KernelExitResult::FullClose {
                    reason: "EMA Macro Breakdown".to_string(),
                    exit_price: snap.close,
                });
            }
            if !is_long && snap.close > snap.ema_macro {
                return Some(KernelExitResult::FullClose {
                    reason: "EMA Macro Breakout".to_string(),
                    exit_price: snap.close,
                });
            }
            None
        }
        "exit.smart.stagnation" => {
            if snap.atr < (atr * 0.70) {
                let is_underwater = if is_long {
                    snap.close < entry
                } else {
                    snap.close > entry
                };
                if is_underwater && pos.symbol.to_uppercase() != "PAXG" {
                    return Some(KernelExitResult::FullClose {
                        reason: format!(
                            "Stagnation Exit (Low Vol: {:.2} < {:.2})",
                            snap.atr,
                            atr * 0.70
                        ),
                        exit_price: snap.close,
                    });
                }
            }
            None
        }
        "exit.smart.funding_headwind" => {
            check_funding_headwind_kernel(pos, snap, profit_atr_val, duration_hours)
        }
        "exit.smart.tsme" => {
            if snap.adx > 50.0 {
                let tsme_min_profit = params.tsme_min_profit_atr;
                let gate_profit_ok = profit_atr_val >= tsme_min_profit;
                let gate_slope_ok = if params.tsme_require_adx_slope_negative {
                    snap.adx_slope < 0.0
                } else {
                    true
                };

                if gate_profit_ok && gate_slope_ok {
                    let is_exhausted = if is_long {
                        snap.macd_hist < snap.prev_macd_hist
                            && snap.prev_macd_hist < snap.prev2_macd_hist
                    } else {
                        snap.macd_hist > snap.prev_macd_hist
                            && snap.prev_macd_hist > snap.prev2_macd_hist
                    };

                    if is_exhausted {
                        return Some(KernelExitResult::FullClose {
                            reason: format!(
                                "Trend Saturation Momentum Exhaustion (ADX: {:.1}, ADX_slope: {:.2})",
                                snap.adx, snap.adx_slope
                            ),
                            exit_price: snap.close,
                        });
                    }
                }
            }
            None
        }
        "exit.smart.mmde" => {
            if profit_atr_val > 1.5 && snap.adx > 35.0 {
                let is_diverging = if is_long {
                    snap.macd_hist < snap.prev_macd_hist
                        && snap.prev_macd_hist < snap.prev2_macd_hist
                        && snap.prev2_macd_hist < snap.prev3_macd_hist
                } else {
                    snap.macd_hist > snap.prev_macd_hist
                        && snap.prev_macd_hist > snap.prev2_macd_hist
                        && snap.prev2_macd_hist > snap.prev3_macd_hist
                };

                if is_diverging {
                    return Some(KernelExitResult::FullClose {
                        reason: format!(
                            "MACD Persistent Divergence (Profit: {profit_atr_val:.2} ATR)"
                        ),
                        exit_price: snap.close,
                    });
                }
            }
            None
        }
        "exit.smart.rsi_overextension" => {
            if !params.enable_rsi_overextension_exit {
                return None;
            }
            check_rsi_overextension_kernel(pos, snap, params, profit_atr_val)
        }
        _ => None,
    }
}

/// Funding headwind sub-check for kernel positions.
/// Faithfully replicates the full AFL/TDH/TLFB/MFE/ABF/HCFB/MTF/TCFB/VSFT/CTEB/ETFS/TAES/DFG/PVS/PBFB/TWFS chain.
fn check_funding_headwind_kernel(
    pos: &Position,
    snap: &IndicatorSnapshot,
    profit_atr_val: f64,
    duration_hours: f64,
) -> Option<KernelExitResult> {
    let funding_rate = snap.funding_rate;
    if funding_rate == 0.0 {
        return None;
    }

    let is_long = pos.side == PositionSide::Long;
    let entry = pos.avg_entry_price;
    // C5: ATR zero-guard — prevent NaN/Inf from division by zero
    let atr = effective_atr(entry, pos.entry_atr).max(1e-12);

    let is_headwind = if is_long {
        funding_rate > 0.0
    } else {
        funding_rate < 0.0
    };
    if !is_headwind {
        return None;
    }

    let price_diff_atr = (snap.close - entry).abs() / atr;

    // ── AFL (Adaptive Funding Ladder) ────────────────────────────────────
    let mut headwind_threshold = if funding_rate.abs() > 0.0001 {
        0.15
    } else if funding_rate.abs() > 0.00006 {
        0.25
    } else if funding_rate.abs() > 0.00004 {
        0.40
    } else if funding_rate.abs() > 0.00002 {
        0.60
    } else if funding_rate.abs() < 0.00001 {
        0.95 // NZF: near-zero funding buffer
    } else {
        0.80
    };

    // ── Volatility-Adjusted Sensitivity ──────────────────────────────────
    if snap.atr > (atr * 1.2) {
        headwind_threshold *= 0.6;
    }

    // ── TDH (Time-Decay Headwind) with floor ─────────────────────────────
    if duration_hours > 1.0 {
        let decay_factor = (1.0 - (duration_hours - 1.0) / 11.0).max(0.0);
        headwind_threshold = (headwind_threshold * decay_factor).max(0.35);
    }

    // ── TLFB (Trend Loyalty Funding Buffer) ──────────────────────────────
    let is_trend_valid = if is_long {
        snap.ema_fast > snap.ema_slow
    } else {
        snap.ema_fast < snap.ema_slow
    };
    if is_trend_valid && snap.adx > 25.0 {
        headwind_threshold = headwind_threshold.max(0.75);
    }

    // ── MFE (Momentum-Filtered Funding Exit) ─────────────────────────────
    let is_momentum_improving = if is_long {
        snap.macd_hist > snap.prev_macd_hist
    } else {
        snap.macd_hist < snap.prev_macd_hist
    };
    if is_momentum_improving {
        headwind_threshold *= 1.5;
        headwind_threshold = headwind_threshold.max(0.50);
    }

    // ── ABF (ADX-Boosted Funding Threshold) ──────────────────────────────
    if snap.adx > 35.0 {
        headwind_threshold *= 1.4;
    }

    // ── HCFB (High-Confidence Funding Buffer) ────────────────────────────
    if pos.confidence == Some(2) {
        headwind_threshold *= 1.25;
    }

    // ── MTF (Macro-Trend Filtered Funding Exit) ──────────────────────────
    let is_macro_aligned = if snap.ema_macro > 0.0 {
        if is_long {
            snap.close > snap.ema_macro
        } else {
            snap.close < snap.ema_macro
        }
    } else {
        false
    };
    if is_macro_aligned {
        headwind_threshold *= 1.3;
    }

    // ── TCFB (Triple Confirmation Funding Buffer) ────────────────────────
    if is_momentum_improving && is_macro_aligned && pos.confidence == Some(2) {
        headwind_threshold *= 1.5;
    }

    // ── VSFT (Volatility-Scaled Funding Tolerance) ───────────────────────
    if snap.close > 0.0 && (snap.atr / snap.close) < 0.002 {
        headwind_threshold *= 1.25;
    }

    // ── CTEB (Counter-Trend Exhaustion Buffer) ───────────────────────────
    if (!is_long && snap.rsi > 65.0) || (is_long && snap.rsi < 35.0) {
        headwind_threshold *= 1.3;
    }

    // ── ETFS (Extreme Trend Funding Shield) ──────────────────────────────
    if snap.adx > 45.0 {
        headwind_threshold *= 1.6;
    }

    // ── TAES (Trend Acceleration Exit Shield) ────────────────────────────
    if snap.adx_slope > 1.0 {
        headwind_threshold *= 1.4;
    }

    // ── DFG (Dynamic Funding Guard) ──────────────────────────────────────
    if snap.adx > 40.0 {
        headwind_threshold *= 1.25;
    }

    // ── PVS (Profit-Vol Shield) ──────────────────────────────────────────
    if profit_atr_val > 1.5 && snap.atr_slope > 0.0 {
        headwind_threshold *= 1.5;
    }

    // ── PBFB (Profit-Based Funding Buffer) ───────────────────────────────
    if profit_atr_val > 3.0 {
        headwind_threshold *= 2.0;
    } else if profit_atr_val > 2.0 {
        headwind_threshold *= 1.5;
    }

    // ── TWFS (Trend Weakening Funding Sensitivity) ───────────────────────
    if snap.adx_slope < 0.0 {
        headwind_threshold *= 0.75;
    }

    // ── Final check: underwater with headwind ────────────────────────────
    let is_underwater = if is_long {
        snap.close < entry
    } else {
        snap.close > entry
    };
    if is_underwater && price_diff_atr > headwind_threshold {
        return Some(KernelExitResult::FullClose {
            reason: format!(
                "Funding Headwind Exit (FR: {funding_rate:.6}, Thr: {headwind_threshold:.2}, Dur: {duration_hours:.1}h)"
            ),
            exit_price: snap.close,
        });
    }

    None
}

/// RSI overextension sub-check for kernel positions.
fn check_rsi_overextension_kernel(
    pos: &Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    profit_atr_val: f64,
) -> Option<KernelExitResult> {
    let is_long = pos.side == PositionSide::Long;
    let is_low_conf = pos.confidence == Some(0);

    let sw = params.rsi_exit_profit_atr_switch.max(0.0);

    let (rsi_ub, rsi_lb) = if profit_atr_val < sw {
        // Low-profit regime: wider thresholds (less aggressive exit).
        let mut ub = params.rsi_exit_ub_lo_profit;
        let mut lb = params.rsi_exit_lb_lo_profit;
        if is_low_conf {
            if params.rsi_exit_ub_lo_profit_low_conf > 0.0 {
                ub = params.rsi_exit_ub_lo_profit_low_conf;
            }
            if params.rsi_exit_lb_lo_profit_low_conf > 0.0 {
                lb = params.rsi_exit_lb_lo_profit_low_conf;
            }
        }
        (ub, lb)
    } else {
        // High-profit regime: tighter thresholds (protect profits).
        let mut ub = params.rsi_exit_ub_hi_profit;
        let mut lb = params.rsi_exit_lb_hi_profit;
        if is_low_conf {
            if params.rsi_exit_ub_hi_profit_low_conf > 0.0 {
                ub = params.rsi_exit_ub_hi_profit_low_conf;
            }
            if params.rsi_exit_lb_hi_profit_low_conf > 0.0 {
                lb = params.rsi_exit_lb_hi_profit_low_conf;
            }
        }
        (ub, lb)
    };

    if is_long && snap.rsi > rsi_ub {
        return Some(KernelExitResult::FullClose {
            reason: format!("RSI Overbought ({:.1}, Thr: {rsi_ub})", snap.rsi),
            exit_price: snap.close,
        });
    }
    if !is_long && snap.rsi < rsi_lb {
        return Some(KernelExitResult::FullClose {
            reason: format!("RSI Oversold ({:.1}, Thr: {rsi_lb})", snap.rsi),
            exit_price: snap.close,
        });
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

use crate::decision_kernel::{ExitContext, ThresholdRecord};

fn record_threshold(
    thresholds: &mut Vec<ThresholdRecord>,
    name: &str,
    actual: f64,
    threshold: f64,
    passed: bool,
) {
    if let Some(existing) = thresholds.iter_mut().find(|record| record.name == name) {
        existing.actual = actual;
        existing.threshold = threshold;
        existing.passed = passed;
    } else {
        thresholds.push(ThresholdRecord {
            name: name.to_string(),
            actual,
            threshold,
            passed,
        });
    }
}

fn stop_loss_triggered(
    pos: &Position,
    snap: &IndicatorSnapshot,
    current_time_ms: i64,
    _behaviour_id: &str,
    sl_price: f64,
    thresholds: &mut Vec<ThresholdRecord>,
) -> bool {
    let sl_hit = match pos.side {
        PositionSide::Long => snap.close <= sl_price,
        PositionSide::Short => snap.close >= sl_price,
    };
    maybe_debug_stop_snapshot(
        pos,
        snap,
        current_time_ms,
        pos.avg_entry_price,
        effective_atr(pos.avg_entry_price, pos.entry_atr),
        sl_price,
        sl_hit,
    );
    let sl_distance = match pos.side {
        PositionSide::Long => snap.close - sl_price,
        PositionSide::Short => sl_price - snap.close,
    };
    record_threshold(thresholds, "sl_distance", sl_distance, 0.0, !sl_hit);
    sl_hit
}

#[allow(clippy::too_many_arguments)]
fn build_exit_evaluation(
    pos: &Position,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    entry: f64,
    atr: f64,
    pa: f64,
    duration_bars: u64,
    result: KernelExitResult,
    exit_type: &str,
    exit_reason: String,
    thresholds: Vec<ThresholdRecord>,
    sl_price: Option<f64>,
    behaviour_trace: Vec<BehaviourTrace>,
) -> ExitEvaluation {
    let tp_price = if behaviour_plan.is_enabled("exit.take_profit.full") {
        Some(compute_full_tp_price(pos.side, entry, atr, params))
    } else {
        None
    };
    let exit_bounds = build_exit_bounds(pos, params, behaviour_plan, entry, atr, sl_price);
    ExitEvaluation {
        result: result.clone(),
        threshold_records: thresholds,
        exit_context: Some(ExitContext {
            exit_type: exit_type.to_string(),
            exit_reason,
            exit_price: match result {
                KernelExitResult::Hold => snap_unreachable_exit_price(),
                KernelExitResult::FullClose { exit_price, .. }
                | KernelExitResult::PartialClose { exit_price, .. } => exit_price,
            },
            entry_price: entry,
            sl_price,
            tp_price,
            trailing_sl: pos.trailing_sl,
            profit_atr: pa,
            duration_bars,
        }),
        exit_bounds,
        behaviour_trace,
    }
}

fn snap_unreachable_exit_price() -> f64 {
    0.0
}

#[allow(clippy::too_many_arguments)]
fn evaluate_stop_loss_behaviour(
    pos: &Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    current_time_ms: i64,
    entry: f64,
    atr: f64,
    pa: f64,
    duration_bars: u64,
    behaviour_id: &str,
    state: &mut StopLossState,
    thresholds: &mut Vec<ThresholdRecord>,
    behaviour_trace: &mut Vec<BehaviourTrace>,
) -> Option<ExitEvaluation> {
    let enabled = behaviour_plan.is_enabled(behaviour_id);
    if !enabled {
        behaviour_trace.push(trace(
            behaviour_id,
            false,
            "disabled",
            "disabled by behaviour plan",
        ));
        return None;
    }

    match behaviour_id {
        "exit.stop_loss.base" => {
            state.sl_price = Some(recompute_sl_price(
                pos.side, entry, atr, snap, params, state,
            ));
            let sl_price = state.sl_price.expect("base stop-loss price must exist");
            behaviour_trace.push(trace(
                behaviour_id,
                true,
                "executed",
                format!("sl_mult={:.4} sl_price={sl_price:.4}", state.sl_mult),
            ));
            if stop_loss_triggered(
                pos,
                snap,
                current_time_ms,
                behaviour_id,
                sl_price,
                thresholds,
            ) {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "triggered",
                    format!("close {:.4} breached sl {:.4}", snap.close, sl_price),
                ));
                let result = KernelExitResult::FullClose {
                    reason: "Stop Loss".to_string(),
                    exit_price: snap.close,
                };
                return Some(build_exit_evaluation(
                    pos,
                    params,
                    behaviour_plan,
                    entry,
                    atr,
                    pa,
                    duration_bars,
                    result,
                    "stop_loss",
                    format!("SL at {sl_price:.4}, close {:.4}", snap.close),
                    thresholds.clone(),
                    state.sl_price,
                    behaviour_trace.clone(),
                ));
            }
        }
        "exit.stop_loss.ase" => {
            let underwater = match pos.side {
                PositionSide::Long => snap.close < entry,
                PositionSide::Short => snap.close > entry,
            };
            if snap.adx_slope < 0.0 && underwater {
                state.sl_mult *= 0.8;
                if state.sl_price.is_some() {
                    state.sl_price = Some(recompute_sl_price(
                        pos.side, entry, atr, snap, params, state,
                    ));
                }
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "executed",
                    format!("tightened sl_mult to {:.4}", state.sl_mult),
                ));
            } else {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "skipped",
                    "ASE preconditions did not pass",
                ));
            }
        }
        "exit.stop_loss.dase" => {
            if snap.adx > 40.0 && pa > 0.5 {
                state.sl_mult *= 1.15;
                if state.sl_price.is_some() {
                    state.sl_price = Some(recompute_sl_price(
                        pos.side, entry, atr, snap, params, state,
                    ));
                }
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "executed",
                    format!("widened sl_mult to {:.4}", state.sl_mult),
                ));
            } else {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "skipped",
                    "DASE preconditions did not pass",
                ));
            }
        }
        "exit.stop_loss.slb" => {
            if snap.adx > 45.0 {
                state.sl_mult *= 1.10;
                if state.sl_price.is_some() {
                    state.sl_price = Some(recompute_sl_price(
                        pos.side, entry, atr, snap, params, state,
                    ));
                }
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "executed",
                    format!("buffered sl_mult to {:.4}", state.sl_mult),
                ));
            } else {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "skipped",
                    "SLB preconditions did not pass",
                ));
            }
        }
        "exit.stop_loss.breakeven" => {
            if !params.enable_breakeven_stop || params.breakeven_start_atr <= 0.0 {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "config_disabled",
                    "breakeven stop disabled in params",
                ));
                return None;
            }
            let be_start = atr * params.breakeven_start_atr;
            let active = match pos.side {
                PositionSide::Long => (snap.close - entry) >= be_start,
                PositionSide::Short => (entry - snap.close) >= be_start,
            };
            state.breakeven_active = active;
            if state.sl_price.is_some() {
                state.sl_price = Some(recompute_sl_price(
                    pos.side, entry, atr, snap, params, state,
                ));
            }
            behaviour_trace.push(trace(
                behaviour_id,
                true,
                if active { "executed" } else { "skipped" },
                format!("breakeven_active={active}"),
            ));
        }
        _ => {}
    }

    if let Some(sl_price) = state.sl_price {
        if behaviour_id != "exit.stop_loss.base"
            && stop_loss_triggered(
                pos,
                snap,
                current_time_ms,
                behaviour_id,
                sl_price,
                thresholds,
            )
        {
            behaviour_trace.push(trace(
                behaviour_id,
                true,
                "triggered",
                format!("close {:.4} breached sl {:.4}", snap.close, sl_price),
            ));
            let result = KernelExitResult::FullClose {
                reason: "Stop Loss".to_string(),
                exit_price: snap.close,
            };
            return Some(build_exit_evaluation(
                pos,
                params,
                behaviour_plan,
                entry,
                atr,
                pa,
                duration_bars,
                result,
                "stop_loss",
                format!("SL at {sl_price:.4}, close {:.4}", snap.close),
                thresholds.clone(),
                state.sl_price,
                behaviour_trace.clone(),
            ));
        }
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn evaluate_trailing_behaviour(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    entry: f64,
    atr: f64,
    pa: f64,
    duration_bars: u64,
    behaviour_id: &str,
    state: &mut TrailingState,
    sl_price: Option<f64>,
    thresholds: &mut Vec<ThresholdRecord>,
    behaviour_trace: &mut Vec<BehaviourTrace>,
) -> Option<ExitEvaluation> {
    let enabled = behaviour_plan.is_enabled(behaviour_id);
    if !enabled {
        behaviour_trace.push(trace(
            behaviour_id,
            false,
            "disabled",
            "disabled by behaviour plan",
        ));
        return None;
    }

    match behaviour_id {
        "exit.trailing.low_conf_override" => {
            if pos.confidence == Some(0) {
                if params.trailing_start_atr_low_conf > 0.0 {
                    state.start_atr = params.trailing_start_atr_low_conf;
                }
                if params.trailing_distance_atr_low_conf > 0.0 {
                    state.distance_atr = params.trailing_distance_atr_low_conf;
                }
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "executed",
                    format!(
                        "start_atr={:.4} distance_atr={:.4}",
                        state.start_atr, state.distance_atr
                    ),
                ));
            } else {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "skipped",
                    "position is not low confidence",
                ));
            }
        }
        "exit.trailing.vol_buffer" => {
            if !params.enable_vol_buffered_trailing {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "config_disabled",
                    "vol-buffered trailing disabled in params",
                ));
            } else {
                state.vol_buffer_active = true;
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "executed",
                    format!("bb_width_ratio={:.4}", snap.bb_width_ratio),
                ));
            }
        }
        "exit.trailing.base" => {
            state.base_seen = true;
            behaviour_trace.push(trace(
                behaviour_id,
                true,
                "executed",
                format!(
                    "start_atr={:.4} distance_atr={:.4} current_trailing={:?}",
                    state.start_atr, state.distance_atr, pos.trailing_sl
                ),
            ));
        }
        _ => {}
    }

    if !state.base_seen {
        return None;
    }

    let new_tsl = compute_trailing(ComputeTrailingInput {
        side: pos.side,
        atr,
        current_trailing: pos.trailing_sl,
        snap,
        trailing_state: *state,
        profit_atr_val: pa,
    });
    pos.trailing_sl = new_tsl;

    if let Some(tsl_price) = new_tsl {
        let triggered = match pos.side {
            PositionSide::Long => snap.close <= tsl_price,
            PositionSide::Short => snap.close >= tsl_price,
        };
        let tsl_distance = match pos.side {
            PositionSide::Long => snap.close - tsl_price,
            PositionSide::Short => tsl_price - snap.close,
        };
        record_threshold(
            thresholds,
            "trailing_sl_distance",
            tsl_distance,
            0.0,
            !triggered,
        );
        if triggered {
            behaviour_trace.push(trace(
                behaviour_id,
                true,
                "triggered",
                format!("close {:.4} breached trailing {:.4}", snap.close, tsl_price),
            ));
            let result = KernelExitResult::FullClose {
                reason: "Trailing Stop".to_string(),
                exit_price: snap.close,
            };
            return Some(build_exit_evaluation(
                pos,
                params,
                behaviour_plan,
                entry,
                atr,
                pa,
                duration_bars,
                result,
                "trailing",
                format!("Trailing SL at {tsl_price:.4}, close {:.4}", snap.close),
                thresholds.clone(),
                sl_price,
                behaviour_trace.clone(),
            ));
        }
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn evaluate_take_profit_behaviour(
    pos: &Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    entry: f64,
    atr: f64,
    pa: f64,
    duration_bars: u64,
    behaviour_id: &str,
    sl_price: Option<f64>,
    thresholds: &mut Vec<ThresholdRecord>,
    behaviour_trace: &mut Vec<BehaviourTrace>,
) -> Option<ExitEvaluation> {
    let enabled = behaviour_plan.is_enabled(behaviour_id);
    if !enabled {
        behaviour_trace.push(trace(
            behaviour_id,
            false,
            "disabled",
            "disabled by behaviour plan",
        ));
        return None;
    }

    let result = match behaviour_id {
        "exit.take_profit.partial" => {
            if !params.enable_partial_tp {
                behaviour_trace.push(trace(
                    behaviour_id,
                    true,
                    "config_disabled",
                    "partial take-profit disabled in params",
                ));
                return None;
            }
            check_take_profit_partial(
                pos.side,
                entry,
                atr,
                pos.quantity,
                pos.tp1_taken,
                snap,
                params,
            )
        }
        "exit.take_profit.full" => {
            check_take_profit_full(pos.side, entry, atr, pos.tp1_taken, snap, params)
        }
        _ => None,
    };

    let threshold_price = if behaviour_id == "exit.take_profit.partial" {
        compute_partial_tp_price(pos, entry, atr, params)
            .unwrap_or_else(|| compute_full_tp_price(pos.side, entry, atr, params))
    } else {
        compute_full_tp_price(pos.side, entry, atr, params)
    };
    let distance = match pos.side {
        PositionSide::Long => threshold_price - snap.close,
        PositionSide::Short => snap.close - threshold_price,
    };
    record_threshold(thresholds, "tp_distance", distance, 0.0, distance > 0.0);

    if let Some(result) = result {
        let (exit_type, exit_reason) = match &result {
            KernelExitResult::PartialClose { reason, .. } => {
                ("take_profit_partial", reason.clone())
            }
            KernelExitResult::FullClose { reason, .. } => ("take_profit", reason.clone()),
            KernelExitResult::Hold => unreachable!(),
        };
        behaviour_trace.push(trace(behaviour_id, true, "triggered", exit_reason.clone()));
        return Some(build_exit_evaluation(
            pos,
            params,
            behaviour_plan,
            entry,
            atr,
            pa,
            duration_bars,
            result,
            exit_type,
            exit_reason,
            thresholds.clone(),
            sl_price,
            behaviour_trace.clone(),
        ));
    }

    behaviour_trace.push(trace(
        behaviour_id,
        true,
        "executed",
        "take-profit threshold not reached",
    ));
    None
}

#[allow(clippy::too_many_arguments)]
fn evaluate_smart_exit_behaviour(
    pos: &Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    behaviour_plan: &BehaviourGroupPlan,
    entry: f64,
    atr: f64,
    pa: f64,
    duration_bars: u64,
    duration_hours: f64,
    behaviour_id: &str,
    sl_price: Option<f64>,
    behaviour_trace: &mut Vec<BehaviourTrace>,
) -> Option<ExitEvaluation> {
    let enabled = behaviour_plan.is_enabled(behaviour_id);
    if !enabled {
        behaviour_trace.push(trace(
            behaviour_id,
            false,
            "disabled",
            "disabled by behaviour plan",
        ));
        return None;
    }

    if behaviour_id == "exit.smart.ema_macro_breakdown"
        && (!params.require_macro_alignment || snap.ema_macro <= 0.0)
    {
        behaviour_trace.push(trace(
            behaviour_id,
            true,
            "config_disabled",
            "macro alignment exit inactive in params/snapshot",
        ));
        return None;
    }
    if behaviour_id == "exit.smart.rsi_overextension" && !params.enable_rsi_overextension_exit {
        behaviour_trace.push(trace(
            behaviour_id,
            true,
            "config_disabled",
            "RSI overextension exit disabled in params",
        ));
        return None;
    }

    if let Some(result) = check_smart_exit(behaviour_id, pos, snap, params, pa, duration_hours) {
        let reason = match &result {
            KernelExitResult::FullClose { reason, .. } => reason.clone(),
            KernelExitResult::PartialClose { reason, .. } => reason.clone(),
            KernelExitResult::Hold => String::new(),
        };
        behaviour_trace.push(trace(behaviour_id, true, "triggered", reason.clone()));
        return Some(build_exit_evaluation(
            pos,
            params,
            behaviour_plan,
            entry,
            atr,
            pa,
            duration_bars,
            result,
            "smart_exit",
            reason,
            Vec::new(),
            sl_price,
            behaviour_trace.clone(),
        ));
    }

    behaviour_trace.push(trace(
        behaviour_id,
        true,
        "executed",
        "smart exit did not trigger",
    ));
    None
}

/// Evaluate exit conditions for an existing kernel position.
///
/// Priority is resolved by the active behaviour plan order after the glitch guard.
/// Updates `pos.trailing_sl` in-place (ratcheted).
pub fn evaluate_exits(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    current_time_ms: i64,
) -> KernelExitResult {
    evaluate_exits_with_diagnostics(pos, snap, params, current_time_ms).result
}

pub fn evaluate_exits_with_behaviour_plan(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    current_time_ms: i64,
    behaviour_plan: &BehaviourGroupPlan,
) -> KernelExitResult {
    evaluate_exits_with_behaviour_plan_and_diagnostics(
        pos,
        snap,
        params,
        current_time_ms,
        behaviour_plan,
    )
    .result
}

/// Evaluate exit conditions and return enriched diagnostics (threshold records + exit context).
pub fn evaluate_exits_with_diagnostics(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    current_time_ms: i64,
) -> ExitEvaluation {
    let behaviour_plan = default_exit_behaviour_plan();
    evaluate_exits_with_behaviour_plan_and_diagnostics(
        pos,
        snap,
        params,
        current_time_ms,
        &behaviour_plan,
    )
}

pub fn evaluate_exits_with_behaviour_plan_and_diagnostics(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    current_time_ms: i64,
    behaviour_plan: &BehaviourGroupPlan,
) -> ExitEvaluation {
    let mut thresholds = Vec::new();
    let mut behaviour_trace = Vec::new();

    if params.block_exits_on_extreme_dev && snap.prev_close > 0.0 {
        let price_change_pct = (snap.close - snap.prev_close).abs() / snap.prev_close;
        let atr_dev = if snap.atr > 0.0 {
            (snap.close - snap.prev_close).abs() / snap.atr
        } else {
            0.0
        };
        let is_glitch = price_change_pct > params.glitch_price_dev_pct
            || (snap.atr > 0.0
                && (snap.close - snap.prev_close).abs() > snap.atr * params.glitch_atr_mult);
        thresholds.push(ThresholdRecord {
            name: "glitch_price_dev".into(),
            actual: price_change_pct,
            threshold: params.glitch_price_dev_pct,
            passed: price_change_pct <= params.glitch_price_dev_pct,
        });
        thresholds.push(ThresholdRecord {
            name: "glitch_atr_mult".into(),
            actual: atr_dev,
            threshold: params.glitch_atr_mult,
            passed: atr_dev <= params.glitch_atr_mult,
        });
        if is_glitch {
            let entry = pos.avg_entry_price;
            let atr = effective_atr(entry, pos.entry_atr);
            let mut stop_state = StopLossState {
                sl_mult: params.sl_atr_mult,
                sl_price: None,
                breakeven_active: false,
            };
            if behaviour_plan.is_enabled("exit.stop_loss.base") {
                stop_state.sl_price = Some(recompute_sl_price(
                    pos.side,
                    entry,
                    atr,
                    snap,
                    params,
                    &stop_state,
                ));
            }
            return ExitEvaluation {
                result: KernelExitResult::Hold,
                threshold_records: thresholds,
                exit_context: None,
                exit_bounds: build_exit_bounds(
                    pos,
                    params,
                    behaviour_plan,
                    entry,
                    atr,
                    stop_state.sl_price,
                ),
                behaviour_trace,
            };
        }
    }

    let entry = pos.avg_entry_price;
    let atr = effective_atr(entry, pos.entry_atr);
    let pa = profit_atr(pos.side, entry, snap.close, atr);
    let duration_bars = if current_time_ms > pos.opened_at_ms {
        ((current_time_ms - pos.opened_at_ms) / 1000).max(0) as u64
    } else {
        0
    };
    let duration_hours = if current_time_ms > pos.opened_at_ms {
        (current_time_ms - pos.opened_at_ms) as f64 / 3_600_000.0
    } else {
        0.0
    };

    let mut stop_state = StopLossState {
        sl_mult: params.sl_atr_mult,
        sl_price: None,
        breakeven_active: false,
    };
    let mut trailing_state = TrailingState {
        start_atr: params.trailing_start_atr,
        distance_atr: params.trailing_distance_atr,
        vol_buffer_active: false,
        base_seen: false,
    };

    for behaviour_id in behaviour_plan.ordered_ids() {
        if behaviour_id.starts_with("exit.stop_loss.") {
            if let Some(eval) = evaluate_stop_loss_behaviour(
                pos,
                snap,
                params,
                behaviour_plan,
                current_time_ms,
                entry,
                atr,
                pa,
                duration_bars,
                behaviour_id,
                &mut stop_state,
                &mut thresholds,
                &mut behaviour_trace,
            ) {
                return eval;
            }
            continue;
        }

        if behaviour_id.starts_with("exit.trailing.") {
            if let Some(eval) = evaluate_trailing_behaviour(
                pos,
                snap,
                params,
                behaviour_plan,
                entry,
                atr,
                pa,
                duration_bars,
                behaviour_id,
                &mut trailing_state,
                stop_state.sl_price,
                &mut thresholds,
                &mut behaviour_trace,
            ) {
                return eval;
            }
            continue;
        }

        if behaviour_id.starts_with("exit.take_profit.") {
            if let Some(eval) = evaluate_take_profit_behaviour(
                pos,
                snap,
                params,
                behaviour_plan,
                entry,
                atr,
                pa,
                duration_bars,
                behaviour_id,
                stop_state.sl_price,
                &mut thresholds,
                &mut behaviour_trace,
            ) {
                return eval;
            }
            continue;
        }

        if behaviour_id.starts_with("exit.smart.") {
            if let Some(eval) = evaluate_smart_exit_behaviour(
                pos,
                snap,
                params,
                behaviour_plan,
                entry,
                atr,
                pa,
                duration_bars,
                duration_hours,
                behaviour_id,
                stop_state.sl_price,
                &mut behaviour_trace,
            ) {
                let mut eval = eval;
                eval.threshold_records = thresholds;
                return eval;
            }
        }
    }

    ExitEvaluation {
        result: KernelExitResult::Hold,
        threshold_records: thresholds,
        exit_context: None,
        exit_bounds: build_exit_bounds(
            pos,
            params,
            behaviour_plan,
            entry,
            atr,
            stop_state.sl_price,
        ),
        behaviour_trace,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_snap(close: f64) -> IndicatorSnapshot {
        IndicatorSnapshot {
            close,
            high: close,
            low: close,
            open: close,
            volume: 0.0,
            t: 0,
            ema_slow: close,
            ema_fast: close,
            ema_macro: close,
            adx: 30.0,
            adx_pos: 15.0,
            adx_neg: 15.0,
            adx_slope: 0.0,
            bb_upper: close * 1.02,
            bb_lower: close * 0.98,
            bb_width: 0.04,
            bb_width_avg: 0.04,
            bb_width_ratio: 1.0,
            atr: close * 0.01,
            atr_slope: 0.0,
            avg_atr: close * 0.01,
            rsi: 50.0,
            stoch_rsi_k: 50.0,
            stoch_rsi_d: 50.0,
            macd_hist: 0.0,
            prev_macd_hist: 0.0,
            prev2_macd_hist: 0.0,
            prev3_macd_hist: 0.0,
            vol_sma: 100.0,
            vol_trend: false,
            prev_close: close,
            prev_ema_fast: close,
            prev_ema_slow: close,
            bar_count: 200,
            funding_rate: 0.0,
        }
    }

    fn long_pos(entry: f64) -> Position {
        Position {
            symbol: "BTC".to_string(),
            side: PositionSide::Long,
            quantity: 1.0,
            avg_entry_price: entry,
            opened_at_ms: 0,
            updated_at_ms: 0,
            notional_usd: entry,
            margin_usd: entry / 3.0,
            confidence: Some(2), // High
            entry_atr: Some(entry * 0.01),
            entry_adx_threshold: None,
            adds_count: 0,
            tp1_taken: false,
            trailing_sl: None,
            mae_usd: 0.0,
            mfe_usd: 0.0,
            last_funding_ms: None,
        }
    }

    fn short_pos(entry: f64) -> Position {
        Position {
            symbol: "BTC".to_string(),
            side: PositionSide::Short,
            quantity: 1.0,
            avg_entry_price: entry,
            opened_at_ms: 0,
            updated_at_ms: 0,
            notional_usd: entry,
            margin_usd: entry / 3.0,
            confidence: Some(2),
            entry_atr: Some(entry * 0.01),
            entry_adx_threshold: None,
            adds_count: 0,
            tp1_taken: false,
            trailing_sl: None,
            mae_usd: 0.0,
            mfe_usd: 0.0,
            last_funding_ms: None,
        }
    }

    fn default_params() -> ExitParams {
        ExitParams::default()
    }

    fn custom_plan(order: &[&str], disabled: &[&str]) -> BehaviourGroupPlan {
        let base = default_exit_behaviour_plan();
        let mut items = Vec::new();
        for requested in order {
            if let Some(item) = base.item(requested) {
                items.push(item.clone());
            }
        }
        for item in base.items {
            if !order.iter().any(|requested| *requested == item.id) {
                items.push(item);
            }
        }
        for item in &mut items {
            if disabled.iter().any(|requested| *requested == item.id) {
                item.enabled = false;
            }
        }
        BehaviourGroupPlan { items }
    }

    // ── Stop Loss tests ─────────────────────────────────────────────────

    #[test]
    fn sl_triggers_long() {
        let mut pos = long_pos(100.0);
        // SL = 100 - 1.0 * 2.0 = 98.0; close = 97.5 → trigger
        let snap = default_snap(97.5);
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected FullClose(Stop Loss), got {:?}", other),
        }
    }

    #[test]
    fn sl_triggers_short() {
        let mut pos = short_pos(100.0);
        // SL = 100 + 1.0 * 2.0 = 102.0; close = 102.5 → trigger
        let snap = default_snap(102.5);
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected FullClose(Stop Loss), got {:?}", other),
        }
    }

    #[test]
    fn sl_no_trigger_above_level() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(99.0); // above SL of 98.0
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    // ── SL dynamic adjustment tests ─────────────────────────────────────

    #[test]
    fn sl_ase_tightens_when_underwater() {
        // ASE: adx_slope < 0, underwater → sl_mult *= 0.8 → 1.6
        // SL = 100 - 1.0 * 1.6 = 98.4; close = 98.3 → trigger
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(98.3);
        snap.adx_slope = -1.0;
        let params = ExitParams {
            enable_breakeven_stop: false,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected SL with ASE, got {:?}", other),
        }
    }

    #[test]
    fn sl_dase_widens_when_strong_trend_profitable() {
        // DASE: ADX > 40, profit > 0.5 ATR → sl_mult *= 1.15 → 2.30
        // SL = 100 - 1.0 * 2.30 = 97.70
        // Breakeven: profit 1.0 ATR >= 0.7 → BE SL = 100.05
        // Final SL = max(97.70, 100.05) = 100.05; close = 101.0 → no trigger
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(101.0);
        snap.adx = 42.0;
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        // Should NOT trigger SL (price above 100.05)
        assert!(
            !matches!(result, KernelExitResult::FullClose { ref reason, .. } if reason == "Stop Loss"),
            "SL should not trigger at 101.0 with DASE + breakeven"
        );
    }

    #[test]
    fn sl_slb_widens_when_adx_saturated() {
        // SLB: ADX > 45 → sl_mult = 2.0 * 1.10 = 2.20
        // SL = 100 - 1.0 * 2.20 = 97.80; close = 98.0 → no trigger
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(98.0);
        snap.adx = 50.0;
        let params = ExitParams {
            enable_breakeven_stop: false,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    #[test]
    fn sl_breakeven_moves_sl_when_profitable() {
        // profit = 0.8 ATR >= breakeven_start_atr (0.7)
        // BE SL = entry + 0.05 * 1.0 = 100.05
        // Base SL = 100 - 2.0 = 98.0; final = max(98, 100.05) = 100.05
        // close = 100.0 → below 100.05 → trigger SL
        let mut pos = long_pos(100.0);
        let snap2 = default_snap(100.04); // profit = 0.04, below 0.7 ATR
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        // At 100.04, no breakeven, base SL = 98.0, no trigger.
        let result = evaluate_exits(&mut pos, &snap2, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);

        // At 100.8, breakeven fires: BE SL = 100.05, close = 100.8 > 100.05 → no SL trigger
        let snap3 = default_snap(100.8);
        let result2 = evaluate_exits(&mut pos, &snap3, &params, 0);
        assert!(
            !matches!(result2, KernelExitResult::FullClose { ref reason, .. } if reason == "Stop Loss"),
        );
    }

    // ── Trailing stop tests ─────────────────────────────────────────────

    #[test]
    fn trailing_not_active_below_threshold() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(100.5); // profit = 0.5 ATR, threshold = 1.0
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        evaluate_exits(&mut pos, &snap, &params, 0);
        assert!(pos.trailing_sl.is_none());
    }

    #[test]
    fn trailing_activates_at_threshold() {
        let mut pos = long_pos(100.0);
        // profit = 2.0 ATR (above trailing_start_atr=1.0)
        // effective_dist = 0.8 (no adjustments fire at default ADX=30, profit NOT > 2.0)
        // Wait, profit_atr = 2.0, and the condition is `> 2.0` not `>= 2.0`. So
        // profit_atr = 2.0 doesn't fire high-profit tightening.
        // trailing_sl = 102.0 - 1.0 * 0.8 = 101.2
        let snap = default_snap(102.0);
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        evaluate_exits(&mut pos, &snap, &params, 0);
        let tsl = pos.trailing_sl.expect("trailing should activate");
        assert!((tsl - 101.2).abs() < 0.01);
    }

    #[test]
    fn trailing_ratchets_up_for_long() {
        let mut pos = long_pos(100.0);
        pos.trailing_sl = Some(101.8);
        let snap = default_snap(102.0); // candidate = 101.2, prev = 101.8
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        evaluate_exits(&mut pos, &snap, &params, 0);
        let tsl = pos.trailing_sl.unwrap();
        assert!(
            (tsl - 101.8).abs() < 0.01,
            "ratchet should keep 101.8, got {}",
            tsl
        );
    }

    #[test]
    fn trailing_triggers_close() {
        let mut pos = long_pos(100.0);
        pos.trailing_sl = Some(101.0);
        let snap = default_snap(100.9); // below trailing_sl=101.0
        let params = ExitParams {
            enable_breakeven_stop: false,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Trailing Stop"),
            other => panic!("Expected Trailing Stop, got {:?}", other),
        }
    }

    // ── Take Profit tests ───────────────────────────────────────────────

    #[test]
    fn tp_partial_close_on_first_hit() {
        let mut pos = long_pos(100.0);
        // tp_atr_mult=4.0 → TP at 104.0; partial uses same level
        let snap = default_snap(104.0);
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        match result {
            KernelExitResult::PartialClose {
                reason, fraction, ..
            } => {
                assert_eq!(reason, "Take Profit (Partial)");
                assert!((fraction - 0.5).abs() < 0.01);
            }
            other => panic!("Expected PartialClose, got {:?}", other),
        }
    }

    #[test]
    fn tp_full_close_when_partial_disabled() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(104.0);
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Take Profit"),
            other => panic!("Expected FullClose(Take Profit), got {:?}", other),
        }
    }

    #[test]
    fn tp_hold_after_partial_taken() {
        let mut pos = long_pos(100.0);
        pos.tp1_taken = true;
        let snap = default_snap(104.0);
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        // tp_partial_atr_mult=0.0 → trailing manages after tp1 taken
        assert_eq!(result, KernelExitResult::Hold);
    }

    #[test]
    fn tp_full_after_partial_with_separate_level() {
        let mut pos = long_pos(100.0);
        pos.tp1_taken = true;
        // tp_partial_atr_mult=3.0, tp_atr_mult=4.0
        let snap = default_snap(104.0); // at full TP level
        let params = ExitParams {
            tp_partial_atr_mult: 3.0,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Take Profit"),
            other => panic!("Expected full close at full TP, got {:?}", other),
        }
    }

    #[test]
    fn behaviour_plan_can_disable_stop_loss_base() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(97.5);
        let plan = custom_plan(&[], &["exit.stop_loss.base"]);

        let result =
            evaluate_exits_with_behaviour_plan(&mut pos, &snap, &default_params(), 0, &plan);
        assert_eq!(result, KernelExitResult::Hold);

        let mut pos = long_pos(100.0);
        let eval = evaluate_exits_with_behaviour_plan_and_diagnostics(
            &mut pos,
            &snap,
            &default_params(),
            0,
            &plan,
        );
        assert!(eval
            .behaviour_trace
            .iter()
            .any(|item| item.id == "exit.stop_loss.base" && item.status == "disabled"));
    }

    #[test]
    fn behaviour_plan_can_reorder_full_tp_ahead_of_partial() {
        let snap = default_snap(104.0);

        let mut default_pos = long_pos(100.0);
        let default_result = evaluate_exits(&mut default_pos, &snap, &default_params(), 0);
        match default_result {
            KernelExitResult::PartialClose { ref reason, .. } => {
                assert_eq!(reason, "Take Profit (Partial)")
            }
            other => panic!("Expected default partial TP, got {:?}", other),
        }

        let mut reordered_pos = long_pos(100.0);
        let plan = custom_plan(&["exit.take_profit.full", "exit.take_profit.partial"], &[]);
        let reordered_result = evaluate_exits_with_behaviour_plan(
            &mut reordered_pos,
            &snap,
            &default_params(),
            0,
            &plan,
        );
        match reordered_result {
            KernelExitResult::FullClose { ref reason, .. } => assert_eq!(reason, "Take Profit"),
            other => panic!("Expected reordered full TP, got {:?}", other),
        }
    }

    #[test]
    fn behaviour_plan_can_reorder_smart_exit_ahead_of_take_profit() {
        let mut snap = default_snap(104.5);
        snap.ema_fast = 95.0;
        snap.ema_slow = 105.0;

        let mut default_pos = long_pos(100.0);
        let default_result = evaluate_exits(&mut default_pos, &snap, &default_params(), 0);
        match default_result {
            KernelExitResult::PartialClose { ref reason, .. } => {
                assert_eq!(reason, "Take Profit (Partial)")
            }
            other => panic!("Expected default TP priority, got {:?}", other),
        }

        let mut reordered_pos = long_pos(100.0);
        let plan = custom_plan(
            &[
                "exit.smart.trend_breakdown",
                "exit.take_profit.partial",
                "exit.take_profit.full",
            ],
            &[],
        );
        let reordered_result = evaluate_exits_with_behaviour_plan(
            &mut reordered_pos,
            &snap,
            &default_params(),
            0,
            &plan,
        );
        match reordered_result {
            KernelExitResult::FullClose { ref reason, .. } => {
                assert!(reason.contains("Trend Breakdown"), "got: {reason}");
            }
            other => panic!("Expected reordered smart exit priority, got {:?}", other),
        }
    }

    // ── No exit on normal conditions ────────────────────────────────────

    #[test]
    fn hold_on_normal_conditions() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(101.0); // small profit, no exit conditions
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    #[test]
    fn short_hold_on_normal_conditions() {
        let mut pos = short_pos(100.0);
        let snap = default_snap(99.0);
        let params = ExitParams {
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    #[test]
    fn short_tp_partial() {
        let mut pos = short_pos(100.0);
        // TP at 100 - 1.0*4.0 = 96.0
        let snap = default_snap(96.0);
        let result = evaluate_exits(&mut pos, &snap, &default_params(), 0);
        match result {
            KernelExitResult::PartialClose {
                reason, fraction, ..
            } => {
                assert_eq!(reason, "Take Profit (Partial)");
                assert!((fraction - 0.5).abs() < 0.01);
            }
            other => panic!("Expected PartialClose for short TP, got {:?}", other),
        }
    }

    #[test]
    fn sl_priority_over_trailing() {
        // When both SL and trailing would trigger, SL wins (higher priority)
        let mut pos = long_pos(100.0);
        pos.trailing_sl = Some(99.0); // trailing at 99.0
                                      // SL = 98.0 (base); close = 97.5 → both SL and trailing trigger
        let snap = default_snap(97.5);
        let params = ExitParams {
            enable_breakeven_stop: false,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("SL should have priority, got {:?}", other),
        }
    }

    #[test]
    fn exit_params_default_round_trip() {
        let params = ExitParams::default();
        let json = serde_json::to_string(&params).unwrap();
        let deser: ExitParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, deser);
    }

    // ── Glitch guard tests (AQC-712) ────────────────────────────────────

    #[test]
    fn test_glitch_guard_blocks_exit_on_price_spike() {
        let mut pos = long_pos(200.0);
        let mut snap = default_snap(150.0);
        snap.prev_close = 100.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: true,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(
            result,
            KernelExitResult::Hold,
            "Glitch guard should block exit on price spike"
        );
    }

    #[test]
    fn test_glitch_guard_blocks_exit_on_atr_spike() {
        let mut pos = long_pos(200.0);
        let mut snap = default_snap(115.0);
        snap.prev_close = 100.0;
        snap.atr = 1.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: true,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(
            result,
            KernelExitResult::Hold,
            "Glitch guard should block exit on ATR spike"
        );
    }

    #[test]
    fn test_glitch_guard_allows_exit_normal_move() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(97.5);
        snap.prev_close = 100.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: true,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected SL to trigger on normal move, got {:?}", other),
        }
    }

    #[test]
    fn test_glitch_guard_disabled_allows_extreme_move() {
        let mut pos = long_pos(200.0);
        let mut snap = default_snap(150.0);
        snap.prev_close = 100.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: false,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected SL when glitch guard disabled, got {:?}", other),
        }
    }

    #[test]
    fn test_glitch_guard_prev_close_zero_skips() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(97.5);
        snap.prev_close = 0.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: true,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => assert_eq!(reason, "Stop Loss"),
            other => panic!("Expected SL when prev_close=0 skips guard, got {:?}", other),
        }
    }

    // ── Smart exit tests ────────────────────────────────────────────────

    fn smart_exit_params() -> ExitParams {
        ExitParams {
            enable_partial_tp: false,
            smart_exit_adx_exhaustion_lt: 0.0,
            require_macro_alignment: false,
            ..default_params()
        }
    }

    #[test]
    fn test_smart_exit_trend_breakdown_ema_cross() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(100.0);
        snap.ema_fast = 99.0;
        snap.ema_slow = 100.0;
        snap.adx = 20.0;
        let params = smart_exit_params();
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("Trend Breakdown"), "got: {reason}");
            }
            other => panic!("Expected FullClose(Trend Breakdown), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_weak_cross_suppression() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(100.0);
        snap.ema_fast = 99.95;
        snap.ema_slow = 100.0;
        snap.adx = 30.0;
        let params = smart_exit_params();
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    #[test]
    fn test_smart_exit_trend_exhaustion_adx_below_threshold() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(100.0);
        snap.adx = 15.0;
        let params = ExitParams {
            enable_partial_tp: false,
            smart_exit_adx_exhaustion_lt: 20.0,
            require_macro_alignment: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("Trend Exhaustion"), "got: {reason}");
            }
            other => panic!("Expected FullClose(Trend Exhaustion), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_stagnation_low_vol_underwater() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(99.0); // underwater
        snap.atr = 0.5; // 50% of entry_atr (1.0), below 70% threshold
        snap.adx = 30.0; // above exhaustion
        let params = smart_exit_params();
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("Stagnation Exit"), "got: {reason}");
            }
            other => panic!("Expected FullClose(Stagnation Exit), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_stagnation_skips_paxg() {
        let mut pos = long_pos(100.0);
        pos.symbol = "PAXG".to_string();
        let mut snap = default_snap(99.0);
        snap.atr = 0.5;
        snap.adx = 30.0;
        let params = smart_exit_params();
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        // Should NOT trigger stagnation for PAXG
        assert!(
            !matches!(result, KernelExitResult::FullClose { ref reason, .. } if reason.contains("Stagnation")),
            "PAXG should be exempt from stagnation exit, got {:?}",
            result
        );
    }

    #[test]
    fn test_smart_exit_tsme_momentum_contraction() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(102.0);
        snap.adx = 55.0;
        snap.adx_slope = -1.0;
        snap.macd_hist = 0.5;
        snap.prev_macd_hist = 0.8;
        snap.prev2_macd_hist = 1.1;
        let params = smart_exit_params();
        // profit_atr = (102-100)/1.0 = 2.0 >= tsme_min_profit_atr (1.0)
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("Trend Saturation"), "got: {reason}");
            }
            other => panic!("Expected FullClose(TSME), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_mmde_persistent_divergence() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(102.0);
        snap.adx = 40.0;
        snap.macd_hist = 0.3;
        snap.prev_macd_hist = 0.5;
        snap.prev2_macd_hist = 0.7;
        snap.prev3_macd_hist = 0.9;
        let params = smart_exit_params();
        // profit_atr = (102-100)/1.0 = 2.0 > 1.5, ADX 40 > 35
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(
                    reason.contains("MACD Persistent Divergence"),
                    "got: {reason}"
                );
            }
            other => panic!("Expected FullClose(MMDE), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_rsi_overextension_long() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(100.0);
        snap.rsi = 85.0; // above 80 (lo-profit threshold)
        snap.adx = 30.0;
        let params = smart_exit_params();
        // profit_atr = 0.0 < rsi_exit_profit_atr_switch (1.5) → lo-profit regime → rsi_ub = 80
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        match result {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("RSI Overbought"), "got: {reason}");
            }
            other => panic!("Expected FullClose(RSI Overbought), got {:?}", other),
        }
    }

    #[test]
    fn test_smart_exit_rsi_profit_switch_tighter_threshold() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(100.0);
        snap.rsi = 72.0; // between 70 (hi-profit) and 80 (lo-profit)
        snap.adx = 30.0;
        let params = smart_exit_params();

        // Low profit: rsi_ub = 80 → 72 < 80 → no trigger
        // profit_atr = 0.0 (close=100, entry=100, atr=1.0)
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);

        // High profit: simulate by setting close above entry so profit_atr > switch (1.5)
        let mut snap2 = snap;
        snap2.close = 101.6; // profit_atr = 1.6/1.0 = 1.6 > 1.5 → hi-profit → rsi_ub = 70
        snap2.rsi = 72.0; // 72 > 70 → trigger
        let result2 = evaluate_exits(&mut pos, &snap2, &params, 0);
        match result2 {
            KernelExitResult::FullClose { reason, .. } => {
                assert!(reason.contains("RSI Overbought"), "got: {reason}");
            }
            other => panic!(
                "Expected FullClose(RSI Overbought) with hi-profit, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_smart_exit_no_trigger_normal_conditions() {
        let mut pos = long_pos(100.0);
        let mut snap = default_snap(101.0);
        snap.ema_fast = 101.0;
        snap.ema_slow = 100.0; // EMA aligned for Long
        snap.adx = 30.0; // above any exhaustion threshold
        snap.rsi = 55.0; // normal
        let params = smart_exit_params();
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold);
    }

    // ── ExitBounds tests ────────────────────────────────────────────────

    #[test]
    fn test_exit_bounds_long_basic() {
        // entry=100, atr=1.0, sl_mult=2.0, tp_mult=4.0
        // Use close=100.3 (profit_atr=0.3, below trailing/breakeven thresholds)
        let mut pos = long_pos(100.0);
        let snap = default_snap(100.3);
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        assert_eq!(eval.result, KernelExitResult::Hold);
        let bounds = eval.exit_bounds.expect("bounds should be Some");
        // TP = entry + atr * tp_atr_mult = 100 + 1.0*4.0 = 104
        assert!((bounds.upper_full - 104.0).abs() < 0.01);
        // SL ~98 (no trailing at this profit level), must be below entry
        assert!(
            bounds.lower_full < 100.0,
            "lower_full={}",
            bounds.lower_full
        );
        // Partial TP: tp_partial_atr_mult=0 → falls back to tp_atr_mult=4.0 → 104
        assert!(bounds.upper_partial.is_some());
    }

    #[test]
    fn test_exit_bounds_short_basic() {
        // SHORT: entry=100, atr=1.0, TP below entry, SL above entry
        // Use close=99.7 (profit_atr=0.3, below trailing/breakeven thresholds)
        let mut pos = short_pos(100.0);
        let snap = default_snap(99.7);
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        assert_eq!(eval.result, KernelExitResult::Hold);
        let bounds = eval.exit_bounds.expect("bounds should be Some");
        // TP = entry - atr * tp_atr_mult = 100 - 4.0 = 96
        assert!((bounds.upper_full - 96.0).abs() < 0.01);
        // SL = entry + atr * sl_atr_mult = 100 + 2.0 = 102 (no trailing)
        assert!(
            bounds.lower_full > 100.0,
            "lower_full={}",
            bounds.lower_full
        );
    }

    #[test]
    fn test_exit_bounds_trailing_active() {
        // LONG with trailing_sl set → lower_full picks max(sl, trailing)
        let mut pos = long_pos(100.0);
        pos.trailing_sl = Some(99.5); // trailing above SL (98.0)
        let snap = default_snap(102.0);
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        let bounds = eval.exit_bounds.expect("bounds should be Some");
        // lower_full = max(sl_price, 99.5); sl_price ~98.0 → lower_full ~99.5
        assert!(
            bounds.lower_full >= 99.0,
            "trailing should raise lower_full, got {}",
            bounds.lower_full
        );
    }

    #[test]
    fn test_exit_bounds_tp1_taken() {
        // When tp1_taken, upper_partial should be None
        let mut pos = long_pos(100.0);
        pos.tp1_taken = true;
        let snap = default_snap(101.0);
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        let bounds = eval.exit_bounds.expect("bounds should be Some");
        assert!(
            bounds.upper_partial.is_none(),
            "tp1_taken → upper_partial should be None"
        );
    }

    #[test]
    fn test_exit_bounds_on_sl_hit() {
        // Even when SL fires, bounds should be populated
        let mut pos = long_pos(100.0);
        let snap = default_snap(97.5); // below SL ~98
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        assert!(matches!(eval.result, KernelExitResult::FullClose { .. }));
        let bounds = eval
            .exit_bounds
            .expect("bounds should be present even on SL hit");
        assert!(bounds.upper_full > 100.0);
        assert!(bounds.lower_full < 100.0);
    }

    #[test]
    fn test_exit_bounds_on_hold() {
        // Price in middle, nothing fires → Hold with bounds
        let mut pos = long_pos(100.0);
        let snap = default_snap(100.5);
        let params = default_params();
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        assert_eq!(eval.result, KernelExitResult::Hold);
        assert!(eval.exit_bounds.is_some());
    }

    #[test]
    fn test_exit_bounds_partial_tp_disabled() {
        let mut pos = long_pos(100.0);
        let snap = default_snap(101.0);
        let mut params = default_params();
        params.enable_partial_tp = false;
        let eval = evaluate_exits_with_diagnostics(&mut pos, &snap, &params, 1000);
        let bounds = eval.exit_bounds.expect("bounds should be Some");
        assert!(
            bounds.upper_partial.is_none(),
            "partial TP disabled → upper_partial None"
        );
    }
}
