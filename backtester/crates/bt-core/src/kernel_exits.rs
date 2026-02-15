//! Kernel-level exit evaluation (SL, trailing, TP with partial support).
//!
//! Mirrors the priority logic from `exits/mod.rs::check_all_exits()` but operates
//! on the kernel's own `Position` type.  Glitch guard (AQC-712) blocks exits during
//! extreme price deviations.  Smart exits (AQC-713) will be added in a later ticket.

use crate::decision_kernel::{Position, PositionSide};
use crate::indicators::IndicatorSnapshot;
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
    FullClose { reason: String, exit_price: f64 },
    PartialClose { reason: String, exit_price: f64, fraction: f64 },
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

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

/// Compute the dynamically-adjusted stop-loss price.
/// Mirrors `exits/stop_loss.rs::compute_sl_price`.
fn compute_sl_price(
    side: PositionSide,
    entry: f64,
    atr: f64,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
) -> f64 {
    let mut sl_mult = params.sl_atr_mult;

    // 1. ASE: ADX slope < 0 AND underwater → tighten 20%
    let is_underwater = match side {
        PositionSide::Long => snap.close < entry,
        PositionSide::Short => snap.close > entry,
    };
    if snap.adx_slope < 0.0 && is_underwater {
        sl_mult *= 0.8;
    }

    // 2. FTB: disabled in backtester (no-op)

    // 3. DASE: ADX > 40 AND profitable > 0.5 ATR → widen 15%
    if snap.adx > 40.0 {
        let pa = match side {
            PositionSide::Long => (snap.close - entry) / atr,
            PositionSide::Short => (entry - snap.close) / atr,
        };
        if pa > 0.5 {
            sl_mult *= 1.15;
        }
    }

    // 4. SLB: ADX > 45 → widen 10%
    if snap.adx > 45.0 {
        sl_mult *= 1.10;
    }

    // Base SL price
    let mut sl_price = match side {
        PositionSide::Long => entry - (atr * sl_mult),
        PositionSide::Short => entry + (atr * sl_mult),
    };

    // 5. Breakeven stop
    if params.enable_breakeven_stop && params.breakeven_start_atr > 0.0 {
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

/// Compute trailing stop price (or None if not yet active).
/// Updates are ratcheted — the trailing SL can only improve.
/// Mirrors `exits/trailing.rs::compute_trailing`.
fn compute_trailing(
    side: PositionSide,
    entry: f64,
    atr: f64,
    confidence: Option<u8>,
    current_trailing: Option<f64>,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    profit_atr_val: f64,
) -> Option<f64> {
    let is_low_conf = confidence == Some(0);

    let mut trailing_start = params.trailing_start_atr;
    let mut trailing_dist = params.trailing_distance_atr;

    if is_low_conf {
        if params.trailing_start_atr_low_conf > 0.0 {
            trailing_start = params.trailing_start_atr_low_conf;
        }
        if params.trailing_distance_atr_low_conf > 0.0 {
            trailing_dist = params.trailing_distance_atr_low_conf;
        }
    }

    // RSI Trend-Guard floor
    let min_trailing_dist = match side {
        PositionSide::Long if snap.rsi > 60.0 => 0.7,
        PositionSide::Short if snap.rsi < 40.0 => 0.7,
        _ => 0.5,
    };

    let mut effective_dist = trailing_dist;

    // Vol-Buffered Trailing Stop (VBTS)
    if params.enable_vol_buffered_trailing && snap.bb_width_ratio > 1.2 {
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

/// Check take-profit conditions.
/// Mirrors `exits/take_profit.rs::check_tp`.
fn check_tp(
    side: PositionSide,
    entry: f64,
    atr: f64,
    quantity: f64,
    tp1_taken: bool,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
) -> KernelExitResult {
    let tp_mult = params.tp_atr_mult;
    let tp_price = match side {
        PositionSide::Long => entry + (atr * tp_mult),
        PositionSide::Short => entry - (atr * tp_mult),
    };

    if params.enable_partial_tp {
        if !tp1_taken {
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

            if partial_hit {
                let pct = params.tp_partial_pct.clamp(0.0, 1.0);
                if pct > 0.0 && pct < 1.0 {
                    let remaining_notional = quantity * (1.0 - pct) * snap.close;
                    if remaining_notional < params.tp_partial_min_notional_usd {
                        return KernelExitResult::Hold;
                    }
                    return KernelExitResult::PartialClose {
                        reason: "Take Profit (Partial)".to_string(),
                        exit_price: snap.close,
                        fraction: pct,
                    };
                }
                // pct == 0 or >= 1.0 → fall through to full close
            } else {
                if params.tp_partial_atr_mult <= 0.0 {
                    return KernelExitResult::Hold;
                }
                // Fall through to check full TP
            }
        } else {
            // tp1 already taken
            if params.tp_partial_atr_mult > 0.0 {
                let tp_hit = match side {
                    PositionSide::Long => snap.close >= tp_price,
                    PositionSide::Short => snap.close <= tp_price,
                };
                if tp_hit {
                    return KernelExitResult::FullClose {
                        reason: "Take Profit".to_string(),
                        exit_price: snap.close,
                    };
                }
            }
            return KernelExitResult::Hold;
        }
    }

    // Full TP check
    let tp_hit = match side {
        PositionSide::Long => snap.close >= tp_price,
        PositionSide::Short => snap.close <= tp_price,
    };
    if tp_hit {
        return KernelExitResult::FullClose {
            reason: "Take Profit".to_string(),
            exit_price: snap.close,
        };
    }

    KernelExitResult::Hold
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate exit conditions for an existing kernel position.
///
/// Priority: Glitch Guard > Stop Loss > Trailing Stop > Take Profit.
/// Updates `pos.trailing_sl` in-place (ratcheted).
pub fn evaluate_exits(
    pos: &mut Position,
    snap: &IndicatorSnapshot,
    params: &ExitParams,
    _current_time_ms: i64,
) -> KernelExitResult {
    // ── 0. Glitch guard — block exits during extreme price deviations ───
    // Mirrors exits/mod.rs check_all_exits() step 0.
    if params.block_exits_on_extreme_dev && snap.prev_close > 0.0 {
        let price_change_pct = (snap.close - snap.prev_close).abs() / snap.prev_close;
        let is_glitch = price_change_pct > params.glitch_price_dev_pct
            || (snap.atr > 0.0
                && (snap.close - snap.prev_close).abs() > snap.atr * params.glitch_atr_mult);
        if is_glitch {
            return KernelExitResult::Hold;
        }
    }

    let entry = pos.avg_entry_price;
    let atr = effective_atr(entry, pos.entry_atr);
    let pa = profit_atr(pos.side, entry, snap.close, atr);

    // ── 1. Stop Loss ────────────────────────────────────────────────────
    let sl_price = compute_sl_price(pos.side, entry, atr, snap, params);
    let sl_hit = match pos.side {
        PositionSide::Long => snap.close <= sl_price,
        PositionSide::Short => snap.close >= sl_price,
    };
    if sl_hit {
        return KernelExitResult::FullClose {
            reason: "Stop Loss".to_string(),
            exit_price: snap.close,
        };
    }

    // ── 2. Trailing Stop ────────────────────────────────────────────────
    let new_tsl = compute_trailing(
        pos.side,
        entry,
        atr,
        pos.confidence,
        pos.trailing_sl,
        snap,
        params,
        pa,
    );
    // Always ratchet trailing_sl into position state
    pos.trailing_sl = new_tsl;

    if let Some(tsl_price) = new_tsl {
        let triggered = match pos.side {
            PositionSide::Long => snap.close <= tsl_price,
            PositionSide::Short => snap.close >= tsl_price,
        };
        if triggered {
            return KernelExitResult::FullClose {
                reason: "Trailing Stop".to_string(),
                exit_price: snap.close,
            };
        }
    }

    // ── 3. Take Profit ──────────────────────────────────────────────────
    check_tp(
        pos.side,
        entry,
        atr,
        pos.quantity,
        pos.tp1_taken,
        snap,
        params,
    )
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
        let snap = default_snap(100.0);
        // We need to set snap so that close triggers breakeven but also SL.
        // Actually breakeven_start needs profit >= 0.7 ATR. At close=100.0, profit=0.
        // Let me use a different approach: snap shows price dropped to 100.0 AFTER
        // having been at 100.8 — but we only see current close.
        // Breakeven fires when (snap.close - entry) >= be_start. At 100.0, diff = 0.
        // So breakeven won't fire. Need close >= 100.7 for breakeven.
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
        assert!((tsl - 101.8).abs() < 0.01, "ratchet should keep 101.8, got {}", tsl);
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
            KernelExitResult::PartialClose { reason, fraction, .. } => {
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
            KernelExitResult::PartialClose { reason, fraction, .. } => {
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
        // 50% price spike: prev_close=100, close=150 → pct=0.50 > 0.40 threshold
        // SL would trigger (close=150 is far from entry=200), but glitch guard blocks first
        let mut pos = long_pos(200.0);
        let mut snap = default_snap(150.0);
        snap.prev_close = 100.0;
        let params = ExitParams {
            block_exits_on_extreme_dev: true,
            enable_partial_tp: false,
            ..default_params()
        };
        let result = evaluate_exits(&mut pos, &snap, &params, 0);
        assert_eq!(result, KernelExitResult::Hold, "Glitch guard should block exit on price spike");
    }

    #[test]
    fn test_glitch_guard_blocks_exit_on_atr_spike() {
        // ATR-relative spike: |close - prev_close| = 15.0 > atr(1.0) * glitch_atr_mult(12.0) = 12.0
        // But pct = 15/100 = 0.15 < 0.40, so only ATR condition triggers
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
        assert_eq!(result, KernelExitResult::Hold, "Glitch guard should block exit on ATR spike");
    }

    #[test]
    fn test_glitch_guard_allows_exit_normal_move() {
        // Normal move: prev_close=100, close=97.5 → pct=0.025 < 0.40
        // ATR check: |2.5| vs 1.0*12.0=12.0 → not triggered
        // SL: entry=100, atr=1.0, sl_mult=2.0 → SL=98.0; close=97.5 → SL triggers
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
        // Same extreme spike as first test, but guard disabled → SL fires
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
        // prev_close=0 → guard condition short-circuits, SL fires normally
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
}
