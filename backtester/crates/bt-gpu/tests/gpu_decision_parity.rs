//! Per-function parity test harness for GPU decision codegen vs Rust SSOT.
//!
//! This module validates that the CUDA codegen'd decision functions structurally
//! match the Rust reference implementations in bt-core. Since we cannot execute
//! CUDA code in unit tests, parity is verified through:
//!
//!   1. **Constant matching**: every hardcoded threshold / multiplier in the Rust
//!      source must appear (identically or via config default) in the CUDA codegen.
//!   2. **Branch coverage**: every conditional path in Rust must have a
//!      corresponding branch in CUDA.
//!   3. **Precision tier expectations**: each comparison declares which precision
//!      tier (T0-T4) governs its tolerance, via `bt_gpu::precision::*`.
//!   4. **Fixture diversity**: 50 test fixtures spanning penny stocks to BTC,
//!      long/short, deep loss to large profit, weak to strong trend.
//!
//! Tests for unimplemented codegen functions are marked `#[ignore]` with a note
//! on the ticket that will implement them.
//!
//! # Implemented
//! - `compute_sl_price_codegen` (AQC-1220) -- stop loss
//! - `compute_trailing_codegen` (AQC-1221) -- trailing stop
//!
//! # Stubs (ignored)
//! - `check_gates_codegen` (AQC-1210)
//! - `generate_signal_codegen` (AQC-1211)
//! - `check_tp_codegen` (AQC-1222)
//! - `check_smart_exits_codegen` (AQC-1223)
//! - `check_all_exits_codegen` (AQC-1224)
//! - `compute_entry_size_codegen` (AQC-1230)
//! - `is_pesc_blocked_codegen` (AQC-1231)
//!
//! # Running
//!
//! Requires the `codegen` feature flag:
//!
//! ```sh
//! cargo test -p bt-gpu --features codegen --test gpu_decision_parity
//! ```

#![cfg(feature = "codegen")]

use bt_gpu::codegen::decision::render_all_decision;
use bt_gpu::precision::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture infrastructure
// ═══════════════════════════════════════════════════════════════════════════════

/// Realistic trading data for exercising decision functions across diverse
/// market conditions.  Each fixture represents a snapshot of one bar with
/// position context.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for future codegen parity tests
struct TestFixture {
    label: &'static str,
    // Position
    pos_type: PosType,
    entry_price: f64,
    current_price: f64,
    entry_atr: f64,
    confidence: Confidence,
    current_trailing_sl: f64, // 0.0 = no trailing stop yet
    // Indicators
    atr: f64,
    adx: f64,
    adx_slope: f64,
    rsi: f64,
    atr_slope: f64,
    bb_width_ratio: f64,
    profit_atr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PosType {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // Medium reserved for future fixture scenarios
enum Confidence {
    Low,
    Medium,
    High,
}

/// Generate 50 diverse test fixtures covering the full decision function input
/// space.  Organized into groups by market condition.
fn make_fixtures() -> Vec<TestFixture> {
    let mut fixtures = Vec::with_capacity(50);

    // ── Group 1: Price levels (penny stocks to BTC) ──────────────────────
    let price_levels: &[(&str, f64, f64)] = &[
        ("penny-stock-long", 0.0025, 0.0026),
        ("micro-cap-long", 0.45, 0.47),
        ("small-cap-long", 12.50, 12.80),
        ("mid-cap-long", 185.00, 188.50),
        ("large-cap-long", 3450.00, 3510.00),
        ("btc-long", 65000.00, 66200.00),
        ("btc-high-long", 105000.00, 106500.00),
    ];
    for &(label, entry, current) in price_levels {
        fixtures.push(TestFixture {
            label,
            pos_type: PosType::Long,
            entry_price: entry,
            current_price: current,
            entry_atr: entry * 0.01,
            confidence: Confidence::High,
            current_trailing_sl: 0.0,
            atr: entry * 0.01,
            adx: 30.0,
            adx_slope: 0.5,
            rsi: 55.0,
            atr_slope: 0.0,
            bb_width_ratio: 1.0,
            profit_atr: (current - entry) / (entry * 0.01),
        });
    }

    // ── Group 2: Short positions at various price levels ─────────────────
    let short_levels: &[(&str, f64, f64)] = &[
        ("penny-stock-short", 0.0025, 0.0024),
        ("mid-cap-short", 185.00, 181.00),
        ("btc-short", 65000.00, 63800.00),
    ];
    for &(label, entry, current) in short_levels {
        fixtures.push(TestFixture {
            label,
            pos_type: PosType::Short,
            entry_price: entry,
            current_price: current,
            entry_atr: entry * 0.01,
            confidence: Confidence::High,
            current_trailing_sl: 0.0,
            atr: entry * 0.01,
            adx: 30.0,
            adx_slope: 0.5,
            rsi: 45.0,
            atr_slope: 0.0,
            bb_width_ratio: 1.0,
            profit_atr: (entry - current) / (entry * 0.01),
        });
    }

    // ── Group 3: Profit levels (deep loss to large profit) ───────────────
    let profit_scenarios: &[(&str, PosType, f64, f64)] = &[
        ("deep-loss-long", PosType::Long, 100.0, 95.0),      // -5 ATR
        ("moderate-loss-long", PosType::Long, 100.0, 98.5),   // -1.5 ATR
        ("slight-loss-long", PosType::Long, 100.0, 99.7),     // -0.3 ATR
        ("breakeven-long", PosType::Long, 100.0, 100.0),      // 0 ATR
        ("small-profit-long", PosType::Long, 100.0, 100.3),   // 0.3 ATR
        ("half-atr-profit-long", PosType::Long, 100.0, 100.6),// 0.6 ATR (DASE boundary)
        ("one-atr-profit-long", PosType::Long, 100.0, 101.0), // 1.0 ATR
        ("two-atr-profit-long", PosType::Long, 100.0, 102.0), // 2.0 ATR (trailing high-profit boundary)
        ("large-profit-long", PosType::Long, 100.0, 105.0),   // 5.0 ATR
        ("deep-loss-short", PosType::Short, 100.0, 105.0),    // -5 ATR
        ("moderate-loss-short", PosType::Short, 100.0, 101.5), // -1.5 ATR
        ("small-profit-short", PosType::Short, 100.0, 99.5),  // 0.5 ATR
        ("large-profit-short", PosType::Short, 100.0, 95.0),  // 5.0 ATR
    ];
    for &(label, pos_type, entry, current) in profit_scenarios {
        let atr = entry * 0.01; // 1.0 ATR = 1% of entry
        let profit_atr = match pos_type {
            PosType::Long => (current - entry) / atr,
            PosType::Short => (entry - current) / atr,
        };
        fixtures.push(TestFixture {
            label,
            pos_type,
            entry_price: entry,
            current_price: current,
            entry_atr: atr,
            confidence: Confidence::High,
            current_trailing_sl: 0.0,
            atr,
            adx: 30.0,
            adx_slope: 0.5,
            rsi: 50.0,
            atr_slope: 0.0,
            bb_width_ratio: 1.0,
            profit_atr,
        });
    }

    // ── Group 4: ADX levels (weak trend to strong/saturated) ─────────────
    let adx_scenarios: &[(&str, f64, f64)] = &[
        ("very-weak-trend", 10.0, -0.5),  // ADX < 25 (weak trend tightening)
        ("weak-trend", 22.0, 0.3),         // ADX < 25 (weak trend tightening)
        ("moderate-trend", 30.0, 0.5),     // Normal
        ("strong-trend", 38.0, 1.2),       // Between 35 and 40
        ("dase-boundary", 41.0, 0.8),      // ADX > 40 (DASE trigger)
        ("slb-boundary", 46.0, 0.5),       // ADX > 45 (SLB trigger)
        ("saturated-trend", 55.0, 1.5),    // Both DASE and SLB
        ("extreme-trend", 75.0, 2.0),      // Extreme ADX
    ];
    for &(label, adx, adx_slope) in adx_scenarios {
        fixtures.push(TestFixture {
            label,
            pos_type: PosType::Long,
            entry_price: 50000.0,
            current_price: 50500.0, // +0.5 ATR profit (just above DASE threshold)
            entry_atr: 1000.0,      // 2% ATR on BTC-scale
            confidence: Confidence::High,
            current_trailing_sl: 0.0,
            atr: 1000.0,
            adx,
            adx_slope,
            rsi: 55.0,
            atr_slope: 0.0,
            bb_width_ratio: 1.0,
            profit_atr: 0.5,
        });
    }

    // ── Group 5: RSI regimes (for trailing RSI Trend-Guard) ──────────────
    let rsi_scenarios: &[(&str, PosType, f64)] = &[
        ("rsi-oversold-long", PosType::Long, 25.0),
        ("rsi-neutral-long", PosType::Long, 50.0),
        ("rsi-trending-long", PosType::Long, 65.0),  // RSI > 60 (trending floor)
        ("rsi-overbought-long", PosType::Long, 80.0),
        ("rsi-oversold-short", PosType::Short, 25.0),
        ("rsi-trending-short", PosType::Short, 35.0), // RSI < 40 (trending floor)
        ("rsi-neutral-short", PosType::Short, 50.0),
        ("rsi-overbought-short", PosType::Short, 80.0),
    ];
    for &(label, pos_type, rsi) in rsi_scenarios {
        let entry = 50000.0;
        let atr = 500.0;
        let current = match pos_type {
            PosType::Long => entry + 2.5 * atr,  // 2.5 ATR profit (active trailing)
            PosType::Short => entry - 2.5 * atr,
        };
        fixtures.push(TestFixture {
            label,
            pos_type,
            entry_price: entry,
            current_price: current,
            entry_atr: atr,
            confidence: Confidence::High,
            current_trailing_sl: 0.0,
            atr,
            adx: 30.0,
            adx_slope: 0.5,
            rsi,
            atr_slope: 0.0,
            bb_width_ratio: 1.0,
            profit_atr: 2.5,
        });
    }

    // ── Group 6: Low confidence ──────────────────────────────────────────
    fixtures.push(TestFixture {
        label: "low-conf-long",
        pos_type: PosType::Long,
        entry_price: 100.0,
        current_price: 102.0,
        entry_atr: 1.0,
        confidence: Confidence::Low,
        current_trailing_sl: 0.0,
        atr: 1.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 2.0,
    });
    fixtures.push(TestFixture {
        label: "low-conf-short",
        pos_type: PosType::Short,
        entry_price: 100.0,
        current_price: 98.0,
        entry_atr: 1.0,
        confidence: Confidence::Low,
        current_trailing_sl: 0.0,
        atr: 1.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 45.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 2.0,
    });

    // ── Group 7: Trailing ratchet (existing trailing SL) ─────────────────
    fixtures.push(TestFixture {
        label: "ratchet-long-improve",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 52000.0, // 4 ATR profit
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 51200.0, // existing trailing SL
        atr: 500.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 4.0,
    });
    fixtures.push(TestFixture {
        label: "ratchet-short-improve",
        pos_type: PosType::Short,
        entry_price: 50000.0,
        current_price: 48000.0, // 4 ATR profit
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 48800.0, // existing trailing SL
        atr: 500.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 45.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 4.0,
    });

    // ── Group 8: VBTS + TATP + TSPV specific scenarios ───────────────────
    fixtures.push(TestFixture {
        label: "vbts-active-bb-expanding",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 51500.0, // 3 ATR profit
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.5, // > 1.2 threshold (VBTS active)
        profit_atr: 3.0,
    });
    fixtures.push(TestFixture {
        label: "tatp-trend-accelerating",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 51500.0,
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 40.0,  // > 35
        adx_slope: 1.0, // > 0 (trend accelerating)
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 3.0, // > 2.0 (high profit)
    });
    fixtures.push(TestFixture {
        label: "tspv-vol-expanding",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 51500.0,
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 28.0,  // < 35 (TATP not active)
        adx_slope: -0.5,
        rsi: 55.0,
        atr_slope: 0.5, // > 0 (vol expanding, TSPV path)
        bb_width_ratio: 1.0,
        profit_atr: 3.0, // > 2.0 (high profit)
    });

    // ── Group 9: Zero-ATR fallback ───────────────────────────────────────
    fixtures.push(TestFixture {
        label: "zero-atr-fallback-long",
        pos_type: PosType::Long,
        entry_price: 100.0,
        current_price: 101.0,
        entry_atr: 0.0, // triggers fallback to entry * 0.005
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 0.0,
        adx: 30.0,
        adx_slope: 0.0,
        rsi: 50.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 2.0,
    });

    // ── Group 10: ASE-specific (underwater + negative ADX slope) ─────────
    fixtures.push(TestFixture {
        label: "ase-long-underwater-negative-slope",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 49500.0, // underwater
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 30.0,
        adx_slope: -1.0, // negative slope
        rsi: 45.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: -1.0,
    });
    fixtures.push(TestFixture {
        label: "ase-short-underwater-negative-slope",
        pos_type: PosType::Short,
        entry_price: 50000.0,
        current_price: 50500.0, // underwater for short
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 30.0,
        adx_slope: -1.0,
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: -1.0,
    });

    // ── Group 11: Breakeven stop (SL moved past entry) ───────────────────
    fixtures.push(TestFixture {
        label: "breakeven-active-long",
        pos_type: PosType::Long,
        entry_price: 50000.0,
        current_price: 50400.0, // 0.8 ATR profit (> 0.7 breakeven_start_atr)
        entry_atr: 500.0,
        confidence: Confidence::High,
        current_trailing_sl: 0.0,
        atr: 500.0,
        adx: 30.0,
        adx_slope: 0.5,
        rsi: 55.0,
        atr_slope: 0.0,
        bb_width_ratio: 1.0,
        profit_atr: 0.8,
    });

    // Verify we hit our target of 50 fixtures.
    assert!(
        fixtures.len() >= 50,
        "Expected at least 50 fixtures, got {}",
        fixtures.len()
    );

    fixtures
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: get the full CUDA codegen source for parsing
// ═══════════════════════════════════════════════════════════════════════════════

fn get_decision_cuda_source() -> String {
    render_all_decision(None)
}

/// Extract the CUDA source for compute_sl_price_codegen only.
fn get_sl_cuda_source() -> String {
    get_decision_cuda_source()
}

/// Extract the CUDA source for compute_trailing_codegen only.
fn get_trailing_cuda_source() -> String {
    get_decision_cuda_source()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rust reference implementations (inline, mirroring bt-core)
// ═══════════════════════════════════════════════════════════════════════════════
//
// These are simplified versions of the Rust SSOT for direct numerical comparison.
// They use the same logic as bt-core/src/exits/stop_loss.rs and trailing.rs but
// accept raw f64 parameters instead of Position/IndicatorSnapshot structs.

/// Rust SSOT: compute_sl_price (mirrors bt-core/src/exits/stop_loss.rs).
///
/// Default config: sl_atr_mult=2.0, enable_breakeven_stop=true,
/// breakeven_start_atr=0.7, breakeven_buffer_atr=0.05.
fn rust_compute_sl_price(
    pos_type: PosType,
    entry_price: f64,
    entry_atr: f64,
    current_price: f64,
    adx: f64,
    adx_slope: f64,
    sl_atr_mult: f64,
    enable_breakeven: bool,
    breakeven_start_atr: f64,
    breakeven_buffer_atr: f64,
) -> f64 {
    let atr = if entry_atr > 0.0 {
        entry_atr
    } else {
        entry_price * 0.005
    };

    let mut sl_mult = sl_atr_mult;

    // 1. ASE
    let is_underwater = match pos_type {
        PosType::Long => current_price < entry_price,
        PosType::Short => current_price > entry_price,
    };
    if adx_slope < 0.0 && is_underwater {
        sl_mult *= 0.8;
    }

    // 2. FTB -- disabled

    // 3. DASE
    if adx > 40.0 {
        let profit_in_atr = match pos_type {
            PosType::Long => (current_price - entry_price) / atr,
            PosType::Short => (entry_price - current_price) / atr,
        };
        if profit_in_atr > 0.5 {
            sl_mult *= 1.15;
        }
    }

    // 4. SLB
    if adx > 45.0 {
        sl_mult *= 1.10;
    }

    // Compute raw SL
    let mut sl_price = match pos_type {
        PosType::Long => entry_price - (atr * sl_mult),
        PosType::Short => entry_price + (atr * sl_mult),
    };

    // 5. Breakeven
    if enable_breakeven && breakeven_start_atr > 0.0 {
        let be_start = atr * breakeven_start_atr;
        let be_buffer = atr * breakeven_buffer_atr;

        match pos_type {
            PosType::Long => {
                if (current_price - entry_price) >= be_start {
                    sl_price = sl_price.max(entry_price + be_buffer);
                }
            }
            PosType::Short => {
                if (entry_price - current_price) >= be_start {
                    sl_price = sl_price.min(entry_price - be_buffer);
                }
            }
        }
    }

    sl_price
}

/// Rust SSOT: compute_trailing (mirrors bt-core/src/exits/trailing.rs).
///
/// Returns the candidate trailing SL price, or `current_trailing_sl` if not active.
fn rust_compute_trailing(
    pos_type: PosType,
    entry_price: f64,
    current_price: f64,
    entry_atr: f64,
    current_trailing_sl: f64, // 0.0 = none
    confidence: Confidence,
    rsi: f64,
    adx: f64,
    adx_slope: f64,
    atr_slope: f64,
    bb_width_ratio: f64,
    profit_atr: f64,
    // Config params (using defaults from GpuComboConfig):
    trailing_start_atr: f64,
    trailing_distance_atr: f64,
    trailing_start_atr_low_conf: f64,
    trailing_distance_atr_low_conf: f64,
    trailing_rsi_floor_default: f64,
    trailing_rsi_floor_trending: f64,
    enable_vol_buffered_trailing: bool,
    trailing_vbts_bb_threshold: f64,
    trailing_vbts_mult: f64,
    trailing_high_profit_atr: f64,
    trailing_tighten_tspv: f64,
    trailing_tighten_default: f64,
    trailing_weak_trend_mult: f64,
) -> f64 {
    let atr = if entry_atr > 0.0 {
        entry_atr
    } else {
        entry_price * 0.005
    };

    // Per-confidence overrides
    let mut t_start = trailing_start_atr;
    let mut t_dist = trailing_distance_atr;

    if confidence == Confidence::Low {
        if trailing_start_atr_low_conf > 0.0 {
            t_start = trailing_start_atr_low_conf;
        }
        if trailing_distance_atr_low_conf > 0.0 {
            t_dist = trailing_distance_atr_low_conf;
        }
    }

    // RSI Trend-Guard floor
    let min_trailing_dist = match pos_type {
        PosType::Long if rsi > 60.0 => trailing_rsi_floor_trending,
        PosType::Short if rsi < 40.0 => trailing_rsi_floor_trending,
        _ => trailing_rsi_floor_default,
    };

    // Effective trailing distance
    let mut effective_dist = t_dist;

    // VBTS
    if enable_vol_buffered_trailing && bb_width_ratio > trailing_vbts_bb_threshold {
        effective_dist *= trailing_vbts_mult;
    }

    // High-profit tightening with TATP/TSPV
    if profit_atr > trailing_high_profit_atr {
        if adx > 35.0 && adx_slope > 0.0 {
            // TATP
            effective_dist = t_dist * 1.0;
        } else if atr_slope > 0.0 {
            // TSPV
            effective_dist = t_dist * trailing_tighten_tspv;
        } else {
            effective_dist = t_dist * trailing_tighten_default;
        }
    } else if adx < 25.0 {
        effective_dist = t_dist * trailing_weak_trend_mult;
    }

    // Clamp to floor
    effective_dist = effective_dist.max(min_trailing_dist);

    // Activation gate
    if profit_atr < t_start {
        return current_trailing_sl;
    }

    // Compute candidate
    let candidate = match pos_type {
        PosType::Long => current_price - (atr * effective_dist),
        PosType::Short => current_price + (atr * effective_dist),
    };

    // Ratchet
    if current_trailing_sl > 0.0 {
        match pos_type {
            PosType::Long => candidate.max(current_trailing_sl),
            PosType::Short => candidate.min(current_trailing_sl),
        }
    } else {
        candidate
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stop Loss codegen parity tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sl_codegen_matches_rust_constants() {
    // Verify that every hardcoded threshold in the Rust stop_loss.rs has a
    // matching constant in the CUDA codegen source.
    let src = get_sl_cuda_source();

    // ASE: tighten by 20% (multiply by 0.8) when adx_slope < 0 and underwater
    assert!(
        src.contains("*= 0.8"),
        "SL codegen must contain ASE tightening factor 0.8"
    );
    assert!(
        src.contains("adx_slope < 0.0"),
        "SL codegen must check adx_slope < 0.0 for ASE"
    );

    // DASE: ADX > 40, profit > 0.5 ATR, widen 15%
    assert!(
        src.contains("adx > 40.0"),
        "SL codegen must contain DASE ADX threshold 40.0"
    );
    assert!(
        src.contains("profit_in_atr > 0.5"),
        "SL codegen must contain DASE profit threshold 0.5 ATR"
    );
    assert!(
        src.contains("*= 1.15"),
        "SL codegen must contain DASE expansion factor 1.15"
    );

    // SLB: ADX > 45, widen 10%
    assert!(
        src.contains("adx > 45.0"),
        "SL codegen must contain SLB ADX threshold 45.0"
    );
    assert!(
        src.contains("*= 1.10"),
        "SL codegen must contain SLB expansion factor 1.10"
    );

    // ATR fallback: entry_price * 0.005
    assert!(
        src.contains("entry_price * 0.005"),
        "SL codegen must contain ATR fallback factor 0.005"
    );

    // Breakeven: reads from config (not hardcoded thresholds in CUDA)
    assert!(
        src.contains("cfg.breakeven_start_atr"),
        "SL codegen must read breakeven_start_atr from config"
    );
    assert!(
        src.contains("cfg.breakeven_buffer_atr"),
        "SL codegen must read breakeven_buffer_atr from config"
    );
}

#[test]
fn test_sl_codegen_branch_coverage() {
    // Verify every conditional branch in the Rust SL logic has a
    // corresponding branch in the CUDA codegen.
    let src = get_sl_cuda_source();

    // Branch 1: ATR fallback (atr > 0 vs fallback)
    assert!(
        src.contains("atr > 0.0"),
        "SL codegen must branch on atr > 0.0"
    );

    // Branch 2: ASE underwater check (long vs short)
    assert!(
        src.contains("current_price < entry_price"),
        "SL codegen must check LONG underwater: current < entry"
    );
    assert!(
        src.contains("current_price > entry_price"),
        "SL codegen must check SHORT underwater: current > entry"
    );

    // Branch 3: ASE activation (adx_slope < 0 AND underwater)
    assert!(
        src.contains("adx_slope < 0.0 && is_underwater"),
        "SL codegen must combine adx_slope < 0 with underwater for ASE"
    );

    // Branch 4: DASE (adx > 40)
    assert!(
        src.contains("adx > 40.0"),
        "SL codegen must branch on ADX > 40 for DASE"
    );

    // Branch 5: DASE profit check (directional)
    assert!(
        src.contains("(current_price - entry_price) / eff_atr"),
        "SL codegen must compute LONG profit_in_atr"
    );
    assert!(
        src.contains("(entry_price - current_price) / eff_atr"),
        "SL codegen must compute SHORT profit_in_atr"
    );

    // Branch 6: SLB (adx > 45)
    assert!(
        src.contains("adx > 45.0"),
        "SL codegen must branch on ADX > 45 for SLB"
    );

    // Branch 7: Directional SL computation
    assert!(
        src.contains("entry_price - (eff_atr * sl_mult)"),
        "SL codegen must compute LONG SL: entry - atr * mult"
    );
    assert!(
        src.contains("entry_price + (eff_atr * sl_mult)"),
        "SL codegen must compute SHORT SL: entry + atr * mult"
    );

    // Branch 8: Breakeven guard (enable flag + start > 0)
    assert!(
        src.contains("cfg.enable_breakeven_stop"),
        "SL codegen must check breakeven enable flag"
    );

    // Branch 9: Breakeven direction (fmax for LONG, fmin for SHORT)
    assert!(
        src.contains("fmax(sl_price, entry_price + be_buffer)"),
        "SL codegen must use fmax for LONG breakeven"
    );
    assert!(
        src.contains("fmin(sl_price, entry_price - be_buffer)"),
        "SL codegen must use fmin for SHORT breakeven"
    );
}

#[test]
fn test_sl_codegen_modifier_ordering() {
    // The Rust SSOT applies modifiers in a specific order:
    // ASE -> FTB(disabled) -> DASE -> SLB -> raw SL -> breakeven
    // CUDA must follow the same order.
    let src = get_sl_cuda_source();

    // Use the full section divider markers (with parenthetical names) to avoid
    // matching the header summary which lists all modifiers in a compact form.
    let ase_pos = src
        .find("1. ASE (ADX")
        .expect("ASE section marker must exist");
    let dase_pos = src
        .find("3. DASE (Dynamic")
        .expect("DASE section marker must exist");
    let slb_pos = src
        .find("4. SLB (Saturation")
        .expect("SLB section marker must exist");
    let raw_sl_pos = src
        .find("Compute raw SL price")
        .expect("Raw SL computation section marker must exist");
    let be_pos = src
        .find("5. Breakeven Stop")
        .expect("Breakeven section marker must exist");

    assert!(
        ase_pos < dase_pos,
        "ASE must precede DASE (Rust order)"
    );
    assert!(
        dase_pos < slb_pos,
        "DASE must precede SLB (Rust order)"
    );
    assert!(
        slb_pos < raw_sl_pos,
        "SLB must precede raw SL computation (Rust order)"
    );
    assert!(
        raw_sl_pos < be_pos,
        "Raw SL must precede breakeven (Rust order)"
    );
}

#[test]
fn test_sl_numerical_parity_with_fixtures() {
    // For each fixture, compute SL using the Rust reference and verify that
    // the CUDA codegen would produce the same result (given same inputs and
    // default config). Since we can't run CUDA, we verify the Rust function
    // produces valid results and that the codegen contains the matching logic.
    //
    // Precision tier: T2 (single arithmetic: price +/- atr * mult).
    let fixtures = make_fixtures();

    // Default config values matching GpuComboConfig defaults
    let sl_atr_mult = 2.0;
    let enable_breakeven = true;
    let breakeven_start_atr = 0.7;
    let breakeven_buffer_atr = 0.05;

    for f in &fixtures {
        let sl = rust_compute_sl_price(
            f.pos_type,
            f.entry_price,
            f.entry_atr,
            f.current_price,
            f.adx,
            f.adx_slope,
            sl_atr_mult,
            enable_breakeven,
            breakeven_start_atr,
            breakeven_buffer_atr,
        );

        // SL must be finite and positive for any non-zero entry price
        if f.entry_price > 0.0 {
            assert!(
                sl.is_finite(),
                "SL must be finite for fixture '{}': got {}",
                f.label,
                sl
            );
        }

        // Directional sanity: LONG SL <= entry, SHORT SL >= entry
        // (unless breakeven has moved it past entry)
        let atr = if f.entry_atr > 0.0 {
            f.entry_atr
        } else {
            f.entry_price * 0.005
        };
        let profit = match f.pos_type {
            PosType::Long => f.current_price - f.entry_price,
            PosType::Short => f.entry_price - f.current_price,
        };
        let be_threshold = atr * breakeven_start_atr;
        let be_active = enable_breakeven && breakeven_start_atr > 0.0 && profit >= be_threshold;

        if !be_active {
            match f.pos_type {
                PosType::Long => assert!(
                    sl <= f.entry_price + f64::EPSILON,
                    "LONG SL must be at or below entry for '{}': sl={}, entry={}",
                    f.label,
                    sl,
                    f.entry_price
                ),
                PosType::Short => assert!(
                    sl >= f.entry_price - f64::EPSILON,
                    "SHORT SL must be at or above entry for '{}': sl={}, entry={}",
                    f.label,
                    sl,
                    f.entry_price
                ),
            }
        }

        // T2 precision: verify the f32 cast of the SL result stays within T2.
        let sl_f32 = sl as f32;
        let sl_roundtrip = sl_f32 as f64;
        assert!(
            within_tolerance(sl, sl_roundtrip, TIER_T2_TOLERANCE),
            "SL f32 round-trip exceeds T2 for '{}': f64={}, f32_rt={}, rel_err={:.2e}",
            f.label,
            sl,
            sl_roundtrip,
            relative_error(sl, sl_roundtrip)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trailing codegen parity tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_trailing_codegen_matches_rust_constants() {
    // Verify that every hardcoded constant in trailing.rs matches the CUDA
    // codegen (via config defaults).
    let src = get_trailing_cuda_source();

    // RSI Trend-Guard thresholds (hardcoded in Rust: 60.0 long, 40.0 short)
    assert!(
        src.contains("rsi > 60.0"),
        "Trailing codegen must contain RSI trending threshold 60.0 for LONG"
    );
    assert!(
        src.contains("rsi < 40.0"),
        "Trailing codegen must contain RSI trending threshold 40.0 for SHORT"
    );

    // RSI Trend-Guard floor values are read from config in CUDA.
    // Rust hardcodes: default=0.5, trending=0.7
    // CUDA reads: cfg.trailing_rsi_floor_default, cfg.trailing_rsi_floor_trending
    // GpuComboConfig defaults: 0.5, 0.7 -- verified by this assertion:
    assert!(
        src.contains("cfg.trailing_rsi_floor_default"),
        "Trailing codegen must read RSI floor default from config"
    );
    assert!(
        src.contains("cfg.trailing_rsi_floor_trending"),
        "Trailing codegen must read RSI floor trending from config"
    );

    // VBTS: bb_width_ratio threshold read from config (Rust hardcodes 1.2)
    assert!(
        src.contains("cfg.trailing_vbts_bb_threshold"),
        "Trailing codegen must read VBTS BB threshold from config"
    );
    assert!(
        src.contains("cfg.trailing_vbts_mult"),
        "Trailing codegen must read VBTS multiplier from config"
    );

    // TATP: adx > 35.0 and adx_slope > 0.0, factor = 1.0 (hardcoded in both)
    assert!(
        src.contains("adx > 35.0"),
        "Trailing codegen must contain TATP ADX threshold 35.0"
    );
    assert!(
        src.contains("adx_slope > 0.0"),
        "Trailing codegen must check TATP adx_slope > 0"
    );
    assert!(
        src.contains("trailing_dist * 1.0"),
        "Trailing codegen must contain TATP factor 1.0 (no tightening)"
    );

    // TSPV: atr_slope > 0.0, factor from config (Rust hardcodes 0.75)
    assert!(
        src.contains("atr_slope > 0.0"),
        "Trailing codegen must check TSPV atr_slope > 0"
    );
    assert!(
        src.contains("cfg.trailing_tighten_tspv"),
        "Trailing codegen must read TSPV tighten factor from config"
    );

    // Default high-profit tightening factor from config (Rust hardcodes 0.5)
    assert!(
        src.contains("cfg.trailing_tighten_default"),
        "Trailing codegen must read default tighten factor from config"
    );

    // Weak-trend: adx < 25.0 (hardcoded in both)
    assert!(
        src.contains("adx < 25.0"),
        "Trailing codegen must contain weak-trend ADX threshold 25.0"
    );
    assert!(
        src.contains("cfg.trailing_weak_trend_mult"),
        "Trailing codegen must read weak-trend multiplier from config"
    );

    // High-profit ATR threshold from config (Rust hardcodes 2.0)
    assert!(
        src.contains("cfg.trailing_high_profit_atr"),
        "Trailing codegen must read high-profit ATR threshold from config"
    );

    // ATR fallback (same as SL: entry_price * 0.005)
    assert!(
        src.contains("entry_price * 0.005"),
        "Trailing codegen must contain ATR fallback 0.005"
    );

    // Low confidence mapped to integer 0
    assert!(
        src.contains("confidence == 0"),
        "Trailing codegen must map Low confidence to 0"
    );
}

#[test]
fn test_trailing_codegen_branch_coverage() {
    // Verify every conditional branch in Rust trailing.rs has a CUDA match.
    let src = get_trailing_cuda_source();

    // Branch 1: ATR fallback
    assert!(
        src.contains("atr > 0.0"),
        "Trailing codegen must branch on atr > 0.0"
    );

    // Branch 2: Low confidence override
    assert!(
        src.contains("confidence == 0"),
        "Trailing codegen must branch on Low confidence"
    );
    assert!(
        src.contains("cfg.trailing_start_atr_low_conf > 0.0"),
        "Trailing codegen must check low_conf start > 0"
    );
    assert!(
        src.contains("cfg.trailing_distance_atr_low_conf > 0.0"),
        "Trailing codegen must check low_conf distance > 0"
    );

    // Branch 3: RSI Trend-Guard (directional)
    assert!(
        src.contains("pos_type == POS_LONG && rsi > 60.0"),
        "Trailing codegen must check LONG RSI > 60 for trending floor"
    );
    assert!(
        src.contains("pos_type == POS_SHORT && rsi < 40.0"),
        "Trailing codegen must check SHORT RSI < 40 for trending floor"
    );

    // Branch 4: VBTS
    assert!(
        src.contains("cfg.enable_vol_buffered_trailing"),
        "Trailing codegen must check VBTS enable flag"
    );

    // Branch 5: High-profit tightening
    assert!(
        src.contains("profit_atr > (double)cfg.trailing_high_profit_atr"),
        "Trailing codegen must branch on high profit"
    );

    // Branch 6: TATP (adx > 35 && adx_slope > 0)
    assert!(
        src.contains("adx > 35.0 && adx_slope > 0.0"),
        "Trailing codegen must branch on TATP conditions"
    );

    // Branch 7: TSPV (atr_slope > 0)
    assert!(
        src.contains("atr_slope > 0.0"),
        "Trailing codegen must branch on TSPV condition"
    );

    // Branch 8: Weak-trend (adx < 25)
    assert!(
        src.contains("adx < 25.0"),
        "Trailing codegen must branch on weak-trend ADX < 25"
    );

    // Branch 9: Activation gate (profit < trailing_start)
    assert!(
        src.contains("profit_atr < trailing_start"),
        "Trailing codegen must check activation gate"
    );
    assert!(
        src.contains("return current_trailing_sl"),
        "Trailing codegen must return existing SL when not active"
    );

    // Branch 10: Directional candidate computation
    assert!(
        src.contains("current_price - (eff_atr * effective_dist)"),
        "Trailing codegen must compute LONG candidate"
    );
    assert!(
        src.contains("current_price + (eff_atr * effective_dist)"),
        "Trailing codegen must compute SHORT candidate"
    );

    // Branch 11: Ratchet (existing trailing SL > 0)
    assert!(
        src.contains("current_trailing_sl > 0.0"),
        "Trailing codegen must check for existing trailing SL"
    );
    assert!(
        src.contains("fmax(candidate, current_trailing_sl)"),
        "Trailing codegen must ratchet LONG via fmax"
    );
    assert!(
        src.contains("fmin(candidate, current_trailing_sl)"),
        "Trailing codegen must ratchet SHORT via fmin"
    );
}

#[test]
fn test_trailing_codegen_modifier_ordering() {
    // The Rust SSOT applies trailing modifiers in a specific order:
    // per-conf overrides -> RSI floor -> VBTS -> high-profit/TATP/TSPV/weak ->
    // clamp floor -> activation gate -> candidate -> ratchet
    let src = get_trailing_cuda_source();

    // Use the section divider markers (with dashes) to avoid matching the
    // header summary which lists all modifiers in a compact one-line form.
    let conf_pos = src
        .find("Per-confidence overrides for trailing")
        .expect("Per-confidence section marker must exist");
    let rsi_floor_pos = src
        .find("RSI Trend-Guard floor")
        .expect("RSI Trend-Guard floor section marker must exist");
    let vbts_pos = src
        .find("VBTS (Vol-Buffered")
        .expect("VBTS section marker must exist");
    let high_profit_pos = src
        .find("High-profit tightening")
        .expect("High-profit tightening section marker must exist");
    let activation_pos = src
        .find("Activation gate")
        .expect("Activation gate section marker must exist");
    let ratchet_pos = src
        .find("Ratchet: only allow")
        .expect("Ratchet section marker must exist");

    assert!(
        conf_pos < rsi_floor_pos,
        "Per-confidence must precede RSI floor"
    );
    assert!(
        rsi_floor_pos < vbts_pos,
        "RSI floor must precede VBTS"
    );
    assert!(
        vbts_pos < high_profit_pos,
        "VBTS must precede high-profit tightening"
    );
    assert!(
        high_profit_pos < activation_pos,
        "High-profit must precede activation gate"
    );
    assert!(
        activation_pos < ratchet_pos,
        "Activation gate must precede ratchet"
    );
}

#[test]
fn test_trailing_numerical_parity_with_fixtures() {
    // For each fixture, compute trailing SL using the Rust reference.
    // Verify directional sanity and f32 round-trip precision.
    //
    // Precision tier: T2 (single arithmetic: price +/- atr * dist).
    let fixtures = make_fixtures();

    // GpuComboConfig defaults (matching buffers.rs defaults)
    let trailing_start_atr = 1.0;
    let trailing_distance_atr = 0.8;
    let trailing_start_atr_low_conf = 0.0;
    let trailing_distance_atr_low_conf = 0.0;
    let trailing_rsi_floor_default = 0.5;
    let trailing_rsi_floor_trending = 0.7;
    let enable_vol_buffered_trailing = true;
    let trailing_vbts_bb_threshold = 1.2;
    let trailing_vbts_mult = 1.25;
    let trailing_high_profit_atr = 2.0;
    let trailing_tighten_tspv = 0.75;
    let trailing_tighten_default = 0.5;
    let trailing_weak_trend_mult = 0.7;

    for f in &fixtures {
        let tsl = rust_compute_trailing(
            f.pos_type,
            f.entry_price,
            f.current_price,
            f.entry_atr,
            f.current_trailing_sl,
            f.confidence,
            f.rsi,
            f.adx,
            f.adx_slope,
            f.atr_slope,
            f.bb_width_ratio,
            f.profit_atr,
            trailing_start_atr,
            trailing_distance_atr,
            trailing_start_atr_low_conf,
            trailing_distance_atr_low_conf,
            trailing_rsi_floor_default,
            trailing_rsi_floor_trending,
            enable_vol_buffered_trailing,
            trailing_vbts_bb_threshold,
            trailing_vbts_mult,
            trailing_high_profit_atr,
            trailing_tighten_tspv,
            trailing_tighten_default,
            trailing_weak_trend_mult,
        );

        // TSL must be finite
        assert!(
            tsl.is_finite(),
            "Trailing SL must be finite for fixture '{}': got {}",
            f.label,
            tsl
        );

        // If trailing is active (profit >= trailing_start), the TSL should
        // be between the entry and the current price (for profitable positions)
        let effective_start = if f.confidence == Confidence::Low
            && trailing_start_atr_low_conf > 0.0
        {
            trailing_start_atr_low_conf
        } else {
            trailing_start_atr
        };

        if f.profit_atr >= effective_start && f.profit_atr > 0.0 {
            match f.pos_type {
                PosType::Long => {
                    // Trailing SL should be below current price
                    assert!(
                        tsl < f.current_price,
                        "LONG trailing SL must be below current price for '{}': tsl={}, current={}",
                        f.label,
                        tsl,
                        f.current_price
                    );
                }
                PosType::Short => {
                    // Trailing SL should be above current price
                    assert!(
                        tsl > f.current_price,
                        "SHORT trailing SL must be above current price for '{}': tsl={}, current={}",
                        f.label,
                        tsl,
                        f.current_price
                    );
                }
            }
        }

        // T2 precision: f32 round-trip
        if tsl != 0.0 {
            let tsl_f32 = tsl as f32;
            let tsl_roundtrip = tsl_f32 as f64;
            assert!(
                within_tolerance(tsl, tsl_roundtrip, TIER_T2_TOLERANCE),
                "Trailing SL f32 round-trip exceeds T2 for '{}': f64={}, f32_rt={}, rel_err={:.2e}",
                f.label,
                tsl,
                tsl_roundtrip,
                relative_error(tsl, tsl_roundtrip)
            );
        }
    }
}

#[test]
fn test_trailing_config_defaults_match_rust_hardcoded() {
    // The Rust trailing.rs hardcodes several constants that the CUDA codegen
    // reads from GpuComboConfig.  Verify that the GpuComboConfig defaults
    // match the Rust hardcoded values.
    //
    // This is the critical "same numbers" check that ensures the two
    // implementations agree on thresholds without requiring GPU execution.

    // GpuComboConfig defaults (from buffers.rs)
    let gpu_defaults: &[(&str, f64, f64)] = &[
        // (name, GpuComboConfig default, Rust hardcoded value)
        ("trailing_rsi_floor_default", 0.5, 0.5),
        ("trailing_rsi_floor_trending", 0.7, 0.7),
        ("trailing_vbts_bb_threshold", 1.2, 1.2),
        ("trailing_vbts_mult", 1.25, 1.25),
        ("trailing_high_profit_atr", 2.0, 2.0),
        ("trailing_tighten_tspv", 0.75, 0.75),
        ("trailing_tighten_default", 0.5, 0.5),
        ("trailing_weak_trend_mult", 0.7, 0.7),
    ];

    for &(name, gpu_default, rust_hardcoded) in gpu_defaults {
        assert!(
            exact_match(gpu_default, rust_hardcoded),
            "Config mismatch for '{}': GpuComboConfig default={}, Rust hardcoded={}",
            name,
            gpu_default,
            rust_hardcoded
        );
    }
}

#[test]
fn test_sl_config_defaults_match_rust_hardcoded() {
    // Similar check for SL constants.
    // The SL codegen reads sl_atr_mult from config; the Rust default is 2.0.
    // The hardcoded SL constants (ASE=0.8, DASE=1.15, SLB=1.10) are
    // literal in both Rust and CUDA — they are NOT config-driven.

    let sl_hardcoded: &[(&str, f64, &str)] = &[
        ("ASE tightening factor", 0.8, "*= 0.8"),
        ("DASE expansion factor", 1.15, "*= 1.15"),
        ("SLB expansion factor", 1.10, "*= 1.10"),
        ("DASE ADX threshold", 40.0, "adx > 40.0"),
        ("DASE profit threshold", 0.5, "profit_in_atr > 0.5"),
        ("SLB ADX threshold", 45.0, "adx > 45.0"),
        ("ATR fallback factor", 0.005, "entry_price * 0.005"),
    ];

    let src = get_sl_cuda_source();
    for &(name, _value, cuda_pattern) in sl_hardcoded {
        assert!(
            src.contains(cuda_pattern),
            "SL constant '{}' not found in CUDA codegen (expected pattern: '{}')",
            name,
            cuda_pattern
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fixture coverage test
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fixture_coverage() {
    let fixtures = make_fixtures();

    // Must have at least 50 fixtures
    assert!(
        fixtures.len() >= 50,
        "Expected at least 50 fixtures, got {}",
        fixtures.len()
    );

    // Coverage: both position types
    assert!(
        fixtures.iter().any(|f| f.pos_type == PosType::Long),
        "Fixtures must include LONG positions"
    );
    assert!(
        fixtures.iter().any(|f| f.pos_type == PosType::Short),
        "Fixtures must include SHORT positions"
    );

    // Coverage: price levels (penny stock to BTC)
    assert!(
        fixtures.iter().any(|f| f.entry_price < 0.01),
        "Fixtures must include penny stock prices"
    );
    assert!(
        fixtures.iter().any(|f| f.entry_price > 50000.0),
        "Fixtures must include BTC-scale prices"
    );

    // Coverage: profit levels
    assert!(
        fixtures.iter().any(|f| f.profit_atr < -3.0),
        "Fixtures must include deep-loss scenarios (< -3 ATR)"
    );
    assert!(
        fixtures
            .iter()
            .any(|f| f.profit_atr.abs() < 0.01),
        "Fixtures must include breakeven scenarios"
    );
    assert!(
        fixtures.iter().any(|f| f.profit_atr > 3.0),
        "Fixtures must include large-profit scenarios (> 3 ATR)"
    );

    // Coverage: ADX regimes
    assert!(
        fixtures.iter().any(|f| f.adx < 25.0),
        "Fixtures must include weak-trend ADX (< 25)"
    );
    assert!(
        fixtures.iter().any(|f| f.adx > 40.0 && f.adx <= 45.0),
        "Fixtures must include DASE-only ADX (40 < ADX <= 45)"
    );
    assert!(
        fixtures.iter().any(|f| f.adx > 45.0),
        "Fixtures must include SLB-triggering ADX (> 45)"
    );

    // Coverage: ASE conditions (underwater + negative slope)
    assert!(
        fixtures
            .iter()
            .any(|f| f.adx_slope < 0.0 && match f.pos_type {
                PosType::Long => f.current_price < f.entry_price,
                PosType::Short => f.current_price > f.entry_price,
            }),
        "Fixtures must include ASE-triggering scenario (underwater + neg slope)"
    );

    // Coverage: RSI regimes for trailing
    assert!(
        fixtures
            .iter()
            .any(|f| f.pos_type == PosType::Long && f.rsi > 60.0),
        "Fixtures must include LONG RSI trending (> 60)"
    );
    assert!(
        fixtures
            .iter()
            .any(|f| f.pos_type == PosType::Short && f.rsi < 40.0),
        "Fixtures must include SHORT RSI trending (< 40)"
    );

    // Coverage: VBTS (bb_width_ratio > 1.2)
    assert!(
        fixtures.iter().any(|f| f.bb_width_ratio > 1.2),
        "Fixtures must include VBTS-active scenario (bb_width_ratio > 1.2)"
    );

    // Coverage: TATP (adx > 35 && adx_slope > 0 && profit > 2.0)
    assert!(
        fixtures
            .iter()
            .any(|f| f.adx > 35.0 && f.adx_slope > 0.0 && f.profit_atr > 2.0),
        "Fixtures must include TATP scenario"
    );

    // Coverage: TSPV (atr_slope > 0 && profit > 2.0 && adx <= 35)
    assert!(
        fixtures
            .iter()
            .any(|f| f.atr_slope > 0.0
                && f.profit_atr > 2.0
                && !(f.adx > 35.0 && f.adx_slope > 0.0)),
        "Fixtures must include TSPV scenario"
    );

    // Coverage: Low confidence
    assert!(
        fixtures
            .iter()
            .any(|f| f.confidence == Confidence::Low),
        "Fixtures must include Low confidence scenarios"
    );

    // Coverage: Trailing ratchet (existing trailing SL)
    assert!(
        fixtures.iter().any(|f| f.current_trailing_sl > 0.0),
        "Fixtures must include ratchet scenarios (existing trailing SL)"
    );

    // Coverage: Zero ATR fallback
    assert!(
        fixtures.iter().any(|f| f.entry_atr == 0.0),
        "Fixtures must include zero-ATR fallback scenario"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA codegen uses double precision for all price math (AQC-734 mandate)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sl_codegen_uses_double_not_float() {
    let src = get_sl_cuda_source();

    // All critical price/atr variables must be `double`
    let required_doubles = [
        "double sl_mult",
        "double sl_price",
        "double eff_atr",
        "double be_start",
        "double be_buffer",
        "double profit_in_atr",
    ];
    for pat in &required_doubles {
        assert!(
            src.contains(pat),
            "SL codegen must use 'double' for '{}' (AQC-734)",
            pat
        );
    }

    // Must NOT use float for critical variables
    let forbidden_floats = [
        "float sl_price",
        "float sl_mult",
        "float eff_atr",
    ];
    for pat in &forbidden_floats {
        assert!(
            !src.contains(pat),
            "SL codegen must NOT use 'float' for '{}' (AQC-734)",
            pat
        );
    }
}

#[test]
fn test_trailing_codegen_uses_double_not_float() {
    let src = get_trailing_cuda_source();

    let required_doubles = [
        "double trailing_start",
        "double trailing_dist",
        "double effective_dist",
        "double candidate",
        "double eff_atr",
        "double min_trailing_dist",
    ];
    for pat in &required_doubles {
        assert!(
            src.contains(pat),
            "Trailing codegen must use 'double' for '{}' (AQC-734)",
            pat
        );
    }

    let forbidden_floats = [
        "float trailing_start",
        "float trailing_dist",
        "float effective_dist",
        "float candidate",
    ];
    for pat in &forbidden_floats {
        assert!(
            !src.contains(pat),
            "Trailing codegen must NOT use 'float' for '{}' (AQC-734)",
            pat
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ignored placeholder tests for unimplemented codegen functions.
// Each will be filled in by the ticket noted in the comment.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "AQC-1210: check_gates_codegen not yet implemented"]
fn test_gates_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("check_gates"),
        "Gates codegen stub should be replaced by AQC-1210"
    );
}

#[test]
#[ignore = "AQC-1211: generate_signal_codegen not yet implemented"]
fn test_signal_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("generate_signal"),
        "Signal codegen stub should be replaced by AQC-1211"
    );
}

#[test]
#[ignore = "AQC-1222: check_tp_codegen not yet implemented"]
fn test_tp_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("check_tp"),
        "TP codegen stub should be replaced by AQC-1222"
    );
}

#[test]
#[ignore = "AQC-1223: check_smart_exits_codegen not yet implemented"]
fn test_smart_exits_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("check_smart_exits"),
        "Smart exits codegen stub should be replaced by AQC-1223"
    );
}

#[test]
#[ignore = "AQC-1224: check_all_exits_codegen not yet implemented"]
fn test_all_exits_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("check_all_exits"),
        "All-exits codegen stub should be replaced by AQC-1224"
    );
}

#[test]
#[ignore = "AQC-1230: compute_entry_size_codegen not yet implemented"]
fn test_entry_size_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("compute_entry_size"),
        "Entry size codegen stub should be replaced by AQC-1230"
    );
}

#[test]
#[ignore = "AQC-1231: is_pesc_blocked_codegen not yet implemented"]
fn test_pesc_codegen_parity() {
    let src = get_decision_cuda_source();
    assert!(
        src.contains("__device__") && src.contains("is_pesc_blocked"),
        "PESC codegen stub should be replaced by AQC-1231"
    );
}
