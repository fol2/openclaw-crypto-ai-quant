//! GPU buffer structs with bytemuck Pod/Zeroable for direct GPU upload.
//!
//! All structs are `#[repr(C)]` and 16-byte aligned for CUDA compatibility.
//! The runtime uses mixed precision:
//! - raw OHLCV candles are preserved as f64 for indicator parity
//! - snapshots/trade config remain f32 for throughput
//!
//! **M12 — Precision budget:** decision kernels still consume f32 snapshots
//! while the CPU reference engine is f64. See [`crate::precision`] for formal
//! tolerance tiers (T0–T4) used by the parity test suite.

use bytemuck::{Pod, Zeroable};

/// Hard symbol ceiling imposed by GPU kernel state layout.
pub const GPU_MAX_SYMBOLS: usize = 52;
/// Fixed-size ring buffer capacity for per-combo GPU execution trace events.
pub const GPU_TRACE_CAP: usize = 1024;
/// Trace symbol selector sentinel: capture events for all symbols.
pub const GPU_TRACE_SYMBOL_ALL: u32 = u32::MAX;
/// Default anomaly Bollinger-Band width-ratio threshold (not part of sweep axes).
pub const DEFAULT_ANOMALY_BB_WIDTH_RATIO_GT: f32 = 3.0;

// ═══════════════════════════════════════════════════════════════════════════
// GpuSnapshot — precomputed indicator values per (bar, symbol)
// ═══════════════════════════════════════════════════════════════════════════

/// Precomputed indicator snapshot for one (bar, symbol) pair.
/// Layout: `snapshots[bar_idx * num_symbols + sym_idx]`
///
/// 160 bytes, aligned to 16.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSnapshot {
    // OHLCV
    pub close: f32,
    pub high: f32,
    pub low: f32,
    pub open: f32,
    pub volume: f32,
    pub t_sec: u32,

    // EMAs
    pub ema_fast: f32,
    pub ema_slow: f32,
    pub ema_macro: f32,

    // ADX
    pub adx: f32,
    pub adx_slope: f32,
    pub adx_pos: f32,
    pub adx_neg: f32,

    // ATR
    pub atr: f32,
    pub atr_slope: f32,
    pub avg_atr: f32,

    // Bollinger Bands
    pub bb_upper: f32,
    pub bb_lower: f32,
    pub bb_width: f32,
    pub bb_width_ratio: f32,

    // RSI
    pub rsi: f32,
    pub stoch_k: f32,
    pub stoch_d: f32,

    // MACD
    pub macd_hist: f32,
    pub prev_macd_hist: f32,
    pub prev2_macd_hist: f32,
    pub prev3_macd_hist: f32,

    // Volume
    pub vol_sma: f32,
    pub vol_trend: u32, // bool as u32

    // Lagged
    pub prev_close: f32,
    pub prev_ema_fast: f32,
    pub prev_ema_slow: f32,

    // Precomputed on CPU (engine-level)
    pub ema_slow_slope_pct: f32,

    // Meta
    pub bar_count: u32,
    pub valid: u32, // 0 = no bar for this symbol at this timestamp

    // Funding
    pub funding_rate: f32,

    // Padding to 160 bytes (40 × 4 bytes)
    pub _pad: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuSnapshot>() == 160);

// ═══════════════════════════════════════════════════════════════════════════
// GpuRawCandle — raw OHLCV bar (uploaded to GPU once, ~6 MB total)
// ═══════════════════════════════════════════════════════════════════════════

/// Raw OHLCV candle for GPU upload. Missing bars have close=0.
/// Layout: `candles[bar_idx * num_symbols + sym_idx]`
///
/// 56 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRawCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub t_sec: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<GpuRawCandle>() == 56);

// ═══════════════════════════════════════════════════════════════════════════
// GpuIndicatorConfig — indicator window parameters per indicator combo
// ═══════════════════════════════════════════════════════════════════════════

/// Indicator window parameters for GPU-side indicator computation.
/// One per indicator combo (uploaded per VRAM batch).
///
/// 80 bytes (20 × u32).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuIndicatorConfig {
    pub ema_fast_window: u32,
    pub ema_slow_window: u32,
    pub ema_macro_window: u32,
    pub adx_window: u32,
    pub bb_window: u32,
    pub bb_width_avg_window: u32,
    pub atr_window: u32,
    pub rsi_window: u32,
    pub vol_sma_window: u32,
    pub vol_trend_window: u32,
    pub stoch_rsi_window: u32,
    pub stoch_rsi_smooth1: u32,
    pub stoch_rsi_smooth2: u32,
    pub avg_atr_window: u32,
    pub slow_drift_slope_window: u32,
    pub lookback: u32,
    pub use_stoch_rsi: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<GpuIndicatorConfig>() == 80);

fn checked_u32_from_usize(name: &str, val: usize) -> u32 {
    u32::try_from(val).unwrap_or_else(|_| {
        eprintln!(
            "[gpu-config] {name} value {val} exceeds u32::MAX; clamping to {}",
            u32::MAX
        );
        u32::MAX
    })
}

impl GpuIndicatorConfig {
    /// Build from StrategyConfig using canonical runtime indicator windows.
    pub fn from_strategy_config(cfg: &bt_core::config::StrategyConfig, lookback: usize) -> Self {
        let ic = &cfg.indicators;
        Self {
            ema_fast_window: checked_u32_from_usize(
                "indicators.ema_fast_window",
                ic.ema_fast_window,
            ),
            ema_slow_window: checked_u32_from_usize(
                "indicators.ema_slow_window",
                ic.ema_slow_window,
            ),
            ema_macro_window: checked_u32_from_usize(
                "indicators.ema_macro_window",
                ic.ema_macro_window,
            ),
            adx_window: checked_u32_from_usize("indicators.adx_window", ic.adx_window),
            bb_window: checked_u32_from_usize("indicators.bb_window", ic.bb_window),
            bb_width_avg_window: checked_u32_from_usize(
                "indicators.bb_width_avg_window",
                ic.bb_width_avg_window,
            ),
            atr_window: checked_u32_from_usize("indicators.atr_window", ic.atr_window),
            rsi_window: checked_u32_from_usize("indicators.rsi_window", ic.rsi_window),
            vol_sma_window: checked_u32_from_usize("indicators.vol_sma_window", ic.vol_sma_window),
            vol_trend_window: checked_u32_from_usize(
                "indicators.vol_trend_window",
                ic.vol_trend_window,
            ),
            stoch_rsi_window: checked_u32_from_usize(
                "indicators.stoch_rsi_window",
                ic.stoch_rsi_window,
            ),
            stoch_rsi_smooth1: checked_u32_from_usize(
                "indicators.stoch_rsi_smooth1",
                ic.stoch_rsi_smooth1,
            ),
            stoch_rsi_smooth2: checked_u32_from_usize(
                "indicators.stoch_rsi_smooth2",
                ic.stoch_rsi_smooth2,
            ),
            avg_atr_window: checked_u32_from_usize(
                "thresholds.entry.ave_avg_atr_window",
                cfg.effective_ave_avg_atr_window(),
            ),
            slow_drift_slope_window: checked_u32_from_usize(
                "thresholds.entry.slow_drift_slope_window",
                cfg.thresholds.entry.slow_drift_slope_window,
            ),
            lookback: checked_u32_from_usize("lookback", lookback),
            use_stoch_rsi: cfg.filters.use_stoch_rsi_filter as u32,
            _pad: [0; 3],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IndicatorParams — uniform params for indicator kernel
// ═══════════════════════════════════════════════════════════════════════════

/// Global params for the indicator/breadth CUDA kernels.
///
/// 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndicatorParams {
    pub num_ind_combos: u32,
    pub num_symbols: u32,
    pub num_bars: u32,
    pub btc_sym_idx: u32,
    pub _pad: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<IndicatorParams>() == 32);

// ═══════════════════════════════════════════════════════════════════════════
// GpuComboConfig — flattened trade parameters per combo
// ═══════════════════════════════════════════════════════════════════════════

/// Flattened trade parameters for one sweep combo.
/// Only trade-affecting fields (not indicator windows).
///
/// 652 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuComboConfig {
    // Core sizing [0-3]
    pub allocation_pct: f32,
    pub sl_atr_mult: f32,
    pub tp_atr_mult: f32,
    pub leverage: f32,

    // REEF [4-9]
    pub enable_reef_filter: u32,
    pub reef_long_rsi_block_gt: f32,
    pub reef_short_rsi_block_lt: f32,
    pub reef_adx_threshold: f32,
    pub reef_long_rsi_extreme_gt: f32,
    pub reef_short_rsi_extreme_lt: f32,

    // Dynamic leverage + smart-exit contract [10-13]
    // [10] smart-exit behaviour mask (bit-per-behaviour, see constants below)
    pub smart_exit_behaviour_mask: u32,
    pub leverage_low: f32,
    pub leverage_medium: f32,
    pub leverage_high: f32,
    // [14] non-smart exit behaviour mask (bit-per-behaviour, see constants below)
    pub exit_behaviour_mask: u32,
    // Trailing config (repurposed pad slots) [15]
    pub trailing_rsi_floor_default: f32,

    // Execution [16-19]
    pub slippage_bps: f32,
    pub min_notional_usd: f32,
    pub bump_to_min_notional: u32,
    pub max_total_margin_pct: f32,
    // Trailing config (repurposed pad slots) [20-21]
    pub trailing_rsi_floor_trending: f32,
    pub trailing_vbts_bb_threshold: f32,

    // Dynamic sizing [22-29]
    pub enable_dynamic_sizing: u32,
    pub confidence_mult_high: f32,
    pub confidence_mult_medium: f32,
    pub confidence_mult_low: f32,
    pub adx_sizing_min_mult: f32,
    pub adx_sizing_full_adx: f32,
    pub vol_baseline_pct: f32,
    pub vol_scalar_min: f32,

    // [30] + trailing config [31]
    pub vol_scalar_max: f32,
    pub trailing_vbts_mult: f32,

    // Pyramiding [32-38] + trailing config [39]
    pub enable_pyramiding: u32,
    pub max_adds_per_symbol: u32,
    pub add_fraction_of_base_margin: f32,
    pub add_cooldown_minutes: u32,
    pub add_min_profit_atr: f32,
    pub add_min_confidence: u32, // 0=Low, 1=Medium, 2=High
    pub entry_min_confidence: u32,
    pub trailing_high_profit_atr: f32,

    // Partial TP [40-45]
    pub enable_partial_tp: u32,
    pub tp_partial_pct: f32,
    pub tp_partial_min_notional_usd: f32,
    pub trailing_start_atr: f32,
    pub trailing_distance_atr: f32,
    pub tp_partial_atr_mult: f32,

    // SSF + breakeven [46-49]
    pub enable_ssf_filter: u32,
    pub enable_breakeven_stop: u32,
    pub breakeven_start_atr: f32,
    pub breakeven_buffer_atr: f32,

    // Per-confidence trailing overrides [50-53]
    pub trailing_start_atr_low_conf: f32,
    pub trailing_distance_atr_low_conf: f32,
    pub smart_exit_adx_exhaustion_lt: f32,
    pub smart_exit_adx_exhaustion_lt_low_conf: f32,

    // RSI overextension exit [54-63]
    pub enable_rsi_overextension_exit: u32,
    pub rsi_exit_profit_atr_switch: f32,
    pub rsi_exit_ub_lo_profit: f32,
    pub rsi_exit_ub_hi_profit: f32,
    pub rsi_exit_lb_lo_profit: f32,
    pub rsi_exit_lb_hi_profit: f32,
    pub rsi_exit_ub_lo_profit_low_conf: f32,
    pub rsi_exit_ub_hi_profit_low_conf: f32,
    pub rsi_exit_lb_lo_profit_low_conf: f32,
    pub rsi_exit_lb_hi_profit_low_conf: f32,

    // Reentry cooldown [64-66] + trailing config [67]
    pub reentry_cooldown_minutes: u32,
    pub reentry_cooldown_min_mins: u32,
    pub reentry_cooldown_max_mins: u32,
    pub trailing_tighten_default: f32,

    // Volatility-buffered trailing + TSME [68-70] + trailing config [71]
    pub enable_vol_buffered_trailing: u32,
    pub tsme_min_profit_atr: f32,
    pub tsme_require_adx_slope_negative: u32,
    pub trailing_tighten_tspv: f32,

    // ATR floor / signal reversal / glitch [72-76] + trailing config [77]
    pub min_atr_pct: f32,
    pub reverse_entry_signal: u32,
    pub block_exits_on_extreme_dev: u32,
    pub glitch_price_dev_pct: f32,
    pub glitch_atr_mult: f32,
    pub trailing_weak_trend_mult: f32,

    // Rate limits + entry flags [78-81]
    pub max_open_positions: u32,
    pub max_entry_orders_per_loop: u32,
    pub enable_slow_drift_entries: u32,
    pub slow_drift_require_macd_sign: u32,

    // Filters (gates) [82-91]
    pub enable_ranging_filter: u32,
    pub enable_anomaly_filter: u32,
    pub enable_extension_filter: u32,
    pub require_adx_rising: u32,
    pub adx_rising_saturation: f32,
    pub require_volume_confirmation: u32,
    pub vol_confirm_include_prev: u32,
    pub use_stoch_rsi_filter: u32,
    pub require_btc_alignment: u32,
    pub require_macro_alignment: u32,

    // Market regime [92-97]
    pub enable_regime_filter: u32,
    pub enable_auto_reverse: u32,
    pub auto_reverse_breadth_low: f32,
    pub auto_reverse_breadth_high: f32,
    pub breadth_block_short_above: f32,
    pub breadth_block_long_below: f32,

    // Entry thresholds [98-119]
    pub min_adx: f32,
    pub high_conf_volume_mult: f32,
    pub btc_adx_override: f32,
    pub max_dist_ema_fast: f32,
    pub ave_atr_ratio_gt: f32,
    pub ave_adx_mult: f32,
    pub dre_min_adx: f32,
    pub dre_max_adx: f32,
    pub dre_long_rsi_limit_low: f32,
    pub dre_long_rsi_limit_high: f32,
    pub dre_short_rsi_limit_low: f32,
    pub dre_short_rsi_limit_high: f32,
    pub macd_mode: u32, // 0=Accel, 1=Sign, 2=None
    pub pullback_min_adx: f32,
    pub pullback_rsi_long_min: f32,
    pub pullback_rsi_short_max: f32,
    pub pullback_require_macd_sign: u32,
    pub pullback_confidence: u32,
    pub slow_drift_min_slope_pct: f32,
    pub slow_drift_min_adx: f32,
    pub slow_drift_rsi_long_min: f32,
    pub slow_drift_rsi_short_max: f32,

    // Ranging/anomaly thresholds [120-125]
    pub ranging_adx_lt: f32,
    pub ranging_bb_width_ratio_lt: f32,
    pub anomaly_bb_width_ratio_gt: f32,
    pub slow_drift_ranging_slope_override: f32,
    pub snapshot_offset: u32, // byte offset into concatenated snapshots array (in elements, not bytes)
    pub breadth_offset: u32,  // offset into concatenated breadth/btc_bullish arrays (in elements)

    // TP momentum [126-127]
    pub tp_strong_adx_gt: f32,
    pub tp_weak_adx_lt: f32,

    // === Decision codegen fields (AQC-1250) === [128-138]
    // Pullback entry (mode 2)
    pub enable_pullback_entries: u32,
    // Anomaly filter (CPU uses price_change + ema_dev; GPU previously used only bb_width)
    pub anomaly_price_change_pct: f32,
    pub anomaly_ema_dev_pct: f32,
    // Ranging filter (previously hardcoded in CUDA)
    pub ranging_rsi_low: f32,
    pub ranging_rsi_high: f32,
    pub ranging_min_signals: u32,
    // StochRSI thresholds (previously hardcoded 0.85 / 0.15)
    pub stoch_rsi_block_long_gt: f32,
    pub stoch_rsi_block_short_lt: f32,
    // AVE (Adaptive Volatility Entry) gate
    pub ave_enabled: u32,
    // TP multipliers (confidence-dependent TP)
    pub tp_mult_strong: f32,
    pub tp_mult_weak: f32,

    // Entry/exit cooldown (seconds) [139-140]
    pub entry_cooldown_s: u32,
    pub exit_cooldown_s: u32,

    // Signal mode behaviour contract [141-144]
    pub signal_mode_behaviour_mask: u32,
    pub signal_mode_order_0: u32,
    pub signal_mode_order_1: u32,
    pub signal_mode_order_2: u32,

    // Exit behaviour order contract [145-162]
    pub exit_order_0: u32,
    pub exit_order_1: u32,
    pub exit_order_2: u32,
    pub exit_order_3: u32,
    pub exit_order_4: u32,
    pub exit_order_5: u32,
    pub exit_order_6: u32,
    pub exit_order_7: u32,
    pub exit_order_8: u32,
    pub exit_order_9: u32,
    pub exit_order_10: u32,
    pub exit_order_11: u32,
    pub exit_order_12: u32,
    pub exit_order_13: u32,
    pub exit_order_14: u32,
    pub exit_order_15: u32,
    pub exit_order_16: u32,
    pub exit_order_17: u32,
}

const _: () = assert!(std::mem::size_of::<GpuComboConfig>() == 652);

// ═══════════════════════════════════════════════════════════════════════════
// GpuPosition — per-symbol position state
// ═══════════════════════════════════════════════════════════════════════════

/// Per-symbol position (embedded in GpuComboState).
///
/// 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPosition {
    pub active: u32, // 0=empty, 1=LONG, 2=SHORT
    pub _pad0: u32,
    pub entry_price: f64,
    pub size: f64,
    pub confidence: u32, // 0=Low, 1=Medium, 2=High
    pub _pad1: u32,
    pub entry_atr: f64,
    pub entry_adx_threshold: f64,
    pub trailing_sl: f64, // 0.0 = not set
    pub leverage: f64,
    pub margin_used: f64,
    pub adds_count: u32,
    pub tp1_taken: u32,
    pub open_time_sec: u32,
    pub last_add_time_sec: u32,
    pub kernel_margin_used: f64,
}

const _: () = assert!(std::mem::size_of::<GpuPosition>() == 96);

/// Compact trace event emitted by the GPU sweep kernel.
///
/// Stored in `GpuComboState::trace_events` ring buffer.
///
/// 40 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTraceEvent {
    pub t_sec: u32,
    pub sym: u32,
    pub kind: u32,
    pub side: u32,
    pub reason: u32,
    pub price: f32,
    pub size: f32,
    pub _pad0: u32,
    pub pnl: f64,
}

const _: () = assert!(std::mem::size_of::<GpuTraceEvent>() == 40);

// ═══════════════════════════════════════════════════════════════════════════
// GpuComboState — mutable state per combo (positions + accumulators)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-combo mutable state, lives in GPU storage buffer.
///
/// Contains up to 52 symbol positions (matching max watchlist size)
/// plus PESC cooldown arrays and result accumulators.
///
/// Accumulator fields use f64 (double) for precision over 10K+ trade accumulations.
///
/// Size:
/// 16 (header)
/// + 52*96 (positions)
/// + 5*52*4 (PESC + entry/exit cooldown maps)
/// + 16 + 1024*40 (trace control + ring)
/// + 56 (accumulators) + 8 (pad)
///   = 47,088 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuComboState {
    // Account state (f64 balance = 8 bytes, absorbs old _state_pad)
    pub balance: f64,
    pub num_open: u32,
    pub entries_this_bar: u32,

    // Positions (fixed-size array)
    pub positions: [GpuPosition; 52],

    // PESC state (per-symbol)
    pub pesc_close_time_sec: [u32; 52],
    pub pesc_close_type: [u32; 52],        // 0=none, 1=LONG, 2=SHORT
    pub pesc_close_reason: [u32; 52],      // 0=none, 1=signal_flip, 2=other
    pub last_entry_attempt_sec: [u32; 52], // per-symbol successful entry/add timestamp
    pub last_exit_attempt_sec: [u32; 52],  // per-symbol successful exit/partial timestamp

    // Optional GPU event trace (single-combo/symbol diagnostics)
    pub trace_enabled: u32,                           // 0=off, 1=on
    pub trace_symbol: u32,                            // symbol index, or GPU_TRACE_SYMBOL_ALL
    pub trace_count: u32, // number of valid entries in ring (<= GPU_TRACE_CAP)
    pub trace_head: u32,  // monotonic write cursor
    pub trace_events: [GpuTraceEvent; GPU_TRACE_CAP], // fixed ring buffer

    // Result accumulators (f64 for precision)
    pub total_pnl: f64,
    pub total_fees: f64,
    pub total_trades: u32,
    pub total_wins: u32,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub max_drawdown: f64,
    pub peak_equity: f64,
    pub kernel_cash: f64,
}

// Manual impls because bytemuck derive doesn't support [T; 52].
//
// SAFETY invariants for `Pod`/`Zeroable`:
// - `GpuComboState` is `#[repr(C)]` and contains only Pod field types
//   (primitives or arrays of `GpuPosition`/`GpuTraceEvent`, which are Pod).
// - Layout is validated by compile-time size/alignment checks below.
// - Zeroed bytes are valid for all fields (numeric zero values are legal sentinels).
//
// If you add/remove/retype fields, update the expected-size formula and tests in this module.
unsafe impl Pod for GpuComboState {}
unsafe impl Zeroable for GpuComboState {}

const GPU_COMBO_STATE_EXPECTED_LAYOUT_BYTES: usize = std::mem::size_of::<f64>()
    + std::mem::size_of::<u32>() * 2
    + std::mem::size_of::<[GpuPosition; 52]>()
    + std::mem::size_of::<[u32; 52]>() * 5
    + std::mem::size_of::<u32>() * 4
    + std::mem::size_of::<[GpuTraceEvent; GPU_TRACE_CAP]>()
    + std::mem::size_of::<f64>() * 6
    + std::mem::size_of::<u32>() * 2
    + std::mem::size_of::<f64>();

const _: () =
    assert!(std::mem::size_of::<GpuComboState>() == GPU_COMBO_STATE_EXPECTED_LAYOUT_BYTES);
const _: () = assert!(std::mem::size_of::<GpuComboState>() == 47088);
const _: () = assert!(std::mem::align_of::<GpuComboState>() == 8);
const _: () = assert!(std::mem::size_of::<GpuComboState>().is_multiple_of(16));

// ═══════════════════════════════════════════════════════════════════════════
// GpuResult — readback result per combo
// ═══════════════════════════════════════════════════════════════════════════

/// Compact result struct read back from GPU after sweep completes.
///
/// 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuResult {
    pub final_balance: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub total_trades: u32,
    pub total_wins: u32,
    pub total_losses: u32,
    pub _pad0: u32,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub max_drawdown_pct: f64,
}

const _: () = assert!(std::mem::size_of::<GpuResult>() == 64);

// ═══════════════════════════════════════════════════════════════════════════
// GpuParams — uniform parameters for the compute shader
// ═══════════════════════════════════════════════════════════════════════════

/// Global parameters passed as uniform to the compute shader.
///
/// 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub num_combos: u32,
    pub num_symbols: u32,
    pub num_bars: u32,
    pub btc_sym_idx: u32,  // u32::MAX when unavailable
    pub paxg_sym_idx: u32, // u32::MAX when unavailable
    pub chunk_start: u32,
    pub chunk_end: u32,
    pub initial_balance_bits: u32,   // f32 bits
    pub maker_fee_rate_bits: u32,    // f32 bits (from config, default 3.5 bps)
    pub taker_fee_rate_bits: u32,    // f32 bits (from config, default 3.5 bps)
    pub max_sub_per_bar: u32,        // 0 = no sub-bars (backwards compatible)
    pub trade_end_bar: u32,          // last bar index for result write-back (scoped trade range)
    pub debug_t_sec: u32,            // 0 = disabled; otherwise enable debug logs at this timestamp
    pub funding_enabled: u32,        // 1 = apply funding settlements in trade kernel
    pub entry_interval_sec: u32, // entry sub-bar interval in seconds (for signal-on-close alignment)
    pub signal_on_candle_close: u32, // 1 = evaluate sub-bar entries on candle close semantics
}

const _: () = assert!(std::mem::size_of::<GpuParams>() == 64);

// ═══════════════════════════════════════════════════════════════════════════
// Funding buffers (precomputed CPU hourly settlements, uploaded once)
// ═══════════════════════════════════════════════════════════════════════════

/// Span into the flat funding-rate buffer for one `(bar, symbol)` slot.
///
/// Layout key:
/// - index array shape: `[num_bars * num_symbols]`
/// - slot index: `bar_idx * num_symbols + sym_idx`
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFundingSpan {
    pub offset: u32,
    pub len: u32,
}

const _: () = assert!(std::mem::size_of::<GpuFundingSpan>() == 8);

// ═══════════════════════════════════════════════════════════════════════════
// Conversion helpers
// ═══════════════════════════════════════════════════════════════════════════

pub const GPU_EXIT_MASK_STOP_LOSS_ASE: u32 = 1 << 0;
pub const GPU_EXIT_MASK_STOP_LOSS_DASE: u32 = 1 << 1;
pub const GPU_EXIT_MASK_STOP_LOSS_SLB: u32 = 1 << 2;
pub const GPU_EXIT_MASK_STOP_LOSS_BASE: u32 = 1 << 3;
pub const GPU_EXIT_MASK_STOP_LOSS_BREAKEVEN: u32 = 1 << 4;
pub const GPU_EXIT_MASK_TRAILING_LOW_CONF_OVERRIDE: u32 = 1 << 5;
pub const GPU_EXIT_MASK_TRAILING_VOL_BUFFER: u32 = 1 << 6;
pub const GPU_EXIT_MASK_TRAILING_BASE: u32 = 1 << 7;
pub const GPU_EXIT_MASK_TAKE_PROFIT_PARTIAL: u32 = 1 << 8;
pub const GPU_EXIT_MASK_TAKE_PROFIT_FULL: u32 = 1 << 9;
pub const GPU_EXIT_MASK_ALL: u32 = GPU_EXIT_MASK_STOP_LOSS_ASE
    | GPU_EXIT_MASK_STOP_LOSS_DASE
    | GPU_EXIT_MASK_STOP_LOSS_SLB
    | GPU_EXIT_MASK_STOP_LOSS_BASE
    | GPU_EXIT_MASK_STOP_LOSS_BREAKEVEN
    | GPU_EXIT_MASK_TRAILING_LOW_CONF_OVERRIDE
    | GPU_EXIT_MASK_TRAILING_VOL_BUFFER
    | GPU_EXIT_MASK_TRAILING_BASE
    | GPU_EXIT_MASK_TAKE_PROFIT_PARTIAL
    | GPU_EXIT_MASK_TAKE_PROFIT_FULL;

pub const GPU_SMART_EXIT_MASK_TREND_BREAKDOWN: u32 = 1 << 0;
pub const GPU_SMART_EXIT_MASK_TREND_EXHAUSTION: u32 = 1 << 1;
pub const GPU_SMART_EXIT_MASK_EMA_MACRO_BREAKDOWN: u32 = 1 << 2;
pub const GPU_SMART_EXIT_MASK_STAGNATION: u32 = 1 << 3;
pub const GPU_SMART_EXIT_MASK_FUNDING_HEADWIND: u32 = 1 << 4;
pub const GPU_SMART_EXIT_MASK_TSME: u32 = 1 << 5;
pub const GPU_SMART_EXIT_MASK_MMDE: u32 = 1 << 6;
pub const GPU_SMART_EXIT_MASK_RSI_OVEREXTENSION: u32 = 1 << 7;
pub const GPU_SMART_EXIT_MASK_ALL: u32 = GPU_SMART_EXIT_MASK_TREND_BREAKDOWN
    | GPU_SMART_EXIT_MASK_TREND_EXHAUSTION
    | GPU_SMART_EXIT_MASK_EMA_MACRO_BREAKDOWN
    | GPU_SMART_EXIT_MASK_STAGNATION
    | GPU_SMART_EXIT_MASK_FUNDING_HEADWIND
    | GPU_SMART_EXIT_MASK_TSME
    | GPU_SMART_EXIT_MASK_MMDE
    | GPU_SMART_EXIT_MASK_RSI_OVEREXTENSION;

pub const GPU_SIGNAL_MODE_MASK_STANDARD_TREND: u32 = 1 << 0;
pub const GPU_SIGNAL_MODE_MASK_PULLBACK: u32 = 1 << 1;
pub const GPU_SIGNAL_MODE_MASK_SLOW_DRIFT: u32 = 1 << 2;
pub const GPU_SIGNAL_MODE_MASK_ALL: u32 = GPU_SIGNAL_MODE_MASK_STANDARD_TREND
    | GPU_SIGNAL_MODE_MASK_PULLBACK
    | GPU_SIGNAL_MODE_MASK_SLOW_DRIFT;

pub const GPU_SIGNAL_MODE_ID_STANDARD_TREND: u32 = 0;
pub const GPU_SIGNAL_MODE_ID_PULLBACK: u32 = 1;
pub const GPU_SIGNAL_MODE_ID_SLOW_DRIFT: u32 = 2;

pub const GPU_EXIT_ORDER_ID_STOP_LOSS_ASE: u32 = 0;
pub const GPU_EXIT_ORDER_ID_STOP_LOSS_DASE: u32 = 1;
pub const GPU_EXIT_ORDER_ID_STOP_LOSS_SLB: u32 = 2;
pub const GPU_EXIT_ORDER_ID_STOP_LOSS_BASE: u32 = 3;
pub const GPU_EXIT_ORDER_ID_STOP_LOSS_BREAKEVEN: u32 = 4;
pub const GPU_EXIT_ORDER_ID_TRAILING_LOW_CONF_OVERRIDE: u32 = 5;
pub const GPU_EXIT_ORDER_ID_TRAILING_VOL_BUFFER: u32 = 6;
pub const GPU_EXIT_ORDER_ID_TRAILING_BASE: u32 = 7;
pub const GPU_EXIT_ORDER_ID_TAKE_PROFIT_PARTIAL: u32 = 8;
pub const GPU_EXIT_ORDER_ID_TAKE_PROFIT_FULL: u32 = 9;
pub const GPU_EXIT_ORDER_ID_SMART_TREND_BREAKDOWN: u32 = 10;
pub const GPU_EXIT_ORDER_ID_SMART_TREND_EXHAUSTION: u32 = 11;
pub const GPU_EXIT_ORDER_ID_SMART_EMA_MACRO_BREAKDOWN: u32 = 12;
pub const GPU_EXIT_ORDER_ID_SMART_STAGNATION: u32 = 13;
pub const GPU_EXIT_ORDER_ID_SMART_FUNDING_HEADWIND: u32 = 14;
pub const GPU_EXIT_ORDER_ID_SMART_TSME: u32 = 15;
pub const GPU_EXIT_ORDER_ID_SMART_MMDE: u32 = 16;
pub const GPU_EXIT_ORDER_ID_SMART_RSI_OVEREXTENSION: u32 = 17;

/// Validate that an f64 value fits in f32 without becoming infinite.
/// Returns `Err` if a finite f64 overflows to infinity in f32.
fn checked_f32(name: &str, val: f64) -> Result<f32, String> {
    let f = val as f32;
    if !f.is_finite() && val.is_finite() {
        return Err(format!("{name}: value {val} overflows f32"));
    }
    Ok(f)
}

fn build_exit_behaviour_mask(plan: &bt_core::behaviour::ResolvedBehaviourPlan) -> u32 {
    let mut mask = GPU_EXIT_MASK_ALL;
    for item in &plan.exits.items {
        if item.enabled {
            continue;
        }
        match item.id.as_str() {
            "exit.stop_loss.ase" => mask &= !GPU_EXIT_MASK_STOP_LOSS_ASE,
            "exit.stop_loss.dase" => mask &= !GPU_EXIT_MASK_STOP_LOSS_DASE,
            "exit.stop_loss.slb" => mask &= !GPU_EXIT_MASK_STOP_LOSS_SLB,
            "exit.stop_loss.base" => mask &= !GPU_EXIT_MASK_STOP_LOSS_BASE,
            "exit.stop_loss.breakeven" => mask &= !GPU_EXIT_MASK_STOP_LOSS_BREAKEVEN,
            "exit.trailing.low_conf_override" => mask &= !GPU_EXIT_MASK_TRAILING_LOW_CONF_OVERRIDE,
            "exit.trailing.vol_buffer" => mask &= !GPU_EXIT_MASK_TRAILING_VOL_BUFFER,
            "exit.trailing.base" => mask &= !GPU_EXIT_MASK_TRAILING_BASE,
            "exit.take_profit.partial" => mask &= !GPU_EXIT_MASK_TAKE_PROFIT_PARTIAL,
            "exit.take_profit.full" => mask &= !GPU_EXIT_MASK_TAKE_PROFIT_FULL,
            _ => {}
        }
    }
    mask
}

fn build_smart_exit_behaviour_mask(plan: &bt_core::behaviour::ResolvedBehaviourPlan) -> u32 {
    let mut mask = GPU_SMART_EXIT_MASK_ALL;
    for item in &plan.exits.items {
        if item.enabled {
            continue;
        }
        match item.id.as_str() {
            "exit.smart.trend_breakdown" => mask &= !GPU_SMART_EXIT_MASK_TREND_BREAKDOWN,
            "exit.smart.trend_exhaustion" => mask &= !GPU_SMART_EXIT_MASK_TREND_EXHAUSTION,
            "exit.smart.ema_macro_breakdown" => mask &= !GPU_SMART_EXIT_MASK_EMA_MACRO_BREAKDOWN,
            "exit.smart.stagnation" => mask &= !GPU_SMART_EXIT_MASK_STAGNATION,
            "exit.smart.funding_headwind" => mask &= !GPU_SMART_EXIT_MASK_FUNDING_HEADWIND,
            "exit.smart.tsme" => mask &= !GPU_SMART_EXIT_MASK_TSME,
            "exit.smart.mmde" => mask &= !GPU_SMART_EXIT_MASK_MMDE,
            "exit.smart.rsi_overextension" => mask &= !GPU_SMART_EXIT_MASK_RSI_OVEREXTENSION,
            _ => {}
        }
    }
    mask
}

fn build_signal_mode_behaviour_mask(plan: &bt_core::behaviour::ResolvedBehaviourPlan) -> u32 {
    let mut mask = GPU_SIGNAL_MODE_MASK_ALL;
    for item in &plan.signal_modes.items {
        if item.enabled {
            continue;
        }
        match item.id.as_str() {
            "signal.mode.standard_trend" => mask &= !GPU_SIGNAL_MODE_MASK_STANDARD_TREND,
            "signal.mode.pullback" => mask &= !GPU_SIGNAL_MODE_MASK_PULLBACK,
            "signal.mode.slow_drift" => mask &= !GPU_SIGNAL_MODE_MASK_SLOW_DRIFT,
            _ => {}
        }
    }
    mask
}

fn encode_signal_mode_order(
    plan: &bt_core::behaviour::ResolvedBehaviourPlan,
) -> Result<[u32; 3], String> {
    let mut out = [GPU_SIGNAL_MODE_ID_STANDARD_TREND; 3];
    for (idx, id) in plan.signal_modes.ordered_ids().enumerate() {
        if idx >= out.len() {
            return Err(format!(
                "GPU signal mode order only supports 3 behaviours, got extra `{id}`"
            ));
        }
        out[idx] = match id {
            "signal.mode.standard_trend" => GPU_SIGNAL_MODE_ID_STANDARD_TREND,
            "signal.mode.pullback" => GPU_SIGNAL_MODE_ID_PULLBACK,
            "signal.mode.slow_drift" => GPU_SIGNAL_MODE_ID_SLOW_DRIFT,
            _ => {
                return Err(format!(
                    "unknown signal mode behaviour `{id}` in GPU order encoding"
                ))
            }
        };
    }
    Ok(out)
}

fn encode_exit_order(
    plan: &bt_core::behaviour::ResolvedBehaviourPlan,
) -> Result<[u32; 18], String> {
    let mut out = [GPU_EXIT_ORDER_ID_STOP_LOSS_ASE; 18];
    for (idx, id) in plan.exits.ordered_ids().enumerate() {
        if idx >= out.len() {
            return Err(format!(
                "GPU exit order only supports 18 behaviours, got extra `{id}`"
            ));
        }
        out[idx] = match id {
            "exit.stop_loss.ase" => GPU_EXIT_ORDER_ID_STOP_LOSS_ASE,
            "exit.stop_loss.dase" => GPU_EXIT_ORDER_ID_STOP_LOSS_DASE,
            "exit.stop_loss.slb" => GPU_EXIT_ORDER_ID_STOP_LOSS_SLB,
            "exit.stop_loss.base" => GPU_EXIT_ORDER_ID_STOP_LOSS_BASE,
            "exit.stop_loss.breakeven" => GPU_EXIT_ORDER_ID_STOP_LOSS_BREAKEVEN,
            "exit.trailing.low_conf_override" => GPU_EXIT_ORDER_ID_TRAILING_LOW_CONF_OVERRIDE,
            "exit.trailing.vol_buffer" => GPU_EXIT_ORDER_ID_TRAILING_VOL_BUFFER,
            "exit.trailing.base" => GPU_EXIT_ORDER_ID_TRAILING_BASE,
            "exit.take_profit.partial" => GPU_EXIT_ORDER_ID_TAKE_PROFIT_PARTIAL,
            "exit.take_profit.full" => GPU_EXIT_ORDER_ID_TAKE_PROFIT_FULL,
            "exit.smart.trend_breakdown" => GPU_EXIT_ORDER_ID_SMART_TREND_BREAKDOWN,
            "exit.smart.trend_exhaustion" => GPU_EXIT_ORDER_ID_SMART_TREND_EXHAUSTION,
            "exit.smart.ema_macro_breakdown" => GPU_EXIT_ORDER_ID_SMART_EMA_MACRO_BREAKDOWN,
            "exit.smart.stagnation" => GPU_EXIT_ORDER_ID_SMART_STAGNATION,
            "exit.smart.funding_headwind" => GPU_EXIT_ORDER_ID_SMART_FUNDING_HEADWIND,
            "exit.smart.tsme" => GPU_EXIT_ORDER_ID_SMART_TSME,
            "exit.smart.mmde" => GPU_EXIT_ORDER_ID_SMART_MMDE,
            "exit.smart.rsi_overextension" => GPU_EXIT_ORDER_ID_SMART_RSI_OVEREXTENSION,
            _ => {
                return Err(format!(
                    "unknown exit behaviour `{id}` in GPU order encoding"
                ))
            }
        };
    }
    Ok(out)
}

impl GpuComboConfig {
    /// Convert a `StrategyConfig` (f64) into a `GpuComboConfig` (f32).
    ///
    /// Returns `Err` if any price, size, or leverage value overflows f32.
    pub fn from_strategy_config(cfg: &bt_core::config::StrategyConfig) -> Result<Self, String> {
        let resolved = bt_core::execution_contract::resolve_gpu_execution_config(cfg, None)
            .map_err(|err| format!("execution contract resolution failed: {err}"))?;
        let exit_behaviour_mask = build_exit_behaviour_mask(&resolved.behaviour_plan);
        let smart_exit_behaviour_mask = build_smart_exit_behaviour_mask(&resolved.behaviour_plan);
        let signal_mode_behaviour_mask = build_signal_mode_behaviour_mask(&resolved.behaviour_plan);
        let signal_mode_order = encode_signal_mode_order(&resolved.behaviour_plan)?;
        let exit_order = encode_exit_order(&resolved.behaviour_plan)?;
        let cfg = &resolved.effective_cfg;
        let tc = &cfg.trade;
        let fc = &cfg.filters;
        let mc = &cfg.market_regime;
        let et = &cfg.thresholds.entry;
        let rt = &cfg.thresholds.ranging;
        let at = &cfg.thresholds.anomaly;
        let tp = &cfg.thresholds.tp_and_momentum;
        macro_rules! checked_f32_field {
            ($expr:expr) => {
                checked_f32(stringify!($expr), $expr)?
            };
        }

        Ok(Self {
            allocation_pct: checked_f32("allocation_pct", tc.allocation_pct)?,
            sl_atr_mult: checked_f32("sl_atr_mult", tc.sl_atr_mult)?,
            tp_atr_mult: checked_f32("tp_atr_mult", tc.tp_atr_mult)?,
            leverage: checked_f32("leverage", tc.leverage)?,

            enable_reef_filter: tc.enable_reef_filter as u32,
            reef_long_rsi_block_gt: checked_f32_field!(tc.reef_long_rsi_block_gt),
            reef_short_rsi_block_lt: checked_f32_field!(tc.reef_short_rsi_block_lt),
            reef_adx_threshold: checked_f32_field!(tc.reef_adx_threshold),
            reef_long_rsi_extreme_gt: checked_f32_field!(tc.reef_long_rsi_extreme_gt),
            reef_short_rsi_extreme_lt: checked_f32_field!(tc.reef_short_rsi_extreme_lt),

            smart_exit_behaviour_mask,
            leverage_low: checked_f32("leverage_low", tc.leverage_low)?,
            leverage_medium: checked_f32("leverage_medium", tc.leverage_medium)?,
            leverage_high: checked_f32("leverage_high", tc.leverage_high)?,
            exit_behaviour_mask,
            trailing_rsi_floor_default: 0.5,

            slippage_bps: checked_f32("slippage_bps", tc.slippage_bps)?,
            min_notional_usd: checked_f32("min_notional_usd", tc.min_notional_usd)?,
            bump_to_min_notional: tc.bump_to_min_notional as u32,
            max_total_margin_pct: checked_f32("max_total_margin_pct", tc.max_total_margin_pct)?,
            trailing_rsi_floor_trending: 0.7,
            trailing_vbts_bb_threshold: 1.2,

            enable_dynamic_sizing: tc.enable_dynamic_sizing as u32,
            confidence_mult_high: checked_f32("confidence_mult_high", tc.confidence_mult_high)?,
            confidence_mult_medium: checked_f32(
                "confidence_mult_medium",
                tc.confidence_mult_medium,
            )?,
            confidence_mult_low: checked_f32("confidence_mult_low", tc.confidence_mult_low)?,
            adx_sizing_min_mult: checked_f32_field!(tc.adx_sizing_min_mult),
            adx_sizing_full_adx: checked_f32_field!(tc.adx_sizing_full_adx),
            vol_baseline_pct: checked_f32_field!(tc.vol_baseline_pct),
            vol_scalar_min: checked_f32_field!(tc.vol_scalar_min),
            vol_scalar_max: checked_f32_field!(tc.vol_scalar_max),
            trailing_vbts_mult: 1.25,

            enable_pyramiding: tc.enable_pyramiding as u32,
            max_adds_per_symbol: tc.max_adds_per_symbol as u32,
            add_fraction_of_base_margin: checked_f32(
                "add_fraction_of_base_margin",
                tc.add_fraction_of_base_margin,
            )?,
            add_cooldown_minutes: tc.add_cooldown_minutes as u32,
            add_min_profit_atr: checked_f32_field!(tc.add_min_profit_atr),
            add_min_confidence: tc.add_min_confidence as u32,
            entry_min_confidence: tc.entry_min_confidence as u32,
            trailing_high_profit_atr: 2.0,

            enable_partial_tp: tc.enable_partial_tp as u32,
            tp_partial_pct: checked_f32("tp_partial_pct", tc.tp_partial_pct)?,
            tp_partial_min_notional_usd: checked_f32(
                "tp_partial_min_notional_usd",
                tc.tp_partial_min_notional_usd,
            )?,
            trailing_start_atr: checked_f32_field!(tc.trailing_start_atr),
            trailing_distance_atr: checked_f32_field!(tc.trailing_distance_atr),
            tp_partial_atr_mult: checked_f32_field!(tc.tp_partial_atr_mult),

            enable_ssf_filter: tc.enable_ssf_filter as u32,
            enable_breakeven_stop: tc.enable_breakeven_stop as u32,
            breakeven_start_atr: checked_f32_field!(tc.breakeven_start_atr),
            breakeven_buffer_atr: checked_f32_field!(tc.breakeven_buffer_atr),
            trailing_start_atr_low_conf: checked_f32_field!(tc.trailing_start_atr_low_conf),
            trailing_distance_atr_low_conf: checked_f32_field!(tc.trailing_distance_atr_low_conf),
            smart_exit_adx_exhaustion_lt: checked_f32_field!(tc.smart_exit_adx_exhaustion_lt),
            smart_exit_adx_exhaustion_lt_low_conf: checked_f32_field!(
                tc.smart_exit_adx_exhaustion_lt_low_conf
            ),

            enable_rsi_overextension_exit: tc.enable_rsi_overextension_exit as u32,
            rsi_exit_profit_atr_switch: checked_f32_field!(tc.rsi_exit_profit_atr_switch),
            rsi_exit_ub_lo_profit: checked_f32_field!(tc.rsi_exit_ub_lo_profit),
            rsi_exit_ub_hi_profit: checked_f32_field!(tc.rsi_exit_ub_hi_profit),
            rsi_exit_lb_lo_profit: checked_f32_field!(tc.rsi_exit_lb_lo_profit),
            rsi_exit_lb_hi_profit: checked_f32_field!(tc.rsi_exit_lb_hi_profit),
            rsi_exit_ub_lo_profit_low_conf: checked_f32_field!(tc.rsi_exit_ub_lo_profit_low_conf),
            rsi_exit_ub_hi_profit_low_conf: checked_f32_field!(tc.rsi_exit_ub_hi_profit_low_conf),
            rsi_exit_lb_lo_profit_low_conf: checked_f32_field!(tc.rsi_exit_lb_lo_profit_low_conf),
            rsi_exit_lb_hi_profit_low_conf: checked_f32_field!(tc.rsi_exit_lb_hi_profit_low_conf),

            reentry_cooldown_minutes: tc.reentry_cooldown_minutes as u32,
            reentry_cooldown_min_mins: tc.reentry_cooldown_min_mins as u32,
            reentry_cooldown_max_mins: tc.reentry_cooldown_max_mins as u32,
            trailing_tighten_default: 0.5,

            enable_vol_buffered_trailing: tc.enable_vol_buffered_trailing as u32,
            tsme_min_profit_atr: checked_f32_field!(tc.tsme_min_profit_atr),
            tsme_require_adx_slope_negative: tc.tsme_require_adx_slope_negative as u32,
            trailing_tighten_tspv: 0.75,

            min_atr_pct: checked_f32_field!(tc.min_atr_pct),
            reverse_entry_signal: tc.reverse_entry_signal as u32,
            block_exits_on_extreme_dev: tc.block_exits_on_extreme_dev as u32,
            glitch_price_dev_pct: checked_f32_field!(tc.glitch_price_dev_pct),
            glitch_atr_mult: checked_f32_field!(tc.glitch_atr_mult),
            trailing_weak_trend_mult: 0.7,

            max_open_positions: tc.max_open_positions as u32,
            max_entry_orders_per_loop: tc.max_entry_orders_per_loop as u32,
            enable_slow_drift_entries: et.enable_slow_drift_entries as u32,
            slow_drift_require_macd_sign: et.slow_drift_require_macd_sign as u32,

            enable_ranging_filter: fc.enable_ranging_filter as u32,
            enable_anomaly_filter: fc.enable_anomaly_filter as u32,
            enable_extension_filter: fc.enable_extension_filter as u32,
            require_adx_rising: fc.require_adx_rising as u32,
            adx_rising_saturation: checked_f32_field!(fc.adx_rising_saturation),
            require_volume_confirmation: fc.require_volume_confirmation as u32,
            vol_confirm_include_prev: fc.vol_confirm_include_prev as u32,
            use_stoch_rsi_filter: fc.use_stoch_rsi_filter as u32,
            require_btc_alignment: fc.require_btc_alignment as u32,
            require_macro_alignment: fc.require_macro_alignment as u32,

            enable_regime_filter: mc.enable_regime_filter as u32,
            enable_auto_reverse: mc.enable_auto_reverse as u32,
            auto_reverse_breadth_low: checked_f32_field!(mc.auto_reverse_breadth_low),
            auto_reverse_breadth_high: checked_f32_field!(mc.auto_reverse_breadth_high),
            breadth_block_short_above: checked_f32_field!(mc.breadth_block_short_above),
            breadth_block_long_below: checked_f32_field!(mc.breadth_block_long_below),
            min_adx: checked_f32_field!(et.min_adx),
            high_conf_volume_mult: checked_f32_field!(et.high_conf_volume_mult),
            btc_adx_override: checked_f32_field!(et.btc_adx_override),
            max_dist_ema_fast: checked_f32_field!(et.max_dist_ema_fast),
            ave_atr_ratio_gt: checked_f32_field!(et.ave_atr_ratio_gt),
            ave_adx_mult: checked_f32_field!(et.ave_adx_mult),
            dre_min_adx: checked_f32_field!(et.min_adx),
            dre_max_adx: checked_f32_field!(tp.adx_strong_gt),
            dre_long_rsi_limit_low: checked_f32_field!(tp.rsi_long_weak),
            dre_long_rsi_limit_high: checked_f32_field!(tp.rsi_long_strong),
            dre_short_rsi_limit_low: checked_f32_field!(tp.rsi_short_weak),
            dre_short_rsi_limit_high: checked_f32_field!(tp.rsi_short_strong),
            macd_mode: match et.macd_hist_entry_mode {
                bt_core::config::MacdMode::Accel => 0,
                bt_core::config::MacdMode::Sign => 1,
                bt_core::config::MacdMode::None => 2,
            },
            pullback_min_adx: checked_f32_field!(et.pullback_min_adx),
            pullback_rsi_long_min: checked_f32_field!(et.pullback_rsi_long_min),
            pullback_rsi_short_max: checked_f32_field!(et.pullback_rsi_short_max),
            pullback_require_macd_sign: et.pullback_require_macd_sign as u32,
            pullback_confidence: et.pullback_confidence as u32,
            slow_drift_min_slope_pct: checked_f32_field!(et.slow_drift_min_slope_pct),
            slow_drift_min_adx: checked_f32_field!(et.slow_drift_min_adx),
            slow_drift_rsi_long_min: checked_f32_field!(et.slow_drift_rsi_long_min),
            slow_drift_rsi_short_max: checked_f32_field!(et.slow_drift_rsi_short_max),
            ranging_adx_lt: checked_f32_field!(rt.adx_below),
            ranging_bb_width_ratio_lt: checked_f32_field!(rt.bb_width_ratio_below),
            anomaly_bb_width_ratio_gt: DEFAULT_ANOMALY_BB_WIDTH_RATIO_GT,
            slow_drift_ranging_slope_override: checked_f32_field!(et.slow_drift_min_slope_pct),
            snapshot_offset: 0,
            breadth_offset: 0,

            tp_strong_adx_gt: checked_f32_field!(tp.adx_strong_gt),
            tp_weak_adx_lt: checked_f32_field!(tp.adx_weak_lt),

            // Decision codegen fields (AQC-1250)
            enable_pullback_entries: et.enable_pullback_entries as u32,
            anomaly_price_change_pct: checked_f32_field!(at.price_change_pct_gt),
            anomaly_ema_dev_pct: checked_f32_field!(at.ema_fast_dev_pct_gt),
            ranging_rsi_low: checked_f32_field!(rt.rsi_low),
            ranging_rsi_high: checked_f32_field!(rt.rsi_high),
            ranging_min_signals: rt.min_signals as u32,
            stoch_rsi_block_long_gt: checked_f32_field!(
                cfg.thresholds.stoch_rsi.block_long_if_k_gt
            ),
            stoch_rsi_block_short_lt: checked_f32_field!(
                cfg.thresholds.stoch_rsi.block_short_if_k_lt
            ),
            ave_enabled: et.ave_enabled as u32,
            tp_mult_strong: checked_f32_field!(tp.tp_mult_strong),
            tp_mult_weak: checked_f32_field!(tp.tp_mult_weak),
            entry_cooldown_s: tc.entry_cooldown_s as u32,
            exit_cooldown_s: tc.exit_cooldown_s as u32,
            signal_mode_behaviour_mask,
            signal_mode_order_0: signal_mode_order[0],
            signal_mode_order_1: signal_mode_order[1],
            signal_mode_order_2: signal_mode_order[2],
            exit_order_0: exit_order[0],
            exit_order_1: exit_order[1],
            exit_order_2: exit_order[2],
            exit_order_3: exit_order[3],
            exit_order_4: exit_order[4],
            exit_order_5: exit_order[5],
            exit_order_6: exit_order[6],
            exit_order_7: exit_order[7],
            exit_order_8: exit_order[8],
            exit_order_9: exit_order[9],
            exit_order_10: exit_order[10],
            exit_order_11: exit_order[11],
            exit_order_12: exit_order[12],
            exit_order_13: exit_order[13],
            exit_order_14: exit_order[14],
            exit_order_15: exit_order[15],
            exit_order_16: exit_order[16],
            exit_order_17: exit_order[17],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GpuComboConfig, GpuComboState, GpuIndicatorConfig, GPU_COMBO_STATE_EXPECTED_LAYOUT_BYTES,
        GPU_EXIT_MASK_STOP_LOSS_BREAKEVEN, GPU_EXIT_MASK_TAKE_PROFIT_PARTIAL,
        GPU_EXIT_MASK_TRAILING_LOW_CONF_OVERRIDE, GPU_EXIT_MASK_TRAILING_VOL_BUFFER,
        GPU_EXIT_ORDER_ID_SMART_TREND_BREAKDOWN, GPU_EXIT_ORDER_ID_STOP_LOSS_ASE,
        GPU_EXIT_ORDER_ID_STOP_LOSS_BASE, GPU_EXIT_ORDER_ID_TAKE_PROFIT_FULL,
        GPU_EXIT_ORDER_ID_TAKE_PROFIT_PARTIAL, GPU_SIGNAL_MODE_ID_PULLBACK,
        GPU_SIGNAL_MODE_ID_SLOW_DRIFT, GPU_SIGNAL_MODE_ID_STANDARD_TREND,
        GPU_SIGNAL_MODE_MASK_STANDARD_TREND, GPU_SMART_EXIT_MASK_ALL,
    };
    use bt_core::config::{
        BehaviourGroupConfig, BehaviourProfileConfig, PipelineConfig, PipelineProfileConfig,
        RuntimeConfig, StrategyConfig,
    };
    use bytemuck::{checked, Zeroable};
    use std::collections::BTreeMap;

    #[test]
    fn test_gpu_indicator_config_uses_threshold_ave_window() {
        let mut cfg = StrategyConfig::default();
        cfg.indicators.ave_avg_atr_window = 3;
        cfg.thresholds.entry.ave_avg_atr_window = 17;

        let out = GpuIndicatorConfig::from_strategy_config(&cfg, 200);
        assert_eq!(out.avg_atr_window, 17);
    }

    #[test]
    fn test_gpu_indicator_config_clamps_large_usize_windows() {
        let mut cfg = StrategyConfig::default();
        cfg.indicators.ema_fast_window = usize::MAX;
        cfg.thresholds.entry.ave_avg_atr_window = usize::MAX;

        let out = GpuIndicatorConfig::from_strategy_config(&cfg, usize::MAX);
        assert_eq!(out.ema_fast_window, u32::MAX);
        assert_eq!(out.avg_atr_window, u32::MAX);
        assert_eq!(out.lookback, u32::MAX);
    }

    /// AQC-1250: round-trip test — StrategyConfig → GpuComboConfig for the 11 new
    /// decision codegen fields.
    #[test]
    fn test_codegen_fields_roundtrip() {
        let mut cfg = StrategyConfig::default();

        // Set non-default values for all 11 new fields
        cfg.thresholds.entry.enable_pullback_entries = true;
        cfg.thresholds.anomaly.price_change_pct_gt = 0.25;
        cfg.thresholds.anomaly.ema_fast_dev_pct_gt = 0.75;
        cfg.thresholds.ranging.rsi_low = 42.0;
        cfg.thresholds.ranging.rsi_high = 58.0;
        cfg.thresholds.ranging.min_signals = 3;
        cfg.thresholds.stoch_rsi.block_long_if_k_gt = 0.90;
        cfg.thresholds.stoch_rsi.block_short_if_k_lt = 0.10;
        cfg.thresholds.entry.ave_enabled = false;
        cfg.thresholds.tp_and_momentum.tp_mult_strong = 8.5;
        cfg.thresholds.tp_and_momentum.tp_mult_weak = 2.5;

        let gpu = GpuComboConfig::from_strategy_config(&cfg).unwrap();

        assert_eq!(gpu.enable_pullback_entries, 1);
        assert!((gpu.anomaly_price_change_pct - 0.25).abs() < f32::EPSILON);
        assert!((gpu.anomaly_ema_dev_pct - 0.75).abs() < f32::EPSILON);
        assert!((gpu.ranging_rsi_low - 42.0).abs() < f32::EPSILON);
        assert!((gpu.ranging_rsi_high - 58.0).abs() < f32::EPSILON);
        assert_eq!(gpu.ranging_min_signals, 3);
        assert!((gpu.stoch_rsi_block_long_gt - 0.90).abs() < f32::EPSILON);
        assert!((gpu.stoch_rsi_block_short_lt - 0.10).abs() < f32::EPSILON);
        assert_eq!(gpu.ave_enabled, 0);
        assert!((gpu.tp_mult_strong - 8.5).abs() < f32::EPSILON);
        assert!((gpu.tp_mult_weak - 2.5).abs() < f32::EPSILON);
    }

    /// AQC-1250: default StrategyConfig maps disabled features to 0 in GPU struct.
    #[test]
    fn test_codegen_fields_defaults() {
        let cfg = StrategyConfig::default();
        let gpu = GpuComboConfig::from_strategy_config(&cfg).unwrap();

        // enable_pullback_entries defaults to false → 0
        assert_eq!(gpu.enable_pullback_entries, 0);
        // ave_enabled defaults to true → 1
        assert_eq!(gpu.ave_enabled, 1);
        // Default anomaly thresholds
        assert!((gpu.anomaly_price_change_pct - 0.10).abs() < f32::EPSILON);
        assert!((gpu.anomaly_ema_dev_pct - 0.50).abs() < f32::EPSILON);
        // Default ranging RSI zone
        assert!((gpu.ranging_rsi_low - 47.0).abs() < f32::EPSILON);
        assert!((gpu.ranging_rsi_high - 53.0).abs() < f32::EPSILON);
        assert_eq!(gpu.ranging_min_signals, 2);
        // Default stochRSI thresholds
        assert!((gpu.stoch_rsi_block_long_gt - 0.85).abs() < f32::EPSILON);
        assert!((gpu.stoch_rsi_block_short_lt - 0.15).abs() < f32::EPSILON);
        // Default TP multipliers
        assert!((gpu.tp_mult_strong - 7.0).abs() < f32::EPSILON);
        assert!((gpu.tp_mult_weak - 3.0).abs() < f32::EPSILON);
        // Entry/exit cooldowns are propagated
        assert_eq!(gpu.entry_cooldown_s, cfg.trade.entry_cooldown_s as u32);
        assert_eq!(gpu.exit_cooldown_s, cfg.trade.exit_cooldown_s as u32);
    }

    /// H11: verify that extreme f64 values that overflow f32 are caught.
    #[test]
    fn test_checked_f32_overflow_detected() {
        let mut cfg = StrategyConfig::default();
        cfg.trade.leverage = f64::MAX; // way beyond f32::MAX
        let err = GpuComboConfig::from_strategy_config(&cfg);
        assert!(
            err.is_err(),
            "expected overflow error for f64::MAX leverage"
        );
        assert!(
            err.unwrap_err().contains("leverage"),
            "error should mention the field name"
        );
    }

    #[test]
    fn test_checked_f32_overflow_detected_for_previously_unchecked_field() {
        let mut cfg = StrategyConfig::default();
        cfg.trade.trailing_start_atr = f64::MAX;
        let err = GpuComboConfig::from_strategy_config(&cfg);
        assert!(
            err.is_err(),
            "expected overflow error for trailing_start_atr"
        );
        assert!(
            err.unwrap_err().contains("tc.trailing_start_atr"),
            "error should mention the overflowing field path"
        );
    }

    /// H11: normal values still convert successfully.
    #[test]
    fn test_checked_f32_normal_values_pass() {
        let cfg = StrategyConfig::default();
        assert!(
            GpuComboConfig::from_strategy_config(&cfg).is_ok(),
            "default config should not overflow"
        );
    }

    #[test]
    fn test_gpu_combo_config_uses_resolved_execution_contract() {
        let profile = "gpu_modular";
        let cfg = StrategyConfig {
            runtime: RuntimeConfig {
                profile: profile.to_string(),
                ..RuntimeConfig::default()
            },
            pipeline: PipelineConfig {
                default_profile: "production".to_string(),
                profiles: BTreeMap::from([(
                    profile.to_string(),
                    PipelineProfileConfig {
                        behaviours: BehaviourProfileConfig {
                            entry_sizing: BehaviourGroupConfig {
                                disabled: vec![
                                    "entry.sizing.dynamic".to_string(),
                                    "entry.sizing.confidence_multiplier".to_string(),
                                    "entry.sizing.adx_multiplier".to_string(),
                                    "entry.sizing.volatility_scalar".to_string(),
                                    "entry.sizing.min_notional_bump".to_string(),
                                ],
                                ..BehaviourGroupConfig::default()
                            },
                            entry_progression: BehaviourGroupConfig {
                                disabled: vec![
                                    "entry.progression.pyramiding".to_string(),
                                    "entry.progression.add_cooldown".to_string(),
                                ],
                                ..BehaviourGroupConfig::default()
                            },
                            risk: BehaviourGroupConfig {
                                disabled: vec![
                                    "risk.entry_cooldown".to_string(),
                                    "risk.exit_cooldown".to_string(),
                                    "risk.pesc".to_string(),
                                ],
                                ..BehaviourGroupConfig::default()
                            },
                            ..BehaviourProfileConfig::default()
                        },
                        ..PipelineProfileConfig::default()
                    },
                )]),
            },
            ..StrategyConfig::default()
        };

        let resolved = bt_core::execution_contract::resolve_execution_config(&cfg, None)
            .expect("execution contract must resolve");
        let gpu = GpuComboConfig::from_strategy_config(&resolved.effective_cfg)
            .expect("resolved config must lower into GPU combo config");

        assert_eq!(gpu.enable_dynamic_sizing, 0);
        assert!((gpu.confidence_mult_high - 1.0).abs() < f32::EPSILON);
        assert!((gpu.confidence_mult_medium - 1.0).abs() < f32::EPSILON);
        assert!((gpu.confidence_mult_low - 1.0).abs() < f32::EPSILON);
        assert!((gpu.adx_sizing_min_mult - 1.0).abs() < f32::EPSILON);
        assert!((gpu.adx_sizing_full_adx - 0.0).abs() < f32::EPSILON);
        assert!((gpu.vol_baseline_pct - 0.0).abs() < f32::EPSILON);
        assert!((gpu.vol_scalar_min - 1.0).abs() < f32::EPSILON);
        assert!((gpu.vol_scalar_max - 1.0).abs() < f32::EPSILON);
        assert_eq!(gpu.bump_to_min_notional, 0);
        assert_eq!(gpu.enable_pyramiding, 0);
        assert_eq!(gpu.add_cooldown_minutes, 0);
        assert_eq!(gpu.entry_cooldown_s, 0);
        assert_eq!(gpu.exit_cooldown_s, 0);
        assert_eq!(gpu.reentry_cooldown_minutes, 0);
    }

    #[test]
    fn test_gpu_combo_config_lowers_builtin_parity_exit_isolation() {
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = "parity_exit_isolation".to_string();

        let gpu = GpuComboConfig::from_strategy_config(&cfg)
            .expect("parity_exit_isolation should lower into GPU config");

        assert_eq!(gpu.enable_breakeven_stop, 0);
        assert_eq!(gpu.enable_partial_tp, 0);
        assert_eq!(gpu.enable_vol_buffered_trailing, 0);
        assert_eq!(gpu.trailing_start_atr_low_conf, 0.0);
        assert_eq!(gpu.trailing_distance_atr_low_conf, 0.0);
        assert_eq!(
            gpu.exit_behaviour_mask & GPU_EXIT_MASK_STOP_LOSS_BREAKEVEN,
            0
        );
        assert_eq!(
            gpu.exit_behaviour_mask & GPU_EXIT_MASK_TRAILING_LOW_CONF_OVERRIDE,
            0
        );
        assert_eq!(
            gpu.exit_behaviour_mask & GPU_EXIT_MASK_TRAILING_VOL_BUFFER,
            0
        );
        assert_eq!(
            gpu.exit_behaviour_mask & GPU_EXIT_MASK_TAKE_PROFIT_PARTIAL,
            0
        );
        assert_eq!(gpu.smart_exit_behaviour_mask & GPU_SMART_EXIT_MASK_ALL, 0);
    }

    #[test]
    fn test_gpu_combo_config_lowers_signal_mode_reorder_and_disable() {
        let profile = "gpu_signal_order";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    signal_modes: BehaviourGroupConfig {
                        order: vec![
                            "signal.mode.slow_drift".to_string(),
                            "signal.mode.standard_trend".to_string(),
                            "signal.mode.pullback".to_string(),
                        ],
                        disabled: vec!["signal.mode.standard_trend".to_string()],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let gpu = GpuComboConfig::from_strategy_config(&cfg)
            .expect("signal mode reorder/disable should lower into GPU config");
        assert_eq!(
            gpu.signal_mode_behaviour_mask & GPU_SIGNAL_MODE_MASK_STANDARD_TREND,
            0
        );
        assert_eq!(gpu.signal_mode_order_0, GPU_SIGNAL_MODE_ID_SLOW_DRIFT);
        assert_eq!(gpu.signal_mode_order_1, GPU_SIGNAL_MODE_ID_STANDARD_TREND);
        assert_eq!(gpu.signal_mode_order_2, GPU_SIGNAL_MODE_ID_PULLBACK);
    }

    #[test]
    fn test_gpu_combo_config_accepts_supported_exit_reorder() {
        let profile = "gpu_custom_exit_order";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    exits: BehaviourGroupConfig {
                        order: vec![
                            "exit.stop_loss.ase".to_string(),
                            "exit.stop_loss.dase".to_string(),
                            "exit.stop_loss.slb".to_string(),
                            "exit.stop_loss.base".to_string(),
                            "exit.stop_loss.breakeven".to_string(),
                            "exit.trailing.low_conf_override".to_string(),
                            "exit.trailing.vol_buffer".to_string(),
                            "exit.trailing.base".to_string(),
                            "exit.smart.trend_breakdown".to_string(),
                            "exit.take_profit.full".to_string(),
                            "exit.take_profit.partial".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let gpu = GpuComboConfig::from_strategy_config(&cfg)
            .expect("supported exit reorder should lower into GPU config");
        assert_eq!(gpu.exit_order_8, GPU_EXIT_ORDER_ID_SMART_TREND_BREAKDOWN);
        assert_eq!(gpu.exit_order_9, GPU_EXIT_ORDER_ID_TAKE_PROFIT_FULL);
        assert_eq!(gpu.exit_order_10, GPU_EXIT_ORDER_ID_TAKE_PROFIT_PARTIAL);
    }

    #[test]
    fn test_gpu_combo_config_accepts_stop_loss_internal_reorder() {
        let profile = "gpu_bad_stop_order";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    exits: BehaviourGroupConfig {
                        order: vec![
                            "exit.stop_loss.base".to_string(),
                            "exit.stop_loss.ase".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let gpu = GpuComboConfig::from_strategy_config(&cfg)
            .expect("stop-loss internal reorder should lower into GPU config");
        assert_eq!(gpu.exit_order_0, GPU_EXIT_ORDER_ID_STOP_LOSS_BASE);
        assert_eq!(gpu.exit_order_1, GPU_EXIT_ORDER_ID_STOP_LOSS_ASE);
    }

    #[test]
    fn test_gpu_combo_state_layout_invariants() {
        assert_eq!(
            std::mem::size_of::<GpuComboState>(),
            GPU_COMBO_STATE_EXPECTED_LAYOUT_BYTES
        );
        assert_eq!(std::mem::size_of::<GpuComboState>(), 47088);
        assert_eq!(std::mem::align_of::<GpuComboState>(), 8);
    }

    #[test]
    fn test_gpu_combo_state_checked_cast_slice_roundtrip() {
        let states = [GpuComboState::zeroed(), GpuComboState::zeroed()];
        let bytes: &[u8] = checked::cast_slice(&states);
        assert_eq!(
            bytes.len(),
            states.len() * std::mem::size_of::<GpuComboState>()
        );

        let roundtrip: &[GpuComboState] = checked::cast_slice(bytes);
        assert_eq!(roundtrip.len(), states.len());
        assert_eq!(roundtrip[0].num_open, 0);
        assert_eq!(roundtrip[1].trace_count, 0);
    }
}
