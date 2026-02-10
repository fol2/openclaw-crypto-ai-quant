//! GPU buffer structs with bytemuck Pod/Zeroable for direct GPU upload.
//!
//! All structs are `#[repr(C)]` and 16-byte aligned for CUDA compatibility.
//! f64 values from the CPU side are cast to f32 for GPU computation.

use bytemuck::{Pod, Zeroable};

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
/// 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRawCandle {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
    pub t_sec: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuRawCandle>() == 32);

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

impl GpuIndicatorConfig {
    /// Build from a StrategyConfig's indicator section.
    pub fn from_strategy_config(cfg: &bt_core::config::StrategyConfig, lookback: usize) -> Self {
        let ic = &cfg.indicators;
        Self {
            ema_fast_window: ic.ema_fast_window as u32,
            ema_slow_window: ic.ema_slow_window as u32,
            ema_macro_window: ic.ema_macro_window as u32,
            adx_window: ic.adx_window as u32,
            bb_window: ic.bb_window as u32,
            bb_width_avg_window: ic.bb_width_avg_window as u32,
            atr_window: ic.atr_window as u32,
            rsi_window: ic.rsi_window as u32,
            vol_sma_window: ic.vol_sma_window as u32,
            vol_trend_window: ic.vol_trend_window as u32,
            stoch_rsi_window: ic.stoch_rsi_window as u32,
            stoch_rsi_smooth1: ic.stoch_rsi_smooth1 as u32,
            stoch_rsi_smooth2: ic.stoch_rsi_smooth2 as u32,
            avg_atr_window: ic.ave_avg_atr_window as u32,
            slow_drift_slope_window: cfg.thresholds.entry.slow_drift_slope_window as u32,
            lookback: lookback as u32,
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
/// 512 bytes (128 × f32), aligned to 16.
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

    // Dynamic leverage [10-15]
    pub enable_dynamic_leverage: u32,
    pub leverage_low: f32,
    pub leverage_medium: f32,
    pub leverage_high: f32,
    pub leverage_max_cap: f32,
    pub _pad0: u32,

    // Execution [16-21]
    pub slippage_bps: f32,
    pub min_notional_usd: f32,
    pub bump_to_min_notional: u32,
    pub max_total_margin_pct: f32,
    pub _pad1: u32,
    pub _pad2: u32,

    // Dynamic sizing [22-29]
    pub enable_dynamic_sizing: u32,
    pub confidence_mult_high: f32,
    pub confidence_mult_medium: f32,
    pub confidence_mult_low: f32,
    pub adx_sizing_min_mult: f32,
    pub adx_sizing_full_adx: f32,
    pub vol_baseline_pct: f32,
    pub vol_scalar_min: f32,

    // [30-31]
    pub vol_scalar_max: f32,
    pub _pad3: u32,

    // Pyramiding [32-39]
    pub enable_pyramiding: u32,
    pub max_adds_per_symbol: u32,
    pub add_fraction_of_base_margin: f32,
    pub add_cooldown_minutes: u32,
    pub add_min_profit_atr: f32,
    pub add_min_confidence: u32, // 0=Low, 1=Medium, 2=High
    pub entry_min_confidence: u32,
    pub _pad4: u32,

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

    // Reentry cooldown [64-67]
    pub reentry_cooldown_minutes: u32,
    pub reentry_cooldown_min_mins: u32,
    pub reentry_cooldown_max_mins: u32,
    pub _pad6: u32,

    // Volatility-buffered trailing + TSME [68-71]
    pub enable_vol_buffered_trailing: u32,
    pub tsme_min_profit_atr: f32,
    pub tsme_require_adx_slope_negative: u32,
    pub _pad7: u32,

    // ATR floor / signal reversal / glitch [72-77]
    pub min_atr_pct: f32,
    pub reverse_entry_signal: u32,
    pub block_exits_on_extreme_dev: u32,
    pub glitch_price_dev_pct: f32,
    pub glitch_atr_mult: f32,
    pub _pad8: u32,

    // Rate limits [78-81]
    pub max_open_positions: u32,
    pub max_entry_orders_per_loop: u32,
    pub _pad9: u32,
    pub _pad10: u32,

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
    pub snapshot_offset: u32,  // byte offset into concatenated snapshots array (in elements, not bytes)
    pub breadth_offset: u32,   // offset into concatenated breadth/btc_bullish arrays (in elements)

    // TP momentum [126-127]
    pub tp_strong_adx_gt: f32,
    pub tp_weak_adx_lt: f32,
}

const _: () = assert!(std::mem::size_of::<GpuComboConfig>() == 512);

// ═══════════════════════════════════════════════════════════════════════════
// GpuPosition — per-symbol position state
// ═══════════════════════════════════════════════════════════════════════════

/// Per-symbol position (embedded in GpuComboState).
///
/// 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPosition {
    pub active: u32,       // 0=empty, 1=LONG, 2=SHORT
    pub entry_price: f32,
    pub size: f32,
    pub confidence: u32,   // 0=Low, 1=Medium, 2=High
    pub entry_atr: f32,
    pub entry_adx_threshold: f32,
    pub trailing_sl: f32,  // 0.0 = not set
    pub leverage: f32,
    pub margin_used: f32,
    pub adds_count: u32,
    pub tp1_taken: u32,
    pub open_time_sec: u32,
    pub last_add_time_sec: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<GpuPosition>() == 64);

// ═══════════════════════════════════════════════════════════════════════════
// GpuComboState — mutable state per combo (positions + accumulators)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-combo mutable state, lives in GPU storage buffer.
///
/// Contains up to 52 symbol positions (matching max watchlist size)
/// plus PESC cooldown arrays and result accumulators.
///
/// Size: 64 + 52*64 + 52*4 + 52*4 + 52*4 + 32 = 3,608 bytes → round to 3,616 (16-aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuComboState {
    // Account state
    pub balance: f32,
    pub num_open: u32,
    pub entries_this_bar: u32,
    pub _state_pad: u32,

    // Positions (fixed-size array)
    pub positions: [GpuPosition; 52],

    // PESC state (per-symbol)
    pub pesc_close_time_sec: [u32; 52],
    pub pesc_close_type: [u32; 52],  // 0=none, 1=LONG, 2=SHORT
    pub pesc_close_reason: [u32; 52], // 0=none, 1=signal_flip, 2=other

    // Result accumulators
    pub total_pnl: f32,
    pub total_fees: f32,
    pub total_trades: u32,
    pub total_wins: u32,
    pub gross_profit: f32,
    pub gross_loss: f32,
    pub max_drawdown: f32,
    pub peak_equity: f32,
}

// Manual impls because bytemuck derive doesn't support [T; 52]
// SAFETY: All fields are f32/u32/arrays of Pod types, all valid as zeroed bytes
unsafe impl Pod for GpuComboState {}
unsafe impl Zeroable for GpuComboState {}

const _: () = assert!(std::mem::size_of::<GpuComboState>() % 16 == 0);

// ═══════════════════════════════════════════════════════════════════════════
// GpuResult — readback result per combo
// ═══════════════════════════════════════════════════════════════════════════

/// Compact result struct read back from GPU after sweep completes.
///
/// 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuResult {
    pub final_balance: f32,
    pub total_pnl: f32,
    pub total_fees: f32,
    pub total_trades: u32,
    pub total_wins: u32,
    pub total_losses: u32,
    pub gross_profit: f32,
    pub gross_loss: f32,
    pub max_drawdown_pct: f32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<GpuResult>() == 48);

// ═══════════════════════════════════════════════════════════════════════════
// GpuParams — uniform parameters for the compute shader
// ═══════════════════════════════════════════════════════════════════════════

/// Global parameters passed as uniform to the compute shader.
///
/// 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub num_combos: u32,
    pub num_symbols: u32,
    pub num_bars: u32,
    pub chunk_start: u32,
    pub chunk_end: u32,
    pub initial_balance_bits: u32, // f32 bits
    pub fee_rate_bits: u32,        // f32 bits (0.00035)
    pub max_sub_per_bar: u32,      // 0 = no sub-bars (backwards compatible)
    pub trade_end_bar: u32,        // last bar index for result write-back (scoped trade range)
}

const _: () = assert!(std::mem::size_of::<GpuParams>() == 36);

// ═══════════════════════════════════════════════════════════════════════════
// Conversion helpers
// ═══════════════════════════════════════════════════════════════════════════

impl GpuComboConfig {
    /// Convert a `StrategyConfig` (f64) into a `GpuComboConfig` (f32).
    pub fn from_strategy_config(cfg: &bt_core::config::StrategyConfig) -> Self {
        let tc = &cfg.trade;
        let fc = &cfg.filters;
        let mc = &cfg.market_regime;
        let et = &cfg.thresholds.entry;
        let rt = &cfg.thresholds.ranging;
        let _at = &cfg.thresholds.anomaly;
        let tp = &cfg.thresholds.tp_and_momentum;

        Self {
            allocation_pct: tc.allocation_pct as f32,
            sl_atr_mult: tc.sl_atr_mult as f32,
            tp_atr_mult: tc.tp_atr_mult as f32,
            leverage: tc.leverage as f32,

            enable_reef_filter: tc.enable_reef_filter as u32,
            reef_long_rsi_block_gt: tc.reef_long_rsi_block_gt as f32,
            reef_short_rsi_block_lt: tc.reef_short_rsi_block_lt as f32,
            reef_adx_threshold: tc.reef_adx_threshold as f32,
            reef_long_rsi_extreme_gt: tc.reef_long_rsi_extreme_gt as f32,
            reef_short_rsi_extreme_lt: tc.reef_short_rsi_extreme_lt as f32,

            enable_dynamic_leverage: tc.enable_dynamic_leverage as u32,
            leverage_low: tc.leverage_low as f32,
            leverage_medium: tc.leverage_medium as f32,
            leverage_high: tc.leverage_high as f32,
            leverage_max_cap: tc.leverage_max_cap as f32,
            _pad0: 0,

            slippage_bps: tc.slippage_bps as f32,
            min_notional_usd: tc.min_notional_usd as f32,
            bump_to_min_notional: tc.bump_to_min_notional as u32,
            max_total_margin_pct: tc.max_total_margin_pct as f32,
            _pad1: 0,
            _pad2: 0,

            enable_dynamic_sizing: tc.enable_dynamic_sizing as u32,
            confidence_mult_high: tc.confidence_mult_high as f32,
            confidence_mult_medium: tc.confidence_mult_medium as f32,
            confidence_mult_low: tc.confidence_mult_low as f32,
            adx_sizing_min_mult: tc.adx_sizing_min_mult as f32,
            adx_sizing_full_adx: tc.adx_sizing_full_adx as f32,
            vol_baseline_pct: tc.vol_baseline_pct as f32,
            vol_scalar_min: tc.vol_scalar_min as f32,

            vol_scalar_max: tc.vol_scalar_max as f32,
            _pad3: 0,

            enable_pyramiding: tc.enable_pyramiding as u32,
            max_adds_per_symbol: tc.max_adds_per_symbol as u32,
            add_fraction_of_base_margin: tc.add_fraction_of_base_margin as f32,
            add_cooldown_minutes: tc.add_cooldown_minutes as u32,
            add_min_profit_atr: tc.add_min_profit_atr as f32,
            add_min_confidence: tc.add_min_confidence as u32,
            entry_min_confidence: tc.entry_min_confidence as u32,
            _pad4: 0,

            enable_partial_tp: tc.enable_partial_tp as u32,
            tp_partial_pct: tc.tp_partial_pct as f32,
            tp_partial_min_notional_usd: tc.tp_partial_min_notional_usd as f32,
            trailing_start_atr: tc.trailing_start_atr as f32,
            trailing_distance_atr: tc.trailing_distance_atr as f32,
            tp_partial_atr_mult: tc.tp_partial_atr_mult as f32,

            enable_ssf_filter: tc.enable_ssf_filter as u32,
            enable_breakeven_stop: tc.enable_breakeven_stop as u32,
            breakeven_start_atr: tc.breakeven_start_atr as f32,
            breakeven_buffer_atr: tc.breakeven_buffer_atr as f32,

            trailing_start_atr_low_conf: tc.trailing_start_atr_low_conf as f32,
            trailing_distance_atr_low_conf: tc.trailing_distance_atr_low_conf as f32,
            smart_exit_adx_exhaustion_lt: tc.smart_exit_adx_exhaustion_lt as f32,
            smart_exit_adx_exhaustion_lt_low_conf: tc.smart_exit_adx_exhaustion_lt_low_conf as f32,

            enable_rsi_overextension_exit: tc.enable_rsi_overextension_exit as u32,
            rsi_exit_profit_atr_switch: tc.rsi_exit_profit_atr_switch as f32,
            rsi_exit_ub_lo_profit: tc.rsi_exit_ub_lo_profit as f32,
            rsi_exit_ub_hi_profit: tc.rsi_exit_ub_hi_profit as f32,
            rsi_exit_lb_lo_profit: tc.rsi_exit_lb_lo_profit as f32,
            rsi_exit_lb_hi_profit: tc.rsi_exit_lb_hi_profit as f32,
            rsi_exit_ub_lo_profit_low_conf: tc.rsi_exit_ub_lo_profit_low_conf as f32,
            rsi_exit_ub_hi_profit_low_conf: tc.rsi_exit_ub_hi_profit_low_conf as f32,
            rsi_exit_lb_lo_profit_low_conf: tc.rsi_exit_lb_lo_profit_low_conf as f32,
            rsi_exit_lb_hi_profit_low_conf: tc.rsi_exit_lb_hi_profit_low_conf as f32,

            reentry_cooldown_minutes: tc.reentry_cooldown_minutes as u32,
            reentry_cooldown_min_mins: tc.reentry_cooldown_min_mins as u32,
            reentry_cooldown_max_mins: tc.reentry_cooldown_max_mins as u32,
            _pad6: 0,

            enable_vol_buffered_trailing: tc.enable_vol_buffered_trailing as u32,
            tsme_min_profit_atr: tc.tsme_min_profit_atr as f32,
            tsme_require_adx_slope_negative: tc.tsme_require_adx_slope_negative as u32,
            _pad7: 0,

            min_atr_pct: tc.min_atr_pct as f32,
            reverse_entry_signal: tc.reverse_entry_signal as u32,
            block_exits_on_extreme_dev: tc.block_exits_on_extreme_dev as u32,
            glitch_price_dev_pct: tc.glitch_price_dev_pct as f32,
            glitch_atr_mult: tc.glitch_atr_mult as f32,
            _pad8: 0,

            max_open_positions: tc.max_open_positions as u32,
            max_entry_orders_per_loop: tc.max_entry_orders_per_loop as u32,
            _pad9: 0,
            _pad10: 0,

            enable_ranging_filter: fc.enable_ranging_filter as u32,
            enable_anomaly_filter: fc.enable_anomaly_filter as u32,
            enable_extension_filter: fc.enable_extension_filter as u32,
            require_adx_rising: fc.require_adx_rising as u32,
            adx_rising_saturation: fc.adx_rising_saturation as f32,
            require_volume_confirmation: fc.require_volume_confirmation as u32,
            vol_confirm_include_prev: fc.vol_confirm_include_prev as u32,
            use_stoch_rsi_filter: fc.use_stoch_rsi_filter as u32,
            require_btc_alignment: fc.require_btc_alignment as u32,
            require_macro_alignment: fc.require_macro_alignment as u32,

            enable_regime_filter: mc.enable_regime_filter as u32,
            enable_auto_reverse: mc.enable_auto_reverse as u32,
            auto_reverse_breadth_low: mc.auto_reverse_breadth_low as f32,
            auto_reverse_breadth_high: mc.auto_reverse_breadth_high as f32,
            breadth_block_short_above: mc.breadth_block_short_above as f32,
            breadth_block_long_below: mc.breadth_block_long_below as f32,

            min_adx: et.min_adx as f32,
            high_conf_volume_mult: et.high_conf_volume_mult as f32,
            btc_adx_override: et.btc_adx_override as f32,
            max_dist_ema_fast: et.max_dist_ema_fast as f32,
            ave_atr_ratio_gt: et.ave_atr_ratio_gt as f32,
            ave_adx_mult: et.ave_adx_mult as f32,
            dre_min_adx: et.min_adx as f32,
            dre_max_adx: tp.adx_strong_gt as f32,
            dre_long_rsi_limit_low: tp.rsi_long_weak as f32,
            dre_long_rsi_limit_high: tp.rsi_long_strong as f32,
            dre_short_rsi_limit_low: tp.rsi_short_weak as f32,
            dre_short_rsi_limit_high: tp.rsi_short_strong as f32,
            macd_mode: match et.macd_hist_entry_mode {
                bt_core::config::MacdMode::Accel => 0,
                bt_core::config::MacdMode::Sign => 1,
                bt_core::config::MacdMode::None => 2,
            },
            pullback_min_adx: et.pullback_min_adx as f32,
            pullback_rsi_long_min: et.pullback_rsi_long_min as f32,
            pullback_rsi_short_max: et.pullback_rsi_short_max as f32,
            pullback_require_macd_sign: et.pullback_require_macd_sign as u32,
            pullback_confidence: et.pullback_confidence as u32,
            slow_drift_min_slope_pct: et.slow_drift_min_slope_pct as f32,
            slow_drift_min_adx: et.slow_drift_min_adx as f32,
            slow_drift_rsi_long_min: et.slow_drift_rsi_long_min as f32,
            slow_drift_rsi_short_max: et.slow_drift_rsi_short_max as f32,

            ranging_adx_lt: rt.adx_below as f32,
            ranging_bb_width_ratio_lt: rt.bb_width_ratio_below as f32,
            anomaly_bb_width_ratio_gt: 3.0, // hardcoded anomaly threshold (not in sweep)
            slow_drift_ranging_slope_override: et.slow_drift_min_slope_pct as f32,
            snapshot_offset: 0,
            breadth_offset: 0,

            tp_strong_adx_gt: tp.adx_strong_gt as f32,
            tp_weak_adx_lt: tp.adx_weak_lt as f32,
        }
    }
}
