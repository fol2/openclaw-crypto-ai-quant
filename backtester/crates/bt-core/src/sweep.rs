//! Parallel parameter sweep engine using rayon.
//!
//! Generates a cartesian product of config overrides from a [`SweepSpec`],
//! runs each combination through `engine::run_simulation` in parallel via
//! rayon, and returns sorted results.

use std::sync::Arc;

use rayon::prelude::*;
use serde::Deserialize;

use crate::candle::{CandleData, FundingRateData};
use crate::config::{Confidence, StrategyConfig};
use crate::engine;
use crate::report::{self, SimReport};

// ---------------------------------------------------------------------------
// Sweep specification (loaded from YAML)
// ---------------------------------------------------------------------------

/// A single axis in the parameter sweep.
#[derive(Debug, Clone, Deserialize)]
pub struct SweepAxis {
    /// Dot-separated config path, e.g. "trade.sl_atr_mult"
    pub path: String,
    /// Values to test along this axis.
    pub values: Vec<f64>,
}

/// Full sweep specification (loaded from YAML).
#[derive(Debug, Clone, Deserialize)]
pub struct SweepSpec {
    pub axes: Vec<SweepAxis>,
    #[serde(default = "default_initial_balance")]
    pub initial_balance: f64,
    #[serde(default = "default_lookback")]
    pub lookback: usize,
}

fn default_initial_balance() -> f64 {
    10000.0
}
fn default_lookback() -> usize {
    200
}

// ---------------------------------------------------------------------------
// Sweep result
// ---------------------------------------------------------------------------

/// Result of one sweep configuration.
#[derive(Debug, Clone)]
pub struct SweepResult {
    pub config_id: String,
    pub report: SimReport,
    pub overrides: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// Config generation (cartesian product)
// ---------------------------------------------------------------------------

/// Generate all combinations (cartesian product) from the sweep axes.
fn generate_combinations(axes: &[SweepAxis]) -> Vec<Vec<(String, f64)>> {
    if axes.is_empty() {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let sub = generate_combinations(&axes[1..]);
    for val in &axes[0].values {
        for combo in &sub {
            let mut new_combo = vec![(axes[0].path.clone(), *val)];
            new_combo.extend(combo.iter().cloned());
            result.push(new_combo);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// apply_override — set a config field by dot-path
// ---------------------------------------------------------------------------

/// Apply a set of overrides to a cloned config.
fn apply_overrides(base: &StrategyConfig, overrides: &[(String, f64)]) -> StrategyConfig {
    let mut cfg = base.clone();
    for (path, value) in overrides {
        apply_one(&mut cfg, path, *value);
    }
    cfg
}

/// Set a single f64 config value by dot-separated path.
///
/// Supports all common sweep parameters. Integer fields (usize) are cast
/// from f64. Unknown paths are logged to stderr.
/// Public wrapper for `apply_one` used by bt-gpu crate.
pub fn apply_one_pub(cfg: &mut StrategyConfig, path: &str, value: f64) {
    apply_one(cfg, path, value);
}

fn apply_one(cfg: &mut StrategyConfig, path: &str, value: f64) {
    match path {
        // === Trade config ===
        "trade.sl_atr_mult" => cfg.trade.sl_atr_mult = value,
        "trade.tp_atr_mult" => cfg.trade.tp_atr_mult = value,
        "trade.trailing_start_atr" => cfg.trade.trailing_start_atr = value,
        "trade.trailing_distance_atr" => cfg.trade.trailing_distance_atr = value,
        "trade.allocation_pct" => cfg.trade.allocation_pct = value,
        "trade.leverage" => cfg.trade.leverage = value,
        "trade.max_open_positions" => cfg.trade.max_open_positions = value as usize,
        "trade.min_atr_pct" => cfg.trade.min_atr_pct = value,
        "trade.slippage_bps" => cfg.trade.slippage_bps = value,
        "trade.min_notional_usd" => cfg.trade.min_notional_usd = value,
        "trade.bump_to_min_notional" => cfg.trade.bump_to_min_notional = value != 0.0,
        "trade.max_total_margin_pct" => cfg.trade.max_total_margin_pct = value,
        "trade.reentry_cooldown_minutes" => cfg.trade.reentry_cooldown_minutes = value as usize,
        "trade.reentry_cooldown_min_mins" => cfg.trade.reentry_cooldown_min_mins = value as usize,
        "trade.reentry_cooldown_max_mins" => cfg.trade.reentry_cooldown_max_mins = value as usize,
        "trade.breakeven_start_atr" => cfg.trade.breakeven_start_atr = value,
        "trade.breakeven_buffer_atr" => cfg.trade.breakeven_buffer_atr = value,
        "trade.confidence_mult_low" => cfg.trade.confidence_mult_low = value,
        "trade.confidence_mult_medium" => cfg.trade.confidence_mult_medium = value,
        "trade.confidence_mult_high" => cfg.trade.confidence_mult_high = value,
        "trade.tsme_min_profit_atr" => cfg.trade.tsme_min_profit_atr = value,
        "trade.rsi_exit_profit_atr_switch" => cfg.trade.rsi_exit_profit_atr_switch = value,
        "trade.rsi_exit_ub_lo_profit" => cfg.trade.rsi_exit_ub_lo_profit = value,
        "trade.rsi_exit_ub_hi_profit" => cfg.trade.rsi_exit_ub_hi_profit = value,
        "trade.rsi_exit_lb_lo_profit" => cfg.trade.rsi_exit_lb_lo_profit = value,
        "trade.rsi_exit_lb_hi_profit" => cfg.trade.rsi_exit_lb_hi_profit = value,
        "trade.rsi_exit_ub_lo_profit_low_conf" => cfg.trade.rsi_exit_ub_lo_profit_low_conf = value,
        "trade.rsi_exit_ub_hi_profit_low_conf" => cfg.trade.rsi_exit_ub_hi_profit_low_conf = value,
        "trade.rsi_exit_lb_lo_profit_low_conf" => cfg.trade.rsi_exit_lb_lo_profit_low_conf = value,
        "trade.rsi_exit_lb_hi_profit_low_conf" => cfg.trade.rsi_exit_lb_hi_profit_low_conf = value,
        "trade.tp_partial_pct" => cfg.trade.tp_partial_pct = value,
        "trade.tp_partial_atr_mult" => cfg.trade.tp_partial_atr_mult = value,
        "trade.add_min_profit_atr" => cfg.trade.add_min_profit_atr = value,
        "trade.add_fraction_of_base_margin" => cfg.trade.add_fraction_of_base_margin = value,
        "trade.max_adds_per_symbol" => cfg.trade.max_adds_per_symbol = value as usize,
        "trade.add_cooldown_minutes" => cfg.trade.add_cooldown_minutes = value as usize,
        "trade.leverage_low" => cfg.trade.leverage_low = value,
        "trade.leverage_medium" => cfg.trade.leverage_medium = value,
        "trade.leverage_high" => cfg.trade.leverage_high = value,
        "trade.leverage_max_cap" => cfg.trade.leverage_max_cap = value,
        "trade.adx_sizing_min_mult" => cfg.trade.adx_sizing_min_mult = value,
        "trade.adx_sizing_full_adx" => cfg.trade.adx_sizing_full_adx = value,
        "trade.vol_baseline_pct" => cfg.trade.vol_baseline_pct = value,
        "trade.vol_scalar_min" => cfg.trade.vol_scalar_min = value,
        "trade.vol_scalar_max" => cfg.trade.vol_scalar_max = value,
        "trade.entry_cooldown_s" => cfg.trade.entry_cooldown_s = value as usize,
        "trade.exit_cooldown_s" => cfg.trade.exit_cooldown_s = value as usize,
        "trade.max_entry_orders_per_loop" => cfg.trade.max_entry_orders_per_loop = value as usize,
        "trade.glitch_price_dev_pct" => cfg.trade.glitch_price_dev_pct = value,
        "trade.glitch_atr_mult" => cfg.trade.glitch_atr_mult = value,
        "trade.reef_long_rsi_block_gt" => cfg.trade.reef_long_rsi_block_gt = value,
        "trade.reef_short_rsi_block_lt" => cfg.trade.reef_short_rsi_block_lt = value,
        "trade.reef_adx_threshold" => cfg.trade.reef_adx_threshold = value,
        "trade.reef_long_rsi_extreme_gt" => cfg.trade.reef_long_rsi_extreme_gt = value,
        "trade.reef_short_rsi_extreme_lt" => cfg.trade.reef_short_rsi_extreme_lt = value,
        "trade.trailing_start_atr_low_conf" => cfg.trade.trailing_start_atr_low_conf = value,
        "trade.trailing_distance_atr_low_conf" => cfg.trade.trailing_distance_atr_low_conf = value,
        "trade.smart_exit_adx_exhaustion_lt" => cfg.trade.smart_exit_adx_exhaustion_lt = value,
        "trade.smart_exit_adx_exhaustion_lt_low_conf" => cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = value,

        // === Confidence enums (0=low, 1=medium, 2=high) ===
        "trade.entry_min_confidence" => {
            cfg.trade.entry_min_confidence = match value as u8 {
                0 => Confidence::Low,
                1 => Confidence::Medium,
                _ => Confidence::High,
            };
        }
        "trade.add_min_confidence" => {
            cfg.trade.add_min_confidence = match value as u8 {
                0 => Confidence::Low,
                1 => Confidence::Medium,
                _ => Confidence::High,
            };
        }
        "thresholds.entry.pullback_confidence" => {
            cfg.thresholds.entry.pullback_confidence = match value as u8 {
                0 => Confidence::Low,
                1 => Confidence::Medium,
                _ => Confidence::High,
            };
        }

        // === Thresholds — entry ===
        "thresholds.entry.min_adx" => cfg.thresholds.entry.min_adx = value,
        "thresholds.entry.high_conf_volume_mult" => cfg.thresholds.entry.high_conf_volume_mult = value,
        "thresholds.entry.btc_adx_override" => cfg.thresholds.entry.btc_adx_override = value,
        "thresholds.entry.max_dist_ema_fast" => cfg.thresholds.entry.max_dist_ema_fast = value,
        "thresholds.entry.ave_atr_ratio_gt" => cfg.thresholds.entry.ave_atr_ratio_gt = value,
        "thresholds.entry.ave_adx_mult" => cfg.thresholds.entry.ave_adx_mult = value,
        "thresholds.entry.ave_avg_atr_window" => cfg.thresholds.entry.ave_avg_atr_window = value as usize,
        "thresholds.entry.pullback_min_adx" => cfg.thresholds.entry.pullback_min_adx = value,
        "thresholds.entry.pullback_rsi_long_min" => cfg.thresholds.entry.pullback_rsi_long_min = value,
        "thresholds.entry.pullback_rsi_short_max" => cfg.thresholds.entry.pullback_rsi_short_max = value,
        "thresholds.entry.slow_drift_slope_window" => cfg.thresholds.entry.slow_drift_slope_window = value as usize,
        "thresholds.entry.slow_drift_min_slope_pct" => cfg.thresholds.entry.slow_drift_min_slope_pct = value,
        "thresholds.entry.slow_drift_min_adx" => cfg.thresholds.entry.slow_drift_min_adx = value,
        "thresholds.entry.slow_drift_rsi_long_min" => cfg.thresholds.entry.slow_drift_rsi_long_min = value,
        "thresholds.entry.slow_drift_rsi_short_max" => cfg.thresholds.entry.slow_drift_rsi_short_max = value,

        // === Thresholds — ranging ===
        "thresholds.ranging.min_signals" => cfg.thresholds.ranging.min_signals = value as usize,
        "thresholds.ranging.adx_below" => cfg.thresholds.ranging.adx_below = value,
        "thresholds.ranging.bb_width_ratio_below" => cfg.thresholds.ranging.bb_width_ratio_below = value,
        "thresholds.ranging.rsi_low" => cfg.thresholds.ranging.rsi_low = value,
        "thresholds.ranging.rsi_high" => cfg.thresholds.ranging.rsi_high = value,

        // === Thresholds — anomaly ===
        "thresholds.anomaly.price_change_pct_gt" => cfg.thresholds.anomaly.price_change_pct_gt = value,
        "thresholds.anomaly.ema_fast_dev_pct_gt" => cfg.thresholds.anomaly.ema_fast_dev_pct_gt = value,

        // === Thresholds — tp_and_momentum ===
        "thresholds.tp_and_momentum.adx_strong_gt" => cfg.thresholds.tp_and_momentum.adx_strong_gt = value,
        "thresholds.tp_and_momentum.adx_weak_lt" => cfg.thresholds.tp_and_momentum.adx_weak_lt = value,
        "thresholds.tp_and_momentum.tp_mult_strong" => cfg.thresholds.tp_and_momentum.tp_mult_strong = value,
        "thresholds.tp_and_momentum.tp_mult_weak" => cfg.thresholds.tp_and_momentum.tp_mult_weak = value,
        "thresholds.tp_and_momentum.rsi_long_strong" => cfg.thresholds.tp_and_momentum.rsi_long_strong = value,
        "thresholds.tp_and_momentum.rsi_long_weak" => cfg.thresholds.tp_and_momentum.rsi_long_weak = value,
        "thresholds.tp_and_momentum.rsi_short_strong" => cfg.thresholds.tp_and_momentum.rsi_short_strong = value,
        "thresholds.tp_and_momentum.rsi_short_weak" => cfg.thresholds.tp_and_momentum.rsi_short_weak = value,

        // === Thresholds — stoch_rsi ===
        "thresholds.stoch_rsi.block_long_if_k_gt" => cfg.thresholds.stoch_rsi.block_long_if_k_gt = value,
        "thresholds.stoch_rsi.block_short_if_k_lt" => cfg.thresholds.stoch_rsi.block_short_if_k_lt = value,

        // === Boolean toggles (0.0 = false, else true) ===
        "trade.enable_dynamic_sizing" => cfg.trade.enable_dynamic_sizing = value != 0.0,
        "trade.enable_dynamic_leverage" => cfg.trade.enable_dynamic_leverage = value != 0.0,
        "trade.enable_pyramiding" => cfg.trade.enable_pyramiding = value != 0.0,
        "trade.enable_partial_tp" => cfg.trade.enable_partial_tp = value != 0.0,
        "trade.enable_ssf_filter" => cfg.trade.enable_ssf_filter = value != 0.0,
        "trade.enable_reef_filter" => cfg.trade.enable_reef_filter = value != 0.0,
        "trade.enable_breakeven_stop" => cfg.trade.enable_breakeven_stop = value != 0.0,
        "trade.enable_rsi_overextension_exit" => cfg.trade.enable_rsi_overextension_exit = value != 0.0,
        "trade.enable_vol_buffered_trailing" => cfg.trade.enable_vol_buffered_trailing = value != 0.0,
        "trade.tsme_require_adx_slope_negative" => cfg.trade.tsme_require_adx_slope_negative = value != 0.0,
        "trade.reverse_entry_signal" => cfg.trade.reverse_entry_signal = value != 0.0,
        "trade.block_exits_on_extreme_dev" => cfg.trade.block_exits_on_extreme_dev = value != 0.0,
        "filters.vol_confirm_include_prev" => cfg.filters.vol_confirm_include_prev = value != 0.0,

        // === Filters ===
        "filters.enable_ranging_filter" => cfg.filters.enable_ranging_filter = value != 0.0,
        "filters.enable_anomaly_filter" => cfg.filters.enable_anomaly_filter = value != 0.0,
        "filters.enable_extension_filter" => cfg.filters.enable_extension_filter = value != 0.0,
        "filters.require_adx_rising" => cfg.filters.require_adx_rising = value != 0.0,
        "filters.adx_rising_saturation" => cfg.filters.adx_rising_saturation = value,
        "filters.require_volume_confirmation" => cfg.filters.require_volume_confirmation = value != 0.0,
        "filters.use_stoch_rsi_filter" => cfg.filters.use_stoch_rsi_filter = value != 0.0,
        "filters.require_btc_alignment" => cfg.filters.require_btc_alignment = value != 0.0,
        "filters.require_macro_alignment" => cfg.filters.require_macro_alignment = value != 0.0,

        // === Entry toggles ===
        "thresholds.entry.enable_pullback_entries" => cfg.thresholds.entry.enable_pullback_entries = value != 0.0,
        "thresholds.entry.enable_slow_drift_entries" => cfg.thresholds.entry.enable_slow_drift_entries = value != 0.0,
        "thresholds.entry.ave_enabled" => cfg.thresholds.entry.ave_enabled = value != 0.0,
        "thresholds.entry.pullback_require_macd_sign" => cfg.thresholds.entry.pullback_require_macd_sign = value != 0.0,
        "thresholds.entry.slow_drift_require_macd_sign" => cfg.thresholds.entry.slow_drift_require_macd_sign = value != 0.0,

        // === Market regime toggles ===
        "market_regime.enable_regime_filter" => cfg.market_regime.enable_regime_filter = value != 0.0,
        "market_regime.enable_auto_reverse" => cfg.market_regime.enable_auto_reverse = value != 0.0,

        // === Indicators ===
        "indicators.ema_slow_window" => cfg.indicators.ema_slow_window = value as usize,
        "indicators.ema_fast_window" => cfg.indicators.ema_fast_window = value as usize,
        "indicators.ema_macro_window" => cfg.indicators.ema_macro_window = value as usize,
        "indicators.adx_window" => cfg.indicators.adx_window = value as usize,
        "indicators.bb_window" => cfg.indicators.bb_window = value as usize,
        "indicators.bb_width_avg_window" => cfg.indicators.bb_width_avg_window = value as usize,
        "indicators.atr_window" => cfg.indicators.atr_window = value as usize,
        "indicators.rsi_window" => cfg.indicators.rsi_window = value as usize,
        "indicators.vol_sma_window" => cfg.indicators.vol_sma_window = value as usize,
        "indicators.vol_trend_window" => cfg.indicators.vol_trend_window = value as usize,
        "indicators.stoch_rsi_window" => cfg.indicators.stoch_rsi_window = value as usize,
        "indicators.stoch_rsi_smooth1" => cfg.indicators.stoch_rsi_smooth1 = value as usize,
        "indicators.stoch_rsi_smooth2" => cfg.indicators.stoch_rsi_smooth2 = value as usize,
        "indicators.ave_avg_atr_window" => cfg.indicators.ave_avg_atr_window = value as usize,

        // === Market regime ===
        "market_regime.breadth_block_short_above" => cfg.market_regime.breadth_block_short_above = value,
        "market_regime.breadth_block_long_below" => cfg.market_regime.breadth_block_long_below = value,
        "market_regime.auto_reverse_breadth_low" => cfg.market_regime.auto_reverse_breadth_low = value,
        "market_regime.auto_reverse_breadth_high" => cfg.market_regime.auto_reverse_breadth_high = value,

        _ => eprintln!("[sweep] Unknown override path: {path}"),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run a parameter sweep: generate all config combinations, run each in parallel.
///
/// Results are sorted by `total_pnl` descending (best first).
///
/// `from_ts`/`to_ts` are forwarded to the simulation engine to restrict trading to a
/// specific time window.
pub fn run_sweep(
    base_cfg: &StrategyConfig,
    spec: &SweepSpec,
    candles: &CandleData,
    exit_candles: Option<&CandleData>,
    entry_candles: Option<&CandleData>,
    funding_rates: Option<&FundingRateData>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Vec<SweepResult> {
    let combos = generate_combinations(&spec.axes);
    let total = combos.len();
    eprintln!(
        "[sweep] Generated {total} config combinations across {} axes",
        spec.axes.len()
    );

    let candles = Arc::new(candles.clone());
    let exit_candles_arc = exit_candles.map(|ec| Arc::new(ec.clone()));
    let entry_candles_arc = entry_candles.map(|ec| Arc::new(ec.clone()));
    let funding_rates_arc = funding_rates.map(|fr| Arc::new(fr.clone()));
    let initial_balance = spec.initial_balance;
    let lookback = spec.lookback;

    let mut results: Vec<SweepResult> = combos
        .par_iter()
        .enumerate()
        .map(|(i, overrides)| {
            let cfg = apply_overrides(base_cfg, overrides);
            let config_id = format!("sweep_{:04}", i);

            let sim = engine::run_simulation(
                &candles,
                &cfg,
                initial_balance,
                lookback,
                exit_candles_arc.as_deref(),
                entry_candles_arc.as_deref(),
                funding_rates_arc.as_deref(),
                None, // sweeps always start clean (no init-state)
                from_ts,
                to_ts,
            );

            let rpt = report::build_report(
                &sim.trades,
                &sim.signals,
                &sim.equity_curve,
                &sim.gate_stats,
                initial_balance,
                sim.final_balance,
                &config_id,
                false, // don't embed per-trade detail in sweep
                false, // don't embed equity curve in sweep
            );

            SweepResult {
                config_id,
                report: rpt,
                overrides: overrides.clone(),
            }
        })
        .collect();

    // Sort by total PnL descending (best first).
    results.sort_by(|a, b| {
        b.report
            .total_pnl
            .partial_cmp(&a.report.total_pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Load a sweep specification from a YAML file.
pub fn load_sweep_spec(path: &str) -> Result<SweepSpec, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read sweep spec: {e}"))?;
    serde_yaml::from_str(&raw).map_err(|e| format!("Failed to parse sweep spec: {e}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_combinations_empty() {
        let combos = generate_combinations(&[]);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_generate_combinations_single_axis() {
        let axes = vec![SweepAxis {
            path: "trade.sl_atr_mult".to_string(),
            values: vec![1.0, 2.0, 3.0],
        }];
        let combos = generate_combinations(&axes);
        assert_eq!(combos.len(), 3);
        assert!((combos[0][0].1 - 1.0).abs() < 1e-9);
        assert!((combos[1][0].1 - 2.0).abs() < 1e-9);
        assert!((combos[2][0].1 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_combinations_two_axes() {
        let axes = vec![
            SweepAxis {
                path: "trade.sl_atr_mult".to_string(),
                values: vec![1.0, 2.0],
            },
            SweepAxis {
                path: "trade.tp_atr_mult".to_string(),
                values: vec![3.0, 4.0, 5.0],
            },
        ];
        let combos = generate_combinations(&axes);
        // 2 x 3 = 6
        assert_eq!(combos.len(), 6);
        // Each combo has 2 entries
        for combo in &combos {
            assert_eq!(combo.len(), 2);
        }
    }

    #[test]
    fn test_apply_overrides_trade_fields() {
        let base = StrategyConfig::default();
        let overrides = vec![
            ("trade.sl_atr_mult".to_string(), 1.5),
            ("trade.tp_atr_mult".to_string(), 6.0),
            ("trade.leverage".to_string(), 5.0),
            ("trade.max_open_positions".to_string(), 10.0),
        ];
        let cfg = apply_overrides(&base, &overrides);
        assert!((cfg.trade.sl_atr_mult - 1.5).abs() < 1e-9);
        assert!((cfg.trade.tp_atr_mult - 6.0).abs() < 1e-9);
        assert!((cfg.trade.leverage - 5.0).abs() < 1e-9);
        assert_eq!(cfg.trade.max_open_positions, 10);
    }

    #[test]
    fn test_apply_overrides_thresholds() {
        let base = StrategyConfig::default();
        let overrides = vec![
            ("thresholds.entry.min_adx".to_string(), 18.0),
            ("thresholds.entry.slow_drift_min_slope_pct".to_string(), 0.001),
            ("thresholds.ranging.adx_below".to_string(), 25.0),
        ];
        let cfg = apply_overrides(&base, &overrides);
        assert!((cfg.thresholds.entry.min_adx - 18.0).abs() < 1e-9);
        assert!((cfg.thresholds.entry.slow_drift_min_slope_pct - 0.001).abs() < 1e-9);
        assert!((cfg.thresholds.ranging.adx_below - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_overrides_filters() {
        let base = StrategyConfig::default();
        let overrides = vec![("filters.adx_rising_saturation".to_string(), 35.0)];
        let cfg = apply_overrides(&base, &overrides);
        assert!((cfg.filters.adx_rising_saturation - 35.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_overrides_indicators() {
        let base = StrategyConfig::default();
        let overrides = vec![
            ("indicators.ema_slow_window".to_string(), 100.0),
            ("indicators.ema_fast_window".to_string(), 30.0),
        ];
        let cfg = apply_overrides(&base, &overrides);
        assert_eq!(cfg.indicators.ema_slow_window, 100);
        assert_eq!(cfg.indicators.ema_fast_window, 30);
    }

    #[test]
    fn test_apply_overrides_preserves_untouched() {
        let base = StrategyConfig::default();
        let overrides = vec![("trade.leverage".to_string(), 10.0)];
        let cfg = apply_overrides(&base, &overrides);
        // Changed
        assert!((cfg.trade.leverage - 10.0).abs() < 1e-9);
        // Preserved
        assert!((cfg.trade.allocation_pct - base.trade.allocation_pct).abs() < 1e-9);
        assert!((cfg.trade.sl_atr_mult - base.trade.sl_atr_mult).abs() < 1e-9);
        assert_eq!(cfg.indicators.ema_slow_window, base.indicators.ema_slow_window);
    }

    #[test]
    fn test_sweep_spec_deserialization() {
        let yaml = r#"
axes:
  - path: trade.sl_atr_mult
    values: [1.0, 1.5, 2.0]
  - path: trade.tp_atr_mult
    values: [3.0, 4.0]
initial_balance: 5000.0
lookback: 100
"#;
        let spec: SweepSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.axes.len(), 2);
        assert!((spec.initial_balance - 5000.0).abs() < 1e-9);
        assert_eq!(spec.lookback, 100);
        assert_eq!(spec.axes[0].values.len(), 3);
        assert_eq!(spec.axes[1].values.len(), 2);
    }

    #[test]
    fn test_sweep_spec_defaults() {
        let yaml = r#"
axes:
  - path: trade.leverage
    values: [2.0]
"#;
        let spec: SweepSpec = serde_yaml::from_str(yaml).unwrap();
        assert!((spec.initial_balance - 10000.0).abs() < 1e-9);
        assert_eq!(spec.lookback, 200);
    }
}
