//! Classify sweep axes into indicator-affecting vs trade-only.
//!
//! Indicator axes change EMA/ADX/BB/ATR/RSI window parameters, requiring
//! a full indicator recomputation. Trade axes only affect sizing, SL/TP,
//! trailing, etc. and can be swept on GPU without recomputing indicators.

use bt_core::sweep::SweepAxis;

/// Indicator config paths that require indicator recomputation.
pub const INDICATOR_PATHS: &[&str] = &[
    "indicators.ema_slow_window",
    "indicators.ema_fast_window",
    "indicators.ema_macro_window",
    "indicators.adx_window",
    "indicators.bb_window",
    "indicators.bb_width_avg_window",
    "indicators.atr_window",
    "indicators.rsi_window",
    "indicators.vol_sma_window",
    "indicators.vol_trend_window",
    "indicators.stoch_rsi_window",
    "indicators.stoch_rsi_smooth1",
    "indicators.stoch_rsi_smooth2",
    "indicators.ave_avg_atr_window",
    // Threshold paths that affect indicator-derived values computed at gate time
    "thresholds.entry.ave_avg_atr_window",
    "thresholds.entry.slow_drift_slope_window",
];

/// Split sweep axes into (indicator_axes, trade_axes).
pub fn split_axes(axes: &[SweepAxis]) -> (Vec<SweepAxis>, Vec<SweepAxis>) {
    let mut indicator = Vec::new();
    let mut trade = Vec::new();

    for axis in axes {
        if is_indicator_axis(&axis.path) {
            indicator.push(axis.clone());
        } else {
            trade.push(axis.clone());
        }
    }

    (indicator, trade)
}

fn is_indicator_axis(path: &str) -> bool {
    INDICATOR_PATHS.contains(&path)
}

/// Generate all combinations (cartesian product) from axes.
pub fn generate_combinations(axes: &[SweepAxis]) -> Vec<Vec<(String, f64)>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_axes() {
        let axes = vec![
            SweepAxis { path: "trade.sl_atr_mult".to_string(), values: vec![1.5, 2.0] },
            SweepAxis { path: "indicators.ema_slow_window".to_string(), values: vec![30.0, 50.0] },
            SweepAxis { path: "trade.tp_atr_mult".to_string(), values: vec![3.0, 5.0] },
        ];
        let (ind, trade) = split_axes(&axes);
        assert_eq!(ind.len(), 1);
        assert_eq!(ind[0].path, "indicators.ema_slow_window");
        assert_eq!(trade.len(), 2);
    }

    #[test]
    fn test_generate_combinations() {
        let axes = vec![
            SweepAxis { path: "a".to_string(), values: vec![1.0, 2.0] },
            SweepAxis { path: "b".to_string(), values: vec![10.0, 20.0] },
        ];
        let combos = generate_combinations(&axes);
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_empty_axes() {
        let combos = generate_combinations(&[]);
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0].len(), 0);
    }
}
