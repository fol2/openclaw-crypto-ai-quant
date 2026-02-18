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
    let mut combos: Vec<Vec<(String, f64)>> = vec![Vec::new()];
    for axis in axes {
        let mut next = Vec::with_capacity(combos.len().saturating_mul(axis.values.len()));
        for combo in &combos {
            for value in &axis.values {
                let mut expanded = combo.clone();
                expanded.push((axis.path.clone(), *value));
                next.push(expanded);
            }
        }
        combos = next;
    }
    combos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_axes() {
        let axes = vec![
            SweepAxis {
                path: "trade.sl_atr_mult".to_string(),
                values: vec![1.5, 2.0],
                gate: None,
            },
            SweepAxis {
                path: "indicators.ema_slow_window".to_string(),
                values: vec![30.0, 50.0],
                gate: None,
            },
            SweepAxis {
                path: "trade.tp_atr_mult".to_string(),
                values: vec![3.0, 5.0],
                gate: None,
            },
        ];
        let (ind, trade) = split_axes(&axes);
        assert_eq!(ind.len(), 1);
        assert_eq!(ind[0].path, "indicators.ema_slow_window");
        assert_eq!(trade.len(), 2);
    }

    #[test]
    fn test_generate_combinations() {
        let axes = vec![
            SweepAxis {
                path: "a".to_string(),
                values: vec![1.0, 2.0],
                gate: None,
            },
            SweepAxis {
                path: "b".to_string(),
                values: vec![10.0, 20.0],
                gate: None,
            },
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

    #[test]
    fn test_generate_combinations_many_axes_no_recursion() {
        let axes: Vec<SweepAxis> = (0..256)
            .map(|i| SweepAxis {
                path: format!("axis{i}"),
                values: vec![i as f64],
                gate: None,
            })
            .collect();
        let combos = generate_combinations(&axes);
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0].len(), 256);
        assert_eq!(combos[0][0], ("axis0".to_string(), 0.0));
        assert_eq!(combos[0][255], ("axis255".to_string(), 255.0));
    }
}
