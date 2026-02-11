//! CPU-side indicator precomputation.
//!
//! Runs IndicatorBank for each symbol across all bars, producing a flat
//! Vec<GpuSnapshot> in [bar_idx × num_symbols + sym_idx] layout.
//! Also produces per-bar breadth% and BTC bullish flag.

use bt_core::candle::CandleData;
use bt_core::config::StrategyConfig;
use bt_core::indicators::IndicatorBank;

use bytemuck::Zeroable;
use crate::buffers::GpuSnapshot;

/// Result of CPU precomputation for one indicator config.
pub struct PrecomputeResult {
    /// Flat snapshot array: `snapshots[bar_idx * num_symbols + sym_idx]`
    pub snapshots: Vec<GpuSnapshot>,
    /// Per-bar market breadth percentage (0-100).
    pub breadth: Vec<f32>,
    /// Per-bar BTC bullish flag (1 = bullish, 0 = bearish/unknown).
    pub btc_bullish: Vec<u32>,
    /// Number of unique timestamps (bars).
    pub num_bars: usize,
    /// Number of symbols.
    pub num_symbols: usize,
    /// Sorted unique timestamps (ms).
    pub timestamps: Vec<i64>,
    /// EMA slow slope history per symbol per bar for slow-drift signal.
    pub ema_slow_slopes: Vec<f32>,
}

/// Precompute indicator snapshots for all symbols across all timestamps.
pub fn precompute_snapshots(
    candles: &CandleData,
    cfg: &StrategyConfig,
    symbols: &[String],
    lookback: usize,
) -> PrecomputeResult {
    use rustc_hash::FxHashMap;

    let num_symbols = symbols.len();

    // Build unified timeline
    let mut all_ts: Vec<i64> = candles
        .values()
        .flat_map(|bars| bars.iter().map(|b| b.t))
        .collect();
    all_ts.sort_unstable();
    all_ts.dedup();
    let num_bars = all_ts.len();

    // Build per-symbol bar index: symbol -> (timestamp -> bar_index)
    let mut bar_indices: FxHashMap<&str, FxHashMap<i64, usize>> = FxHashMap::default();
    for (sym, bars) in candles {
        let mut idx = FxHashMap::default();
        for (i, bar) in bars.iter().enumerate() {
            idx.insert(bar.t, i);
        }
        bar_indices.insert(sym.as_str(), idx);
    }

    // Symbol index lookup
    let sym_to_idx: FxHashMap<&str, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // Initialize indicator banks per symbol
    let mut banks: Vec<IndicatorBank> = symbols
        .iter()
        .map(|_| IndicatorBank::new(&cfg.indicators, cfg.filters.use_stoch_rsi_filter))
        .collect();

    // EMA slow history per symbol (for slope computation)
    let slope_window = cfg.thresholds.entry.slow_drift_slope_window;
    let mut ema_slow_histories: Vec<Vec<f64>> = vec![Vec::new(); num_symbols];

    // Output arrays
    let total_slots = num_bars * num_symbols;
    let mut snapshots = vec![GpuSnapshot::zeroed(); total_slots];
    let mut breadth = vec![50.0f32; num_bars];
    // 0 = bearish, 1 = bullish, 2 = unavailable.
    let mut btc_bullish = vec![2u32; num_bars];
    let mut ema_slow_slopes = vec![0.0f32; total_slots];

    // BTC symbol index
    let btc_idx = sym_to_idx.get("BTC").or_else(|| sym_to_idx.get("BTCUSDT"));

    for (bar_idx, &ts) in all_ts.iter().enumerate() {
        let mut bullish_count = 0u32;
        let mut total_count = 0u32;

        for (sym_idx, sym) in symbols.iter().enumerate() {
            let slot = bar_idx * num_symbols + sym_idx;

            // Check if this symbol has a bar at this timestamp
            let bar = bar_indices
                .get(sym.as_str())
                .and_then(|idx| idx.get(&ts))
                .and_then(|&i| candles.get(sym).and_then(|bars| bars.get(i)));

            if let Some(bar) = bar {
                let snap = banks[sym_idx].update(bar);

                // Track EMA slow history for slope
                ema_slow_histories[sym_idx].push(snap.ema_slow);

                // Compute EMA slow slope
                let slope = if ema_slow_histories[sym_idx].len() >= slope_window
                    && snap.close > 0.0
                {
                    let h = &ema_slow_histories[sym_idx];
                    let current = h[h.len() - 1];
                    let past = h[h.len() - slope_window];
                    ((current - past) / snap.close) as f32
                } else {
                    0.0
                };

                // Check warmup
                let is_warm = snap.bar_count >= lookback;

                snapshots[slot] = GpuSnapshot {
                    close: snap.close as f32,
                    high: snap.high as f32,
                    low: snap.low as f32,
                    open: snap.open as f32,
                    volume: snap.volume as f32,
                    t_sec: (ts / 1000) as u32,
                    ema_fast: snap.ema_fast as f32,
                    ema_slow: snap.ema_slow as f32,
                    ema_macro: snap.ema_macro as f32,
                    adx: snap.adx as f32,
                    adx_slope: snap.adx_slope as f32,
                    adx_pos: snap.adx_pos as f32,
                    adx_neg: snap.adx_neg as f32,
                    atr: snap.atr as f32,
                    atr_slope: snap.atr_slope as f32,
                    avg_atr: snap.avg_atr as f32,
                    bb_upper: snap.bb_upper as f32,
                    bb_lower: snap.bb_lower as f32,
                    bb_width: snap.bb_width as f32,
                    bb_width_ratio: snap.bb_width_ratio as f32,
                    rsi: snap.rsi as f32,
                    stoch_k: snap.stoch_rsi_k as f32,
                    stoch_d: snap.stoch_rsi_d as f32,
                    macd_hist: snap.macd_hist as f32,
                    prev_macd_hist: snap.prev_macd_hist as f32,
                    prev2_macd_hist: snap.prev2_macd_hist as f32,
                    prev3_macd_hist: snap.prev3_macd_hist as f32,
                    vol_sma: snap.vol_sma as f32,
                    vol_trend: snap.vol_trend as u32,
                    prev_close: snap.prev_close as f32,
                    prev_ema_fast: snap.prev_ema_fast as f32,
                    prev_ema_slow: snap.prev_ema_slow as f32,
                    ema_slow_slope_pct: slope,
                    bar_count: snap.bar_count as u32,
                    valid: if is_warm { 1 } else { 0 },
                    funding_rate: snap.funding_rate as f32,
                    _pad: [0; 4],
                };
                ema_slow_slopes[slot] = slope;

                // Market breadth: EMA_fast > EMA_slow = bullish
                if snap.bar_count >= 2 {
                    total_count += 1;
                    if snap.prev_ema_fast > snap.prev_ema_slow {
                        bullish_count += 1;
                    }
                }
            }
            // else: snapshots[slot] remains zeroed (valid=0)
        }

        // Breadth
        breadth[bar_idx] = if total_count > 0 {
            (bullish_count as f32 / total_count as f32) * 100.0
        } else {
            50.0
        };

        // BTC bullish
        if let Some(&bi) = btc_idx {
            let bank = &banks[bi];
            if bank.bar_count >= 2 {
                btc_bullish[bar_idx] = if bank.prev_close > bank.prev_ema_slow { 1 } else { 0 };
            }
        }
    }

    PrecomputeResult {
        snapshots,
        breadth,
        btc_bullish,
        num_bars,
        num_symbols,
        timestamps: all_ts,
        ema_slow_slopes,
    }
}
