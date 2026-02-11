//! Raw candle preparation for GPU upload.
//!
//! Builds a flat `Vec<GpuRawCandle>` in [bar_idx × num_symbols + sym_idx] layout
//! from the CandleData HashMap. This is pure data layout — no indicator computation.
//! The resulting buffer is ~6 MB for 5000 bars × 51 symbols and is uploaded to GPU once.

use bt_core::candle::CandleData;
use bytemuck::Zeroable;
use rustc_hash::FxHashMap;

use crate::buffers::GpuRawCandle;

/// Result of sub-bar candle preparation for GPU upload.
pub struct SubBarResult {
    /// Flat candle array: `candles[(bar_idx * max_sub + sub_idx) * num_symbols + sym_idx]`
    /// Missing slots have close=0.
    pub candles: Vec<GpuRawCandle>,
    /// Per (bar, symbol) count of valid sub-bars: `sub_counts[bar_idx * num_symbols + sym_idx]`
    pub sub_counts: Vec<u32>,
    /// Maximum number of sub-bars per main bar (rectangle width).
    pub max_sub_per_bar: u32,
    /// Number of main bars.
    pub num_bars: usize,
    /// Number of symbols.
    pub num_symbols: usize,
}

/// Result of raw candle preparation.
pub struct RawCandleResult {
    /// Flat candle array: `candles[bar_idx * num_symbols + sym_idx]`
    /// Missing bars have close=0.
    pub candles: Vec<GpuRawCandle>,
    /// Number of unique timestamps (bars).
    pub num_bars: usize,
    /// Number of symbols.
    pub num_symbols: usize,
    /// Sorted unique timestamps (ms).
    pub timestamps: Vec<i64>,
}

/// Find the bar index range `[start, end)` that falls within `[from_ts, to_ts]`.
///
/// Uses `partition_point` (binary search on sorted timestamps).
/// When both bounds are `None`, returns `(0, len)` — full range (backwards compatible).
pub fn find_trade_bar_range(
    timestamps: &[i64],
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> (u32, u32) {
    let start = match from_ts {
        Some(ft) => timestamps.partition_point(|&t| t < ft) as u32,
        None => 0,
    };
    let end = match to_ts {
        Some(tt) => timestamps.partition_point(|&t| t <= tt) as u32,
        None => timestamps.len() as u32,
    };
    (start, end)
}

/// Build flat raw candle buffer for GPU upload.
///
/// This is O(bars × symbols) data layout with zero computation.
/// Missing (bar, symbol) pairs get close=0 (indicator kernel skips them).
pub fn prepare_raw_candles(candles: &CandleData, symbols: &[String]) -> RawCandleResult {
    let num_symbols = symbols.len();

    // Build unified timeline from all symbols
    let mut all_ts: Vec<i64> = candles
        .values()
        .flat_map(|bars| bars.iter().map(|b| b.t))
        .collect();
    all_ts.sort_unstable();
    all_ts.dedup();
    let num_bars = all_ts.len();

    // Build per-symbol bar index: symbol → (timestamp → bar_index_in_source)
    let mut bar_indices: FxHashMap<&str, FxHashMap<i64, usize>> = FxHashMap::default();
    for (sym, bars) in candles {
        let mut idx = FxHashMap::default();
        for (i, bar) in bars.iter().enumerate() {
            idx.insert(bar.t, i);
        }
        bar_indices.insert(sym.as_str(), idx);
    }

    // Allocate flat buffer
    let total_slots = num_bars * num_symbols;
    let mut raw = vec![GpuRawCandle::zeroed(); total_slots];

    // Fill: for each (bar, symbol), copy OHLCV or leave zeroed
    for (bar_idx, &ts) in all_ts.iter().enumerate() {
        for (sym_idx, sym) in symbols.iter().enumerate() {
            let bar = bar_indices
                .get(sym.as_str())
                .and_then(|idx| idx.get(&ts))
                .and_then(|&i| candles.get(sym).and_then(|bars| bars.get(i)));

            if let Some(bar) = bar {
                raw[bar_idx * num_symbols + sym_idx] = GpuRawCandle {
                    open: bar.o as f32,
                    high: bar.h as f32,
                    low: bar.l as f32,
                    close: bar.c as f32,
                    volume: bar.v as f32,
                    t_sec: (ts / 1000) as u32,
                    _pad: [0; 2],
                };
            }
            // else: stays zeroed (close=0 → indicator kernel skips)
        }
    }

    RawCandleResult {
        candles: raw,
        num_bars,
        num_symbols,
        timestamps: all_ts,
    }
}

/// Prepare sub-bar candles aligned to main bar ranges for GPU upload.
///
/// For each main bar `i`, sub-bars fall in the half-open range `(ts[i], ts[i+1]]`
/// (last bar uses `(ts[last], +inf)`). This matches bt-core's sub-bar scan:
/// it evaluates sub-bars *after* the current indicator bar using the current
/// indicator snapshot. Sub-bars at or before `ts[0]` are ignored.
///
/// Sub-bars are sorted chronologically and packed into a rectangular layout
/// padded with zeroed candles.
///
/// Layout: `candles[(bar_idx * max_sub + sub_idx) * num_symbols + sym_idx]`
///
/// This uses the same `GpuRawCandle` struct as main candles — only OHLCV + t_sec
/// are meaningful; the kernel reads just price fields for SL/TP checks.
pub fn prepare_sub_bar_candles(
    main_timestamps: &[i64],
    sub_candles: &CandleData,
    symbols: &[String],
) -> SubBarResult {
    let num_bars = main_timestamps.len();
    let num_symbols = symbols.len();

    if num_bars == 0 || num_symbols == 0 {
        return SubBarResult {
            candles: Vec::new(),
            sub_counts: vec![0u32; num_bars * num_symbols],
            max_sub_per_bar: 0,
            num_bars,
            num_symbols,
        };
    }

    // For each (bar, symbol), collect sorted sub-bar timestamps.
    // Main bar i covers (main_ts[i], main_ts[i+1]] in ms.
    // Last bar covers (main_ts[last], +inf).

    // Step 1: For each symbol, sort sub-bars by timestamp and assign to main bars via binary search
    // sub_bars_by_bar_sym[bar_idx][sym_idx] = Vec<&Bar>
    let mut sub_bars_by_bar_sym: Vec<Vec<Vec<&bt_core::candle::OhlcvBar>>> =
        vec![vec![Vec::new(); num_symbols]; num_bars];

    for (sym_idx, sym) in symbols.iter().enumerate() {
        if let Some(bars) = sub_candles.get(sym) {
            for bar in bars {
                // Find which main bar this sub-bar belongs to for CPU-parity windows:
                //   (main_ts[i], main_ts[i+1]].
                //
                // We use lower_bound(main_ts, t) - 1:
                // - t == main_ts[k]   -> bar k-1
                // - main_ts[k] < t <= main_ts[k+1] -> bar k
                // - t > main_ts[last] -> last bar
                // - t <= main_ts[0]   -> ignored
                let main_idx = match main_timestamps.binary_search(&bar.t) {
                    Ok(0) => continue,
                    Ok(i) => i - 1,
                    Err(0) => continue,
                    Err(i) if i >= num_bars => num_bars - 1,
                    Err(i) => i - 1,
                };
                sub_bars_by_bar_sym[main_idx][sym_idx].push(bar);
            }
        }
    }

    // Step 2: Find max sub-bars per main bar (across all symbols)
    let mut max_sub: u32 = 0;
    for bar_idx in 0..num_bars {
        for sym_idx in 0..num_symbols {
            let count = sub_bars_by_bar_sym[bar_idx][sym_idx].len() as u32;
            if count > max_sub {
                max_sub = count;
            }
        }
    }

    if max_sub == 0 {
        return SubBarResult {
            candles: Vec::new(),
            sub_counts: vec![0u32; num_bars * num_symbols],
            max_sub_per_bar: 0,
            num_bars,
            num_symbols,
        };
    }

    // Step 3: Allocate rectangular buffer and fill
    let total_slots = num_bars * (max_sub as usize) * num_symbols;
    let mut flat = vec![GpuRawCandle::zeroed(); total_slots];
    let mut sub_counts = vec![0u32; num_bars * num_symbols];

    for bar_idx in 0..num_bars {
        for sym_idx in 0..num_symbols {
            let subs = &mut sub_bars_by_bar_sym[bar_idx][sym_idx];
            // Sort chronologically (should already be sorted, but ensure)
            subs.sort_by_key(|b| b.t);

            let count = subs.len();
            sub_counts[bar_idx * num_symbols + sym_idx] = count as u32;

            for (sub_idx, bar) in subs.iter().enumerate() {
                let flat_idx = (bar_idx * (max_sub as usize) + sub_idx) * num_symbols + sym_idx;
                flat[flat_idx] = GpuRawCandle {
                    open: bar.o as f32,
                    high: bar.h as f32,
                    low: bar.l as f32,
                    close: bar.c as f32,
                    volume: bar.v as f32,
                    t_sec: (bar.t / 1000) as u32,
                    _pad: [0; 2],
                };
            }
        }
    }

    eprintln!(
        "[sub-bar] Prepared {} bars × {} max_sub × {} symbols = {} slots ({:.1} MB)",
        num_bars,
        max_sub,
        num_symbols,
        total_slots,
        (total_slots * std::mem::size_of::<GpuRawCandle>()) as f64 / 1e6,
    );

    SubBarResult {
        candles: flat,
        sub_counts,
        max_sub_per_bar: max_sub,
        num_bars,
        num_symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::candle::OhlcvBar;

    #[test]
    fn prepare_sub_bars_matches_cpu_exit_windows() {
        let main_timestamps = vec![1_000, 2_000, 3_000];
        let symbols = vec!["ETH".to_string()];

        // Regression guard for AQC-168:
        // if these windows drift from bt-core, GPU sub-bar exits evaluate SL using
        // the wrong indicator snapshot and can flip sl_atr_mult directional behaviour.
        // Include boundary cases: <= first ts (ignored), exact boundaries, between boundaries,
        // and beyond the last timestamp (should map to last bar).
        let sub_times = vec![900_i64, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500];
        let mut sub: CandleData = CandleData::default();
        sub.insert(
            "ETH".to_string(),
            sub_times
                .iter()
                .map(|&t| OhlcvBar {
                    t,
                    t_close: t,
                    o: t as f64,
                    h: t as f64,
                    l: t as f64,
                    c: t as f64,
                    v: 1.0,
                    n: 1,
                })
                .collect(),
        );

        let out = prepare_sub_bar_candles(&main_timestamps, &sub, &symbols);

        // Expected CPU-parity assignment:
        // bar0 (1000,2000] -> 1500,2000
        // bar1 (2000,3000] -> 2500,3000
        // bar2 (3000,+inf) -> 3500
        assert_eq!(out.sub_counts, vec![2, 2, 1]);

        let max_sub = out.max_sub_per_bar as usize;
        let ns = symbols.len();
        let close_at = |bar_idx: usize, sub_idx: usize| -> i64 {
            let idx = (bar_idx * max_sub + sub_idx) * ns;
            out.candles[idx].close as i64
        };

        assert_eq!(close_at(0, 0), 1_500);
        assert_eq!(close_at(0, 1), 2_000);
        assert_eq!(close_at(1, 0), 2_500);
        assert_eq!(close_at(1, 1), 3_000);
        assert_eq!(close_at(2, 0), 3_500);
    }
}
