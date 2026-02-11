pub mod axis_split;
pub mod buffers;
pub mod gpu_host;
pub mod layout;
#[allow(dead_code)]
mod precompute;
pub mod raw_candles;
pub mod tpe_sweep;

/// Check if a set of overrides produces a degenerate config that the GPU f32
/// kernel can evaluate but produces phantom signals not reproducible in f64.
///
/// Current checks:
/// - `ema_fast_window >= ema_slow_window` → alignment and close/ema gates
///   become anti-correlated, producing zero real signals.
pub fn is_degenerate_overrides(overrides: &[(String, f64)]) -> bool {
    let mut ema_fast: Option<f64> = None;
    let mut ema_slow: Option<f64> = None;
    for (path, val) in overrides {
        match path.as_str() {
            "indicators.ema_fast_window" => ema_fast = Some(*val),
            "indicators.ema_slow_window" => ema_slow = Some(*val),
            _ => {}
        }
    }
    if let (Some(fast), Some(slow)) = (ema_fast, ema_slow) {
        if fast.round() >= slow.round() {
            return true;
        }
    }
    false
}

use bt_core::candle::{CandleData, FundingRateData};
use bt_core::config::StrategyConfig;
use bt_core::sweep::SweepSpec;

pub use buffers::GpuResult;
pub use layout::GpuSweepResult;

/// Bar chunk size for TDR mitigation in trade kernel.
const BAR_CHUNK_SIZE: u32 = 500;

/// Run a GPU-accelerated parameter sweep.
///
/// **All-GPU pipeline**: Raw candles uploaded once → indicator kernel computes
/// indicators on GPU → breadth kernel → trade kernel. No CPU precompute,
/// no host RAM for snapshots, minimal host↔device transfers.
///
/// Pipeline:
/// 1. CPU: prepare raw candle flat buffer (~6 MB, layout only)
/// 2. Upload raw candles to GPU ONCE
/// 3. For each VRAM batch of K indicator combos:
///    a. Launch indicator_kernel (K × S threads) → snapshots in VRAM
///    b. Launch breadth_kernel (K × B threads) → breadth/btc_bullish in VRAM
///    c. Build K × T GpuComboConfigs with per-combo offsets
///    d. Launch sweep_engine_kernel (K × T threads) → trade results
///    e. Read back results (tiny: K × T × 48 bytes)
/// 4. Sort and return results
pub fn run_gpu_sweep(
    candles: &CandleData,
    base_cfg: &StrategyConfig,
    spec: &SweepSpec,
    _funding: Option<&FundingRateData>,
    sub_candles: Option<&CandleData>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Vec<GpuSweepResult> {
    let (indicator_axes, trade_axes) = axis_split::split_axes(&spec.axes);

    let indicator_combos = axis_split::generate_combinations(&indicator_axes);
    let trade_combos = axis_split::generate_combinations(&trade_axes);

    let symbols: Vec<String> = {
        let mut s: Vec<String> = candles.keys().cloned().collect();
        s.sort();
        s
    };
    let num_symbols = symbols.len();

    // BTC symbol index for breadth kernel
    let btc_sym_idx = symbols.iter().position(|s| s == "BTC")
        .or_else(|| symbols.iter().position(|s| s == "BTCUSDT"))
        .unwrap_or(0) as u32;

    // ── 1. Prepare raw candles (CPU, layout only, ~10ms) ─────────────────
    let raw = raw_candles::prepare_raw_candles(candles, &symbols);
    let num_bars = raw.num_bars as u32;

    // Compute trade bar range from time scope
    let (trade_start, trade_end) = raw_candles::find_trade_bar_range(&raw.timestamps, from_ts, to_ts);

    eprintln!(
        "[GPU] {} ind × {} trade = {} total combos, {} bars × {} symbols",
        indicator_combos.len(),
        trade_combos.len(),
        indicator_combos.len() * trade_combos.len(),
        num_bars,
        num_symbols,
    );

    // ── 2. Init CUDA device, upload raw candles ONCE ─────────────────────
    let device_state = gpu_host::GpuDeviceState::new();
    let candles_gpu = device_state.dev.htod_sync_copy(&raw.candles).unwrap();

    let candle_upload_mb = (raw.candles.len() * std::mem::size_of::<buffers::GpuRawCandle>()) as f64 / 1e6;
    eprintln!("[GPU] Raw candles uploaded: {:.1} MB (one-time)", candle_upload_mb);

    // Prepare sub-bar candles for GPU (if provided)
    let sub_bar_result = sub_candles.map(|sc| {
        raw_candles::prepare_sub_bar_candles(&raw.timestamps, sc, &symbols)
    });
    let sub_candles_gpu: Option<cudarc::driver::CudaSlice<buffers::GpuRawCandle>> =
        sub_bar_result.as_ref().and_then(|sbr| {
            if sbr.max_sub_per_bar == 0 || sbr.candles.is_empty() {
                None
            } else {
                Some(device_state.dev.htod_sync_copy(&sbr.candles).unwrap())
            }
        });
    let sub_counts_gpu: Option<cudarc::driver::CudaSlice<u32>> =
        sub_bar_result.as_ref().and_then(|sbr| {
            if sbr.max_sub_per_bar == 0 || sbr.sub_counts.is_empty() {
                None
            } else {
                Some(device_state.dev.htod_sync_copy(&sbr.sub_counts).unwrap())
            }
        });
    let max_sub_per_bar = sub_bar_result
        .as_ref()
        .map(|sbr| sbr.max_sub_per_bar)
        .unwrap_or(0);

    if max_sub_per_bar > 0 {
        eprintln!("[GPU] Sub-bar candles: max_sub={}, GPU buffer uploaded", max_sub_per_bar);
    }

    // ── 3. Calculate VRAM budget ─────────────────────────────────────────
    let total_vram = device_state.total_vram_bytes();
    let free_vram = device_state.free_vram_bytes();

    // Per-indicator-combo VRAM cost (snapshots + breadth + btc_bullish)
    let snapshot_elements = (num_bars as usize) * num_symbols;
    let snapshot_bytes_per_ind: usize =
        snapshot_elements * std::mem::size_of::<buffers::GpuSnapshot>() // 160 each
        + (num_bars as usize) * std::mem::size_of::<f32>()             // breadth
        + (num_bars as usize) * std::mem::size_of::<u32>();            // btc_bullish

    // Per-trade-combo VRAM cost (config + state + result)
    let combo_bytes: usize =
        std::mem::size_of::<buffers::GpuComboConfig>()
        + std::mem::size_of::<buffers::GpuComboState>()
        + std::mem::size_of::<buffers::GpuResult>()
        + 32; // params overhead

    let t = trade_combos.len();
    let per_ind_total_vram = snapshot_bytes_per_ind + t * combo_bytes;

    // Hard cap: max 10 GB per snapshot allocation (large contiguous allocs
    // can fail even with plenty of total free VRAM due to fragmentation)
    let max_snapshot_alloc: usize = 10 * 1024 * 1024 * 1024; // 10 GB
    let max_ind_by_snapshot = if snapshot_bytes_per_ind > 0 {
        max_snapshot_alloc / snapshot_bytes_per_ind
    } else {
        indicator_combos.len()
    };

    // Also respect reported free VRAM (50% to leave room for configs/states/overhead)
    let usable_vram = (free_vram * 50) / 100;
    let max_ind_by_vram = if per_ind_total_vram > 0 {
        usable_vram / per_ind_total_vram
    } else {
        indicator_combos.len()
    };

    let max_ind_per_vram = max_ind_by_snapshot.min(max_ind_by_vram).max(1);
    let batch_size = max_ind_per_vram.min(indicator_combos.len());

    eprintln!(
        "[GPU] VRAM: {:.1} GB total, {:.1} GB free",
        total_vram as f64 / 1e9,
        free_vram as f64 / 1e9,
    );
    eprintln!(
        "[GPU] Per-ind snapshot: {:.1} MB, batch size: {} ind/batch (VRAM limit: {}, snap cap: {})",
        snapshot_bytes_per_ind as f64 / 1e6,
        batch_size, max_ind_by_vram, max_ind_by_snapshot,
    );
    eprintln!("[GPU] Pipeline: ALL-GPU (indicator + breadth + trade kernels)");

    // ── 4. Main sweep loop ───────────────────────────────────────────────
    let mut all_results: Vec<GpuSweepResult> = Vec::new();
    let mut done = 0usize;
    let total_ind = indicator_combos.len();

    for chunk in indicator_combos.chunks(batch_size) {
        let k = chunk.len();

        // a. Build GpuIndicatorConfigs for this batch
        let ind_configs: Vec<buffers::GpuIndicatorConfig> = chunk
            .iter()
            .map(|ind_combo| {
                let mut cfg = base_cfg.clone();
                for (path, value) in ind_combo {
                    bt_core::sweep::apply_one_pub(&mut cfg, path, *value);
                }
                buffers::GpuIndicatorConfig::from_strategy_config(&cfg, spec.lookback)
            })
            .collect();

        // b. Create indicator buffers + dispatch indicator & breadth kernels
        let mut ind_bufs = gpu_host::IndicatorBuffers::new(
            &device_state,
            &candles_gpu,
            &ind_configs,
            num_bars,
            num_symbols as u32,
            btc_sym_idx,
        );
        gpu_host::dispatch_indicator_kernels(&device_state, &mut ind_bufs);

        // c. Build K×T GpuComboConfigs with per-combo offsets
        let snapshot_stride = (num_bars as usize) * num_symbols;
        let breadth_stride = num_bars as usize;

        // Also need per-indicator StrategyConfig for trade params
        let ind_cfgs: Vec<(Vec<(String, f64)>, StrategyConfig)> = chunk
            .iter()
            .map(|ind_combo| {
                let mut cfg = base_cfg.clone();
                for (path, value) in ind_combo {
                    bt_core::sweep::apply_one_pub(&mut cfg, path, *value);
                }
                (ind_combo.clone(), cfg)
            })
            .collect();

        let total_combos = k * t;
        let mut gpu_configs = Vec::with_capacity(total_combos);
        let mut combo_meta: Vec<(usize, usize)> = Vec::with_capacity(total_combos);

        for (ind_idx, (_ind_combo, ind_cfg)) in ind_cfgs.iter().enumerate() {
            let snap_off = (ind_idx * snapshot_stride) as u32;
            let br_off = (ind_idx * breadth_stride) as u32;

            for (trade_idx, trade_overrides) in trade_combos.iter().enumerate() {
                let mut cfg = ind_cfg.clone();
                for (path, value) in trade_overrides {
                    bt_core::sweep::apply_one_pub(&mut cfg, path, *value);
                }
                let mut gpu_cfg = buffers::GpuComboConfig::from_strategy_config(&cfg);
                gpu_cfg.snapshot_offset = snap_off;
                gpu_cfg.breadth_offset = br_off;
                gpu_configs.push(gpu_cfg);
                combo_meta.push((ind_idx, trade_idx));
            }
        }

        // d. Dispatch trade kernel using VRAM-resident indicator data
        let mut trade_bufs = gpu_host::BatchBuffers::from_indicator_buffers(
            &device_state,
            &ind_bufs,
            &gpu_configs,
            spec.initial_balance as f32,
        );
        trade_bufs.max_sub_per_bar = max_sub_per_bar;
        trade_bufs.sub_candles = sub_candles_gpu.clone();
        trade_bufs.sub_counts = sub_counts_gpu.clone();

        let gpu_results = gpu_host::dispatch_and_readback(
            &device_state,
            &mut trade_bufs,
            num_bars,
            BAR_CHUNK_SIZE,
            trade_start,
            trade_end,
        );

        // e. Map results back
        for (i, result) in gpu_results.iter().enumerate() {
            let (ind_idx, trade_idx) = combo_meta[i];
            let (ind_combo, _) = &ind_cfgs[ind_idx];
            let trade_overrides = &trade_combos[trade_idx];

            let mut all_overrides = ind_combo.clone();
            all_overrides.extend(trade_overrides.iter().cloned());

            // Skip degenerate combos (e.g. ema_fast >= ema_slow) that produce
            // phantom f32 signals not reproducible in f64 CPU replay.
            if is_degenerate_overrides(&all_overrides) {
                continue;
            }

            let config_id = all_overrides
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");

            let pf = if result.gross_loss.abs() > 0.001 {
                result.gross_profit as f64 / result.gross_loss.abs() as f64
            } else if result.gross_profit > 0.0 {
                999.0
            } else {
                0.0
            };

            let wr = if result.total_trades > 0 {
                result.total_wins as f64 / result.total_trades as f64
            } else {
                0.0
            };

            all_results.push(GpuSweepResult {
                config_id,
                total_pnl: result.total_pnl as f64,
                final_balance: result.final_balance as f64,
                total_trades: result.total_trades,
                total_wins: result.total_wins,
                win_rate: wr,
                profit_factor: pf,
                max_drawdown_pct: result.max_drawdown_pct as f64,
                overrides: all_overrides,
            });
        }

        done += chunk.len();
        if total_ind > 1 {
            eprintln!("[GPU] Progress: {}/{} indicator configs done", done, total_ind);
        }
        // ind_bufs + trade_bufs dropped here → frees VRAM for next batch
    }

    all_results.sort_by(|a, b| {
        b.total_pnl
            .partial_cmp(&a.total_pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results
}
