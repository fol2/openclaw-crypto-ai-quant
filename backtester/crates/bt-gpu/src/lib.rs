pub mod axis_split;
pub mod buffers;
pub mod gpu_host;
pub mod layout;
pub mod precision;
/// Legacy CPU-side indicator precompute path, superseded by the all-GPU
/// indicator kernel pipeline. Retained for reference and potential fallback.
#[allow(dead_code)]
mod precompute;
pub mod raw_candles;
pub mod tpe_sweep;

#[cfg(feature = "codegen")]
#[path = "../codegen/mod.rs"]
pub mod codegen;

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

/// Build deterministic symbol ordering and enforce GPU kernel symbol cap.
pub(crate) fn sorted_symbols_with_kernel_cap(
    candles: &CandleData,
    log_prefix: &str,
) -> Vec<String> {
    let mut symbols: Vec<String> = candles.keys().cloned().collect();
    symbols.sort();
    if symbols.len() > buffers::GPU_MAX_SYMBOLS {
        eprintln!(
            "{log_prefix} Warning: {} symbols loaded, truncating to {} (kernel state limit)",
            symbols.len(),
            buffers::GPU_MAX_SYMBOLS,
        );
        symbols.truncate(buffers::GPU_MAX_SYMBOLS);
    }
    symbols
}

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
    let (results, _states, _symbols) = run_gpu_sweep_internal(
        candles,
        base_cfg,
        spec,
        _funding,
        sub_candles,
        from_ts,
        to_ts,
        false,
    );
    results
}

/// Run a GPU sweep and also return final GPU combo states plus the symbol
/// ordering used by the kernel.
///
/// Intended for diagnostic tooling (for example axis parity trace capture).
pub fn run_gpu_sweep_with_states(
    candles: &CandleData,
    base_cfg: &StrategyConfig,
    spec: &SweepSpec,
    _funding: Option<&FundingRateData>,
    sub_candles: Option<&CandleData>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> (Vec<GpuSweepResult>, Vec<buffers::GpuComboState>, Vec<String>) {
    let (results, states, symbols) = run_gpu_sweep_internal(
        candles,
        base_cfg,
        spec,
        _funding,
        sub_candles,
        from_ts,
        to_ts,
        true,
    );
    (results, states.unwrap_or_default(), symbols)
}

fn run_gpu_sweep_internal(
    candles: &CandleData,
    base_cfg: &StrategyConfig,
    spec: &SweepSpec,
    _funding: Option<&FundingRateData>,
    sub_candles: Option<&CandleData>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    capture_states: bool,
) -> (
    Vec<GpuSweepResult>,
    Option<Vec<buffers::GpuComboState>>,
    Vec<String>,
) {
    let (indicator_axes, trade_axes) = axis_split::split_axes(&spec.axes);

    let indicator_combos = axis_split::generate_combinations(&indicator_axes);
    let trade_combos = axis_split::generate_combinations(&trade_axes);

    let symbols = sorted_symbols_with_kernel_cap(candles, "[GPU]");
    let num_symbols = symbols.len();

    // BTC symbol index for breadth kernel.
    // Use u32::MAX sentinel when unavailable, so kernels can treat alignment as unknown.
    let btc_sym_idx = symbols
        .iter()
        .position(|s| s == "BTC")
        .or_else(|| symbols.iter().position(|s| s == "BTCUSDT"))
        .map(|idx| idx as u32)
        .unwrap_or(u32::MAX);

    // ── 1. Prepare raw candles (CPU, layout only, ~10ms) ─────────────────
    let raw = raw_candles::prepare_raw_candles(candles, &symbols);
    let num_bars = raw.num_bars as u32;

    // Compute trade bar range from time scope
    let (trade_start, trade_end) =
        raw_candles::find_trade_bar_range(&raw.timestamps, from_ts, to_ts);

    eprintln!(
        "[GPU] {} ind × {} trade = {} total combos, {} bars × {} symbols",
        indicator_combos.len(),
        trade_combos.len(),
        indicator_combos.len() * trade_combos.len(),
        num_bars,
        num_symbols,
    );

    // ── 2. Init CUDA device, upload raw candles ONCE ─────────────────────
    // C6: graceful fallback — return empty results if GPU init fails
    let device_state = match gpu_host::GpuDeviceState::new() {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("[GPU] {e} — GPU sweep unavailable, returning empty results");
            return (
                Vec::new(),
                if capture_states {
                    Some(Vec::new())
                } else {
                    None
                },
                symbols,
            );
        }
    };
    let candles_gpu = device_state.dev.htod_sync_copy(&raw.candles).unwrap();

    let candle_upload_mb =
        (raw.candles.len() * std::mem::size_of::<buffers::GpuRawCandle>()) as f64 / 1e6;
    eprintln!(
        "[GPU] Raw candles uploaded: {:.1} MB (one-time)",
        candle_upload_mb
    );

    // Prepare sub-bar candles for GPU (if provided)
    let sub_bar_result =
        sub_candles.map(|sc| raw_candles::prepare_sub_bar_candles(&raw.timestamps, sc, &symbols));
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
        eprintln!(
            "[GPU] Sub-bar candles: max_sub={}, GPU buffer uploaded",
            max_sub_per_bar
        );
    }

    // ── 3. Calculate VRAM budget ─────────────────────────────────────────
    let total_vram = device_state.total_vram_bytes();
    let free_vram = device_state.free_vram_bytes();

    // Per-indicator-combo VRAM cost (snapshots + breadth + btc_bullish)
    let snapshot_elements = (num_bars as usize) * num_symbols;
    let snapshot_bytes_per_ind: usize = snapshot_elements * std::mem::size_of::<buffers::GpuSnapshot>() // 160 each
        + (num_bars as usize) * std::mem::size_of::<f32>()             // breadth
        + (num_bars as usize) * std::mem::size_of::<u32>(); // btc_bullish

    // Per-trade-combo VRAM cost (config + state + result)
    let combo_bytes: usize = std::mem::size_of::<buffers::GpuComboConfig>()
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
        batch_size,
        max_ind_by_vram,
        max_ind_by_snapshot,
    );
    eprintln!("[GPU] Pipeline: ALL-GPU (indicator + breadth + trade kernels)");

    // ── 4. Main sweep loop ───────────────────────────────────────────────
    let mut all_results: Vec<(GpuSweepResult, Option<buffers::GpuComboState>)> = Vec::new();
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
        // H11: handle GPU memory allocation failure gracefully
        let mut ind_bufs = match gpu_host::IndicatorBuffers::new(
            &device_state,
            &candles_gpu,
            &ind_configs,
            num_bars,
            num_symbols as u32,
            btc_sym_idx,
        ) {
            Ok(bufs) => bufs,
            Err(e) => {
                eprintln!("[GPU] Indicator buffer allocation failed: {e} — skipping batch");
                done += chunk.len();
                continue;
            }
        };
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
        let combo_base = done * t;
        let mut trade_bufs = gpu_host::BatchBuffers::from_indicator_buffers(
            &device_state,
            &ind_bufs,
            &gpu_configs,
            spec.initial_balance as f32,
            combo_base,
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
        let gpu_states = if capture_states {
            Some(gpu_host::readback_states(&device_state, &trade_bufs))
        } else {
            None
        };

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

            let mapped = GpuSweepResult {
                config_id,
                output_mode: "gpu".to_string(),
                total_pnl: result.total_pnl as f64,
                final_balance: result.final_balance as f64,
                total_trades: result.total_trades,
                total_wins: result.total_wins,
                win_rate: wr,
                profit_factor: pf,
                max_drawdown_pct: result.max_drawdown_pct as f64,
                overrides: all_overrides,
            };
            let state_opt = gpu_states.as_ref().map(|states| states[i]);
            all_results.push((mapped, state_opt));
        }

        done += chunk.len();
        if total_ind > 1 {
            eprintln!(
                "[GPU] Progress: {}/{} indicator configs done",
                done, total_ind
            );
        }
        // ind_bufs + trade_bufs dropped here → frees VRAM for next batch
    }

    all_results.sort_by(|a, b| {
        b.0.total_pnl
            .partial_cmp(&a.0.total_pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if capture_states {
        let mut results = Vec::with_capacity(all_results.len());
        let mut states = Vec::with_capacity(all_results.len());
        for (result, state) in all_results {
            results.push(result);
            states.push(state.expect("missing captured state"));
        }
        (results, Some(states), symbols)
    } else {
        let results = all_results.into_iter().map(|(result, _)| result).collect();
        (results, None, symbols)
    }
}

#[cfg(test)]
mod tests {
    use bt_core::candle::{CandleData, OhlcvBar};

    fn extract_fn_block<'a>(source: &'a str, signature: &str) -> &'a str {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("Function signature not found: {signature}"));
        let brace_start = source[start..]
            .find('{')
            .map(|idx| start + idx)
            .expect("Function body start not found");

        let mut depth = 0usize;
        for (idx, ch) in source[brace_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let end = brace_start + idx + 1;
                        return &source[start..end];
                    }
                }
                _ => {}
            }
        }
        panic!("Function body end not found: {signature}");
    }

    #[test]
    fn tp_mult_kernel_sources_are_fixed_to_trade_tp_atr_mult() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

        let cu_src = std::fs::read_to_string(root.join("kernels/sweep_engine.cu"))
            .expect("Failed to read CUDA sweep engine source");
        let cu_fn = extract_fn_block(&cu_src, "__device__ float get_tp_mult(");
        assert!(
            cu_fn.contains("return cfg->tp_atr_mult;"),
            "CUDA get_tp_mult must return cfg->tp_atr_mult"
        );
        assert!(
            !cu_fn.contains("return 7.0f;") && !cu_fn.contains("return 3.0f;"),
            "CUDA get_tp_mult must not hardcode ADX TP multipliers"
        );

        // DEPRECATED (AQC-1241): WGSL shader is no longer maintained.
        // This assertion is kept to prevent silent regressions in the
        // archived shader, but no new parity checks should be added here.
        let wgsl_src = std::fs::read_to_string(root.join("shaders/sweep_engine.wgsl"))
            .expect("Failed to read WGSL sweep engine source (deprecated, retained for reference)");
        let wgsl_fn = extract_fn_block(&wgsl_src, "fn get_tp_mult(");
        assert!(
            wgsl_fn.contains("return (*cfg).tp_atr_mult;"),
            "WGSL get_tp_mult must return (*cfg).tp_atr_mult"
        );
        assert!(
            !wgsl_fn.contains("return 7.0;") && !wgsl_fn.contains("return 3.0;"),
            "WGSL get_tp_mult must not hardcode ADX TP multipliers"
        );
    }

    #[test]
    fn sorted_symbols_with_kernel_cap_truncates_deterministically() {
        let mut candles: CandleData = CandleData::default();
        for idx in 0..(crate::buffers::GPU_MAX_SYMBOLS + 9) {
            candles.insert(
                format!("SYM{:03}", idx),
                vec![OhlcvBar {
                    t: 0,
                    t_close: 0,
                    o: 1.0,
                    h: 1.0,
                    l: 1.0,
                    c: 1.0,
                    v: 1.0,
                    n: 1,
                }],
            );
        }

        let out = crate::sorted_symbols_with_kernel_cap(&candles, "[test]");
        assert_eq!(out.len(), crate::buffers::GPU_MAX_SYMBOLS);
        assert_eq!(out.first().map(String::as_str), Some("SYM000"));
        assert_eq!(out.last().map(String::as_str), Some("SYM051"));
    }

    #[test]
    fn sorted_symbols_with_kernel_cap_keeps_all_when_within_limit() {
        let mut candles: CandleData = CandleData::default();
        candles.insert(
            "BTC".to_string(),
            vec![OhlcvBar {
                t: 0,
                t_close: 0,
                o: 1.0,
                h: 1.0,
                l: 1.0,
                c: 1.0,
                v: 1.0,
                n: 1,
            }],
        );
        candles.insert(
            "ETH".to_string(),
            vec![OhlcvBar {
                t: 0,
                t_close: 0,
                o: 1.0,
                h: 1.0,
                l: 1.0,
                c: 1.0,
                v: 1.0,
                n: 1,
            }],
        );

        let out = crate::sorted_symbols_with_kernel_cap(&candles, "[test]");
        assert_eq!(out, vec!["BTC".to_string(), "ETH".to_string()]);
    }
}
