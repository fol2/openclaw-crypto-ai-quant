//! TPE (Tree-structured Parzen Estimator) Bayesian optimization for GPU sweep.
//!
//! Uses the `tpe` crate for intelligent parameter sampling instead of grid search.
//! Each TPE trial batch is evaluated on GPU using the same all-GPU pipeline.
//!
//! **Hybrid approach**: TPE samples N configs -> GPU batch evaluate -> tell results -> repeat.
//! First batch is essentially random; subsequent batches are increasingly informed.
//!
//! **Optimization layers**:
//! - Layer 1: Fixed VRAM budget from total_vram (not fluctuating free_vram) + arena buffers
//! - Layer 2: Indicator config dedup (many TPE trials share identical rounded int params)
//! - Layer 3: TPE observation pruning (caps ask/tell complexity at O(4000))
//! - Layer 4: Double-buffer pipeline — TPE ask() overlaps with GPU evaluation

use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;

use bt_core::candle::{CandleData, FundingRateData};
use bt_core::config::StrategyConfig;
use bt_core::sweep::SweepSpec;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

use bytemuck::Zeroable;

use crate::axis_split::INDICATOR_PATHS;
use crate::buffers;
use crate::gpu_host;
use crate::layout::{GpuSweepResult, MinPnlHeapEntry};
use crate::raw_candles;
use crate::BAR_CHUNK_SIZE;

/// Maximum TPE observations before pruning kicks in.
/// Lower = faster ask() calls (each ask is O(observations) per axis).
/// 1000 balances Bayesian quality vs ask() speed for 34-axis spaces.
const MAX_TPE_OBSERVATIONS: usize = 1000;
/// After pruning, keep only this many best observations.
const PRUNED_OBSERVATIONS: usize = 500;

/// Configuration for TPE-based sweep.
pub struct TpeConfig {
    /// Total number of trials to evaluate.
    pub trials: usize,
    /// Number of trials per GPU batch (higher = more GPU throughput, less TPE efficiency).
    pub batch_size: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            trials: 5000,
            batch_size: 256,
            seed: 42,
        }
    }
}

// =============================================================================
// Double-buffer pipeline messages
// =============================================================================

/// TPE thread → GPU thread: sampled params for one batch.
struct TpeBatchRequest {
    batch_idx: usize,
    batch_n: usize,
    trial_overrides: Vec<Vec<(String, f64)>>,
    trial_raw_values: Vec<Vec<f64>>,
}

/// GPU thread → TPE thread: evaluation results for one batch.
struct GpuBatchResult {
    #[allow(dead_code)]
    batch_idx: usize,
    results: Vec<buffers::GpuResult>,
    trial_overrides: Vec<Vec<(String, f64)>>,
    trial_raw_values: Vec<Vec<f64>>,
}

// =============================================================================
// Per-axis optimizer state
// =============================================================================

/// Per-axis optimizer state.
struct AxisOptimizer {
    path: String,
    min_val: f64,
    max_val: f64,
    optimizer: tpe::TpeOptimizer,
    is_indicator: bool,
    is_integer: bool,
}

// =============================================================================
// TPE worker (runs on dedicated thread)
// =============================================================================

/// TPE worker: runs on a dedicated thread. Owns all TpeOptimizer state.
///
/// **Pre-fetch pipeline** — always stays one batch ahead of GPU:
///   1. Sample batch 0 (priming) → send to GPU
///   2. Immediately sample batch 1 (speculative, pre-fetch) → send to GPU
///   3. Wait for GPU results of batch 0
///   4. tell() batch 0 results
///   5. Sample batch 2 (now informed by batch 0) → send to GPU
///   6. Wait for GPU results of batch 1 → tell() → sample batch 3 → send ...
///   7. Repeat until all trials exhausted
///
/// GPU never starves because there's always a pre-fetched batch in the channel.
/// Quality loss: batch 1 is sampled before batch 0 results (same as random for
/// the first batch; subsequent pre-fetches lag by one tell cycle, not two).
fn tpe_worker(
    mut axis_opts: Vec<AxisOptimizer>,
    mut rng: StdRng,
    tpe_cfg_trials: usize,
    tpe_cfg_batch_size: usize,
    result_rx: crossbeam_channel::Receiver<GpuBatchResult>,
    request_tx: crossbeam_channel::Sender<Option<TpeBatchRequest>>,
) {
    let mut observation_cache: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut obs_count: usize = 0;
    let mut trials_sampled: usize = 0;
    let mut batch_idx: usize = 0;
    let mut pending_results: usize = 0;

    // --- Helper: sample and send one batch if trials remain ---
    let send_batch =
        |axis_opts: &mut Vec<AxisOptimizer>,
         rng: &mut StdRng,
         trials_sampled: &mut usize,
         batch_idx: &mut usize,
         pending: &mut usize,
         tx: &crossbeam_channel::Sender<Option<TpeBatchRequest>>|
         -> bool {
            if *trials_sampled >= tpe_cfg_trials {
                return false;
            }
            let batch_n = tpe_cfg_batch_size.min(tpe_cfg_trials - *trials_sampled);
            let (overrides, raw_values) = sample_batch(axis_opts, rng, batch_n);
            *trials_sampled += batch_n;

            let ok = tx
                .send(Some(TpeBatchRequest {
                    batch_idx: *batch_idx,
                    batch_n,
                    trial_overrides: overrides,
                    trial_raw_values: raw_values,
                }))
                .is_ok();
            if ok {
                *batch_idx += 1;
                *pending += 1;
            }
            ok
        };

    // --- Priming: send batch 0 ---
    send_batch(
        &mut axis_opts,
        &mut rng,
        &mut trials_sampled,
        &mut batch_idx,
        &mut pending_results,
        &request_tx,
    );

    // --- Pre-fetch: send batch 1 immediately (speculative, before any results) ---
    send_batch(
        &mut axis_opts,
        &mut rng,
        &mut trials_sampled,
        &mut batch_idx,
        &mut pending_results,
        &request_tx,
    );

    // --- Main loop: receive results, tell, pre-fetch next ---
    while pending_results > 0 {
        let gpu_result = match result_rx.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        pending_results -= 1;

        // tell() the results from the completed batch
        tell_results(
            &mut axis_opts,
            &gpu_result,
            &mut observation_cache,
            &mut obs_count,
        );

        // Pre-fetch: sample and send next batch (overlaps with GPU processing
        // the batch that's already in the channel)
        if trials_sampled < tpe_cfg_trials {
            send_batch(
                &mut axis_opts,
                &mut rng,
                &mut trials_sampled,
                &mut batch_idx,
                &mut pending_results,
                &request_tx,
            );
        }
    }

    // Signal GPU thread: no more batches
    let _ = request_tx.send(None);
}

/// Sample one batch of trial parameters from TPE.
fn sample_batch(
    axis_opts: &mut [AxisOptimizer],
    rng: &mut StdRng,
    batch_n: usize,
) -> (Vec<Vec<(String, f64)>>, Vec<Vec<f64>>) {
    // Generate deterministic per-axis RNG seeds from main rng
    let axis_seeds: Vec<u64> = (0..axis_opts.len()).map(|_| rng.gen()).collect();

    // Parallel ask: each axis generates all batch_n samples independently
    let axis_samples: Vec<Vec<f64>> = axis_opts
        .par_iter_mut()
        .zip(axis_seeds.par_iter())
        .map(|(axis, &seed)| {
            let mut local_rng = StdRng::seed_from_u64(seed);
            (0..batch_n)
                .map(|_| {
                    if (axis.max_val - axis.min_val).abs() < 1e-12 {
                        axis.min_val
                    } else {
                        axis.optimizer.ask(&mut local_rng).unwrap()
                    }
                })
                .collect()
        })
        .collect();

    // Transpose [axis][trial] → [trial][axis] and apply clamp/round
    let mut trial_overrides: Vec<Vec<(String, f64)>> = Vec::with_capacity(batch_n);
    let mut trial_raw_values: Vec<Vec<f64>> = Vec::with_capacity(batch_n);

    for t in 0..batch_n {
        let mut overrides = Vec::with_capacity(axis_opts.len());
        let mut raw_vals = Vec::with_capacity(axis_opts.len());

        for (a, axis) in axis_opts.iter().enumerate() {
            let raw_val = axis_samples[a][t];
            raw_vals.push(raw_val);

            let val = if axis.is_integer {
                raw_val.round().clamp(axis.min_val, axis.max_val)
            } else {
                raw_val.clamp(axis.min_val, axis.max_val)
            };
            overrides.push((axis.path.clone(), val));
        }

        trial_overrides.push(overrides);
        trial_raw_values.push(raw_vals);
    }

    (trial_overrides, trial_raw_values)
}

/// Process GPU results: tell TPE + prune observations.
fn tell_results(
    axis_opts: &mut [AxisOptimizer],
    gpu_result: &GpuBatchResult,
    observation_cache: &mut Vec<(Vec<f64>, f64)>,
    obs_count: &mut usize,
) {
    for (i, result) in gpu_result.results.iter().enumerate() {
        let pnl = result.total_pnl as f64;
        let objective = if crate::is_degenerate_overrides(&gpu_result.trial_overrides[i]) {
            1e18
        } else {
            let trade_penalty = if result.total_trades < 20 { 0.5 } else { 1.0 };
            -(pnl * trade_penalty)
        };

        for (j, axis) in axis_opts.iter_mut().enumerate() {
            if (axis.max_val - axis.min_val).abs() < 1e-12 {
                continue;
            }
            let _ = axis
                .optimizer
                .tell(gpu_result.trial_raw_values[i][j], objective);
        }

        observation_cache.push((gpu_result.trial_raw_values[i].clone(), objective));
        *obs_count += 1;
    }

    // Layer 3: TPE observation pruning
    if *obs_count > MAX_TPE_OBSERVATIONS {
        let t_prune = Instant::now();

        observation_cache
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        observation_cache.truncate(PRUNED_OBSERVATIONS);
        *obs_count = observation_cache.len();

        for axis in axis_opts.iter_mut() {
            if (axis.max_val - axis.min_val).abs() < 1e-12 {
                continue;
            }
            axis.optimizer = tpe::TpeOptimizer::new(
                tpe::parzen_estimator(),
                tpe::range(axis.min_val, axis.max_val).unwrap(),
            );
        }

        for (raw_vals, objective) in observation_cache.iter() {
            for (j, axis) in axis_opts.iter_mut().enumerate() {
                if (axis.max_val - axis.min_val).abs() < 1e-12 {
                    continue;
                }
                let _ = axis.optimizer.tell(raw_vals[j], *objective);
            }
        }

        let prune_ms = t_prune.elapsed().as_millis();
        eprintln!(
            "[TPE] Pruned observations: {} -> {} ({}ms)",
            MAX_TPE_OBSERVATIONS, PRUNED_OBSERVATIONS, prune_ms,
        );
    }
}

// =============================================================================
// GPU-side result builder
// =============================================================================

/// Build GpuSweepResult entries from raw GPU results + trial overrides.
fn build_sweep_results(
    gpu_results: &[buffers::GpuResult],
    trial_overrides: &[Vec<(String, f64)>],
    best_pnl: &mut f64,
    best_trial: &mut usize,
    trials_done: usize,
) -> Vec<GpuSweepResult> {
    let mut out = Vec::with_capacity(gpu_results.len());

    for (i, result) in gpu_results.iter().enumerate() {
        let pnl = result.total_pnl as f64;
        let overrides = &trial_overrides[i];

        let config_id = overrides
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

        if pnl > *best_pnl {
            *best_pnl = pnl;
            *best_trial = trials_done + i;
        }

        out.push(GpuSweepResult {
            config_id,
            output_mode: "gpu_tpe".to_string(),
            total_pnl: pnl,
            final_balance: result.final_balance as f64,
            total_trades: result.total_trades,
            total_wins: result.total_wins,
            win_rate: wr,
            profit_factor: pf,
            max_drawdown_pct: result.max_drawdown_pct as f64,
            overrides: overrides.clone(),
        });
    }

    out
}

// =============================================================================
// Main entry point
// =============================================================================

/// Run a TPE-guided GPU sweep with double-buffer pipeline.
///
/// TPE sampling (CPU) overlaps with GPU evaluation using a dedicated TPE thread
/// and crossbeam channels. Protocol:
///   1. TPE thread samples batch 0 (priming)
///   2. GPU evaluates batch 0 while TPE thread idles
///   3. GPU sends results → TPE thread tells batch 0, samples batch 1
///   4. GPU evaluates batch 1 while TPE samples batch 2 (full overlap)
///   5. Repeat until all trials done
pub fn run_tpe_sweep(
    candles: &CandleData,
    base_cfg: &StrategyConfig,
    spec: &SweepSpec,
    _funding: Option<&FundingRateData>,
    tpe_cfg: &TpeConfig,
    sub_candles: Option<&CandleData>,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    top_k: usize,
) -> Vec<GpuSweepResult> {
    let rng = StdRng::seed_from_u64(tpe_cfg.seed);

    // -- 1. Create per-axis TPE optimizers ----------------------------------------
    let mut axis_opts: Vec<AxisOptimizer> = Vec::new();
    for axis in &spec.axes {
        let min_val = axis.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = axis
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-12 {
            axis_opts.push(AxisOptimizer {
                path: axis.path.clone(),
                min_val,
                max_val,
                optimizer: tpe::TpeOptimizer::new(
                    tpe::parzen_estimator(),
                    tpe::range(min_val - 0.5, max_val + 0.5).unwrap(),
                ),
                is_indicator: is_indicator_path(&axis.path),
                is_integer: is_integer_axis(&axis.path),
            });
            continue;
        }

        let optimizer = tpe::TpeOptimizer::new(
            tpe::parzen_estimator(),
            tpe::range(min_val, max_val).unwrap(),
        );

        axis_opts.push(AxisOptimizer {
            path: axis.path.clone(),
            min_val,
            max_val,
            optimizer,
            is_indicator: is_indicator_path(&axis.path),
            is_integer: is_integer_axis(&axis.path),
        });
    }

    let has_indicator_axes = axis_opts.iter().any(|a| a.is_indicator);

    eprintln!(
        "[TPE] {} axes ({} indicator, {} trade), {} trials, batch={}",
        axis_opts.len(),
        axis_opts.iter().filter(|a| a.is_indicator).count(),
        axis_opts.iter().filter(|a| !a.is_indicator).count(),
        tpe_cfg.trials,
        tpe_cfg.batch_size,
    );
    for a in &axis_opts {
        eprintln!(
            "  - {} [{:.4}..{:.4}] {}{}",
            a.path,
            a.min_val,
            a.max_val,
            if a.is_indicator { "(indicator) " } else { "" },
            if a.is_integer { "(int)" } else { "" },
        );
    }

    // -- 2. Init GPU, upload raw candles ------------------------------------------
    let symbols = crate::sorted_symbols_with_kernel_cap(candles, "[TPE]");
    let num_symbols = symbols.len();
    let btc_sym_idx = symbols
        .iter()
        .position(|s| s == "BTC")
        .or_else(|| symbols.iter().position(|s| s == "BTCUSDT"))
        .map(|idx| idx as u32)
        .unwrap_or(u32::MAX);

    let raw = raw_candles::prepare_raw_candles(candles, &symbols);
    let num_bars = raw.num_bars as u32;

    let (trade_start, trade_end) =
        raw_candles::find_trade_bar_range(&raw.timestamps, from_ts, to_ts);
    eprintln!(
        "[TPE] Trade bar range: {}..{} ({} of {} bars)",
        trade_start,
        trade_end,
        trade_end - trade_start,
        num_bars,
    );

    // C6: graceful fallback — return empty results if GPU init fails
    let device_state = match gpu_host::GpuDeviceState::new() {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("[TPE] {e} — GPU sweep unavailable, returning empty results");
            return Vec::new();
        }
    };
    let candles_gpu = device_state.dev.htod_sync_copy(&raw.candles).unwrap();

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
            "[TPE] Sub-bar candles: max_sub={}, GPU buffer uploaded",
            max_sub_per_bar,
        );
    }

    eprintln!(
        "[TPE] {} bars x {} symbols, CUDA device ready",
        num_bars, num_symbols,
    );

    // -- Layer 1: Fixed VRAM budget from total_vram (40%) -------------------------
    let total_vram = device_state.total_vram_bytes();
    let snapshot_stride = (num_bars as usize) * (num_symbols as usize);
    let breadth_stride = num_bars as usize;

    let snap_bytes_per_ind = snapshot_stride * std::mem::size_of::<buffers::GpuSnapshot>();
    let breadth_bytes_per_ind = breadth_stride * std::mem::size_of::<f32>();
    let btc_bytes_per_ind = breadth_stride * std::mem::size_of::<u32>();
    let ind_bytes_per_slot = snap_bytes_per_ind + breadth_bytes_per_ind + btc_bytes_per_ind;

    let vram_budget = 22usize * 1024 * 1024 * 1024;
    let arena_cap = if ind_bytes_per_slot > 0 {
        (vram_budget / ind_bytes_per_slot).max(1)
    } else {
        tpe_cfg.batch_size
    };

    let arena_cap = if snap_bytes_per_ind > 0 {
        let max_by_alloc = (20usize * 1024 * 1024 * 1024) / snap_bytes_per_ind;
        arena_cap.min(max_by_alloc).max(1)
    } else {
        arena_cap
    };

    eprintln!(
        "[TPE] VRAM budget: {:.1} GB (of {:.1} GB total), arena_cap={} ind slots ({:.1} MB/slot)",
        vram_budget as f64 / 1e9,
        total_vram as f64 / 1e9,
        arena_cap,
        ind_bytes_per_slot as f64 / 1e6,
    );

    // Pre-allocate arena buffers
    let mut snap_arena: CudaSlice<buffers::GpuSnapshot> = device_state
        .dev
        .alloc_zeros::<buffers::GpuSnapshot>(arena_cap * snapshot_stride)
        .unwrap();
    let mut breadth_arena: CudaSlice<f32> = device_state
        .dev
        .alloc_zeros::<f32>(arena_cap * breadth_stride)
        .unwrap();
    let mut btc_arena: CudaSlice<u32> = device_state
        .dev
        .alloc_zeros::<u32>(arena_cap * breadth_stride)
        .unwrap();

    // -- 3. Double-buffer pipeline ------------------------------------------------
    // Request channel: bounded(2) so TPE can always have one pre-fetched batch
    // queued while GPU processes the current one. This is the core of the
    // double-buffer: GPU finishes → immediately grabs pre-fetched batch → zero stall.
    let (request_tx, request_rx) = crossbeam_channel::bounded::<Option<TpeBatchRequest>>(2);
    let (result_tx, result_rx) = crossbeam_channel::bounded::<GpuBatchResult>(1);

    let tpe_trials = tpe_cfg.trials;
    let tpe_batch_size = tpe_cfg.batch_size;

    // Spawn TPE thread — takes ownership of axis_opts + rng
    let tpe_handle = std::thread::Builder::new()
        .name("tpe-sampler".into())
        .spawn(move || {
            tpe_worker(
                axis_opts,
                rng,
                tpe_trials,
                tpe_batch_size,
                result_rx,
                request_tx,
            );
        })
        .expect("Failed to spawn TPE thread");

    // GPU thread: receive batches, evaluate, send results
    // Bounded top-K heap: keeps only the best results by PnL to prevent OOM
    let effective_top_k = if top_k == 0 { usize::MAX } else { top_k };
    let mut top_heap: BinaryHeap<MinPnlHeapEntry> = BinaryHeap::with_capacity(
        effective_top_k.min(tpe_cfg.trials).min(100_000) + 1,
    );
    let mut best_pnl = f64::NEG_INFINITY;
    let mut best_trial = 0usize;
    let mut trials_done = 0usize;

    eprintln!("[TPE] Pipeline mode: double-buffer (TPE thread + GPU thread)");

    while let Ok(Some(request)) = request_rx.recv() {
        let t_gpu = Instant::now();

        // Evaluate batch on GPU
        let gpu_results = if has_indicator_axes {
            evaluate_mixed_batch_arena(
                &device_state,
                &candles_gpu,
                base_cfg,
                &request.trial_overrides,
                num_bars,
                num_symbols as u32,
                btc_sym_idx,
                spec.lookback,
                spec.initial_balance as f32,
                arena_cap,
                snapshot_stride,
                breadth_stride,
                &mut snap_arena,
                &mut breadth_arena,
                &mut btc_arena,
                max_sub_per_bar,
                sub_candles_gpu.as_ref(),
                sub_counts_gpu.as_ref(),
                trade_start,
                trade_end,
            )
        } else {
            evaluate_trade_only_batch(
                &device_state,
                &candles_gpu,
                base_cfg,
                &request.trial_overrides,
                num_bars,
                num_symbols as u32,
                btc_sym_idx,
                spec.lookback,
                spec.initial_balance as f32,
                max_sub_per_bar,
                sub_candles_gpu.as_ref(),
                sub_counts_gpu.as_ref(),
                trade_start,
                trade_end,
            )
        };

        let gpu_ms = t_gpu.elapsed().as_millis();

        // Build GpuSweepResult on GPU thread (no need to send back through channel)
        let batch_results = build_sweep_results(
            &gpu_results,
            &request.trial_overrides,
            &mut best_pnl,
            &mut best_trial,
            trials_done,
        );

        trials_done += request.batch_n;

        eprintln!(
            "[TPE] {}/{} trials done, best PnL: ${:.2} (trial #{}) | gpu={}ms batch={}",
            trials_done, tpe_trials, best_pnl, best_trial, gpu_ms, request.batch_idx,
        );

        for r in batch_results {
            if top_heap.len() < effective_top_k {
                top_heap.push(MinPnlHeapEntry(r));
            } else if let Some(worst) = top_heap.peek() {
                if r.total_pnl > worst.0.total_pnl || worst.0.total_pnl.is_nan() {
                    top_heap.pop();
                    top_heap.push(MinPnlHeapEntry(r));
                }
            }
        }

        // Send results to TPE thread (tell + ask next batch)
        // This returns immediately — TPE thread will process while we wait for next request
        if result_tx
            .send(GpuBatchResult {
                batch_idx: request.batch_idx,
                results: gpu_results,
                trial_overrides: request.trial_overrides,
                trial_raw_values: request.trial_raw_values,
            })
            .is_err()
        {
            break; // TPE thread exited
        }
    }

    // Wait for TPE thread to finish
    tpe_handle.join().expect("TPE thread panicked");

    // Drain heap and sort by PnL descending
    let mut results: Vec<GpuSweepResult> = top_heap.into_iter().map(|e| e.0).collect();
    results.sort_by(|a, b| {
        b.total_pnl
            .partial_cmp(&a.total_pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

// =============================================================================
// GPU batch evaluation helpers
// =============================================================================

/// Evaluate a batch of trials where all share the same indicator config
/// (no indicator axes in the sweep). Single indicator dispatch, N trade combos.
fn evaluate_trade_only_batch(
    ds: &gpu_host::GpuDeviceState,
    candles_gpu: &CudaSlice<buffers::GpuRawCandle>,
    base_cfg: &StrategyConfig,
    trial_overrides: &[Vec<(String, f64)>],
    num_bars: u32,
    num_symbols: u32,
    btc_sym_idx: u32,
    lookback: usize,
    initial_balance: f32,
    max_sub_per_bar: u32,
    sub_candles_gpu: Option<&CudaSlice<buffers::GpuRawCandle>>,
    sub_counts_gpu: Option<&CudaSlice<u32>>,
    trade_start: u32,
    trade_end: u32,
) -> Vec<buffers::GpuResult> {
    let ind_config = buffers::GpuIndicatorConfig::from_strategy_config(base_cfg, lookback);

    // H11: handle GPU memory allocation failure gracefully
    let mut ind_bufs = match gpu_host::IndicatorBuffers::new(
        ds,
        candles_gpu,
        &[ind_config],
        num_bars,
        num_symbols,
        btc_sym_idx,
    ) {
        Ok(bufs) => bufs,
        Err(e) => {
            eprintln!("[TPE] Indicator buffer allocation failed: {e}");
            return Vec::new();
        }
    };
    if let Err(e) = gpu_host::dispatch_indicator_kernels(ds, &mut ind_bufs) {
        eprintln!("[TPE] Indicator kernel dispatch failed: {e}");
        return Vec::new();
    }

    let gpu_configs: Vec<buffers::GpuComboConfig> = trial_overrides
        .iter()
        .map(|overrides| {
            let mut cfg = base_cfg.clone();
            for (path, value) in overrides {
                bt_core::sweep::apply_one_pub(&mut cfg, path, *value);
            }
            let mut gpu_cfg = buffers::GpuComboConfig::from_strategy_config(&cfg);
            gpu_cfg.snapshot_offset = 0;
            gpu_cfg.breadth_offset = 0;
            gpu_cfg
        })
        .collect();

    let mut trade_bufs = match gpu_host::BatchBuffers::from_indicator_buffers(
        ds,
        &ind_bufs,
        &gpu_configs,
        initial_balance,
        0,
    ) {
        Ok(bufs) => bufs,
        Err(e) => {
            eprintln!("[TPE] Trade buffer allocation failed: {e}");
            return Vec::new();
        }
    };
    trade_bufs.max_sub_per_bar = max_sub_per_bar;
    trade_bufs.sub_candles = sub_candles_gpu.cloned();
    trade_bufs.sub_counts = sub_counts_gpu.cloned();

    match gpu_host::dispatch_and_readback(
        ds,
        &mut trade_bufs,
        num_bars,
        BAR_CHUNK_SIZE,
        trade_start,
        trade_end,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[TPE] Trade kernel dispatch/readback failed: {e}");
            Vec::new()
        }
    }
}

/// Evaluate a batch of trials where indicator params may vary per trial.
/// Uses Layer 1 (arena buffers) + Layer 2 (indicator config dedup).
fn evaluate_mixed_batch_arena(
    ds: &gpu_host::GpuDeviceState,
    candles_gpu: &CudaSlice<buffers::GpuRawCandle>,
    base_cfg: &StrategyConfig,
    trial_overrides: &[Vec<(String, f64)>],
    num_bars: u32,
    num_symbols: u32,
    btc_sym_idx: u32,
    lookback: usize,
    initial_balance: f32,
    arena_cap: usize,
    snapshot_stride: usize,
    breadth_stride: usize,
    snap_arena: &mut CudaSlice<buffers::GpuSnapshot>,
    breadth_arena: &mut CudaSlice<f32>,
    btc_arena: &mut CudaSlice<u32>,
    max_sub_per_bar: u32,
    sub_candles_gpu: Option<&CudaSlice<buffers::GpuRawCandle>>,
    sub_counts_gpu: Option<&CudaSlice<u32>>,
    trade_start: u32,
    trade_end: u32,
) -> Vec<buffers::GpuResult> {
    let n = trial_overrides.len();

    let trial_cfgs: Vec<StrategyConfig> = trial_overrides
        .iter()
        .map(|overrides| {
            let mut cfg = base_cfg.clone();
            for (path, value) in overrides {
                bt_core::sweep::apply_one_pub(&mut cfg, path, *value);
            }
            cfg
        })
        .collect();

    let trial_ind_cfgs: Vec<buffers::GpuIndicatorConfig> = trial_cfgs
        .iter()
        .map(|cfg| buffers::GpuIndicatorConfig::from_strategy_config(cfg, lookback))
        .collect();

    // Layer 2: Indicator config dedup
    let mut dedup_map: HashMap<[u8; 80], usize> = HashMap::new();
    let mut unique_ind_cfgs: Vec<buffers::GpuIndicatorConfig> = Vec::new();
    let mut trial_to_unique: Vec<usize> = Vec::with_capacity(n);

    for ind_cfg in &trial_ind_cfgs {
        let bytes: &[u8] = bytemuck::bytes_of(ind_cfg);
        let mut key = [0u8; 80];
        key.copy_from_slice(bytes);

        let unique_idx = if let Some(&idx) = dedup_map.get(&key) {
            idx
        } else {
            let idx = unique_ind_cfgs.len();
            dedup_map.insert(key, idx);
            unique_ind_cfgs.push(*ind_cfg);
            idx
        };
        trial_to_unique.push(unique_idx);
    }

    let num_unique = unique_ind_cfgs.len();
    if num_unique < n {
        eprintln!(
            "[TPE-dedup] {} trials -> {} unique indicator configs ({:.0}% dedup)",
            n,
            num_unique,
            (1.0 - num_unique as f64 / n as f64) * 100.0,
        );
    }

    let mut all_results: Vec<buffers::GpuResult> = vec![buffers::GpuResult::zeroed(); n];

    let num_groups = (num_unique + arena_cap - 1) / arena_cap;
    for group_idx in 0..num_groups {
        let group_start = group_idx * arena_cap;
        let group_end = (group_start + arena_cap).min(num_unique);
        let group_ind_cfgs = &unique_ind_cfgs[group_start..group_end];

        dispatch_indicator_arena(
            ds,
            candles_gpu,
            group_ind_cfgs,
            num_bars,
            num_symbols,
            btc_sym_idx,
            snap_arena,
            breadth_arena,
            btc_arena,
        );

        let mut group_trial_indices: Vec<usize> = Vec::new();
        let mut gpu_configs: Vec<buffers::GpuComboConfig> = Vec::new();

        for (trial_idx, &unique_slot) in trial_to_unique.iter().enumerate() {
            if unique_slot >= group_start && unique_slot < group_end {
                let local_slot = unique_slot - group_start;
                let mut gpu_cfg =
                    buffers::GpuComboConfig::from_strategy_config(&trial_cfgs[trial_idx]);
                gpu_cfg.snapshot_offset = (local_slot * snapshot_stride) as u32;
                gpu_cfg.breadth_offset = (local_slot * breadth_stride) as u32;
                gpu_configs.push(gpu_cfg);
                group_trial_indices.push(trial_idx);
            }
        }

        if gpu_configs.is_empty() {
            continue;
        }

        let trade_results = dispatch_trade_arena(
            ds,
            snap_arena,
            breadth_arena,
            btc_arena,
            &gpu_configs,
            initial_balance,
            num_symbols,
            btc_sym_idx,
            num_bars,
            max_sub_per_bar,
            sub_candles_gpu,
            sub_counts_gpu,
            trade_start,
            trade_end,
        );

        for (local_idx, &trial_idx) in group_trial_indices.iter().enumerate() {
            all_results[trial_idx] = trade_results[local_idx];
        }
    }

    all_results
}

/// Dispatch indicator_kernel + breadth_kernel into pre-allocated arena buffers.
fn dispatch_indicator_arena(
    ds: &gpu_host::GpuDeviceState,
    candles_gpu: &CudaSlice<buffers::GpuRawCandle>,
    ind_configs: &[buffers::GpuIndicatorConfig],
    num_bars: u32,
    num_symbols: u32,
    btc_sym_idx: u32,
    snap_arena: &mut CudaSlice<buffers::GpuSnapshot>,
    breadth_arena: &mut CudaSlice<f32>,
    btc_arena: &mut CudaSlice<u32>,
) {
    let k = ind_configs.len() as u32;
    let block_size = 64u32;

    let ind_configs_gpu = ds.dev.htod_sync_copy(ind_configs).unwrap();

    let params = buffers::IndicatorParams {
        num_ind_combos: k,
        num_symbols,
        num_bars,
        btc_sym_idx,
        _pad: [0; 4],
    };
    let ind_params_gpu = ds.dev.htod_sync_copy(&[params]).unwrap();

    let ind_threads = k * num_symbols;
    let ind_grid = (ind_threads + block_size - 1) / block_size;

    let ind_func: CudaFunction = ds
        .dev
        .get_func("indicators", "indicator_kernel")
        .expect("indicator_kernel not found in PTX");

    let ind_cfg = LaunchConfig {
        grid_dim: (ind_grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        ind_func.launch(
            ind_cfg,
            (
                &ind_params_gpu,
                &ind_configs_gpu,
                candles_gpu,
                &mut *snap_arena,
            ),
        )
    }
    .expect("indicator_kernel launch failed");

    let br_threads = k * num_bars;
    let br_grid = (br_threads + block_size - 1) / block_size;

    let br_func: CudaFunction = ds
        .dev
        .get_func("indicators", "breadth_kernel")
        .expect("breadth_kernel not found in PTX");

    let br_cfg = LaunchConfig {
        grid_dim: (br_grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        br_func.launch(
            br_cfg,
            (
                &ind_params_gpu,
                &*snap_arena,
                &mut *breadth_arena,
                &mut *btc_arena,
            ),
        )
    }
    .expect("breadth_kernel launch failed");
}

/// Dispatch trade sweep kernel using pre-allocated arena snapshot/breadth/btc buffers.
fn dispatch_trade_arena(
    ds: &gpu_host::GpuDeviceState,
    snap_arena: &mut CudaSlice<buffers::GpuSnapshot>,
    breadth_arena: &mut CudaSlice<f32>,
    btc_arena: &mut CudaSlice<u32>,
    gpu_configs: &[buffers::GpuComboConfig],
    initial_balance: f32,
    num_symbols: u32,
    btc_sym_idx: u32,
    num_bars: u32,
    max_sub_per_bar: u32,
    sub_candles_gpu: Option<&CudaSlice<buffers::GpuRawCandle>>,
    sub_counts_gpu: Option<&CudaSlice<u32>>,
    trade_start: u32,
    trade_end: u32,
) -> Vec<buffers::GpuResult> {
    let num_combos = gpu_configs.len() as u32;
    let block_size = 64u32;
    let grid_size = (num_combos + block_size - 1) / block_size;

    let configs_gpu = ds.dev.htod_sync_copy(gpu_configs).unwrap();

    let mut states_host = vec![buffers::GpuComboState::zeroed(); gpu_configs.len()];
    for s in &mut states_host {
        s.balance = initial_balance as f64;
        s.peak_equity = initial_balance as f64;
    }
    let mut states_gpu = ds.dev.htod_sync_copy(&states_host).unwrap();
    let mut results_gpu = ds
        .dev
        .alloc_zeros::<buffers::GpuResult>(gpu_configs.len())
        .unwrap();

    let sentinel_candle: CudaSlice<buffers::GpuRawCandle>;
    let sentinel_counts: CudaSlice<u32>;
    let (sc_ref, sn_ref) = match (sub_candles_gpu, sub_counts_gpu) {
        (Some(sc), Some(sn)) => (sc, sn),
        _ => {
            sentinel_candle = ds.dev.alloc_zeros::<buffers::GpuRawCandle>(1).unwrap();
            sentinel_counts = ds.dev.alloc_zeros::<u32>(1).unwrap();
            (&sentinel_candle, &sentinel_counts)
        }
    };

    let effective_chunk = if max_sub_per_bar > 0 {
        BAR_CHUNK_SIZE.min(50)
    } else {
        BAR_CHUNK_SIZE
    };
    let trade_range = trade_end - trade_start;
    let num_chunks = (trade_range + effective_chunk - 1) / effective_chunk;

    for chunk_idx in 0..num_chunks {
        let chunk_start = trade_start + chunk_idx * effective_chunk;
        let chunk_end = (chunk_start + effective_chunk).min(trade_end);

        let params_host = buffers::GpuParams {
            num_combos,
            num_symbols,
            num_bars,
            btc_sym_idx,
            chunk_start,
            chunk_end,
            initial_balance_bits: initial_balance.to_bits(),
            maker_fee_rate_bits: 0.00035f32.to_bits(),
            taker_fee_rate_bits: 0.00035f32.to_bits(),
            max_sub_per_bar,
            trade_end_bar: trade_end,
        };
        let params_gpu = ds.dev.htod_sync_copy(&[params_host]).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let func: CudaFunction = ds
            .dev
            .get_func("sweep", "sweep_engine_kernel")
            .expect("sweep_engine_kernel not found in PTX");

        unsafe {
            func.launch(
                cfg,
                (
                    &params_gpu,
                    &*snap_arena,
                    &*breadth_arena,
                    &*btc_arena,
                    &configs_gpu,
                    &mut states_gpu,
                    &mut results_gpu,
                    sc_ref,
                    sn_ref,
                ),
            )
        }
        .expect("sweep_engine_kernel launch failed");
    }

    ds.dev.dtoh_sync_copy(&results_gpu).unwrap()
}

// =============================================================================
// Helpers
// =============================================================================

fn is_indicator_path(path: &str) -> bool {
    INDICATOR_PATHS.contains(&path)
}

/// Integer axes are indicator windows and a few other discrete params.
fn is_integer_axis(path: &str) -> bool {
    path.starts_with("indicators.")
        || path.contains("_window")
        || path == "trade.max_open_positions"
        || path == "trade.max_entry_orders_per_loop"
        || path == "trade.pyramid_max_adds"
}
