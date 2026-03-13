use bt_core::candle::{CandleData, OhlcvBar};
use bt_core::config::{
    BehaviourGroupConfig, BehaviourProfileConfig, Confidence, MacdMode, PipelineConfig,
    PipelineProfileConfig, RuntimeConfig, StrategyConfig,
};
use bt_core::engine::{run_simulation, RunSimulationInput};
use bt_core::sweep::{SweepAxis, SweepSpec};
use bt_gpu::buffers::{GpuComboState, GpuTraceEvent, GPU_TRACE_CAP};
use bt_gpu::run_gpu_sweep_with_states;
use bt_gpu::tpe_sweep::{run_tpe_sweep, TpeConfig};
use cudarc::driver::CudaDevice;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::Mutex;

const TRACE_KIND_CLOSE: u32 = 3;
const TRACE_KIND_PARTIAL: u32 = 4;
const TRACE_REASON_PARTIAL: u32 = 8;
const TRACE_REASON_EXIT_TRAILING: u32 = 4;
const TRACE_REASON_EXIT_TP: u32 = 5;

static GPU_TRACE_ENV_LOCK: Mutex<()> = Mutex::new(());

fn bar_1h(t: i64, close: f64) -> OhlcvBar {
    OhlcvBar {
        t,
        t_close: t + 3_600_000,
        o: close,
        h: close + 1.0,
        l: close - 1.0,
        c: close,
        v: 1_000.0,
        n: 1,
    }
}

fn sub_bar_3m(t: i64, close: f64) -> OhlcvBar {
    OhlcvBar {
        t,
        t_close: t + 180_000,
        o: close,
        h: close,
        l: close,
        c: close,
        v: 1_000.0,
        n: 1,
    }
}

fn linear_sub_bars(start_t: i64, from_close: f64, to_close: f64) -> Vec<OhlcvBar> {
    (0..20)
        .map(|slot| {
            let alpha = (slot as f64 + 1.0) / 20.0;
            let close = from_close + (to_close - from_close) * alpha;
            sub_bar_3m(start_t + ((slot as i64) + 1) * 180_000, close)
        })
        .collect()
}

fn permissive_cfg(profile: &str) -> StrategyConfig {
    let mut cfg = StrategyConfig::default();
    cfg.filters.enable_ranging_filter = false;
    cfg.filters.enable_anomaly_filter = false;
    cfg.filters.enable_extension_filter = false;
    cfg.filters.require_adx_rising = false;
    cfg.filters.require_btc_alignment = false;
    cfg.filters.use_stoch_rsi_filter = false;
    cfg.filters.require_macro_alignment = false;
    cfg.thresholds.entry.min_adx = -1.0;
    cfg.thresholds.entry.macd_hist_entry_mode = MacdMode::None;
    cfg.trade.entry_min_confidence = Confidence::Low;
    cfg.trade.enable_ssf_filter = false;
    cfg.trade.enable_reef_filter = false;
    cfg.trade.enable_dynamic_sizing = false;
    cfg.trade.leverage = 1.0;
    cfg.trade.allocation_pct = 0.10;
    cfg.trade.slippage_bps = 0.0;
    cfg.trade.enable_pyramiding = false;
    cfg.trade.min_atr_pct = 0.0;
    cfg.indicators.ema_fast_window = 2;
    cfg.indicators.ema_slow_window = 3;
    cfg.indicators.adx_window = 3;
    cfg.indicators.bb_window = 3;
    cfg.runtime = RuntimeConfig {
        profile: profile.to_string(),
        ..RuntimeConfig::default()
    };
    cfg
}

fn collect_trace_events(state: &GpuComboState) -> Vec<GpuTraceEvent> {
    let count = state.trace_count.min(GPU_TRACE_CAP as u32) as usize;
    let start = (state.trace_head as usize).saturating_sub(count) % GPU_TRACE_CAP;
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(state.trace_events[(start + idx) % GPU_TRACE_CAP]);
    }
    out
}

fn first_exit_trace(state: &GpuComboState) -> GpuTraceEvent {
    collect_trace_events(state)
        .into_iter()
        .find(|ev| ev.kind == TRACE_KIND_CLOSE || ev.kind == TRACE_KIND_PARTIAL)
        .expect("expected at least one GPU exit trace event")
}

fn exit_trace_events(state: &GpuComboState) -> Vec<GpuTraceEvent> {
    collect_trace_events(state)
        .into_iter()
        .filter(|ev| ev.kind == TRACE_KIND_CLOSE || ev.kind == TRACE_KIND_PARTIAL)
        .collect()
}

fn with_gpu_trace<T>(symbol_idx: u32, f: impl FnOnce() -> T) -> T {
    let _guard = GPU_TRACE_ENV_LOCK.lock().expect("trace env mutex poisoned");
    const KEYS: [&str; 3] = [
        "AQC_GPU_TRACE",
        "AQC_GPU_TRACE_COMBO",
        "AQC_GPU_TRACE_SYMBOL",
    ];
    let mut prev: Vec<(&str, Option<OsString>)> = Vec::with_capacity(KEYS.len());
    for key in KEYS {
        prev.push((key, std::env::var_os(key)));
    }

    std::env::set_var("AQC_GPU_TRACE", "1");
    std::env::set_var("AQC_GPU_TRACE_COMBO", "0");
    std::env::set_var("AQC_GPU_TRACE_SYMBOL", symbol_idx.to_string());

    let out = f();

    for (key, val) in prev {
        if let Some(v) = val {
            std::env::set_var(key, v);
        } else {
            std::env::remove_var(key);
        }
    }
    out
}

fn sweep_spec() -> SweepSpec {
    SweepSpec {
        axes: vec![],
        initial_balance: 1_000.0,
        lookback: 0,
    }
}

fn build_sub_bar_trailing_fixture() -> (CandleData, CandleData) {
    let hour = 3_600_000i64;
    let mut closes = vec![100.0; 70];
    closes.extend([
        100.5, 101.0, 101.8, 102.6, 103.4, 104.2, 105.0, 105.8, 106.6, 107.4, 108.2,
    ]);
    let main = closes
        .iter()
        .enumerate()
        .map(|(idx, close)| bar_1h(idx as i64 * hour, *close))
        .collect::<Vec<_>>();
    let mut sub = Vec::new();
    for idx in 1..(closes.len() - 1) {
        sub.extend(linear_sub_bars(
            (idx as i64 - 1) * hour,
            closes[idx - 1],
            closes[idx],
        ));
    }
    let start = (closes.len() as i64 - 2) * hour;
    for (slot, close) in [
        109.0, 110.0, 111.0, 110.6, 110.0, 109.4, 108.8, 108.2, 107.8, 107.4,
    ]
    .into_iter()
    .enumerate()
    {
        sub.push(sub_bar_3m(start + ((slot as i64) + 1) * 180_000, close));
    }

    let mut candles: CandleData = FxHashMap::default();
    candles.insert("BTC".to_string(), main);
    let mut sub_candles: CandleData = FxHashMap::default();
    sub_candles.insert("BTC".to_string(), sub);
    (candles, sub_candles)
}

fn build_sub_bar_tp_continuation_fixture() -> (CandleData, CandleData) {
    let hour = 3_600_000i64;
    let mut closes = vec![100.0; 70];
    closes.extend([
        100.4, 100.8, 101.2, 101.6, 102.0, 102.4, 102.8, 103.2, 103.6, 104.0, 104.4,
    ]);
    let main = closes
        .iter()
        .enumerate()
        .map(|(idx, close)| bar_1h(idx as i64 * hour, *close))
        .collect::<Vec<_>>();
    let mut sub = Vec::new();
    for idx in 1..(closes.len() - 1) {
        sub.extend(linear_sub_bars(
            (idx as i64 - 1) * hour,
            closes[idx - 1],
            closes[idx],
        ));
    }
    let start = (closes.len() as i64 - 2) * hour;
    for (slot, close) in [105.0, 106.5, 108.0, 109.5, 111.0, 112.5, 113.0]
        .into_iter()
        .enumerate()
    {
        sub.push(sub_bar_3m(start + ((slot as i64) + 1) * 180_000, close));
    }

    let mut candles: CandleData = FxHashMap::default();
    candles.insert("BTC".to_string(), main);
    let mut sub_candles: CandleData = FxHashMap::default();
    sub_candles.insert("BTC".to_string(), sub);
    (candles, sub_candles)
}

fn sweep_engine_source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels/sweep_engine.cu"
    ))
    .expect("sweep_engine.cu should be readable from crate tests")
}

#[test]
fn sweep_engine_uses_ordered_exit_helper_in_actual_loops() {
    let src = sweep_engine_source();
    assert!(
        src.contains("OrderedExitDecision evaluate_ordered_exits("),
        "sweep_engine must define the shared ordered exit helper"
    );
    assert_eq!(
        src.matches("OrderedExitDecision exit_decision = evaluate_ordered_exits(")
            .count(),
        2,
        "main-bar and sub-bar loops must both route through the ordered helper"
    );
    assert!(
        !src.contains("if (check_stop_loss(pos, snap, &cfg))"),
        "main-bar loop must not keep the legacy fixed stop-loss call path"
    );
    assert!(
        !src.contains("if (check_stop_loss(pos, hybrid, &cfg))"),
        "sub-bar loop must not keep the legacy fixed stop-loss call path"
    );
    assert!(
        !src.contains("unsigned int tp_result = check_tp(pos, snap, &cfg, tp_mult);"),
        "main-bar loop must not keep the legacy fixed TP path"
    );
    assert!(
        !src.contains("unsigned int tp_result = check_tp(pos, hybrid, &cfg, tp_mult);"),
        "sub-bar loop must not keep the legacy fixed TP path"
    );
}

#[test]
fn cpu_gpu_main_bar_exit_order_prefers_full_before_partial_take_profit() {
    if let Err(err) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", err);
        return;
    }

    let profile = "gpu_mainbar_full_before_partial";
    let mut cfg = permissive_cfg(profile);
    cfg.trade.tp_atr_mult = 20.0;
    cfg.trade.enable_partial_tp = true;
    cfg.trade.tp_partial_pct = 0.5;
    cfg.trade.tp_partial_atr_mult = 19.5;
    cfg.trade.tp_partial_min_notional_usd = 1.0;
    cfg.trade.enable_vol_buffered_trailing = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    cfg.trade.tsme_min_profit_atr = 999.0;
    cfg.pipeline = PipelineConfig {
        default_profile: "production".to_string(),
        profiles: BTreeMap::from([(
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
                            "exit.take_profit.full".to_string(),
                            "exit.take_profit.partial".to_string(),
                        ],
                        disabled: vec![
                            "exit.trailing.low_conf_override".to_string(),
                            "exit.trailing.vol_buffer".to_string(),
                            "exit.trailing.base".to_string(),
                            "exit.smart.trend_exhaustion".to_string(),
                            "exit.smart.ema_macro_breakdown".to_string(),
                            "exit.smart.stagnation".to_string(),
                            "exit.smart.funding_headwind".to_string(),
                            "exit.smart.tsme".to_string(),
                            "exit.smart.mmde".to_string(),
                            "exit.smart.rsi_overextension".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        )]),
    };

    let hour = 3_600_000i64;
    let mut bars = Vec::new();
    for idx in 0..180 {
        bars.push(bar_1h(idx as i64 * hour, 100.0));
    }
    for idx in 180..199 {
        bars.push(bar_1h(idx as i64 * hour, 100.0 + (idx - 179) as f64));
    }
    bars.push(bar_1h(199 * hour, 160.0));
    let mut candles: CandleData = FxHashMap::default();
    candles.insert("BTC".to_string(), bars);

    let cpu = run_simulation(RunSimulationInput {
        candles: &candles,
        cfg: &cfg,
        initial_balance: 1_000.0,
        lookback: 0,
        exit_candles: None,
        entry_candles: None,
        funding_rates: None,
        init_state: None,
        from_ts: None,
        to_ts: None,
    });
    let cpu_exit = cpu
        .trades
        .iter()
        .find(|trade| trade.is_close())
        .expect("CPU run should emit a close");
    assert!(
        cpu_exit.action.starts_with("CLOSE"),
        "CPU main-bar path should full-close before partial reduce, got `{}`",
        cpu_exit.action
    );
    assert!(
        cpu_exit.reason.contains("Take Profit"),
        "CPU should close via full take profit first, got `{}`",
        cpu_exit.reason
    );

    let (_rows, states, _symbols) = with_gpu_trace(0, || {
        run_gpu_sweep_with_states(&candles, &cfg, &sweep_spec(), None, None, None, None)
    });
    let state = states
        .first()
        .copied()
        .expect("GPU run should return one combo state");
    let gpu_exit = first_exit_trace(&state);
    assert_eq!(
        gpu_exit.reason, TRACE_REASON_EXIT_TP,
        "GPU main-bar loop must honour full-before-partial take-profit order"
    );
}

#[test]
fn tpe_smoke_respects_reordered_exit_profile_via_shared_gpu_contract() {
    if let Err(err) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", err);
        return;
    }

    let profile = "gpu_tpe_full_before_partial";
    let mut cfg = permissive_cfg(profile);
    cfg.trade.tp_atr_mult = 20.0;
    cfg.trade.enable_partial_tp = true;
    cfg.trade.tp_partial_pct = 0.5;
    cfg.trade.tp_partial_atr_mult = 19.5;
    cfg.trade.tp_partial_min_notional_usd = 1.0;
    cfg.trade.enable_vol_buffered_trailing = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    cfg.trade.tsme_min_profit_atr = 999.0;
    cfg.pipeline = PipelineConfig {
        default_profile: "production".to_string(),
        profiles: BTreeMap::from([(
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
                            "exit.take_profit.full".to_string(),
                            "exit.take_profit.partial".to_string(),
                        ],
                        disabled: vec![
                            "exit.trailing.low_conf_override".to_string(),
                            "exit.trailing.vol_buffer".to_string(),
                            "exit.trailing.base".to_string(),
                            "exit.smart.trend_exhaustion".to_string(),
                            "exit.smart.ema_macro_breakdown".to_string(),
                            "exit.smart.stagnation".to_string(),
                            "exit.smart.funding_headwind".to_string(),
                            "exit.smart.tsme".to_string(),
                            "exit.smart.mmde".to_string(),
                            "exit.smart.rsi_overextension".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        )]),
    };

    let hour = 3_600_000i64;
    let mut bars = Vec::new();
    for idx in 0..180 {
        bars.push(bar_1h(idx as i64 * hour, 100.0));
    }
    for idx in 180..199 {
        bars.push(bar_1h(idx as i64 * hour, 100.0 + (idx - 179) as f64));
    }
    bars.push(bar_1h(199 * hour, 160.0));
    let mut candles: CandleData = FxHashMap::default();
    candles.insert("BTC".to_string(), bars);

    let spec = SweepSpec {
        axes: vec![SweepAxis {
            path: "trade.leverage".to_string(),
            values: vec![1.0],
            gate: None,
        }],
        initial_balance: 1_000.0,
        lookback: 0,
    };

    let tpe = run_tpe_sweep(
        &candles,
        &cfg,
        &spec,
        None,
        &TpeConfig {
            trials: 1,
            batch_size: 1,
            seed: 42,
        },
        None,
        None,
        None,
        1,
    );
    assert_eq!(tpe.completed_trials, 1, "TPE should complete one trial");
    assert_eq!(tpe.results.len(), 1, "TPE should return one top-k result");
    let tpe_result = &tpe.results[0];
    assert_eq!(tpe_result.output_mode, "gpu_tpe");
    assert_eq!(
        tpe_result.overrides,
        vec![("trade.leverage".to_string(), 1.0)]
    );

    let (gpu_rows, states, _symbols) = with_gpu_trace(0, || {
        run_gpu_sweep_with_states(&candles, &cfg, &spec, None, None, None, None)
    });
    assert_eq!(
        gpu_rows.len(),
        1,
        "direct GPU sweep should produce one result"
    );
    assert_eq!(
        states.len(),
        1,
        "direct GPU sweep should return one combo state"
    );
    let gpu_result = &gpu_rows[0];

    assert_eq!(tpe_result.config_id, gpu_result.config_id);
    assert_eq!(tpe_result.total_trades, gpu_result.total_trades);
    assert!((tpe_result.final_balance - gpu_result.final_balance).abs() <= 1e-9);
    assert!((tpe_result.profit_factor - gpu_result.profit_factor).abs() <= 1e-9);
    assert!((tpe_result.max_drawdown_pct - gpu_result.max_drawdown_pct).abs() <= 1e-9);

    let first_exit = first_exit_trace(&states[0]);
    assert_eq!(first_exit.kind, TRACE_KIND_CLOSE);
    assert_eq!(
        first_exit.reason, TRACE_REASON_EXIT_TP,
        "reordered exit profile should reach GPU kernel via TPE and close with full TP first"
    );
}

#[test]
fn cpu_gpu_sub_bar_trailing_exit_matches_cpu_reason() {
    if let Err(err) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", err);
        return;
    }

    let profile = "gpu_sub_bar_trailing";
    let mut cfg = permissive_cfg(profile);
    cfg.engine.entry_interval = "3m".to_string();
    cfg.engine.exit_interval = "3m".to_string();
    cfg.trade.sl_atr_mult = 99.0;
    cfg.trade.tp_atr_mult = 99.0;
    cfg.trade.enable_partial_tp = false;
    cfg.trade.trailing_start_atr = 0.5;
    cfg.trade.trailing_distance_atr = 0.5;
    cfg.trade.enable_vol_buffered_trailing = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    cfg.trade.tsme_min_profit_atr = 999.0;
    cfg.pipeline = PipelineConfig {
        default_profile: "production".to_string(),
        profiles: BTreeMap::from([(
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
                            "exit.take_profit.partial".to_string(),
                            "exit.take_profit.full".to_string(),
                        ],
                        disabled: vec![
                            "exit.smart.trend_exhaustion".to_string(),
                            "exit.smart.ema_macro_breakdown".to_string(),
                            "exit.smart.stagnation".to_string(),
                            "exit.smart.funding_headwind".to_string(),
                            "exit.smart.tsme".to_string(),
                            "exit.smart.mmde".to_string(),
                            "exit.smart.rsi_overextension".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        )]),
    };

    let (candles, sub_candles) = build_sub_bar_trailing_fixture();

    let cpu = run_simulation(RunSimulationInput {
        candles: &candles,
        cfg: &cfg,
        initial_balance: 1_000.0,
        lookback: 0,
        exit_candles: Some(&sub_candles),
        entry_candles: Some(&sub_candles),
        funding_rates: None,
        init_state: None,
        from_ts: None,
        to_ts: None,
    });
    let cpu_exit = cpu
        .trades
        .iter()
        .rev()
        .find(|trade| trade.is_close())
        .unwrap_or_else(|| {
            panic!(
                "CPU run should emit at least one close; trades={:?}",
                cpu.trades
                    .iter()
                    .map(|trade| format!(
                        "{}:{}@{}",
                        trade.action, trade.reason, trade.timestamp_ms
                    ))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        cpu_exit.reason, "Trailing Stop",
        "CPU sub-bar fixture should close via trailing stop, got `{}`",
        cpu_exit.reason
    );

    let (_rows, states, _symbols) = with_gpu_trace(0, || {
        run_gpu_sweep_with_states(
            &candles,
            &cfg,
            &sweep_spec(),
            None,
            Some(&sub_candles),
            None,
            None,
        )
    });
    let state = states
        .first()
        .copied()
        .expect("GPU run should return one combo state");
    let gpu_exit = exit_trace_events(&state)
        .into_iter()
        .last()
        .expect("GPU run should emit at least one exit trace");
    assert_eq!(
        gpu_exit.reason, TRACE_REASON_EXIT_TRAILING,
        "GPU sub-bar fixture should close via trailing trace reason"
    );
}

#[test]
fn cpu_gpu_sub_bar_partial_then_full_tp_continuation_matches_cpu_sequence() {
    if let Err(err) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", err);
        return;
    }

    let profile = "gpu_sub_bar_tp_continuation";
    let mut cfg = permissive_cfg(profile);
    cfg.engine.entry_interval = "3m".to_string();
    cfg.engine.exit_interval = "3m".to_string();
    cfg.trade.sl_atr_mult = 99.0;
    cfg.trade.tp_atr_mult = 2.5;
    cfg.trade.enable_partial_tp = true;
    cfg.trade.tp_partial_pct = 0.5;
    cfg.trade.tp_partial_atr_mult = 1.5;
    cfg.trade.tp_partial_min_notional_usd = 1.0;
    cfg.trade.trailing_start_atr = 99.0;
    cfg.trade.trailing_distance_atr = 99.0;
    cfg.trade.enable_vol_buffered_trailing = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    cfg.trade.tsme_min_profit_atr = 999.0;
    cfg.pipeline = PipelineConfig {
        default_profile: "production".to_string(),
        profiles: BTreeMap::from([(
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
                            "exit.take_profit.partial".to_string(),
                            "exit.take_profit.full".to_string(),
                        ],
                        disabled: vec![
                            "exit.smart.trend_exhaustion".to_string(),
                            "exit.smart.ema_macro_breakdown".to_string(),
                            "exit.smart.stagnation".to_string(),
                            "exit.smart.funding_headwind".to_string(),
                            "exit.smart.tsme".to_string(),
                            "exit.smart.mmde".to_string(),
                            "exit.smart.rsi_overextension".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        )]),
    };

    let (candles, sub_candles) = build_sub_bar_tp_continuation_fixture();

    let cpu = run_simulation(RunSimulationInput {
        candles: &candles,
        cfg: &cfg,
        initial_balance: 1_000.0,
        lookback: 0,
        exit_candles: Some(&sub_candles),
        entry_candles: Some(&sub_candles),
        funding_rates: None,
        init_state: None,
        from_ts: None,
        to_ts: None,
    });
    let cpu_exit_sequence = cpu
        .trades
        .iter()
        .filter(|trade| trade.is_close())
        .map(|trade| trade.reason.as_str())
        .collect::<Vec<_>>();
    assert!(
        cpu_exit_sequence
            .windows(2)
            .any(|window| window == ["Take Profit (Partial)", "Take Profit"]),
        "CPU sub-bar fixture should partial-then-full close, got {:?}",
        cpu.trades
            .iter()
            .map(|trade| format!("{}:{}@{}", trade.action, trade.reason, trade.timestamp_ms))
            .collect::<Vec<_>>()
    );

    let (_rows, states, _symbols) = with_gpu_trace(0, || {
        run_gpu_sweep_with_states(
            &candles,
            &cfg,
            &sweep_spec(),
            None,
            Some(&sub_candles),
            None,
            None,
        )
    });
    let state = states
        .first()
        .copied()
        .expect("GPU run should return one combo state");
    let gpu_exit_sequence = exit_trace_events(&state)
        .into_iter()
        .map(|event| (event.kind, event.reason))
        .collect::<Vec<_>>();
    assert!(
        gpu_exit_sequence.windows(2).any(|window| {
            window
                == [
                    (TRACE_KIND_PARTIAL, TRACE_REASON_PARTIAL),
                    (TRACE_KIND_CLOSE, TRACE_REASON_EXIT_TP),
                ]
        }),
        "GPU sub-bar fixture should trace partial-then-full TP continuation, got {:?}",
        gpu_exit_sequence
    );
}
