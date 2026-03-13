use bt_core::candle::{CandleData, OhlcvBar};
use bt_core::config::{
    BehaviourGroupConfig, BehaviourProfileConfig, Confidence, MacdMode, PipelineConfig,
    PipelineProfileConfig, RuntimeConfig, StrategyConfig,
};
use bt_core::engine::{run_simulation, RunSimulationInput};
use bt_core::sweep::SweepSpec;
use bt_gpu::buffers::{GpuComboState, GpuTraceEvent, GPU_TRACE_CAP};
use bt_gpu::run_gpu_sweep_with_states;
use cudarc::driver::CudaDevice;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::Mutex;

const TRACE_KIND_CLOSE: u32 = 3;
const TRACE_KIND_PARTIAL: u32 = 4;
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
