use bt_core::candle::{CandleData, OhlcvBar};
use bt_core::config::{Confidence, MacdMode, StrategyConfig};
use bt_core::sweep::{SweepAxis, SweepResult, SweepSpec};
use bt_gpu::{layout::GpuSweepResult, run_gpu_sweep};
use cudarc::driver::CudaDevice;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug)]
struct Metrics {
    total_pnl: f64,
    total_trades: u32,
}

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

fn bar_3m(t: i64, close: f64) -> OhlcvBar {
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

fn main_candles() -> CandleData {
    let mut candles: CandleData = FxHashMap::default();
    candles.insert(
        "BTC".to_string(),
        vec![
            bar_1h(0, 100.0),
            bar_1h(3_600_000, 102.0),
            bar_1h(7_200_000, 103.0),
            bar_1h(10_800_000, 104.0),
        ],
    );
    candles
}

fn base_cfg() -> StrategyConfig {
    let mut cfg = StrategyConfig::default();

    // Make entry permissive and deterministic for very short fixtures.
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

    // Disable non-target paths that can mask end-of-backtest close behaviour.
    cfg.trade.enable_ssf_filter = false;
    cfg.trade.enable_reef_filter = false;
    cfg.trade.enable_dynamic_sizing = false;
    cfg.trade.enable_dynamic_leverage = false;
    cfg.trade.enable_pyramiding = false;
    cfg.trade.enable_partial_tp = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    cfg.trade.trailing_start_atr = 100.0;
    cfg.trade.trailing_distance_atr = 1.0;
    cfg.trade.block_exits_on_extreme_dev = false;

    cfg.trade.leverage = 1.0;
    cfg.trade.allocation_pct = 0.10;
    cfg.trade.slippage_bps = 0.0;
    cfg.trade.min_atr_pct = 0.0;

    // Small windows keep fixture short and stable.
    cfg.indicators.ema_fast_window = 2;
    cfg.indicators.ema_slow_window = 3;
    cfg.indicators.adx_window = 2;
    cfg.indicators.atr_window = 2;
    cfg.indicators.rsi_window = 1;
    cfg.indicators.bb_window = 2;
    cfg.indicators.bb_width_avg_window = 2;
    cfg.indicators.vol_sma_window = 2;
    cfg.indicators.vol_trend_window = 2;

    cfg
}

fn axis_value(overrides: &[(String, f64)], axis: &str) -> f64 {
    overrides
        .iter()
        .find_map(|(path, value)| if path == axis { Some(*value) } else { None })
        .unwrap_or_else(|| panic!("axis {axis} missing in overrides: {:?}", overrides))
}

fn assign_pair(slot: &mut [Option<Metrics>; 2], value: f64, low: f64, high: f64, metrics: Metrics) {
    if (value - low).abs() < 1e-9 {
        slot[0] = Some(metrics);
    } else if (value - high).abs() < 1e-9 {
        slot[1] = Some(metrics);
    } else {
        panic!("unexpected axis value {value} (expected {low} or {high})");
    }
}

fn collect_cpu_metrics(results: Vec<SweepResult>, axis: &str, low: f64, high: f64) -> [Metrics; 2] {
    let mut out: [Option<Metrics>; 2] = [None, None];
    for r in results {
        let value = axis_value(&r.overrides, axis);
        assign_pair(
            &mut out,
            value,
            low,
            high,
            Metrics {
                total_pnl: r.report.total_pnl,
                total_trades: r.report.total_trades,
            },
        );
    }
    [
        out[0].unwrap_or_else(|| panic!("missing CPU result for {axis}={low}")),
        out[1].unwrap_or_else(|| panic!("missing CPU result for {axis}={high}")),
    ]
}

fn collect_gpu_metrics(
    results: Vec<GpuSweepResult>,
    axis: &str,
    low: f64,
    high: f64,
) -> [Metrics; 2] {
    let mut out: [Option<Metrics>; 2] = [None, None];
    for r in results {
        let value = axis_value(&r.overrides, axis);
        assign_pair(
            &mut out,
            value,
            low,
            high,
            Metrics {
                total_pnl: r.total_pnl,
                total_trades: r.total_trades,
            },
        );
    }
    [
        out[0].unwrap_or_else(|| panic!("missing GPU result for {axis}={low}")),
        out[1].unwrap_or_else(|| panic!("missing GPU result for {axis}={high}")),
    ]
}

fn run_axis_case(
    axis: &str,
    values: [f64; 2],
    cfg: &StrategyConfig,
    candles: &CandleData,
    exit_candles: &CandleData,
) -> ([Metrics; 2], [Metrics; 2]) {
    let spec = SweepSpec {
        axes: vec![SweepAxis {
            path: axis.to_string(),
            values: values.to_vec(),
        }],
        initial_balance: 1_000.0,
        lookback: 0,
    };

    let cpu = bt_core::sweep::run_sweep(
        cfg,
        &spec,
        candles,
        Some(exit_candles),
        None,
        None,
        None,
        None,
    );
    let gpu = run_gpu_sweep(candles, cfg, &spec, None, Some(exit_candles), None, None);

    (
        collect_cpu_metrics(cpu, axis, values[0], values[1]),
        collect_gpu_metrics(gpu, axis, values[0], values[1]),
    )
}

#[test]
fn eob_forced_close_parity_for_sl_and_tp_axes() {
    if let Err(e) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", e);
        return;
    }

    let candles = main_candles();

    // SL axis: low SL exits in sub-bar; high SL must rely on end-of-backtest force close.
    let mut sl_exit_candles: CandleData = FxHashMap::default();
    sl_exit_candles.insert(
        "BTC".to_string(),
        vec![
            bar_3m(3_660_000, 103.0), // deterministic long-entry trigger
            bar_3m(7_380_000, 90.0),  // low-SL hit; high-SL survives to EOB close
        ],
    );

    let mut sl_cfg = base_cfg();
    sl_cfg.trade.tp_atr_mult = 8.0; // Keep TP out of the way.
    let (cpu_sl, gpu_sl) = run_axis_case(
        "trade.sl_atr_mult",
        [0.2, 8.0],
        &sl_cfg,
        &candles,
        &sl_exit_candles,
    );

    // TP axis: low TP exits in sub-bar; high TP must rely on end-of-backtest force close.
    let mut tp_exit_candles: CandleData = FxHashMap::default();
    tp_exit_candles.insert(
        "BTC".to_string(),
        vec![
            bar_3m(3_660_000, 103.0), // deterministic long-entry trigger
            bar_3m(7_380_000, 105.0), // low-TP hit; high-TP survives to EOB close
        ],
    );

    let mut tp_cfg = base_cfg();
    tp_cfg.trade.sl_atr_mult = 8.0; // Keep SL out of the way.
    let (cpu_tp, gpu_tp) = run_axis_case(
        "trade.tp_atr_mult",
        [0.5, 4.0],
        &tp_cfg,
        &candles,
        &tp_exit_candles,
    );

    // In both cases, the high value branch depends on force-close at end-of-backtest.
    assert_eq!(cpu_sl[1].total_trades, 1, "cpu_sl={:?}", cpu_sl);
    assert_eq!(gpu_sl[1].total_trades, 1, "gpu_sl={:?}", gpu_sl);
    assert_eq!(cpu_tp[1].total_trades, 1, "cpu_tp={:?}", cpu_tp);
    assert_eq!(gpu_tp[1].total_trades, 1, "gpu_tp={:?}", gpu_tp);

    // Directional parity along each axis.
    let cpu_sl_delta = cpu_sl[1].total_pnl - cpu_sl[0].total_pnl;
    let gpu_sl_delta = gpu_sl[1].total_pnl - gpu_sl[0].total_pnl;
    assert!(cpu_sl_delta.abs() > 1e-6);
    assert!(gpu_sl_delta.abs() > 1e-6);
    assert_eq!(cpu_sl_delta > 0.0, gpu_sl_delta > 0.0);

    let cpu_tp_delta = cpu_tp[1].total_pnl - cpu_tp[0].total_pnl;
    let gpu_tp_delta = gpu_tp[1].total_pnl - gpu_tp[0].total_pnl;
    assert!(cpu_tp_delta.abs() > 1e-6);
    assert!(gpu_tp_delta.abs() > 1e-6);
    assert_eq!(cpu_tp_delta > 0.0, gpu_tp_delta > 0.0);
}
