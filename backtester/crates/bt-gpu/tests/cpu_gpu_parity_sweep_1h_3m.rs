use bt_core::candle::{CandleData, OhlcvBar};
use bt_core::config::{Confidence, MacdMode, StrategyConfig};
use bt_core::sweep::SweepSpec;
use bt_gpu::run_gpu_sweep;
use cudarc::driver::CudaDevice;
use rustc_hash::FxHashMap;

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

#[test]
fn cpu_gpu_parity_sweep_1h_3m_tiny_fixture() {
    if let Err(e) = CudaDevice::new(0) {
        eprintln!("Skipping: CUDA unavailable: {:?}", e);
        return;
    }

    let mut candles: CandleData = FxHashMap::default();
    candles.insert(
        "BTC".to_string(),
        vec![
            bar_1h(0, 100.0),
            bar_1h(3_600_000, 102.0), // triggers a BUY under the relaxed config below
        ],
    );

    let mut exit_candles: CandleData = FxHashMap::default();
    // With two main bars, the second bar's range is (3_600_000, +inf], so this is in-range.
    exit_candles.insert("BTC".to_string(), vec![bar_3m(3_780_000, 96.0)]);

    let mut cfg = StrategyConfig::default();

    // Make entry permissive and deterministic (avoid depending on ADX/MACD/StochRSI).
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

    // Disable entry-time trade filters that can suppress trades on short fixtures.
    cfg.trade.enable_ssf_filter = false;
    cfg.trade.enable_reef_filter = false;

    // Reduce sources of CPU f64 vs GPU f32 drift.
    cfg.trade.enable_dynamic_sizing = false;
    cfg.trade.enable_dynamic_leverage = false;
    cfg.trade.leverage = 1.0;
    cfg.trade.allocation_pct = 0.10;
    cfg.trade.slippage_bps = 0.0;
    cfg.trade.enable_pyramiding = false;
    cfg.trade.enable_rsi_overextension_exit = false;
    cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
    cfg.trade.min_atr_pct = 0.0;

    // Small windows to make the fixture short and avoid extended warmup.
    cfg.indicators.ema_fast_window = 2;
    cfg.indicators.ema_slow_window = 3;
    cfg.indicators.adx_window = 2;
    cfg.indicators.atr_window = 2;
    cfg.indicators.rsi_window = 1;
    cfg.indicators.bb_window = 2;
    cfg.indicators.bb_width_avg_window = 2;
    cfg.indicators.vol_sma_window = 2;
    cfg.indicators.vol_trend_window = 2;

    let spec = SweepSpec {
        axes: vec![],
        initial_balance: 1_000.0,
        lookback: 0,
    };

    let cpu = bt_core::sweep::run_sweep(
        &cfg,
        &spec,
        &candles,
        Some(&exit_candles),
        None,
        None,
        None,
        None,
    );
    assert_eq!(cpu.len(), 1);
    let cpu_rpt = &cpu[0].report;
    assert_eq!(
        cpu_rpt.total_trades, 1,
        "Fixture should produce exactly one trade on CPU"
    );

    let gpu = run_gpu_sweep(&candles, &cfg, &spec, None, Some(&exit_candles), None, None);
    assert_eq!(gpu.len(), 1);
    let gpu_rpt = &gpu[0];
    assert_eq!(
        gpu_rpt.total_trades, 1,
        "Fixture should produce exactly one trade on GPU"
    );

    // Parity: allow minor differences due to f32 vs f64 math.
    let eps = 0.10;
    assert!(
        (cpu_rpt.final_balance - gpu_rpt.final_balance).abs() <= eps,
        "final_balance mismatch (cpu={}, gpu={}, eps={})",
        cpu_rpt.final_balance,
        gpu_rpt.final_balance,
        eps
    );
    assert!(
        (cpu_rpt.total_pnl - gpu_rpt.total_pnl).abs() <= eps,
        "total_pnl mismatch (cpu={}, gpu={}, eps={})",
        cpu_rpt.total_pnl,
        gpu_rpt.total_pnl,
        eps
    );
    assert_eq!(cpu_rpt.total_wins, gpu_rpt.total_wins);
}
