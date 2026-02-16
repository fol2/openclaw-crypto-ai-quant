//! Deterministic tp_atr_mult directional parity regression.
//!
//! Uses the real CUDA sweep runtime when a CUDA device is available. The test
//! is skipped only when CUDA is genuinely unavailable.

use std::path::PathBuf;

use bt_core::candle::{CandleData, OhlcvBar};
use bt_core::sweep::{SweepAxis, SweepSpec};
use bt_core::{config, engine};
use cudarc::driver::result::DriverError;
use cudarc::driver::{sys::CUresult, CudaDevice};

const INITIAL_BALANCE: f64 = 1000.0;
const TP_AXIS_VALUES: [f64; 4] = [3.0, 4.0, 5.0, 6.0];

fn build_tp_isolation_fixture() -> (CandleData, CandleData) {
    let closes = [100.0, 100.7, 101.3, 101.9, 102.5, 103.1, 103.7, 104.3];
    let rel_path = [
        0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.10, 1.45, 1.75, 2.05, 2.20, 2.35, 2.45, 2.50, 2.35,
        2.10, 1.85, 1.60, 1.35, 1.20,
    ];

    let mut main: CandleData = CandleData::default();
    let mut sub: CandleData = CandleData::default();
    let mut main_bars: Vec<OhlcvBar> = Vec::new();
    let mut sub_bars: Vec<OhlcvBar> = Vec::new();

    for (i, close) in closes.iter().enumerate() {
        let t = (i as i64) * 3_600_000;
        let open: f64 = if i == 0 { *close } else { closes[i - 1] };
        let high = open.max(*close) + 0.12;
        let low = open.min(*close) - 0.12;

        main_bars.push(OhlcvBar {
            t,
            t_close: t + 3_600_000,
            o: open,
            h: high,
            l: low,
            c: *close,
            v: 1000.0,
            n: 1,
        });
    }

    for i in 1..closes.len() {
        let base = closes[i - 1];
        let start = ((i - 1) as i64) * 3_600_000;

        for (j, rel) in rel_path.iter().enumerate() {
            let px = base + rel;
            let t = start + ((j as i64) + 1) * 180_000;

            sub_bars.push(OhlcvBar {
                t,
                t_close: t + 180_000,
                o: px,
                h: px + 0.03,
                l: px - 0.03,
                c: px,
                v: 900.0,
                n: 1,
            });
        }
    }

    main.insert("BTC".to_string(), main_bars);
    sub.insert("BTC".to_string(), sub_bars);
    (main, sub)
}

fn sign(x: f64) -> i8 {
    let eps = 1e-9;
    if x > eps {
        1
    } else if x < -eps {
        -1
    } else {
        0
    }
}

fn cuda_is_unavailable(err: DriverError) -> bool {
    matches!(
        err.0,
        CUresult::CUDA_ERROR_NO_DEVICE
            | CUresult::CUDA_ERROR_INVALID_DEVICE
            | CUresult::CUDA_ERROR_DEVICE_UNAVAILABLE
    )
}

fn ensure_cuda_available_or_skip() -> bool {
    match CudaDevice::new(0) {
        Ok(_device) => true,
        Err(err) if cuda_is_unavailable(err) => {
            eprintln!("Skipping tp_atr_mult GPU directional parity test: CUDA unavailable ({err:?})");
            false
        }
        Err(err) => panic!(
            "CUDA preflight failed before parity test; this is a configuration/runtime error: {err:?}"
        ),
    }
}

fn run_cpu_pnl_by_axis(
    base_cfg: &bt_core::config::StrategyConfig,
    candles: &CandleData,
    sub_candles: &CandleData,
) -> Vec<f64> {
    let mut cpu_pnl_by_axis_value: Vec<f64> = Vec::new();

    for tp_mult in TP_AXIS_VALUES {
        let mut cfg = base_cfg.clone();
        cfg.trade.tp_atr_mult = tp_mult;

        let sim = engine::run_simulation(
            candles,
            &cfg,
            INITIAL_BALANCE,
            0,
            Some(sub_candles),
            Some(sub_candles),
            None,
            None,
            None,
            None,
        );

        cpu_pnl_by_axis_value.push(sim.final_balance - INITIAL_BALANCE);
    }

    cpu_pnl_by_axis_value
}

fn run_gpu_pnl_by_axis(
    base_cfg: &bt_core::config::StrategyConfig,
    candles: &CandleData,
    sub_candles: &CandleData,
) -> Vec<f64> {
    let spec = SweepSpec {
        axes: vec![SweepAxis {
            path: "trade.tp_atr_mult".to_string(),
            values: TP_AXIS_VALUES.to_vec(),
            gate: None,
        }],
        initial_balance: INITIAL_BALANCE,
        lookback: 0,
    };

    let gpu_results = bt_gpu::run_gpu_sweep(
        candles,
        base_cfg,
        &spec,
        None,
        Some(sub_candles),
        None,
        None,
    );

    let mut pnl_by_axis: Vec<Option<f64>> = vec![None; TP_AXIS_VALUES.len()];

    for result in gpu_results {
        let tp_mult = result
            .overrides
            .iter()
            .find_map(|(path, value)| (path == "trade.tp_atr_mult").then_some(*value))
            .unwrap_or_else(|| panic!("GPU result is missing trade.tp_atr_mult override"));

        let axis_idx = TP_AXIS_VALUES
            .iter()
            .position(|axis_value| (tp_mult - axis_value).abs() < 1e-9)
            .unwrap_or_else(|| {
                panic!("Unexpected tp_atr_mult override from GPU runtime: {tp_mult}")
            });

        assert!(
            pnl_by_axis[axis_idx].is_none(),
            "Duplicate GPU result for tp_atr_mult={tp_mult}"
        );

        pnl_by_axis[axis_idx] = Some(result.final_balance - INITIAL_BALANCE);
    }

    pnl_by_axis
        .into_iter()
        .enumerate()
        .map(|(idx, maybe_pnl)| {
            maybe_pnl.unwrap_or_else(|| {
                panic!("Missing GPU result for tp_atr_mult={}", TP_AXIS_VALUES[idx])
            })
        })
        .collect()
}

#[test]
fn tp_atr_mult_first_to_last_direction_matches_gpu_runtime() {
    if !ensure_cuda_available_or_skip() {
        return;
    }

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cfg_path = crate_root.join("../../testdata/gpu_cpu_parity/tp_atr_mult_strategy.yaml");

    let base_cfg = config::load_config(&cfg_path.to_string_lossy(), None, false);
    let (candles, sub_candles) = build_tp_isolation_fixture();

    let cpu_pnl_by_axis_value = run_cpu_pnl_by_axis(&base_cfg, &candles, &sub_candles);
    let gpu_pnl_by_axis_value = run_gpu_pnl_by_axis(&base_cfg, &candles, &sub_candles);

    let cpu_delta =
        cpu_pnl_by_axis_value[cpu_pnl_by_axis_value.len() - 1] - cpu_pnl_by_axis_value[0];
    let gpu_delta =
        gpu_pnl_by_axis_value[gpu_pnl_by_axis_value.len() - 1] - gpu_pnl_by_axis_value[0];

    assert_ne!(
        sign(cpu_delta),
        0,
        "CPU fixture did not react to tp_atr_mult (first->last delta is ~0)."
    );
    assert_ne!(
        sign(gpu_delta),
        0,
        "GPU runtime fixture is degenerate (first->last delta is ~0)."
    );
    assert_eq!(
        sign(cpu_delta),
        sign(gpu_delta),
        "Directional parity regression for tp_atr_mult: cpu_delta={:.6}, gpu_delta={:.6}, cpu={:?}, gpu={:?}",
        cpu_delta,
        gpu_delta,
        cpu_pnl_by_axis_value,
        gpu_pnl_by_axis_value,
    );
}
