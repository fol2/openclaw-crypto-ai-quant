# Synthetic Signal Capture Progress

Deterministic synthetic fixtures used to verify whether CPU and GPU both react to each target parameter.

- Schema version: 3
- Total target axes: 14
- Selected in this run: 14
- Executed: 14
- Pass: 12
- Fail: 2
- Pending: 0

## Coverage Matrix

| Axis | Scenario | Status | Pass | Notes |
|---|---|---|---:|---|
| `trade.sl_atr_mult` | `sl_isolation` | `fail` | no | Expectation: Changing SL multiplier should alter aggregate PnL.<br>Expectation check: total_pnl change from 1.5 -> 3; CPU delta=-7.869540, GPU delta=10.677996.<br>Direction mismatch: CPU/GPU first->last delta directions do not match (CPU=decrease, GPU=increase).<br>CPU and GPU both reacted to axis changes.<br>Low CPU/GPU PnL-direction agreement (33.3%, mismatches=2).<br>Low CPU/GPU trade-direction agreement (33.3%, mismatches=2). |
| `trade.tp_atr_mult` | `tp_isolation` | `fail` | no | Expectation: Changing TP multiplier should alter aggregate PnL.<br>Expectation check: total_pnl change from 3 -> 6; CPU delta=0.274905, GPU delta=-55.851318.<br>Direction mismatch: CPU/GPU first->last delta directions do not match (CPU=increase, GPU=decrease).<br>CPU and GPU both reacted to axis changes.<br>Low CPU/GPU trade-direction agreement (33.3%, mismatches=2). |
| `trade.allocation_pct` | `allocation_dual` | `pass` | yes | Expectation: Higher allocation should increase PnL under positive trend fixture.<br>Expectation check: total_pnl increase from 0.1 -> 0.2; CPU delta=37.828537, GPU delta=23.234703.<br>CPU and GPU both reacted to axis changes. |
| `trade.trailing_start_atr` | `trailing_single` | `pass` | yes | Expectation: Trailing activation threshold should change realised PnL.<br>Expectation check: total_pnl change from 1.5 -> 2.5; CPU delta=-0.771809, GPU delta=-11.685560.<br>CPU and GPU both reacted to axis changes. |
| `trade.trailing_distance_atr` | `trailing_single` | `pass` | yes | Expectation: Trailing distance should change realised PnL.<br>Expectation check: total_pnl change from 0.5 -> 1; CPU delta=-6.859345, GPU delta=-3.919406.<br>CPU and GPU both reacted to axis changes. |
| `trade.enable_partial_tp` | `partial_tp_toggle` | `pass` | yes | Expectation: Enabling partial TP should change trade count in partial-only fixture.<br>Expectation check: total_trades change from 0 -> 1; CPU delta=2.000000, GPU delta=3.000000.<br>CPU and GPU both reacted to axis changes. |
| `trade.tp_partial_pct` | `partial_tp_pct` | `pass` | yes | Expectation: Partial TP percentage should change realised PnL.<br>Expectation check: total_pnl change from 0.25 -> 0.75; CPU delta=-9.402049, GPU delta=-17.534626.<br>CPU and GPU both reacted to axis changes. |
| `trade.tp_partial_atr_mult` | `partial_tp_atr_mult` | `pass` | yes | Expectation: Partial TP ATR trigger level should change realised PnL.<br>Expectation check: total_pnl change from 0.5 -> 2; CPU delta=5.373798, GPU delta=8.105225.<br>CPU and GPU both reacted to axis changes. |
| `trade.max_total_margin_pct` | `margin_cap_isolation` | `pass` | yes | Expectation: More margin headroom should increase realised PnL in profitable multi-symbol fixture.<br>Expectation check: total_pnl increase from 0.2 -> 0.8; CPU delta=106.861168, GPU delta=69.350414.<br>CPU and GPU both reacted to axis changes. |
| `trade.max_open_positions` | `max_positions_isolation` | `pass` | yes | Expectation: Allowing more concurrent positions should increase executed trades.<br>Expectation check: total_trades increase from 1 -> 4; CPU delta=4.000000, GPU delta=2.000000.<br>CPU and GPU both reacted to axis changes. |
| `trade.leverage` | `leverage_isolation` | `pass` | yes | Expectation: Higher leverage should increase realised PnL in profitable fixture.<br>Expectation check: total_pnl increase from 1 -> 5; CPU delta=170.257059, GPU delta=404.461899.<br>CPU and GPU both reacted to axis changes. |
| `trade.smart_exit_adx_exhaustion_lt` | `smart_exit_high_conf` | `pass` | yes | Expectation: Raising ADX exhaustion threshold should alter exit churn.<br>Expectation check: total_trades change from 0 -> 200; CPU delta=6.000000, GPU delta=7.000000.<br>CPU and GPU both reacted to axis changes. |
| `trade.smart_exit_adx_exhaustion_lt_low_conf` | `smart_exit_low_conf` | `pass` | yes | Expectation: Low-confidence ADX exhaustion override should alter low-confidence exit churn.<br>Expectation check: total_trades change from 0 -> 200; CPU delta=1.000000, GPU delta=1.000000.<br>CPU and GPU both reacted to axis changes. |
| `filters.require_macro_alignment` | `macro_alignment_gate` | `pass` | yes | Expectation: Macro alignment gate false/true should alter captured trades.<br>Expectation check: total_trades change from 0 -> 1; CPU delta=-3.000000, GPU delta=-4.000000.<br>CPU and GPU both reacted to axis changes.<br>Macro gate false->true trade-direction change is consistent across CPU and GPU. |

## Executed Experiment Details

### `trade.sl_atr_mult` on `sl_isolation`

Deep downside spikes with TP/trailing effectively disabled to isolate SL multipliers.

- Status: `fail`
- Pass: no
- Notes: Expectation: Changing SL multiplier should alter aggregate PnL.; Expectation check: total_pnl change from 1.5 -> 3; CPU delta=-7.869540, GPU delta=10.677996.; Direction mismatch: CPU/GPU first->last delta directions do not match (CPU=decrease, GPU=increase).; CPU and GPU both reacted to axis changes.; Low CPU/GPU PnL-direction agreement (33.3%, mismatches=2).; Low CPU/GPU trade-direction agreement (33.3%, mismatches=2).
- Expectation first->last delta: total_pnl change (CPU -7.869540, GPU 10.677996, pass=no)
- Baseline: CPU trades=2, pnl=-7.8422; GPU trades=3, pnl=-33.7364
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/3 (33.3%, mismatches=2)
- CPU/GPU trade-count direction agreement across axis steps: 1/3 (33.3%, mismatches=2)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 1.5 | -7.8422 | 2 | -33.7364 | 3 |
| 2 | -8.8259 | 2 | -28.4203 | 2 |
| 2.5 | -11.7770 | 2 | -34.3167 | 2 |
| 3 | -15.7117 | 2 | -23.0584 | 1 |

### `trade.tp_atr_mult` on `tp_isolation`

High-upside spikes with trailing/SL effectively disabled to isolate TP multipliers.

- Status: `fail`
- Pass: no
- Notes: Expectation: Changing TP multiplier should alter aggregate PnL.; Expectation check: total_pnl change from 3 -> 6; CPU delta=0.274905, GPU delta=-55.851318.; Direction mismatch: CPU/GPU first->last delta directions do not match (CPU=increase, GPU=decrease).; CPU and GPU both reacted to axis changes.; Low CPU/GPU trade-direction agreement (33.3%, mismatches=2).
- Expectation first->last delta: total_pnl change (CPU 0.274905, GPU -55.851318, pass=no)
- Baseline: CPU trades=3, pnl=42.3717; GPU trades=4, pnl=99.3173
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 3/3 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 1/3 (33.3%, mismatches=2)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 3 | 42.3717 | 3 | 99.3173 | 4 |
| 4 | 33.8731 | 2 | 60.1925 | 2 |
| 5 | 31.6051 | 2 | 35.5722 | 1 |
| 6 | 42.6466 | 1 | 43.4660 | 1 |

### `trade.allocation_pct` on `allocation_dual`

Dual-symbol synchronous path to force portfolio sizing effects.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Higher allocation should increase PnL under positive trend fixture.; Expectation check: total_pnl increase from 0.1 -> 0.2; CPU delta=37.828537, GPU delta=23.234703.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl increase (CPU 37.828537, GPU 23.234703, pass=yes)
- Baseline: CPU trades=4, pnl=37.7721; GPU trades=2, pnl=23.2355
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 2/2 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0.1 | 37.7721 | 4 | 23.2355 | 2 |
| 0.15 | 56.6793 | 4 | 34.8530 | 2 |
| 0.2 | 75.6006 | 4 | 46.4702 | 2 |

### `trade.trailing_start_atr` on `trailing_single`

Single-symbol trend/retrace path to force trailing threshold decisions.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Trailing activation threshold should change realised PnL.; Expectation check: total_pnl change from 1.5 -> 2.5; CPU delta=-0.771809, GPU delta=-11.685560.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl change (CPU -0.771809, GPU -11.685560, pass=yes)
- Baseline: CPU trades=2, pnl=18.9977; GPU trades=1, pnl=11.6856
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 2/2 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 1.5 | 18.9977 | 2 | 11.6856 | 1 |
| 2 | 19.0487 | 2 | 16.5847 | 1 |
| 2.5 | 18.2259 | 1 | 0.0000 | 0 |

### `trade.trailing_distance_atr` on `trailing_single`

Single-symbol trend/retrace path to force trailing threshold decisions.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Trailing distance should change realised PnL.; Expectation check: total_pnl change from 0.5 -> 1; CPU delta=-6.859345, GPU delta=-3.919406.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl change (CPU -6.859345, GPU -3.919406, pass=yes)
- Baseline: CPU trades=2, pnl=18.9977; GPU trades=1, pnl=11.6856
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/1 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 1/1 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0.5 | 18.9977 | 2 | 11.6856 | 1 |
| 1 | 12.1384 | 2 | 7.7662 | 1 |

### `trade.enable_partial_tp` on `partial_tp_toggle`

Partial-TP toggle fixture with full TP effectively disabled. When enabled, partial reduce can trigger; when disabled, the same path cannot emit a partial action.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Enabling partial TP should change trade count in partial-only fixture.; Expectation check: total_trades change from 0 -> 1; CPU delta=2.000000, GPU delta=3.000000.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_trades change (CPU 2.000000, GPU 3.000000, pass=yes)
- Baseline: CPU trades=2, pnl=33.8731; GPU trades=2, pnl=60.1925
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/1 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 1/1 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0 | 33.8731 | 2 | 60.1925 | 2 |
| 1 | 24.4723 | 4 | 42.6555 | 5 |

### `trade.tp_partial_pct` on `partial_tp_pct`

Partial-TP percentage fixture with partial TP forced on and dedicated partial trigger enabled. Only tp_partial_pct changes closed fraction at the first TP event.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Partial TP percentage should change realised PnL.; Expectation check: total_pnl change from 0.25 -> 0.75; CPU delta=-9.402049, GPU delta=-17.534626.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl change (CPU -9.402049, GPU -17.534626, pass=yes)
- Baseline: CPU trades=4, pnl=29.1730; GPU trades=5, pnl=51.4234
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 2/2 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0.25 | 29.1730 | 4 | 51.4234 | 5 |
| 0.5 | 24.4723 | 4 | 42.6555 | 5 |
| 0.75 | 19.7710 | 4 | 33.8888 | 5 |

### `trade.tp_partial_atr_mult` on `partial_tp_atr_mult`

Partial-TP ATR-multiplier fixture with partial TP fixed on and constant partial fraction. Only tp_partial_atr_mult changes when partial reduction is triggered.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Partial TP ATR trigger level should change realised PnL.; Expectation check: total_pnl change from 0.5 -> 2; CPU delta=5.373798, GPU delta=8.105225.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl change (CPU 5.373798, GPU 8.105225, pass=yes)
- Baseline: CPU trades=4, pnl=27.8334; GPU trades=5, pnl=49.2160
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 2/2 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0.5 | 27.8334 | 4 | 49.2160 | 5 |
| 1 | 29.1730 | 4 | 51.4234 | 5 |
| 2 | 33.2072 | 4 | 57.3213 | 5 |

### `trade.max_total_margin_pct` on `margin_cap_isolation`

Three-symbol profitable fixture with aggressive per-position allocation. Only max_total_margin_pct controls how much portfolio headroom is available.

- Status: `pass`
- Pass: yes
- Notes: Expectation: More margin headroom should increase realised PnL in profitable multi-symbol fixture.; Expectation check: total_pnl increase from 0.2 -> 0.8; CPU delta=106.861168, GPU delta=69.350414.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl increase (CPU 106.861168, GPU 69.350414, pass=yes)
- Baseline: CPU trades=4, pnl=44.2133; GPU trades=1, pnl=23.3711
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 3/3 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/3 (66.7%, mismatches=1)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0.2 | 44.2133 | 4 | 23.3711 | 1 |
| 0.4 | 76.0170 | 6 | 46.6690 | 2 |
| 0.6 | 113.7382 | 6 | 69.7669 | 2 |
| 0.8 | 151.0745 | 6 | 92.7215 | 3 |

### `trade.max_open_positions` on `max_positions_isolation`

Three-symbol synchronous-entry fixture with ample margin headroom. Only max_open_positions throttles how many concurrent entries can be opened.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Allowing more concurrent positions should increase executed trades.; Expectation check: total_trades increase from 1 -> 4; CPU delta=4.000000, GPU delta=2.000000.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_trades increase (CPU 4.000000, GPU 2.000000, pass=yes)
- Baseline: CPU trades=2, pnl=26.6298; GPU trades=1, pnl=14.0227
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 2/2 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 1 | 26.6298 | 2 | 14.0227 | 1 |
| 2 | 49.1247 | 4 | 27.8825 | 2 |
| 4 | 67.6087 | 6 | 41.5833 | 3 |

### `trade.leverage` on `leverage_isolation`

Single-symbol profitable TP fixture with dynamic leverage disabled. Only leverage rescales position notional and realised PnL.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Higher leverage should increase realised PnL in profitable fixture.; Expectation check: total_pnl increase from 1 -> 5; CPU delta=170.257059, GPU delta=404.461899.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_pnl increase (CPU 170.257059, GPU 404.461899, pass=yes)
- Baseline: CPU trades=3, pnl=42.3717; GPU trades=4, pnl=99.3173
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 3/3 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 3/3 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 1 | 42.3717 | 3 | 99.3173 | 4 |
| 2 | 84.8204 | 3 | 199.3505 | 4 |
| 3 | 127.3463 | 3 | 300.1029 | 4 |
| 5 | 212.6287 | 3 | 503.7792 | 4 |

### `trade.smart_exit_adx_exhaustion_lt` on `smart_exit_high_conf`

High-confidence smart-exit fixture with SL/TP/trailing neutralised. Only smart_exit_adx_exhaustion_lt controls ADX-based early exit pressure.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Raising ADX exhaustion threshold should alter exit churn.; Expectation check: total_trades change from 0 -> 200; CPU delta=6.000000, GPU delta=7.000000.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_trades change (CPU 6.000000, GPU 7.000000, pass=yes)
- Baseline: CPU trades=1, pnl=18.2259; GPU trades=0, pnl=0.0000
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/2 (50.0%, mismatches=1)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0 | 18.2259 | 1 | 0.0000 | 0 |
| 20 | 18.2259 | 1 | 0.0000 | 0 |
| 200 | 7.0995 | 7 | 12.7061 | 7 |

### `trade.smart_exit_adx_exhaustion_lt_low_conf` on `smart_exit_low_conf`

Low-confidence smart-exit fixture that blocks standard entries and forces slow-drift entries. Only smart_exit_adx_exhaustion_lt_low_conf should influence ADX-exhaustion exits.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Low-confidence ADX exhaustion override should alter low-confidence exit churn.; Expectation check: total_trades change from 0 -> 200; CPU delta=1.000000, GPU delta=1.000000.; CPU and GPU both reacted to axis changes.
- Expectation first->last delta: total_trades change (CPU 1.000000, GPU 1.000000, pass=yes)
- Baseline: CPU trades=1, pnl=28.4318; GPU trades=0, pnl=0.0000
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/2 (50.0%, mismatches=1)
- CPU/GPU trade-count direction agreement across axis steps: 2/2 (100.0%, mismatches=0)

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| 0 | 28.4318 | 1 | 0.0000 | 0 |
| 20 | 28.4318 | 1 | 0.0000 | 0 |
| 200 | 26.2653 | 2 | 9.4517 | 1 |

### `filters.require_macro_alignment` on `macro_alignment_gate`

Dedicated false/true macro-alignment gate toggle. Reuses tp_isolation fixture (ema_slow_window == ema_macro_window) so enabling the gate consistently suppresses directional alignment on both CPU and GPU.

- Status: `pass`
- Pass: yes
- Notes: Expectation: Macro alignment gate false/true should alter captured trades.; Expectation check: total_trades change from 0 -> 1; CPU delta=-3.000000, GPU delta=-4.000000.; CPU and GPU both reacted to axis changes.; Macro gate false->true trade-direction change is consistent across CPU and GPU.
- Expectation first->last delta: total_trades change (CPU -3.000000, GPU -4.000000, pass=yes)
- Baseline: CPU trades=3, pnl=42.3717; GPU trades=4, pnl=99.3173
- CPU triggered: yes
- GPU triggered: yes
- CPU/GPU PnL direction agreement across axis steps: 1/1 (100.0%, mismatches=0)
- CPU/GPU trade-count direction agreement across axis steps: 1/1 (100.0%, mismatches=0)
- Macro gate false->true trade delta (CPU/GPU): -3.0/-4.0
- Macro gate consistent trade-direction change: yes

| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |
|---:|---:|---:|---:|---:|
| false | 42.3717 | 3 | 99.3173 | 4 |
| true | 0.0000 | 0 | 0.0000 | 0 |
