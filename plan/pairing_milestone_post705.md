# GPU/CPU Pairing Milestone Evidence (Post-#705)

Date: 19 February 2026

## Scope

This note records the first pairing milestone after PR #705 (`gpu: align sub-bar trade start with scoped window`) was merged to `master`.

The test scope is:

- Mode: GPU TPE sweep paired to CPU replay
- Interval: 30m
- Entry interval: 3m (sub-bar entry/exit enabled)
- Funding mode: enabled
- State basis: immutable snapshot with balance-from handling enabled

## Artefacts

- `artifacts/gpu_tpe_30m3m_funding_snapshot_harness_post705_rebuild_10k20k_20260219T1130Z/pairing/trials_10000/pairing_report.json`
- `artifacts/gpu_tpe_30m3m_funding_snapshot_harness_post705_rebuild_10k20k_20260219T1130Z/pairing/trials_20000/pairing_report.json`

## Results

### Seed 10k sample (10 rows)

- `trade_diff_zero`: `10/10`
- `max_abs_trades_diff`: `0`
- `max_abs_pnl_diff`: `0.003995300290227988`

### Seed 20k sample (10 rows)

- `trade_diff_zero`: `10/10`
- `max_abs_trades_diff`: `0`
- `max_abs_pnl_diff`: `0.10703584229497665`

## Interpretation

- Trade-count parity is now deterministic on sampled rows (`10/10` for both seeds).
- Remaining work is PnL residual tightening, with the current worst sampled residual at `0.10703584229497665`.
- Next investigation target is the row with the largest residual in seed 20k (`sample_10`).

## Notes

- This document records milestone evidence only; it does not claim full live/paper/backtester canonical alignment completion.
