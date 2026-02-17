# Live Replay Bundle

## Objective

Create a deterministic replay bundle for a fixed live window so all alignment runs use the same inputs and commands.

## Build Bundle

```bash
python tools/build_live_replay_bundle.py \
  --live-db ./trading_engine_live.db \
  --paper-db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_1h.db \
  --funding-db ./backtester/data/funding_rates_1h.db \
  --interval 1h \
  --from-ts 1770700000000 \
  --to-ts 1771200000000 \
  --bundle-dir /tmp/live_replay_bundle_1h
```

## Bundle Contents

- `replay_bundle_manifest.json`
- `live_baseline_trades.jsonl`
- `backtester_replay_report.json`
- `run_01_export_and_seed.sh`
- `run_02_replay.sh`
- `run_03_audit.sh`
- `run_04_trade_reconcile.sh`
- `run_05_action_reconcile.sh`

## Usage Sequence

1. Run `run_01_export_and_seed.sh`
2. Run `run_02_replay.sh`
3. Run `run_03_audit.sh`
4. Run `run_04_trade_reconcile.sh`
5. Run `run_05_action_reconcile.sh`

This keeps snapshot seeding, replay execution, state alignment audit, exit-trade reconciliation, and action-level reconciliation tied to one immutable bundle manifest.

## Execution Context

The generated scripts auto-detect bundle-local artefact paths via `BUNDLE_DIR` and use `REPO_ROOT` for tool/binary paths.

- default `REPO_ROOT` is the current shell working directory
- set `REPO_ROOT=/path/to/openclaw-crypto-ai-quant` when running outside repo root
