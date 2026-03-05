# Paper Seed from Canonical Snapshot

## Objective

Apply a live-canonical snapshot into the paper DB so `PaperTrader.load_state()` reconstructs the same position baseline as the replay seed.

## Export Canonical Snapshot

```bash
python tools/export_live_canonical_snapshot.py \
  --source live \
  --output /tmp/live_init_state_v2.json
```

## Apply Snapshot to Paper DB

```bash
cargo run -p aiq-runtime -- \
  snapshot seed-paper \
  --snapshot /tmp/live_init_state_v2.json \
  --target-db ./trading_engine.db \
  --strict-replace \
  --json
```

This performs:

- synthetic `trades` seed rows (`reason=state_sync_seed`)
- balance seed row (`reason=state_sync_balance_seed`)
- `position_state` refresh aligned to seeded open trade IDs
- `position_state_history` seed rows for bootstrap provenance
- `runtime_cooldowns` seed (entry/exit cooldown maps from snapshot runtime state)

`--strict-replace` is the deterministic mode and clears existing seed targets before writing.  
Without `--strict-replace`, the Rust command fails closed when the paper DB still has open positions outside the snapshot surface.

## Deterministic Replay Pairing

Use the same snapshot as the backtester seed:

```bash
cd backtester
./target/release/mei-backtester replay \
  --candles-db ../candles_dbs/candles_1h.db \
  --init-state /tmp/live_init_state_v2.json
```

With this pairing, paper and backtester start from the same canonical seed artefact.

## Legacy Tooling

`tools/apply_canonical_snapshot_to_paper.py` is now a legacy/frozen reference path.  
New paper bootstrap workflows should use `aiq-runtime snapshot seed-paper`.
