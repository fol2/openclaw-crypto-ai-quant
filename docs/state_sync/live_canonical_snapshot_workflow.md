# Live-Canonical Snapshot Workflow

## Objective

Seed `paper` and `backtester` from one canonical continuation contract so replay and parity checks are deterministic and auditable.

The continuation surface is moving toward a Rust-owned model:

- `aiq-runtime snapshot export-paper` is now the preferred paper export path
- `aiq-runtime snapshot seed-paper` is now the preferred paper bootstrap path
- live canonical export remains valid until the Rust live adapter owns exchange truth
- `init-state v2` is the canonical contract because it carries runtime cooldown markers

## Snapshot Export

Paper snapshot export now has a Rust-owned path:

```bash
cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot export-paper --db trading_engine.db --output /tmp/paper_init_state_v2.json

cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot validate --path /tmp/paper_init_state_v2.json --json

cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot seed-paper --snapshot /tmp/paper_init_state_v2.json --target-db trading_engine.db --strict-replace --json
```

Live canonical export remains:

Export a v2 init-state snapshot from the live database:

```bash
python tools/export_live_canonical_snapshot.py \
  --source live \
  --output /tmp/live_init_state_v2.json
```

The output includes:

- `balance`
- `positions`
  - for `--as-of-ts` snapshots, position runtime fields (`adds_count`, `tp1_taken`, `last_add_time_ms`) are reconstructed from historical `ADD`/`REDUCE` fills
  - `last_funding_time_ms` is carried from persisted position state when available
  - `margin_used` is recalculated from reconstructed `size * entry_price / leverage` so reduced positions carry correct margin headroom
- `runtime`
  - `entry_attempt_ms_by_symbol`
  - `exit_attempt_ms_by_symbol`
  - `last_close_info_by_symbol`
- `runtime.entry_attempt_ms_by_symbol`
- `runtime.exit_attempt_ms_by_symbol`
- canonical metadata (`open_orders`, `cursors`, warnings)

When using the Rust paper exporter, runtime markers come from `runtime_cooldowns` when available.

## Backtester Replay Seed

Use the exported file directly with replay:

```bash
cd backtester
./target/release/mei-backtester replay \
  --candles-db ../candles_dbs/candles_1h.db \
  --init-state /tmp/live_init_state_v2.json
```

`bt-core` accepts v1 and v2 init-state schemas. With v2, cooldown markers are restored into runtime state.

## Paper Alignment Notes

This snapshot is the canonical seed artefact for paper/backtester alignment. For paper DB mirroring, continue using `tools/mirror_live_state.py` where needed for OMS table projection.

For paper-only continuation, prefer the Rust snapshot + `seed-paper` path so the Python export and seed surfaces can be retired mode by mode.

## Audit Expectations

For each replay/alignment run, record:

- snapshot file path + hash
- source DB path
- replay window (`from_ts`, `to_ts`)
- first divergence event (if any)
- mismatch classification (logic, numeric, state, data, non-simulatable OMS)
