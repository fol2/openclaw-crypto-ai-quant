# Init-State V2 Contract

## Purpose

`init-state v2` is the canonical continuation format for Rust-owned replay and paper seeding.

Unlike v1, v2 carries both:

- `balance + positions`
- runtime continuation markers under `runtime`

## Required Fields

Top-level fields:

- `version`
- `source`
- `exported_at_ms`
- `balance`
- `positions`
- `runtime`

Position fields:

- `symbol`
- `side`
- `size`
- `entry_price`
- `entry_atr`
- `trailing_sl`
- `confidence`
- `leverage`
- `margin_used`
- `adds_count`
- `tp1_taken`
- `open_time_ms`
- `last_add_time_ms`
- `entry_adx_threshold`

Runtime fields:

- `entry_attempt_ms_by_symbol`
- `exit_attempt_ms_by_symbol`

## Validation Rules

- `version` must be `2`
- `balance` must be non-negative
- `side` must be `long` or `short`
- `confidence` must be `low`, `medium`, or `high`
- `leverage` and `size` must be positive
- the file must be loadable by `bt-core::init_state::load`

## Current Commands

Paper export:

```bash
cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot export-paper --db trading_engine.db --output /tmp/paper_init_state_v2.json
```

Validation:

```bash
cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot validate --path /tmp/paper_init_state_v2.json --json
```

Paper seed:

```bash
cargo run --manifest-path Cargo.toml -p aiq-runtime -- \
  snapshot seed-paper --snapshot /tmp/paper_init_state_v2.json --target-db trading_engine.db --strict-replace --json
```

## Paper Seed Rules

- `snapshot seed-paper` is a write path and must validate the snapshot before writing.
- The Rust seed path currently rewrites these paper DB projection targets:
  - `trades`
  - `position_state`
  - `position_state_history`
  - `runtime_cooldowns`
- `--strict-replace` is the deterministic bootstrap mode.
- Without `--strict-replace`, the command fails closed if stale open paper positions would remain outside the snapshot.

## Backward Compatibility

- `bt-core` continues to read v1 and v2
- new Rust exporters should write v2 only
- Python v1 export should be treated as legacy compatibility, not the forward contract
