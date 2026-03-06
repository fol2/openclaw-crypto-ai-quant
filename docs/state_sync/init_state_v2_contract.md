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
- `last_funding_time_ms`
- `last_add_time_ms`
- `entry_adx_threshold`

Runtime fields:

- `entry_attempt_ms_by_symbol`
- `exit_attempt_ms_by_symbol`
- `last_close_info_by_symbol`

`runtime.last_close_info_by_symbol` value shape:

```json
{
  "ETH": {
    "timestamp_ms": 1772676400000,
    "side": "short",
    "reason": "Signal Trigger"
  }
}
```

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
  - `runtime_last_closes`
- `--strict-replace` is the deterministic bootstrap mode.
- Without `--strict-replace`, the command fails closed if stale open paper positions would remain outside the snapshot.

## Paper Runtime Restore Rules

- `aiq-runtime paper doctor` is the current Rust-owned bootstrap/restore shell.
- `aiq-runtime paper run-once` extends the same restored state into one single-shot execution step.
- The paper shell restores state from the paper DB through the same continuation contract:
  - `trades`
  - `position_state`
  - `runtime_cooldowns`
  - `runtime_last_closes`
- A healthy paper restore requires:
  - valid `init-state v2` semantics
  - deterministic symbol normalisation
  - no duplicate restored symbols
  - runtime cooldown markers carried into the Rust-owned in-memory state
  - last close metadata carried into kernel PESC state when present

## Paper Runtime Execution Rules

- `paper run-once` must start from the same restored state that `paper doctor` reports.
- `paper cycle` must also start from the same restored state, but it executes one explicit multi-symbol cycle with a required `--step-close-ts-ms`.
- Both commands remain shell surfaces only in this phase; long-running paper loops remain out of scope.
- Default write timestamps follow execution time for DB parity; pass `--exported-at-ms` when you need reproducible artefacts.
- DB projection after a successful `paper run-once` or `paper cycle` step is limited to the Rust-owned paper projection surface for this phase:
  - `trades`
  - `position_state`
  - `runtime_cooldowns`
  - `runtime_last_closes`
- `paper cycle` also records a rerun guard row in `runtime_cycle_steps` and fails closed if the same step identity is applied twice.

## Backward Compatibility

- `bt-core` continues to read v1 and v2
- new Rust exporters should write v2 only
- Python v1 export should be treated as legacy compatibility, not the forward contract
