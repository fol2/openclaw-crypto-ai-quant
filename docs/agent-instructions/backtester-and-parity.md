# Backtester and Parity Instructions

Load this file when the task involves replay, sweep, GPU parity, indicator
validation, or backtester-owned code paths.

## Ownership

Backtesting ownership lives in:

- `backtester/crates/bt-cli`
- `backtester/crates/bt-core`
- `backtester/crates/bt-gpu`
- `backtester/crates/risk-core`

Important source files:

- `backtester/crates/bt-core/src/engine.rs`
- `backtester/crates/bt-core/src/decision_kernel.rs`
- `backtester/crates/bt-core/src/behaviour.rs`
- `backtester/crates/bt-signals/src/behaviour.rs`

## First Commands To Reach For

```bash
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- replay --candles-db candles_dbs/candles_1h.db
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- sweep --sweep-config backtester/sweeps/smoke.yaml
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- dump-indicators
```

## Critical Validation Rules

- Keep compared backtests on identical date ranges across intervals.
- Always query actual DB coverage; do not assume fixed history windows.
- Rust indicator outputs must match Python `ta` within `0.00005` absolute
  error.
- Validate indicator changes with `dump-indicators` against Python `ta`.

See also:

- [`docs/current_authoritative_paths.md`](../current_authoritative_paths.md)
- [`docs/runbook.md`](../runbook.md)
