# Rust Runtime Foundation Programme

## Objective

Create the first Rust-owned runtime foundation that can evolve into the single execution path for:

- live trading
- paper trading
- replay
- backtest
- CPU sweep

This programme does not finish the migration. It creates the shared contracts, crates, and documentation needed to deliver the migration as an atomic PR train.

## Deliverables In This Foundation Slice

- Root Rust workspace for non-backtester runtime crates.
- `aiq-runtime-core` pipeline planning crate.
- `aiq-runtime` bootstrap CLI for config loading and pipeline inspection.
- Additive YAML support for `runtime` and `pipeline`.
- ADR, team charter, and legacy ledger to guide later subagent work.

## Runtime Contract

- Existing YAML merge order remains unchanged.
- `runtime.profile` selects the active runtime pipeline profile.
- `pipeline.default_profile` provides the fallback profile.
- `pipeline.profiles.<name>` may override:
  - `ranker`
  - `stage_order`
  - `enabled_stages`
  - `disabled_stages`

Built-in profiles reserved by the Rust runtime:

- `production`
- `parity_baseline`
- `stage_debug`

## PR Train Guidance

1. Extend the shared Rust config and runtime planning surface.
2. Move paper execution and persistence onto Rust.
3. Move live execution, OMS, and reconciliation onto Rust.
4. Replace Python parity orchestration with Rust-native parity tooling.
5. Remove Python runtime code from the active tree once production cutover is complete.

## Non-Goals For This Slice

- No production cutover yet.
- No CUDA parity rewrite yet.
- No DB v2 migration yet.
- No Python deletion yet.
