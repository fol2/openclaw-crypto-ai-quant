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

## Delivered Follow-on Slice

The next delivered slice extends the foundation with Rust-owned paper bootstrap tooling:

- `snapshot seed-paper`
- stricter snapshot validation
- `paper doctor` restore shell for Rust-owned paper bootstrap reports

The current delivered slice extends that again with the first Rust-owned paper execution shell:

- `paper run-once`
- single-step paper execution reporting with optional `--exported-at-ms` pinning for reproducible artefacts
- Rust-owned write-back of the paper projection surface for one execution cycle

The current delivered slice extends that further with a repeatable Rust paper orchestration shell:

- `paper cycle`
- explicit `--step-close-ts-ms` cycle identity and rerun guard
- multi-symbol cycle execution across explicit symbols plus open paper positions
- still no daemon/systemd cutover

The current delivered slice extends that again with a bounded Rust paper catch-up shell:

- `paper loop`
- resumes from `runtime_cycle_steps` when prior Rust cycle state exists
- requires `--start-step-close-ts-ms` only for the first bootstrap run on a fresh paper DB
- executes up to `--max-steps` unapplied cycle steps and exits
- optional `--follow` mode keeps polling after catch-up instead of exiting immediately idle
- still no daemon/systemd cutover

The current delivered slice extends that again with an opt-in Rust paper
daemon orchestration surface:

- `paper daemon`
- owns the outer scheduler instead of delegating long-running follow behaviour back to `paper loop`
- reuses the same `paper cycle` step identity and rerun guard through `runtime_cycle_steps`
- optional `--watch-symbols-file` reloads a symbols manifest without restarting, while retaining the last good manifest on invalid or runtime-invalid malformed reloads
- active symbols remain `manifest ∪ open paper positions`, so exit lanes are not dropped during watchlist changes
- long-running orchestration only; still no paper/systemd cutover

The current delivered slice extends that again with a small daemon/watchlist
ownership step:

- `paper loop` only loads `--symbols-file` once at start-up and stays a bounded shell
- `paper daemon --watch-symbols-file` owns the later file-refresh path without changing step identity
- an initially empty daemon watchlist is treated as an idle lane in follow mode rather than a hard startup failure
- explicit `--symbols` are still unioned with the watched file contents
- still no paper/systemd cutover

The current delivered slice extends that again with a read-only Rust paper
service manifest surface:

- `paper manifest`
- resolves the current daemon service contract from existing `AI_QUANT_*` env vars plus optional CLI overrides
- derives the candle DB path from `AI_QUANT_CANDLES_DB_DIR` + resolved interval when needed
- emits the resolved Rust daemon command, warnings, and runtime bootstrap metadata without executing any paper steps

The current delivered slice extends that again with a service-grade launch and
resume contract for the same manifest surface:

- `paper manifest`
- accepts opt-in watchlist/bootstrap launch inputs (`--watch-symbols-file`, `--start-step-close-ts-ms`) as part of the deterministic daemon plan
- reports whether the lane is blocked, idle with no symbols, bootstrap-ready, bootstrap-blocked, resumable, or merely caught up and waiting
- exposes the current active symbols plus `last_applied_step_close_ts_ms`, `next_due_step_close_ts_ms`, and `latest_common_close_ts_ms` when the paper DB and candles DB are inspectable
- resolves the daemon `status_path` so later service supervision can observe the same lane lifecycle contract

The current delivered slice extends that again with a daemon lifecycle status
surface:

- `paper daemon`
- writes a durable lane status JSON while the daemon is running and after it stops
- derives the default `status_path` from the resolved daemon lock path when `AI_QUANT_STATUS_PATH` is unset
- keeps the lifecycle status contract read-only with respect to trading semantics; it does not widen DB projections or claim systemd cutover

The current delivered slice extends that again with effective paper-service
config parity:

- Rust paper surfaces now apply `AI_QUANT_PROMOTED_ROLE` against the latest promoted YAML before building the active runtime config
- `AI_QUANT_STRATEGY_MODE` remains the first strategy-mode selector, with `AI_QUANT_STRATEGY_MODE_FILE` as the same file-backed fallback used by the Python paper service
- `paper manifest`, `paper doctor`, `paper cycle`, `paper loop`, and `paper daemon` now share the same Rust-owned effective config document instead of reporting those service inputs as metadata only
- still no paper/systemd cutover

Python paper execution is still the active runtime path, and the opt-in Rust
paper daemon wrapper does not change that. Python paper bootstrap is no longer
the only continuity surface.

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
