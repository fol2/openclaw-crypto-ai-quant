# Architecture

## Overview

The repository is organised around four Rust-owned execution surfaces:

1. `runtime/aiq-runtime` for paper/live trading control and daemon ownership
2. `backtester/` for replay, sweeps, and indicator validation
3. `ws_sidecar/` for Hyperliquid market-data ingestion
4. `hub/` for operator dashboard and service inspection

## Runtime Path

`aiq-runtime` is the authoritative trading entrypoint.

- `paper` subcommands own paper effective-config, manifests, service state, and daemon execution
- `live` subcommands own live effective-config, manifests, service state, and daemon execution
- `snapshot` subcommands own paper snapshot export and paper seeding
- `pipeline` and `doctor` expose runtime inspection contracts

The runtime contract is now split into:

- a **stage plan** that controls coarse pipeline ownership boundaries
- a **behaviour plan** that controls concrete gate, signal, exit, sizing,
  progression, and risk behaviours within those stages

This means parity-debugging lanes no longer need code edits just to suppress or
re-sequence one internal behaviour.

That behaviour plan is now enforced all the way down to exit execution. The
runtime and backtester honour the configured exit ordering across stop-loss,
trailing, take-profit, and smart-exit paths, and diagnostics carry a
behaviour-level trace for the executed path.

Shell wrappers in `scripts/` and service examples in `systemd/` call these
Rust-owned surfaces directly.

## Backtester Path

`backtester/` is a Cargo workspace with:

- `bt-core`: simulation engine, config, state, indicators
- `bt-signals`: shared decision logic
- `bt-data`: SQLite loaders
- `bt-cli`: replay/sweep CLI
- `bt-gpu`: CUDA sweep acceleration
- `risk-core`: shared risk primitives

## Data Flow

1. `ws_sidecar` ingests Hyperliquid feeds and persists candles
2. `aiq-runtime` resolves config and executes paper/live cycles
3. `bt-cli` replays the same strategy/config contract offline
4. `hub` exposes service state, backtest/sweep controls, logs, and monitoring views

Hub sweep jobs now persist per-run JSONL artefacts under
`artifacts/sweeps/<job-id>.jsonl`. The Hub sweep route reads structured stdout
when the backtester returns it directly and otherwise falls back to that JSONL
file so the UI can still inspect candidate-family sweeps across `paper1`,
`paper2`, and `paper3`.

Hub backtest jobs now request equity-curve output from the Rust replay path,
and the Hub backtest API normalises legacy result keys such as `total_pnl` and
`max_drawdown_pct` into the page contract before rendering. This keeps older
and newer replay artefacts readable through one operator-facing surface.

The Hub dashboard now normalises legacy `paper` mode selections onto the
canonical candidate family before querying monitor APIs, and its websocket
client accepts topic labels exposed through `topic`, `channel`, `stream`, or
`type`, with payloads carried in `data`, `payload`, `body`, or `message`.
This keeps older and newer Hub feed envelopes readable through the same
operator-facing dashboard surface.

Hub symbol-detail transaction views use the trading DB as the primary source of
position entries and fall back to the latest reconstructed trade journey when a
live-authoritative position has no ledger `open_trade_id`. This keeps
`OPEN`/`ADD`/`REDUCE`/`CLOSE` legs visible in detail and journey review flows
even when the live snapshot is authoritative.

## Behaviour-Modular Contract

Behaviour-level configuration lives under
`pipeline.profiles.<profile>.behaviours` in the strategy YAML merge contract.

Current first-class behaviour groups are:

- `gates`
- `signal_modes`
- `signal_confidence`
- `exits`
- `engine`
- `entry_sizing`
- `entry_progression`
- `risk`

Each group supports:

- `order`: preferred execution order
- `enabled`: explicit allow-list
- `disabled`: explicit block-list

The shipped example config keeps `production` as the default profile and adds
two opt-in parity lanes:

- `parity_baseline`: explicit behaviour ordering for reproducible parity/debug inspection
- `parity_exit_isolation`: parity baseline with exit modifiers and smart exits disabled

## Persistent State

- trading SQLite DBs for paper/live runtime state
- candle SQLite DBs and optional partition directories
- runtime status files used by service inspection and paper monitor freshness fallback
- snapshot JSON files used for continuation and replay seeding

Paper monitor health continues to prefer the legacy `runtime_logs` heartbeat
when available, but now falls back to the paper daemon status files when that
heartbeat is stale or missing. Live `OFF` remains a service-state concept owned
by the Hub/systemd inspection path rather than by the paper status-file
fallback.

## Removed Surfaces

The zero-Python cutover removed the legacy runtime, tool, and test tree. The
repository now treats Rust as the only implementation language for active
execution ownership.
