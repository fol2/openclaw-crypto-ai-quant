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
4. `hub` exposes service state, backtest controls, logs, and monitoring views

## Persistent State

- trading SQLite DBs for paper/live runtime state
- candle SQLite DBs and optional partition directories
- runtime status files used by service inspection
- snapshot JSON files used for continuation and replay seeding

## Removed Surfaces

The zero-Python cutover removed the legacy runtime, tool, and test tree. The
repository now treats Rust as the only implementation language for active
execution ownership.
