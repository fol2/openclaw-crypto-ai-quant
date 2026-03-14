# openclaw-crypto-ai-quant

Rust-native crypto perpetual futures stack for Hyperliquid DEX.

The repository is now zero-Python. Production trading, backtesting, market-data
ingestion, and operator-facing control surfaces all live in Rust or shell.
Historical Python runtime and tooling surfaces have been removed from the tree.

## Components

| Component | Path | Purpose |
|---|---|---|
| Runtime CLI | `runtime/aiq-runtime` | Paper/live control plane, daemon ownership, snapshots, service inspection |
| Runtime core | `runtime/aiq-runtime-core` | Shared stage + behaviour pipeline and bootstrap contracts |
| Backtester CLI | `backtester/crates/bt-cli` | Replay, sweep, and indicator dump commands |
| Backtester core | `backtester/crates/bt-core` | Simulation engine, config, indicators, state |
| GPU sweep | `backtester/crates/bt-gpu` | CUDA sweep acceleration and parity helpers |
| Data loader | `backtester/crates/bt-data` | SQLite candle and universe-history loading |
| WS sidecar | `ws_sidecar/` | Hyperliquid stream ingestion and candle persistence |
| Hub | `hub/` | Rust + frontend operator dashboard, symbol-detail trade review, and service controls |

## Quick Start

```bash
git clone https://github.com/fol2/openclaw-crypto-ai-quant.git
cd openclaw-crypto-ai-quant

cargo build --workspace

# Paper lane
cargo run -p aiq-runtime -- paper daemon --lane paper1 --project-dir "$PWD"

# Inspect live contract without executing orders
cargo run -p aiq-runtime -- live manifest --project-dir "$PWD" --json
```

For release-style backtester binaries:

```bash
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli --features gpu
```

## Development

```bash
# Full workspace
cargo check --workspace
cargo test --workspace

# Formatting and linting
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Backtester-only
cargo test --manifest-path backtester/Cargo.toml
```

Optional local helpers used by some scripts:

- `jq`
- `sqlite3`
- CUDA toolkit for GPU sweeps

## Behaviour-Modular Runtime

The Rust runtime now supports both:

- **stage-modular control** for coarse pipeline ownership
- **behaviour-modular control** for fine-grained decision behaviour debugging

Behaviour profiles live under `pipeline.profiles.<name>.behaviours` and can
explicitly reorder or disable gate, signal, exit, sizing, progression, and risk
behaviours without editing Rust code. Exit behaviour plans now control the
actual stop-loss, trailing, take-profit, and smart-exit sequence rather than
stopping at stage-level ownership.

Shipped example configs include two opt-in parity lanes:

- `parity_baseline`: explicit production-like behaviour ordering with broker/fill stages disabled
- `parity_exit_isolation`: parity baseline plus disabled exit modifiers for stop-loss/trailing/full-TP debugging

Inspect them with:

```bash
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_baseline --json
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_exit_isolation --json
```

## Operations

Shell wrappers and systemd examples are provided for the Rust-owned runtime:

- `scripts/run_paper_lane.sh`
- `scripts/run_live.sh`
- `scripts/run_paper.sh`
- `systemd/openclaw-ai-quant-hub.service.example`
- `systemd/openclaw-ai-quant-trader-v8-paper*.service.example`
- `systemd/openclaw-ai-quant-live-v8.service.example`
- `systemd/openclaw-ai-quant-live-db-sync-v8.service.example`
- `systemd/openclaw-ai-quant-live-db-sync-v8.timer.example`
- `systemd/openclaw-ai-quant-funding-v8.service.example`
- `systemd/openclaw-ai-quant-prune-runtime-logs-v8.service.example`
- `systemd/openclaw-ai-quant-ws-sidecar.service.example`

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/current_authoritative_paths.md](docs/current_authoritative_paths.md)
- [docs/runbook.md](docs/runbook.md)
- [runtime/README.md](runtime/README.md)
- [backtester/README.md](backtester/README.md)

## Licence

MIT
