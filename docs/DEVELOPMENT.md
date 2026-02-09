# AI Quant — Development Guide

This repository contains both Python (trading engine, strategies, monitoring) and Rust (backtester, WS sidecar) components.

## Python Development (uv / ruff / pytest)

This project is managed by `uv`:

- Project config: `./pyproject.toml`
- Lockfile: `./uv.lock`
- Dev venv (uv-managed): `./.venv/` (ignored by git)
- Runtime venv (systemd scripts): `./venv/` (unchanged; used by `run_live.sh` / `run_paper.sh`)

### Setup

```bash
uv sync --dev
```

### Lint / Format

```bash
uv run ruff check engine strategy exchange live tools tests
uv run ruff format engine strategy exchange live tools tests
```

### Tests

```bash
uv run pytest
```

Notes:
- `pytest` enforces 100% coverage for `engine/sqlite_logger.py` and `monitor/heartbeat.py`.

## Rust Development (Backtester + WS Sidecar)

### Build

```bash
# Backtester (CPU)
cd backtester && cargo build --release

# Backtester (GPU, requires CUDA toolkit)
cd backtester && cargo build --release -p bt-cli --features gpu

# WS Sidecar
cd ws_sidecar && cargo build --release
```

### Test

```bash
cd backtester && cargo test
cd ws_sidecar && cargo test
```

### Lint

```bash
cargo fmt --check
cargo clippy -- -D warnings
```

### Notes

- Cargo.lock files are tracked in version control for reproducibility.
- GPU builds require NVIDIA CUDA Toolkit installed on the system.

## Project Structure

- `engine/` — Core trading engine (Python)
- `strategy/` — Strategy implementations (Python)
- `exchange/` — Exchange adapters (Python)
- `live/` — Live trading (Python)
- `tools/` — Operational tools (Python)
- `config/` — Runtime configuration (YAML)
- `backtester/` — High-performance backtesting framework (Rust)
- `ws_sidecar/` — WebSocket data streaming sidecar (Rust)
- `monitor/` — Read-only web dashboard (Python)
- `tests/` — Unit and integration tests
- `config/strategy_overrides.yaml` — Per-symbol strategy parameters (hot-reloads)
