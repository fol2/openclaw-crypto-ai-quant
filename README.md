# openclaw-crypto-ai-quant

AI-powered crypto perpetual futures trading engine for [Hyperliquid DEX](https://hyperliquid.xyz), with a high-performance Rust backtester featuring CPU and CUDA GPU acceleration.

## Features

- **Unified paper / live trading** via a single daemon (`engine.daemon`)
- **Mei Alpha v1 strategy**: multi-indicator, confidence-ranked entries with ATR-based risk management
- **Rust decision kernel** (`bt-signals`): shared signal logic across backtester, GPU sweep, and live trading (via PyO3 bridge)
- **Kernel orchestrator + shadow mode**: gradual Python → Rust cutover with parallel decision tracking
- **Signal ranking**: `score = confidence_rank × 100 + ADX`; two-phase collect-rank-execute
- **Market breadth filter**, auto-reverse, regime gating, ATR floor
- **Rust backtester**: CPU replay and CUDA GPU parameter sweeps (60K combos in ~3 s)
- **TPE Bayesian optimisation** for parameter search
- **Multi-interval backtesting** (entry on 1h, exit on 1m/3m/5m)
- **Strategy factory pipeline**: nightly sweep → validate → deploy → paper → promote → live ramp
- **Strategy lifecycle state machine**: candidate → validated → paper → live_small → live_full → paused → retired
- **Risk manager**: daily loss limits, drawdown kill-switch, rate limiting, exposure caps, slippage guard
- **Hot-reloadable YAML configuration** with merge chain (`defaults ← global YAML ← per-symbol ← live`)
- **Config deploy pipeline**: sweep → deploy → export state → replay with init-state
- **Funding rate simulation** in backtester and paper trader
- **Ensemble runner**: parallel multi-strategy daemons with independent sizing budgets
- **Alerting**: Discord / Telegram via `openclaw message send`
- **Real-time monitor dashboard** (Python) + **Hub dashboard** (Rust + Svelte)
- **WS sidecar** (Rust): streams Hyperliquid market data over Unix socket; maintains candle DBs
- **Deterministic replay alignment gate**: CI-integrated parity checks between paper/live and backtester
- **113 test files**, ~22K lines of Python tests

## Architecture

```
                    ┌─────────────┐
                    │  Hyperliquid │
                    │     DEX      │
                    └──────┬───────┘
                           │ WS + REST
                    ┌──────▼───────┐
                    │  WS Sidecar  │  (Rust)
                    │ (market data) │
                    └──────┬───────┘
                           │ Unix socket
              ┌────────────▼────────────┐
              │     Unified Engine      │
              │       (engine/)         │
              ├──────────┬──────────────┤
              │  Paper   │    Live      │
              │ Trader   │   Trader     │
              ├──────────┴──────────────┤
              │   Kernel Orchestrator   │
              │  (Rust bt-runtime/PyO3) │
              ├─────────────────────────┤
              │     Risk Manager        │
              ├─────────────────────────┤
              │   Order Mgmt System     │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │  Monitor   │   │  Hub (Rust  │   │  Alerting   │
   │ Dashboard  │   │  + Svelte)  │   │  (Discord/  │
   │ (Python)   │   │             │   │  Telegram)  │
   └────────────┘   └─────────────┘   └─────────────┘

Standalone:
   ┌──────────────────────────────────────────────┐
   │          Rust Backtester (CPU / CUDA GPU)     │
   │  bt-core · bt-signals · bt-gpu · risk-core   │
   └──────────────────────────────────────────────┘

Nightly pipeline:
   ┌──────────────────────────────────────────────┐
   │  Strategy Factory (factory_run.py / cycle)    │
   │  sweep → validate → deploy → paper → promote  │
   └──────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/fol2/openclaw-crypto-ai-quant.git
cd openclaw-crypto-ai-quant

# Python setup (requires Python >=3.12 and uv)
uv sync --dev
source .venv/bin/activate

# Configure
cp .env.example .env
# Paper mode does NOT require secrets.
# For dry_live/live, keep secrets OUTSIDE the repo:
mkdir -p ~/.config/openclaw
cp config/secrets.json.example ~/.config/openclaw/ai-quant-secrets.json
chmod 600 ~/.config/openclaw/ai-quant-secrets.json
# Edit .env and secrets file with your values

# Run paper trader
AI_QUANT_MODE=paper python -m engine.daemon
```

## Backtester

The Rust backtester provides high-performance simulation with CPU and GPU acceleration. All execution paths share a single decision kernel (`bt-signals`).

```bash
# Build (CPU)
python3 tools/build_mei_backtester.py

# Build (GPU, requires CUDA toolkit)
python3 tools/build_mei_backtester.py --gpu

# Single replay
mei-backtester replay --candles-db candles_dbs/candles_1h.db

# Parameter sweep
mei-backtester sweep --sweep-config sweeps/smoke.yaml

# GPU sweep
mei-backtester sweep --gpu --sweep-spec sweeps/allgpu_60k.yaml

# TPE Bayesian optimisation
mei-backtester sweep --gpu --tpe --tpe-trials 5000 --sweep-spec sweep.yaml
```

See [backtester/README.md](backtester/README.md) for detailed documentation including candle DB partitioning, universe filtering, GPU parity lanes, and the config deploy pipeline.

## Strategy Factory

`factory_run.py` drives the nightly strategy generation pipeline. All outputs are written under the artifacts root (default: `artifacts/`) using a date-scoped layout:

```
artifacts/
  YYYY-MM-DD/
    run_<run_id>/
      data_checks/
      sweeps/
      configs/
      replays/
      reports/
      logs/
      run_metadata.json
```

The factory lifecycle is: **sweep → validate (OOS + slippage stress) → deploy to paper → gate checks → promote to live → ramp (25% → 50% → 100%)**. See [docs/strategy_lifecycle.md](docs/strategy_lifecycle.md) and [docs/success_metrics.md](docs/success_metrics.md) for details.

## Configuration

Strategy configuration is managed through `config/strategy_overrides.yaml`, which supports hot-reload at runtime via mtime polling. The merge order is:

```
_DEFAULT_STRATEGY_CONFIG ← global YAML ← symbols.<SYM> YAML ← live YAML (if live mode)
```

Key configuration sections:

- **trade**: Position sizing, leverage, SL/TP multipliers, pyramiding, partial TP, dynamic sizing
- **market_regime**: Breadth thresholds, auto-reverse, regime filter
- **filters**: Ranging regime, anomaly, BTC alignment, market breadth entry gates
- **thresholds.entry**: Min ADX, pullback %, slow drift
- **indicators**: EMA, ADX, Bollinger Bands window sizes
- **engine**: Candle interval, entry/exit intervals

For live/dry_live operation, review `.env.example` and `systemd/ai-quant-live.env.example`. Notable safety controls:

- `AI_QUANT_LIVE_ENABLE` + `AI_QUANT_LIVE_CONFIRM`: required to send real orders
- `AI_QUANT_KILL_SWITCH`: `close_only` or `halt_all`
- `AI_QUANT_KILL_SWITCH_FILE`: file-based kill-switch (no restart needed)
- `AI_QUANT_OMS_REQUIRE_INTENT_FOR_ENTRY`: fail-closed entry dedupe (default: enabled)
- `AI_QUANT_RISK_MAX_DRAWDOWN_PCT`: equity drawdown % auto-kill

## Deployment

Systemd user service templates are provided in `systemd/` for production deployment:

| Service | Template | Purpose |
|---------|----------|---------|
| Paper trader | `openclaw-ai-quant-trader.service.example` | Paper trading daemon |
| Live trader | `openclaw-ai-quant-live.service.example` | Live trading daemon |
| WS sidecar | `openclaw-ai-quant-ws-sidecar.service.example` | Market data WebSocket |
| Monitor | `openclaw-ai-quant-monitor.service.example` | Real-time monitoring dashboard |
| Factory | `openclaw-ai-quant-factory.{service,timer}.example` | Nightly strategy sweep |
| Log pruning | `openclaw-ai-quant-prune-runtime-logs.{service,timer}.example` | SQLite log retention |
| Replay gate | `openclaw-ai-quant-replay-alignment-gate.{service,timer}.example` | Deterministic replay checks |
| Funding sync | `openclaw-ai-quant-funding-v8.{service,timer}.example` | Funding rate backfill |

Copy relevant templates to `~/.config/systemd/user/`, customise paths and environment variables, then enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now <unit-name>
```

## Project Structure

```
.
├── engine/                    # Unified trading engine (Python)
│   ├── core.py                # UnifiedEngine — main trading loop
│   ├── daemon.py              # Entrypoint daemon (paper / dry_live / live)
│   ├── market_data.py         # MarketDataHub — candle + mid data
│   ├── strategy_manager.py    # YAML hot-reload via mtime polling
│   ├── oms.py                 # Order Management System (live dedupe + audit)
│   ├── oms_reconciler.py      # OMS reconciliation
│   ├── risk.py                # RiskManager — rate limits, drawdown, kill-switch
│   ├── alerting.py            # Discord / Telegram notifications
│   ├── event_logger.py        # Decision + trade event logging
│   ├── promoted_config.py     # Promoted config loading
│   ├── rest_client.py         # Hyperliquid REST client
│   ├── sqlite_logger.py       # SQLite trade/candle logger
│   └── systemd_watchdog.py    # sd_notify watchdog
├── strategy/                  # Strategy implementations
│   ├── mei_alpha_v1.py        # Signals, confidence, PaperTrader
│   ├── kernel_orchestrator.py # Rust kernel orchestrator (PyO3 bridge)
│   ├── broker_adapter.py      # Kernel OrderIntent → exchange orders
│   ├── shadow_mode.py         # Shadow mode decision tracking
│   ├── reconciler.py          # Position reconciliation
│   └── event_id.py            # Deterministic event IDs
├── exchange/                  # Exchange adapters
│   ├── executor.py            # HyperliquidLiveExecutor (SDK interface)
│   ├── ws.py                  # WebSocket client
│   ├── sidecar.py             # WS sidecar Unix socket client
│   ├── meta.py                # Hyperliquid metadata + sz rounding
│   └── market_watch.py        # Market watch utilities
├── live/                      # Live trading
│   └── trader.py              # LiveTrader
├── monitor/                   # Real-time dashboard (Python)
│   ├── server.py              # HTTP + SSE server
│   └── heartbeat.py           # Daemon heartbeat checks
├── hub/                       # Hub dashboard (Rust + Svelte)
│   ├── src/                   # Axum backend
│   └── frontend/              # Svelte frontend
├── config/                    # Runtime configuration
│   ├── strategy_overrides.yaml.example
│   ├── strategy_overrides.live.example.yaml
│   ├── strategy_overrides.promoted_*.example.yaml
│   ├── ensemble.example.yaml
│   └── secrets.json.example
├── tools/                     # Operational tools (60+ scripts)
│   ├── deploy_sweep.py        # Deploy sweep results to YAML
│   ├── export_state.py        # Export paper/live state to JSON
│   ├── factory_cycle.py       # Strategy factory automation
│   ├── promote_to_live.py     # Paper → live promotion
│   ├── paper_deploy.py        # Paper deployment
│   ├── flat_now.py            # Emergency flatten + pause
│   ├── manual_trade.py        # Manual trade entry/exit
│   ├── validate_config.py     # Config parity validation
│   ├── reality_check.py       # Quick trading diagnostics
│   ├── rollback_to_last_good.py
│   └── ...                    # Audit, analysis, CI tools
├── backtester/                # Rust backtester (Cargo workspace)
│   ├── crates/bt-core/        # Simulation engine + indicators
│   ├── crates/bt-signals/     # Decision kernel (shared signal logic)
│   ├── crates/bt-data/        # SQLite candle loader
│   ├── crates/bt-cli/         # CLI (replay, sweep, dump-indicators)
│   ├── crates/bt-gpu/         # CUDA GPU sweep + TPE
│   ├── crates/bt-runtime/     # PyO3 bridge to Python
│   ├── crates/risk-core/      # Rust risk primitives
│   └── sweeps/                # Sweep configs + runner scripts
├── ws_sidecar/                # Rust WS sidecar
├── analysis/                  # Post-trade analysis
├── research/                  # Strategy research modules
├── schemas/                   # JSON schemas (GPU candidate, etc.)
├── scripts/                   # Shell scripts (CI gates, run_paper/live)
├── sweep_full/                # Full sweep runner scripts
├── systemd/                   # Service + timer templates
├── tests/                     # Python tests (113 files, ~22K lines)
├── docs/                      # Documentation
├── plan/                      # Planning documents
├── factory_run.py             # Strategy factory entrypoint
├── diag_*.py                  # Diagnostic scripts
├── pyproject.toml             # Python project config (uv)
├── VERSION                    # Single source of truth for version
├── AGENTS.md                  # AI agent instructions
└── CLAUDE.md                  # Claude Code pointer
```

## Development

### Python

```bash
uv sync --dev                  # Install dependencies
uv run pytest                  # Run tests
uv run ruff check engine strategy exchange live tools tests monitor
uv run ruff format engine strategy exchange live tools tests monitor
```

### Rust

```bash
cd backtester && cargo build --release   # Backtester
cd ws_sidecar && cargo build --release   # WS sidecar
cd hub && cargo build --release          # Hub dashboard
cargo test && cargo fmt --check && cargo clippy -- -D warnings
```

### Release Process

Version is governed by `VERSION` (single source of truth). All `pyproject.toml` and `Cargo.toml` versions must match. See [docs/release_process.md](docs/release_process.md) for the tag-driven release flow.

```bash
tools/release/set_version.sh 0.1.1    # Bump version everywhere
tools/release/check_versions.sh        # Verify consistency
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design and component interactions
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Development setup and guidelines
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — Contribution guidelines
- [docs/release_process.md](docs/release_process.md) — Version governance and tag-driven release flow
- [docs/runbook.md](docs/runbook.md) — Operations runbook (emergency stop, rollback, diagnostics)
- [docs/strategy_lifecycle.md](docs/strategy_lifecycle.md) — Config state machine (candidate → live)
- [docs/success_metrics.md](docs/success_metrics.md) — Risk limits and promotion criteria
- [backtester/README.md](backtester/README.md) — Backtester documentation
- [monitor/README.md](monitor/README.md) — Dashboard documentation
- [engine/README.md](engine/README.md) — Engine internals

## License

MIT
