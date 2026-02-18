# openclaw-crypto-ai-quant

AI-powered crypto perpetual futures trading engine for Hyperliquid DEX, with GPU-accelerated backtesting.

## Features

- Unified paper/live trading via single daemon
- Mei Alpha v1 strategy: multi-indicator, confidence-ranked entries
- Signal ranking: score = confidence_rank * 100 + ADX
- Market breadth filter, auto-reverse, ATR floor
- Rust backtester: 149K bars x 51 symbols in ~0.075s
- CUDA GPU sweep: 60K param combos in ~3s
- TPE Bayesian optimization for parameter search
- Multi-interval backtesting (entry on 1h, exit on 1m/3m/5m)
- Strategy hot-reload via YAML
- Config deploy pipeline: sweep → deploy → export state → replay
- Funding rate simulation
- Real-time monitor dashboard
- WS sidecar for Hyperliquid market data

## Architecture

```
                    ┌─────────────┐
                    │  Hyperliquid │
                    │     DEX      │
                    └──────┬───────┘
                           │ WS + REST
                    ┌──────▼───────┐
                    │  WS Sidecar  │ (Rust)
                    │  (market data)│
                    └──────┬───────┘
                           │ Unix socket
              ┌────────────▼────────────┐
              │    Unified Engine       │
              │      (engine/)          │
              ├─────────┬───────────────┤
              │  Paper   │    Live      │
              │ Trader   │   Trader     │
              └─────────┴───────────────┘
                           │
              ┌────────────▼────────────┐
              │   Monitor Dashboard     │
              └─────────────────────────┘

   Standalone:
              ┌─────────────────────────┐
              │   Rust Backtester       │
              │  (CPU / CUDA GPU)       │
              └─────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/fol2/openclaw-crypto-ai-quant.git
cd openclaw-crypto-ai-quant

# Python setup (recommended: uv)
uv sync --dev
source .venv/bin/activate

# Configure
cp .env.example .env
# Paper mode does NOT require secrets.
# For dry_live/live, keep secrets OUTSIDE the repo:
mkdir -p ~/.config/openclaw
cp config/secrets.json.example ~/.config/openclaw/ai-quant-secrets.json
chmod 600 ~/.config/openclaw/ai-quant-secrets.json
# Edit .env and ~/.config/openclaw/ai-quant-secrets.json with your values

# Run paper trader
AI_QUANT_MODE=paper python -m engine.daemon
```

## Backtester

The Rust backtester provides high-performance simulation with CPU and GPU acceleration.

```bash
# Build (CPU)
cd backtester && cargo build --release

# Single replay
./target/release/mei-backtester replay --candles-db ../candles_dbs/candles_1h.db

# Parameter sweep
./target/release/mei-backtester sweep --sweep-config sweeps/smoke.yaml

# GPU sweep (requires CUDA)
cargo build --release -p bt-cli --features gpu
./target/release/mei-backtester sweep --gpu --sweep-spec sweeps/allgpu_60k.yaml

# TPE Bayesian optimization
./target/release/mei-backtester sweep --gpu --tpe --tpe-trials 5000 --sweep-spec sweep.yaml
```

See [backtester/README.md](backtester/README.md) for detailed documentation.

## Strategy Factory Artifacts

`factory_run.py` writes all outputs under the artifacts root directory (default: `artifacts/`) using a date-scoped layout:

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

This layout is designed for reproducibility. A run directory should be self-contained: you can inspect the inputs, commands, and outputs without guessing what was executed.

## Configuration

Strategy configuration is managed through `config/strategy_overrides.yaml`, which supports hot-reload at runtime. The unified daemon watches this file and applies changes without restart.

Key configuration sections:

- **trade**: Position sizing, leverage, stop-loss/take-profit multipliers, pyramiding rules
- **market_regime**: Volatility and trend classification thresholds
- **filters**: Market breadth, volume, correlation filters
- **indicators**: ADX, RSI, MACD, Bollinger Bands, and other technical indicator parameters

Both the backtester and live engine read from the same YAML file, ensuring consistent parameter values across simulation and production.

For live/dry_live operation, review `.env.example` and `systemd/ai-quant-live.env.example`. Notable safety controls:

- `AI_QUANT_LIVE_ENABLE` + `AI_QUANT_LIVE_CONFIRM`: required to send real orders
- `AI_QUANT_OMS_REQUIRE_INTENT_FOR_ENTRY`: when enabled (default), entry orders fail-closed if an OMS intent cannot be created (prevents untracked duplicates)

## Deployment

Systemd service templates are provided in the `systemd/` directory for production deployment:

- `openclaw-ai-quant-paper.service` - Paper trading daemon
- `openclaw-ai-quant-live.service` - Live trading daemon
- `openclaw-ai-quant-ws-sidecar.service` - WebSocket sidecar for market data
- `openclaw-ai-quant-monitor.service` - Real-time monitoring dashboard

Copy the relevant templates to `/etc/systemd/system/`, customize paths and environment variables, then enable and start the services.

## Project Structure

```
.
├── engine/                 # Unified trading engine (Python)
│   ├── core.py             # Main trading loop (UnifiedEngine)
│   ├── daemon.py           # Entrypoint daemon
│   ├── market_data.py      # Candle/mid data hub
│   ├── strategy_manager.py # YAML hot-reload
│   ├── oms.py              # Order Management System
│   ├── rest_client.py      # Hyperliquid REST client
│   └── ...
├── strategy/               # Strategy implementations
│   └── mei_alpha_v1.py     # Signals, confidence, PaperTrader
├── exchange/               # Exchange adapters
│   ├── ws.py               # WebSocket client
│   ├── sidecar.py          # WS sidecar client
│   ├── meta.py             # Hyperliquid metadata
│   ├── executor.py         # HyperliquidLiveExecutor
│   └── market_watch.py     # Market watch
├── live/                   # Live trading
│   └── trader.py           # LiveTrader
├── config/                 # Runtime configuration
│   └── strategy_overrides.yaml
├── tools/                  # Operational tools
│   ├── deploy_sweep.py     # Deploy sweep results to YAML
│   ├── export_state.py     # Export paper/live state to JSON
│   └── ...
├── backtester/             # Rust backtester (Cargo workspace)
│   ├── crates/bt-core/     # Simulation engine + indicators
│   ├── crates/bt-data/     # SQLite candle loader
│   ├── crates/bt-cli/      # CLI (replay, sweep, dump-indicators)
│   ├── crates/bt-gpu/      # CUDA GPU sweep + TPE
│   └── sweeps/             # Sweep configs + runner scripts
├── ws_sidecar/             # Rust WS sidecar
├── monitor/                # Real-time dashboard
├── scripts/                # Shell scripts (run_paper.sh, run_live.sh)
├── systemd/                # Service templates
└── docs/                   # Documentation
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and component interactions
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development setup and guidelines
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Contribution guidelines
- [docs/release_process.md](docs/release_process.md) - Version governance and tag-driven release flow
- [backtester/README.md](backtester/README.md) - Backtester documentation
- [monitor/README.md](monitor/README.md) - Dashboard documentation

## License

MIT
