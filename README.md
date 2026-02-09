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
              │  (quant_trader_v5)      │
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

# Python setup
python3 -m venv venv
source venv/bin/activate
pip install -e .
# Or with uv:
uv sync

# Configure
cp .env.example .env
cp secrets.json.example secrets.json
# Edit .env and secrets.json with your values

# Run paper trader
AI_QUANT_MODE=paper python -m quant_trader_v5.run_unified_daemon
```

## Backtester

The Rust backtester provides high-performance simulation with CPU and GPU acceleration.

```bash
# Build (CPU)
cd backtester && cargo build --release

# Single replay
./target/release/mei-backtester replay --candles-db ../candles_dbs/candles_1h.db

# Parameter sweep
./target/release/mei-backtester sweep --sweep-config sweep_gpu_smoke.yaml

# GPU sweep (requires CUDA)
cargo build --release -p bt-cli --features gpu
./target/release/mei-backtester sweep --gpu --sweep-spec sweep_gpu_60k_allgpu.yaml

# TPE Bayesian optimization
./target/release/mei-backtester sweep --gpu --tpe --tpe-trials 5000 --sweep-spec sweep.yaml
```

See [backtester/README.md](backtester/README.md) for detailed documentation.

## Configuration

Strategy configuration is managed through `strategy_overrides.yaml`, which supports hot-reload at runtime. The unified daemon watches this file and applies changes without restart.

Key configuration sections:

- **trade**: Position sizing, leverage, stop-loss/take-profit multipliers, pyramiding rules
- **market_regime**: Volatility and trend classification thresholds
- **filters**: Market breadth, volume, correlation filters
- **indicators**: ADX, RSI, MACD, Bollinger Bands, and other technical indicator parameters

Both the backtester and live engine read from the same YAML file, ensuring consistent parameter values across simulation and production.

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
├── mei_alpha_v1.py          # Strategy: signals, confidence, PaperTrader
├── live_trader.py           # Live order execution via Hyperliquid SDK
├── execution_live.py        # HyperliquidLiveExecutor
├── quant_trader_v5/         # Unified engine (paper + live)
│   ├── engine.py            # Main trading loop
│   ├── market_data.py       # Candle/mid data hub
│   ├── strategy_manager.py  # YAML hot-reload
│   └── ...
├── backtester/              # Rust backtester (Cargo workspace)
│   ├── crates/bt-core/      # Simulation engine + indicators
│   ├── crates/bt-data/      # SQLite candle loader
│   ├── crates/bt-cli/       # CLI (replay, sweep, dump-indicators)
│   └── crates/bt-gpu/       # CUDA GPU sweep + TPE
├── ws_sidecar/              # Rust WS sidecar
├── monitor/                 # Real-time dashboard
├── strategy_overrides.yaml  # Strategy config (hot-reloads)
├── deploy_sweep.py          # Deploy sweep results to YAML
├── export_state.py          # Export paper/live state to JSON
├── systemd/                 # Service templates
└── .env.example             # Environment variable template
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and component interactions
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [backtester/README.md](backtester/README.md) - Backtester documentation
- [monitor/README.md](monitor/README.md) - Dashboard documentation

## License

MIT
