# AI Agent Instructions for openclaw-crypto-ai-quant

This guide provides comprehensive instructions for AI coding assistants (Claude Code, Copilot, Cursor, etc.) working on this codebase.

## Project Overview

This is a crypto perpetual futures trading engine for Hyperliquid DEX. The system features:

- Python-based strategy logic and execution engine
- Rust-based backtester with optional GPU acceleration
- Unified daemon supporting both paper and live trading modes
- Real-time market data via WebSocket sidecar
- Hot-reloadable YAML configuration system

## Architecture

### Core Components

- **Strategy configuration**: `strategy_overrides.yaml` (hot-reloads via mtime polling)
- **Strategy defaults**: `mei_alpha_v1.py` → `_DEFAULT_STRATEGY_CONFIG` dict
- **Unified engine**: `quant_trader_v5/engine.py` → `UnifiedEngine.run_forever()`
- **Paper trader**: `mei_alpha_v1.py` → `PaperTrader` class
- **Live trader**: `live_trader.py` → `LiveTrader` class

### Databases

- **Paper DB**: `trading_engine.db` (SQLite)
- **Live DB**: `trading_engine_live.db` (SQLite)
- **Candle DBs**: `candles_dbs/candles_{1m,3m,5m,15m,30m,1h}.db`

### Candle Database Coverage

Different candle databases cover different time spans. When running backtests across multiple intervals, ALWAYS restrict all runs to the same date range for valid comparisons:

- `1m`: 3.5 days
- `3m`: 10 days
- `5m`: 17 days
- `15m`: 52 days
- `30m`: 104 days
- `1h`: 208 days

## Configuration System

### Merge Order

Configuration values are merged in this order (later values override earlier):

```
_DEFAULT_STRATEGY_CONFIG ← global YAML ← symbols.<SYM> YAML ← live YAML (if live mode)
```

### Key Configuration Sections

#### `trade`
- `sl_atr_mult`, `tp_atr_mult`: Stop-loss and take-profit multipliers
- `leverage`: Base leverage
- `allocation_pct`: Margin allocated per position (notional = margin × leverage)
- `min_atr_pct`: Minimum ATR percentage threshold (entries below this are skipped)
- `reentry_cooldown_minutes`: Cooldown period before re-entering same symbol
- `enable_pyramiding`: Enable/disable pyramid ADD orders
- `enable_partial_tp`: Enable/disable partial take-profit
- `enable_dynamic_leverage`: Dynamically scale leverage by confidence
- `enable_dynamic_sizing`: Dynamically scale position size by ADX/volatility

#### `market_regime`
- `enable_regime_filter`: Enable market regime filter
- `enable_auto_reverse`: Auto-reverse signals based on market breadth
- `bearish_breadth_threshold_pct`: Threshold for bearish regime detection
- `bullish_breadth_threshold_pct`: Threshold for bullish regime detection

#### `filters`
Entry gate filters:
- `enable_ranging_regime_filter`: Block entries in ranging markets
- `enable_anomaly_filter`: Block entries with anomaly detection
- `enable_btc_alignment_filter`: Require BTC direction alignment
- `enable_market_breadth_filter`: Filter by market-wide breadth

#### `thresholds.entry`
- `min_adx`: Minimum ADX required for entry
- `pullback_pct`: Required pullback percentage
- `slow_drift`: Slow drift threshold

#### `indicators`
Window sizes for technical indicators:
- `ema_fast`, `ema_slow`: EMA periods
- `adx_window`: ADX period
- `bb_window`: Bollinger Bands period
- `bb_std`: Bollinger Bands standard deviation

#### `engine`
- `interval`: Candle interval (e.g., `1h`, `3m`, `5m`)
- `entry_interval`: Interval for entry signal generation
- `exit_interval`: Interval for exit signal generation

**IMPORTANT**: The `engine.interval` parameter is NOT hot-reloadable. Changing it requires a service restart.

### Configuration Defaults

The code-level default for `entry_min_confidence` is `"high"`, which blocks 70-90% of entry signals. To allow all confidence levels (low/medium/high), explicitly set `entry_min_confidence: low` in your YAML configuration.

## Signal Flow

### Signal Generation and Processing

1. **Strategy analysis**: `mei_alpha_v1.analyze(df, sym, btc_bullish)` → `(signal, confidence, now_series)`
2. **Signal reversal** (optional): Manual or auto-reverse based on market breadth
3. **Regime filter**: Block trades against market tide
4. **ATR floor enforcement**: Skip if `ATR% < min_atr_pct`
5. **Phase 1**: Collect entry candidates (exits run immediately per-symbol, not ranked)
6. **Phase 2**: Rank entries by score, execute in order

### Signal Ranking Algorithm

Entry signals are ranked using this formula:

```
score = confidence_rank * 100 + ADX
```

Where:
- High confidence: `confidence_rank = 2` → `score = 200 + ADX`
- Medium confidence: `confidence_rank = 1` → `score = 100 + ADX`
- Low confidence: `confidence_rank = 0` → `score = 0 + ADX`

Tiebreaker: Symbol name alphabetical order (ascending)

**Important**: Exits are executed immediately per-symbol and are NOT ranked. Pyramid ADD orders are also executed immediately when a same-direction position exists, NOT ranked.

### Audit Trail

The `now_series` dictionary carries audit flags for debugging:

- `_reversed_entry`: Signal was reversed (manual or auto)
- `_regime_blocked`: Entry blocked by regime filter
- `_atr_floored`: Entry skipped due to ATR floor
- `_market_breadth_pct`: Current market breadth percentage

## Rust Backtester

### Location and Structure

- **Location**: `backtester/` directory
- **Cargo workspace**: `bt-core`, `bt-data`, `bt-cli`, `bt-gpu`

### Building

```bash
# CPU-only build
cargo build --release -p bt-cli

# GPU-accelerated build (requires CUDA toolkit)
cargo build --release -p bt-cli --features gpu
```

On WSL2, you may need to set:
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Commands

- `replay`: Single-run backtest
- `sweep`: Parameter sweep
- `dump-indicators`: Export indicators for validation

### Indicator Parity Requirements

All 17 indicators in the Rust backtester MUST match the Python `ta` library within **0.00005 absolute error**.

#### Critical Implementation Details

1. **EMA seeding**: Use `adjust=False` (pandas `ewm` method), NOT SMA seed. This matches `ta.trend.ema_indicator`.

2. **`vol_trend`**: Calculate as `SMA(vol, 5) > SMA(vol, 20)`, NOT `current_vol > SMA(vol, 5)`.

3. **`bb_width`**: Calculate as `(bb_high - bb_low) / Close`, NOT `(bb_high - bb_low) / bb_middle`.

### Validation Workflow

When modifying indicator logic in Rust:

1. Run `dump-indicators` to export Rust output
2. Compare against Python `ta` library output
3. Ensure all indicators match within 0.00005 absolute error
4. Document any intentional deviations

## Services (systemd)

### Service Management

- **Paper trader**: `systemctl --user restart openclaw-ai-quant-trader`
- **Live trader**: `systemctl --user restart openclaw-ai-quant-live`
- **WebSocket sidecar**: `systemctl --user restart openclaw-ai-quant-ws-sidecar`
- **Monitor**: `systemctl --user restart openclaw-ai-quant-monitor`

### Hot-reload Behavior

- **YAML configuration changes**: Hot-reload automatically (no restart required)
- **Python code changes**: Require service restart
- **Engine interval changes**: Require service restart (not hot-reloadable)

## Development

### Python Environment

- **Python version**: >=3.12
- **Package manager**: `uv`

### Setup

```bash
uv sync --dev
```

### Testing

```bash
uv run pytest
```

**Coverage requirement**: 100% coverage for `sqlite_logger` and `heartbeat` modules.

### Linting and Formatting

```bash
# Lint
uv run ruff check quant_trader_v5 tests

# Format
uv run ruff format quant_trader_v5 tests
```

### Rust Development

```bash
# Run tests
cargo test

# Format
cargo fmt

# Lint
cargo clippy
```

## Critical Rules for AI Assistants

### Security

1. **NEVER commit sensitive files**: `secrets.json`, `.env`, or any file containing private keys, wallet addresses, or API keys.

2. **Live trading safety**: Never disable kill switches (`AI_QUANT_KILL_SWITCH`, `AI_QUANT_HARD_KILL_SWITCH`) without explicit user confirmation.

### Configuration Management

3. **NEVER auto-tune strategy config via automated scripts**. Always use suggestion-only mode. This prevents accidental deployment of untested parameters. (Lesson from emergency rollback incident.)

4. **Code defaults vs YAML overrides**: The code-level default for `entry_min_confidence` is `"high"`, which blocks most entries. Remember to set `entry_min_confidence: low` in YAML to allow all confidence levels.

### Backtesting

5. **Date range consistency**: When comparing backtest results across different intervals, ALWAYS restrict all runs to the same date range. Different candle databases have different coverage spans.

6. **Indicator parity**: Backtester indicator calculations MUST match Python `ta` library within 0.00005 absolute error. Validate using `dump-indicators` after any indicator changes.

### Debugging

7. **Audit flags**: The `now_series` dictionary carries debugging flags (`_reversed_entry`, `_regime_blocked`, `_atr_floored`, `_market_breadth_pct`). Check these when investigating unexpected behavior.

8. **When validating Rust indicators**: Always use `dump-indicators` and compare against Python `ta` output. Document the validation process.

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `mei_alpha_v1.py` | Strategy signals + PaperTrader |
| `live_trader.py` | Live order execution wrapper |
| `execution_live.py` | HyperliquidLiveExecutor (SDK interface) |
| `quant_trader_v5/engine.py` | Main trading loop |
| `quant_trader_v5/strategy_manager.py` | YAML hot-reload |
| `quant_trader_v5/market_data.py` | Candle/mid data hub |
| `quant_trader_v5/oms.py` | Order Management System |
| `strategy_overrides.yaml` | Live strategy config |
| `backtester/crates/bt-core/src/engine.rs` | Backtester simulation |
| `backtester/crates/bt-gpu/src/gpu_host.rs` | CUDA GPU dispatch |
| `deploy_sweep.py` | Deploy sweep results to YAML |
| `export_state.py` | Export trader state to JSON |

## Additional Resources

- **Strategy version history**: `strategy_changelog.json`
- **Project management**: `pyproject.toml`
- **Dependencies**: Managed via `uv` (see `pyproject.toml` for list)

## Common Patterns

### Reading Strategy Config

```python
# In Python code:
from quant_trader_v5.strategy_manager import StrategyManager

strategy = StrategyManager(yaml_path="strategy_overrides.yaml")
config = strategy.get_config(symbol="BTC")
```

### Accessing Market Data

```python
# In Python code:
from quant_trader_v5.market_data import MarketDataHub

market = MarketDataHub()
df = market.get_candles(symbol="BTC", interval="1h", lookback_bars=200)
```

### Running Backtests

```bash
# Single run
./target/release/bt-cli replay --config strategy_overrides.yaml --start 2024-01-01 --end 2024-01-10

# Parameter sweep
./target/release/bt-cli sweep --config sweep_config.yaml --start 2024-01-01 --end 2024-01-10
```

## Troubleshooting

### Common Issues

1. **Service won't start**: Check systemd logs with `journalctl --user -u openclaw-ai-quant-trader -f`

2. **YAML changes not taking effect**: Verify mtime polling is working. Check engine logs for "Config reloaded" messages.

3. **Indicator mismatch in backtester**: Run `dump-indicators` and compare against Python `ta` output. Check EMA seeding, `vol_trend`, and `bb_width` implementations.

4. **GPU build fails**: Ensure CUDA toolkit is installed. On WSL2, set `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.

5. **Unexpected entry filtering**: Check `entry_min_confidence` setting. Code default is `"high"`, which blocks most entries. Set to `"low"` in YAML to allow all confidence levels.

## Best Practices

1. **Always read the code before suggesting changes**. Understand the existing implementation.

2. **Test configuration changes in paper mode first** before deploying to live trading.

3. **Validate backtester changes** using `dump-indicators` and comparison against Python `ta` library.

4. **Document significant changes** in commit messages and consider updating `strategy_changelog.json`.

5. **Preserve test coverage**. All changes to `sqlite_logger` and `heartbeat` must maintain 100% coverage.

6. **Use YAML for strategy tuning**, not Python code edits. YAML changes hot-reload; code changes require restart.

7. **When in doubt, ask the user** before making risky changes (especially for live trading or automated parameter tuning).
