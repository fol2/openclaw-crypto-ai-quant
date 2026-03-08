# Unified Trading Engine

The core Python daemon now owns `dry_live` and `live`, while `paper` mode is a
legacy recovery-only path after the Rust paper cutover.

## Running

```bash
# Paper trading legacy recovery/debug only
AI_QUANT_MODE=paper python -m engine.daemon

# Dry live (real data, no real orders)
AI_QUANT_MODE=dry_live python -m engine.daemon

# Live trading (requires safety gates)
AI_QUANT_MODE=live python -m engine.daemon
```

File lock prevents duplicate daemons: `ai_quant_paper.lock` / `ai_quant_live.lock` (configurable via `AI_QUANT_LOCK_PATH`).

## Modules

| Module | Purpose |
|--------|---------|
| `daemon.py` | Entrypoint for live/dry_live plus the legacy paper recovery path |
| `core.py` | `UnifiedEngine` — main trading loop, two-phase collect-rank-execute |
| `strategy_manager.py` | Hot-reloads the resolver-selected strategy YAML path via mtime polling for the active Python daemon compatibility path |
| `market_data.py` | `MarketDataHub` — candle + mid data from WS sidecar / SQLite / REST fallback |
| `risk.py` | `RiskManager` — rate limits, drawdown kill-switch, exposure caps, slippage guard |
| `oms.py` | `LiveOms` — durable intent/order/fill ledger for live trading |
| `oms_reconciler.py` | OMS state reconciliation against exchange positions/fills |
| `alerting.py` | Discord / Telegram notifications via `openclaw message send` |
| `event_logger.py` | Decision + trade event logging for audit trail |
| `promoted_config.py` | Compatibility shim that calls the shared Rust `paper effective-config` resolver for paper start-up and factory materialisation |
| `sqlite_logger.py` | Trade, candle, and position state persistence |
| `rest_client.py` | Hyperliquid REST API client |
| `systemd_watchdog.py` | `sd_notify` integration for systemd services |
| `kernel_shadow_report.py` | Shadow mode decision report generation |
| `openclaw_cli.py` | CLI wrapper for `openclaw` commands |
| `utils.py` | Shared utilities |

## Design

### Candle Key Polling

Instead of rebuilding DataFrames on every loop iteration, the engine polls a cheap per-symbol candle key:

- Close time of the last closed candle when `AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=1` (default)
- Open time of the latest candle when `AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=0`

Only when the key changes does the engine fetch a full candles DataFrame and run `mei_alpha_v1.analyze()`. This makes 100+ symbol watchlists practical on a small box.

### Order Management System (LiveOms)

Durable OMS for live trading:

- `OrderIntent` rows created at submit time (restart-safe dedupe for OPEN intents per candle)
- Orders + Fills tables (fills deduped by `fill_hash` + `fill_tid`)
- Fill-to-intent matching via `client_order_id`, with time-proximity fallback
- Trades `meta_json` enriched with `oms.intent_id` for debugging

### Configuration Hot-Reload

`StrategyManager` watches the resolver-selected strategy YAML path via mtime
polling. In the active Python compatibility path this remains the hot-reload
surface, but the effective-config owner is now Rust:
`aiq-runtime paper effective-config` resolves promoted-role discovery,
strategy-mode selection, config identity, and the materialised YAML path before
legacy paper recovery or factory materialisation continue.
The `engine.interval` parameter is NOT hot-reloadable (requires service
restart).

## Environment Variables

### Mode Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `AI_QUANT_MODE` | `paper` / `dry_live` / `live` | Trading mode |
| `AI_QUANT_SIGNAL_ON_CANDLE_CLOSE` | `1` (default) / `0` | Signal timing |

### Live Safety Gates

| Variable | Description |
|----------|-------------|
| `AI_QUANT_LIVE_ENABLE` | Must be `1` for live mode |
| `AI_QUANT_LIVE_CONFIRM` | Must be `I_UNDERSTAND_THIS_CAN_LOSE_MONEY` |
| `AI_QUANT_KILL_SWITCH` | `close_only` or `halt_all` |
| `AI_QUANT_KILL_SWITCH_FILE` | File-based kill-switch (polled, no restart needed) |
| `AI_QUANT_HARD_KILL_SWITCH` | Blocks ALL orders including exits |

### Optional

| Variable | Description |
|----------|-------------|
| `AI_QUANT_LOCK_PATH` | Custom lock file path |
| `AI_QUANT_STRATEGY_YAML` | Base strategy YAML input for the Rust resolver; paper start-up then switches to the resolver-selected materialised path |
| `AI_QUANT_PROMOTED_ROLE` | Promoted config selector (`primary` / `fallback` / `conservative`) |
| `AI_QUANT_STRATEGY_MODE` | Strategy-mode selector; env wins over `AI_QUANT_STRATEGY_MODE_FILE` |
| `AI_QUANT_STRATEGY_MODE_FILE` | File-backed strategy-mode fallback when the env var is unset |
| `AI_QUANT_RUNTIME_BIN` | Optional absolute path to the `aiq-runtime` binary for paper start-up / factory control-plane resolution |
| `AI_QUANT_WS_ENABLE_BBO` | Enable BBO subscription (`0` to disable) |

See `.env.example` for the full list.
