# Python Runtime Compatibility

Production paper and live services now run through the Rust `aiq-runtime`.
This directory remains for compatibility helpers, monitoring integrations, and
test coverage. The old Python daemon path has been retired.

## Running

```bash
# Production paper lane
./scripts/run_paper_lane.sh paper1

# Production live lane
./scripts/run_live.sh

```

File lock prevents duplicate daemons: `ai_quant_paper.lock` / `ai_quant_live.lock` (configurable via `AI_QUANT_LOCK_PATH`).

## Modules

| Module | Purpose |
|--------|---------|
| `daemon.py` | Retired Python daemon shim that fails fast and points callers to Rust service wrappers |
| `core.py` | `UnifiedEngine` compatibility loop plus shared helper types kept for tests and parity tooling |
| `strategy_manager.py` | Hot-reloads the resolver-selected strategy YAML path for Python compatibility paths and helper consumers |
| `market_data.py` | `MarketDataHub` — candle + mid data from WS sidecar / SQLite / REST fallback |
| `risk.py` | Legacy Python `RiskManager` helpers retained for archival tooling and tests |
| `oms.py` | Legacy Python OMS helpers retained for archival tooling and tests |
| `oms_reconciler.py` | OMS state reconciliation against exchange positions/fills |
| `alerting.py` | Discord / Telegram notifications via `openclaw message send` |
| `event_logger.py` | Decision + trade event logging for audit trail |
| `promoted_config.py` | Compatibility shim that calls the shared Rust effective-config resolver for paper, dry-live, live, and factory control-plane materialisation |
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
polling for compatibility consumers, but the effective-config owner is now Rust:
`aiq-runtime paper effective-config` and `aiq-runtime live effective-config`
resolve the shared Rust-owned startup contract for paper and live-facing
runtime consumers, covering promoted-role discovery, strategy-mode selection,
config identity, and the materialised YAML path before legacy Python runtime
consumers continue.
The `engine.interval` parameter is NOT hot-reloadable (requires service
restart).

## Environment Variables

### Mode Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `AI_QUANT_MODE` | `paper` / `dry_live` / `live` | Legacy mode selector retained only for compatibility helpers |
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
| `AI_QUANT_STRATEGY_YAML` | Base strategy YAML input for the Rust resolver; paper, dry-live, and live start-up then switch to the resolver-selected materialised path |
| `AI_QUANT_PROMOTED_ROLE` | Promoted config selector (`primary` / `fallback` / `conservative`) |
| `AI_QUANT_STRATEGY_MODE` | Strategy-mode selector; env wins over `AI_QUANT_STRATEGY_MODE_FILE` |
| `AI_QUANT_STRATEGY_MODE_FILE` | File-backed strategy-mode fallback when the env var is unset |
| `AI_QUANT_RUNTIME_BIN` | Optional absolute path to the `aiq-runtime` binary for paper/live start-up and factory control-plane resolution |
| `AI_QUANT_WS_ENABLE_BBO` | Enable BBO subscription (`0` to disable) |

See `.env.example` for the full list.
