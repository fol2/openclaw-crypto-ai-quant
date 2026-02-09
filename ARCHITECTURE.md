# Hyperliquid Trading Engine - System Architecture

## 1. The Trader (Paper)
- **Daemon:** `dev/ai_quant/quant_trader_v5/run_unified_daemon.py` (unified loop; paper mode via `AI_QUANT_MODE=paper`)
- **Hot reload:** Strategy YAML (`strategy_overrides.yaml`) hot-reloads by mtime via `StrategyManager` (no module reload loop).
- **Service:** `openclaw-ai-quant-trader.service` (systemd user service)
- **Role:**
  - Streams Hyperliquid market data via websocket:
    - `allMids` for live mids (exit/entry pricing)
    - `bbo` for best bid/ask (more realistic fill prices)
    - `candle` for 1h candles (strategy indicators)
  - Generates signals via `mei_alpha_v1.analyze(...)`.
  - Manages paper positions with ATR-based SL/TP + trailing stops.
  - Keeps **1 net position per symbol** (perps-style), with multiple fills/tranches:
    - `ADD` (scale-in) and `REDUCE` (partial close), plus `CLOSE` for final exit.
  - Simulates perps trading fees (taker by default) for realistic net PnL.
  - Simulates perps hourly funding (optional, enabled by default) for more realistic PnL when holding positions.
  - Logs to SQLite DB: `dev/ai_quant/trading_engine.db` (`trades`, `position_state`, `candles`).
  - Sends Discord alerts via `openclaw message send`.
  - Optional per-symbol overrides via `dev/ai_quant/strategy_overrides.yaml`.
  - Watchlist is `StrategyManager.get_watchlist()` (default: top 50 by 24h notional volume; configurable via `AI_QUANT_TOP_N` or `AI_QUANT_SYMBOLS`).
- **Single-instance:** File lock `dev/ai_quant/ai_quant_paper.lock` prevents duplicate paper daemons (override via `AI_QUANT_LOCK_PATH`).

## 2. The Trader (Live)
- **Daemon:** `dev/ai_quant/quant_trader_v5/run_unified_daemon.py` (unified loop; live via `AI_QUANT_MODE=live` or `dry_live`)
- **Service:** `openclaw-ai-quant-live.service` (systemd user service)
- **Mode:** set `AI_QUANT_MODE=live` (or `dry_live` to never send orders)
- **Role:**
  - Shares the same strategy logic (`mei_alpha_v1.analyze(...)` + `PaperTrader.check_exit_conditions(...)`) as paper trading.
  - Uses Hyperliquid WS streams for market data (`allMids`, `bbo`, `candle`) and user streams (`userFills`, `orderUpdates`, `userFundings`, `userNonFundingLedgerUpdates`).
  - Places *real* perps orders via the Hyperliquid SDK (`Exchange.market_open/market_close`).
  - Logs **actual fills** (price + fee + realized PnL) into SQLite (`AI_QUANT_DB_PATH`, recommended: `trading_engine_live.db`).
  - Reconciles positions/equity from `Info.user_state(...)` periodically (belt-and-suspenders).
- **Single-instance:** File lock `dev/ai_quant/ai_quant_live.lock` prevents duplicate live daemons (override via `AI_QUANT_LOCK_PATH`).
- **Safety gates:**
  - Live orders require explicit enable flags:
    - `AI_QUANT_LIVE_ENABLE=1`
    - `AI_QUANT_LIVE_CONFIRM=I_UNDERSTAND_THIS_CAN_LOSE_MONEY`
  - Close-only switch (disables new entries/adds; exits still allowed):
    - `AI_QUANT_KILL_SWITCH=1`
  - Hard stop (disables ALL orders, including exits):
    - `AI_QUANT_HARD_KILL_SWITCH=1`
  - Live-only sizing caps in YAML (see `strategy_overrides.yaml` → `live:`).

### Candle History
- The websocket `candle` stream only pushes the *current* candle, so the engine uses a local candle cache in SQLite.
- Utility scripts (including candle bootstrap) are archived to reduce clutter in the main folder.
- Seed once on a fresh DB with: `dev/ai_quant/venv/bin/python3 dev/ai_quant/archive/housekeeping_*/scripts/bootstrap_candles.py`

## 3. The Analyst (Review & Refinement)
- **Role:** Mei (me) during my "Self-Improvement" or manual review sessions.
- **Workflow:**
  1. **Audit:** Review `dev/ai_quant/trading_engine.db` (`trades`, `audit_events`, `runtime_logs`).
  2. **Analyze:** Identify why trades failed (e.g., "Stop loss was too tight for this volatility").
  3. **Refine:** Rewrite the logic in `mei_alpha_v1.py` to improve the pattern detection.
  4. **Deploy:** YAML edits apply automatically (mtime hot reload). Python code edits require restarting the service.

## 4. The Human (Commander)
- **Role:** James.
- **Powers:** 
  - Direct me to change assets.
  - Force a trade closure or entry.
  - Critique the logic and demand "sophistication" (Item 1, 2, 3).
