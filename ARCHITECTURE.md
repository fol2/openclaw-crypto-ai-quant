# Hyperliquid Trading Engine - System Architecture

## 1. The Trader (Paper)
- **Daemon:** `./quant_trader_v5/run_unified_daemon.py` (unified loop; paper mode via `AI_QUANT_MODE=paper`)
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
  - Logs to SQLite DB: `./trading_engine.db` (`trades`, `position_state`, `candles`).
  - Sends Discord alerts via `openclaw message send`.
  - Optional per-symbol overrides via `./strategy_overrides.yaml`.
  - Watchlist is `StrategyManager.get_watchlist()` (default: top 50 by 24h notional volume; configurable via `AI_QUANT_TOP_N` or `AI_QUANT_SYMBOLS`).
- **Single-instance:** File lock `./ai_quant_paper.lock` prevents duplicate paper daemons (override via `AI_QUANT_LOCK_PATH`).

## 2. The Trader (Live)
- **Daemon:** `./quant_trader_v5/run_unified_daemon.py` (unified loop; live via `AI_QUANT_MODE=live` or `dry_live`)
- **Service:** `openclaw-ai-quant-live.service` (systemd user service)
- **Mode:** set `AI_QUANT_MODE=live` (or `dry_live` to never send orders)
- **Role:**
  - Shares the same strategy logic (`mei_alpha_v1.analyze(...)` + `PaperTrader.check_exit_conditions(...)`) as paper trading.
  - Uses Hyperliquid WS streams for market data (`allMids`, `bbo`, `candle`) and user streams (`userFills`, `orderUpdates`, `userFundings`, `userNonFundingLedgerUpdates`).
  - Places real perps orders via the Hyperliquid SDK (`Exchange.market_open/market_close`).
  - Logs actual fills (price + fee + realized PnL) into SQLite (`AI_QUANT_DB_PATH`, recommended: `trading_engine_live.db`).
  - Reconciles positions/equity from `Info.user_state(...)` periodically (belt-and-suspenders).
- **Single-instance:** File lock `./ai_quant_live.lock` prevents duplicate live daemons (override via `AI_QUANT_LOCK_PATH`).
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
- The websocket `candle` stream only pushes the current candle, so the engine uses a local candle cache in SQLite.
- The WS sidecar handles candle bootstrapping automatically when initializing a fresh database.

## 3. Analyst Mode
- **Role:** Cron-based analyst that runs hourly, reads database data, generates performance reports, and delivers them to Discord.
- **Workflow:**
  1. **Audit:** Review `./trading_engine.db` (`trades`, `audit_events`, `runtime_logs`).
  2. **Analyze:** Identify patterns in trade performance (e.g., "Stop loss was too tight for this volatility").
  3. **Report:** Generate summary statistics, win rate, drawdown analysis, and send to Discord.
  4. **Read-only:** The analyst cannot modify configuration files or strategy parameters. It only reads and reports.
- **Constraints:** No config changes allowed. Strategy refinement requires manual code or YAML edits.

## 4. Manual Overrides
- **Role:** The operator can manually intervene to adjust trading behavior.
- **Powers:**
  - Direct the engine to change assets or watchlist composition.
  - Force a trade closure or entry via override commands.
  - Adjust strategy parameters via `./strategy_overrides.yaml` (hot-reloads automatically).
  - Critique the logic and demand refinements to detection patterns or risk management.

## 5. WS Sidecar
- **Role:** High-performance Rust binary that streams Hyperliquid WebSocket data and maintains candle databases.
- **Features:**
  - Streams real-time market data from Hyperliquid.
  - Maintains candle DBs for 6 intervals (1m, 5m, 15m, 1h, 4h, 1d).
  - Communicates with the trading engine via Unix socket.
  - Bootstraps historical candles automatically on fresh databases.
- **Location:** `ws_sidecar/`

## 6. Monitor Dashboard
- **Role:** Read-only web dashboard for real-time monitoring of trading activity.
- **Features:**
  - Reads SQLite databases (`trading_engine.db`, `trading_engine_live.db`).
  - Displays live mids from the WS sidecar.
  - Shows position state, recent trades, performance metrics.
  - No write access to databases or configuration.
- **Location:** `monitor/` (see `monitor/README.md` for details).

## 7. Backtester
- **Role:** High-performance historical simulation engine for strategy validation.
- **Features:**
  - CPU mode for fast local testing.
  - CUDA GPU mode for accelerated multi-strategy sweep.
  - Simulates order fills, fees, slippage, and funding.
  - Outputs performance metrics (Sharpe, Sortino, max drawdown, win rate).
- **Location:** `backtester/` (see `backtester/README.md` for details).

## 8. Signal Ranking
- **Mechanism:** Two-phase collect-rank-execute system.
- **Scoring formula:**
  - `score = confidence_rank * 100 + ADX`
  - Higher score = stronger signal priority.
  - Tiebreak: alphabetical by symbol name.
- **Purpose:** Prevents simultaneous entries across highly correlated assets. Only the top-ranked signal executes per cycle.

## 9. Market Breadth & Auto-Reverse
- **Market Breadth:**
  - Calculated as the percentage of watchlist symbols with `EMA_fast > EMA_slow`.
  - Acts as a regime filter to block trades against the prevailing market tide.
  - Example: In a bearish market (breadth < 30%), long entries may be suppressed.
- **Auto-Reverse:**
  - Fades noise in choppy, range-bound markets.
  - When conditions meet specific criteria (e.g., low volatility, tight range), the engine may reverse position direction to fade false breakouts.
  - Designed to reduce whipsaw losses during sideways price action.
