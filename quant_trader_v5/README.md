quant_trader_v5 (drop-in refactor)

What you get

1) One daemon loop for paper and live
Run a single entrypoint and switch modes using AI_QUANT_MODE.

2) StrategyManager singleton
Edits to strategy_overrides.yaml are picked up automatically by mtime polling.
No importlib.reload(mei_alpha_v1) needed.

3) MarketDataHub
Reads mids and candles from Hyperliquid WS first.
If WS is stale or empty it falls back to SQLite candles table, then REST candleSnapshot.

4) UnifiedEngine improvements for large watchlists (v5)
Instead of rebuilding DataFrames for every symbol on every loop, the engine polls a cheap per-symbol candle key:
- close time of the last closed candle when AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=1 (default)
- open time of the latest candle when AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=0

Only when the key changes do we fetch a full candles DF and run mei_alpha_v1.analyze.
This makes 100+ symbols practical on a small box, especially on 1h or 15m intervals.

5) LiveOms (v6)
Adds a durable Order Management System (OMS) ledger for live trading:
- OrderIntent rows created at submit time (with restart-safe dedupe for OPEN intents per candle)
- Orders + Fills tables (fills deduped by fill_hash+fill_tid)
- Fill to intent matching via client_order_id when available, with a time-proximity fallback
- Trades meta_json is enriched with oms.intent_id so you can debug "why did this fill happen".

How to run

Paper:
  AI_QUANT_MODE=paper python -m quant_trader_v5.run_unified_daemon

Dry live:
  AI_QUANT_MODE=dry_live python -m quant_trader_v5.run_unified_daemon

Live:
  AI_QUANT_MODE=live python -m quant_trader_v5.run_unified_daemon

Recommended env for 100 symbols

- Keep AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=1.
- Disable BBO subscription if you do not need it (requires hyperliquid_ws patch):
    AI_QUANT_WS_ENABLE_BBO=0
- Consider disabling candle subscription and using DB/REST if you only trade on candle close (advanced):
    AI_QUANT_WS_ENABLE_CANDLE=0

Notes

- This package depends on your existing mei_alpha_v1.py and hyperliquid_ws.py.
- For best performance on large universes, patch hyperliquid_ws.py with the included WS toggles.
