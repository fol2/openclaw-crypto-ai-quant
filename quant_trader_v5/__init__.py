"""quant_trader_v5

A hardened, unified daemon architecture that fixes the main pain points in the current app:

- One strategy/config singleton (YAML hot-reload, no more reloading mei_alpha_v1 every loop)
- One market-data hub (Hyperliquid WS with REST fallback, plus DB backfill)
- One engine loop for paper + live (no duplicate daemons)

v5 focus: scale and stability for large symbol universes
- Candle-key polling so we do not build candle DataFrames for every symbol on every loop.
- Entry and exit fast-path to reduce work when most symbols are idle.
- Helpers to fetch the "last closed candle key" cheaply (WS first, DB fallback).

This package is designed to be dropped next to your existing scripts.
"""

from .strategy_manager import StrategyManager
from .market_data import MarketDataHub, PriceQuote
from .engine import UnifiedEngine
from .oms import LiveOms
from .risk import RiskManager
from .oms_reconciler import LiveOmsReconciler

__all__ = [
    "StrategyManager",
    "MarketDataHub",
    "PriceQuote",
    "UnifiedEngine",
    "LiveOms",
    "RiskManager",
    "LiveOmsReconciler",
]
