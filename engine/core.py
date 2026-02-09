from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .market_data import MarketDataHub
from .strategy_manager import StrategyManager
from .utils import now_ms


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _interval_to_ms(interval: str) -> int:
    s = str(interval or "").strip().lower()
    if not s:
        return 0
    try:
        if s.endswith("m"):
            return int(float(s[:-1]) * 60.0 * 1000.0)
        if s.endswith("h"):
            return int(float(s[:-1]) * 60.0 * 60.0 * 1000.0)
        if s.endswith("d"):
            return int(float(s[:-1]) * 24.0 * 60.0 * 60.0 * 1000.0)
        # Fallback: assume seconds.
        return int(float(s) * 1000.0)
    except Exception:
        return 0


class ModePlugin(Protocol):
    def before_iteration(self) -> None: ...

    def after_iteration(self) -> None: ...


@dataclass
class EngineStats:
    loops: int = 0
    loop_errors: int = 0
    ws_restarts: int = 0
    last_heartbeat_s: float = 0.0
    last_ws_restart_s: float = 0.0


class UnifiedEngine:
    """One orchestration loop for both paper and live.

    v5 changes focus on scaling and stability:
    - Do not build candle DataFrames for every symbol on every loop.
      Instead poll a cheap per-symbol "candle key" (close time or open time),
      and only pull a full DF when the key changes.
    - Avoid fetching mid/funding for symbols that have nothing to do this loop.
      Exits always run for open positions. Entries only run for real entry candidates.

    The engine delegates:
    - Strategy math to mei_alpha_v1.analyze
    - Execution and position accounting to PaperTrader / LiveTrader
    """

    def __init__(
        self,
        *,
        trader: Any,
        strategy: StrategyManager,
        market: MarketDataHub,
        interval: str,
        lookback_bars: int,
        mode_plugin: ModePlugin | None = None,
    ):
        self.trader = trader
        self.strategy = strategy
        self.market = market
        self.interval = str(interval)
        self.lookback_bars = int(lookback_bars)
        self.mode_plugin = mode_plugin

        self.stats = EngineStats()

        # Entry de-dup. Key is derived from the candle we are acting on.
        self._last_entry_key: dict[str, int] = {}
        self._last_entry_key_open_pos_count: dict[str, int] = {}

        # Cached strategy outputs to avoid re-running ta.*.
        # Stored per symbol:
        # - key: candle key we last analyzed (close ms if signal_on_close else open ms)
        # - sig/conf/now: outputs of mei_alpha_v1.analyze
        self._analysis_cache: dict[str, dict[str, Any]] = {}

        # BTC anchor context cache.
        self._btc_ctx: dict[str, Any] = {"key": None, "btc_bullish": None}

        # Market Breadth: % of watchlist with bullish EMA alignment (computed from cached analysis).
        self._market_breadth_pct: float | None = None

        # WS staleness control
        self._stale_strikes: int = 0
        self._ws_restart_window_started_s: float = 0.0
        self._ws_restart_count_in_window: int = 0

        # ── Config: YAML `engine:` section + env var fallback ──

        # --- Interval: YAML engine.interval → env AI_QUANT_INTERVAL → constructor arg ---
        # NOT hot-reloadable (WS subscriptions, candle readiness tied to it).
        _init_engine_cfg = (self.strategy.get_config("__GLOBAL__") or {}).get("engine") or {}
        _yaml_interval = _init_engine_cfg.get("interval")
        if _yaml_interval is not None:
            self.interval = str(_yaml_interval).strip().lower()
        # else: keep constructor value (from env var or default)

        # --- Helper: read engine config with YAML first, env fallback ---
        def _ecfg(cfg: dict) -> dict:
            """Get fresh engine: section from a config dict."""
            return cfg.get("engine") or {}

        def _cfg_float(ecfg: dict, yaml_key: str, env_key: str, default: float) -> float:
            v = ecfg.get(yaml_key)
            if v is not None:
                return float(v)
            e = os.getenv(f"AI_QUANT_{env_key}", "")
            return float(e) if e.strip() else default

        def _cfg_int(ecfg: dict, yaml_key: str, env_key: str, default: int) -> int:
            return int(_cfg_float(ecfg, yaml_key, env_key, float(default)))

        def _cfg_str(ecfg: dict, yaml_key: str, env_key: str, default: str) -> str:
            v = ecfg.get(yaml_key)
            if v is not None:
                return str(v).strip().lower()
            return str(os.getenv(f"AI_QUANT_{env_key}", default) or default).strip().lower()

        def _cfg_bool(ecfg: dict, yaml_key: str, env_key: str, default: bool) -> bool:
            v = ecfg.get(yaml_key)
            if v is not None:
                return bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes")
            return _env_bool(f"AI_QUANT_{env_key}", default)

        self._cfg_float = _cfg_float
        self._cfg_int = _cfg_int
        self._cfg_str = _cfg_str
        self._cfg_bool = _cfg_bool
        self._ecfg = _ecfg

        # --- Init-time engine config (read once) ---
        self._heartbeat_every_s = _cfg_float(_init_engine_cfg, "heartbeat_every_s", "HEARTBEAT_SECS", 30.0)
        self._loop_target_s = _cfg_float(_init_engine_cfg, "loop_target_s", "LOOP_TARGET_SECS", 5.0)

        self._ws_stale_mids_s = float(os.getenv("AI_QUANT_WS_STALE_MIDS_S", "60"))
        self._ws_stale_candle_s = float(os.getenv("AI_QUANT_WS_STALE_CANDLES_S", str(2 * 60 * 60)))
        self._ws_stale_bbo_s = float(os.getenv("AI_QUANT_WS_STALE_BBO_S", "120"))

        self._ws_stale_strikes_to_restart = int(os.getenv("AI_QUANT_WS_STALE_STRIKES", "2"))
        self._ws_restart_cooldown_s = float(os.getenv("AI_QUANT_WS_RESTART_COOLDOWN_S", "60"))
        self._ws_restart_window_s = float(os.getenv("AI_QUANT_WS_RESTART_WINDOW_S", "600"))
        self._ws_restart_max_in_window = int(os.getenv("AI_QUANT_WS_RESTART_MAX_IN_WINDOW", "5"))

        # In "sidecar-only" market-data mode we must not fetch HL REST meta in the engine loop.
        self._rest_enabled = _env_bool("AI_QUANT_REST_ENABLE", True)

        # Performance and behavior knobs (init-time, not hot-reloadable)
        self._signal_on_candle_close = _cfg_bool(_init_engine_cfg, "signal_on_candle_close", "SIGNAL_ON_CANDLE_CLOSE", True)
        self._candle_close_grace_ms = _cfg_int(_init_engine_cfg, "candle_close_grace_ms", "CANDLE_CLOSE_GRACE_MS", 2000)

        # --- Hot-reloadable engine params: entry_interval, exit_interval ---
        # These are refreshed each loop via _refresh_engine_config().
        self._entry_interval: str = ""   # e.g. "3m" — determines reanalyze cadence
        self._exit_interval: str = ""    # e.g. "3m" — determines exit candle DB
        self._reanalyze_interval_s: float = 0.0
        self._last_analyze_ts: dict[str, int] = {}  # symbol → wall-clock bucket
        self._exit_reanalyze_interval_s: float = 0.0
        self._last_exit_ts: dict[str, int] = {}  # symbol → wall-clock bucket (exit)
        self._refresh_engine_config()  # populate from initial YAML

        # Optional: audit sampling for NEUTRAL signals (helps debug "why no entries" regimes).
        try:
            self._neutral_audit_sample_every_s = float(os.getenv("AI_QUANT_NEUTRAL_AUDIT_SAMPLE_EVERY_S", "0") or 0.0)
        except Exception:
            self._neutral_audit_sample_every_s = 0.0
        try:
            self._neutral_audit_sample_symbols = int(os.getenv("AI_QUANT_NEUTRAL_AUDIT_SAMPLE_SYMBOLS", "5") or 5)
        except Exception:
            self._neutral_audit_sample_symbols = 5
        self._neutral_audit_sample_every_s = float(max(0.0, self._neutral_audit_sample_every_s))
        self._neutral_audit_sample_symbols = int(max(0, min(250, self._neutral_audit_sample_symbols)))
        self._neutral_audit_last_s: float = 0.0
        self._neutral_audit_cursor: int = 0

        # Optional: print analyze() gates/tags periodically, even when signal is NEUTRAL.
        # Useful for "why didn't we trade this big move" debugging without writing to SQLite.
        raw_debug_syms = str(os.getenv("AI_QUANT_DEBUG_GATES_SYMBOLS", "") or "").strip()
        self._debug_gates_symbols: set[str] = set()
        if raw_debug_syms:
            self._debug_gates_symbols = {s.strip().upper() for s in raw_debug_syms.split(",") if s.strip()}
        try:
            self._debug_gates_every_s = float(os.getenv("AI_QUANT_DEBUG_GATES_EVERY_S", "0") or 0.0)
        except Exception:
            self._debug_gates_every_s = 0.0
        self._debug_gates_every_s = float(max(0.0, self._debug_gates_every_s))

        # Entry timing guard (close-mode only).
        # If the engine restarts (or stalls) it may act on an older closed candle key.
        # This prevents "late" entries when the candle-close signal is too old.
        try:
            max_delay_ms = int(float(os.getenv("AI_QUANT_ENTRY_MAX_DELAY_MS", "0") or 0.0))
        except Exception:
            max_delay_ms = 0
        if max_delay_ms <= 0:
            try:
                max_delay_ms = int(float(os.getenv("AI_QUANT_ENTRY_MAX_DELAY_S", "0") or 0.0) * 1000.0)
            except Exception:
                max_delay_ms = 0
        # Clamp to a sane range (0 disables).
        self._entry_max_delay_ms = int(max(0, min(max_delay_ms, 7 * 24 * 60 * 60 * 1000)))

        # Optional: if a symbol was blocked by max_open_positions, allow re-try within the same candle
        # only when open position count decreased.
        self._entry_retry_on_capacity = _env_bool("AI_QUANT_ENTRY_RETRY_ON_CAPACITY", False)

        self._interval_ms = _interval_to_ms(self.interval)

    def _ws_is_stale(self, *, symbols: list[str]) -> tuple[bool, str]:
        h = self.market.ws_health(symbols=symbols, interval=self.interval)

        mids_age = getattr(h, "mids_age_s", None)
        candle_age = getattr(h, "candle_age_s", None)
        bbo_age = getattr(h, "bbo_age_s", None)

        if mids_age is None:
            return True, "mids_age_s is None"
        try:
            if float(mids_age) > self._ws_stale_mids_s:
                return True, f"mids_age_s={float(mids_age):.1f}s"
        except Exception:
            return True, "mids_age_s not numeric"

        try:
            if candle_age is not None and float(candle_age) > self._ws_stale_candle_s:
                return True, f"candle_age_s={float(candle_age):.1f}s"
        except Exception:
            pass

        # BBO can be optional. Treat it as stale only when enabled (threshold > 0) and numeric.
        try:
            if self._ws_stale_bbo_s > 0 and bbo_age is not None and float(bbo_age) > self._ws_stale_bbo_s:
                return True, f"bbo_age_s={float(bbo_age):.1f}s"
        except Exception:
            pass

        return False, "ok"

    def _maybe_restart_ws(self, *, active_symbols: list[str], candle_limit: int, user: str | None) -> None:
        stale, reason = self._ws_is_stale(symbols=active_symbols)
        if stale:
            self._stale_strikes += 1
        else:
            self._stale_strikes = 0

        if self._stale_strikes < self._ws_stale_strikes_to_restart:
            return

        now_s = time.time()

        if self.stats.last_ws_restart_s and (now_s - self.stats.last_ws_restart_s) < self._ws_restart_cooldown_s:
            return

        if (not self._ws_restart_window_started_s) or ((now_s - self._ws_restart_window_started_s) > self._ws_restart_window_s):
            self._ws_restart_window_started_s = now_s
            self._ws_restart_count_in_window = 0

        self._ws_restart_count_in_window += 1
        self.stats.last_ws_restart_s = now_s

        print(
            f"🔄 WS restart. stale_strikes={self._stale_strikes} reason={reason}. "
            f"restart_count_in_window={self._ws_restart_count_in_window}/{self._ws_restart_max_in_window}"
        )

        self._stale_strikes = 0
        self.stats.ws_restarts += 1

        try:
            self.market.restart_ws(
                symbols=active_symbols,
                interval=self.interval,
                candle_limit=int(candle_limit),
                user=user,
            )
        except Exception:
            print(f"⚠️ WS restart failed\n{traceback.format_exc()}")
            return

        if self._ws_restart_count_in_window >= self._ws_restart_max_in_window:
            raise SystemExit(
                f"Too many WS restarts within {self._ws_restart_window_s:.0f}s window. Exiting for supervisor restart."
            )

    def _prepare_df_for_analysis(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        """Returns the dataframe slice we want to feed into analyze().

        If AI_QUANT_SIGNAL_ON_CANDLE_CLOSE=1, we act on the most recent closed candle.
        The HL candle stream frequently updates the in-progress candle. Dropping the last row
        when it is not yet closed makes entries stable and avoids intrabar noise.
        """
        if df is None or df.empty:
            return df

        if not self._signal_on_candle_close:
            return df

        if "T" in df.columns:
            try:
                t_close = df["T"].iloc[-1]
                if pd.notna(t_close):
                    if now_ms() < (int(t_close) - int(self._candle_close_grace_ms)):
                        if len(df) >= 6:
                            return df.iloc[:-1]
            except Exception:
                return df

        return df

    def _analysis_key(self, df: pd.DataFrame | None) -> int | None:
        if df is None or df.empty:
            return None

        if self._signal_on_candle_close and "T" in df.columns:
            try:
                t_close = df["T"].iloc[-1]
                if pd.notna(t_close):
                    return int(t_close)
            except Exception:
                pass

        try:
            return int(df["timestamp"].iloc[-1])
        except Exception:
            return None

    def _entry_is_too_late(self, *, entry_key: int, now_ts_ms: int) -> bool:
        """Returns True when acting on an old candle close should be skipped."""
        if not self._signal_on_candle_close:
            return False
        max_delay = int(self._entry_max_delay_ms or 0)
        if max_delay <= 0:
            return False
        try:
            delay = int(now_ts_ms) - int(entry_key)
        except Exception:
            return False
        if delay < 0:
            return False
        return delay > max_delay

    def _candle_key_hint(self, symbol: str) -> int | None:
        """Cheap per-symbol candle key used to decide whether we need to fetch a full DF.

        In close-mode, returns the close time of the last closed candle.
        In open-mode, returns the open time of the latest candle (may be in-progress).
        """
        try:
            if self._signal_on_candle_close:
                return self.market.get_last_closed_candle_key(
                    symbol,
                    interval=self.interval,
                    grace_ms=int(self._candle_close_grace_ms),
                )
            return self.market.get_latest_candle_open_key(symbol, interval=self.interval)
        except Exception:
            return None

    def _refresh_engine_config(self) -> None:
        """Re-read hot-reloadable engine params from YAML (entry_interval, exit_interval).

        Called at init and after every strategy.maybe_reload() in the main loop.
        """
        ecfg = self._ecfg((self.strategy.get_config("__GLOBAL__") or {}))

        # entry_interval — determines reanalyze cadence
        entry_iv = self._cfg_str(ecfg, "entry_interval", "ENTRY_INTERVAL", "")
        if not entry_iv or entry_iv == self.interval:
            # No sub-bar entry interval — reanalyze every main-interval bar
            self._entry_interval = self.interval
            self._reanalyze_interval_s = 0.0  # disabled — rely on candle-close key change
        else:
            self._entry_interval = entry_iv
            ms = _interval_to_ms(entry_iv)
            self._reanalyze_interval_s = float(ms) / 1000.0 if ms > 0 else 0.0

        # exit_interval — determines which candle DB to use for exit price
        # AND the cadence for exit checks (separate from entry reanalyze cadence).
        exit_iv = self._cfg_str(ecfg, "exit_interval", "EXIT_PRICE_SOURCE", "mid")
        # back-compat: strip "_candle" suffix (e.g. "3m_candle" → "3m")
        exit_iv = exit_iv.replace("_candle", "")
        self._exit_interval = exit_iv
        if exit_iv and exit_iv != "mid":
            exit_ms = _interval_to_ms(exit_iv)
            self._exit_reanalyze_interval_s = float(exit_ms) / 1000.0 if exit_ms > 0 else 0.0
        else:
            # No sub-bar exit interval — fall back to entry cadence
            self._exit_reanalyze_interval_s = self._reanalyze_interval_s

    def _reanalyze_due(self, symbol: str) -> bool:
        """Return True when the wall-clock interval bucket has changed for *symbol*.

        Aligned to :00 boundaries so that a 60s interval triggers at the start
        of each new minute — matching exactly when 1m candles close.
        """
        if self._reanalyze_interval_s <= 0:
            return False
        current_bucket = int(time.time()) // int(self._reanalyze_interval_s)
        last_bucket = self._last_analyze_ts.get(symbol, -1)
        return current_bucket != last_bucket

    def _mark_analyzed(self, symbol: str) -> None:
        """Record the current wall-clock bucket for *symbol*."""
        if self._reanalyze_interval_s <= 0:
            return
        self._last_analyze_ts[symbol] = int(time.time()) // int(self._reanalyze_interval_s)

    def _exit_reanalyze_due(self, symbol: str) -> bool:
        """Return True when the exit-interval wall-clock bucket has changed for *symbol*."""
        if self._exit_reanalyze_interval_s <= 0:
            return False
        current_bucket = int(time.time()) // int(self._exit_reanalyze_interval_s)
        last_bucket = self._last_exit_ts.get(symbol, -1)
        if current_bucket != last_bucket:
            self._last_exit_ts[symbol] = current_bucket
            return True
        return False

    def _attach_strategy_snapshot(self, *, symbol: str, now_series: Any) -> None:
        try:
            audit = now_series.get("audit")
        except Exception:
            audit = None

        if not isinstance(audit, dict):
            return

        snap = self.strategy.snapshot
        try:
            audit2 = dict(audit)
            audit2["strategy"] = {
                "version": str(snap.version or ""),
                "overrides_sha1": str(snap.overrides_sha1 or ""),
            }
            now_series["audit"] = audit2
        except Exception:
            return

    def run_forever(self) -> None:
        import strategy.mei_alpha_v1 as mei_alpha_v1

        try:
            import exchange.meta as hyperliquid_meta
        except Exception:
            hyperliquid_meta = None

        while True:
            loop_start = time.time()
            self.stats.loops += 1

            try:
                # Hot reload YAML only.
                self.strategy.maybe_reload()
                self._refresh_engine_config()

                if self.mode_plugin is not None:
                    self.mode_plugin.before_iteration()

                watchlist = self.strategy.get_watchlist()

                # v5.076: Exclude stable/pegged assets from watchlist (hot-reloadable).
                try:
                    _global_cfg = mei_alpha_v1.get_strategy_config("__GLOBAL__") or {}
                    _wl_exclude_raw = _global_cfg.get("watchlist_exclude") or []
                    if isinstance(_wl_exclude_raw, list) and _wl_exclude_raw:
                        _wl_exclude = {str(s).upper() for s in _wl_exclude_raw if str(s).strip()}
                        if _wl_exclude:
                            watchlist = [s for s in watchlist if s.upper() not in _wl_exclude]
                except Exception:
                    pass

                watch_set = {s.upper() for s in watchlist}

                neutral_sample_syms: set[str] = set()
                if self._neutral_audit_sample_every_s > 0:
                    now_s = time.time()
                    if (now_s - float(self._neutral_audit_last_s or 0.0)) >= float(self._neutral_audit_sample_every_s):
                        self._neutral_audit_last_s = now_s
                        wl = [str(s or "").upper() for s in (watchlist or []) if str(s or "").strip()]
                        if wl and self._neutral_audit_sample_symbols > 0:
                            k = max(1, min(int(self._neutral_audit_sample_symbols), len(wl)))
                            start = int(self._neutral_audit_cursor) % len(wl)
                            for i in range(k):
                                neutral_sample_syms.add(wl[(start + i) % len(wl)])
                            self._neutral_audit_cursor = (start + k) % len(wl)

                try:
                    open_syms = list((self.trader.positions or {}).keys())
                except Exception:
                    open_syms = []

                active_symbols = list(dict.fromkeys(["BTC"] + watchlist + open_syms))

                user = getattr(getattr(self.trader, "secrets", None), "main_address", None)
                candle_limit = self.lookback_bars + 50

                try:
                    self.market.ensure(
                        symbols=active_symbols,
                        interval=self.interval,
                        candle_limit=candle_limit,
                        user=user,
                    )
                except Exception:
                    pass

                try:
                    self._maybe_restart_ws(active_symbols=active_symbols, candle_limit=candle_limit, user=user)
                except SystemExit:
                    raise
                except Exception:
                    print(f"⚠️ WS health check failed\n{traceback.format_exc()}")

                # Candle readiness gate (sidecar only):
                # - Exits still run (using cached indicators + fresh price).
                # - Entries/adds are paused until candles are fully backfilled for the selected interval.
                not_ready_set: set[str] = set()
                try:
                    _ready, not_ready = self.market.candles_ready(symbols=active_symbols, interval=self.interval)
                    not_ready_set = {str(s).upper() for s in (not_ready or [])}
                except Exception:
                    # If we cannot determine readiness, be conservative and pause entries for watchlist symbols.
                    not_ready_set = {str(s).upper() for s in watchlist}

                # If we are sampling NEUTRAL audits this loop, also persist a single candle-readiness snapshot
                # so "no trades" can be distinguished between "no signals" vs "candles not ready".
                if neutral_sample_syms and not_ready_set:
                    # Optional: include a small health sample to make the audit actionable (e.g. new listings
                    # that legitimately have fewer historical candles).
                    not_ready_health = None
                    try:
                        sample_syms = sorted(list(not_ready_set))[:10]
                        health = self.market.candles_health(symbols=sample_syms, interval=self.interval)
                        if isinstance(health, dict):
                            items = health.get("items")
                            if isinstance(items, list):
                                out = []
                                for it in items:
                                    if not isinstance(it, dict):
                                        continue
                                    out.append(
                                        {
                                            "symbol": it.get("symbol"),
                                            "ready": it.get("ready"),
                                            "have_count": it.get("have_count"),
                                            "expected_count": it.get("expected_count"),
                                            "min_t": it.get("min_t"),
                                            "max_t": it.get("max_t"),
                                            "last_ok_backfill_ms": it.get("last_ok_backfill_ms"),
                                            "last_err_backfill": it.get("last_err_backfill"),
                                        }
                                    )
                                not_ready_health = out
                    except Exception:
                        not_ready_health = None

                    try:
                        mei_alpha_v1.log_audit_event(
                            "ENGINE",
                            "CANDLES_NOT_READY_SAMPLE",
                            data={
                                "mode": str(os.getenv("AI_QUANT_MODE", "paper") or "paper"),
                                "interval": str(self.interval),
                                "watchlist_n": int(len(watchlist or [])),
                                "active_symbols_n": int(len(active_symbols or [])),
                                "lookback_bars": int(self.lookback_bars),
                                "candle_limit": int(candle_limit),
                                "ready_n": int(max(0, len(active_symbols or []) - len(not_ready_set))),
                                "not_ready_n": int(len(not_ready_set)),
                                "not_ready": sorted(list(not_ready_set))[:200],
                                "not_ready_health": not_ready_health,
                            },
                        )
                    except Exception:
                        pass

                # BTC context (anchor) only refreshes when BTC candle key changes.
                btc_key_hint = self._candle_key_hint("BTC")
                btc_df_raw: pd.DataFrame | None = None
                btc_df: pd.DataFrame | None = None
                btc_bullish = self._btc_ctx.get("btc_bullish")

                if (
                    ("BTC" not in not_ready_set)
                    and (self._reanalyze_due("BTC") or (btc_key_hint is None) or (btc_key_hint != self._btc_ctx.get("key")))
                ):
                    btc_df_raw = self.market.get_candles_df("BTC", interval=self.interval, min_rows=self.lookback_bars)
                    btc_df = self._prepare_df_for_analysis(btc_df_raw)
                    btc_bullish = None
                    if btc_df is not None and not btc_df.empty:
                        try:
                            btc_ema_slow = mei_alpha_v1.ta.trend.ema_indicator(btc_df["Close"], window=50).iloc[-1]
                            if mei_alpha_v1.pd.notna(btc_ema_slow):
                                btc_bullish = btc_df["Close"].iloc[-1] > btc_ema_slow
                        except Exception:
                            btc_bullish = None

                    self._btc_ctx["key"] = btc_key_hint
                    self._btc_ctx["btc_bullish"] = btc_bullish
                    self._mark_analyzed("BTC")

                # Market Breadth: % of watchlist with bullish EMA alignment (Fast > Slow).
                # Computed from previous loop's cached analysis to avoid double-work.
                _breadth_pos = 0
                _breadth_total = 0
                for _bsym in watch_set:
                    _bc = self._analysis_cache.get(_bsym)
                    if not isinstance(_bc, dict):
                        continue
                    _bnow = _bc.get("now")
                    if _bnow is None:
                        continue
                    try:
                        _ef = float(_bnow.get("EMA_fast", 0.0) or 0.0)
                        _es = float(_bnow.get("EMA_slow", 0.0) or 0.0)
                        if _ef > 0 and _es > 0:
                            _breadth_total += 1
                            if _ef > _es:
                                _breadth_pos += 1
                    except Exception:
                        continue
                self._market_breadth_pct = ((_breadth_pos / _breadth_total) * 100.0) if _breadth_total > 0 else None

                # Phase 1: Loop active symbols — exits run immediately, entries are collected.
                _CONF_RANK = {"low": 0, "medium": 1, "high": 2}
                entry_candidates: list[dict[str, Any]] = []

                for symbol in active_symbols:
                    try:
                        sym_u = str(symbol or "").upper().strip()
                        if not sym_u:
                            continue

                        pos_open = sym_u in (self.trader.positions or {})
                        is_watch = sym_u in watch_set

                        # Use a cheap key hint first. Avoid pulling candles unless it changed.
                        key_hint = self._candle_key_hint(sym_u)

                        cached = self._analysis_cache.get(sym_u)
                        cached_key = cached.get("key") if isinstance(cached, dict) else None

                        is_entry_boundary = self._reanalyze_due(sym_u)
                        is_exit_boundary = self._exit_reanalyze_due(sym_u)
                        need_analyze = is_entry_boundary or (cached is None) or (key_hint is None) or (cached_key != key_hint)

                        sig = None
                        conf = None
                        now_series = None

                        if sym_u in not_ready_set and pos_open and (not need_analyze):
                            # Candle backfill in progress (or unhealthy): don't recompute indicators every loop.
                            # For open positions we still run exits using cached indicators + fresh price.
                            sig = cached.get("sig") if isinstance(cached, dict) else None
                            conf = cached.get("conf") if isinstance(cached, dict) else None
                            now_series = cached.get("now") if isinstance(cached, dict) else None
                            if now_series is None:
                                now_series = {"Close": None, "is_anomaly": False}
                            if sig is None:
                                sig = "NEUTRAL"
                            if conf is None:
                                conf = "N/A"

                        elif need_analyze:
                            # Only fetch candles when we truly need a recompute.
                            if sym_u == "BTC" and btc_df_raw is not None:
                                df_raw = btc_df_raw
                            else:
                                df_raw = self.market.get_candles_df(sym_u, interval=self.interval, min_rows=self.lookback_bars)

                            df = self._prepare_df_for_analysis(df_raw)
                            if df is None or df.empty or len(df) < int(self.lookback_bars):
                                # If we have an open position but cannot fetch candles (WS down, missing DB seed, invalid symbol),
                                # still run exit checks using the best available (cached) indicators + fresh price.
                                if pos_open:
                                    cached_now = cached.get("now") if isinstance(cached, dict) else None
                                    now_series = cached_now if cached_now is not None else {"Close": None, "is_anomaly": False}
                                    sig = "NEUTRAL"
                                    conf = "N/A"
                                else:
                                    continue

                            key = self._analysis_key(df)
                            if now_series is None:
                                sig, conf, now_series = mei_alpha_v1.analyze(df.copy(), sym_u, btc_bullish=btc_bullish)

                            self._analysis_cache[sym_u] = {
                                "key": int(key) if key is not None else (int(key_hint) if key_hint is not None else None),
                                "sig": str(sig),
                                "conf": str(conf),
                                "now": now_series,
                                "computed_at_s": time.time(),
                            }
                            self._mark_analyzed(sym_u)
                            cached = self._analysis_cache[sym_u]
                        else:
                            sig = cached.get("sig") if isinstance(cached, dict) else None
                            conf = cached.get("conf") if isinstance(cached, dict) else None
                            now_series = cached.get("now") if isinstance(cached, dict) else None

                        if now_series is None:
                            continue

                        sig_u = str(sig or "").upper()

                        # Reverse entry signal: manual toggle OR auto-reverse based on Breadth.
                        _reversed_entry = False
                        if sig_u in ("BUY", "SELL"):
                            try:
                                _full_cfg = mei_alpha_v1.get_strategy_config(sym_u) or {}
                                _trade_cfg = _full_cfg.get("trade") or {}
                                _regime_cfg = _full_cfg.get("market_regime") or {}

                                _should_reverse = bool(_trade_cfg.get("reverse_entry_signal", False))

                                # Auto-reverse: override manual flag based on Breadth zone.
                                if bool(_regime_cfg.get("enable_auto_reverse", False)) and self._market_breadth_pct is not None:
                                    _ar_lo = float(_regime_cfg.get("auto_reverse_breadth_low", 10.0))
                                    _ar_hi = float(_regime_cfg.get("auto_reverse_breadth_high", 90.0))
                                    _b = float(self._market_breadth_pct)
                                    # Choppy zone → reverse ON; Trending zone → reverse OFF.
                                    _should_reverse = (_ar_lo <= _b <= _ar_hi)

                                if _should_reverse:
                                    _orig_sig = sig_u
                                    sig_u = "SELL" if sig_u == "BUY" else "BUY"
                                    _reversed_entry = True
                                    print(f"🔁 {sym_u} signal REVERSED: {_orig_sig} → {sig_u} (Breadth={self._market_breadth_pct:.1f}%)" if self._market_breadth_pct is not None else f"🔁 {sym_u} signal REVERSED: {_orig_sig} → {sig_u}")
                            except Exception:
                                pass
                        if _reversed_entry and now_series is not None:
                            try:
                                now_series["_reversed_entry"] = True
                            except Exception:
                                pass

                        # Market Regime Bias: block entries against the market tide.
                        _regime_blocked = False
                        if sig_u in ("BUY", "SELL") and self._market_breadth_pct is not None:
                            try:
                                _regime_cfg = (mei_alpha_v1.get_strategy_config(sym_u) or {}).get("market_regime") or {}
                                if bool(_regime_cfg.get("enable_regime_filter", False)):
                                    _breadth = float(self._market_breadth_pct)
                                    _block_short_above = float(_regime_cfg.get("breadth_block_short_above", 90.0))
                                    _block_long_below = float(_regime_cfg.get("breadth_block_long_below", 10.0))
                                    if sig_u == "SELL" and _breadth > _block_short_above:
                                        print(f"🌊 {sym_u} SELL blocked by Market Regime: Breadth={_breadth:.1f}% > {_block_short_above:.0f}%")
                                        sig_u = "NEUTRAL"
                                        _regime_blocked = True
                                    elif sig_u == "BUY" and _breadth < _block_long_below:
                                        print(f"🌊 {sym_u} BUY blocked by Market Regime: Breadth={_breadth:.1f}% < {_block_long_below:.0f}%")
                                        sig_u = "NEUTRAL"
                                        _regime_blocked = True
                            except Exception:
                                pass

                        # ATR Floor: enforce minimum ATR as % of price to prevent noise stops.
                        if now_series is not None:
                            try:
                                _atr_raw = float(now_series.get("ATR") or 0.0)
                                _close_px = float(now_series.get("Close") or 0.0)
                                _min_atr_pct = float((mei_alpha_v1.get_trade_params(sym_u) or {}).get("min_atr_pct", 0.003) or 0.003)
                                if _close_px > 0 and _min_atr_pct > 0:
                                    _atr_floor = _close_px * _min_atr_pct
                                    if _atr_raw < _atr_floor:
                                        now_series["ATR"] = _atr_floor
                                        now_series["_atr_floored"] = True
                            except Exception:
                                pass

                        # Inject breadth + regime info into now_series for audit trail.
                        if now_series is not None:
                            try:
                                now_series["_market_breadth_pct"] = self._market_breadth_pct
                                if _regime_blocked:
                                    now_series["_regime_blocked"] = True
                            except Exception:
                                pass

                        # Optional gate debugging (even when NEUTRAL).
                        if self._debug_gates_every_s > 0 and self._debug_gates_symbols:
                            try:
                                want = ("*" in self._debug_gates_symbols) or (sym_u in self._debug_gates_symbols)
                            except Exception:
                                want = False
                            if want:
                                now_s = time.time()
                                last_t = None
                                try:
                                    last_t = cached.get("debug_last_print_s") if isinstance(cached, dict) else None
                                except Exception:
                                    last_t = None
                                if (last_t is None) or ((now_s - float(last_t)) >= float(self._debug_gates_every_s)):
                                    try:
                                        audit = None
                                        try:
                                            audit = now_series.get("audit") if hasattr(now_series, "get") else None
                                        except Exception:
                                            audit = None
                                        if isinstance(audit, dict):
                                            gates = audit.get("gates")
                                            tags = audit.get("tags")
                                            print(f"🔎 gates {sym_u} sig={sig_u} conf={conf} gates={gates} tags={tags}")
                                        else:
                                            print(f"🔎 gates {sym_u} sig={sig_u} conf={conf} audit_missing=1")
                                    except Exception:
                                        pass
                                    if isinstance(cached, dict):
                                        cached["debug_last_print_s"] = now_s

                        # Exits: run for open positions at exit-interval boundaries (aligned with backtester).
                        if pos_open and is_exit_boundary:
                            if self._rest_enabled and hyperliquid_meta is not None:
                                try:
                                    now_series["funding_rate"] = float(hyperliquid_meta.get_funding_rate(sym_u) or 0.0)
                                except Exception:
                                    now_series["funding_rate"] = 0.0
                            else:
                                now_series["funding_rate"] = 0.0

                            self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)

                            # Exit price sourcing: "mid" = WS mid price;
                            # any other value (e.g. "3m", "1m") = latest candle close from that interval DB.
                            # Reads from YAML engine.exit_interval (hot-reloadable via _refresh_engine_config).
                            quote = None
                            _exit_iv = self._exit_interval  # e.g. "3m", "mid"
                            if _exit_iv and _exit_iv != "mid":
                                try:
                                    df_exit = self.market.get_candles_df(sym_u, interval=_exit_iv, min_rows=1)
                                    if df_exit is not None and not df_exit.empty:
                                        current_price = float(df_exit["Close"].iloc[-1])
                                    else:
                                        # Fallback to mid price when candle unavailable
                                        quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                        current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))
                                except Exception:
                                    quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                    current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))
                            else:
                                quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))

                            try:
                                audit = now_series.get("audit")
                                if isinstance(audit, dict):
                                    audit2 = dict(audit)
                                    if _exit_iv and _exit_iv != "mid" and quote is None:
                                        audit2["quote"] = {"source": f"candle_{_exit_iv}_close", "age_s": 0.0}
                                    elif quote is not None:
                                        audit2["quote"] = {
                                            "source": str(getattr(quote, "source", "")),
                                            "age_s": float(getattr(quote, "age_s", 0.0) or 0.0),
                                        }
                                    now_series["audit"] = audit2
                            except Exception:
                                pass

                            try:
                                self.trader.check_exit_conditions(
                                    sym_u,
                                    current_price,
                                    now_ms(),
                                    is_anomaly=bool(now_series.get("is_anomaly", False)),
                                    dynamic_tp_mult=now_series.get("tp_mult"),
                                    indicators=now_series,
                                )
                            except AttributeError:
                                mei_alpha_v1.PaperTrader.check_exit_conditions(
                                    self.trader,
                                    sym_u,
                                    current_price,
                                    now_ms(),
                                    is_anomaly=bool(now_series.get("is_anomaly", False)),
                                    dynamic_tp_mult=now_series.get("tp_mult"),
                                    indicators=now_series,
                                )

                        # Optional NEUTRAL sampling: record gates/tags periodically for a few symbols.
                        if is_watch and sig_u == "NEUTRAL" and neutral_sample_syms and (sym_u in neutral_sample_syms):
                            try:
                                audit = now_series.get("audit")
                                if isinstance(audit, dict):
                                    mei_alpha_v1.log_audit_event(
                                        sym_u,
                                        "ANALYZE_NEUTRAL_SAMPLE",
                                        data={
                                            "interval": str(self.interval),
                                            "candle_key": int(key_hint) if key_hint is not None else None,
                                            "audit": audit,
                                        },
                                    )
                            except Exception:
                                pass
                            try:
                                neutral_sample_syms.discard(sym_u)
                            except Exception:
                                pass

                        # Entries: only watchlist symbols, only when signal is not neutral.
                        # Collect valid entry candidates; actual execution is deferred to
                        # the ranking phase after all symbols have been evaluated.
                        if (not is_watch) or sig_u == "NEUTRAL":
                            continue

                        # Determine the key used for entry de-dup.
                        entry_key = None
                        try:
                            entry_key = int(key_hint) if key_hint is not None else int((cached or {}).get("key"))  # type: ignore[arg-type]
                        except Exception:
                            entry_key = None

                        open_pos_count = 0
                        try:
                            open_pos_count = int(len(self.trader.positions or {}))
                        except Exception:
                            open_pos_count = 0

                        # Timing guard: don't enter on an old candle-close signal (common after restarts).
                        if entry_key is not None:
                            now_ts = now_ms()
                            if self._entry_is_too_late(entry_key=int(entry_key), now_ts_ms=int(now_ts)):
                                try:
                                    delay_s = max(0.0, (float(now_ts) - float(entry_key)) / 1000.0)
                                    max_s = float(self._entry_max_delay_ms) / 1000.0
                                    print(
                                        f"🕒 skip {sym_u} entry: stale candle-close signal "
                                        f"delay={delay_s:.0f}s > max={max_s:.0f}s key={int(entry_key)}"
                                    )
                                except Exception:
                                    print(f"🕒 skip {sym_u} entry: stale candle-close signal key={int(entry_key)}")
                                self._last_entry_key[sym_u] = int(entry_key)
                                self._last_entry_key_open_pos_count[sym_u] = open_pos_count
                                continue

                            last_key = self._last_entry_key.get(sym_u)
                            if last_key is not None and int(last_key) == int(entry_key):
                                if self._entry_retry_on_capacity:
                                    last_open_count = self._last_entry_key_open_pos_count.get(sym_u)
                                    if last_open_count is not None and open_pos_count < int(last_open_count):
                                        self._last_entry_key_open_pos_count[sym_u] = open_pos_count
                                    else:
                                        continue
                                else:
                                    continue

                        entry_candidates.append({
                            "symbol": sym_u,
                            "signal": sig_u,
                            "confidence": conf,
                            "adx": float(now_series.get("ADX", 0) or 0),
                            "entry_key": entry_key,
                            "now_series": now_series,
                            "open_pos_count": open_pos_count,
                        })

                    except SystemExit:
                        raise
                    except Exception:
                        print(f"⚠️ Engine symbol error: {symbol}\n{traceback.format_exc()}")
                        continue

                # Phase 2: Rank entry candidates by score (conf_rank * 100 + ADX),
                # tiebreak by symbol name (ascending). Execute in descending score order.
                if entry_candidates:
                    # Two-pass stable sort: first by symbol asc (tiebreak),
                    # then by score desc (primary). Python's sort is stable,
                    # so equal-score items retain alphabetical order.
                    entry_candidates.sort(key=lambda c: c["symbol"])
                    entry_candidates.sort(
                        key=lambda c: _CONF_RANK.get(c["confidence"], 0) * 100 + c["adx"],
                        reverse=True,
                    )

                    for cand in entry_candidates:
                        try:
                            sym_u = cand["symbol"]
                            sig_u = cand["signal"]
                            conf = cand["confidence"]
                            entry_key = cand["entry_key"]
                            now_series = cand["now_series"]
                            open_pos_count = cand["open_pos_count"]

                            # Entry price sourcing: use entry_interval candle close
                            # (aligned with backtester), fallback to mid-price.
                            quote = None
                            _entry_iv = self._entry_interval
                            if _entry_iv and _entry_iv != self.interval and _entry_iv != "mid":
                                try:
                                    df_entry = self.market.get_candles_df(sym_u, interval=_entry_iv, min_rows=1)
                                    if df_entry is not None and not df_entry.empty:
                                        current_price = float(df_entry["Close"].iloc[-1])
                                    else:
                                        quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                        current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))
                                except Exception:
                                    quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                    current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))
                            else:
                                quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                current_price = float(quote.price) if quote is not None else float(now_series.get("Close"))

                            # If no key, fall back to executing immediately (legacy behavior).
                            if entry_key is None:
                                self.trader.execute_trade(
                                    sym_u,
                                    sig_u,
                                    current_price,
                                    now_ms(),
                                    conf,
                                    atr=float(now_series.get("ATR") or 0.0),
                                    indicators=now_series,
                                )
                                continue

                            if self._rest_enabled and hyperliquid_meta is not None:
                                try:
                                    now_series["funding_rate"] = float(hyperliquid_meta.get_funding_rate(sym_u) or 0.0)
                                except Exception:
                                    now_series["funding_rate"] = 0.0
                            else:
                                now_series["funding_rate"] = 0.0

                            self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)

                            try:
                                audit = now_series.get("audit")
                                if isinstance(audit, dict):
                                    audit2 = dict(audit)
                                    if _entry_iv and _entry_iv != self.interval and _entry_iv != "mid" and quote is None:
                                        audit2["quote"] = {"source": f"candle_{_entry_iv}_close", "age_s": 0.0}
                                    elif quote is not None:
                                        audit2["quote"] = {
                                            "source": str(getattr(quote, "source", "")),
                                            "age_s": float(getattr(quote, "age_s", 0.0) or 0.0),
                                        }
                                    now_series["audit"] = audit2
                            except Exception:
                                pass

                            self._last_entry_key[sym_u] = int(entry_key)
                            self._last_entry_key_open_pos_count[sym_u] = open_pos_count

                            self.trader.execute_trade(
                                sym_u,
                                sig_u,
                                current_price,
                                int(entry_key),
                                conf,
                                atr=float(now_series.get("ATR") or 0.0),
                                indicators=now_series,
                            )
                        except SystemExit:
                            raise
                        except Exception:
                            print(f"⚠️ Engine ranked-entry error: {cand.get('symbol')}\n{traceback.format_exc()}")
                            continue

                if self.mode_plugin is not None:
                    self.mode_plugin.after_iteration()

                if (time.time() - self.stats.last_heartbeat_s) >= self._heartbeat_every_s:
                    snap = self.strategy.snapshot
                    h = self.market.health(symbols=active_symbols, interval=self.interval)

                    try:
                        open_pos = len(self.trader.positions or {})
                    except Exception:
                        open_pos = 0

                    loop_s = time.time() - loop_start
                    _rc = (mei_alpha_v1.get_strategy_config("BTC") or {}).get("market_regime") or {}
                    _auto_rev_on = (
                        bool(_rc.get("enable_auto_reverse", False))
                        and self._market_breadth_pct is not None
                        and float(_rc.get("auto_reverse_breadth_low", 20.0)) <= self._market_breadth_pct <= float(_rc.get("auto_reverse_breadth_high", 80.0))
                    )
                    print(
                        f"🫀 engine ok. loops={self.stats.loops} errors={self.stats.loop_errors} "
                        f"symbols={len(active_symbols)} open_pos={open_pos} loop={loop_s:.2f}s "
                        f"ws_connected={h.get('connected')} ws_thread_alive={h.get('thread_alive')} "
                        f"ws_restarts={self.stats.ws_restarts} "
                        f"signal_on_close={int(self._signal_on_candle_close)} entry_iv={self._entry_interval} exit_iv={self._exit_interval} reanalyze_s={self._reanalyze_interval_s:.0f} exit_reanalyze_s={self._exit_reanalyze_interval_s:.0f} "
                        f"breadth={f'{self._market_breadth_pct:.1f}%' if self._market_breadth_pct is not None else 'n/a'} "
                        f"auto_rev={'ON' if _auto_rev_on else 'OFF'} "
                        f"strategy_sha1={str(snap.overrides_sha1 or '')[:8]} version={snap.version or 'n/a'}"
                    )
                    self.stats.last_heartbeat_s = time.time()

            except SystemExit:
                raise
            except Exception:
                self.stats.loop_errors += 1
                print(f"🔥 Engine loop error\n{traceback.format_exc()}")
            finally:
                # Align sleep to wall-clock boundaries (e.g., :00 for 60s target).
                # This ensures loops fire at predictable times matching candle closes.
                now = time.time()
                interval = int(self._loop_target_s) or 1
                next_boundary = ((int(now) // interval) + 1) * interval
                sleep_s = max(0.1, next_boundary - now)
                time.sleep(sleep_s)
