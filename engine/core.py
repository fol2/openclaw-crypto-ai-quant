from __future__ import annotations

import os
import json
import time
import traceback
import importlib
from importlib import util
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

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


@dataclass
class KernelDecision:
    symbol: str
    action: str
    signal: str
    confidence: str
    score: float = 0.0
    now_series: dict[str, Any] | None = None
    target_size: float | None = None
    entry_key: int | None = None
    reason: str | None = None
    open_pos_count: int = 0

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | Any) -> "KernelDecision | None":
        if not isinstance(raw, Mapping):
            return None

        sym = str(raw.get("symbol", "")).strip().upper()
        if not sym:
            return None

        act = _normalise_kernel_action(raw.get("action"), raw_kind=raw.get("kind"))
        if act not in {"OPEN", "ADD", "CLOSE", "REDUCE"}:
            return None

        signal = _normalise_kernel_intent_signal(
            raw.get("signal"),
            raw_action=raw.get("action"),
            raw_kind=raw.get("kind"),
            raw_side=raw.get("side"),
        )
        confidence = str(raw.get("confidence", "N/A"))

        try:
            score = float(raw.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        now_series = raw.get("now_series")
        if not isinstance(now_series, dict):
            now_series = {}

        target_size = _normalise_kernel_target_size(raw.get("quantity"), raw.get("notional_usd"), raw.get("price"))
        if target_size is None:
            target_size = _normalise_kernel_target_size(
                raw.get("target_size"),
                raw.get("notional_hint_usd"),
                raw.get("price"),
            )

        entry_key = raw.get(
            "entry_key",
            raw.get("intent_id", raw.get("entry_candle_key", raw.get("candle_key"))),
        )
        try:
            entry_key = int(entry_key) if entry_key is not None else None
        except Exception:
            entry_key = None

        reason = raw.get("reason")
        if isinstance(reason, str):
            reason = reason.strip() or None
        else:
            reason = None

        try:
            open_pos_count = int(raw.get("open_pos_count", 0) or 0)
        except Exception:
            open_pos_count = 0

        return cls(
            symbol=sym,
            action=act,
            signal=signal,
            confidence=confidence,
            score=score,
            now_series=now_series,
            target_size=target_size,
            entry_key=entry_key,
            reason=reason,
            open_pos_count=open_pos_count,
        )


class DecisionProvider(Protocol):
    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: Any,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: Any,
        now_ms: int,
    ) -> Iterable[KernelDecision]:
        ...


class KernelDecisionFileProvider:
    def __init__(self, path: str | None):
        self.path = str(path).strip() if path else None

    def _load_raw(self) -> list[dict[str, Any]]:
        path = self.path
        if not path:
            return []

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return []
        except Exception:
            return []

        if isinstance(raw, dict):
            if isinstance(raw.get("decisions"), list):
                return [item for item in raw["decisions"] if isinstance(item, Mapping)]
            if "symbol" in raw:
                return [raw]  # single-item payload
            return []

        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Mapping)]

        return []

    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: Any,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: Any,
        now_ms: int,
    ) -> Iterable[KernelDecision]:
        del symbols, watchlist, open_symbols, market, interval, lookback_bars, mode, not_ready_symbols, strategy, now_ms
        for raw in self._load_raw():
            dec = KernelDecision.from_raw(raw)
            if dec is None:
                continue
            yield dec


def _load_kernel_runtime_module(module_name: str = "bt_runtime"):
    """Import and return the Rust runtime binding module."""
    def _try_import() -> object:
        return importlib.import_module(module_name)

    try:
        return _try_import()
    except ModuleNotFoundError as exc:
        if exc.name not in {module_name, None}:
            raise

        candidates = [
            Path(os.getenv("AI_QUANT_BT_RUNTIME_PATH") or ""),
            Path(__file__).resolve().parents[1] / "backtester" / "target" / "release" / "deps",
            Path(__file__).resolve().parents[1] / "backtester" / "target" / "release",
            Path(__file__).resolve().parents[1] / "backtester" / "target" / "debug" / "deps",
            Path(__file__).resolve().parents[1] / "backtester" / "target" / "debug",
        ]
        for candidate in candidates:
            candidate_str = str(candidate)
            if not candidate.exists() or candidate_str in sys.path:
                continue
            sys.path.insert(0, candidate_str)

        try:
            return _try_import()
        except ModuleNotFoundError as second_exc:
            if second_exc.name not in {module_name, None}:
                raise

        for candidate_dir in candidates:
            if not candidate_dir.is_dir():
                continue
            for pattern in [
                f"lib{module_name}*.so",
                f"{module_name}*.so",
                f"{module_name}.so",
            ]:
                for candidate in candidate_dir.glob(pattern):
                    if not candidate.is_file():
                        continue
                    if not candidate.name.startswith(("lib" + module_name, module_name)):
                        continue
                    spec = util.spec_from_file_location(module_name, str(candidate))
                    if spec is None or spec.loader is None:
                        continue
                    module = util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module

        raise


def _normalise_kernel_signal(raw_signal: Any) -> str:
    sig = str(raw_signal or "").strip().upper()
    if sig == "LONG":
        return "BUY"
    if sig == "SHORT":
        return "SELL"
    return sig


def _normalise_kernel_side(raw_side: Any) -> str:
    side = str(raw_side or "").strip().lower()
    if side == "long":
        return "BUY"
    if side == "short":
        return "SELL"
    return str(raw_side or "NEUTRAL").strip().upper()


def _normalise_kernel_intent_signal(
    raw_signal: Any,
    *,
    raw_action: Any,
    raw_kind: Any,
    raw_side: Any,
) -> str:
    has_signal = not (raw_signal is None or str(raw_signal).strip() == "")

    signal = _normalise_kernel_signal(raw_signal)
    if signal in {"BUY", "SELL", "NEUTRAL"}:
        return signal

    if raw_side is not None and str(raw_side).strip() != "":
        side = _normalise_kernel_side(raw_side)
        if side in {"BUY", "SELL", "NEUTRAL"}:
            return side

    if not has_signal:
        action = _normalise_kernel_action(raw_action, raw_kind)
        if action:
            return action

    return "NEUTRAL"


def _normalise_kernel_action(raw_action: Any, raw_kind: Any = None) -> str:
    action = str(raw_action or "").strip().upper()
    if action in {"OPEN", "ADD", "CLOSE", "REDUCE"}:
        return action
    if action == "REVERSE":
        return "CLOSE"

    kind = str(raw_kind or "").strip().lower()
    if kind == "open":
        return "OPEN"
    if kind == "add":
        return "ADD"
    if kind == "close":
        return "CLOSE"
    if kind == "reverse":
        return "CLOSE"

    return ""


def _normalise_kernel_target_size(raw_size: Any, raw_notional: Any, raw_price: Any) -> float | None:
    try:
        notional = float(raw_notional)
        if notional > 0.0:
            price = float(raw_price)
            if price <= 0.0:
                return None
            return notional / price
    except Exception:
        pass

    try:
        quantity = float(raw_size)
        if quantity > 0.0:
            return quantity
    except Exception:
        pass

    try:
        notional = float(raw_notional)
        if notional <= 0.0:
            return None
    except Exception:
        return None

    try:
        price = float(raw_price)
        if price <= 0.0:
            return None
        return notional / price
    except Exception:
        return None


def _normalise_kernel_price(raw_price: Any, default: float = 0.0) -> float:
    try:
        price = float(raw_price)
        if price > 0.0:
            return price
    except Exception:
        pass
    return default


def _normalise_kernel_notional(raw_notional: Any, raw_size: Any, price: float) -> float:
    try:
        raw_value = float(raw_notional)
        if raw_value > 0.0:
            return raw_value
    except Exception:
        pass
    try:
        size_value = float(raw_size)
        if size_value > 0.0 and price > 0.0:
            return size_value * price
    except Exception:
        pass
    return 0.0


def _normalise_kernel_event_id(raw_id: Any, fallback: int) -> int:
    try:
        parsed = int(raw_id)
        if parsed >= 0:
            return parsed
    except Exception:
        pass
    return fallback


class KernelDecisionRustBindingProvider:
    """Call the Rust decision kernel via the `bt_runtime` extension."""

    def __init__(self, module_name: str = "bt_runtime", path: str | None = None) -> None:
        self._runtime = _load_kernel_runtime_module(module_name)
        self._path = str(path).strip() if path else None
        self._state_json = self._runtime.default_kernel_state_json(10_000.0, now_ms())
        self._params_json = self._runtime.default_kernel_params_json()
        self._event_id = 1

    def _load_raw_events(self) -> list[dict[str, Any]]:
        if not self._path:
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return []
        except Exception:
            return []

        if isinstance(raw, dict):
            if isinstance(raw.get("events"), list):
                return [item for item in raw["events"] if isinstance(item, Mapping)]
            if "schema_version" in raw and "symbol" in raw:
                return [raw]
            return []
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Mapping)]
        return []

    def _build_market_event(self, raw: Mapping[str, Any], *, now_ms_value: int) -> str | None:
        symbol = str(raw.get("symbol", "")).strip().upper()
        if not symbol:
            return None

        signal = _normalise_kernel_intent_signal(
            raw.get("signal"),
            raw_action=raw.get("action"),
            raw_kind=raw.get("kind"),
            raw_side=raw.get("side"),
        )
        if signal not in {"BUY", "SELL", "NEUTRAL"}:
            return None

        price = _normalise_kernel_price(
            raw.get("price"),
            default=0.0,
        )
        if price <= 0.0:
            return None

        notional_hint = _normalise_kernel_notional(
            raw.get("notional_hint_usd"),
            raw.get("target_size"),
            price,
        )

        timestamp_ms = raw.get("timestamp_ms")
        try:
            timestamp_ms = int(timestamp_ms)
            if timestamp_ms <= 0:
                timestamp_ms = now_ms_value
        except Exception:
            timestamp_ms = now_ms_value

        event_id = _normalise_kernel_event_id(raw.get("event_id"), self._event_id)
        self._event_id = max(self._event_id, event_id + 1)

        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp_ms": timestamp_ms,
            "symbol": symbol,
            "signal": signal.lower(),
            "price": price,
            "notional_hint_usd": notional_hint if notional_hint > 0.0 else None,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _intent_to_decision(
        self,
        decision: Mapping[str, Any],
        event_raw: Mapping[str, Any],
    ) -> KernelDecision | None:
        symbol = str(decision.get("symbol", "")).strip().upper()
        if not symbol:
            return None

        result = KernelDecision.from_raw(
            {
                "symbol": symbol,
                "kind": decision.get("kind"),
                "side": decision.get("side"),
                "quantity": decision.get("quantity"),
                "price": decision.get("price"),
                "notional_usd": decision.get("notional_usd"),
                "notional_hint_usd": decision.get("notional_hint_usd"),
                "intent_id": decision.get("intent_id"),
                "signal": event_raw.get("signal"),
                "confidence": event_raw.get("confidence", "N/A"),
                "now_series": event_raw.get("now_series"),
                "entry_key": event_raw.get("entry_key"),
            }
        )
        if result is None:
            return None

        result.reason = f"kernel:{str(decision.get('kind', 'unknown')).strip().lower()}"
        return result

    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: Any,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: Any,
        now_ms: int,
    ) -> Iterable[KernelDecision]:
        del symbols, watchlist, open_symbols, market, interval, lookback_bars, mode, not_ready_symbols, strategy
        raw_events = self._load_raw_events()
        if not raw_events:
            return []

        decisions: list[KernelDecision] = []
        state_json = self._state_json

        for raw in raw_events:
            event_json = self._build_market_event(raw, now_ms_value=int(now_ms or 0))
            if event_json is None:
                continue
            response = self._runtime.step_decision(state_json, event_json, self._params_json)
            try:
                envelope = json.loads(response)
            except Exception:
                continue
            if not bool(envelope.get("ok", False)):
                continue
            decision_payload = envelope.get("decision")
            if not isinstance(decision_payload, dict):
                continue

            state_json = json.dumps(decision_payload.get("state"), ensure_ascii=False)
            self._state_json = state_json
            for raw_intent in decision_payload.get("intents", []):
                if not isinstance(raw_intent, Mapping):
                    continue
                dec = self._intent_to_decision(raw_intent, raw)
                if dec is not None:
                    decisions.append(dec)

        return decisions


class NoopDecisionProvider:
    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: Any,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: Any,
        now_ms: int,
    ) -> Iterable[KernelDecision]:
        del symbols, watchlist, open_symbols, market, interval, lookback_bars, mode, not_ready_symbols, strategy, now_ms
        return []


_CONF_RANK = {"high": 2, "medium": 1, "low": 0}


class PythonAnalyzeDecisionProvider:
    """Generate entry decisions via mei_alpha_v1.analyze() — Python signal path.

    Restores the pre-TKT-004 signal generation that was removed when kernel
    decision routing was introduced (415ccef).  The Rust backtester computes
    indicators and signals internally; this provider gives the live/paper
    engine an equivalent signal source so the SSOT decision path is not broken.

    When AI_QUANT_KERNEL_DECISION_FILE is configured, use
    KernelDecisionRustBindingProvider instead (file-based kernel decisions).
    """

    def __init__(self) -> None:
        self._btc_bullish: bool | None = None
        self._btc_key: int | None = None

    # ------------------------------------------------------------------
    # DecisionProvider protocol
    # ------------------------------------------------------------------
    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: Any,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: Any,
        now_ms: int,
    ) -> Iterable[KernelDecision]:
        decisions: list[KernelDecision] = []
        open_set = {s.upper() for s in (open_symbols or [])}

        # --- BTC context (require_btc_alignment filter) ----------------
        btc_bullish = self._btc_bullish
        try:
            btc_df = market.get_candles_df("BTC", interval=interval, min_rows=lookback_bars)
            if btc_df is not None and not btc_df.empty and len(btc_df) >= lookback_bars:
                btc_ema_slow = strategy.ta.trend.ema_indicator(
                    btc_df["Close"], window=50,
                ).iloc[-1]
                if strategy.pd.notna(btc_ema_slow):
                    btc_bullish = bool(btc_df["Close"].iloc[-1] > btc_ema_slow)
                    self._btc_bullish = btc_bullish
        except Exception:
            pass

        # --- Per-symbol signal generation ------------------------------
        for sym in watchlist:
            sym_u = sym.upper().strip()
            if not sym_u or sym_u in not_ready_symbols:
                continue

            try:
                df_raw = market.get_candles_df(
                    sym_u, interval=interval, min_rows=lookback_bars,
                )
                if df_raw is None or df_raw.empty or len(df_raw) < lookback_bars:
                    continue

                df = df_raw.tail(lookback_bars).copy()
                sig, conf, now_series = strategy.analyze(
                    df, sym_u, btc_bullish=btc_bullish,
                )

                sig_u = str(sig or "").upper()
                if sig_u not in ("BUY", "SELL"):
                    continue

                if not isinstance(now_series, dict):
                    now_series = {}

                # ATR floor: enforce minimum ATR as % of price.
                try:
                    _atr_raw = float(now_series.get("ATR") or 0.0)
                    _close_px = float(now_series.get("Close") or 0.0)
                    _min_atr_pct = float(
                        (strategy.get_trade_params(sym_u) or {}).get(
                            "min_atr_pct", 0.003,
                        ) or 0.003,
                    )
                    if _close_px > 0 and _min_atr_pct > 0:
                        _atr_floor = _close_px * _min_atr_pct
                        if _atr_raw < _atr_floor:
                            now_series["ATR"] = _atr_floor
                            now_series["_atr_floored"] = True
                except Exception:
                    pass

                # Candle key for dedup (engine handles actual dedup).
                entry_key: int | None = None
                try:
                    if "T" in df.columns:
                        entry_key = int(df["T"].iloc[-1])
                    else:
                        entry_key = int(df["timestamp"].iloc[-1])
                except Exception:
                    pass

                price = float(now_series.get("Close") or 0)
                if price <= 0:
                    continue

                action = "ADD" if sym_u in open_set else "OPEN"
                adx = float(now_series.get("ADX") or 0)
                score = _CONF_RANK.get(str(conf or "").lower(), 0) * 100 + adx

                decisions.append(
                    KernelDecision(
                        symbol=sym_u,
                        action=action,
                        signal=sig_u,
                        confidence=str(conf or "N/A"),
                        score=score,
                        now_series=now_series,
                        entry_key=entry_key,
                        reason=f"python_analyze:{sig_u.lower()}",
                    )
                )
            except Exception:
                continue

        return decisions


def _build_default_decision_provider() -> DecisionProvider:
    path = os.getenv("AI_QUANT_KERNEL_DECISION_FILE")
    provider_mode = str(os.getenv("AI_QUANT_KERNEL_DECISION_PROVIDER", "") or "").strip().lower()

    if provider_mode in {"none", "noop"}:
        return NoopDecisionProvider()

    if provider_mode == "python":
        print("📊 Decision provider: PythonAnalyzeDecisionProvider (explicit)")
        return PythonAnalyzeDecisionProvider()

    if provider_mode == "file":
        if not path:
            raise RuntimeError(
                "AI_QUANT_KERNEL_DECISION_PROVIDER=file requires AI_QUANT_KERNEL_DECISION_FILE to be set."
            )
        return KernelDecisionFileProvider(path)

    if provider_mode == "rust":
        if path:
            try:
                return KernelDecisionRustBindingProvider(path=path)
            except Exception as exc:
                raise RuntimeError(
                    "AI_QUANT_KERNEL_DECISION_PROVIDER=rust is configured, but the Rust decision kernel "
                    "extension is unavailable. Set AI_QUANT_KERNEL_DECISION_PROVIDER=none or provide "
                    "AI_QUANT_KERNEL_DECISION_FILE."
                ) from exc
        # No decision file: Rust binding cannot generate signals from candle data
        # on its own.  Fall back to Python analyze path which replicates the same
        # indicator / filter logic that the Rust backtester uses internally.
        print(
            "⚠️ AI_QUANT_KERNEL_DECISION_PROVIDER=rust but AI_QUANT_KERNEL_DECISION_FILE "
            "not set. Falling back to PythonAnalyzeDecisionProvider (mei_alpha_v1.analyze)."
        )
        return PythonAnalyzeDecisionProvider()

    if path:
        try:
            return KernelDecisionRustBindingProvider(path=path)
        except Exception:
            return KernelDecisionFileProvider(path)

    # Auto mode: try Rust binding with file, then Python analyze, then crash.
    try:
        return KernelDecisionRustBindingProvider(path=None)
    except Exception:
        print(
            "⚠️ Decision provider auto-mode: Rust kernel extension unavailable and "
            "AI_QUANT_KERNEL_DECISION_FILE not configured. "
            "Falling back to PythonAnalyzeDecisionProvider (mei_alpha_v1.analyze)."
        )
        return PythonAnalyzeDecisionProvider()


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
        mode: str | None = None,
        mode_plugin: ModePlugin | None = None,
        decision_provider: DecisionProvider | None = None,
    ):
        self.trader = trader
        self.strategy = strategy
        self.market = market
        self.interval = str(interval)
        self.lookback_bars = int(lookback_bars)
        self.mode = str(mode or os.getenv("AI_QUANT_MODE", "paper") or "paper").strip().lower()
        self.mode_plugin = mode_plugin
        self.decision_provider = decision_provider or _build_default_decision_provider()

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

        # Global regime gate (trend vs chop). When OFF, entries are blocked (exits still run).
        self._regime_gate_on: bool = True
        self._regime_gate_reason: str = "disabled"
        self._regime_gate_last_key: int | None = None
        self._regime_gate_last_logged_on: bool | None = None

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

    def _update_regime_gate(
        self,
        *,
        mei_alpha_v1: Any,
        btc_key_hint: int | None,
        btc_df: pd.DataFrame | None,
    ) -> None:
        """Update the global regime gate state.

        The gate is designed to flip only on a schedule (typically once per main bar),
        keyed off the BTC candle key. When the gate is OFF, entries are blocked
        (exits still run).
        """
        try:
            key = int(btc_key_hint) if btc_key_hint is not None else None
        except Exception:
            key = None

        # Avoid recomputing every loop; BTC key only changes once per main bar (in close-mode).
        if key is not None and self._regime_gate_last_key is not None and int(key) == int(self._regime_gate_last_key):
            return
        if key is None and self._regime_gate_last_key is None and self.stats.loops > 1:
            # No key available: keep the previous state to avoid flapping.
            return
        self._regime_gate_last_key = key

        cfg = mei_alpha_v1.get_strategy_config("BTC") or {}
        rc = cfg.get("market_regime") if isinstance(cfg, dict) else None
        rc = rc if isinstance(rc, dict) else {}

        enabled = bool(rc.get("enable_regime_gate", False))
        fail_open = bool(rc.get("regime_gate_fail_open", False))

        if not enabled:
            new_on = True
            new_reason = "disabled"
        else:
            # Breadth chop zone: inside => gate OFF; outside => potential trend regime.
            try:
                chop_lo = float(rc.get("regime_gate_breadth_low", rc.get("auto_reverse_breadth_low", 10.0)))
            except Exception:
                chop_lo = 10.0
            try:
                chop_hi = float(rc.get("regime_gate_breadth_high", rc.get("auto_reverse_breadth_high", 90.0)))
            except Exception:
                chop_hi = 90.0

            try:
                btc_adx_min = float(rc.get("regime_gate_btc_adx_min", 20.0))
            except Exception:
                btc_adx_min = 20.0
            try:
                btc_atr_pct_min = float(rc.get("regime_gate_btc_atr_pct_min", 0.003))
            except Exception:
                btc_atr_pct_min = 0.003

            breadth = self._market_breadth_pct
            try:
                b = float(breadth) if breadth is not None else None
            except Exception:
                b = None

            if b is None:
                new_on = bool(fail_open)
                new_reason = "breadth_missing"
            elif chop_lo <= b <= chop_hi:
                new_on = False
                new_reason = "breadth_chop"
            else:
                # BTC metrics: prefer cached analysis output, fall back to lightweight indicator compute.
                adx: float | None = None
                atr_pct: float | None = None

                btc_now = None
                try:
                    bc = self._analysis_cache.get("BTC")
                    btc_now = bc.get("now") if isinstance(bc, dict) else None
                except Exception:
                    btc_now = None

                if isinstance(btc_now, dict):
                    try:
                        adx = float(btc_now.get("ADX") or 0.0)
                    except Exception:
                        adx = None
                    try:
                        atr = float(btc_now.get("ATR") or 0.0)
                        close = float(btc_now.get("Close") or 0.0)
                        if close > 0:
                            atr_pct = atr / close
                    except Exception:
                        atr_pct = None

                if (adx is None or atr_pct is None) and btc_df is not None and (not btc_df.empty):
                    ind = cfg.get("indicators") if isinstance(cfg, dict) else None
                    ind = ind if isinstance(ind, dict) else {}
                    try:
                        adx_w = int(ind.get("adx_window", 14))
                    except Exception:
                        adx_w = 14
                    try:
                        atr_w = int(ind.get("atr_window", 14))
                    except Exception:
                        atr_w = 14

                    try:
                        adx_obj = mei_alpha_v1.ta.trend.ADXIndicator(
                            btc_df["High"],
                            btc_df["Low"],
                            btc_df["Close"],
                            window=int(adx_w),
                        )
                        adx_s = adx_obj.adx()
                        if not adx_s.empty:
                            adx = float(adx_s.iloc[-1])
                    except Exception:
                        pass

                    try:
                        atr_s = mei_alpha_v1.ta.volatility.average_true_range(
                            btc_df["High"],
                            btc_df["Low"],
                            btc_df["Close"],
                            window=int(atr_w),
                        )
                        if not atr_s.empty:
                            atr = float(atr_s.iloc[-1])
                            close = float(btc_df["Close"].iloc[-1])
                            if close > 0:
                                atr_pct = atr / close
                    except Exception:
                        pass

                if adx is None or atr_pct is None:
                    new_on = bool(fail_open)
                    new_reason = "btc_metrics_missing"
                elif float(adx) < float(btc_adx_min):
                    new_on = False
                    new_reason = "btc_adx_low"
                elif float(atr_pct) < float(btc_atr_pct_min):
                    new_on = False
                    new_reason = "btc_atr_low"
                else:
                    new_on = True
                    new_reason = "trend_ok"

        self._regime_gate_on = bool(new_on)
        self._regime_gate_reason = str(new_reason or "none").strip() or "none"

        if self._regime_gate_last_logged_on is None or bool(self._regime_gate_on) != bool(self._regime_gate_last_logged_on):
            self._regime_gate_last_logged_on = bool(self._regime_gate_on)
            try:
                mei_alpha_v1.log_audit_event(
                    "ENGINE",
                    "REGIME_GATE_STATE",
                    data={
                        "interval": str(self.interval),
                        "gate_on": bool(self._regime_gate_on),
                        "reason": str(self._regime_gate_reason),
                        "btc_key": key,
                        "breadth_pct": self._market_breadth_pct,
                    },
                )
            except Exception:
                pass
            try:
                print(f"🧭 regime gate {'ON' if self._regime_gate_on else 'OFF'} reason={self._regime_gate_reason}")
            except Exception:
                pass

    def _decision_execute_trade(
        self,
        symbol: str,
        signal: str,
        price: float,
        timestamp: int,
        confidence: str,
        *,
        atr: float = 0.0,
        indicators=None,
        action: str = "",
        target_size: float | None = None,
        reason: str | None = None,
    ) -> None:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return

        act = str(action or "").strip().upper()
        if act not in {"OPEN", "ADD", "CLOSE", "REDUCE"}:
            return

        try:
            target_size_f = float(target_size) if target_size is not None else None
        except Exception:
            target_size_f = None

        try:
            return self.trader.execute_trade(
                symbol,
                signal,
                price,
                timestamp,
                confidence,
                atr=atr,
                indicators=indicators,
                action=act,
                target_size=target_size_f,
                reason=reason,
            )
        except TypeError:
            # Fall back only to explicit action handlers to avoid silently degrading to signal-only legacy routing.
            if act == "OPEN":
                return
            if act == "ADD":
                fn = getattr(self.trader, "add_to_position", None)
                if callable(fn):
                    try:
                        return fn(sym, price, timestamp, confidence, atr=atr, indicators=indicators, target_size=target_size_f)
                    except TypeError:
                        return fn(sym, price, timestamp, confidence, atr=atr, indicators=indicators)
                return

            if act in {"CLOSE", "REDUCE"}:
                if sym not in ((self.trader.positions or {}) if isinstance(getattr(self.trader, "positions", None), dict) else {}):
                    return
                pos = (self.trader.positions or {}).get(sym, {})
                try:
                    pos_size = float(pos.get("size") or 0.0)
                except Exception:
                    pos_size = 0.0
                try:
                    reduce_sz = float(target_size_f) if target_size_f is not None else pos_size
                except Exception:
                    reduce_sz = pos_size
                reduce_sz = float(max(0.0, min(pos_size, reduce_sz)))

                if act == "CLOSE":
                    fn = getattr(self.trader, "close_position", None)
                    if callable(fn):
                        try:
                            return fn(
                                sym,
                                price,
                                timestamp,
                                reason=str(reason or "Kernel CLOSE"),
                                meta={"reason": str(reason or "").strip() or None},
                            )
                        except TypeError:
                            return fn(sym, price, timestamp, str(reason or "Kernel CLOSE"))
                    return
                fn = getattr(self.trader, "reduce_position", None)
                if callable(fn):
                    return fn(
                        sym,
                        reduce_sz,
                        price,
                        timestamp,
                        str(reason or "Kernel REDUCE"),
                        confidence=confidence,
                        meta={"reason": str(reason or "").strip() or None},
                    )
                return

        if act == "ADD":
            fn = getattr(self.trader, "add_to_position", None)
            if callable(fn):
                try:
                    return fn(sym, price, timestamp, confidence, atr=atr, indicators=indicators, target_size=target_size_f)
                except TypeError:
                    return fn(sym, price, timestamp, confidence, atr=atr, indicators=indicators)
            return

        if act in {"CLOSE", "REDUCE"}:
            if sym not in ((self.trader.positions or {}) if isinstance(getattr(self.trader, "positions", None), dict) else {}):
                return
            pos = (self.trader.positions or {}).get(sym, {})
            try:
                pos_size = float(pos.get("size") or 0.0)
            except Exception:
                pos_size = 0.0
            try:
                reduce_sz = float(target_size_f) if target_size_f is not None else pos_size
            except Exception:
                reduce_sz = pos_size
            reduce_sz = float(max(0.0, min(pos_size, reduce_sz)))

            if act == "CLOSE":
                fn = getattr(self.trader, "close_position", None)
                if callable(fn):
                    try:
                        return fn(
                            sym,
                            price,
                            timestamp,
                            reason=str(reason or "Kernel CLOSE"),
                            meta={"reason": str(reason or "").strip() or None},
                        )
                    except TypeError:
                        return fn(sym, price, timestamp, str(reason or "Kernel CLOSE"))
                return
            fn = getattr(self.trader, "reduce_position", None)
            if callable(fn):
                return fn(
                    sym,
                    reduce_sz,
                    price,
                    timestamp,
                    str(reason or "Kernel REDUCE"),
                    confidence=confidence,
                    meta={"reason": str(reason or "").strip() or None},
                )
            return

        # Legacy "OPEN" fallback: if the trader does not support action-aware execute_trade.
        return self.trader.execute_trade(
            sym,
            signal,
            price,
            timestamp,
            confidence,
            atr=atr,
            indicators=indicators,
        )

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
                                "mode": str(getattr(self, "mode", os.getenv("AI_QUANT_MODE", "paper") or "paper")),
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

                # Regime gate: a global binary switch (trend OK vs chop) that blocks entries when OFF.
                try:
                    self._update_regime_gate(mei_alpha_v1=mei_alpha_v1, btc_key_hint=btc_key_hint, btc_df=btc_df)
                except Exception:
                    pass

                # Phase 1: Keep exit logic for open positions.
                for sym_u in sorted((self.trader.positions or {}).keys()):
                    try:
                        cached = self._analysis_cache.get(str(sym_u).upper()) or {}
                        if not isinstance(cached, dict):
                            cached = {}

                        now_series = cached.get("now") if isinstance(cached.get("now"), dict) else {}
                        if not isinstance(now_series, dict):
                            now_series = {}
                        is_exit_boundary = self._exit_reanalyze_due(sym_u)
                        if not is_exit_boundary:
                            continue

                        if self._rest_enabled and hyperliquid_meta is not None:
                            try:
                                now_series["funding_rate"] = float(hyperliquid_meta.get_funding_rate(sym_u) or 0.0)
                            except Exception:
                                now_series["funding_rate"] = 0.0
                        else:
                            now_series["funding_rate"] = 0.0

                        self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)

                        quote = None
                        _exit_iv = self._exit_interval
                        if _exit_iv and _exit_iv != "mid":
                            try:
                                df_exit = self.market.get_candles_df(sym_u, interval=_exit_iv, min_rows=1)
                                if df_exit is not None and not df_exit.empty:
                                    current_price = float(df_exit["Close"].iloc[-1])
                                else:
                                    quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                    current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)
                            except Exception:
                                quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)
                        else:
                            quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                            current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)

                        try:
                            if isinstance(now_series, dict):
                                if quote is not None:
                                    try:
                                        now_series["quote"] = {
                                            "source": str(getattr(quote, "source", "")),
                                            "age_s": float(getattr(quote, "age_s", 0.0) or 0.0),
                                        }
                                    except Exception:
                                        pass
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
                    except SystemExit:
                        raise
                    except Exception:
                        print(f"⚠️ Engine exit logic error: {sym_u}\n{traceback.format_exc()}")
                        continue

                # Phase 2: Execute explicit kernel decisions (OPEN/ADD/CLOSE/REDUCE) ordered by score.
                decision_exec: list[KernelDecision] = []
                try:
                    for dec in self.decision_provider.get_decisions(
                        symbols=watchlist,
                        watchlist=watchlist,
                        open_symbols=open_syms,
                        market=self.market,
                        interval=self.interval,
                        lookback_bars=self.lookback_bars,
                        mode=self.mode,
                        not_ready_symbols=not_ready_set,
                        strategy=self.strategy,
                        now_ms=now_ms(),
                    ):
                        if dec is None:
                            continue
                        sym_u = str(dec.symbol).upper().strip()
                        if not sym_u:
                            continue

                        act = str(dec.action or "").strip().upper()
                        if act not in {"OPEN", "ADD", "CLOSE", "REDUCE"}:
                            continue

                        now_series = dec.now_series if isinstance(dec.now_series, dict) else {}
                        try:
                            now_series = dict(now_series)
                        except Exception:
                            now_series = {}

                        if not isinstance(now_series, dict):
                            now_series = {}
                        if not now_series:
                            now_series = {"Close": None, "is_anomaly": False}

                        self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)

                        # Store the latest kernel output for reuse by downstream checks / audit.
                        self._analysis_cache[sym_u] = {
                            "key": int(dec.entry_key) if dec.entry_key is not None else None,
                            "sig": str(dec.signal),
                            "conf": str(dec.confidence),
                            "now": dict(now_series),
                            "computed_at_s": time.time(),
                            "action": act,
                        }

                        # Open-side decisions are always watchlist-controlled.
                        if act in {"OPEN", "ADD"} and not sym_u in watch_set:
                            continue

                        # Skip decision entries where candle keys are not ready.
                        if act in {"OPEN", "ADD"} and sym_u in not_ready_set:
                            print(f"🕒 skip {sym_u} {act}: candles not ready")
                            continue

                        decision_exec.append(dec)
                except Exception:
                    print(f"⚠️ Engine decision provider error: {traceback.format_exc()}")

                decision_exec.sort(
                    key=lambda item: (
                        -(item.score or 0.0),
                        str(item.symbol),
                    )
                )

                for dec in decision_exec:
                    try:
                        sym_u = str(dec.symbol).upper().strip()
                        act = str(dec.action or "").strip().upper()
                        if act not in {"OPEN", "ADD", "CLOSE", "REDUCE"}:
                            continue

                        signal = str(dec.signal or "").upper().strip()
                        conf = str(dec.confidence or "N/A")
                        entry_key = dec.entry_key
                        open_pos_count = int(dec.open_pos_count or 0)

                        now_series = dec.now_series if isinstance(dec.now_series, dict) else {}
                        try:
                            now_series = dict(now_series)
                        except Exception:
                            now_series = {}
                        if not isinstance(now_series, dict):
                            now_series = {}
                        if not now_series:
                            now_series = {"Close": None, "is_anomaly": False}
                        self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)

                        if self._rest_enabled and hyperliquid_meta is not None:
                            try:
                                now_series["funding_rate"] = float(hyperliquid_meta.get_funding_rate(sym_u) or 0.0)
                            except Exception:
                                now_series["funding_rate"] = 0.0
                        else:
                            now_series["funding_rate"] = 0.0

                        quote = None
                        if act in {"OPEN", "ADD"}:
                            _entry_iv = self._entry_interval
                            if _entry_iv and _entry_iv != self.interval and _entry_iv != "mid":
                                try:
                                    df_entry = self.market.get_candles_df(sym_u, interval=_entry_iv, min_rows=1)
                                    if df_entry is not None and not df_entry.empty:
                                        current_price = float(df_entry["Close"].iloc[-1])
                                    else:
                                        quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                        current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)
                                except Exception:
                                    quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                    current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)
                            else:
                                quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                                current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)
                        else:
                            quote = self.market.get_mid_price(sym_u, max_age_s=10.0, interval=self.interval)
                            current_price = float(quote.price) if quote is not None else float(now_series.get("Close") or 0.0)

                        if act in {"OPEN", "ADD"}:
                            if entry_key is not None:
                                now_ts = now_ms()
                                if self._entry_is_too_late(entry_key=int(entry_key), now_ts_ms=int(now_ts)):
                                    print(
                                        f"🕒 skip {sym_u} {act}: stale candle-close signal "
                                        f"key={int(entry_key)}"
                                    )
                                    continue

                                last_key = self._last_entry_key.get(sym_u)
                                if last_key is not None and int(last_key) == int(entry_key):
                                    if self._entry_retry_on_capacity:
                                        last_open_count = self._last_entry_key_open_pos_count.get(sym_u)
                                        if last_open_count is not None and int(len(self.trader.positions or {})) < int(last_open_count):
                                            self._last_entry_key_open_pos_count[sym_u] = int(open_pos_count)
                                        else:
                                            continue
                                    else:
                                        continue

                            if self._rest_enabled and hyperliquid_meta is not None:
                                try:
                                    now_series["funding_rate"] = float(hyperliquid_meta.get_funding_rate(sym_u) or 0.0)
                                except Exception:
                                    now_series["funding_rate"] = 0.0
                            else:
                                now_series["funding_rate"] = 0.0

                            self._attach_strategy_snapshot(symbol=sym_u, now_series=now_series)
                            if entry_key is not None:
                                self._last_entry_key[sym_u] = int(entry_key)
                                self._last_entry_key_open_pos_count[sym_u] = open_pos_count

                            if isinstance(now_series, dict):
                                try:
                                    if quote is not None:
                                        now_series["quote"] = {
                                            "source": str(getattr(quote, "source", "")),
                                            "age_s": float(getattr(quote, "age_s", 0.0) or 0.0),
                                        }
                                except Exception:
                                    pass

                            self._decision_execute_trade(
                                sym_u,
                                signal,
                                current_price,
                                int(entry_key or now_ms()),
                                conf,
                                atr=float(now_series.get("ATR") or 0.0),
                                indicators=now_series,
                                action=act,
                                target_size=dec.target_size,
                                reason=dec.reason,
                            )

                        elif act in {"CLOSE", "REDUCE"}:
                            self._decision_execute_trade(
                                sym_u,
                                signal,
                                current_price,
                                now_ms(),
                                conf,
                                atr=float(now_series.get("ATR") or 0.0),
                                indicators=now_series,
                                action=act,
                                target_size=dec.target_size,
                                reason=dec.reason,
                            )
                    except SystemExit:
                        raise
                    except Exception:
                        print(f"⚠️ Engine decision execution error: {dec.symbol}\n{traceback.format_exc()}")
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
                    _btc_cfg = mei_alpha_v1.get_strategy_config("BTC") or {}
                    _rc = (_btc_cfg.get("market_regime") or {}) if isinstance(_btc_cfg, dict) else {}
                    _tc = (_btc_cfg.get("trade") or {}) if isinstance(_btc_cfg, dict) else {}
                    try:
                        _size_mult = float(_tc.get("size_multiplier", 1.0))
                    except Exception:
                        _size_mult = 1.0
                    _auto_rev_on = (
                        bool(_rc.get("enable_auto_reverse", False))
                        and self._market_breadth_pct is not None
                        and float(_rc.get("auto_reverse_breadth_low", 20.0)) <= self._market_breadth_pct <= float(_rc.get("auto_reverse_breadth_high", 80.0))
                    )
                    cfg_id = ""
                    try:
                        from .event_logger import current_config_id

                        cfg_id = str(current_config_id() or "")
                    except Exception:
                        cfg_id = ""

                    risk = getattr(self.trader, "risk", None)
                    try:
                        kill_mode = str(getattr(risk, "kill_mode", "off") or "off").strip().lower()
                    except Exception:
                        kill_mode = "off"
                    try:
                        kill_reason = str(getattr(risk, "kill_reason", "") or "").strip() or "none"
                    except Exception:
                        kill_reason = "none"

                    try:
                        strategy_mode = str(os.getenv("AI_QUANT_STRATEGY_MODE", "") or "").strip().lower() or "none"
                    except Exception:
                        strategy_mode = "none"

                    slip_enabled = 0
                    slip_n = 0
                    slip_win = 0
                    slip_thr_bps_s = "0"
                    slip_last_bps_s = "none"
                    slip_median_bps_s = "none"
                    try:
                        fn = getattr(risk, "slippage_guard_stats", None) if risk is not None else None
                        st = fn() if callable(fn) else {}
                        if isinstance(st, dict):
                            slip_enabled = 1 if bool(st.get("enabled")) else 0
                            slip_n = int(st.get("n") or 0)
                            slip_win = int(st.get("window_fills") or 0)
                            thr = st.get("threshold_median_bps")
                            if thr is not None:
                                slip_thr_bps_s = f"{float(thr):.3f}"
                            last = st.get("last_bps")
                            if last is not None:
                                slip_last_bps_s = f"{float(last):.3f}"
                            med = st.get("median_bps")
                            if med is not None:
                                slip_median_bps_s = f"{float(med):.3f}"
                    except Exception:
                        pass
                    print(
                        f"🫀 engine ok. loops={self.stats.loops} errors={self.stats.loop_errors} "
                        f"symbols={len(active_symbols)} open_pos={open_pos} loop={loop_s:.2f}s "
                        f"size_mult={float(_size_mult):g} "
                        f"ws_connected={h.get('connected')} ws_thread_alive={h.get('thread_alive')} "
                        f"ws_restarts={self.stats.ws_restarts} "
                        f"kill={kill_mode} kill_reason={kill_reason} "
                        f"strategy_mode={strategy_mode} "
                        f"regime_gate={'on' if self._regime_gate_on else 'off'} regime_reason={self._regime_gate_reason} "
                        f"slip_enabled={slip_enabled} slip_n={slip_n} slip_win={slip_win} slip_thr_bps={slip_thr_bps_s} "
                        f"slip_last_bps={slip_last_bps_s} slip_median_bps={slip_median_bps_s} "
                        f"signal_on_close={int(self._signal_on_candle_close)} entry_iv={self._entry_interval} exit_iv={self._exit_interval} reanalyze_s={self._reanalyze_interval_s:.0f} exit_reanalyze_s={self._exit_reanalyze_interval_s:.0f} "
                        f"breadth={f'{self._market_breadth_pct:.1f}%' if self._market_breadth_pct is not None else 'n/a'} "
                        f"auto_rev={'ON' if _auto_rev_on else 'OFF'} "
                        f"config_id={cfg_id or 'none'} "
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
