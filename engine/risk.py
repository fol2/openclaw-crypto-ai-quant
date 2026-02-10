from __future__ import annotations

import datetime
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else str(raw)


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


class TokenBucket:
    """Simple token bucket rate limiter.

    - capacity: maximum tokens
    - refill_per_s: tokens added per second
    """

    def __init__(self, *, capacity: float, refill_per_s: float):
        self.capacity = float(max(0.0, capacity))
        self.refill_per_s = float(max(0.0, refill_per_s))
        self._tokens = float(self.capacity)
        self._last_s = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_s)
        self._last_s = now
        if dt <= 0 or self.refill_per_s <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + (dt * self.refill_per_s))

    def allow(self, *, cost: float = 1.0) -> bool:
        c = float(max(0.0, cost))
        self._refill()
        if self._tokens >= c:
            self._tokens -= c
            return True
        return False


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    kill_mode: str | None = None


class RiskManager:
    """Production guardrails: kill-switch + rate limiting + basic circuit breakers.

    Design goals:
    - Never throw in trading path.
    - Default behaviour is conservative but not annoying.
    - Kill-switch is close-only by default: it blocks OPEN/ADD but allows CLOSE/REDUCE.
    """

    def __init__(self):
        # Kill switch
        self._kill_mode: str = "off"  # off | close_only | halt_all
        self._kill_reason: str | None = None
        self._kill_since_s: float | None = None

        self._kill_file = _env_str("AI_QUANT_KILL_SWITCH_FILE", "")
        self._kill_env = _env_str("AI_QUANT_KILL_SWITCH", "").strip()
        self._kill_mode_env = _env_str("AI_QUANT_KILL_SWITCH_MODE", "close_only").strip().lower()

        # Rate limits (entry orders)
        entry_per_min = float(max(1.0, _env_float("AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN", 30.0)))
        self._entry_bucket = TokenBucket(capacity=entry_per_min, refill_per_s=entry_per_min / 60.0)

        entry_sym_per_min = float(max(1.0, _env_float("AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN_PER_SYMBOL", 6.0)))
        self._entry_symbol_per_min = float(entry_sym_per_min)
        self._entry_symbol_events: dict[str, deque[float]] = defaultdict(deque)

        # Rate limits (exit orders)
        exit_per_min = float(max(1.0, _env_float("AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN", 120.0)))
        self._exit_bucket = TokenBucket(capacity=exit_per_min, refill_per_s=exit_per_min / 60.0)

        # Cancels
        cancel_per_min = float(max(1.0, _env_float("AI_QUANT_RISK_MAX_CANCELS_PER_MIN", 120.0)))
        self._cancel_bucket = TokenBucket(capacity=cancel_per_min, refill_per_s=cancel_per_min / 60.0)

        # Notional throttle (entry only by default)
        self._notional_window_s = float(max(10.0, _env_float("AI_QUANT_RISK_NOTIONAL_WINDOW_S", 60.0)))
        self._max_notional_per_window = float(max(0.0, _env_float("AI_QUANT_RISK_MAX_NOTIONAL_PER_WINDOW_USD", 0.0)))
        self._notional_events: deque[tuple[float, float]] = deque()  # (ts_s, notional)

        # Min spacing between orders (global)
        # Default is disabled (0) to avoid surprising behaviour changes; enable explicitly via env.
        raw_gap = os.getenv("AI_QUANT_RISK_MIN_ORDER_GAP_MS")
        if raw_gap is None:
            self._min_order_gap_ms = 0
        else:
            try:
                self._min_order_gap_ms = int(max(0.0, float(str(raw_gap).strip())))
            except Exception:
                self._min_order_gap_ms = 0
        self._last_order_ts_ms: int | None = None

        # Drawdown kill-switch
        # NOTE: This value can be sourced from YAML (preferred) or env. It is refreshed periodically
        # in refresh() so operators can tune risk controls without code changes.
        self._max_drawdown_pct = float(max(0.0, _env_float("AI_QUANT_RISK_MAX_DRAWDOWN_PCT", 0.0)))
        self._equity_peak: float | None = None
        self._drawdown_reduce_policy: str = "none"  # none | close_all

        # Daily realised PnL kill-switch (UTC day).
        self._max_daily_loss_usd = float(max(0.0, _env_float("AI_QUANT_RISK_MAX_DAILY_LOSS_USD", 0.0)))
        self._daily_utc_day: str | None = None
        self._daily_realised_pnl_usd: float = 0.0
        self._daily_fees_usd: float = 0.0

        # Slippage anomaly guard (entry fills). Uses median of recent fills vs mid/BBO.
        self._slippage_guard_enabled = _env_bool("AI_QUANT_RISK_SLIPPAGE_GUARD_ENABLED", False)
        self._slippage_guard_window_fills = int(max(1, _env_int("AI_QUANT_RISK_SLIPPAGE_GUARD_WINDOW_FILLS", 20)))
        self._slippage_guard_max_median_bps = float(
            max(0.0, _env_float("AI_QUANT_RISK_SLIPPAGE_GUARD_MAX_MEDIAN_BPS", 0.0))
        )
        self._slippage_bps: deque[float] = deque()

        # Diagnostics
        self._blocked_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kill(self, *, mode: str = "close_only", reason: str = "manual") -> None:
        m = str(mode or "close_only").strip().lower()
        if m not in {"close_only", "halt_all"}:
            m = "close_only"
        r = str(reason or "manual")
        # Idempotent: avoid resetting kill_since_s on every refresh loop when the reason is stable
        # (e.g. kill file present, drawdown still breached).
        if self._kill_mode == m and self._kill_reason == r and self._kill_since_s is not None:
            return
        self._kill_mode = m
        self._kill_reason = r
        self._kill_since_s = time.time()

    def clear_kill(self, *, reason: str = "manual_clear") -> None:
        # Reset any cached state so we do not immediately re-trigger on stale peaks once the
        # operator explicitly resumes.
        self._equity_peak = None
        self._daily_utc_day = None
        self._daily_realised_pnl_usd = 0.0
        self._daily_fees_usd = 0.0
        self._slippage_bps.clear()
        self._kill_mode = "off"
        self._kill_reason = str(reason or "manual_clear")
        self._kill_since_s = None

    @property
    def kill_mode(self) -> str:
        return self._kill_mode

    @property
    def kill_reason(self) -> str | None:
        return self._kill_reason

    @property
    def kill_since_s(self) -> float | None:
        return self._kill_since_s

    @property
    def drawdown_reduce_policy(self) -> str:
        return self._drawdown_reduce_policy

    def refresh(self, *, trader: Any | None = None) -> None:
        """Call periodically (for example every 5-10s) to re-evaluate kill conditions."""
        try:
            self._refresh_risk_config()
        except Exception:
            pass

        try:
            self._refresh_manual_kill()
        except Exception:
            pass

        try:
            self._refresh_drawdown(trader)
        except Exception:
            pass

    def note_fill(
        self,
        *,
        ts_ms: int,
        symbol: str,
        action: str,
        pnl_usd: float,
        fee_usd: float,
        fill_price: float | None = None,
        side: str | None = None,
        ref_mid: float | None = None,
        ref_bid: float | None = None,
        ref_ask: float | None = None,
    ) -> None:
        """Update rolling risk state based on a newly-ingested live fill.

        This is designed to be called only for NEW fills (after DB dedupe), and must never throw.
        """
        try:
            self._refresh_daily_loss(ts_ms=ts_ms, symbol=symbol, action=action, pnl_usd=pnl_usd, fee_usd=fee_usd)
        except Exception:
            return
        try:
            self._refresh_slippage_guard(
                ts_ms=ts_ms,
                symbol=symbol,
                action=action,
                fill_price=fill_price,
                side=side,
                ref_mid=ref_mid,
                ref_bid=ref_bid,
                ref_ask=ref_ask,
            )
        except Exception:
            return

    def allow_order(
        self,
        *,
        symbol: str,
        action: str,
        side: str,
        notional_usd: float,
        leverage: float | None,
        intent_id: str | None = None,
        reduce_risk: bool = False,
    ) -> RiskDecision:
        """Decide whether an order is allowed.

        reduce_risk=True is used for CLOSE/REDUCE. It bypasses some entry-only throttles.
        """
        sym = str(symbol or "").strip().upper()
        ac = str(action or "").strip().upper()

        # Kill switch behaviour.
        if self._kill_mode == "halt_all":
            return self._block("halt_all", sym=sym, action=ac, intent_id=intent_id)
        if self._kill_mode == "close_only" and (not reduce_risk) and ac in {"OPEN", "ADD"}:
            return self._block("close_only", sym=sym, action=ac, intent_id=intent_id)

        # Order spacing.
        if self._min_order_gap_ms > 0:
            now_ms = int(time.time() * 1000)
            last = self._last_order_ts_ms
            if last is not None and (now_ms - int(last)) < int(self._min_order_gap_ms):
                return self._block("min_order_gap", sym=sym, action=ac, intent_id=intent_id)

        # Entry throttles.
        if (not reduce_risk) and ac in {"OPEN", "ADD"}:
            if not self._entry_bucket.allow(cost=1.0):
                return self._block("entry_rate", sym=sym, action=ac, intent_id=intent_id)
            if not self._allow_symbol_event(sym=sym, is_entry=True):
                return self._block("entry_rate_symbol", sym=sym, action=ac, intent_id=intent_id)
            if self._max_notional_per_window > 0:
                if not self._allow_notional(float(notional_usd or 0.0)):
                    return self._block("entry_notional_rate", sym=sym, action=ac, intent_id=intent_id)

        # Exit throttles (still limited, but higher by default).
        if reduce_risk or ac in {"CLOSE", "REDUCE"}:
            if not self._exit_bucket.allow(cost=1.0):
                return self._block("exit_rate", sym=sym, action=ac, intent_id=intent_id)

        return RiskDecision(allowed=True, reason="ok", kill_mode=self._kill_mode if self._kill_mode != "off" else None)

    def note_order_sent(
        self,
        *,
        symbol: str,
        action: str,
        notional_usd: float,
        reduce_risk: bool = False,
    ) -> None:
        """Record that we actually sent an order (call after a successful submit)."""
        try:
            self._last_order_ts_ms = int(time.time() * 1000)
        except Exception:
            self._last_order_ts_ms = None

        sym = str(symbol or "").strip().upper()
        ac = str(action or "").strip().upper()

        # Track per-symbol event counts for entry spam protection.
        if (not reduce_risk) and ac in {"OPEN", "ADD"}:
            now = time.time()
            dq = self._entry_symbol_events[sym]
            dq.append(now)
            self._prune_times(dq, window_s=60.0, now_s=now)

        # Track notional.
        if (not reduce_risk) and ac in {"OPEN", "ADD"} and self._max_notional_per_window > 0:
            now = time.time()
            self._notional_events.append((now, float(abs(notional_usd or 0.0))))
            self._prune_notional(now_s=now)

    def allow_cancel(self, *, symbol: str, exchange_order_id: str | None = None) -> RiskDecision:
        sym = str(symbol or "").strip().upper()
        if not self._cancel_bucket.allow(cost=1.0):
            return self._block("cancel_rate", sym=sym, action="CANCEL", intent_id=exchange_order_id)
        return RiskDecision(allowed=True, reason="ok", kill_mode=self._kill_mode if self._kill_mode != "off" else None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_risk_config(self) -> None:
        """Pull risk controls from YAML (preferred) with env var overrides.

        This is best-effort and must never throw.
        """
        cfg: dict[str, Any] = {}
        try:
            from .strategy_manager import StrategyManager

            cfg = StrategyManager.get().get_config("__GLOBAL__") or {}
        except Exception:
            cfg = {}

        risk_cfg = cfg.get("risk") if isinstance(cfg, dict) else None
        risk_cfg = risk_cfg if isinstance(risk_cfg, dict) else {}

        # Drawdown threshold: YAML risk.max_drawdown_pct, overridden by env when set.
        try:
            dd = risk_cfg.get("max_drawdown_pct")
            dd_pct = float(dd) if dd is not None else None
        except Exception:
            dd_pct = None
        env_dd = os.getenv("AI_QUANT_RISK_MAX_DRAWDOWN_PCT")
        if env_dd is not None and str(env_dd).strip() != "":
            try:
                dd_pct = float(str(env_dd).strip())
            except Exception:
                pass
        if dd_pct is None:
            dd_pct = float(self._max_drawdown_pct or 0.0)
        self._max_drawdown_pct = float(max(0.0, dd_pct))

        # Optional: reduce-risk policy on drawdown kill.
        pol = risk_cfg.get("drawdown_reduce_policy")
        env_pol = os.getenv("AI_QUANT_RISK_DRAWDOWN_REDUCE_POLICY")
        if env_pol is not None and str(env_pol).strip() != "":
            pol = env_pol
        pol_s = str(pol or "none").strip().lower()
        if pol_s not in {"none", "close_all"}:
            pol_s = "none"
        self._drawdown_reduce_policy = pol_s

        # Daily loss threshold (USD): YAML risk.max_daily_loss_usd, overridden by env when set.
        try:
            dl = risk_cfg.get("max_daily_loss_usd")
            dl_usd = float(dl) if dl is not None else None
        except Exception:
            dl_usd = None
        env_dl = os.getenv("AI_QUANT_RISK_MAX_DAILY_LOSS_USD")
        if env_dl is not None and str(env_dl).strip() != "":
            try:
                dl_usd = float(str(env_dl).strip())
            except Exception:
                pass
        if dl_usd is None:
            dl_usd = float(self._max_daily_loss_usd or 0.0)
        self._max_daily_loss_usd = float(max(0.0, dl_usd))

        # Slippage anomaly guard.
        sg = risk_cfg.get("slippage_guard")
        sg = sg if isinstance(sg, dict) else {}

        enabled = sg.get("enabled")
        env_enabled = os.getenv("AI_QUANT_RISK_SLIPPAGE_GUARD_ENABLED")
        if env_enabled is not None and str(env_enabled).strip() != "":
            enabled = env_enabled
        try:
            self._slippage_guard_enabled = (
                bool(enabled)
                if not isinstance(enabled, str)
                else str(enabled).strip().lower() in {"1", "true", "yes", "y", "on"}
            )
        except Exception:
            self._slippage_guard_enabled = False

        window = sg.get("window_fills")
        env_window = os.getenv("AI_QUANT_RISK_SLIPPAGE_GUARD_WINDOW_FILLS")
        if env_window is not None and str(env_window).strip() != "":
            window = env_window
        try:
            self._slippage_guard_window_fills = int(max(1, int(float(window or 20))))
        except Exception:
            self._slippage_guard_window_fills = 20

        max_median = sg.get("max_median_bps")
        env_max_median = os.getenv("AI_QUANT_RISK_SLIPPAGE_GUARD_MAX_MEDIAN_BPS")
        if env_max_median is not None and str(env_max_median).strip() != "":
            max_median = env_max_median
        try:
            self._slippage_guard_max_median_bps = float(max(0.0, float(max_median or 0.0)))
        except Exception:
            self._slippage_guard_max_median_bps = 0.0

    def _refresh_manual_kill(self) -> None:
        # Env kill
        raw = _env_str("AI_QUANT_KILL_SWITCH", "").strip()
        if raw:
            v = raw.strip().lower()
            if v in {"clear", "resume", "off", "unpause"}:
                self.clear_kill(reason="env_clear")
                return
            if v in {"1", "true", "yes", "y", "on", "close_only", "close"}:
                mode = "close_only"
                try:
                    if self._kill_mode_env in {"close_only", "halt_all"}:
                        mode = self._kill_mode_env
                except Exception:
                    mode = "close_only"
                self.kill(mode=mode, reason="env")
            elif v in {"halt", "halt_all", "2", "stop", "full"}:
                self.kill(mode="halt_all", reason="env")
        else:
            # If env cleared, do not auto-clear: only clear explicitly.
            pass

        # File kill
        kill_file = _env_str("AI_QUANT_KILL_SWITCH_FILE", "").strip() or self._kill_file
        if kill_file:
            try:
                if os.path.exists(kill_file):
                    mode = "close_only"
                    try:
                        with open(kill_file, "r", encoding="utf-8") as f:
                            txt = (f.read() or "").strip().lower()
                        if any(tok in txt for tok in ("clear", "resume", "off", "unpause")):
                            self.clear_kill(reason=f"file_clear:{kill_file}")
                            return
                        if "halt" in txt or "full" in txt:
                            mode = "halt_all"
                        elif "close" in txt:
                            mode = "close_only"
                    except Exception:
                        pass
                    self.kill(mode=mode, reason=f"file:{kill_file}")
            except Exception:
                pass

    def _refresh_drawdown(self, trader: Any | None) -> None:
        if self._max_drawdown_pct <= 0:
            return

        equity = None
        # Best-effort extract equity from trader.
        try:
            if trader is not None:
                fn = getattr(trader, "get_live_balance", None)
                if callable(fn):
                    equity = float(fn() or 0.0)
        except Exception:
            equity = None

        try:
            if equity is None and trader is not None:
                equity = float(getattr(trader, "_account_value_usd", 0.0) or 0.0)
        except Exception:
            equity = None

        if equity is None or equity <= 0:
            return

        if self._equity_peak is None or equity > float(self._equity_peak):
            self._equity_peak = float(equity)
            return

        peak = float(self._equity_peak or 0.0)
        if peak <= 0:
            return
        dd = max(0.0, (peak - equity) / peak) * 100.0
        if dd >= float(self._max_drawdown_pct):
            # Close-only kill.
            self.kill(mode="close_only", reason="drawdown")
            self._audit(
                symbol="SYSTEM",
                event="RISK_KILL_DRAWDOWN",
                level="warn",
                data={
                    "drawdown_pct": float(dd),
                    "threshold_pct": float(self._max_drawdown_pct),
                    "equity": float(equity),
                    "peak": float(peak),
                },
            )

    def _refresh_daily_loss(self, *, ts_ms: int, symbol: str, action: str, pnl_usd: float, fee_usd: float) -> None:
        if self._max_daily_loss_usd <= 0:
            return

        ac = str(action or "").strip().upper()
        if ac not in {"CLOSE", "REDUCE"}:
            return

        t_ms = int(ts_ms or 0)
        if t_ms <= 0:
            return

        day = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).date().isoformat()
        if not day:
            return

        if self._daily_utc_day != day:
            # Reset at UTC day boundary.
            self._daily_utc_day = day
            self._daily_realised_pnl_usd = 0.0
            self._daily_fees_usd = 0.0

        try:
            self._daily_realised_pnl_usd += float(pnl_usd or 0.0)
        except Exception:
            pass
        try:
            self._daily_fees_usd += float(fee_usd or 0.0)
        except Exception:
            pass

        net = float(self._daily_realised_pnl_usd) - float(self._daily_fees_usd)
        if net <= -float(self._max_daily_loss_usd):
            # Kill reason must be stable within the day so kill_since_s remains useful.
            self.kill(mode="close_only", reason=f"daily_loss:{day}")
            self._audit(
                symbol=str(symbol or "SYSTEM").strip().upper() or "SYSTEM",
                event="RISK_KILL_DAILY_LOSS",
                level="warn",
                data={
                    "utc_day": str(day),
                    "realised_pnl_usd": float(self._daily_realised_pnl_usd),
                    "fees_usd": float(self._daily_fees_usd),
                    "net_pnl_usd": float(net),
                    "threshold_usd": float(self._max_daily_loss_usd),
                },
            )

    def _refresh_slippage_guard(
        self,
        *,
        ts_ms: int,
        symbol: str,
        action: str,
        fill_price: float | None,
        side: str | None,
        ref_mid: float | None,
        ref_bid: float | None,
        ref_ask: float | None,
    ) -> None:
        if (not self._slippage_guard_enabled) or self._slippage_guard_max_median_bps <= 0:
            return

        ac = str(action or "").strip().upper()
        if ac not in {"OPEN", "ADD"}:
            return

        try:
            px = float(fill_price) if fill_price is not None else 0.0
        except Exception:
            px = 0.0
        if px <= 0:
            return

        sd = str(side or "").strip().upper()
        if sd not in {"BUY", "SELL"}:
            return

        ref_kind = None
        ref_px = None
        if sd == "BUY":
            if ref_ask is not None and float(ref_ask) > 0:
                ref_kind = "ask"
                ref_px = float(ref_ask)
            elif ref_mid is not None and float(ref_mid) > 0:
                ref_kind = "mid"
                ref_px = float(ref_mid)
        else:
            if ref_bid is not None and float(ref_bid) > 0:
                ref_kind = "bid"
                ref_px = float(ref_bid)
            elif ref_mid is not None and float(ref_mid) > 0:
                ref_kind = "mid"
                ref_px = float(ref_mid)

        if ref_px is None or ref_px <= 0:
            return

        if sd == "BUY":
            slip_bps = ((px - ref_px) / ref_px) * 10000.0
        else:
            slip_bps = ((ref_px - px) / ref_px) * 10000.0
        slip_bps = float(max(0.0, slip_bps))

        self._slippage_bps.append(slip_bps)
        while len(self._slippage_bps) > int(self._slippage_guard_window_fills):
            try:
                self._slippage_bps.popleft()
            except Exception:
                break

        win = int(self._slippage_guard_window_fills)
        if len(self._slippage_bps) < win:
            return

        vals = sorted(float(x) for x in self._slippage_bps)
        if not vals:
            return
        mid_i = len(vals) // 2
        if len(vals) % 2 == 1:
            median = float(vals[mid_i])
        else:
            median = float((vals[mid_i - 1] + vals[mid_i]) / 2.0)

        if median <= float(self._slippage_guard_max_median_bps):
            return

        # Avoid spamming the same kill event.
        if self._kill_mode == "close_only" and self._kill_reason == "slippage_guard" and self._kill_since_s is not None:
            return

        self.kill(mode="close_only", reason="slippage_guard")
        self._audit(
            symbol=str(symbol or "SYSTEM").strip().upper() or "SYSTEM",
            event="RISK_KILL_SLIPPAGE",
            level="warn",
            data={
                "slippage_bps_last": float(slip_bps),
                "slippage_ref_kind": str(ref_kind or ""),
                "slippage_ref_px": float(ref_px),
                "slippage_window_fills": int(win),
                "slippage_median_bps": float(median),
                "threshold_median_bps": float(self._slippage_guard_max_median_bps),
            },
        )

    def _allow_symbol_event(self, *, sym: str, is_entry: bool) -> bool:
        if not sym:
            return True
        if not is_entry:
            return True
        now = time.time()
        dq = self._entry_symbol_events[sym]
        self._prune_times(dq, window_s=60.0, now_s=now)
        return len(dq) < int(max(1.0, float(self._entry_symbol_per_min)))

    def _allow_notional(self, add_notional: float) -> bool:
        now = time.time()
        self._prune_notional(now_s=now)
        total = 0.0
        for _, n in self._notional_events:
            total += float(n)
        if (total + float(abs(add_notional or 0.0))) > float(self._max_notional_per_window):
            return False
        return True

    def _prune_times(self, dq: deque[float], *, window_s: float, now_s: float) -> None:
        cutoff = float(now_s) - float(window_s)
        while dq and float(dq[0]) < cutoff:
            dq.popleft()

    def _prune_notional(self, *, now_s: float) -> None:
        cutoff = float(now_s) - float(self._notional_window_s)
        while self._notional_events and float(self._notional_events[0][0]) < cutoff:
            self._notional_events.popleft()

    def _block(self, code: str, *, sym: str, action: str, intent_id: str | None) -> RiskDecision:
        self._blocked_counts[str(code)] += 1
        reason = str(code)
        if code in {"close_only", "halt_all"}:
            # Include the root reason.
            reason = f"{code}:{self._kill_reason or 'manual'}"
        self._audit(
            symbol=sym or "SYSTEM",
            event="RISK_BLOCK",
            level="warn",
            data={
                "code": str(code),
                "action": str(action),
                "intent_id": intent_id,
                "kill_mode": self._kill_mode,
                "kill_reason": self._kill_reason,
            },
        )
        return RiskDecision(allowed=False, reason=reason, kill_mode=self._kill_mode if self._kill_mode != "off" else None)

    def _audit(self, *, symbol: str, event: str, level: str, data: dict[str, Any] | None) -> None:
        try:
            import strategy.mei_alpha_v1 as mei_alpha_v1

            mei_alpha_v1.log_audit_event(symbol=symbol, event=event, level=level, data=data)
        except Exception:
            # Never block trading on audit logging.
            return
