"""Unified daemon entrypoint (paper + live).

Usage:
  AI_QUANT_MODE=paper python -m engine.daemon
  AI_QUANT_MODE=dry_live python -m engine.daemon
  AI_QUANT_MODE=live python -m engine.daemon

What this replaces:
- trader_daemon.py (paper loop)
- trader_daemon_live.py (live loop)

The goal is to keep ONE orchestration loop, and let PaperTrader / LiveTrader focus on:
- strategy decision logic
- execution (paper vs real orders)

YAML config hot-reload is handled by StrategyManager, without reloading mei_alpha_v1 each loop.
"""

from __future__ import annotations

import os
import time
from collections import deque

from .core import UnifiedEngine
from .market_data import MarketDataHub
from .strategy_manager import StrategyManager
from .oms import LiveOms
from .oms_reconciler import LiveOmsReconciler
from .risk import RiskManager


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _mode() -> str:
    return str(os.getenv("AI_QUANT_MODE", "paper") or "paper").strip().lower()

def _default_db_path() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "trading_engine.db"))
    except Exception:
        return "trading_engine.db"


def _db_path() -> str:
    return str(os.getenv("AI_QUANT_DB_PATH", _default_db_path()) or _default_db_path())

def _hl_timeout_s() -> float:
    # Never allow infinite REST hangs in the main loop.
    raw = os.getenv("AI_QUANT_HL_TIMEOUT_S", "10")
    try:
        v = float(raw)
    except Exception:
        v = 10.0
    return float(max(0.5, min(30.0, v)))


def acquire_lock_or_exit(lock_path: str):
    """Prevent multiple daemons from running.

    Uses fcntl when available (Linux). Falls back to a simple pid file otherwise.
    """
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        # Fallback: best-effort pid file.
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
        except Exception:
            pass
        return lock_file

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


class PaperPlugin:
    def __init__(self, trader):
        self.trader = trader
        self._last_funding_check = 0.0
        self._funding_every_s = float(os.getenv("AI_QUANT_PAPER_FUNDING_SECS", "60"))

    def before_iteration(self) -> None:
        # Reset per-loop entry budget (mirrors LivePlugin).
        try:
            import strategy.mei_alpha_v1 as mei_alpha_v1
            cfg = mei_alpha_v1.get_strategy_config("BTC") or {}
            budget = int((cfg.get("trade") or {}).get("max_entry_orders_per_loop", 0))
            self.trader._entry_budget_remaining = budget if budget > 0 else None
        except Exception:
            pass

        # If your paper trader applies funding, do it here on a schedule.
        now = time.time()
        if (now - self._last_funding_check) >= self._funding_every_s:
            self._last_funding_check = now
            fn = getattr(self.trader, "apply_funding_payments", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def after_iteration(self) -> None:
        return


class LivePlugin:
    """Hooks for live mode.

    - periodic state sync (account + open positions)
    - drain WS user events for fills/funding/orderUpdates
    - periodic REST backfill of fills as a safety net
    """

    def __init__(
        self,
        *,
        trader,
        secrets_main_address: str,
        oms: LiveOms | None = None,
        risk: RiskManager | None = None,
        reconciler: LiveOmsReconciler | None = None,
    ):
        self.trader = trader
        self.main_address = secrets_main_address
        self._oms = oms
        self._risk = risk
        self._reconciler = reconciler

        # Attach OMS onto the trader so order submission can record OrderIntents.
        if self._oms is not None:
            try:
                setattr(self.trader, "oms", self._oms)
            except Exception:
                pass

        # Attach RiskManager onto the trader so execution path can enforce guardrails.
        if self._risk is not None:
            try:
                setattr(self.trader, "risk", self._risk)
            except Exception:
                pass

        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        import exchange.ws as hyperliquid_ws
        import live.trader as live_trader

        self._ws = hyperliquid_ws.hl_ws
        self._lt = live_trader
        self._rest_info = Info(constants.MAINNET_API_URL, skip_ws=True, timeout=_hl_timeout_s())

        self._last_state_sync = 0.0
        self._state_sync_s = float(os.getenv("AI_QUANT_LIVE_STATE_SYNC_SECS", "10"))

        self._last_rest_fills_sync = 0.0
        self._rest_sync_s = float(os.getenv("AI_QUANT_LIVE_REST_FILLS_SYNC_SECS", "60"))
        self._last_rest_fills_err_s = 0.0

        # On start, backfill a small window (dedup happens in DB).
        self._last_rest_fills_ms = int(time.time() * 1000) - int(float(os.getenv("AI_QUANT_LIVE_BACKFILL_MS", str(2 * 60 * 60 * 1000))))

        # Buffer WS fills so transient DB errors don't drop fills (WS queues are drained).
        self._pending_ws_fills: list[dict] = []
        self._err_last_s: dict[str, float] = {}
        self._err_log_every_s = float(os.getenv("AI_QUANT_LIVE_ERR_LOG_EVERY_S", "30"))
        try:
            self._max_pending_ws_fills = int(float(os.getenv("AI_QUANT_LIVE_MAX_PENDING_FILLS", "50000")))
        except Exception:
            self._max_pending_ws_fills = 50000

        # Health guard (alerts + optional kill-switch).
        self._health_enabled = _env_bool("AI_QUANT_HEALTH_GUARD", True)
        self._health_alert_cooldown_s = float(os.getenv("AI_QUANT_HEALTH_ALERT_COOLDOWN_S", "300"))
        self._health_last_alert_s: dict[str, float] = {}

        self._fill_ingest_fail_streak = 0
        self._fill_ingest_fail_to_kill = int(float(os.getenv("AI_QUANT_HEALTH_FILL_INGEST_FAILS_TO_KILL", "5")))
        self._fill_ingest_kill = _env_bool("AI_QUANT_HEALTH_FILL_INGEST_KILL", True)
        self._fill_ingest_kill_mode = str(os.getenv("AI_QUANT_HEALTH_FILL_INGEST_KILL_MODE", "close_only") or "close_only").strip().lower()

        self._unmatched_window_s = float(os.getenv("AI_QUANT_HEALTH_UNMATCHED_WINDOW_S", "900"))
        self._unmatched_alert_n = int(float(os.getenv("AI_QUANT_HEALTH_UNMATCHED_ALERT_N", "1")))
        self._unmatched_kill = _env_bool("AI_QUANT_HEALTH_UNMATCHED_KILL", False)
        self._unmatched_kill_mode = str(os.getenv("AI_QUANT_HEALTH_UNMATCHED_KILL_MODE", "close_only") or "close_only").strip().lower()
        self._unmatched_ts: deque[float] = deque()

        self._pending_fills_to_kill = int(float(os.getenv("AI_QUANT_HEALTH_PENDING_FILLS_TO_KILL", "20000")))
        self._pending_fills_kill = _env_bool("AI_QUANT_HEALTH_PENDING_FILLS_KILL", True)
        self._pending_fills_kill_mode = str(os.getenv("AI_QUANT_HEALTH_PENDING_FILLS_KILL_MODE", "close_only") or "close_only").strip().lower()

        # Signal backfill: repair missing `signals` rows from recent `trades` (live DB).
        # This keeps the monitor "SIGNAL" column populated even if signal logging was missed
        # (older versions, transient DB locks, REST-only ingestion).
        self._signal_backfill_enabled = _env_bool("AI_QUANT_SIGNAL_BACKFILL", True)
        try:
            self._signal_backfill_every_s = float(os.getenv("AI_QUANT_SIGNAL_BACKFILL_SECS", "300"))
        except Exception:
            self._signal_backfill_every_s = 300.0
        try:
            self._signal_backfill_lookback_h = float(os.getenv("AI_QUANT_SIGNAL_BACKFILL_LOOKBACK_H", "48"))
        except Exception:
            self._signal_backfill_lookback_h = 48.0
        try:
            self._signal_backfill_max_trades = int(float(os.getenv("AI_QUANT_SIGNAL_BACKFILL_MAX_TRADES", "2000")))
        except Exception:
            self._signal_backfill_max_trades = 2000
        try:
            self._signal_backfill_dedup_s = float(os.getenv("AI_QUANT_SIGNAL_BACKFILL_DEDUP_S", "600"))
        except Exception:
            self._signal_backfill_dedup_s = 600.0
        self._last_signal_backfill_s = 0.0

        # Per-loop entry budget (OPEN/ADD) to prevent long loops caused by burst order submissions.
        # 0 disables (unlimited).
        try:
            self._max_entry_orders_per_loop = int(float(os.getenv("AI_QUANT_LIVE_MAX_ENTRY_ORDERS_PER_LOOP", "6")))
        except Exception:
            self._max_entry_orders_per_loop = 6
        self._max_entry_orders_per_loop = max(0, int(self._max_entry_orders_per_loop))

    def _log_exc(self, key: str, msg: str) -> None:
        now = time.time()
        last = float(self._err_last_s.get(key, 0.0) or 0.0)
        if (now - last) < float(self._err_log_every_s or 0.0):
            return
        self._err_last_s[key] = now
        try:
            import traceback

            print(f"⚠️ {msg}\n{traceback.format_exc()}")
        except Exception:
            try:
                print(f"⚠️ {msg}")
            except Exception:
                pass

    def _health_alert(self, key: str, message: str, *, kill: bool, kill_mode: str) -> None:
        if not self._health_enabled:
            return
        now = time.time()
        last = float(self._health_last_alert_s.get(key, 0.0) or 0.0)
        if (now - last) < float(self._health_alert_cooldown_s or 0.0):
            return
        self._health_last_alert_s[key] = now

        msg = f"🚨 HEALTH[{key}] {message}"
        try:
            print(msg)
        except Exception:
            pass

        try:
            import strategy.mei_alpha_v1 as mei_alpha_v1

            mei_alpha_v1.log_audit_event(
                symbol="SYSTEM",
                event="HEALTH_ALERT",
                level="warn",
                data={"key": str(key), "message": str(message), "kill": bool(kill), "kill_mode": str(kill_mode)},
            )
        except Exception:
            pass

        # Best-effort Discord notification (same channel as live fills).
        try:
            if hasattr(self._lt, "_live_notify_enabled") and callable(getattr(self._lt, "_live_notify_enabled")):
                if self._lt._live_notify_enabled():
                    target = self._lt._live_discord_target()
                    if target:
                        self._lt._send_discord_message(target=target, message=msg)
        except Exception:
            pass

        # Optional kill-switch.
        if kill and self._risk is not None:
            try:
                mode = str(kill_mode or "close_only").strip().lower()
                if mode not in {"close_only", "halt_all"}:
                    mode = "close_only"
                self._risk.kill(mode=mode, reason=f"health:{key}")
            except Exception:
                pass

    def before_iteration(self) -> None:
        now = time.time()

        # Reset per-loop entry budget on the trader (best-effort).
        try:
            if self._max_entry_orders_per_loop > 0:
                setattr(self.trader, "_entry_budget_remaining", int(self._max_entry_orders_per_loop))
            else:
                setattr(self.trader, "_entry_budget_remaining", None)
        except Exception:
            pass

        # Periodic account/position reconciliation.
        if (now - self._last_state_sync) >= self._state_sync_s:
            self._last_state_sync = now

            # Refresh risk state (kill-switch, drawdown, etc).
            try:
                if self._risk is not None:
                    self._risk.refresh(trader=self.trader)
            except Exception:
                self._log_exc("risk_refresh", "RiskManager.refresh failed")

            try:
                self.trader.sync_from_exchange(force=False)
            except Exception:
                self._log_exc("sync_from_exchange", "trader.sync_from_exchange failed")

            # Periodic OMS maintenance (expire stale intents).
            try:
                if self._oms is not None:
                    self._oms.reconcile(trader=self.trader)
            except Exception:
                self._log_exc("oms_reconcile", "OMS reconcile() failed")

        # True OMS reconcile loop (open orders, stale cancels, partial-fill tidy).
        try:
            if self._reconciler is not None:
                self._reconciler.maybe_run(trader=self.trader)
        except Exception:
            self._log_exc("oms_reconciler", "OMS reconciler maybe_run() failed")

        # Drain WS user fills into a local buffer so transient DB errors don't lose fills.
        try:
            fills = self._ws.drain_user_fills(max_items=5000)
            if fills:
                self._pending_ws_fills.extend([f for f in fills if isinstance(f, dict)])
                if self._pending_fills_to_kill > 0 and len(self._pending_ws_fills) >= self._pending_fills_to_kill:
                    self._health_alert(
                        "pending_fills",
                        f"pending_ws_fills={len(self._pending_ws_fills)} exceeds {self._pending_fills_to_kill}",
                        kill=bool(self._pending_fills_kill),
                        kill_mode=str(self._pending_fills_kill_mode or "close_only"),
                    )
                if self._max_pending_ws_fills > 0 and len(self._pending_ws_fills) > self._max_pending_ws_fills:
                    # Keep the newest; REST backfill should be able to recover the dropped ones.
                    drop = len(self._pending_ws_fills) - self._max_pending_ws_fills
                    self._pending_ws_fills = self._pending_ws_fills[drop:]
                    print(
                        f"⚠️ WS fills buffer exceeded {self._max_pending_ws_fills}; "
                        f"dropped {drop} oldest (REST backfill should recover)"
                    )
        except Exception:
            self._log_exc("drain_user_fills", "WS drain_user_fills failed")

        if self._pending_ws_fills:
            try:
                pending = list(self._pending_ws_fills)
                if self._oms is not None:
                    inserted = self._oms.process_user_fills(trader=self.trader, fills=pending)
                else:
                    inserted = self._lt.process_user_fills(self.trader, pending)

                if inserted:
                    try:
                        self.trader.sync_from_exchange(force=True)
                    except Exception:
                        self._log_exc("sync_after_fills", "sync_from_exchange(force=True) failed after fills")

                # Health guard: unmatched fills and ingest failures.
                self._fill_ingest_fail_streak = 0
                if self._oms is not None:
                    try:
                        st = getattr(self._oms, "last_ingest_stats", None) or {}
                        unmatched_new = int(st.get("unmatched_new") or 0)
                        if unmatched_new > 0:
                            for _ in range(unmatched_new):
                                self._unmatched_ts.append(now)

                        cutoff = float(now) - float(self._unmatched_window_s or 0.0)
                        while self._unmatched_ts and float(self._unmatched_ts[0]) < cutoff:
                            self._unmatched_ts.popleft()

                        if self._unmatched_alert_n > 0 and len(self._unmatched_ts) >= self._unmatched_alert_n:
                            samples = st.get("unmatched_samples")
                            extra = ""
                            try:
                                if isinstance(samples, list) and samples:
                                    extra = f" samples={samples[:3]}"
                            except Exception:
                                extra = ""
                            self._health_alert(
                                "unmatched_fills",
                                f"unmatched_new={unmatched_new} window_s={self._unmatched_window_s} count_in_window={len(self._unmatched_ts)}{extra}",
                                kill=bool(self._unmatched_kill),
                                kill_mode=str(self._unmatched_kill_mode or "close_only"),
                            )
                    except Exception:
                        pass

                # Only drop buffered fills after successful processing.
                self._pending_ws_fills.clear()
            except Exception:
                self._fill_ingest_fail_streak += 1
                self._log_exc(
                    "process_user_fills",
                    f"process_user_fills failed; keeping {len(self._pending_ws_fills)} fills for retry",
                )
                if self._fill_ingest_fail_to_kill > 0 and self._fill_ingest_fail_streak >= self._fill_ingest_fail_to_kill:
                    self._health_alert(
                        "fill_ingest_failed",
                        f"fail_streak={self._fill_ingest_fail_streak} pending_ws_fills={len(self._pending_ws_fills)}",
                        kill=bool(self._fill_ingest_kill),
                        kill_mode=str(self._fill_ingest_kill_mode or "close_only"),
                    )

        try:
            fundings = self._ws.drain_user_fundings(max_items=2000)
            if fundings:
                self._lt.process_user_fundings(self.trader, fundings)
        except Exception:
            self._log_exc("user_fundings", "WS drain/process userFundings failed")

        try:
            ledger = self._ws.drain_user_ledger_updates(max_items=2000)
            if ledger:
                self._lt.process_ws_events("userNonFundingLedgerUpdates", ledger)
        except Exception:
            self._log_exc("user_ledger", "WS drain/process userNonFundingLedgerUpdates failed")

        try:
            order_updates = self._ws.drain_order_updates(max_items=2000)
            if order_updates:
                self._lt.process_ws_events("orderUpdates", order_updates)
        except Exception:
            self._log_exc("order_updates", "WS drain/process orderUpdates failed")

        # REST fill backfill safety. Useful if WS drops for a while.
        if self._rest_sync_s > 0 and (now - self._last_rest_fills_sync) >= self._rest_sync_s:
            self._last_rest_fills_sync = now
            now_ms = int(time.time() * 1000)
            start_ms = max(0, int(self._last_rest_fills_ms) - (5 * 60 * 1000))  # overlap
            try:
                rest_fills = self._rest_info.user_fills_by_time(
                    self.main_address,
                    start_ms,
                    now_ms,
                    aggregate_by_time=False,
                ) or []
                if rest_fills:
                    if self._oms is not None:
                        inserted = self._oms.process_user_fills(trader=self.trader, fills=rest_fills)
                    else:
                        inserted = self._lt.process_user_fills(self.trader, rest_fills)
                    if inserted:
                        try:
                            self.trader.sync_from_exchange(force=True)
                        except Exception:
                            self._log_exc("sync_after_rest_fills", "sync_from_exchange(force=True) failed after REST fills")
                # Advance the cursor only after a successful REST call.
                self._last_rest_fills_ms = now_ms
            except Exception:
                # Do not advance the cursor on failures; we'll retry on the next cycle.
                if (now - self._last_rest_fills_err_s) >= 30.0:
                    self._last_rest_fills_err_s = now
                    try:
                        import traceback

                        print(f"⚠️ REST fills backfill failed\n{traceback.format_exc()}")
                    except Exception:
                        pass

        # Repair missing signals from recent trades (best-effort, throttled).
        if (
            self._signal_backfill_enabled
            and self._oms is not None
            and (self._signal_backfill_every_s > 0)
            and ((now - self._last_signal_backfill_s) >= self._signal_backfill_every_s)
        ):
            self._last_signal_backfill_s = now
            try:
                n = self._oms.backfill_signals_from_trades(
                    lookback_h=float(self._signal_backfill_lookback_h),
                    max_trades=int(self._signal_backfill_max_trades),
                    dedup_s=float(self._signal_backfill_dedup_s),
                )
                if n:
                    print(f"🧩 backfilled {int(n)} missing signals from recent trades")
            except Exception:
                self._log_exc("signal_backfill", "signal backfill failed")

    def after_iteration(self) -> None:
        return


def main() -> None:
    mode = _mode()
    try:
        from .sqlite_logger import install_sqlite_stdio_logger

        install_sqlite_stdio_logger(db_path=_db_path(), mode=mode)
    except Exception:
        pass

    import strategy.mei_alpha_v1 as mei_alpha_v1

    default_lock_name = "ai_quant_paper.lock" if mode == "paper" else "ai_quant_live.lock"
    lock_path = os.getenv(
        "AI_QUANT_LOCK_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", default_lock_name),
    )
    _lock = acquire_lock_or_exit(lock_path)

    strategy = StrategyManager.get()
    market_db_path = os.getenv("AI_QUANT_MARKET_DB_PATH", mei_alpha_v1.DB_PATH)
    market = MarketDataHub(
        db_path=market_db_path,
        stale_mid_s=float(os.getenv("AI_QUANT_WS_STALE_MIDS_S", "60")),
        stale_bbo_s=float(os.getenv("AI_QUANT_WS_STALE_BBO_S", "120")),
        stale_candle_s=float(os.getenv("AI_QUANT_WS_STALE_CANDLES_S", str(2 * 60 * 60))),
        db_timeout_s=float(os.getenv("AI_QUANT_DB_TIMEOUT_S", "30")),
    )

    if mode == "paper":
        # PaperTrader loads its last state from the DB. To reset, delete the DB or use a fresh DB path.
        trader = mei_alpha_v1.PaperTrader()
        plugin = PaperPlugin(trader)

    elif mode in {"live", "dry_live"}:
        import live.trader as live_trader

        secrets_path = os.getenv(
            "AI_QUANT_SECRETS_PATH",
            os.path.join(os.path.dirname(__file__), "..", "secrets.json"),
        )
        secrets = live_trader.load_live_secrets(secrets_path)

        executor = live_trader.HyperliquidLiveExecutor(
            secret_key=secrets.secret_key,
            main_address=secrets.main_address,
            timeout_s=_hl_timeout_s(),
        )
        trader = live_trader.LiveTrader(executor=executor)
        # Used by the engine to subscribe to WS user channels.
        trader.secrets = secrets

        try:
            live_trader._ensure_live_tables()
        except Exception:
            pass

        oms = LiveOms(db_path=mei_alpha_v1.DB_PATH)
        risk = RiskManager()
        reconciler = LiveOmsReconciler(oms=oms, executor=executor, main_address=secrets.main_address, risk=risk)

        # Dry-live should be read-only: no cancels.
        if mode == "dry_live":
            try:
                reconciler.cancel_stale = False
            except Exception:
                pass

        plugin = LivePlugin(
            trader=trader,
            secrets_main_address=secrets.main_address,
            oms=oms,
            risk=risk,
            reconciler=reconciler,
        )

    else:
        raise SystemExit(f"Unknown AI_QUANT_MODE={mode}")

    interval = str(os.getenv("AI_QUANT_INTERVAL", mei_alpha_v1.INTERVAL) or mei_alpha_v1.INTERVAL).strip()
    try:
        lookback_bars = int(os.getenv("AI_QUANT_LOOKBACK_BARS", str(mei_alpha_v1.LOOKBACK_HOURS)))
    except Exception:
        lookback_bars = int(mei_alpha_v1.LOOKBACK_HOURS)
    lookback_bars = int(max(50, min(10_000, lookback_bars)))

    engine = UnifiedEngine(
        trader=trader,
        strategy=strategy,
        market=market,
        interval=interval,
        lookback_bars=lookback_bars,
        mode_plugin=plugin,
    )
    print(f"🚀 Unified engine started. mode={mode}")
    engine.run_forever()


if __name__ == "__main__":
    main()
