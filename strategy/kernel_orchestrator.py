"""AQC-823: Kernel orchestrator -- Python signal generation cutover.

Replaces the Python ``analyze()`` / ``check_exit_conditions()`` signal
generation paths with a thin orchestrator that feeds data to the Rust
decision kernel and routes resulting ``OrderIntents`` to the broker
adapter.

Classes
-------
KernelOrchestrator
    Feed candles/prices to the kernel, execute intents, log decisions.

Data classes
------------
KernelDecision
    Parsed response from a single kernel step.

Enum
----
LegacyMode
    Kernel operation mode (KERNEL_ONLY, SHADOW).

Module helpers
--------------
build_evaluate_event(snap, gate_result, ema_slow_slope_pct, symbol, price)
    Construct a MarketEvent JSON for ``signal="evaluate"``.
build_price_update_event(symbol, price)
    Construct a MarketEvent JSON for ``signal="price_update"``.
get_legacy_mode(config)
    Read cutover mode from env / config.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy bt_runtime import
# ---------------------------------------------------------------------------

try:
    import bt_runtime as _bt_runtime

    _BT_RUNTIME_AVAILABLE = True
except ImportError:
    _bt_runtime = None  # type: ignore[assignment]
    _BT_RUNTIME_AVAILABLE = False

# ---------------------------------------------------------------------------
# Lazy imports from sibling strategy modules — these are kept lazy so that
# the module can be imported even when heavy dependencies (pandas, ta) are
# absent (e.g. in lightweight test environments).
# ---------------------------------------------------------------------------


def _import_mei_helpers():
    """Import indicator/gate/entry helpers from ``mei_alpha_v1``."""
    from strategy.mei_alpha_v1 import (
        build_entry_params,
        build_gate_result,
        build_indicator_snapshot,
        compute_ema_slow_slope,
    )
    return build_indicator_snapshot, build_gate_result, build_entry_params, compute_ema_slow_slope


def _import_extract_action():
    from strategy.parity_test import extract_action
    return extract_action


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Kernel schema version — must match ``KERNEL_SCHEMA_VERSION`` in Rust.
KERNEL_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# LegacyMode enum
# ---------------------------------------------------------------------------


class LegacyMode(str, enum.Enum):
    """Cutover mode for kernel operation.

    * ``KERNEL_ONLY`` -- pure kernel decisions (default).
    * ``SHADOW``      -- kernel decides but a secondary path runs in parallel for comparison.

    The former ``LEGACY`` mode (Python decides) was removed in AQC-825.
    """

    KERNEL_ONLY = "kernel_only"
    SHADOW = "shadow"


def get_legacy_mode(config: dict[str, Any] | None = None) -> LegacyMode:
    """Read the cutover mode from environment or config.

    Resolution order:

    1. ``KERNEL_LEGACY_MODE`` environment variable (if set and valid).
    2. ``config["legacy_mode"]`` (if provided and valid).
    3. Default: ``LegacyMode.KERNEL_ONLY``.

    Parameters
    ----------
    config : dict, optional
        Config dict that may contain a ``"legacy_mode"`` key.

    Returns
    -------
    LegacyMode
    """
    env_val = os.environ.get("KERNEL_LEGACY_MODE", "").strip().lower()
    if env_val:
        try:
            return LegacyMode(env_val)
        except ValueError:
            logger.warning(
                "[orchestrator] invalid KERNEL_LEGACY_MODE=%r, falling back to config/default",
                env_val,
            )

    if config is not None:
        cfg_val = str(config.get("legacy_mode", "")).strip().lower()
        if cfg_val:
            try:
                return LegacyMode(cfg_val)
            except ValueError:
                logger.warning(
                    "[orchestrator] invalid config legacy_mode=%r, falling back to default",
                    cfg_val,
                )

    return LegacyMode.KERNEL_ONLY


# ---------------------------------------------------------------------------
# KernelDecision dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KernelDecision:
    """Parsed response from a single kernel step.

    Attributes
    ----------
    ok : bool
        ``True`` if the kernel returned a successful response.
    state_json : str
        New kernel state JSON after the step.
    intents : list[dict]
        ``OrderIntent`` dicts from the kernel decision.
    fills : list[dict]
        ``FillEvent`` dicts from the kernel decision (paper fills).
    diagnostics : dict
        Full diagnostics dict from the kernel.
    action : str
        Extracted action string (``"BUY"`` / ``"SELL"`` / ``"HOLD"`` / ``"CLOSE"``).
    raw_json : str
        Full kernel response JSON for audit trail.
    """

    ok: bool
    state_json: str
    intents: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    action: str
    raw_json: str


# ---------------------------------------------------------------------------
# MarketEvent builders
# ---------------------------------------------------------------------------


def build_evaluate_event(
    snap: dict[str, Any],
    gate_result: dict[str, Any],
    ema_slow_slope_pct: float,
    symbol: str,
    price: float | None = None,
) -> dict[str, Any]:
    """Construct a MarketEvent dict for ``signal="evaluate"``.

    Parameters
    ----------
    snap : dict
        IndicatorSnapshot dict (from ``build_indicator_snapshot``).
    gate_result : dict
        GateResult dict (from ``build_gate_result``).
    ema_slow_slope_pct : float
        EMA-slow slope as a fraction of current price.
    symbol : str
        Trading symbol (e.g. ``"ETH"``).
    price : float, optional
        Override close price.  Defaults to ``snap["close"]``.

    Returns
    -------
    dict
        MarketEvent ready for ``json.dumps()`` and kernel ingestion.
    """
    ts = int(snap.get("t", int(time.time() * 1000)))
    close_price = float(price if price is not None else snap.get("close", 0.0))

    return {
        "schema_version": KERNEL_SCHEMA_VERSION,
        "event_id": ts,
        "timestamp_ms": ts,
        "symbol": str(symbol).upper(),
        "signal": "evaluate",
        "price": close_price,
        "indicators": snap,
        "gate_result": gate_result,
        "ema_slow_slope_pct": float(ema_slow_slope_pct),
    }


def build_price_update_event(
    symbol: str,
    price: float,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Construct a MarketEvent dict for ``signal="price_update"``.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. ``"ETH"``).
    price : float
        Current price.
    timestamp_ms : int, optional
        Epoch ms timestamp.  Defaults to now.

    Returns
    -------
    dict
        MarketEvent ready for ``json.dumps()`` and kernel ingestion.
    """
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

    return {
        "schema_version": KERNEL_SCHEMA_VERSION,
        "event_id": ts,
        "timestamp_ms": ts,
        "symbol": str(symbol).upper(),
        "signal": "price_update",
        "price": float(price),
    }


# ---------------------------------------------------------------------------
# KernelOrchestrator
# ---------------------------------------------------------------------------


class KernelOrchestrator:
    """Thin orchestrator replacing Python signal generation.

    Feeds candle data (via indicator helpers) to the Rust decision kernel,
    parses the response into ``KernelDecision`` objects, and optionally
    routes ``OrderIntents`` through the broker adapter.

    Parameters
    ----------
    config : dict, optional
        Strategy / orchestrator configuration.
    broker_adapter : object, optional
        A ``BrokerAdapter`` instance for executing intents on the exchange.
    reconciler : object, optional
        A ``PositionReconciler`` instance for kernel-vs-exchange sync.
    shadow_tracker : object, optional
        A ``ShadowDecisionTracker`` for shadow-mode comparison logging.
    db_path : str, optional
        Path to the SQLite database for decision logging.  If ``None``,
        falls back to ``DB_PATH`` from ``mei_alpha_v1``.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        broker_adapter: Any | None = None,
        reconciler: Any | None = None,
        shadow_tracker: Any | None = None,
        db_path: str | None = None,
    ):
        self._config = config or {}
        self._broker_adapter = broker_adapter
        self._reconciler = reconciler
        self._shadow_tracker = shadow_tracker
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Main entry: process a candle
    # ------------------------------------------------------------------

    def process_candle(
        self,
        symbol: str,
        candle_df: Any,
        kernel_state_json: str,
        params_json: str,
        exit_params_json: str | None = None,
        btc_bullish: bool | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> KernelDecision:
        """Evaluate entry signal from a candle DataFrame via the kernel.

        Steps:
            1. Build IndicatorSnapshot from ``candle_df``.
            2. Build GateResult from the snapshot.
            3. Build EntryParams from config.
            4. Compute EMA-slow slope.
            5. Construct MarketEvent with ``signal="evaluate"``.
            6. Call ``bt_runtime.step_decision()`` (or ``step_full()`` if
               *exit_params_json* provided).
            7. Parse response into ``KernelDecision``.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        candle_df : pd.DataFrame
            Candle data with Open/High/Low/Close/Volume.
        kernel_state_json : str
            Current kernel ``StrategyState`` JSON.
        params_json : str
            Current ``KernelParams`` JSON.
        exit_params_json : str, optional
            If provided, ``step_full()`` is used (entry + exit in one step).
        btc_bullish : bool or None
            BTC trend direction.
        cfg : dict, optional
            Strategy config override.

        Returns
        -------
        KernelDecision
        """
        if not _BT_RUNTIME_AVAILABLE or _bt_runtime is None:
            raise RuntimeError(
                "FATAL: bt_runtime extension is not available.  The Rust kernel is "
                "REQUIRED for live/paper trading (AQC-825).  Build bt_runtime or check "
                "your AI_QUANT_BT_RUNTIME_PATH configuration."
            )

        effective_cfg = cfg or self._config

        # 1-4. Build helpers
        (
            build_indicator_snapshot,
            build_gate_result,
            build_entry_params,
            compute_ema_slow_slope,
        ) = _import_mei_helpers()

        snap = build_indicator_snapshot(
            candle_df,
            symbol=symbol,
            config=effective_cfg.get("indicators"),
        )
        gate_result = build_gate_result(
            snap,
            symbol,
            cfg=effective_cfg,
            btc_bullish=btc_bullish,
            ema_slow_slope_pct=compute_ema_slow_slope(candle_df, effective_cfg),
        )
        entry_params = build_entry_params(effective_cfg)
        ema_slow_slope_pct = compute_ema_slow_slope(candle_df, effective_cfg)

        # 5. Construct MarketEvent
        event = build_evaluate_event(
            snap, gate_result, ema_slow_slope_pct, symbol,
        )
        event_json = json.dumps(event, default=_json_default)

        # Merge entry_params into params
        params_dict = json.loads(params_json)
        params_dict["entry_params"] = entry_params
        merged_params_json = json.dumps(params_dict)

        # 6. Call kernel
        try:
            if exit_params_json is not None:
                raw = _bt_runtime.step_full(
                    kernel_state_json, event_json, merged_params_json, exit_params_json,
                )
            else:
                raw = _bt_runtime.step_decision(
                    kernel_state_json, event_json, merged_params_json,
                )
        except Exception as exc:
            logger.error("[orchestrator] kernel call failed: %s", exc)
            return KernelDecision(
                ok=False,
                state_json=kernel_state_json,
                intents=[],
                fills=[],
                diagnostics={"error": str(exc)},
                action="HOLD",
                raw_json="{}",
            )

        return self._parse_kernel_response(raw, kernel_state_json)

    # ------------------------------------------------------------------
    # Price update (exit evaluation)
    # ------------------------------------------------------------------

    def process_price_update(
        self,
        symbol: str,
        price: float,
        kernel_state_json: str,
        params_json: str,
        exit_params_json: str,
        timestamp_ms: int | None = None,
    ) -> KernelDecision:
        """Evaluate exit conditions via a PriceUpdate through the kernel.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        price : float
            Current price.
        kernel_state_json : str
            Current kernel ``StrategyState`` JSON.
        params_json : str
            Current ``KernelParams`` JSON.
        exit_params_json : str
            ``ExitParams`` JSON for the kernel.
        timestamp_ms : int, optional
            Override timestamp.

        Returns
        -------
        KernelDecision
        """
        if not _BT_RUNTIME_AVAILABLE or _bt_runtime is None:
            raise RuntimeError(
                "FATAL: bt_runtime extension is not available.  The Rust kernel is "
                "REQUIRED for live/paper trading (AQC-825).  Build bt_runtime or check "
                "your AI_QUANT_BT_RUNTIME_PATH configuration."
            )

        event = build_price_update_event(symbol, price, timestamp_ms=timestamp_ms)
        event_json = json.dumps(event)

        try:
            raw = _bt_runtime.step_full(
                kernel_state_json, event_json, params_json, exit_params_json,
            )
        except Exception as exc:
            logger.error("[orchestrator] kernel price_update failed: %s", exc)
            return KernelDecision(
                ok=False,
                state_json=kernel_state_json,
                intents=[],
                fills=[],
                diagnostics={"error": str(exc)},
                action="HOLD",
                raw_json="{}",
            )

        return self._parse_kernel_response(raw, kernel_state_json)

    # ------------------------------------------------------------------
    # Execute decision through broker
    # ------------------------------------------------------------------

    def execute_decision(
        self,
        decision: KernelDecision,
        dry_run: bool = False,
        symbol_info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute intents from a ``KernelDecision`` through the broker adapter.

        Parameters
        ----------
        decision : KernelDecision
            Parsed kernel decision containing intents.
        dry_run : bool
            If ``True``, log intents but do not execute.
        symbol_info : dict, optional
            Symbol metadata for size rounding.

        Returns
        -------
        list[dict]
            ``FillEvent`` dicts from the broker adapter.  Empty list if
            ``dry_run`` is ``True``, no intents, or no broker adapter.
        """
        if not decision.intents:
            logger.debug("[orchestrator] execute_decision: no intents to execute")
            return []

        if dry_run:
            logger.info(
                "[orchestrator] DRY RUN: would execute %d intents: %s",
                len(decision.intents),
                json.dumps(decision.intents, default=str),
            )
            return []

        if self._broker_adapter is None:
            logger.warning("[orchestrator] no broker_adapter configured; cannot execute intents")
            return []

        fills = self._broker_adapter.execute_intents(
            decision.intents, symbol_info=symbol_info,
        )
        logger.info(
            "[orchestrator] executed %d/%d intents, got %d fills",
            len(fills),
            len(decision.intents),
            len(fills),
        )
        return fills

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        kernel_state_json: str,
        exchange_positions: dict[str, dict[str, Any]],
    ) -> Any:
        """Reconcile kernel positions with exchange positions.

        Delegates to the configured ``PositionReconciler``.

        Parameters
        ----------
        kernel_state_json : str
            Current kernel ``StrategyState`` JSON (contains ``positions``).
        exchange_positions : dict
            Mapping of symbol -> exchange position dict.

        Returns
        -------
        ReconciliationReport or None
            The reconciliation report, or ``None`` if no reconciler configured.
        """
        if self._reconciler is None:
            logger.warning("[orchestrator] no reconciler configured; skipping reconciliation")
            return None

        # Extract kernel positions from state
        try:
            state = json.loads(kernel_state_json)
            kernel_positions = state.get("positions", {})
        except (json.JSONDecodeError, TypeError):
            logger.error("[orchestrator] failed to parse kernel state for reconciliation")
            return None

        return self._reconciler.reconcile(kernel_positions, exchange_positions)

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def log_decision(self, decision: KernelDecision, symbol: str = "") -> str | None:
        """Log a kernel decision to the ``decision_events`` SQLite table.

        Parameters
        ----------
        decision : KernelDecision
            The decision to log.
        symbol : str
            Trading symbol for the event row.

        Returns
        -------
        str or None
            The decision event ULID if logged, else ``None``.
        """
        db_path = self._db_path
        if db_path is None:
            try:
                from strategy.mei_alpha_v1 import DB_PATH
                db_path = DB_PATH
            except ImportError:
                logger.warning("[orchestrator] cannot import DB_PATH; skipping decision logging")
                return None

        import sqlite3

        event_id = _generate_event_id()
        ts = int(time.time() * 1000)

        context = {
            "ok": decision.ok,
            "action": decision.action,
            "intents_count": len(decision.intents),
            "fills_count": len(decision.fills),
            "diagnostics": decision.diagnostics,
        }

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute(
                """
                INSERT INTO decision_events
                    (id, timestamp_ms, symbol, event_type, status,
                     decision_phase, triggered_by, action_taken,
                     context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts,
                    str(symbol).upper(),
                    "kernel_decision",
                    "executed" if decision.ok else "error",
                    "kernel_evaluation",
                    "kernel_orchestrator",
                    decision.action,
                    json.dumps(context, separators=(",", ":")),
                ),
            )
            conn.commit()
            return event_id
        except Exception:
            logger.exception("[orchestrator] failed to log decision to DB")
            return None
        finally:
            if conn is not None:
                conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_kernel_response(
        self, raw_json: str, fallback_state_json: str,
    ) -> KernelDecision:
        """Parse a raw kernel JSON response into a ``KernelDecision``.

        Parameters
        ----------
        raw_json : str
            Full kernel response envelope.
        fallback_state_json : str
            State to use if parsing fails.

        Returns
        -------
        KernelDecision
        """
        try:
            envelope = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("[orchestrator] failed to parse kernel response: %s", exc)
            return KernelDecision(
                ok=False,
                state_json=fallback_state_json,
                intents=[],
                fills=[],
                diagnostics={"error": f"JSON parse error: {exc}"},
                action="HOLD",
                raw_json=raw_json if isinstance(raw_json, str) else "{}",
            )

        ok = bool(envelope.get("ok", False))

        if not ok:
            error = envelope.get("error", {})
            return KernelDecision(
                ok=False,
                state_json=fallback_state_json,
                intents=[],
                fills=[],
                diagnostics={"error": error},
                action="HOLD",
                raw_json=raw_json,
            )

        decision = envelope.get("decision", {})
        intents = decision.get("intents", [])
        fills = decision.get("fills", [])
        diagnostics = decision.get("diagnostics", {})

        # Extract new state
        state = decision.get("state", {})
        state_json = json.dumps(state, separators=(",", ":"), sort_keys=True)

        # Extract action
        extract_action = _import_extract_action()
        action = extract_action(raw_json)

        return KernelDecision(
            ok=True,
            state_json=state_json,
            intents=intents,
            fills=fills,
            diagnostics=diagnostics,
            action=action,
            raw_json=raw_json,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_event_id() -> str:
    """Generate a ULID for decision event IDs.

    Mirrors ``_generate_event_id()`` from sibling modules to avoid
    circular imports.
    """
    import random

    t = int(time.time() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = _ULID_ALPHABET[t & 0x1F] + time_part
        t >>= 5
    rand_part = "".join(random.choices(_ULID_ALPHABET, k=16))
    return time_part + rand_part


def _json_default(o: Any) -> Any:
    """JSON serializer fallback for numpy/datetime types."""
    try:
        if hasattr(o, "item"):
            return o.item()
    except Exception:
        pass
    return str(o)
