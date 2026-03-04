#!/usr/bin/env python3
"""Audit action-level alignment between live baseline and backtester replay report.

This report compares canonical action events from:
- `live_baseline_trades.jsonl`
- replay JSON output generated with `mei-backtester replay --trades --output ...`

It validates event parity (OPEN/ADD/REDUCE/CLOSE) and numeric parity, while
classifying funding-only gaps as accepted non-simulatable residuals.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from reason_codes import classify_reason_code
except ModuleNotFoundError:  # pragma: no cover - module execution path
    from tools.reason_codes import classify_reason_code

SIDE_ACTIONS = {"OPEN", "ADD", "REDUCE", "CLOSE"}
FUNDING_ACTION = "FUNDING"
END_OF_BACKTEST_REASON_CODES = {"exit_end_of_backtest"}
ORDER_FAIL_EVENTS = {
    "LIVE_ORDER_FAIL_UPDATE_LEVERAGE",
    "LIVE_ORDER_FAIL_MARKET_OPEN",
    "LIVE_ORDER_FAIL_MARKET_CLOSE",
}
DEFAULT_TOL = 1e-9
DEFAULT_ORDER_FAIL_MATCH_WINDOW_MS = 300_000
COMPARE_SURFACE_ARTEFACT_KINDS = {
    "matched_funding_pair",
    "missing_backtester_funding_action",
    "missing_live_funding_action",
    "missing_live_action_end_of_backtest",
    "missing_live_action_live_order_fail",
}
COMPARE_SURFACE_ARTEFACT_CLASSES = {"non-simulatable_exchange_oms_effect", "state_initialisation_gap"}
COMPARE_SURFACE_ARTEFACT_CLASS_BY_KIND = {
    "matched_funding_pair": "non-simulatable_exchange_oms_effect",
    "missing_backtester_funding_action": "non-simulatable_exchange_oms_effect",
    "missing_live_funding_action": "non-simulatable_exchange_oms_effect",
    "missing_live_action_end_of_backtest": "state_initialisation_gap",
    "missing_live_action_live_order_fail": "non-simulatable_exchange_oms_effect",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit live-vs-backtester action-level parity from replay bundle artefacts.")
    parser.add_argument("--live-baseline", required=True, help="Path to live_baseline_trades.jsonl")
    parser.add_argument("--backtester-replay-report", required=True, help="Path to backtester replay JSON report (with trades)")
    parser.add_argument(
        "--live-order-fail-events",
        help=(
            "Optional path to live order-fail event JSONL exported from audit_events "
            "(events beginning with LIVE_ORDER_FAIL_)."
        ),
    )
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument(
        "--timestamp-bucket-ms",
        type=int,
        default=1,
        help="Match timestamp bucket in milliseconds (1 = exact millisecond match)",
    )
    parser.add_argument(
        "--timestamp-bucket-anchor",
        choices=("floor", "ceil"),
        default="floor",
        help="Timestamp bucket anchor for match key generation",
    )
    parser.add_argument("--price-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for price comparison")
    parser.add_argument("--size-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for size comparison")
    parser.add_argument("--pnl-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for pnl comparison")
    parser.add_argument("--fee-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for fee comparison")
    parser.add_argument("--balance-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for balance comparison")
    parser.add_argument(
        "--allow-compare-surface-artefacts",
        action="store_true",
        default=False,
        help=(
            "Opt-in: expose gate pass for artefact-only mismatch windows "
            "(strict fail-closed remains the default signal)."
        ),
    )
    parser.add_argument(
        "--order-fail-match-window-ms",
        type=int,
        default=DEFAULT_ORDER_FAIL_MATCH_WINDOW_MS,
        help="Absolute timestamp window for matching missing live actions to LIVE_ORDER_FAIL events",
    )
    return parser


def _parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except Exception:
        return 0.0


def _parse_timestamp_ms(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > 10_000_000_000:
            return iv
        return max(0, iv * 1000)

    raw = str(value).strip()
    if not raw:
        return 0
    if raw.isdigit():
        iv = int(raw)
        if iv > 10_000_000_000:
            return iv
        return iv * 1000
    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(ts.timestamp() * 1000)
    except Exception:
        return 0


def _normalise_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalise_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    return ""


def _normalise_confidence(value: Any) -> str:
    conf = str(value or "").strip().lower()
    if conf in {"low", "medium", "high"}:
        return conf
    return conf


def _normalise_signal(value: Any) -> str:
    signal = str(value or "").strip().upper()
    if signal in {"BUY", "SELL"}:
        return signal
    return ""


def _almost_equal(left: float, right: float, tol: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def _bucket_timestamp_ms(ts_ms: int, bucket_ms: int, anchor: str) -> int:
    if bucket_ms <= 1:
        return int(ts_ms)
    if str(anchor).strip().lower() == "ceil":
        if ts_ms >= 0:
            return int(((ts_ms + bucket_ms - 1) // bucket_ms) * bucket_ms)
        return int(-(((-ts_ms) // bucket_ms) * bucket_ms))
    if ts_ms >= 0:
        return int((ts_ms // bucket_ms) * bucket_ms)
    return int(-(((-ts_ms) // bucket_ms) * bucket_ms))


def _canonical_live_action(action: Any, side: Any) -> str:
    act = str(action or "").strip().upper()
    if act in SIDE_ACTIONS:
        s = _normalise_side(side)
        if not s:
            return ""
        return f"{act}_{s}"
    if act == FUNDING_ACTION:
        return FUNDING_ACTION
    return ""


def _canonical_backtester_action(action: Any) -> str:
    act = str(action or "").strip().upper()
    if act == FUNDING_ACTION:
        return FUNDING_ACTION
    if any(act == f"{base}_LONG" or act == f"{base}_SHORT" for base in SIDE_ACTIONS):
        return act
    return ""


def _is_exit_action_code(action_code: str) -> bool:
    code = str(action_code or "").strip().upper()
    return code.startswith("CLOSE_") or code.startswith("REDUCE_")


def _extract_side_from_action_code(action_code: str) -> str:
    code = str(action_code or "").strip().upper()
    if code.endswith("_LONG"):
        return "LONG"
    if code.endswith("_SHORT"):
        return "SHORT"
    return ""


def _summarise_action_scope_contract(
    live_actions: list[dict[str, Any]],
    backtester_actions: list[dict[str, Any]],
    *,
    matched_pairs: int,
) -> dict[str, Any]:
    def _collect(rows: list[dict[str, Any]]) -> dict[str, Any]:
        symbols: set[str] = set()
        sides: set[str] = set()
        action_codes: set[str] = set()
        symbol_sides: set[str] = set()
        non_funding_rows = 0
        for row in rows:
            action_code = str(row.get("action_code") or "").strip().upper()
            if not action_code or action_code == FUNDING_ACTION:
                continue
            symbol = _normalise_symbol(row.get("symbol"))
            if not symbol:
                continue
            non_funding_rows += 1
            action_codes.add(action_code)
            symbols.add(symbol)
            side = _extract_side_from_action_code(action_code)
            if side:
                sides.add(side)
                symbol_sides.add(f"{symbol}:{side}")
        return {
            "non_funding_rows": int(non_funding_rows),
            "symbols": sorted(symbols),
            "sides": sorted(sides),
            "action_codes": sorted(action_codes),
            "symbol_sides": sorted(symbol_sides),
        }

    live_scope = _collect(live_actions)
    backtester_scope = _collect(backtester_actions)
    shared_symbols = sorted(set(live_scope["symbols"]) & set(backtester_scope["symbols"]))
    shared_sides = sorted(set(live_scope["sides"]) & set(backtester_scope["sides"]))
    shared_action_codes = sorted(set(live_scope["action_codes"]) & set(backtester_scope["action_codes"]))
    shared_symbol_sides = sorted(set(live_scope["symbol_sides"]) & set(backtester_scope["symbol_sides"]))
    mismatch = bool(
        int(matched_pairs) == 0
        and int(live_scope["non_funding_rows"]) > 0
        and int(backtester_scope["non_funding_rows"]) > 0
        and len(shared_symbol_sides) == 0
    )
    if not mismatch:
        mismatch_kind = ""
    elif len(shared_symbols) == 0:
        mismatch_kind = "symbol_scope_disjoint"
    else:
        mismatch_kind = "symbol_side_scope_disjoint"
    return {
        "matched_pairs": int(matched_pairs),
        "live_non_funding_rows": int(live_scope["non_funding_rows"]),
        "backtester_non_funding_rows": int(backtester_scope["non_funding_rows"]),
        "live_symbols": live_scope["symbols"],
        "backtester_symbols": backtester_scope["symbols"],
        "shared_symbols": shared_symbols,
        "live_sides": live_scope["sides"],
        "backtester_sides": backtester_scope["sides"],
        "shared_sides": shared_sides,
        "live_action_codes": live_scope["action_codes"],
        "backtester_action_codes": backtester_scope["action_codes"],
        "shared_action_codes": shared_action_codes,
        "live_symbol_sides": live_scope["symbol_sides"],
        "backtester_symbol_sides": backtester_scope["symbol_sides"],
        "shared_symbol_sides": shared_symbol_sides,
        "mismatch": mismatch,
        "mismatch_kind": mismatch_kind,
    }


def _apply_scope_disjoint_state_gap_classification(
    mismatches: list[dict[str, Any]],
    *,
    scope_contract: Mapping[str, Any] | None,
) -> int:
    if not isinstance(scope_contract, Mapping):
        return 0
    mismatch_kind = str(scope_contract.get("mismatch_kind") or "").strip().lower()
    shared_symbol_sides = scope_contract.get("shared_symbol_sides") or []
    if mismatch_kind != "symbol_side_scope_disjoint" or len(shared_symbol_sides) != 0:
        return 0

    reclassified = 0
    for row in mismatches:
        if str(row.get("classification") or "").strip().lower() != "deterministic_logic_divergence":
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in {"missing_backtester_action", "missing_live_action"}:
            continue
        row["classification"] = "state_initialisation_gap"
        row["scope_contract_gap"] = {
            "kind": "symbol_side_scope_disjoint",
            "shared_symbol_sides": list(shared_symbol_sides),
        }
        reclassified += 1
    return reclassified


def _collapse_split_fill_actions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_fill_hash: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        fill_hash = str(row.get("fill_hash") or "").strip().lower()
        action_code = str(row.get("action_code") or "").strip().upper()
        symbol = str(row.get("symbol") or "").strip().upper()
        if fill_hash and symbol and _is_exit_action_code(action_code):
            by_fill_hash[(symbol, action_code, fill_hash)].append(row)
        else:
            passthrough.append(dict(row))

    collapsed_rows: list[dict[str, Any]] = list(passthrough)
    collapsed_groups = 0
    collapsed_member_rows = 0

    for (_symbol, _action_code, fill_hash), group in by_fill_hash.items():
        if len(group) <= 1:
            collapsed_rows.append(dict(group[0]))
            continue

        sorted_group = sorted(
            group,
            key=lambda r: (
                int(r.get("timestamp_ms") or 0),
                int(r.get("source_id") or 0),
                int(r.get("line_no") or 0),
            ),
        )
        collapsed_groups += 1
        collapsed_member_rows += len(sorted_group)

        total_size = sum(float(item.get("size") or 0.0) for item in sorted_group)
        total_pnl = sum(float(item.get("pnl_usd") or 0.0) for item in sorted_group)
        total_fee = sum(float(item.get("fee_usd") or 0.0) for item in sorted_group)
        total_abs_size = sum(abs(float(item.get("size") or 0.0)) for item in sorted_group)
        if total_abs_size > 0.0:
            weighted_price = sum(
                float(item.get("price") or 0.0) * abs(float(item.get("size") or 0.0))
                for item in sorted_group
            ) / total_abs_size
        else:
            weighted_price = float(sorted_group[-1].get("price") or 0.0)

        merged = dict(sorted_group[-1])
        merged["timestamp_ms"] = max(int(item.get("timestamp_ms") or 0) for item in sorted_group)
        merged["source_id"] = max(int(item.get("source_id") or 0) for item in sorted_group)
        merged["line_no"] = max(int(item.get("line_no") or 0) for item in sorted_group)
        merged["price"] = float(weighted_price)
        merged["size"] = float(total_size)
        merged["pnl_usd"] = float(total_pnl)
        merged["fee_usd"] = float(total_fee)
        merged["fill_hash"] = fill_hash
        merged["split_fill_collapsed"] = True
        merged["split_fill_member_count"] = len(sorted_group)
        merged["split_fill_member_source_ids"] = [int(item.get("source_id") or 0) for item in sorted_group]
        collapsed_rows.append(merged)

    collapsed_rows.sort(
        key=lambda r: (
            int(r.get("timestamp_ms") or 0),
            str(r.get("symbol") or ""),
            str(r.get("action_code") or ""),
            int(r.get("source_id") or 0),
            int(r.get("line_no") or 0),
        )
    )
    stats = {
        "split_fill_groups_collapsed": int(collapsed_groups),
        "split_fill_rows_collapsed": int(max(0, collapsed_member_rows - collapsed_groups)),
    }
    return collapsed_rows, stats


def _summarise_funding_evidence(mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    per_symbol: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "matched_pairs": 0,
            "unmatched_live": 0,
            "unmatched_backtester": 0,
            "matched_live_pnl_usd": 0.0,
            "matched_backtester_pnl_usd": 0.0,
        }
    )

    for row in mismatches:
        if str(row.get("action_code") or "").strip().upper() != FUNDING_ACTION:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        bucket = per_symbol[symbol]
        bucket["symbol"] = symbol
        kind = str(row.get("kind") or "").strip().lower()
        if kind == "matched_funding_pair":
            bucket["matched_pairs"] += 1
            live_obj = row.get("live") if isinstance(row.get("live"), dict) else {}
            backtester_obj = row.get("backtester") if isinstance(row.get("backtester"), dict) else {}
            bucket["matched_live_pnl_usd"] += float(live_obj.get("pnl_usd") or 0.0)
            bucket["matched_backtester_pnl_usd"] += float(backtester_obj.get("pnl_usd") or 0.0)
        elif kind == "missing_backtester_funding_action":
            bucket["unmatched_live"] += 1
        elif kind == "missing_live_funding_action":
            bucket["unmatched_backtester"] += 1

    rows = sorted(per_symbol.values(), key=lambda item: item["symbol"])
    return {
        "matched_pairs": int(sum(int(item["matched_pairs"]) for item in rows)),
        "unmatched_live": int(sum(int(item["unmatched_live"]) for item in rows)),
        "unmatched_backtester": int(sum(int(item["unmatched_backtester"]) for item in rows)),
        "symbols": [str(item["symbol"]) for item in rows if str(item["symbol"])],
        "by_symbol": rows,
    }


def _summarise_mismatch_breakdown(mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    def _count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            key = str(row.get(field) or "unknown")
            counts[key] += 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    surface_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    logic_rows: list[dict[str, Any]] = []
    classification_kind_drift_rows: list[dict[str, Any]] = []

    for row in mismatches:
        kind = str(row.get("kind") or "").strip().lower()
        classification = str(row.get("classification") or "").strip().lower()
        kind_is_surface = kind in COMPARE_SURFACE_ARTEFACT_KINDS
        class_is_surface = classification in COMPARE_SURFACE_ARTEFACT_CLASSES
        class_is_state = classification == "state_initialisation_gap"

        if class_is_surface:
            surface_rows.append(row)
        elif class_is_state:
            state_rows.append(row)
        else:
            logic_rows.append(row)
        if (kind_is_surface != class_is_surface) and not class_is_state:
            classification_kind_drift_rows.append(
                {
                    "kind": kind or "unknown",
                    "classification": classification or "unknown",
                    "symbol": str(row.get("symbol") or "unknown"),
                }
            )

    total = len(mismatches)
    surface_total = len(surface_rows)
    state_total = len(state_rows)
    logic_total = len(logic_rows)
    return {
        "total": int(total),
        "compare_surface_artefact_total": int(surface_total),
        "state_scope_residual_total": int(state_total),
        "logic_divergence_total": int(logic_total),
        "compare_surface_artefact_ratio": float(surface_total / total) if total else 0.0,
        "logic_divergence_ratio": float(logic_total / total) if total else 0.0,
        "compare_surface_artefact_by_kind": _count(surface_rows, "kind"),
        "compare_surface_artefact_by_symbol": _count(surface_rows, "symbol"),
        "logic_divergence_by_kind": _count(logic_rows, "kind"),
        "logic_divergence_by_action_code": _count(logic_rows, "action_code"),
        "logic_divergence_by_symbol": _count(logic_rows, "symbol"),
        "classification_kind_drift_total": int(len(classification_kind_drift_rows)),
        "classification_kind_drift_by_kind": _count(classification_kind_drift_rows, "kind"),
        "classification_kind_drift_by_classification": _count(classification_kind_drift_rows, "classification"),
    }


def _normalise_compare_surface_classifications(mismatches: list[dict[str, Any]]) -> int:
    reclassified = 0
    for row in mismatches:
        kind = str(row.get("kind") or "").strip().lower()
        canonical = COMPARE_SURFACE_ARTEFACT_CLASS_BY_KIND.get(kind)
        if not canonical:
            continue
        current = str(row.get("classification") or "").strip().lower()
        if current == canonical:
            continue
        if current:
            row["classification_original"] = current
        row["classification"] = canonical
        reclassified += 1
    return reclassified


def _load_live_actions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    total_actions = 0
    unknown_actions = 0

    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            total_actions += 1

            action_code = _canonical_live_action(row.get("action"), row.get("type"))
            if not action_code:
                unknown_actions += 1
                continue

            symbol = _normalise_symbol(row.get("symbol"))
            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms <= 0:
                ts_ms = _parse_timestamp_ms(row.get("timestamp"))
            if not symbol or ts_ms <= 0:
                continue

            actions.append(
                {
                    "source": "live",
                    "source_id": int(row.get("id") or 0),
                    "line_no": line_no,
                    "symbol": symbol,
                    "timestamp_ms": ts_ms,
                    "action_code": action_code,
                    "price": _parse_float(row.get("price")),
                    "size": _parse_float(row.get("size")),
                    "pnl_usd": _parse_float(row.get("pnl")),
                    "fee_usd": _parse_float(row.get("fee_usd")),
                    "balance": _parse_float(row.get("balance")),
                    "confidence": _normalise_confidence(row.get("confidence")),
                    "reason": str(row.get("reason") or ""),
                    "reason_code": (
                        str(row.get("reason_code") or "").strip().lower()
                        or classify_reason_code(action_code, str(row.get("reason") or ""))
                    ),
                    "fill_hash": str(row.get("fill_hash") or "").strip().lower(),
                }
            )

    actions.sort(key=lambda r: (r["timestamp_ms"], r["symbol"], r["action_code"], r["source_id"], r["line_no"]))
    counts = {
        "live_total_actions": total_actions,
        "live_canonical_actions": len(actions),
        "live_unknown_actions": unknown_actions,
    }
    return actions, counts


def _load_backtester_actions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trades = payload.get("trades")
    if not isinstance(trades, list):
        raise ValueError("backtester replay report does not include `trades`; run replay with `--trades --output`")

    actions: list[dict[str, Any]] = []
    unknown_actions = 0
    for idx, row in enumerate(trades, start=1):
        action_code = _canonical_backtester_action(row.get("action"))
        if not action_code:
            unknown_actions += 1
            continue

        symbol = _normalise_symbol(row.get("symbol"))
        ts_ms = _parse_timestamp_ms(row.get("timestamp"))
        if not symbol or ts_ms <= 0:
            continue

        reason_text = str(row.get("reason") or "")
        reason_code = str(row.get("reason_code") or "").strip().lower()
        if not reason_code:
            reason_code = classify_reason_code(action_code, reason_text)

        actions.append(
            {
                "source": "backtester",
                "source_id": idx,
                "row_no": idx,
                "symbol": symbol,
                "timestamp_ms": ts_ms,
                "action_code": action_code,
                "price": _parse_float(row.get("price")),
                "size": _parse_float(row.get("size")),
                "pnl_usd": _parse_float(row.get("pnl")),
                "fee_usd": _parse_float(row.get("fee")),
                "balance": _parse_float(row.get("balance")),
                "confidence": _normalise_confidence(row.get("confidence")),
                "reason": reason_text,
                "reason_code": reason_code,
            }
        )

    actions.sort(key=lambda r: (r["timestamp_ms"], r["symbol"], r["action_code"], r["source_id"], r["row_no"]))
    counts = {
        "backtester_total_actions": len(trades),
        "backtester_canonical_actions": len(actions),
        "backtester_unknown_actions": unknown_actions,
    }
    return actions, counts


def _load_live_order_fail_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    total_rows = 0
    skipped_rows = 0

    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except Exception:
                skipped_rows += 1
                continue

            event = str(row.get("event") or "").strip().upper()
            symbol = _normalise_symbol(row.get("symbol"))
            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms <= 0:
                ts_ms = _parse_timestamp_ms(row.get("timestamp"))
            if event not in ORDER_FAIL_EVENTS or not symbol or ts_ms <= 0:
                skipped_rows += 1
                continue

            data = row.get("data")
            if not isinstance(data, dict):
                data = {}

            events.append(
                {
                    "source": "live_order_fail",
                    "source_id": int(row.get("id") or 0),
                    "line_no": line_no,
                    "timestamp_ms": ts_ms,
                    "symbol": symbol,
                    "event": event,
                    "signal": _normalise_signal(data.get("signal")),
                    "kind": str(data.get("kind") or "").strip().upper(),
                    "submit_err_kind": str(data.get("submit_err_kind") or "").strip().lower(),
                    "submit_err": str(data.get("submit_err") or ""),
                    "data": data,
                }
            )

    events.sort(key=lambda r: (r["timestamp_ms"], r["symbol"], r["event"], int(r["source_id"])))
    counts = {
        "live_order_fail_total_rows": total_rows,
        "live_order_fail_events": len(events),
        "live_order_fail_skipped_rows": skipped_rows,
    }
    return events, counts


def _is_order_fail_compatible(action_code: str, event_row: dict[str, Any]) -> bool:
    code = str(action_code or "").strip().upper()
    if "_" not in code:
        return False
    base, side = code.split("_", 1)
    event = str(event_row.get("event") or "").strip().upper()
    kind = str(event_row.get("kind") or "").strip().upper()
    signal = str(event_row.get("signal") or "").strip().upper()

    if event not in ORDER_FAIL_EVENTS:
        return False

    if base == "ADD":
        return event in {"LIVE_ORDER_FAIL_UPDATE_LEVERAGE", "LIVE_ORDER_FAIL_MARKET_OPEN"} and kind == "ADD"

    if base == "OPEN":
        if event not in {"LIVE_ORDER_FAIL_UPDATE_LEVERAGE", "LIVE_ORDER_FAIL_MARKET_OPEN"}:
            return False
        if kind == "ADD":
            return False
        if signal == "BUY":
            return side == "LONG"
        if signal == "SELL":
            return side == "SHORT"
        return False

    if base in {"CLOSE", "REDUCE"}:
        return event == "LIVE_ORDER_FAIL_MARKET_CLOSE"

    return False


def _match_live_order_fail_event(
    live_order_fail_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    consumed_event_ids: set[int],
    symbol: str,
    action_code: str,
    target_ts_ms: int,
    window_ms: int,
) -> dict[str, Any] | None:
    candidates = live_order_fail_by_symbol.get(symbol, [])
    best: dict[str, Any] | None = None
    best_delta: int | None = None

    for row in candidates:
        source_id = int(row.get("source_id") or 0)
        if source_id > 0 and source_id in consumed_event_ids:
            continue
        if not _is_order_fail_compatible(action_code, row):
            continue
        delta = abs(int(row.get("timestamp_ms") or 0) - int(target_ts_ms))
        if delta > window_ms:
            continue
        if best is None or best_delta is None or delta < best_delta:
            best = row
            best_delta = delta

    if best is not None:
        source_id = int(best.get("source_id") or 0)
        if source_id > 0:
            consumed_event_ids.add(source_id)
    return best


def _group_events(
    rows: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
    timestamp_bucket_anchor: str,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["symbol"],
            row["action_code"],
            _bucket_timestamp_ms(int(row["timestamp_ms"]), timestamp_bucket_ms, timestamp_bucket_anchor),
        )
        grouped[key].append(row)
    for values in grouped.values():
        values.sort(
            key=lambda r: (
                float(r["size"]),
                float(r["pnl_usd"]),
                float(r["fee_usd"]),
                float(r["price"]),
                float(r["balance"]),
                int(r["source_id"]),
            )
        )
    return grouped


def _compare_actions(
    live_actions: list[dict[str, Any]],
    backtester_actions: list[dict[str, Any]],
    live_order_fail_events: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
    timestamp_bucket_anchor: str,
    price_tol: float,
    size_tol: float,
    pnl_tol: float,
    fee_tol: float,
    balance_tol: float,
    order_fail_match_window_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    matched_pairs = 0
    numeric_mismatch = 0
    confidence_mismatch = 0
    reason_code_mismatch = 0
    unmatched_live = 0
    unmatched_backtester = 0
    non_simulatable_residuals = 0
    state_scope_residuals = 0
    order_fail_residuals = 0
    funding_matched_pairs = 0
    funding_unmatched_live = 0
    funding_unmatched_backtester = 0
    consumed_order_fail_event_ids: set[int] = set()
    live_order_fail_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_order_fail_events:
        live_order_fail_by_symbol[str(row.get("symbol") or "")].append(row)

    per_symbol: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "live_actions": 0,
            "backtester_actions": 0,
            "matched_pairs": 0,
            "live_pnl_usd": 0.0,
            "backtester_pnl_usd": 0.0,
            "live_fee_usd": 0.0,
            "backtester_fee_usd": 0.0,
        }
    )

    for row in live_actions:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["live_actions"] += 1
        stats["live_pnl_usd"] += float(row["pnl_usd"])
        stats["live_fee_usd"] += float(row["fee_usd"])

    for row in backtester_actions:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["backtester_actions"] += 1
        stats["backtester_pnl_usd"] += float(row["pnl_usd"])
        stats["backtester_fee_usd"] += float(row["fee_usd"])

    live_groups = _group_events(
        live_actions,
        timestamp_bucket_ms=timestamp_bucket_ms,
        timestamp_bucket_anchor=timestamp_bucket_anchor,
    )
    bt_groups = _group_events(
        backtester_actions,
        timestamp_bucket_ms=timestamp_bucket_ms,
        timestamp_bucket_anchor=timestamp_bucket_anchor,
    )
    all_keys = sorted(set(live_groups.keys()) | set(bt_groups.keys()))

    for key in all_keys:
        symbol, action_code, ts_ms = key
        live_rows = live_groups.get(key, [])
        bt_rows = bt_groups.get(key, [])
        pair_count = min(len(live_rows), len(bt_rows))

        for idx in range(pair_count):
            lrow = live_rows[idx]
            brow = bt_rows[idx]
            per_symbol[symbol]["matched_pairs"] += 1
            matched_pairs += 1

            if action_code == FUNDING_ACTION:
                funding_matched_pairs += 1
                non_simulatable_residuals += 1
                mismatches.append(
                    {
                        "classification": COMPARE_SURFACE_ARTEFACT_CLASS_BY_KIND["matched_funding_pair"],
                        "kind": "matched_funding_pair",
                        "symbol": symbol,
                        "action_code": action_code,
                        "match_key_timestamp_ms": ts_ms,
                        "live_timestamp_ms": lrow["timestamp_ms"],
                        "backtester_timestamp_ms": brow["timestamp_ms"],
                        "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                        "backtester_ref": {"row_no": brow["row_no"]},
                        "live": {
                            "size": lrow["size"],
                            "pnl_usd": lrow["pnl_usd"],
                            "fee_usd": lrow["fee_usd"],
                            "reason": lrow["reason"],
                        },
                        "backtester": {
                            "size": brow["size"],
                            "pnl_usd": brow["pnl_usd"],
                            "fee_usd": brow["fee_usd"],
                            "reason_code": brow["reason_code"],
                            "reason": brow["reason"],
                        },
                        "delta": {
                            "size": lrow["size"] - brow["size"],
                            "pnl_usd": lrow["pnl_usd"] - brow["pnl_usd"],
                            "fee_usd": lrow["fee_usd"] - brow["fee_usd"],
                        },
                    }
                )
                continue

            numeric_checks = {
                "price": _almost_equal(lrow["price"], brow["price"], price_tol),
                "size": _almost_equal(lrow["size"], brow["size"], size_tol),
                "pnl_usd": _almost_equal(lrow["pnl_usd"], brow["pnl_usd"], pnl_tol),
                "fee_usd": _almost_equal(lrow["fee_usd"], brow["fee_usd"], fee_tol),
                "balance": _almost_equal(lrow["balance"], brow["balance"], balance_tol),
            }
            has_numeric_mismatch = not all(numeric_checks.values())

            live_conf = str(lrow["confidence"] or "")
            bt_conf = str(brow["confidence"] or "")
            has_confidence_mismatch = bool(live_conf and bt_conf and live_conf != bt_conf)
            live_reason_code = str(lrow["reason_code"] or "")
            bt_reason_code = str(brow["reason_code"] or "")
            has_reason_code_mismatch = bool(live_reason_code and bt_reason_code and live_reason_code != bt_reason_code)

            if not has_numeric_mismatch and not has_confidence_mismatch and not has_reason_code_mismatch:
                continue

            if has_numeric_mismatch:
                numeric_mismatch += 1
            if has_confidence_mismatch:
                confidence_mismatch += 1
            if has_reason_code_mismatch:
                reason_code_mismatch += 1

            classification = "numeric_policy_divergence" if has_numeric_mismatch else "deterministic_logic_divergence"
            if has_numeric_mismatch:
                kind = "action_numeric_mismatch"
            elif has_reason_code_mismatch:
                kind = "reason_code_mismatch"
            else:
                kind = "confidence_mismatch"

            mismatches.append(
                {
                    "classification": classification,
                    "kind": kind,
                    "symbol": symbol,
                    "action_code": action_code,
                    "match_key_timestamp_ms": ts_ms,
                    "live_timestamp_ms": lrow["timestamp_ms"],
                    "backtester_timestamp_ms": brow["timestamp_ms"],
                    "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                    "backtester_ref": {"row_no": brow["row_no"]},
                    "live": {
                        "price": lrow["price"],
                        "size": lrow["size"],
                        "pnl_usd": lrow["pnl_usd"],
                        "fee_usd": lrow["fee_usd"],
                        "balance": lrow["balance"],
                        "confidence": live_conf,
                        "reason_code": live_reason_code,
                        "reason": lrow["reason"],
                    },
                    "backtester": {
                        "price": brow["price"],
                        "size": brow["size"],
                        "pnl_usd": brow["pnl_usd"],
                        "fee_usd": brow["fee_usd"],
                        "balance": brow["balance"],
                        "confidence": bt_conf,
                        "reason_code": bt_reason_code,
                        "reason": brow["reason"],
                    },
                    "delta": {
                        "price": lrow["price"] - brow["price"],
                        "size": lrow["size"] - brow["size"],
                        "pnl_usd": lrow["pnl_usd"] - brow["pnl_usd"],
                        "fee_usd": lrow["fee_usd"] - brow["fee_usd"],
                        "balance": lrow["balance"] - brow["balance"],
                    },
                    "tolerance": {
                        "price_tol": price_tol,
                        "size_tol": size_tol,
                        "pnl_tol": pnl_tol,
                        "fee_tol": fee_tol,
                        "balance_tol": balance_tol,
                    },
                    "flags": {
                        "numeric_mismatch": has_numeric_mismatch,
                        "confidence_mismatch": has_confidence_mismatch,
                        "reason_code_mismatch": has_reason_code_mismatch,
                    },
                }
            )

        if len(live_rows) > pair_count:
            for lrow in live_rows[pair_count:]:
                if action_code == FUNDING_ACTION:
                    funding_unmatched_live += 1
                    unmatched_live += 1
                    mismatches.append(
                        {
                            "classification": COMPARE_SURFACE_ARTEFACT_CLASS_BY_KIND["missing_backtester_funding_action"],
                            "kind": "missing_backtester_funding_action",
                            "symbol": symbol,
                            "action_code": action_code,
                            "match_key_timestamp_ms": ts_ms,
                            "live_timestamp_ms": lrow["timestamp_ms"],
                            "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                            "live": {
                                "pnl_usd": lrow["pnl_usd"],
                                "fee_usd": lrow["fee_usd"],
                                "reason": lrow["reason"],
                            },
                        }
                    )
                else:
                    unmatched_live += 1
                    mismatches.append(
                        {
                            "classification": "deterministic_logic_divergence",
                            "kind": "missing_backtester_action",
                            "symbol": symbol,
                            "action_code": action_code,
                            "match_key_timestamp_ms": ts_ms,
                            "live_timestamp_ms": lrow["timestamp_ms"],
                            "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                            "live": {
                                "price": lrow["price"],
                                "size": lrow["size"],
                                "pnl_usd": lrow["pnl_usd"],
                                "fee_usd": lrow["fee_usd"],
                                "balance": lrow["balance"],
                                "confidence": lrow["confidence"],
                                "reason_code": lrow["reason_code"],
                                "reason": lrow["reason"],
                            },
                        }
                    )

        if len(bt_rows) > pair_count:
            for brow in bt_rows[pair_count:]:
                if action_code == FUNDING_ACTION:
                    funding_unmatched_backtester += 1
                    unmatched_backtester += 1
                    mismatches.append(
                        {
                            "classification": COMPARE_SURFACE_ARTEFACT_CLASS_BY_KIND["missing_live_funding_action"],
                            "kind": "missing_live_funding_action",
                            "symbol": symbol,
                            "action_code": action_code,
                            "match_key_timestamp_ms": ts_ms,
                            "backtester_timestamp_ms": brow["timestamp_ms"],
                            "backtester_ref": {"row_no": brow["row_no"]},
                            "backtester": {
                                "pnl_usd": brow["pnl_usd"],
                                "fee_usd": brow["fee_usd"],
                                "reason_code": brow["reason_code"],
                                "reason": brow["reason"],
                            },
                        }
                    )
                else:
                    reason_code = str(brow.get("reason_code") or "").strip().lower()
                    if reason_code in END_OF_BACKTEST_REASON_CODES:
                        state_scope_residuals += 1
                        mismatches.append(
                            {
                                "classification": "state_initialisation_gap",
                                "kind": "missing_live_action_end_of_backtest",
                                "symbol": symbol,
                                "action_code": action_code,
                                "match_key_timestamp_ms": ts_ms,
                                "backtester_timestamp_ms": brow["timestamp_ms"],
                                "backtester_ref": {"row_no": brow["row_no"]},
                                "backtester": {
                                    "price": brow["price"],
                                    "size": brow["size"],
                                    "pnl_usd": brow["pnl_usd"],
                                    "fee_usd": brow["fee_usd"],
                                    "balance": brow["balance"],
                                    "confidence": brow["confidence"],
                                    "reason_code": brow["reason_code"],
                                    "reason": brow["reason"],
                                },
                            }
                        )
                        continue

                    matched_fail = _match_live_order_fail_event(
                        live_order_fail_by_symbol,
                        consumed_event_ids=consumed_order_fail_event_ids,
                        symbol=symbol,
                        action_code=action_code,
                        target_ts_ms=int(brow["timestamp_ms"]),
                        window_ms=max(1, int(order_fail_match_window_ms)),
                    )
                    if matched_fail is not None:
                        non_simulatable_residuals += 1
                        order_fail_residuals += 1
                        mismatches.append(
                            {
                                "classification": "non-simulatable_exchange_oms_effect",
                                "kind": "missing_live_action_live_order_fail",
                                "symbol": symbol,
                                "action_code": action_code,
                                "match_key_timestamp_ms": ts_ms,
                                "backtester_timestamp_ms": brow["timestamp_ms"],
                                "backtester_ref": {"row_no": brow["row_no"]},
                                "backtester": {
                                    "price": brow["price"],
                                    "size": brow["size"],
                                    "pnl_usd": brow["pnl_usd"],
                                    "fee_usd": brow["fee_usd"],
                                    "balance": brow["balance"],
                                    "confidence": brow["confidence"],
                                    "reason_code": brow["reason_code"],
                                    "reason": brow["reason"],
                                },
                                "live_order_fail_event": {
                                    "id": matched_fail.get("source_id"),
                                    "timestamp_ms": matched_fail.get("timestamp_ms"),
                                    "event": matched_fail.get("event"),
                                    "kind": matched_fail.get("kind"),
                                    "signal": matched_fail.get("signal"),
                                    "submit_err_kind": matched_fail.get("submit_err_kind"),
                                },
                            }
                        )
                        continue

                    unmatched_backtester += 1
                    mismatches.append(
                        {
                            "classification": "deterministic_logic_divergence",
                            "kind": "missing_live_action",
                            "symbol": symbol,
                            "action_code": action_code,
                            "match_key_timestamp_ms": ts_ms,
                            "backtester_timestamp_ms": brow["timestamp_ms"],
                            "backtester_ref": {"row_no": brow["row_no"]},
                            "backtester": {
                                "price": brow["price"],
                                "size": brow["size"],
                                "pnl_usd": brow["pnl_usd"],
                                "fee_usd": brow["fee_usd"],
                                "balance": brow["balance"],
                                "confidence": brow["confidence"],
                                "reason_code": brow["reason_code"],
                                "reason": brow["reason"],
                            },
                        }
                    )

    per_symbol_rows = sorted(
        (
            {
                **values,
                "pnl_delta_usd": values["live_pnl_usd"] - values["backtester_pnl_usd"],
                "fee_delta_usd": values["live_fee_usd"] - values["backtester_fee_usd"],
            }
            for values in per_symbol.values()
        ),
        key=lambda row: row["symbol"],
    )

    summary = {
        "matched_pairs": matched_pairs,
        "numeric_mismatch": numeric_mismatch,
        "confidence_mismatch": confidence_mismatch,
        "reason_code_mismatch": reason_code_mismatch,
        "unmatched_live": unmatched_live,
        "unmatched_backtester": unmatched_backtester,
        "non_simulatable_residuals": non_simulatable_residuals,
        "state_scope_residuals": state_scope_residuals,
        "order_fail_residuals": order_fail_residuals,
        "funding_matched_pairs": funding_matched_pairs,
        "funding_unmatched_live": funding_unmatched_live,
        "funding_unmatched_backtester": funding_unmatched_backtester,
    }
    return mismatches, summary, per_symbol_rows


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_baseline = Path(args.live_baseline).expanduser().resolve()
    backtester_report = Path(args.backtester_replay_report).expanduser().resolve()
    live_order_fail_events_path = Path(args.live_order_fail_events).expanduser().resolve() if args.live_order_fail_events else None
    output = Path(args.output).expanduser().resolve()

    if not live_baseline.exists():
        parser.error(f"live baseline not found: {live_baseline}")
    if not backtester_report.exists():
        parser.error(f"backtester replay report not found: {backtester_report}")
    if live_order_fail_events_path is not None and not live_order_fail_events_path.exists():
        parser.error(f"live order-fail events file not found: {live_order_fail_events_path}")

    live_actions, live_counts = _load_live_actions(live_baseline)
    live_actions, live_split_fill_stats = _collapse_split_fill_actions(live_actions)
    try:
        backtester_actions, backtester_counts = _load_backtester_actions(backtester_report)
    except ValueError as exc:
        parser.error(str(exc))
    if live_order_fail_events_path is not None:
        live_order_fail_events, order_fail_counts = _load_live_order_fail_events(live_order_fail_events_path)
    else:
        live_order_fail_events, order_fail_counts = [], {
            "live_order_fail_total_rows": 0,
            "live_order_fail_events": 0,
            "live_order_fail_skipped_rows": 0,
        }

    mismatches, compare_summary, per_symbol_rows = _compare_actions(
        live_actions,
        backtester_actions,
        live_order_fail_events,
        timestamp_bucket_ms=max(1, int(args.timestamp_bucket_ms)),
        timestamp_bucket_anchor=str(args.timestamp_bucket_anchor or "floor"),
        price_tol=float(args.price_tol),
        size_tol=float(args.size_tol),
        pnl_tol=float(args.pnl_tol),
        fee_tol=float(args.fee_tol),
        balance_tol=float(args.balance_tol),
        order_fail_match_window_ms=max(1, int(args.order_fail_match_window_ms)),
    )
    scope_contract = _summarise_action_scope_contract(
        live_actions,
        backtester_actions,
        matched_pairs=int(compare_summary.get("matched_pairs") or 0),
    )
    classification_reclassified_count = _normalise_compare_surface_classifications(mismatches)
    scope_classification_reclassified_count = _apply_scope_disjoint_state_gap_classification(
        mismatches,
        scope_contract=scope_contract,
    )
    classification_reclassified_count += int(scope_classification_reclassified_count)

    mismatch_counts: dict[str, int] = defaultdict(int)
    mismatch_kind_counts: dict[str, int] = defaultdict(int)
    mismatch_action_code_counts: dict[str, int] = defaultdict(int)
    for item in mismatches:
        mismatch_counts[str(item.get("classification") or "unknown")] += 1
        mismatch_kind_counts[str(item.get("kind") or "unknown")] += 1
        mismatch_action_code_counts[str(item.get("action_code") or "unknown")] += 1

    accepted_classes = {"non-simulatable_exchange_oms_effect", "state_initialisation_gap"}
    accepted_residuals = [m for m in mismatches if str(m.get("classification") or "") in accepted_classes]
    mismatch_breakdown = _summarise_mismatch_breakdown(mismatches)

    logic_divergence_free = int(mismatch_breakdown["logic_divergence_total"]) == 0
    strict_alignment_pass = (
        compare_summary["numeric_mismatch"] == 0
        and compare_summary["confidence_mismatch"] == 0
        and compare_summary["reason_code_mismatch"] == 0
        and logic_divergence_free
    )
    compare_surface_artefact_total = int(mismatch_breakdown["compare_surface_artefact_total"])
    artefact_only_mismatch = bool(logic_divergence_free and compare_surface_artefact_total > 0)
    gate_pass_if_allow_compare_surface_artefacts = bool(strict_alignment_pass or artefact_only_mismatch)
    selected_gate_pass = bool(
        gate_pass_if_allow_compare_surface_artefacts
        if args.allow_compare_surface_artefacts
        else strict_alignment_pass
    )
    selected_gate_mode = "allow_compare_surface_artefacts" if args.allow_compare_surface_artefacts else "strict_fail_closed"

    report = {
        "schema_version": 1,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "inputs": {
            "live_baseline": str(live_baseline),
            "backtester_replay_report": str(backtester_report),
            "live_order_fail_events": str(live_order_fail_events_path) if live_order_fail_events_path else None,
            "timestamp_bucket_ms": max(1, int(args.timestamp_bucket_ms)),
            "timestamp_bucket_anchor": str(args.timestamp_bucket_anchor or "floor"),
            "price_tol": float(args.price_tol),
            "size_tol": float(args.size_tol),
            "pnl_tol": float(args.pnl_tol),
            "fee_tol": float(args.fee_tol),
            "balance_tol": float(args.balance_tol),
            "order_fail_match_window_ms": max(1, int(args.order_fail_match_window_ms)),
            "allow_compare_surface_artefacts": bool(args.allow_compare_surface_artefacts),
        },
        "counts": {
            **live_counts,
            "live_canonical_actions_pre_split_fill_collapse": int(live_counts.get("live_canonical_actions") or 0),
            "live_canonical_actions": len(live_actions),
            **live_split_fill_stats,
            **backtester_counts,
            **order_fail_counts,
            **compare_summary,
            "compare_surface_artefact_total": int(mismatch_breakdown["compare_surface_artefact_total"]),
            "logic_divergence_total": int(mismatch_breakdown["logic_divergence_total"]),
            "classification_reclassified_count": int(classification_reclassified_count),
            "scope_shared_symbols": len(scope_contract.get("shared_symbols") or []),
            "scope_shared_symbol_sides": len(scope_contract.get("shared_symbol_sides") or []),
            "mismatch_total": len(mismatches),
        },
        "status": {
            "strict_alignment_pass": strict_alignment_pass,
            "accepted_residuals_only": strict_alignment_pass and bool(accepted_residuals),
            "logic_divergence_free": bool(logic_divergence_free),
            "artefact_only_mismatch": bool(artefact_only_mismatch),
            "scope_contract_mismatch": bool(scope_contract.get("mismatch")),
            "gate_pass_strict_fail_closed": bool(strict_alignment_pass),
            "gate_pass_if_allow_compare_surface_artefacts": bool(gate_pass_if_allow_compare_surface_artefacts),
            "selected_gate_mode": selected_gate_mode,
            "selected_gate_pass": bool(selected_gate_pass),
            "allow_compare_surface_artefacts_enabled": bool(args.allow_compare_surface_artefacts),
        },
        "mismatch_counts_by_classification": dict(sorted(mismatch_counts.items(), key=lambda x: x[0])),
        "mismatch_counts_by_kind": dict(sorted(mismatch_kind_counts.items(), key=lambda x: x[0])),
        "mismatch_counts_by_action_code": dict(sorted(mismatch_action_code_counts.items(), key=lambda x: x[0])),
        "mismatch_breakdown": mismatch_breakdown,
        "funding_pair_evidence": _summarise_funding_evidence(mismatches),
        "scope_contract": scope_contract,
        "accepted_residuals": accepted_residuals,
        "per_symbol": per_symbol_rows,
        "mismatches": mismatches,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
