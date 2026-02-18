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
from typing import Any

from reason_codes import classify_reason_code

SIDE_ACTIONS = {"OPEN", "ADD", "REDUCE", "CLOSE"}
FUNDING_ACTION = "FUNDING"
DEFAULT_TOL = 1e-9


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit live-vs-backtester action-level parity from replay bundle artefacts.")
    parser.add_argument("--live-baseline", required=True, help="Path to live_baseline_trades.jsonl")
    parser.add_argument("--backtester-replay-report", required=True, help="Path to backtester replay JSON report (with trades)")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument(
        "--timestamp-bucket-ms",
        type=int,
        default=1,
        help="Match timestamp bucket in milliseconds (1 = exact millisecond match)",
    )
    parser.add_argument("--price-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for price comparison")
    parser.add_argument("--size-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for size comparison")
    parser.add_argument("--pnl-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for pnl comparison")
    parser.add_argument("--fee-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for fee comparison")
    parser.add_argument("--balance-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for balance comparison")
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


def _almost_equal(left: float, right: float, tol: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def _bucket_timestamp_ms(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 1:
        return int(ts_ms)
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
                    "reason_code": classify_reason_code(action_code, str(row.get("reason") or "")),
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


def _group_events(
    rows: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["symbol"], row["action_code"], _bucket_timestamp_ms(int(row["timestamp_ms"]), timestamp_bucket_ms))
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
    *,
    timestamp_bucket_ms: int,
    price_tol: float,
    size_tol: float,
    pnl_tol: float,
    fee_tol: float,
    balance_tol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    matched_pairs = 0
    numeric_mismatch = 0
    confidence_mismatch = 0
    reason_code_mismatch = 0
    unmatched_live = 0
    unmatched_backtester = 0
    non_simulatable_residuals = 0

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

    live_groups = _group_events(live_actions, timestamp_bucket_ms=timestamp_bucket_ms)
    bt_groups = _group_events(backtester_actions, timestamp_bucket_ms=timestamp_bucket_ms)
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
                    non_simulatable_residuals += 1
                    mismatches.append(
                        {
                            "classification": "non-simulatable_exchange_oms_effect",
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
                    non_simulatable_residuals += 1
                    mismatches.append(
                        {
                            "classification": "non-simulatable_exchange_oms_effect",
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
    }
    return mismatches, summary, per_symbol_rows


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_baseline = Path(args.live_baseline).expanduser().resolve()
    backtester_report = Path(args.backtester_replay_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not live_baseline.exists():
        parser.error(f"live baseline not found: {live_baseline}")
    if not backtester_report.exists():
        parser.error(f"backtester replay report not found: {backtester_report}")

    live_actions, live_counts = _load_live_actions(live_baseline)
    try:
        backtester_actions, backtester_counts = _load_backtester_actions(backtester_report)
    except ValueError as exc:
        parser.error(str(exc))

    mismatches, compare_summary, per_symbol_rows = _compare_actions(
        live_actions,
        backtester_actions,
        timestamp_bucket_ms=max(1, int(args.timestamp_bucket_ms)),
        price_tol=float(args.price_tol),
        size_tol=float(args.size_tol),
        pnl_tol=float(args.pnl_tol),
        fee_tol=float(args.fee_tol),
        balance_tol=float(args.balance_tol),
    )

    mismatch_counts: dict[str, int] = defaultdict(int)
    for item in mismatches:
        mismatch_counts[str(item.get("classification") or "unknown")] += 1

    accepted_residuals = [m for m in mismatches if m.get("classification") == "non-simulatable_exchange_oms_effect"]

    strict_alignment_pass = (
        compare_summary["numeric_mismatch"] == 0
        and compare_summary["confidence_mismatch"] == 0
        and compare_summary["reason_code_mismatch"] == 0
        and compare_summary["unmatched_live"] == 0
        and compare_summary["unmatched_backtester"] == 0
    )

    report = {
        "schema_version": 1,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "inputs": {
            "live_baseline": str(live_baseline),
            "backtester_replay_report": str(backtester_report),
            "timestamp_bucket_ms": max(1, int(args.timestamp_bucket_ms)),
            "price_tol": float(args.price_tol),
            "size_tol": float(args.size_tol),
            "pnl_tol": float(args.pnl_tol),
            "fee_tol": float(args.fee_tol),
            "balance_tol": float(args.balance_tol),
        },
        "counts": {
            **live_counts,
            **backtester_counts,
            **compare_summary,
            "mismatch_total": len(mismatches),
        },
        "status": {
            "strict_alignment_pass": strict_alignment_pass,
            "accepted_residuals_only": strict_alignment_pass and bool(accepted_residuals),
        },
        "mismatch_counts_by_classification": dict(sorted(mismatch_counts.items(), key=lambda x: x[0])),
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
