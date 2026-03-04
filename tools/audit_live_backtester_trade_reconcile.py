#!/usr/bin/env python3
"""Audit trade-level alignment between live baseline and backtester replay output.

This report compares only simulatable exit actions (CLOSE/REDUCE) and classifies
mismatches using the project alignment taxonomy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional runtime dependency
    yaml = None

SIMULATABLE_EXIT_ACTIONS = {"CLOSE", "REDUCE"}
NON_SIMULATABLE_ACTIONS = {"FUNDING"}
END_OF_BACKTEST_REASON_CODES = {"exit_end_of_backtest"}
LIVE_ENTRY_ACTIONS = {"OPEN", "ADD"}
BACKTESTER_ENTRY_ACTIONS = {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
DEFAULT_ENTRY_MIN_CONFIDENCE = "high"
DEFAULT_TOL = 1e-9


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit live-vs-backtester trade alignment from replay bundle artefacts.")
    parser.add_argument("--live-baseline", required=True, help="Path to live_baseline_trades.jsonl")
    parser.add_argument("--backtester-trades", required=True, help="Path to backtester_trades.csv")
    parser.add_argument(
        "--bundle-manifest",
        default=None,
        help="Optional replay_bundle_manifest.json path for execution-model assumptions.",
    )
    parser.add_argument(
        "--backtester-replay-report",
        default=None,
        help="Optional backtester replay JSON for policy-mismatch evidence (auto-discovered from manifest when omitted).",
    )
    parser.add_argument(
        "--strategy-config-snapshot",
        default=None,
        help="Optional locked strategy config YAML for entry confidence policy evidence (auto-discovered from manifest when omitted).",
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
    parser.add_argument("--size-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for exit_size comparison")
    parser.add_argument("--pnl-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for pnl comparison")
    parser.add_argument("--fee-tol", type=float, default=DEFAULT_TOL, help="Absolute tolerance for fee comparison")
    return parser


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    confidence = str(value or "").strip().lower()
    if confidence in CONFIDENCE_RANK:
        return confidence
    return ""


def _confidence_rank(value: Any) -> int:
    confidence = _normalise_confidence(value)
    if not confidence:
        return -1
    return int(CONFIDENCE_RANK[confidence])


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


def _resolve_optional_path(raw: str | None, *, base_dir: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _extract_side_from_backtester_action(action: Any) -> str:
    action_s = str(action or "").strip().upper()
    if action_s.endswith("_LONG"):
        return "LONG"
    if action_s.endswith("_SHORT"):
        return "SHORT"
    return ""


def _load_live_simulatable_exits(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exits: list[dict[str, Any]] = []

    total_actions = 0
    total_non_simulatable_actions = 0
    non_simulatable_pnl = 0.0
    info_only_actions = 0

    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            total_actions += 1

            action = str(row.get("action") or "").strip().upper()
            symbol = _normalise_symbol(row.get("symbol"))
            side = _normalise_side(row.get("type"))
            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms <= 0:
                ts_ms = _parse_timestamp_ms(row.get("timestamp"))

            if action in NON_SIMULATABLE_ACTIONS:
                total_non_simulatable_actions += 1
                non_simulatable_pnl += _parse_float(row.get("pnl"))
                continue

            if action not in SIMULATABLE_EXIT_ACTIONS:
                info_only_actions += 1
                continue

            if not symbol or not side or ts_ms <= 0:
                continue

            exits.append(
                {
                    "source": "live",
                    "source_id": int(row.get("id") or 0),
                    "line_no": line_no,
                    "symbol": symbol,
                    "side": side,
                    "exit_ts_ms": ts_ms,
                    "exit_size": _parse_float(row.get("size")),
                    "pnl_usd": _parse_float(row.get("pnl")),
                    "fee_usd": _parse_float(row.get("fee_usd")),
                    "action": action,
                }
            )

    exits.sort(key=lambda r: (r["exit_ts_ms"], r["symbol"], r["source_id"], r["line_no"]))

    counts = {
        "live_total_actions": total_actions,
        "live_simulatable_exit_actions": len(exits),
        "live_non_simulatable_actions": total_non_simulatable_actions,
        "live_non_simulatable_pnl_usd": non_simulatable_pnl,
        "live_info_only_actions": info_only_actions,
    }
    return exits, counts


def _load_live_entry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            action = str(row.get("action") or "").strip().upper()
            if action not in LIVE_ENTRY_ACTIONS:
                continue
            symbol = _normalise_symbol(row.get("symbol"))
            side = _normalise_side(row.get("type"))
            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms <= 0:
                ts_ms = _parse_timestamp_ms(row.get("timestamp"))
            if not symbol or not side or ts_ms <= 0:
                continue
            rows.append(
                {
                    "source_id": int(row.get("id") or 0),
                    "line_no": line_no,
                    "symbol": symbol,
                    "side": side,
                    "timestamp_ms": ts_ms,
                    "action": action,
                    "confidence": _normalise_confidence(row.get("confidence")),
                }
            )
    rows.sort(key=lambda r: (int(r["timestamp_ms"]), str(r["symbol"]), int(r["source_id"]), int(r["line_no"])))
    return rows


def _load_backtester_exits(path: Path) -> list[dict[str, Any]]:
    exits: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row_no, row in enumerate(reader, start=2):
            symbol = _normalise_symbol(row.get("symbol"))
            side = _normalise_side(row.get("side"))
            ts_ms = int(_parse_float(row.get("exit_ts_ms")))
            if not symbol or not side or ts_ms <= 0:
                continue
            exits.append(
                {
                    "source": "backtester",
                    "source_id": str(row.get("trade_id") or ""),
                    "row_no": row_no,
                    "symbol": symbol,
                    "side": side,
                    "exit_ts_ms": ts_ms,
                    "exit_size": _parse_float(row.get("exit_size")),
                    "pnl_usd": _parse_float(row.get("pnl_usd")),
                    "fee_usd": _parse_float(row.get("fee_usd")),
                    "reason_code": str(row.get("reason_code") or ""),
                    "reason": str(row.get("reason") or ""),
                }
            )
    exits.sort(key=lambda r: (r["exit_ts_ms"], r["symbol"], str(r["source_id"]), r["row_no"]))
    return exits


def _load_backtester_entry_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path)
    raw_rows = payload.get("trades")
    if not isinstance(raw_rows, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or "").strip().upper()
        if action not in BACKTESTER_ENTRY_ACTIONS:
            continue
        symbol = _normalise_symbol(row.get("symbol"))
        side = _extract_side_from_backtester_action(action)
        ts_ms = _parse_timestamp_ms(row.get("timestamp"))
        if not symbol or not side or ts_ms <= 0:
            continue
        rows.append(
            {
                "row_no": idx,
                "symbol": symbol,
                "side": side,
                "timestamp_ms": ts_ms,
                "action": action,
                "confidence": _normalise_confidence(row.get("confidence")),
                "reason_code": str(row.get("reason_code") or ""),
            }
        )

    rows.sort(key=lambda r: (int(r["timestamp_ms"]), str(r["symbol"]), int(r["row_no"])))
    return rows


def _extract_entry_min_confidence_from_cfg_block(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    trade_obj = payload.get("trade")
    if isinstance(trade_obj, dict):
        conf = _normalise_confidence(trade_obj.get("entry_min_confidence"))
        if conf:
            return conf
    return _normalise_confidence(payload.get("entry_min_confidence"))


def _load_locked_entry_confidence_policy(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or yaml is None:
        return {
            "strategy_config_snapshot": str(path) if path is not None else None,
            "policy_available": False,
            "global_min_confidence": "",
            "symbol_min_confidence": {},
            "policy_source": "unavailable",
            "used_runtime_default": False,
            "error": None,
        }

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed YAML path
        return {
            "strategy_config_snapshot": str(path),
            "policy_available": False,
            "global_min_confidence": "",
            "symbol_min_confidence": {},
            "policy_source": "yaml_parse_error",
            "used_runtime_default": False,
            "error": str(exc),
        }

    cfg = loaded if isinstance(loaded, dict) else {}
    global_min_confidence = _extract_entry_min_confidence_from_cfg_block(cfg)

    symbol_min_confidence: dict[str, str] = {}
    symbols_obj = cfg.get("symbols")
    if isinstance(symbols_obj, dict):
        for raw_symbol, symbol_cfg in symbols_obj.items():
            symbol = _normalise_symbol(raw_symbol)
            if not symbol:
                continue
            symbol_conf = _extract_entry_min_confidence_from_cfg_block(symbol_cfg)
            if symbol_conf:
                symbol_min_confidence[symbol] = symbol_conf

    used_runtime_default = False
    policy_source = "strategy_snapshot"
    if not global_min_confidence and not symbol_min_confidence:
        global_min_confidence = DEFAULT_ENTRY_MIN_CONFIDENCE
        used_runtime_default = True
        policy_source = "runtime_default"

    return {
        "strategy_config_snapshot": str(path),
        "policy_available": bool(global_min_confidence or symbol_min_confidence),
        "global_min_confidence": global_min_confidence,
        "symbol_min_confidence": dict(sorted(symbol_min_confidence.items(), key=lambda kv: kv[0])),
        "policy_source": policy_source,
        "used_runtime_default": bool(used_runtime_default),
        "error": None,
    }


def _confidence_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        conf = _normalise_confidence(row.get("confidence"))
        if conf:
            counts[conf] += 1
        else:
            counts["unknown"] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _apply_entry_confidence_policy_residual_classification(
    mismatches: list[dict[str, Any]],
    *,
    live_entry_rows: list[dict[str, Any]],
    backtester_entry_rows: list[dict[str, Any]],
    locked_entry_policy: dict[str, Any],
) -> dict[str, Any]:
    live_entries_by_symbol_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in live_entry_rows:
        live_entries_by_symbol_side[(str(row["symbol"]), str(row["side"]))].append(row)

    backtester_entries_by_symbol_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in backtester_entry_rows:
        backtester_entries_by_symbol_side[(str(row["symbol"]), str(row["side"]))].append(row)

    global_min_conf = _normalise_confidence((locked_entry_policy or {}).get("global_min_confidence"))
    symbol_min_conf = dict((locked_entry_policy or {}).get("symbol_min_confidence") or {})
    policy_available = bool((locked_entry_policy or {}).get("policy_available"))

    reclassified_count = 0
    sample_rows: list[dict[str, Any]] = []
    unresolved_missing_backtester_exit = 0

    for mismatch in mismatches:
        cls = str(mismatch.get("classification") or "").strip().lower()
        kind = str(mismatch.get("kind") or "").strip().lower()
        if cls != "deterministic_logic_divergence" or kind != "missing_backtester_exit":
            continue

        unresolved_missing_backtester_exit += 1

        symbol = _normalise_symbol(mismatch.get("symbol"))
        side = _normalise_side(mismatch.get("side"))
        if not symbol or not side:
            continue

        live_symbol_side_entries = live_entries_by_symbol_side.get((symbol, side), [])
        if not live_symbol_side_entries:
            continue
        backtester_symbol_side_entries = backtester_entries_by_symbol_side.get((symbol, side), [])
        if backtester_symbol_side_entries:
            continue
        if not policy_available:
            continue

        required_min_conf = _normalise_confidence(symbol_min_conf.get(symbol) or global_min_conf)
        required_min_rank = _confidence_rank(required_min_conf)
        if required_min_rank < 0:
            continue

        live_below_threshold = [
            row
            for row in live_symbol_side_entries
            if 0 <= _confidence_rank(row.get("confidence")) < required_min_rank
        ]
        if not live_below_threshold:
            continue

        backtester_below_threshold = [
            row
            for row in backtester_entry_rows
            if 0 <= _confidence_rank(row.get("confidence")) < required_min_rank
        ]
        if backtester_below_threshold:
            continue

        mismatch["classification"] = "policy_mismatch_residual"
        mismatch["policy_mismatch"] = {
            "kind": "entry_confidence_gate",
            "required_min_confidence": required_min_conf,
            "live_entries_below_threshold": len(live_below_threshold),
            "live_entry_confidences": sorted(
                {str(row.get("confidence") or "") for row in live_symbol_side_entries if str(row.get("confidence") or "")}
            ),
            "backtester_entries_symbol_side": len(backtester_symbol_side_entries),
            "backtester_global_entries_below_threshold": len(backtester_below_threshold),
            "evidence": {
                "live_entry_ids": [int(row.get("source_id") or 0) for row in live_below_threshold[:3]],
                "live_entry_lines": [int(row.get("line_no") or 0) for row in live_below_threshold[:3]],
            },
        }
        reclassified_count += 1
        unresolved_missing_backtester_exit -= 1
        if len(sample_rows) < 10:
            sample_rows.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "live_exit_ts_ms": int(mismatch.get("live_exit_ts_ms") or 0),
                    "required_min_confidence": required_min_conf,
                    "live_entry_confidences": mismatch["policy_mismatch"]["live_entry_confidences"],
                    "live_entry_ids": mismatch["policy_mismatch"]["evidence"]["live_entry_ids"],
                }
            )

    evidence_complete = bool(
        reclassified_count > 0
        and policy_available
        and bool(global_min_conf or symbol_min_conf)
        and bool(backtester_entry_rows)
        and bool(sample_rows)
    )

    return {
        "detected": bool(reclassified_count > 0),
        "kind": "entry_confidence_gate",
        "evidence_complete": bool(evidence_complete),
        "reclassified_mismatch_count": int(reclassified_count),
        "unresolved_missing_backtester_exit_count": int(max(0, unresolved_missing_backtester_exit)),
        "locked_entry_policy": {
            "strategy_config_snapshot": (locked_entry_policy or {}).get("strategy_config_snapshot"),
            "policy_available": bool(policy_available),
            "global_min_confidence": global_min_conf,
            "symbol_min_confidence": dict(sorted(symbol_min_conf.items(), key=lambda kv: kv[0])),
            "policy_source": (locked_entry_policy or {}).get("policy_source"),
            "used_runtime_default": bool((locked_entry_policy or {}).get("used_runtime_default")),
            "error": (locked_entry_policy or {}).get("error"),
        },
        "live_entry_confidence_counts": _confidence_counts(live_entry_rows),
        "backtester_entry_confidence_counts": _confidence_counts(backtester_entry_rows),
        "samples": sample_rows,
    }


def _build_event_groups(
    rows: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
    timestamp_bucket_anchor: str,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["symbol"],
            row["side"],
            _bucket_timestamp_ms(int(row["exit_ts_ms"]), timestamp_bucket_ms, timestamp_bucket_anchor),
        )
        grouped[key].append(row)

    for values in grouped.values():
        values.sort(
            key=lambda r: (
                float(r["exit_size"]),
                float(r["pnl_usd"]),
                float(r["fee_usd"]),
                str(r["source_id"]),
            )
        )

    return grouped


def _compare_exits(
    live_exits: list[dict[str, Any]],
    backtester_exits: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
    timestamp_bucket_anchor: str,
    size_tol: float,
    pnl_tol: float,
    fee_tol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    matched_pairs = 0

    per_symbol: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "live_exit_count": 0,
            "backtester_exit_count": 0,
            "matched_pairs": 0,
            "live_pnl_usd": 0.0,
            "backtester_pnl_usd": 0.0,
            "live_fee_usd": 0.0,
            "backtester_fee_usd": 0.0,
        }
    )

    for row in live_exits:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["live_exit_count"] += 1
        stats["live_pnl_usd"] += float(row["pnl_usd"])
        stats["live_fee_usd"] += float(row["fee_usd"])

    for row in backtester_exits:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["backtester_exit_count"] += 1
        stats["backtester_pnl_usd"] += float(row["pnl_usd"])
        stats["backtester_fee_usd"] += float(row["fee_usd"])

    live_groups = _build_event_groups(
        live_exits,
        timestamp_bucket_ms=timestamp_bucket_ms,
        timestamp_bucket_anchor=timestamp_bucket_anchor,
    )
    bt_groups = _build_event_groups(
        backtester_exits,
        timestamp_bucket_ms=timestamp_bucket_ms,
        timestamp_bucket_anchor=timestamp_bucket_anchor,
    )
    all_keys = sorted(set(live_groups.keys()) | set(bt_groups.keys()))

    unmatched_live = 0
    unmatched_backtester = 0
    numeric_mismatch = 0
    state_scope_residuals = 0

    for key in all_keys:
        symbol, side, ts_ms = key
        live_rows = live_groups.get(key, [])
        bt_rows = bt_groups.get(key, [])

        pair_count = min(len(live_rows), len(bt_rows))
        for idx in range(pair_count):
            lrow = live_rows[idx]
            brow = bt_rows[idx]
            per_symbol[symbol]["matched_pairs"] += 1
            matched_pairs += 1

            size_match = _almost_equal(lrow["exit_size"], brow["exit_size"], size_tol)
            pnl_match = _almost_equal(lrow["pnl_usd"], brow["pnl_usd"], pnl_tol)
            fee_match = _almost_equal(lrow["fee_usd"], brow["fee_usd"], fee_tol)
            if size_match and pnl_match and fee_match:
                continue

            numeric_mismatch += 1
            mismatches.append(
                {
                    "classification": "numeric_policy_divergence",
                    "kind": "exit_numeric_mismatch",
                    "symbol": symbol,
                    "side": side,
                    "match_key_exit_ts_ms": ts_ms,
                    "live_exit_ts_ms": lrow["exit_ts_ms"],
                    "backtester_exit_ts_ms": brow["exit_ts_ms"],
                    "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                    "backtester_ref": {"trade_id": brow["source_id"], "row_no": brow["row_no"]},
                    "live": {
                        "exit_size": lrow["exit_size"],
                        "pnl_usd": lrow["pnl_usd"],
                        "fee_usd": lrow["fee_usd"],
                        "action": lrow["action"],
                    },
                    "backtester": {
                        "exit_size": brow["exit_size"],
                        "pnl_usd": brow["pnl_usd"],
                        "fee_usd": brow["fee_usd"],
                        "reason_code": brow["reason_code"],
                    },
                    "delta": {
                        "exit_size": lrow["exit_size"] - brow["exit_size"],
                        "pnl_usd": lrow["pnl_usd"] - brow["pnl_usd"],
                        "fee_usd": lrow["fee_usd"] - brow["fee_usd"],
                    },
                    "tolerance": {
                        "size_tol": size_tol,
                        "pnl_tol": pnl_tol,
                        "fee_tol": fee_tol,
                    },
                }
            )

        if len(live_rows) > pair_count:
            for lrow in live_rows[pair_count:]:
                unmatched_live += 1
                mismatches.append(
                    {
                        "classification": "deterministic_logic_divergence",
                        "kind": "missing_backtester_exit",
                        "symbol": symbol,
                        "side": side,
                        "match_key_exit_ts_ms": ts_ms,
                        "live_exit_ts_ms": lrow["exit_ts_ms"],
                        "live_ref": {"id": lrow["source_id"], "line_no": lrow["line_no"]},
                        "live": {
                            "exit_size": lrow["exit_size"],
                            "pnl_usd": lrow["pnl_usd"],
                            "fee_usd": lrow["fee_usd"],
                            "action": lrow["action"],
                        },
                    }
                )

        if len(bt_rows) > pair_count:
            for brow in bt_rows[pair_count:]:
                reason_code = str(brow.get("reason_code") or "").strip().lower()
                if reason_code in END_OF_BACKTEST_REASON_CODES:
                    state_scope_residuals += 1
                    mismatches.append(
                        {
                            "classification": "state_initialisation_gap",
                            "kind": "missing_live_exit_end_of_backtest",
                            "symbol": symbol,
                            "side": side,
                            "match_key_exit_ts_ms": ts_ms,
                            "backtester_exit_ts_ms": brow["exit_ts_ms"],
                            "backtester_ref": {"trade_id": brow["source_id"], "row_no": brow["row_no"]},
                            "backtester": {
                                "exit_size": brow["exit_size"],
                                "pnl_usd": brow["pnl_usd"],
                                "fee_usd": brow["fee_usd"],
                                "reason_code": brow["reason_code"],
                            },
                        }
                    )
                    continue

                unmatched_backtester += 1
                mismatches.append(
                    {
                        "classification": "deterministic_logic_divergence",
                        "kind": "missing_live_exit",
                        "symbol": symbol,
                        "side": side,
                        "match_key_exit_ts_ms": ts_ms,
                        "backtester_exit_ts_ms": brow["exit_ts_ms"],
                        "backtester_ref": {"trade_id": brow["source_id"], "row_no": brow["row_no"]},
                        "backtester": {
                            "exit_size": brow["exit_size"],
                            "pnl_usd": brow["pnl_usd"],
                            "fee_usd": brow["fee_usd"],
                            "reason_code": brow["reason_code"],
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
        "unmatched_live": unmatched_live,
        "unmatched_backtester": unmatched_backtester,
        "state_scope_residuals": state_scope_residuals,
    }
    return mismatches, summary, per_symbol_rows


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_baseline = Path(args.live_baseline).expanduser().resolve()
    backtester_trades = Path(args.backtester_trades).expanduser().resolve()
    backtester_replay_report_override = (
        Path(args.backtester_replay_report).expanduser().resolve() if args.backtester_replay_report else None
    )
    strategy_config_snapshot_override = (
        Path(args.strategy_config_snapshot).expanduser().resolve() if args.strategy_config_snapshot else None
    )
    bundle_manifest = Path(args.bundle_manifest).expanduser().resolve() if args.bundle_manifest else None
    output = Path(args.output).expanduser().resolve()

    if not live_baseline.exists():
        parser.error(f"live baseline not found: {live_baseline}")
    if not backtester_trades.exists():
        parser.error(f"backtester trades not found: {backtester_trades}")
    if backtester_replay_report_override is not None and not backtester_replay_report_override.exists():
        parser.error(f"backtester replay report not found: {backtester_replay_report_override}")
    if strategy_config_snapshot_override is not None and not strategy_config_snapshot_override.exists():
        parser.error(f"strategy config snapshot not found: {strategy_config_snapshot_override}")
    if bundle_manifest is not None and not bundle_manifest.exists():
        parser.error(f"bundle manifest not found: {bundle_manifest}")

    manifest_obj: dict[str, Any] = {}
    alignment_assumptions: dict[str, Any] = {}
    bbo_fill_model_enabled = False
    backtester_replay_report_path = backtester_replay_report_override
    strategy_config_snapshot_path = strategy_config_snapshot_override
    if bundle_manifest is not None:
        manifest_obj = _load_json_object(bundle_manifest)
        raw_assumptions = manifest_obj.get("alignment_assumptions")
        if isinstance(raw_assumptions, dict):
            alignment_assumptions = raw_assumptions
            bbo_obj = raw_assumptions.get("bbo_fill_model")
            if isinstance(bbo_obj, dict):
                bbo_fill_model_enabled = bool(bbo_obj.get("enabled_any"))
        artefacts = manifest_obj.get("artefacts")
        if isinstance(artefacts, dict):
            if backtester_replay_report_path is None:
                backtester_replay_report_path = _resolve_optional_path(
                    artefacts.get("backtester_replay_report_json"),
                    base_dir=bundle_manifest.parent,
                )
            if strategy_config_snapshot_path is None:
                strategy_config_snapshot_path = _resolve_optional_path(
                    artefacts.get("strategy_config_snapshot_file"),
                    base_dir=bundle_manifest.parent,
                )

    if backtester_replay_report_path is not None and not backtester_replay_report_path.exists():
        parser.error(f"backtester replay report not found: {backtester_replay_report_path}")
    if strategy_config_snapshot_path is not None and not strategy_config_snapshot_path.exists():
        parser.error(f"strategy config snapshot not found: {strategy_config_snapshot_path}")

    live_exits, live_counts = _load_live_simulatable_exits(live_baseline)
    live_entry_rows = _load_live_entry_rows(live_baseline)
    backtester_exits = _load_backtester_exits(backtester_trades)
    backtester_entry_rows = (
        _load_backtester_entry_rows(backtester_replay_report_path) if backtester_replay_report_path is not None else []
    )
    locked_entry_policy = _load_locked_entry_confidence_policy(strategy_config_snapshot_path)

    mismatches, compare_summary, per_symbol_rows = _compare_exits(
        live_exits,
        backtester_exits,
        timestamp_bucket_ms=max(1, int(args.timestamp_bucket_ms)),
        timestamp_bucket_anchor=str(args.timestamp_bucket_anchor or "floor"),
        size_tol=float(args.size_tol),
        pnl_tol=float(args.pnl_tol),
        fee_tol=float(args.fee_tol),
    )

    policy_mismatch_analysis = _apply_entry_confidence_policy_residual_classification(
        mismatches,
        live_entry_rows=live_entry_rows,
        backtester_entry_rows=backtester_entry_rows,
        locked_entry_policy=locked_entry_policy,
    )

    mismatch_counts: dict[str, int] = defaultdict(int)
    for item in mismatches:
        mismatch_counts[str(item.get("classification") or "unknown")] += 1
    policy_mismatch_residuals = [
        row for row in mismatches if str(row.get("classification") or "").strip().lower() == "policy_mismatch_residual"
    ]
    deterministic_unexplained = int(mismatch_counts.get("deterministic_logic_divergence", 0))

    non_simulatable_residuals: list[dict[str, Any]] = []
    if live_counts["live_non_simulatable_actions"] > 0:
        non_simulatable_residuals.append(
            {
                "classification": "non-simulatable_exchange_oms_effect",
                "kind": "funding_events_not_in_replay_export",
                "count": int(live_counts["live_non_simulatable_actions"]),
                "pnl_usd": float(live_counts["live_non_simulatable_pnl_usd"]),
            }
        )
    if bbo_fill_model_enabled:
        non_simulatable_residuals.append(
            {
                "classification": "non-simulatable_exchange_oms_effect",
                "kind": "bbo_fill_model_execution_assumption",
                "detail": (
                    "bundle manifest reports bbo_fill_model.enabled_any=true; "
                    "numeric price/PnL drift may include execution-model residuals"
                ),
                "numeric_mismatch_count": int(compare_summary["numeric_mismatch"]),
            }
        )
    accepted_residuals = list(non_simulatable_residuals)
    accepted_residuals.extend(
        [m for m in mismatches if str(m.get("classification") or "") == "state_initialisation_gap"]
    )

    strict_alignment_pass = (
        compare_summary["numeric_mismatch"] == 0
        and compare_summary["unmatched_live"] == 0
        and compare_summary["unmatched_backtester"] == 0
    )
    policy_mismatch_residual_only = (
        not strict_alignment_pass
        and compare_summary["numeric_mismatch"] == 0
        and deterministic_unexplained == 0
        and len(policy_mismatch_residuals) > 0
        and (len(policy_mismatch_residuals) + int(compare_summary["state_scope_residuals"]) == len(mismatches))
    )

    report = {
        "schema_version": 1,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "inputs": {
            "live_baseline": str(live_baseline),
            "backtester_trades": str(backtester_trades),
            "backtester_replay_report": str(backtester_replay_report_path) if backtester_replay_report_path else None,
            "strategy_config_snapshot": str(strategy_config_snapshot_path) if strategy_config_snapshot_path else None,
            "bundle_manifest": str(bundle_manifest) if bundle_manifest is not None else None,
            "timestamp_bucket_ms": max(1, int(args.timestamp_bucket_ms)),
            "timestamp_bucket_anchor": str(args.timestamp_bucket_anchor or "floor"),
            "size_tol": float(args.size_tol),
            "pnl_tol": float(args.pnl_tol),
            "fee_tol": float(args.fee_tol),
        },
        "counts": {
            **live_counts,
            "backtester_exit_rows": len(backtester_exits),
            **compare_summary,
            "policy_mismatch_residuals": len(policy_mismatch_residuals),
            "deterministic_unexplained": deterministic_unexplained,
            "mismatch_total": len(mismatches),
        },
        "status": {
            "strict_alignment_pass": strict_alignment_pass,
            "policy_mismatch_residual_only": bool(policy_mismatch_residual_only),
            "accepted_residuals_only": strict_alignment_pass and bool(accepted_residuals),
        },
        "execution_model_assumptions": {
            "bundle_manifest_present": bundle_manifest is not None,
            "bbo_fill_model_enabled": bool(bbo_fill_model_enabled),
            "alignment_assumptions": alignment_assumptions,
        },
        "policy_mismatch_analysis": policy_mismatch_analysis,
        "mismatch_counts_by_classification": dict(sorted(mismatch_counts.items(), key=lambda x: x[0])),
        "policy_mismatch_residuals": policy_mismatch_residuals,
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
