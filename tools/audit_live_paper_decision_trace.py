#!/usr/bin/env python3
"""Audit decision-event trace alignment between live and paper SQLite databases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit live-vs-paper decision_events trace parity.")
    parser.add_argument("--live-db", required=True, help="Path to live SQLite DB")
    parser.add_argument("--paper-db", required=True, help="Path to paper SQLite DB")
    parser.add_argument(
        "--paper-min-id-exclusive",
        type=int,
        help="Optional lower-bound filter: include only paper decision rows with trade_id > this value.",
    )
    parser.add_argument(
        "--paper-seed-watermark",
        help=(
            "Optional JSON watermark file generated during seed step "
            "(expects key pre_seed_max_trade_id)."
        ),
    )
    parser.add_argument(
        "--bundle-manifest",
        help=(
            "Optional replay bundle manifest. When set with "
            "--require-single-run-fingerprint, enforce one live run_fingerprint "
            "within the audited window."
        ),
    )
    parser.add_argument(
        "--require-single-run-fingerprint",
        action="store_true",
        default=False,
        help="Fail-closed when manifest.live_run_fingerprint_provenance reports drift within the window.",
    )
    parser.add_argument(
        "--max-live-run-fingerprint-distinct",
        type=int,
        default=1,
        help="Maximum allowed distinct live run_fingerprint in manifest provenance window.",
    )
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument("--from-ts", type=int, help="Filter start timestamp (ms, inclusive)")
    parser.add_argument("--to-ts", type=int, help="Filter end timestamp (ms, inclusive)")
    parser.add_argument(
        "--timestamp-bucket-ms",
        type=int,
        default=1,
        help="Decision event match bucket in milliseconds (1 = exact millisecond match)",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true", default=False, help="Return exit code 1 on strict mismatch")
    parser.add_argument(
        "--allow-paper-preseed-unlinked-residuals",
        action="store_true",
        default=False,
        help=(
            "Treat paper_preseed_decision_rows_out_of_scope residuals as non-blocking. "
            "Default is fail-closed."
        ),
    )
    parser.add_argument(
        "--include-runtime-only-blocked",
        action="store_true",
        default=False,
        help=(
            "Include runtime-only blocked decisions (reason_code=execution_would) in parity comparison. "
            "Default excludes them because backtester has no equivalent."
        ),
    )
    parser.add_argument(
        "--include-funding-events",
        action="store_true",
        default=False,
        help=(
            "Include funding decision events in parity comparison. Default excludes them because "
            "live decision_events currently does not emit funding rows."
        ),
    )
    return parser


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for row in rows:
        if str(row["name"] if isinstance(row, sqlite3.Row) else row[1]).strip() == column_name:
            return True
    return False


def _bucket_timestamp_ms(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 1:
        return int(ts_ms)
    if ts_ms >= 0:
        return int((ts_ms // bucket_ms) * bucket_ms)
    return int(-(((-ts_ms) // bucket_ms) * bucket_ms))


def _norm_text(value: Any, *, lower: bool = True) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.lower() if lower else text


def _resolve_paper_min_id_exclusive(
    *,
    raw_min_id: int | None,
    watermark_path_raw: str | None,
) -> int | None:
    resolved: int | None = int(raw_min_id) if raw_min_id is not None else None
    if not watermark_path_raw:
        return resolved

    watermark_path = Path(str(watermark_path_raw)).expanduser().resolve()
    if not watermark_path.exists():
        raise ValueError(f"paper seed watermark file not found: {watermark_path}")

    try:
        payload = json.loads(watermark_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid paper seed watermark JSON: {watermark_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid paper seed watermark payload: {watermark_path}")

    raw_watermark = payload.get("pre_seed_max_trade_id")
    if raw_watermark is None:
        raise ValueError(
            f"paper seed watermark missing pre_seed_max_trade_id: {watermark_path}"
        )
    try:
        watermark_value = int(raw_watermark)
    except Exception as exc:
        raise ValueError(
            f"invalid pre_seed_max_trade_id in paper seed watermark: {watermark_path}"
        ) from exc

    if resolved is None:
        return int(watermark_value)
    return max(int(resolved), int(watermark_value))


def _load_manifest_run_fingerprint_provenance(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundle manifest must be a JSON object")
    provenance = payload.get("live_run_fingerprint_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("manifest.live_run_fingerprint_provenance is missing")
    return provenance


def _run_fingerprint_guard_issue(
    *,
    detail: str,
    run_fingerprint_distinct: int | None = None,
    max_allowed: int | None = None,
    rows_sampled: int | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "classification": "state_initialisation_gap",
        "kind": "live_run_fingerprint_drift_within_window",
        "detail": detail,
    }
    if run_fingerprint_distinct is not None:
        issue["run_fingerprint_distinct"] = int(run_fingerprint_distinct)
    if max_allowed is not None:
        issue["max_live_run_fingerprint_distinct"] = int(max_allowed)
    if rows_sampled is not None:
        issue["rows_sampled"] = int(rows_sampled)
    return issue


def _load_decision_rows(
    db_path: Path,
    *,
    source_name: str,
    from_ts: int | None,
    to_ts: int | None,
    include_runtime_only_blocked: bool,
    include_funding_events: bool,
    paper_min_trade_id_exclusive: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    conn = _connect_ro(db_path)
    try:
        if not _table_exists(conn, "decision_events"):
            return [], {"table_present": False, "row_count": 0}, [
                {
                    "classification": "state_initialisation_gap",
                    "kind": "missing_decision_events_table",
                    "source": source_name,
                    "db_path": str(db_path),
                }
            ]
        config_col = (
            "config_fingerprint"
            if _column_exists(conn, "decision_events", "config_fingerprint")
            else "NULL AS config_fingerprint"
        )
        reason_code_col = (
            "reason_code"
            if _column_exists(conn, "decision_events", "reason_code")
            else "NULL AS reason_code"
        )

        clauses: list[str] = []
        params: list[Any] = []
        if from_ts is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(int(from_ts))
        if to_ts is not None:
            clauses.append("timestamp_ms <= ?")
            params.append(int(to_ts))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        sql = (
            "SELECT id, timestamp_ms, symbol, event_type, status, decision_phase, "
            f"triggered_by, action_taken, rejection_reason, {reason_code_col}, {config_col}, trade_id "
            f"FROM decision_events {where_sql} ORDER BY timestamp_ms ASC, id ASC"
        )
        rows = conn.execute(sql, tuple(params)).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    funding_excluded = 0
    runtime_only_excluded = 0
    filtered_by_trade_id = 0
    for row in rows:
        ts_ms = int(row["timestamp_ms"] or 0)
        symbol = _norm_text(row["symbol"], lower=False).upper()
        event_type = _norm_text(row["event_type"])
        if ts_ms <= 0 or not symbol:
            continue
        if not include_funding_events and event_type == "funding":
            funding_excluded += 1
            continue

        trade_id_raw = row["trade_id"]
        trade_id_value: int | None = None
        if trade_id_raw is not None:
            try:
                trade_id_value = int(trade_id_raw)
            except Exception:
                trade_id_value = None
        if (
            source_name == "paper"
            and paper_min_trade_id_exclusive is not None
            and trade_id_value is not None
            and trade_id_value <= int(paper_min_trade_id_exclusive)
        ):
            filtered_by_trade_id += 1
            continue

        status = _norm_text(row["status"])
        reason_code = _norm_text(row["reason_code"])
        rejection_reason = _norm_text(row["rejection_reason"], lower=False)
        rejection_reason_l = _norm_text(rejection_reason)
        runtime_only_blocked = bool(
            status == "blocked"
            and (
                reason_code == "execution_would"
                or (not reason_code and rejection_reason_l.startswith("would_send:"))
            )
        )
        if (
            not include_runtime_only_blocked
            and runtime_only_blocked
        ):
            runtime_only_excluded += 1
            continue

        out.append(
            {
                "source": source_name,
                "source_id": _norm_text(row["id"], lower=False),
                "timestamp_ms": ts_ms,
                "symbol": symbol,
                "event_type": event_type,
                "status": status,
                "decision_phase": _norm_text(row["decision_phase"]),
                "triggered_by": _norm_text(row["triggered_by"]),
                "action_taken": _norm_text(row["action_taken"]),
                "rejection_reason": rejection_reason,
                "reason_code": reason_code,
                "config_fingerprint": _norm_text(row["config_fingerprint"]),
                "trade_linked": trade_id_value is not None,
                "trade_id": trade_id_value,
            }
        )

    counts = {
        "table_present": True,
        "row_count": len(out),
        "funding_excluded": int(funding_excluded),
        "runtime_only_excluded": int(runtime_only_excluded),
        "filtered_by_trade_id": int(filtered_by_trade_id),
    }
    return out, counts, []


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
) -> dict[tuple[str, str, str, str, str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["symbol"],
            row["event_type"],
            row["status"],
            row["decision_phase"],
            row["action_taken"],
            row["triggered_by"],
            _bucket_timestamp_ms(int(row["timestamp_ms"]), timestamp_bucket_ms),
        )
        grouped[key].append(row)

    for items in grouped.values():
        items.sort(key=lambda r: (r["rejection_reason"], int(bool(r["trade_linked"])), r["source_id"]))
    return grouped


def _compare_rows(
    live_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    *,
    timestamp_bucket_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    matched_pairs = 0
    rejection_reason_mismatch = 0
    config_fingerprint_mismatch = 0
    trade_linkage_mismatch = 0
    unmatched_live = 0
    unmatched_paper = 0

    per_symbol: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "live_rows": 0,
            "paper_rows": 0,
            "matched_pairs": 0,
        }
    )

    for row in live_rows:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["live_rows"] += 1

    for row in paper_rows:
        stats = per_symbol[row["symbol"]]
        stats["symbol"] = row["symbol"]
        stats["paper_rows"] += 1

    live_groups = _group_rows(live_rows, timestamp_bucket_ms=timestamp_bucket_ms)
    paper_groups = _group_rows(paper_rows, timestamp_bucket_ms=timestamp_bucket_ms)
    all_keys = sorted(set(live_groups.keys()) | set(paper_groups.keys()))

    for key in all_keys:
        symbol, event_type, status, phase, action_taken, triggered_by, match_ts = key
        lrows = live_groups.get(key, [])
        prows = paper_groups.get(key, [])
        pair_count = min(len(lrows), len(prows))

        for idx in range(pair_count):
            lrow = lrows[idx]
            prow = prows[idx]
            matched_pairs += 1
            per_symbol[symbol]["matched_pairs"] += 1

            reason_l = _norm_text(lrow.get("rejection_reason"), lower=False)
            reason_p = _norm_text(prow.get("rejection_reason"), lower=False)
            if reason_l and reason_p and reason_l != reason_p:
                rejection_reason_mismatch += 1
                mismatches.append(
                    {
                        "classification": "deterministic_logic_divergence",
                        "kind": "decision_rejection_reason_mismatch",
                        "symbol": symbol,
                        "event_type": event_type,
                        "status": status,
                        "decision_phase": phase,
                        "action_taken": action_taken,
                        "triggered_by": triggered_by,
                        "match_key_timestamp_ms": match_ts,
                        "live_timestamp_ms": lrow["timestamp_ms"],
                        "paper_timestamp_ms": prow["timestamp_ms"],
                        "live_ref": {"id": lrow["source_id"]},
                        "paper_ref": {"id": prow["source_id"]},
                        "live_rejection_reason": reason_l,
                        "paper_rejection_reason": reason_p,
                    }
                )

            linked_l = bool(lrow.get("trade_linked"))
            linked_p = bool(prow.get("trade_linked"))
            if linked_l != linked_p:
                trade_linkage_mismatch += 1
                mismatches.append(
                    {
                        "classification": "state_initialisation_gap",
                        "kind": "decision_trade_linkage_mismatch",
                        "symbol": symbol,
                        "event_type": event_type,
                        "status": status,
                        "decision_phase": phase,
                        "action_taken": action_taken,
                        "triggered_by": triggered_by,
                        "match_key_timestamp_ms": match_ts,
                        "live_timestamp_ms": lrow["timestamp_ms"],
                        "paper_timestamp_ms": prow["timestamp_ms"],
                        "live_ref": {"id": lrow["source_id"]},
                        "paper_ref": {"id": prow["source_id"]},
                        "live_trade_linked": linked_l,
                        "paper_trade_linked": linked_p,
                    }
                )

            cfg_l = _norm_text(lrow.get("config_fingerprint"))
            cfg_p = _norm_text(prow.get("config_fingerprint"))
            if cfg_l != cfg_p:
                config_fingerprint_mismatch += 1
                mismatches.append(
                    {
                        "classification": "state_initialisation_gap",
                        "kind": "decision_config_fingerprint_mismatch",
                        "symbol": symbol,
                        "event_type": event_type,
                        "status": status,
                        "decision_phase": phase,
                        "action_taken": action_taken,
                        "triggered_by": triggered_by,
                        "match_key_timestamp_ms": match_ts,
                        "live_timestamp_ms": lrow["timestamp_ms"],
                        "paper_timestamp_ms": prow["timestamp_ms"],
                        "live_ref": {"id": lrow["source_id"]},
                        "paper_ref": {"id": prow["source_id"]},
                        "live_config_fingerprint": cfg_l,
                        "paper_config_fingerprint": cfg_p,
                    }
                )

        if len(lrows) > pair_count:
            for lrow in lrows[pair_count:]:
                unmatched_live += 1
                mismatches.append(
                    {
                        "classification": "deterministic_logic_divergence",
                        "kind": "missing_paper_decision_event",
                        "symbol": symbol,
                        "event_type": event_type,
                        "status": status,
                        "decision_phase": phase,
                        "action_taken": action_taken,
                        "triggered_by": triggered_by,
                        "match_key_timestamp_ms": match_ts,
                        "live_timestamp_ms": lrow["timestamp_ms"],
                        "live_ref": {"id": lrow["source_id"]},
                    }
                )

        if len(prows) > pair_count:
            for prow in prows[pair_count:]:
                unmatched_paper += 1
                mismatches.append(
                    {
                        "classification": "deterministic_logic_divergence",
                        "kind": "missing_live_decision_event",
                        "symbol": symbol,
                        "event_type": event_type,
                        "status": status,
                        "decision_phase": phase,
                        "action_taken": action_taken,
                        "triggered_by": triggered_by,
                        "match_key_timestamp_ms": match_ts,
                        "paper_timestamp_ms": prow["timestamp_ms"],
                        "paper_ref": {"id": prow["source_id"]},
                        "paper_trade_linked": bool(prow.get("trade_linked")),
                        "paper_trade_id": prow.get("trade_id"),
                    }
                )

    summary = {
        "matched_pairs": matched_pairs,
        "rejection_reason_mismatch": rejection_reason_mismatch,
        "config_fingerprint_mismatch": config_fingerprint_mismatch,
        "trade_linkage_mismatch": trade_linkage_mismatch,
        "unmatched_live": unmatched_live,
        "unmatched_paper": unmatched_paper,
    }
    per_symbol_rows = sorted(per_symbol.values(), key=lambda row: row["symbol"])
    return mismatches, summary, per_symbol_rows


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_db = Path(args.live_db).expanduser().resolve()
    paper_db = Path(args.paper_db).expanduser().resolve()
    bundle_manifest = Path(args.bundle_manifest).expanduser().resolve() if args.bundle_manifest else None
    output = Path(args.output).expanduser().resolve()
    from_ts = int(args.from_ts) if args.from_ts is not None else None
    to_ts = int(args.to_ts) if args.to_ts is not None else None
    timestamp_bucket_ms = max(1, int(args.timestamp_bucket_ms))
    include_runtime_only_blocked = bool(args.include_runtime_only_blocked)
    include_funding_events = bool(args.include_funding_events)
    allow_paper_preseed_unlinked_residuals = bool(args.allow_paper_preseed_unlinked_residuals)
    try:
        paper_min_id_exclusive = _resolve_paper_min_id_exclusive(
            raw_min_id=int(args.paper_min_id_exclusive)
            if args.paper_min_id_exclusive is not None
            else None,
            watermark_path_raw=str(args.paper_seed_watermark).strip()
            if args.paper_seed_watermark
            else None,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not live_db.exists():
        parser.error(f"live DB not found: {live_db}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")
    if bundle_manifest is not None and not bundle_manifest.exists():
        parser.error(f"bundle manifest not found: {bundle_manifest}")
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        parser.error("from-ts must be <= to-ts")

    live_rows, live_counts, live_load_issues = _load_decision_rows(
        live_db,
        source_name="live",
        from_ts=from_ts,
        to_ts=to_ts,
        include_runtime_only_blocked=include_runtime_only_blocked,
        include_funding_events=include_funding_events,
        paper_min_trade_id_exclusive=None,
    )
    paper_rows, paper_counts, paper_load_issues = _load_decision_rows(
        paper_db,
        source_name="paper",
        from_ts=from_ts,
        to_ts=to_ts,
        include_runtime_only_blocked=include_runtime_only_blocked,
        include_funding_events=include_funding_events,
        paper_min_trade_id_exclusive=paper_min_id_exclusive,
    )

    mismatches: list[dict[str, Any]] = []
    mismatches.extend(live_load_issues)
    mismatches.extend(paper_load_issues)

    compare_summary = {
        "matched_pairs": 0,
        "rejection_reason_mismatch": 0,
        "config_fingerprint_mismatch": 0,
        "trade_linkage_mismatch": 0,
        "unmatched_live": 0,
        "unmatched_paper": 0,
    }
    per_symbol_rows: list[dict[str, Any]] = []

    if not live_load_issues and not paper_load_issues:
        cmp_mismatches, compare_summary, per_symbol_rows = _compare_rows(
            live_rows,
            paper_rows,
            timestamp_bucket_ms=timestamp_bucket_ms,
        )
        mismatches.extend(cmp_mismatches)

    run_fingerprint_guard_issues: list[dict[str, Any]] = []
    run_fingerprint_guard_detail: dict[str, Any] = {
        "enabled": bool(args.require_single_run_fingerprint),
        "bundle_manifest": str(bundle_manifest) if bundle_manifest is not None else None,
        "max_live_run_fingerprint_distinct": max(1, int(args.max_live_run_fingerprint_distinct)),
        "ok": True,
        "live_run_fingerprint_provenance": None,
    }
    if bool(args.require_single_run_fingerprint):
        if bundle_manifest is None:
            run_fingerprint_guard_issues.append(
                _run_fingerprint_guard_issue(
                    detail="run_fingerprint guard requested but --bundle-manifest is missing",
                )
            )
            run_fingerprint_guard_detail["ok"] = False
        else:
            try:
                provenance = _load_manifest_run_fingerprint_provenance(bundle_manifest)
            except Exception as exc:
                run_fingerprint_guard_issues.append(
                    _run_fingerprint_guard_issue(
                        detail=f"unable to read manifest live_run_fingerprint_provenance: {exc}",
                    )
                )
                run_fingerprint_guard_detail["ok"] = False
            else:
                run_fingerprint_guard_detail["live_run_fingerprint_provenance"] = provenance
                run_fp_distinct = int(provenance.get("run_fingerprint_distinct") or 0)
                rows_sampled = int(provenance.get("rows_sampled") or 0)
                max_allowed = max(1, int(args.max_live_run_fingerprint_distinct))
                if run_fp_distinct > max_allowed:
                    run_fingerprint_guard_issues.append(
                        _run_fingerprint_guard_issue(
                            detail=(
                                f"manifest live run_fingerprint drift exceeded limit: "
                                f"distinct={run_fp_distinct} max={max_allowed}"
                            ),
                            run_fingerprint_distinct=run_fp_distinct,
                            max_allowed=max_allowed,
                            rows_sampled=rows_sampled,
                        )
                    )
                    run_fingerprint_guard_detail["ok"] = False
    mismatches.extend(run_fingerprint_guard_issues)

    mismatch_counts: dict[str, int] = defaultdict(int)
    for item in mismatches:
        mismatch_counts[str(item.get("classification") or "unknown")] += 1

    strict_alignment_pass = (
        compare_summary["rejection_reason_mismatch"] == 0
        and compare_summary["config_fingerprint_mismatch"] == 0
        and compare_summary["trade_linkage_mismatch"] == 0
        and compare_summary["unmatched_live"] == 0
        and compare_summary["unmatched_paper"] == 0
        and not live_load_issues
        and not paper_load_issues
        and not run_fingerprint_guard_issues
    )
    paper_window_not_replayed = (
        int(live_counts.get("row_count") or 0) > 0
        and int(paper_counts.get("row_count") or 0) == 0
        and compare_summary["matched_pairs"] == 0
        and compare_summary["unmatched_live"] == int(live_counts.get("row_count") or 0)
        and compare_summary["unmatched_paper"] == 0
        and not live_load_issues
        and not paper_load_issues
    )
    if paper_window_not_replayed:
        strict_alignment_pass = False

    accepted_residuals: list[dict[str, Any]] = []
    missing_table_issues = [
        row
        for row in mismatches
        if str(row.get("kind") or "").strip().lower() == "missing_decision_events_table"
    ]
    non_missing_table_mismatches = [
        row
        for row in mismatches
        if str(row.get("kind") or "").strip().lower() != "missing_decision_events_table"
    ]
    paper_preseed_scope_only = (
        paper_min_id_exclusive is not None
        and int(paper_counts.get("filtered_by_trade_id") or 0) > 0
        and compare_summary["rejection_reason_mismatch"] == 0
        and compare_summary["config_fingerprint_mismatch"] == 0
        and compare_summary["trade_linkage_mismatch"] == 0
        and compare_summary["unmatched_live"] == 0
        and compare_summary["unmatched_paper"] > 0
        and len(non_missing_table_mismatches) == int(compare_summary["unmatched_paper"])
        and all(
            str(row.get("kind") or "").strip().lower() == "missing_live_decision_event"
            for row in non_missing_table_mismatches
        )
        and all(
            not bool(row.get("paper_trade_linked"))
            for row in non_missing_table_mismatches
        )
    )
    if paper_preseed_scope_only:
        accepted_residuals.append(
            {
                "classification": "state_initialisation_gap",
                "kind": "paper_preseed_decision_rows_out_of_scope",
                "paper_unmatched_rows": int(compare_summary["unmatched_paper"]),
                "paper_rows_excluded_by_trade_id": int(paper_counts.get("filtered_by_trade_id") or 0),
                "paper_min_trade_id_exclusive": int(paper_min_id_exclusive),
            }
        )
        if allow_paper_preseed_unlinked_residuals:
            strict_alignment_pass = not run_fingerprint_guard_issues

    decision_table_unavailable_but_empty = (
        len(missing_table_issues) > 0
        and len(non_missing_table_mismatches) == 0
        and int(live_counts.get("row_count") or 0) == 0
        and int(paper_counts.get("row_count") or 0) == 0
        and compare_summary["matched_pairs"] == 0
        and compare_summary["rejection_reason_mismatch"] == 0
        and compare_summary["config_fingerprint_mismatch"] == 0
        and compare_summary["trade_linkage_mismatch"] == 0
        and compare_summary["unmatched_live"] == 0
        and compare_summary["unmatched_paper"] == 0
    )
    if decision_table_unavailable_but_empty:
        accepted_residuals = list(missing_table_issues)
        strict_alignment_pass = not run_fingerprint_guard_issues

    report = {
        "schema_version": 2,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "inputs": {
            "live_db": str(live_db),
            "paper_db": str(paper_db),
            "bundle_manifest": str(bundle_manifest) if bundle_manifest is not None else None,
            "paper_min_id_exclusive": int(paper_min_id_exclusive)
            if paper_min_id_exclusive is not None
            else None,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "timestamp_bucket_ms": timestamp_bucket_ms,
            "include_runtime_only_blocked": include_runtime_only_blocked,
            "include_funding_events": include_funding_events,
            "allow_paper_preseed_unlinked_residuals": allow_paper_preseed_unlinked_residuals,
            "require_single_run_fingerprint": bool(args.require_single_run_fingerprint),
            "max_live_run_fingerprint_distinct": max(1, int(args.max_live_run_fingerprint_distinct)),
        },
        "counts": {
            "live_decision_rows": int(live_counts.get("row_count") or 0),
            "paper_decision_rows": int(paper_counts.get("row_count") or 0),
            "live_funding_events_excluded": int(live_counts.get("funding_excluded") or 0),
            "paper_funding_events_excluded": int(paper_counts.get("funding_excluded") or 0),
            "live_runtime_only_excluded": int(live_counts.get("runtime_only_excluded") or 0),
            "paper_runtime_only_excluded": int(paper_counts.get("runtime_only_excluded") or 0),
            "paper_filtered_by_trade_id": int(paper_counts.get("filtered_by_trade_id") or 0),
            **compare_summary,
            "mismatch_total": len(mismatches),
        },
        "status": {
            "strict_alignment_pass": strict_alignment_pass,
            "accepted_residuals_only": strict_alignment_pass and bool(accepted_residuals),
            "paper_window_not_replayed": bool(paper_window_not_replayed),
        },
        "mismatch_counts_by_classification": dict(sorted(mismatch_counts.items(), key=lambda x: x[0])),
        "run_fingerprint_guard": run_fingerprint_guard_detail,
        "accepted_residuals": accepted_residuals,
        "per_symbol": per_symbol_rows,
        "mismatches": mismatches,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output.as_posix())

    if args.fail_on_mismatch and not strict_alignment_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
