#!/usr/bin/env python3
"""Audit event-order parity between live baseline and paper replay trades.

This compares canonical simulatable action events (`OPEN/ADD/REDUCE/CLOSE`) from:
- bundle `live_baseline_trades.jsonl` (live canonical baseline)
- paper `trades` table

Funding rows are treated as accepted non-simulatable residuals and excluded from
strict event-order parity checks. Simulatable events are compared in original
source order (live baseline line order vs paper trade-id order).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

SIDE_ACTIONS = {"OPEN", "ADD", "REDUCE", "CLOSE"}
FUNDING_ACTION = "FUNDING"
TRADE_ACTIONS = SIDE_ACTIONS | {FUNDING_ACTION}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit event-order parity between live baseline and paper replay actions."
    )
    parser.add_argument("--live-baseline", required=True, help="Path to live_baseline_trades.jsonl")
    parser.add_argument("--paper-db", required=True, help="Path to paper SQLite DB")
    parser.add_argument(
        "--paper-min-id-exclusive",
        type=int,
        help="Optional lower-bound filter: include only paper trades with id > this value.",
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
        help="Compare ordering at this timestamp bucket in milliseconds (1 = exact millisecond).",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        default=False,
        help="Return exit code 1 when strict order parity fails.",
    )
    return parser


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


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


def _bucket_timestamp_ms(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 1:
        return int(ts_ms)
    if ts_ms >= 0:
        return int((ts_ms // bucket_ms) * bucket_ms)
    return int(-(((-ts_ms) // bucket_ms) * bucket_ms))


def _canonical_action(action: Any, side: Any) -> str:
    act = str(action or "").strip().upper()
    if act in SIDE_ACTIONS:
        s = _normalise_side(side)
        if not s:
            return ""
        return f"{act}_{s}"
    if act == FUNDING_ACTION:
        return FUNDING_ACTION
    return ""


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


def _load_live_sequence(
    path: Path,
    *,
    from_ts: int | None,
    to_ts: int | None,
    timestamp_bucket_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    unknown_actions = 0
    funding_events = 0
    total_rows = 0
    funding_by_symbol: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1

            action = str(row.get("action") or "").strip().upper()
            if action not in TRADE_ACTIONS:
                continue

            ts_ms = _parse_timestamp_ms(row.get("timestamp_ms"))
            if ts_ms <= 0:
                ts_ms = _parse_timestamp_ms(row.get("timestamp"))
            if from_ts is not None and ts_ms < from_ts:
                continue
            if to_ts is not None and ts_ms > to_ts:
                continue

            action_code = _canonical_action(action, row.get("type"))
            if not action_code:
                unknown_actions += 1
                continue
            if action_code == FUNDING_ACTION:
                funding_events += 1
                symbol = _normalise_symbol(row.get("symbol"))
                bucket = funding_by_symbol.get(symbol)
                if bucket is None:
                    bucket = {"symbol": symbol, "count": 0, "pnl_usd": 0.0}
                    funding_by_symbol[symbol] = bucket
                bucket["count"] = int(bucket["count"]) + 1
                bucket["pnl_usd"] = float(bucket["pnl_usd"]) + _parse_float(row.get("pnl"))
                continue

            symbol = _normalise_symbol(row.get("symbol"))
            if not symbol or ts_ms <= 0:
                continue

            bucket_ts = _bucket_timestamp_ms(ts_ms, timestamp_bucket_ms)
            events.append(
                {
                    "source": "live",
                    "sequence_idx": len(events),
                    "source_id": int(row.get("id") or 0),
                    "line_no": line_no,
                    "symbol": symbol,
                    "timestamp_ms": ts_ms,
                    "bucket_ts_ms": bucket_ts,
                    "action_code": action_code,
                }
            )

    return events, {
        "live_total_rows": int(total_rows),
        "live_simulatable_events": len(events),
        "live_unknown_actions": int(unknown_actions),
        "live_funding_events": int(funding_events),
        "live_funding_by_symbol": sorted(funding_by_symbol.values(), key=lambda item: str(item.get("symbol") or "")),
    }


def _load_paper_sequence(
    paper_db: Path,
    *,
    paper_min_id_exclusive: int | None,
    from_ts: int | None,
    to_ts: int | None,
    timestamp_bucket_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conn = _connect_ro(paper_db)
    try:
        rows = conn.execute(
            "SELECT id, timestamp, symbol, action, type, reason, pnl "
            "FROM trades ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()

    events: list[dict[str, Any]] = []
    unknown_actions = 0
    funding_events = 0
    total_rows = 0
    filtered_by_id = 0
    funding_by_symbol: dict[str, dict[str, Any]] = {}

    for row in rows:
        total_rows += 1
        source_id = int(row["id"] or 0)
        if paper_min_id_exclusive is not None and source_id <= paper_min_id_exclusive:
            filtered_by_id += 1
            continue
        action = str(row["action"] or "").strip().upper()
        if action not in TRADE_ACTIONS:
            continue

        ts_ms = _parse_timestamp_ms(row["timestamp"])
        if from_ts is not None and ts_ms < from_ts:
            continue
        if to_ts is not None and ts_ms > to_ts:
            continue

        action_code = _canonical_action(action, row["type"])
        if not action_code:
            unknown_actions += 1
            continue
        if action_code == FUNDING_ACTION:
            funding_events += 1
            symbol = _normalise_symbol(row["symbol"])
            bucket = funding_by_symbol.get(symbol)
            if bucket is None:
                bucket = {"symbol": symbol, "count": 0, "pnl_usd": 0.0}
                funding_by_symbol[symbol] = bucket
            bucket["count"] = int(bucket["count"]) + 1
            bucket["pnl_usd"] = float(bucket["pnl_usd"]) + _parse_float(row["pnl"])
            continue

        symbol = _normalise_symbol(row["symbol"])
        if not symbol or ts_ms <= 0:
            continue

        bucket_ts = _bucket_timestamp_ms(ts_ms, timestamp_bucket_ms)
        events.append(
            {
                "source": "paper",
                "sequence_idx": len(events),
                "source_id": source_id,
                "symbol": symbol,
                "timestamp_ms": ts_ms,
                "bucket_ts_ms": bucket_ts,
                "action_code": action_code,
                "reason": str(row["reason"] or ""),
            }
        )

    return events, {
        "paper_total_rows": int(total_rows),
        "paper_filtered_by_id": int(filtered_by_id),
        "paper_simulatable_events": len(events),
        "paper_unknown_actions": int(unknown_actions),
        "paper_funding_events": int(funding_events),
        "paper_funding_by_symbol": sorted(funding_by_symbol.values(), key=lambda item: str(item.get("symbol") or "")),
    }


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


def _compare_event_order(
    live_events: list[dict[str, Any]],
    paper_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mismatches: list[dict[str, Any]] = []
    matched = 0
    order_mismatch = 0
    unmatched_live = 0
    unmatched_paper = 0

    limit = max(len(live_events), len(paper_events))
    for idx in range(limit):
        live_row = live_events[idx] if idx < len(live_events) else None
        paper_row = paper_events[idx] if idx < len(paper_events) else None
        if live_row is None and paper_row is not None:
            unmatched_paper += 1
            mismatches.append(
                {
                    "kind": "extra_paper_event",
                    "index": idx,
                    "paper": paper_row,
                }
            )
            continue
        if paper_row is None and live_row is not None:
            unmatched_live += 1
            mismatches.append(
                {
                    "kind": "missing_paper_event",
                    "index": idx,
                    "live": live_row,
                }
            )
            continue

        assert live_row is not None
        assert paper_row is not None
        same = (
            live_row["symbol"] == paper_row["symbol"]
            and live_row["action_code"] == paper_row["action_code"]
            and int(live_row["bucket_ts_ms"]) == int(paper_row["bucket_ts_ms"])
        )
        if same:
            matched += 1
        else:
            order_mismatch += 1
            mismatches.append(
                {
                    "kind": "order_mismatch",
                    "index": idx,
                    "live": live_row,
                    "paper": paper_row,
                }
            )

    return mismatches, {
        "matched_events": int(matched),
        "order_mismatch": int(order_mismatch),
        "unmatched_live_events": int(unmatched_live),
        "unmatched_paper_events": int(unmatched_paper),
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_baseline = Path(args.live_baseline).expanduser().resolve()
    paper_db = Path(args.paper_db).expanduser().resolve()
    bundle_manifest = Path(args.bundle_manifest).expanduser().resolve() if args.bundle_manifest else None
    output_path = Path(args.output).expanduser().resolve()
    from_ts = int(args.from_ts) if args.from_ts is not None else None
    to_ts = int(args.to_ts) if args.to_ts is not None else None
    timestamp_bucket_ms = max(1, int(args.timestamp_bucket_ms))
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

    if not live_baseline.exists():
        parser.error(f"live baseline not found: {live_baseline}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")
    if bundle_manifest is not None and not bundle_manifest.exists():
        parser.error(f"bundle manifest not found: {bundle_manifest}")
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        parser.error("from-ts must be <= to-ts")

    live_events, live_counts = _load_live_sequence(
        live_baseline,
        from_ts=from_ts,
        to_ts=to_ts,
        timestamp_bucket_ms=timestamp_bucket_ms,
    )
    paper_events, paper_counts = _load_paper_sequence(
        paper_db,
        paper_min_id_exclusive=paper_min_id_exclusive,
        from_ts=from_ts,
        to_ts=to_ts,
        timestamp_bucket_ms=timestamp_bucket_ms,
    )

    mismatches, compare_counts = _compare_event_order(live_events, paper_events)
    strict_pass = (
        compare_counts["order_mismatch"] == 0
        and compare_counts["unmatched_live_events"] == 0
        and compare_counts["unmatched_paper_events"] == 0
    )

    accepted_residuals: list[dict[str, Any]] = []
    unknown_actions_present = (
        live_counts["live_unknown_actions"] > 0 or paper_counts["paper_unknown_actions"] > 0
    )
    live_funding_rows = list(live_counts.get("live_funding_by_symbol") or [])
    paper_funding_rows = list(paper_counts.get("paper_funding_by_symbol") or [])
    live_funding_by_symbol = {
        str(item.get("symbol") or ""): {
            "count": int(item.get("count") or 0),
            "pnl_usd": float(item.get("pnl_usd") or 0.0),
        }
        for item in live_funding_rows
    }
    paper_funding_by_symbol = {
        str(item.get("symbol") or ""): {
            "count": int(item.get("count") or 0),
            "pnl_usd": float(item.get("pnl_usd") or 0.0),
        }
        for item in paper_funding_rows
    }
    funding_symbols = sorted(set(live_funding_by_symbol.keys()) | set(paper_funding_by_symbol.keys()))
    funding_by_symbol: list[dict[str, Any]] = []
    funding_matched_pairs = 0
    funding_unmatched_live = 0
    funding_unmatched_paper = 0
    matched_live_pnl_usd = 0.0
    matched_paper_pnl_usd = 0.0
    for symbol in funding_symbols:
        live_item = live_funding_by_symbol.get(symbol, {"count": 0, "pnl_usd": 0.0})
        paper_item = paper_funding_by_symbol.get(symbol, {"count": 0, "pnl_usd": 0.0})
        live_count = int(live_item.get("count") or 0)
        paper_count = int(paper_item.get("count") or 0)
        matched_count = min(live_count, paper_count)
        unmatched_live_count = max(0, live_count - matched_count)
        unmatched_paper_count = max(0, paper_count - matched_count)
        funding_matched_pairs += matched_count
        funding_unmatched_live += unmatched_live_count
        funding_unmatched_paper += unmatched_paper_count
        live_pnl = float(live_item.get("pnl_usd") or 0.0)
        paper_pnl = float(paper_item.get("pnl_usd") or 0.0)
        live_share = (float(matched_count) / float(live_count)) if live_count > 0 else 0.0
        paper_share = (float(matched_count) / float(paper_count)) if paper_count > 0 else 0.0
        matched_live_pnl_usd += live_pnl * live_share
        matched_paper_pnl_usd += paper_pnl * paper_share
        funding_by_symbol.append(
            {
                "symbol": symbol,
                "live_count": live_count,
                "paper_count": paper_count,
                "matched_pairs": matched_count,
                "unmatched_live": unmatched_live_count,
                "unmatched_paper": unmatched_paper_count,
                "live_pnl_usd": live_pnl,
                "paper_pnl_usd": paper_pnl,
            }
        )
    if funding_matched_pairs > 0:
        accepted_residuals.append(
            {
                "kind": "matched_funding_pairs_non_simulatable",
                "classification": "non-simulatable_exchange_oms_effect",
                "matched_pairs": int(funding_matched_pairs),
                "live_funding_events": int(live_counts["live_funding_events"]),
                "paper_funding_events": int(paper_counts["paper_funding_events"]),
                "matched_live_pnl_usd": float(matched_live_pnl_usd),
                "matched_paper_pnl_usd": float(matched_paper_pnl_usd),
                "symbols": [item["symbol"] for item in funding_by_symbol],
                "by_symbol": funding_by_symbol,
            }
        )
    if funding_unmatched_live > 0 or funding_unmatched_paper > 0:
        strict_pass = False
        mismatches.append(
            {
                "classification": "deterministic_logic_divergence",
                "kind": "funding_unmatched_across_surfaces",
                "live_funding_events": int(live_counts["live_funding_events"]),
                "paper_funding_events": int(paper_counts["paper_funding_events"]),
                "unmatched_live_funding_events": int(funding_unmatched_live),
                "unmatched_paper_funding_events": int(funding_unmatched_paper),
                "symbols": [item["symbol"] for item in funding_by_symbol],
                "by_symbol": funding_by_symbol,
            }
        )
    funding_contract_clean = funding_unmatched_live == 0 and funding_unmatched_paper == 0

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
    if run_fingerprint_guard_issues:
        strict_pass = False

    # If paper has no simulatable events in the audited window, but live does,
    # treat this as a replay/state-window gap residual instead of a hard order bug.
    #
    # This keeps deterministic replay harnesses usable in historical windows
    # where live baseline exists but paper was not running/replayed for that span.
    paper_window_not_replayed = (
        paper_counts["paper_simulatable_events"] == 0
        and live_counts["live_simulatable_events"] > 0
        and compare_counts["matched_events"] == 0
        and compare_counts["order_mismatch"] == 0
        and compare_counts["unmatched_live_events"] == live_counts["live_simulatable_events"]
        and compare_counts["unmatched_paper_events"] == 0
        and not unknown_actions_present
    )
    if paper_window_not_replayed:
        accepted_residuals.append(
            {
                "kind": "paper_window_not_replayed",
                "classification": "state_initialisation_gap",
                "live_simulatable_events": live_counts["live_simulatable_events"],
                "paper_simulatable_events": paper_counts["paper_simulatable_events"],
            }
        )
        strict_pass = funding_contract_clean and not run_fingerprint_guard_issues

    # When live baseline has no simulatable events but paper contains only
    # snapshot seed-close rows, treat as expected state-initialisation residual.
    paper_seed_only_window = (
        live_counts["live_simulatable_events"] == 0
        and paper_counts["paper_simulatable_events"] > 0
        and compare_counts["matched_events"] == 0
        and compare_counts["order_mismatch"] == 0
        and compare_counts["unmatched_live_events"] == 0
        and compare_counts["unmatched_paper_events"] == paper_counts["paper_simulatable_events"]
        and not unknown_actions_present
        and all(
            str(ev.get("reason") or "").strip().lower() == "state_sync_seed_close"
            for ev in paper_events
        )
    )
    if paper_seed_only_window:
        accepted_residuals.append(
            {
                "kind": "paper_seed_close_only_window",
                "classification": "state_initialisation_gap",
                "paper_simulatable_events": paper_counts["paper_simulatable_events"],
            }
        )
        strict_pass = funding_contract_clean and not run_fingerprint_guard_issues

    report = {
        "schema_version": 1,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "inputs": {
            "live_baseline": str(live_baseline),
            "paper_db": str(paper_db),
            "paper_min_id_exclusive": int(paper_min_id_exclusive)
            if paper_min_id_exclusive is not None
            else None,
            "bundle_manifest": str(bundle_manifest) if bundle_manifest is not None else None,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "timestamp_bucket_ms": timestamp_bucket_ms,
            "require_single_run_fingerprint": bool(args.require_single_run_fingerprint),
            "max_live_run_fingerprint_distinct": max(1, int(args.max_live_run_fingerprint_distinct)),
        },
        "counts": {
            **live_counts,
            **paper_counts,
            **compare_counts,
            "mismatch_count": len(mismatches),
            "accepted_residual_count": len(accepted_residuals),
            "funding_matched_pairs": int(funding_matched_pairs),
            "funding_unmatched_live": int(funding_unmatched_live),
            "funding_unmatched_paper": int(funding_unmatched_paper),
        },
        "status": {
            "order_parity_pass": bool(strict_pass),
            "strict_alignment_pass": bool(strict_pass),
        },
        "run_fingerprint_guard": run_fingerprint_guard_detail,
        "funding_pair_evidence": {
            "matched_pairs": int(funding_matched_pairs),
            "unmatched_live": int(funding_unmatched_live),
            "unmatched_paper": int(funding_unmatched_paper),
            "matched_live_pnl_usd": float(matched_live_pnl_usd),
            "matched_paper_pnl_usd": float(matched_paper_pnl_usd),
            "symbols": [item["symbol"] for item in funding_by_symbol],
            "by_symbol": funding_by_symbol,
        },
        "accepted_residuals": accepted_residuals,
        "mismatches": mismatches,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path.as_posix())

    if args.fail_on_mismatch and not strict_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
