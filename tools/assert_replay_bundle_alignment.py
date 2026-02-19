#!/usr/bin/env python3
"""Evaluate replay bundle alignment reports and return a single pass/fail gate.

This tool reads:
- replay bundle manifest
- state alignment report
- trade reconciliation report
- action reconciliation report
- live/paper action reconciliation report (optional/required)
- live/paper decision trace reconciliation report (optional/required)
- event-order parity report (optional/required)
- GPU parity report (optional/required)

and emits one deterministic gate result for CI/manual workflows, including
market-data provenance checks for the replay candle window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from candles_provenance import build_candles_window_provenance
except ModuleNotFoundError:  # pragma: no cover - module execution path
    from tools.candles_provenance import build_candles_window_provenance

try:
    import yaml
except Exception:  # pragma: no cover - optional runtime dependency
    yaml = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assert replay bundle alignment reports with one strict gate.")
    parser.add_argument("--bundle-dir", required=True, help="Replay bundle directory")
    parser.add_argument(
        "--bundle-manifest",
        default="replay_bundle_manifest.json",
        help="Replay bundle manifest filename/path",
    )
    parser.add_argument(
        "--candles-db",
        help="Optional candles DB override path for provenance validation",
    )
    parser.add_argument("--state-report", default="state_alignment_report.json", help="State alignment report filename/path")
    parser.add_argument("--trade-report", default="trade_reconcile_report.json", help="Trade reconcile report filename/path")
    parser.add_argument("--action-report", default="action_reconcile_report.json", help="Action reconcile report filename/path")
    parser.add_argument(
        "--live-paper-report",
        default="live_paper_action_reconcile_report.json",
        help="Optional live/paper action reconcile report filename/path",
    )
    parser.add_argument(
        "--require-live-paper",
        action="store_true",
        default=False,
        help="Fail when live/paper action report is missing",
    )
    parser.add_argument(
        "--live-paper-decision-trace-report",
        default="live_paper_decision_trace_reconcile_report.json",
        help="Optional live/paper decision trace reconcile report filename/path",
    )
    parser.add_argument(
        "--require-live-paper-decision-trace",
        action="store_true",
        default=False,
        help="Fail when live/paper decision trace report is missing",
    )
    parser.add_argument(
        "--event-order-report",
        default="event_order_parity_report.json",
        help="Optional event-order parity report filename/path",
    )
    parser.add_argument(
        "--require-event-order",
        action="store_true",
        default=False,
        help="Fail when event-order parity report is missing",
    )
    parser.add_argument(
        "--gpu-parity-report",
        default="gpu_smoke_parity_report.json",
        help="Optional GPU smoke parity report filename/path",
    )
    parser.add_argument(
        "--require-gpu-parity",
        action="store_true",
        default=False,
        help="Fail when GPU parity report is missing",
    )
    parser.add_argument("--output", help="Optional output path for gate report JSON")
    parser.add_argument(
        "--strict-no-residuals",
        action="store_true",
        default=False,
        help="Fail if accepted residuals are present in trade/action reports",
    )
    parser.add_argument(
        "--skip-candles-provenance-check",
        action="store_true",
        default=False,
        help="Skip manifest candle-window provenance validation",
    )
    parser.add_argument(
        "--require-runtime-strategy-provenance",
        action="store_true",
        default=False,
        help="Fail when manifest runtime strategy provenance is missing",
    )
    parser.add_argument(
        "--max-strategy-sha1-distinct",
        type=int,
        default=None,
        help="Maximum allowed distinct runtime strategy_sha1 within the replay window",
    )
    parser.add_argument(
        "--require-oms-strategy-provenance",
        action="store_true",
        default=False,
        help="Fail when manifest OMS strategy provenance is missing",
    )
    parser.add_argument(
        "--max-oms-strategy-sha1-distinct",
        type=int,
        default=None,
        help="Maximum allowed distinct OMS strategy_sha1 within the replay window",
    )
    parser.add_argument(
        "--require-locked-strategy-match",
        action="store_true",
        default=False,
        help="Fail when locked strategy hash cannot be matched against runtime/OMS provenance",
    )
    return parser


def _resolve_report_path(bundle_dir: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (bundle_dir / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _hash_json_canonical(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except Exception:
        payload = repr(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compute_locked_strategy_sha1(bundle_dir: Path, manifest: dict[str, Any]) -> tuple[str, Path]:
    artefacts = manifest.get("artefacts") or {}
    snapshot_raw = str((artefacts.get("strategy_config_snapshot_file") or "strategy_overrides.locked.yaml")).strip()
    snapshot_path = Path(snapshot_raw).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = (bundle_dir / snapshot_path).resolve()
    else:
        snapshot_path = snapshot_path.resolve()

    if yaml is None or not snapshot_path.exists():
        return "", snapshot_path
    try:
        raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "", snapshot_path
    obj = raw if isinstance(raw, dict) else {}
    return _hash_json_canonical(obj), snapshot_path


def _gpu_lane_pass_map(report: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    lanes = report.get("lanes")
    lane_map: dict[str, bool] = {}
    invalid_lanes: list[str] = []
    if isinstance(lanes, dict):
        for lane_name, lane_obj in lanes.items():
            if isinstance(lane_obj, dict):
                ranking = lane_obj.get("ranking") or {}
                raw_all_pass = (ranking or {}).get("all_pass")
                if isinstance(raw_all_pass, bool):
                    lane_map[str(lane_name)] = raw_all_pass
                else:
                    invalid_lanes.append(str(lane_name))
    elif isinstance(lanes, list):
        for idx, lane_obj in enumerate(lanes, start=1):
            if isinstance(lane_obj, dict):
                lane_name = str(lane_obj.get("lane") or f"lane_{idx}")
                ranking = lane_obj.get("ranking") or {}
                raw_all_pass = (ranking or {}).get("all_pass")
                if isinstance(raw_all_pass, bool):
                    lane_map[lane_name] = raw_all_pass
                else:
                    invalid_lanes.append(lane_name)
    return lane_map, invalid_lanes


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        parser.error(f"bundle directory not found: {bundle_dir}")

    manifest_path = _resolve_report_path(bundle_dir, args.bundle_manifest)
    state_path = _resolve_report_path(bundle_dir, args.state_report)
    trade_path = _resolve_report_path(bundle_dir, args.trade_report)
    action_path = _resolve_report_path(bundle_dir, args.action_report)
    live_paper_path = _resolve_report_path(bundle_dir, args.live_paper_report)
    live_paper_decision_trace_path = _resolve_report_path(bundle_dir, args.live_paper_decision_trace_report)
    event_order_path = _resolve_report_path(bundle_dir, args.event_order_report)
    gpu_parity_path = _resolve_report_path(bundle_dir, args.gpu_parity_report)

    failures: list[dict[str, Any]] = []
    candles_provenance_checked = not bool(args.skip_candles_provenance_check)
    candles_provenance_ok = not candles_provenance_checked

    manifest: dict[str, Any] | None = None
    if not manifest_path.exists():
        failures.append(
            {
                "code": "missing_bundle_manifest",
                "classification": "market_data_alignment_gap",
                "detail": str(manifest_path),
            }
        )
    else:
        loaded = _load_json(manifest_path)
        if isinstance(loaded, dict):
            manifest = loaded
        else:
            failures.append(
                {
                    "code": "invalid_bundle_manifest",
                    "classification": "market_data_alignment_gap",
                    "detail": "bundle manifest is not a JSON object",
                }
            )

    strategy_runtime_stability_ok = True
    strategy_oms_stability_ok = True
    locked_strategy_match_ok = True
    if manifest is not None:
        runtime_strategy = manifest.get("runtime_strategy_provenance")
        oms_strategy = manifest.get("oms_strategy_provenance")
        locked_strategy = manifest.get("locked_strategy_provenance")
        if not isinstance(runtime_strategy, dict):
            strategy_runtime_stability_ok = False
            if args.require_runtime_strategy_provenance:
                failures.append(
                    {
                        "code": "missing_runtime_strategy_provenance",
                        "classification": "state_initialisation_gap",
                        "detail": "manifest.runtime_strategy_provenance is missing",
                    }
                )
        else:
            distinct_sha = _as_int(runtime_strategy.get("strategy_sha1_distinct"), 0)
            strategy_rows_sampled = _as_int(runtime_strategy.get("strategy_rows_sampled"), 0)
            timeline = runtime_strategy.get("strategy_sha1_timeline")
            timeline_count = len(timeline) if isinstance(timeline, list) else 0
            if args.require_runtime_strategy_provenance and (strategy_rows_sampled <= 0 or timeline_count <= 0):
                strategy_runtime_stability_ok = False
                failures.append(
                    {
                        "code": "empty_runtime_strategy_provenance",
                        "classification": "state_initialisation_gap",
                        "detail": "runtime strategy provenance is present but contains no sampled strategy_sha1 rows",
                        "counts": {
                            "strategy_rows_sampled": strategy_rows_sampled,
                            "strategy_sha1_timeline_count": timeline_count,
                            "runtime_rows_in_window": _as_int(runtime_strategy.get("runtime_rows_in_window"), 0),
                        },
                    }
                )
            max_distinct = args.max_strategy_sha1_distinct
            if max_distinct is not None and distinct_sha > int(max_distinct):
                strategy_runtime_stability_ok = False
                failures.append(
                    {
                        "code": "strategy_config_drift_within_window",
                        "classification": "state_initialisation_gap",
                        "detail": (
                            f"runtime strategy_sha1 drift exceeded limit: "
                            f"distinct={distinct_sha} > max={int(max_distinct)}"
                        ),
                        "counts": {
                            "strategy_sha1_distinct": distinct_sha,
                            "strategy_version_distinct": _as_int(runtime_strategy.get("strategy_version_distinct"), 0),
                            "strategy_rows_sampled": strategy_rows_sampled,
                            "runtime_rows_in_window": _as_int(runtime_strategy.get("runtime_rows_in_window"), 0),
                        },
                    }
                )

        if not isinstance(oms_strategy, dict):
            strategy_oms_stability_ok = False
            if args.require_oms_strategy_provenance or args.require_locked_strategy_match:
                failures.append(
                    {
                        "code": "missing_oms_strategy_provenance",
                        "classification": "state_initialisation_gap",
                        "detail": "manifest.oms_strategy_provenance is missing",
                    }
                )
        else:
            oms_distinct_sha = _as_int(oms_strategy.get("strategy_sha1_distinct"), 0)
            oms_rows_sampled = _as_int(oms_strategy.get("oms_rows_sampled"), 0)
            oms_timeline = oms_strategy.get("strategy_sha1_timeline")
            oms_timeline_count = len(oms_timeline) if isinstance(oms_timeline, list) else 0
            if (args.require_oms_strategy_provenance or args.require_locked_strategy_match) and (
                oms_rows_sampled <= 0 or oms_timeline_count <= 0
            ):
                strategy_oms_stability_ok = False
                failures.append(
                    {
                        "code": "empty_oms_strategy_provenance",
                        "classification": "state_initialisation_gap",
                        "detail": "OMS strategy provenance is present but contains no sampled strategy_sha1 rows",
                        "counts": {
                            "oms_rows_sampled": oms_rows_sampled,
                            "strategy_sha1_timeline_count": oms_timeline_count,
                        },
                    }
                )
            max_oms_distinct = args.max_oms_strategy_sha1_distinct
            if max_oms_distinct is not None and oms_distinct_sha > int(max_oms_distinct):
                strategy_oms_stability_ok = False
                failures.append(
                    {
                        "code": "strategy_config_drift_within_oms_window",
                        "classification": "state_initialisation_gap",
                        "detail": (
                            f"OMS strategy_sha1 drift exceeded limit: "
                            f"distinct={oms_distinct_sha} > max={int(max_oms_distinct)}"
                        ),
                        "counts": {
                            "strategy_sha1_distinct": oms_distinct_sha,
                            "strategy_version_distinct": _as_int(oms_strategy.get("strategy_version_distinct"), 0),
                            "oms_rows_sampled": oms_rows_sampled,
                        },
                    }
                )

        if args.require_locked_strategy_match:
            if not isinstance(locked_strategy, dict):
                locked_strategy_match_ok = False
                failures.append(
                    {
                        "code": "missing_locked_strategy_provenance",
                        "classification": "state_initialisation_gap",
                        "detail": "manifest.locked_strategy_provenance is missing",
                    }
                )
            else:
                locked_sha = str(locked_strategy.get("strategy_overrides_sha1") or "").strip().lower()
                if len(locked_sha) < 8:
                    locked_strategy_match_ok = False
                    failures.append(
                        {
                            "code": "invalid_locked_strategy_sha1",
                            "classification": "state_initialisation_gap",
                            "detail": "locked strategy hash is missing or too short",
                        }
                    )
                else:
                    computed_locked_sha, snapshot_path = _compute_locked_strategy_sha1(bundle_dir, manifest)
                    if len(computed_locked_sha) < 8:
                        locked_strategy_match_ok = False
                        failures.append(
                            {
                                "code": "locked_strategy_snapshot_hash_unavailable",
                                "classification": "state_initialisation_gap",
                                "detail": (
                                    f"unable to compute locked strategy hash from snapshot file: {snapshot_path}"
                                ),
                            }
                        )
                    elif computed_locked_sha != locked_sha:
                        locked_strategy_match_ok = False
                        failures.append(
                            {
                                "code": "locked_strategy_manifest_snapshot_hash_mismatch",
                                "classification": "state_initialisation_gap",
                                "detail": "manifest locked strategy hash does not match snapshot YAML canonical hash",
                                "manifest_locked_sha1_prefix8": locked_sha[:8],
                                "snapshot_locked_sha1_prefix8": computed_locked_sha[:8],
                                "snapshot_path": str(snapshot_path),
                            }
                        )

                    runtime_timeline = (
                        runtime_strategy.get("strategy_sha1_timeline")
                        if isinstance(runtime_strategy, dict)
                        else []
                    )
                    runtime_rows = runtime_timeline if isinstance(runtime_timeline, list) else []
                    runtime_prefix_match = any(
                        locked_sha.startswith(str(row.get("strategy_sha1") or "").strip().lower())
                        for row in runtime_rows
                        if str(row.get("strategy_sha1") or "").strip()
                    )
                    if not runtime_prefix_match:
                        locked_strategy_match_ok = False
                        failures.append(
                            {
                                "code": "locked_strategy_runtime_prefix_mismatch",
                                "classification": "state_initialisation_gap",
                                "detail": "locked strategy hash does not match any runtime strategy_sha1 prefix",
                                "locked_strategy_sha1_prefix8": locked_sha[:8],
                            }
                        )

                    oms_timeline = (
                        oms_strategy.get("strategy_sha1_timeline")
                        if isinstance(oms_strategy, dict)
                        else []
                    )
                    oms_rows = oms_timeline if isinstance(oms_timeline, list) else []
                    oms_exact_match = any(
                        str(row.get("strategy_sha1") or "").strip().lower() == locked_sha
                        for row in oms_rows
                    )
                    if not oms_exact_match:
                        locked_strategy_match_ok = False
                        failures.append(
                            {
                                "code": "locked_strategy_oms_sha1_mismatch",
                                "classification": "state_initialisation_gap",
                                "detail": "locked strategy hash does not match any OMS strategy_sha1 in window",
                                "locked_strategy_sha1_prefix8": locked_sha[:8],
                            }
                        )

    if candles_provenance_checked and manifest is not None:
        manifest_inputs = manifest.get("inputs") or {}
        manifest_provenance = manifest.get("candles_provenance") or {}
        candles_db_raw = str(args.candles_db or manifest_inputs.get("candles_db") or "").strip()
        if not candles_db_raw:
            failures.append(
                {
                    "code": "missing_candles_db_for_provenance",
                    "classification": "market_data_alignment_gap",
                    "detail": "candles DB path is missing in manifest and no --candles-db override was provided",
                }
            )
        else:
            candles_db_path = Path(candles_db_raw).expanduser().resolve()
            if not candles_db_path.exists():
                failures.append(
                    {
                        "code": "candles_db_not_found_for_provenance",
                        "classification": "market_data_alignment_gap",
                        "detail": str(candles_db_path),
                    }
                )
            else:
                interval = str(manifest_provenance.get("interval") or manifest_inputs.get("interval") or "").strip()
                from_ts = _as_int(manifest_provenance.get("from_ts", manifest_inputs.get("from_ts")))
                to_ts = _as_int(manifest_provenance.get("to_ts", manifest_inputs.get("to_ts")))
                if not interval or from_ts > to_ts:
                    failures.append(
                        {
                            "code": "invalid_manifest_candles_provenance",
                            "classification": "market_data_alignment_gap",
                            "detail": "manifest candles provenance has invalid interval/from_ts/to_ts",
                        }
                    )
                else:
                    actual_provenance = build_candles_window_provenance(
                        candles_db_path,
                        interval=interval,
                        from_ts=from_ts,
                        to_ts=to_ts,
                    )
                    expected_hash = str(manifest_provenance.get("window_hash_sha256") or "").strip().lower()
                    expected_universe_hash = str(manifest_provenance.get("universe_hash_sha256") or "").strip().lower()
                    raw_expected_symbols = manifest_provenance.get("symbols")
                    expected_symbols = (
                        [str(s or "").strip().upper() for s in raw_expected_symbols]
                        if isinstance(raw_expected_symbols, list)
                        else []
                    )
                    expected_row_count_raw = manifest_provenance.get("row_count")

                    missing_manifest_fields: list[str] = []
                    if not expected_hash:
                        missing_manifest_fields.append("candles_provenance.window_hash_sha256")
                    if not expected_universe_hash:
                        missing_manifest_fields.append("candles_provenance.universe_hash_sha256")
                    if not isinstance(raw_expected_symbols, list):
                        missing_manifest_fields.append("candles_provenance.symbols")
                    if expected_row_count_raw is None:
                        missing_manifest_fields.append("candles_provenance.row_count")

                    if missing_manifest_fields:
                        failures.append(
                            {
                                "code": "invalid_manifest_candles_provenance_fields",
                                "classification": "market_data_alignment_gap",
                                "detail": "manifest candles provenance is missing required lock fields",
                                "missing_fields": missing_manifest_fields,
                            }
                        )
                    else:
                        expected_row_count = _as_int(expected_row_count_raw, -1)
                        hash_mismatch = expected_hash != actual_provenance["window_hash_sha256"]
                        universe_hash_mismatch = expected_universe_hash != actual_provenance["universe_hash_sha256"]
                        symbol_mismatch = expected_symbols != actual_provenance["symbols"]
                        row_count_mismatch = expected_row_count != _as_int(actual_provenance["row_count"], -1)

                        if hash_mismatch or universe_hash_mismatch or symbol_mismatch or row_count_mismatch:
                            failures.append(
                                {
                                    "code": "candles_provenance_mismatch",
                                    "classification": "market_data_alignment_gap",
                                    "detail": "recomputed candle window provenance does not match manifest",
                                    "expected": {
                                        "window_hash_sha256": expected_hash,
                                        "universe_hash_sha256": expected_universe_hash,
                                        "symbol_count": len(expected_symbols),
                                        "symbols": expected_symbols,
                                        "row_count": expected_row_count,
                                    },
                                    "actual": {
                                        "window_hash_sha256": actual_provenance["window_hash_sha256"],
                                        "universe_hash_sha256": actual_provenance["universe_hash_sha256"],
                                        "symbol_count": _as_int(actual_provenance["symbol_count"]),
                                        "symbols": actual_provenance["symbols"],
                                        "row_count": _as_int(actual_provenance["row_count"]),
                                    },
                                }
                            )
                        else:
                            candles_provenance_ok = True

    state_report: dict[str, Any] | None = None
    if not state_path.exists():
        failures.append(
            {
                "code": "missing_state_report",
                "classification": "state_initialisation_gap",
                "detail": str(state_path),
            }
        )
    else:
        state_report = _load_json(state_path)
        if not bool(state_report.get("ok")):
            failures.append(
                {
                    "code": "state_alignment_failed",
                    "classification": "state_initialisation_gap",
                    "detail": "state alignment report is not ok",
                    "diff_count": int((state_report.get("summary") or {}).get("diff_count") or 0),
                }
            )

    trade_report: dict[str, Any] | None = None
    if not trade_path.exists():
        failures.append(
            {
                "code": "missing_trade_report",
                "classification": "deterministic_logic_divergence",
                "detail": str(trade_path),
            }
        )
    else:
        trade_report = _load_json(trade_path)
        trade_status = bool(((trade_report.get("status") or {}).get("strict_alignment_pass")))
        if not trade_status:
            failures.append(
                {
                    "code": "trade_alignment_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": "trade reconciliation strict alignment failed",
                    "counts": trade_report.get("counts") or {},
                }
            )
        if args.strict_no_residuals:
            trade_residuals = list(trade_report.get("accepted_residuals") or [])
            if trade_residuals:
                failures.append(
                    {
                        "code": "trade_residuals_present",
                        "classification": "non-simulatable_exchange_oms_effect",
                        "detail": "trade reconciliation has accepted residuals",
                        "count": len(trade_residuals),
                    }
                )

    action_report: dict[str, Any] | None = None
    if not action_path.exists():
        failures.append(
            {
                "code": "missing_action_report",
                "classification": "deterministic_logic_divergence",
                "detail": str(action_path),
            }
        )
    else:
        action_report = _load_json(action_path)
        action_status = bool(((action_report.get("status") or {}).get("strict_alignment_pass")))
        if not action_status:
            failures.append(
                {
                    "code": "action_alignment_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": "action reconciliation strict alignment failed",
                    "counts": action_report.get("counts") or {},
                }
            )
        if args.strict_no_residuals:
            action_residuals = list(action_report.get("accepted_residuals") or [])
            if action_residuals:
                failures.append(
                    {
                        "code": "action_residuals_present",
                        "classification": "non-simulatable_exchange_oms_effect",
                        "detail": "action reconciliation has accepted residuals",
                        "count": len(action_residuals),
                    }
                )

    live_paper_report: dict[str, Any] | None = None
    if not live_paper_path.exists():
        if args.require_live_paper:
            failures.append(
                {
                    "code": "missing_live_paper_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(live_paper_path),
                }
            )
    else:
        live_paper_report = _load_json(live_paper_path)
        live_paper_status = bool(((live_paper_report.get("status") or {}).get("strict_alignment_pass")))
        if not live_paper_status:
            failures.append(
                {
                    "code": "live_paper_alignment_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": "live/paper reconciliation strict alignment failed",
                    "counts": live_paper_report.get("counts") or {},
                }
            )
        if args.strict_no_residuals:
            live_paper_residuals = list(live_paper_report.get("accepted_residuals") or [])
            if live_paper_residuals:
                failures.append(
                    {
                        "code": "live_paper_residuals_present",
                        "classification": "non-simulatable_exchange_oms_effect",
                        "detail": "live/paper reconciliation has accepted residuals",
                        "count": len(live_paper_residuals),
                    }
                )

    live_paper_decision_trace_report: dict[str, Any] | None = None
    if not live_paper_decision_trace_path.exists():
        if args.require_live_paper_decision_trace:
            failures.append(
                {
                    "code": "missing_live_paper_decision_trace_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(live_paper_decision_trace_path),
                }
            )
    else:
        live_paper_decision_trace_report = _load_json(live_paper_decision_trace_path)
        decision_trace_status = bool(
            ((live_paper_decision_trace_report.get("status") or {}).get("strict_alignment_pass"))
        )
        if not decision_trace_status:
            failures.append(
                {
                    "code": "live_paper_decision_trace_alignment_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": "live/paper decision trace strict alignment failed",
                    "counts": live_paper_decision_trace_report.get("counts") or {},
                }
            )

    event_order_report: dict[str, Any] | None = None
    if not event_order_path.exists():
        if args.require_event_order:
            failures.append(
                {
                    "code": "missing_event_order_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(event_order_path),
                }
            )
    else:
        event_order_report = _load_json(event_order_path)
        event_order_ok = bool((event_order_report.get("status") or {}).get("order_parity_pass"))
        if not event_order_ok:
            failures.append(
                {
                    "code": "event_order_parity_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": "event order parity check failed",
                    "counts": event_order_report.get("counts") or {},
                }
            )
        if args.strict_no_residuals:
            event_order_residuals = list(event_order_report.get("accepted_residuals") or [])
            if event_order_residuals:
                failures.append(
                    {
                        "code": "event_order_residuals_present",
                        "classification": "state_initialisation_gap",
                        "detail": "event-order reconciliation has accepted residuals",
                        "count": len(event_order_residuals),
                    }
                )

    gpu_parity_report: dict[str, Any] | None = None
    gpu_lane_status: dict[str, bool] = {}
    if not gpu_parity_path.exists():
        if args.require_gpu_parity:
            failures.append(
                {
                    "code": "missing_gpu_parity_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(gpu_parity_path),
                }
            )
    else:
        loaded_gpu = _load_json(gpu_parity_path)
        if not isinstance(loaded_gpu, dict):
            failures.append(
                {
                    "code": "invalid_gpu_parity_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": "GPU parity report is not a JSON object",
                }
            )
        else:
            gpu_parity_report = loaded_gpu
            gpu_lane_status, invalid_gpu_lanes = _gpu_lane_pass_map(gpu_parity_report)
            if not gpu_lane_status:
                failures.append(
                    {
                        "code": "invalid_gpu_parity_report_lanes",
                        "classification": "deterministic_logic_divergence",
                        "detail": "GPU parity report does not contain lane ranking status",
                    }
                )
            elif invalid_gpu_lanes:
                failures.append(
                    {
                        "code": "invalid_gpu_parity_lane_values",
                        "classification": "deterministic_logic_divergence",
                        "detail": "GPU parity report has non-boolean ranking.all_pass values",
                        "lanes": invalid_gpu_lanes,
                    }
                )
            else:
                has_lane_a = any("lane_a" in name.lower() for name in gpu_lane_status)
                has_lane_b = any("lane_b" in name.lower() for name in gpu_lane_status)
                if not (has_lane_a and has_lane_b):
                    failures.append(
                        {
                            "code": "invalid_gpu_parity_lane_coverage",
                            "classification": "deterministic_logic_divergence",
                            "detail": "GPU parity report is missing required lane_a/lane_b coverage",
                            "lanes": gpu_lane_status,
                        }
                    )
                elif not all(gpu_lane_status.values()):
                    failures.append(
                        {
                            "code": "gpu_parity_failed",
                            "classification": "deterministic_logic_divergence",
                            "detail": "GPU parity lane ranking assertions failed",
                            "lanes": gpu_lane_status,
                        }
                    )

    ok = len(failures) == 0

    report = {
        "ok": ok,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "bundle_dir": str(bundle_dir),
        "inputs": {
            "bundle_manifest": str(manifest_path),
            "candles_db_override": str(Path(args.candles_db).expanduser().resolve()) if args.candles_db else None,
            "skip_candles_provenance_check": bool(args.skip_candles_provenance_check),
            "require_runtime_strategy_provenance": bool(args.require_runtime_strategy_provenance),
            "max_strategy_sha1_distinct": int(args.max_strategy_sha1_distinct)
            if args.max_strategy_sha1_distinct is not None
            else None,
            "require_oms_strategy_provenance": bool(args.require_oms_strategy_provenance),
            "max_oms_strategy_sha1_distinct": int(args.max_oms_strategy_sha1_distinct)
            if args.max_oms_strategy_sha1_distinct is not None
            else None,
            "require_locked_strategy_match": bool(args.require_locked_strategy_match),
            "state_report": str(state_path),
            "trade_report": str(trade_path),
            "action_report": str(action_path),
            "live_paper_report": str(live_paper_path),
            "require_live_paper": bool(args.require_live_paper),
            "live_paper_decision_trace_report": str(live_paper_decision_trace_path),
            "require_live_paper_decision_trace": bool(args.require_live_paper_decision_trace),
            "event_order_report": str(event_order_path),
            "require_event_order": bool(args.require_event_order),
            "gpu_parity_report": str(gpu_parity_path),
            "require_gpu_parity": bool(args.require_gpu_parity),
            "strict_no_residuals": bool(args.strict_no_residuals),
        },
        "checks": {
            "manifest_present": manifest is not None,
            "strategy_runtime_stability_ok": strategy_runtime_stability_ok,
            "strategy_oms_stability_ok": strategy_oms_stability_ok,
            "locked_strategy_match_ok": locked_strategy_match_ok,
            "candles_provenance_checked": candles_provenance_checked,
            "candles_provenance_ok": candles_provenance_ok,
            "state_ok": bool(state_report.get("ok")) if state_report is not None else False,
            "trade_ok": bool((trade_report.get("status") or {}).get("strict_alignment_pass")) if trade_report else False,
            "action_ok": bool((action_report.get("status") or {}).get("strict_alignment_pass")) if action_report else False,
            "live_paper_ok": bool((live_paper_report.get("status") or {}).get("strict_alignment_pass"))
            if live_paper_report
            else (not bool(args.require_live_paper)),
            "live_paper_decision_trace_ok": bool(
                (live_paper_decision_trace_report.get("status") or {}).get("strict_alignment_pass")
            )
            if live_paper_decision_trace_report
            else (not bool(args.require_live_paper_decision_trace)),
            "event_order_ok": bool((event_order_report.get("status") or {}).get("order_parity_pass"))
            if event_order_report
            else (not bool(args.require_event_order)),
            "gpu_parity_ok": all(gpu_lane_status.values()) if gpu_lane_status else (not bool(args.require_gpu_parity)),
            "gpu_parity_lane_status": gpu_lane_status,
            "trade_residual_count": len((trade_report or {}).get("accepted_residuals") or []),
            "action_residual_count": len((action_report or {}).get("accepted_residuals") or []),
            "live_paper_residual_count": len((live_paper_report or {}).get("accepted_residuals") or []),
        },
        "failure_count": len(failures),
        "failures": failures,
    }

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_path = Path(args.output).expanduser().resolve() if args.output else (bundle_dir / "alignment_gate_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(output_path.as_posix())

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
