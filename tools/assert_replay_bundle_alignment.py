#!/usr/bin/env python3
"""Evaluate replay bundle alignment reports and return a single pass/fail gate.

This tool reads:
- replay bundle manifest
- state alignment report
- trade reconciliation report
- action reconciliation report
- live/paper action reconciliation report (optional/required)
- live/paper decision trace reconciliation report (optional/required)

and emits one deterministic gate result for CI/manual workflows, including
market-data provenance checks for the replay candle window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

try:
    from candles_provenance import build_candles_window_provenance
except ModuleNotFoundError:  # pragma: no cover - module execution path
    from tools.candles_provenance import build_candles_window_provenance


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
                    expected_hash = str(manifest_provenance.get("window_hash_sha256") or "")
                    expected_universe_hash = str(manifest_provenance.get("universe_hash_sha256") or "")
                    expected_symbols = [str(s) for s in (manifest_provenance.get("symbols") or [])]
                    expected_row_count = manifest_provenance.get("row_count")

                    hash_mismatch = bool(expected_hash) and expected_hash != actual_provenance["window_hash_sha256"]
                    universe_hash_mismatch = bool(expected_universe_hash) and (
                        expected_universe_hash != actual_provenance["universe_hash_sha256"]
                    )
                    symbol_mismatch = bool(expected_symbols) and expected_symbols != actual_provenance["symbols"]
                    row_count_mismatch = (
                        expected_row_count is not None and _as_int(expected_row_count) != _as_int(actual_provenance["row_count"])
                    )

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
                                    "row_count": _as_int(expected_row_count, -1),
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

    ok = len(failures) == 0

    report = {
        "ok": ok,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "bundle_dir": str(bundle_dir),
        "inputs": {
            "bundle_manifest": str(manifest_path),
            "candles_db_override": str(Path(args.candles_db).expanduser().resolve()) if args.candles_db else None,
            "skip_candles_provenance_check": bool(args.skip_candles_provenance_check),
            "state_report": str(state_path),
            "trade_report": str(trade_path),
            "action_report": str(action_path),
            "live_paper_report": str(live_paper_path),
            "require_live_paper": bool(args.require_live_paper),
            "live_paper_decision_trace_report": str(live_paper_decision_trace_path),
            "require_live_paper_decision_trace": bool(args.require_live_paper_decision_trace),
            "strict_no_residuals": bool(args.strict_no_residuals),
        },
        "checks": {
            "manifest_present": manifest is not None,
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
