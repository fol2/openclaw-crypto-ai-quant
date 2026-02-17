#!/usr/bin/env python3
"""Evaluate replay bundle alignment reports and return a single pass/fail gate.

This tool reads:
- state alignment report
- trade reconciliation report
- action reconciliation report
- live/paper action reconciliation report (optional/required)
- live/paper decision trace reconciliation report (optional/required)

and emits one deterministic gate result for CI/manual workflows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assert replay bundle alignment reports with one strict gate.")
    parser.add_argument("--bundle-dir", required=True, help="Replay bundle directory")
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
    return parser


def _resolve_report_path(bundle_dir: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (bundle_dir / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    if not bundle_dir.exists():
        parser.error(f"bundle directory not found: {bundle_dir}")

    state_path = _resolve_report_path(bundle_dir, args.state_report)
    trade_path = _resolve_report_path(bundle_dir, args.trade_report)
    action_path = _resolve_report_path(bundle_dir, args.action_report)
    live_paper_path = _resolve_report_path(bundle_dir, args.live_paper_report)
    live_paper_decision_trace_path = _resolve_report_path(bundle_dir, args.live_paper_decision_trace_report)

    failures: list[dict[str, Any]] = []

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
