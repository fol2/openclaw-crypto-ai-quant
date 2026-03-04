from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest

from tools import assert_replay_bundle_alignment as alignment_gate
from tools import build_live_replay_bundle as replay_bundle_builder
from tools import run_paper_deterministic_replay as paper_harness


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_base_gate_bundle(
    bundle_dir: Path,
    *,
    manifest_inputs: dict | None = None,
    manifest_extra: dict | None = None,
    seed_apply_report: dict | None = None,
    action_report: dict | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "state_alignment_report.json", {"ok": True, "summary": {"diff_count": 0}})
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {"status": {"strict_alignment_pass": True}, "accepted_residuals": [], "counts": {}},
    )
    _write_json(
        bundle_dir / "action_reconcile_report.json",
        action_report or {"status": {"strict_alignment_pass": True}, "accepted_residuals": [], "counts": {}},
    )
    _write_json(
        bundle_dir / "backtester_replay_report.json",
        {"config_fingerprint": "a" * 64},
    )
    if seed_apply_report is not None:
        _write_json(bundle_dir / "paper_seed_apply_report.json", seed_apply_report)

    manifest = {
        "schema_version": 1,
        "inputs": {
            "from_ts": 1_000,
            "to_ts": 2_000,
            "snapshot_strict_replace": False,
        },
        "artefacts": {
            "backtester_replay_report_json": "backtester_replay_report.json",
        },
    }
    if isinstance(manifest_inputs, dict):
        manifest["inputs"].update(manifest_inputs)
    if isinstance(manifest_extra, dict):
        manifest.update(manifest_extra)
    _write_json(bundle_dir / "replay_bundle_manifest.json", manifest)


def test_live_run_fingerprint_provenance_summariser_respects_declared_window() -> None:
    payload = replay_bundle_builder._summarise_live_run_fingerprint_provenance(
        [
            {"timestamp_ms": 999, "run_fingerprint": "fp-before"},
            {"timestamp_ms": 1_000, "run_fingerprint": "fp-main"},
            {"timestamp_ms": 1_500, "run_fingerprint": "fp-main"},
            {"timestamp_ms": 2_000, "run_fingerprint": "fp-end"},
            {"timestamp_ms": 2_001, "run_fingerprint": "fp-after"},
        ],
        from_ts=1_000,
        to_ts=2_000,
    )

    assert payload["window_from_ts"] == 1_000
    assert payload["window_to_ts"] == 2_000
    assert payload["rows_sampled"] == 3
    assert payload["run_fingerprint_distinct"] == 2


def test_alignment_gate_fails_on_seed_strict_replace_contract_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        manifest_inputs={"snapshot_strict_replace": True},
        seed_apply_report={"strict_replace": False},
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "snapshot_strict_replace_contract_mismatch" in failure_codes


def test_alignment_gate_fails_on_live_run_fingerprint_provenance_window_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        manifest_extra={
            "live_run_fingerprint_provenance": {
                "rows_sampled": 8,
                "run_fingerprint_distinct": 1,
                "run_fingerprint_timeline": [],
                "window_from_ts": 1_000,
                "window_to_ts": 2_999,
            }
        },
        seed_apply_report={"strict_replace": False},
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "live_run_fingerprint_provenance_window_mismatch" in failure_codes


def test_alignment_gate_action_artefact_only_residuals_fail_closed_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(),
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}
    action_failure = next(
        (row for row in report.get("failures") or [] if str(row.get("code") or "") == "action_alignment_failed"),
        {},
    )

    assert exit_code == 1
    assert "action_alignment_failed" in failure_codes
    assert report["checks"]["action_gate_mode"] == "strict_fail_closed"
    assert report["checks"]["action_ok"] is False
    assert report["checks"]["action_strict_ok"] is False
    assert report["checks"]["action_opt_in_ok"] is True
    assert report["checks"]["action_artefact_only_mismatch"] is True
    assert action_failure.get("gate_mode") == "strict_fail_closed"


def test_alignment_gate_action_artefact_only_residuals_pass_with_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(),
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-action-artefact-residuals",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["ok"] is True
    assert report["checks"]["action_gate_mode"] == "allow_action_artefact_residuals"
    assert report["checks"]["action_ok"] is True
    assert report["checks"]["action_strict_ok"] is False
    assert report["checks"]["action_opt_in_ok"] is True
    assert report["checks"]["action_artefact_only_mismatch"] is True


def test_alignment_gate_action_artefact_opt_in_requires_hard_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(include_hard_evidence=False),
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-action-artefact-residuals",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "action_artefact_opt_in_unproven" in failure_codes
    assert report["checks"]["action_opt_in_ok"] is False


def test_alignment_gate_action_artefact_opt_in_rejects_deterministic_classification_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(deterministic_class_count=1),
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-action-artefact-residuals",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "action_artefact_opt_in_unproven" in failure_codes
    assert report["checks"]["action_opt_in_ok"] is False


def test_alignment_gate_action_artefact_opt_in_rejects_kind_classification_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(classification_kind_drift_total=1),
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-action-artefact-residuals",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "action_artefact_opt_in_unproven" in failure_codes
    assert report["checks"]["action_opt_in_ok"] is False


@pytest.mark.parametrize(
    ("axis", "report_name", "residual_failure_code", "extra_args"),
    [
        ("trade", "trade_reconcile_report.json", "trade_residuals_present", []),
        ("action", "action_reconcile_report.json", "action_residuals_present", []),
        (
            "live_paper",
            "live_paper_action_reconcile_report.json",
            "live_paper_residuals_present",
            ["--require-live-paper"],
        ),
    ],
)
def test_alignment_gate_axis_contract_marks_blocking_residuals_per_axis(
    tmp_path: Path,
    monkeypatch,
    axis: str,
    report_name: str,
    residual_failure_code: str,
    extra_args: list[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_json(
        bundle_dir / report_name,
        {
            "status": {"strict_alignment_pass": True},
            "accepted_residuals": [
                {"classification": "deterministic_logic_divergence", "kind": f"blocking_{axis}_residual"}
            ],
            "counts": {},
        },
    )
    output = bundle_dir / "alignment_gate_report.json"
    argv = [
        "assert_replay_bundle_alignment.py",
        "--bundle-dir",
        str(bundle_dir),
        "--skip-candles-provenance-check",
        "--strict-no-residuals",
        "--output",
        str(output),
    ]
    argv.extend(extra_args)

    monkeypatch.setattr(sys, "argv", argv)
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}
    axis_contract = (report.get("contract") or {}).get("axes", {}).get(axis, {})

    assert exit_code == 1
    assert residual_failure_code in failure_codes
    assert report["contract"]["fail_codes"] == [axis]
    assert axis_contract.get("report_present") is True
    assert axis_contract.get("tool_strict_pass") is True
    assert axis_contract.get("gate_ok") is False
    assert axis_contract.get("strict_no_residuals_checked") is True
    assert int(axis_contract.get("accepted_residual_count") or 0) == 1
    assert int(axis_contract.get("blocking_residual_count") or 0) == 1
    assert residual_failure_code in set(axis_contract.get("failure_codes") or [])


def test_alignment_gate_axis_contract_marks_optional_live_paper_missing_as_green(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    axis_contract = (report.get("contract") or {}).get("axes", {}).get("live_paper", {})

    assert exit_code == 0
    assert report["contract"]["fail_codes"] == []
    assert axis_contract.get("required") is False
    assert axis_contract.get("report_present") is False
    assert axis_contract.get("tool_strict_pass") is None
    assert axis_contract.get("gate_ok") is True
    assert axis_contract.get("strict_no_residuals_checked") is False
    assert report["checks"]["live_paper_required"] is False
    assert report["checks"]["live_paper_report_present"] is False
    assert report["checks"]["live_paper_tool_strict_ok"] is None
    assert report["checks"]["live_paper_ok"] is True
    assert report["checks"]["live_paper_gate_ok"] is True
    assert report["checks"]["live_paper_decision_trace_required"] is False
    assert report["checks"]["live_paper_decision_trace_report_present"] is False
    assert report["checks"]["live_paper_decision_trace_tool_strict_ok"] is None
    assert report["checks"]["live_paper_decision_trace_ok"] is True
    assert report["checks"]["live_paper_decision_trace_gate_ok"] is True


def test_alignment_gate_axis_contract_tracks_residual_counts_without_strict_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": True},
            "accepted_residuals": [{"classification": "deterministic_logic_divergence", "kind": "trade_residual"}],
            "counts": {},
        },
    )
    _write_json(
        bundle_dir / "action_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": True},
            "accepted_residuals": [{"classification": "state_initialisation_gap", "kind": "action_residual"}],
            "counts": {},
        },
    )
    _write_json(
        bundle_dir / "live_paper_action_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": True},
            "accepted_residuals": [{"classification": "deterministic_logic_divergence", "kind": "live_paper_residual"}],
            "counts": {},
        },
    )
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--require-live-paper",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    axes = ((report.get("contract") or {}).get("axes") or {})

    assert exit_code == 0
    assert report["ok"] is True
    assert report["checks"]["trade_residual_count"] == 1
    assert report["checks"]["action_residual_count"] == 1
    assert report["checks"]["live_paper_residual_count"] == 1
    assert int((axes.get("trade") or {}).get("accepted_residual_count") or 0) == 1
    assert int((axes.get("trade") or {}).get("blocking_residual_count") or 0) == 1
    assert int((axes.get("action") or {}).get("accepted_residual_count") or 0) == 1
    assert int((axes.get("action") or {}).get("blocking_residual_count") or 0) == 0
    assert int((axes.get("live_paper") or {}).get("accepted_residual_count") or 0) == 1
    assert int((axes.get("live_paper") or {}).get("blocking_residual_count") or 0) == 1
    assert (axes.get("trade") or {}).get("strict_no_residuals_checked") is False
    assert (axes.get("action") or {}).get("strict_no_residuals_checked") is False
    assert (axes.get("live_paper") or {}).get("strict_no_residuals_checked") is False


def _write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_trade_policy_mismatch_report(
    bundle_dir: Path,
    *,
    evidence_complete: bool = True,
    policy_only: bool = True,
    provenance_contract_ok: bool = True,
) -> None:
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {
            "status": {
                "strict_alignment_pass": False,
                "policy_mismatch_residual_only": bool(policy_only),
            },
            "counts": {
                "mismatch_total": 2,
                "policy_mismatch_residuals": 2,
                "deterministic_unexplained": 0,
            },
            "policy_mismatch_analysis": {
                "detected": True,
                "kind": "entry_confidence_gate",
                "evidence_complete": bool(evidence_complete),
                "reclassified_mismatch_count": 2,
                "locked_entry_policy": {
                    "global_min_confidence": "high",
                    "symbol_min_confidence": {},
                    "policy_source": "strategy_snapshot_explicit",
                    "provenance_contract_ok": bool(provenance_contract_ok),
                },
            },
            "policy_mismatch_residuals": [
                {"classification": "policy_mismatch_residual", "kind": "missing_backtester_exit"},
                {"classification": "policy_mismatch_residual", "kind": "missing_backtester_exit"},
            ],
            "accepted_residuals": [],
        },
    )


def _action_artefact_report(
    *,
    include_hard_evidence: bool = True,
    deterministic_class_count: int = 0,
    classification_kind_drift_total: int = 0,
) -> dict:
    mismatch_total = 109
    deterministic_class_count = max(0, int(deterministic_class_count))
    non_sim_class_count = max(0, mismatch_total - deterministic_class_count)
    report = {
        "status": {
            "strict_alignment_pass": False,
            "gate_pass_if_allow_compare_surface_artefacts": True,
            "artefact_only_mismatch": True,
            "logic_divergence_free": True,
        },
        "accepted_residuals": [],
        "counts": {"mismatch_total": mismatch_total},
        "mismatch_counts_by_classification": {
            "non-simulatable_exchange_oms_effect": non_sim_class_count,
            "deterministic_logic_divergence": deterministic_class_count,
        },
    }
    if include_hard_evidence:
        report["mismatch_breakdown"] = {
            "total": mismatch_total,
            "compare_surface_artefact_total": mismatch_total,
            "logic_divergence_total": 0,
            "classification_kind_drift_total": max(0, int(classification_kind_drift_total)),
        }
    return report


def test_alignment_gate_trade_policy_mismatch_stays_fail_closed_without_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_trade_policy_mismatch_report(bundle_dir)
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "trade_alignment_failed" in failure_codes
    assert report.get("checks", {}).get("trade_strict_ok") is False
    assert report.get("checks", {}).get("trade_ok") is False
    assert report.get("checks", {}).get("trade_gate_ok") is False


def test_alignment_gate_trade_policy_mismatch_requires_hard_evidence_on_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_trade_policy_mismatch_report(bundle_dir, evidence_complete=False)
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--allow-trade-policy-mismatch-residual",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "trade_policy_mismatch_opt_in_unproven" in failure_codes


def test_alignment_gate_trade_policy_mismatch_opt_in_requires_provenance_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_trade_policy_mismatch_report(bundle_dir, evidence_complete=True, provenance_contract_ok=False)
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--allow-trade-policy-mismatch-residual",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "trade_policy_mismatch_opt_in_unproven" in failure_codes


def test_alignment_gate_allows_trade_policy_mismatch_when_opted_in_and_proven(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_trade_policy_mismatch_report(bundle_dir, evidence_complete=True, policy_only=True)
    output = bundle_dir / "alignment_gate_report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--allow-trade-policy-mismatch-residual",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report.get("ok") is True
    assert report.get("checks", {}).get("trade_policy_mismatch_opt_in_applied") is True
    assert report.get("checks", {}).get("trade_strict_ok") is False
    assert report.get("checks", {}).get("trade_ok") is True
    assert report.get("checks", {}).get("trade_gate_ok") is True
    assert report.get("checks", {}).get("trade_tool_strict_ok") is False


def test_paper_harness_runs_bundle_gate_script_and_sets_strict_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output = bundle_dir / "paper_deterministic_replay_run.json"

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07c_gpu_parity.sh",
        "run_07b_event_order_parity.sh",
    ):
        _write_script(
            bundle_dir / script_name,
            (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
                'echo "$(basename "$0")" >> "$BUNDLE_DIR/steps.log"\n'
            ),
        )

    _write_script(
        bundle_dir / "run_08_assert_alignment.sh",
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'echo "strict=${STRICT_NO_RESIDUALS:-} allow_action_artefacts=${AQC_ALLOW_ACTION_ARTEFACT_RESIDUALS:-}" > "$BUNDLE_DIR/gate_env.txt"\n'
            'echo "$(basename "$0")" >> "$BUNDLE_DIR/steps.log"\n'
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper_deterministic_replay.py",
            "--bundle-dir",
            str(bundle_dir),
            "--repo-root",
            str(tmp_path),
            "--strict-no-residuals",
            "--output",
            str(output),
        ],
    )

    exit_code = paper_harness.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    gate_env = (bundle_dir / "gate_env.txt").read_text(encoding="utf-8").strip()
    steps = report.get("steps") or []
    alignment_step = next((row for row in steps if row.get("step") == "alignment_gate"), {})

    assert exit_code == 0
    assert report.get("ok") is True
    assert gate_env == "strict=1 allow_action_artefacts=0"
    assert report.get("allow_action_artefact_residuals") is False
    assert "run_08_assert_alignment.sh" in str(alignment_step.get("command") or "")


def test_paper_harness_sets_action_artefact_opt_in_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output = bundle_dir / "paper_deterministic_replay_run.json"

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07c_gpu_parity.sh",
        "run_07b_event_order_parity.sh",
    ):
        _write_script(
            bundle_dir / script_name,
            (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
                'echo "$(basename "$0")" >> "$BUNDLE_DIR/steps.log"\n'
            ),
        )

    _write_script(
        bundle_dir / "run_08_assert_alignment.sh",
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'echo "strict=${STRICT_NO_RESIDUALS:-} allow_action_artefacts=${AQC_ALLOW_ACTION_ARTEFACT_RESIDUALS:-}" > "$BUNDLE_DIR/gate_env.txt"\n'
            'echo "$(basename "$0")" >> "$BUNDLE_DIR/steps.log"\n'
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper_deterministic_replay.py",
            "--bundle-dir",
            str(bundle_dir),
            "--repo-root",
            str(tmp_path),
            "--allow-action-artefact-residuals",
            "--output",
            str(output),
        ],
    )

    exit_code = paper_harness.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    gate_env = (bundle_dir / "gate_env.txt").read_text(encoding="utf-8").strip()
    steps = report.get("steps") or []
    alignment_step = next((row for row in steps if row.get("step") == "alignment_gate"), {})

    assert exit_code == 0
    assert report.get("ok") is True
    assert report.get("allow_action_artefact_residuals") is True
    assert gate_env == "strict=0 allow_action_artefacts=1"
    assert "run_08_assert_alignment.sh" in str(alignment_step.get("command") or "")
