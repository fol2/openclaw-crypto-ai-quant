from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys

import pytest

from tools import assert_replay_bundle_alignment as alignment_gate
from tools import build_live_replay_bundle as replay_bundle_builder
from tools import run_paper_deterministic_replay as paper_harness


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _init_live_db_for_resolution(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                action TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE runtime_logs (
                id INTEGER PRIMARY KEY,
                ts_ms INTEGER,
                message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE oms_intents (
                intent_id TEXT PRIMARY KEY,
                created_ts_ms INTEGER,
                decision_ts_ms INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_trade_row(path: Path, *, trade_id: int, timestamp: str, action: str = "OPEN") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO trades (id, timestamp, action) VALUES (?, ?, ?)",
            (int(trade_id), str(timestamp), str(action)),
        )
        conn.commit()
    finally:
        conn.close()


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


def test_live_replay_scope_summariser_tracks_symbol_side_pairs() -> None:
    payload = replay_bundle_builder._summarise_live_replay_scope(
        [
            {"symbol": "POL", "action": "OPEN", "type": "SHORT"},
            {"symbol": "POL", "action": "CLOSE", "type": "SHORT"},
            {"symbol": "NEAR", "action": "OPEN", "type": "LONG"},
            {"symbol": "POL", "action": "FUNDING", "type": ""},
            {"symbol": "BAD", "action": "UNKNOWN", "type": "LONG"},
        ]
    )

    assert payload["live_total_rows"] == 5
    assert payload["live_canonical_side_action_rows"] == 3
    assert payload["live_funding_rows"] == 1
    assert payload["symbols"] == ["NEAR", "POL"]
    assert payload["sides"] == ["LONG", "SHORT"]
    assert payload["symbol_sides"] == ["NEAR:LONG", "POL:SHORT"]


def test_live_db_window_resolver_auto_switches_to_single_covered_candidate(tmp_path: Path) -> None:
    requested = tmp_path / "trading_engine_v8_live.db"
    alternate = tmp_path / "trading_engine_live.db"
    _init_live_db_for_resolution(requested)
    _init_live_db_for_resolution(alternate)
    _insert_trade_row(alternate, trade_id=1, timestamp="2026-03-03T12:00:00+00:00", action="OPEN")

    resolved, resolution = replay_bundle_builder._resolve_live_db_for_window(
        requested_live_db=requested.resolve(),
        from_ts=1772535600000,
        to_ts=1772542800000,
        live_window_to_ts=1772544599999,
        strict_live_db_path=False,
        allow_empty_live_window=False,
    )

    assert resolved == alternate.resolve()
    assert resolution["auto_switched"] is True
    assert resolution["resolution_reason"] == "auto_switched_higher_window_coverage_candidate"
    requested_cov = resolution["coverage_by_db"][str(requested.resolve())]
    alternate_cov = resolution["coverage_by_db"][str(alternate.resolve())]
    assert requested_cov["has_any_rows"] is False
    assert alternate_cov["has_any_rows"] is True
    assert alternate_cov["live_baseline_trades"] == 1


def test_live_db_window_resolver_fails_when_multiple_alternates_have_coverage(tmp_path: Path) -> None:
    requested = tmp_path / "custom_live.db"
    candidate_live = tmp_path / "trading_engine_live.db"
    candidate_v8 = tmp_path / "trading_engine_v8_live.db"
    _init_live_db_for_resolution(requested)
    _init_live_db_for_resolution(candidate_live)
    _init_live_db_for_resolution(candidate_v8)
    _insert_trade_row(candidate_live, trade_id=1, timestamp="2026-03-03T12:00:00+00:00", action="OPEN")
    _insert_trade_row(candidate_v8, trade_id=2, timestamp="2026-03-03T12:30:00+00:00", action="OPEN")

    with pytest.raises(ValueError, match="multiple live DB candidates share the best replay-window coverage"):
        replay_bundle_builder._resolve_live_db_for_window(
            requested_live_db=requested.resolve(),
            from_ts=1772535600000,
            to_ts=1772542800000,
            live_window_to_ts=1772544599999,
            strict_live_db_path=False,
            allow_empty_live_window=False,
        )


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


def test_alignment_gate_runtime_window_empty_does_not_emit_locked_runtime_prefix_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
    )

    snapshot_path = bundle_dir / "strategy_overrides.locked.yaml"
    snapshot_path.write_text("global:\n  engine:\n    interval: 30m\n", encoding="utf-8")
    obj = {"global": {"engine": {"interval": "30m"}}}
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest()
    sha1_legacy = hashlib.sha1(payload).hexdigest()

    manifest = json.loads((bundle_dir / "replay_bundle_manifest.json").read_text(encoding="utf-8"))
    manifest["artefacts"]["strategy_config_snapshot_file"] = snapshot_path.name
    manifest["runtime_strategy_provenance"] = {
        "window_from_ts": 1_000,
        "window_to_ts": 2_000,
        "runtime_rows_in_window": 0,
        "strategy_rows_sampled": 0,
        "strategy_sha1_distinct": 0,
        "strategy_version_distinct": 0,
        "strategy_sha1_timeline": [],
    }
    manifest["oms_strategy_provenance"] = {
        "window_from_ts": 1_000,
        "window_to_ts": 2_000,
        "oms_rows_in_window": 0,
        "oms_rows_sampled": 0,
        "strategy_sha1_distinct": 0,
        "strategy_version_distinct": 0,
        "strategy_sha1_timeline": [],
    }
    manifest["locked_strategy_provenance"] = {
        "strategy_overrides_sha1": sha256,
        "strategy_overrides_sha1_legacy": sha1_legacy,
        "strategy_overrides_snapshot_sha1": sha256,
        "strategy_overrides_snapshot_sha1_legacy": sha1_legacy,
    }
    (bundle_dir / "replay_bundle_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    output = bundle_dir / "alignment_gate_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--require-runtime-strategy-provenance",
            "--require-locked-strategy-match",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )
    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "runtime_strategy_provenance_window_empty" in failure_codes
    assert "locked_strategy_runtime_prefix_mismatch" not in failure_codes


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


def test_alignment_gate_action_paper_window_not_replayed_is_blocking_even_with_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        seed_apply_report={"strict_replace": False},
        action_report=_action_artefact_report(paper_window_not_replayed=True),
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
    assert "action_paper_window_not_replayed" in failure_codes
    assert report["checks"]["action_paper_window_not_replayed"] is True
    assert report["checks"]["action_ok"] is False


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


def test_alignment_gate_decision_trace_empty_paper_window_stays_blocking_under_strict_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(
        bundle_dir,
        manifest_inputs={"snapshot_strict_replace": True},
        seed_apply_report={"strict_replace": True},
    )
    _write_json(
        bundle_dir / "live_paper_decision_trace_reconcile_report.json",
        {
            "status": {
                "strict_alignment_pass": False,
                "paper_window_not_replayed": True,
            },
            "counts": {
                "live_decision_rows": 12,
                "paper_decision_rows": 0,
            },
            "accepted_residuals": [],
            "mismatches": [],
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
            "--require-live-paper-decision-trace",
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "live_paper_decision_trace_paper_window_not_replayed" in failure_codes
    assert report["checks"]["live_paper_decision_trace_paper_window_not_replayed"] is True
    assert report["checks"]["live_paper_decision_trace_gate_ok"] is False


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
    runtime_provenance_contract_ok: bool = True,
    runtime_source_contract_ok: bool = True,
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
                "runtime_entry_policy": {
                    "policy_source": "decision_events_context_json",
                    "provenance_contract_ok": bool(runtime_provenance_contract_ok),
                    "source_contract_ok": bool(runtime_source_contract_ok),
                    "rows_with_non_fallback_confidence_source": 2,
                    "rows_with_policy": 2,
                    "match_verified": 2,
                    "match_fallback": 0,
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
    paper_window_not_replayed: bool = False,
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
            "paper_window_not_replayed": bool(paper_window_not_replayed),
        },
        "accepted_residuals": [],
        "counts": {
            "mismatch_total": mismatch_total,
            "live_simulatable_actions": 12,
            "paper_simulatable_actions": 0 if paper_window_not_replayed else 12,
        },
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


@pytest.mark.parametrize(
    ("strict_env", "allow_action_env", "expected_flags"),
    [
        (None, None, []),
        ("1", "1", ["--strict-no-residuals", "--allow-action-artefact-residuals"]),
    ],
)
def test_bundle_run_09_script_wires_harness_flags_without_blank_args(
    tmp_path: Path,
    strict_env: str | None,
    allow_action_env: str | None,
    expected_flags: list[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    stub_harness = tools_dir / "run_paper_deterministic_replay.py"
    _write_script(
        stub_harness,
        (
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "args = list(sys.argv[1:])\n"
            "out_path = ''\n"
            "for idx, token in enumerate(args):\n"
            "    if token == '--output' and idx + 1 < len(args):\n"
            "        out_path = args[idx + 1]\n"
            "        break\n"
            "if not out_path:\n"
            "    raise SystemExit('missing --output')\n"
            "Path(out_path).write_text(json.dumps({'argv': args}), encoding='utf-8')\n"
        ),
    )

    run_09 = bundle_dir / "run_09_paper_deterministic_replay.sh"
    _write_script(
        run_09,
        replay_bundle_builder._render_run_09_paper_harness_script(
            paper_harness_report_name="paper_deterministic_replay_run.json"
        ),
    )

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo_root)
    if strict_env is not None:
        env["STRICT_NO_RESIDUALS"] = strict_env
    if allow_action_env is not None:
        env["AQC_ALLOW_ACTION_ARTEFACT_RESIDUALS"] = allow_action_env

    proc = subprocess.run(
        ["bash", str(run_09)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    report = json.loads((bundle_dir / "paper_deterministic_replay_run.json").read_text(encoding="utf-8"))
    argv = [str(v) for v in (report.get("argv") or [])]

    assert argv[:4] == [
        "--bundle-dir",
        str(bundle_dir),
        "--repo-root",
        str(repo_root),
    ]
    assert argv[-2:] == ["--output", str(bundle_dir / "paper_deterministic_replay_run.json")]
    assert all(arg != "" and arg.strip() != "" for arg in argv)
    for flag in expected_flags:
        assert flag in argv


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


def test_alignment_gate_emits_scope_contract_failures_for_trade_and_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": False, "scope_contract_mismatch": True},
            "accepted_residuals": [],
            "counts": {"matched_pairs": 0},
            "scope_contract": {"mismatch": True, "mismatch_kind": "symbol_side_scope_disjoint"},
        },
    )
    _write_json(
        bundle_dir / "action_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": False, "scope_contract_mismatch": True},
            "accepted_residuals": [],
            "counts": {"matched_pairs": 0},
            "scope_contract": {"mismatch": True, "mismatch_kind": "symbol_side_scope_disjoint"},
            "mismatch_breakdown": {
                "total": 1,
                "compare_surface_artefact_total": 0,
                "logic_divergence_total": 1,
                "classification_kind_drift_total": 0,
            },
            "mismatch_counts_by_classification": {"deterministic_logic_divergence": 1},
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
            "--skip-candles-provenance-check",
            "--output",
            str(output),
        ],
    )

    exit_code = alignment_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    failure_codes = {str(row.get("code") or "") for row in report.get("failures") or []}

    assert exit_code == 1
    assert "trade_replay_scope_contract_mismatch" in failure_codes
    assert "action_replay_scope_contract_mismatch" in failure_codes
    assert report.get("checks", {}).get("trade_scope_contract_mismatch") is True
    assert report.get("checks", {}).get("action_scope_contract_mismatch") is True


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


def test_alignment_gate_trade_policy_mismatch_opt_in_requires_runtime_provenance_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_base_gate_bundle(bundle_dir, seed_apply_report={"strict_replace": False})
    _write_trade_policy_mismatch_report(
        bundle_dir,
        evidence_complete=True,
        provenance_contract_ok=True,
        runtime_provenance_contract_ok=False,
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
        "run_03b_mirror_live_window_to_paper.sh",
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
    executed_scripts = [
        line.strip()
        for line in (bundle_dir / "steps.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = report.get("steps") or []
    alignment_step = next((row for row in steps if row.get("step") == "alignment_gate"), {})
    step_names = [str(row.get("step") or "") for row in steps]
    step_indices = [int(row.get("step_index") or 0) for row in steps]

    assert exit_code == 0
    assert report.get("ok") is True
    assert gate_env == "strict=1 allow_action_artefacts=0"
    assert report.get("allow_action_artefact_residuals") is False
    assert report.get("planned_steps") == [
        "export_and_seed",
        "replay",
        "state_audit",
        "paper_window_mirror",
        "trade_reconcile",
        "action_reconcile",
        "live_paper_action_reconcile",
        "live_paper_decision_trace_reconcile",
        "event_order_parity",
        "gpu_parity",
        "alignment_gate",
    ]
    assert step_names == report.get("planned_steps")
    assert step_indices == list(range(1, len(step_indices) + 1))
    assert executed_scripts == [
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_03b_mirror_live_window_to_paper.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07b_event_order_parity.sh",
        "run_07c_gpu_parity.sh",
        "run_08_assert_alignment.sh",
    ]
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
        "run_03b_mirror_live_window_to_paper.sh",
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


def test_paper_harness_wires_db_overrides_into_steps_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output = bundle_dir / "paper_deterministic_replay_run.json"

    live_db = tmp_path / "live.sqlite"
    paper_db = tmp_path / "paper.sqlite"
    candles_db = tmp_path / "candles.sqlite"
    funding_db = tmp_path / "funding.sqlite"
    live_db.write_text("", encoding="utf-8")
    paper_db.write_text("", encoding="utf-8")
    candles_db.write_text("", encoding="utf-8")
    funding_db.write_text("", encoding="utf-8")

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_03b_mirror_live_window_to_paper.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07b_event_order_parity.sh",
        "run_07c_gpu_parity.sh",
    ):
        _write_script(
            bundle_dir / script_name,
            (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
                'echo "live=${LIVE_DB:-} paper=${PAPER_DB:-} candles=${CANDLES_DB:-} funding=${FUNDING_DB:-}" > "$BUNDLE_DIR/pre_gate_env.txt"\n'
            ),
        )

    _write_script(
        bundle_dir / "run_08_assert_alignment.sh",
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'echo "live=${LIVE_DB:-} paper=${PAPER_DB:-} candles=${CANDLES_DB:-} funding=${FUNDING_DB:-} strict=${STRICT_NO_RESIDUALS:-} action=${AQC_ALLOW_ACTION_ARTEFACT_RESIDUALS:-}" > "$BUNDLE_DIR/gate_env.txt"\n'
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
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--candles-db",
            str(candles_db),
            "--funding-db",
            str(funding_db),
            "--allow-action-artefact-residuals",
            "--output",
            str(output),
        ],
    )

    exit_code = paper_harness.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    pre_gate_env = (bundle_dir / "pre_gate_env.txt").read_text(encoding="utf-8").strip()
    gate_env = (bundle_dir / "gate_env.txt").read_text(encoding="utf-8").strip()

    expected_env_prefix = (
        f"live={live_db.resolve()} paper={paper_db.resolve()} "
        f"candles={candles_db.resolve()} funding={funding_db.resolve()}"
    )

    assert exit_code == 0
    assert report.get("ok") is True
    assert pre_gate_env == expected_env_prefix
    assert gate_env == f"{expected_env_prefix} strict=0 action=1"
    assert report.get("effective_inputs") == {
        "live_db": str(live_db.resolve()),
        "paper_db": str(paper_db.resolve()),
        "candles_db": str(candles_db.resolve()),
        "funding_db": str(funding_db.resolve()),
    }


def test_build_bundle_mirror_script_enables_replace_id_collisions_and_persists_live_db_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    requested_live_db = tmp_path / "trading_engine_v8_live.db"
    resolved_live_db = tmp_path / "trading_engine_live.db"
    paper_db = tmp_path / "paper.db"
    candles_db = tmp_path / "candles_30m.db"
    strategy_cfg = tmp_path / "strategy_overrides.live.yaml"
    bundle_dir = tmp_path / "bundle"

    for path in (requested_live_db, resolved_live_db, paper_db, candles_db):
        path.write_bytes(b"")
    strategy_cfg.write_text(
        (
            "global:\n"
            "  engine:\n"
            "    interval: 30m\n"
            "    entry_interval: 30m\n"
            "    exit_interval: 30m\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay_bundle_builder,
        "_resolve_live_db_for_window",
        lambda **_: (
            resolved_live_db.resolve(),
            {
                "requested_live_db": str(requested_live_db.resolve()),
                "resolved_live_db": str(resolved_live_db.resolve()),
                "resolution_reason": "auto_switched_single_window_coverage_candidate",
                "auto_switched": True,
                "strict_live_db_path": False,
                "allow_empty_live_window": False,
                "window_from_ts": 0,
                "window_to_ts": 0,
                "window_live_to_ts": 1_799_999,
                "coverage_by_db": {
                    str(requested_live_db.resolve()): {"has_any_rows": False},
                    str(resolved_live_db.resolve()): {"has_any_rows": True, "live_baseline_trades": 1},
                },
            },
        ),
    )
    monkeypatch.setattr(replay_bundle_builder, "_load_live_baseline_trades", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(replay_bundle_builder, "_load_live_order_fail_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        replay_bundle_builder,
        "_load_runtime_strategy_provenance",
        lambda *_args, **_kwargs: {
            "window_from_ts": 0,
            "window_to_ts": 0,
            "runtime_rows_in_window": 0,
            "strategy_rows_sampled": 0,
            "strategy_sha1_distinct": 0,
            "strategy_version_distinct": 0,
            "strategy_sha1_timeline": [],
        },
    )
    monkeypatch.setattr(
        replay_bundle_builder,
        "_load_oms_strategy_provenance",
        lambda *_args, **_kwargs: {
            "window_from_ts": 0,
            "window_to_ts": 0,
            "oms_rows_in_window": 0,
            "oms_rows_sampled": 0,
            "strategy_sha1_distinct": 0,
            "strategy_version_distinct": 0,
            "strategy_sha1_timeline": [],
        },
    )
    monkeypatch.setattr(
        replay_bundle_builder,
        "build_candles_window_provenance",
        lambda *_args, **_kwargs: {
            "interval": "30m",
            "from_ts": 0,
            "to_ts": 0,
            "row_count": 0,
            "symbol_count": 0,
            "symbols": [],
            "window_hash_sha256": "0" * 64,
            "universe_hash_sha256": "1" * 64,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_live_replay_bundle.py",
            "--live-db",
            str(requested_live_db),
            "--paper-db",
            str(paper_db),
            "--candles-db",
            str(candles_db),
            "--strategy-config",
            str(strategy_cfg),
            "--interval",
            "30m",
            "--from-ts",
            "0",
            "--to-ts",
            "0",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-empty-live-window",
            "--enable-live-paper-mirror",
        ],
    )

    rc = replay_bundle_builder.main()
    assert rc == 0

    mirror_script = (bundle_dir / "run_03b_mirror_live_window_to_paper.sh").read_text(encoding="utf-8")
    manifest = json.loads((bundle_dir / "replay_bundle_manifest.json").read_text(encoding="utf-8"))

    assert "--replace-id-collisions" in mirror_script
    assert manifest["inputs"]["requested_live_db"] == str(requested_live_db.resolve())
    assert manifest["inputs"]["live_db"] == str(resolved_live_db.resolve())
    assert manifest["inputs"]["mirror_replace_id_collisions"] is True
    assert manifest["live_db_resolution"]["auto_switched"] is True
