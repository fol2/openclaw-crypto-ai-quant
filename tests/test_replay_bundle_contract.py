from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

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
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "state_alignment_report.json", {"ok": True, "summary": {"diff_count": 0}})
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {"status": {"strict_alignment_pass": True}, "accepted_residuals": [], "counts": {}},
    )
    _write_json(
        bundle_dir / "action_reconcile_report.json",
        {"status": {"strict_alignment_pass": True}, "accepted_residuals": [], "counts": {}},
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


def _write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


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
            'echo "strict=${STRICT_NO_RESIDUALS:-}" > "$BUNDLE_DIR/gate_env.txt"\n'
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
    assert gate_env == "strict=1"
    assert "run_08_assert_alignment.sh" in str(alignment_step.get("command") or "")
