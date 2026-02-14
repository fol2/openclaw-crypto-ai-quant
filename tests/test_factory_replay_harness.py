from __future__ import annotations

import json
import factory_run

from pathlib import Path

import pytest

from factory_run import _run_replay_equivalence_check


TRACE = {
    "decision_diagnostics": [
        {
            "event_id": 1,
            "source": "fixture",
            "timestamp_ms": 1700000000000,
            "symbol": "ETH",
            "signal": "BUY",
            "requested_notional_usd": 1000.0,
            "schema_version": 1,
            "intents": [],
            "fills": [],
            "warnings": [],
            "errors": [],
            "applied_to_kernel_state": True,
        }
    ]
}


def _write_trace(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _force_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_MODE", mode)


def test_factory_run_replay_equivalence_happy_path(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    _write_trace(baseline, TRACE)
    _write_trace(replay, TRACE)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "1")
    _force_mode(monkeypatch, "backtest")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok
    assert summary["replay_equivalence_status"] == "pass"
    assert summary["replay_equivalence_mode"] == "backtest"
    assert summary["replay_equivalence_failure_code"] == ""
    assert summary["replay_equivalence_report_path"] != ""

    report_path = Path(summary["replay_equivalence_report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["mode"] == "backtest"


def test_factory_run_replay_equivalence_detects_mismatch_when_strict(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    changed = json.loads(json.dumps(TRACE))
    changed["decision_diagnostics"][0]["event_id"] = 2

    _write_trace(baseline, TRACE)
    _write_trace(replay, changed)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "1")
    _force_mode(monkeypatch, "backtest")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert not ok
    assert summary["replay_equivalence_status"] == "fail"
    assert summary["replay_equivalence_mode"] == "backtest"
    assert summary["replay_equivalence_count"] >= 1
    assert summary["replay_equivalence_failure_code"] == "mismatch"


def test_factory_run_replay_equivalence_runs_as_warning_when_not_strict(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    changed = json.loads(json.dumps(TRACE))
    changed["decision_diagnostics"][0]["event_id"] = 3

    _write_trace(baseline, TRACE)
    _write_trace(replay, changed)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "0")
    _force_mode(monkeypatch, "backtest")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok
    assert summary["replay_equivalence_status"] == "fail"
    assert summary["replay_equivalence_mode"] == "backtest"
    assert summary["replay_equivalence_failure_code"] == "mismatch"


def test_factory_run_replay_equivalence_not_run_without_baseline(monkeypatch, tmp_path) -> None:
    replay = tmp_path / "replay.json"
    _write_trace(replay, TRACE)

    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", raising=False)
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "0")
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_LIVE_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_PAPER_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BACKTEST_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BACKTEST_STRICT", raising=False)
    _force_mode(monkeypatch, "backtest")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok
    assert summary["replay_equivalence_status"] == "not_run"
    assert summary["replay_equivalence_count"] == 0
    assert summary["replay_equivalence_mode"] == "backtest"
    assert summary["replay_equivalence_diffs"] == []
    assert summary["replay_equivalence_failure_code"] == "not_run"


def test_factory_run_replay_equivalence_missing_baseline_fails_in_strict_mode(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "missing.json"
    replay = tmp_path / "replay.json"
    _write_trace(replay, TRACE)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "1")
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_LIVE_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_PAPER_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BACKTEST_BASELINE", raising=False)
    monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BACKTEST_STRICT", raising=False)
    _force_mode(monkeypatch, "backtest")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert not ok
    assert summary["replay_equivalence_status"] == "missing_baseline"
    assert summary["replay_equivalence_count"] == 0
    assert summary["replay_equivalence_mode"] == "backtest"
    assert summary["replay_equivalence_failure_code"] == "missing_baseline"


@pytest.mark.parametrize(
    ("mode", "mode_baseline", "mode_strict", "global_strict", "strict_result"),
    [
        ("live", True, "1", "0", False),
        ("paper", True, "0", "1", True),
        ("backtest", False, None, "1", False),
        ("backtest", False, None, "0", True),
    ],
)
def test_factory_run_replay_equivalence_mode_specific_strictness_matrix(
    monkeypatch,
    tmp_path,
    mode: str,
    mode_baseline: bool,
    mode_strict: str | None,
    global_strict: str,
    strict_result: bool,
) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    changed = json.loads(json.dumps(TRACE))
    changed["decision_diagnostics"][0]["event_id"] = 4

    _write_trace(baseline, TRACE)
    _write_trace(replay, changed)

    if mode_baseline:
        monkeypatch.setenv(f"AI_QUANT_REPLAY_EQUIVALENCE_{mode.upper()}_BASELINE", str(baseline))
        monkeypatch.delenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", raising=False)
    else:
        monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
        monkeypatch.delenv(f"AI_QUANT_REPLAY_EQUIVALENCE_{mode.upper()}_BASELINE", raising=False)

    if mode_strict is None:
        monkeypatch.delenv(f"AI_QUANT_REPLAY_EQUIVALENCE_{mode.upper()}_STRICT", raising=False)
    else:
        monkeypatch.setenv(f"AI_QUANT_REPLAY_EQUIVALENCE_{mode.upper()}_STRICT", mode_strict)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", global_strict)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_MODE", mode)

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok is strict_result
    assert summary["replay_equivalence_mode"] == mode
    assert summary["replay_equivalence_status"] == "fail"
    assert summary["replay_equivalence_failure_code"] == "mismatch"


def _write_run_metadata(*, run_dir: Path, candidate_id: str, config_path: str, replay_report: str) -> None:
    (run_dir / "replays").mkdir(parents=True, exist_ok=True)
    (run_dir / "replays" / replay_report).write_text("{}\n", encoding="utf-8")
    candidate = {
        "config_id": candidate_id,
        "path": str(config_path),
        "replay_report_path": str(run_dir / "replays" / replay_report),
        "pipeline_stage": "candidate_validation",
        "sweep_stage": "gpu",
        "replay_stage": "cpu_replay",
        "validation_gate": "replay_only",
        "canonical_cpu_verified": True,
        "candidate_mode": True,
    }
    run_meta = {
        "run_id": "baseline-run",
        "candidate_configs": [candidate],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta), encoding="utf-8")


def test_factory_run_resolve_baseline_aligns_to_candidate_run_metadata(tmp_path: Path) -> None:
    baseline_run = tmp_path / "baseline_run"
    right_run = tmp_path / "right_run"
    baseline_run.mkdir()
    right_run.mkdir()

    _write_run_metadata(
        run_dir=baseline_run,
        candidate_id="cfg-alpha",
        config_path=str(tmp_path / "candidate_alpha.yaml"),
        replay_report="candidate_alpha.replay.json",
    )
    (tmp_path / "candidate_alpha.yaml").write_text("candidate: alpha\n", encoding="utf-8")

    right_report = right_run / "candidate_alpha.replay.json"
    right_report.write_text("{}", encoding="utf-8")

    baseline = factory_run._resolve_replay_equivalence_baseline_path(
        "backtest",
        baseline_path=baseline_run / "replays" / "candidate_alpha.replay.json",
        right_report=right_report,
        summary={"config_id": "cfg-alpha", "config_path": str(tmp_path / "candidate_alpha.yaml")},
    )
    assert baseline == (baseline_run / "replays" / "candidate_alpha.replay.json")


def test_factory_run_resolve_baseline_aligns_by_filename_fallback(tmp_path: Path) -> None:
    raw_dir = tmp_path / "baseline_dir"
    raw_dir.mkdir()
    baseline_dir = raw_dir / "replays"
    baseline_dir.mkdir()

    (baseline_dir / "candidate_alpha.replay.json").write_text("{}", encoding="utf-8")
    (baseline_dir / "candidate_beta.replay.json").write_text("{}", encoding="utf-8")

    right_report = tmp_path / "candidate_beta.replay.json"
    right_report.write_text("{}", encoding="utf-8")

    resolved = factory_run._resolve_replay_equivalence_baseline_path(
        "backtest",
        baseline_path=baseline_dir / "candidate_alpha.replay.json",
        right_report=right_report,
        summary={"config_id": "unknown", "config_path": "/tmp/missing.yaml"},
    )
    assert resolved == baseline_dir / "candidate_beta.replay.json"


def test_factory_run_resolve_baseline_run_dir_uses_metadata_match(tmp_path: Path) -> None:
    baseline_run = tmp_path / "baseline_run"
    right_dir = tmp_path / "right"
    baseline_run.mkdir()
    right_dir.mkdir()

    _write_run_metadata(
        run_dir=baseline_run,
        candidate_id="cfg-beta",
        config_path=str(tmp_path / "candidate_beta.yaml"),
        replay_report="candidate_beta.replay.json",
    )
    (baseline_run / "replays" / "candidate_beta.replay.json").write_text("{}", encoding="utf-8")
    right_report = right_dir / "candidate_beta.replay.json"
    right_report.write_text("{}", encoding="utf-8")

    (tmp_path / "candidate_beta.yaml").write_text("candidate: beta\n", encoding="utf-8")

    resolved = factory_run._resolve_replay_equivalence_baseline_path(
        "backtest",
        baseline_path=baseline_run,
        right_report=right_report,
        summary={"config_id": "cfg-beta", "config_path": str(tmp_path / "candidate_beta.yaml")},
    )
    assert resolved == baseline_run / "replays" / "candidate_beta.replay.json"
