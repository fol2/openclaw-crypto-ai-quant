from __future__ import annotations

import json

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


def test_factory_run_replay_equivalence_happy_path(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    _write_trace(baseline, TRACE)
    _write_trace(replay, TRACE)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "1")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok
    assert summary["replay_equivalence_status"] == "pass"
    assert summary["replay_equivalence_report_path"] != ""

    report_path = Path(summary["replay_equivalence_report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True


def test_factory_run_replay_equivalence_detects_mismatch_when_strict(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    changed = json.loads(json.dumps(TRACE))
    changed["decision_diagnostics"][0]["event_id"] = 2

    _write_trace(baseline, TRACE)
    _write_trace(replay, changed)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "1")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert not ok
    assert summary["replay_equivalence_status"] == "fail"
    assert summary["replay_equivalence_count"] >= 1


def test_factory_run_replay_equivalence_runs_as_warning_when_not_strict(monkeypatch, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    replay = tmp_path / "replay.json"
    changed = json.loads(json.dumps(TRACE))
    changed["decision_diagnostics"][0]["event_id"] = 3

    _write_trace(baseline, TRACE)
    _write_trace(replay, changed)

    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_BASELINE", str(baseline))
    monkeypatch.setenv("AI_QUANT_REPLAY_EQUIVALENCE_STRICT", "0")

    summary: dict = {}
    ok = _run_replay_equivalence_check(right_report=replay, summary=summary)

    assert ok
    assert summary["replay_equivalence_status"] == "fail"
