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
