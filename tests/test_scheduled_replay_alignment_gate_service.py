from __future__ import annotations

import json
from pathlib import Path

import tools.run_scheduled_replay_alignment_gate_service as replay_gate_service


def test_is_fresh_blocker_update_accepts_blocked_schema_v2_payload(tmp_path: Path) -> None:
    report_path = tmp_path / "scheduled_alignment_gate_run.json"
    blocker_path = tmp_path / "release_blocker.json"
    report_path.write_text("{}", encoding="utf-8")
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_ms": 5_000,
                "blocked": True,
                "report_path": str(report_path),
            }
        ),
        encoding="utf-8",
    )

    assert replay_gate_service._is_fresh_blocker_update(blocker_path, start_ms=4_000) is True


def test_is_fresh_blocker_update_rejects_fallback_exception_payload(tmp_path: Path) -> None:
    blocker_path = tmp_path / "release_blocker.json"
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_ms": 5_000,
                "blocked": True,
                "reason_codes": ["scheduler_unhandled_exception"],
            }
        ),
        encoding="utf-8",
    )

    assert replay_gate_service._is_fresh_blocker_update(blocker_path, start_ms=4_000) is False


def test_main_normalises_blocked_exit_to_success(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "scheduled_alignment_gate_run.json"
    blocker_path = tmp_path / "release_blocker.json"
    report_path.write_text("{}", encoding="utf-8")
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at_ms": 5_000,
                "blocked": True,
                "report_path": str(report_path),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_QUANT_REPLAY_GATE_BLOCKER_FILE", str(blocker_path))
    monkeypatch.setattr(replay_gate_service, "_now_ms", lambda: 4_000)

    class _Proc:
        returncode = 1

    monkeypatch.setattr(replay_gate_service.subprocess, "run", lambda *args, **kwargs: _Proc())

    assert replay_gate_service.main([]) == 0


def test_main_preserves_non_blocker_failures(tmp_path: Path, monkeypatch) -> None:
    blocker_path = tmp_path / "release_blocker.json"
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_ms": 5_000,
                "blocked": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_QUANT_REPLAY_GATE_BLOCKER_FILE", str(blocker_path))
    monkeypatch.setattr(replay_gate_service, "_now_ms", lambda: 4_000)

    class _Proc:
        returncode = 1

    monkeypatch.setattr(replay_gate_service.subprocess, "run", lambda *args, **kwargs: _Proc())

    assert replay_gate_service.main([]) == 1
