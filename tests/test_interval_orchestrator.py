from types import SimpleNamespace

import pytest

from tools.interval_orchestrator import orchestrate_interval_restart


def test_orchestrate_restart_clears_pause_file_on_success(tmp_path, monkeypatch):
    pause_file = tmp_path / "kill.txt"

    calls = []

    def fake_run(argv, capture_output, text, check):  # noqa: ARG001
        calls.append(list(argv))
        if argv[:3] == ["systemctl", "--user", "restart"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[:4] == ["systemctl", "--user", "is-active", "--quiet"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr("subprocess.run", fake_run)

    res = orchestrate_interval_restart(
        ws_service="ws",
        trader_service="trader",
        pause_file=pause_file,
        pause_mode="close_only",
        resume_on_success=True,
        verify_sleep_s=0.0,
    )

    assert all(r.ok for r in res)
    assert not pause_file.exists()
    assert any(c[:3] == ["systemctl", "--user", "restart"] for c in calls)


def test_orchestrate_restart_leaves_pause_file_on_restart_failure(tmp_path, monkeypatch):
    pause_file = tmp_path / "kill.txt"

    def fake_run(argv, capture_output, text, check):  # noqa: ARG001
        if argv[:3] == ["systemctl", "--user", "restart"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        if argv[:4] == ["systemctl", "--user", "is-active", "--quiet"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr("subprocess.run", fake_run)

    res = orchestrate_interval_restart(
        ws_service="ws",
        trader_service="trader",
        pause_file=pause_file,
        pause_mode="close_only",
        resume_on_success=True,
        verify_sleep_s=0.0,
    )

    assert any(not r.ok for r in res)
    assert pause_file.exists()


def test_orchestrate_restart_leaves_pause_file_on_is_active_failure(tmp_path, monkeypatch):
    pause_file = tmp_path / "kill.txt"

    def fake_run(argv, capture_output, text, check):  # noqa: ARG001
        if argv[:3] == ["systemctl", "--user", "restart"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[:4] == ["systemctl", "--user", "is-active", "--quiet"]:
            # Fail for trader
            svc = argv[4]
            return SimpleNamespace(returncode=0 if svc == "ws" else 3, stdout="", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr("subprocess.run", fake_run)

    res = orchestrate_interval_restart(
        ws_service="ws",
        trader_service="trader",
        pause_file=pause_file,
        pause_mode="close_only",
        resume_on_success=True,
        verify_sleep_s=0.0,
    )

    assert any(not r.ok for r in res)
    assert pause_file.exists()

