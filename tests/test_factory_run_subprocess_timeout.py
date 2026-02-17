from __future__ import annotations

import sys
from pathlib import Path

import factory_run


def test_run_cmd_reports_timeout_and_non_zero_exit(tmp_path: Path) -> None:
    out_path = tmp_path / "cmd.stdout.txt"
    err_path = tmp_path / "cmd.stderr.txt"

    res = factory_run._run_cmd(
        [sys.executable, "-c", "import time; time.sleep(1.0)"],
        cwd=tmp_path,
        stdout_path=out_path,
        stderr_path=err_path,
        timeout_s=0.1,
    )

    assert res.timed_out is True
    assert res.exit_code == 124
    assert res.timeout_s is not None
    assert res.timeout_s == 0.1
    assert res.elapsed_s < 1.0
    assert "timed out" in err_path.read_text(encoding="utf-8").lower()


def test_run_cmd_non_timeout_path(tmp_path: Path) -> None:
    out_path = tmp_path / "ok.stdout.txt"
    err_path = tmp_path / "ok.stderr.txt"

    res = factory_run._run_cmd(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        stdout_path=out_path,
        stderr_path=err_path,
        timeout_s=2.0,
    )

    assert res.timed_out is False
    assert res.exit_code == 0
    assert res.timeout_s == 2.0
    assert out_path.read_text(encoding="utf-8").strip() == "ok"


def test_run_cmd_timeout_can_be_disabled(tmp_path: Path) -> None:
    out_path = tmp_path / "no_timeout.stdout.txt"
    err_path = tmp_path / "no_timeout.stderr.txt"

    res = factory_run._run_cmd(
        [sys.executable, "-c", "import time; time.sleep(0.2); print('done')"],
        cwd=tmp_path,
        stdout_path=out_path,
        stderr_path=err_path,
        timeout_s=0,
    )

    assert res.timed_out is False
    assert res.exit_code == 0
    assert res.timeout_s is None
    assert out_path.read_text(encoding="utf-8").strip() == "done"


def test_run_cmd_respects_env_default_timeout_disable(
    tmp_path: Path, monkeypatch
) -> None:
    out_path = tmp_path / "env_disable.stdout.txt"
    err_path = tmp_path / "env_disable.stderr.txt"
    monkeypatch.setenv("AI_QUANT_FACTORY_CMD_TIMEOUT_S", "0")

    res = factory_run._run_cmd(
        [sys.executable, "-c", "import time; time.sleep(0.2); print('done')"],
        cwd=tmp_path,
        stdout_path=out_path,
        stderr_path=err_path,
        timeout_s=None,
    )

    assert res.timed_out is False
    assert res.exit_code == 0
    assert res.timeout_s is None
    assert out_path.read_text(encoding="utf-8").strip() == "done"
