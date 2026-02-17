import sys

from tools.sensitivity_check import _run_cmd


def test_run_cmd_success_reports_no_timeout(tmp_path):
    stdout_path = tmp_path / "ok.stdout.txt"
    stderr_path = tmp_path / "ok.stderr.txt"
    res = _run_cmd(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=5.0,
    )
    assert res.exit_code == 0
    assert res.timed_out is False
    assert "ok" in stdout_path.read_text(encoding="utf-8")


def test_run_cmd_timeout_returns_124(tmp_path):
    stdout_path = tmp_path / "timeout.stdout.txt"
    stderr_path = tmp_path / "timeout.stderr.txt"
    res = _run_cmd(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=0.1,
    )
    assert res.exit_code == 124
    assert res.timed_out is True

