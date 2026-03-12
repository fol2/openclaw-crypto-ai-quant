#!/usr/bin/env python3
"""Systemd-friendly entrypoint for the scheduled replay alignment gate.

The scheduled gate intentionally exits with status 1 when the gate is blocked.
That is useful for deployment tooling, but it leaves the systemd oneshot unit
in a failed state even when the scheduler completed normally and wrote a fresh
release blocker report. This wrapper keeps genuine crashes red while treating a
fresh, well-formed blocker update as an operationally successful run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _resolve_blocker_path(*, cwd: Path) -> Path:
    raw = str(os.getenv("AI_QUANT_REPLAY_GATE_BLOCKER_FILE", "") or "").strip()
    if not raw:
        bundle_root = str(os.getenv("AI_QUANT_REPLAY_GATE_BUNDLE_ROOT", "") or "").strip()
        if bundle_root:
            raw = str(Path(bundle_root).expanduser() / "release_blocker.json")
        else:
            raw = "/tmp/openclaw-ai-quant/replay_gate/release_blocker.json"

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_fresh_blocker_update(path: Path, *, start_ms: int) -> bool:
    try:
        payload = _read_json(path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    try:
        schema_version = int(payload.get("schema_version") or 0)
        generated_at_ms = int(payload.get("generated_at_ms") or 0)
    except Exception:
        return False

    report_path_raw = str(payload.get("report_path") or "").strip()
    if not report_path_raw:
        return False

    report_path = Path(report_path_raw).expanduser()
    if not report_path.is_absolute():
        report_path = (path.parent / report_path).resolve()

    return (
        payload.get("blocked") is True and schema_version >= 2 and generated_at_ms >= start_ms and report_path.exists()
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = Path(__file__).with_name("run_scheduled_replay_alignment_gate.py")
    cmd = [sys.executable, "-u", str(target), *(argv or sys.argv[1:])]
    start_ms = _now_ms()
    proc = subprocess.run(cmd, cwd=repo_root)
    rc = int(proc.returncode)

    if rc == 1 and _is_fresh_blocker_update(_resolve_blocker_path(cwd=repo_root), start_ms=start_ms):
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
