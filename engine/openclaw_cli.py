from __future__ import annotations

import os
import shutil
import subprocess
import time
from functools import lru_cache


@lru_cache(maxsize=1)
def resolve_openclaw_bin() -> str:
    """Return an executable path for the `openclaw` CLI.

    Priority:
    1) `AI_QUANT_OPENCLAW_BIN` (explicit override)
    2) `PATH` lookup (`shutil.which`)
    3) common user-level install locations
    """
    candidates: list[str] = []

    override = str(os.getenv("AI_QUANT_OPENCLAW_BIN", "")).strip()
    if override:
        candidates.append(override)

    which_bin = shutil.which("openclaw")
    if which_bin:
        candidates.append(which_bin)

    home = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home, ".local", "bin", "openclaw"),
            "/home/linuxbrew/.linuxbrew/bin/openclaw",
            "/usr/local/bin/openclaw",
            "/usr/bin/openclaw",
        ]
    )

    seen: set[str] = set()
    for raw in candidates:
        path = os.path.expanduser(str(raw or "").strip())
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Final fallback: keep previous behaviour; subprocess will raise if unavailable.
    return "openclaw"


@lru_cache(maxsize=1)
def openclaw_exec_env() -> dict[str, str]:
    """Return process env with PATH entries needed by OpenClaw CLI wrappers."""
    env = dict(os.environ)
    path_raw = str(env.get("PATH", "") or "")

    head = [
        os.path.expanduser("~/.local/bin"),
        "/home/linuxbrew/.linuxbrew/bin",
    ]
    tail = [p for p in path_raw.split(":") if p]

    merged: list[str] = []
    seen: set[str] = set()
    for p in head + tail:
        p2 = str(p or "").strip()
        if not p2 or p2 in seen:
            continue
        seen.add(p2)
        merged.append(p2)

    env["PATH"] = ":".join(merged)
    return env


def send_openclaw_message(
    *,
    channel: str,
    target: str,
    message: str,
    timeout_s: float = 6.0,
) -> subprocess.CompletedProcess[str]:
    """Send a message via OpenClaw CLI with small retry/backoff for transient failures."""
    cmd = [
        resolve_openclaw_bin(),
        "message",
        "send",
        "--channel",
        str(channel),
        "--target",
        str(target),
        "--message",
        str(message),
    ]

    try:
        retry_count = int(float(os.getenv("AI_QUANT_DISCORD_SEND_RETRIES", "2") or 2))
    except Exception:
        retry_count = 2
    retry_count = max(0, min(5, retry_count))
    attempts = retry_count + 1

    try:
        timeout_s = float(timeout_s)
    except Exception:
        timeout_s = 6.0
    timeout_s = max(1.0, min(30.0, timeout_s))

    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True,
                timeout=timeout_s,
                env=openclaw_exec_env(),
            )
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            stderr_l = str(exc.stderr or "").lower()
            # Retry only for likely-transient cases.
            transient = exc.returncode in {9, 137, 143} or "rate limit" in stderr_l or "429" in stderr_l
            if not transient:
                raise
        except Exception as exc:
            last_exc = exc
            raise

        if i < attempts - 1:
            time.sleep(min(2.0, 0.35 * (i + 1)))

    assert last_exc is not None
    raise last_exc
