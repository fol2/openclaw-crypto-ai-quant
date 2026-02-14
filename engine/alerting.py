from __future__ import annotations

import os
import queue
import subprocess
import threading
from typing import Any

from .openclaw_cli import send_openclaw_message


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else str(raw)


def _instance_label() -> str:
    label = _env_str("AI_QUANT_DISCORD_LABEL", "").strip()
    if label:
        return label
    return _env_str("AI_QUANT_INSTANCE_TAG", "").strip()


def _decorate_message(message: str) -> str:
    msg = str(message)
    label = _instance_label()
    if not label:
        return msg
    prefix = f"[{label}]"
    if msg.startswith(prefix):
        return msg
    return f"{prefix} {msg}"


def _redact_target(target: str) -> str:
    """Best-effort redaction for alert targets.

    Some channels may use webhook URLs or other secrets as "targets". Never print full targets
    to stdout/stderr.
    """
    t = str(target or "").strip()
    if not t:
        return ""
    # Short, human-readable targets are likely safe (#channel, @chat, etc).
    if len(t) <= 32 and ("://" not in t) and ("/" not in t):
        return t
    if len(t) <= 12:
        return "…"
    return t[:6] + "…" + t[-4:]


def parse_targets(raw: str) -> list[tuple[str, str]]:
    """Parse AI_QUANT_ALERT_TARGETS into [(channel, target), ...].

    Format: comma-separated `channel:target` items.
    Example:
      discord:#alerts,telegram:@my_channel
    """
    out: list[tuple[str, str]] = []
    for part in str(raw or "").split(","):
        s = str(part or "").strip()
        if not s:
            continue
        if ":" not in s:
            continue
        ch, tgt = s.split(":", 1)
        ch2 = str(ch or "").strip().lower()
        tgt2 = str(tgt or "").strip()
        if not ch2 or not tgt2:
            continue
        out.append((ch2, tgt2))
    return out


def targets() -> list[tuple[str, str]]:
    return parse_targets(_env_str("AI_QUANT_ALERT_TARGETS", "").strip())


def enabled() -> bool:
    raw_enabled = os.getenv("AI_QUANT_ALERT_ENABLED")
    if raw_enabled is not None and str(raw_enabled).strip() != "":
        return _env_bool("AI_QUANT_ALERT_ENABLED", False) and bool(targets())
    return bool(targets())


def _send_one_sync(*, channel: str, target: str, message: str) -> None:
    msg = _decorate_message(str(message or "").strip())
    if not msg:
        return
    if _env_bool("AI_QUANT_ALERT_DRY_RUN", False):
        try:
            print(f"🟡 ALERT DRY RUN channel={channel} target={_redact_target(target)} message={msg}")
        except Exception:
            pass
        return

    try:
        timeout_s = float(max(1.0, min(30.0, _env_float("AI_QUANT_ALERT_SEND_TIMEOUT_S", 6.0))))
    except Exception:
        timeout_s = 6.0

    send_openclaw_message(channel=str(channel), target=str(target), message=msg, timeout_s=timeout_s)


def send_openclaw_message(*, channel: str, target: str, message: str, timeout_s: float = 6.0) -> None:
    """Send a single OpenClaw message synchronously.

    This keeps compatibility with older call sites that previously imported
    `send_openclaw_message` from a removed module.
    """
    msg = str(message or "").strip()
    if not msg:
        return
    ch = str(channel or "").strip()
    tgt = str(target or "").strip()
    if not ch or not tgt:
        return

    try:
        timeout = float(timeout_s)
    except Exception:
        timeout = 6.0
    timeout = max(1.0, min(30.0, timeout))

    subprocess.run(
        [
            "openclaw",
            "message",
            "send",
            "--channel",
            ch,
            "--target",
            tgt,
            "--message",
            msg,
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=timeout,
    )


_ALERT_QUEUE_LOCK = threading.RLock()
_ALERT_QUEUE_MAX = max(10, min(5000, _env_int("AI_QUANT_ALERT_QUEUE_MAX", 200)))
_ALERT_QUEUE: queue.Queue[tuple[str, str, str]] = queue.Queue(maxsize=_ALERT_QUEUE_MAX)
_ALERT_WORKER_STARTED = False


def _alert_worker() -> None:
    while True:
        channel, target, message = _ALERT_QUEUE.get()
        try:
            _send_one_sync(channel=channel, target=target, message=message)
        except Exception as e:
            try:
                print(f"⚠️ Failed to send alert (channel={channel} target={_redact_target(target)}): {e}")
            except Exception:
                pass
        finally:
            try:
                _ALERT_QUEUE.task_done()
            except Exception:
                pass


def _ensure_worker_started() -> None:
    global _ALERT_WORKER_STARTED
    with _ALERT_QUEUE_LOCK:
        if _ALERT_WORKER_STARTED:
            return
        t = threading.Thread(target=_alert_worker, name="aiq_alert_sender", daemon=True)
        t.start()
        _ALERT_WORKER_STARTED = True


def send_alert(message: str, *, extra: dict[str, Any] | None = None) -> None:
    """Best-effort alert send that must NOT stall the trading loop."""
    if not enabled():
        return

    msg = str(message or "").strip()
    if not msg:
        return

    # Optional structured extras for future formatting; currently appended as a compact one-liner.
    try:
        if isinstance(extra, dict) and extra:
            parts = [f"{k}={v}" for k, v in extra.items() if str(k).strip()]
            if parts:
                msg = msg + "\n" + " ".join(parts)
    except Exception:
        pass

    tgs = targets()
    if not tgs:
        return

    try:
        async_enabled = _env_bool("AI_QUANT_ALERT_ASYNC", True)
    except Exception:
        async_enabled = True

    if not async_enabled:
        for ch, tgt in tgs:
            try:
                _send_one_sync(channel=ch, target=tgt, message=msg)
            except Exception:
                continue
        return

    _ensure_worker_started()
    for ch, tgt in tgs:
        try:
            _ALERT_QUEUE.put_nowait((str(ch), str(tgt), msg))
        except Exception:
            # Drop when overloaded; correctness > notifications.
            try:
                print("⚠️ Alert queue full; dropping message")
            except Exception:
                pass
