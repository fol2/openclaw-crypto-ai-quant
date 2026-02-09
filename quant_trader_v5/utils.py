from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges `override` into `base`.

    Rules:
    - dict + dict: merge recursively
    - everything else: override replaces base

    Notes:
    - This mutates `base` and returns it.
    - Lists are replaced (not concatenated).
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return base
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None


def sha1_json(obj: Any) -> str:
    """Deterministic hash for nested dict/list primitives."""
    try:
        b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except Exception:
        b = repr(obj).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return ""


@dataclass(frozen=True)
class Backoff:
    base_s: float = 1.0
    max_s: float = 30.0
    jitter_pct: float = 0.25

    def delay(self, attempt: int) -> float:
        """Exponential backoff with jitter.

        attempt is 1-indexed.
        """
        a = max(1, int(attempt))
        d = min(self.max_s, self.base_s * (2 ** (a - 1)))
        j = max(0.0, float(self.jitter_pct))
        lo = d * (1.0 - j)
        hi = d * (1.0 + j)
        return random.uniform(lo, hi)


def now_ms() -> int:
    return int(time.time() * 1000)
