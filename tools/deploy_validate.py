#!/usr/bin/env python3
"""Deployment-time YAML validation (AQC-706).

This validator is designed to fail fast before writing a config to a running engine.

It checks:
- YAML parses and has the expected `global` mapping shape.
- A minimal set of required fields exist with sane types/ranges.

It does not attempt to validate strategy performance, only configuration structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _get_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_config_obj(obj: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["YAML root must be a mapping"]

    glob = obj.get("global")
    if not isinstance(glob, dict):
        return ["Missing required mapping: global"]

    def req_map(path: str) -> dict[str, Any] | None:
        v = _get_path(obj, path)
        if not isinstance(v, dict):
            errs.append(f"Missing required mapping: {path}")
            return None
        return v

    def req_number(path: str, *, min_v: float | None = None, max_v: float | None = None) -> None:
        v = _get_path(obj, path)
        if not _is_number(v):
            errs.append(f"Missing/invalid number: {path}")
            return
        fv = float(v)
        if min_v is not None and fv < float(min_v):
            errs.append(f"Out of range (min {min_v}): {path}={fv}")
        if max_v is not None and fv > float(max_v):
            errs.append(f"Out of range (max {max_v}): {path}={fv}")

    def req_int(path: str, *, min_v: int | None = None, max_v: int | None = None) -> None:
        v = _get_path(obj, path)
        if isinstance(v, bool) or not isinstance(v, int):
            # Do not accept floats here: keep windows/limits unambiguous.
            errs.append(f"Missing/invalid int: {path}")
            return
        iv = int(v)
        if min_v is not None and iv < int(min_v):
            errs.append(f"Out of range (min {min_v}): {path}={iv}")
        if max_v is not None and iv > int(max_v):
            errs.append(f"Out of range (max {max_v}): {path}={iv}")

    def req_bool(path: str) -> None:
        v = _get_path(obj, path)
        if not isinstance(v, bool):
            errs.append(f"Missing/invalid bool: {path}")

    # Required mappings
    req_map("global.trade")
    req_map("global.indicators")
    req_map("global.thresholds.entry")

    # Core trade sizing/execution fields
    req_number("global.trade.allocation_pct", min_v=0.0, max_v=1.0)
    req_number("global.trade.leverage", min_v=0.0)
    req_number("global.trade.sl_atr_mult", min_v=0.0)
    req_number("global.trade.tp_atr_mult", min_v=0.0)
    req_number("global.trade.slippage_bps", min_v=0.0)
    req_int("global.trade.max_open_positions", min_v=1)
    req_number("global.trade.max_total_margin_pct", min_v=0.0, max_v=1.0)
    req_number("global.trade.min_notional_usd", min_v=0.0)
    req_number("global.trade.min_atr_pct", min_v=0.0, max_v=1.0)
    req_bool("global.trade.bump_to_min_notional")

    # Key indicator windows
    req_int("global.indicators.adx_window", min_v=1)
    req_int("global.indicators.ema_fast_window", min_v=1)
    req_int("global.indicators.ema_slow_window", min_v=1)
    req_int("global.indicators.bb_window", min_v=1)
    req_int("global.indicators.atr_window", min_v=1)

    # Entry threshold sanity
    req_number("global.thresholds.entry.min_adx", min_v=0.0)

    # Invariants
    fast = _get_path(obj, "global.indicators.ema_fast_window")
    slow = _get_path(obj, "global.indicators.ema_slow_window")
    if isinstance(fast, int) and isinstance(slow, int) and fast >= slow:
        errs.append("Invalid invariant: global.indicators.ema_fast_window must be < ema_slow_window")

    return errs


def validate_yaml_text(text: str) -> list[str]:
    try:
        obj = yaml.safe_load(text) or {}
    except Exception as e:
        return [f"YAML parse error: {type(e).__name__}: {e}"]
    return validate_config_obj(obj)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a strategy overrides YAML before deploying.")
    ap.add_argument("--config", required=True, help="Path to a YAML config file to validate.")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    text = cfg_path.read_text(encoding="utf-8")
    errs = validate_yaml_text(text)
    if errs:
        for e in errs:
            print(f"[invalid] {e}")
        return 1
    print("[ok] config validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

