#!/usr/bin/env python3
"""Compare CPU vs GPU sweep JSONL outputs for parity.

This tool matches rows by their sweep overrides (CPU uses a JSON object, GPU uses
a list of key/value pairs) and compares key result metrics.

Usage:
    python tools/compare_sweep_outputs.py --cpu cpu.jsonl --gpu gpu.jsonl

Exit codes:
    0: All matched rows are within tolerances (and no missing rows, unless ignored)
    1: At least one row exceeds tolerances, or row sets differ
    2: Invalid inputs (parse errors, missing fields, etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS: tuple[str, ...] = (
    "final_balance",
    "total_pnl",
    "total_trades",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
)


def _parse_metric_kv(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Expected NAME=VALUE format")
    name, raw = text.split("=", 1)
    name = name.strip()
    raw = raw.strip()
    if not name:
        raise argparse.ArgumentTypeError("Metric name cannot be empty")
    try:
        val = float(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float value: {raw}") from e
    return name, val


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object per line")
            rows.append(obj)
    return rows


def _normalise_overrides(raw_overrides: Any) -> list[tuple[str, Any]]:
    """Accept both list-of-pairs and dict formats."""
    if isinstance(raw_overrides, dict):
        return [(str(k), v) for k, v in raw_overrides.items()]
    if isinstance(raw_overrides, list):
        out: list[tuple[str, Any]] = []
        for pair in raw_overrides:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError("Invalid overrides list item (expected [key, value])")
            k, v = pair
            out.append((str(k), v))
        return out
    raise ValueError("Invalid overrides type (expected object or list-of-pairs)")


def _canon_value_for_key(v: Any, round_dp: int | None) -> Any:
    """Canonicalise override values to improve CPU/GPU matching reliability."""
    if v is None:
        return None

    # Most sweep axes are numeric; normalise bool/int/float to float to avoid JSON
    # number formatting differences turning into type mismatches.
    if isinstance(v, bool):
        fv = float(int(v))
    elif isinstance(v, (int, float)):
        fv = float(v)
    else:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)

    if not math.isfinite(fv):
        return str(fv)

    if round_dp is not None:
        fv = round(fv, round_dp)
    if fv == 0.0:
        fv = 0.0  # normalise -0.0
    return fv


def _canon_overrides_key(raw_overrides: Any, round_dp: int | None) -> tuple[tuple[str, Any], ...]:
    items = _normalise_overrides(raw_overrides)
    norm: list[tuple[str, Any]] = [(k, _canon_value_for_key(v, round_dp)) for k, v in items]
    norm.sort(key=lambda kv: kv[0])
    return tuple(norm)


def _format_overrides(key: tuple[tuple[str, Any], ...]) -> str:
    parts: list[str] = []
    for k, v in key:
        if isinstance(v, float):
            parts.append(f"{k}={v:.12g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _as_float(row: dict[str, Any], metric: str, src: str, overrides_str: str) -> float:
    if metric not in row:
        raise ValueError(f"{src}: missing metric '{metric}' for overrides: {overrides_str}")
    v = row[metric]
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{src}: non-numeric metric '{metric}'={v!r} for overrides: {overrides_str}") from e


def _within_tol(cpu: float, gpu: float, abs_tol: float, rel_tol: float) -> tuple[bool, float, float]:
    diff = abs(cpu - gpu)
    scale = max(abs(cpu), abs(gpu))
    allowed = max(abs_tol, rel_tol * scale)
    return diff <= allowed, diff, allowed


def _iter_sorted(keys: Iterable[tuple[tuple[str, Any], ...]]):
    # Sort by string form to keep output deterministic.
    return sorted(keys, key=_format_overrides)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CPU vs GPU sweep JSONL outputs for parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cpu", required=True, type=Path, help="Path to CPU sweep JSONL output")
    parser.add_argument("--gpu", required=True, type=Path, help="Path to GPU sweep JSONL output")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated metrics to compare (default: {','.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Default absolute tolerance (overridden by --abs-tol-metric)",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.0,
        help="Default relative tolerance (overridden by --rel-tol-metric)",
    )
    parser.add_argument(
        "--abs-tol-metric",
        action="append",
        default=[],
        type=_parse_metric_kv,
        help="Per-metric absolute tolerance, e.g. --abs-tol-metric total_pnl=0.01",
    )
    parser.add_argument(
        "--rel-tol-metric",
        action="append",
        default=[],
        type=_parse_metric_kv,
        help="Per-metric relative tolerance, e.g. --rel-tol-metric profit_factor=0.001",
    )
    parser.add_argument(
        "--override-round-dp",
        type=int,
        default=12,
        help="Round override values to this many decimal places when matching rows (default: 12)",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Do not fail if CPU/GPU row sets differ; compare only the intersection",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all matched rows (not only those exceeding tolerances)",
    )
    parser.add_argument(
        "--max-printed",
        type=int,
        default=50,
        help="Maximum number of differing rows to print (default: 50; ignored with --show-all)",
    )

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown_metrics = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown_metrics:
        print(f"Error: unknown metric(s): {', '.join(unknown_metrics)}", file=sys.stderr)
        print(f"Supported metrics: {', '.join(DEFAULT_METRICS)}", file=sys.stderr)
        return 2

    abs_tols: dict[str, float] = {m: args.abs_tol for m in metrics}
    rel_tols: dict[str, float] = {m: args.rel_tol for m in metrics}

    # Default: trade counts must match exactly unless overridden.
    if "total_trades" in abs_tols and abs_tols["total_trades"] == args.abs_tol:
        abs_tols["total_trades"] = 0.0
        rel_tols["total_trades"] = 0.0

    for name, val in args.abs_tol_metric:
        if name not in abs_tols:
            print(f"Error: --abs-tol-metric uses unknown metric: {name}", file=sys.stderr)
            return 2
        abs_tols[name] = val
    for name, val in args.rel_tol_metric:
        if name not in rel_tols:
            print(f"Error: --rel-tol-metric uses unknown metric: {name}", file=sys.stderr)
            return 2
        rel_tols[name] = val

    try:
        cpu_rows = _read_jsonl(args.cpu)
        gpu_rows = _read_jsonl(args.gpu)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    def index_rows(rows: list[dict[str, Any]], src: str) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
        idx: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
        for i, row in enumerate(rows, start=1):
            if "overrides" not in row:
                raise ValueError(f"{src}: row {i}: missing 'overrides'")
            key = _canon_overrides_key(row["overrides"], args.override_round_dp)
            if key in idx:
                raise ValueError(f"{src}: duplicate overrides row: {_format_overrides(key)}")
            idx[key] = row
        return idx

    try:
        cpu_idx = index_rows(cpu_rows, "CPU")
        gpu_idx = index_rows(gpu_rows, "GPU")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    cpu_keys = set(cpu_idx.keys())
    gpu_keys = set(gpu_idx.keys())

    missing_in_gpu = cpu_keys - gpu_keys
    missing_in_cpu = gpu_keys - cpu_keys
    matched = cpu_keys & gpu_keys

    failures = 0
    printed = 0

    if missing_in_gpu:
        failures += len(missing_in_gpu) if not args.ignore_missing else 0
        print(f"Missing in GPU: {len(missing_in_gpu)} row(s)", file=sys.stderr)
        for key in _iter_sorted(missing_in_gpu):
            print(f"  {_format_overrides(key)}", file=sys.stderr)
        print(file=sys.stderr)

    if missing_in_cpu:
        failures += len(missing_in_cpu) if not args.ignore_missing else 0
        print(f"Missing in CPU: {len(missing_in_cpu)} row(s)", file=sys.stderr)
        for key in _iter_sorted(missing_in_cpu):
            print(f"  {_format_overrides(key)}", file=sys.stderr)
        print(file=sys.stderr)

    max_abs_diff: dict[str, float] = {m: 0.0 for m in metrics}

    for key in _iter_sorted(matched):
        cpu_row = cpu_idx[key]
        gpu_row = gpu_idx[key]
        overrides_str = _format_overrides(key)

        row_failed = False
        metric_lines: list[str] = []
        for m in metrics:
            try:
                cpu_v = _as_float(cpu_row, m, "CPU", overrides_str)
                gpu_v = _as_float(gpu_row, m, "GPU", overrides_str)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2

            ok, diff, allowed = _within_tol(cpu_v, gpu_v, abs_tols[m], rel_tols[m])
            max_abs_diff[m] = max(max_abs_diff[m], diff)
            if not ok:
                row_failed = True
                failures += 1

            metric_lines.append(f"  {m}: cpu={cpu_v:.12g} gpu={gpu_v:.12g} diff={diff:.12g} (allowed={allowed:.12g})")

        if args.show_all or row_failed:
            if not args.show_all and printed >= args.max_printed:
                continue
            printed += 1
            header = "DIFF" if row_failed else "OK"
            print(f"{header}: {overrides_str}")
            for line in metric_lines:
                print(line)
            print()

    print("Summary:", file=sys.stderr)
    print(f"  CPU rows: {len(cpu_rows)}", file=sys.stderr)
    print(f"  GPU rows: {len(gpu_rows)}", file=sys.stderr)
    print(f"  Matched rows: {len(matched)}", file=sys.stderr)
    print(f"  Failures: {failures}", file=sys.stderr)
    print("  Max abs diffs:", file=sys.stderr)
    for m in metrics:
        print(f"    {m}: {max_abs_diff[m]:.12g}", file=sys.stderr)

    if failures > 0 and not args.show_all and printed >= args.max_printed:
        print(
            f"\nNote: output truncated to --max-printed {args.max_printed} row(s).",
            file=sys.stderr,
        )

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
