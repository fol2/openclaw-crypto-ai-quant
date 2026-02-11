#!/usr/bin/env python3
"""Generate a 17-phase sweep set from full_144v coverage.

Design goals:
- Keep full axis coverage from `full_144v.yaml` (every axis appears once).
- Keep total grid size near the legacy 17-phase scale (~18k combos per interval).
- Preserve broader resolution (3-point min/mid/max) on core axes from `full_34axis`
  plus selected high-impact risk/exit/regime dimensions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Axis:
    path: str
    values: list[float]


def _load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"invalid YAML root: {path}")
    return obj


def _uniq(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for raw in values:
        v = float(raw)
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def _reduce_values(path: str, values: list[float], *, keep_three: set[str]) -> list[float]:
    vals = _uniq([float(v) for v in values])
    if not vals:
        raise ValueError(f"axis has empty values: {path}")
    if len(vals) <= 2:
        return vals
    if path in keep_three:
        mid = vals[len(vals) // 2]
        return _uniq([vals[0], mid, vals[-1]])
    return _uniq([vals[0], vals[-1]])


def _combo_count(axes: list[Axis]) -> int:
    n = 1
    for axis in axes:
        n *= len(axis.values)
    return int(n)


def _phase_id(i: int) -> str:
    return f"p{i:02d}_144v"


def _dump_yaml(path: Path, obj: dict[str, Any]) -> None:
    text = yaml.safe_dump(obj, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def build(
    *,
    full_144v_path: Path,
    full_34_path: Path,
    out_dir: Path,
    source_spec_ref: str,
    reference_spec_ref: str,
    phase_count: int = 17,
    max_axes_per_phase: int = 9,
) -> dict[str, Any]:
    src_144 = _load_yaml(full_144v_path)
    src_34 = _load_yaml(full_34_path)

    axes_144 = src_144.get("axes", [])
    axes_34 = src_34.get("axes", [])
    if not isinstance(axes_144, list) or not isinstance(axes_34, list):
        raise ValueError("sweep specs must define `axes` as a list")

    baseline_paths = {
        str(item.get("path", "")).strip()
        for item in axes_34
        if isinstance(item, dict) and str(item.get("path", "")).strip()
    }

    extra_keep_three = {
        "trade.smart_exit_adx_exhaustion_lt",
        "trade.smart_exit_adx_exhaustion_lt_low_conf",
        "trade.rsi_exit_profit_atr_switch",
        "trade.rsi_exit_ub_hi_profit",
        "trade.rsi_exit_lb_hi_profit",
        "thresholds.tp_and_momentum.tp_mult_strong",
        "thresholds.tp_and_momentum.tp_mult_weak",
        "market_regime.breadth_block_short_above",
        "market_regime.breadth_block_long_below",
        "market_regime.auto_reverse_breadth_low",
        "market_regime.auto_reverse_breadth_high",
        "indicators.ema_macro_window",
    }

    enum_three = {
        "trade.entry_min_confidence",
        "trade.add_min_confidence",
        "thresholds.entry.pullback_confidence",
        "thresholds.entry.macd_hist_entry_mode",
    }

    keep_three = baseline_paths | extra_keep_three | enum_three

    reduced: list[Axis] = []
    seen_paths: set[str] = set()
    for item in axes_144:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        raw_values = item.get("values", [])
        if not path or not isinstance(raw_values, list):
            continue
        if path in seen_paths:
            raise ValueError(f"duplicate axis path in full_144v: {path}")
        seen_paths.add(path)
        values = _reduce_values(path, [float(v) for v in raw_values], keep_three=keep_three)
        reduced.append(Axis(path=path, values=values))

    if len(reduced) == 0:
        raise ValueError("no axes found in full_144v")

    phases: list[list[Axis]] = [[] for _ in range(phase_count)]
    phase_weights = [0.0 for _ in range(phase_count)]

    items = sorted(reduced, key=lambda a: (-len(a.values), a.path))
    for axis in items:
        # Prefer the least-loaded phase while respecting max axes per phase.
        choices = [i for i, arr in enumerate(phases) if len(arr) < max_axes_per_phase]
        if not choices:
            # Safety fallback: if constraints are impossible, place in absolute least-loaded phase.
            choices = list(range(phase_count))
        idx = min(choices, key=lambda i: (phase_weights[i], len(phases[i])))
        phases[idx].append(axis)
        phase_weights[idx] += math.log(len(axis.values))

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_phases: list[dict[str, Any]] = []
    total_combo = 0
    all_paths_written: set[str] = set()

    for i, arr in enumerate(phases, start=1):
        arr_sorted = sorted(arr, key=lambda a: a.path)
        spec = {
            "initial_balance": float(src_144.get("initial_balance", 10000.0)),
            "lookback": int(src_144.get("lookback", 200)),
            "axes": [{"path": a.path, "values": [float(v) for v in a.values]} for a in arr_sorted],
        }
        phase_file = out_dir / f"{_phase_id(i)}.yaml"
        header = (
            f"# {_phase_id(i)}\n"
            "# Auto-generated from full_144v coverage.\n"
            "# Values are reduced for phase-grid practicality.\n"
        )
        phase_file.write_text(header + yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        combo = _combo_count(arr_sorted)
        total_combo += combo
        for axis in arr_sorted:
            all_paths_written.add(axis.path)

        manifest_phases.append(
            {
                "id": _phase_id(i),
                "file": str(phase_file.name),
                "axis_count": len(arr_sorted),
                "combo": combo,
                "two_value_axes": sum(1 for a in arr_sorted if len(a.values) == 2),
                "three_value_axes": sum(1 for a in arr_sorted if len(a.values) == 3),
                "paths": [a.path for a in arr_sorted],
            }
        )

    all_paths_src = {a.path for a in reduced}
    if all_paths_written != all_paths_src:
        miss = sorted(all_paths_src - all_paths_written)
        extra = sorted(all_paths_written - all_paths_src)
        raise ValueError(f"path coverage mismatch: missing={len(miss)} extra={len(extra)}")

    manifest = {
        "source_spec": str(source_spec_ref),
        "reference_spec": str(reference_spec_ref),
        "phase_count": int(phase_count),
        "total_axes": len(reduced),
        "total_combo": int(total_combo),
        "notes": [
            "All full_144v axes are included once across the 17 phases.",
            "Most axes use 2-point min/max values; selected key axes use 3-point min/mid/max.",
            "This keeps runtime near the legacy 17-phase scale while preserving 144v coverage.",
        ],
        "phases": manifest_phases,
    }
    _dump_yaml(out_dir / "manifest.yaml", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate 17-phase 144v sweep specs.")
    ap.add_argument(
        "--full-144v",
        default="backtester/sweeps/full_144v.yaml",
        help="Path to the full 144v sweep spec.",
    )
    ap.add_argument(
        "--full-34",
        default="backtester/sweeps/full_34axis.yaml",
        help="Path to the baseline 34-axis sweep spec.",
    )
    ap.add_argument(
        "--out-dir",
        default="backtester/sweeps/full_144v_17phase",
        help="Output directory for generated phase specs.",
    )
    ap.add_argument("--phase-count", type=int, default=17, help="Number of phases (default: 17).")
    ap.add_argument("--max-axes-per-phase", type=int, default=9, help="Max axes per phase.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    full_144v = (root / str(args.full_144v)).resolve()
    full_34 = (root / str(args.full_34)).resolve()
    out_dir = (root / str(args.out_dir)).resolve()

    manifest = build(
        full_144v_path=full_144v,
        full_34_path=full_34,
        out_dir=out_dir,
        source_spec_ref=str(args.full_144v),
        reference_spec_ref=str(args.full_34),
        phase_count=int(args.phase_count),
        max_axes_per_phase=int(args.max_axes_per_phase),
    )

    print(
        f"Generated {manifest['phase_count']} phases, "
        f"{manifest['total_axes']} axes, total_combo={manifest['total_combo']}"
    )
    for ph in manifest["phases"]:
        print(
            f"{ph['id']}: axes={ph['axis_count']} combo={ph['combo']} "
            f"(2v={ph['two_value_axes']}, 3v={ph['three_value_axes']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
