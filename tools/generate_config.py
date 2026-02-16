#!/usr/bin/env python3
"""Generate a standalone YAML config from base config + sweep overrides.

Reads a sweep JSONL results file, picks a result by rank/criteria,
applies its overrides to a base YAML config, and writes a complete
standalone config suitable for CPU replay or deployment.

Usage:
    # Best PnL (default)
    python generate_config.py --sweep-results sweep.jsonl --base-config base.yaml -o out.yaml

    # Best drawdown (lowest DD)
    python generate_config.py --sweep-results sweep.jsonl --base-config base.yaml --sort-by dd -o out.yaml

    # Top 10 comparison table
    python generate_config.py --sweep-results sweep.jsonl --show-top 10

    # Pick by rank (e.g., 3rd best PnL)
    python generate_config.py --sweep-results sweep.jsonl --base-config base.yaml --rank 3 -o out.yaml

    # Balanced composite score
    python generate_config.py --sweep-results sweep.jsonl --base-config base.yaml --sort-by balanced -o out.yaml
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import yaml

try:
    # When invoked via `python3 factory_run.py`, repo root is on sys.path.
    from tools.config_id import config_id_from_obj
except ImportError:  # pragma: no cover
    # When invoked via `python3 tools/generate_config.py`, sys.path[0] is `tools/`.
    from config_id import config_id_from_obj  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Type maps for correct YAML serialisation
# ---------------------------------------------------------------------------

# Fields that must be integers
INT_FIELDS = {
    "max_open_positions", "max_adds_per_symbol", "add_cooldown_minutes",
    "reentry_cooldown_minutes", "reentry_cooldown_min_mins", "reentry_cooldown_max_mins",
    "max_entry_orders_per_loop", "entry_cooldown_s", "exit_cooldown_s",
    "adx_window", "ema_fast_window", "ema_slow_window",
    "bb_window", "ema_macro_window", "atr_window", "rsi_window",
    "bb_width_avg_window", "vol_sma_window", "vol_trend_window",
    "stoch_rsi_window", "stoch_rsi_smooth1", "stoch_rsi_smooth2",
    "slow_drift_slope_window", "min_signals", "ave_avg_atr_window",
    "loop_target_s",
}

# Fields that must be booleans
BOOL_FIELDS = {
    "enable_dynamic_leverage", "enable_dynamic_sizing", "enable_pyramiding",
    "enable_partial_tp", "enable_vol_buffered_trailing", "enable_reef_filter",
    "enable_ssf_filter", "enable_breakeven_stop", "enable_rsi_overextension_exit",
    "reverse_entry_signal", "block_exits_on_extreme_dev", "bump_to_min_notional",
    "use_bbo_for_fills", "tsme_require_adx_slope_negative",
    "enable_extension_filter", "enable_ranging_filter", "enable_anomaly_filter",
    "require_btc_alignment", "require_volume_confirmation", "require_macro_alignment",
    "require_adx_rising", "vol_confirm_include_prev", "use_stoch_rsi_filter",
    "enable_regime_filter", "enable_auto_reverse",
    "ave_enabled", "enable_pullback_entries", "pullback_require_macd_sign",
    "enable_slow_drift_entries", "slow_drift_require_macd_sign",
    "signal_on_candle_close",
}

# Fields that are confidence enums (0=low, 1=medium, 2=high)
CONFIDENCE_FIELDS = {
    "entry_min_confidence", "add_min_confidence", "pullback_confidence",
}
CONFIDENCE_MAP = {0: "low", 1: "medium", 2: "high"}

# Fields that are categorical enums encoded as continuous floats by the sweep optimizer
CATEGORICAL_FIELDS = {
    "macd_hist_entry_mode": {0: "accel", 1: "sign", 2: "none"},
}

# Fields that are string enums (keep as-is)
STRING_FIELDS = {
    "interval", "entry_interval", "exit_interval",
}


def _round_half_away_from_zero(x: float) -> int:
    """Round half values away from zero.

    Python's built-in round() uses bankers rounding (e.g. round(0.5) == 0).
    For config coercion we prefer the more intuitive behaviour:
    - 0.5 -> 1
    - -0.5 -> -1
    """
    xf = float(x)
    if xf >= 0:
        return int(math.floor(xf + 0.5))
    return int(math.ceil(xf - 0.5))


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(min(int(hi), max(int(lo), int(v))))


def coerce_value(param_name: str, raw_value):
    """Convert a raw sweep value to the correct Python type for YAML output."""
    if param_name in CATEGORICAL_FIELDS:
        mapping = CATEGORICAL_FIELDS[param_name]
        if isinstance(raw_value, str) and raw_value in mapping.values():
            return raw_value  # already a valid string
        idx = max(0, min(len(mapping) - 1, round(float(raw_value))))
        return mapping.get(idx, list(mapping.values())[0])

    if param_name in STRING_FIELDS:
        return str(raw_value)

    if param_name in CONFIDENCE_FIELDS:
        if isinstance(raw_value, str):
            s = str(raw_value).strip().lower()
            if s in {"low", "medium", "high"}:
                return s
            return raw_value
        try:
            idx = _round_half_away_from_zero(float(raw_value))
        except Exception:
            idx = 0
        idx = _clamp_int(idx, 0, 2)
        return CONFIDENCE_MAP.get(idx, "low")

    if param_name in BOOL_FIELDS:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            s = str(raw_value).strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        try:
            return float(raw_value) >= 0.5
        except Exception:
            return bool(raw_value)

    if param_name in INT_FIELDS:
        try:
            return _round_half_away_from_zero(float(raw_value))
        except Exception:
            return int(raw_value)

    # Default: float (round to 6 significant digits for readability)
    if isinstance(raw_value, (int, float)):
        val = float(raw_value)
        if val == 0:
            return 0.0
        if abs(val) >= 1:
            return round(val, 6)
        digits = 6 - int(math.floor(math.log10(abs(val)))) - 1
        return round(val, max(0, digits))
    return raw_value


# ---------------------------------------------------------------------------
# Load & sort sweep results
# ---------------------------------------------------------------------------

SORT_KEYS = {
    "pnl":      lambda r: r.get("total_pnl", 0),
    "dd":       lambda r: -r.get("max_drawdown_pct", 1),  # lower DD = better
    "pf":       lambda r: r.get("profit_factor", 0),
    "wr":       lambda r: r.get("win_rate", 0),
    "sharpe":   lambda r: r.get("sharpe_ratio", 0),
    "trades":   lambda r: r.get("total_trades", 0),
}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(x)


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def balanced_score(r: dict) -> float:
    """Composite score: normalised PnL + PF + Sharpe - DD penalty."""
    pnl = _safe_float(r.get("total_pnl", 0.0), 0.0)
    pf = min(_safe_float(r.get("profit_factor", 0.0), 0.0), 10.0)  # cap PF to avoid 8-trade outliers
    sharpe = _safe_float(r.get("sharpe_ratio", 0.0), 0.0)
    dd = _safe_float(r.get("max_drawdown_pct", 1.0), 1.0)
    trades = _safe_int(r.get("total_trades", 0), 0)
    trade_penalty = 0.5 if trades < 20 else 1.0
    return (pnl * 0.3 + pf * 20 + sharpe * 15 - dd * 100) * trade_penalty


def load_sweep_results(path: str, sort_by: str = "pnl",
                       min_trades: int = 0) -> list[dict]:
    """Load JSONL sweep results sorted by chosen metric."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    # Filter by minimum trades
    if min_trades > 0:
        results = [r for r in results if r.get("total_trades", 0) >= min_trades]

    if sort_by == "balanced":
        results.sort(key=balanced_score, reverse=True)
    else:
        key_fn = SORT_KEYS.get(sort_by, SORT_KEYS["pnl"])
        results.sort(key=key_fn, reverse=True)

    return results


# ---------------------------------------------------------------------------
# YAML operations
# ---------------------------------------------------------------------------

def _load_yaml(path: str):
    """Load YAML safely. Returns a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at root of YAML: {path}")
    return data


def _set_nested(data: dict, dotpath: str, value):
    """Set a value in a nested dict using dot notation (e.g., 'trade.sl_atr_mult')."""
    if not dotpath.startswith("global."):
        dotpath = "global." + dotpath
    keys = dotpath.split(".")
    d = data
    for k in keys[:-1]:
        if k not in d or d[k] is None:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _get_nested(data: dict, dotpath: str, default=None):
    """Get a value from nested dict using dot notation."""
    if not dotpath.startswith("global."):
        dotpath = "global." + dotpath
    keys = dotpath.split(".")
    d = data
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Normalise overrides from sweep JSONL
# ---------------------------------------------------------------------------

def normalise_overrides(raw_overrides) -> list[tuple[str, float]]:
    """Accept both list-of-pairs and dict formats."""
    if isinstance(raw_overrides, dict):
        return list(raw_overrides.items())
    if isinstance(raw_overrides, list):
        return [(k, v) for k, v in raw_overrides]
    return []


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_row(r: dict, rank: int) -> str:
    pnl = _safe_float(r.get("total_pnl", 0.0), 0.0)
    bal = _safe_float(r.get("final_balance", 0.0), 0.0)
    trades = _safe_int(r.get("total_trades", 0), 0)
    wr = _safe_float(r.get("win_rate", 0.0), 0.0) * 100
    pf = _safe_float(r.get("profit_factor", 0.0), 0.0)
    dd = _safe_float(r.get("max_drawdown_pct", 0.0), 0.0) * 100
    sharpe = _safe_float(r.get("sharpe_ratio", 0.0), 0.0)
    bscore = balanced_score(r)
    return (f"  #{rank:<3d}  PnL ${pnl:>9.2f}  Bal ${bal:>9.2f}  "
            f"Trades {trades:>4d}  WR {wr:>5.1f}%  PF {pf:>5.2f}  "
            f"DD {dd:>5.1f}%  Sharpe {sharpe:>5.2f}  Score {bscore:>6.1f}")


def show_top(results: list[dict], n: int, sort_label: str):
    print(f"\n--- Top {n} by {sort_label} ---\n", file=sys.stderr)
    for i, r in enumerate(results[:n]):
        print(format_row(r, i + 1), file=sys.stderr)
    print(file=sys.stderr)


def show_diff(base_data: dict, overrides: list[tuple[str, float]]):
    """Show what changes vs base config."""
    print("\n--- Config Overrides ---\n", file=sys.stderr)
    for path, raw_val in sorted(overrides):
        param_name = path.split(".")[-1]
        coerced = coerce_value(param_name, raw_val)
        old_val = _get_nested(base_data, path, "???")
        marker = " <--" if old_val != coerced else ""
        print(f"  {path}: {old_val} -> {coerced}{marker}", file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate standalone YAML config from base + sweep overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sweep-results", required=True,
                        help="Path to sweep results JSONL file")
    parser.add_argument("--base-config", default=None,
                        help="Base YAML config to apply overrides to")
    parser.add_argument("--sort-by", default="pnl",
                        choices=["pnl", "dd", "pf", "wr", "sharpe", "trades", "balanced"],
                        help="Sort metric for ranking (default: pnl)")
    parser.add_argument("--rank", type=int, default=1,
                        help="Pick Nth best result by sort metric (default: 1)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output YAML path (default: stdout)")
    parser.add_argument("--show-top", type=int, default=0, metavar="N",
                        help="Show top N results and exit (no config generation)")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Minimum trade count to include (default: 20)")
    parser.add_argument("--show-diff", action="store_true",
                        help="Show diff of overrides vs base config")
    args = parser.parse_args()

    # Load & sort results
    results = load_sweep_results(args.sweep_results, sort_by=args.sort_by,
                                 min_trades=args.min_trades)
    if not results:
        print("[generate] No results found in sweep file.", file=sys.stderr)
        sys.exit(1)

    print(f"[generate] Loaded {len(results)} results, sorted by {args.sort_by}",
          file=sys.stderr)

    # Show-top mode: display table and exit
    if args.show_top > 0:
        show_top(results, args.show_top, args.sort_by)
        return

    # Pick result
    idx = args.rank - 1
    if idx < 0 or idx >= len(results):
        print(f"[generate] Rank {args.rank} out of range (1..{len(results)}).",
              file=sys.stderr)
        sys.exit(1)

    selected = results[idx]
    raw_overrides = normalise_overrides(selected.get("overrides", []))
    if not raw_overrides:
        print("[generate] Selected result has no overrides.", file=sys.stderr)
        sys.exit(1)

    # Show selected result summary
    print(f"\n[generate] Selected #{args.rank} by {args.sort_by}:", file=sys.stderr)
    print(format_row(selected, args.rank), file=sys.stderr)

    # Need base config for generation
    if not args.base_config:
        print("[generate] Error: --base-config required to generate YAML.",
              file=sys.stderr)
        sys.exit(1)

    # Load base config
    base_data = _load_yaml(args.base_config)

    # Optionally show diff
    if args.show_diff:
        show_diff(base_data, raw_overrides)

    # Apply overrides with proper type coercion
    applied = 0
    for path, raw_val in raw_overrides:
        param_name = path.split(".")[-1]
        coerced = coerce_value(param_name, raw_val)
        _set_nested(base_data, path, coerced)
        applied += 1

    # Warn about potentially problematic config values
    leverage_max_cap = _get_nested(base_data, "global.trade.leverage_max_cap", 0)
    if isinstance(leverage_max_cap, (int, float)) and 0 < leverage_max_cap < 1.5:
        print(f"⚠️  leverage_max_cap={leverage_max_cap:.4f} effectively disables "
              f"dynamic leverage (caps all leverage to ~{leverage_max_cap:.1f}x)",
              file=sys.stderr)

    # Update header comment
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg_id = config_id_from_obj(base_data)
    pnl = _safe_float(selected.get("total_pnl", 0.0), 0.0)
    trades = _safe_int(selected.get("total_trades", 0), 0)
    wr = _safe_float(selected.get("win_rate", 0.0), 0.0) * 100
    pf = _safe_float(selected.get("profit_factor", 0.0), 0.0)
    dd = _safe_float(selected.get("max_drawdown_pct", 0.0), 0.0) * 100

    # Write output
    if args.output:
        out_path = args.output
    else:
        out_path = "/dev/stdout"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Generated by generate_config.py at {now}\n")
        f.write(f"# config_id: {cfg_id}\n")
        f.write(f"# Source: {os.path.basename(args.sweep_results)} "
                f"rank #{args.rank} by {args.sort_by}\n")
        f.write(f"# Backtest: PnL ${pnl:.2f} | {trades} trades | "
                f"WR {wr:.1f}% | PF {pf:.2f} | DD {dd:.1f}%\n")
        f.write(f"# Base: {os.path.basename(args.base_config)}\n")
        f.write(f"# Overrides applied: {applied}\n")
        yaml.safe_dump(base_data, f, sort_keys=False)

    if args.output:
        print(f"\n[generate] Config written to {args.output}", file=sys.stderr)
        print(f"[generate] config_id: {cfg_id}", file=sys.stderr)
        print(f"[generate] Applied {applied} overrides from sweep results.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
