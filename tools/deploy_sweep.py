#!/usr/bin/env python3
"""Deploy optimised sweep results to live/paper config with optional position close + restart.

Usage:
    python deploy_sweep.py --sweep-results sweep_results.jsonl --rank 1 --dry-run
    python deploy_sweep.py --sweep-results sweep_results.jsonl --rank 1 --close-paper --restart --yes
    python deploy_sweep.py --sweep-results sweep_results.jsonl --rank 1 --close-live --close-paper --restart --yes
"""

import argparse
import copy
import json
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
YAML_PATH = os.path.join(PROJECT_DIR, "config", "strategy_overrides.yaml")
CHANGELOG_PATH = os.path.join(PROJECT_DIR, "strategy_changelog.json")
PAPER_DB = os.path.join(PROJECT_DIR, "trading_engine.db")
LIVE_DB = os.path.join(PROJECT_DIR, "trading_engine_live.db")
SECRETS_PATH = os.path.join(PROJECT_DIR, "secrets.json")


# ---------------------------------------------------------------------------
# Parse sweep results
# ---------------------------------------------------------------------------

def load_sweep_results(path: str) -> list[dict]:
    """Load JSONL sweep results, sort by total_pnl descending."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    results.sort(key=lambda r: r.get("total_pnl", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# YAML operations (ruamel.yaml preserves comments)
# ---------------------------------------------------------------------------

def _load_yaml(path: str):
    """Load YAML preserving comments. Returns (yaml_obj, data)."""
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return yaml, data


def _backup_yaml(path: str):
    ts = int(time.time())
    backup = f"{path}.bak.{ts}"
    import shutil
    shutil.copy2(path, backup)
    print(f"[deploy] Backed up YAML → {backup}", file=sys.stderr)
    return backup


def _set_nested(data: dict, dotpath: str, value):
    """Set a value in a nested dict using dot notation.

    e.g. 'trade.sl_atr_mult' → data['global']['trade']['sl_atr_mult']
    """
    # Prepend 'global.' if not already present (sweep overrides use short paths)
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
# Show diff
# ---------------------------------------------------------------------------

def show_diff(current_yaml: dict, overrides: dict[str, float]):
    """Display what would change."""
    print("\n--- Config Changes ---\n", file=sys.stderr)
    for path, new_val in sorted(overrides.items()):
        old_val = _get_nested(current_yaml, path, "???")
        marker = " ✓" if old_val != new_val else " (same)"
        print(f"  {path}: {old_val} → {new_val}{marker}", file=sys.stderr)
    print(file=sys.stderr)


def show_metrics(result: dict):
    """Display backtest metrics for the selected sweep result."""
    print("--- Backtest Metrics ---\n", file=sys.stderr)
    for key in ["total_pnl", "total_trades", "win_rate", "profit_factor",
                 "sharpe_ratio", "max_drawdown_pct", "final_balance"]:
        val = result.get(key, "N/A")
        if isinstance(val, float):
            if "pct" in key or "rate" in key:
                print(f"  {key}: {val*100:.1f}%", file=sys.stderr)
            else:
                print(f"  {key}: {val:.2f}", file=sys.stderr)
        else:
            print(f"  {key}: {val}", file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Close positions
# ---------------------------------------------------------------------------

def close_live_positions(*, max_retries: int = 3):
    """Close all live positions via Hyperliquid API."""
    sys.path.insert(0, PROJECT_DIR)
    from exchange.executor import load_live_secrets, HyperliquidLiveExecutor

    secrets = load_live_secrets(SECRETS_PATH)
    executor = HyperliquidLiveExecutor(
        secret_key=secrets.secret_key,
        main_address=secrets.main_address,
    )

    positions = executor.get_positions(force=True)
    if not positions:
        print("[deploy] No live positions to close.", file=sys.stderr)
        return True

    print(f"[deploy] Closing {len(positions)} live position(s)...", file=sys.stderr)

    for sym, pdata in positions.items():
        is_long = pdata["type"] == "LONG"
        size = pdata["size"]
        # To close: sell if long, buy if short
        is_buy = not is_long

        success = False
        for attempt in range(1, max_retries + 1):
            print(f"  [{sym}] Attempt {attempt}/{max_retries}: "
                  f"closing {pdata['type']} size={size:.6f}", file=sys.stderr)
            res = executor.market_close(
                sym,
                is_buy=is_buy,
                sz=size,
                slippage_pct=0.02,
            )
            if res is not None:
                print(f"  [{sym}] Close submitted. Waiting 5s...", file=sys.stderr)
                time.sleep(5)
                # Verify
                remaining = executor.get_positions(force=True)
                if sym not in remaining:
                    print(f"  [{sym}] Closed successfully.", file=sys.stderr)
                    success = True
                    break
                else:
                    print(f"  [{sym}] Still open after close attempt.", file=sys.stderr)
            else:
                print(f"  [{sym}] Close rejected/failed.", file=sys.stderr)
                time.sleep(2)

        if not success:
            print(f"  [{sym}] FAILED to close after {max_retries} attempts. ABORTING.", file=sys.stderr)
            return False

    print("[deploy] All live positions closed.", file=sys.stderr)
    return True


def close_paper_positions():
    """Close all paper positions by inserting CLOSE trades and clearing position_state."""
    if not os.path.exists(PAPER_DB):
        print("[deploy] Paper DB not found — nothing to close.", file=sys.stderr)
        return True

    conn = sqlite3.connect(PAPER_DB, timeout=10)
    conn.row_factory = sqlite3.Row

    # Get open positions (simplified: just read position_state)
    rows = conn.execute("SELECT symbol FROM position_state").fetchall()
    if not rows:
        print("[deploy] No paper positions to close.", file=sys.stderr)
        conn.close()
        return True

    now_iso = datetime.now(timezone.utc).isoformat()
    # Get latest balance
    bal_row = conn.execute("SELECT balance FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    balance = float(bal_row["balance"]) if bal_row else 0.0

    for row in rows:
        sym = row["symbol"]
        print(f"  [{sym}] Inserting CLOSE trade for paper position.", file=sys.stderr)
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, type, action, price, size, notional, "
            "reason, confidence, pnl, fee_usd, balance, entry_atr, leverage, margin_used) "
            "VALUES (?, ?, 'SYSTEM', 'CLOSE', 0, 0, 0, 'deploy_sweep close', 'N/A', 0, 0, ?, 0, 0, 0)",
            (now_iso, sym, balance),
        )

    conn.execute("DELETE FROM position_state")
    conn.commit()
    conn.close()

    print(f"[deploy] Closed {len(rows)} paper position(s).", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

def update_changelog(overrides: dict[str, float], metrics: dict, version: str):
    """Append entry to strategy_changelog.json."""
    changelog = {"current_version": version, "history": []}
    if os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
            changelog = json.load(f)

    changes = [f"DEPLOY: Applied sweep-optimised config (rank #{metrics.get('_rank', 1)})."]
    for path, val in sorted(overrides.items()):
        changes.append(f"  {path}: → {val}")

    pnl = metrics.get("total_pnl", 0)
    trades = metrics.get("total_trades", 0)
    wr = metrics.get("win_rate", 0) * 100
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    changes.append(
        f"Backtest verification: ${pnl:.2f} PnL / {trades} trades / "
        f"{wr:.1f}% WR / PF {pf:.2f} / Sharpe {sharpe:.3f}"
    )

    entry = {
        "version": version,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "changes": changes,
    }

    changelog["current_version"] = version
    changelog["history"].insert(0, entry)

    with open(CHANGELOG_PATH, "w", encoding="utf-8") as f:
        json.dump(changelog, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[deploy] Changelog updated → {version}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Restart services
# ---------------------------------------------------------------------------

def restart_services():
    """Restart paper + live trading services."""
    for svc in ["openclaw-ai-quant-trader", "openclaw-ai-quant-live"]:
        print(f"[deploy] Restarting {svc}...", file=sys.stderr)
        result = subprocess.run(
            ["systemctl", "--user", "restart", svc],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  {svc}: restarted OK", file=sys.stderr)
        else:
            print(f"  {svc}: restart FAILED — {result.stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deploy sweep-optimised config")
    parser.add_argument("--sweep-results", required=True, help="Path to sweep results JSONL file")
    parser.add_argument("--rank", type=int, default=1, help="Pick Nth best result (1 = best PnL)")
    parser.add_argument("--close-live", action="store_true", help="Close all live positions before deploy")
    parser.add_argument("--close-paper", action="store_true", help="Close all paper positions before deploy")
    parser.add_argument("--restart", action="store_true", help="Restart paper + live services after deploy")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--version", type=str, default=None,
                        help="Version tag for changelog (auto-incremented if omitted)")
    args = parser.parse_args()

    # Load sweep results
    results = load_sweep_results(args.sweep_results)
    if not results:
        print("[deploy] No results found in sweep file.", file=sys.stderr)
        sys.exit(1)

    idx = args.rank - 1
    if idx < 0 or idx >= len(results):
        print(f"[deploy] Rank {args.rank} out of range (1..{len(results)}).", file=sys.stderr)
        sys.exit(1)

    selected = results[idx]
    overrides = selected.get("overrides", {})
    if not overrides:
        print("[deploy] Selected result has no overrides.", file=sys.stderr)
        sys.exit(1)

    # Ensure override values are numeric
    overrides = {k: float(v) for k, v in overrides.items()}

    # Load current YAML
    yaml, current_data = _load_yaml(YAML_PATH)

    # Show diff + metrics
    show_diff(current_data, overrides)
    show_metrics(selected)

    if args.dry_run:
        print("[deploy] --dry-run: no changes applied.", file=sys.stderr)
        return

    # Confirm
    if not args.yes:
        answer = input("Apply these changes? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("[deploy] Aborted.", file=sys.stderr)
            return

    # Close positions
    if args.close_live:
        if not close_live_positions():
            print("[deploy] Live close failed. Aborting deploy.", file=sys.stderr)
            sys.exit(1)

    if args.close_paper:
        if not close_paper_positions():
            print("[deploy] Paper close failed. Aborting deploy.", file=sys.stderr)
            sys.exit(1)

    # Backup + apply YAML changes
    _backup_yaml(YAML_PATH)

    for path, value in overrides.items():
        # Convert float to int if it's a whole number and looks like an integer param
        int_params = {
            "max_open_positions", "max_adds_per_symbol", "add_cooldown_minutes",
            "reentry_cooldown_minutes", "reentry_cooldown_min_mins", "reentry_cooldown_max_mins",
            "max_entry_orders_per_loop", "adx_window", "ema_fast_window", "ema_slow_window",
            "bb_window", "ema_macro_window", "atr_window", "rsi_window", "bb_width_avg_window",
            "vol_sma_window", "vol_trend_window", "stoch_rsi_window", "stoch_rsi_smooth1",
            "stoch_rsi_smooth2", "slow_drift_slope_window", "min_signals",
        }
        param_name = path.split(".")[-1]
        if param_name in int_params and value == int(value):
            value = int(value)
        _set_nested(current_data, path, value)

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(current_data, f)

    print(f"[deploy] YAML updated: {YAML_PATH}", file=sys.stderr)

    # Determine version
    version = args.version
    if not version:
        if os.path.exists(CHANGELOG_PATH):
            with open(CHANGELOG_PATH, "r") as f:
                cl = json.load(f)
            cur = cl.get("current_version", "v0.000")
            # Auto-increment: v6.000 → v6.001
            try:
                prefix, num = cur.rsplit(".", 1)
                version = f"{prefix}.{int(num) + 1:03d}"
            except Exception:
                version = cur + ".1"
        else:
            version = "v6.001"

    selected["_rank"] = args.rank
    update_changelog(overrides, selected, version)

    # Restart
    if args.restart:
        restart_services()

    print(f"\n[deploy] Done. Version: {version}", file=sys.stderr)


if __name__ == "__main__":
    main()
