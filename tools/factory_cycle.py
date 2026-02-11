#!/usr/bin/env python3
"""Run a full factory cycle and optionally auto-deploy the best config to paper.

This tool is designed for unattended runs (e.g. a systemd timer) on the `major-v8`
worktree. It orchestrates:

1) `factory_run.py` (data checks → sweep → candidate generation → CPU validation → registry)
2) Selection of the best non-rejected candidate
3) Optional paper deployment via `tools/paper_deploy.py` (atomic YAML write + optional restart)

The goal is to make the strategy factory loop runnable as a single command for a
parallel v8 instance, without touching production `master`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

AIQ_ROOT = Path(__file__).resolve().parents[1]

# When invoked as `python3 tools/factory_cycle.py`, sys.path[0] is `tools/`.
# Ensure the repo root is importable so `import factory_run` works consistently.
if str(AIQ_ROOT) not in sys.path:
    sys.path.insert(0, str(AIQ_ROOT))

import factory_run  # noqa: E402  (needs sys.path fix above)

try:
    from tools.paper_deploy import deploy_paper_config
    from tools.registry_index import default_registry_db_path
except ImportError:  # pragma: no cover
    from paper_deploy import deploy_paper_config  # type: ignore[no-redef]
    from registry_index import default_registry_db_path  # type: ignore[no-redef]

def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML mapping at root: {path}")
    return obj


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base (mutates base)."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return base
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def _norm_mode_key(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s in {"mode1", "m1"}:
        return "primary"
    if s in {"mode2", "m2"}:
        return "fallback"
    if s in {"mode3", "m3", "safety", "safe"}:
        return "conservative"
    if s in {"halt", "pause", "paused"}:
        return "flat"
    return s


def _apply_strategy_mode_overlay(*, base: dict[str, Any], strategy_mode: str) -> dict[str, Any]:
    """Return an effective YAML dict with modes.<strategy_mode> merged into global/symbols.

    Rust backtester ignores `modes:` currently, so the overlay must be materialised into
    the effective `global:` section before sweeps/replays to keep "what you replay" in
    sync with "what you trade".
    """
    mode_key = _norm_mode_key(strategy_mode)
    if not mode_key:
        return base

    modes = base.get("modes") if isinstance(base.get("modes"), dict) else {}
    if not isinstance(modes, dict):
        return base

    mode_over = modes.get(mode_key)
    if mode_over is None:
        mode_over = modes.get(str(mode_key).upper())
    if mode_over is None:
        mode_over = modes.get(str(mode_key).lower())
    if not isinstance(mode_over, dict):
        raise KeyError(f"strategy mode not found in YAML: {mode_key}")

    out = json.loads(json.dumps(base))  # cheap deep copy for YAML primitives
    if "global" not in out or not isinstance(out.get("global"), dict):
        out["global"] = {}
    if "symbols" not in out or not isinstance(out.get("symbols"), dict):
        out["symbols"] = {}

    if "global" in mode_over or "symbols" in mode_over:
        glob = mode_over.get("global")
        sym = mode_over.get("symbols")
        if isinstance(glob, dict):
            _deep_merge(out["global"], glob)  # type: ignore[arg-type]
        if isinstance(sym, dict):
            # Merge per-symbol overlays if present.
            for k, v in sym.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                k2 = k.strip().upper()
                if not k2:
                    continue
                if k2 not in out["symbols"] or not isinstance(out["symbols"].get(k2), dict):
                    out["symbols"][k2] = {}
                _deep_merge(out["symbols"][k2], v)  # type: ignore[arg-type]
        return out

    # Shorthand: modes.<mode>: {trade: {...}, engine: {...}, ...}
    _deep_merge(out["global"], mode_over)  # type: ignore[arg-type]
    return out


def _yaml_engine_interval(yaml_obj: dict[str, Any]) -> str:
    glob = yaml_obj.get("global") if isinstance(yaml_obj.get("global"), dict) else {}
    eng = glob.get("engine") if isinstance(glob.get("engine"), dict) else {}
    v = str(eng.get("interval", "") or "").strip()
    return v


@dataclass(frozen=True)
class Candidate:
    config_id: str
    config_path: str
    score_v1: float | None
    total_pnl: float
    max_drawdown_pct: float
    total_trades: int
    profit_factor: float
    rejected: bool
    reject_reason: str


def _parse_candidate(it: dict[str, Any]) -> Candidate | None:
    try:
        config_id = str(it.get("config_id", "")).strip()
        config_path = str(it.get("config_path", "")).strip()
        if not config_id or not config_path:
            return None
        score_v1 = it.get("score_v1", None)
        score = None if score_v1 is None else float(score_v1)
        return Candidate(
            config_id=config_id,
            config_path=config_path,
            score_v1=score,
            total_pnl=float(it.get("total_pnl", 0.0)),
            max_drawdown_pct=float(it.get("max_drawdown_pct", 0.0)),
            total_trades=int(it.get("total_trades", 0)),
            profit_factor=float(it.get("profit_factor", 0.0)),
            rejected=bool(it.get("rejected", False)),
            reject_reason=str(it.get("reject_reason", "") or "").strip(),
        )
    except Exception:
        return None


def _select_best_candidate(items: list[dict[str, Any]]) -> Candidate | None:
    parsed = [c for c in (_parse_candidate(it) for it in items) if c is not None]
    parsed_ok = [c for c in parsed if not bool(c.rejected)]
    if not parsed_ok:
        return None

    # Prefer score_v1 when present. If score_v1 is missing for all candidates, fall back to total_pnl.
    any_score = any(c.score_v1 is not None for c in parsed_ok)
    if any_score:
        parsed_ok.sort(key=lambda c: float(c.score_v1 or float("-inf")), reverse=True)
    else:
        parsed_ok.sort(key=lambda c: float(c.total_pnl), reverse=True)
    return parsed_ok[0]


def _query_run_dir(*, registry_db: Path, run_id: str) -> Path:
    con = sqlite3.connect(str(registry_db), timeout=2.0)
    try:
        row = con.execute("SELECT run_dir FROM runs WHERE run_id = ? LIMIT 1", (str(run_id),)).fetchone()
        if not row:
            raise FileNotFoundError(f"run_id not found in registry: {run_id}")
        run_dir = str(row[0] or "").strip()
        if not run_dir:
            raise FileNotFoundError(f"run_dir missing in registry for run_id: {run_id}")
        return Path(run_dir).expanduser().resolve()
    finally:
        con.close()


def _mark_deployed(*, registry_db: Path, run_id: str, config_id: str) -> None:
    con = sqlite3.connect(str(registry_db), timeout=2.0)
    try:
        with con:
            con.execute(
                "UPDATE run_configs SET deployed = 1 WHERE run_id = ? AND config_id = ?",
                (str(run_id), str(config_id)),
            )
    finally:
        con.close()


def _send_discord(*, target: str, message: str) -> None:
    tgt = str(target or "").strip()
    msg = str(message or "").strip()
    if not tgt or not msg:
        return
    try:
        timeout_s = float(os.getenv("AI_QUANT_DISCORD_SEND_TIMEOUT_S", "6") or 6)
    except Exception:
        timeout_s = 6.0
    timeout_s = max(1.0, min(30.0, timeout_s))
    try:
        subprocess.run(
            ["openclaw", "message", "send", "--channel", "discord", "--target", tgt, "--message", msg],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        # Notifications must never block the pipeline.
        return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run factory_run and optionally auto-deploy the best config to paper.")
    ap.add_argument("--run-id", default="", help="Run id (default: nightly_<UTC timestamp>).")
    ap.add_argument(
        "--profile",
        default="daily",
        choices=sorted(factory_run.PROFILE_DEFAULTS.keys()),
        help="Factory profile for trials/candidates (default: daily).",
    )
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root (default: artifacts).")
    ap.add_argument("--config", default="config/strategy_overrides.yaml", help="Base strategy YAML config path.")
    ap.add_argument(
        "--strategy-mode",
        default="",
        help="Optional modes.<mode> overlay to materialise into global for this run (e.g. primary, fallback).",
    )

    ap.add_argument("--sweep-spec", default="backtester/sweeps/allgpu_60k.yaml", help="Sweep spec YAML path.")
    ap.add_argument("--interval", default="", help="Main interval for sweep/replay (default: from effective YAML).")
    ap.add_argument("--candles-db", default="candles_dbs", help="Candle DB path (dir or glob).")
    ap.add_argument("--funding-db", default="candles_dbs/funding_rates.db", help="Funding DB path.")

    ap.add_argument("--gpu", action="store_true", help="Use GPU sweep (requires CUDA build/runtime).")
    ap.add_argument("--tpe", action="store_true", help="Use TPE Bayesian optimisation for GPU sweeps (requires --gpu).")
    ap.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation.")
    ap.add_argument("--slippage-stress", action="store_true", help="Enable slippage stress validation.")
    ap.add_argument("--concentration-checks", action="store_true", help="Enable concentration checks.")
    ap.add_argument("--sensitivity-checks", action="store_true", help="Enable sensitivity checks.")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run-id from artifacts (factory_run --resume).")

    ap.add_argument("--no-deploy", action="store_true", help="Run factory only; do not deploy.")
    ap.add_argument(
        "--yaml-path",
        default=str(AIQ_ROOT / "config" / "strategy_overrides.yaml"),
        help="Target strategy overrides YAML path for deployment (default: config/strategy_overrides.yaml).",
    )
    ap.add_argument("--deploy-reason", default="", help="Reason recorded in the paper deploy artefact.")
    ap.add_argument(
        "--restart",
        default="auto",
        choices=["auto", "always", "never"],
        help="Restart policy for deployment (default: auto).",
    )
    ap.add_argument("--service", default="openclaw-ai-quant-trader", help="systemd user service name for paper trader.")
    ap.add_argument("--ws-service", default="openclaw-ai-quant-ws-sidecar", help="systemd user service name for WS sidecar.")
    ap.add_argument("--pause-file", default="", help="Optional kill-switch file path to pause trading during restart.")
    ap.add_argument(
        "--pause-mode",
        default="close_only",
        choices=["close_only", "halt_all"],
        help="Pause mode to write into pause file (default: close_only).",
    )
    ap.add_argument("--leave-paused", action="store_true", help="Do not clear the pause file after a successful restart.")
    ap.add_argument("--verify-sleep-s", type=float, default=2.0, help="Seconds to wait before verifying service health.")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify YAML or restart services; still runs factory.")

    ap.add_argument(
        "--discord-target",
        default="",
        help="Optional Discord target to notify (uses `openclaw message send`).",
    )
    args = ap.parse_args(argv)

    run_id = str(args.run_id).strip() or f"nightly_{_utc_compact()}"
    artifacts_dir = Path(str(args.artifacts_dir)).expanduser().resolve()
    base_cfg_path = Path(str(args.config)).expanduser().resolve()

    # Materialise strategy-mode overlay into an effective base YAML (required for Rust backtester parity).
    effective_cfg_path = base_cfg_path
    if str(args.strategy_mode or "").strip():
        base_obj = _read_yaml(base_cfg_path)
        eff_obj = _apply_strategy_mode_overlay(base=base_obj, strategy_mode=str(args.strategy_mode))
        out_dir = (artifacts_dir / "_effective_configs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        effective_cfg_path = (out_dir / f"{run_id}.yaml").resolve()
        effective_cfg_path.write_text(yaml.safe_dump(eff_obj, sort_keys=False) + "\n", encoding="utf-8")

    # Default interval comes from YAML if not provided explicitly.
    interval = str(args.interval).strip()
    if not interval:
        try:
            eff_obj2 = _read_yaml(effective_cfg_path)
            interval = _yaml_engine_interval(eff_obj2) or "1h"
        except Exception:
            interval = "1h"

    factory_argv: list[str] = [
        "--run-id",
        run_id,
        "--profile",
        str(args.profile),
        "--artifacts-dir",
        str(artifacts_dir),
        "--config",
        str(effective_cfg_path),
        "--interval",
        str(interval),
        "--candles-db",
        str(args.candles_db),
        "--funding-db",
        str(args.funding_db),
        "--sweep-spec",
        str(args.sweep_spec),
    ]
    if bool(args.resume):
        factory_argv.append("--resume")
    if bool(args.gpu):
        factory_argv.append("--gpu")
    if bool(args.tpe):
        factory_argv.append("--tpe")
    if bool(args.walk_forward):
        factory_argv.append("--walk-forward")
    if bool(args.slippage_stress):
        factory_argv.append("--slippage-stress")
    if bool(args.concentration_checks):
        factory_argv.append("--concentration-checks")
    if bool(args.sensitivity_checks):
        factory_argv.append("--sensitivity-checks")

    rc = int(factory_run.main(factory_argv))
    if rc != 0:
        _send_discord(
            target=str(args.discord_target),
            message=f"Factory cycle FAILED run_id={run_id} (exit={rc}). Check systemd logs / artifacts.",
        )
        return rc

    registry_db = default_registry_db_path(artifacts_root=artifacts_dir)
    run_dir = _query_run_dir(registry_db=registry_db, run_id=run_id)
    report_path = run_dir / "reports" / "report.json"
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    items = rep.get("items", []) if isinstance(rep, dict) else []
    if not isinstance(items, list):
        items = []

    best = _select_best_candidate(items)
    if best is None:
        _send_discord(
            target=str(args.discord_target),
            message=f"Factory cycle OK but produced no deployable candidates (all rejected). run_id={run_id}",
        )
        return 2

    selection = {
        "version": "factory_cycle_selection_v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "selected": best.__dict__,
        "effective_config_path": str(effective_cfg_path),
        "interval": str(interval),
        "deployed": False,
    }
    (run_dir / "reports" / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if bool(args.no_deploy):
        _send_discord(
            target=str(args.discord_target),
            message=f"Factory cycle OK (no-deploy). run_id={run_id} selected={best.config_id[:12]}",
        )
        return 0

    pause_file = Path(args.pause_file).expanduser().resolve() if str(args.pause_file).strip() else None
    reason = str(args.deploy_reason).strip() or f"auto factory_cycle run_id={run_id} profile={args.profile}"

    try:
        deploy_dir = deploy_paper_config(
            config_id=str(best.config_id),
            artifacts_dir=artifacts_dir,
            yaml_path=Path(str(args.yaml_path)),
            out_dir=None,
            reason=reason,
            restart=str(args.restart),
            service=str(args.service),
            dry_run=bool(args.dry_run),
            validate=True,
            ws_service=str(args.ws_service),
            pause_file=pause_file,
            pause_mode=str(args.pause_mode),
            resume_on_success=not bool(args.leave_paused),
            verify_sleep_s=float(args.verify_sleep_s),
        )
    except Exception as e:
        _send_discord(
            target=str(args.discord_target),
            message=f"Factory cycle deploy FAILED run_id={run_id} config_id={best.config_id[:12]} error={type(e).__name__}: {e}",
        )
        raise

    if not bool(args.dry_run):
        try:
            _mark_deployed(registry_db=registry_db, run_id=run_id, config_id=str(best.config_id))
        except Exception:
            pass

    selection["deployed"] = not bool(args.dry_run)
    selection["deploy_dir"] = str(deploy_dir)
    (run_dir / "reports" / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    msg = (
        f"Factory cycle DEPLOYED run_id={run_id}\n"
        f"- config_id: {best.config_id}\n"
        f"- score_v1: {best.score_v1 if best.score_v1 is not None else 'n/a'}\n"
        f"- total_pnl: {best.total_pnl:.2f}\n"
        f"- pf: {best.profit_factor:.3f}\n"
        f"- dd: {best.max_drawdown_pct * 100:.2f}%\n"
        f"- trades: {best.total_trades}\n"
    )
    _send_discord(target=str(args.discord_target), message=msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
