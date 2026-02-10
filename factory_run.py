#!/usr/bin/env python3
"""Nightly strategy factory orchestrator.

This script runs the end-to-end workflow:
1) Data checks (candles + funding)
2) Sweep (GPU optional)
3) Generate a shortlist of candidate configs
4) Replay/validate candidates on CPU
5) Emit a ranked summary report and store all artifacts under a run directory

The goal is reproducibility: every run has a `run_id` and a self-contained artifact folder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.config_id import config_id_from_yaml_file
except ImportError:  # pragma: no cover
    from config_id import config_id_from_yaml_file  # type: ignore[no-redef]

try:
    from tools.registry_index import default_registry_db_path, ingest_run_dir
except ImportError:  # pragma: no cover
    from registry_index import default_registry_db_path, ingest_run_dir  # type: ignore[no-redef]


AIQ_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CmdResult:
    argv: list[str]
    cwd: str
    exit_code: int
    elapsed_s: float
    stdout_path: str | None
    stderr_path: str | None


def _run_cmd(
    argv: list[str],
    *,
    cwd: Path,
    stdout_path: Path | None,
    stderr_path: Path | None,
    env: dict[str, str] | None = None,
) -> CmdResult:
    t0 = time.time()

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_f = stdout_path.open("w", encoding="utf-8")
    else:
        stdout_f = subprocess.DEVNULL  # type: ignore[assignment]

    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_f = stderr_path.open("w", encoding="utf-8")
    else:
        stderr_f = subprocess.DEVNULL  # type: ignore[assignment]

    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
            check=False,
            text=True,
        )
        exit_code = int(proc.returncode)
    finally:
        if hasattr(stdout_f, "close"):
            stdout_f.close()  # type: ignore[call-arg]
        if hasattr(stderr_f, "close"):
            stderr_f.close()  # type: ignore[call-arg]

    return CmdResult(
        argv=list(argv),
        cwd=str(cwd),
        exit_code=exit_code,
        elapsed_s=float(time.time() - t0),
        stdout_path=str(stdout_path) if stdout_path is not None else None,
        stderr_path=str(stderr_path) if stderr_path is not None else None,
    )


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(AIQ_ROOT)).decode("utf-8").strip()
        return out
    except Exception:
        return ""


def _resolve_backtester_cmd() -> list[str]:
    env_bin = os.getenv("MEI_BACKTESTER_BIN", "").strip()
    if env_bin:
        return [env_bin]

    rel = AIQ_ROOT / "backtester" / "target" / "release" / "mei-backtester"
    if rel.exists():
        return [str(rel)]

    # Fallback: build+run via cargo.
    return ["cargo", "run", "-p", "bt-cli", "--bin", "mei-backtester", "--"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarise_replay_report(path: Path) -> dict[str, Any]:
    d = _load_json(path)
    return {
        "path": str(path),
        "final_balance": float(d.get("final_balance", 0.0)),
        "total_pnl": float(d.get("total_pnl", 0.0)),
        "total_trades": int(d.get("total_trades", 0)),
        "win_rate": float(d.get("win_rate", 0.0)),
        "profit_factor": float(d.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(d.get("max_drawdown_pct", 0.0)),
        "total_fees": float(d.get("total_fees", 0.0)),
    }


def _render_ranked_report_md(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Factory Run Report")
    lines.append("")
    if not items:
        lines.append("No replay reports were produced.")
        lines.append("")
        return "\n".join(lines)

    items_sorted = sorted(items, key=lambda x: float(x.get("total_pnl", 0.0)), reverse=True)

    lines.append("## Ranked Candidates (by total_pnl)")
    lines.append("")
    lines.append("| Rank | total_pnl | max_dd_pct | trades | win_rate | profit_factor | config_id | report |")
    lines.append("| ---: | --------: | ---------: | -----: | -------: | ------------: | :------ | :----- |")
    for i, it in enumerate(items_sorted, start=1):
        cfg_id = str(it.get("config_id", ""))
        cfg_id_short = cfg_id[:12] if len(cfg_id) > 12 else cfg_id
        lines.append(
            "| {rank} | {pnl:.2f} | {dd:.4f} | {trades} | {wr:.4f} | {pf:.4f} | `{cfg_id}` | `{path}` |".format(
                rank=i,
                pnl=float(it.get("total_pnl", 0.0)),
                dd=float(it.get("max_drawdown_pct", 0.0)),
                trades=int(it.get("total_trades", 0)),
                wr=float(it.get("win_rate", 0.0)),
                pf=float(it.get("profit_factor", 0.0)),
                cfg_id=cfg_id_short,
                path=str(it.get("path", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _run_date_utc(generated_at_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(generated_at_ms / 1000))


def resolve_run_dir(*, artifacts_root: Path, run_id: str, generated_at_ms: int) -> Path:
    """Resolve a standard run directory path under the artifacts root.

    Layout:
      artifacts_root/YYYY-MM-DD/run_<run_id>/
    """

    run_date = _run_date_utc(generated_at_ms)
    return (artifacts_root / run_date / f"run_{run_id}").resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the nightly strategy factory workflow and store artifacts.")
    ap.add_argument("--run-id", required=True, help="Unique identifier for this run (used in artifact paths).")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts).")

    ap.add_argument("--config", default="config/strategy_overrides.yaml", help="Base strategy YAML config path.")
    ap.add_argument("--interval", default="1h", help="Main interval for sweep/replay (default: 1h).")
    ap.add_argument("--candles-db", default=None, help="Optional candle DB path override.")
    ap.add_argument("--funding-db", default=None, help="Optional funding DB path for replay/sweep.")

    ap.add_argument("--sweep-spec", default="backtester/sweeps/smoke.yaml", help="Sweep spec YAML path.")
    ap.add_argument("--gpu", action="store_true", help="Use GPU sweep (requires CUDA build/runtime).")
    ap.add_argument("--top-n", type=int, default=0, help="Only print top N sweep results (0 = no summary).")

    ap.add_argument("--num-candidates", type=int, default=3, help="How many candidate configs to generate (default: 3).")
    ap.add_argument("--sort-by", default="balanced", choices=["pnl", "dd", "pf", "wr", "sharpe", "trades", "balanced"])

    args = ap.parse_args(argv)

    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("--run-id cannot be empty")

    artifacts_root = (AIQ_ROOT / str(args.artifacts_dir)).resolve()
    generated_at_ms = int(time.time() * 1000)
    run_dir = resolve_run_dir(artifacts_root=artifacts_root, run_id=run_id, generated_at_ms=generated_at_ms)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_ms": generated_at_ms,
        "run_date_utc": _run_date_utc(generated_at_ms),
        "run_dir": str(run_dir),
        "git_head": _git_head_sha(),
        "args": vars(args),
        "steps": [],
    }

    # ------------------------------------------------------------------
    # 1) Data checks
    # ------------------------------------------------------------------
    candle_check = _run_cmd(
        ["python3", "tools/check_candle_dbs.py", "--json-indent", "2"],
        cwd=AIQ_ROOT,
        stdout_path=run_dir / "data_checks" / "candle_dbs.json",
        stderr_path=run_dir / "data_checks" / "candle_dbs.stderr.txt",
    )
    meta["steps"].append({"name": "check_candle_dbs", **candle_check.__dict__})
    if candle_check.exit_code != 0:
        _write_json(run_dir / "run_metadata.json", meta)
        return int(candle_check.exit_code)

    funding_check = _run_cmd(
        ["python3", "tools/check_funding_rates_db.py"],
        cwd=AIQ_ROOT,
        stdout_path=run_dir / "data_checks" / "funding_rates.json",
        stderr_path=run_dir / "data_checks" / "funding_rates.stderr.txt",
    )
    meta["steps"].append({"name": "check_funding_rates_db", **funding_check.__dict__})
    if funding_check.exit_code != 0:
        _write_json(run_dir / "run_metadata.json", meta)
        return int(funding_check.exit_code)

    # ------------------------------------------------------------------
    # 2) Sweep
    # ------------------------------------------------------------------
    bt_cmd = _resolve_backtester_cmd()
    sweep_out = run_dir / "sweeps" / "sweep_results.jsonl"
    sweep_argv = bt_cmd + [
        "sweep",
        "--config",
        str(args.config),
        "--sweep-spec",
        str(args.sweep_spec),
        "--interval",
        str(args.interval),
        "--output",
        str(sweep_out),
        "--top-n",
        str(int(args.top_n)),
    ]
    if args.candles_db:
        sweep_argv += ["--candles-db", str(args.candles_db)]
    if args.funding_db:
        sweep_argv += ["--funding-db", str(args.funding_db)]
    if bool(args.gpu):
        sweep_argv += ["--gpu"]

    sweep_res = _run_cmd(
        sweep_argv,
        cwd=AIQ_ROOT / "backtester",
        stdout_path=run_dir / "sweeps" / "sweep.stdout.txt",
        stderr_path=run_dir / "sweeps" / "sweep.stderr.txt",
    )
    meta["steps"].append({"name": "sweep", **sweep_res.__dict__})
    if sweep_res.exit_code != 0:
        _write_json(run_dir / "run_metadata.json", meta)
        return int(sweep_res.exit_code)

    # ------------------------------------------------------------------
    # 3) Candidate config generation
    # ------------------------------------------------------------------
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths: list[Path] = []
    candidate_config_ids: dict[str, str] = {}
    for rank in range(1, int(args.num_candidates) + 1):
        out_yaml = configs_dir / f"candidate_{args.sort_by}_rank{rank}.yaml"
        gen_argv = [
            "python3",
            "tools/generate_config.py",
            "--sweep-results",
            str(sweep_out),
            "--base-config",
            str(args.config),
            "--sort-by",
            str(args.sort_by),
            "--rank",
            str(rank),
            "-o",
            str(out_yaml),
        ]
        gen_res = _run_cmd(
            gen_argv,
            cwd=AIQ_ROOT,
            stdout_path=run_dir / "configs" / f"generate_config_rank{rank}.stdout.txt",
            stderr_path=run_dir / "configs" / f"generate_config_rank{rank}.stderr.txt",
        )
        meta["steps"].append({"name": f"generate_config_rank{rank}", **gen_res.__dict__})
        if gen_res.exit_code != 0:
            _write_json(run_dir / "run_metadata.json", meta)
            return int(gen_res.exit_code)
        candidate_paths.append(out_yaml)
        candidate_config_ids[str(out_yaml)] = config_id_from_yaml_file(out_yaml)

    meta["candidate_configs"] = [
        {"path": p, "config_id": candidate_config_ids[p]} for p in sorted(candidate_config_ids.keys())
    ]

    # ------------------------------------------------------------------
    # 4) CPU replay / validation (minimal v1: run replay once per candidate)
    # ------------------------------------------------------------------
    replays_dir = run_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    replay_reports: list[dict[str, Any]] = []
    for cfg_path in candidate_paths:
        out_json = replays_dir / f"{cfg_path.stem}.replay.json"
        replay_argv = bt_cmd + [
            "replay",
            "--config",
            str(cfg_path),
            "--interval",
            str(args.interval),
            "--output",
            str(out_json),
        ]
        if args.candles_db:
            replay_argv += ["--candles-db", str(args.candles_db)]
        if args.funding_db:
            replay_argv += ["--funding-db", str(args.funding_db)]

        replay_res = _run_cmd(
            replay_argv,
            cwd=AIQ_ROOT / "backtester",
            stdout_path=run_dir / "replays" / f"{cfg_path.stem}.stdout.txt",
            stderr_path=run_dir / "replays" / f"{cfg_path.stem}.stderr.txt",
        )
        meta["steps"].append({"name": f"replay_{cfg_path.stem}", **replay_res.__dict__})
        if replay_res.exit_code != 0:
            _write_json(run_dir / "run_metadata.json", meta)
            return int(replay_res.exit_code)

        summary = _summarise_replay_report(out_json)
        summary["config_path"] = str(cfg_path)
        summary["config_id"] = candidate_config_ids[str(cfg_path)]
        replay_reports.append(summary)

    # ------------------------------------------------------------------
    # 5) Final report
    # ------------------------------------------------------------------
    report_md = _render_ranked_report_md(replay_reports)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "report.md").write_text(report_md, encoding="utf-8")
    _write_json(run_dir / "reports" / "report.json", {"items": replay_reports})

    # Persist metadata before registry ingestion (ingest reads run_metadata.json).
    _write_json(run_dir / "run_metadata.json", meta)

    # ------------------------------------------------------------------
    # 6) Registry index
    # ------------------------------------------------------------------
    registry_db = default_registry_db_path(artifacts_root=artifacts_root)
    meta["registry_db"] = str(registry_db)
    try:
        res = ingest_run_dir(registry_db=registry_db, run_dir=run_dir)
        meta["registry_ingest"] = res.__dict__
    except Exception as e:
        meta["registry_error"] = str(e)
        _write_json(run_dir / "run_metadata.json", meta)
        return 1

    _write_json(run_dir / "run_metadata.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
