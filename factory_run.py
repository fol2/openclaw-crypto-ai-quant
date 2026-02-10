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
import hashlib
import json
import os
import platform
import shutil
import sqlite3
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"path": str(p), "exists": False}

    out: dict[str, Any] = {"path": str(p), "exists": True, "is_file": p.is_file(), "is_dir": p.is_dir()}
    if p.is_file():
        st = p.stat()
        out.update(
            {
                "size_bytes": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "sha256": _sha256_file(p),
            }
        )
    return out


def _capture_repro_metadata(*, run_dir: Path, artifacts_root: Path, bt_cmd: list[str], meta: dict[str, Any]) -> None:
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort system/environment metadata. Failures should not block the pipeline.
    env_keys = [
        "MEI_BACKTESTER_BIN",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PYTHONHASHSEED",
    ]
    meta["repro"] = {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "argv0": sys.argv[0] if sys.argv else "",
        },
        "env": {k: os.getenv(k, "") for k in env_keys},
        "backtester_cmd": list(bt_cmd),
        "files": {
            "pyproject_toml": _file_fingerprint(AIQ_ROOT / "pyproject.toml"),
            "uv_lock": _file_fingerprint(AIQ_ROOT / "uv.lock"),
            "sweep_spec": _file_fingerprint(AIQ_ROOT / str(meta.get("args", {}).get("sweep_spec", ""))),
        },
        "cmds": [],
    }

    # Capture versions as command outputs into artifacts.
    version_cmds: list[tuple[str, list[str]]] = [
        ("uname", ["uname", "-a"]),
        ("git_status", ["git", "status", "--porcelain=v1"]),
        ("cargo_version", ["cargo", "--version"]),
        ("rustc_version", ["rustc", "--version"]),
        ("nvidia_smi", ["nvidia-smi", "-L"]),
        ("nvcc_version", ["nvcc", "--version"]),
    ]

    for name, argv in version_cmds:
        out_path = repro_dir / f"{name}.stdout.txt"
        err_path = repro_dir / f"{name}.stderr.txt"
        try:
            res = _run_cmd(argv, cwd=AIQ_ROOT, stdout_path=out_path, stderr_path=err_path)
            meta["repro"]["cmds"].append({"name": name, **res.__dict__})
        except Exception as e:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("", encoding="utf-8")
            err_path.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            meta["repro"]["cmds"].append(
                {
                    "name": name,
                    "argv": list(argv),
                    "cwd": str(AIQ_ROOT),
                    "exit_code": 127,
                    "elapsed_s": 0.0,
                    "stdout_path": str(out_path),
                    "stderr_path": str(err_path),
                    "exception": f"{type(e).__name__}: {e}",
                }
            )

    # Fingerprint the resolved backtester binary when it is a path.
    bt0 = str(bt_cmd[0]) if bt_cmd else ""
    bt_path = Path(bt0) if bt0 and ("/" in bt0 or bt0.endswith(".exe")) else None
    if bt_path and bt_path.exists():
        meta["repro"]["files"]["mei_backtester_bin"] = _file_fingerprint(bt_path)

    meta["repro"]["artifacts_root"] = str(artifacts_root)


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


def _utc_compact(ts_ms: int) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(int(ts_ms) / 1000))


def _registry_lookup_run_dir(*, artifacts_root: Path, run_id: str) -> Path | None:
    registry_db = default_registry_db_path(artifacts_root=artifacts_root)
    if not registry_db.exists():
        return None
    uri = f"file:{registry_db}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        row = con.execute("SELECT run_dir FROM runs WHERE run_id = ? LIMIT 1", (str(run_id),)).fetchone()
        if not row:
            return None
        run_dir = str(row[0] or "").strip()
        return Path(run_dir).expanduser().resolve() if run_dir else None
    finally:
        con.close()


def _scan_lookup_run_dir(*, artifacts_root: Path, run_id: str) -> Path | None:
    """Fallback when registry is missing: scan artifacts for a matching run_id."""
    target = str(run_id).strip()
    if not target:
        return None
    for date_dir in sorted([p for p in artifacts_root.iterdir() if p.is_dir()]):
        for run_dir in sorted([p for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]):
            meta_path = run_dir / "run_metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = _load_json(meta_path)
            except Exception:
                continue
            if str(meta.get("run_id", "")).strip() == target:
                return run_dir.resolve()
    return None


def _find_existing_run_dir(*, artifacts_root: Path, run_id: str) -> Path:
    p = _registry_lookup_run_dir(artifacts_root=artifacts_root, run_id=run_id)
    if p is None:
        p = _scan_lookup_run_dir(artifacts_root=artifacts_root, run_id=run_id)
    if p is None:
        raise FileNotFoundError(f"Could not locate run_id={run_id} under {artifacts_root}")
    return p


def _reproduce_run(*, artifacts_root: Path, source_run_id: str) -> int:
    """Re-run CPU replay/reporting for an existing run_id.

    This creates a new run directory with a derived run_id. It does not modify the original run directory.
    """
    source_run_id = str(source_run_id).strip()
    if not source_run_id:
        raise SystemExit("--reproduce cannot be empty")

    source_run_dir = _find_existing_run_dir(artifacts_root=artifacts_root, run_id=source_run_id)
    source_meta = _load_json(source_run_dir / "run_metadata.json")
    source_args = source_meta.get("args", {}) if isinstance(source_meta.get("args", {}), dict) else {}

    interval = str(source_args.get("interval", "1h"))
    candles_db = source_args.get("candles_db")
    funding_db = source_args.get("funding_db")

    generated_at_ms = int(time.time() * 1000)
    new_run_id = f"repro_{source_run_id}_{_utc_compact(generated_at_ms)}"
    run_dir = resolve_run_dir(artifacts_root=artifacts_root, run_id=new_run_id, generated_at_ms=generated_at_ms)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "run_id": new_run_id,
        "generated_at_ms": generated_at_ms,
        "run_date_utc": _run_date_utc(generated_at_ms),
        "run_dir": str(run_dir),
        "git_head": _git_head_sha(),
        "args": source_args,
        "steps": [],
        "reproduce_of_run_id": source_run_id,
        "reproduce_source_run_dir": str(source_run_dir),
        "reproduce_source_git_head": str(source_meta.get("git_head", "")).strip(),
    }

    bt_cmd = _resolve_backtester_cmd()
    _capture_repro_metadata(run_dir=run_dir, artifacts_root=artifacts_root, bt_cmd=bt_cmd, meta=meta)
    _write_json(run_dir / "run_metadata.json", meta)

    # ------------------------------------------------------------------
    # Copy configs from the source run
    # ------------------------------------------------------------------
    src_candidates = source_meta.get("candidate_configs", [])
    if not isinstance(src_candidates, list) or not src_candidates:
        raise FileNotFoundError(f"Missing candidate_configs in {source_run_dir / 'run_metadata.json'}")

    # Prefer reproducing only the candidates that were actually selected for replay.
    # Older runs may not have `selected`, so fall back to all entries.
    src_candidates_all = src_candidates
    src_candidates = [
        it
        for it in src_candidates_all
        if isinstance(it, dict) and ("selected" not in it or bool(it.get("selected")))
    ]
    if not src_candidates:
        src_candidates = [it for it in src_candidates_all if isinstance(it, dict)]

    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths: list[Path] = []
    candidate_config_ids: dict[str, str] = {}
    copied_candidates: list[dict[str, Any]] = []

    for it in src_candidates:
        if not isinstance(it, dict):
            continue
        src_path = str(it.get("path", "")).strip()
        src_cfg_id = str(it.get("config_id", "")).strip()
        if not src_path or not src_cfg_id:
            continue

        src = Path(src_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Missing source config file: {src}")

        dst = (configs_dir / src.name).resolve()
        shutil.copy2(src, dst)

        dst_cfg_id = config_id_from_yaml_file(dst)
        if dst_cfg_id != src_cfg_id:
            raise ValueError(f"config_id mismatch for {dst}: expected {src_cfg_id}, got {dst_cfg_id}")

        candidate_paths.append(dst)
        candidate_config_ids[str(dst)] = dst_cfg_id
        copied_candidates.append({"path": str(dst), "config_id": dst_cfg_id, "source_path": str(src)})

    if not candidate_paths:
        raise ValueError("No candidate configs found to reproduce")

    meta["candidate_configs"] = copied_candidates

    # ------------------------------------------------------------------
    # CPU replay / validation (v1: run replay once per candidate)
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
            interval,
            "--output",
            str(out_json),
        ]
        if candles_db:
            replay_argv += ["--candles-db", str(candles_db)]
        if funding_db:
            replay_argv += ["--funding-db", str(funding_db)]

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

        entry = entry_by_path.get(str(cfg_path))
        if entry is not None:
            entry["replay_report_path"] = str(out_json)
            entry["replay_stdout_path"] = str(replay_res.stdout_path or "")
            entry["replay_stderr_path"] = str(replay_res.stderr_path or "")

    # ------------------------------------------------------------------
    # Reports + registry
    # ------------------------------------------------------------------
    report_md = _render_ranked_report_md(replay_reports)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "report.md").write_text(report_md, encoding="utf-8")
    _write_json(run_dir / "reports" / "report.json", {"items": replay_reports})

    _write_json(run_dir / "run_metadata.json", meta)

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the nightly strategy factory workflow and store artifacts.")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--run-id", help="Unique identifier for this run (used in artifact paths).")
    mx.add_argument("--reproduce", metavar="RUN_ID", help="Reproduce an existing run_id (CPU replay + reports only).")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts).")

    ap.add_argument("--config", default="config/strategy_overrides.yaml", help="Base strategy YAML config path.")
    ap.add_argument("--interval", default="1h", help="Main interval for sweep/replay (default: 1h).")
    ap.add_argument("--candles-db", default=None, help="Optional candle DB path override.")
    ap.add_argument("--funding-db", default=None, help="Optional funding DB path for replay/sweep.")

    ap.add_argument("--sweep-spec", default="backtester/sweeps/smoke.yaml", help="Sweep spec YAML path.")
    ap.add_argument("--gpu", action="store_true", help="Use GPU sweep (requires CUDA build/runtime).")
    ap.add_argument("--tpe", action="store_true", help="Use TPE Bayesian optimisation for GPU sweeps (requires --gpu).")
    ap.add_argument("--tpe-trials", type=int, default=5000, help="Number of TPE trials (default: 5000).")
    ap.add_argument("--tpe-batch", type=int, default=256, help="Trials per GPU batch (default: 256).")
    ap.add_argument("--tpe-seed", type=int, default=42, help="RNG seed for TPE reproducibility (default: 42).")
    ap.add_argument("--top-n", type=int, default=0, help="Only print top N sweep results (0 = no summary).")

    ap.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="How many candidate configs to generate when shortlist is disabled (default: 3).",
    )
    ap.add_argument(
        "--sort-by",
        default="balanced",
        choices=["pnl", "dd", "pf", "wr", "sharpe", "trades", "balanced"],
        help="Sort metric for candidate selection when shortlist is disabled (default: balanced).",
    )
    ap.add_argument(
        "--shortlist-modes",
        default="dd,balanced",
        help="Comma-separated sort modes to generate per sweep (default: dd,balanced).",
    )
    ap.add_argument(
        "--shortlist-per-mode",
        type=int,
        default=10,
        help="How many unique configs to keep per mode (default: 10). Set 0 to disable shortlist.",
    )
    ap.add_argument(
        "--shortlist-max-rank",
        type=int,
        default=50,
        help="Max rank to scan per mode while deduplicating (default: 50).",
    )

    args = ap.parse_args(argv)

    artifacts_root = (AIQ_ROOT / str(args.artifacts_dir)).resolve()
    if args.reproduce:
        return _reproduce_run(artifacts_root=artifacts_root, source_run_id=str(args.reproduce))

    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("--run-id cannot be empty")

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

    bt_cmd = _resolve_backtester_cmd()
    _capture_repro_metadata(run_dir=run_dir, artifacts_root=artifacts_root, bt_cmd=bt_cmd, meta=meta)
    _write_json(run_dir / "run_metadata.json", meta)

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
    if bool(getattr(args, "tpe", False)):
        if not bool(args.gpu):
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)
            (run_dir / "reports" / "report.md").write_text(
                "# Factory Run Report\n\nError: --tpe requires --gpu.\n", encoding="utf-8"
            )
            _write_json(run_dir / "run_metadata.json", meta)
            return 1
        sweep_argv += [
            "--tpe",
            "--tpe-trials",
            str(int(getattr(args, "tpe_trials", 5000))),
            "--tpe-batch",
            str(int(getattr(args, "tpe_batch", 256))),
            "--tpe-seed",
            str(int(getattr(args, "tpe_seed", 42))),
        ]

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

    shortlist_per_mode = int(getattr(args, "shortlist_per_mode", 0) or 0)
    shortlist_max_rank = int(getattr(args, "shortlist_max_rank", 0) or 0)
    shortlist_modes_raw = str(getattr(args, "shortlist_modes", "") or "").strip()

    candidate_paths: list[Path] = []
    candidate_config_ids: dict[str, str] = {}
    candidate_entries: list[dict[str, Any]] = []
    entry_by_path: dict[str, dict[str, Any]] = {}

    # Multi-mode shortlist generation (AQC-403). Deduplicate across modes by config_id.
    if shortlist_per_mode > 0:
        allowed_modes = {"pnl", "dd", "pf", "wr", "sharpe", "trades", "balanced"}
        modes = [m.strip() for m in shortlist_modes_raw.split(",") if m.strip()]
        modes = [m for m in modes if m in allowed_modes]
        if not modes:
            modes = ["dd", "balanced"]
        if shortlist_max_rank <= 0:
            shortlist_max_rank = max(10, shortlist_per_mode * 5)

        seen_ids: set[str] = set()
        for mode in modes:
            kept = 0
            for rank in range(1, shortlist_max_rank + 1):
                out_yaml = configs_dir / f"candidate_{mode}_rank{rank}.yaml"
                gen_argv = [
                    "python3",
                    "tools/generate_config.py",
                    "--sweep-results",
                    str(sweep_out),
                    "--base-config",
                    str(args.config),
                    "--sort-by",
                    str(mode),
                    "--rank",
                    str(rank),
                    "-o",
                    str(out_yaml),
                ]
                gen_res = _run_cmd(
                    gen_argv,
                    cwd=AIQ_ROOT,
                    stdout_path=run_dir / "configs" / f"generate_config_{mode}_rank{rank}.stdout.txt",
                    stderr_path=run_dir / "configs" / f"generate_config_{mode}_rank{rank}.stderr.txt",
                )
                meta["steps"].append({"name": f"generate_config_{mode}_rank{rank}", **gen_res.__dict__})
                if gen_res.exit_code != 0:
                    _write_json(run_dir / "run_metadata.json", meta)
                    return int(gen_res.exit_code)

                cfg_id = config_id_from_yaml_file(out_yaml)
                candidate_config_ids[str(out_yaml)] = cfg_id

                is_dup = cfg_id in seen_ids
                selected = (not is_dup) and kept < shortlist_per_mode
                if selected:
                    seen_ids.add(cfg_id)
                    candidate_paths.append(out_yaml)
                    kept += 1

                entry = {
                    "path": str(out_yaml),
                    "config_id": cfg_id,
                    "sort_by": mode,
                    "rank": int(rank),
                    "selected": bool(selected),
                    "deduped": bool(is_dup),
                    "interval": str(args.interval),
                    "base_config_path": str(args.config),
                    "sweep_spec_path": str(args.sweep_spec),
                    "sweep_results_path": str(sweep_out),
                    "sweep_stdout_path": str(sweep_res.stdout_path or ""),
                    "sweep_stderr_path": str(sweep_res.stderr_path or ""),
                    "generate_stdout_path": str(gen_res.stdout_path or ""),
                    "generate_stderr_path": str(gen_res.stderr_path or ""),
                }
                candidate_entries.append(entry)
                entry_by_path[str(out_yaml)] = entry

                if kept >= shortlist_per_mode:
                    break

    # Single-mode generation (legacy): args.num_candidates + args.sort_by.
    else:
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
            cfg_id = config_id_from_yaml_file(out_yaml)
            candidate_config_ids[str(out_yaml)] = cfg_id
            entry = {
                "path": str(out_yaml),
                "config_id": cfg_id,
                "sort_by": str(args.sort_by),
                "rank": int(rank),
                "selected": True,
                "deduped": False,
                "interval": str(args.interval),
                "base_config_path": str(args.config),
                "sweep_spec_path": str(args.sweep_spec),
                "sweep_results_path": str(sweep_out),
                "sweep_stdout_path": str(sweep_res.stdout_path or ""),
                "sweep_stderr_path": str(sweep_res.stderr_path or ""),
                "generate_stdout_path": str(gen_res.stdout_path or ""),
                "generate_stderr_path": str(gen_res.stderr_path or ""),
            }
            candidate_entries.append(entry)
            entry_by_path[str(out_yaml)] = entry

    meta["candidate_configs"] = candidate_entries

    if not candidate_paths:
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (run_dir / "reports" / "report.md").write_text("# Factory Run Report\n\nNo candidate configs were selected.\n", encoding="utf-8")
        _write_json(run_dir / "run_metadata.json", meta)
        return 1

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
