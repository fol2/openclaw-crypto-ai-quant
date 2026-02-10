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


PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    # Very fast profile for verifying the pipeline end-to-end.
    "smoke": {
        "tpe_trials": 2000,
        "num_candidates": 2,
        "shortlist_per_mode": 3,
        "shortlist_max_rank": 20,
    },
    # Default weekday run profile.
    "daily": {
        "tpe_trials": 5000,
        "num_candidates": 3,
        "shortlist_per_mode": 10,
        "shortlist_max_rank": 50,
    },
    # Heavier profile for deeper sweeps and larger shortlists.
    "deep": {
        "tpe_trials": 500000,
        "num_candidates": 5,
        "shortlist_per_mode": 20,
        "shortlist_max_rank": 200,
    },
}


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


def _gpu_compute_processes(*, stdout_path: Path, stderr_path: Path) -> tuple[CmdResult, list[str]]:
    """Return a list of running CUDA compute processes (best-effort).

    Uses nvidia-smi compute-apps query; if the command fails, callers should treat the GPU as unavailable.
    """

    res = _run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=AIQ_ROOT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if res.exit_code != 0:
        return res, []

    try:
        lines = [ln.strip() for ln in stdout_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        lines = []
    return res, lines


def _ensure_gpu_idle_or_exit(*, run_dir: Path, meta: dict[str, Any], wait_s: int, poll_s: int) -> int:
    """Gate GPU sweeps to avoid interfering with other GPU workloads.

    If `wait_s` is >0, this will poll for an idle GPU until the timeout; otherwise it exits immediately.
    """

    wait_s = int(wait_s or 0)
    poll_s = int(poll_s or 0)
    if wait_s < 0:
        wait_s = 0
    if poll_s <= 0:
        poll_s = 10

    t0 = time.time()
    deadline = t0 + float(wait_s)
    attempt = 0

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    while True:
        attempt += 1
        out_path = run_dir / "logs" / "gpu_check.stdout.txt"
        err_path = run_dir / "logs" / "gpu_check.stderr.txt"
        res, procs = _gpu_compute_processes(stdout_path=out_path, stderr_path=err_path)
        meta["steps"].append({"name": f"gpu_check_attempt{attempt}", **res.__dict__})

        if res.exit_code != 0:
            meta["gpu_check"] = {
                "idle": False,
                "error": f"nvidia-smi failed with exit_code={res.exit_code}",
                "wait_s": wait_s,
                "poll_s": poll_s,
                "attempts": attempt,
            }
            _write_json(run_dir / "run_metadata.json", meta)
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)
            (run_dir / "reports" / "report.md").write_text(
                "# Factory Run Report\n\nError: GPU requested but nvidia-smi failed.\n",
                encoding="utf-8",
            )
            return 1

        idle = not bool(procs)
        meta["gpu_check"] = {
            "idle": bool(idle),
            "processes": list(procs),
            "wait_s": wait_s,
            "poll_s": poll_s,
            "attempts": attempt,
            "waited_s": int(time.time() - t0),
        }
        _write_json(run_dir / "run_metadata.json", meta)

        if idle:
            return 0

        if time.time() >= deadline:
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)
            details = "\n".join([f"- {p}" for p in procs]) if procs else "- (unknown)\n"
            (run_dir / "reports" / "report.md").write_text(
                "# Factory Run Report\n\n"
                "Error: GPU appears busy (compute processes detected). Try again later.\n\n"
                "Detected processes:\n"
                f"{details}\n",
                encoding="utf-8",
            )
            return 2

        time.sleep(float(max(1, poll_s)))


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


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_reject_reason(it: dict[str, Any], reason: str) -> None:
    reason = str(reason or "").strip()
    if not reason:
        return
    if not bool(it.get("rejected")):
        it["rejected"] = True
        it["reject_reason"] = reason
        return
    prev = str(it.get("reject_reason", "") or "").strip()
    if not prev:
        it["reject_reason"] = reason
        return
    if reason in prev:
        return
    it["reject_reason"] = f"{prev}; {reason}"


def _compute_score_v1(
    it: dict[str, Any],
    *,
    min_trades: int,
    trades_penalty_weight: float,
) -> dict[str, Any] | None:
    """Compute stability score_v1 (AQC-504).

    score_v1 = median(OOS_daily_return)
             - 2.0 * max(OOS_drawdown_pct)
             - 0.5 * slippage_fragility
             - penalty_if_trades_too_low

    Notes:
    - OOS metrics come from walk-forward validation (AQC-501).
    - slippage_fragility is a balance-normalised PnL drop from slippage stress (AQC-502).
    """

    # Require walk-forward metrics to avoid comparing incomparable candidates.
    if "wf_median_oos_daily_return" not in it or "wf_max_oos_drawdown_pct" not in it:
        return None
    # Require slippage fragility (otherwise the score is optimistic).
    if "slippage_fragility" not in it:
        return None

    try:
        oos_daily_ret = float(it.get("wf_median_oos_daily_return", 0.0))
        oos_max_dd = float(it.get("wf_max_oos_drawdown_pct", 0.0))
        slip_frag = float(it.get("slippage_fragility", 0.0))
        trades = int(it.get("total_trades", 0))
    except Exception:
        return None

    min_trades = int(min_trades)
    if min_trades < 0:
        min_trades = 0
    trades_penalty_weight = float(trades_penalty_weight)
    if trades_penalty_weight < 0:
        trades_penalty_weight = 0.0

    deficit = max(0, int(min_trades) - int(trades))
    trades_penalty = (deficit / float(max(1, min_trades))) * trades_penalty_weight if min_trades > 0 else 0.0

    dd_weight = 2.0
    slippage_weight = 0.5
    score = float(oos_daily_ret) - dd_weight * float(oos_max_dd) - slippage_weight * float(slip_frag) - float(trades_penalty)

    return {
        "version": "score_v1",
        "score": float(score),
        "components": {
            "median_oos_daily_return": float(oos_daily_ret),
            "max_oos_drawdown_pct": float(oos_max_dd),
            "slippage_fragility": float(slip_frag),
            "total_trades": int(trades),
            "min_trades": int(min_trades),
            "trades_penalty": float(trades_penalty),
            "weights": {
                "drawdown": float(dd_weight),
                "slippage": float(slippage_weight),
                "trades_penalty_weight": float(trades_penalty_weight),
            },
        },
    }


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

    kept = [it for it in items if not bool(it.get("rejected"))]
    rejected = [it for it in items if bool(it.get("rejected"))]

    if not kept:
        lines.append("No non-rejected replay reports were produced.")
        lines.append("")
    else:
        scored = [it for it in kept if isinstance(it.get("score_v1"), (int, float))]
        unscored = [it for it in kept if not isinstance(it.get("score_v1"), (int, float))]
        if scored:
            items_sorted = sorted(scored, key=lambda x: float(x.get("score_v1", float("-inf"))), reverse=True)

            lines.append("## Ranked Candidates (by score_v1, excluding rejected)")
            lines.append("")
            lines.append(
                "| Rank | score_v1 | oos_med_daily_ret | oos_max_dd_pct | slip_frag | trades_pen | trades | total_pnl | config_id | replay |"
            )
            lines.append(
                "| ---: | -------: | ---------------: | ------------: | --------: | --------: | -----: | --------: | :------ | :----- |"
            )
            for i, it in enumerate(items_sorted, start=1):
                cfg_id = str(it.get("config_id", ""))
                cfg_id_short = cfg_id[:12] if len(cfg_id) > 12 else cfg_id
                comps = it.get("score_v1_components", {})
                trades_pen = float(comps.get("trades_penalty", 0.0)) if isinstance(comps, dict) else 0.0
                lines.append(
                    "| {rank} | {score:.6f} | {oos:.6f} | {dd:.4f} | {slip:.6f} | {tpen:.6f} | {trades} | {pnl:.2f} | `{cfg_id}` | `{path}` |".format(
                        rank=i,
                        score=float(it.get("score_v1", 0.0)),
                        oos=float(it.get("wf_median_oos_daily_return", 0.0)),
                        dd=float(it.get("wf_max_oos_drawdown_pct", 0.0)),
                        slip=float(it.get("slippage_fragility", 0.0)),
                        tpen=float(trades_pen),
                        trades=int(it.get("total_trades", 0)),
                        pnl=float(it.get("total_pnl", 0.0)),
                        cfg_id=cfg_id_short,
                        path=str(it.get("path", "")),
                    )
                )
            lines.append("")
            if unscored:
                lines.append(f"Note: {len(unscored)} candidate(s) had no score_v1 (missing validation artefacts).")
                lines.append("")
        else:
            items_sorted = sorted(kept, key=lambda x: float(x.get("total_pnl", 0.0)), reverse=True)

            lines.append("## Ranked Candidates (by total_pnl, excluding rejected)")
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

    if rejected:
        lines.append("## Rejected Candidates")
        lines.append("")
        lines.append(
            "| total_pnl | config_id | reason | replay | slippage_fragility | reject_bps | pnl_drop_reject_bps | pnl_reject_bps |"
        )
        lines.append("| --------: | :------ | :----- | :----- | -----------------: | ---------: | ------------------: | ------------: |")
        rej_sorted = sorted(rejected, key=lambda x: float(x.get("total_pnl", 0.0)), reverse=True)
        for it in rej_sorted:
            cfg_id = str(it.get("config_id", ""))
            cfg_id_short = cfg_id[:12] if len(cfg_id) > 12 else cfg_id
            reason = str(it.get("reject_reason", "rejected"))
            lines.append(
                "| {pnl:.2f} | `{cfg}` | {reason} | `{path}` | {frag:.6f} | {rbps:.0f} | {drop:.2f} | {pnl_r:.2f} |".format(
                    pnl=float(it.get("total_pnl", 0.0)),
                    cfg=cfg_id_short,
                    reason=reason,
                    path=str(it.get("path", "")),
                    frag=float(it.get("slippage_fragility", 0.0)),
                    rbps=float(it.get("slippage_reject_bps", 0.0)),
                    drop=float(it.get("pnl_drop_at_reject_bps", 0.0)),
                    pnl_r=float(it.get("slippage_pnl_at_reject_bps", 0.0)),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _render_validation_report_md(items: list[dict[str, Any]], *, score_min_trades: int) -> str:
    lines: list[str] = []
    lines.append("# Validation Report")
    lines.append("")
    lines.append("This report lists per-candidate validation signals and human-readable reasons.")
    lines.append("")
    if not items:
        lines.append("No candidates were produced.")
        lines.append("")
        return "\n".join(lines)

    def _has_num(v: Any) -> bool:
        return isinstance(v, (int, float))

    def _sort_key(it: dict[str, Any]) -> tuple[int, float]:
        if _has_num(it.get("score_v1")):
            return (2, float(it.get("score_v1", 0.0)))
        return (1, float(it.get("total_pnl", 0.0)))

    def _reasons(it: dict[str, Any]) -> list[str]:
        out: list[str] = []
        rr = str(it.get("reject_reason", "") or "").strip()
        if rr:
            out.append(rr)

        try:
            if "wf_median_oos_daily_return" in it and float(it.get("wf_median_oos_daily_return", 0.0)) < 0.0:
                out.append("OOS negative (median daily return < 0)")
        except Exception:
            pass

        try:
            if int(it.get("total_trades", 0)) < int(score_min_trades):
                out.append(f"too few trades (trades < {int(score_min_trades)})")
        except Exception:
            pass

        if not _has_num(it.get("score_v1")):
            out.append("missing score_v1 (enable --walk-forward and --slippage-stress)")

        # Deduplicate while preserving order.
        seen: set[str] = set()
        dedup: list[str] = []
        for r in out:
            if r in seen:
                continue
            seen.add(r)
            dedup.append(r)
        return dedup

    items_sorted = sorted(items, key=_sort_key, reverse=True)

    for i, it in enumerate(items_sorted, start=1):
        cfg_id = str(it.get("config_id", "")).strip()
        cfg_id_short = cfg_id[:12] if len(cfg_id) > 12 else cfg_id
        status = "REJECT" if bool(it.get("rejected")) else "ACCEPT"

        lines.append(f"## Candidate {i}: `{cfg_id_short}`")
        lines.append("")
        lines.append(f"- Status: {status}")
        if _has_num(it.get("score_v1")):
            lines.append(f"- score_v1: {float(it.get('score_v1', 0.0)):.6f}")
        else:
            lines.append("- score_v1: n/a")
        lines.append(f"- total_pnl: {float(it.get('total_pnl', 0.0)):.2f}")
        lines.append(f"- max_drawdown_pct: {float(it.get('max_drawdown_pct', 0.0)):.4f}")
        lines.append(f"- trades: {int(it.get('total_trades', 0))}")
        if _has_num(it.get("wf_median_oos_daily_return")):
            lines.append(f"- OOS median daily return: {float(it.get('wf_median_oos_daily_return', 0.0)):.6f}")
        if _has_num(it.get("wf_max_oos_drawdown_pct")):
            lines.append(f"- OOS max drawdown pct: {float(it.get('wf_max_oos_drawdown_pct', 0.0)):.4f}")
        if _has_num(it.get("slippage_fragility")):
            lines.append(f"- slippage fragility: {float(it.get('slippage_fragility', 0.0)):.6f}")
        if _has_num(it.get("top1_pnl_pct")):
            lines.append(f"- top1 pnl pct: {float(it.get('top1_pnl_pct', 0.0)):.4f}")
        if _has_num(it.get("top5_pnl_pct")):
            lines.append(f"- top5 pnl pct: {float(it.get('top5_pnl_pct', 0.0)):.4f}")
        if _has_num(it.get("long_pnl_usd")) or _has_num(it.get("short_pnl_usd")):
            lines.append(
                f"- long/short pnl: {float(it.get('long_pnl_usd', 0.0)):.2f} / {float(it.get('short_pnl_usd', 0.0)):.2f}"
            )
        rs = _reasons(it)
        lines.append(f"- Reasons: {'; '.join(rs) if rs else 'none'}")
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
    entry_by_path: dict[str, dict[str, Any]] = {}

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
        entry = {"path": str(dst), "config_id": dst_cfg_id, "source_path": str(src)}
        copied_candidates.append(entry)
        entry_by_path[str(dst)] = entry

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
        _write_json(run_dir / "run_metadata.json", meta)

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

    validation_md = _render_validation_report_md(
        replay_reports, score_min_trades=int(getattr(args, "score_min_trades", 30) or 30)
    )
    (run_dir / "reports" / "validation_report.md").write_text(validation_md, encoding="utf-8")

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
    args = _parse_cli_args(argv)

    artifacts_root = (AIQ_ROOT / str(args.artifacts_dir)).resolve()
    if args.reproduce and args.resume:
        raise SystemExit("--resume cannot be used with --reproduce")
    if args.reproduce:
        return _reproduce_run(artifacts_root=artifacts_root, source_run_id=str(args.reproduce))

    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("--run-id cannot be empty")

    existing_run_dir: Path | None = None
    try:
        existing_run_dir = _find_existing_run_dir(artifacts_root=artifacts_root, run_id=run_id)
    except FileNotFoundError:
        existing_run_dir = None

    now_ms = int(time.time() * 1000)

    if existing_run_dir is not None:
        if not bool(args.resume):
            raise SystemExit(f"run_id already exists; use --resume to continue: {run_id}")
        run_dir = existing_run_dir
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "run_metadata.json"
        meta_obj = _load_json(meta_path) if meta_path.exists() else {}
        meta = meta_obj if isinstance(meta_obj, dict) else {}

        # Safety: do not allow resuming with different core parameters.
        orig_args = meta.get("args", {}) if isinstance(meta.get("args", {}), dict) else {}
        guard_keys = {
            "config",
            "interval",
            "candles_db",
            "funding_db",
            "sweep_spec",
            "gpu",
            "concentration_checks",
            "conc_max_top1_pnl_pct",
            "conc_max_top5_pnl_pct",
            "conc_min_symbols_traded",
            "walk_forward",
            "walk_forward_splits_json",
            "walk_forward_min_test_days",
            "slippage_stress",
            "slippage_stress_bps",
            "slippage_stress_reject_bps",
            "score_min_trades",
            "score_trades_penalty_weight",
            "tpe",
            "tpe_trials",
            "tpe_batch",
            "tpe_seed",
            "top_n",
            "num_candidates",
            "sort_by",
            "shortlist_modes",
            "shortlist_per_mode",
            "shortlist_max_rank",
        }
        mismatches: list[str] = []
        for k in sorted(guard_keys):
            if k not in orig_args:
                continue
            if getattr(args, k, None) != orig_args.get(k):
                mismatches.append(k)
        if mismatches:
            raise SystemExit(f"--resume args mismatch for keys: {', '.join(mismatches)}")

        # Use original args when available to keep the run reproducible.
        for k, v in orig_args.items():
            if k == "resume":
                continue
            if hasattr(args, k):
                setattr(args, k, v)

        meta.setdefault("run_id", run_id)
        meta.setdefault("run_dir", str(run_dir))
        meta.setdefault("steps", [])
        meta["resumed_at_ms"] = now_ms
        meta["resumed_git_head"] = _git_head_sha()
    else:
        generated_at_ms = now_ms
        run_dir = resolve_run_dir(artifacts_root=artifacts_root, run_id=run_id, generated_at_ms=generated_at_ms)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        args_meta = dict(vars(args))
        args_meta.pop("resume", None)
        meta = {
            "run_id": run_id,
            "generated_at_ms": generated_at_ms,
            "run_date_utc": _run_date_utc(generated_at_ms),
            "run_dir": str(run_dir),
            "git_head": _git_head_sha(),
            "args": args_meta,
            "steps": [],
        }

    # Resolve backtester cmd once and persist early metadata.
    bt_cmd = _resolve_backtester_cmd()
    if "repro" not in meta:
        _capture_repro_metadata(run_dir=run_dir, artifacts_root=artifacts_root, bt_cmd=bt_cmd, meta=meta)
    _write_json(run_dir / "run_metadata.json", meta)

    # ------------------------------------------------------------------
    # 1) Data checks
    # ------------------------------------------------------------------
    candle_out = run_dir / "data_checks" / "candle_dbs.json"
    candle_err = run_dir / "data_checks" / "candle_dbs.stderr.txt"
    if bool(args.resume) and _is_nonempty_file(candle_out):
        meta["steps"].append(
            {
                "name": "check_candle_dbs_skip",
                "argv": [],
                "cwd": str(AIQ_ROOT),
                "exit_code": 0,
                "elapsed_s": 0.0,
                "stdout_path": str(candle_out),
                "stderr_path": str(candle_err),
            }
        )
    else:
        candle_check = _run_cmd(
            ["python3", "tools/check_candle_dbs.py", "--json-indent", "2"],
            cwd=AIQ_ROOT,
            stdout_path=candle_out,
            stderr_path=candle_err,
        )
        meta["steps"].append({"name": "check_candle_dbs", **candle_check.__dict__})
        if candle_check.exit_code != 0:
            _write_json(run_dir / "run_metadata.json", meta)
            return int(candle_check.exit_code)
    _write_json(run_dir / "run_metadata.json", meta)

    funding_out = run_dir / "data_checks" / "funding_rates.json"
    funding_err = run_dir / "data_checks" / "funding_rates.stderr.txt"
    if bool(args.resume) and _is_nonempty_file(funding_out):
        meta["steps"].append(
            {
                "name": "check_funding_rates_db_skip",
                "argv": [],
                "cwd": str(AIQ_ROOT),
                "exit_code": 0,
                "elapsed_s": 0.0,
                "stdout_path": str(funding_out),
                "stderr_path": str(funding_err),
            }
        )
    else:
        funding_check = _run_cmd(
            ["python3", "tools/check_funding_rates_db.py"],
            cwd=AIQ_ROOT,
            stdout_path=funding_out,
            stderr_path=funding_err,
        )
        meta["steps"].append({"name": "check_funding_rates_db", **funding_check.__dict__})
        if funding_check.exit_code != 0:
            _write_json(run_dir / "run_metadata.json", meta)
            return int(funding_check.exit_code)
    _write_json(run_dir / "run_metadata.json", meta)

    # ------------------------------------------------------------------
    # 2) Sweep
    # ------------------------------------------------------------------
    sweep_out = run_dir / "sweeps" / "sweep_results.jsonl"
    if bool(args.resume) and _is_nonempty_file(sweep_out):
        sweep_res = CmdResult(
            argv=[],
            cwd=str(AIQ_ROOT / "backtester"),
            exit_code=0,
            elapsed_s=0.0,
            stdout_path=str(run_dir / "sweeps" / "sweep.stdout.txt"),
            stderr_path=str(run_dir / "sweeps" / "sweep.stderr.txt"),
        )
        meta["steps"].append({"name": "sweep_skip", **sweep_res.__dict__})
    else:
        if bool(args.gpu):
            rc = _ensure_gpu_idle_or_exit(
                run_dir=run_dir,
                meta=meta,
                wait_s=int(getattr(args, "gpu_wait_s", 0) or 0),
                poll_s=int(getattr(args, "gpu_poll_s", 0) or 0),
            )
            if rc != 0:
                return int(rc)

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

    _write_json(run_dir / "run_metadata.json", meta)

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
    skip_generation = False

    if bool(args.resume):
        existing_entries = meta.get("candidate_configs", [])
        if isinstance(existing_entries, list) and existing_entries:
            candidate_entries = [it for it in existing_entries if isinstance(it, dict)]
            for it in candidate_entries:
                p = str(it.get("path", "")).strip()
                if p:
                    entry_by_path[p] = it

            for it in candidate_entries:
                p_raw = str(it.get("path", "")).strip()
                if not p_raw:
                    continue
                if "selected" in it and not bool(it.get("selected")):
                    continue
                p = Path(p_raw).expanduser().resolve()
                if not _is_nonempty_file(p):
                    continue
                candidate_paths.append(p)
                cid = str(it.get("config_id", "")).strip()
                if not cid:
                    cid = config_id_from_yaml_file(p)
                    it["config_id"] = cid
                candidate_config_ids[str(p)] = cid

            skip_generation = bool(candidate_paths)

    # Multi-mode shortlist generation (AQC-403). Deduplicate across modes by config_id.
    if not skip_generation and shortlist_per_mode > 0:
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
    elif not skip_generation:
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
    _write_json(run_dir / "run_metadata.json", meta)

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

        replay_stdout = run_dir / "replays" / f"{cfg_path.stem}.stdout.txt"
        replay_stderr = run_dir / "replays" / f"{cfg_path.stem}.stderr.txt"

        if bool(args.resume) and _is_nonempty_file(out_json):
            replay_res = CmdResult(
                argv=[],
                cwd=str(AIQ_ROOT / "backtester"),
                exit_code=0,
                elapsed_s=0.0,
                stdout_path=str(replay_stdout),
                stderr_path=str(replay_stderr),
            )
            meta["steps"].append({"name": f"replay_{cfg_path.stem}_skip", **replay_res.__dict__})
        else:
            replay_res = _run_cmd(
                replay_argv,
                cwd=AIQ_ROOT / "backtester",
                stdout_path=replay_stdout,
                stderr_path=replay_stderr,
            )
            meta["steps"].append({"name": f"replay_{cfg_path.stem}", **replay_res.__dict__})
            if replay_res.exit_code != 0:
                _write_json(run_dir / "run_metadata.json", meta)
                return int(replay_res.exit_code)

        summary = _summarise_replay_report(out_json)
        summary["config_path"] = str(cfg_path)
        summary["config_id"] = candidate_config_ids[str(cfg_path)]

        if bool(getattr(args, "concentration_checks", False)):
            cc_dir = run_dir / "concentration" / str(cfg_path.stem)
            cc_dir.mkdir(parents=True, exist_ok=True)
            cc_summary_path = cc_dir / "summary.json"

            cc_stdout = cc_dir / "concentration.stdout.txt"
            cc_stderr = cc_dir / "concentration.stderr.txt"

            if bool(args.resume) and _is_nonempty_file(cc_summary_path):
                cc_res = CmdResult(
                    argv=[],
                    cwd=str(AIQ_ROOT),
                    exit_code=0,
                    elapsed_s=0.0,
                    stdout_path=str(cc_stdout),
                    stderr_path=str(cc_stderr),
                )
                meta["steps"].append({"name": f"concentration_{cfg_path.stem}_skip", **cc_res.__dict__})
            else:
                cc_argv = [
                    "python3",
                    "tools/concentration_checks.py",
                    "--replay-report",
                    str(out_json),
                    "--output",
                    str(cc_summary_path),
                    "--max-top1-pnl-pct",
                    str(float(getattr(args, "conc_max_top1_pnl_pct", 0.65) or 0.65)),
                    "--max-top5-pnl-pct",
                    str(float(getattr(args, "conc_max_top5_pnl_pct", 0.90) or 0.90)),
                    "--min-symbols-traded",
                    str(int(getattr(args, "conc_min_symbols_traded", 5) or 5)),
                ]

                cc_res = _run_cmd(cc_argv, cwd=AIQ_ROOT, stdout_path=cc_stdout, stderr_path=cc_stderr)
                meta["steps"].append({"name": f"concentration_{cfg_path.stem}", **cc_res.__dict__})
                if cc_res.exit_code != 0:
                    _write_json(run_dir / "run_metadata.json", meta)
                    return int(cc_res.exit_code)

            summary["concentration_summary_path"] = str(cc_summary_path)
            try:
                cc_obj = _load_json(cc_summary_path)
                summary["concentration_reject"] = bool(cc_obj.get("reject", False)) if isinstance(cc_obj, dict) else False
                metrics = cc_obj.get("metrics", {}) if isinstance(cc_obj, dict) else {}
                if isinstance(metrics, dict):
                    summary["symbols_traded"] = int(metrics.get("symbols_traded", 0) or 0)
                    summary["top1_pnl_pct"] = float(metrics.get("top1_pnl_pct", 0.0) or 0.0)
                    summary["top5_pnl_pct"] = float(metrics.get("top5_pnl_pct", 0.0) or 0.0)
                    summary["long_pnl_usd"] = float(metrics.get("long_pnl_usd", 0.0) or 0.0)
                    summary["short_pnl_usd"] = float(metrics.get("short_pnl_usd", 0.0) or 0.0)

                if bool(summary.get("concentration_reject")):
                    reasons = cc_obj.get("reject_reasons", []) if isinstance(cc_obj, dict) else []
                    if isinstance(reasons, list) and reasons:
                        _append_reject_reason(summary, "concentration: " + ", ".join([str(r) for r in reasons]))
                    else:
                        _append_reject_reason(summary, "concentration")
            except Exception:
                summary["concentration_error"] = "failed to load concentration summary.json"

        if bool(getattr(args, "walk_forward", False)):
            wf_dir = run_dir / "walk_forward" / str(cfg_path.stem)
            wf_dir.mkdir(parents=True, exist_ok=True)
            wf_summary_path = wf_dir / "summary.json"

            wf_stdout = wf_dir / "walk_forward.stdout.txt"
            wf_stderr = wf_dir / "walk_forward.stderr.txt"

            if bool(args.resume) and _is_nonempty_file(wf_summary_path):
                wf_res = CmdResult(
                    argv=[],
                    cwd=str(AIQ_ROOT),
                    exit_code=0,
                    elapsed_s=0.0,
                    stdout_path=str(wf_stdout),
                    stderr_path=str(wf_stderr),
                )
                meta["steps"].append({"name": f"walk_forward_{cfg_path.stem}_skip", **wf_res.__dict__})
            else:
                wf_argv = [
                    "python3",
                    "tools/walk_forward_validate.py",
                    "--config",
                    str(cfg_path),
                    "--interval",
                    str(args.interval),
                    "--min-test-days",
                    str(int(getattr(args, "walk_forward_min_test_days", 1) or 1)),
                    "--out-dir",
                    str(wf_dir),
                    "--output",
                    str(wf_summary_path),
                ]
                if getattr(args, "walk_forward_splits_json", None):
                    wf_argv += ["--splits-json", str(args.walk_forward_splits_json)]
                if args.candles_db:
                    wf_argv += ["--candles-db", str(args.candles_db)]
                if args.funding_db:
                    wf_argv += ["--funding-db", str(args.funding_db)]

                wf_res = _run_cmd(wf_argv, cwd=AIQ_ROOT, stdout_path=wf_stdout, stderr_path=wf_stderr)
                meta["steps"].append({"name": f"walk_forward_{cfg_path.stem}", **wf_res.__dict__})
                if wf_res.exit_code != 0:
                    _write_json(run_dir / "run_metadata.json", meta)
                    return int(wf_res.exit_code)

            summary["walk_forward_summary_path"] = str(wf_summary_path)
            try:
                wf_obj = _load_json(wf_summary_path)
                agg = wf_obj.get("aggregate", {}) if isinstance(wf_obj, dict) else {}
                if isinstance(agg, dict):
                    summary["wf_median_oos_daily_return"] = float(agg.get("median_oos_daily_return", 0.0))
                    summary["wf_max_oos_drawdown_pct"] = float(agg.get("max_oos_drawdown_pct", 0.0))
                    summary["wf_walk_forward_score_v1"] = float(agg.get("walk_forward_score_v1", 0.0))
            except Exception:
                # Keep the factory pipeline resilient: store the path and continue.
                summary["wf_error"] = "failed to load walk_forward summary.json"

        if bool(getattr(args, "slippage_stress", False)):
            ss_dir = run_dir / "slippage_stress" / str(cfg_path.stem)
            ss_dir.mkdir(parents=True, exist_ok=True)
            ss_summary_path = ss_dir / "summary.json"

            ss_stdout = ss_dir / "slippage_stress.stdout.txt"
            ss_stderr = ss_dir / "slippage_stress.stderr.txt"

            if bool(args.resume) and _is_nonempty_file(ss_summary_path):
                ss_res = CmdResult(
                    argv=[],
                    cwd=str(AIQ_ROOT),
                    exit_code=0,
                    elapsed_s=0.0,
                    stdout_path=str(ss_stdout),
                    stderr_path=str(ss_stderr),
                )
                meta["steps"].append({"name": f"slippage_stress_{cfg_path.stem}_skip", **ss_res.__dict__})
            else:
                ss_argv = [
                    "python3",
                    "tools/slippage_stress.py",
                    "--config",
                    str(cfg_path),
                    "--interval",
                    str(args.interval),
                    "--slippage-bps",
                    str(getattr(args, "slippage_stress_bps", "10,20,30") or "10,20,30"),
                    "--reject-flip-bps",
                    str(float(getattr(args, "slippage_stress_reject_bps", 20.0) or 20.0)),
                    "--out-dir",
                    str(ss_dir),
                    "--output",
                    str(ss_summary_path),
                ]
                if args.candles_db:
                    ss_argv += ["--candles-db", str(args.candles_db)]
                if args.funding_db:
                    ss_argv += ["--funding-db", str(args.funding_db)]

                ss_res = _run_cmd(ss_argv, cwd=AIQ_ROOT, stdout_path=ss_stdout, stderr_path=ss_stderr)
                meta["steps"].append({"name": f"slippage_stress_{cfg_path.stem}", **ss_res.__dict__})
                if ss_res.exit_code != 0:
                    _write_json(run_dir / "run_metadata.json", meta)
                    return int(ss_res.exit_code)

            summary["slippage_stress_summary_path"] = str(ss_summary_path)
            try:
                ss_obj = _load_json(ss_summary_path)
                agg = ss_obj.get("aggregate", {}) if isinstance(ss_obj, dict) else {}
                if isinstance(agg, dict):
                    reject_bps = float(getattr(args, "slippage_stress_reject_bps", 20.0) or 20.0)
                    summary["slippage_fragility"] = float(agg.get("slippage_fragility", 0.0))
                    summary["slippage_flip_sign_at_reject_bps"] = bool(agg.get("flip_sign_at_reject_bps", False))
                    summary["slippage_reject"] = bool(agg.get("reject", False))
                    summary["slippage_reject_bps"] = float(reject_bps)
                    summary["pnl_drop_at_reject_bps"] = float(agg.get("pnl_drop_at_reject_bps", 0.0))
                    summary["slippage_pnl_at_reject_bps"] = float(agg.get("pnl_at_reject_bps", 0.0))

                    # Canonical key for AQC-504 scoring (when using the default 20 bps).
                    if abs(float(reject_bps) - 20.0) < 1e-9:
                        summary["pnl_drop_when_slippage_20bps"] = float(summary.get("pnl_drop_at_reject_bps", 0.0))
                    if bool(summary.get("slippage_reject")):
                        _append_reject_reason(summary, f"slippage flip at {float(reject_bps):g} bps")
            except Exception:
                summary["slippage_error"] = "failed to load slippage_stress summary.json"

        score_obj = _compute_score_v1(
            summary,
            min_trades=int(getattr(args, "score_min_trades", 30) or 30),
            trades_penalty_weight=float(getattr(args, "score_trades_penalty_weight", 0.05) or 0.05),
        )
        if score_obj is not None:
            summary["score_v1"] = float(score_obj.get("score", 0.0))
            summary["score_v1_components"] = score_obj.get("components", {})

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


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    profile = str(getattr(args, "profile", "") or "daily").strip()
    defaults = PROFILE_DEFAULTS.get(profile)
    if defaults is None:
        raise SystemExit(f"Unknown --profile: {profile}")

    for k, v in defaults.items():
        if not hasattr(args, k):
            continue
        if getattr(args, k) is None:
            setattr(args, k, int(v))


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the nightly strategy factory workflow and store artifacts.")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--run-id", help="Unique identifier for this run (used in artifact paths).")
    mx.add_argument("--reproduce", metavar="RUN_ID", help="Reproduce an existing run_id (CPU replay + reports only).")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts).")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run_id from artifacts.")

    ap.add_argument(
        "--profile",
        default="daily",
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Preset defaults for trials and candidate counts (default: daily).",
    )

    ap.add_argument("--config", default="config/strategy_overrides.yaml", help="Base strategy YAML config path.")
    ap.add_argument("--interval", default="1h", help="Main interval for sweep/replay (default: 1h).")
    ap.add_argument("--candles-db", default=None, help="Optional candle DB path override.")
    ap.add_argument("--funding-db", default=None, help="Optional funding DB path for replay/sweep.")

    ap.add_argument("--sweep-spec", default="backtester/sweeps/smoke.yaml", help="Sweep spec YAML path.")
    ap.add_argument("--gpu", action="store_true", help="Use GPU sweep (requires CUDA build/runtime).")
    ap.add_argument(
        "--gpu-wait-s",
        type=int,
        default=0,
        help="Wait up to N seconds for an idle GPU before exiting (default: 0).",
    )
    ap.add_argument(
        "--gpu-poll-s",
        type=int,
        default=10,
        help="Polling interval in seconds while waiting for GPU idle (default: 10).",
    )
    ap.add_argument("--tpe", action="store_true", help="Use TPE Bayesian optimisation for GPU sweeps (requires --gpu).")
    ap.add_argument("--tpe-trials", type=int, default=None, help="Number of TPE trials (default depends on --profile).")
    ap.add_argument("--tpe-batch", type=int, default=256, help="Trials per GPU batch (default: 256).")
    ap.add_argument("--tpe-seed", type=int, default=42, help="RNG seed for TPE reproducibility (default: 42).")
    ap.add_argument("--top-n", type=int, default=0, help="Only print top N sweep results (0 = no summary).")

    ap.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="How many candidate configs to generate when shortlist is disabled (default depends on --profile).",
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
        default=None,
        help="How many unique configs to keep per mode (default depends on --profile). Set 0 to disable shortlist.",
    )
    ap.add_argument(
        "--shortlist-max-rank",
        type=int,
        default=None,
        help="Max rank to scan per mode while deduplicating (default depends on --profile).",
    )

    ap.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation for each candidate.")
    ap.add_argument("--walk-forward-splits-json", default=None, help="Optional JSON file defining walk-forward splits.")
    ap.add_argument(
        "--walk-forward-min-test-days",
        type=int,
        default=1,
        help="Minimum out-of-sample test length in days for walk-forward splits (default: 1).",
    )

    ap.add_argument("--slippage-stress", action="store_true", help="Run slippage stress replays for each candidate.")
    ap.add_argument(
        "--slippage-stress-bps",
        default="10,20,30",
        help="Comma-separated slippage bps levels for stress tests (default: 10,20,30).",
    )
    ap.add_argument(
        "--slippage-stress-reject-bps",
        type=float,
        default=20.0,
        help="Reject when PnL flips sign at this slippage level (default: 20).",
    )

    ap.add_argument(
        "--concentration-checks",
        action="store_true",
        help="Run concentration/diversification checks from replay output (per-symbol breakdown).",
    )
    ap.add_argument(
        "--conc-max-top1-pnl-pct",
        type=float,
        default=0.65,
        help="Reject if top-1 symbol contributes more than this fraction of positive PnL (default: 0.65).",
    )
    ap.add_argument(
        "--conc-max-top5-pnl-pct",
        type=float,
        default=0.90,
        help="Reject if top-5 symbols contribute more than this fraction of positive PnL (default: 0.90).",
    )
    ap.add_argument(
        "--conc-min-symbols-traded",
        type=int,
        default=5,
        help="Reject if fewer than this many symbols had at least one trade (default: 5).",
    )

    ap.add_argument(
        "--score-min-trades",
        type=int,
        default=30,
        help="Minimum trade count used by score_v1 trades penalty (default: 30).",
    )
    ap.add_argument(
        "--score-trades-penalty-weight",
        type=float,
        default=0.05,
        help="Maximum score_v1 trades penalty when trades=0 (default: 0.05).",
    )

    return ap


def _parse_cli_args(argv: list[str] | None) -> argparse.Namespace:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    _apply_profile_defaults(args)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
