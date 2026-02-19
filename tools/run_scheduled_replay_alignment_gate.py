#!/usr/bin/env python3
"""Run scheduled replay-bundle alignment checks and maintain a release blocker.

This orchestrator builds a deterministic replay bundle for a recent live window,
runs the full paper deterministic replay harness, and writes:

- a per-run report JSON
- a release-blocker JSON (blocked when the gate fails)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
from typing import Any

_STRATEGY_SHA1_RE = re.compile(r"strategy_sha1=([0-9a-fA-F]{7,40})")


def _now_ms() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _resolve_path(raw: str, *, base: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _read_live_baseline_count(manifest_path: Path) -> int:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return -1
    if not isinstance(payload, dict):
        return -1
    counts = payload.get("counts") or {}
    if not isinstance(counts, dict):
        return -1
    try:
        return int(counts.get("live_baseline_trades") or 0)
    except Exception:
        return -1


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_to_log(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"$ {' '.join(shlex.quote(x) for x in cmd)}\n")
        fp.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(proc.returncode)


def _db_has_table(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (str(table_name),),
            ).fetchone()
            return bool(row is not None)
        finally:
            conn.close()
    except Exception:
        return False


def _candles_interval_coverage_ms(candles_db: Path, interval: str) -> tuple[int | None, int | None]:
    if not candles_db.exists():
        return None, None
    try:
        conn = sqlite3.connect(f"file:{candles_db}?mode=ro", uri=True, timeout=10)
        try:
            row = conn.execute(
                "SELECT MIN(t), MAX(t) FROM candles WHERE interval = ?",
                (str(interval),),
            ).fetchone()
            if not row:
                return None, None
            lo_raw, hi_raw = row[0], row[1]
            if lo_raw is None or hi_raw is None:
                return None, None
            return int(lo_raw), int(hi_raw)
        finally:
            conn.close()
    except Exception:
        return None, None


def _runtime_strategy_window_stats(live_db: Path, *, from_ts: int, to_ts: int) -> dict[str, Any]:
    if not live_db.exists():
        return {
            "sampled_rows": 0,
            "strategy_sha1_distinct": 0,
            "last_segment_from_ts": None,
            "last_segment_to_ts": None,
            "last_segment_sha1": None,
            "timeline": [],
        }
    try:
        conn = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT ts_ms, message FROM runtime_logs WHERE ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms ASC, id ASC",
                (int(from_ts), int(to_ts)),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return {
            "sampled_rows": 0,
            "strategy_sha1_distinct": 0,
            "last_segment_from_ts": None,
            "last_segment_to_ts": None,
            "last_segment_sha1": None,
            "timeline": [],
        }

    timeline: list[dict[str, Any]] = []
    sampled_rows = 0
    distinct_sha: set[str] = set()

    for row in rows:
        ts_ms = int(row["ts_ms"] or 0)
        message = str(row["message"] or "")
        if ts_ms <= 0 or not message:
            continue
        m = _STRATEGY_SHA1_RE.search(message)
        if m is None:
            continue
        sha = str(m.group(1) or "").strip().lower()
        if not sha:
            continue
        sampled_rows += 1
        distinct_sha.add(sha)

        if timeline and str(timeline[-1]["strategy_sha1"]) == sha:
            timeline[-1]["last_ts_ms"] = ts_ms
            timeline[-1]["sample_count"] = int(timeline[-1]["sample_count"]) + 1
        else:
            timeline.append(
                {
                    "strategy_sha1": sha,
                    "first_ts_ms": ts_ms,
                    "last_ts_ms": ts_ms,
                    "sample_count": 1,
                }
            )

    last_segment = timeline[-1] if timeline else None
    return {
        "sampled_rows": int(sampled_rows),
        "strategy_sha1_distinct": len(distinct_sha),
        "last_segment_from_ts": int(last_segment["first_ts_ms"]) if last_segment else None,
        "last_segment_to_ts": int(last_segment["last_ts_ms"]) if last_segment else None,
        "last_segment_sha1": str(last_segment["strategy_sha1"]) if last_segment else None,
        "timeline": timeline,
    }


def _ts_ms_to_iso_utc(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=dt.timezone.utc).isoformat()


def _count_live_trades_in_window(live_db: Path, *, from_ts: int, to_ts: int) -> int:
    if not live_db.exists() or from_ts > to_ts:
        return 0
    from_iso = _ts_ms_to_iso_utc(int(from_ts))
    to_iso = _ts_ms_to_iso_utc(int(to_ts))
    try:
        conn = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True, timeout=10)
        try:
            row = conn.execute(
                "SELECT COUNT(1) FROM trades WHERE timestamp >= ? AND timestamp <= ?",
                (from_iso, to_iso),
            ).fetchone()
            if not row:
                return 0
            return int(row[0] or 0)
        finally:
            conn.close()
    except Exception:
        return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    ap = argparse.ArgumentParser(description="Run scheduled replay alignment gate checks.")
    ap.add_argument("--repo-root", default=str(repo_root), help="Repo root path.")
    ap.add_argument(
        "--live-db",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_LIVE_DB", "./trading_engine_live.db") or "./trading_engine_live.db"),
        help="Live SQLite DB path.",
    )
    ap.add_argument(
        "--paper-db",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_PAPER_DB", "./trading_engine.db") or "./trading_engine.db"),
        help="Paper SQLite DB path.",
    )
    ap.add_argument(
        "--candles-db",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_CANDLES_DB", "") or ""),
        help="Candles SQLite DB path (required unless set via AI_QUANT_REPLAY_GATE_CANDLES_DB).",
    )
    ap.add_argument(
        "--funding-db",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_FUNDING_DB", "") or ""),
        help="Optional funding SQLite DB path.",
    )
    ap.add_argument(
        "--interval",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_INTERVAL", "1h") or "1h"),
        help="Replay interval.",
    )
    ap.add_argument(
        "--window-minutes",
        type=int,
        default=_env_int("AI_QUANT_REPLAY_GATE_WINDOW_MINUTES", 240),
        help="Replay window size in minutes (default: 240).",
    )
    ap.add_argument(
        "--lag-minutes",
        type=int,
        default=_env_int("AI_QUANT_REPLAY_GATE_LAG_MINUTES", 2),
        help="Tail lag in minutes before now for replay end timestamp (default: 2).",
    )
    ap.add_argument(
        "--bundle-root",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_BUNDLE_ROOT", "/tmp/openclaw-ai-quant/replay_gate") or "/tmp/openclaw-ai-quant/replay_gate"),
        help="Root directory for scheduled replay bundles.",
    )
    strict_group = ap.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict-no-residuals",
        dest="strict_no_residuals",
        action="store_true",
        help="Fail when accepted residuals are present in alignment reports.",
    )
    strict_group.add_argument(
        "--allow-residuals",
        dest="strict_no_residuals",
        action="store_false",
        help="Allow accepted residuals from alignment reports.",
    )
    ap.set_defaults(strict_no_residuals=_env_bool("AI_QUANT_REPLAY_GATE_STRICT_NO_RESIDUALS", True))
    ap.add_argument(
        "--min-live-trades",
        type=int,
        default=_env_int("AI_QUANT_REPLAY_GATE_MIN_LIVE_TRADES", 1),
        help="Minimum live baseline trade count required for a valid scheduled check (default: 1).",
    )
    ap.add_argument(
        "--release-blocker-file",
        default=str(os.getenv("AI_QUANT_REPLAY_GATE_BLOCKER_FILE", "") or ""),
        help="Optional release-blocker status path (default: <bundle-root>/release_blocker.json).",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Optional run report JSON path (default: <bundle-dir>/scheduled_alignment_gate_run.json).",
    )
    auto_clamp_default = _env_bool("AI_QUANT_REPLAY_GATE_AUTO_STRATEGY_WINDOW_CLAMP", True)
    clamp_group = ap.add_mutually_exclusive_group()
    clamp_group.add_argument(
        "--auto-strategy-window-clamp",
        dest="auto_strategy_window_clamp",
        action="store_true",
        help="When strategy SHA drifts in the requested window, clamp to the latest contiguous stable SHA segment.",
    )
    clamp_group.add_argument(
        "--disable-auto-strategy-window-clamp",
        dest="auto_strategy_window_clamp",
        action="store_false",
        help="Disable auto-clamping to the latest stable strategy SHA segment.",
    )
    ap.set_defaults(auto_strategy_window_clamp=auto_clamp_default)
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    repo_root = _resolve_path(str(args.repo_root), base=Path.cwd())
    resolve_base = repo_root if repo_root.exists() else Path.cwd()
    live_db = _resolve_path(str(args.live_db), base=resolve_base)
    paper_db = _resolve_path(str(args.paper_db), base=resolve_base)
    candles_db_raw = str(args.candles_db or "").strip()
    candles_db = _resolve_path(candles_db_raw, base=resolve_base) if candles_db_raw else Path("")
    funding_db_raw = str(args.funding_db or "").strip()
    funding_db = _resolve_path(funding_db_raw, base=resolve_base) if funding_db_raw else None

    interval = str(args.interval or "").strip()

    now_ms = _now_ms()
    lag_minutes = int(args.lag_minutes)
    window_minutes = int(args.window_minutes)
    min_live_trades = int(args.min_live_trades)
    requested_to_ts = int(now_ms - lag_minutes * 60_000)
    requested_from_ts = int(requested_to_ts - window_minutes * 60_000)
    to_ts = int(requested_to_ts)
    from_ts = int(requested_from_ts)
    coverage_from_ts: int | None = None
    coverage_to_ts: int | None = None
    strategy_window_stats: dict[str, Any] | None = None
    strategy_window_adjustment: dict[str, Any] | None = None

    bundle_root = _resolve_path(str(args.bundle_root), base=resolve_base)
    try:
        bundle_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        bundle_root = Path("/tmp/openclaw-ai-quant/replay_gate").resolve()
        bundle_root.mkdir(parents=True, exist_ok=True)
    run_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    interval_tag = interval if interval else "unknown_interval"
    bundle_dir = (bundle_root / f"bundle_{interval_tag}_{from_ts}_{to_ts}_{run_tag}").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        _resolve_path(str(args.output), base=resolve_base)
        if str(args.output or "").strip()
        else (bundle_dir / "scheduled_alignment_gate_run.json").resolve()
    )
    blocker_path = (
        _resolve_path(str(args.release_blocker_file), base=resolve_base)
        if str(args.release_blocker_file or "").strip()
        else (bundle_root / "release_blocker.json").resolve()
    )
    build_log = (bundle_dir / "scheduled_build_bundle.log").resolve()
    harness_log = (bundle_dir / "scheduled_harness.log").resolve()
    manifest_path = (bundle_dir / "replay_bundle_manifest.json").resolve()
    harness_report_path = (bundle_dir / "paper_deterministic_replay_run.json").resolve()
    gate_report_path = (bundle_dir / "alignment_gate_report.json").resolve()

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo_root)
    failures: list[dict[str, Any]] = []

    if not repo_root.exists():
        failures.append(
            {
                "code": "repo_root_not_found",
                "classification": "state_initialisation_gap",
                "detail": str(repo_root),
            }
        )
    if not candles_db_raw:
        failures.append(
            {
                "code": "missing_candles_db",
                "classification": "state_initialisation_gap",
                "detail": "candles DB path is required (--candles-db or AI_QUANT_REPLAY_GATE_CANDLES_DB)",
            }
        )
    if not live_db.exists():
        failures.append(
            {
                "code": "live_db_not_found",
                "classification": "state_initialisation_gap",
                "detail": str(live_db),
            }
        )
    if not paper_db.exists():
        failures.append(
            {
                "code": "paper_db_not_found",
                "classification": "state_initialisation_gap",
                "detail": str(paper_db),
            }
        )
    if candles_db_raw and not candles_db.exists():
        failures.append(
            {
                "code": "candles_db_not_found",
                "classification": "state_initialisation_gap",
                "detail": str(candles_db),
            }
        )
    if funding_db is not None and not funding_db.exists():
        failures.append(
            {
                "code": "funding_db_not_found",
                "classification": "state_initialisation_gap",
                "detail": str(funding_db),
            }
        )
    if window_minutes <= 0:
        failures.append(
            {
                "code": "invalid_window_minutes",
                "classification": "state_initialisation_gap",
                "detail": f"window_minutes={window_minutes} must be > 0",
            }
        )
    if lag_minutes < 0:
        failures.append(
            {
                "code": "invalid_lag_minutes",
                "classification": "state_initialisation_gap",
                "detail": f"lag_minutes={lag_minutes} must be >= 0",
            }
        )
    if min_live_trades < 0:
        failures.append(
            {
                "code": "invalid_min_live_trades",
                "classification": "state_initialisation_gap",
                "detail": f"min_live_trades={min_live_trades} must be >= 0",
            }
        )
    if not interval:
        failures.append(
            {
                "code": "missing_interval",
                "classification": "state_initialisation_gap",
                "detail": "interval must not be empty",
            }
        )
    if requested_from_ts > requested_to_ts:
        failures.append(
            {
                "code": "invalid_replay_window",
                "classification": "state_initialisation_gap",
                "detail": f"from_ts={requested_from_ts} > to_ts={requested_to_ts}",
            }
        )
    if not failures:
        strategy_window_stats = _runtime_strategy_window_stats(
            live_db,
            from_ts=int(from_ts),
            to_ts=int(to_ts),
        )
        if bool(args.auto_strategy_window_clamp):
            distinct_sha = int(strategy_window_stats.get("strategy_sha1_distinct") or 0)
            timeline = strategy_window_stats.get("timeline")
            timeline_rows = timeline if isinstance(timeline, list) else []
            if distinct_sha > 1 and timeline_rows:
                selected_segment: dict[str, Any] | None = None
                selected_from_ts: int | None = None
                selected_to_ts: int | None = None
                selected_trade_rows = 0
                selection_mode = "latest_segment_fallback"

                for segment in reversed(timeline_rows):
                    seg_from_raw = segment.get("first_ts_ms")
                    seg_to_raw = segment.get("last_ts_ms")
                    if not isinstance(seg_from_raw, int) or not isinstance(seg_to_raw, int):
                        continue
                    seg_from_ts = max(int(from_ts), int(seg_from_raw))
                    seg_to_ts = min(int(to_ts), int(seg_to_raw))
                    if seg_from_ts >= seg_to_ts:
                        continue
                    trade_rows = _count_live_trades_in_window(
                        live_db,
                        from_ts=int(seg_from_ts),
                        to_ts=int(seg_to_ts),
                    )
                    if trade_rows >= int(min_live_trades):
                        selected_segment = segment
                        selected_from_ts = int(seg_from_ts)
                        selected_to_ts = int(seg_to_ts)
                        selected_trade_rows = int(trade_rows)
                        selection_mode = "latest_segment_with_min_live_trades"
                        break

                if selected_segment is None:
                    fallback = timeline_rows[-1]
                    seg_from_raw = fallback.get("first_ts_ms")
                    seg_to_raw = fallback.get("last_ts_ms")
                    if isinstance(seg_from_raw, int) and isinstance(seg_to_raw, int):
                        seg_from_ts = max(int(from_ts), int(seg_from_raw))
                        seg_to_ts = min(int(to_ts), int(seg_to_raw))
                        if seg_from_ts < seg_to_ts:
                            selected_segment = fallback
                            selected_from_ts = int(seg_from_ts)
                            selected_to_ts = int(seg_to_ts)
                            selected_trade_rows = _count_live_trades_in_window(
                                live_db,
                                from_ts=int(seg_from_ts),
                                to_ts=int(seg_to_ts),
                            )

                if selected_segment is not None and selected_from_ts is not None and selected_to_ts is not None:
                    if selected_from_ts >= selected_to_ts:
                        failures.append(
                            {
                                "code": "invalid_strategy_window_clamp",
                                "classification": "state_initialisation_gap",
                                "detail": (
                                    f"clamped window is empty: from_ts={selected_from_ts} >= to_ts={selected_to_ts} "
                                    f"(strategy_sha1={selected_segment.get('strategy_sha1')})"
                                ),
                            }
                        )
                    else:
                        strategy_window_adjustment = {
                            "applied": True,
                            "reason": "strategy_sha1_drift_detected",
                            "strategy_sha1_distinct": distinct_sha,
                            "selection_mode": selection_mode,
                            "selected_segment_sha1": str(selected_segment.get("strategy_sha1") or "").strip().lower(),
                            "selected_segment_first_ts_ms": int(selected_segment.get("first_ts_ms") or selected_from_ts),
                            "selected_segment_last_ts_ms": int(selected_segment.get("last_ts_ms") or selected_to_ts),
                            "selected_live_trade_rows": int(selected_trade_rows),
                            "original_from_ts": int(from_ts),
                            "original_to_ts": int(to_ts),
                            "clamped_from_ts": int(selected_from_ts),
                            "clamped_to_ts": int(selected_to_ts),
                        }
                        from_ts = int(selected_from_ts)
                        to_ts = int(selected_to_ts)

        coverage_from_ts, coverage_to_ts = _candles_interval_coverage_ms(candles_db, interval)
        if coverage_from_ts is None or coverage_to_ts is None:
            failures.append(
                {
                    "code": "missing_interval_coverage",
                    "classification": "market_data_alignment_gap",
                    "detail": f"no candles coverage found for interval={interval} in {candles_db}",
                }
            )
        else:
            if to_ts > coverage_to_ts:
                to_ts = int(coverage_to_ts)
                from_ts = int(to_ts - window_minutes * 60_000)
            if from_ts < coverage_from_ts:
                from_ts = int(coverage_from_ts)
            if from_ts > to_ts:
                failures.append(
                    {
                        "code": "invalid_clamped_replay_window",
                        "classification": "market_data_alignment_gap",
                        "detail": (
                            f"effective window outside coverage: from_ts={from_ts}, to_ts={to_ts}, "
                            f"coverage={coverage_from_ts}..{coverage_to_ts}"
                        ),
                    }
                )

    build_cmd = [
        "python3",
        str((repo_root / "tools" / "build_live_replay_bundle.py").resolve()),
        "--live-db",
        str(live_db),
        "--paper-db",
        str(paper_db),
        "--candles-db",
        str(candles_db),
        "--interval",
        interval,
        "--from-ts",
        str(from_ts),
        "--to-ts",
        str(to_ts),
        "--bundle-dir",
        str(bundle_dir),
    ]
    if funding_db is not None:
        build_cmd.extend(["--funding-db", str(funding_db)])

    build_rc: int | None = None
    live_baseline_trades = -1
    if not failures:
        build_rc = _run_to_log(build_cmd, cwd=repo_root, env=env, log_path=build_log)
    if build_rc is not None and build_rc != 0:
        failures.append(
            {
                "code": "bundle_build_failed",
                "classification": "state_initialisation_gap",
                "detail": f"build_live_replay_bundle.py exited with code {build_rc}",
            }
        )
    elif build_rc is not None and not manifest_path.exists():
        failures.append(
            {
                "code": "bundle_manifest_missing",
                "classification": "state_initialisation_gap",
                "detail": str(manifest_path),
            }
        )
    elif build_rc is not None:
        live_baseline_trades = _read_live_baseline_count(manifest_path)
        if live_baseline_trades < min_live_trades:
            failures.append(
                {
                    "code": "insufficient_live_baseline_trades",
                    "classification": "state_initialisation_gap",
                    "detail": f"live_baseline_trades={live_baseline_trades} below min_live_trades={min_live_trades}",
                }
            )

    harness_rc: int | None = None
    harness_report: dict[str, Any] | None = None
    gate_report: dict[str, Any] | None = None

    if not failures:
        harness_cmd = [
            "python3",
            str((repo_root / "tools" / "run_paper_deterministic_replay.py").resolve()),
            "--bundle-dir",
            str(bundle_dir),
            "--repo-root",
            str(repo_root),
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--candles-db",
            str(candles_db),
            "--output",
            str(harness_report_path),
        ]
        if funding_db is not None:
            harness_cmd.extend(["--funding-db", str(funding_db)])
        if bool(args.strict_no_residuals):
            harness_cmd.append("--strict-no-residuals")

        harness_rc = _run_to_log(harness_cmd, cwd=repo_root, env=env, log_path=harness_log)
        if harness_rc != 0:
            failures.append(
                {
                    "code": "deterministic_replay_harness_failed",
                    "classification": "deterministic_logic_divergence",
                    "detail": f"run_paper_deterministic_replay.py exited with code {harness_rc}",
                }
            )

        if harness_report_path.exists():
            try:
                loaded = _read_json(harness_report_path)
            except Exception as exc:
                failures.append(
                    {
                        "code": "invalid_harness_report_json",
                        "classification": "deterministic_logic_divergence",
                        "detail": str(exc),
                    }
                )
            else:
                if isinstance(loaded, dict):
                    harness_report = loaded
                    if not bool(loaded.get("ok")):
                        failures.append(
                            {
                                "code": "deterministic_replay_harness_report_failed",
                                "classification": "deterministic_logic_divergence",
                                "detail": f"failed_step={loaded.get('failed_step')}",
                            }
                        )
                else:
                    failures.append(
                        {
                            "code": "invalid_harness_report",
                            "classification": "deterministic_logic_divergence",
                            "detail": "paper_deterministic_replay_run.json is not a JSON object",
                        }
                    )
        else:
            failures.append(
                {
                    "code": "missing_harness_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(harness_report_path),
                }
            )

        if gate_report_path.exists():
            try:
                loaded = _read_json(gate_report_path)
            except Exception as exc:
                failures.append(
                    {
                        "code": "invalid_alignment_gate_report_json",
                        "classification": "deterministic_logic_divergence",
                        "detail": str(exc),
                    }
                )
            else:
                if isinstance(loaded, dict):
                    gate_report = loaded
                    if not bool(loaded.get("ok")):
                        failures.append(
                            {
                                "code": "alignment_gate_failed",
                                "classification": "deterministic_logic_divergence",
                                "detail": f"failures={len(loaded.get('failures') or [])}",
                            }
                        )
                else:
                    failures.append(
                        {
                            "code": "invalid_alignment_gate_report",
                            "classification": "deterministic_logic_divergence",
                            "detail": "alignment_gate_report.json is not a JSON object",
                        }
                    )
        else:
            failures.append(
                {
                    "code": "missing_alignment_gate_report",
                    "classification": "deterministic_logic_divergence",
                    "detail": str(gate_report_path),
                }
            )

    ok = len(failures) == 0
    generated_at_ms = _now_ms()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_ms": generated_at_ms,
        "ok": ok,
        "repo_root": str(repo_root),
        "window": {
            "interval": interval,
            "requested_from_ts": requested_from_ts,
            "requested_to_ts": requested_to_ts,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "coverage_from_ts": coverage_from_ts,
            "coverage_to_ts": coverage_to_ts,
            "window_minutes": int(args.window_minutes),
            "lag_minutes": int(args.lag_minutes),
            "auto_strategy_window_clamp": bool(args.auto_strategy_window_clamp),
            "strategy_window_stats": strategy_window_stats,
            "strategy_window_adjustment": strategy_window_adjustment,
        },
        "paths": {
            "bundle_root": str(bundle_root),
            "bundle_dir": str(bundle_dir),
            "live_db": str(live_db),
            "paper_db": str(paper_db),
            "candles_db": str(candles_db),
            "funding_db": str(funding_db) if funding_db is not None else None,
            "manifest": str(manifest_path),
            "harness_report": str(harness_report_path),
            "alignment_gate_report": str(gate_report_path),
            "build_log": str(build_log),
            "harness_log": str(harness_log),
            "release_blocker_file": str(blocker_path),
        },
        "min_live_trades": min_live_trades,
        "live_baseline_trades": int(live_baseline_trades),
        "strict_no_residuals": bool(args.strict_no_residuals),
        "build_exit_code": None if build_rc is None else int(build_rc),
        "harness_exit_code": None if harness_rc is None else int(harness_rc),
        "failures": failures,
        "db_checks": {
            "live_has_trades_table": _db_has_table(live_db, "trades"),
            "paper_has_trades_table": _db_has_table(paper_db, "trades"),
        },
    }
    try:
        _write_json(report_path, report)
    except Exception as exc:
        ok = False
        failures.append(
            {
                "code": "write_report_failed",
                "classification": "state_initialisation_gap",
                "detail": str(exc),
            }
        )
        report["ok"] = False
        report["failures"] = failures

    previous_last_passed_ms: int | None = None
    if blocker_path.exists():
        try:
            previous = _read_json(blocker_path)
            if isinstance(previous, dict):
                prev_last_pass = previous.get("last_passed_at_ms")
                if isinstance(prev_last_pass, int):
                    previous_last_passed_ms = prev_last_pass
        except Exception:
            previous_last_passed_ms = None

    blocker_payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_ms": generated_at_ms,
        "blocked": (not ok),
        "reason_codes": [str(f.get("code") or "") for f in failures],
        "report_path": str(report_path),
        "bundle_dir": str(bundle_dir),
        "window": {
            "interval": interval,
            "requested_from_ts": requested_from_ts,
            "requested_to_ts": requested_to_ts,
            "from_ts": from_ts,
            "to_ts": to_ts,
        },
    }
    if ok:
        blocker_payload["last_passed_at_ms"] = generated_at_ms
    elif previous_last_passed_ms is not None:
        blocker_payload["last_passed_at_ms"] = previous_last_passed_ms

    try:
        _write_json(blocker_path, blocker_payload)
    except Exception:
        fallback_blocker_path = Path("/tmp/openclaw-ai-quant/replay_gate/release_blocker.json").resolve()
        fallback_blocker_path.parent.mkdir(parents=True, exist_ok=True)
        blocker_payload["write_fallback_from"] = str(blocker_path)
        _write_json(fallback_blocker_path, blocker_payload)
        blocker_path = fallback_blocker_path
    print(str(report_path))
    print(str(blocker_path))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - emergency fail-closed path
        fallback_blocker_raw = str(os.getenv("AI_QUANT_REPLAY_GATE_BLOCKER_FILE", "") or "").strip()
        if not fallback_blocker_raw:
            bundle_root_raw = str(os.getenv("AI_QUANT_REPLAY_GATE_BUNDLE_ROOT", "") or "").strip()
            if bundle_root_raw:
                fallback_blocker_raw = str((Path(bundle_root_raw).expanduser() / "release_blocker.json"))
            else:
                fallback_blocker_raw = "/tmp/openclaw-ai-quant/replay_gate/release_blocker.json"

        fallback_blocker_path = Path(fallback_blocker_raw).expanduser()
        if not fallback_blocker_path.is_absolute():
            fallback_blocker_path = (Path.cwd() / fallback_blocker_path).resolve()
        fallback_payload = {
            "schema_version": 1,
            "generated_at_ms": _now_ms(),
            "blocked": True,
            "reason_codes": ["scheduler_unhandled_exception"],
            "detail": str(exc),
        }
        try:
            _write_json(fallback_blocker_path, fallback_payload)
        except Exception:
            pass
        raise
