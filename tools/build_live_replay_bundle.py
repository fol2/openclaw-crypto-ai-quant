#!/usr/bin/env python3
"""Build a deterministic replay bundle for live-canonical replication.

The bundle contains:
- live baseline trades for a fixed window
- executable command scripts for snapshot export, paper seeding, replay, and audit
- a manifest with strict parameters and file hashes for traceability
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import sqlite3
from pathlib import Path
from typing import Any

try:
    from candles_provenance import build_candles_window_provenance
except ModuleNotFoundError:  # pragma: no cover - module execution path
    from tools.candles_provenance import build_candles_window_provenance

TRADE_ACTIONS = {"OPEN", "ADD", "REDUCE", "CLOSE", "FUNDING"}


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_timestamp_ms(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > 10_000_000_000:
            return iv
        return max(0, iv * 1000)

    raw = str(value).strip()
    if not raw:
        return 0

    if raw.isdigit():
        iv = int(raw)
        if iv > 10_000_000_000:
            return iv
        return iv * 1000

    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(ts.timestamp() * 1000)
    except Exception:
        return 0


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a live replay bundle with deterministic artefacts.")
    parser.add_argument("--live-db", default="./trading_engine_live.db", help="Path to live SQLite DB")
    parser.add_argument("--paper-db", default="./trading_engine.db", help="Path to paper SQLite DB")
    parser.add_argument("--candles-db", required=True, help="Path to candles SQLite DB used for replay")
    parser.add_argument("--funding-db", default=None, help="Optional funding SQLite DB used for replay funding events")
    parser.add_argument("--interval", default="1h", help="Replay interval (for command template)")
    parser.add_argument("--from-ts", type=int, required=True, help="Replay start timestamp (ms, inclusive)")
    parser.add_argument("--to-ts", type=int, required=True, help="Replay end timestamp (ms, inclusive)")
    parser.add_argument("--bundle-dir", required=True, help="Output bundle directory")
    parser.add_argument(
        "--snapshot-name",
        default="live_init_state_v2.json",
        help="Snapshot file name to generate inside the bundle",
    )
    return parser


def _load_live_baseline_trades(live_db: Path, *, from_ts: int, to_ts: int) -> list[dict[str, Any]]:
    conn = _connect_ro(live_db)
    try:
        rows = conn.execute(
            "SELECT id, timestamp, symbol, action, type, price, size, pnl, balance, reason, confidence, fee_usd, leverage, margin_used "
            "FROM trades ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        action = str(row["action"] or "").strip().upper()
        if action not in TRADE_ACTIONS:
            continue

        ts_ms = _parse_timestamp_ms(row["timestamp"])
        if ts_ms < from_ts or ts_ms > to_ts:
            continue

        out.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "timestamp_ms": ts_ms,
                "symbol": str(row["symbol"] or "").strip().upper(),
                "action": action,
                "type": str(row["type"] or "").strip().upper(),
                "price": float(row["price"] or 0.0),
                "size": float(row["size"] or 0.0),
                "pnl": float(row["pnl"] or 0.0),
                "balance": float(row["balance"] or 0.0),
                "reason": str(row["reason"] or ""),
                "confidence": str(row["confidence"] or "").strip().lower(),
                "fee_usd": float(row["fee_usd"] or 0.0),
                "leverage": float(row["leverage"] or 0.0),
                "margin_used": float(row["margin_used"] or 0.0),
            }
        )

    return out


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_db = Path(args.live_db).expanduser().resolve()
    paper_db = Path(args.paper_db).expanduser().resolve()
    candles_db = Path(args.candles_db).expanduser().resolve()
    funding_db = Path(args.funding_db).expanduser().resolve() if args.funding_db else None
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    snapshot_name = Path(args.snapshot_name).name

    if not live_db.exists():
        parser.error(f"live DB not found: {live_db}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")
    if not candles_db.exists():
        parser.error(f"candles DB not found: {candles_db}")
    if funding_db is not None and not funding_db.exists():
        parser.error(f"funding DB not found: {funding_db}")
    if args.from_ts > args.to_ts:
        parser.error("from-ts must be <= to-ts")

    bundle_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = bundle_dir / snapshot_name
    live_trades_path = bundle_dir / "live_baseline_trades.jsonl"
    audit_report_path = bundle_dir / "state_alignment_report.json"
    replay_trades_csv = bundle_dir / "backtester_trades.csv"
    replay_report_path = bundle_dir / "backtester_replay_report.json"
    trade_reconcile_path = bundle_dir / "trade_reconcile_report.json"
    action_reconcile_path = bundle_dir / "action_reconcile_report.json"
    live_paper_action_reconcile_path = bundle_dir / "live_paper_action_reconcile_report.json"
    live_paper_decision_trace_reconcile_path = bundle_dir / "live_paper_decision_trace_reconcile_report.json"
    event_order_parity_path = bundle_dir / "event_order_parity_report.json"
    gpu_parity_report_path = bundle_dir / "gpu_smoke_parity_report.json"
    alignment_gate_path = bundle_dir / "alignment_gate_report.json"
    paper_harness_report_path = bundle_dir / "paper_deterministic_replay_run.json"
    manifest_path = bundle_dir / "replay_bundle_manifest.json"

    baseline_trades = _load_live_baseline_trades(
        live_db,
        from_ts=int(args.from_ts),
        to_ts=int(args.to_ts),
    )
    baseline_trade_exit_count = sum(1 for row in baseline_trades if row.get("action") in {"CLOSE", "REDUCE"})
    baseline_trade_funding_count = sum(1 for row in baseline_trades if row.get("action") == "FUNDING")
    candles_provenance = build_candles_window_provenance(
        candles_db,
        interval=str(args.interval),
        from_ts=int(args.from_ts),
        to_ts=int(args.to_ts),
    )

    with live_trades_path.open("w", encoding="utf-8") as fp:
        for row in baseline_trades:
            fp.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    cmd_export_seed = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"LIVE_DB=\"${{LIVE_DB:-{shlex.quote(str(live_db))}}}\"\n"
        f"PAPER_DB=\"${{PAPER_DB:-{shlex.quote(str(paper_db))}}}\"\n"
        f"SNAPSHOT_PATH=\"$BUNDLE_DIR/{snapshot_name}\"\n"
        "python3 \"$REPO_ROOT/tools/export_live_canonical_snapshot.py\" --source live --db-path \"$LIVE_DB\" --output \"$SNAPSHOT_PATH\"\n"
        "python3 \"$REPO_ROOT/tools/apply_canonical_snapshot_to_paper.py\" --snapshot \"$SNAPSHOT_PATH\" --target-db \"$PAPER_DB\""
    )

    if funding_db is not None:
        funding_env = f"FUNDING_DB=\"${{FUNDING_DB:-{shlex.quote(str(funding_db))}}}\"\n"
        funding_arg = " --funding-db \"$FUNDING_DB\""
    else:
        funding_env = ""
        funding_arg = ""

    cmd_replay = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"CANDLES_DB=\"${{CANDLES_DB:-{shlex.quote(str(candles_db))}}}\"\n"
        f"{funding_env}"
        f"SNAPSHOT_PATH=\"$BUNDLE_DIR/{snapshot_name}\"\n"
        "cd \"$REPO_ROOT/backtester\"\n"
        f"./target/release/mei-backtester replay --candles-db \"$CANDLES_DB\" "
        f"--interval {shlex.quote(str(args.interval))} --from-ts {int(args.from_ts)} --to-ts {int(args.to_ts)} "
        f"--init-state \"$SNAPSHOT_PATH\" --export-trades \"$BUNDLE_DIR/{replay_trades_csv.name}\" "
        f"--output \"$BUNDLE_DIR/{replay_report_path.name}\" --trades{funding_arg}"
    )

    cmd_audit = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"LIVE_DB=\"${{LIVE_DB:-{shlex.quote(str(live_db))}}}\"\n"
        f"PAPER_DB=\"${{PAPER_DB:-{shlex.quote(str(paper_db))}}}\"\n"
        f"SNAPSHOT_PATH=\"$BUNDLE_DIR/{snapshot_name}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_state_sync_alignment.py\" --live-db \"$LIVE_DB\" "
        f"--paper-db \"$PAPER_DB\" --snapshot \"$SNAPSHOT_PATH\" --output \"$BUNDLE_DIR/{audit_report_path.name}\""
    )

    cmd_trade_reconcile = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_live_backtester_trade_reconcile.py\" "
        f"--live-baseline \"$BUNDLE_DIR/{live_trades_path.name}\" "
        f"--backtester-trades \"$BUNDLE_DIR/{replay_trades_csv.name}\" "
        f"--output \"$BUNDLE_DIR/{trade_reconcile_path.name}\""
    )

    cmd_action_reconcile = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_live_backtester_action_reconcile.py\" "
        f"--live-baseline \"$BUNDLE_DIR/{live_trades_path.name}\" "
        f"--backtester-replay-report \"$BUNDLE_DIR/{replay_report_path.name}\" "
        f"--output \"$BUNDLE_DIR/{action_reconcile_path.name}\""
    )

    cmd_live_paper_action_reconcile = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"LIVE_DB=\"${{LIVE_DB:-{shlex.quote(str(live_db))}}}\"\n"
        f"PAPER_DB=\"${{PAPER_DB:-{shlex.quote(str(paper_db))}}}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_live_paper_action_reconcile.py\" "
        "--live-db \"$LIVE_DB\" "
        "--paper-db \"$PAPER_DB\" "
        f"--from-ts {int(args.from_ts)} --to-ts {int(args.to_ts)} "
        f"--output \"$BUNDLE_DIR/{live_paper_action_reconcile_path.name}\""
    )

    cmd_live_paper_decision_trace_reconcile = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"LIVE_DB=\"${{LIVE_DB:-{shlex.quote(str(live_db))}}}\"\n"
        f"PAPER_DB=\"${{PAPER_DB:-{shlex.quote(str(paper_db))}}}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_live_paper_decision_trace.py\" "
        "--live-db \"$LIVE_DB\" "
        "--paper-db \"$PAPER_DB\" "
        f"--from-ts {int(args.from_ts)} --to-ts {int(args.to_ts)} "
        f"--output \"$BUNDLE_DIR/{live_paper_decision_trace_reconcile_path.name}\""
    )

    cmd_event_order_parity = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"PAPER_DB=\"${{PAPER_DB:-{shlex.quote(str(paper_db))}}}\"\n"
        "python3 \"$REPO_ROOT/tools/audit_live_baseline_paper_order_parity.py\" "
        f"--live-baseline \"$BUNDLE_DIR/{live_trades_path.name}\" "
        "--paper-db \"$PAPER_DB\" "
        f"--from-ts {int(args.from_ts)} --to-ts {int(args.to_ts)} "
        "--fail-on-mismatch "
        f"--output \"$BUNDLE_DIR/{event_order_parity_path.name}\""
    )

    cmd_gpu_parity = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"CANDLES_DB=\"${{CANDLES_DB:-{shlex.quote(str(candles_db))}}}\"\n"
        f"{funding_env}"
        "python3 \"$REPO_ROOT/tools/run_bundle_gpu_parity.py\" "
        "--bundle-dir \"$BUNDLE_DIR\" "
        "--repo-root \"$REPO_ROOT\" "
        "--candles-db \"$CANDLES_DB\" "
        f"{funding_arg} "
        f"--output \"$BUNDLE_DIR/{gpu_parity_report_path.name}\""
    )

    cmd_alignment_gate = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        f"CANDLES_DB=\"${{CANDLES_DB:-{shlex.quote(str(candles_db))}}}\"\n"
        "python3 \"$REPO_ROOT/tools/assert_replay_bundle_alignment.py\" "
        "--bundle-dir \"$BUNDLE_DIR\" "
        "--candles-db \"$CANDLES_DB\" "
        f"--live-paper-report \"$BUNDLE_DIR/{live_paper_action_reconcile_path.name}\" "
        "--require-live-paper "
        f"--live-paper-decision-trace-report \"$BUNDLE_DIR/{live_paper_decision_trace_reconcile_path.name}\" "
        "--require-live-paper-decision-trace "
        f"--event-order-report \"$BUNDLE_DIR/{event_order_parity_path.name}\" "
        "--require-event-order "
        f"--gpu-parity-report \"$BUNDLE_DIR/{gpu_parity_report_path.name}\" "
        "--require-gpu-parity "
        f"--output \"$BUNDLE_DIR/{alignment_gate_path.name}\""
    )

    cmd_paper_harness = (
        "set -euo pipefail\n"
        "BUNDLE_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd)}\"\n"
        "STRICT_FLAG=\"\"\n"
        "if [ \"${STRICT_NO_RESIDUALS:-0}\" = \"1\" ]; then STRICT_FLAG=\"--strict-no-residuals\"; fi\n"
        "python3 \"$REPO_ROOT/tools/run_paper_deterministic_replay.py\" "
        "--bundle-dir \"$BUNDLE_DIR\" "
        "--repo-root \"$REPO_ROOT\" "
        "$STRICT_FLAG "
        f"--output \"$BUNDLE_DIR/{paper_harness_report_path.name}\""
    )

    (bundle_dir / "run_01_export_and_seed.sh").write_text(cmd_export_seed + "\n", encoding="utf-8")
    (bundle_dir / "run_02_replay.sh").write_text(cmd_replay + "\n", encoding="utf-8")
    (bundle_dir / "run_03_audit.sh").write_text(cmd_audit + "\n", encoding="utf-8")
    (bundle_dir / "run_04_trade_reconcile.sh").write_text(cmd_trade_reconcile + "\n", encoding="utf-8")
    (bundle_dir / "run_05_action_reconcile.sh").write_text(cmd_action_reconcile + "\n", encoding="utf-8")
    (bundle_dir / "run_06_live_paper_action_reconcile.sh").write_text(cmd_live_paper_action_reconcile + "\n", encoding="utf-8")
    (bundle_dir / "run_07_live_paper_decision_trace_reconcile.sh").write_text(
        cmd_live_paper_decision_trace_reconcile + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "run_07b_event_order_parity.sh").write_text(cmd_event_order_parity + "\n", encoding="utf-8")
    (bundle_dir / "run_07c_gpu_parity.sh").write_text(cmd_gpu_parity + "\n", encoding="utf-8")
    (bundle_dir / "run_08_assert_alignment.sh").write_text(cmd_alignment_gate + "\n", encoding="utf-8")
    (bundle_dir / "run_09_paper_deterministic_replay.sh").write_text(cmd_paper_harness + "\n", encoding="utf-8")

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07b_event_order_parity.sh",
        "run_07c_gpu_parity.sh",
        "run_08_assert_alignment.sh",
        "run_09_paper_deterministic_replay.sh",
    ):
        script_path = bundle_dir / script_name
        script_path.chmod(0o755)

    manifest = {
        "schema_version": 1,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "bundle_dir": str(bundle_dir),
        "inputs": {
            "live_db": str(live_db),
            "paper_db": str(paper_db),
            "candles_db": str(candles_db),
            "funding_db": str(funding_db) if funding_db is not None else None,
            "interval": str(args.interval),
            "from_ts": int(args.from_ts),
            "to_ts": int(args.to_ts),
        },
        "input_hashes": {
            "candles_db_sha256": _hash_file(candles_db),
            "funding_db_sha256": _hash_file(funding_db) if funding_db is not None else None,
        },
        "candles_provenance": candles_provenance,
        "artefacts": {
            "snapshot_file": snapshot_path.name,
            "live_baseline_trades": live_trades_path.name,
            "live_baseline_trades_sha256": _hash_file(live_trades_path),
            "audit_report_file": audit_report_path.name,
            "backtester_trades_csv": replay_trades_csv.name,
            "backtester_replay_report_json": replay_report_path.name,
            "trade_reconcile_report_file": trade_reconcile_path.name,
            "action_reconcile_report_file": action_reconcile_path.name,
            "live_paper_action_reconcile_report_file": live_paper_action_reconcile_path.name,
            "live_paper_decision_trace_reconcile_report_file": live_paper_decision_trace_reconcile_path.name,
            "event_order_parity_report_file": event_order_parity_path.name,
            "gpu_parity_report_file": gpu_parity_report_path.name,
            "alignment_gate_report_file": alignment_gate_path.name,
            "paper_deterministic_replay_run_report_file": paper_harness_report_path.name,
        },
        "counts": {
            "live_baseline_trades": len(baseline_trades),
            "live_baseline_simulatable_exits": baseline_trade_exit_count,
            "live_baseline_funding_events": baseline_trade_funding_count,
        },
        "commands": {
            "export_and_seed": cmd_export_seed,
            "replay": cmd_replay,
            "audit": cmd_audit,
            "trade_reconcile": cmd_trade_reconcile,
            "action_reconcile": cmd_action_reconcile,
            "live_paper_action_reconcile": cmd_live_paper_action_reconcile,
            "live_paper_decision_trace_reconcile": cmd_live_paper_decision_trace_reconcile,
            "event_order_parity": cmd_event_order_parity,
            "gpu_parity": cmd_gpu_parity,
            "alignment_gate": cmd_alignment_gate,
            "paper_deterministic_replay_harness": cmd_paper_harness,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
