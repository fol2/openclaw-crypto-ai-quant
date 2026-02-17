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
import sqlite3
from pathlib import Path
from typing import Any

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
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()

    if not live_db.exists():
        parser.error(f"live DB not found: {live_db}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")
    if not candles_db.exists():
        parser.error(f"candles DB not found: {candles_db}")
    if args.from_ts > args.to_ts:
        parser.error("from-ts must be <= to-ts")

    bundle_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = bundle_dir / args.snapshot_name
    live_trades_path = bundle_dir / "live_baseline_trades.jsonl"
    audit_report_path = bundle_dir / "state_alignment_report.json"
    replay_trades_csv = bundle_dir / "backtester_trades.csv"
    manifest_path = bundle_dir / "replay_bundle_manifest.json"

    baseline_trades = _load_live_baseline_trades(
        live_db,
        from_ts=int(args.from_ts),
        to_ts=int(args.to_ts),
    )

    with live_trades_path.open("w", encoding="utf-8") as fp:
        for row in baseline_trades:
            fp.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    cmd_export_seed = (
        f"python tools/export_live_canonical_snapshot.py --source live --db-path {json.dumps(str(live_db))} "
        f"--output {json.dumps(str(snapshot_path))}\n"
        f"python tools/apply_canonical_snapshot_to_paper.py --snapshot {json.dumps(str(snapshot_path))} "
        f"--target-db {json.dumps(str(paper_db))}"
    )

    cmd_replay = (
        "cd backtester\n"
        f"./target/release/mei-backtester replay --candles-db {json.dumps(str(candles_db))} "
        f"--interval {json.dumps(str(args.interval))} --from-ts {int(args.from_ts)} --to-ts {int(args.to_ts)} "
        f"--init-state {json.dumps(str(snapshot_path))} --export-trades {json.dumps(str(replay_trades_csv))}"
    )

    cmd_audit = (
        f"python tools/audit_state_sync_alignment.py --live-db {json.dumps(str(live_db))} "
        f"--paper-db {json.dumps(str(paper_db))} --snapshot {json.dumps(str(snapshot_path))} "
        f"--output {json.dumps(str(audit_report_path))}"
    )

    (bundle_dir / "run_01_export_and_seed.sh").write_text(cmd_export_seed + "\n", encoding="utf-8")
    (bundle_dir / "run_02_replay.sh").write_text(cmd_replay + "\n", encoding="utf-8")
    (bundle_dir / "run_03_audit.sh").write_text(cmd_audit + "\n", encoding="utf-8")

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
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
            "interval": str(args.interval),
            "from_ts": int(args.from_ts),
            "to_ts": int(args.to_ts),
        },
        "artefacts": {
            "snapshot_path": str(snapshot_path),
            "live_baseline_trades": str(live_trades_path),
            "live_baseline_trades_sha256": _hash_file(live_trades_path),
            "audit_report_path": str(audit_report_path),
            "backtester_trades_csv": str(replay_trades_csv),
        },
        "counts": {
            "live_baseline_trades": len(baseline_trades),
        },
        "commands": {
            "export_and_seed": cmd_export_seed,
            "replay": cmd_replay,
            "audit": cmd_audit,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
