#!/usr/bin/env python3
"""Prune SQLite runtime logs to a bounded retention window (AQC-1104).

The engine can optionally mirror stdout/stderr into the `runtime_logs` table via
`engine.sqlite_logger`. This table is useful for dashboards and debugging, but it
will grow without bound if left unattended.

This tool deletes rows older than N days (UTC) and is designed to run from a
systemd timer or cron job.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path


AIQ_ROOT = Path(__file__).resolve().parents[1]


def _default_db_path() -> Path:
    return (AIQ_ROOT / "trading_engine.db").resolve()


def _db_path() -> Path:
    p = str(os.getenv("AI_QUANT_DB_PATH", "") or "").strip()
    return Path(p).expanduser().resolve() if p else _default_db_path()


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (str(name),),
    ).fetchone()
    return row is not None


def prune_runtime_logs(*, db_path: Path, keep_days: float, dry_run: bool, vacuum: bool) -> int:
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.exists():
        # Nothing to do.
        return 0

    keep_s = float(max(0.0, float(keep_days))) * 86400.0
    cutoff_ts_ms = int((time.time() - keep_s) * 1000.0)

    con = sqlite3.connect(str(db_path), timeout=2.0)
    try:
        if not _table_exists(con, "runtime_logs"):
            return 0

        row = con.execute("SELECT COUNT(*) FROM runtime_logs WHERE ts_ms < ?", (int(cutoff_ts_ms),)).fetchone()
        n = int(row[0] if row else 0)
        if n <= 0:
            return 0

        if dry_run:
            print(f"[prune_runtime_logs] would_delete={n} cutoff_ts_ms={cutoff_ts_ms} db={db_path}")
            return 0

        con.execute("DELETE FROM runtime_logs WHERE ts_ms < ?", (int(cutoff_ts_ms),))
        con.commit()
        print(f"[prune_runtime_logs] deleted={n} cutoff_ts_ms={cutoff_ts_ms} db={db_path}")

        if vacuum:
            # VACUUM can be expensive; keep it opt-in.
            con.execute("VACUUM")
            con.commit()
        return 0
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prune runtime_logs rows older than N days.")
    ap.add_argument("--db", default="", help="SQLite DB path (default: AI_QUANT_DB_PATH or ./trading_engine.db).")
    ap.add_argument(
        "--keep-days",
        type=float,
        default=float(os.getenv("AI_QUANT_RUNTIME_LOG_KEEP_DAYS", "14") or 14),
        help="Retention window in days (default: 14).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted without writing.")
    ap.add_argument("--vacuum", action="store_true", help="Run VACUUM after deleting rows (slow).")
    args = ap.parse_args(argv)

    path = Path(str(args.db)).expanduser().resolve() if str(args.db).strip() else _db_path()
    return int(
        prune_runtime_logs(
            db_path=path,
            keep_days=float(args.keep_days),
            dry_run=bool(args.dry_run),
            vacuum=bool(args.vacuum),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

