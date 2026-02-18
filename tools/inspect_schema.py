#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


DB_PAPER = Path("/home/fol2hk/.openclaw/workspace/dev/ai_quant/trading_engine.db")
DB_LIVE = Path("/home/fol2hk/.openclaw/workspace/dev/ai_quant/trading_engine_live.db")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str) -> str:
    ident = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(ident):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return ident


def print_schema(db_path: Path, table_name: str) -> None:
    safe_table = _validate_identifier(table_name)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({safe_table})")
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Schema for {safe_table} in {db_path}:")
    for row in rows:
        print(row)


def main() -> int:
    print_schema(DB_PAPER, "trades")
    print_schema(DB_LIVE, "position_state")
    print_schema(DB_LIVE, "oms_intents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
