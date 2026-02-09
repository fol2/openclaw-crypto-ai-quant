#!/usr/bin/env python3
"""Fetch historical funding rates from Hyperliquid and store in SQLite.

Usage:
    python fetch_funding_rates.py [--days 30] [--db candles_dbs/funding_rates.db]

The script:
  1. Reads the symbol list from AI_QUANT_SYMBOLS env or the sidecar universe env file.
  2. For each symbol, fetches funding_history from the Hyperliquid REST API.
  3. Inserts into a SQLite database (INSERT OR IGNORE for idempotency).

Run manually or via cron (every 4-6 hours) to keep the DB up to date.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

from hyperliquid.info import Info
from hyperliquid.utils import constants


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    time INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    premium REAL,
    PRIMARY KEY (symbol, time)
);
"""

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "candles_dbs",
    "funding_rates.db",
)


def get_symbols() -> list[str]:
    """Get trading symbols from env or default universe file."""
    raw = os.getenv("AI_QUANT_SYMBOLS", "")
    if raw.strip():
        return [s.strip().upper() for s in raw.split(",") if s.strip()]

    # Fallback: read from universe env file
    env_file = os.path.expanduser("~/.config/openclaw/ai-quant-universe.env")
    if os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("AI_QUANT_SIDECAR_SYMBOLS="):
                    val = line.split("=", 1)[1]
                    return [s.strip().upper() for s in val.split(",") if s.strip()]

    # Absolute fallback
    return ["BTC", "ETH", "SOL"]


def get_last_time(conn: sqlite3.Connection, symbol: str) -> int | None:
    """Get the latest timestamp for a symbol in the DB."""
    row = conn.execute(
        "SELECT MAX(time) FROM funding_rates WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def fetch_and_store(
    info: Info,
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> int:
    """Fetch funding history for one symbol and insert into DB. Returns rows inserted."""
    try:
        data = info.funding_history(symbol, start_ms, end_ms)
    except Exception as e:
        print(f"  WARN: API error for {symbol}: {e}", file=sys.stderr)
        return 0

    if not data:
        return 0

    rows = []
    for entry in data:
        try:
            t = int(entry["time"])
            rate = float(entry["fundingRate"])
            premium = float(entry.get("premium", 0.0)) if "premium" in entry else None
            rows.append((symbol, t, rate, premium))
        except (KeyError, ValueError, TypeError) as e:
            print(f"  WARN: bad entry for {symbol}: {entry} ({e})", file=sys.stderr)
            continue

    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO funding_rates (symbol, time, funding_rate, premium) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid funding rates")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to backfill (default: 30)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    symbols = get_symbols()
    print(f"[funding] {len(symbols)} symbols, {args.days} days lookback")
    print(f"[funding] DB: {args.db}")

    # Ensure DB directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(DB_SCHEMA)

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    now_ms = int(time.time() * 1000)
    default_start_ms = now_ms - args.days * 24 * 3600 * 1000

    total_inserted = 0
    for i, sym in enumerate(symbols):
        last = get_last_time(conn, sym)
        start_ms = (last + 1) if last is not None else default_start_ms

        if start_ms >= now_ms:
            print(f"  [{i+1}/{len(symbols)}] {sym}: up to date")
            continue

        count = fetch_and_store(info, conn, sym, start_ms, now_ms)
        total_inserted += count
        print(f"  [{i+1}/{len(symbols)}] {sym}: +{count} rows (from {start_ms})")

        # Rate limit: be polite to HL API
        time.sleep(0.2)

    conn.close()
    print(f"\n[funding] Done. Total inserted: {total_inserted}")


if __name__ == "__main__":
    main()
