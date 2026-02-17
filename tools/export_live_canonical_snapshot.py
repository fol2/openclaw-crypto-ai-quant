#!/usr/bin/env python3
"""Export a live-canonical snapshot for paper/backtester state sync.

This tool produces a JSON payload that is directly usable as backtester
`--init-state` input (v2 schema) while carrying extra canonical metadata.

Examples:
    python tools/export_live_canonical_snapshot.py --source live --output /tmp/live_init_state.json
    python tools/export_live_canonical_snapshot.py --source paper --output /tmp/paper_init_state.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_DB = PROJECT_DIR / "trading_engine_live.db"
DEFAULT_PAPER_DB = PROJECT_DIR / "trading_engine.db"

ENTRY_ATTEMPT_ACTIONS = ("OPEN", "ADD")
EXIT_ATTEMPT_ACTIONS = ("CLOSE", "REDUCE")


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _parse_timestamp_ms(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > 10_000_000_000:
            return iv
        if iv > 0:
            return iv * 1000
        return 0

    raw = str(value).strip()
    if not raw:
        return 0

    if raw.isdigit():
        iv = int(raw)
        if iv > 10_000_000_000:
            return iv
        return iv * 1000

    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except Exception:
        return 0


def _as_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _max_id(conn: sqlite3.Connection, table_name: str) -> int | None:
    if not _table_exists(conn, table_name):
        return None
    try:
        row = conn.execute(f"SELECT MAX(id) AS max_id FROM {table_name}").fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    value = row["max_id"]
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _reconstruct_positions_and_balance(conn: sqlite3.Connection) -> tuple[float, list[dict[str, Any]]]:
    row = conn.execute("SELECT balance FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    balance = float(row["balance"] or 0.0) if row else 0.0

    sql_open = """
    SELECT t.id AS open_id, t.timestamp AS open_ts, t.symbol, t.type AS pos_type,
           t.price AS open_px, t.size AS open_sz, t.confidence,
           t.entry_atr, t.leverage, t.margin_used
    FROM trades t
    INNER JOIN (
        SELECT symbol, MAX(id) AS open_id
        FROM trades WHERE action = 'OPEN' GROUP BY symbol
    ) lo ON t.id = lo.open_id
    LEFT JOIN (
        SELECT symbol, MAX(id) AS close_id
        FROM trades WHERE action = 'CLOSE' GROUP BY symbol
    ) lc ON t.symbol = lc.symbol
    WHERE lc.close_id IS NULL OR t.id > lc.close_id
    """

    positions: list[dict[str, Any]] = []
    for row in conn.execute(sql_open).fetchall():
        symbol = str(row["symbol"] or "").strip().upper()
        if not symbol:
            continue

        pos_type = str(row["pos_type"] or "").strip().upper()
        if pos_type not in {"LONG", "SHORT"}:
            continue

        open_id = int(row["open_id"])
        avg_entry = float(row["open_px"] or 0.0)
        net_size = float(row["open_sz"] or 0.0)
        if avg_entry <= 0.0 or net_size <= 0.0:
            continue

        entry_atr = float(row["entry_atr"] or 0.0)
        confidence = str((row["confidence"] or "medium")).strip().lower() or "medium"
        leverage = float(row["leverage"] or 1.0)
        if leverage <= 0.0:
            leverage = 1.0

        margin_used = float(row["margin_used"] or 0.0)
        if margin_used <= 0.0:
            margin_used = abs(net_size) * avg_entry / leverage

        fills = conn.execute(
            "SELECT action, price, size, entry_atr FROM trades "
            "WHERE symbol = ? AND id > ? AND action IN ('ADD', 'REDUCE') ORDER BY id ASC",
            (symbol, open_id),
        ).fetchall()

        for fill in fills:
            action = str(fill["action"] or "").strip().upper()
            px = float(fill["price"] or 0.0)
            sz = float(fill["size"] or 0.0)
            fill_atr = float(fill["entry_atr"] or 0.0)
            if px <= 0.0 or sz <= 0.0:
                continue

            if action == "ADD":
                new_total = net_size + sz
                if new_total > 0:
                    avg_entry = ((avg_entry * net_size) + (px * sz)) / new_total
                    if fill_atr > 0 and entry_atr > 0:
                        entry_atr = ((entry_atr * net_size) + (fill_atr * sz)) / new_total
                    elif fill_atr > 0:
                        entry_atr = fill_atr
                    net_size = new_total
            elif action == "REDUCE":
                net_size -= sz
                if net_size <= 0:
                    net_size = 0.0
                    break

        if net_size <= 0.0:
            continue

        trailing_sl = None
        adds_count = 0
        tp1_taken = False
        last_add_time_ms = 0
        entry_adx_threshold = 0.0

        if _table_exists(conn, "position_state"):
            ps_row = conn.execute(
                "SELECT open_trade_id, trailing_sl, adds_count, tp1_taken, last_add_time, entry_adx_threshold "
                "FROM position_state WHERE symbol = ? LIMIT 1",
                (symbol,),
            ).fetchone()
            if ps_row:
                open_trade_id = ps_row["open_trade_id"]
                if open_trade_id is None or int(open_trade_id) == open_id:
                    trailing_sl = float(ps_row["trailing_sl"]) if ps_row["trailing_sl"] is not None else None
                    adds_count = int(ps_row["adds_count"] or 0)
                    tp1_taken = bool(ps_row["tp1_taken"] or 0)
                    last_add_time_ms = int(ps_row["last_add_time"] or 0)
                    entry_adx_threshold = float(ps_row["entry_adx_threshold"] or 0.0)

        side = "long" if pos_type == "LONG" else "short"
        positions.append(
            {
                "symbol": symbol,
                "side": side,
                "size": round(net_size, 10),
                "entry_price": round(avg_entry, 10),
                "entry_atr": round(entry_atr, 10),
                "trailing_sl": round(trailing_sl, 10) if trailing_sl is not None else None,
                "confidence": confidence,
                "leverage": float(leverage),
                "margin_used": round(margin_used, 10),
                "adds_count": int(adds_count),
                "tp1_taken": bool(tp1_taken),
                "open_time_ms": _parse_timestamp_ms(row["open_ts"]),
                "last_add_time_ms": int(last_add_time_ms),
                "entry_adx_threshold": float(entry_adx_threshold),
            }
        )

    positions.sort(key=lambda p: str(p.get("symbol") or ""))
    return balance, positions


def _load_attempt_markers(conn: sqlite3.Connection) -> tuple[dict[str, int], dict[str, int]]:
    entry_attempt_ms: dict[str, int] = {}
    exit_attempt_ms: dict[str, int] = {}

    if not _table_exists(conn, "trades"):
        return entry_attempt_ms, exit_attempt_ms

    entry_q = (
        "SELECT symbol, MAX(timestamp) AS ts FROM trades "
        "WHERE action IN ({}) GROUP BY symbol".format(
            ",".join("?" for _ in ENTRY_ATTEMPT_ACTIONS)
        )
    )
    for row in conn.execute(entry_q, ENTRY_ATTEMPT_ACTIONS).fetchall():
        symbol = str(row["symbol"] or "").strip().upper()
        if not symbol:
            continue
        ts_ms = _parse_timestamp_ms(row["ts"])
        if ts_ms > 0:
            entry_attempt_ms[symbol] = ts_ms

    exit_q = (
        "SELECT symbol, MAX(timestamp) AS ts FROM trades "
        "WHERE action IN ({}) GROUP BY symbol".format(
            ",".join("?" for _ in EXIT_ATTEMPT_ACTIONS)
        )
    )
    for row in conn.execute(exit_q, EXIT_ATTEMPT_ACTIONS).fetchall():
        symbol = str(row["symbol"] or "").strip().upper()
        if not symbol:
            continue
        ts_ms = _parse_timestamp_ms(row["ts"])
        if ts_ms > 0:
            exit_attempt_ms[symbol] = ts_ms

    return entry_attempt_ms, exit_attempt_ms


def _load_open_orders(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "oms_open_orders"):
        return []

    rows = conn.execute("SELECT * FROM oms_open_orders ORDER BY symbol ASC").fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({k: _as_json_value(row[k]) for k in row.keys()})
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a live-canonical snapshot for paper/backtester sync. "
            "The output can be passed directly to mei-backtester --init-state."
        )
    )
    parser.add_argument(
        "--source",
        choices=("live", "paper"),
        default="live",
        help="Source mode used to select default DB path.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional explicit SQLite DB path. Overrides --source default.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (DEFAULT_LIVE_DB if args.source == "live" else DEFAULT_PAPER_DB).resolve()

    if not db_path.exists():
        parser.error(f"DB path not found: {db_path}")

    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    warnings: list[str] = []

    conn = _connect_ro(db_path)
    try:
        balance, positions = _reconstruct_positions_and_balance(conn)
        entry_attempt_ms, exit_attempt_ms = _load_attempt_markers(conn)
        open_orders = _load_open_orders(conn)

        cursors = {
            "trades_max_id": _max_id(conn, "trades"),
            "decision_events_max_id": _max_id(conn, "decision_events"),
            "ws_events_max_id": _max_id(conn, "ws_events"),
            "oms_intents_max_id": _max_id(conn, "oms_intents"),
            "oms_orders_max_id": _max_id(conn, "oms_orders"),
            "oms_fills_max_id": _max_id(conn, "oms_fills"),
        }

        if balance <= 0.0:
            warnings.append("balance_non_positive")

        snapshot = {
            "version": 2,
            "source": f"{args.source}_db",
            "exported_at_ms": now_ms,
            "balance": float(balance),
            "positions": positions,
            "runtime": {
                "entry_attempt_ms_by_symbol": entry_attempt_ms,
                "exit_attempt_ms_by_symbol": exit_attempt_ms,
            },
            "canonical": {
                "db_path": str(db_path),
                "open_orders": open_orders,
                "cursors": cursors,
                "warnings": warnings,
                "notes": [
                    "This snapshot is live-canonical for deterministic replay seeding.",
                    "Non-simulatable exchange effects must be tracked separately in audit reports.",
                ],
            },
        }
    finally:
        conn.close()

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
