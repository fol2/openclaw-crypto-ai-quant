#!/usr/bin/env python3
"""Audit alignment across live DB, paper DB, and optional canonical snapshot.

The report is intended for financial-grade state-sync tracing before replay/parity runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_TOL = 1e-9


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
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


def _reconstruct_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "trades"):
        return []

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
        pos_type = str(row["pos_type"] or "").strip().upper()
        if not symbol or pos_type not in {"LONG", "SHORT"}:
            continue

        open_id = int(row["open_id"])
        avg_entry = float(row["open_px"] or 0.0)
        net_size = float(row["open_sz"] or 0.0)
        if avg_entry <= 0.0 or net_size <= 0.0:
            continue

        entry_atr = float(row["entry_atr"] or 0.0)
        leverage = float(row["leverage"] or 1.0)
        if leverage <= 0.0:
            leverage = 1.0
        margin_used = float(row["margin_used"] or (abs(net_size) * avg_entry / leverage))

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
                if new_total > 0.0:
                    avg_entry = ((avg_entry * net_size) + (px * sz)) / new_total
                    if fill_atr > 0 and entry_atr > 0:
                        entry_atr = ((entry_atr * net_size) + (fill_atr * sz)) / new_total
                    elif fill_atr > 0:
                        entry_atr = fill_atr
                    net_size = new_total
            elif action == "REDUCE":
                net_size -= sz
                if net_size <= 0.0:
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

        positions.append(
            {
                "symbol": symbol,
                "side": "long" if pos_type == "LONG" else "short",
                "size": round(net_size, 10),
                "entry_price": round(avg_entry, 10),
                "entry_atr": round(entry_atr, 10),
                "trailing_sl": round(trailing_sl, 10) if trailing_sl is not None else None,
                "confidence": str(row["confidence"] or "medium").strip().lower() or "medium",
                "leverage": float(leverage),
                "margin_used": round(margin_used, 10),
                "adds_count": int(adds_count),
                "tp1_taken": bool(tp1_taken),
                "open_time_ms": _parse_timestamp_ms(row["open_ts"]),
                "last_add_time_ms": int(last_add_time_ms),
                "entry_adx_threshold": float(entry_adx_threshold),
            }
        )

    positions.sort(key=lambda x: x["symbol"])
    return positions


def _load_balance(conn: sqlite3.Connection) -> float:
    if not _table_exists(conn, "trades"):
        return 0.0
    row = conn.execute("SELECT balance FROM trades ORDER BY id DESC LIMIT 1").fetchone()
    return float(row["balance"] or 0.0) if row else 0.0


def _load_open_orders(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "oms_open_orders"):
        return []
    rows = conn.execute("SELECT * FROM oms_open_orders ORDER BY symbol ASC").fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({k: row[k] for k in row.keys()})
    return out


def _index_positions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("symbol") or "").upper(): r for r in rows if str(r.get("symbol") or "").strip()}


def _almost_equal(left: float, right: float, tol: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def _compare_positions(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
    tol: float,
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    lidx = _index_positions(left)
    ridx = _index_positions(right)
    symbols = sorted(set(lidx.keys()) | set(ridx.keys()))

    numeric_fields = (
        "size",
        "entry_price",
        "entry_atr",
        "trailing_sl",
        "leverage",
        "margin_used",
        "entry_adx_threshold",
    )
    scalar_fields = (
        "side",
        "confidence",
        "adds_count",
        "tp1_taken",
        "open_time_ms",
        "last_add_time_ms",
    )

    for symbol in symbols:
        lrow = lidx.get(symbol)
        rrow = ridx.get(symbol)
        if lrow is None:
            diffs.append(
                {
                    "classification": "state_initialisation_gap",
                    "kind": "position_missing",
                    "symbol": symbol,
                    "left": left_name,
                    "right": right_name,
                    "detail": "missing_on_left",
                }
            )
            continue
        if rrow is None:
            diffs.append(
                {
                    "classification": "state_initialisation_gap",
                    "kind": "position_missing",
                    "symbol": symbol,
                    "left": left_name,
                    "right": right_name,
                    "detail": "missing_on_right",
                }
            )
            continue

        for field in scalar_fields:
            if lrow.get(field) != rrow.get(field):
                diffs.append(
                    {
                        "classification": "state_initialisation_gap",
                        "kind": "position_field_mismatch",
                        "symbol": symbol,
                        "field": field,
                        "left": lrow.get(field),
                        "right": rrow.get(field),
                        "left_source": left_name,
                        "right_source": right_name,
                    }
                )

        for field in numeric_fields:
            lv = lrow.get(field)
            rv = rrow.get(field)
            if lv is None and rv is None:
                continue
            if lv is None or rv is None:
                diffs.append(
                    {
                        "classification": "state_initialisation_gap",
                        "kind": "position_field_mismatch",
                        "symbol": symbol,
                        "field": field,
                        "left": lv,
                        "right": rv,
                        "left_source": left_name,
                        "right_source": right_name,
                    }
                )
                continue
            if not _almost_equal(float(lv), float(rv), tol):
                diffs.append(
                    {
                        "classification": "numeric_policy_divergence",
                        "kind": "position_numeric_mismatch",
                        "symbol": symbol,
                        "field": field,
                        "left": float(lv),
                        "right": float(rv),
                        "left_source": left_name,
                        "right_source": right_name,
                        "abs_delta": abs(float(lv) - float(rv)),
                    }
                )

    return diffs


def _build_state(db_path: Path) -> dict[str, Any]:
    conn = _connect_ro(db_path)
    try:
        return {
            "db_path": str(db_path),
            "balance": _load_balance(conn),
            "positions": _reconstruct_positions(conn),
            "open_orders": _load_open_orders(conn),
        }
    finally:
        conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit live/paper/snapshot state alignment.")
    parser.add_argument("--live-db", required=True, help="Path to live SQLite DB")
    parser.add_argument("--paper-db", required=True, help="Path to paper SQLite DB")
    parser.add_argument("--snapshot", help="Optional canonical snapshot JSON path")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL, help="Absolute numeric tolerance")
    parser.add_argument("--output", help="Optional output JSON path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_db = Path(args.live_db).expanduser().resolve()
    paper_db = Path(args.paper_db).expanduser().resolve()
    if not live_db.exists():
        parser.error(f"live DB not found: {live_db}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")

    live_state = _build_state(live_db)
    paper_state = _build_state(paper_db)

    snapshot_state: dict[str, Any] | None = None
    if args.snapshot:
        snapshot_path = Path(args.snapshot).expanduser().resolve()
        if not snapshot_path.exists():
            parser.error(f"snapshot not found: {snapshot_path}")
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot_state = {
            "snapshot_path": str(snapshot_path),
            "balance": float(payload.get("balance") or 0.0),
            "positions": payload.get("positions") or [],
            "open_orders": (payload.get("canonical") or {}).get("open_orders") or [],
        }

    diffs: list[dict[str, Any]] = []

    if not _almost_equal(live_state["balance"], paper_state["balance"], args.tolerance):
        diffs.append(
            {
                "classification": "numeric_policy_divergence",
                "kind": "balance_mismatch",
                "left_source": "live_db",
                "right_source": "paper_db",
                "left": float(live_state["balance"]),
                "right": float(paper_state["balance"]),
                "abs_delta": abs(float(live_state["balance"]) - float(paper_state["balance"])),
            }
        )

    diffs.extend(
        _compare_positions(
            live_state["positions"],
            paper_state["positions"],
            left_name="live_db",
            right_name="paper_db",
            tol=float(args.tolerance),
        )
    )

    if snapshot_state is not None:
        if not _almost_equal(live_state["balance"], snapshot_state["balance"], args.tolerance):
            diffs.append(
                {
                    "classification": "state_initialisation_gap",
                    "kind": "snapshot_balance_mismatch",
                    "left_source": "live_db",
                    "right_source": "snapshot",
                    "left": float(live_state["balance"]),
                    "right": float(snapshot_state["balance"]),
                    "abs_delta": abs(float(live_state["balance"]) - float(snapshot_state["balance"])),
                }
            )

        diffs.extend(
            _compare_positions(
                live_state["positions"],
                snapshot_state["positions"],
                left_name="live_db",
                right_name="snapshot",
                tol=float(args.tolerance),
            )
        )

    report = {
        "ok": len(diffs) == 0,
        "generated_at_ms": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "tolerance": float(args.tolerance),
        "summary": {
            "live_balance": float(live_state["balance"]),
            "paper_balance": float(paper_state["balance"]),
            "live_positions": len(live_state["positions"]),
            "paper_positions": len(paper_state["positions"]),
            "live_open_orders": len(live_state["open_orders"]),
            "paper_open_orders": len(paper_state["open_orders"]),
            "diff_count": len(diffs),
        },
        "diffs": diffs,
    }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(out_path.as_posix())
    else:
        print(payload)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
