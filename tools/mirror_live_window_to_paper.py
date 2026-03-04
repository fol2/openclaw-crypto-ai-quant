#!/usr/bin/env python3
"""Mirror live window actions and decision rows into a paper SQLite database.

This utility is intended for deterministic replay harness workflows where paper
coverage for a historical live window must be explicit and reproducible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

TRADE_ACTIONS = {"OPEN", "ADD", "REDUCE", "CLOSE", "FUNDING"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror live window trade + decision rows into paper DB."
    )
    parser.add_argument("--live-db", required=True, help="Path to live SQLite DB")
    parser.add_argument("--paper-db", required=True, help="Path to paper SQLite DB")
    parser.add_argument("--from-ts", type=int, required=True, help="Window start timestamp (ms, inclusive)")
    parser.add_argument("--to-ts", type=int, required=True, help="Window end timestamp (ms, inclusive)")
    parser.add_argument(
        "--replace-window",
        action="store_true",
        default=False,
        help=(
            "Delete paper rows inside the window before applying mirror rows. "
            "Recommended for deterministic harness runs."
        ),
    )
    parser.add_argument(
        "--skip-decision-events",
        action="store_true",
        default=False,
        help="Mirror trades only; do not mirror decision_events rows.",
    )
    parser.add_argument(
        "--exclude-funding-events",
        action="store_true",
        default=False,
        help="Exclude FUNDING actions from trade mirroring.",
    )
    parser.add_argument(
        "--allow-overwrite-existing",
        action="store_true",
        default=False,
        help=(
            "Allow overwriting colliding paper ids even when they were not produced "
            "by this mirror tool."
        ),
    )
    parser.add_argument(
        "--mirror-tag",
        default="live_window_replay",
        help="Mirror tag written into meta/context payloads (default: live_window_replay)",
    )
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    return parser


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_rw(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"] if isinstance(row, sqlite3.Row) else row[1]).strip() for row in rows}


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
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except Exception:
        return 0


def _in_window(ts_ms: int, from_ts: int, to_ts: int) -> bool:
    return int(from_ts) <= int(ts_ms) <= int(to_ts)


def _chunked(values: list[Any], size: int = 400) -> list[list[Any]]:
    out: list[list[Any]] = []
    for idx in range(0, len(values), max(1, int(size))):
        out.append(values[idx : idx + max(1, int(size))])
    return out


def _load_rows_with_nullable_columns(
    conn: sqlite3.Connection,
    *,
    table: str,
    requested_columns: list[str],
    where_sql: str = "",
    params: tuple[Any, ...] = (),
) -> list[sqlite3.Row]:
    present = _table_columns(conn, table)
    select_parts: list[str] = []
    for col in requested_columns:
        if col in present:
            select_parts.append(col)
        else:
            select_parts.append(f"NULL AS {col}")
    sql = f"SELECT {', '.join(select_parts)} FROM {table} {where_sql}"
    return conn.execute(sql, params).fetchall()


def _merge_json_marker(raw_value: Any, *, marker: dict[str, Any]) -> str:
    base: dict[str, Any] = {}
    raw_text = str(raw_value or "").strip()
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                base = dict(parsed)
            else:
                base = {"_raw_payload": raw_text}
        except Exception:
            base = {"_raw_payload": raw_text}
    base.update(marker)
    return json.dumps(base, sort_keys=True, separators=(",", ":"))


def _has_marker(raw_value: Any, *, mirror_tag: str) -> bool:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return False
    try:
        payload = json.loads(raw_text)
    except Exception:
        return False
    return isinstance(payload, dict) and str(payload.get("mirror_source") or "").strip() == str(mirror_tag)


def _load_live_trades(
    conn: sqlite3.Connection,
    *,
    from_ts: int,
    to_ts: int,
    include_funding: bool,
) -> list[dict[str, Any]]:
    requested_cols = [
        "id",
        "timestamp",
        "symbol",
        "type",
        "action",
        "price",
        "size",
        "notional",
        "reason",
        "confidence",
        "pnl",
        "fee_usd",
        "fee_token",
        "fee_rate",
        "balance",
        "entry_atr",
        "leverage",
        "margin_used",
        "meta_json",
        "fill_hash",
        "fill_tid",
        "run_fingerprint",
        "reason_code",
    ]
    rows = _load_rows_with_nullable_columns(
        conn,
        table="trades",
        requested_columns=requested_cols,
        where_sql="ORDER BY id ASC",
    )

    allowed_actions = set(TRADE_ACTIONS if include_funding else (TRADE_ACTIONS - {"FUNDING"}))
    out: list[dict[str, Any]] = []
    for row in rows:
        action = str(row["action"] or "").strip().upper()
        if action not in allowed_actions:
            continue
        ts_ms = _parse_timestamp_ms(row["timestamp"])
        if not _in_window(ts_ms, from_ts, to_ts):
            continue
        out.append({key: row[key] for key in requested_cols})
    return out


def _load_live_decision_events(
    conn: sqlite3.Connection,
    *,
    from_ts: int,
    to_ts: int,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "decision_events"):
        return []
    requested_cols = [
        "id",
        "timestamp_ms",
        "symbol",
        "event_type",
        "status",
        "decision_phase",
        "parent_decision_id",
        "trade_id",
        "triggered_by",
        "action_taken",
        "rejection_reason",
        "reason_code",
        "config_fingerprint",
        "run_fingerprint",
        "context_json",
    ]
    rows = _load_rows_with_nullable_columns(
        conn,
        table="decision_events",
        requested_columns=requested_cols,
        where_sql="WHERE timestamp_ms >= ? AND timestamp_ms <= ? ORDER BY timestamp_ms ASC, id ASC",
        params=(int(from_ts), int(to_ts)),
    )
    return [{key: row[key] for key in requested_cols} for row in rows]


def _paper_trade_ids_in_window(
    conn: sqlite3.Connection,
    *,
    from_ts: int,
    to_ts: int,
) -> list[int]:
    rows = conn.execute(
        "SELECT id, timestamp, action FROM trades ORDER BY id ASC"
    ).fetchall()
    out: list[int] = []
    for row in rows:
        action = str(row["action"] or "").strip().upper()
        if action not in TRADE_ACTIONS:
            continue
        ts_ms = _parse_timestamp_ms(row["timestamp"])
        if _in_window(ts_ms, from_ts, to_ts):
            out.append(int(row["id"]))
    return out


def _delete_trade_ids(conn: sqlite3.Connection, ids: list[int]) -> int:
    if not ids:
        return 0
    deleted = 0
    for chunk in _chunked(ids):
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(f"DELETE FROM trades WHERE id IN ({placeholders})", tuple(chunk))
        deleted += int(cur.rowcount or 0)
    return int(deleted)


def _fetch_existing_json_by_id(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    ids: list[Any],
    json_column: str,
) -> dict[str, str]:
    if not ids:
        return {}
    out: dict[str, str] = {}
    for chunk in _chunked(list(ids)):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT {id_column}, {json_column} FROM {table} WHERE {id_column} IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        for row in rows:
            out[str(row[id_column])] = str(row[json_column] or "")
    return out


def _upsert_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    rows: list[dict[str, Any]],
    paper_columns: set[str],
) -> int:
    if not rows:
        return 0
    insert_cols = [col for col in rows[0].keys() if col in paper_columns]
    placeholders = ",".join("?" for _ in insert_cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(insert_cols)}) VALUES ({placeholders})"
    payload = [tuple(row.get(col) for col in insert_cols) for row in rows]
    conn.executemany(sql, payload)
    return len(rows)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    live_db = Path(args.live_db).expanduser().resolve()
    paper_db = Path(args.paper_db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    from_ts = int(args.from_ts)
    to_ts = int(args.to_ts)
    replace_window = bool(args.replace_window)
    include_funding = not bool(args.exclude_funding_events)
    mirror_decisions = not bool(args.skip_decision_events)
    mirror_tag = str(args.mirror_tag or "").strip() or "live_window_replay"

    if not live_db.exists():
        parser.error(f"live DB not found: {live_db}")
    if not paper_db.exists():
        parser.error(f"paper DB not found: {paper_db}")
    if from_ts > to_ts:
        parser.error("from-ts must be <= to-ts")

    live_conn = _connect_ro(live_db)
    try:
        live_trade_rows = _load_live_trades(
            live_conn,
            from_ts=from_ts,
            to_ts=to_ts,
            include_funding=include_funding,
        )
        live_decision_rows = (
            _load_live_decision_events(live_conn, from_ts=from_ts, to_ts=to_ts)
            if mirror_decisions
            else []
        )
    finally:
        live_conn.close()

    generated_at_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    marker = {
        "mirror_source": mirror_tag,
        "mirror_window_from_ts": int(from_ts),
        "mirror_window_to_ts": int(to_ts),
    }

    deleted_trade_rows = 0
    deleted_decision_rows = 0
    trade_id_collisions = 0
    decision_id_collisions = 0
    trade_collision_non_marker = 0
    decision_collision_non_marker = 0
    mirrored_trade_rows = 0
    mirrored_decision_rows = 0

    paper_conn = _connect_rw(paper_db)
    try:
        if not _table_exists(paper_conn, "trades"):
            raise RuntimeError("paper DB missing trades table")
        paper_trade_cols = _table_columns(paper_conn, "trades")
        if not paper_trade_cols:
            raise RuntimeError("paper DB trades table has no columns")

        has_decision_table = _table_exists(paper_conn, "decision_events")
        paper_decision_cols = _table_columns(paper_conn, "decision_events") if has_decision_table else set()

        if replace_window:
            paper_trade_ids = _paper_trade_ids_in_window(paper_conn, from_ts=from_ts, to_ts=to_ts)
            deleted_trade_rows = _delete_trade_ids(paper_conn, paper_trade_ids)
            if has_decision_table:
                cur = paper_conn.execute(
                    "DELETE FROM decision_events WHERE timestamp_ms >= ? AND timestamp_ms <= ?",
                    (int(from_ts), int(to_ts)),
                )
                deleted_decision_rows = int(cur.rowcount or 0)

        trade_ids = [int(row["id"]) for row in live_trade_rows if row.get("id") is not None]
        existing_trade_json = _fetch_existing_json_by_id(
            paper_conn,
            table="trades",
            id_column="id",
            ids=[int(v) for v in trade_ids],
            json_column="meta_json",
        )
        trade_id_collisions = len(existing_trade_json)
        trade_collision_non_marker = sum(
            1 for raw in existing_trade_json.values() if not _has_marker(raw, mirror_tag=mirror_tag)
        )

        decision_ids = [str(row["id"]) for row in live_decision_rows if str(row.get("id") or "").strip()]
        existing_decision_json = (
            _fetch_existing_json_by_id(
                paper_conn,
                table="decision_events",
                id_column="id",
                ids=decision_ids,
                json_column="context_json",
            )
            if has_decision_table and decision_ids
            else {}
        )
        decision_id_collisions = len(existing_decision_json)
        decision_collision_non_marker = sum(
            1 for raw in existing_decision_json.values() if not _has_marker(raw, mirror_tag=mirror_tag)
        )

        if (
            not bool(args.allow_overwrite_existing)
            and (trade_collision_non_marker > 0 or decision_collision_non_marker > 0)
        ):
            raise RuntimeError(
                "detected colliding paper ids without mirror marker; "
                "rerun with --replace-window or --allow-overwrite-existing"
            )

        trade_payload: list[dict[str, Any]] = []
        for row in live_trade_rows:
            payload = dict(row)
            payload["meta_json"] = _merge_json_marker(payload.get("meta_json"), marker=marker)
            trade_payload.append(payload)

        decision_payload: list[dict[str, Any]] = []
        if mirror_decisions and has_decision_table:
            for row in live_decision_rows:
                payload = dict(row)
                payload["context_json"] = _merge_json_marker(payload.get("context_json"), marker=marker)
                decision_payload.append(payload)

        mirrored_trade_rows = _upsert_rows(
            paper_conn,
            table="trades",
            rows=trade_payload,
            paper_columns=paper_trade_cols,
        )
        if mirror_decisions and has_decision_table:
            mirrored_decision_rows = _upsert_rows(
                paper_conn,
                table="decision_events",
                rows=decision_payload,
                paper_columns=paper_decision_cols,
            )

        paper_conn.commit()
    except Exception as exc:
        paper_conn.rollback()
        report = {
            "schema_version": 1,
            "generated_at_ms": generated_at_ms,
            "inputs": {
                "live_db": str(live_db),
                "paper_db": str(paper_db),
                "from_ts": int(from_ts),
                "to_ts": int(to_ts),
                "replace_window": replace_window,
                "skip_decision_events": bool(args.skip_decision_events),
                "exclude_funding_events": bool(args.exclude_funding_events),
                "allow_overwrite_existing": bool(args.allow_overwrite_existing),
                "mirror_tag": mirror_tag,
            },
            "counts": {
                "live_trade_rows_window": int(len(live_trade_rows)),
                "live_decision_rows_window": int(len(live_decision_rows)),
                "paper_trade_rows_deleted_window": int(deleted_trade_rows),
                "paper_decision_rows_deleted_window": int(deleted_decision_rows),
                "trade_id_collisions": int(trade_id_collisions),
                "decision_id_collisions": int(decision_id_collisions),
                "trade_collision_non_marker": int(trade_collision_non_marker),
                "decision_collision_non_marker": int(decision_collision_non_marker),
                "mirrored_trade_rows": int(mirrored_trade_rows),
                "mirrored_decision_rows": int(mirrored_decision_rows),
            },
            "status": {
                "ok": False,
                "error": str(exc),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path.as_posix())
        print(f"[mirror] failed: {exc}", file=sys.stderr)
        return 1
    finally:
        paper_conn.close()

    report = {
        "schema_version": 1,
        "generated_at_ms": generated_at_ms,
        "inputs": {
            "live_db": str(live_db),
            "paper_db": str(paper_db),
            "from_ts": int(from_ts),
            "to_ts": int(to_ts),
            "replace_window": replace_window,
            "skip_decision_events": bool(args.skip_decision_events),
            "exclude_funding_events": bool(args.exclude_funding_events),
            "allow_overwrite_existing": bool(args.allow_overwrite_existing),
            "mirror_tag": mirror_tag,
        },
        "counts": {
            "live_trade_rows_window": int(len(live_trade_rows)),
            "live_decision_rows_window": int(len(live_decision_rows)),
            "paper_trade_rows_deleted_window": int(deleted_trade_rows),
            "paper_decision_rows_deleted_window": int(deleted_decision_rows),
            "trade_id_collisions": int(trade_id_collisions),
            "decision_id_collisions": int(decision_id_collisions),
            "trade_collision_non_marker": int(trade_collision_non_marker),
            "decision_collision_non_marker": int(decision_collision_non_marker),
            "mirrored_trade_rows": int(mirrored_trade_rows),
            "mirrored_decision_rows": int(mirrored_decision_rows),
        },
        "status": {
            "ok": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
