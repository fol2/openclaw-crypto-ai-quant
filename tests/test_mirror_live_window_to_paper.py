from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from tools import mirror_live_window_to_paper as mirror_tool


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                type TEXT,
                action TEXT,
                price REAL,
                size REAL,
                notional REAL,
                reason TEXT,
                confidence TEXT,
                pnl REAL,
                fee_usd REAL,
                fee_token TEXT,
                fee_rate REAL,
                balance REAL,
                entry_atr REAL,
                leverage REAL,
                margin_used REAL,
                meta_json TEXT,
                fill_hash TEXT,
                fill_tid INTEGER,
                run_fingerprint TEXT,
                reason_code TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                parent_decision_id TEXT,
                trade_id INTEGER,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                context_json TEXT,
                config_fingerprint TEXT,
                run_fingerprint TEXT,
                reason_code TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_mirror_replaces_window_and_copies_trades_and_decisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    output = tmp_path / "mirror.json"
    _init_db(live_db)
    _init_db(paper_db)

    live_conn = sqlite3.connect(live_db)
    try:
        live_conn.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, type, action, price, size, notional, reason, confidence,
                pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage, margin_used,
                meta_json, fill_hash, fill_tid, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                101,
                "2026-03-03T12:00:00+00:00",
                "ADA",
                "LONG",
                "OPEN",
                1.0,
                10.0,
                10.0,
                "entry",
                "high",
                0.0,
                0.0,
                "USDC",
                0.0,
                1000.0,
                0.0,
                1.0,
                10.0,
                '{"src":"live"}',
                "",
                None,
                "fp-live",
                "entry_signal",
            ),
        )
        live_conn.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, type, action, price, size, notional, reason, confidence,
                pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage, margin_used,
                meta_json, fill_hash, fill_tid, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                102,
                "2026-03-03T12:30:00+00:00",
                "ADA",
                "LONG",
                "FUNDING",
                1.0,
                0.0,
                0.0,
                "funding",
                "low",
                -0.1,
                0.0,
                "USDC",
                0.0,
                999.9,
                0.0,
                1.0,
                10.0,
                "{}",
                "",
                None,
                "fp-live",
                "funding",
            ),
        )
        live_conn.execute(
            """
            INSERT INTO decision_events (
                id, timestamp_ms, symbol, event_type, status, decision_phase, parent_decision_id,
                trade_id, triggered_by, action_taken, rejection_reason, context_json,
                config_fingerprint, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "de-live-1",
                1772539200000,
                "ADA",
                "entry",
                "executed",
                "execution",
                None,
                101,
                "entry_interval",
                "open_long",
                "",
                '{"ctx":"live"}',
                "cfg",
                "fp-live",
                "entry_signal",
            ),
        )
        live_conn.commit()
    finally:
        live_conn.close()

    paper_conn = sqlite3.connect(paper_db)
    try:
        paper_conn.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, type, action, price, size, notional, reason, confidence,
                pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage, margin_used,
                meta_json, fill_hash, fill_tid, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "2026-03-03T12:10:00+00:00",
                "DOGE",
                "SHORT",
                "OPEN",
                0.0,
                0.0,
                0.0,
                "old",
                "low",
                0.0,
                0.0,
                "USDC",
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                '{"old":true}',
                "",
                None,
                "fp-old",
                "entry_signal",
            ),
        )
        paper_conn.execute(
            """
            INSERT INTO decision_events (
                id, timestamp_ms, symbol, event_type, status, decision_phase, parent_decision_id,
                trade_id, triggered_by, action_taken, rejection_reason, context_json,
                config_fingerprint, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "de-old-1",
                1772539200500,
                "DOGE",
                "entry",
                "executed",
                "execution",
                None,
                1,
                "entry_interval",
                "open_short",
                "",
                '{"old":true}',
                "cfg-old",
                "fp-old",
                "entry_signal",
            ),
        )
        paper_conn.commit()
    finally:
        paper_conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror_live_window_to_paper.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--from-ts",
            "1772539100000",
            "--to-ts",
            "1772541200000",
            "--replace-window",
            "--output",
            str(output),
        ],
    )
    rc = mirror_tool.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["status"]["ok"] is True
    assert report["counts"]["paper_trade_rows_deleted_window"] == 1
    assert report["counts"]["paper_decision_rows_deleted_window"] == 1
    assert report["counts"]["mirrored_trade_rows"] == 2
    assert report["counts"]["mirrored_decision_rows"] == 1

    verify = sqlite3.connect(paper_db)
    try:
        rows = verify.execute(
            "SELECT id, action, meta_json FROM trades WHERE id IN (101, 102) ORDER BY id ASC"
        ).fetchall()
        assert [row[0] for row in rows] == [101, 102]
        assert [row[1] for row in rows] == ["OPEN", "FUNDING"]
        for _, _, raw_meta in rows:
            payload = json.loads(raw_meta)
            assert payload["mirror_source"] == "live_window_replay"

        old_trade = verify.execute("SELECT COUNT(*) FROM trades WHERE id = 1").fetchone()
        assert int(old_trade[0]) == 0

        de = verify.execute(
            "SELECT id, context_json FROM decision_events WHERE id = 'de-live-1'"
        ).fetchone()
        assert de is not None
        ctx = json.loads(str(de[1]))
        assert ctx["mirror_source"] == "live_window_replay"

        old_de = verify.execute(
            "SELECT COUNT(*) FROM decision_events WHERE id = 'de-old-1'"
        ).fetchone()
        assert int(old_de[0]) == 0
    finally:
        verify.close()


def test_mirror_fails_on_non_marker_id_collision_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    output = tmp_path / "mirror_collision.json"
    _init_db(live_db)
    _init_db(paper_db)

    live_conn = sqlite3.connect(live_db)
    try:
        live_conn.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, type, action, price, size, notional, reason, confidence,
                pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage, margin_used,
                meta_json, fill_hash, fill_tid, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                9,
                "2026-03-03T12:00:00+00:00",
                "ADA",
                "LONG",
                "OPEN",
                1.0,
                1.0,
                1.0,
                "entry",
                "high",
                0.0,
                0.0,
                "USDC",
                0.0,
                1000.0,
                0.0,
                1.0,
                1.0,
                "{}",
                "",
                None,
                "fp-live",
                "entry_signal",
            ),
        )
        live_conn.commit()
    finally:
        live_conn.close()

    paper_conn = sqlite3.connect(paper_db)
    try:
        paper_conn.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, type, action, price, size, notional, reason, confidence,
                pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage, margin_used,
                meta_json, fill_hash, fill_tid, run_fingerprint, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                9,
                "2026-03-01T00:00:00+00:00",
                "BTC",
                "SHORT",
                "OPEN",
                0.0,
                0.0,
                0.0,
                "paper-row",
                "low",
                0.0,
                0.0,
                "USDC",
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                '{"source":"paper_live"}',
                "",
                None,
                "fp-paper",
                "entry_signal",
            ),
        )
        paper_conn.commit()
    finally:
        paper_conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror_live_window_to_paper.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--from-ts",
            "1772539100000",
            "--to-ts",
            "1772540000000",
            "--output",
            str(output),
        ],
    )
    rc = mirror_tool.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["status"]["ok"] is False
    assert report["counts"]["trade_collision_non_marker"] == 1
