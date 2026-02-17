from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import tools.reality_check as reality_check


def _setup_reality_db(db_path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "CREATE TABLE audit_events (symbol TEXT, event TEXT, timestamp TEXT)"
        )
        con.execute(
            "CREATE TABLE trades ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "symbol TEXT, timestamp TEXT, action TEXT, type TEXT, price REAL, size REAL, reason TEXT, confidence TEXT)"
        )
        con.commit()
    finally:
        con.close()


def test_print_db_reports_missing_database(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.db"
    reality_check._print_db("Paper", missing, symbol="BTC", hours=2.0)
    out = capsys.readouterr().out
    assert "missing db" in out


def test_audit_counts_and_recent_trades_parse_expected_rows(tmp_path) -> None:
    db_path = tmp_path / "reality.db"
    _setup_reality_db(db_path)

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    old_iso = (now_dt - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO audit_events (symbol, event, timestamp) VALUES (?, ?, ?)",
            ("BTC", "entry_signal", now_iso),
        )
        con.execute(
            "INSERT INTO audit_events (symbol, event, timestamp) VALUES (?, ?, ?)",
            ("BTC", "entry_signal", now_iso),
        )
        con.execute(
            "INSERT INTO audit_events (symbol, event, timestamp) VALUES (?, ?, ?)",
            ("BTC", "", now_iso),
        )
        con.execute(
            "INSERT INTO audit_events (symbol, event, timestamp) VALUES (?, ?, ?)",
            ("BTC", "entry_signal", old_iso),
        )
        con.execute(
            "INSERT INTO audit_events (symbol, event, timestamp) VALUES (?, ?, ?)",
            ("ETH", "entry_signal", now_iso),
        )
        con.execute(
            "INSERT INTO trades (symbol, timestamp, action, type, price, size, reason, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("BTC", now_iso, "OPEN", "market_open", 100.0, 0.1, "test-one", "high"),
        )
        con.execute(
            "INSERT INTO trades (symbol, timestamp, action, type, price, size, reason, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("BTC", now_iso, "CLOSE", "market_close", 101.0, 0.1, "test-two", "medium"),
        )
        con.execute(
            "INSERT INTO trades (symbol, timestamp, action, type, price, size, reason, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ETH", now_iso, "OPEN", "market_open", 10.0, 1.0, "other-symbol", "low"),
        )
        con.commit()
    finally:
        con.close()

    ro = reality_check._connect(db_path)
    try:
        counts = reality_check._audit_counts(ro, symbol="BTC", hours=2.0)
        assert counts == [("entry_signal", 2)]

        trades = reality_check._recent_trades(ro, symbol="BTC", limit=10)
        assert len(trades) == 2
        assert trades[0]["action"] == "CLOSE"
        assert trades[0]["reason"] == "test-two"
        assert trades[1]["action"] == "OPEN"
    finally:
        ro.close()


def test_print_db_includes_trade_lines_for_parsed_rows(tmp_path, capsys) -> None:
    db_path = tmp_path / "reality.db"
    _setup_reality_db(db_path)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO trades (symbol, timestamp, action, type, price, size, reason, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("BTC", now_iso, "OPEN", "market_open", 100.0, 0.2, "print-check", "high"),
        )
        con.commit()
    finally:
        con.close()

    reality_check._print_db("Paper", db_path, symbol="BTC", hours=2.0)
    out = capsys.readouterr().out
    assert "recent trades for BTC" in out
    assert "reason=print-check" in out


def test_print_db_reports_open_failure(tmp_path, capsys, monkeypatch) -> None:
    db_path = tmp_path / "exists.db"
    db_path.write_text("", encoding="utf-8")

    def _raise_open_error(_db_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(reality_check, "_connect", _raise_open_error)
    reality_check._print_db("Paper", db_path, symbol="BTC", hours=2.0)
    out = capsys.readouterr().out
    assert "db_open_failed: boom" in out
