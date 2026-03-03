import sqlite3

from engine.oms import LiveOms


class _DummyExecutor:
    def get_positions(self, force: bool = False):
        return {}


class _DummyTrader:
    def __init__(self) -> None:
        self.executor = _DummyExecutor()
        self.balance = 10_000.0
        self.positions = {}

    def pop_pending(self, symbol: str):
        return {}


def _ensure_trades_schema(db_path: str) -> None:
    """Create the minimal trades schema needed by OMS fill ingestion tests."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            type TEXT,
            action TEXT,
            price REAL,
            size REAL,
            notional REAL,
            reason TEXT,
            confidence TEXT,
            reason_code TEXT,
            pnl REAL,
            fee_usd REAL,
            fee_token TEXT,
            fee_rate REAL,
            balance REAL,
            entry_atr REAL,
            leverage REAL,
            margin_used REAL,
            meta_json TEXT,
            run_fingerprint TEXT,
            fill_hash TEXT,
            fill_tid INTEGER
        )
        """
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_fill_hash_tid ON trades(fill_hash, fill_tid)"
    )
    con.commit()
    con.close()


def _insert_pending_decision(db_path: str, *, decision_id: str, symbol: str = "BTC", event_type: str = "entry_signal") -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_events (
            id TEXT PRIMARY KEY,
            timestamp_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            decision_phase TEXT NOT NULL,
            parent_decision_id TEXT,
            trade_id INTEGER,
            triggered_by TEXT,
            action_taken TEXT,
            rejection_reason TEXT,
            reason_code TEXT,
            config_fingerprint TEXT,
            run_fingerprint TEXT,
            context_json TEXT
        )
        """
    )
    cur.execute(
        """
        INSERT OR REPLACE INTO decision_events (
            id, timestamp_ms, symbol, event_type, status, decision_phase,
            parent_decision_id, trade_id, triggered_by, action_taken,
            rejection_reason, reason_code, config_fingerprint, run_fingerprint, context_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(decision_id),
            1_700_000_000_000,
            str(symbol),
            str(event_type),
            "hold",
            "execution",
            None,
            None,
            "schedule",
            "open",
            None,
            None,
            None,
            None,
            None,
        ),
    )
    con.commit()
    con.close()


def test_mark_submit_unknown_makes_intent_matchable_by_time_proximity(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    oms = LiveOms(db_path=str(db_path))
    intent = oms.create_intent(
        symbol="BTC",
        action="ADD",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta=None,
        dedupe_open=False,
    )

    oms.mark_submit_unknown(
        intent,
        symbol="BTC",
        side="BUY",
        order_type="market_open",
        reduce_only=False,
        requested_size=1.0,
        error="timeout",
    )

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT status, sent_ts_ms FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    row = cur.fetchone()
    con.close()
    assert row is not None
    status, sent_ts_ms = row
    assert status == "UNKNOWN"
    assert sent_ts_ms is not None and int(sent_ts_ms) > 0

    found = oms.store.find_pending_intent(
        symbol="BTC",
        action="ADD",
        side="BUY",
        t_ms=int(sent_ts_ms),
        ttl_ms=60_000,
    )
    assert found == intent.intent_id


def test_expire_old_sent_intents_expires_unknown_intents(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    oms = LiveOms(db_path=str(db_path))
    intent = oms.create_intent(
        symbol="BTC",
        action="ADD",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta=None,
        dedupe_open=False,
    )
    oms.mark_submit_unknown(
        intent,
        symbol="BTC",
        side="BUY",
        order_type="market_open",
        reduce_only=False,
        requested_size=1.0,
        error="timeout",
    )

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT sent_ts_ms FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    sent_ts_ms = int(cur.fetchone()[0])
    con.close()

    n = oms.store.expire_old_sent_intents(older_than_ms=sent_ts_ms + 1)
    assert n == 1

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT status FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    status = cur.fetchone()[0]
    con.close()
    assert status == "EXPIRED"


def test_mark_failed_finalises_pending_decision(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    decision_id = "dec_failed_1"
    _insert_pending_decision(str(db_path), decision_id=decision_id, symbol="BTC", event_type="entry_signal")

    oms = LiveOms(db_path=str(db_path))
    intent = oms.create_intent(
        symbol="BTC",
        action="OPEN",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta={"decision": {"event_id": decision_id, "action": "open"}},
        dedupe_open=False,
    )

    oms.mark_failed(intent, error="market_open rejected")

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT status FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    intent_status = cur.fetchone()[0]
    cur.execute(
        "SELECT status, decision_phase, rejection_reason, reason_code FROM decision_events WHERE id = ? LIMIT 1",
        (decision_id,),
    )
    row = cur.fetchone()
    con.close()

    assert intent_status == "REJECTED"
    assert row is not None
    assert row[0] == "rejected"
    assert row[1] == "execution"
    assert "market_open rejected" in str(row[2] or "")
    assert row[3] == "execution_rejected"


def test_reconcile_expire_unknown_finalises_pending_decision(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    decision_id = "dec_unknown_expired_1"
    _insert_pending_decision(str(db_path), decision_id=decision_id, symbol="BTC", event_type="entry_signal")

    oms = LiveOms(db_path=str(db_path), expire_sent_after_ms=1000)
    intent = oms.create_intent(
        symbol="BTC",
        action="OPEN",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta={"decision": {"event_id": decision_id, "action": "open"}},
        dedupe_open=False,
    )

    oms.mark_submit_unknown(
        intent,
        symbol="BTC",
        side="BUY",
        order_type="market_open",
        reduce_only=False,
        requested_size=1.0,
        error="timeout",
    )
    oms.store.update_intent(intent.intent_id, sent_ts_ms=1)
    oms.reconcile(trader=None)

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT status FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    intent_status = cur.fetchone()[0]
    cur.execute(
        "SELECT status, decision_phase, rejection_reason, reason_code FROM decision_events WHERE id = ? LIMIT 1",
        (decision_id,),
    )
    row = cur.fetchone()
    con.close()

    assert intent_status == "EXPIRED"
    assert row is not None
    assert row[0] == "rejected"
    assert row[1] == "execution"
    assert "expired without fill" in str(row[2] or "").lower()
    assert row[3] == "execution_submit_unknown_expired"


def test_reconcile_expire_partial_marks_decision_executed(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    decision_id = "dec_partial_expired_1"
    _insert_pending_decision(str(db_path), decision_id=decision_id, symbol="BTC", event_type="entry_signal")

    oms = LiveOms(db_path=str(db_path), expire_sent_after_ms=1000)
    intent = oms.create_intent(
        symbol="BTC",
        action="OPEN",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta={"decision": {"event_id": decision_id, "action": "open"}},
        dedupe_open=False,
    )

    oms.store.update_intent(intent.intent_id, status="PARTIAL", sent_ts_ms=1, last_error="remaining expired")
    oms.reconcile(trader=None)

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT status FROM oms_intents WHERE intent_id = ? LIMIT 1", (intent.intent_id,))
    intent_status = cur.fetchone()[0]
    cur.execute(
        "SELECT status, decision_phase, rejection_reason, reason_code FROM decision_events WHERE id = ? LIMIT 1",
        (decision_id,),
    )
    row = cur.fetchone()
    con.close()

    assert intent_status == "EXPIRED"
    assert row is not None
    assert row[0] == "executed"
    assert row[1] == "execution"
    assert "residual expired" in str(row[2] or "").lower()
    assert row[3] == "execution_partial_expired"


def test_late_fill_promotes_rejected_decision_to_executed(tmp_path, monkeypatch):
    db_path = tmp_path / "oms.db"
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(db_path))

    decision_id = "dec_late_fill_1"
    _insert_pending_decision(str(db_path), decision_id=decision_id, symbol="BTC", event_type="entry_signal")
    _ensure_trades_schema(str(db_path))

    oms = LiveOms(db_path=str(db_path))
    intent = oms.create_intent(
        symbol="BTC",
        action="OPEN",
        side="BUY",
        requested_size=1.0,
        requested_notional=100.0,
        leverage=3.0,
        decision_ts=1700000000000,
        reason="test",
        confidence="high",
        entry_atr=None,
        meta={"decision": {"event_id": decision_id, "action": "open"}},
        dedupe_open=False,
    )

    # Simulate a rejected dispatch first.
    oms.mark_failed(intent, error="market_open rejected")

    # Then simulate an eventual late fill for the same order.
    oms.mark_sent(
        intent,
        symbol="BTC",
        side="BUY",
        order_type="market_open",
        reduce_only=False,
        requested_size=1.0,
        result={"oid": "9001"},
    )
    inserted = oms.process_user_fills(
        trader=_DummyTrader(),
        fills=[
            {
                "coin": "BTC",
                "tid": 123456,
                "hash": "0xlatefill123",
                "time": 1700000001000,
                "px": "100.0",
                "sz": "1.0",
                "dir": "Open Long",
                "startPosition": "0",
                "fee": "0.1",
                "closedPnl": "0.0",
                "oid": "9001",
            }
        ],
    )
    assert inserted == 1

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        "SELECT status, decision_phase, rejection_reason, reason_code, trade_id FROM decision_events WHERE id = ? LIMIT 1",
        (decision_id,),
    )
    row = cur.fetchone()
    con.close()

    assert row is not None
    assert row[0] == "executed"
    assert row[1] == "execution"
    # Late fills should clear stale rejection metadata.
    assert row[2] in (None, "")
    assert row[3] in (None, "")
    assert int(row[4]) > 0
