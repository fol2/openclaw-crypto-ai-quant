"""Tests for decision trace API endpoints in monitor (AQC-805).

Verifies the four new endpoints:
  - GET /api/v2/decisions          — list with filters + pagination
  - GET /api/v2/decisions/{id}     — single decision with context + gates
  - GET /api/v2/trades/{id}/decision-trace — entry-to-exit chain
  - GET /api/v2/decisions/{id}/gates — gate evaluations for a decision
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest


def _stub_missing_modules():
    """Insert stub modules so heavy optional deps don't block import."""
    stubbed: list[str] = []
    optional_packages = [
        "websocket",
        "ta",
        "ta.momentum",
        "ta.trend",
        "ta.volatility",
        "ta.volume",
        "exchange",
        "exchange.ws",
        "exchange.meta",
        "exchange.sidecar",
        "engine",
        "engine.alerting",
        "engine.kernel_shadow_report",
        "engine.event_logger",
        "hyperliquid",
        "hyperliquid.utils",
        "hyperliquid.info",
        "hyperliquid.exchange",
        "eth_account",
        "bt_runtime",
        "heartbeat",
    ]
    for name in optional_packages:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__dict__.setdefault("send_openclaw_message", lambda *a, **kw: None)
            mod.__dict__.setdefault("ShadowReport", type("ShadowReport", (), {}))
            mod.__dict__.setdefault("emit_event", lambda *a, **kw: None)
            mod.__dict__.setdefault("parse_last_heartbeat", lambda *a, **kw: {"ok": False})
            sys.modules[name] = mod
            stubbed.append(name)
    return stubbed


@pytest.fixture()
def monitor_db(tmp_path, monkeypatch):
    """Create a temp DB with decision traceability tables and seed data.

    Returns (db_path, module) where module is monitor.server reloaded to
    point at the temp DB.
    """
    db_path = tmp_path / "trading_engine.db"
    monkeypatch.setenv("AIQ_MONITOR_PAPER_DB", str(db_path))

    stubbed = _stub_missing_modules()

    # Create the DB and schema.
    con = sqlite3.connect(str(db_path))
    con.executescript("""
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
            context_json TEXT
        );
        CREATE TABLE IF NOT EXISTS decision_context (
            decision_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            price REAL, rsi REAL, adx REAL, adx_slope REAL,
            macd_hist REAL, ema_fast REAL, ema_slow REAL, ema_macro REAL,
            bb_width_ratio REAL, stoch_k REAL, stoch_d REAL,
            atr REAL, atr_slope REAL, volume REAL, vol_sma REAL,
            rsi_entry_threshold REAL,
            min_adx_threshold REAL,
            sl_price REAL, tp_price REAL, trailing_sl REAL,
            gate_ranging INTEGER, gate_anomaly INTEGER, gate_extension INTEGER,
            gate_adx INTEGER, gate_volume INTEGER, gate_adx_rising INTEGER,
            gate_btc_alignment INTEGER,
            bullish_alignment INTEGER, bearish_alignment INTEGER,
            FOREIGN KEY (decision_id) REFERENCES decision_events(id)
        );
        CREATE TABLE IF NOT EXISTS gate_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL,
            gate_name TEXT NOT NULL,
            gate_passed INTEGER NOT NULL,
            metric_value REAL,
            threshold_value REAL,
            operator TEXT,
            explanation TEXT,
            FOREIGN KEY (decision_id) REFERENCES decision_events(id)
        );
        CREATE TABLE IF NOT EXISTS decision_lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_decision_id TEXT,
            entry_trade_id INTEGER,
            exit_decision_id TEXT,
            exit_trade_id INTEGER,
            exit_reason TEXT,
            duration_ms INTEGER,
            FOREIGN KEY (signal_decision_id) REFERENCES decision_events(id),
            FOREIGN KEY (exit_decision_id) REFERENCES decision_events(id)
        );
        CREATE INDEX IF NOT EXISTS idx_de_symbol_ts ON decision_events(symbol, timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_de_event_type ON decision_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_de_trade_id ON decision_events(trade_id);
        CREATE INDEX IF NOT EXISTS idx_ge_decision ON gate_evaluations(decision_id);
        CREATE INDEX IF NOT EXISTS idx_dl_entry ON decision_lineage(entry_trade_id);
        CREATE INDEX IF NOT EXISTS idx_dl_exit ON decision_lineage(exit_trade_id);
    """)
    con.commit()
    con.close()

    # Import (or reload) the monitor server module.
    mod_name = "monitor.server"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        import monitor.server as mod  # type: ignore[import-untyped]

    yield str(db_path), mod

    for name in stubbed:
        sys.modules.pop(name, None)


def _seed_decision_events(db_path: str, events: list[dict]) -> None:
    """Insert decision_events rows."""
    con = sqlite3.connect(db_path)
    for e in events:
        con.execute(
            """
            INSERT INTO decision_events
            (id, timestamp_ms, symbol, event_type, status, decision_phase,
             parent_decision_id, trade_id, triggered_by, action_taken,
             rejection_reason, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e["id"], e["timestamp_ms"], e["symbol"], e["event_type"],
                e["status"], e["decision_phase"],
                e.get("parent_decision_id"), e.get("trade_id"),
                e.get("triggered_by"), e.get("action_taken"),
                e.get("rejection_reason"), e.get("context_json"),
            ),
        )
    con.commit()
    con.close()


def _seed_decision_context(db_path: str, rows: list[dict]) -> None:
    """Insert decision_context rows."""
    con = sqlite3.connect(db_path)
    for r in rows:
        con.execute(
            """
            INSERT INTO decision_context
            (decision_id, symbol, price, rsi, adx, atr, gate_adx, gate_volume,
             bullish_alignment, bearish_alignment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["decision_id"], r["symbol"], r.get("price"), r.get("rsi"),
                r.get("adx"), r.get("atr"), r.get("gate_adx"), r.get("gate_volume"),
                r.get("bullish_alignment"), r.get("bearish_alignment"),
            ),
        )
    con.commit()
    con.close()


def _seed_gate_evaluations(db_path: str, rows: list[dict]) -> None:
    """Insert gate_evaluations rows."""
    con = sqlite3.connect(db_path)
    for r in rows:
        con.execute(
            """
            INSERT INTO gate_evaluations
            (decision_id, gate_name, gate_passed, metric_value, threshold_value, operator, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["decision_id"], r["gate_name"], r["gate_passed"],
                r.get("metric_value"), r.get("threshold_value"),
                r.get("operator"), r.get("explanation"),
            ),
        )
    con.commit()
    con.close()


def _seed_lineage(db_path: str, rows: list[dict]) -> None:
    """Insert decision_lineage rows."""
    con = sqlite3.connect(db_path)
    for r in rows:
        con.execute(
            """
            INSERT INTO decision_lineage
            (signal_decision_id, entry_trade_id, exit_decision_id, exit_trade_id, exit_reason, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("signal_decision_id"), r.get("entry_trade_id"),
                r.get("exit_decision_id"), r.get("exit_trade_id"),
                r.get("exit_reason"), r.get("duration_ms"),
            ),
        )
    con.commit()
    con.close()


# ── GET /api/v2/decisions (list) ─────────────────────────────────────────


class TestDecisionsList:

    def test_empty_table(self, monitor_db):
        db_path, mod = monitor_db
        result = mod.build_decisions_list("paper")
        assert result["data"] == []
        assert result["total"] == 0
        assert result["limit"] == 100
        assert result["offset"] == 0

    def test_list_all(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "D001", "timestamp_ms": now_ms - 2000, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
            {"id": "D002", "timestamp_ms": now_ms - 1000, "symbol": "BTC", "event_type": "gate_block", "status": "blocked", "decision_phase": "gate_evaluation"},
            {"id": "D003", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "fill", "status": "executed", "decision_phase": "execution"},
        ])
        result = mod.build_decisions_list("paper")
        assert result["total"] == 3
        assert len(result["data"]) == 3
        # Results ordered by timestamp_ms DESC.
        assert result["data"][0]["id"] == "D003"
        assert result["data"][2]["id"] == "D001"

    def test_filter_by_symbol(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "D001", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
            {"id": "D002", "timestamp_ms": now_ms, "symbol": "BTC", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
        ])
        result = mod.build_decisions_list("paper", symbol="ETH")
        assert result["total"] == 1
        assert result["data"][0]["symbol"] == "ETH"

    def test_filter_by_time_window(self, monitor_db):
        db_path, mod = monitor_db
        _seed_decision_events(db_path, [
            {"id": "D001", "timestamp_ms": 1000, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
            {"id": "D002", "timestamp_ms": 2000, "symbol": "ETH", "event_type": "fill", "status": "executed", "decision_phase": "execution"},
            {"id": "D003", "timestamp_ms": 3000, "symbol": "ETH", "event_type": "exit_check", "status": "executed", "decision_phase": "execution"},
        ])
        result = mod.build_decisions_list("paper", start_ms=1500, end_ms=2500)
        assert result["total"] == 1
        assert result["data"][0]["id"] == "D002"

    def test_filter_by_event_type(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "D001", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
            {"id": "D002", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "gate_block", "status": "blocked", "decision_phase": "gate_evaluation"},
        ])
        result = mod.build_decisions_list("paper", event_type="gate_block")
        assert result["total"] == 1
        assert result["data"][0]["event_type"] == "gate_block"

    def test_filter_by_status(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "D001", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"},
            {"id": "D002", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "gate_block", "status": "blocked", "decision_phase": "gate_evaluation"},
        ])
        result = mod.build_decisions_list("paper", status="blocked")
        assert result["total"] == 1
        assert result["data"][0]["status"] == "blocked"

    def test_pagination(self, monitor_db):
        db_path, mod = monitor_db
        events = [
            {"id": f"D{i:03d}", "timestamp_ms": 1000 + i, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation"}
            for i in range(10)
        ]
        _seed_decision_events(db_path, events)

        # Page 1: limit=3, offset=0
        result = mod.build_decisions_list("paper", limit=3, offset=0)
        assert result["total"] == 10
        assert len(result["data"]) == 3
        assert result["limit"] == 3
        assert result["offset"] == 0

        # Page 2: limit=3, offset=3
        result2 = mod.build_decisions_list("paper", limit=3, offset=3)
        assert result2["total"] == 10
        assert len(result2["data"]) == 3
        # No overlap with page 1.
        ids_p1 = {r["id"] for r in result["data"]}
        ids_p2 = {r["id"] for r in result2["data"]}
        assert ids_p1.isdisjoint(ids_p2)

    def test_limit_clamped(self, monitor_db):
        db_path, mod = monitor_db
        result = mod.build_decisions_list("paper", limit=5000)
        assert result["limit"] == 1000

        result2 = mod.build_decisions_list("paper", limit=-5)
        assert result2["limit"] == 1

    def test_db_missing(self, monitor_db, monkeypatch):
        _db_path, mod = monitor_db
        monkeypatch.setenv("AIQ_MONITOR_PAPER_DB", "/nonexistent/path/db.sqlite")
        result = mod.build_decisions_list("paper")
        assert result["data"] == []
        assert result["total"] == 0
        assert result.get("error") == "db_missing"


# ── GET /api/v2/decisions/{id} (detail) ──────────────────────────────────


class TestDecisionDetail:

    def test_not_found(self, monitor_db):
        _db_path, mod = monitor_db
        result = mod.build_decision_detail("paper", "NONEXISTENT")
        assert result is None

    def test_found_with_context_and_gates(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {
                "id": "D100", "timestamp_ms": now_ms, "symbol": "ETH",
                "event_type": "entry_signal", "status": "executed",
                "decision_phase": "signal_generation", "triggered_by": "schedule",
                "action_taken": "open_long",
                "context_json": json.dumps({"rsi": 42.5}),
            },
        ])
        _seed_decision_context(db_path, [
            {"decision_id": "D100", "symbol": "ETH", "price": 2500.0, "rsi": 42.5, "adx": 28.0, "atr": 35.0, "gate_adx": 1, "gate_volume": 1, "bullish_alignment": 1, "bearish_alignment": 0},
        ])
        _seed_gate_evaluations(db_path, [
            {"decision_id": "D100", "gate_name": "adx", "gate_passed": 1, "metric_value": 28.0, "threshold_value": 25.0, "operator": ">", "explanation": "ADX OK"},
            {"decision_id": "D100", "gate_name": "volume", "gate_passed": 1, "metric_value": 1500000.0, "threshold_value": 1000000.0, "operator": ">", "explanation": "Volume OK"},
        ])

        result = mod.build_decision_detail("paper", "D100")
        assert result is not None
        assert result["decision"]["id"] == "D100"
        assert result["decision"]["symbol"] == "ETH"
        assert result["decision"]["event_type"] == "entry_signal"
        assert len(result["context"]) == 1
        assert result["context"][0]["price"] == 2500.0
        assert len(result["gates"]) == 2
        assert result["gates"][0]["gate_name"] == "adx"
        assert result["gates"][1]["gate_name"] == "volume"

    def test_found_without_context_or_gates(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "D200", "timestamp_ms": now_ms, "symbol": "BTC", "event_type": "fill", "status": "executed", "decision_phase": "execution"},
        ])
        result = mod.build_decision_detail("paper", "D200")
        assert result is not None
        assert result["decision"]["id"] == "D200"
        assert result["context"] == []
        assert result["gates"] == []


# ── GET /api/v2/trades/{id}/decision-trace ───────────────────────────────


class TestTradeDecisionTrace:

    def test_no_lineage(self, monitor_db):
        db_path, mod = monitor_db
        result = mod.build_trade_decision_trace("paper", 999)
        assert result["chain"] == []
        assert result["lineage"] is None

    def test_full_chain(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)

        # Signal -> Fill (entry) -> Exit decision
        _seed_decision_events(db_path, [
            {"id": "SIG01", "timestamp_ms": now_ms - 3000, "symbol": "ETH", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation", "trade_id": 500},
            {"id": "FILL01", "timestamp_ms": now_ms - 2000, "symbol": "ETH", "event_type": "fill", "status": "executed", "decision_phase": "execution", "trade_id": 500, "parent_decision_id": "SIG01"},
            {"id": "EXIT01", "timestamp_ms": now_ms - 1000, "symbol": "ETH", "event_type": "exit_check", "status": "executed", "decision_phase": "execution", "trade_id": 501},
        ])
        _seed_lineage(db_path, [
            {"signal_decision_id": "SIG01", "entry_trade_id": 500, "exit_decision_id": "EXIT01", "exit_trade_id": 501, "exit_reason": "stop_loss", "duration_ms": 2000},
        ])

        result = mod.build_trade_decision_trace("paper", 500)
        assert result["lineage"] is not None
        assert result["lineage"]["entry_trade_id"] == 500
        assert result["lineage"]["exit_trade_id"] == 501
        assert result["lineage"]["exit_reason"] == "stop_loss"
        assert result["lineage"]["duration_ms"] == 2000

        # Chain should contain all 3 decision events in chronological order.
        assert len(result["chain"]) == 3
        chain_ids = [e["id"] for e in result["chain"]]
        assert chain_ids == ["SIG01", "FILL01", "EXIT01"]

    def test_chain_via_exit_trade(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "SIG02", "timestamp_ms": now_ms - 2000, "symbol": "BTC", "event_type": "entry_signal", "status": "executed", "decision_phase": "signal_generation", "trade_id": 600},
            {"id": "EXIT02", "timestamp_ms": now_ms - 1000, "symbol": "BTC", "event_type": "exit_check", "status": "executed", "decision_phase": "execution", "trade_id": 601},
        ])
        _seed_lineage(db_path, [
            {"signal_decision_id": "SIG02", "entry_trade_id": 600, "exit_decision_id": "EXIT02", "exit_trade_id": 601, "exit_reason": "signal_flip", "duration_ms": 1000},
        ])

        # Query via exit trade_id.
        result = mod.build_trade_decision_trace("paper", 601)
        assert result["lineage"] is not None
        chain_ids = {e["id"] for e in result["chain"]}
        assert "EXIT02" in chain_ids

    def test_db_missing(self, monitor_db, monkeypatch):
        _db_path, mod = monitor_db
        monkeypatch.setenv("AIQ_MONITOR_PAPER_DB", "/nonexistent/path/db.sqlite")
        result = mod.build_trade_decision_trace("paper", 100)
        assert result["chain"] == []
        assert result.get("error") == "db_missing"


# ── GET /api/v2/decisions/{id}/gates ─────────────────────────────────────


class TestDecisionGates:

    def test_not_found(self, monitor_db):
        _db_path, mod = monitor_db
        result = mod.build_decision_gates("paper", "NONEXISTENT")
        assert result is None

    def test_gates_returned(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "G001", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "gate_block", "status": "blocked", "decision_phase": "gate_evaluation"},
        ])
        _seed_gate_evaluations(db_path, [
            {"decision_id": "G001", "gate_name": "adx_rising", "gate_passed": 0, "metric_value": 18.5, "threshold_value": 25.0, "operator": ">", "explanation": "ADX too low"},
            {"decision_id": "G001", "gate_name": "volume", "gate_passed": 1, "metric_value": 1500000.0, "threshold_value": 1000000.0, "operator": ">", "explanation": "Volume OK"},
            {"decision_id": "G001", "gate_name": "ranging", "gate_passed": 0, "metric_value": 0.85, "threshold_value": 0.80, "operator": "<", "explanation": "In ranging regime"},
        ])

        result = mod.build_decision_gates("paper", "G001")
        assert result is not None
        assert len(result["gates"]) == 3
        assert result["gates"][0]["gate_name"] == "adx_rising"
        assert result["gates"][0]["gate_passed"] == 0
        assert result["gates"][1]["gate_name"] == "volume"
        assert result["gates"][1]["gate_passed"] == 1
        assert result["gates"][2]["gate_name"] == "ranging"

    def test_decision_exists_no_gates(self, monitor_db):
        db_path, mod = monitor_db
        now_ms = int(time.time() * 1000)
        _seed_decision_events(db_path, [
            {"id": "G002", "timestamp_ms": now_ms, "symbol": "ETH", "event_type": "fill", "status": "executed", "decision_phase": "execution"},
        ])
        result = mod.build_decision_gates("paper", "G002")
        assert result is not None
        assert result["gates"] == []
