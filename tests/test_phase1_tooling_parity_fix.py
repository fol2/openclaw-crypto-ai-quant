from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from tools import audit_live_backtester_action_reconcile as live_bt_action
from tools import audit_live_backtester_trade_reconcile as live_bt_trade
from tools import audit_live_baseline_paper_order_parity as event_order_parity
from tools import audit_live_paper_action_reconcile as live_paper_action
from tools import audit_live_paper_decision_trace as live_paper_decision


def _mk_action(
    *,
    symbol: str,
    action_code: str,
    ts_ms: int,
    source_id: int,
    size: float,
    pnl_usd: float = 0.0,
    fee_usd: float = 0.0,
    price: float = 0.0,
    balance: float = 0.0,
    fill_hash: str = "",
) -> dict:
    return {
        "source": "test",
        "source_id": source_id,
        "line_no": source_id,
        "row_no": source_id,
        "symbol": symbol,
        "timestamp_ms": ts_ms,
        "action_code": action_code,
        "price": price,
        "size": size,
        "pnl_usd": pnl_usd,
        "fee_usd": fee_usd,
        "balance": balance,
        "confidence": "medium",
        "reason": "",
        "reason_code": "funding" if action_code == "FUNDING" else "signal_trigger",
        "fill_hash": fill_hash,
    }


def _write_decision_events_db(path: Path, *, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE decision_events (
            id TEXT PRIMARY KEY,
            timestamp_ms INTEGER,
            symbol TEXT,
            event_type TEXT,
            action_taken TEXT,
            context_json TEXT
        )
        """
    )
    for idx, row in enumerate(rows, start=1):
        conn.execute(
            """
            INSERT INTO decision_events (id, timestamp_ms, symbol, event_type, action_taken, context_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(row.get("id") or f"D{idx}"),
                int(row.get("timestamp_ms") or 0),
                str(row.get("symbol") or ""),
                str(row.get("event_type") or "entry_signal"),
                str(row.get("action_taken") or ""),
                json.dumps(row.get("context") or {}),
            ),
        )
    conn.commit()
    conn.close()


def test_live_backtester_split_fill_collapse_by_fill_hash() -> None:
    rows = [
        _mk_action(
            symbol="ETH",
            action_code="CLOSE_SHORT",
            ts_ms=1000,
            source_id=10,
            size=1.0,
            pnl_usd=2.0,
            fee_usd=0.10,
            price=100.0,
            fill_hash="hx",
        ),
        _mk_action(
            symbol="ETH",
            action_code="CLOSE_SHORT",
            ts_ms=1100,
            source_id=11,
            size=2.0,
            pnl_usd=3.0,
            fee_usd=0.20,
            price=101.0,
            fill_hash="hx",
        ),
        _mk_action(
            symbol="ETH",
            action_code="OPEN_SHORT",
            ts_ms=900,
            source_id=9,
            size=3.0,
            price=99.0,
        ),
    ]

    collapsed, stats = live_bt_action._collapse_split_fill_actions(rows)

    assert stats["split_fill_groups_collapsed"] == 1
    assert stats["split_fill_rows_collapsed"] == 1
    collapsed_close = [r for r in collapsed if r["action_code"] == "CLOSE_SHORT"]
    assert len(collapsed_close) == 1
    assert collapsed_close[0]["size"] == 3.0
    assert collapsed_close[0]["pnl_usd"] == 5.0
    assert collapsed_close[0]["fee_usd"] == pytest.approx(0.30)


def test_action_scope_contract_flags_symbol_side_disjoint() -> None:
    live_rows = [
        _mk_action(symbol="POL", action_code="OPEN_SHORT", ts_ms=1000, source_id=1, size=1.0),
        _mk_action(symbol="POL", action_code="CLOSE_SHORT", ts_ms=2000, source_id=2, size=1.0, pnl_usd=1.0),
    ]
    bt_rows = [
        _mk_action(symbol="POL", action_code="OPEN_LONG", ts_ms=1000, source_id=10, size=1.0),
        _mk_action(symbol="POL", action_code="CLOSE_LONG", ts_ms=2000, source_id=11, size=1.0, pnl_usd=1.0),
    ]

    scope = live_bt_action._summarise_action_scope_contract(live_rows, bt_rows, matched_pairs=0)

    assert scope["mismatch"] is True
    assert scope["mismatch_kind"] == "symbol_side_scope_disjoint"
    assert scope["shared_symbols"] == ["POL"]
    assert scope["shared_symbol_sides"] == []


def test_trade_scope_contract_flags_symbol_side_disjoint() -> None:
    live_rows = [
        {
            "symbol": "POL",
            "side": "SHORT",
            "exit_ts_ms": 2000,
            "exit_size": 1.0,
            "pnl_usd": 1.0,
            "fee_usd": 0.0,
            "source_id": "l1",
            "row_no": 1,
            "action": "CLOSE",
        }
    ]
    bt_rows = [
        {
            "symbol": "POL",
            "side": "LONG",
            "exit_ts_ms": 2000,
            "exit_size": 1.0,
            "pnl_usd": 1.0,
            "fee_usd": 0.0,
            "source_id": "b1",
            "row_no": 1,
            "reason_code": "exit_filter",
        }
    ]

    scope = live_bt_trade._summarise_exit_scope_contract(live_rows, bt_rows, matched_pairs=0)

    assert scope["mismatch"] is True
    assert scope["mismatch_kind"] == "symbol_side_scope_disjoint"
    assert scope["shared_exit_symbols"] == ["POL"]
    assert scope["shared_exit_symbol_sides"] == []


def test_live_backtester_funding_only_matched_pairs_are_residuals() -> None:
    live_rows = [
        _mk_action(symbol="ETH", action_code="FUNDING", ts_ms=1000, source_id=1, size=1.0, pnl_usd=0.2),
        _mk_action(symbol="BTC", action_code="FUNDING", ts_ms=2000, source_id=2, size=1.0, pnl_usd=0.3),
    ]
    bt_rows = [
        _mk_action(symbol="ETH", action_code="FUNDING", ts_ms=1000, source_id=1, size=1.0, pnl_usd=0.1),
    ]

    mismatches, summary, _ = live_bt_action._compare_actions(
        live_rows,
        bt_rows,
        [],
        timestamp_bucket_ms=1,
        timestamp_bucket_anchor="floor",
        price_tol=1e-9,
        size_tol=1e-9,
        pnl_tol=1e-9,
        fee_tol=1e-9,
        balance_tol=1e-9,
        order_fail_match_window_ms=60_000,
    )

    assert summary["funding_matched_pairs"] == 1
    assert summary["funding_unmatched_live"] == 1
    assert summary["non_simulatable_residuals"] == 1
    assert summary["unmatched_live"] == 1
    matched = [m for m in mismatches if m.get("kind") == "matched_funding_pair"]
    unmatched = [m for m in mismatches if m.get("kind") == "missing_backtester_funding_action"]
    assert len(matched) == 1
    assert len(unmatched) == 1
    assert unmatched[0]["classification"] == "non-simulatable_exchange_oms_effect"


def test_live_backtester_missing_live_funding_is_classified_as_surface_artefact() -> None:
    live_rows = [
        _mk_action(symbol="ETH", action_code="FUNDING", ts_ms=1000, source_id=1, size=1.0, pnl_usd=0.2),
    ]
    bt_rows = [
        _mk_action(symbol="ETH", action_code="FUNDING", ts_ms=1000, source_id=1, size=1.0, pnl_usd=0.1),
        _mk_action(symbol="BTC", action_code="FUNDING", ts_ms=2000, source_id=2, size=1.0, pnl_usd=0.3),
    ]

    mismatches, summary, _ = live_bt_action._compare_actions(
        live_rows,
        bt_rows,
        [],
        timestamp_bucket_ms=1,
        timestamp_bucket_anchor="floor",
        price_tol=1e-9,
        size_tol=1e-9,
        pnl_tol=1e-9,
        fee_tol=1e-9,
        balance_tol=1e-9,
        order_fail_match_window_ms=60_000,
    )

    assert summary["funding_matched_pairs"] == 1
    assert summary["funding_unmatched_backtester"] == 1
    assert summary["unmatched_backtester"] == 1
    unmatched = [m for m in mismatches if m.get("kind") == "missing_live_funding_action"]
    assert len(unmatched) == 1
    assert unmatched[0]["classification"] == "non-simulatable_exchange_oms_effect"


def test_live_backtester_mismatch_breakdown_tracks_kind_classification_drift() -> None:
    breakdown = live_bt_action._summarise_mismatch_breakdown(
        [
            {
                "classification": "deterministic_logic_divergence",
                "kind": "missing_backtester_funding_action",
                "symbol": "ETH",
                "action_code": "FUNDING",
            },
            {
                "classification": "deterministic_logic_divergence",
                "kind": "missing_backtester_action",
                "symbol": "ETH",
                "action_code": "OPEN_SHORT",
            },
        ]
    )

    assert breakdown["compare_surface_artefact_total"] == 1
    assert breakdown["logic_divergence_total"] == 1
    assert breakdown["classification_kind_drift_total"] == 1
    assert breakdown["classification_kind_drift_by_kind"] == {"missing_backtester_funding_action": 1}
    assert breakdown["classification_kind_drift_by_classification"] == {"deterministic_logic_divergence": 1}


def test_trade_reconcile_classifies_entry_confidence_policy_mismatch_residual(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    live_baseline = bundle_dir / "live_baseline_trades.jsonl"
    live_baseline.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "timestamp_ms": 1_000,
                        "symbol": "ETH",
                        "action": "OPEN",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 0.0,
                        "fee_usd": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "timestamp_ms": 2_000,
                        "symbol": "ETH",
                        "action": "CLOSE",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 1.0,
                        "fee_usd": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    backtester_trades = bundle_dir / "backtester_trades.csv"
    backtester_trades.write_text(
        "trade_id,position_id,entry_ts_ms,exit_ts_ms,symbol,side,entry_price,exit_price,exit_size,pnl_usd,fee_usd,mae_pct,mfe_pct,reason_code,reason\n",
        encoding="utf-8",
    )

    replay_report = bundle_dir / "backtester_replay_report.json"
    replay_report.write_text(
        json.dumps(
            {
                "config_fingerprint": "a" * 64,
                "trades": [
                    {
                        "action": "OPEN_LONG",
                        "symbol": "ETH",
                        "confidence": "high",
                        "timestamp": 1_200,
                        "reason_code": "entry_signal_sub_bar",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    live_db = bundle_dir / "trading_engine_live.db"
    _write_decision_events_db(
        live_db,
        rows=[
            {
                "id": "DE1",
                "timestamp_ms": 1_000_000,
                "symbol": "ETH",
                "action_taken": "open",
                "context": {
                    "confidence": "medium",
                    "confidence_source": "event_payload",
                    "entry_min_confidence_policy": "high",
                    "action": "OPEN",
                    "source": "kernel_candle",
                },
            }
        ],
    )

    strategy_snapshot = bundle_dir / "strategy_overrides.locked.yaml"
    strategy_snapshot.write_text("global:\n  trade:\n    entry_min_confidence: high\n", encoding="utf-8")

    manifest = bundle_dir / "replay_bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artefacts": {
                    "backtester_replay_report_json": replay_report.name,
                    "strategy_config_snapshot_file": strategy_snapshot.name,
                },
                "inputs": {
                    "live_db": live_db.name,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = bundle_dir / "trade_reconcile_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_backtester_trade_reconcile.py",
            "--live-baseline",
            str(live_baseline),
            "--backtester-trades",
            str(backtester_trades),
            "--bundle-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    exit_code = live_bt_trade.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["mismatch_total"] == 1
    assert report["counts"]["policy_mismatch_residuals"] == 1
    assert report["counts"]["deterministic_unexplained"] == 0
    assert report["status"]["strict_alignment_pass"] is False
    assert report["status"]["policy_mismatch_residual_only"] is True
    assert report["policy_mismatch_analysis"]["detected"] is True
    assert report["policy_mismatch_analysis"]["evidence_complete"] is True
    assert report["mismatch_counts_by_classification"]["policy_mismatch_residual"] == 1
    assert report["mismatches"][0]["classification"] == "policy_mismatch_residual"
    runtime_policy = report["policy_mismatch_analysis"]["runtime_entry_policy"]
    assert runtime_policy["provenance_contract_ok"] is True
    assert runtime_policy["match_verified"] == 1
    assert runtime_policy["match_fallback"] == 0


def test_trade_reconcile_does_not_attribute_policy_mismatch_when_entry_policy_provenance_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    live_baseline = bundle_dir / "live_baseline_trades.jsonl"
    live_baseline.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "timestamp_ms": 1_000,
                        "symbol": "ETH",
                        "action": "OPEN",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 0.0,
                        "fee_usd": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "timestamp_ms": 2_000,
                        "symbol": "ETH",
                        "action": "CLOSE",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 1.0,
                        "fee_usd": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    backtester_trades = bundle_dir / "backtester_trades.csv"
    backtester_trades.write_text(
        "trade_id,position_id,entry_ts_ms,exit_ts_ms,symbol,side,entry_price,exit_price,exit_size,pnl_usd,fee_usd,mae_pct,mfe_pct,reason_code,reason\n",
        encoding="utf-8",
    )

    replay_report = bundle_dir / "backtester_replay_report.json"
    replay_report.write_text(
        json.dumps(
            {
                "config_fingerprint": "a" * 64,
                "trades": [
                    {
                        "action": "OPEN_LONG",
                        "symbol": "ETH",
                        "confidence": "high",
                        "timestamp": 1_200,
                        "reason_code": "entry_signal_sub_bar",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    strategy_snapshot = bundle_dir / "strategy_overrides.locked.yaml"
    strategy_snapshot.write_text("trade:\n  leverage: 5\n", encoding="utf-8")

    manifest = bundle_dir / "replay_bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artefacts": {
                    "backtester_replay_report_json": replay_report.name,
                    "strategy_config_snapshot_file": strategy_snapshot.name,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = bundle_dir / "trade_reconcile_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_backtester_trade_reconcile.py",
            "--live-baseline",
            str(live_baseline),
            "--backtester-trades",
            str(backtester_trades),
            "--bundle-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    exit_code = live_bt_trade.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["mismatch_total"] == 1
    assert report["counts"]["policy_mismatch_residuals"] == 0
    assert report["counts"]["deterministic_unexplained"] == 1
    assert report["status"]["policy_mismatch_residual_only"] is False
    assert report["policy_mismatch_analysis"]["detected"] is False
    assert report["policy_mismatch_analysis"]["evidence_complete"] is False
    locked_policy = report["policy_mismatch_analysis"]["locked_entry_policy"]
    assert locked_policy["policy_available"] is False
    assert locked_policy["provenance_contract_ok"] is False
    assert locked_policy["policy_source"] == "strategy_snapshot_missing_entry_min_confidence"
    assert report["mismatches"][0]["classification"] == "deterministic_logic_divergence"


def test_trade_reconcile_does_not_attribute_policy_mismatch_when_runtime_confidence_is_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    live_baseline = bundle_dir / "live_baseline_trades.jsonl"
    live_baseline.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "timestamp_ms": 1_000,
                        "symbol": "ETH",
                        "action": "OPEN",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 0.0,
                        "fee_usd": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "timestamp_ms": 2_000,
                        "symbol": "ETH",
                        "action": "CLOSE",
                        "type": "SHORT",
                        "confidence": "medium",
                        "size": 1.0,
                        "pnl": 1.0,
                        "fee_usd": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    backtester_trades = bundle_dir / "backtester_trades.csv"
    backtester_trades.write_text(
        "trade_id,position_id,entry_ts_ms,exit_ts_ms,symbol,side,entry_price,exit_price,exit_size,pnl_usd,fee_usd,mae_pct,mfe_pct,reason_code,reason\n",
        encoding="utf-8",
    )

    replay_report = bundle_dir / "backtester_replay_report.json"
    replay_report.write_text(
        json.dumps(
            {
                "config_fingerprint": "a" * 64,
                "trades": [
                    {
                        "action": "OPEN_LONG",
                        "symbol": "ETH",
                        "confidence": "high",
                        "timestamp": 1_200,
                        "reason_code": "entry_signal_sub_bar",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    live_db = bundle_dir / "trading_engine_live.db"
    _write_decision_events_db(
        live_db,
        rows=[
            {
                "id": "DE1",
                "timestamp_ms": 1_000_000,
                "symbol": "ETH",
                "action_taken": "open",
                "context": {
                    "confidence": "medium",
                    "confidence_source": "fallback_na_medium",
                    "entry_min_confidence_policy": "high",
                    "action": "OPEN",
                    "source": "kernel_candle",
                },
            }
        ],
    )

    strategy_snapshot = bundle_dir / "strategy_overrides.locked.yaml"
    strategy_snapshot.write_text("global:\n  trade:\n    entry_min_confidence: high\n", encoding="utf-8")

    manifest = bundle_dir / "replay_bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artefacts": {
                    "backtester_replay_report_json": replay_report.name,
                    "strategy_config_snapshot_file": strategy_snapshot.name,
                },
                "inputs": {
                    "live_db": live_db.name,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = bundle_dir / "trade_reconcile_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_backtester_trade_reconcile.py",
            "--live-baseline",
            str(live_baseline),
            "--backtester-trades",
            str(backtester_trades),
            "--bundle-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    exit_code = live_bt_trade.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["mismatch_total"] == 1
    assert report["counts"]["policy_mismatch_residuals"] == 0
    assert report["counts"]["deterministic_unexplained"] == 1
    assert report["policy_mismatch_analysis"]["detected"] is False
    assert report["policy_mismatch_analysis"]["evidence_complete"] is False
    runtime_policy = report["policy_mismatch_analysis"]["runtime_entry_policy"]
    assert runtime_policy["source_contract_ok"] is False
    assert runtime_policy["provenance_contract_ok"] is False
    assert runtime_policy["match_fallback"] == 1
    assert report["mismatches"][0]["classification"] == "deterministic_logic_divergence"


def test_trade_policy_loader_reads_global_strategy_schema(tmp_path: Path) -> None:
    strategy_snapshot = tmp_path / "strategy_overrides.locked.yaml"
    strategy_snapshot.write_text(
        (
            "global:\n"
            "  trade:\n"
            "    entry_min_confidence: high\n"
            "symbols:\n"
            "  ETH:\n"
            "    trade:\n"
            "      entry_min_confidence: medium\n"
        ),
        encoding="utf-8",
    )

    policy = live_bt_trade._load_locked_entry_confidence_policy(strategy_snapshot)

    assert policy["policy_available"] is True
    assert policy["provenance_contract_ok"] is True
    assert policy["global_min_confidence"] == "high"
    assert policy["symbol_min_confidence"]["ETH"] == "medium"
    assert policy["policy_source"] == "strategy_snapshot_explicit"


def test_decision_trace_filters_paper_rows_by_trade_id_watermark(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE decision_events (
            id TEXT PRIMARY KEY,
            timestamp_ms INTEGER,
            symbol TEXT,
            event_type TEXT,
            status TEXT,
            decision_phase TEXT,
            triggered_by TEXT,
            action_taken TEXT,
            rejection_reason TEXT,
            reason_code TEXT,
            config_fingerprint TEXT,
            trade_id INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('D1', 1000, 'ETH', 'entry_signal', 'accepted', 'execution', 'kernel',
                'open_short', '', '', 'abc', 10)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('D2', 1001, 'ETH', 'entry_signal', 'accepted', 'execution', 'kernel',
                'open_short', '', '', 'abc', 200)
        """
    )
    conn.commit()
    conn.close()

    rows, counts, issues = live_paper_decision._load_decision_rows(
        db,
        source_name="paper",
        from_ts=None,
        to_ts=None,
        include_runtime_only_blocked=False,
        include_funding_events=True,
        paper_min_trade_id_exclusive=100,
    )
    assert issues == []
    assert counts["filtered_by_trade_id"] == 1
    assert counts["row_count"] == 1
    assert rows[0]["source_id"] == "D2"


def test_decision_trace_excludes_funding_events_by_default(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE decision_events (
            id TEXT PRIMARY KEY,
            timestamp_ms INTEGER,
            symbol TEXT,
            event_type TEXT,
            status TEXT,
            decision_phase TEXT,
            triggered_by TEXT,
            action_taken TEXT,
            rejection_reason TEXT,
            reason_code TEXT,
            config_fingerprint TEXT,
            trade_id INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('F1', 1000, 'ETH', 'funding', 'executed', 'execution', 'schedule',
                'apply_funding', '', 'funding_payment', 'abc', NULL)
        """
    )
    conn.commit()
    conn.close()

    rows, counts, issues = live_paper_decision._load_decision_rows(
        db,
        source_name="paper",
        from_ts=None,
        to_ts=None,
        include_runtime_only_blocked=False,
        include_funding_events=False,
        paper_min_trade_id_exclusive=None,
    )
    assert issues == []
    assert counts["funding_excluded"] == 1
    assert counts["row_count"] == 0
    assert rows == []


def test_decision_trace_main_accepts_preseed_paper_only_unlinked_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                reason_code TEXT,
                config_fingerprint TEXT,
                trade_id INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('L1', 1000, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', NULL)
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P1', 1000, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P2', 1001, 'ETH', 'gate_block', 'blocked', 'risk_check', 'schedule',
                'blocked', 'margin cap', 'exit_filter', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P3', 999, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', 10)
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_decision_trace.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--paper-min-id-exclusive",
            "100",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_decision.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["mismatch_total"] == 1
    assert report["counts"]["unmatched_paper"] == 1
    assert report["status"]["strict_alignment_pass"] is False
    assert report["status"]["accepted_residuals_only"] is False
    assert any(
        str(row.get("kind") or "") == "paper_preseed_decision_rows_out_of_scope"
        for row in report["accepted_residuals"]
    )


def test_decision_trace_main_preseed_unlinked_residuals_are_opt_in_non_blocking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                reason_code TEXT,
                config_fingerprint TEXT,
                trade_id INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('L1', 1000, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', NULL)
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P1', 1000, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P2', 1001, 'ETH', 'gate_block', 'blocked', 'risk_check', 'schedule',
                'blocked', 'margin cap', 'exit_filter', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P3', 999, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', 10)
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_decision_trace.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--paper-min-id-exclusive",
            "100",
            "--allow-paper-preseed-unlinked-residuals",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_decision.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["mismatch_total"] == 1
    assert report["counts"]["unmatched_paper"] == 1
    assert report["status"]["strict_alignment_pass"] is True
    assert report["status"]["accepted_residuals_only"] is True
    assert report["inputs"]["allow_paper_preseed_unlinked_residuals"] is True
    assert any(
        str(row.get("kind") or "") == "paper_preseed_decision_rows_out_of_scope"
        for row in report["accepted_residuals"]
    )


def test_decision_trace_main_fails_strict_when_run_fingerprint_guard_drifts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                reason_code TEXT,
                config_fingerprint TEXT,
                trade_id INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "live_run_fingerprint_provenance": {
                    "rows_sampled": 12,
                    "run_fingerprint_distinct": 2,
                    "run_fingerprint_timeline": [],
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_decision_trace.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--allow-paper-preseed-unlinked-residuals",
            "--bundle-manifest",
            str(manifest),
            "--require-single-run-fingerprint",
            "--max-live-run-fingerprint-distinct",
            "1",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_decision.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["status"]["strict_alignment_pass"] is False
    assert any(
        str(row.get("kind") or "") == "live_run_fingerprint_drift_within_window"
        for row in report["mismatches"]
    )


def test_decision_trace_opt_in_does_not_mask_deterministic_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                reason_code TEXT,
                config_fingerprint TEXT,
                trade_id INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('L1', 1000, 'ETH', 'gate_block', 'blocked', 'risk_check', 'schedule',
                'blocked', 'margin 1 > cap', 'exit_filter', 'abc', NULL)
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P1', 1000, 'ETH', 'gate_block', 'blocked', 'risk_check', 'schedule',
                'blocked', 'margin 2 > cap', 'exit_filter', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P2', 1001, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('P3', 999, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open', '', 'entry_signal', 'abc', 10)
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_decision_trace.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--paper-min-id-exclusive",
            "100",
            "--allow-paper-preseed-unlinked-residuals",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_decision.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["inputs"]["allow_paper_preseed_unlinked_residuals"] is True
    assert report["counts"]["rejection_reason_mismatch"] == 1
    assert report["status"]["strict_alignment_pass"] is False
    assert report["status"]["accepted_residuals_only"] is False
    assert any(
        str(row.get("kind") or "") == "decision_rejection_reason_mismatch"
        for row in report["mismatches"]
    )


def test_decision_trace_window_not_replayed_stays_blocking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                decision_phase TEXT,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                reason_code TEXT,
                config_fingerprint TEXT,
                trade_id INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO decision_events
        (id, timestamp_ms, symbol, event_type, status, decision_phase, triggered_by,
         action_taken, rejection_reason, reason_code, config_fingerprint, trade_id)
        VALUES ('L1', 1000, 'ETH', 'entry_signal', 'executed', 'execution', 'schedule',
                'open_short', '', 'entry_signal', 'abc', 101)
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_decision_trace.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--fail-on-mismatch",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_decision.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["counts"]["live_decision_rows"] == 1
    assert report["counts"]["paper_decision_rows"] == 0
    assert report["status"]["paper_window_not_replayed"] is True
    assert report["status"]["strict_alignment_pass"] is False


def test_event_order_funding_contract_marks_unmatched_as_mismatch(tmp_path: Path, monkeypatch) -> None:
    live_baseline = tmp_path / "live_baseline.jsonl"
    live_baseline.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "timestamp_ms": 1000,
                        "symbol": "ETH",
                        "action": "FUNDING",
                        "type": "SHORT",
                        "pnl": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "timestamp_ms": 2000,
                        "symbol": "BTC",
                        "action": "FUNDING",
                        "type": "SHORT",
                        "pnl": 0.2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paper_db = tmp_path / "paper.db"
    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            reason TEXT,
            pnl REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO trades (id, timestamp, symbol, action, type, reason, pnl)
        VALUES (10, '1970-01-01T00:00:01+00:00', 'ETH', 'FUNDING', 'SHORT', 'funding', 0.05)
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "event_order_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_baseline_paper_order_parity.py",
            "--live-baseline",
            str(live_baseline),
            "--paper-db",
            str(paper_db),
            "--output",
            str(output),
        ],
    )

    exit_code = event_order_parity.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["funding_matched_pairs"] == 1
    assert report["counts"]["funding_unmatched_live"] == 1
    assert report["status"]["strict_alignment_pass"] is False
    assert any(
        str(row.get("kind") or "") == "funding_unmatched_across_surfaces"
        for row in report["mismatches"]
    )


def test_event_order_paper_window_not_replayed_with_unmatched_funding_is_non_blocking_with_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_baseline = tmp_path / "live_baseline.jsonl"
    live_baseline.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "timestamp_ms": 1000,
                        "symbol": "ETH",
                        "action": "OPEN",
                        "type": "SHORT",
                        "pnl": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "timestamp_ms": 1100,
                        "symbol": "ETH",
                        "action": "FUNDING",
                        "type": "SHORT",
                        "pnl": 0.1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paper_db = tmp_path / "paper.db"
    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            reason TEXT,
            pnl REAL
        )
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "event_order_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_baseline_paper_order_parity.py",
            "--live-baseline",
            str(live_baseline),
            "--paper-db",
            str(paper_db),
            "--fail-on-mismatch",
            "--output",
            str(output),
        ],
    )

    exit_code = event_order_parity.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["status"]["strict_alignment_pass"] is True
    assert any(
        str(row.get("kind") or "") == "paper_window_not_replayed"
        for row in report["accepted_residuals"]
    )
    assert any(
        str(row.get("kind") or "") == "funding_unmatched_across_surfaces"
        for row in report["mismatches"]
    )
    assert report["counts"]["funding_unmatched_live"] == 1
    assert report["counts"]["mismatch_count"] >= 1


def test_live_paper_action_main_collapses_split_fills(tmp_path: Path, monkeypatch) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    schema = """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            balance REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            fee_usd REAL,
            fill_hash TEXT
        )
    """
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(schema)
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO trades
        (id, timestamp, symbol, action, type, price, size, pnl, balance, reason, reason_code, confidence, fee_usd, fill_hash)
        VALUES
        (1, '1970-01-01T00:00:01.000+00:00', 'ETH', 'CLOSE', 'SHORT', 100.0, 1.0, 1.0, 100.0, 'exit', 'exit_signal', 'medium', 0.0, 'fhx'),
        (2, '1970-01-01T00:00:01.100+00:00', 'ETH', 'CLOSE', 'SHORT', 100.0, 2.0, 2.0, 100.0, 'exit', 'exit_signal', 'medium', 0.0, 'fhx')
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(paper_db)
    conn.execute(
        """
        INSERT INTO trades
        (id, timestamp, symbol, action, type, price, size, pnl, balance, reason, reason_code, confidence, fee_usd, fill_hash)
        VALUES
        (11, '1970-01-01T00:00:01.100+00:00', 'ETH', 'CLOSE', 'SHORT', 100.0, 3.0, 3.0, 100.0, 'exit', 'exit_signal', 'medium', 0.0, 'fhx')
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "live_paper_action_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_action_reconcile.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_action.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["counts"]["split_fill_groups_collapsed"] == 1
    assert report["counts"]["split_fill_rows_collapsed"] == 1
    assert report["status"]["strict_alignment_pass"] is True


def test_live_paper_action_main_run_fingerprint_drift_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    schema = """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            balance REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            fee_usd REAL,
            fill_hash TEXT
        )
    """
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(schema)
        conn.commit()
        conn.close()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "live_run_fingerprint_provenance": {
                    "rows_sampled": 5,
                    "run_fingerprint_distinct": 2,
                    "run_fingerprint_timeline": [],
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "live_paper_action_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_action_reconcile.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--bundle-manifest",
            str(manifest),
            "--require-single-run-fingerprint",
            "--max-live-run-fingerprint-distinct",
            "1",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_action.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["status"]["strict_alignment_pass"] is False
    assert report["run_fingerprint_guard"]["ok"] is False
    assert any(
        str(row.get("kind") or "") == "live_run_fingerprint_drift_within_window"
        for row in report["mismatches"]
    )


def test_live_paper_action_detects_window_not_replayed_with_funding_gaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    schema = """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            balance REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            fee_usd REAL,
            fill_hash TEXT
        )
    """
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(schema)
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO trades
        (id, timestamp, symbol, action, type, price, size, pnl, balance, reason, reason_code, confidence, fee_usd, fill_hash)
        VALUES
        (1, '1970-01-01T00:00:01.000+00:00', 'ETH', 'OPEN', 'SHORT', 100.0, 1.0, 0.0, 100.0, 'entry', 'signal_trigger', 'medium', 0.0, ''),
        (2, '1970-01-01T00:00:01.100+00:00', 'ETH', 'FUNDING', 'SHORT', 100.0, 1.0, 0.1, 100.1, 'funding', 'funding', 'medium', 0.0, '')
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "live_paper_action_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_action_reconcile.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--fail-on-mismatch",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_action.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["status"]["strict_alignment_pass"] is False
    assert report["status"]["paper_window_not_replayed"] is True
    assert report["counts"]["mismatch_total"] == 2
    assert report["counts"]["paper_window_not_replayed_artefact_mismatch_total"] == 2
    assert report["counts"]["non_blocking_evidence_total"] == 0
    assert report["counts"]["true_mismatch_total"] == 0
    assert report["counts"]["live_simulatable_actions"] == 1
    assert report["counts"]["paper_simulatable_actions"] == 0
    assert report["counts"]["unmatched_live_simulatable"] == 1
    assert report["counts"]["funding_unmatched_live"] == 1
    assert any(
        str(row.get("kind") or "") == "paper_window_not_replayed"
        for row in report["accepted_residuals"]
    )


def test_live_paper_action_window_not_replayed_opt_in_is_ignored_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "live.db"
    paper_db = tmp_path / "paper.db"
    schema = """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            type TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            balance REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            fee_usd REAL,
            fill_hash TEXT
        )
    """
    for path in (live_db, paper_db):
        conn = sqlite3.connect(path)
        conn.execute(schema)
        conn.commit()
        conn.close()

    conn = sqlite3.connect(live_db)
    conn.execute(
        """
        INSERT INTO trades
        (id, timestamp, symbol, action, type, price, size, pnl, balance, reason, reason_code, confidence, fee_usd, fill_hash)
        VALUES
        (1, '1970-01-01T00:00:01.000+00:00', 'ETH', 'OPEN', 'SHORT', 100.0, 1.0, 0.0, 100.0, 'entry', 'signal_trigger', 'medium', 0.0, ''),
        (2, '1970-01-01T00:00:01.100+00:00', 'ETH', 'FUNDING', 'SHORT', 100.0, 1.0, 0.1, 100.1, 'funding', 'funding', 'medium', 0.0, '')
        """
    )
    conn.commit()
    conn.close()

    output = tmp_path / "live_paper_action_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_live_paper_action_reconcile.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--allow-paper-window-not-replayed",
            "--fail-on-mismatch",
            "--output",
            str(output),
        ],
    )

    exit_code = live_paper_action.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["status"]["strict_alignment_pass"] is False
    assert report["status"]["paper_window_not_replayed"] is True
    assert report["status"]["paper_window_not_replayed_opt_in_ignored"] is True
    assert report["inputs"]["allow_paper_window_not_replayed"] is True
    assert report["counts"]["mismatch_total"] == 2
    assert report["counts"]["paper_window_not_replayed_artefact_mismatch_total"] == 2
    assert report["counts"]["non_blocking_evidence_total"] == 0
    assert report["counts"]["true_mismatch_total"] == 0
    assert report["counts"]["funding_unmatched_live"] == 1
    assert any(
        str(row.get("kind") or "") == "paper_window_not_replayed"
        and bool(row.get("non_blocking_opt_in_enabled")) is True
        for row in report["accepted_residuals"]
    )
