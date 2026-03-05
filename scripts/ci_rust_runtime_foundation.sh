#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

echo "[runtime-foundation] cargo check --workspace"
cargo check --workspace

echo "[runtime-foundation] cargo test -p aiq-runtime-core"
cargo test -p aiq-runtime-core

echo "[runtime-foundation] cargo test -p aiq-runtime"
cargo test -p aiq-runtime

echo "[runtime-foundation] bt-core config + init-state compatibility"
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_load_yaml_runtime_pipeline_overrides
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_parse_valid_v2_json_with_runtime
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_into_sim_state_with_runtime_filters_unknown_symbols

echo "[runtime-foundation] runtime CLI smoke"
cargo run -q -p aiq-runtime -- doctor --json >/tmp/aiq-runtime-doctor.json
cargo run -q -p aiq-runtime -- pipeline --json >/tmp/aiq-runtime-pipeline.json

python3 - <<'PY'
import json
import sqlite3
from pathlib import Path

snap_path = Path("/tmp/aiq-runtime-seed-snapshot.json")
db_path = Path("/tmp/aiq-runtime-paper.db")

payload = {
    "version": 2,
    "source": "paper",
    "exported_at_ms": 1772676900000,
    "balance": 1000.0,
    "positions": [
        {
            "symbol": "BTC",
            "side": "long",
            "size": 2.0,
            "entry_price": 100.0,
            "entry_atr": 5.0,
            "trailing_sl": 95.0,
            "confidence": "high",
            "leverage": 4.0,
            "margin_used": 50.0,
            "adds_count": 1,
            "tp1_taken": False,
            "open_time_ms": 1772676500000,
            "last_add_time_ms": 1772676600000,
            "entry_adx_threshold": 23.5,
        }
    ],
    "runtime": {
        "entry_attempt_ms_by_symbol": {"BTC": 1772676500000},
        "exit_attempt_ms_by_symbol": {"BTC": 1772676550000},
    },
}
snap_path.write_text(json.dumps(payload), encoding="utf-8")

conn = sqlite3.connect(db_path)
conn.executescript(
    """
    CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        type TEXT,
        action TEXT,
        price REAL,
        size REAL,
        notional REAL,
        reason TEXT,
        reason_code TEXT,
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
        run_fingerprint TEXT,
        fill_hash TEXT,
        fill_tid INTEGER
    );
    CREATE TABLE position_state (
        symbol TEXT PRIMARY KEY,
        open_trade_id INTEGER,
        trailing_sl REAL,
        last_funding_time INTEGER,
        adds_count INTEGER,
        tp1_taken INTEGER,
        last_add_time INTEGER,
        entry_adx_threshold REAL,
        updated_at TEXT
    );
    CREATE TABLE position_state_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_ts_ms INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        open_trade_id INTEGER,
        trailing_sl REAL,
        last_funding_time INTEGER,
        adds_count INTEGER,
        tp1_taken INTEGER,
        last_add_time INTEGER,
        entry_adx_threshold REAL,
        event_type TEXT NOT NULL,
        run_fingerprint TEXT
    );
    CREATE TABLE runtime_cooldowns (
        symbol TEXT PRIMARY KEY,
        last_entry_attempt_s REAL,
        last_exit_attempt_s REAL,
        updated_at TEXT
    );
    """
)
conn.close()
PY

cargo run -q -p aiq-runtime -- snapshot validate --path /tmp/aiq-runtime-seed-snapshot.json --json >/tmp/aiq-runtime-snapshot-validate.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper.db --strict-replace --json >/tmp/aiq-runtime-seed-paper.json

echo "[runtime-foundation] ok"
