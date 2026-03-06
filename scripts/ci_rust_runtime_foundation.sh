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
loop_db_path = Path("/tmp/aiq-runtime-paper-loop.db")
bars_path = Path("/tmp/aiq-runtime-candles.db")
for path in (
    snap_path,
    db_path,
    loop_db_path,
    bars_path,
    Path("/tmp/aiq-runtime-doctor.json"),
    Path("/tmp/aiq-runtime-pipeline.json"),
    Path("/tmp/aiq-runtime-snapshot-validate.json"),
    Path("/tmp/aiq-runtime-seed-paper.json"),
    Path("/tmp/aiq-runtime-paper-run-once.json"),
    Path("/tmp/aiq-runtime-paper-cycle.json"),
    Path("/tmp/aiq-runtime-paper-loop.json"),
    Path("/tmp/aiq-runtime-paper-loop-resume.json"),
    Path("/tmp/aiq-runtime-paper-loop-idle.json"),
    Path("/tmp/aiq-runtime-paper-loop-follow.json"),
):
    if path.exists():
        path.unlink()

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
            "last_funding_time_ms": 1772676580000,
            "last_add_time_ms": 1772676600000,
            "entry_adx_threshold": 23.5,
        }
    ],
    "runtime": {
        "entry_attempt_ms_by_symbol": {"BTC": 1772676500000},
        "exit_attempt_ms_by_symbol": {"BTC": 1772676550000},
        "last_close_info_by_symbol": {
            "ETH": {
                "timestamp_ms": 1772676400000,
                "side": "short",
                "reason": "Signal Trigger"
            }
        },
    },
}
snap_path.write_text(json.dumps(payload), encoding="utf-8")

for db in (db_path, loop_db_path):
    conn = sqlite3.connect(db)
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

conn = sqlite3.connect(bars_path)
conn.executescript(
    """
    CREATE TABLE candles (
        symbol TEXT,
        interval TEXT,
        t INTEGER,
        t_close INTEGER,
        o REAL,
        h REAL,
        l REAL,
        c REAL,
        v REAL,
        n INTEGER
    );
    """
)
base = 1772670000000
for symbol, start, drift in (("ETH", 100.0, 0.25), ("BTC", 50000.0, 20.0)):
    price = start
    for idx in range(420):
        t = base + idx * 1800000
        open_ = price
        close = price + drift
        high = max(open_, close) + 0.5
        low = min(open_, close) - 0.5
        volume = 1000.0 + idx
        conn.execute(
            "INSERT INTO candles VALUES (?, '30m', ?, ?, ?, ?, ?, ?, ?, 1)",
            (symbol, t, t + 1800000, open_, high, low, close, volume),
        )
        price = close
conn.commit()
conn.close()
PY

cargo run -q -p aiq-runtime -- snapshot validate --path /tmp/aiq-runtime-seed-snapshot.json --json >/tmp/aiq-runtime-snapshot-validate.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper.db --strict-replace --json >/tmp/aiq-runtime-seed-paper.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper-loop.db --strict-replace --json >/tmp/aiq-runtime-seed-paper-loop.json
cargo run -q -p aiq-runtime -- paper doctor --db /tmp/aiq-runtime-paper.db --json >/tmp/aiq-runtime-paper-doctor.json
cargo run -q -p aiq-runtime -- paper run-once --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --target-symbol ETH --exported-at-ms 1772676900000 --dry-run --json >/tmp/aiq-runtime-paper-run-once.json
cargo run -q -p aiq-runtime -- paper cycle --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --step-close-ts-ms 1773426000000 --exported-at-ms 1772676900000 --json >/tmp/aiq-runtime-paper-cycle.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --start-step-close-ts-ms 1773422400000 --max-steps 2 --json >/tmp/aiq-runtime-paper-loop.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --max-steps 2 --json >/tmp/aiq-runtime-paper-loop-resume.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --max-steps 1 --json >/tmp/aiq-runtime-paper-loop-idle.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --follow --idle-sleep-ms 1 --max-idle-polls 1 --max-steps 1 --json >/tmp/aiq-runtime-paper-loop-follow.json
cargo run -q -p aiq-runtime -- paper doctor --db /tmp/aiq-runtime-paper.db --live --json >/tmp/aiq-runtime-paper-doctor-live.json
cargo run -q -p aiq-runtime -- paper run-once --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --target-symbol ETH --exported-at-ms 1772676900000 --live --dry-run --json >/tmp/aiq-runtime-paper-run-once-live.json
if cargo run -q -p aiq-runtime -- paper cycle --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --step-close-ts-ms 1773426000000 --exported-at-ms 1772676900000 --json >/tmp/aiq-runtime-paper-cycle-rerun.json 2>/tmp/aiq-runtime-paper-cycle-rerun.stderr; then
  echo "paper cycle rerun guard did not fail closed" >&2
  exit 1
fi

python3 - <<'PY'
import json
import sqlite3
from pathlib import Path

report = json.loads(Path("/tmp/aiq-runtime-paper-run-once.json").read_text(encoding="utf-8"))
assert report["snapshot_exported_at_ms"] == 1772676900000
assert report["symbol"] == "ETH"
cycle = json.loads(Path("/tmp/aiq-runtime-paper-cycle.json").read_text(encoding="utf-8"))
assert cycle["step_close_ts_ms"] == 1773426000000
assert cycle["runtime_step_recorded"] is True
loop = json.loads(Path("/tmp/aiq-runtime-paper-loop.json").read_text(encoding="utf-8"))
assert loop["executed_steps"] == 2
assert [step["step_close_ts_ms"] for step in loop["steps"]] == [1773422400000, 1773424200000]
assert [step["snapshot_exported_at_ms"] for step in loop["steps"]] == [1773422400000, 1773424200000]
loop_resume = json.loads(Path("/tmp/aiq-runtime-paper-loop-resume.json").read_text(encoding="utf-8"))
assert loop_resume["executed_steps"] == 1
assert [step["step_close_ts_ms"] for step in loop_resume["steps"]] == [1773426000000]
loop_idle = json.loads(Path("/tmp/aiq-runtime-paper-loop-idle.json").read_text(encoding="utf-8"))
assert loop_idle["executed_steps"] == 0
assert loop_idle["latest_common_close_ts_ms"] == 1773426000000
loop_follow = json.loads(Path("/tmp/aiq-runtime-paper-loop-follow.json").read_text(encoding="utf-8"))
assert loop_follow["executed_steps"] == 0
assert loop_follow["follow"] is True
assert loop_follow["idle_polls"] == 1
assert any("follow exhausted" in warning for warning in loop_follow["warnings"])
with sqlite3.connect("/tmp/aiq-runtime-paper-loop.db") as conn:
    runtime_steps = conn.execute("SELECT COUNT(*) FROM runtime_cycle_steps").fetchone()[0]
assert runtime_steps == 3
doctor = json.loads(Path("/tmp/aiq-runtime-paper-doctor.json").read_text(encoding="utf-8"))
assert doctor["paper_bootstrap"]["runtime_close_markers"] == 1
PY

echo "[runtime-foundation] ok"
