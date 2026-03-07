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
import os
import signal
import sqlite3
import time
from pathlib import Path

snap_path = Path("/tmp/aiq-runtime-seed-snapshot.json")
db_path = Path("/tmp/aiq-runtime-paper.db")
loop_db_path = Path("/tmp/aiq-runtime-paper-loop.db")
bars_path = Path("/tmp/aiq-runtime-candles.db")
gap_db_path = Path("/tmp/aiq-runtime-paper-loop-gap.db")
gap_bars_path = Path("/tmp/aiq-runtime-paper-loop-gap-candles.db")
for path in (
    snap_path,
    db_path,
    loop_db_path,
    bars_path,
    gap_db_path,
    gap_bars_path,
    Path("/tmp/aiq-runtime-doctor.json"),
    Path("/tmp/aiq-runtime-pipeline.json"),
    Path("/tmp/aiq-runtime-paper-manifest.json"),
    Path("/tmp/aiq-runtime-paper-effective-config-base.json"),
    Path("/tmp/aiq-runtime-paper-effective-config-primary.json"),
    Path("/tmp/aiq-runtime-paper-effective-config-fallback.json"),
    Path("/tmp/aiq-runtime-paper-manifest-resume.json"),
    Path("/tmp/aiq-runtime-paper-status-bootstrap.json"),
    Path("/tmp/aiq-runtime-paper-service-bootstrap.json"),
    Path("/tmp/aiq-runtime-paper-bootstrap.status.json"),
    Path("/tmp/aiq-runtime-paper-status-stopped.json"),
    Path("/tmp/aiq-runtime-paper-service-stopped.json"),
    Path("/tmp/aiq-runtime-paper-service-apply.json"),
    Path("/tmp/aiq-runtime-paper-daemon.json"),
    Path("/tmp/aiq-runtime-paper-daemon.status.json"),
    Path("/tmp/aiq-runtime-snapshot-validate.json"),
    Path("/tmp/aiq-runtime-seed-paper.json"),
    Path("/tmp/aiq-runtime-seed-paper-loop.json"),
    Path("/tmp/aiq-runtime-seed-paper-loop-gap.json"),
    Path("/tmp/aiq-runtime-paper-run-once.json"),
    Path("/tmp/aiq-runtime-paper-cycle.json"),
    Path("/tmp/aiq-runtime-paper-loop.json"),
    Path("/tmp/aiq-runtime-paper-loop-resume.json"),
    Path("/tmp/aiq-runtime-paper-loop-idle.json"),
    Path("/tmp/aiq-runtime-paper-loop-follow.json"),
    Path("/tmp/aiq-runtime-paper-loop-gap.stderr"),
    Path("/tmp/aiq-runtime-effective-config.yaml"),
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

for db in (db_path, loop_db_path, gap_db_path):
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
for symbol, start, drift in (("ETH", 100.0, 0.25), ("SOL", 150.0, 0.4), ("BTC", 50000.0, 20.0)):
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

conn = sqlite3.connect(gap_bars_path)
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
for symbol, start, drift in (("ETH", 100.0, 0.25), ("BTC", 50000.0, 20.0)):
    price = start
    for idx in range(100):
        if idx == 90:
            price += drift
            continue
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

Path("/tmp/aiq-runtime-effective-config.yaml").write_text(
    "\n".join(
        [
            "global:",
            "  engine:",
            "    interval: 30m",
            "modes:",
            "  primary:",
            "    global:",
            "      engine:",
            "        interval: 5m",
            "  fallback:",
            "    global:",
            "      engine:",
            "        interval: 1h",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

AI_QUANT_STRATEGY_MODE= \
cargo run -q -p aiq-runtime -- paper effective-config --config /tmp/aiq-runtime-effective-config.yaml --json >/tmp/aiq-runtime-paper-effective-config-base.json
AI_QUANT_STRATEGY_MODE=primary \
cargo run -q -p aiq-runtime -- paper effective-config --config /tmp/aiq-runtime-effective-config.yaml --json >/tmp/aiq-runtime-paper-effective-config-primary.json
AI_QUANT_STRATEGY_MODE=fallback \
cargo run -q -p aiq-runtime -- paper effective-config --config /tmp/aiq-runtime-effective-config.yaml --json >/tmp/aiq-runtime-paper-effective-config-fallback.json

python3 - <<'PY'
import json
from pathlib import Path

base = json.loads(Path("/tmp/aiq-runtime-paper-effective-config-base.json").read_text(encoding="utf-8"))
primary = json.loads(Path("/tmp/aiq-runtime-paper-effective-config-primary.json").read_text(encoding="utf-8"))
fallback = json.loads(Path("/tmp/aiq-runtime-paper-effective-config-fallback.json").read_text(encoding="utf-8"))

assert base["interval"] == "30m"
assert primary["interval"] == "5m"
assert fallback["interval"] == "1h"
assert base["config_id"] != primary["config_id"]
assert primary["config_id"] != fallback["config_id"]
PY

AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example \
AI_QUANT_DB_PATH=/tmp/aiq-runtime-paper.db \
AI_QUANT_CANDLES_DB_PATH=/tmp/aiq-runtime-candles.db \
AI_QUANT_SYMBOLS=ETH,SOL \
AI_QUANT_LOOKBACK_BARS=200 \
cargo run -q -p aiq-runtime -- paper manifest --json >/tmp/aiq-runtime-paper-manifest.json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example \
AI_QUANT_DB_PATH=/tmp/aiq-runtime-paper.db \
AI_QUANT_CANDLES_DB_PATH=/tmp/aiq-runtime-candles.db \
AI_QUANT_SYMBOLS=ETH,SOL \
cargo run -q -p aiq-runtime -- paper status --status-path /tmp/aiq-runtime-paper-bootstrap.status.json --json >/tmp/aiq-runtime-paper-status-bootstrap.json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example \
AI_QUANT_DB_PATH=/tmp/aiq-runtime-paper.db \
AI_QUANT_CANDLES_DB_PATH=/tmp/aiq-runtime-candles.db \
AI_QUANT_SYMBOLS=ETH,SOL \
cargo run -q -p aiq-runtime -- paper service --status-path /tmp/aiq-runtime-paper-bootstrap.status.json --json >/tmp/aiq-runtime-paper-service-bootstrap.json
cargo run -q -p aiq-runtime -- snapshot validate --path /tmp/aiq-runtime-seed-snapshot.json --json >/tmp/aiq-runtime-snapshot-validate.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper.db --strict-replace --json >/tmp/aiq-runtime-seed-paper.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper-loop.db --strict-replace --json >/tmp/aiq-runtime-seed-paper-loop.json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/aiq-runtime-seed-snapshot.json --target-db /tmp/aiq-runtime-paper-loop-gap.db --strict-replace --json >/tmp/aiq-runtime-seed-paper-loop-gap.json
cargo run -q -p aiq-runtime -- paper doctor --db /tmp/aiq-runtime-paper.db --json >/tmp/aiq-runtime-paper-doctor.json
cargo run -q -p aiq-runtime -- paper run-once --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --target-symbol ETH --exported-at-ms 1772676900000 --dry-run --json >/tmp/aiq-runtime-paper-run-once.json
cargo run -q -p aiq-runtime -- paper cycle --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --step-close-ts-ms 1773426000000 --exported-at-ms 1772676900000 --json >/tmp/aiq-runtime-paper-cycle.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --start-step-close-ts-ms 1773422400000 --max-steps 2 --json >/tmp/aiq-runtime-paper-loop.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --max-steps 2 --json >/tmp/aiq-runtime-paper-loop-resume.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --max-steps 1 --json >/tmp/aiq-runtime-paper-loop-idle.json
cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --follow --idle-sleep-ms 1 --max-idle-polls 1 --max-steps 1 --json >/tmp/aiq-runtime-paper-loop-follow.json
cargo run -q -p aiq-runtime -- paper daemon --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --status-path /tmp/aiq-runtime-paper-daemon.status.json --idle-sleep-ms 1 --max-idle-polls 1 --json >/tmp/aiq-runtime-paper-daemon.json
cargo run -q -p aiq-runtime -- paper status --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --status-path /tmp/aiq-runtime-paper-daemon.status.json --json >/tmp/aiq-runtime-paper-status-stopped.json
cargo run -q -p aiq-runtime -- paper service --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --status-path /tmp/aiq-runtime-paper-daemon.status.json --json >/tmp/aiq-runtime-paper-service-stopped.json
cargo run -q -p aiq-runtime -- paper service apply --db /tmp/aiq-runtime-paper-loop.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --status-path /tmp/aiq-runtime-paper-daemon.status.json --action resume --json >/tmp/aiq-runtime-paper-service-apply.json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example \
AI_QUANT_DB_PATH=/tmp/aiq-runtime-paper-loop.db \
AI_QUANT_CANDLES_DB_PATH=/tmp/aiq-runtime-candles.db \
AI_QUANT_SYMBOLS=ETH \
cargo run -q -p aiq-runtime -- paper manifest --json >/tmp/aiq-runtime-paper-manifest-resume.json
cargo run -q -p aiq-runtime -- paper doctor --db /tmp/aiq-runtime-paper.db --live --json >/tmp/aiq-runtime-paper-doctor-live.json
cargo run -q -p aiq-runtime -- paper run-once --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --target-symbol ETH --exported-at-ms 1772676900000 --live --dry-run --json >/tmp/aiq-runtime-paper-run-once-live.json
if cargo run -q -p aiq-runtime -- paper cycle --db /tmp/aiq-runtime-paper.db --candles-db /tmp/aiq-runtime-candles.db --symbols ETH --step-close-ts-ms 1773426000000 --exported-at-ms 1772676900000 --json >/tmp/aiq-runtime-paper-cycle-rerun.json 2>/tmp/aiq-runtime-paper-cycle-rerun.stderr; then
  echo "paper cycle rerun guard did not fail closed" >&2
  exit 1
fi
if cargo run -q -p aiq-runtime -- paper loop --db /tmp/aiq-runtime-paper-loop-gap.db --candles-db /tmp/aiq-runtime-paper-loop-gap-candles.db --symbols ETH --start-step-close-ts-ms 1772832000000 --max-steps 2 --json >/tmp/aiq-runtime-paper-loop-gap.json 2>/tmp/aiq-runtime-paper-loop-gap.stderr; then
  echo "paper loop gap guard did not fail closed" >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import signal
import sqlite3
import time
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
assert sum(1 for warning in loop_follow["warnings"] if "paper loop idle:" in warning) == 1
assert any("follow exhausted" in warning for warning in loop_follow["warnings"])
assert "exact candle close" in Path("/tmp/aiq-runtime-paper-loop-gap.stderr").read_text(encoding="utf-8")
conn = sqlite3.connect("/tmp/aiq-runtime-paper-loop.db")
try:
    recorded_steps = conn.execute("SELECT COUNT(*) FROM runtime_cycle_steps").fetchone()[0]
finally:
    conn.close()
assert recorded_steps == 3
doctor = json.loads(Path("/tmp/aiq-runtime-paper-doctor.json").read_text(encoding="utf-8"))
assert doctor["paper_bootstrap"]["runtime_close_markers"] == 1
manifest = json.loads(Path("/tmp/aiq-runtime-paper-manifest.json").read_text(encoding="utf-8"))
assert manifest["interval"] == "30m"
assert manifest["lookback_bars"] == 200
assert manifest["symbols"] == ["ETH", "SOL"]
assert manifest["candles_db"] == "/tmp/aiq-runtime-candles.db"
assert manifest["watch_symbols_file"] is False
assert manifest["resume"]["launch_state"] == "bootstrap_required"
assert manifest["resume"]["launch_ready"] is False
assert manifest["status_path"].endswith(".status.json")
assert "--status-path" in manifest["daemon_command"]
status_bootstrap = json.loads(Path("/tmp/aiq-runtime-paper-status-bootstrap.json").read_text(encoding="utf-8"))
assert status_bootstrap["service_state"] == "bootstrap_required"
assert status_bootstrap["status_file_present"] is False
service_bootstrap = json.loads(Path("/tmp/aiq-runtime-paper-service-bootstrap.json").read_text(encoding="utf-8"))
assert service_bootstrap["desired_action"] == "hold"
assert "start-step-close-ts-ms" in service_bootstrap["action_reason"]
manifest_resume = json.loads(Path("/tmp/aiq-runtime-paper-manifest-resume.json").read_text(encoding="utf-8"))
assert manifest_resume["resume"]["launch_state"] == "caught_up_idle"
assert manifest_resume["resume"]["launch_ready"] is True
assert manifest_resume["resume"]["last_applied_step_close_ts_ms"] == 1773426000000
assert manifest_resume["resume"]["next_due_step_close_ts_ms"] == 1773427800000
status_stopped = json.loads(Path("/tmp/aiq-runtime-paper-status-stopped.json").read_text(encoding="utf-8"))
assert status_stopped["service_state"] == "stopped"
assert status_stopped["status_file_present"] is True
assert status_stopped["daemon_status"]["running"] is False
service_stopped = json.loads(Path("/tmp/aiq-runtime-paper-service-stopped.json").read_text(encoding="utf-8"))
assert service_stopped["desired_action"] == "start"
assert service_stopped["status"]["service_state"] == "stopped"
assert service_stopped["daemon_command"][0:3] == ["aiq-runtime", "paper", "daemon"]
service_apply = json.loads(Path("/tmp/aiq-runtime-paper-service-apply.json").read_text(encoding="utf-8"))
assert service_apply["requested_action"] == "resume"
assert service_apply["applied_action"] == "start"
spawned_pid = service_apply["spawned_pid"]
assert isinstance(spawned_pid, int) and spawned_pid > 0
status_path = Path("/tmp/aiq-runtime-paper-daemon.status.json")
deadline = time.time() + 5.0
while True:
    running_status = json.loads(status_path.read_text(encoding="utf-8"))
    if running_status["pid"] == spawned_pid and running_status["running"] is True:
        break
    assert time.time() <= deadline, "paper service apply did not publish a running status in time"
    time.sleep(0.05)
os.kill(spawned_pid, signal.SIGTERM)
deadline = time.time() + 5.0
while True:
    stopped_status = json.loads(status_path.read_text(encoding="utf-8"))
    if stopped_status["pid"] == spawned_pid and stopped_status["running"] is False:
        break
    assert time.time() <= deadline, "paper service apply cleanup did not publish a stopped status in time"
    time.sleep(0.05)
PY

echo "[runtime-foundation] ok"
