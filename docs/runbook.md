# Operations Runbook

Procedures for common operational scenarios. When the bot misbehaves, follow the relevant section below.

## Services Reference

| Service | Unit name | Purpose |
|---------|-----------|---------|
| Paper trader (primary) | `openclaw-ai-quant-trader-v8-paper1` | Primary paper trading daemon |
| Paper trader (candidate #2) | `openclaw-ai-quant-trader-v8-paper2` | Candidate paper trading daemon |
| Paper trader (candidate #3) | `openclaw-ai-quant-trader-v8-paper3` | Candidate paper trading daemon |
| Live trader | `openclaw-ai-quant-live-v8` | Live trading daemon |
| WS sidecar | `openclaw-ai-quant-ws-sidecar` | Market data WebSocket |
| Monitor | `openclaw-ai-quant-monitor` | Real-time dashboard |

All services are systemd user units. Manage with `systemctl --user <action> <unit>`.

### Optional timers (nightly factory + log pruning + replay alignment gate)

Example service/timer templates live under `systemd/`:

- `openclaw-ai-quant-factory-v8.{service,timer}.example` runs `factory_run.py` nightly.
- `openclaw-ai-quant-prune-runtime-logs-v8.{service,timer}.example` prunes SQLite `runtime_logs` daily.
- `openclaw-ai-quant-replay-alignment-gate.{service,timer}.example` runs deterministic replay alignment checks and writes a release-blocker status file.

Install (example):

```bash
mkdir -p ~/.config/systemd/user
install -m 0644 systemd/openclaw-ai-quant-factory-v8.service.example \
  ~/.config/systemd/user/openclaw-ai-quant-factory-v8.service
install -m 0644 systemd/openclaw-ai-quant-factory-v8.timer.example \
  ~/.config/systemd/user/openclaw-ai-quant-factory-v8.timer
install -m 0644 systemd/openclaw-ai-quant-prune-runtime-logs-v8.service.example \
  ~/.config/systemd/user/openclaw-ai-quant-prune-runtime-logs-v8.service
install -m 0644 systemd/openclaw-ai-quant-prune-runtime-logs-v8.timer.example \
  ~/.config/systemd/user/openclaw-ai-quant-prune-runtime-logs-v8.timer
install -m 0644 systemd/openclaw-ai-quant-replay-alignment-gate.service.example \
  ~/.config/systemd/user/openclaw-ai-quant-replay-alignment-gate.service
install -m 0644 systemd/openclaw-ai-quant-replay-alignment-gate.timer.example \
  ~/.config/systemd/user/openclaw-ai-quant-replay-alignment-gate.timer
systemctl --user daemon-reload
systemctl --user enable --now openclaw-ai-quant-factory-v8.timer
systemctl --user enable --now openclaw-ai-quant-prune-runtime-logs-v8.timer
systemctl --user enable --now openclaw-ai-quant-replay-alignment-gate.timer
```

Configure runtime log retention via:

```
AI_QUANT_RUNTIME_LOG_KEEP_DAYS=14
```

Example replay alignment gate environment (`~/.config/openclaw/ai-quant-replay-gate.env`):

```bash
AI_QUANT_REPLAY_GATE_CANDLES_DB=/home/<user>/openclaw-plugins/ai_quant/candles_dbs/candles_1h.db
AI_QUANT_REPLAY_GATE_FUNDING_DB=/home/<user>/openclaw-plugins/ai_quant/backtester/data/funding_rates_1h.db
AI_QUANT_REPLAY_GATE_INTERVAL=1h
AI_QUANT_REPLAY_GATE_WINDOW_MINUTES=240
AI_QUANT_REPLAY_GATE_LAG_MINUTES=2
AI_QUANT_REPLAY_GATE_MIN_LIVE_TRADES=1
AI_QUANT_REPLAY_GATE_STRICT_NO_RESIDUALS=1
AI_QUANT_REPLAY_GATE_BUNDLE_ROOT=/tmp/openclaw-ai-quant/replay_gate
AI_QUANT_REPLAY_GATE_BLOCKER_FILE=/tmp/openclaw-ai-quant/replay_gate/release_blocker.json
```

Manual trigger:

```bash
systemctl --user start openclaw-ai-quant-replay-alignment-gate.service
```

Check blocker status:

```bash
cat /tmp/openclaw-ai-quant/replay_gate/release_blocker.json
```

Deployment tooling is fail-closed against this blocker by default:

- `python tools/paper_deploy.py ...`
- `python tools/deploy_sweep.py ...`
- `python tools/promote_to_live.py --apply ...`

These commands now abort when the blocker is red/missing/stale. Override only for emergency/manual workflows:

```bash
python tools/paper_deploy.py ... --ignore-replay-gate
python tools/deploy_sweep.py ... --ignore-replay-gate
python tools/promote_to_live.py --apply ... --ignore-replay-gate
```

### Secrets management

Do not store secrets in the repo. Recommended locations:

- Hyperliquid key material: `~/.config/openclaw/ai-quant-secrets.json` (chmod 600)
- Service environment: `~/.config/openclaw/ai-quant-live-v8.env`
  - Put alerting targets (e.g. `AI_QUANT_ALERT_TARGETS`) here, especially if using webhook URLs.

---

## 1. Pause Trading (Emergency Stop)

Use when: unexpected losses, erratic behaviour, exchange issues, or any situation requiring immediate halt.

### Option A: Kill-switch via environment (close-only)

Blocks new entries but allows exits to close positions.

```bash
# Close-only mode (recommended default)
export AI_QUANT_KILL_SWITCH=close_only
systemctl --user restart openclaw-ai-quant-live-v8

# Full halt — blocks ALL orders including exits
export AI_QUANT_KILL_SWITCH=halt_all
systemctl --user restart openclaw-ai-quant-live-v8
```

### Option B: Kill-switch via file (no restart needed)

The `RiskManager` polls this file periodically.

```bash
# Close-only
echo "close_only" > /tmp/ai-quant-kill

# Full halt
echo "halt_all" > /tmp/ai-quant-kill
```

Set the file path in the service env:
```
AI_QUANT_KILL_SWITCH_FILE=/tmp/ai-quant-kill
```

### Option C: Stop the service entirely

```bash
# Stop live trading
systemctl --user stop openclaw-ai-quant-live-v8

# Stop paper trading
systemctl --user stop openclaw-ai-quant-trader-v8-paper1
```

### Clearing the kill-switch

```bash
# Remove file-based kill
rm /tmp/ai-quant-kill

# Clear env-based kill — unset and restart
unset AI_QUANT_KILL_SWITCH
systemctl --user restart openclaw-ai-quant-live-v8
```

### Verification

```bash
# Check service status
systemctl --user status openclaw-ai-quant-live-v8

# Check logs for kill-switch activation
journalctl --user -u openclaw-ai-quant-live-v8 --since "10 min ago" | grep -i kill
```

---

## 1a. Emergency Flatten ("Flat Now") (AQC-808)

Use when: you need to *both* pause entries and flatten positions under a known unwind policy.

Policy (default):

- Write a kill-switch file in `close_only` mode (pause new entries, allow exits).
- Flatten positions:
  - Paper: clear `position_state` and insert `SYSTEM CLOSE` trades.
  - Live (optional): submit market-close (reduce-only IOC) per open position with retries.
- Leave the kill-switch in place until post-incident review is complete.

### Paper flatten + pause (safe default)

```bash
python tools/flat_now.py \
  --kill-file /tmp/ai-quant-kill \
  --pause-mode close_only \
  --paper \
  --yes
```

### Live flatten + pause (explicit, destructive)

Live flatten requires `--yes`. Secrets are loaded from `AI_QUANT_SECRETS_PATH` (or `~/.config/openclaw/ai-quant-secrets.json`).

```bash
python tools/flat_now.py \
  --kill-file /tmp/ai-quant-kill \
  --pause-mode close_only \
  --live \
  --yes
```

To flatten both paper and live in one command:

```bash
python tools/flat_now.py \
  --kill-file /tmp/ai-quant-kill \
  --pause-mode close_only \
  --paper \
  --live \
  --yes
```

Notes:

- Avoid `halt_all` when you want exits to run in-engine; it blocks *all* orders including exits.
- This tool does not clear the kill-switch automatically. Clear it manually once you are confident it is safe.

## 1b. Strategy Mode Switching (AQC-1002)

The paper service contract supports an optional strategy-mode selector. Rust
paper surfaces resolve the selector in the same order as the current Python
paper service: `AI_QUANT_STRATEGY_MODE` first, then
`AI_QUANT_STRATEGY_MODE_FILE` when the env var is unset.

The engine supports an optional strategy-mode overlay selected via `AI_QUANT_STRATEGY_MODE`:

- `primary`: 30m/5m
- `fallback`: 1h/5m
- `conservative`: 1h/15m
- `flat`: safety profile (use with a kill-switch when needed)

Mode overlays are defined under `modes:` in `config/strategy_overrides.yaml`.
For paper lanes, mode changes now re-resolve through the Rust `paper effective-config`
contract before `StrategyManager` switches to the new materialised YAML path.

### Manual mode change

```bash
export AI_QUANT_STRATEGY_MODE=fallback

# Restart is required if the mode changes global.engine.interval.
systemctl --user restart openclaw-ai-quant-live-v8
```

### Automatic step-down on kill events

When enabled (`AI_QUANT_MODE_SWITCH_ENABLE=1`), the live daemon steps down one mode on each new kill event
and persists the selected mode to `AI_QUANT_STRATEGY_MODE_FILE` (default: `artifacts/state/strategy_mode.txt`).

If your systemd unit is configured to auto-restart on exit, you can also enable:

- `AI_QUANT_MODE_SWITCH_EXIT_ON_RESTART_REQUIRED=1`

This makes the daemon exit when switching to/from `primary` (where an interval change is likely), allowing
systemd to restart it with the new persisted mode.

---

## 1c. Ensemble Runner (AQC-1003)

The engine supports running a small ensemble (2-3 strategies) by launching multiple daemons with different configs.

This is implemented as a process runner (`tools/ensemble_runner.py`), not an in-process multi-strategy engine.

### Key points

- Each strategy has its own sizing budget via config overrides (e.g. `global.trade.size_multiplier`).
- Global risk caps (portfolio heat/exposure/kill-switch) are enforced by the existing RiskManager logic.
- Start with `dry_live` until you are confident the ensemble behaves as expected.

### Example

Edit the example spec:

- `config/ensemble.example.yaml`

Dry-run the plan:

```bash
python tools/ensemble_runner.py --spec config/ensemble.example.yaml
```

Launch the ensemble in dry-live:

```bash
python tools/ensemble_runner.py \
  --spec config/ensemble.example.yaml \
  --mode dry_live \
  --yes
```

## 2. Roll Back Config

Use when: a bad config was deployed and needs to be reverted to last-known-good.

### Step 1: Identify the previous config

```bash
# Check git log for config changes
git log --oneline -10 -- config/strategy_overrides.yaml
```

### Step 2: Restore the previous version

```bash
# Restore from git (replace <commit> with the last-known-good commit)
git checkout <commit> -- config/strategy_overrides.yaml
```

### Step 3: Verify the rollback

```bash
# Diff against current to confirm the change
git diff config/strategy_overrides.yaml

# Validate config parity
uv run python tools/validate_config.py
```

### Step 4: Reload

YAML config changes hot-reload automatically via mtime polling. No restart is needed unless `engine.interval` was changed.

```bash
# Confirm hot-reload happened (look for "Config reloaded" in logs)
journalctl --user -u openclaw-ai-quant-live-v8 --since "2 min ago" | grep -i reload
```

If `engine.interval` changed, a restart is required:

```bash
systemctl --user restart openclaw-ai-quant-live-v8
```

---

## 3. Verify Database Integrity

Use when: suspecting data corruption, missing candles, or stale market data.

### Candle database freshness

```bash
# Check all candle DBs — row counts and time range
for db in candles_dbs/candles_*.db; do
    echo "=== $db ==="
    sqlite3 "$db" "SELECT COUNT(*) AS rows, datetime(MIN(t)/1000, 'unixepoch') AS earliest, datetime(MAX(t)/1000, 'unixepoch') AS latest, datetime(MAX(t_close)/1000, 'unixepoch') AS latest_close FROM candles;"
done
```

### Candle retention beyond 5,000 bars (AQC-203)

Hyperliquid only backfills roughly the last 5,000 bars per interval via `/info` (`candleSnapshot`). To build multi-month 3m/5m histories, the WS sidecar can keep the local candle DB append-only (no pruning) for selected intervals.

#### Migration / rollout

1. Deploy a `openclaw-ai-quant-ws-sidecar` build that includes AQC-203.
2. Ensure the service environment contains:
   - `AI_QUANT_CANDLE_PRUNE_DISABLE_INTERVALS=3m,5m`
   - Set an empty value to re-enable pruning for all intervals: `AI_QUANT_CANDLE_PRUNE_DISABLE_INTERVALS=`
3. Restart the sidecar service:
   - `systemctl --user restart openclaw-ai-quant-ws-sidecar`

No schema changes are required. Existing DB files are reused.

Notes:
- `Append-only beyond 5,000` is prospective. You cannot backfill older-than-5,000 bars from Hyperliquid, but once collected the sidecar will retain them locally.

#### Verification

```bash
# Per-symbol row counts should exceed 5,000 over time
sqlite3 candles_dbs/candles_3m.db "SELECT symbol, COUNT(*) AS rows FROM candles GROUP BY symbol ORDER BY rows DESC LIMIT 10;"
sqlite3 candles_dbs/candles_5m.db "SELECT symbol, COUNT(*) AS rows FROM candles GROUP BY symbol ORDER BY rows DESC LIMIT 10;"

# Duplicate check (should be 0; PK is (symbol, interval, t))
sqlite3 candles_dbs/candles_3m.db "SELECT COUNT(*) AS dupes FROM (SELECT symbol, interval, t, COUNT(*) AS c FROM candles GROUP BY symbol, interval, t HAVING c > 1);"
sqlite3 candles_dbs/candles_5m.db "SELECT COUNT(*) AS dupes FROM (SELECT symbol, interval, t, COUNT(*) AS c FROM candles GROUP BY symbol, interval, t HAVING c > 1);"
```

#### Disk growth expectations

3m adds ~480 rows/day/symbol, 5m adds ~288 rows/day/symbol. With pruning disabled for both, plan disk accordingly and monitor `candles_dbs/candles_{3m,5m}.db` file sizes.

### Candle DB partitioning / archival (AQC-204)

If you retain multi-month histories (especially 3m/5m), a single `candles_{interval}.db` file can grow large. Partitioning keeps the "hot" DB small while preserving long history in monthly archive DBs.

This repo uses monthly SQLite partitions created by `tools/partition_candles_db.py`:

- Hot DB (written by the WS sidecar): `candles_dbs/candles_{interval}.db`
- Archive partitions: `candles_dbs/partitions/{interval}/candles_{interval}_YYYY-MM.db`

#### Partitioning procedure

Prefer to stop the sidecar before applying changes:

```bash
systemctl --user stop openclaw-ai-quant-ws-sidecar
```

Dry-run (no writes):

```bash
uv run python tools/partition_candles_db.py --interval 5m
```

Apply copy-only:

```bash
uv run python tools/partition_candles_db.py --interval 5m --keep-days 120 --apply
```

Apply copy + delete (shrinks the hot DB over time):

```bash
uv run python tools/partition_candles_db.py --interval 5m --keep-days 120 --apply --delete
```

Optional: run `VACUUM` after deletion (slow, but compacts the file):

```bash
uv run python tools/partition_candles_db.py --interval 5m --keep-days 120 --apply --delete --vacuum
```

Restart the sidecar afterwards:

```bash
systemctl --user start openclaw-ai-quant-ws-sidecar
```

#### Backtester usage across partitions

The backtester can load candles from a comma-separated list of DB paths and/or a directory containing partition DBs.

Example (hot DB + partitions dir):

```bash
./target/release/mei-backtester replay \
  --interval 5m \
  --candles-db candles_dbs/candles_5m.db,candles_dbs/partitions/5m \
  --config config/strategy_overrides.yaml
```

### BBO snapshot database (AQC-206)

Optional, sampled best-bid/best-ask (BBO) snapshots are stored in SQLite for slippage modelling and post-trade analysis.

#### Enablement

- Enable BBO subscriptions: `AI_QUANT_WS_ENABLE_BBO=1`
- Optional: drive mid-price updates from BBO ticks (lower latency for monitor/UI): `AI_QUANT_MIDS_FROM_BBO=1`
- Enable snapshots: `AI_QUANT_BBO_SNAPSHOTS_ENABLE=1`

Tuning knobs:
- `AI_QUANT_BBO_SNAPSHOTS_DB_PATH` (default: `$AI_QUANT_CANDLES_DB_DIR/bbo_snapshots.db`)
- `AI_QUANT_BBO_SNAPSHOTS_SAMPLE_MS` (per-symbol insert throttle; default: 1000)
- `AI_QUANT_BBO_SNAPSHOTS_MAX_QUEUE` (bounded in-memory queue; default: 20000; drops when full)
- `AI_QUANT_BBO_SNAPSHOTS_RETENTION_HOURS` (time-based retention; default: 24)
- `AI_QUANT_BBO_SNAPSHOTS_RETENTION_SWEEP_SECS` (sweep interval; default: 600; min: 30)
- `AI_QUANT_MIDS_FROM_BBO_FALLBACK_AGE_S` (when BBO-driven mids are enabled, use `allMids` fallback once BBO age exceeds this threshold; default: 5.0)

Snapshots are only written for symbols the sidecar is subscribed to (the union of live client requests and `AI_QUANT_SIDECAR_SYMBOLS`).

#### Verification

```bash
# Row counts + time range per symbol
sqlite3 candles_dbs/bbo_snapshots.db \
  "SELECT symbol, COUNT(*) AS rows, datetime(MIN(ts_ms)/1000, 'unixepoch') AS earliest, datetime(MAX(ts_ms)/1000, 'unixepoch') AS latest FROM bbo_snapshots GROUP BY symbol ORDER BY rows DESC LIMIT 10;"

# Quick retention sanity check (should trend to 0 once retention < 48h)
sqlite3 candles_dbs/bbo_snapshots.db \
  "SELECT COUNT(*) AS older_than_48h FROM bbo_snapshots WHERE ts_ms < (strftime('%s','now') - 48*3600) * 1000;"
```

### Trading database integrity

```bash
# Paper DB
sqlite3 trading_engine.db "PRAGMA integrity_check;"

# Live DB
sqlite3 trading_engine_live.db "PRAGMA integrity_check;"
```

### Reality check (quick diagnostics)

```bash
# Check recent trading activity for a specific symbol
uv run python tools/reality_check.py --symbol BTC --hours 2
```

### Funding rate database

```bash
# Check freshness
sqlite3 candles_dbs/funding_rates.db \
    "SELECT symbol, COUNT(*) AS rows, MAX(time) AS latest FROM funding_rates GROUP BY symbol ORDER BY latest DESC LIMIT 10;"

# Automated freshness + anomaly checks (JSON on stdout; summary on stderr)
uv run python tools/check_funding_rates_db.py --lookback-hours 72 --max-gap-hours 4

# Backfill if stale
uv run python tools/fetch_funding_rates.py --days 7
```

### Universe history database (AQC-205)

Tracks when symbols appear/disappear in the Hyperliquid perp universe to support survivorship-bias-aware backtests.

```bash
# Sync current universe snapshot (recommended: run hourly via cron)
uv run python tools/sync_universe_history.py

# Inspect derived listing/delisting bounds
sqlite3 candles_dbs/universe_history.db \
    "SELECT symbol, first_seen_ms, last_seen_ms FROM universe_listings ORDER BY last_seen_ms DESC LIMIT 20;"
```

Backtester integration:

```bash
# Replay with universe filtering enabled (keeps symbols whose listing interval overlaps the backtest window)
./target/release/mei-backtester replay \
    --candles-db candles_dbs/candles_5m.db \
    --config config/strategy_overrides.yaml \
    --universe-filter

# Sweep with the same filter
./target/release/mei-backtester sweep \
    --candles-db candles_dbs/candles_5m.db \
    --sweep-spec backtester/sweeps/smoke.yaml \
    --universe-filter
```

Notes:
- The universe filter uses `universe_listings.first_seen_ms` / `last_seen_ms` derived from local snapshots. If you have not been running the sync script for the period you are testing, the filter may exclude symbols unexpectedly.
- The universe DB defaults to `<candles_db_dir>/universe_history.db`, so it follows your `--candles-db` location (useful when running the backtester from within `backtester/`).

### WebSocket sidecar health

```bash
# Check sidecar is running and connected
systemctl --user status openclaw-ai-quant-ws-sidecar

# Check sidecar logs for disconnects/errors
journalctl --user -u openclaw-ai-quant-ws-sidecar --since "1 hour ago" | grep -iE "error|disconnect|reconnect"
```

---

## 4. Rerun Validation

Use when: re-validating a config after a pause, before re-deploying, or as part of nightly checks.

### Single config replay (CPU)

```bash
# Replay with current config on 5m candles
./target/release/mei-backtester replay \
    --candles-db candles_dbs/candles_5m.db \
    --config config/strategy_overrides.yaml
```

### Slippage stress test

```bash
# Test at 10, 20, 30 bps slippage
for bps in 10 20 30; do
    echo "=== Slippage: ${bps} bps ==="
    ./target/release/mei-backtester replay \
        --candles-db candles_dbs/candles_5m.db \
        --config config/strategy_overrides.yaml \
        --slippage-bps "$bps"
done
```

### Config parity validation

```bash
# Verify Python defaults match Rust defaults and YAML is fully explicit
uv run python tools/validate_config.py
```

### Indicator parity check

```bash
# Export Rust indicators and compare against Python ta library
./target/release/mei-backtester dump-indicators \
    --candles-db candles_dbs/candles_1h.db \
    --symbol BTC
```

---

## 5. Restart Services

Use when: deploying code changes, recovering from crashes, or after system maintenance.

### Restart order (recommended)

Services should be restarted in dependency order:

```bash
# 1. WebSocket sidecar first (data source)
systemctl --user restart openclaw-ai-quant-ws-sidecar

# 2. Wait for sidecar to connect (check logs)
sleep 5
journalctl --user -u openclaw-ai-quant-ws-sidecar --since "10 sec ago" | tail -3

# 3. Trading daemon
systemctl --user restart openclaw-ai-quant-live-v8    # or openclaw-ai-quant-trader-v8-paper1 for paper

# 4. Monitor (optional, non-critical)
systemctl --user restart openclaw-ai-quant-monitor
```

### Full stop (all services)

```bash
systemctl --user stop openclaw-ai-quant-live-v8
systemctl --user stop openclaw-ai-quant-trader-v8-paper1
systemctl --user stop openclaw-ai-quant-monitor
systemctl --user stop openclaw-ai-quant-ws-sidecar
```

### Full start (all services)

```bash
systemctl --user start openclaw-ai-quant-ws-sidecar
sleep 5
systemctl --user start openclaw-ai-quant-trader-v8-paper1     # paper
# systemctl --user start openclaw-ai-quant-live-v8     # live (uncomment when ready)
systemctl --user start openclaw-ai-quant-monitor
```

### Post-restart health check

```bash
# Verify all services are active
systemctl --user status openclaw-ai-quant-ws-sidecar openclaw-ai-quant-live-v8 openclaw-ai-quant-monitor

# Check for errors in the last minute
journalctl --user -u openclaw-ai-quant-live-v8 --since "1 min ago" | grep -iE "error|exception|traceback"

# Confirm config was loaded
journalctl --user -u openclaw-ai-quant-live-v8 --since "1 min ago" | grep -i "config"
```

---

## 6. Export State for Debugging

Use when: you need a snapshot of the current trader state for analysis, replay, or deterministic paper seeding.

```bash
# Export paper state with the Rust continuation path
cargo run -p aiq-runtime -- \
  snapshot export-paper --db ./trading_engine.db --output /tmp/paper_state.json

# Validate the exported snapshot before replay or seeding
cargo run -p aiq-runtime -- \
  snapshot validate --path /tmp/paper_state.json --json

# Export live state
uv run python tools/export_state.py --source live --output /tmp/live_state.json

# Export trade history to CSV
uv run python tools/export_csv.py
```

### Seed paper from a validated snapshot

Use when: you need deterministic paper/bootstrap parity from a known `init-state v2` artefact.

```bash
cargo run -p aiq-runtime -- \
  snapshot seed-paper \
  --snapshot /tmp/paper_state.json \
  --target-db ./trading_engine.db \
  --strict-replace \
  --json
```

Safe operator expectations:

- `snapshot validate` must succeed before `snapshot seed-paper`
- `--strict-replace` is the deterministic bootstrap mode
- the current Rust seed path rewrites `trades`, `position_state`, `position_state_history`, `runtime_cooldowns`, and `runtime_last_closes`
- if `--strict-replace` is omitted and stale open paper positions would remain, the command fails closed

### Inspect Rust paper bootstrap state

Use when: you need to prove that Rust can restore paper balance, positions, and cooldown markers from the current paper DB before replacing the Python bootstrap path.

```bash
cargo run -p aiq-runtime -- \
  paper doctor \
  --db ./trading_engine.db \
  --json
```

Expected operator signals:

- `runtime_bootstrap.mode = "paper"`
- a resolved pipeline profile is present
- `paper_bootstrap.position_count` matches the DB-backed paper positions
- `paper_bootstrap.runtime_entry_markers` / `runtime_exit_markers` match `runtime_cooldowns`

### Execute one Rust paper step

Use when: you want to prove that Rust can restore paper state, make one decision from local candle data, and project the resulting paper writes back into SQLite without starting a daemon.

```bash
cargo run -p aiq-runtime -- \
  paper run-once \
  --db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_30m.db \
  --target-symbol ETH \
  --exported-at-ms 1772676900000 \
  --dry-run \
  --json
```

Operational expectations:

- `paper run-once` is single-shot, not a continuous loop
- `--dry-run` should be the default diagnostic path during bring-up
- `--candles-db` must contain both the target symbol and the BTC anchor symbol at the resolved `engine.interval`
- default write timestamps follow execution time; pass `--exported-at-ms` when you need reproducible report/write artefacts for parity debugging
- a healthy report shows restored paper state, decision/fill counts, projected action codes, and whether DB writes were skipped or applied

### Execute one repeatable Rust paper cycle

Use when: you want one full repeatable Rust paper iteration across an explicit symbol set plus any already-open paper positions, without starting a daemon.

```bash
cargo run -p aiq-runtime -- \
  paper cycle \
  --db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_30m.db \
  --symbols ETH,SOL \
  --step-close-ts-ms 1773426000000 \
  --exported-at-ms 1772676900000 \
  --dry-run \
  --json
```

Operational expectations:

- `paper cycle` is still a shell, not a long-running daemon
- `--step-close-ts-ms` is the repeatable cycle identity and drives decision/cooldown timestamps
- write mode records a rerun guard in `runtime_cycle_steps`; reapplying the same step fails closed
- explicit `--symbols` are unioned with any open paper positions so exits are never skipped
- `BTC` may still be part of the active cycle even when it is also the anchor symbol
- `--candles-db` must contain target bars plus the BTC anchor symbol at the resolved `engine.interval`
- all resolved cycle symbols must share the same `engine.interval`; mixed per-symbol interval overrides fail closed
- the write path refreshes `trades`, `position_state`, `runtime_cooldowns`, and `runtime_last_closes` inside one immediate transaction

### Execute one bounded Rust paper loop shell

Use when: you want the Rust runtime to catch up one or more unapplied paper cycle steps and then exit, without starting a daemon.

```bash
cargo run -p aiq-runtime -- \
  paper loop \
  --db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_30m.db \
  --symbols ETH,SOL \
  --start-step-close-ts-ms 1773424200000 \
  --max-steps 2 \
  --dry-run \
  --json
```

Operational expectations:

- `paper loop` is still a shell, not a long-running daemon
- the command resumes from `runtime_cycle_steps` when prior Rust cycle state exists
- `--start-step-close-ts-ms` is required only when no prior matching `runtime_cycle_steps` rows exist for the current config fingerprint / interval / live lane
- each executed loop step reuses the same `paper cycle` contract and records the same rerun guard rows on write mode
- when `--symbols-file` is supplied, the loop loads that file once at start-up and keeps a bounded manifest for the life of the shell
- dry-run uses an isolated temporary paper DB copy so multi-step previews carry forward projected Rust state without mutating the real paper DB
- when `--exported-at-ms` is omitted, each planned step uses its own `step_close_ts_ms` as the snapshot export timestamp for deterministic catch-up artefacts
- every due step must have an exact candle close for each active symbol and the BTC anchor; missing bar closes fail closed instead of being silently marked as applied
- the loop stops cleanly when the next due step is newer than the latest common candle close across the explicit symbols, open paper positions, and BTC anchor

Follow-mode example:

```bash
cargo run -p aiq-runtime -- \
  paper loop \
  --db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_30m.db \
  --symbols ETH,SOL \
  --follow \
  --idle-sleep-ms 5000 \
  --max-idle-polls 12 \
  --json
```

Follow-mode expectations:

- `--follow` keeps the shell alive after catch-up and polls for the next due step instead of returning immediately idle
- `--idle-sleep-ms` controls how long the shell sleeps between no-work polls
- `--max-idle-polls 0` means unbounded follow mode; any positive value caps the number of idle polls before the shell exits with a warning
- `--max-idle-polls 1` exits on the first no-work poll, so `idle_polls` reports `1` and the shell does not sleep again before returning
- `--symbols-file` still behaves as a one-shot start-up manifest in follow mode; use `paper daemon --watch-symbols-file` when you need reloadable watchlist orchestration
- `paper loop --follow` still shares the same step identity and write contract as `paper cycle`, but it is no longer the owner of watchlist reload semantics

### Run the opt-in Rust paper daemon wrapper

Use when: you want a long-running Rust paper orchestration lane that keeps the
existing `paper cycle` contract alive between due steps, without claiming
Python daemon cutover. This is also the first Rust lane that can optionally
watch a symbols manifest file, pick up later watchlist refreshes without
restarting, and retain the last good manifest if a reload is invalid or
runtime-invalid malformed.

```bash
cargo run -p aiq-runtime -- \
  paper daemon \
  --db ./trading_engine.db \
  --candles-db ./candles_dbs/candles_30m.db \
  --symbols-file ./tmp/paper-watchlist.txt \
  --watch-symbols-file \
  --status-path ./artifacts/state/paper-daemon.status.json \
  --idle-sleep-ms 5000 \
  --max-idle-polls 0 \
  --json
```

Operational expectations:

- `paper daemon` is opt-in orchestration only; Python `engine.daemon` remains the active paper runtime path in this phase
- the daemon reuses the same restored state contract as `paper doctor` and the same rerun guard / DB write contract as `paper cycle`
- `--start-step-close-ts-ms` is required only when no prior matching `runtime_cycle_steps` rows exist for the current config fingerprint / interval / live lane
- `--symbols-file` loads the daemon manifest once at start-up; add `--watch-symbols-file` when that manifest should reload on file changes without restarting the process
- reload failures do not clear the lane: the daemon retains the last good manifest and emits a warning instead, including runtime-invalid but still UTF-8-clean payloads that fail daemon preflight
- active symbols remain `manifest ∪ open paper positions`, so exit coverage is preserved while the manifest changes
- `--lock-path` may be supplied when the Rust daemon lane needs an isolated lock namespace; changing the lock path does not widen DB projections
- `--status-path` may be supplied when operators want the daemon lifecycle JSON in a specific location; otherwise the daemon derives it from the resolved lock path or `AI_QUANT_STATUS_PATH`
- `--dry-run` remains the safest bring-up path while the surface is still opt-in
- manual `paper daemon` launch is still supported, but the preferred conventional lane entrypoint is now `paper lane daemon --lane <paper1|paper2|paper3|livepaper>` so the example service contract is resolved by Rust instead of being retyped by hand
- if you only need bounded catch-up or a short follow poll budget, use `paper loop` directly instead of `paper daemon`

### Run the conventional Rust paper lane wrapper

Use when: you want the standard `paper1` / `paper2` / `paper3` / `livepaper`
service contract resolved from a worktree root, without reconstructing the YAML
/ DB / lock / status mapping by hand. This is the launch path used by the
example paper systemd units in this phase.

```bash
mkdir -p ./artifacts/state
touch ./artifacts/state/paper_watchlist_paper1.txt

cargo run -p aiq-runtime -- \
  paper lane daemon \
  --lane paper1 \
  --project-dir . \
  --symbols-file ./artifacts/state/paper_watchlist_paper1.txt \
  --watch-symbols-file \
  --lookback-bars 200 \
  --json
```

Conventional lane expectations:

- `paper lane` resolves `paper1`, `paper2`, `paper3`, and `livepaper` onto the conventional `config/strategy_overrides.<lane>.yaml`, `trading_engine_v8_<lane>.db`, `ai_quant_paper_v8_<lane>.lock`, `ai_quant_paper_v8_<lane>.status.json`, and `artifacts/state/strategy_mode_v8_<lane>.txt` contracts under the selected `--project-dir`
- `paper lane daemon` still executes the same Rust `paper daemon` write path; it simply binds the example lane mapping before launch
- `paper lane manifest` / `status` / `service` / `apply` mirror the underlying Rust service planner and supervisor surfaces for the same conventional lane contract
- `paper1` / `paper2` / `paper3` automatically inject the conventional promoted-role and strategy-mode pair (`primary`, `fallback`, `conservative`); `livepaper` intentionally leaves both unset
- `--symbols-file` plus `--watch-symbols-file` remains the preferred lane watchlist path for the example services because an absent watchlist file would otherwise fail the daemon at start-up
- this cutover only moves the example paper service launch ownership into Rust; active production paper cutover is still a later step

### Inspect the Rust paper daemon service manifest

Use when: you want to verify how the current `AI_QUANT_*` service env would map
onto the Rust paper daemon before attempting any systemd cutover.

```bash
AI_QUANT_STRATEGY_YAML=./config/strategy_overrides.paper1.yaml \
AI_QUANT_DB_PATH=./trading_engine_v8_paper1.db \
AI_QUANT_CANDLES_DB_DIR=./candles_dbs \
AI_QUANT_SYMBOLS=BTC,ETH,SOL \
AI_QUANT_LOOKBACK_BARS=200 \
AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS=1773424200000 \
cargo run -p aiq-runtime -- \
  paper manifest \
  --watch-symbols-file \
  --json
```

Manifest expectations:

- `paper effective-config` is the narrower read-only control-plane surface for Python paper start-up and factory materialisation; `paper manifest` builds on top of that same resolver contract and adds daemon launch/resume inspection
- if `paper` start-up or factory materialisation drifts, run `aiq-runtime paper effective-config --json` first; set `AI_QUANT_RUNTIME_BIN` explicitly when the Python wrapper cannot discover the runtime binary
- `paper manifest` is read-only; it never writes the paper DB or starts follow-mode polling
- `paper manifest` is the preflight launch contract; `paper status` shows the current lane state, `paper service` previews the recommended supervision action, and `paper service apply` is the mutating surface that can enact that recommendation
- `--config`, `--db`, `--candles-db`, `--symbols`, `--symbols-file`, `--watch-symbols-file`, `--lookback-bars`, `--start-step-close-ts-ms`, `--lock-path`, and `--status-path` may override the corresponding env-derived values
- `AI_QUANT_PROMOTED_ROLE` is applied to the effective Rust paper config before the manifest resolves interval, pipeline profile, or daemon command
- `AI_QUANT_STRATEGY_MODE` remains the first strategy-mode selector; when it is unset, `AI_QUANT_STRATEGY_MODE_FILE` provides the same persisted fallback used by the Python paper service
- if `AI_QUANT_CANDLES_DB_PATH` is unset, the manifest derives the candle DB path from `AI_QUANT_CANDLES_DB_DIR` plus the resolved config interval
- if `AI_QUANT_INTERVAL` disagrees with the resolved config interval, the manifest returns a warning instead of silently changing the Rust runtime interval
- `base_config_path` is the operator-selected YAML, while `active_yaml_path` and `effective_yaml_path` are the materialised files Rust will actually use after promoted-config resolution and optional mode overlay
- Python paper start-up and factory deployment metadata must agree with the same resolver-derived `config_id`, interval, promoted-config path, and strategy-mode source for the target lane
- the resolved `status_path` is the daemon lifecycle JSON path for the current launch contract; when unset explicitly, it is derived from the resolved lock path
- the emitted `daemon_command` is the exact Rust paper daemon launch contract for the current env/CLI combination; it is intended for operator review, not as evidence of cutover
- `resume.launch_state` tells you whether the lane would fail closed (`blocked` / `bootstrap_required`), launch idle without symbols, start a fresh bootstrap, resume a due step, or simply stay caught up waiting for the next bar close
- `resume.last_applied_step_close_ts_ms`, `resume.next_due_step_close_ts_ms`, and `resume.latest_common_close_ts_ms` expose the restart/resume cursor when the paper DB and candle DB are both inspectable

### Inspect the Rust paper daemon service status

Use when: you want one read-only view that combines the current Rust launch
contract with the persisted daemon lifecycle JSON, so you can tell whether the
lane is running, stale, stopped, or needs a restart because the live daemon
contract drifted from the current config/env plan.

```bash
AI_QUANT_STRATEGY_YAML=./config/strategy_overrides.paper1.yaml \
AI_QUANT_DB_PATH=./trading_engine_v8_paper1.db \
AI_QUANT_CANDLES_DB_DIR=./candles_dbs \
AI_QUANT_SYMBOLS=BTC,ETH,SOL \
AI_QUANT_STATUS_STALE_AFTER_MS=30000 \
cargo run -p aiq-runtime -- \
  paper status \
  --json
```

Status expectations:

- `paper status` is read-only; it never starts the daemon or writes the paper DB
- `paper status` is the current lane-state view; pair it with `paper manifest` when you need the preflight contract or with `paper service` / `paper service apply` when deciding what to do next
- when no daemon lifecycle JSON exists yet, `service_state` falls back to the current launch contract (`blocked`, `bootstrap_required`, `bootstrap_ready`, `resume_ready`, or `caught_up_idle`)
- when a matching daemon status JSON exists and still reports `running=true`, `service_state` becomes `running`
- when the running status JSON is older than the configured staleness threshold, `service_state` becomes `status_stale`
- when the running daemon status reports `ok=false` or carries runtime errors, `service_state` becomes `restart_required` so later supervision can fail closed instead of silently monitoring an unhealthy lane
- when the running daemon status no longer matches the current launch contract (for example config fingerprint, profile, DB paths, BTC anchor, lookback, explicit symbols, the bootstrap step while the lane is still fresh, lock path, or symbols-file wiring drifted), `service_state` becomes `restart_required`
- when the persisted daemon lifecycle JSON reports `running=false`, `service_state` becomes `stopped`

### Inspect the Rust paper daemon service action

Use when: you want a read-only supervision answer for the current Rust lane,
without actually starting, stopping, or restarting the daemon.

```bash
AI_QUANT_STRATEGY_YAML=./config/strategy_overrides.paper1.yaml \
AI_QUANT_DB_PATH=./trading_engine_v8_paper1.db \
AI_QUANT_CANDLES_DB_DIR=./candles_dbs \
AI_QUANT_SYMBOLS=BTC,ETH,SOL \
AI_QUANT_STATUS_STALE_AFTER_MS=30000 \
cargo run -p aiq-runtime -- \
  paper service \
  --json
```

Service-action expectations:

- `paper service` is read-only; it never starts, stops, or restarts the daemon
- `paper service` is the recommendation preview; `paper service apply` is the separate mutating entrypoint that can enact that recommendation against the Rust paper daemon
- `desired_action = hold` means the current lane should stay unsupervised until the launch contract becomes valid
- `desired_action = start` means the current lane is launch-ready but no healthy Rust daemon is supervising it right now; this also covers watchlist-owned idle lanes that are allowed to start and wait for symbols later
- `desired_action = restart` means the lane is supervised by a stale or drifted daemon contract and later orchestration should recycle it
- `desired_action = monitor` means the current daemon status still matches the live launch contract and supervision can keep watching it
- `action_reason` gives the operator-facing explanation for that recommendation without requiring manual JSON diffing
- `daemon_command` is carried through as the exact launch command the later service surface would supervise

### Apply the Rust paper daemon service action

Use when: you want Rust to supervise the current paper lane directly without
claiming Python paper or systemd cutover yet.

```bash
AI_QUANT_STRATEGY_YAML=./config/strategy_overrides.paper1.yaml \
AI_QUANT_DB_PATH=./trading_engine_v8_paper1.db \
AI_QUANT_CANDLES_DB_DIR=./candles_dbs \
AI_QUANT_SYMBOLS=BTC,ETH,SOL \
AI_QUANT_STATUS_STALE_AFTER_MS=30000 \
cargo run -p aiq-runtime -- \
  paper service apply \
  --action auto \
  --json
```

Service-apply expectations:

- `paper service` remains read-only; only `paper service apply` mutates daemon lifecycle
- `--action auto` reuses the current read-only `paper service` recommendation
- `--action start` / `resume` launches the current `daemon_command` only when the lane is launch-ready and the lock is free
- `--action restart` recycles a stale, unhealthy, or drifted Rust daemon owner; when the lane lock is already free it collapses to `start`
- `--action stop` only signals a daemon when the current Rust status JSON and the active lock owner prove the same Rust pid owns the lane
- `hold` and healthy `monitor` lanes are no-op outcomes under `--action auto`
- the apply surface reuses the same `lock_path`, `status_path`, and launch contract from `paper manifest` / `paper status`, and it fails closed when status JSON is corrupt, daemon ownership cannot be proven, or the lane is not launch-ready

---

## 7. Factory Stage Gate (dry -> smoke -> real)

Use when: promoting configuration candidates from nightly factory output.

Use this section for all automated rollouts. Do not deploy to live without a successful gate sequence.

### Precondition

- Keep the strategy config under version control.
- Ensure `factory_cycle` can write selection/evidence files to `artifacts/`.
- Confirm the following keys are present in `reports/selection.json` after each stage:
  - `selection_stage`
  - `deploy_stage`
  - `promotion_stage`
- Confirm candidate evidence is attached in:
  - `candidate_configs` inside `run_metadata.json`
  - `items` inside `reports/report.json`
  - `evidence_bundle_paths` in `selection.json`
- Confirm replay equivalence proof exists for promotion candidates:
  - `selected.canonical_cpu_verified == true`
  - `selected.replay_equivalence_status == "pass"`
  - `selected.candidate_mode == true`
  - `selected.schema_version == 1`
  - `selected.replay_equivalence_report_path` file exists
  - `selected.replay_equivalence_count` is recorded

### Stage command

```bash
./scripts/run_factory_stage_gate.sh \
  --run-prefix v8_factory_gate \
  --dry-profile smoke \
  --smoke-profile smoke \
  --real-profile daily \
  --config config/strategy_overrides.yaml \
  --artifacts-dir artifacts
```

### Stage definitions

1. Dry stage (`dry`)
   - Runs `factory_cycle` with `--no-deploy`.
   - Expected:
     - `selection_stage == "selected"`
     - `deploy_stage == "no_deploy"` (or `"skipped"`)
     - `promotion_stage == "skipped"`
2. Smoke stage (`smoke`)
   - Runs `factory_cycle` with `--no-deploy`.
   - Expected:
     - `selection_stage == "selected"`
     - `deploy_stage == "no_deploy"` (or `"skipped"`)
     - `promotion_stage == "skipped"`
3. Real stage (`real`)
   - Runs `factory_cycle` without `--no-deploy` unless deployment is intentionally blocked.
   - If deployment is intentionally blocked, behaviour must match dry/smoke.
   - Expected:
     - `selection_stage == "selected"`
     - `deploy_stage != "pending"`
     - `promotion_stage != "pending"`

### Evidence bundle

`run_factory_stage_gate.sh` writes a consolidated manifest:

```text
artifacts/<run-prefix>_<timestamp>.evidence.json
```

Each stage entry includes:

- `run_id`
- `stage`
- `selection_stage`
- `deploy_stage`
- `promotion_stage`
- `evidence_bundle_paths`
- `selected.canonical_cpu_verified`
- `selected.replay_equivalence_status`

### Hard gate acceptance (recommended before promotion)

Use this checklist before moving from `smoke` to `real` and before approving any real deployment:

- `selection.json` must pass:

  ```bash
  python3 scripts/validate_factory_selection_gate.py \
    --selection-json artifacts/<run_id>/reports/selection.json \
    --stage smoke
  ```

- `selection.json` must contain:
  - `evidence_bundle_paths.run_metadata_json` (existing and readable)
  - `evidence_bundle_paths.report_json`
  - `evidence_bundle_paths.selection_json`
  - `evidence_bundle_paths.selection_md`
  - `selected.canonical_cpu_verified == true`
  - `selected.pipeline_stage`, `selected.sweep_stage`, `selected.replay_stage`, `selected.validation_gate`
  - `selected.candidate_mode == true`
  - `selected.schema_version == 1`
  - `selected.replay_equivalence_report_path` exists and path exists
  - `selected.replay_report_path` exists and path exists
  - `selected.replay_equivalence_count` is an integer >= 0

- `run_metadata.json` must contain the same `selected.config_id` row, and replay proof paths must match when present.

Retain this manifest with the promotion ticket.

---

## Quick Reference: Environment Variables (Risk)

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_QUANT_KILL_SWITCH` | (unset) | `close_only` or `halt_all` |
| `AI_QUANT_KILL_SWITCH_FILE` | (unset) | Path to kill-switch file |
| `AI_QUANT_RISK_MAX_DRAWDOWN_PCT` | 0 (disabled) | Equity drawdown % to trigger kill |
| `AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN` | 30 | Global entry rate limit |
| `AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN_PER_SYMBOL` | 6 | Per-symbol entry rate limit |
| `AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN` | 120 | Exit rate limit |
| `AI_QUANT_RISK_MAX_CANCELS_PER_MIN` | 120 | Cancel rate limit |
| `AI_QUANT_RISK_MAX_NOTIONAL_PER_WINDOW_USD` | 0 (disabled) | Max entry notional per window |
| `AI_QUANT_RISK_MIN_ORDER_GAP_MS` | 0 (disabled) | Minimum ms between any two orders |

## References

- [Success Metrics & Guardrails](success_metrics.md) — risk thresholds and kill-switch triggers
- [Strategy Lifecycle](strategy_lifecycle.md) — config state machine and rotation rules
- [Architecture](ARCHITECTURE.md) — system design, components, and data flow
- [Release Process](release_process.md) — version governance and tag-driven releases
