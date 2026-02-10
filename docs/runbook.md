# Operations Runbook

Procedures for common operational scenarios. When the bot misbehaves, follow the relevant section below.

## Services Reference

| Service | Unit name | Purpose |
|---------|-----------|---------|
| Paper trader | `openclaw-ai-quant-trader` | Paper trading daemon |
| Live trader | `openclaw-ai-quant-live` | Live trading daemon |
| WS sidecar | `openclaw-ai-quant-ws-sidecar` | Market data WebSocket |
| Monitor | `openclaw-ai-quant-monitor` | Real-time dashboard |

All services are systemd user units. Manage with `systemctl --user <action> <unit>`.

---

## 1. Pause Trading (Emergency Stop)

Use when: unexpected losses, erratic behaviour, exchange issues, or any situation requiring immediate halt.

### Option A: Kill-switch via environment (close-only)

Blocks new entries but allows exits to close positions.

```bash
# Close-only mode (recommended default)
export AI_QUANT_KILL_SWITCH=close_only
systemctl --user restart openclaw-ai-quant-live

# Full halt — blocks ALL orders including exits
export AI_QUANT_KILL_SWITCH=halt_all
systemctl --user restart openclaw-ai-quant-live
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
systemctl --user stop openclaw-ai-quant-live

# Stop paper trading
systemctl --user stop openclaw-ai-quant-trader
```

### Clearing the kill-switch

```bash
# Remove file-based kill
rm /tmp/ai-quant-kill

# Clear env-based kill — unset and restart
unset AI_QUANT_KILL_SWITCH
systemctl --user restart openclaw-ai-quant-live
```

### Verification

```bash
# Check service status
systemctl --user status openclaw-ai-quant-live

# Check logs for kill-switch activation
journalctl --user -u openclaw-ai-quant-live --since "10 min ago" | grep -i kill
```

---

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
journalctl --user -u openclaw-ai-quant-live --since "2 min ago" | grep -i reload
```

If `engine.interval` changed, a restart is required:

```bash
systemctl --user restart openclaw-ai-quant-live
```

---

## 3. Verify Database Integrity

Use when: suspecting data corruption, missing candles, or stale market data.

### Candle database freshness

```bash
# Check all candle DBs — look for gaps or stale data (basic)
for db in candles_dbs/candles_*.db; do
    echo "=== $db ==="
    sqlite3 "$db" "SELECT COUNT(*) AS rows, datetime(MIN(t)/1000, 'unixepoch') AS earliest, datetime(MAX(t)/1000, 'unixepoch') AS latest, datetime(MAX(t_close)/1000, 'unixepoch') AS latest_close FROM candles;"
done

# Automated freshness + gap checks (JSON on stdout; summary on stderr)
uv run python tools/check_candle_dbs.py --lookback-hours 24
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

### BBO snapshot database (AQC-206)

Optional, sampled best-bid/best-ask (BBO) snapshots are stored in SQLite for slippage modelling and post-trade analysis.

#### Enablement

- Enable BBO subscriptions: `AI_QUANT_WS_ENABLE_BBO=1`
- Enable snapshots: `AI_QUANT_BBO_SNAPSHOTS_ENABLE=1`

Tuning knobs:
- `AI_QUANT_BBO_SNAPSHOTS_DB_PATH` (default: `$AI_QUANT_CANDLES_DB_DIR/bbo_snapshots.db`)
- `AI_QUANT_BBO_SNAPSHOTS_SAMPLE_MS` (per-symbol insert throttle; default: 1000)
- `AI_QUANT_BBO_SNAPSHOTS_MAX_QUEUE` (bounded in-memory queue; default: 20000; drops when full)
- `AI_QUANT_BBO_SNAPSHOTS_RETENTION_HOURS` (time-based retention; default: 24)
- `AI_QUANT_BBO_SNAPSHOTS_RETENTION_SWEEP_SECS` (sweep interval; default: 600; min: 30)

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
systemctl --user restart openclaw-ai-quant-live    # or openclaw-ai-quant-trader for paper

# 4. Monitor (optional, non-critical)
systemctl --user restart openclaw-ai-quant-monitor
```

### Full stop (all services)

```bash
systemctl --user stop openclaw-ai-quant-live
systemctl --user stop openclaw-ai-quant-trader
systemctl --user stop openclaw-ai-quant-monitor
systemctl --user stop openclaw-ai-quant-ws-sidecar
```

### Full start (all services)

```bash
systemctl --user start openclaw-ai-quant-ws-sidecar
sleep 5
systemctl --user start openclaw-ai-quant-trader     # paper
# systemctl --user start openclaw-ai-quant-live     # live (uncomment when ready)
systemctl --user start openclaw-ai-quant-monitor
```

### Post-restart health check

```bash
# Verify all services are active
systemctl --user status openclaw-ai-quant-ws-sidecar openclaw-ai-quant-live openclaw-ai-quant-monitor

# Check for errors in the last minute
journalctl --user -u openclaw-ai-quant-live --since "1 min ago" | grep -iE "error|exception|traceback"

# Confirm config was loaded
journalctl --user -u openclaw-ai-quant-live --since "1 min ago" | grep -i "config"
```

---

## 6. Export State for Debugging

Use when: you need a snapshot of the current trader state for analysis or to seed a backtest.

```bash
# Export paper state
uv run python tools/export_state.py --source paper --output /tmp/paper_state.json

# Export live state
uv run python tools/export_state.py --source live --output /tmp/live_state.json

# Export trade history to CSV
uv run python tools/export_csv.py
```

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
- [AGENTS.md](AGENTS.md) — full architecture and configuration reference
