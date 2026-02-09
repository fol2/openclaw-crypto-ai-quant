# AI Quant Monitor (read-only)

Local dashboard for monitoring **Live** + **Paper** daemons:

- Reads SQLite DBs + daemon runtime logs (no secrets).
- Reads mid ticks from the local Rust **WS sidecar** (no direct Hyperliquid WS/REST calls).
- No order placement. No cancels. Read-only.

## Run (manual)

From `workspace/dev/ai_quant/`:

```bash
venv/bin/python3 -u monitor/server.py
```

Open:

```text
http://127.0.0.1:61010/
```

## Run (systemd user service)

Install:

```bash
cp systemd/openclaw-ai-quant-monitor.service.example ~/.config/systemd/user/openclaw-ai-quant-monitor.service
systemctl --user daemon-reload
systemctl --user enable --now openclaw-ai-quant-monitor.service
```

Logs:

```bash
journalctl --user -u openclaw-ai-quant-monitor.service -f
```

## Env vars

- `AIQ_MONITOR_BIND` (default `127.0.0.1`)
- `AIQ_MONITOR_PORT` (default `61010`)
- `AI_QUANT_WS_SIDECAR_SOCK` (optional; defaults to `%t/openclaw-ai-quant-ws.sock`)
- `AIQ_MONITOR_MIDS_POLL_MS` (default `1000`)
- `AIQ_MONITOR_MIDS_MAX_AGE_S` (default `60`)
- `AIQ_MONITOR_INTERVAL` (default `AI_QUANT_INTERVAL` else `1m`)

Optional overrides:

- `AIQ_MONITOR_LIVE_DB` (default `trading_engine_live.db`)
- `AIQ_MONITOR_PAPER_DB` (default `trading_engine.db`)

Notes:

- Heartbeats are read from SQLite `runtime_logs` (preferred). If unavailable, use the systemd journal (`journalctl --user -u openclaw-ai-quant-*.service`).
