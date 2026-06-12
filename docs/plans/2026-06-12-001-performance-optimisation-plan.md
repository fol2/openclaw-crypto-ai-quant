---
title: Hub Performance Optimisation Plan
created: 2026-06-12
type: performance-optimisation
origin: user request to fully optimise hub load for a small number of viewers
---

# Hub Performance Optimisation Plan

## Problem Frame

The hub is consuming multiple CPU cores and returning slow or failed responses for ordinary dashboard use. The target operating model is one to three viewers using the dashboard without creating sustained high CPU load, API timeouts, or `500` responses.

The investigation shows the live trading database is not the primary source of load. The performance failure is caused by a combination of high-frequency market-data fanout, overlapping frontend polling, repeated candle database reads, and route handlers that run blocking database work on the async runtime.

## Requirements

- Keep live trading behaviour unchanged unless explicitly requested separately.
- Restore responsive dashboard operation for one to three viewers.
- Bound market-data broadcast work so `mids` and BBO updates cannot saturate the hub.
- Prevent the frontend from starting duplicate heavy REST requests while a previous request is still in flight.
- Reduce candle query cost for snapshot, trend, and candle views.
- Add or adjust tests around the changed behaviour.
- Provide before and after operational checks using local hub endpoints.

## Scope

In scope:
- Hub REST and WebSocket performance fixes.
- Frontend request backpressure for dashboard and grid views.
- SQLite read-path indexes for candle queries.
- Operational verification against local WSL services.

Out of scope:
- Starting or enabling `openclaw-ai-quant-live-v8.service`.
- Changing trading strategy, order execution, or risk logic.
- Destructive database pruning or archival.
- Large data migration of historical BBO snapshots.

## Existing Evidence

- `trading_engine_v8_live.db` is about 185 MB and key live tables are small.
- `exit_tunnel` direct SQLite queries complete effectively instantly.
- `candles_1m.db` is about 1.34 GB and recent-symbol queries scan the candle index and sort through a temporary B-tree.
- `aiq-hub` has sustained multi-core CPU usage, many Tokio worker threads, and slow heavy endpoints.
- Sidecar `wait_mids` can produce frequent changed updates; hub publishes each update without an application-level throttle.
- WebSocket subscription handling creates a duplicate receiver for each subscribed topic.
- Dashboard and grid polling can overlap when a previous heavy request has not finished.

## Key Decisions

Use market-data gateway patterns: coalesce and throttle fanout, isolate blocking reads from the async runtime, and make dashboard polling single-flight. The hub should not serialise and broadcast a full market-data payload for every sidecar tick.

Prefer narrow SQLite index additions over pruning. The current failure is not caused by total database size alone; it is caused by specific query shapes and runtime pressure.

Keep read optimisations reversible. Database index additions are non-destructive and can be inspected with `EXPLAIN QUERY PLAN`.

## Implementation Units

### Unit 1: Bound WebSocket Market-Data Fanout

Files:
- `hub/src/main.rs`
- `hub/src/config.rs`
- `hub/src/ws/mod.rs`
- `hub/src/ws/broadcast.rs`

Work:
- Add a configurable minimum publish interval for mids and BBO broadcasts.
- Coalesce sidecar updates within the interval and publish the most recent snapshot.
- Skip JSON serialisation when there are no receivers for a topic.
- Fix duplicate receiver creation in the WebSocket subscribe path.
- Handle unsubscribe messages so clients can release topics.

Tests:
- Add or update hub unit tests for broadcast receiver counting and unsubscribe handling where the local structure supports it.
- Run targeted Rust tests for hub modules touched by this unit.

### Unit 2: Move Heavy Read Work off Async Runtime

Files:
- `hub/src/routes/monitor.rs`
- `hub/src/db/pool.rs`
- `hub/src/state.rs`

Work:
- Wrap blocking snapshot, trend, volume, candle, and tunnel database reads in `spawn_blocking` or an equivalent local pattern.
- Avoid creating a new candle read pool on every request where a cached pool can be held in application state.
- Keep sidecar RPC calls async and separate from blocking SQLite work.

Tests:
- Add targeted tests where route helper extraction makes this practical.
- Run hub tests after the route refactor.

### Unit 3: Optimise Candle Database Reads

Files:
- `hub/src/db/candles.rs`
- `hub/src/db/trading.rs`
- `scripts` or existing maintenance/migration location if present

Work:
- Add read-path indexes for candle queries that filter or order by effective close time.
- Prefer query shapes that use indexed columns instead of `COALESCE(t_close, t)` in hot filters when possible.
- Keep index creation as an explicit maintenance step, not an automatic destructive migration.

Tests:
- Use `sqlite3` with `EXPLAIN QUERY PLAN` on the active candle databases before and after index creation.
- Compare latency for recent-symbol, trend-closes, trend-candles, and 1m candle batch reads.

### Unit 4: Frontend Request Backpressure

Files:
- `hub/frontend/src/pages/Dashboard.svelte`
- `hub/frontend/src/pages/GridView.svelte`
- `hub/frontend/src/components/SymbolDetailPanel.svelte`
- `hub/frontend/src/lib/api.ts`
- `hub/frontend/src/lib/ws.ts`

Work:
- Add in-flight guards for snapshot, trend, candle, volume, and tunnel refresh paths.
- Replace fixed interval overlap with single-flight scheduling.
- Send unsubscribe messages when topic handlers are removed.
- Keep visible behaviour the same except for improved responsiveness under load.

Tests:
- Run frontend build checks.
- Exercise dashboard and grid manually through local hub after restart.

### Unit 5: Operational Validation

Files:
- No production code expected unless validation exposes a missed issue.

Work:
- Restart only `openclaw-ai-quant-hub.service` after code deployment.
- Do not start `openclaw-ai-quant-live-v8.service`.
- Capture endpoint timing for `/api/health`, `/api/snapshot?mode=live`, `/api/trend-closes`, `/api/volumes`, and `/api/tunnel`.
- Capture process CPU and open connection counts before and after.

Tests:
- `systemctl --user status openclaw-ai-quant-hub.service`
- Local `curl` timing probes against `127.0.0.1:61010`
- `top` or `ps` CPU sampling for `aiq-hub`

## Risks

- Throttling market data too aggressively could make prices look less live. Mitigation: start with a small bounded interval suitable for a dashboard, not a trading execution path.
- Route refactoring could change response structure. Mitigation: keep JSON contracts intact and focus on scheduling and query execution.
- Adding indexes to large candle databases may take time and disk space. Mitigation: apply indexes explicitly, measure, and avoid touching live order tables.
- Existing browser tabs may keep stale WebSocket sessions. Mitigation: restart hub and refresh clients after deployment.

## Verification Plan

Before changes:
- Record current CPU for `aiq-hub`.
- Record timings for heavy endpoints.
- Record connection count on port `61010`.

After changes:
- Confirm `aiq-hub` remains active.
- Confirm `openclaw-ai-quant-live-v8.service` remains stopped unless the user explicitly changes that.
- Confirm endpoint timings are materially lower and no longer time out under one to three viewers.
- Confirm dashboard renders without repeated `500` errors.

## Sequencing

1. Fix WebSocket fanout and frontend overlapping polling first to stop the feedback loop.
2. Move blocking database reads off the async runtime.
3. Add candle query indexes and re-check query plans.
4. Build and run targeted checks.
5. Restart hub and capture operational evidence.
