# Hub Performance Optimisation

Date: 2026-06-12

## Problem

The hub returned intermittent HTTP 500s and very slow responses for live dashboard endpoints while only a small number of users were viewing the UI.

The live trading database was not the direct bottleneck. Direct SQLite reads from `trading_engine_v8_live.db` were fast and the live tables were modest in size. The main load came from the hub combining unbounded market-data WebSocket fanout with overlapping frontend polling and slow candle read paths.

## Findings

- `openclaw-ai-quant-hub.service` had reached very high CPU load before the fix. The stopped process recorded 2h 26m CPU time and 718 MB peak memory for the recent run.
- `/api/snapshot?mode=live` and `/api/tunnel?...mode=live` could time out at 20 to 25 seconds while the hub was saturated.
- The sidecar `wait_mids` endpoint can return a full symbol snapshot on frequent market ticks. The hub published every changed response without an application-level publish throttle.
- WebSocket clients could create duplicate per-topic forwarders and did not send unsubscribe messages when the last handler was removed.
- Dashboard, grid, and symbol detail requests could overlap when a previous request was still in flight.
- `candles_dbs/candles_1m.db` was large enough that some recent-symbol and latest-candle queries needed expression indexes for predictable read latency.

## Changes

- Added `AIQ_MONITOR_MIDS_PUBLISH_MIN_MS` with a 250 ms default publish floor.
- Added receiver-count checks so the hub skips JSON serialisation and publish work when no client is subscribed.
- Made WebSocket subscriptions one-forwarder-per-topic per socket and added unsubscribe handling.
- Added frontend single-flight guards for dashboard refreshes, grid snapshot/trend/candle/volume refreshes, and live/journey tunnel fetches.
- Added `scripts/optimise_candle_read_indexes.sh` for non-destructive candle DB read indexes and `ANALYZE`.
- Added a unit test for `BroadcastHub.receiver_count`.

## Validation

- `npm run build` in `hub/frontend`: passed.
- `cargo test -- --test-threads=1` in `hub`: passed, 195 tests.
- `cargo build --release` in `hub`: passed.
- `agent-browser open http://127.0.0.1:61010 && agent-browser snapshot -i`: dashboard rendered live rows.
- Local API probes after restart:
  - `/api/snapshot?mode=live`: HTTP 200, about 0.18 seconds.
  - `/api/trend-closes?interval=5m&limit=60`: HTTP 200, about 0.003 seconds.
  - `/api/volumes`: HTTP 200, about 0.001 seconds.
  - `/api/tunnel?symbol=ZEC&mode=live...`: HTTP 200, about 0.044 seconds.
  - `/api/tunnel?symbol=LIT&mode=live...`: HTTP 200, about 0.001 seconds on follow-up.
- `aiq-hub` CPU after several minutes: about 1.2 percent.
- `openclaw-ai-quant-live-v8.service` remained inactive throughout this work.

## Operational Notes

- The candle index script was run once on the local active candle DBs.
- Tailscale Serve remained configured as `100.89.104.117:61010 -> 127.0.0.1:61010`. A same-host request to the Tailscale IP timed out, so local service verification used `127.0.0.1:61010`; this may be a same-device hairpin limitation rather than a remote-client failure.
- `cargo clippy --all-targets -- -D warnings` is not currently a clean gate because the repository already has existing lint debt outside this change.
