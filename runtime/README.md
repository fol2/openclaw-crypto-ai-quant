# Rust Runtime

`runtime/` contains the Rust-owned trading runtime.

## Crates

- `aiq-runtime-core`: shared pipeline, launch-contract, and bootstrap logic
- `aiq-runtime`: CLI and daemon entrypoints for paper/live ownership

## Main Commands

```bash
cargo run -p aiq-runtime -- pipeline --json
cargo run -p aiq-runtime -- doctor --json

cargo run -p aiq-runtime -- paper effective-config --json
cargo run -p aiq-runtime -- paper manifest --json
cargo run -p aiq-runtime -- paper daemon --lane paper1 --project-dir "$PWD"

cargo run -p aiq-runtime -- live effective-config --project-dir "$PWD" --json
cargo run -p aiq-runtime -- live manifest --json
cargo run -p aiq-runtime -- live daemon --project-dir "$PWD"
cargo run -p aiq-runtime -- live sync-fills --project-dir "$PWD" --json

cargo run -p aiq-runtime -- snapshot export-paper --db trading_engine.db --output /tmp/paper.json
cargo run -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/paper.json --target-db trading_engine.db --strict-replace --json

cargo run -p aiq-runtime --bin aiq-maintenance -- fetch-funding-rates --days 30 --db candles_dbs/funding_rates.db
cargo run -p aiq-runtime --bin aiq-maintenance -- prune-factory-artifacts --project-dir "$PWD" --settings config/factory_defaults.yaml --profile nightly
cargo run -p aiq-runtime --bin aiq-maintenance -- prune-runtime-logs --db trading_engine_v8_live.db --db trading_engine_v8_paper1.db --db trading_engine_v8_paper2.db --db trading_engine_v8_paper3.db
```

Paper supervision now treats `decision_events` as a canonical compatibility
contract. Direct `paper daemon` launches reconcile legacy/full paper DBs
immediately before the first writable cycle, so idle watchlist/bootstrap waits
stay read-only. `paper service apply` still preflights that contract before
supervised start/restart and fails closed on unreconcilable schema drift
instead of letting a daemon enter a restart loop after start-up.

`live sync-fills` fails closed when it encounters unsupported fill shapes, so
hourly automation should treat a failed run as an operator review signal rather
than a silent partial success.

It also refreshes DB-backed exchange account and position snapshots so the Hub
can fall back to current live balance and holdings when an in-memory
Hyperliquid cache is unavailable or stale.

## Ownership

The runtime now owns:

- paper lane effective-config resolution
- paper/live service manifests and status inspection
- paper/live daemon processes
- live fill reconciliation from Hyperliquid back into the local live DB
- snapshot export and seeding contracts
- launch/supervision wrappers used by the remaining systemd examples

The repository no longer ships any alternate runtime language path.

## Behaviour Profiles

The runtime pipeline is now two-layered:

- stage-level control: `ranker`, `stage_order`, `enabled_stages`, `disabled_stages`
- behaviour-level control: `pipeline.profiles.<name>.behaviours.*`

Behaviour profiles allow operator/debug lanes to disable or reorder concrete
decision behaviours beneath a stage, including:

- `gates`
- `signal_modes`
- `signal_confidence`
- `exits`
- `engine`
- `entry_sizing`
- `entry_progression`
- `risk`

The `exits` group is now a real execution contract, not a documentation-only
overlay: operator lanes can disable or re-sequence stop-loss modifiers, trailing
logic, take-profit ordering, and smart exits while keeping the same Rust binary.

Use `cargo run -p aiq-runtime -- pipeline --json` to inspect the resolved stage
and behaviour plan for the active profile.

The shipped example config documents two parity-focused profiles:

- `parity_baseline`: explicit production-like behaviour ordering with broker/fill stages disabled
- `parity_exit_isolation`: parity baseline plus disabled exit modifiers to isolate base stop-loss, trailing, and full take-profit logic

Typical parity inspection commands:

```bash
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_baseline --json
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_exit_isolation --json
```

Paper and live reports also surface behaviour traces so parity lanes can verify
which exit behaviour actually fired on a bar.

The runtime-owned exit tunnel contract now persists `has_upper_full` and
`has_lower_full` flags alongside the numeric bounds, plus `open_time_ms` as the
stable identity of the position instance that produced the row. Runtime emits
those rows from the post-transition position state, with
`ts_ms = step_close_ts_ms + 1`, so a tunnel point becomes effective from the
next bar instead of reusing the pre-close state on the current bar.

Hub chart consumers should use the presence flags to suppress
behaviour-disabled full TP/SL overlays instead of treating missing bounds as
`0.0`. They should also filter by `open_time_ms` whenever the selected journey
or position exposes that opening, so multiple same-symbol reopenings do not
bleed into one tunnel path. Live position fallbacks now preserve opening
timestamps when the Hub can recover them from DB-backed position state, which
lets live tunnel requests stay scoped to the current opening when that
metadata exists.

## Wrappers

Use the shell wrappers when you want stable service-style entrypoints:

- `scripts/run_paper.sh`
- `scripts/run_paper_lane.sh`
- `scripts/run_live.sh`

Use the maintenance binary when you need the kept Rust-owned one-shot jobs:

- `aiq-maintenance fetch-funding-rates`
- `aiq-maintenance prune-factory-artifacts`
- `aiq-maintenance prune-runtime-logs`

## Validation

```bash
cargo test -p aiq-runtime-core
cargo test -p aiq-runtime
./scripts/ci_rust_runtime_foundation.sh
```
