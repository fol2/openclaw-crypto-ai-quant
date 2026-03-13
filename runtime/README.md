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

cargo run -p aiq-runtime -- live effective-config --json
cargo run -p aiq-runtime -- live manifest --json
cargo run -p aiq-runtime -- live daemon --project-dir "$PWD"

cargo run -p aiq-runtime -- snapshot export-paper --db trading_engine.db --output /tmp/paper.json
cargo run -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/paper.json --target-db trading_engine.db --strict-replace --json
```

## Ownership

The runtime now owns:

- paper lane effective-config resolution
- paper/live service manifests and status inspection
- paper/live daemon processes
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

Paper and live reports also surface behaviour traces so parity lanes can verify
which exit behaviour actually fired on a bar.

## Wrappers

Use the shell wrappers when you want stable service-style entrypoints:

- `scripts/run_paper.sh`
- `scripts/run_paper_lane.sh`
- `scripts/run_live.sh`

## Validation

```bash
cargo test -p aiq-runtime-core
cargo test -p aiq-runtime
./scripts/ci_rust_runtime_foundation.sh
```
