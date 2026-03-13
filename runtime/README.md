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
