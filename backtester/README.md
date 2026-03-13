# Mei Backtester

High-performance Rust replay and sweep tooling for the Mei strategy family.

## Build

```bash
# CPU
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli

# GPU
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli --features gpu
```

The release binary is `backtester/target/release/mei-backtester`.

## Commands

```bash
mei-backtester replay --candles-db candles_dbs/candles_1h.db
mei-backtester sweep --sweep-spec backtester/sweeps/smoke.yaml --output /tmp/sweep.jsonl
mei-backtester dump-indicators --symbol BTC --candles-db candles_dbs/candles_1h.db
```

## Features

- replay from blank state or runtime snapshot JSON
- CPU and CUDA GPU sweeps
- universe-history filtering
- parity-oriented sweep lanes
- shared config and indicator logic with the runtime stack

## Useful Scripts

- `backtester/sweeps/run_full_sweep.sh`
- `backtester/sweeps/run_17phase_144v.sh`
- `sweep_full/run_all.sh`
- `sweep_full/run_gpu.sh`
- `sweep_full/run_gpu_60k.sh`
- `sweep_full/run_aligned.sh`

The sweep helper scripts now stay shell-only. They keep JSONL as the canonical
output format; ad-hoc CSV conversion is intentionally left to local `jq`
pipelines when needed.

## Validation

```bash
cargo test --manifest-path backtester/Cargo.toml
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- replay --help
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- sweep --help
```
