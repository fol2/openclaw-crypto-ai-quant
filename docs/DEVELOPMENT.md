# Development

## Prerequisites

- Rust toolchain (`rustup`, stable channel)
- `jq`
- `sqlite3`
- CUDA toolkit if you need GPU sweeps

## Build

```bash
cargo build --workspace
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli
cargo build --release --manifest-path backtester/Cargo.toml -p bt-cli --features gpu
```

## Test

```bash
cargo test --workspace
cargo test --manifest-path backtester/Cargo.toml
./scripts/ci_rust_runtime_foundation.sh
```

## Formatting And Linting

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

## Working Style

- keep production safety changes behind explicit contracts
- prefer JSON outputs for automation surfaces
- keep docs, comments, and user-facing copy in UK English
- avoid reintroducing alternate runtime-language ownership into the active tree
