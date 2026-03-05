# Rust Runtime Foundation

This directory hosts the first Rust-native runtime scaffolding that will
eventually replace the Python trading daemon.

Current contents:

- `aiq-runtime-core`: registry-driven stage graph resolution and runtime
  profile plumbing
- `aiq-runtime`: foundation CLI for config loading, pipeline inspection, and
  future daemon entrypoints

The foundation slice is intentionally narrow. It does not yet execute live or
paper trading, but it establishes the contracts that later slices will build on:

- backward-compatible YAML loading through Rust
- stable stage identifiers for parity debugging
- runtime profile resolution for `production`, `parity_baseline`, and
  `stage_debug`
- a single Rust-owned entry binary that future modes will converge on
