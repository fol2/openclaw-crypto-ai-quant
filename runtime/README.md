# Rust Runtime Foundation

This directory hosts the first Rust-native runtime scaffolding that will
eventually replace the Python trading daemon.

Current contents:

- `aiq-runtime-core`: registry-driven stage graph resolution and runtime
  profile plumbing
- `aiq-runtime`: foundation CLI for config loading, pipeline inspection, and
  runtime migration entrypoints

Current runtime-owned paper surfaces and the paired opt-in wrapper:

| Command | Purpose | Notes |
|---|---|---|
| `snapshot export-paper` | Export a v2 Rust paper continuation snapshot | Bootstrap/export only |
| `snapshot seed-paper` | Seed a paper DB from a v2 snapshot | Deterministic bootstrap path |
| `paper doctor` | Restore Rust-owned paper state and inspect bootstrap markers | Non-mutating |
| `paper run-once` | Execute one single-symbol Rust paper step | Single-shot shell |
| `paper cycle` | Execute one repeatable multi-symbol Rust paper cycle | Explicit `--step-close-ts-ms`, not a daemon |
| `paper loop` | Execute a bounded Rust paper catch-up loop | Resumes from `runtime_cycle_steps`, optional follow polling, and can re-read `--symbols-file` between inspections |
| `paper daemon` | Execute an opt-in long-running Rust paper orchestration wrapper | Wraps `paper loop --follow`, can wait on an empty watchlist file, and keeps the same `paper cycle` write contract; not cutover |

The runtime slice is still intentionally narrow. It does not yet own any live
execution path, and Python paper execution is still the active production
daemon. The paired opt-in `paper daemon` surface is only a long-running wrapper
around the existing follow-mode loop / cycle contracts, but it already
establishes the contracts that later slices will build on:

- backward-compatible YAML loading through Rust
- stable stage identifiers for parity debugging
- runtime profile resolution for `production`, `parity_baseline`, and
  `stage_debug`
- a single Rust-owned entry binary that future modes will converge on
- repeatable paper step / cycle / loop contracts with explicit step identity
- optional follow polling so Rust can stay alive after catch-up and wait for the next due step
- refreshable `--symbols-file` discovery between follow-mode scheduling inspections
- an opt-in daemon wrapper that keeps the same follow-mode loop / cycle contracts alive without claiming service cutover
