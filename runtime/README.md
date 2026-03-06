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
| `paper loop` | Execute a bounded Rust paper catch-up loop | Resumes from `runtime_cycle_steps`, optional follow polling, and only loads `--symbols-file` once at start-up |
| `paper manifest` | Resolve the current Rust paper daemon service contract | Read-only env/CLI manifest, including launch readiness, restart/resume state, and the resolved lane `status_path` |
| `paper status` | Resolve the current Rust paper daemon service state | Read-only manifest + status-file view, including restart-required / stale detection plus health / launch-identity drift detection for the current lane |
| `paper service` | Resolve the current Rust paper daemon supervisor action | Read-only status + launch-contract view that tells supervision whether to hold, start, restart, or monitor the lane while failing closed on unhealthy or drifted daemon status |
| `paper daemon` | Execute an opt-in long-running Rust paper orchestration wrapper | Owns the outer scheduler, can optionally watch `--symbols-file` for reloads, writes a lane status JSON, and keeps the same `paper cycle` write contract; not cutover |

The runtime slice is still intentionally narrow. It does not yet own any live
execution path, and Python paper execution is still the active production
daemon. The paired opt-in `paper daemon` surface now owns the Rust-side outer
scheduler for the existing paper cycle contract, but it still does not claim
paper daemon cutover. Even so, it already establishes the contracts that later
slices will build on:

- backward-compatible YAML loading through Rust
- stable stage identifiers for parity debugging
- runtime profile resolution for `production`, `parity_baseline`, and
  `stage_debug`
- a single Rust-owned entry binary that future modes will converge on
- repeatable paper step / cycle / loop contracts with explicit step identity
- optional follow polling so Rust can stay alive after catch-up and wait for the next due step
- one-shot `--symbols-file` loading for bounded loop shells
- a read-only service manifest so operators can inspect the resolved Rust daemon launch contract, bootstrap requirement, restart/resume state, and lifecycle `status_path` before cutover
- a read-only service status surface so operators can compare the current launch contract against the persisted daemon lifecycle JSON, including daemon health and launch-identity drift, without parsing files by hand
- a read-only supervisor action surface so later service orchestration can decide whether to hold, start, restart, or simply monitor the lane, while still failing closed on unhealthy daemon status
- an opt-in daemon wrapper that owns scheduler/watchlist reload orchestration without claiming service cutover
- a durable daemon `status_path` contract that later service supervision can build on
