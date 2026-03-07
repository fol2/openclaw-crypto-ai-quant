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
| `paper effective-config` | Resolve the shared paper control-plane config contract | Read-only config surface for Python paper start-up and factory materialisation; emits `active_yaml_path`, `effective_yaml_path`, interval, `strategy_overrides_sha1`, and `config_id` |
| `paper doctor` | Restore Rust-owned paper state and inspect bootstrap markers | Non-mutating |
| `paper run-once` | Execute one single-symbol Rust paper step | Single-shot shell |
| `paper cycle` | Execute one repeatable multi-symbol Rust paper cycle | Explicit `--step-close-ts-ms`, not a daemon |
| `paper loop` | Execute a bounded Rust paper catch-up loop | Resumes from `runtime_cycle_steps`, optional follow polling, and only loads `--symbols-file` once at start-up |
| `paper manifest` | Resolve the current Rust paper daemon service contract | Read-only env/CLI manifest, including launch readiness, restart/resume state, resolved lane `status_path`, and effective config selection from promoted-role plus strategy-mode inputs |
| `paper status` | Resolve the current Rust paper daemon service state | Read-only manifest + status-file view, including restart-required / stale detection plus health / launch-identity drift detection for the current lane |
| `paper service` | Resolve the current Rust paper daemon supervisor action | Read-only status + launch-contract view that tells supervision whether to hold, start, restart, or monitor the lane while failing closed on unhealthy or drifted daemon status |
| `paper service apply` | Apply the current Rust paper daemon supervisor action | Opt-in side-effecting supervisor for Rust paper daemon start/restart/resume/stop only; reuses the same manifest/status contract and fails closed on unhealthy, drifted, or unproven lane ownership |
| `paper daemon` | Execute an opt-in long-running Rust paper orchestration wrapper | Owns the outer scheduler, can optionally watch `--symbols-file` for reloads, writes a lane status JSON, and keeps the same `paper cycle` write contract while using the same effective config contract as `paper manifest`; not cutover |
| `paper lane <manifest|status|service|apply|daemon>` | Resolve or launch one of the conventional paper lanes | Maps `paper1` / `paper2` / `paper3` / `livepaper` onto the conventional YAML / DB / lock / status contracts for a worktree, so example systemd units can launch the Rust paper lane directly without retyping the service contract |

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
- a read-only service manifest so operators can inspect the resolved Rust daemon launch contract, bootstrap requirement, restart/resume state, lifecycle `status_path`, promoted config selection, and strategy-mode file fallback before cutover
- a read-only effective-config surface that Python paper start-up and factory materialisation now share with the Rust paper shells
- a read-only service status surface so operators can compare the current launch contract against the persisted daemon lifecycle JSON, including daemon health and launch-identity drift, without parsing files by hand
- a read-only supervisor action surface so operators can inspect whether the lane should be held, started, restarted, or simply monitored
- an opt-in side-effecting `paper service apply` surface that can enact that recommendation for the Rust paper daemon without claiming service cutover
- an opt-in daemon wrapper that owns scheduler/watchlist reload orchestration without claiming service cutover
- a conventional `paper lane` control-plane wrapper that binds the example `paper1` / `paper2` / `paper3` / `livepaper` services to Rust-owned launch contracts from a project root
- a durable daemon `status_path` contract that later service supervision can build on
