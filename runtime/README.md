# Rust Runtime Foundation

This directory hosts the Rust-native runtime that now owns the production paper
lane and is replacing the remaining Python runtime surfaces step by step.

Current contents:

- `aiq-runtime-core`: registry-driven stage graph resolution and runtime
  profile plumbing
- `aiq-runtime`: foundation CLI for config loading, pipeline inspection, and
  runtime migration entrypoints

Current runtime-owned paper surfaces:

| Command | Purpose | Notes |
|---|---|---|
| `snapshot export-paper` | Export a v2 Rust paper continuation snapshot | Bootstrap/export only |
| `snapshot seed-paper` | Seed a paper DB from a v2 snapshot | Deterministic bootstrap path |
| `paper effective-config` | Resolve the shared paper control-plane config contract | Read-only config surface for Python paper start-up and factory materialisation; `config_path` now points at a Rust-owned runtime-facing materialised YAML with defaults, promoted overlays, and strategy-mode overlays already expanded, while `active_yaml_path` / `effective_yaml_path` remain audit artefacts; also accepts `--lane paper1|paper2|paper3|livepaper` plus optional `--project-dir` so Rust can own the conventional lane defaults |
| `paper doctor` | Restore Rust-owned paper state and inspect bootstrap markers | Non-mutating |
| `paper run-once` | Execute one single-symbol Rust paper step | Single-shot shell |
| `paper cycle` | Execute one repeatable multi-symbol Rust paper cycle | Explicit `--step-close-ts-ms`, not a daemon; now honours runtime pipeline profile stage toggles for ranking, execution preview, and persistence while emitting a per-stage trace |
| `paper loop` | Execute a bounded Rust paper catch-up loop | Resumes from `runtime_cycle_steps`, optional follow polling, and only loads `--symbols-file` once at start-up |
| `paper manifest` | Resolve the current Rust paper daemon service contract | Read-only env/CLI manifest, including launch readiness, restart/resume state, resolved lane `status_path`, lane preset metadata, effective config selection from promoted-role plus strategy-mode inputs, and one-shot bootstrap support through `AI_QUANT_PAPER_BOOTSTRAP_FROM_LATEST_COMMON_CLOSE=1` |
| `paper status` | Resolve the current Rust paper daemon service state | Read-only manifest + status-file view, including restart-required / stale detection plus health / launch-identity drift detection for the current lane |
| `paper service` | Resolve the current Rust paper daemon supervisor action | Read-only status + launch-contract view that tells supervision whether to hold, start, restart, or monitor the lane while failing closed on unhealthy or drifted daemon status |
| `paper service apply` | Apply the current Rust paper daemon supervisor action | Side-effecting supervisor for the active Rust paper daemon start/restart/resume/stop path; reuses the same manifest/status contract and fails closed on unhealthy, drifted, or unproven lane ownership |
| `paper daemon` | Execute the long-running Rust paper orchestration wrapper | Owns the active production paper scheduler, accepts `--lane` for `paper1/paper2/paper3/livepaper` plus optional `--project-dir`, can watch a symbols file when one is configured, writes a lane status JSON, and keeps the same `paper cycle` write contract while using the same effective config contract as `paper manifest` |

Current live-facing Rust control-plane surfaces:

| Command | Purpose | Notes |
|---|---|---|
| `live effective-config` | Resolve the shared Rust effective-config contract for live / dry-live start-up | Dedicated live-facing CLI surface; no longer relies on the paper-named compatibility path |
| `live manifest` | Resolve the current live launch contract, config identity, and safety-gate state | Read-only contract surface for the Rust live daemon, including the resolved candle DB, symbols, secrets path, and daemon command |
| `live status` | Resolve the current Rust live daemon service state | Read-only manifest + status-file view, including restart-required / stale detection plus launch-identity drift detection for the live lane |
| `live service` | Resolve the current Rust live daemon supervisor action | Read-only status + launch-contract view that tells supervision whether to hold, start, restart, or monitor the live lane while failing closed on unhealthy or drifted daemon status |
| `live service apply` | Apply the current Rust live daemon supervisor action | Side-effecting supervisor for the Rust live daemon start/restart/resume/stop path; reuses the same manifest/status contract and fails closed on unhealthy, drifted, or unproven lane ownership |
| `live daemon` | Execute the long-running Rust live orchestration wrapper | Owns live state restore, strategy-state build, pipeline execution, OMS intent/order/fill transitions, risk enforcement, broker submission, fill backfill, and status-file publication |

The Rust runtime now owns both the active paper lanes and the authoritative
live service path. Python paper execution remains available only as an explicit
recovery/debug fallback, while Python live / dry-live entrypoints are now
archival recovery surfaces gated behind an explicit opt-in.

The Rust runtime surfaces now establish one shared contract across paper and
live:

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
- a Rust-owned runtime materialisation step so Python paper consumers stop reapplying Python defaults when the resolver already owns the active config identity
- a read-only service status surface so operators can compare the current launch contract against the persisted daemon lifecycle JSON, including daemon health and launch-identity drift, without parsing files by hand
- a read-only supervisor action surface so operators can inspect whether the lane should be held, started, restarted, or simply monitored
- a side-effecting `paper service apply` surface that can enact that recommendation for the active Rust paper daemon
- the active daemon wrapper that owns scheduler/watchlist reload orchestration for the production paper lanes
- a matching `live manifest` / `live status` / `live service` / `live service apply` contract for the production live lane
- a Rust live daemon that owns the exchange-facing runtime loop, OMS/risk transitions, and broker submission contract
- Rust-owned conventional lane presets (`paper1`, `paper2`, `paper3`, `livepaper`) that bundle the default config path, promoted-role / strategy-mode selection, watchlist file path, candle DB directory, DB path, and lock/status paths for paper orchestration
- a durable daemon `status_path` contract that later service supervision can build on
- a live execution slice where the Rust daemon reads the resolved stage graph and configured ranker instead of relying on the retired Python live loop
- a one-shot bootstrap contract where `AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS` or `AI_QUANT_PAPER_BOOTSTRAP_FROM_LATEST_COMMON_CLOSE=1` can unlock the very first Rust launch, and later restarts ignore that bootstrap value once `runtime_cycle_steps` exist
