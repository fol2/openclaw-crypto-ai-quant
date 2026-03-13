# Rust Full-Runtime Cutover Programme

**Date:** 2026-03-07  
**Status:** Delivered programme checkpoint after the Rust paper and live runtime cutovers on `master`; remaining work is compatibility cleanup and deletion follow-through
**Primary goal:** Retire Python as a production runtime language and converge on one Rust-owned end-to-end engine for live, paper, backtest, CPU sweep, and the future GPU parity path.

**Current ownership index:** `docs/current_authoritative_paths.md`
**Final cleanup plan:** `docs/programmes/rust_runtime_final_cleanup_plan.md`

This document now serves two purposes:

1. the delivered record of the cutover programme
2. the remaining cleanup backlog for archival Python compatibility surfaces

Unless a section explicitly says otherwise, future-tense delivery language below
should be read as the historical plan that led to the current Rust-owned
runtime state.

Short operational companion:
`docs/rust_full_runtime_cutover_exec_summary.md`

## 1. North Star

We are no longer planning around a permanent split where Rust owns only the decision kernel and Python remains the control plane.

The target architecture is:

1. one Rust-owned runtime engine
2. one backward-compatible YAML config contract
3. one modular pipeline graph with stage toggles, ordering, and pluggable rankers
4. one parity-debug workflow that can enable or disable stages incrementally
5. one retirement path for Python runtime, Python OMS/risk, Python parity orchestration, and the transitional PyO3 bridge

This changes the migration framing:

- **Old framing:** Rust decision SSOT first, Python orchestration remains.
- **New framing:** Rust owns the full runtime pipeline end to end; CUDA alignment becomes a later downstream alignment task against the Rust engine, not a reason to keep Python in the loop.

## 2. Non-Negotiable Principles

1. `master` remains production-safe and Python-backed until each Rust slice is proven.
2. YAML remains backward-compatible unless the user explicitly approves a breaking change.
3. All new runtime behaviour must be expressed through Rust-owned typed config and pipeline contracts.
4. Stage gating, ranking, and sequencing must be configurable and observable, not hard-coded in one monolithic loop.
5. Python retirement happens by replacement and deletion, not by indefinite coexistence.
6. Every migration slice must improve parity debugging, not make it harder.
7. Housekeeping is part of delivery, not a postscript.

## 3. Current State Summary

## 3.1 What is already true

1. The Rust decision/kernel/backtester SSOT work is broadly complete.
2. `master` now contains the Rust runtime control plane and production service wrappers.
3. The Rust runtime owns the production paper and live service paths:
   - snapshot export/seed
   - paper doctor
   - paper run-once
   - paper cycle
   - paper loop
   - paper effective-config
   - paper manifest
   - paper status
   - paper service
   - paper service apply
   - paper daemon
   - live effective-config
   - live manifest
   - live status
   - live service
   - live service apply
   - live daemon
4. Rust owns effective-config resolution for paper, live, and factory materialisation.
5. Python daemon entrypoints are retired behind explicit archival overrides and are no longer production owners.
6. `aiq-runtime-core` already contains the right architectural seed for the future engine:
   - stable stage registry
   - explicit `StageId`
   - per-profile `stage_order`
   - `enabled_stages`
   - `disabled_stages`
   - pluggable `ranker`
   - built-in profiles such as `production`, `parity_baseline`, and `stage_debug`

## 3.2 What is not yet true

1. Frozen Python replay/parity orchestration is still present as non-authoritative compatibility tooling.
2. Some Python runtime-era helper modules still exist for tests, archival recovery, and non-runtime tools.
3. `bt-runtime` remains as a transitional PyO3 bridge until the final compatibility cleanup no longer needs it.
4. The repo still contains historical plans, validation artefacts, and compatibility surfaces that require archival labelling or deletion follow-through.

## 4. Target Architecture

## 4.1 One all-in-one Rust engine

The final engine must own the complete pipeline:

1. market data normalisation
2. indicator build
3. gate evaluation
4. signal generation
5. ranking
6. risk checks
7. intent generation
8. OMS state transition
9. broker execution
10. fill reconciliation
11. persistence and audit

The same engine contract should back:

1. live runtime
2. paper runtime
3. backtest replay
4. CPU sweep/replay validation
5. later GPU parity and GPU execution mirroring

## 4.2 One config contract, not one code path per mode

The YAML contract remains backward-compatible, but Rust becomes the only effective-config authority.

Additive config should support:

1. pipeline profiles
2. stage enable/disable lists
3. stage order overrides
4. ranker selection
5. execution adapter selection by mode
6. audit sink selection
7. state backend selection
8. parity-debug presets for narrow, reproducible stage isolation

## 4.3 One retirement path

Python runtime surfaces are not “legacy support”. They are explicit deletion targets:

1. `engine/daemon.py`, `engine/core.py`
2. `strategy/mei_alpha_v1.py` runtime ownership surfaces
3. `live/trader.py`, `exchange/executor.py`
4. `engine/oms.py`, `engine/risk.py`
5. Python replay/parity tooling where a Rust equivalent exists
6. `bt-runtime` once Python retirement is complete

## 5. Programme Structure

The programme should run as eight coordinated workstreams.

### Workstream A. Runtime Foundation Landing

Goal: merge the current Rust runtime foundation into `master` via atomic PRs from non-`master` worktrees without claiming full cutover too early.

Deliverables:

1. merge `runtime/aiq-runtime-core` into `master` via atomic PR
2. merge `runtime/aiq-runtime` into `master` via atomic PR
3. merge paper effective-config / manifest / status / service / service apply / daemon contracts via atomic PRs
4. merge a production-safe runtime ledger and cutover checklist into `master` via atomic PRs

Definition of done:

1. `master` contains the Rust runtime control plane
2. the runtime foundation is documented and testable from production-compatible paths
3. no production daemon switch happens in the same PR as the foundation landing

### Workstream B. Modular Engine Contract

Goal: turn the pipeline into a first-class Rust engine graph instead of a hard-coded Python loop.

Deliverables:

1. formal stage registry and stage execution interfaces
2. stable ranker registry
3. per-stage config toggles
4. explicit stage order validation
5. per-stage diagnostics and trace IDs
6. parity-debug profile presets

Definition of done:

1. stage ordering is not hidden inside imperative runtime glue
2. ranking can be swapped without editing the main loop
3. disabling a stage is an intentional config choice, not a code fork

### Workstream C. Rust Effective-Config SSOT

Goal: make Rust the only effective-config merge engine while preserving legacy YAML compatibility.

Deliverables:

1. Rust-owned defaults
2. Rust-owned merge precedence
3. backward-compatible YAML schema extension for `pipeline:` and runtime profiles
4. generated effective-config report shared by live, paper, replay, and factory
5. deprecation warnings for Python-only compatibility keys

Definition of done:

1. Python no longer computes effective config independently
2. config fingerprints come from one Rust path only
3. mode-specific config selection is deterministic and auditable

### Workstream D. Rust Paper Cutover

Goal: replace the active Python paper daemon with the Rust paper daemon and Rust pipeline executor.

Deliverables:

1. Rust paper daemon becomes the active paper service
2. Rust owns scheduler/watchlist reload/state progression
3. Rust owns persistence/audit for paper mode
4. Python paper execution path becomes frozen fallback and then removable

Definition of done:

1. paper production services no longer depend on Python runtime execution
2. paper cutover meets the existing quantitative promotion gate:
   - at least 1 full trading day or 20 trades
   - profit factor >= 1.2
   - max drawdown < 10%
   - slippage stress at 20 bps remains net positive
   - zero kill-switch triggers during the paper run
3. the fail-closed replay alignment gate is green and release blocker status is clear by default unless an emergency override is explicitly approved
4. deletion PRs remove obsolete Python paper runtime surfaces

### Workstream E. Rust Live Cutover

Goal: replace Python live runtime orchestration and execution with Rust-owned live runtime adapters.

Deliverables:

1. Rust live execution adapter
2. Rust broker/exchange adapter contract
3. Rust live OMS transitions and fill reconciliation
4. Rust risk enforcement
5. live service manifest/status/service control surfaces mirroring paper

Definition of done:

1. live trading runtime no longer depends on Python daemon orchestration
2. live cutover uses the existing staged ramp and rollback safeguards:
   - 25% size for 1 day with zero kill-switch triggers
   - 50% size for 1 day with zero kill-switch triggers
   - 100% size only after the ramp gate is green
   - immediate step-down, pause, or flatten path remains documented and validated
3. the fail-closed replay alignment gate remains green and release blocker status is respected by default through the cutover window
4. Python live runtime paths are deletion-ready only after the staged ramp completes successfully

### Workstream F. Parity Harness and Stage-by-Stage Validation

Goal: make parity a native capability of the Rust engine, not a Python-era sidecar.

Deliverables:

1. stage-level parity profiles such as `parity_baseline` and `stage_debug`
2. one-command parity runs for selected stage subsets
3. per-stage diff artefacts for decisions, intents, fills, balances, and OMS transitions
4. direct Rust paper/live/backtest comparison surfaces

Definition of done:

1. parity debugging can isolate the exact stage that diverges
2. “turn on one thing at a time” is a supported workflow, not a manual hack
3. Python parity scripts become optional bridges and then deletion targets

### Workstream G. CUDA Realignment After Runtime Consolidation

Goal: defer large GPU realignment until the Rust full-runtime contract is stable.

Deliverables:

1. define Rust engine parity baseline for CUDA
2. re-run CPU↔CUDA alignment stage by stage
3. migrate GPU parity ownership to the final Rust engine contracts

Definition of done:

1. CUDA work aligns against the full Rust engine, not a Python-shaped runtime
2. GPU parity is simpler because the runtime contract is already unified

### Workstream H. Housekeeping and Repository Recovery

Goal: make the repo understandable again while migration is happening.

Deliverables:

1. runtime surface ledger with statuses: active, frozen, transitional, deletion-ready, removed
2. folder ownership map for Python, Rust, tooling, and docs
3. archival or relocation of obsolete plans and experiments
4. dead compatibility bridge inventory
5. deletion PR cadence tied to each cutover milestone

Definition of done:

1. it is obvious which path is authoritative
2. dead or frozen surfaces are labelled and quarantined
3. new work no longer lands in ambiguous legacy locations

### Phase Roadmap and PR Budgets

This programme must not devolve into unlimited hardening rounds.

For the **core Python-retirement path**, we set a bounded target of **30 PRs maximum** before Python runtime ownership is considered failed or complete for this programme tranche.

That 30-PR target covers:

1. Rust runtime foundation on `master`
2. modular pipeline and config SSOT
3. Rust paper cutover
4. Rust live/OMS/risk cutover
5. Python runtime retirement

It does **not** include the later CUDA realignment tranche, which intentionally starts only after the full Rust runtime contract is stable.

#### Phase 0. Reset and Foundation Landing

Budget: **6 PRs**  
Cumulative budget: **6 / 30**

Goal:
stop the current drift pattern and merge the Rust runtime/control-plane baseline into `master` via atomic PRs.

Target PR themes:

1. runtime workspace merge
2. paper control-plane surfaces merge
3. programme docs + runtime ledger merge
4. effective-config landing path merge
5. pipeline contract bootstrap merge
6. cutover checklist + burn-in lane wiring merge

Hard exit criteria:

1. `runtime/` exists on `master` via approved PR merges
2. runtime ledger exists on `master` via approved PR merges
3. every subsequent PR is traceable to a phase checklist item

#### Phase 1. Modular Pipeline and Config SSOT

Budget: **8 PRs**  
Cumulative budget: **14 / 30**

Goal:
turn the runtime into a stage-graph engine with Rust-owned effective-config semantics.

Target PR themes:

1. stage registry hardening
2. stage executor interfaces
3. ranker registry
4. backward-compatible `pipeline:` YAML contract
5. Rust effective-config as paper authority
6. parity-debug profiles
7. audit/trace surface normalisation
8. Python config mirror freeze and retirement prep

Hard exit criteria:

1. stage ordering is config-driven
2. ranker selection is config-driven
3. Rust owns effective config for paper-facing execution

#### Phase 2. Rust Paper Cutover

Budget: **8 PRs**  
Cumulative budget: **22 / 30**

Goal:
make Rust the active paper runtime and retire Python paper execution.
This goal is now delivered on `master`; the remaining work is deletion and
compatibility cleanup.

Target PR themes:

1. paper daemon execution slice
2. paper persistence/audit ownership
3. paper parity harness slice
4. burn-in service contract
5. rollback contract
6. paper cutover runbook + operator docs
7. Python paper runtime deletion tranche
8. housekeeping/archive tranche for removed paper paths

Hard exit criteria:

1. production paper lane runs on Rust
2. Python paper execution is no longer authoritative
3. the paper gate is met using the current repo thresholds:
   - at least 1 full trading day or 20 trades
   - profit factor >= 1.2
   - max drawdown < 10%
   - net positive outcome under 20 bps slippage stress
   - zero kill-switch triggers
4. the replay alignment gate is green and release blocker status remains clear by default
5. paper deletion tranche is merged, not just planned

#### Phase 3. Rust Live/OMS/Risk Cutover and Python Retirement

Budget: **8 PRs**  
Cumulative budget: **30 / 30**

Goal:
make Rust the end-to-end production runtime and remove Python runtime ownership.
This goal is now delivered on `master`; the remaining work is compatibility
cleanup, helper extraction, and deletion follow-through.

Target PR themes:

1. live execution adapter
2. OMS state transition ownership
3. fill reconciliation ownership
4. risk runtime ownership
5. live cutover service controls
6. Rust-native parity harness finalisation
7. Python live/runtime deletion tranche
8. final retirement docs + repo cleanup tranche

Hard exit criteria:

1. live and paper both run through the Rust runtime
2. the live ramp is completed under the current staged policy:
   - 25% for 1 day with zero kill-switch triggers
   - 50% for 1 day with zero kill-switch triggers
   - 100% only after both prior stages pass
3. the replay alignment gate remains green and release blocker status stays clear by default during cutover
4. Python is no longer a production runtime dependency
5. the remaining Python footprint is either archival or explicitly non-runtime

#### Budget Enforcement Rules

1. A phase may not exceed budget by more than **2 PRs** without an explicit replan approved at the programme level.
2. A phase may not spend more than **2 hardening-only PRs in a row**. The third such PR forces a root-cause review and either a design reset, deletion tranche, or scope merge.
3. If a phase misses its exit criteria at budget, the team must stop local optimisations and open a **phase-reset PR** that explains whether the problem is foundation quality, wrong sequencing, or wrong scope.
4. Every PR must map to exactly one phase goal and one exit-criteria line item.
5. Every successful PR must still follow the mandatory repo flow:
   - implemented from a non-`master` worktree
   - merged into `master` only via atomic PR
   - reviewed by a reviewer subagent before merge
   - merged only after the review is acceptable
   - cleaned up after merge if the branch/worktree belongs to the current session
6. Every **5 PRs** the main agent runs a checkpoint review:
   - budget spent
   - exit criteria status
   - blockers
   - whether to continue, collapse, or rewrite the current slice
7. A mandatory **phase-exit review** is required at cumulative PR `6`, `14`, `22`, and `30`, even if that does not line up with the every-5-PR cadence.
8. Hardening is allowed only when it directly unlocks a phase exit. “Maybe useful later” hardening does not consume roadmap budget.

## 6. Immediate Next Move

The next move should not be “start Rust live adapter immediately”.

The next move should be:

1. merge the Rust runtime foundation into `master` via atomic PRs from non-`master` worktrees
2. promote the modular pipeline contract to a first-class programme artifact
3. define the Python retirement ledger and deletion gates
4. execute the paper cutover first

Recommended first delivery tranche:

1. **PR-01:** merge `runtime/aiq-runtime-core` and `runtime/aiq-runtime` foundation via atomic PR with no service cutover
2. **PR-02:** merge paper control-plane commands via atomic PR (`effective-config`, `manifest`, `status`, `service`, `service apply`, `daemon`)
3. **PR-03:** merge the runtime ledger, programme docs, and cutover/deletion policy via atomic PR
4. **PR-04:** merge Rust effective-config as the shared authority for paper start-up and factory materialisation
5. **PR-05:** merge the first modular pipeline execution slice behind a non-default runtime profile
6. **PR-06:** run a Rust paper burn-in lane with explicit paper-gate evidence, replay-gate evidence, cutover checklist, and rollback contract

## 7. Milestones

### Milestone 0. Programme Reset

Objective:
replace the migration narrative itself so the team optimises for Python retirement, not Python coexistence.

Exit criteria:

1. one canonical programme doc
2. one runtime ledger on `master`
3. one team topology for execution, validation, review, testing, documentation, and housekeeping
4. one bounded PR roadmap with enforced phase budgets
5. one explicit statement that every successful PR still requires reviewer-subagent review before merge

### Milestone 1. Runtime Foundation on `master`

Objective:
make the Rust runtime control plane part of the production baseline.

Exit criteria:

1. `runtime/` exists on `master`
2. `paper effective-config`, `manifest`, `status`, `service`, `service apply`, and `daemon` are buildable and documented
3. no active service cutover yet

### Milestone 2. Modular Rust Pipeline

Objective:
turn the runtime into a stage graph with profile-driven execution.

Exit criteria:

1. stage graph is config-addressable
2. rankers are registry-based
3. parity-debug profiles can isolate subsets of the pipeline

### Milestone 3. Paper Cutover and Python Paper Retirement

Objective:
make Rust the active paper runtime.

Exit criteria:

1. Rust paper daemon runs the production paper lane
2. Python paper daemon is disabled, then removed
3. paper cutover satisfies the current quantitative gate:
   - at least 1 full trading day or 20 trades
   - profit factor >= 1.2
   - max drawdown < 10%
   - net positive at 20 bps slippage stress
   - zero kill-switch triggers
4. replay alignment gate is green and release blocker status is clear by default

### Milestone 4. Live/OMS/Risk Cutover

Objective:
move live execution, OMS, reconciliation, and risk into Rust.

Exit criteria:

1. Rust owns live runtime end to end
2. Python live runtime surfaces become deletion-ready only after the existing 25% / 50% / 100% staged ramp passes
3. replay alignment gate remains green and release blocker status remains clear by default through the cutover window
4. kill-switch, flatten, pause, and rollback runbooks are re-validated

### Milestone 5. Rust-Native Parity Harness

Objective:
replace Python-era replay/parity orchestration with Rust-native tooling.

Exit criteria:

1. stage-by-stage parity runs are Rust-native
2. Python parity scripts are archived or removed

### Milestone 6. CUDA Realignment Against Final Runtime

Objective:
resume CPU↔CUDA alignment only after the Rust runtime contract is stable.

Exit criteria:

1. GPU parity matrix aligns against the final Rust engine
2. residual divergence registry is reduced under one contract

### Milestone 7. Python Retirement Complete

Objective:
remove Python as a production runtime language.

Exit criteria:

1. no production paper or live service requires Python runtime orchestration
2. no authoritative config merge logic lives in Python
3. no authoritative parity harness depends on Python
4. `bt-runtime` is either removed or explicitly retained only for non-runtime compatibility tooling

## 8. Config Enhancement Strategy

The config work must be additive and safe.

### 8.1 Keep backward compatibility

Existing YAML continues to work:

1. defaults
2. global overrides
3. per-symbol overrides
4. live overlays
5. strategy-mode overlays

### 8.2 Add a new additive `pipeline:` contract

Suggested shape:

```yaml
pipeline:
  default_profile: production
  profiles:
    production:
      ranker: confidence_adx
    parity_baseline:
      disabled_stages:
        - broker_execution
        - fill_reconciliation
    stage_debug:
      disabled_stages:
        - ranking
        - risk_checks
        - oms_transition
        - broker_execution
        - fill_reconciliation
        - persistence_audit
```

Later extensions:

1. named ranker parameters
2. stage-local options
3. mode-specific execution adapters
4. audit verbosity controls
5. trace sampling policy

### 8.3 Why this matters

This directly supports:

1. incremental parity rollout
2. precise stage isolation
3. easier root-cause analysis
4. faster debugging of live/paper/backtest drift
5. cleaner future CUDA realignment

## 9. Agentic SDLC Operating Model

The main agent remains the hub and owns:

1. planning
2. worktree creation
3. stream decomposition
4. merge gates
5. cross-stream integration
6. runtime ledger maintenance

Recommended standing teams:

1. **Architecture team**
   - runtime engine graph
   - config SSOT
   - stage/ranker contracts
2. **Paper cutover team**
   - paper daemon
   - persistence
   - paper burn-in and rollback readiness
3. **Live runtime team**
   - exchange adapter
   - OMS
   - risk
   - live supervision
4. **Parity team**
   - stage-by-stage parity harness
   - differential artefacts
   - acceptance gates
5. **Housekeeping team**
   - surface ledger
   - deletion PRs
   - archive/quarantine
   - repo layout hygiene
6. **Documentation team**
   - always-on support team for specs, runbooks, migration notes, deletion notices, and operator docs

## 10. Dedicated Documentation Team Charter

The documentation team should not be an afterthought and should not wait until after implementation.

Its standing responsibilities:

1. maintain the migration programme doc
2. maintain the runtime ledger
3. write and update runbooks per cutover slice
4. publish stage registry, ranker registry, and config contract docs
5. record every Python surface retirement and fallback removal
6. maintain a “current authoritative paths” index
7. update operator docs whenever service contracts change

Required documents:

1. Rust full-runtime cutover programme
2. runtime surface ledger
3. pipeline stage registry reference
4. ranker registry reference
5. paper cutover runbook
6. live cutover runbook
7. parity-debug cookbook
8. housekeeping/archive policy
9. deletion checklist for each retired Python surface

## 11. Housekeeping Rules for This Programme

1. No new runtime logic should be added to Python unless it is temporary safety glue with an explicit retirement note.
2. Every new Rust replacement PR must identify which Python surface becomes frozen, deprecated, or deletion-ready.
3. Every workstream must reserve capacity for deletion and cleanup, not just net-new code.
4. Plans, experiments, and superseded docs must be either archived or marked non-authoritative.
5. Repo layout should make the authoritative runtime path obvious at a glance.

## 12. First 30-Day Execution Plan

### Week 1

1. merge the programme/ledger/docs baseline via atomic PRs
2. merge the current Rust runtime foundation into `master` via atomic PRs
3. define service and config ownership boundaries

### Week 2

1. formalise the stage graph and ranker registry contract
2. wire Rust effective-config as the only authority for paper
3. add parity-debug profiles and trace artefacts

### Week 3

1. start Rust paper daemon burn-in lane
2. compare paper production and Rust paper side-by-side
3. freeze more Python paper runtime surfaces

### Week 4

1. decide whether the current quantitative Rust paper cutover gate is met
2. if yes, prepare the switch and deletion tranche
3. if not, use stage-level parity artefacts to isolate the exact blocker

## 13. What We Will Explicitly Not Do

1. We will not keep Python indefinitely just because some parity tooling is convenient today.
2. We will not attempt a giant one-shot live cutover before paper is fully Rust-owned.
3. We will not let CUDA alignment dominate the sequencing before the full Rust runtime contract is settled.
4. We will not add more hidden runtime branches, fallback engines, or duplicated config merges.

## 14. Success Condition

This programme succeeds only when the answer to all of the following is “yes”:

1. Does one Rust engine own the full runtime pipeline?
2. Can live, paper, backtest, and CPU sweep all run through that same engine contract?
3. Can parity be debugged by enabling or disabling stages one by one?
4. Is YAML still backward-compatible?
5. Have Python runtime surfaces been retired rather than merely bypassed?
6. Is CUDA alignment now a contained downstream task against one stable Rust runtime?
