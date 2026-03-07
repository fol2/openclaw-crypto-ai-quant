# Rust Full-Runtime Cutover: Exec Summary and Checkpoint Checklist

**Date:** 2026-03-07  
**Status:** Working summary for phase execution and every-5-PR checkpoint reviews  
**Companion document:** `docs/rust_full_runtime_cutover_programme.md`
**Execution checklist:** `docs/programmes/rust_runtime_phase0_cutover_checklist.md`

## 1. North Star

Retire Python as a production runtime language.

The target state is one Rust-owned end-to-end engine covering:

1. live
2. paper
3. backtest replay
4. CPU sweep/replay validation
5. later CUDA alignment against the final Rust runtime contract

Python may remain temporarily as compatibility scaffolding, but not as a permanent control plane.

## 2. What This Roadmap Optimises For

1. finish the migration
2. stop unlimited hardening loops
3. force phase exits instead of endless local fixes
4. keep YAML backward-compatible
5. make parity debugging easier through modular stage control
6. delete replaced Python runtime surfaces as we go

## 3. Hard Sequence

The sequence is fixed unless a phase-reset PR says otherwise:

1. merge Rust runtime foundation into `master` via atomic PRs from non-`master` worktrees
2. merge modular pipeline and Rust config authority via the same PR flow
3. cut over paper to Rust
4. cut over live/OMS/risk to Rust
5. retire Python runtime ownership
6. only then focus on the larger Rust↔CUDA realignment tranche

## 4. Bounded Roadmap

Core Python-retirement path budget: **30 PRs maximum**

### Phase 0. Reset and Foundation Landing

- Budget: `6 PRs`
- Cumulative: `6 / 30`
- Exit:
  - `runtime/` exists on `master` via approved PR merges
  - runtime ledger exists on `master` via approved PR merges
  - all subsequent PRs map to a checklist item

### Phase 1. Modular Pipeline and Config SSOT

- Budget: `8 PRs`
- Cumulative: `14 / 30`
- Exit:
  - stage ordering is config-driven
  - ranker selection is config-driven
  - Rust owns effective config for paper-facing execution

### Phase 2. Rust Paper Cutover

- Budget: `8 PRs`
- Cumulative: `22 / 30`
- Exit:
  - production paper lane runs on Rust
  - Python paper execution is no longer authoritative
  - paper gate meets the current quantitative threshold:
    - at least 1 full trading day or 20 trades
    - profit factor >= 1.2
    - max drawdown < 10%
    - net positive under 20 bps slippage stress
    - zero kill-switch triggers
  - replay alignment gate is green and release blocker status is clear by default
  - paper deletion tranche is merged

### Phase 3. Rust Live/OMS/Risk Cutover and Python Retirement

- Budget: `8 PRs`
- Cumulative: `30 / 30`
- Exit:
  - live and paper both run through the Rust runtime
  - live ramp follows the current staged policy:
    - 25% size for 1 day with zero kill-switch triggers
    - 50% size for 1 day with zero kill-switch triggers
    - 100% only after the ramp gate is green
  - replay alignment gate remains green and release blocker status stays clear by default during cutover
  - Python is no longer a production runtime dependency
  - remaining Python is archival or explicitly non-runtime

## 5. Budget Enforcement

1. No phase may exceed budget by more than `2 PRs` without a programme-level replan.
2. No phase may spend more than `2` hardening-only PRs in a row.
3. The third consecutive hardening-only PR forces a root-cause review or phase-reset PR.
4. Every PR must map to one phase and one exit-criteria line item.
5. Every successful PR still requires the mandatory repo flow:
   - implemented from a non-`master` worktree
   - merged into `master` only via atomic PR
   - reviewed by a reviewer subagent before merge
   - merged only after the review is acceptable
6. Every `5 PRs` the main agent must run the checkpoint review below.
7. A mandatory phase-exit review is also required at cumulative PR `6`, `14`, `22`, and `30`.

## 6. First Delivery Tranche

### PR-01

Merge `runtime/aiq-runtime-core` and `runtime/aiq-runtime` foundation into `master` via atomic PR without service cutover.

### PR-02

Merge paper control-plane commands into `master` via atomic PR:

1. `paper effective-config`
2. `paper manifest`
3. `paper status`
4. `paper service`
5. `paper service apply`
6. `paper daemon`

### PR-03

Merge the runtime ledger, programme docs, and cutover/deletion policy via atomic PR.

### PR-04

Make Rust the shared effective-config authority for paper start-up and factory materialisation via atomic PR.

### PR-05

Merge the first modular pipeline execution slice behind a non-default runtime profile.

### PR-06

Run a Rust paper burn-in lane with explicit paper-gate evidence, replay-gate evidence, cutover checklist, and rollback contract.

## 7. Phase Checklist

Use this checklist at the start of each phase.

### Scope

- Is the phase goal explicit and narrow?
- Are the exit criteria binary rather than vague?
- Is the PR budget still intact?
- Is there a named deletion target for the Python surface being replaced?

### Delivery

- Are all planned PRs mapped to one phase outcome?
- Is each PR atomic but still moving the phase forward?
- Are we avoiding speculative hardening that does not unlock the phase exit?
- Has the PR already been assigned its reviewer-subagent merge gate?

### Parity

- Does this phase make stage-by-stage parity easier?
- Can we enable or disable the relevant stages deliberately?
- Are the parity artefacts aligned with the stage graph rather than legacy Python paths?

### Housekeeping

- Has the runtime surface ledger been updated?
- Are frozen or transitional paths labelled clearly?
- Is there a deletion or archival action in this phase, not only net-new code?

## 8. Every-5-PR Checkpoint Review

Run this review after PR `5`, `10`, `15`, `20`, `25`, and `30`.

Also run a mandatory phase-exit review at PR `6`, `14`, `22`, and `30`.

### Snapshot

- Current phase:
- PRs spent in phase:
- Cumulative PRs spent:
- Remaining budget in phase:
- Remaining budget overall:

### Exit Progress

- Exit criterion 1:
- Exit criterion 2:
- Exit criterion 3:

### Evidence

- What was actually merged?
- What is still only in worktrees or draft PRs?
- What changed the critical path?

### Hardening Control

- How many consecutive hardening-only PRs have we spent?
- Did each hardening PR directly unlock an exit criterion?
- If not, should we stop and phase-reset now?

### Quality Diagnosis

- Is the current blockage caused by poor foundations?
- Is it caused by wrong sequencing?
- Is it caused by too much compatibility baggage?
- Is it caused by scope fragmentation across too many atomic PRs?

### Decision

Choose one only:

1. continue current phase
2. collapse remaining scope into fewer PRs
3. open a phase-reset PR
4. cut a deletion tranche now
5. escalate architecture redesign before spending more budget

## 9. Immediate Use

For the next checkpoint, the main agent should track:

1. whether `runtime/` has landed on `master`
2. whether the runtime ledger exists on `master`
3. whether Rust effective-config is becoming the actual authority
4. whether each PR is replacing Python ownership rather than just adding another layer

## 10. Success Test

If a PR does not help us answer “yes” to one of these, it should be challenged:

1. Does this move ownership from Python to Rust?
2. Does this help the modular stage-graph engine exist in production?
3. Does this reduce migration confusion?
4. Does this remove future PR churn rather than create more of it?
