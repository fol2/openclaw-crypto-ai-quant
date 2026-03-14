# Factory Financial-Grade Hardening Plan

## Status

Active.

Prepared on 2026-03-14 after an independent three-reviewer design review of the
current Rust factory cycle.

## Progress

Progress as of 2026-03-14:

- Activated on `master` by PR #991.
- Completed on `master`:
  - PR 1 via PR #993: preserve lane-effective configs and full YAML roots.
  - PR 2 via PR #1000: deterministic role-governed selection, role materiality
    thresholds, and truthful partial-advancement reporting.
  - PR 4 via PR #995, with follow-up fix PR #996: fail-closed paper execution
    evidence, marker rollback hardening, and paper promotion truthfulness.
  - PR 5 via PR #998: manifest-driven `live_small` / `live_full` governance
    with persisted live governance state.
- In progress:
  - PR 3 via PR #1004: explicit train/holdout validation windows, holdout
    artefacts, and train-window parity replays. Implementation, local
    validation, and documentation refresh are complete; PR/reviewer flow is
    still pending.
  - PR 3 follow-up via PR #1006: fix the inclusive train/holdout boundary so
    the two validation windows never share the same replay bar.
- Pending:
  - PR 6.

The `Current State` and `Confirmed Gaps` sections below are the baseline taken
when this plan was prepared. Gaps 1, 3, 4, and 5 are now closed on `master`.
Gap 2 is being closed by the current PR 3 train/holdout work, while the
remaining Gap 6 scheduler/parity items are still open.

## Objective

Close the remaining governance and evidence gaps between the current Rust-owned
factory cycle and a truly financial-grade candidate -> paper -> live lifecycle.

This plan exists so the repository keeps one durable source of truth for the
remaining work even if chat context is discarded.

## Current State

The repository already has a Rust-owned factory executor, Rust Hub integration,
GPU TPE-backed candidate generation, challenger-vs-incumbent comparison for
paper lanes, live-equity balance seeding, and fail-closed paper soak gating.

However, the current implementation is not yet sufficient to call the full
factory lifecycle financial-grade. The main reasons are:

1. some paper lanes can validate one effective config but trade another
2. the current validation stack still reuses the same window as both search and
   so-called "walk-forward" evidence
3. the paper promotion gate does not yet rely on complete paper execution
   evidence
4. the live promotion path does not yet implement the documented ramp and
   rotation lifecycle

## Confirmed Gaps

### 1. Lane materialisation mismatch

The factory currently writes candidate YAMLs that only preserve `global`. This
drops top-level `modes` and any other root overlays that the paper lanes rely
on. `paper2` and `paper3` default to `fallback` and `conservative` strategy
modes, so this creates a real risk that the factory validates one behaviour set
but deploys another.

Required outcome:

- factory-generated YAML keeps the full root structure required for lane parity
- validation, incumbent comparison, deployment, and soak markers all refer to
  the same lane-effective config

### 2. False OOS validation

The current flow computes one common overlap window, uses it for sweep/TPE, then
uses the same window again for replay validation and split-based "walk-forward"
summary generation. That is not true out-of-sample governance.

Required outcome:

- candidate generation and candidate validation use explicitly separated train
  and holdout windows
- artefacts record the train and holdout boundaries
- until true OOS exists, the code and docs must not describe the current metric
  as OOS

### 3. Incomplete paper promotion evidence

The paper promotion gate currently derives profit factor and drawdown from fill
rows and reuses candidate backtest slippage stress as a live-promotion signal.
This is insufficient for execution-quality governance and does not fully honour
the documented paper gate.

Required outcome:

- paper promotion gate reads actual paper execution evidence
- gate includes slippage, funding drag, rejection rate, latency, and kill-switch
  evidence collected during the soak window
- drawdown uses a paper MTM or high-water-mark based equity series rather than
  fill-only balance snapshots

### 4. Soak-marker and rollback truthfulness

The soak gate currently trusts the marker plus a limited identity check. It does
not fully re-derive the current deployed target fingerprint during promotion
assessment, and rollback handling is not yet strong enough to guarantee that
stale marker state can never survive a failed deployment sequence.

Required outcome:

- promotion gate revalidates the currently deployed paper target fingerprint
- rollback restores both YAML and marker state transactionally
- stale or partial marker state must fail closed

### 5. Missing live governance state machine

The documentation describes `paper -> live_small -> live_full` with ramping,
pause, rotation, and retirement criteria. The current implementation promotes a
paper-soaked primary directly to live YAML and optionally restarts the live
service. There is no first-class live ramp, no live incumbent replace gate, and
no rotation evaluator in the factory contract.

Required outcome:

- live promotions enter a governed `live_small` state instead of jumping straight
  to full-size steady state
- ramp stages support 25% -> 50% -> 100% target exposure
- the factory records state transitions, timestamps, and reasons
- live replacement is checked against current live incumbent state before switch
- rotation criteria are enforceable from live evidence

### 6. Selection and scheduler governance gaps

There are still governance gaps around role selection and automation:

- mixed-mode fallback selection is not deterministic enough for strict auditability
- current comparison rules can replace targets on trivial absolute deltas when
  seeded by live equity including unrealised PnL
- all paper targets can block the whole cycle instead of allowing primary to
  advance independently
- daily and deep timers still share the same schedule and rely on the global
  factory lock to avoid collisions
- factory parity gating is still too aggregate-heavy

Required outcome:

- deterministic per-role comparator logic
- challenger materiality thresholds before replacement
- primary lane can progress without secondary hard-blocking
- daily and deep schedules cannot race
- parity enforcement is strong enough to catch per-symbol or event-level drift,
  not only aggregate balance drift

## Guardrails

The following constraints must remain true across all work in this plan:

1. The repository stays Rust-only.
2. The production runtime profile stays on `production` unless an operator
   explicitly chooses a debug or parity lane.
3. Factory changes remain fail-closed for live trading.
4. Live-equity balance seeding remains available because it is the intended
   production default, but replacement logic must not overreact to transient MTM
   noise.
5. Every successful change is delivered as an atomic PR to `master` with reviewer
   sign-off.

## Proposed Atomic PR Sequence

### PR 1. Preserve full YAML roots and lane-effective configs

Scope:

- preserve `global`, `modes`, `symbols`, and any other required top-level root
  data when materialising factory candidates
- materialise exact lane-effective configs for `primary`, `fallback`, and
  `conservative`
- ensure replay validation, incumbent comparison, deployment, and soak evidence
  all use those lane-effective configs
- add regression tests proving `paper2` and `paper3` match their intended
  strategy modes

Acceptance:

- factory-generated fallback and conservative artefacts preserve the root overlays
- paper lane effective-config output matches the config that factory validated
- no lane can silently lose its strategy-mode overlay

### PR 2. Make selection deterministic and role-governed

Scope:

- replace mixed comparator fallback with deterministic per-role ordering
- allow `primary` to advance independently when secondary lanes have no valid
  challenger
- add challenger materiality settings such as minimum return uplift, minimum
  profit-factor uplift, and permitted drawdown slack
- ensure selection and report outputs reflect `incumbent_holds` and partial
  advancement truthfully

Acceptance:

- repeated runs on the same artefacts produce the same role ordering
- a primary challenger can advance without fallback or conservative blocking the
  entire cycle
- reports cannot overstate advancement when no actual replacement happened

### PR 3. Replace pseudo-OOS validation with true train/holdout governance

Scope:

- introduce explicit train and holdout windows for factory validation
- update candidate generation and validation artefacts to record those windows
- rename current metrics if needed until true OOS is implemented fully
- keep GPU TPE, replay, and parity checks aligned with the same modular config
  contract

Acceptance:

- shortlist generation and validation use different time windows
- artefacts expose train/holdout boundaries
- docs no longer describe same-window replay as OOS

### PR 4. Rebuild paper promotion evidence fail-closed

Scope:

- derive paper execution-quality metrics from paper runtime artefacts and DBs
- use MTM or HWM equity evidence for drawdown rather than fill-only balance
- count kill-switch events across the full soak time window, not only fill-linked
  run fingerprints
- revalidate target fingerprint during promotion assessment
- make rollback restore marker state as well as YAML

Acceptance:

- paper promotion gate can fail on real paper slippage, reject rate, latency, or
  funding drag degradation
- kill-switch events without fills still block promotion
- stale soak markers cannot authorise promotion

### PR 5. Implement the live governance state machine

Scope:

- add `live_small` and `live_full` execution states to the factory lifecycle
- support staged sizing ramp with recorded stage transitions
- add live incumbent governance and replacement checks
- record transition reasons, timestamps, and config identity metadata
- add rotation evaluators for the documented live criteria

Acceptance:

- paper promotion no longer implies direct steady-state live replacement
- live ramp state is visible and auditable
- current live config can be paused or held according to the documented rotation
  rules

### PR 6. Harden scheduler and parity guardrails

Scope:

- separate the daily and deep timer cadence, or replace them with a single
  non-racing orchestrator
- fail closed when factory settings are missing or incomplete in a deployment mode
- strengthen factory parity checks from aggregate-only to at least per-symbol
  evidence, with an event-trace lane where practical

Acceptance:

- daily and deep factory jobs cannot collide by schedule design
- deployment mode cannot silently fall back to permissive defaults
- factory parity gate rejects materially divergent symbol-level behaviour

## Recommended Delivery Order

The recommended order is:

1. PR 1 - Complete via PR #993
2. PR 4 - Complete via PR #995 and PR #996
3. PR 5 - Complete via PR #998
4. PR 2 - In progress via PR #1000
5. PR 3 - Pending
6. PR 6 - Pending

Rationale:

- PR 1 removes the most dangerous "validate one config, trade another config"
  risk
- PR 4 tightens the paper evidence model before more live automation is added
- PR 5 is the main financial-grade lifecycle tranche and should not be built on
  top of weak paper evidence
- PR 2 and PR 3 improve governance and research discipline after the deployment
  path is made honest
- PR 6 closes automation and parity loopholes across the finished stack

## Repository Areas Expected To Change

- `runtime/aiq-runtime/src/bin/aiq-factory.rs`
- `runtime/aiq-runtime/src/paper_*`
- `runtime/aiq-runtime-core/`
- `config/factory_defaults.yaml`
- `config/strategy_overrides*.yaml`
- `hub/` factory status and reporting surfaces
- `systemd/openclaw-ai-quant-factory-*.example`
- `docs/runbook.md`
- `docs/strategy_lifecycle.md`
- `docs/success_metrics.md`

## Acceptance For The Whole Plan

This plan is complete only when all of the following are true:

1. Factory validation, paper deployment, and lane runtime all resolve the same
   effective config per role.
2. Factory validation uses genuine holdout evidence, not same-window reuse.
3. Paper promotion relies on real paper execution evidence and time-windowed risk
   evidence.
4. Live promotion uses a governed ramp and live rotation contract.
5. Selection, reporting, and scheduling remain deterministic and audit-friendly.
6. Docs describe exactly the behaviour implemented by code and services.

## Deferred Questions

These questions should be resolved during implementation, not before this plan
enters `active/`:

1. Which exact materiality thresholds should govern challenger replacement under
   live-equity seeding?
2. Which paper execution-quality metrics are already available in authoritative
   DBs, and which need new artefact surfaces?
3. Should live ramp sizing be implemented in config overlays, runtime profile
   controls, or dedicated factory-generated live manifests?
