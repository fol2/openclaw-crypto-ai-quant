# Shared Config Financial-Grade Hardening Plan

## Status

Active.

Prepared on 2026-03-14 after an independent three-reviewer architecture review
of the current modular/shared config control plane and runtime contract.

## Execution Progress

Progress recorded on 2026-03-14:

- PR 1 completed and merged as `#997` (`runtime: harden live config contract defaults`).
  This aligned Hub and runtime live-YAML defaults on
  `config/strategy_overrides.live.yaml`, preserved `live:` overlays during live
  runtime materialisation, and made live config resolution fail closed when the
  real live YAML is missing.
- PR 2 completed and merged as `#999` (`hub: harden auth and mutation routes`).
  This made read auth mandatory outside explicit development mode, added a
  dedicated admin token for mutation routes, replaced permissive CORS with an
  explicit allow-list contract, and restored a working browser auth path for
  the built-in Hub UI and WebSocket client.
- PR 3 completed and merged as `#1001`
  (`hub: make config writes validation-aware`).
  This routed config saves through runtime-grade validation, required
  optimistic-lock proof via `expected_config_id` / `If-Match`, surfaced config
  identity headers to the editor, and added stale-edit rejection for raw YAML
  changes including comment-only edits.
- PR 5 completed and merged as `#1010`
  (`hub: make live apply and rollback transactional`).
  This added a dedicated `POST /api/config/actions/apply-live` contract,
  routed both live apply and live rollback through runtime-supervised
  `aiq-runtime live service apply --json` verification, snapshotting the
  incumbent YAML before mutation, restoring it automatically when apply proof
  failed, and recording before/after `config_id` identities in the emitted live
  apply / rollback artefacts.
- PR 6 implementation completed on 2026-03-14
  (`runtime: bind launch contracts to immutable config artefacts`).
  This switched paper/live manifests to launch daemons from materialised
  `config_path` artefacts, propagated `--expected-config-id` through paper/live
  daemon and service-apply surfaces, persisted daemon `config_id` in paper/live
  status files, made status checks fail closed on `config_id` drift, and
  tightened Hub live apply to pass the approved `config_id` into the runtime
  apply subprocess. At the time of this documentation pass the PR is
  implementation-complete on branch `codex/shared-config-pr6-immutable-launch`
  and is awaiting the required review / merge flow.

Sequence status after merged PR 5 and the current PR 6 implementation pass:

- PR 1: complete
- PR 2: complete
- PR 3: complete
- PR 5: complete
- PR 6: implementation complete; review / merge pending
- PR 4: next planned implementation stage after PR 6 merges
- PR 7: pending
- PR 8: pending
- PR 9: pending

## Current Stage

Current execution stage as of 2026-03-14:

- PR 5 is merged and closed.
- PR 6 implementation is complete and the documentation pass is now describing
  the post-change immutable launch contract rather than a design gap.
- Paper/live manifests now launch daemons from materialised `config_path`
  artefacts, paper/live daemon and service-apply surfaces now accept
  `--expected-config-id`, daemon status files persist `config_id`, and
  paper/live status checks now fail closed on `config_id` drift.
- PR 4 (`Replace false reload semantics with an honest apply model`) is the
  next active implementation stage once PR 6 finishes review and merge.

Current blocker:

- None at the documentation-pass stage. The next action is reviewer coverage
  for PR 6, followed by merge and cleanup before opening PR 4.

## Objective

Close the remaining correctness, security, audit, and operational-governance
gaps between the current config architecture and a truly financial-grade system.

This plan exists so the repository keeps one durable source of truth for the
required work even if chat context is discarded.

## Current State

The repository already has several strong primitives:

- `HubConfig` provides a typed Hub-side config model for monitor, config,
  factory, system, and trade surfaces.
- `PaperEffectiveConfig` materialises active, effective, and runtime-facing YAML
  contracts and emits a stable `config_id`.
- `paper manifest` and `live manifest` produce typed launch contracts and
  runtime fingerprints.
- live runtime launch still passes through explicit safety gates before live
  trading is marked ready.

However, the current end-to-end control plane is not yet financial-grade. The
main reasons are:

1. some control-plane defaults are too permissive for a trading system
2. live config correctness is not fully preserved across Hub and runtime paths
3. config mutation is not yet transactional or config-identity-aware
4. audit, attribution, and operator messaging do not yet track deployed
   `config_id` as the primary source of truth

## Confirmed Gaps

### 1. Control-plane exposure is too permissive

The Hub currently allows authentication to become a no-op when
`AIQ_MONITOR_TOKEN` is unset, and the router still uses permissive CORS. That is
not sufficient for a financial control plane that can mutate config, restart
services, expose live-operational state, and trigger manual trading workflows.

Required outcome:

- authentication is mandatory in non-development deployments
- CORS is narrowed to an explicit allow-list
- mutation endpoints cannot rely on the same undifferentiated access model as
  read-only endpoints

### 2. Live config correctness is not guaranteed

There are multiple correctness defects in the live path:

- Hub rollback defaults point at `config/strategy_overrides.yaml` while the live
  lane defaults point at `config/strategy_overrides.live.yaml`
- `aiq-runtime live effective-config` still documents and falls back to
  `strategy_overrides.yaml` instead of the conventional live YAML path
- `live:` YAML overlays are lost during runtime materialisation because the
  shared effective-config path currently materialises runtime YAML with
  `is_live=false`
- missing live config can silently fall back to `.example` YAML instead of
  failing closed

These are not theoretical concerns. The review confirmed both the lost
`live:`-overlay behaviour and the live-manifest `.example` fallback with local
CLI reproductions.

Required outcome:

- Hub and runtime share one authoritative live-YAML target contract
- all shipped live-facing CLI and manifest surfaces resolve that same default
  contract
- live runtime materialisation preserves `live:` overlays end to end
- live config resolution fails closed when the real live config is missing

### 3. Config mutation is not yet transactional

The current write path validates only YAML syntax, then writes directly to the
target file. It does not yet:

- reuse runtime-grade config validation
- require the caller to prove which current config version they edited
- guard against concurrent last-writer-wins overwrites
- guarantee that restart failure restores the pre-change config state

Rollback has a related issue: the current implementation writes the restored
payload first and only then attempts restart, which can leave file/runtime drift
behind when restart fails.

Required outcome:

- all config changes go through validate -> stage -> apply -> verify
- writes use unique temp files, durable replacement, and optimistic locking
- restart failure restores the prior config automatically

### 4. Runtime launch immutability is now implemented

This gap is closed in the current PR 6 implementation branch.

The runtime launch contract now:

- launches paper/live daemons from materialised `config_path` artefacts rather
  than mutable `base_config_path`
- propagates `--expected-config-id` through daemon launch and service-apply
  surfaces
- persists daemon `config_id` in paper/live status files
- fails closed when paper/live status detects `config_id` drift from the
  current launch contract

### 5. Reload semantics are misleading to operators

The Hub exposes `Save & Reload`, but the backend path only touches file mtime.
Current Rust daemons do not perform a true in-process strategy-YAML reload from
that signal. This creates false operator confidence, especially around
`engine.interval`, which the repository already documents as not hot-reloadable.

Required outcome:

- the UI and API either implement true fail-closed config application semantics
  or stop describing mtime touch as a reload
- restart-required changes are surfaced honestly before apply

### 6. Audit and attribution are not config-id-centric

Current save history, rollback events, and monitoring views are not yet centred
on deployed `config_id`. The monitor still derives `since_config` from file
mtime, rollback events record process-level `USER` / `HOSTNAME` rather than a
real request actor, and save history stores anonymous backup names rather than a
durable deployment identity trail.

Required outcome:

- config audit is append-only and keyed by before/after `config_id`
- request actor, reason, target lane, and validation result are recorded for
  every mutation
- operational reporting uses deployed `config_id`, not file mtime, for
  attribution boundaries

### 7. Sensitive operational metadata needs redaction boundaries

Several surfaces currently expose raw YAML, diff payloads, secrets-path
locations, or daemon command arguments that include secrets-path values. Even
when secret contents are not exposed directly, the control plane should not
default to leaking sensitive host-path and live-operation metadata.

Required outcome:

- sensitive path and command fields are redacted by default
- raw live-config access is explicitly privileged and auditable
- operator diagnostics expose only the minimum required data by default

### 8. Governance workflow is not yet financial-grade

The system does not yet separate read-only access, config editing, and live
approval duties. There is no maker-checker or dual-control boundary for the
highest-risk live mutations.

Required outcome:

- role separation exists at minimum for viewer, config editor, and live
  approver responsibilities
- live rollback or live-cutover actions require stronger approval semantics than
  read-only monitoring

## Guardrails

The following constraints must remain true across all work in this plan:

1. Live-trading guardrails remain fail-closed unless the user explicitly
   approves an emergency bypass.
2. The production runtime profile stays on `production` unless an operator
   explicitly selects a debug or parity lane.
3. Missing live config must not silently degrade to sample or example config.
4. Shared config identity must move towards `config_id` as the authoritative
   deployment contract, not away from it.
5. Every successful change is delivered as one atomic PR to `master`, with a
   documentation subagent sweep before review and a reviewer subagent before
   merge.

## Proposed Atomic PR Sequence

### PR 1. Fix live contract correctness

Scope:

- align Hub live rollback defaults with the live lane's real YAML target
- align `aiq-runtime live effective-config` defaults with the same live YAML
  target contract
- preserve `live:` overlays during runtime config materialisation
- stop live config resolution from falling back to `.example` files
- add regression coverage for live effective-config and live manifest behaviour

Acceptance:

- Hub rollback, live manifest, and `live effective-config` all target the same
  live YAML path by default
- `aiq-runtime live effective-config` preserves `live:` scoped values
- live manifest fails closed when only example YAML exists

### PR 2. Harden default Hub exposure and mutation gating

Scope:

- require authentication for non-development Hub deployments
- replace permissive CORS with an explicit allow-list model
- gate `PUT /api/config`, reload, rollback, and similar mutation endpoints
  behind explicit admin controls
- separate read-only and mutation access paths at the route contract level

Acceptance:

- mutation surfaces are no longer reachable when auth is effectively disabled
- cross-origin browser access is denied by default unless explicitly allowed
- config mutation cannot bypass the admin gate

### PR 3. Make config writes validation-aware and concurrency-safe

Scope:

- route all config writes through runtime-grade config validation
- require expected `config_id` or ETag on mutation requests
- replace fixed temp-file naming with unique staged writes plus durable replace
- add integration coverage for invalid config rejection and concurrent edit
  conflicts

Acceptance:

- syntax-valid but runtime-invalid config is rejected before write
- stale editors cannot silently overwrite newer config
- concurrent writes cannot clobber one another through shared temp filenames

### PR 4. Replace false reload semantics with an honest apply model

Scope:

- remove or rename the current mtime-touch reload semantics
- change UI messaging from ambiguous reload language to truthful apply/restart
  language
- surface restart impact before mutation completes
- preserve clear operator guidance for `engine.interval` and other non-hot
  reloadable changes

Acceptance:

- the product no longer claims a strategy-YAML hot reload that does not exist
- operators can see whether a service restart is required before apply
- documentation and UI copy match real runtime behaviour

### PR 5. Make rollback and apply transactional

Scope:

- stage candidate config changes before replacing the live target
- snapshot the pre-change live payload before any write
- restart and health-check the affected service before final success is recorded
- auto-restore the prior config if restart or post-apply verification fails

Acceptance:

- a failed restart cannot leave config/file/runtime drift behind
- rollback events record both pre-change and restored config identities
- success is reported only after the target runtime is healthy on the intended
  config

### PR 6. Launch daemons from immutable config artefacts

Scope:

- switch daemon launch contracts from mutable base YAML paths to materialised
  config artefacts
- require expected `config_id` during launch/apply workflows
- ensure runtime status and service inspection report the same config identity

Acceptance:

- previewed and launched configs cannot diverge through mutable base-YAML edits
- runtime status can prove which approved config artefact is in force
- live status and live service inspection/apply checks fail when `config_id`
  drifts from the approved launch contract

### PR 7. Make audit and monitoring config-id-centric

Scope:

- record append-only config mutation events keyed by before/after `config_id`
- attach actor identity, reason, lane, validation outcome, and restart result
  to each event
- move `since_config` and similar operational reporting from file mtime to
  deployed `config_id`

Acceptance:

- config history answers who changed what, why, and from which config identity
- monitoring views can attribute performance by deployed config identity

### PR 8. Add redaction boundaries for sensitive operational metadata

Scope:

- redact sensitive path, raw live-config, and daemon-command metadata by default
- require explicit privilege for raw live-config or sensitive diagnostics views
- ensure redacted and non-redacted read paths are separately auditable

Acceptance:

- sensitive operational metadata is not exposed on default read paths
- privileged diagnostic access is explicit and auditable
### PR 9. Add financial-grade approval workflow and role separation

Scope:

- introduce role separation for at least viewer, config editor, and live
  approver responsibilities
- add stronger approval semantics for live rollback and live-cutover actions
- ensure high-risk live mutations are auditable at the actor and approval level

Acceptance:

- the highest-risk live mutations no longer rely on one undifferentiated bearer
- audit logs show both requester and approver where required
- live control-plane actions satisfy a basic maker-checker governance model

## Recommended Delivery Order

The recommended order is:

1. PR 1
2. PR 2
3. PR 3
4. PR 5
5. PR 6
6. PR 4
7. PR 7
8. PR 8
9. PR 9

This order fixes correctness and exposure first, then removes config drift
windows, then hardens governance and operator workflow.
