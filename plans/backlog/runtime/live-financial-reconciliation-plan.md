# Live Financial Reconciliation Plan

## Status

Backlog.

Previously activated on 2026-03-14 and partially executed through PR 3.

Returned to backlog on 2026-03-15 after the merged PR 1/2/3 change set was
functionally rolled back from `master`.

Prepared on 2026-03-14 after a production incident review and three
independent expert critiques covering trading operations, accounting truth, and
real-money risk architecture.

This plan exists so the repository keeps one durable source of truth for the
required work even if chat context is discarded.

## Objective

Replace the current partial live sync model with a financial-grade
reconciliation architecture where:

- external Hyperliquid activity is captured as durable evidence
- holdings, realised cash, equity, margin, and withdrawable figures have clear
  and non-overlapping semantics
- Hub and operator surfaces show graded truth with explicit freshness and
  reconciliation status
- no read or execution path silently treats stale or partial exchange state as
  fully reconciled truth

The target outcome is not merely "fills have been synced". The target outcome
is that exchange truth, internal ledger truth, and operator-visible current
state can be explained, reconciled, and audited end to end.

## Current State

The repository currently has:

- a `live sync-fills` path that can ingest external fills into local OMS and
  trade tables
- a short-term fallback that persists current exchange account and position
  snapshots into the live DB
- Hub read paths that can fall back to those DB snapshots when in-memory
  Hyperliquid state is stale or unavailable

That is useful as a continuity patch, but it is not yet a financial-grade
reconciliation model.

The main issues are:

1. `trades.balance` is not a trustworthy live accounting source
2. exchange `account_value` is currently too easy to confuse with realised cash
3. fills, exchange observations, and Hub truth source are still too tightly
   coupled
4. append-only evidence capture is incomplete
5. reconciliation completeness is not currently graded in a way operators can
   trust for real-money control

## Confirmed Gaps

### 1. Truth sources are mixed together

The current live stack mixes:

- fill-level local ledger rows
- current exchange observations
- UI/operator truth source

into one conceptual bucket.

That creates the exact failure mode already observed in production: holdings
can look correct while balance semantics remain wrong, or a fallback snapshot
can make the system look current without proving that the ledger is complete.

Required outcome:

- exchange observations, ledger events, reconciled projections, and operator
  display state must be separate layers

### 2. Live accounting semantics are currently wrong or overloaded

The current live implementation still allows these semantic failures:

- `account_value` being interpreted as if it were realised cash
- `account_value - withdrawable` being treated like unrealised PnL
- `trades.balance` being used as a live accounting field even though some live
  paths write `NULL` and others write exchange account snapshots

Required outcome:

- realised cash, unrealised PnL, equity, withdrawable, and margin used each
  get their own explicit contract
- live operator surfaces stop presenting exchange equity as realised balance

### 3. The fill cursor is not a completeness proof

The current fill sync window plus overlap model reduces the chance of missed
fills, but it does not prove completeness. Reordered, late-arriving, or
temporarily missing rows can still pass through the windowing model.

Required outcome:

- the system records what was requested, what was received, and what remains
  unreconciled
- operators can tell the difference between "no parse issues observed" and
  "fully reconciled"

### 4. Current snapshot storage is not audit-grade

The short-term fallback currently improves continuity, but it does not yet give
the repository a durable audit trail for:

- which exchange state was observed at which exact time
- which sync run produced it
- whether that state was accepted as authoritative, degraded, or rejected

Required outcome:

- append-only exchange observation events
- append-only fill sync run records
- durable reconciliation status for every run

### 5. Position fallback can still become operationally misleading

Even after the current short-term fixes, a current-state snapshot fallback is
still dangerous if:

- its freshness policy is too loose
- it is treated as execution-grade truth
- the UI does not show source, age, and reconciliation grade prominently

Required outcome:

- stale fallback is informational only
- execution, risk, and approvals do not silently inherit snapshot truth unless
  a stricter contract explicitly allows that

### 6. Non-fill cash events are still missing from the accounting model

A financially correct live ledger needs more than fills. It also needs:

- fees
- funding
- deposits
- withdrawals
- transfers
- rebates
- manual adjustments
- liquidation and ADL events when applicable

Required outcome:

- no live realised-cash figure is treated as audit-grade until those event
  classes are covered or explicitly marked incomplete

### 7. Daily and all-time PnL metrics are not yet trustworthy

Current drawdown, realised balance, and range PnL paths still depend too
heavily on `trades.balance`.

Required outcome:

- performance metrics move onto explicit accounting projections
- until then, live historical metrics are marked degraded or untrusted

### 8. The system needs explicit reconciliation grading

Operators currently need to inspect implementation details to know whether the
system is:

- current
- stale but usable as last-known-good
- partially reconciled
- outright unsafe to trust

Required outcome:

- every live status surface includes source, age, coverage, and reconciliation
  grade

## Target Principles

1. Append-only evidence first.
2. Semantic clarity over convenience.
3. Exchange observations are not the same thing as internal accounting truth.
4. Current-state fallback is informational unless proven otherwise.
5. Reconciliation status must be explicit and operator-visible.
6. No silent downgrade from authoritative to estimated or stale state.
7. Every successful change lands as one atomic PR to `master`.

## Guardrails

The following constraints must remain true across all work in this plan:

1. Live-trading guardrails remain fail-closed unless the user explicitly
   approves an emergency bypass.
2. Snapshot fallback must not silently become execution truth for sizing,
   approvals, or risk controls.
3. A failed or partial sync must not leave behind current-looking authoritative
   state without an explicit degraded or unreconciled marker.
4. Each PR in this sequence must remain atomic and independently reviewable.
5. The production worktree remains on `master`; all implementation work happens
   in separate non-`master` worktrees.

## Interim Operator Policy

Until the early PRs in this plan land, treat the current DB-backed live
snapshot fallback as:

- suitable for continuity of read-only monitoring
- unsuitable as the sole basis for live risk sign-off
- unsuitable as an audit-grade realised balance source

The system should continue to show current exchange-backed holdings and equity,
but operators must not interpret that as proof that the full books and records
are reconciled.

## Proposed Atomic PR Sequence

### PR 1. Split live balance semantics in Hub read paths

Scope:

- stop presenting exchange `account_value` as realised live balance
- introduce explicit naming for exchange-observed equity and related figures
- add source, freshness, and reconciliation-status fields to live balance and
  holdings payloads
- surface degraded or snapshot-backed state clearly in Hub responses and UI

Acceptance:

- live API payloads no longer label exchange equity as realised cash
- operator surfaces show `source`, `as_of`, and freshness state
- stale fallback is clearly marked as last-known-good rather than current truth

### PR 2. Introduce append-only sync run records

Scope:

- add `exchange_sync_runs` or equivalent append-only run headers
- record wallet identity, config/profile context, requested window, received
  counts, unsupported counts, and final run status
- tie each snapshot/fill ingestion action to a `run_id`

Acceptance:

- every sync run leaves a durable header row
- operators can answer when the last successful and last failed runs occurred
- unsupported or degraded runs are queryable without reading logs

### PR 3. Capture exchange account observations as append-only evidence

Scope:

- add append-only `exchange_account_snapshot_events`
- capture a single coherent exchange observation time for each snapshot record
- store raw payload or a raw payload digest plus source metadata

Acceptance:

- account observations are no longer overwrite-only state
- every snapshot row is attributable to one run and one observation time

### PR 4. Capture exchange position observations as append-only evidence

Scope:

- add append-only `exchange_position_snapshot_events`
- stop relying on destructive current-state overwrite as the only persisted
  position snapshot
- materialise a separate current-view table only after successful ingestion

Acceptance:

- the repository can reconstruct which exchange positions were observed for a
  given run
- a current-view table exists, but it is derived from append-only evidence

### PR 5. Stage raw fill evidence before ledger mutation

Scope:

- introduce append-only `exchange_fill_events`
- capture all received fill payloads before trying to reconcile them into OMS
  or trade rows
- distinguish raw evidence capture from reconciliation success

Acceptance:

- unsupported or partially understood fills are still durably captured
- operators can inspect what the exchange returned even when reconciliation is
  incomplete

### PR 6. Make fill reconciliation two-phase and explicitly graded

Scope:

- move from immediate mutable writes to stage -> validate -> reconcile -> commit
- ensure cursor advance happens only when the run is complete by the plan's
  own completeness contract
- add per-run reconciliation grade and exception summary

Acceptance:

- partial reconciliation no longer looks like successful reconciliation
- no current-looking state is committed without an explicit accepted grade

### PR 7. Add current-state projection tables with explicit authority grades

Scope:

- build read-only projections such as:
  - `current_exchange_account_view`
  - `current_exchange_positions_view`
  - `reconciled_live_state`
- include `source`, `as_of_ts_ms`, `age_ms`, `coverage_status`,
  `reconciliation_grade`, and exception counts

Acceptance:

- Hub no longer reads raw snapshot tables directly
- all live fallback state flows through one explicit projection contract

### PR 8. Tighten fallback freshness and operator UX

Scope:

- enforce stricter staleness thresholds for current-state fallback
- add operator-visible banners or payload fields for `current`, `stale`,
  `degraded`, and `unreconciled`
- make position-empty and position-mismatch cases explicit instead of silent

Acceptance:

- a stale or degraded live snapshot cannot silently present itself as current
- operators can see at a glance whether the view is trustworthy

### PR 9. Separate live historical performance from current exchange state

Scope:

- deprecate `trades.balance` as a live accounting source
- stop live daily/range metrics from quietly depending on mutable or mixed
  balance semantics
- introduce explicit temporary semantics such as `exchange_equity_usd`,
  `realised_cash_usd_untrusted`, or similar

Acceptance:

- live historical PnL and drawdown no longer pretend to be audit-grade when
  they are not
- current exchange state and internal accounting state are shown separately

### PR 10. Ingest funding as a first-class event stream

Scope:

- capture funding events into append-only evidence and accounting pathways
- bridge funding into realised-cash or net-realised-PnL calculations
- add reconciliation status for funding completeness

Acceptance:

- funding is no longer an unmodelled source of unexplained equity drift
- operators can see funding contribution separately from trade PnL

### PR 11. Add non-funding cash event capture

Scope:

- ingest deposits, withdrawals, transfers, rebates, and manual adjustments
- record them as append-only cash events
- extend projections to include unexplained delta when not all cash events are
  covered

Acceptance:

- the system can explain changes in cash that do not originate from trade fills
- live realised cash moves closer to audit-grade completeness

### PR 12. Introduce explicit reconciliation exceptions and daily controls

Scope:

- compute and persist:
  - exchange vs local position mismatches
  - fill completeness gaps
  - unexplained equity deltas
  - stale observation incidents
- add daily reconciliation reports and alerts

Acceptance:

- the repository can produce a daily reconciliation report
- failures are first-class data, not just log lines

### PR 13. Add execution-control boundaries

Scope:

- ensure execution, risk approval, and manual trade workflows can only rely on
  state sources explicitly allowed by policy
- prevent a stale or degraded read-path fallback from becoming execution truth

Acceptance:

- snapshot fallback is formally informational unless a later design explicitly
  upgrades it
- risk and approval logic cannot silently consume degraded truth

### PR 14. Deliver the long-term ledger and projection redesign

Scope:

- introduce append-only financial journal inputs for fills, fees, funding,
  cash events, and adjustments
- build deterministic projections for:
  - realised cash ledger
  - open position sub-ledger
  - valuation series
  - equity series
  - drawdown and PnL metrics
- retire live dependence on `trades.balance`

Acceptance:

- live holdings, cash, equity, and performance metrics are all projection-driven
- operator and audit surfaces derive from one coherent accounting model
- the old mixed snapshot/event balance path is formally retired

## Evidence Required Before Real-Money Sign-Off

The plan should not be considered complete until the repository can show:

1. append-only evidence for every sync run, fill ingest, and exchange
   observation
2. explicit reconciliation grading visible to operators and machine consumers
3. daily reconciliation reports covering:
   - fill completeness
   - position agreement
   - funding agreement
   - cash-event agreement
   - unexplained equity delta
4. alerts for stale observations, unsupported fills, identity mismatch, and
   degraded fallback activation
5. replayable audit trails sufficient to reconstruct a specific live state at a
   specific time

## Suggested Plan Activation Rule

Move this document from `plans/backlog/` to `plans/active/` only when work is
 formally prioritised and the first PR in the sequence is ready to start.
