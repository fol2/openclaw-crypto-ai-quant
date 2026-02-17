# Live-Canonical Alignment Master Plan

## 1. Ultimate Objective

Build financial-grade deterministic alignment across `live`, `paper`, and `backtester`, with `live` as the canonical behaviour.

Given the same state and market inputs, `paper` and `backtester` must reproduce what `live` did, except for explicitly tagged non-simulatable OMS/exchange effects.

## 2. Non-Negotiable Principles

- `live` is the source of truth for behavioural parity.
- Every decision and action must be auditable and traceable.
- Unknown drift is unacceptable; every mismatch must be classified.
- Numeric tolerance must be as strict as practical, defaulting to the tightest safe threshold.
- All changes are delivered via atomic PRs into `master`.

## 3. Canonical Data Strategy (Best Available Source)

Use a layered canonical source model:

1. Exchange-confirmed outcomes captured by the live execution path (fills, closures, realised PnL, balance effects).
2. Live OMS intent and decision trail recorded locally.
3. Market data stream used by live at decision time.

If any canonical field is missing today, add explicit capture so replay is complete and deterministic.

## 4. Foundation First: State Sync (Part 1)

Before further parity hardening, implement state synchronisation from `live` to `paper` and `backtester`.

### 4.1 Snapshot Contract

Define a versioned snapshot schema including at minimum:

- account balances and equity components
- per-symbol positions (size, side, entry basis, realised/unrealised context)
- open orders and protective order state
- cooldown/re-entry state
- strategy runtime gates needed for deterministic continuation
- risk and kill-switch flags
- watermark/timestamps needed for deterministic resume

### 4.2 Export/Import Pipeline

- `live`: export canonical snapshot + event cursor.
- `paper`: import snapshot as runtime seed.
- `backtester`: import snapshot as initial engine state.

### 4.3 Replay Harness

Build replay capability:

- Input: snapshot + bounded historical market data window.
- Output: canonical ledger and decision trace.
- Goal: reproduce recent `live` trades under equivalent state.

## 5. Parity Hardening Sequence (After State Sync)

1. Decision parity: gates, thresholds, ranking, reversal, ATR floor, regime filters.
2. Event parity: entry/add/partial/close event ordering and semantics.
3. Numeric parity: realised PnL, fees, balance deltas, rounding policy.
4. GPU parity against CPU SSOT: event-by-event and numeric convergence.

## 6. Mismatch Taxonomy

Every mismatch must be tagged as one of:

- deterministic logic divergence (bug)
- numeric policy divergence (rounding/precision)
- state initialisation gap
- market data alignment gap
- non-simulatable exchange/OMS effect (explicitly accepted)

Only the last category may remain as an accepted residual.

## 7. Acceptance Criteria (Financial Grade)

- `paper` and `backtester` can replay recent `live` trades from synced state.
- All decisions have reason codes and trace artifacts.
- Event parity is exact for simulatable paths.
- Numeric parity meets strict thresholds; exceptions are documented and approved.
- Residual mismatches are fully classified and auditable.

## 8. Delivery Protocol (Atomic)

For each successful change set:

1. Open an atomic PR to `master`.
2. Run reviewer subagent review.
3. Merge only if review is acceptable.
4. Clean up only the current agent/session branch and worktree (local + remote).
5. Continue to the next axis.

## 9. Immediate Next Execution Steps

1. Complete `CLOSE_PNL` numeric parity in current branch (`pr-444-close-pnl-numeric-parity`).
2. Land via atomic PR workflow.
3. Start State Sync foundation PR(s) (snapshot contract, export/import, replay harness).
4. Resume axis-by-axis parity hardening on top of synced-state baseline.
