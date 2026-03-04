# Live vs Paper Action Reconciliation

## Objective

Audit action-level parity between live and paper SQLite trade logs over a selected time window.

Compared canonical actions:

- `OPEN_LONG`, `OPEN_SHORT`
- `ADD_LONG`, `ADD_SHORT`
- `REDUCE_LONG`, `REDUCE_SHORT`
- `CLOSE_LONG`, `CLOSE_SHORT`
- `FUNDING` (matched pairs are preserved as non-blocking evidence; one-sided rows remain mismatch evidence)

## Command

```bash
python tools/audit_live_paper_action_reconcile.py \
  --live-db ./trading_engine_live.db \
  --paper-db ./trading_engine.db \
  --from-ts 1770700000000 \
  --to-ts 1771200000000 \
  --timestamp-bucket-ms 1 \
  --price-tol 1e-9 \
  --size-tol 1e-9 \
  --pnl-tol 1e-9 \
  --fee-tol 1e-9 \
  --balance-tol 1e-9 \
  --output /tmp/live_paper_action_reconcile.json
```

To make this command fail on strict mismatch:

```bash
python tools/audit_live_paper_action_reconcile.py \
  --live-db ./trading_engine_live.db \
  --paper-db ./trading_engine.db \
  --from-ts 1770700000000 \
  --to-ts 1771200000000 \
  --fail-on-mismatch \
  --output /tmp/live_paper_action_reconcile.json
```

Legacy diagnostic flag (does not bypass strict failure for coverage gaps):

```bash
python tools/audit_live_paper_action_reconcile.py \
  --live-db ./trading_engine_live.db \
  --paper-db ./trading_engine.db \
  --from-ts 1770700000000 \
  --to-ts 1771200000000 \
  --allow-paper-window-not-replayed \
  --output /tmp/live_paper_action_reconcile.json
```

## Pass Criteria

`status.strict_alignment_pass = true` only when there are no:

- deterministic logic divergences
- numeric policy divergences
- confidence mismatches
- reason-code mismatches
- unmatched non-funding actions on either side

Default behaviour is fail-closed.

When `paper_window_not_replayed` is detected, strict pass is always false
(fail-closed replay-window coverage contract).

All mismatch evidence remains in `mismatches`, while the report exposes:

- `counts.paper_window_not_replayed_artefact_mismatch_total`
- `counts.non_blocking_evidence_total`
- `counts.true_mismatch_total`

`status.accepted_residuals_only = true` means strict alignment passed and only accepted residual classes remain.
