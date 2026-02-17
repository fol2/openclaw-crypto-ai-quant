# Live vs Paper Action Reconciliation

## Objective

Audit action-level parity between live and paper SQLite trade logs over a selected time window.

Compared canonical actions:

- `OPEN_LONG`, `OPEN_SHORT`
- `ADD_LONG`, `ADD_SHORT`
- `REDUCE_LONG`, `REDUCE_SHORT`
- `CLOSE_LONG`, `CLOSE_SHORT`
- `FUNDING` (tracked as accepted residual when one side is missing)

## Command

```bash
python tools/audit_live_paper_action_reconcile.py \
  --live-db ./trading_engine_live.db \
  --paper-db ./trading_engine.db \
  --from-ts 1770700000000 \
  --to-ts 1771200000000 \
  --timestamp-bucket-ms 1 \
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

## Pass Criteria

`status.strict_alignment_pass = true` only when there are no:

- deterministic logic divergences
- numeric policy divergences

Funding-only one-sided events are captured under `accepted_residuals`.
