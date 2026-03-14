# Success Metrics & Guardrails

Operational metrics and risk limits that define "working" for the active strategy
lifecycle.

These thresholds define the Rust-owned candidate → paper → live promotion path
used by the factory cycle and its operator runbooks.

## Risk Limits

| Metric | Threshold | Action |
|--------|-----------|--------|
| Daily loss limit | -10% of equity (realized PnL, resets UTC 00:00) | Stop all new entries until next UTC day |
| Max equity drawdown | -20% from high water mark (includes unrealized) | Flatten all positions, enter safe mode |
| Max leverage exposure | 5x | Reject entries that would exceed cap |

## Minimum Trade Count Thresholds

| Context | Minimum Trades | Notes |
|---------|---------------|-------|
| Config evaluation (backtest) | 30 | Below this, statistical significance is insufficient to judge a config |
| Rolling live performance | 30 | PF, win rate, Sharpe computed over a sliding window of the last 30 trades |

## Candidate Validation Governance

Factory candidate validation is now split into explicit train and holdout
windows.

- Sweep / TPE search runs on the train window only.
- Promotion gating runs on the trailing holdout window only.
- GPU/CPU parity is rechecked on the train window with a dedicated CPU replay so
  parity does not compare train metrics against holdout metrics.

The validation surface in `config/factory_defaults.yaml` is:

- `validation.holdout_fraction` for the trailing holdout share of common DB coverage
- `validation.holdout_splits` for the number of equal holdout slices summarised in the artefacts

The financial-grade defaults keep the trailing 25% of common coverage as the
holdout window and summarise it in 3 equal slices.

Holdout promotion into `validated` still requires:

- Median holdout daily return > 0
- Slippage stress at 20 bps > 0 on the holdout window
- Minimum holdout trades >= 30
- Holdout top-1 PnL share < 50%

Artefacts now record the resolved `coverage`, `train`, and `holdout` windows in
`run_metadata.json`, candidate validation items, and incumbent/challenger
performance summaries.

## Role-Governed Paper Replacement

Before a validated challenger can replace a deployed paper target, it must win
the deterministic comparator for that role and clear the configured materiality
floor in `config/factory_defaults.yaml`.

| Role | Preferred shortlist | Comparator order | Default materiality floor |
|------|---------------------|------------------|---------------------------|
| `primary` | `efficient` | Higher total PnL, then higher profit factor, then lower drawdown | `min_total_pnl_uplift: 50.0`, `min_profit_factor_uplift: 0.0`, `max_drawdown_slack: 0.50` |
| `fallback` | `growth` | Higher profit factor, then higher total PnL, then lower drawdown | `min_total_pnl_uplift: 0.0`, `min_profit_factor_uplift: 0.05`, `max_drawdown_slack: 0.50` |
| `conservative` | `conservative` | Lower drawdown, then higher profit factor, then higher total PnL | `min_total_pnl_uplift: 0.0`, `min_profit_factor_uplift: 0.0`, `max_drawdown_slack: 0.25` |

If the challenger loses either gate, the factory records `incumbent_holds` and
leaves the current paper target in place. The `primary` lane may still advance
on its own when `fallback` or `conservative` have no deployable replacement; in
that case the report surfaces `selected_partial` / `paper_partial` rather than
claiming a full rollout.

## Promotion Criteria (Paper to Live)

### Gate 1: Paper Minimum Run

- Run for at least **1 full trading day** or **20 trades**, whichever comes first.
- Log fills, slippage, funding, latency, rejection rate.

### Gate 2: Paper Pass Criteria

All must be met:

- Profit factor >= 1.2
- Max drawdown < 10%
- Slippage stress test at 20 bps still net positive
- No kill-switch triggers during paper run

### Gate 3: Live Size Ramp

| Stage | Size | Minimum Duration | Advance Condition |
|-------|------|-----------------|-------------------|
| 1 | 25% of target | 1 day | No kill-switch triggers |
| 2 | 50% of target | 1 day | No kill-switch triggers |
| 3 | 100% of target | — | Steady state |

The factory enforces the ramp by generating stage-specific live manifests rather
than editing the steady-state config in place.

Step down or pause immediately if any kill-switch fires during ramp.

## Rotation Criteria (When to Retire a Config)

Trigger rotation if **any** condition is met:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Rolling profit factor | PF < 1.0 over last 30 trades | Pause config, fall back to safety mode |
| Live drawdown warning | DD > 15% from config-start HWM | Pause config, queue a replacement from the latest validation run |
| Max config age | 14 days since deployment | Pause config, queue a replacement from the latest validation run |

When a config is rotated out, record: config hash, start/stop timestamps, reason for rotation, and final performance summary.

## References

- [Strategy Lifecycle](strategy_lifecycle.md) — state machine and transition triggers
- [Operations Runbook](runbook.md) — emergency stop and kill-switch procedures
