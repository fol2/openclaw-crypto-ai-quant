# Rust Runtime Phase 2 Paper Cutover Checklist

**Date:** 2026-03-08  
**Status:** Active execution checklist for `Phase 2` Rust paper cutover

This checklist exists so `Phase 2` is measured by service ownership, paper-gate
evidence, replay-gate evidence, and deletion progress rather than by unlimited
hardening loops.

## 1. Scope

This checklist covers:

1. Rust paper daemon service ownership
2. first-launch bootstrap contract for existing paper DBs
3. paper gate evidence and replay gate evidence
4. rollback contract and Python paper demotion
5. housekeeping for the retired paper path

It does **not** claim live, OMS, or risk cutover.

## 2. Phase 2 Exit Checklist

- [ ] production paper lane runs on `aiq-runtime paper daemon`
- [ ] Python paper execution is recovery-only, not authoritative
- [ ] first-launch bootstrap contract is recorded:
  - `AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS`, or
  - `AI_QUANT_PAPER_BOOTSTRAP_FROM_LATEST_COMMON_CLOSE=1`
- [ ] replay alignment gate is green and release blocker status is clear by default
- [ ] paper deletion tranche is merged
- [ ] runtime ledger and runbook reflect Rust paper ownership

## 3. Cutover Steps

- [ ] capture `paper effective-config --json`
- [ ] capture `paper manifest --json`
- [ ] capture `paper status --json`
- [ ] capture `paper service --json`
- [ ] install the Rust paper systemd unit from `systemd/openclaw-ai-quant-trader-v8-paper*.service.example`
- [ ] set the first-launch bootstrap contract for the lane
- [ ] restart the lane under systemd
- [ ] verify `paper status --json` reports `service_state=running`
- [ ] verify `paper service --json` reports `desired_action=monitor`
- [ ] verify `systemctl --user status <lane>` shows `aiq-runtime` / `scripts/run_paper_lane.sh`

## 4. Paper Gate Evidence

- [ ] at least 1 full trading day or 20 trades
- [ ] profit factor >= 1.2
- [ ] max drawdown < 10%
- [ ] net positive under 20 bps slippage stress
- [ ] zero kill-switch triggers

## 5. Replay Gate Evidence

- [ ] release blocker file exists
- [ ] release blocker status is green / clear by default
- [ ] scheduled gate snapshots an isolated paper DB for the replay bundle; it must not mutate the active paper lane DB
- [ ] scheduled gate enforces the Phase 2 paper-cutover axes (`state`, `live_paper`, `live_paper_decision_trace`, `event_order`) while keeping backtester trade/action and GPU parity reports as diagnostics for later phases
- [ ] emergency override path is documented separately from the default path

## 6. Rollback Contract

- [ ] stop command recorded
- [ ] previous Python recovery path recorded
- [ ] flatten command recorded if required
- [ ] approver / owner recorded

Minimum rollback note template:

```text
Lane:
Rust unit:
Rust status command:
Stop command:
Flatten command (if required):
Legacy Python recovery command:
Approver:
Rollback trigger:
```

## 7. Housekeeping

- [ ] `docs/housekeeping/legacy-runtime-ledger.md` updated
- [ ] `docs/runbook.md` updated
- [ ] `docs/ARCHITECTURE.md` updated
- [ ] any remaining Python paper references are marked recovery-only or deletion-ready
