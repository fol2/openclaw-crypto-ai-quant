# Rust Runtime Phase 0 Cutover and Burn-In Checklist

**Date:** 2026-03-07  
**Status:** Active execution checklist for `Phase 0` close-out and the first Rust paper burn-in lane

This checklist exists so `Phase 0` is not only a programme statement. Every PR
or operator action that claims `Phase 0` completion should point back to this
artefact and attach concrete evidence.

## 1. Scope

This checklist covers:

1. runtime/control-plane landing on `master`
2. repo-level PR traceability for the migration programme
3. the first Rust paper burn-in lane wiring that later `Phase 2` cutover work
   will reuse

It does **not** claim paper or live cutover by itself.

## 2. Phase 0 Exit Checklist

Mark every line with evidence before calling `Phase 0` complete.

- [ ] `runtime/` exists on `master`
- [ ] runtime ledger exists on `master`
- [ ] PR gate enforces `phase`, `checklist item`, `exit criterion`, and
  `deletion target`
- [ ] the Rust runtime foundation docs are linked from the architecture /
  programme surfaces
- [ ] the first burn-in / rollback checklist exists as a standalone repo
  artefact

## 3. Burn-In Lane Preparation

Before starting a Rust paper burn-in lane:

- [ ] lane selected: `paper1` / `paper2` / `paper3` / `livepaper`
- [ ] runtime profile selected: `production` / `parity_baseline` /
  `stage_debug` / custom profile
- [ ] `paper manifest --json` captured
- [ ] `paper status --json` captured
- [ ] `paper service --json` captured
- [ ] replay alignment blocker verified green
- [ ] rollback owner named
- [ ] rollback command path written down before start

## 4. Burn-In Evidence Pack

Attach or link the following artefacts:

- [ ] `paper effective-config --json`
- [ ] `paper manifest --json`
- [ ] `paper status --json`
- [ ] `paper service --json`
- [ ] paper gate evidence:
  - at least 1 full trading day or 20 trades
  - profit factor >= 1.2
  - max drawdown < 10%
  - net positive under 20 bps slippage stress
  - zero kill-switch triggers
- [ ] replay gate evidence:
  - release blocker clear by default
  - override path documented only for emergencies
- [ ] operator note on whether the lane is:
  - burn-in only
  - cutover-ready
  - rollback-required

## 5. Rollback Contract

The rollback path must be explicit before the lane starts.

- [ ] pause or stop command identified
- [ ] flatten procedure identified if required
- [ ] previous Python-owned fallback path identified
- [ ] owner and approval path identified

Minimum rollback note template:

```text
Lane:
Rust service / manifest:
Fallback Python service:
Pause / stop command:
Flatten command (if required):
Approver:
Reason threshold for rollback:
```

## 6. PR Mapping Rules

For every PR in the Rust full-runtime cutover programme:

1. fill in the PR template `Programme Mapping` section
2. map the PR to one checklist item in the active phase
3. name the deletion target, even if the answer is `n/a`
4. attach reviewer-subagent review evidence before merge

## 7. Phase 0 Decision Rule

`Phase 0` is complete only when:

1. all exit lines in section 2 are checked
2. this checklist has at least one concrete burn-in evidence pack attached or
   referenced
3. the next PR can clearly declare itself a `Phase 1` PR in the template
