# Rust Runtime Final Cleanup Plan

**Date:** 2026-03-13  
**Status:** Active final cleanup plan after the Rust paper and live cutovers  
**Scope:** Remove the remaining Python runtime-era ownership surfaces without breaking operator tooling, config helpers, or rollback workflows

## 1. Decision

**Production-runtime decision:** Rust is the authoritative runtime owner for
paper and live today.

**Deletion decision:** do **not** delete the remaining Python tree in one shot
yet.

The remaining Python footprint is no longer a production runtime dependency, but
it is still used by:

1. operator and emergency tools
2. config/default helper imports
3. archival recovery/debug workflows
4. parity- and helper-oriented tests

The correct next step is a bounded cleanup programme, not a blanket deletion.

## 2. Current Blockers

### 2.1 Operator tooling still imports Python execution helpers

These tools still depend on `exchange/executor.py` and related Python helpers:

- `tools/flat_now.py`
- `tools/export_state.py`
- `tools/deploy_sweep.py`
- `tools/manual_trade.py`

These are the highest-risk blockers because they cover flattening, live-state
inspection, deployment workflows, and manual intervention.

### 2.2 Config and defaults still depend on `strategy/mei_alpha_v1.py`

These paths still read defaults or fallback symbols from the legacy Python
strategy module:

- `engine/strategy_manager.py`
- `tools/validate_config.py`

The problem is not the old `PaperTrader` runtime itself; it is that strategy
defaults, fallback symbols, and helper builders still live beside it.

### 2.3 Some legacy Python modules are now compatibility libraries, not deletion-ready

These modules are no longer production owners, but they still serve tests or
tooling:

- `engine/core.py`
- `engine/oms.py`
- `engine/risk.py`
- `exchange/executor.py`
- `live/trader.py`
- `strategy/mei_alpha_v1.py`

## 3. Cleanup Tranches

### Tranche 0. Runtime Entry Retirement

**Status:** Completed on the current branch.

Delivered:

1. `engine/daemon.py` no longer acts as a production runtime owner for any mode
2. all Python daemon modes now require `AI_QUANT_ALLOW_LEGACY_PYTHON_RUNTIME=1`
3. `live/trader.py` standalone entry messaging now points to Rust-only runtime paths
4. runbooks, architecture docs, and issue templates reflect Rust-owned production paths

Exit gate:

1. no Python entrypoint can be launched accidentally as a production runtime path

### Tranche 1. Operator Tool Migration Off `exchange/executor.py`

**Goal:** make all operationally important tools work without the Python live
exchange adapter.

Files to migrate:

1. `tools/flat_now.py`
2. `tools/export_state.py`
3. `tools/deploy_sweep.py`
4. `tools/manual_trade.py`

Required replacement surfaces:

1. a Rust live-state export path for operator snapshots
2. a Rust flatten / close-live-positions path or a Rust-owned wrapper that can be called from the existing scripts
3. a Rust manual-order preview / submit surface, or an explicit decision to move manual trade support into the Rust runtime CLI
4. deployment tooling that no longer imports `HyperliquidLiveExecutor`

Implementation rule:

1. do not delete `exchange/executor.py` until the tools above stop importing it

Exit gates:

1. no production-facing operator tool imports `exchange/executor.py`
2. emergency flatten workflow is validated through the Rust-owned path
3. live-state export no longer requires the Python executor

### Tranche 2. Extract Strategy Defaults And Helper Surface From `mei_alpha_v1.py`

**Goal:** separate legacy trader classes from the still-needed strategy helpers.

Files to split:

1. `strategy/mei_alpha_v1.py`
2. `engine/strategy_manager.py`
3. `tools/validate_config.py`

Target extraction:

1. move `_DEFAULT_STRATEGY_CONFIG` into a dedicated helper module
2. move `_FALLBACK_SYMBOLS` into a dedicated helper module
3. move any still-needed pure helper functions out of the `PaperTrader`-heavy module
4. keep `PaperTrader` clearly isolated as a legacy recovery-only class until it is removed

Implementation rule:

1. helper extraction must be behaviour-preserving before any deletion PR removes `PaperTrader`

Exit gates:

1. `engine/strategy_manager.py` no longer imports `strategy.mei_alpha_v1`
2. `tools/validate_config.py` no longer imports `strategy.mei_alpha_v1`
3. legacy `PaperTrader` can be reasoned about as a self-contained archival surface

### Tranche 3. Retire Legacy Python Execution Classes

**Goal:** remove the remaining Python execution-loop ownership surfaces once
their helper dependencies are extracted.

Primary deletion targets:

1. `engine/core.py` runtime loop ownership (`UnifiedEngine`)
2. `live/trader.py` runtime class ownership (`LiveTrader`)
3. `strategy/mei_alpha_v1.py` runtime class ownership (`PaperTrader`)
4. runtime-specific portions of `engine/oms.py`

Preconditions:

1. Tranche 1 is complete
2. Tranche 2 is complete
3. any still-needed helper types or pure functions have been extracted first

Implementation rule:

1. if a helper is still needed by tests or tooling, move it before deleting the legacy runtime class that currently hosts it

Exit gates:

1. no production or operator path imports `UnifiedEngine`, `LiveTrader`, or `PaperTrader`
2. legacy runtime classes are either deleted or moved behind a clearly isolated archival package
3. runtime-oriented tests for deleted classes are removed or replaced with Rust-runtime contract tests

### Tranche 4. Replace Frozen Python Parity Orchestration And Retire `bt-runtime`

**Goal:** finish the runtime-language retirement by removing transitional Python
parity ownership and the PyO3 bridge once nothing runtime-facing needs them.

Primary targets:

1. Python replay/parity orchestration under `tools/*parity*` and `tools/*replay*`
2. `backtester/crates/bt-runtime`

Preconditions:

1. Rust-native parity/debug workflow is sufficient for day-to-day investigations
2. no remaining runtime compatibility or recovery path depends on the PyO3 bridge

Exit gates:

1. no authoritative parity workflow depends on Python
2. `bt-runtime` is removed, or retained only with an explicit non-runtime justification

## 4. Recommended PR Slices

Use atomic PRs in this order:

1. Tranche 1a: Rust live-state export + tool migration
2. Tranche 1b: Rust flatten/manual intervention path + tool migration
3. Tranche 2a: extract strategy defaults/fallback symbols
4. Tranche 2b: extract pure strategy helper builders
5. Tranche 3a: delete/archive `engine/daemon.py` runtime path after helper fallout is closed
6. Tranche 3b: delete/archive `live/trader.py` runtime class
7. Tranche 3c: delete/archive `PaperTrader` runtime ownership and Python loop ownership
8. Tranche 4: parity/bridge retirement

## 5. Verification Checklist

Every cleanup tranche must prove all of the following before merge:

1. `rg` shows no remaining imports from the deleted owner surface in production or operator paths
2. targeted tests pass without relying on the retired Python runtime path
3. runbook commands point to Rust-owned surfaces only
4. rollback and emergency workflows remain documented and executable
5. the legacy runtime ledger is updated in the same PR

## 6. Definition Of Done

The final cleanup programme is complete only when all of the following are true:

1. no production or operator path requires `engine/daemon.py`
2. no production or operator path requires `exchange/executor.py`
3. no config/default helper path requires `strategy/mei_alpha_v1.py`
4. no authoritative parity workflow depends on Python
5. `bt-runtime` is removed or explicitly justified as non-runtime-only
6. the remaining Python footprint is limited to non-runtime domains such as monitoring, reporting, or one-off utilities

## 7. Related Documents

- `docs/current_authoritative_paths.md`
- `docs/housekeeping/legacy-runtime-ledger.md`
- `docs/rust_full_runtime_cutover_programme.md`
- `runtime/README.md`
