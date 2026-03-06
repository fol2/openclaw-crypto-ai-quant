# Legacy Runtime Ledger

## Purpose

Track Python runtime surfaces that must be quarantined and later removed as the
Rust runtime replaces them.

## Active Migration Targets

| Surface | Current Owner | Status | Rust Replacement |
|---|---|---|---|
| Python daemon orchestration | `engine/daemon.py`, `engine/core.py` | Active, frozen (paired opt-in Rust paper daemon owns the outer scheduler/watchlist reload path, and a read-only Rust service manifest exists, but no cutover) | `aiq-runtime paper daemon`, `aiq-runtime paper manifest` |
| Python paper execution | `strategy/mei_alpha_v1.py` | Active, frozen (Rust bootstrap + run-once + cycle + loop shells plus paired daemon wrapper) | Rust paper mode |
| Python live execution | `live/trader.py`, `exchange/executor.py` | Active, frozen | Rust live adapter |
| Python OMS / risk runtime | `engine/oms.py`, `engine/risk.py` | Active, frozen | Rust persistence + risk layers |
| Python parity orchestration | `tools/*parity*`, `tools/*replay*` | Active, frozen | Rust parity harness |
| Python paper seed tool | `tools/apply_canonical_snapshot_to_paper.py` | Legacy, frozen compatibility bridge | `aiq-runtime snapshot seed-paper` |
| PyO3 runtime bridge | `backtester/crates/bt-runtime` | Transitional | Remove after Python retirement |

## Deletion Gate

A legacy surface may be removed only when all of the following are true:

1. a Rust replacement exists
2. parity and acceptance checks are green for the affected mode
3. docs and runbooks point to the Rust path
4. the owning PR explicitly records the deletion in this ledger
