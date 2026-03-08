# Legacy Runtime Ledger

## Purpose

Track Python runtime surfaces that must be quarantined and later removed as the
Rust runtime replaces them.

## Active Migration Targets

| Surface | Current Owner | Status | Rust Replacement |
|---|---|---|---|
| Python daemon orchestration | `engine/daemon.py`, `engine/core.py` | Active, frozen for `live` / `dry_live`; paper path is legacy recovery/debug only and no longer the authoritative paper service | `aiq-runtime paper daemon`, `aiq-runtime paper manifest`, `aiq-runtime paper status`, `aiq-runtime paper service`, `aiq-runtime paper service apply`, `aiq-runtime live manifest`, `scripts/run_paper_lane.sh` |
| Python effective-config/default mirror | `strategy/mei_alpha_v1.py`, `engine/strategy_manager.py`, `engine/promoted_config.py` | Active, frozen compatibility shim (`engine/promoted_config.py` legacy helpers are no longer the active owner; `StrategyManager` still hot-reloads the resolver-selected YAML path, but Rust now owns effective-config resolution, runtime materialisation, and config identity for paper, dry-live, and live start-up, and the Python daemon no longer reapplies Python defaults when the Rust-owned contract is active) | Rust effective-config resolver (`aiq-runtime paper effective-config`, `paper manifest`, and the shared runtime config contract) |
| Python paper execution | `strategy/mei_alpha_v1.py` | Recovery-only, deletion-ready (Rust paper daemon is authoritative) | Rust paper mode |
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
