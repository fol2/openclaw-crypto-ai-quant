# Legacy Runtime Ledger

## Purpose

Track Python runtime surfaces that must be quarantined and later removed as the
Rust runtime replaces them.

Production ownership has already moved to Rust. This ledger now tracks the
remaining compatibility and deletion backlog only. See
`docs/current_authoritative_paths.md` for the current production ownership map
and `docs/programmes/rust_runtime_final_cleanup_plan.md` for the active final
cleanup sequence.

## Compatibility And Deletion Targets

| Surface | Current Owner | Status | Rust Replacement |
|---|---|---|---|
| Python daemon orchestration | `engine/daemon.py`, `engine/core.py` | Retired, non-authoritative. `engine/daemon.py` is now a fail-fast shim; production service paths are Rust-only | `aiq-runtime paper daemon`, `aiq-runtime paper manifest`, `aiq-runtime paper status`, `aiq-runtime paper service`, `aiq-runtime paper service apply`, `aiq-runtime live manifest`, `aiq-runtime live status`, `aiq-runtime live service`, `aiq-runtime live service apply`, `aiq-runtime live daemon`, `scripts/run_paper_lane.sh`, `scripts/run_live.sh` |
| Python effective-config/default mirror | `strategy/defaults.py`, `strategy/mei_alpha_v1.py`, `engine/strategy_manager.py`, `engine/promoted_config.py` | Active, frozen compatibility shim. Defaults/fallback symbols are now extracted into `strategy/defaults.py`; `engine/promoted_config.py` legacy helpers are no longer the active owner, and Rust owns effective-config resolution, runtime materialisation, and config identity for paper, dry-live, and live start-up | Rust effective-config resolver (`aiq-runtime paper effective-config`, `paper manifest`, and the shared runtime config contract) |
| Python paper execution | `strategy/mei_alpha_v1.py` (`PaperTrader`) | Recovery-only, deletion-ready (Rust paper daemon is authoritative) | Rust paper mode |
| Python live execution | `live/trader.py` | Archival recovery/debug only; no longer the authoritative production path | Rust live adapter |
| Python execution helpers still used by tools/tests | `engine/core.py`, `engine/oms.py`, `engine/risk.py`, `exchange/executor.py`, `strategy/mei_alpha_v1.py` helpers | Non-runtime compatibility/helpers. Operator tooling has been migrated off `exchange/executor.py`, so the remaining dependency is tests/compatibility only; keep the rest only until helper extraction or deletion tranches are finished | Rust runtime for production ownership; follow-on cleanup tranches for helper retirement |
| Python parity orchestration | `tools/*parity*`, `tools/*replay*` | Active, frozen, non-authoritative | Rust parity harness |
| Python paper seed tool | `tools/apply_canonical_snapshot_to_paper.py` | Legacy, frozen compatibility bridge | `aiq-runtime snapshot seed-paper` |
| PyO3 runtime bridge | `backtester/crates/bt-runtime` | Transitional, non-production | Remove after Python compatibility surfaces no longer depend on it |

## Deletion Gate

A legacy surface may be removed only when all of the following are true:

1. a Rust replacement exists
2. parity and acceptance checks are green for the affected mode
3. docs and runbooks point to the Rust path
4. the owning PR explicitly records the deletion in this ledger
