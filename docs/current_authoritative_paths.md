# Current Authoritative Paths

**Date:** 2026-03-13  
**Status:** Current production ownership map after the Rust paper and live cutovers

This index is the quickest way to answer “which path is authoritative now?”.
It is intentionally short and should stay aligned with the runbook, runtime
README, and legacy runtime ledger.

## Production Runtime

| Concern | Authoritative path | Notes |
|---|---|---|
| Paper service ownership | `systemd/openclaw-ai-quant-trader-v8-paper*.service.example` → `scripts/run_paper_lane.sh` → `aiq-runtime paper daemon` | Production paper lanes run through the Rust daemon |
| Live service ownership | `systemd/openclaw-ai-quant-live-v8.service.example` → `scripts/run_live.sh` → `aiq-runtime live daemon` | Production live lane runs through the Rust daemon |
| Effective-config resolution | `aiq-runtime paper effective-config` and `aiq-runtime live effective-config` | Rust owns config identity, promoted-role resolution, and strategy-mode materialisation |
| Paper supervision | `paper manifest`, `paper status`, `paper service`, `paper service apply` | Read-only planning plus opt-in supervision for Rust paper lanes |
| Live supervision | `live manifest`, `live status`, `live service`, `live service apply` | Read-only planning plus opt-in supervision for the Rust live lane |
| Paper execution loop | `runtime/aiq-runtime/` | `paper doctor`, `paper run-once`, `paper cycle`, `paper loop`, and `paper daemon` are the Rust paper surfaces |
| Live execution, OMS, and risk | `runtime/aiq-runtime/` | Rust live daemon owns exchange-facing state sync, OMS transitions, risk enforcement, broker submission, fill backfill, and lifecycle status publication |
| Market data sidecar | `ws_sidecar/` | Rust WS sidecar remains the authoritative market-data service path |

## Compatibility And Recovery Paths

| Surface | Status | Notes |
|---|---|---|
| `engine/daemon.py` (`paper`, `dry_live`, `live`) | Archival recovery/debug only | Requires `AI_QUANT_ALLOW_LEGACY_PYTHON_RUNTIME=1` for any mode; `AI_QUANT_ALLOW_LEGACY_PYTHON_LIVE=1` remains accepted only for old live/dry-live recovery workflows |
| `strategy/mei_alpha_v1.py` runtime ownership surfaces | Recovery-only fallback | Strategy logic still exists, but production paper execution is Rust-owned |
| `live/trader.py` and `exchange/executor.py` | Archival recovery/debug only | Retained for guarded fallback and investigation workflows |
| `engine/core.py`, `engine/oms.py`, and `engine/risk.py` | Compatibility helpers, not production owners | Retained for archival tests, helper imports, and non-runtime tooling while the final Python cleanup tranche is still open |
| `engine/promoted_config.py` | Frozen compatibility shim | Shells out to Rust effective-config resolution instead of owning config merges |

## Cleanup Backlog

These items are no longer production-runtime owners, but they still need
housekeeping or deletion follow-through:

1. Retire Python replay/parity orchestration once the Rust-native parity harness fully replaces it.
2. Remove `tools/apply_canonical_snapshot_to_paper.py` after all remaining workflows use `aiq-runtime snapshot seed-paper`.
3. Remove `backtester/crates/bt-runtime` once no runtime compatibility tooling depends on the PyO3 bridge.
4. Delete archival Python recovery surfaces only when rollback policy and acceptance evidence no longer require them.

## Related Documents

- `docs/runbook.md`
- `runtime/README.md`
- `docs/ARCHITECTURE.md`
- `docs/housekeeping/legacy-runtime-ledger.md`
- `docs/rust_full_runtime_cutover_programme.md`
