# Current Authoritative Paths

## Runtime Ownership

| Surface | Authoritative path |
|---|---|
| Paper runtime | `runtime/aiq-runtime` (`paper ...`) |
| Live runtime | `runtime/aiq-runtime` (`live ...`) |
| Snapshot export/seed | `runtime/aiq-runtime` (`snapshot ...`) |
| Paper lane wrappers | `scripts/run_paper.sh`, `scripts/run_paper_lane.sh` |
| Live wrapper | `scripts/run_live.sh` |

## Backtesting Ownership

| Surface | Authoritative path |
|---|---|
| Replay | `backtester/crates/bt-cli` |
| Sweep | `backtester/crates/bt-cli` |
| GPU sweep | `backtester/crates/bt-gpu` |
| Indicator parity | `backtester/crates/bt-cli dump-indicators` |

## Operations Ownership

| Surface | Authoritative path |
|---|---|
| Market-data ingestion | `ws_sidecar/` |
| Dashboard and service inspection | `hub/` |
| Release version governance | `VERSION` + Cargo manifests + `tools/release/*.sh` |

## Removed Ownership

Legacy alternate-language runtime, tool, and test surfaces have been removed
from the repository and are no longer part of the active trust chain.
