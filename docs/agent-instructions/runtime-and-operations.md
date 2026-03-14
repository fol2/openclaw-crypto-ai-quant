# Runtime and Operations Instructions

Load this file when the task involves services, manifests, runtime ownership,
operator commands, or production-adjacent runtime behaviour.

## Active Stack

Active execution ownership lives in:

- `runtime/aiq-runtime`
- `runtime/aiq-runtime-core`
- `ws_sidecar/`
- `hub/`

Read [`docs/current_authoritative_paths.md`](../current_authoritative_paths.md)
for the full ownership map.

## First Commands To Reach For

```bash
cargo run -p aiq-runtime -- paper effective-config --lane paper1 --project-dir "$PWD" --json
cargo run -p aiq-runtime -- pipeline --mode paper --json
./scripts/run_paper_lane.sh paper1
./scripts/run_live.sh
```

## Service and Log Reference

Use [`docs/runbook.md`](../runbook.md) for the current operational command set.
The main systemd user services are:

- `openclaw-ai-quant-trader-v8-paper1`
- `openclaw-ai-quant-trader-v8-paper2`
- `openclaw-ai-quant-trader-v8-paper3`
- `openclaw-ai-quant-live-v8`
- `openclaw-ai-quant-ws-sidecar`

## Operator Defaults

- Test risky behaviour changes in paper mode first.
- Treat the production profile as the default live profile.
- Use runtime manifests, effective config, and pipeline inspection before making
  assumptions about live state.
- Keep factory timer cadences non-racing by schedule design; do not rely on the
  global factory lock as the primary collision-avoidance mechanism for
  `daily` vs `deep` runs.
- Treat deployment-enabled factory modes as fail-closed: if deployment,
  selection, profile, validation, or live-governance settings are missing or
  incomplete, fix the settings contract instead of falling back to permissive
  defaults.
