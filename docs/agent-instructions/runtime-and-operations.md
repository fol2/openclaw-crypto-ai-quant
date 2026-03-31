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
- `openclaw-ai-quant-assist-v8` (assist mode — signals + exit tunnel, no trading)
- `openclaw-ai-quant-ws-sidecar`

## Assist Mode

`aiq-runtime live assist` runs the full signal generation and exit tunnel
computation pipeline without placing any trades. It uses a read-only
`HyperliquidInfoClient` that requires only a wallet address and structurally
cannot sign exchange actions. Use assist mode to monitor signals and exit
tunnels while the live trader is stopped, or to shadow a running paper/live
lane without execution side-effects.

The assist service conflicts with the live trader by systemd design
(`Conflicts=openclaw-ai-quant-live-v8.service`), so only one can be active at
a time.

The Hub snapshot API exposes `assist_mode: true` when the assist daemon is
running, and the Hub service allow-list includes
`openclaw-ai-quant-assist-v8`.

## Operator Defaults

- Test risky behaviour changes in paper mode first.
- Treat the production profile as the default live profile.
- Use runtime manifests, effective config, and pipeline inspection before making
  assumptions about live state.
- Keep the single factory timer non-racing by schedule design; do not
  reintroduce competing factory schedules that rely on the global factory lock
  as the primary collision-avoidance mechanism.
- Treat deployment-enabled factory modes as fail-closed: if deployment,
  selection, profile, validation, or live-governance settings are missing or
  incomplete, fix the settings contract instead of falling back to permissive
  defaults.
