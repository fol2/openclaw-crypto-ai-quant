# Configuration and Debugging Instructions

Load this file when the task involves YAML/runtime config, parity lanes,
behaviour traces, or diagnosis of runtime behaviour.

## Configuration Defaults

- Merge order:
  `Rust defaults <- global YAML <- symbols.<SYM> YAML <- live YAML`.
- `engine.interval` is not hot-reloadable; restart the affected service if it
  changes.
- `entry_min_confidence` defaults to `high`; set it explicitly to `low` in YAML
  if all confidence tiers should pass.
- Keep production on `production` unless the user explicitly wants a debug or
  parity lane.

Use `config/strategy_overrides.yaml.example` as the canonical config example.

## Behaviour and Parity

- Inspect resolved stage and behaviour plans before changing code.
- Use `behaviour_trace` as a first-class diagnostic.
- Use `parity_baseline` and `parity_exit_isolation` before ad hoc debug edits.

For deeper background, load:

- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`docs/runbook.md`](../runbook.md)
- `runtime/aiq-runtime-core/src/pipeline.rs`

## Troubleshooting Order

1. Check the relevant service status and logs.
2. Inspect effective config and pipeline resolution.
3. Read emitted `behaviour_trace`.
4. Treat replay-alignment blockers as fail-closed unless the user explicitly
   approves an emergency bypass.
