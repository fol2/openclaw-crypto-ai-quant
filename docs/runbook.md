# Runbook

## Paper

```bash
cargo run -p aiq-runtime -- paper manifest --lane paper1 --project-dir "$PWD" --json
cargo run -p aiq-runtime -- paper daemon --lane paper1 --project-dir "$PWD"
```

## Live

```bash
cargo run -p aiq-runtime -- live manifest --project-dir "$PWD" --json
cargo run -p aiq-runtime -- live daemon --project-dir "$PWD"
```

## Snapshot Operations

```bash
cargo run -p aiq-runtime -- snapshot export-paper --db trading_engine.db --output /tmp/paper.json
cargo run -p aiq-runtime -- snapshot validate --path /tmp/paper.json --json
cargo run -p aiq-runtime -- snapshot seed-paper --snapshot /tmp/paper.json --target-db trading_engine.db --strict-replace --json
```

## Service Wrappers

```bash
./scripts/run_paper_lane.sh paper1
./scripts/run_live.sh
```

## Diagnostics

```bash
cargo run -p aiq-runtime -- doctor --json
cargo run -p aiq-runtime -- pipeline --json
journalctl --user -u openclaw-ai-quant-live-v8 -f
journalctl --user -u openclaw-ai-quant-trader-v8-paper1 -f
```

## Behaviour Debugging

Use the pipeline surface to confirm both stage and behaviour resolution for the
active profile:

```bash
cargo run -p aiq-runtime -- pipeline --mode paper --json
cargo run -p aiq-runtime -- pipeline --mode live --profile parity_baseline --json
```

Look for `behaviours.gates`, `behaviours.signal_modes`, `behaviours.exits`,
`behaviours.entry_sizing`, `behaviours.entry_progression`, and `behaviours.risk`
when validating a parity lane.

For exit-path debugging, confirm both the resolved `behaviours.exits` order and
the emitted `behaviour_trace` in the paper/live report. That trace now tells you
which stop-loss, trailing, take-profit, or smart-exit behaviour executed,
skipped, or was disabled for the bar.
