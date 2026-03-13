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
