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

### Transactional Live Apply and Rollback

The Hub live control-plane now treats live apply and live rollback as
transactional operations. It stages the candidate YAML, snapshots the incumbent
live YAML before mutation, and reports success only after
`aiq-runtime live service apply --json` proves that the final live daemon is:

- running
- healthy
- matched to the current launch contract
- on the intended materialised `config_id`

If the supervised apply proof fails, the Hub restores the incumbent live YAML
and runs a recovery restart against that incumbent contract before returning the
failure response. The emitted artefacts now record both the previous and target
`config_id` values.

Transactional live apply artefacts live under:

```bash
artifacts/applies/live/<timestamp>/
```

Transactional live rollback artefacts live under:

```bash
artifacts/rollbacks/live/<timestamp>/
```

Each directory contains the incumbent snapshot, the staged candidate or
restored payload, and an `event.json` audit record with runtime apply / recovery
proof details.

Use `AI_QUANT_RUNTIME_BIN` when the Hub must call a non-default `aiq-runtime`
binary for live apply and rollback verification subprocesses. When unset, the
Hub falls back to `aiq-runtime` on `PATH`.

Example live apply request:

```bash
curl -sS \
  -H "Authorization: Bearer $AIQ_MONITOR_ADMIN_TOKEN" \
  -H "If-Match: $LIVE_LOCK_ID" \
  -H "Content-Type: application/json" \
  -X POST \
  http://127.0.0.1:8000/api/config/actions/apply-live \
  --data @/tmp/apply-live.json
```

The JSON payload should contain `yaml`, an optional `reason`, and
`restart: "auto"` or `restart: "always"`. Non-dry-run live apply no longer
accepts `restart: "never"` because success now requires runtime proof.

Example live rollback request:

```bash
curl -sS \
  -H "Authorization: Bearer $AIQ_MONITOR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -X POST \
  http://127.0.0.1:8000/api/config/actions/rollback-live \
  --data '{"steps":1,"reason":"operator rollback","restart":"auto"}'
```

Read the current `LIVE_LOCK_ID` from `GET /api/config/raw?file=live` before an
apply request. The response exposes the raw-text lock boundary through
`x-aiq-config-lock-id` and `ETag`.

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

## Factory

Build the Hub with the `factory` feature and install the Rust executor before
enabling policy:

```bash
cargo build --release --manifest-path hub/Cargo.toml --features factory
cargo build --release --manifest-path runtime/aiq-runtime/Cargo.toml --bin aiq-factory
AI_QUANT_FACTORY_ENABLE=1 cargo run --release --manifest-path runtime/aiq-runtime/Cargo.toml --bin aiq-factory -- run --config config/strategy_overrides.yaml --settings config/factory_defaults.yaml --profile daily --json
```

`config/factory_defaults.yaml` now enforces parity fail-closed by default. Live
promotion remains an explicit deployment setting: enable
`deployment.apply_to_live: true` only on the production instance that is meant
to restart `openclaw-ai-quant-live-v8`.

The financial-grade factory default now seeds sweep/replay balance from current
live equity (including unrealised PnL) and compares each challenger against the
currently deployed target config before any paper replacement is applied. Use
`balance.mode: fixed` only for controlled research runs.

Factory validation now splits the common DB coverage into an explicit train
window plus a trailing holdout window. Tune `validation.holdout_fraction` and
`validation.holdout_splits` in `config/factory_defaults.yaml` when operators
need a different holdout share or slice count. Sweep / TPE search and the
dedicated CPU parity replay run on the train window, while candidate gating and
incumbent/challenger comparison use the holdout window only.
The financial-grade defaults reserve the trailing 25% of common coverage as the
holdout window and summarise it in 3 equal holdout slices.

Inspect `artifacts/.../run_metadata.json` and the candidate rows in
`reports/report.json` for the resolved `coverage`, `train`, and `holdout`
boundaries. Candidate rows now expose `train_parity_replay_report_path`,
`holdout_summary_path`, and `holdout_median_daily_return` so operators can
audit exactly which window produced each gate decision.

Paper selection is now deterministic per role. `primary` prefers `efficient`
artefacts ranked by total PnL, then profit factor, then lower drawdown;
`fallback` prefers `growth` artefacts ranked by profit factor, then total PnL,
then lower drawdown; `conservative` prefers `conservative` artefacts ranked by
lower drawdown, then profit factor, then total PnL. Rank and config ID act as
stable tie-breakers so repeated runs on the same artefacts keep the same role
ordering.

Challengers must also clear the role-specific materiality floor in
`config/factory_defaults.yaml` before they can replace an incumbent. The
financial-grade defaults are `primary` `+50.0` total PnL uplift with at most
`0.50` drawdown slack, `fallback` `+0.05` profit-factor uplift with at most
`0.50` drawdown slack, and `conservative` at most `0.25` drawdown slack.

When only the `primary` lane has a deployable challenger, the factory now
allows a truthful partial rollout instead of blocking the whole cycle. Inspect
`reports/selection.json` for `selection_stage: selected_partial`,
`deploy_stage: paper_partial`, `step5_gate_status: partial`, and per-role
deployment statuses such as `incumbent_holds`.

The tracked service examples live under:

```bash
systemd/openclaw-ai-quant-factory-v8.service.example
systemd/openclaw-ai-quant-factory-v8.timer.example
systemd/openclaw-ai-quant-factory-v8-deep.service.example
systemd/openclaw-ai-quant-factory-v8-deep.timer.example
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
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_exit_isolation --json
```

Look for `behaviours.gates`, `behaviours.signal_modes`, `behaviours.exits`,
`behaviours.entry_sizing`, `behaviours.entry_progression`, and `behaviours.risk`
when validating a parity lane.

Use `parity_baseline` when you need production-like behaviour ordering with no
broker execution, and `parity_exit_isolation` when you want to focus on base
stop-loss, trailing, and full take-profit behaviour without modifier noise.

For exit-path debugging, confirm both the resolved `behaviours.exits` order and
the emitted `behaviour_trace` in the paper/live report. That trace now tells you
which stop-loss, trailing, take-profit, or smart-exit behaviour executed,
skipped, or was disabled for the bar.
