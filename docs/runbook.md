# Runbook

## Paper

```bash
cargo run -p aiq-runtime -- paper manifest --lane paper1 --project-dir "$PWD" --json
cargo run -p aiq-runtime -- paper daemon --lane paper1 --project-dir "$PWD"
```

Direct `paper daemon` launches now reconcile `decision_events` against the
canonical Rust contract immediately before the first writable cycle, so idle
watchlist/bootstrap waits remain read-only. `paper service apply` still
preflights the same contract before supervised start/restart. Existing
legacy/full tables are reused in place, while reduced/incompatible tables are
either upgraded transactionally or rejected before the first writable cycle or
supervised start. Operators should treat a `decision_events schema preflight failed` error as a clean
fail-closed deployment signal rather than a daemon crash-loop.

## Live

```bash
cargo run -p aiq-runtime -- live manifest --project-dir "$PWD" --json
cargo run -p aiq-runtime -- live daemon --project-dir "$PWD"
```

## Assist

Assist mode runs signal generation and exit tunnel computation without
placing any trades. It requires only a wallet address (no private key).

```bash
cargo run -p aiq-runtime -- live assist --project-dir "$PWD" --wallet-address 0x...
cargo run -p aiq-runtime -- live assist --project-dir "$PWD" --json
```

The wallet address can also be supplied through `AI_QUANT_WALLET_ADDRESS` or
resolved from the secrets file. The assist daemon writes its status to
`ai_quant_assist.status.json` and the Hub detects it to surface
`assist_mode: true` in the snapshot API.

The `openclaw-ai-quant-assist-v8` systemd service conflicts with
`openclaw-ai-quant-live-v8` so only one can run at a time.

## Hub Manual Trade

Enable Hub manual-trade writes with `AIQ_MANUAL_TRADE_ENABLE=true` in the Hub
unit or environment file.

The Hub manual-trade API does not require `AI_QUANT_LIVE_ENABLE` or
`AI_QUANT_LIVE_CONFIRM` in the Hub process. Those live cutover variables remain
part of the runtime launch contract, not a prerequisite for Hub-side manual
trade requests.

Manual-trade writes still fail closed when `AI_QUANT_HARD_KILL_SWITCH=1`, and
the existing live risk / kill-mode controls continue to gate submissions.

`GET /api/trade/{lookup_id}/result` now includes the latest persisted
`order_status` plus `response` when an OMS order row exists. The `response`
field preserves the latest exchange payload for rejected, resting, retried, or
deduplicated manual-trade submissions, and falls back to raw text when the
payload is not valid JSON.

## Hub Read Auth

Hub read auth now bypasses `AIQ_MONITOR_TOKEN` for trusted LAN and Tailscale
read clients, while mutation routes still require
`AIQ_MONITOR_ADMIN_TOKEN`.

Trusted read sources include:

- RFC1918 IPv4 private ranges
- IPv4 link-local
- Tailscale IPv4 CGNAT `100.64.0.0/10`
- IPv6 link-local, unique-local, and the common Tailscale ULA prefix

When the Hub sits behind a local proxy or gateway, it only trusts
`X-Forwarded-For` / `X-Real-IP` when the direct peer is loopback. Direct
non-loopback clients cannot spoof the trusted-read bypass with forwarded
headers, and bare loopback peers do not bypass read auth on their own.

The Hub also accepts Tailscale Serve identity headers such as
`Tailscale-User-Login` for read-only bypass, but only when the direct peer is
loopback. This covers a local Tailscale HTTP/HTTPS Serve front layer without
opening the same trust boundary to arbitrary local clients.

Raw TCP forwarding to `127.0.0.1:61010` does not provide those trusted
Tailscale identity headers, so it cannot satisfy the Tailscale read-auth
bypass on its own. Use an HTTP/HTTPS Tailscale Serve-style front layer when the
operator wants tailnet reads without `AIQ_MONITOR_TOKEN`.

When an operator explicitly accepts the local-trust tradeoff for this machine,
the Hub can also trust bare loopback peers through these opt-in flags:

- `AIQ_MONITOR_TRUST_LOOPBACK_READ=1`
- `AIQ_MONITOR_TRUST_LOOPBACK_ADMIN=1`

`AIQ_MONITOR_TRUST_LOOPBACK_READ=1` makes bare loopback peers bypass the
read-auth check, so clients on that localhost-forwarded path do not need to
present `AIQ_MONITOR_TOKEN` on those read requests.
`AIQ_MONITOR_TRUST_LOOPBACK_ADMIN=1` also bypasses the admin-auth check for the
same bare loopback peers, so manual-trade and other admin mutation routes do
not need to present `AIQ_MONITOR_ADMIN_TOKEN` on those loopback requests.

Use those flags only when the operator intentionally treats local loopback as a
trusted boundary, for example a machine-specific raw Tailscale TCP forward that
lands on `127.0.0.1:61010`. Any same-host client that can reach the Hub on
localhost will inherit the same read/admin access while those flags are set.
Keep `AIQ_MONITOR_TOKEN` and `AIQ_MONITOR_ADMIN_TOKEN` configured in the Hub
environment. These flags bypass presenting credentials for loopback requests;
they do not remove the need for the Hub to keep both tokens configured.

Keep `AIQ_MONITOR_TOKEN` configured for non-trusted read clients and WebSocket
consumers outside those local/Tailscale ranges. Keep
`AIQ_MONITOR_ADMIN_TOKEN` configured for all admin and manual-trade actions.

### Immutable Launch Contract

The current Rust paper/live launch contract already starts daemons from the
materialised runtime artefact path exposed as `config_path`, not from the
mutable source YAML exposed as `base_config_path`.

`paper manifest` and `live manifest` daemon commands now include both:

- `--config <materialised runtime artefact>`
- `--expected-config-id <approved config_id>`

The daemon status contracts also persist `config_id`, and paper/live status
surfaces fail closed when the running daemon drifts from the current launch
contract identity. `paper service apply` and `live service apply` also verify
the approved `config_id` before supervision and again against the final
resolved launch contract.

### Transactional Live Apply and Rollback

The Hub live control-plane now treats live apply and live rollback as
transactional operations. It stages the candidate YAML, snapshots the incumbent
live YAML before mutation, and reports success only after
`aiq-runtime live service apply --json` proves that the final live daemon is:

- running
- healthy
- matched to the current launch contract
- on the intended materialised `config_id`

The Hub passes the approved candidate `config_id` into that runtime apply
subprocess through `--expected-config-id`, so the supervised apply fails closed
if the live contract re-resolves to a different identity.

If the supervised apply proof fails, the Hub restores the incumbent live YAML
and runs a supervised recovery apply against that incumbent contract before
returning the failure response. Depending on the recovered lane state, that
recovery can resolve as a start, restart, or no-op. The emitted artefacts now
record both the previous and target `config_id` values.

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

### Reload Semantics Retired

The old `POST /api/config/reload` surface is now retired and fails closed with
explicit guidance instead of touching file mtime. The product no longer claims
that strategy YAML changes are hot-reloaded into running Rust daemons.

Operator-facing config semantics are now:

- non-live files: `Save` writes YAML only
- live file: `Apply to Live` previews restart impact, then runs the supervised
  `apply-live` contract

When the resolved live `config_id` changes, the config editor now tells the
operator that a restart is required before confirmation. When the `config_id`
does not change, the editor still warns that stale or stopped lanes may be
supervised during apply.

### Config Audit and Config-ID Attribution

The Hub now appends config mutation events to:

```bash
artifacts/config_audit/config_events.jsonl
```

Each event records the mutation action, target lane / file variant,
before-and-after `config_id`, validation status, apply / rollback result, and a
weak request actor envelope.

The weak actor envelope currently contains:

- auth scope: `admin_token`
- actor label: optional `X-AIQ-Actor` header, falling back to the auth scope
- request source hints such as `X-Forwarded-For`, `X-Real-IP`, and `User-Agent`

Read recent events through:

```bash
curl -sS \
  -H "Authorization: Bearer $AIQ_MONITOR_TOKEN" \
  "http://127.0.0.1:8000/api/config/audit?file=live&limit=20"
```

Monitoring now derives `since_config` from the deployed heartbeat `config_id`
plus the first runtime-log timestamp that carried that same `config_id`, rather
than from YAML file `mtime`. Snapshot responses therefore expose config-centric
attribution boundaries instead of filesystem metadata.

Paper-mode monitor health now treats the daemon status files as the freshness
fallback when the legacy `runtime_logs` heartbeat is stale or missing. The Hub
reads `ai_quant_paper_v8_<lane>.status.json` to refresh paper health
`ts_ms`/cadence metadata and to surface the current paper `config_id` without
requiring a fresh `engine ok` row in the trading DB.

Paper snapshot cards now label the first headline balance as `CASH`. For paper
lanes that value is the ledger cash remainder after reserved margin and fees,
not total account equity. The companion `EQ` figure is the mark-to-market
estimate built from cash plus reserved margin plus unrealised PnL minus
estimated close fees, which keeps open-position monitoring aligned with the
runtime ledger contract.

Live service state remains a separate contract from heartbeat freshness. An
intentionally stopped live lane should still surface as service `OFF` through
`/api/system/services`, and the paper status-file fallback does not change that
live service-state behaviour.

### Default Redaction and Privileged Diagnostics

Default config/system/monitor read paths now redact sensitive operational
metadata such as:

- raw config bodies and raw config diffs
- filesystem paths
- raw subprocess payloads
- raw system log bodies
- raw audit `data_json` blobs

Use the explicit privileged routes when operators genuinely need raw
diagnostics:

```bash
/api/config/raw/privileged
/api/config/diff/privileged
/api/config/audit/privileged
/api/system/logs/raw
```

The Hub System page now tries the privileged raw log route first, then falls
back to the redacted `/api/system/logs` surface when the operator only has a
viewer/editor/approver token. This keeps service log inspection available
without implying raw journal visibility.

The Hub Config page now treats `GET /api/config/files` and
`GET /api/config/history` as metadata-only read surfaces for the file picker
and backup history views. Raw YAML bodies and raw diffs still require an admin
token through the privileged routes above, so editor and approver tokens can
continue to use approval flows without also receiving raw config visibility.

The Hub Statements page uses `GET /api/config/files` and
`GET /api/config/history` as metadata-only surfaces for the config-change tab.
It also normalises legacy `paper` selections onto the canonical candidate
family (`paper1`-`paper3`) before querying journeys, trades, and config diffs.

These privileged diagnostics reads are audited to:

```bash
artifacts/diagnostic_audit/read_events.jsonl
```

Operators may attach a weak actor hint through `X-AIQ-Actor`; when absent, the
Hub falls back to the `admin_token` auth scope label in the diagnostics audit
event.

## Maker-Checker Live Approvals

The Hub now separates three live control-plane roles:

- viewer: read-only access through `AIQ_MONITOR_TOKEN` or the trusted read
  bypass rules
- editor: config save access plus live request creation through
  `AIQ_MONITOR_EDITOR_TOKEN`
- approver: live request approval / rejection through
  `AIQ_MONITOR_APPROVER_TOKEN`

The existing `AIQ_MONITOR_ADMIN_TOKEN` remains reserved for privileged
diagnostics and system actions. It no longer acts as the one undifferentiated
bearer for high-risk live config execution.

The live workflow is now:

1. editor previews live apply with `POST /api/config/actions/apply-live`
   and `dry_run: true`
2. editor creates a pending request through
   `POST /api/config/actions/apply-live/request` or
   `POST /api/config/actions/rollback-live/request`
3. approver lists pending requests through `GET /api/config/approvals`
4. approver executes or rejects through:
   `POST /api/config/approvals/<request_id>/approve`
   `POST /api/config/approvals/<request_id>/reject`

Pending approval requests are stored under:

```bash
artifacts/config_approvals/
```

High-risk config audit events now include both requester and approver identities
when the live request is approved or rejected.

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
cargo build --release --manifest-path runtime/aiq-runtime/Cargo.toml --bin aiq-maintenance
AI_QUANT_FACTORY_ENABLE=1 cargo run --release --manifest-path runtime/aiq-runtime/Cargo.toml --bin aiq-factory -- run --config config/strategy_overrides.yaml --settings config/factory_defaults.yaml --profile daily --json
```

`config/factory_defaults.yaml` now enforces parity fail-closed by default. Live
promotion remains an explicit deployment setting: enable
`deployment.apply_to_live: true` only on the production instance that is meant
to restart `openclaw-ai-quant-live-v8`.

Factory deployment defaults now fail closed when the settings file is missing
or incomplete: `DeploymentSettings::default()` no longer applies paper or live
changes, and it does not restart services until `deployment.apply_to_paper` or
`deployment.apply_to_live` is set explicitly in `config/factory_defaults.yaml`.

The financial-grade factory default now seeds sweep/replay balance from current
live equity (including unrealised PnL) and compares each challenger against the
currently deployed target config before any paper replacement is applied. Use
`balance.mode: fixed` only for controlled research runs.

Factory validation now splits the common DB coverage into an explicit train
window plus a trailing holdout window. Tune `validation.holdout_fraction` and
`validation.holdout_splits` in `config/factory_defaults.yaml` when operators
need a different holdout share or slice count. Sweep / TPE search and the
train parity evidence run on the train window, while candidate gating and
incumbent/challenger comparison use the holdout window only.
Because the backtester treats `--start-ts` and `--end-ts` as inclusive, the
factory now makes the train window end one timestamp before the holdout window
starts so the boundary bar can never leak into both evidence sets.
The financial-grade defaults reserve the trailing 25% of common coverage as the
holdout window and summarise it in 3 equal holdout slices.

Inspect `artifacts/.../run_metadata.json` and the candidate rows in
`reports/report.json` for the resolved `coverage`, `train`, and `holdout`
boundaries. Candidate rows now expose `train_parity_replay_report_path`,
`train_parity_sweep_report_path`, `holdout_summary_path`, and
`holdout_median_daily_return` so operators can audit exactly which window
produced each gate decision.
Backtester replay and sweep JSON artefacts now serialise non-finite
`profit_factor` as the string token `"Infinity"` instead of emitting JSON
`null`. Factory readers remain backward-compatible with older artefacts that
still contain `profit_factor: null`, so historical holdout and parity evidence
can still be inspected after the upgrade.
Train parity now compares a CPU replay against a dedicated single-combo train
parity sweep built from the same lane-effective config rather than against the
original shortlist sweep row.
`step4_parity.comparison_scope` reports whether the evidence is
`aggregate_only` or `aggregate_and_symbol`. When the GPU sweep artefact omits a
per-symbol breakdown, `step4_parity.symbol_checks` stays empty and
`step4_parity.symbol_evidence_note` explains why only aggregate parity was
evaluated.
`reports/selection.json` now distinguishes 3 stages explicitly:

- `deployable_candidates`: the configs that survived the full validation suite
- `role_candidates_by_role`: the strongest challenger per role before
  incumbent/materiality comparison
- `selected` / `selected_candidates_by_role`: the configs actually selected as
  deploy targets after incumbent/materiality checks

If a historical run failed while parsing a replay or sweep report with
`invalid type: null, expected f64`, treat that as a legacy `profit_factor`
serialisation issue. Current factory builds accept that older artefact shape,
while newly emitted artefacts record the explicit `"Infinity"` token.

When the run is blocked before any role selection succeeds, `selected` is
`null` and `best_candidate_preview` carries the best blocked candidate summary
instead. Treat `best_candidate_preview` as diagnostic context only: deploy
decisions are made from the deployable set after validation, not directly from
the shortlist or raw TPE winner.

The Hub Factory page accepts the canonical Rust run ID even when the artefact
directory still carries a `run_` prefix, and it now surfaces `directory_name`,
selection summary fields, candidate evidence, and timer `UnitFileState`
separately. Treat the page timer `enabled` state as the systemd unit-file
setting, not as a synonym for the current `ActiveState`.

Keep nightly factory artefacts bounded with:

```bash
cargo run -p aiq-runtime --bin aiq-maintenance -- prune-factory-artifacts --project-dir "$PWD" --settings config/factory_defaults.yaml --profile nightly
```

The nightly prune keeps only the newest `run_nightly_*` bundle plus any run
still referenced by the current paper soak markers, the live governance state,
or the live YAML `# Base:` header. It also removes stale
`artifacts/_effective_configs/nightly_*.yaml` files that no longer belong to
the latest or currently deployed run set. If a nightly cycle fails before the
service-level post-run housekeeping fires, run the maintenance command
manually. Add `--dry-run` to preview the retained run IDs and aggregate delete
counts first; add `--verbose` when you need the full per-path deletion list.

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
```

The nightly timer stays at `00:50 UTC`. The nightly service example also runs
`aiq-maintenance prune-factory-artifacts --profile nightly` as a post-success
housekeeping step so only the latest nightly artefact bundle and the currently
deployed factory references stay on disk. Build the
`aiq-maintenance` binary alongside `aiq-factory` before enabling that service
example, otherwise systemd will skip the post-step because the example uses a
non-fatal `ExecStartPost=-...` contract.

## Live DB Sync

Use the Rust live fill sync command when Hyperliquid fills must be reconciled
back into the local live SQLite ledger, including fills created outside this
repo.

Manual one-shot run:

```bash
cargo run -p aiq-runtime -- live sync-fills --project-dir "$PWD" --json
```

The command stores a cursor in the live DB and replays a small overlap window
on each run so it can safely run under an hourly timer. For deeper backfills,
override the window explicitly:

```bash
cargo run -p aiq-runtime -- live sync-fills --project-dir "$PWD" --start-ms 1771126743410 --end-ms 1773401649000 --json
```

The command fails closed when it encounters unsupported fill shapes. Treat a
failed hourly run as an operator action item: inspect the emitted warnings
before trusting the local ledger again.

The tracked service examples live under:

```bash
systemd/openclaw-ai-quant-live-db-sync-v8.service.example
systemd/openclaw-ai-quant-live-db-sync-v8.timer.example
```

Typical install flow:

```bash
cargo build --release --manifest-path runtime/aiq-runtime/Cargo.toml
install -d "$HOME/.config/systemd/user"
sed "s|\$PROJECT_DIR|$PWD|g" \
  "$PWD/systemd/openclaw-ai-quant-live-db-sync-v8.service.example" \
  > "$HOME/.config/systemd/user/openclaw-ai-quant-live-db-sync-v8.service"
install -D -m 0644 "$PWD/systemd/openclaw-ai-quant-live-db-sync-v8.timer.example" \
  "$HOME/.config/systemd/user/openclaw-ai-quant-live-db-sync-v8.timer"
systemctl --user daemon-reload
systemctl --user enable --now openclaw-ai-quant-live-db-sync-v8.timer
journalctl --user -u openclaw-ai-quant-live-db-sync-v8.service -f
```

The example service reads `ai-quant-v8.env` plus `ai-quant-live-v8.env`, then
uses the Rust `live sync-fills` cursor contract to auto-check and auto-sync the
ledger every hour.

Successful runs also write exchange account and position snapshots into the
live DB, so Hub read paths can fall back to current live balance and holdings
even when no fresh in-memory Hyperliquid snapshot is available.

Service-level tuning knobs:

- `AI_QUANT_LIVE_FILL_SYNC_LOOKBACK_HOURS`: fallback scan window when no cursor exists yet.
- `AI_QUANT_LIVE_FILL_SYNC_OVERLAP_MINUTES`: overlap replay window for safe hourly re-checks.
- `AI_QUANT_LIVE_FILL_SYNC_CURSOR_KEY`: cursor namespace stored in `runtime_sync_cursors`.

## Diagnostics

```bash
cargo run -p aiq-runtime -- doctor --json
cargo run -p aiq-runtime -- pipeline --json
journalctl --user -u openclaw-ai-quant-live-v8 -f
journalctl --user -u openclaw-ai-quant-assist-v8 -f
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
