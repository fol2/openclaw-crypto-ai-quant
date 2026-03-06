# Rust Runtime Foundation Validation

## Objective

Provide one compact validation matrix for the Rust runtime foundation slice.

This slice is considered healthy only when all of the following surfaces are
green:

- root runtime workspace resolution
- shared runtime config extensions
- Rust paper effective-config parity
- pipeline bootstrap and profile resolution
- snapshot schema validation
- paper snapshot export
- paper snapshot seed
- paper daemon service manifest resolution
- paper daemon service status resolution
- paper daemon service action resolution
- paper runtime bootstrap shell
- paper runtime one-shot execution shell
- paired opt-in paper daemon orchestration wrapper
- daemon lane lifecycle status contract
- bt-core init-state compatibility

## Commands

```bash
# Root runtime workspace
cargo check --workspace
cargo test -p aiq-runtime-core
cargo test -p aiq-runtime

# Shared config / init-state compatibility
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_load_yaml_runtime_pipeline_overrides
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_parse_valid_v2_json_with_runtime
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_into_sim_state_with_runtime_filters_unknown_symbols

# Runtime CLI smoke
cargo run -q -p aiq-runtime -- doctor --json
cargo run -q -p aiq-runtime -- pipeline --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example cargo run -q -p aiq-runtime -- paper effective-config --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example AI_QUANT_DB_PATH=<paper_fixture.db> AI_QUANT_CANDLES_DB_PATH=<candles_fixture.db> AI_QUANT_SYMBOLS=ETH,SOL AI_QUANT_LOOKBACK_BARS=200 cargo run -q -p aiq-runtime -- paper manifest --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example AI_QUANT_PROMOTED_ROLE=primary AI_QUANT_STRATEGY_MODE_FILE=<strategy_mode.txt> AI_QUANT_DB_PATH=<paper_fixture.db> AI_QUANT_CANDLES_DB_PATH=<candles_fixture.db> AI_QUANT_SYMBOLS=ETH cargo run -q -p aiq-runtime -- paper manifest --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example AI_QUANT_DB_PATH=<paper_fixture.db> AI_QUANT_CANDLES_DB_PATH=<candles_fixture.db> AI_QUANT_SYMBOLS=ETH AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS=1773424200000 cargo run -q -p aiq-runtime -- paper manifest --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example AI_QUANT_DB_PATH=<paper_fixture.db> AI_QUANT_CANDLES_DB_PATH=<candles_fixture.db> AI_QUANT_SYMBOLS=ETH AI_QUANT_STATUS_STALE_AFTER_MS=30000 cargo run -q -p aiq-runtime -- paper status --json
AI_QUANT_STRATEGY_YAML=config/strategy_overrides.yaml.example AI_QUANT_DB_PATH=<paper_fixture.db> AI_QUANT_CANDLES_DB_PATH=<candles_fixture.db> AI_QUANT_SYMBOLS=ETH AI_QUANT_STATUS_STALE_AFTER_MS=30000 cargo run -q -p aiq-runtime -- paper service --json
cargo run -q -p aiq-runtime -- snapshot validate --path <snapshot_v2_valid.json> --json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot <snapshot_v2_valid.json> --target-db <paper_fixture.db> --strict-replace --json
cargo run -q -p aiq-runtime -- paper doctor --db <paper_fixture.db> --json
cargo run -q -p aiq-runtime -- paper run-once --db <paper_fixture.db> --candles-db <candles_fixture.db> --target-symbol ETH --exported-at-ms 1772676900000 --dry-run --json
cargo run -q -p aiq-runtime -- paper loop --db <paper_fixture.db> --candles-db <candles_fixture.db> --symbols ETH --start-step-close-ts-ms 1773424200000 --max-steps 2 --dry-run --json
cargo run -q -p aiq-runtime -- paper loop --db <paper_fixture.db> --candles-db <candles_fixture.db> --symbols ETH --follow --idle-sleep-ms 1 --max-idle-polls 1 --max-steps 1 --json
# Opt-in daemon wrapper smoke (paired CLI surface)
cargo run -q -p aiq-runtime -- paper daemon --db <paper_fixture.db> --candles-db <candles_fixture.db> --symbols ETH --start-step-close-ts-ms 1773424200000 --idle-sleep-ms 1 --max-idle-polls 1 --dry-run --json
# Follow-mode symbols-file refresh smoke
cargo test -p aiq-runtime loop_follow_mode_reloads_symbols_file_between_idle_polls
cargo test -p aiq-runtime --test paper_daemon_smoke paper_daemon_reloads_symbols_file_between_idle_polls
cargo run -q -p aiq-runtime -- paper doctor --db <paper_fixture.db> --live --json
cargo run -q -p aiq-runtime -- paper run-once --db <paper_fixture.db> --candles-db <candles_fixture.db> --target-symbol ETH --exported-at-ms 1772676900000 --live --dry-run --json
```

## Acceptance Checks

- `doctor --json` prints a bootstrap object with:
  - a 64-char `config_fingerprint`
  - a resolved pipeline profile
  - explicit stage entries with enabled/disabled state
- `pipeline --json` resolves `production` cleanly against the example YAML when the tracked live YAML is absent.
- `paper manifest --json` resolves the current daemon service/env contract without executing any paper steps, derives a candle DB path when only `AI_QUANT_CANDLES_DB_DIR` is present, emits a deterministic `daemon_command`, reports whether the current lane is blocked, bootstrap-ready, resumable, or merely idle caught up, and resolves the daemon `status_path`.
- `paper effective-config --json` is the narrow read-only control-plane surface for Python paper start-up and factory materialisation, and it must emit the same `active_yaml_path`, `effective_yaml_path`, interval, `strategy_overrides_sha1`, and `config_id` that the broader paper surfaces later consume.
- `paper manifest --json` also applies the same promoted-role / strategy-mode effective config contract used by the Rust paper surfaces, carries `base_config_path`, `active_yaml_path`, `effective_yaml_path`, `strategy_mode_source`, `strategy_overrides_sha1`, and `config_id`, and resolves the interval from that effective config instead of raw env assumptions.
- `paper status --json` combines that same launch contract with the persisted daemon lifecycle JSON and reports whether the lane is running, stale, stopped, restart-required, or merely launch-ready when no daemon status exists yet. Running lanes must now fail closed when the daemon reports unhealthy runtime errors or when the launch identity drifts (`profile`, DB paths, BTC anchor, lookback, explicit symbols, the bootstrap step while the lane is still fresh, or path wiring).
- `paper service --json` reuses the same read-only status view and reports whether later supervision should hold, start, restart, or merely monitor the lane, together with an operator-facing `action_reason`. Idle watchlist-owned lanes that are launch-ready should now map to `start` instead of a permanent `hold`.
- `bt-core` accepts snapshots with `version = 2` and runtime cooldown markers.
- `aiq-runtime` can export a v2 paper snapshot from SQLite and re-validate it through the same Rust snapshot contract.
- `aiq-runtime` can seed a paper DB from a v2 snapshot and report deterministic write counts for `trades`, `position_state`, `runtime_cooldowns`, and `runtime_last_closes`.
- `aiq-runtime paper doctor` can restore Rust-owned paper state from the paper DB and emit a deterministic bootstrap report.
- `aiq-runtime paper run-once` can restore paper state, execute one single-shot step, and report projected action codes and write-back counts.
- `paper run-once` reproducibility requires a fixed `--exported-at-ms`; otherwise write timestamps follow execution time for current DB parity.
- `aiq-runtime paper cycle` can execute one explicit multi-symbol cycle with `--step-close-ts-ms`, record a rerun guard in `runtime_cycle_steps`, and fail closed on duplicate re-apply.
- `aiq-runtime paper loop` can resume from prior `runtime_cycle_steps`, execute up to `--max-steps` unapplied cycle steps, and stop cleanly when the next due step exceeds the latest common candle close.
- `paper loop --follow` can keep polling after catch-up and exit only when its idle poll budget is exhausted or a new due step becomes available.
- `paper loop` only loads `--symbols-file` once at start-up in the current contract; it must not silently drift back into daemon-owned watchlist reload behaviour.
- `paper loop --follow` must continue to honour that one-shot symbols-file load; an empty start-up manifest without open positions is still a fail-closed configuration for the bounded loop surface.
- `paper manifest`, `paper doctor`, `paper cycle`, `paper loop`, and `paper daemon` must all resolve the same effective config when `AI_QUANT_PROMOTED_ROLE` and `AI_QUANT_STRATEGY_MODE` / `AI_QUANT_STRATEGY_MODE_FILE` are present.
- Python paper start-up and factory deployment must resolve the same `config_id`, interval, promoted-config path, and strategy-mode source as `aiq-runtime paper effective-config` / `paper manifest` for the same env.
- the opt-in `aiq-runtime paper daemon` wrapper must reuse the same `paper loop --follow` / `paper cycle` contracts, expose lock metadata, and avoid claiming Python daemon cutover or widening DB projections.
- the opt-in `paper daemon` wrapper must own per-iteration `--watch-symbols-file` refresh behaviour so operators can refresh the Rust symbol lane without restarting the daemon.
- the opt-in `paper daemon` wrapper must also stay alive across an initially empty `--symbols-file`, then execute the next due step once a later watchlist update makes work available.
- the opt-in `paper daemon` wrapper must retain the last good manifest when a later `--watch-symbols-file` payload is invalid or runtime-invalid malformed, and must report that failure without incrementing the successful reload count.
- the opt-in `paper daemon` wrapper must write a durable lifecycle status JSON that shows `running=true` while the daemon holds the lane and flips to `running=false` with a `stopped_at_ms` timestamp on exit.
- multi-step `paper loop --dry-run` previews must carry forward the projected Rust paper state between iterations even though the real paper DB remains untouched.

## Fixture Guidance

- Use a temporary SQLite DB with minimal `trades`, `position_state`, `runtime_cooldowns`, and `runtime_last_closes` tables for paper export tests.
- Use v2 JSON fixtures with `runtime.entry_attempt_ms_by_symbol` and `runtime.exit_attempt_ms_by_symbol` for init-state compatibility checks.
- manifest validation should include one env-driven fixture that resolves `AI_QUANT_STRATEGY_YAML`, `AI_QUANT_DB_PATH`, `AI_QUANT_CANDLES_DB_PATH`, `AI_QUANT_SYMBOLS`, and `AI_QUANT_LOOKBACK_BARS` into a deterministic report, plus one mismatch warning when `AI_QUANT_INTERVAL` disagrees with the resolved config interval.
- manifest validation should also include one service-like fixture that applies `AI_QUANT_PROMOTED_ROLE` and `AI_QUANT_STRATEGY_MODE_FILE`, proves the resolved interval follows the effective config, and emits `promoted_config_path` / `strategy_mode_source` metadata.
- manifest validation should also include one fresh-lane fixture that reports `bootstrap_required`, one bootstrap-ready fixture with `AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS`, and one resumed fixture whose `next_due_step_close_ts_ms` is derived from existing `runtime_cycle_steps`.
- status validation should include one fixture with no daemon status file yet, one stopped daemon status fixture, one stale running status fixture, one unhealthy running status fixture, and one running-but-mismatched status fixture that reports `restart_required`.
- service-action validation should map at least one launch-blocked fixture to `hold`, one launch-ready or stopped fixture to `start`, one launch-ready idle watchlist fixture to `start`, one stale / unhealthy / mismatched running fixture to `restart`, and one healthy running fixture to `monitor`.
- `paper run-once` fixtures must provide bars for both the target symbol and the BTC anchor symbol at the resolved `engine.interval`.
- `paper cycle` validation should include one write run plus one duplicate-step rerun that hard-fails without changing `trades`, `position_state`, `runtime_cooldowns`, `runtime_last_closes`, or `runtime_cycle_steps`.
- `paper loop` validation should include one bootstrap run on a fresh DB with `--start-step-close-ts-ms`, one follow-up resume run without the bootstrap flag, and one idle run that exits with `executed_steps == 0` because the next due step is newer than the latest common candle close.
- follow-mode validation should prove idle polling does not mutate the DB and that `max_idle_polls` counts idle polls directly (`1` means exit on the first no-work poll) so CI catches off-by-one regressions.
- symbols-file refresh validation should prove `paper daemon --watch-symbols-file` can pick up a changed `--symbols-file` on a later iteration, record exactly one successful reload in the JSON report, and keep the same DB write contract.
- empty-watchlist validation should prove a follow-mode daemon can idle on an initially empty watched `--symbols-file`, report that idle state, and then either exhaust its idle budget cleanly or execute once later symbols arrive.
- the opt-in daemon wrapper smoke may reuse that same follow-mode fixture with a dedicated `--lock-path`; it should prove the wrapper reports the chosen lock file and still exits cleanly after the configured idle poll budget.
- lifecycle-status validation should use a dedicated `--status-path`, prove the daemon materialises the JSON while it is still running, and confirm the same file flips to a stopped state after SIGTERM.
- manifest-retention validation should include both invalid UTF-8 payloads and runtime-invalid but UTF-8-clean payloads, and it should also prove that no-op rewrites do not increment the successful reload count.
- `paper loop` validation should also include one gap fixture where a due `step_close_ts_ms` is missing from the candle DB; the command must fail closed instead of backfilling from an older bar and marking the gap as applied.
- Keep fixtures deterministic: one open position, one add, one last-close marker, and one runtime cooldown marker is enough for the foundation slice.
