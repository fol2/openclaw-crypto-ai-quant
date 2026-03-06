# Rust Runtime Foundation Validation

## Objective

Provide one compact validation matrix for the Rust runtime foundation slice.

This slice is considered healthy only when all of the following surfaces are
green:

- root runtime workspace resolution
- shared runtime config extensions
- pipeline bootstrap and profile resolution
- snapshot schema validation
- paper snapshot export
- paper snapshot seed
- paper runtime bootstrap shell
- paper runtime one-shot execution shell
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
cargo run -q -p aiq-runtime -- snapshot validate --path <snapshot_v2_valid.json> --json
cargo run -q -p aiq-runtime -- snapshot seed-paper --snapshot <snapshot_v2_valid.json> --target-db <paper_fixture.db> --strict-replace --json
cargo run -q -p aiq-runtime -- paper doctor --db <paper_fixture.db> --json
cargo run -q -p aiq-runtime -- paper run-once --db <paper_fixture.db> --candles-db <candles_fixture.db> --target-symbol ETH --exported-at-ms 1772676900000 --dry-run --json
cargo run -q -p aiq-runtime -- paper loop --db <paper_fixture.db> --candles-db <candles_fixture.db> --symbols ETH --start-step-close-ts-ms 1773424200000 --max-steps 2 --dry-run --json
cargo run -q -p aiq-runtime -- paper doctor --db <paper_fixture.db> --live --json
cargo run -q -p aiq-runtime -- paper run-once --db <paper_fixture.db> --candles-db <candles_fixture.db> --target-symbol ETH --exported-at-ms 1772676900000 --live --dry-run --json
```

## Acceptance Checks

- `doctor --json` prints a bootstrap object with:
  - a 64-char `config_fingerprint`
  - a resolved pipeline profile
  - explicit stage entries with enabled/disabled state
- `pipeline --json` resolves `production` cleanly against the example YAML when the tracked live YAML is absent.
- `bt-core` accepts snapshots with `version = 2` and runtime cooldown markers.
- `aiq-runtime` can export a v2 paper snapshot from SQLite and re-validate it through the same Rust snapshot contract.
- `aiq-runtime` can seed a paper DB from a v2 snapshot and report deterministic write counts for `trades`, `position_state`, `runtime_cooldowns`, and `runtime_last_closes`.
- `aiq-runtime paper doctor` can restore Rust-owned paper state from the paper DB and emit a deterministic bootstrap report.
- `aiq-runtime paper run-once` can restore paper state, execute one single-shot step, and report projected action codes and write-back counts.
- `paper run-once` reproducibility requires a fixed `--exported-at-ms`; otherwise write timestamps follow execution time for current DB parity.
- `aiq-runtime paper cycle` can execute one explicit multi-symbol cycle with `--step-close-ts-ms`, record a rerun guard in `runtime_cycle_steps`, and fail closed on duplicate re-apply.
- `aiq-runtime paper loop` can resume from prior `runtime_cycle_steps`, execute up to `--max-steps` unapplied cycle steps, and stop cleanly when the next due step exceeds the latest common candle close.
- multi-step `paper loop --dry-run` previews must carry forward the projected Rust paper state between iterations even though the real paper DB remains untouched.

## Fixture Guidance

- Use a temporary SQLite DB with minimal `trades`, `position_state`, `runtime_cooldowns`, and `runtime_last_closes` tables for paper export tests.
- Use v2 JSON fixtures with `runtime.entry_attempt_ms_by_symbol` and `runtime.exit_attempt_ms_by_symbol` for init-state compatibility checks.
- `paper run-once` fixtures must provide bars for both the target symbol and the BTC anchor symbol at the resolved `engine.interval`.
- `paper cycle` validation should include one write run plus one duplicate-step rerun that hard-fails without changing `trades`, `position_state`, `runtime_cooldowns`, `runtime_last_closes`, or `runtime_cycle_steps`.
- `paper loop` validation should include one bootstrap run on a fresh DB with `--start-step-close-ts-ms`, one follow-up resume run without the bootstrap flag, and one idle run that exits with `executed_steps == 0` because the next due step is newer than the latest common candle close.
- `paper loop` validation should also include one gap fixture where a due `step_close_ts_ms` is missing from the candle DB; the command must fail closed instead of backfilling from an older bar and marking the gap as applied.
- Keep fixtures deterministic: one open position, one add, one last-close marker, and one runtime cooldown marker is enough for the foundation slice.
