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
```

## Acceptance Checks

- `doctor --json` prints a bootstrap object with:
  - a 64-char `config_fingerprint`
  - a resolved pipeline profile
  - explicit stage entries with enabled/disabled state
- `pipeline --json` resolves `production` cleanly against the example YAML when the tracked live YAML is absent.
- `bt-core` accepts snapshots with `version = 2` and runtime cooldown markers.
- `aiq-runtime` can export a v2 paper snapshot from SQLite and re-validate it through the same Rust snapshot contract.
- `aiq-runtime` can seed a paper DB from a v2 snapshot and report deterministic write counts for `trades`, `position_state`, and `runtime_cooldowns`.
- `aiq-runtime paper doctor` can restore Rust-owned paper state from the paper DB and emit a deterministic bootstrap report.

## Fixture Guidance

- Use a temporary SQLite DB with minimal `trades`, `position_state`, and `runtime_cooldowns` tables for paper export tests.
- Use v2 JSON fixtures with `runtime.entry_attempt_ms_by_symbol` and `runtime.exit_attempt_ms_by_symbol` for init-state compatibility checks.
- Keep fixtures deterministic: one open position, one add, and one runtime cooldown marker is enough for the foundation slice.
