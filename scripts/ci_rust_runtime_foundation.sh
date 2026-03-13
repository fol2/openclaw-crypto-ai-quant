#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

command -v jq >/dev/null 2>&1 || {
  echo "[runtime-foundation] jq is required" >&2
  exit 1
}

echo "[runtime-foundation] cargo check --workspace"
cargo check --workspace

echo "[runtime-foundation] cargo test -p aiq-runtime-core"
cargo test -p aiq-runtime-core

echo "[runtime-foundation] cargo test -p aiq-runtime"
cargo test -p aiq-runtime

echo "[runtime-foundation] bt-core config + init-state compatibility"
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_load_yaml_runtime_pipeline_overrides
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_parse_valid_v2_json_with_runtime
cargo test --manifest-path backtester/Cargo.toml -p bt-core test_into_sim_state_with_runtime_filters_unknown_symbols

echo "[runtime-foundation] runtime CLI smoke"
cargo run -q -p aiq-runtime -- doctor --json >/tmp/aiq-runtime-doctor.json
cargo run -q -p aiq-runtime -- pipeline --json >/tmp/aiq-runtime-pipeline.json

jq -e '.config_fingerprint | strings | length > 0' /tmp/aiq-runtime-doctor.json >/dev/null
jq -e '.pipeline.stages | arrays | length > 0' /tmp/aiq-runtime-doctor.json >/dev/null
jq -e '.pipeline.stages | arrays | length > 0' /tmp/aiq-runtime-pipeline.json >/dev/null

echo "[runtime-foundation] PASS: Rust runtime foundation checks completed."
