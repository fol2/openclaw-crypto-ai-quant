#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

LANE="${1:-${AI_QUANT_PAPER_LANE:-paper1}}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -n "${AI_QUANT_RUNTIME_BIN:-}" ]]; then
  exec "${AI_QUANT_RUNTIME_BIN}" paper daemon --lane "${LANE}" --project-dir "${PROJECT_DIR}" "$@"
fi

if [[ -x "${PROJECT_DIR}/target/release/aiq-runtime" ]]; then
  exec "${PROJECT_DIR}/target/release/aiq-runtime" paper daemon --lane "${LANE}" --project-dir "${PROJECT_DIR}" "$@"
fi

if command -v aiq-runtime >/dev/null 2>&1; then
  exec "$(command -v aiq-runtime)" paper daemon --lane "${LANE}" --project-dir "${PROJECT_DIR}" "$@"
fi

exec cargo run -q -p aiq-runtime -- paper daemon --lane "${LANE}" --project-dir "${PROJECT_DIR}" "$@"
