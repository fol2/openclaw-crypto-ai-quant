#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PAPER_LANE="${1:-${AI_QUANT_PAPER_LANE:-paper1}}"
shift || true

if [[ -n "${AI_QUANT_RUNTIME_BIN:-}" ]]; then
  exec "${AI_QUANT_RUNTIME_BIN}" paper lane daemon --lane "${PAPER_LANE}" --project-dir "${PROJECT_DIR}" "$@"
fi

if [[ -x "${PROJECT_DIR}/target/release/aiq-runtime" ]]; then
  exec "${PROJECT_DIR}/target/release/aiq-runtime" paper lane daemon --lane "${PAPER_LANE}" --project-dir "${PROJECT_DIR}" "$@"
fi

exec cargo run -q -p aiq-runtime -- paper lane daemon --lane "${PAPER_LANE}" --project-dir "${PROJECT_DIR}" "$@"
