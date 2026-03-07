#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PAPER_LANE="${AI_QUANT_PAPER_LANE:-paper1}"
SERVICE_NAME="openclaw-ai-quant-trader-v8-${PAPER_LANE}.service"

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user restart "${SERVICE_NAME}"
  systemctl --user --no-pager -l status "${SERVICE_NAME}" || true
else
  echo "systemd user instance not available; running Rust paper lane daemon in foreground (${PAPER_LANE})."
  exec "${PROJECT_DIR}/scripts/run_paper_lane.sh" "${PAPER_LANE}" "$@"
fi
