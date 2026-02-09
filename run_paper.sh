#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

SERVICE_NAME="openclaw-ai-quant-trader.service"

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user restart "${SERVICE_NAME}"
  systemctl --user --no-pager -l status "${SERVICE_NAME}" || true
else
  echo "systemd user instance not available; running unified daemon in foreground (paper)."
  cd "${SCRIPT_DIR}"
  exec env AI_QUANT_MODE=paper "${SCRIPT_DIR}/venv/bin/python3" -u -m quant_trader_v5.run_unified_daemon
fi
