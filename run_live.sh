#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

SERVICE_NAME="openclaw-ai-quant-live.service"

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user restart "${SERVICE_NAME}"
  systemctl --user --no-pager -l status "${SERVICE_NAME}" || true
else
  echo "systemd user instance not available; running unified daemon in foreground (live/dry_live per env)."
  ENV_FILE="${HOME}/.config/openclaw/ai-quant-live.env"
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "${ENV_FILE}"
    set +a
  fi
  cd "${SCRIPT_DIR}"
  exec "${SCRIPT_DIR}/venv/bin/python3" -u -m quant_trader_v5.run_unified_daemon
fi
