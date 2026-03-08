#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

SERVICE_NAME="openclaw-ai-quant-live-v8.service"

if [[ -n "${INVOCATION_ID:-}" ]]; then
  :
elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user restart "${SERVICE_NAME}"
  systemctl --user --no-pager -l status "${SERVICE_NAME}" || true
  exit 0
else
  echo "systemd user instance not available; running Rust live daemon in foreground."
fi

for ENV_FILE in \
  "${HOME}/.config/openclaw/ai-quant-universe-v8.env" \
  "${HOME}/.config/openclaw/ai-quant-v8.env" \
  "${HOME}/.config/openclaw/ai-quant-live-v8.env"
do
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "${ENV_FILE}"
    set +a
  fi
done

cd "${PROJECT_DIR}"
export AI_QUANT_MODE="${AI_QUANT_MODE:-live}"
export AI_QUANT_INSTANCE_TAG="${AI_QUANT_INSTANCE_TAG:-v8-LIVE}"
export AI_QUANT_DB_PATH="${AI_QUANT_DB_PATH:-${PROJECT_DIR}/trading_engine_v8_live.db}"
if [[ -n "${AI_QUANT_RUNTIME_BIN:-}" ]]; then
  exec "${AI_QUANT_RUNTIME_BIN}" live daemon --project-dir "${PROJECT_DIR}" "$@"
fi
if [[ -x "${PROJECT_DIR}/target/release/aiq-runtime" ]]; then
  exec "${PROJECT_DIR}/target/release/aiq-runtime" live daemon --project-dir "${PROJECT_DIR}" "$@"
fi
if command -v aiq-runtime >/dev/null 2>&1; then
  exec "$(command -v aiq-runtime)" live daemon --project-dir "${PROJECT_DIR}" "$@"
fi
exec cargo run -q -p aiq-runtime -- live daemon --project-dir "${PROJECT_DIR}" "$@"
