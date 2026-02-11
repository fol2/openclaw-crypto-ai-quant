#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[gpu-parity-gate] SKIP: CUDA toolkit unavailable (nvcc not found)."
    exit 0
fi

# On WSL2, libcuda is often exposed via this path.
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${ROOT_DIR}/backtester"
echo "[gpu-parity-gate] Running tiny GPU runtime parity fixture..."
cargo test -p bt-gpu --test gpu_runtime_parity_tiny_fixture -- --nocapture
