# V8 SSOT Migration Status

**Status**: Complete (all milestones merged to `major-v8`)

## Archive

The legacy main-branch backtester is tagged as `archive/main-backtester-v6`.
It is read-only and retained for 30 days for reference.

## What Changed

V8 establishes a **single source of truth (SSOT)** architecture:

- **Decision Kernel** (`crates/bt-core/src/decision_kernel.rs`): Pure
  deterministic state machine shared by all execution paths
- **GPU Sweep** (`crates/bt-gpu/kernels/sweep_engine.cu`): CUDA kernel
  aligned with CPU kernel semantics (glitch guard, trailing SL, f64 accounting)
- **PyO3 Bridge** (`crates/bt-runtime/src/lib.rs`): Rust<->Python FFI for
  live/paper trading via `step_full()`, `apply_funding()`, `get_equity()`
- **Parity Tests**: GPU<->CPU, Main<->V8, and three-way parity frameworks

## Key Files

| File | Role |
|------|------|
| `crates/bt-core/src/decision_kernel.rs` | SSOT decision kernel |
| `crates/bt-gpu/kernels/sweep_engine.cu` | GPU sweep engine |
| `crates/bt-gpu/src/buffers.rs` | GPU config/state layout (512B config, 4032B state) |
| `crates/bt-runtime/src/lib.rs` | PyO3 bridge |
| `crates/bt-core/tests/gpu_cpu_parity.rs` | GPU<->CPU parity test |
| `crates/bt-core/tests/main_v8_parity.rs` | Main<->V8 parity test |
| `scripts/three_way_parity.py` | Three-way parity report |
