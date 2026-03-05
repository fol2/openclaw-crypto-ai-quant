# ADR-0001: Rust Runtime Foundation

## Status

Accepted

## Context

The repo currently mixes a Rust decision kernel, Python runtime orchestration, Rust market-data services, and CUDA sweep paths. This makes parity work expensive because there is more than one active non-CUDA source of truth.

We need a single Rust-owned runtime path that can eventually serve live, paper, replay, backtest, and CPU sweep with one shared execution contract. We also need a modular pipeline contract so parity can be debugged stage by stage without editing code.

## Decision

- Introduce `aiq-runtime` as the long-term Rust runtime binary.
- Introduce `aiq-runtime-core` as the shared runtime bootstrap and pipeline planning crate.
- Extend the existing YAML contract with additive `runtime` and `pipeline` sections while keeping existing keys backward compatible.
- Use a compiled-in plugin registry and a declarative stage graph for v1.
- Treat Python runtime code as frozen migration reference only. New runtime behaviour should move into Rust instead of expanding the Python path.

## Consequences

- The repo now has an explicit Rust runtime foundation that future PRs can extend without re-deciding the architecture.
- Pipeline profiles become part of the public operational contract and must remain stable once consumed by parity tooling.
- Housekeeping becomes a tracked migration responsibility, not an end-of-project clean-up task.
- CUDA work is intentionally separated: first converge on one Rust runtime contract, then realign Rust and CUDA against that frozen contract.
