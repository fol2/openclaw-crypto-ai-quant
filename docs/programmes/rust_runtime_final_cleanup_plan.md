# Rust Runtime Final Cleanup Plan

## Status

Completed.

## Outcome

The final cleanup tranche closed the remaining mixed-language runtime period and
left the repository with one active implementation stack.

Completed work:

1. removed the legacy runtime and operator code tree
2. removed alternate-language tests and packaging metadata
3. retired service examples and helper scripts that depended on the removed tree
4. rewrote the core operational documentation around the Rust-owned stack
5. converted the remaining kept shell scripts to shell-only helpers

## Acceptance

- no `.py` files remain in the repository
- no `pyproject.toml` or `uv.lock` remain in the repository
- core runtime, backtester, hub, and sidecar surfaces remain Rust-owned
- core documentation points at Rust entrypaths only
