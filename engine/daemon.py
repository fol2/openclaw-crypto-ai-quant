"""Retired Python daemon entrypoint.

Production paper and live execution now run through the Rust `aiq-runtime`
service wrappers:

- `scripts/run_paper_lane.sh`
- `scripts/run_live.sh`

This module is intentionally kept as a small fail-fast shim so stale commands
error clearly instead of silently reviving the old Python runtime path.
"""

from __future__ import annotations


_RETIRED_MESSAGE = (
    "Python daemon runtime is retired. Use `scripts/run_paper_lane.sh` or "
    "`aiq-runtime paper daemon` for paper, and `scripts/run_live.sh` or "
    "`aiq-runtime live daemon` for live."
)


def main() -> None:
    raise SystemExit(_RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
