# AI Quant — Dev Tooling (uv / ruff / pytest)

This folder is a standalone Python project managed by `uv`:

- Project config: `dev/ai_quant/pyproject.toml`
- Lockfile: `dev/ai_quant/uv.lock`
- Dev venv (uv-managed): `dev/ai_quant/.venv/` (ignored by git)
- Runtime venv (systemd scripts): `dev/ai_quant/venv/` (unchanged; used by `run_live.sh` / `run_paper.sh`)

## Setup

```bash
cd workspace/dev/ai_quant
uv sync --dev
```

## Lint / Format

```bash
cd workspace/dev/ai_quant
uv run ruff check quant_trader_v5 tests
uv run ruff format quant_trader_v5 tests
```

## Tests

```bash
cd workspace/dev/ai_quant
uv run pytest
```

Notes:
- `pytest` enforces **100% coverage** for `quant_trader_v5/sqlite_logger.py` and `monitor/heartbeat.py`.

## Safety gate

The live-trader quality gate uses the same Ruff config:

```bash
bash workspace/scripts/validate_quant.sh
```
