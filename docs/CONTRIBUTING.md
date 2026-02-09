# Contributing to openclaw-crypto-ai-quant

We welcome contributions to openclaw-crypto-ai-quant. This guide will help you get started.

## Reporting Issues

Use GitHub Issues to report bugs or request features. When reporting issues, please include:

- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Backtest results if the issue is strategy-related

## Development Setup

Set up your development environment with these commands:

```bash
# Python environment
uv sync --dev

# Rust backtester
cd backtester && cargo build --release

# Rust WebSocket sidecar
cd ws_sidecar && cargo build --release
```

## Code Style

### Python

Python code style is enforced by `ruff` with line-length 120 and Python 3.12 target:

```bash
uv run ruff check engine strategy exchange live tools tests
uv run ruff format engine strategy exchange live tools tests
```

### Rust

Rust code must pass `cargo fmt` and `cargo clippy`:

```bash
cargo fmt
cargo clippy -- -D warnings
```

## Testing

### Python

Run Python tests with:

```bash
uv run pytest
```

100% coverage is required for `sqlite_logger` and `heartbeat` modules.

### Rust

Run Rust tests with:

```bash
cd backtester && cargo test
```

### Strategy Changes

Strategy changes must include backtest evidence (sweep results or replay comparison).

## Pull Requests

1. Fork the repository and create a feature branch
2. Keep PRs focused — one change per PR
3. Ensure all tests pass before submitting
4. Write clear commit messages (conventional commits preferred: `feat:`, `fix:`, `refactor:`, `docs:`)
5. For strategy parameter changes: include before/after backtest metrics in the PR description

## Strategy Changes

Strategy parameter changes affect real trading. Please:

- Run backtests with `mei-backtester replay` showing before/after metrics
- For significant changes, run a parameter sweep and include top results
- Never submit untested parameter changes

## Security

- NEVER commit secrets (private keys, wallet addresses, API keys)
- Use `systemd/ai-quant-live.env.example` and `config/secrets.json.example` as templates
- Report security vulnerabilities privately via GitHub Security Advisories

## Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves significant risk. The authors are not responsible for any financial losses incurred through use of this software.
