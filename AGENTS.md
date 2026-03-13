# AI Agent Instructions for openclaw-crypto-ai-quant

This guide provides comprehensive instructions for AI coding assistants (Claude Code, Copilot, Cursor, etc.) working on this codebase.

## Production Safety and Branching Guardrails (MUST FOLLOW)

These guardrails exist because the production branch (`master`) is live and trades real money.

Do not change this section unless the user explicitly asks to update `AGENTS.md`.

- The production worktree at `/home/fol2hk/openclaw-plugins/ai_quant` MUST stay checked out on `master` at all times.
- Never run `git checkout`, `git switch`, or any branch-changing command inside `/home/fol2hk/openclaw-plugins/ai_quant`.
- Never edit application code in `/home/fol2hk/openclaw-plugins/ai_quant` directly. Treat this folder as production-run only.
- All code changes MUST be done in a separate worktree folder on a non-`master` branch (for example `/home/fol2hk/openclaw-plugins/ai_quant_wt/<ticket-branch>`).
- All AQC ticket work MUST be delivered through atomic PRs targeting `master` (one logical change per PR, no batching of unrelated fixes).
- Do not commit ticket changes directly on `master`. Use a dedicated ticket branch/worktree, open a PR, complete review, then merge.
- Multi-agent collaboration is expected: multiple active worktrees/branches/PRs may coexist at the same time.
- Mandatory PR flow for every successful code update:
  1. Create an atomic PR to `master`.
  2. Run a reviewer subagent to review the PR.
  3. Merge only after the review is acceptable.
  4. Continue to the next task only after merge completion.
  5. After merge, clean up only branches/worktrees created by the current agent/session (local + remote); never delete branches/worktrees owned by other concurrent agents.

## Project Overview

This is a Rust-native crypto perpetual futures trading stack for Hyperliquid DEX.

Active execution ownership now lives in:

- `runtime/aiq-runtime` for paper/live daemon control, manifests, snapshots, and pipeline inspection
- `backtester/` for replay, sweeps, GPU acceleration, and indicator validation
- `ws_sidecar/` for market-data ingestion and candle persistence
- `hub/` for operator-facing dashboard, service inspection, and backtest controls

The repository is zero-Python. Historical alternate-language runtime and tool
surfaces are no longer part of the active trust chain.

## Architecture

### Core Components

- **Runtime CLI**: `runtime/aiq-runtime` — paper/live daemon entrypoints, manifests, effective-config, snapshots
- **Runtime core**: `runtime/aiq-runtime-core` — stage plan, behaviour plan, bootstrap contracts
- **Decision kernel**: `backtester/crates/bt-core/src/decision_kernel.rs` — deterministic transition logic
- **Behaviour registry**: `backtester/crates/bt-core/src/behaviour.rs` and `backtester/crates/bt-signals/src/behaviour.rs`
- **Backtester CLI**: `backtester/crates/bt-cli` — replay, sweep, indicator dump
- **GPU sweep**: `backtester/crates/bt-gpu` — CUDA sweep acceleration and parity fixture tooling
- **Risk primitives**: `backtester/crates/risk-core`
- **Market-data sidecar**: `ws_sidecar/`
- **Operator dashboard**: `hub/`

### Databases

- **Paper DB**: `trading_engine.db` (SQLite)
- **Live DB**: `trading_engine_live.db` (SQLite)
- **Candle DBs**: `candles_dbs/candles_{1m,3m,5m,15m,30m,1h}.db`
- **Candle partitions**: `candles_dbs/partitions/{interval}/candles_{interval}_YYYY-MM.db`
- **Funding rates DB**: `candles_dbs/funding_rates.db`
- **BBO snapshots DB**: `candles_dbs/bbo_snapshots.db` (optional)
- **Universe history DB**: `candles_dbs/universe_history.db`
- **Market data DB**: `market_data.db`
- **Runtime snapshots**: JSON exports produced by `aiq-runtime snapshot ...`

### Candle Database Coverage

Hyperliquid only provides roughly the last 5,000 bars per interval for REST backfill (`candleSnapshot`). The figures below are the approximate *backfill window* implied by that limit.

The WS sidecar can retain an append-only local history beyond that window for selected intervals (see `AI_QUANT_CANDLE_PRUNE_DISABLE_INTERVALS`), so do not assume these DBs have fixed coverage. Always query the actual min/max timestamps when choosing backtest ranges.

When running backtests across multiple intervals, ALWAYS restrict all runs to the same date range for valid comparisons:

- `1m`: 3.5 days
- `3m`: 10 days
- `5m`: 17 days
- `15m`: 52 days
- `30m`: 104 days
- `1h`: 208 days

## Configuration System

### Merge Order

Configuration values are merged in this order (later values override earlier):

```
Rust defaults ← global YAML ← symbols.<SYM> YAML ← live YAML (if live mode)
```

### Key Configuration Sections

#### `trade`
- `sl_atr_mult`, `tp_atr_mult`: Stop-loss and take-profit multipliers
- `leverage`: Base leverage
- `allocation_pct`: Margin allocated per position (notional = margin × leverage)
- `min_atr_pct`: Minimum ATR percentage threshold (entries below this are skipped)
- `reentry_cooldown_minutes`: Cooldown period before re-entering same symbol
- `enable_pyramiding`: Enable/disable pyramid ADD orders
- `enable_partial_tp`: Enable/disable partial take-profit
- `enable_dynamic_sizing`: Dynamically scale position size by ADX/volatility

#### `market_regime`
- `enable_regime_filter`: Enable market regime filter
- `enable_auto_reverse`: Auto-reverse signals based on market breadth
- `bearish_breadth_threshold_pct`: Threshold for bearish regime detection
- `bullish_breadth_threshold_pct`: Threshold for bullish regime detection

#### `filters`
Entry gate filters:
- `enable_ranging_regime_filter`: Block entries in ranging markets
- `enable_anomaly_filter`: Block entries with anomaly detection
- `enable_btc_alignment_filter`: Require BTC direction alignment
- `enable_market_breadth_filter`: Filter by market-wide breadth

#### `thresholds.entry`
- `min_adx`: Minimum ADX required for entry
- `pullback_pct`: Required pullback percentage
- `slow_drift`: Slow drift threshold

#### `indicators`
Window sizes for technical indicators:
- `ema_fast`, `ema_slow`: EMA periods
- `adx_window`: ADX period
- `bb_window`: Bollinger Bands period
- `bb_std`: Bollinger Bands standard deviation

#### `engine`
- `interval`: Candle interval (e.g., `1h`, `3m`, `5m`)
- `entry_interval`: Interval for entry signal generation
- `exit_interval`: Interval for exit signal generation

**IMPORTANT**: The `engine.interval` parameter is NOT hot-reloadable. Changing it requires a service restart.

#### `runtime`
- `profile`: Active runtime profile. Empty means follow `pipeline.default_profile`.
- `state_backend`: Persistence backend identifier
- `audit_sink`: Audit sink identifier

#### `pipeline`
- `default_profile`: Fallback profile when `runtime.profile` is empty
- `profiles.<name>.ranker`: Stage-level ranking contract
- `profiles.<name>.stage_order`: Full stage permutation override
- `profiles.<name>.enabled_stages` / `disabled_stages`: Stage allow/block lists
- `profiles.<name>.behaviours.*`: Behaviour-level order/allow/block controls

#### `pipeline.profiles.<name>.behaviours`

Current first-class behaviour groups are:

- `gates`
- `signal_modes`
- `signal_confidence`
- `exits`
- `engine`
- `entry_sizing`
- `entry_progression`
- `risk`

Each group supports:

- `order`
- `enabled`
- `disabled`

### Parity Profiles

Shipped example configs now include two opt-in parity lanes:

- `parity_baseline`: explicit production-like behaviour ordering with broker/fill stages disabled
- `parity_exit_isolation`: parity baseline plus disabled exit modifiers to isolate base stop-loss, trailing, and full take-profit paths

Keep production on `production` unless you are intentionally running a debug or parity lane.

### Configuration Defaults

The code-level default for `entry_min_confidence` is `"high"`, which blocks 70-90% of entry signals. To allow all confidence levels (low/medium/high), explicitly set `entry_min_confidence: low` in your YAML configuration.

### Strategy Mode Switching

The engine supports an optional strategy-mode overlay selected via `AI_QUANT_STRATEGY_MODE`:

- `primary`: 30m/5m
- `fallback`: 1h/5m
- `conservative`: 1h/15m
- `flat`: safety profile

Mode overlays are defined under `modes:` in `config/strategy_overrides.yaml`. With `AI_QUANT_MODE_SWITCH_ENABLE=1`, the daemon auto-steps-down on kill events.

## Signal Flow

### Signal Generation and Processing

1. `aiq-runtime` loads the merged strategy config and resolves the active stage + behaviour plan.
2. `ws_sidecar` and SQLite sources provide candle, funding, and optional market-data context.
3. Gate evaluation and signal generation run through the shared Rust signal path.
4. Entry sizing, progression, and risk checks honour the resolved behaviour plan.
5. Exit evaluation runs immediately per symbol, including configured stop-loss, trailing, take-profit, and smart-exit ordering.
6. Ranking, OMS transitions, and broker execution only run when their stages are enabled for the active profile.

### Signal Ranking Algorithm

Entry signals are ranked using this formula:

```
score = confidence_rank * 100 + ADX
```

Where:
- High confidence: `confidence_rank = 2` → `score = 200 + ADX`
- Medium confidence: `confidence_rank = 1` → `score = 100 + ADX`
- Low confidence: `confidence_rank = 0` → `score = 0 + ADX`

Tiebreaker: Symbol name alphabetical order (ascending)

**Important**: Exits are executed immediately per-symbol and are NOT ranked. Pyramid ADD orders are also executed immediately when a same-direction position exists, NOT ranked.

### Audit Trail

Paper/live runtime outputs and decision diagnostics carry behaviour traces for debugging. Use them to confirm:

- which gate behaviours ran or were disabled
- which signal mode produced the entry
- which exit behaviour triggered, skipped, or was disabled
- whether risk cooldowns or exposure guards blocked the action

## Rust Backtester

### Location and Structure

- **Location**: `backtester/` directory
- **Cargo workspace crates**:
  - `bt-core`: Simulation engine, decision kernel, indicators, config, state management
  - `bt-signals`: Signal generation (entry logic, gates, confidence) — shared across all paths
  - `bt-data`: SQLite candle loader (supports multi-DB and partition directories)
  - `bt-cli`: CLI entry point (replay, sweep, dump-indicators)
  - `bt-gpu`: CUDA GPU sweep + TPE Bayesian optimisation
  - `risk-core`: Shared, pure risk primitives (entry sizing, confidence tiers)

### Building

```bash
# CPU-only
cd backtester && cargo build --release -p bt-cli

# GPU build
cd backtester && cargo build --release -p bt-cli --features gpu
```

On WSL2, you may need to set:
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Commands

- `replay`: Single-run backtest (supports `--init-state` for exported positions)
- `sweep`: Parallel parameter sweep (CPU or `--gpu`)
- `dump-indicators`: Export indicators for validation
- `--universe-filter`: Filter symbols by listing history from universe DB
- `--parity-mode`: GPU parity lanes (identical-symbol-universe / production)

### Indicator Parity Requirements

All indicators in the Rust backtester MUST match the external Python `ta`
reference implementation within **0.00005 absolute error**.

#### Critical Implementation Details

1. **EMA seeding**: Use `adjust=False` (pandas `ewm` method), NOT SMA seed. This matches `ta.trend.ema_indicator`.

2. **`vol_trend`**: Calculate as `SMA(vol, 5) > SMA(vol, 20)`, NOT `current_vol > SMA(vol, 5)`.

3. **`bb_width`**: Calculate as `(bb_high - bb_low) / Close`, NOT `(bb_high - bb_low) / bb_middle`.

### Validation Workflow

When modifying indicator logic in Rust:

1. Run `dump-indicators` to export Rust output
2. Compare against Python `ta` library output
3. Ensure all indicators match within 0.00005 absolute error
4. Document any intentional deviations

## Risk Manager

`engine/risk.py` → `RiskManager` provides runtime safety:

| Control | Env Variable | Default |
|---------|-------------|---------|
| Kill-switch (env) | `AI_QUANT_KILL_SWITCH` | (unset) |
| Kill-switch (file) | `AI_QUANT_KILL_SWITCH_FILE` | (unset) |
| Max drawdown % | `AI_QUANT_RISK_MAX_DRAWDOWN_PCT` | 0 (disabled) |
| Entry rate limit (global) | `AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN` | 30 |
| Entry rate limit (per-symbol) | `AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN_PER_SYMBOL` | 6 |
| Exit rate limit | `AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN` | 120 |
| Cancel rate limit | `AI_QUANT_RISK_MAX_CANCELS_PER_MIN` | 120 |
| Max notional per window | `AI_QUANT_RISK_MAX_NOTIONAL_PER_WINDOW_USD` | 0 (disabled) |
| Min order gap | `AI_QUANT_RISK_MIN_ORDER_GAP_MS` | 0 (disabled) |

Kill-switch modes: `close_only` (no new entries, exits allowed) or `halt_all` (no orders at all).

## Services (systemd)

### Service Management

- **Paper trader (primary)**: `systemctl --user restart openclaw-ai-quant-trader-v8-paper1`
- **Paper trader (candidate #2)**: `systemctl --user restart openclaw-ai-quant-trader-v8-paper2`
- **Paper trader (candidate #3)**: `systemctl --user restart openclaw-ai-quant-trader-v8-paper3`
- **Live trader**: `systemctl --user restart openclaw-ai-quant-live-v8`
- **WebSocket sidecar**: `systemctl --user restart openclaw-ai-quant-ws-sidecar`
- **Monitor**: `systemctl --user restart openclaw-ai-quant-monitor`
- **Log pruning timer**: `systemctl --user restart openclaw-ai-quant-prune-runtime-logs-v8.timer`
- **Replay gate timer**: `systemctl --user restart openclaw-ai-quant-replay-alignment-gate.timer`

### Hot-reload Behaviour

- **YAML configuration changes**: Hot-reload automatically (no restart required)
- **Engine interval changes**: Require service restart (not hot-reloadable)
- **Strategy mode changes** that affect `engine.interval`: Require service restart

## Development

### Rust Development

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Backtester-specific validation:

```bash
cargo test --manifest-path backtester/Cargo.toml -p bt-core
cargo test --manifest-path backtester/Cargo.toml -p bt-gpu --features codegen --test gpu_decision_parity -- --nocapture
```

## Critical Rules for AI Assistants

### Security

1. **NEVER commit sensitive files**: `secrets.json`, `.env`, or any file containing private keys, wallet addresses, or API keys.

2. **Live trading safety**: Never disable kill switches (`AI_QUANT_KILL_SWITCH`, `AI_QUANT_HARD_KILL_SWITCH`) without explicit user confirmation.

### Configuration Management

3. **NEVER auto-tune strategy config via automated scripts**. Always use suggestion-only mode. This prevents accidental deployment of untested parameters. (Lesson from emergency rollback incident.)

4. **Code defaults vs YAML overrides**: The code-level default for `entry_min_confidence` is `"high"`, which blocks most entries. Remember to set `entry_min_confidence: low` in YAML to allow all confidence levels.

5. **Production defaults stay on `production`**. Parity profiles are opt-in and must never become the implicit live profile.

### Backtesting

6. **Date range consistency**: When comparing backtest results across different intervals, ALWAYS restrict all runs to the same date range. Different candle databases have different coverage spans.

7. **Indicator parity**: Backtester indicator calculations MUST match the Python `ta` reference implementation within 0.00005 absolute error. Validate using `dump-indicators` after any indicator changes.

### Debugging

8. **Behaviour traces are first-class diagnostics**. When parity or runtime behaviour looks wrong, inspect the resolved pipeline profile and the emitted `behaviour_trace` before changing code.

9. **Use parity profiles for isolation before patching logic**. `parity_baseline` confirms stage/behaviour resolution; `parity_exit_isolation` strips exit modifiers so you can focus on base stop-loss/trailing/full TP behaviour.

10. **When validating Rust indicators**: Always use `dump-indicators` and compare against Python `ta` output. Document the validation process.

### Deployment

11. **Replay alignment gate**: Treat replay-alignment blockers as fail-closed. Only bypass them in explicitly approved emergency workflows.

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `config/strategy_overrides.yaml.example` | Canonical runtime/backtester config example, including parity profiles |
| `runtime/aiq-runtime-core/src/pipeline.rs` | Stage profile resolution and built-in stage-debug/parity contracts |
| `runtime/aiq-runtime/src/main.rs` | Runtime CLI entrypoint |
| `backtester/crates/bt-core/src/engine.rs` | Backtester simulation engine |
| `backtester/crates/bt-core/src/decision_kernel.rs` | Shared decision kernel |
| `backtester/crates/bt-core/src/behaviour.rs` | Behaviour groups resolved below the stage plan |
| `backtester/crates/bt-signals/` | Signal generation and behaviour traces |
| `backtester/crates/bt-gpu/src/gpu_host.rs` | CUDA GPU dispatch |
| `backtester/crates/risk-core/` | Rust risk primitives |
| `ws_sidecar/` | Market-data ingestion and persistence |
| `hub/` | Dashboard, system status, and operator controls |
| `systemd/` | Service and timer examples for the Rust-owned surfaces |

## Additional Resources

- **Version**: `VERSION` (single source of truth; all Cargo manifests and release checks must match)
- **Operations runbook**: `docs/runbook.md`
- **Strategy lifecycle**: `docs/strategy_lifecycle.md`
- **Success metrics**: `docs/success_metrics.md`
- **Release process**: `docs/release_process.md`

## Common Patterns

### Reading Strategy Config

```bash
cargo run -p aiq-runtime -- paper effective-config --lane paper1 --project-dir "$PWD" --json
cargo run -p aiq-runtime -- live effective-config --project-dir "$PWD" --json
```

### Inspecting the Active Pipeline

```bash
cargo run -p aiq-runtime -- pipeline --mode paper --json
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_baseline --json
cargo run -p aiq-runtime -- pipeline --mode paper --profile parity_exit_isolation --json
```

### Running Backtests

```bash
# Single run
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- replay --candles-db candles_dbs/candles_1h.db

# Parameter sweep
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- sweep --sweep-config backtester/sweeps/smoke.yaml

# Replay with exported state
cargo run -p aiq-runtime -- snapshot export-paper --db trading_engine.db --output /tmp/state.json
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- replay --init-state /tmp/state.json --trades
```

### Emergency Stop

```bash
# File-based kill-switch (no restart needed)
echo "close_only" > /tmp/ai-quant-kill

# Inspect live service state
cargo run -p aiq-runtime -- live manifest --project-dir "$PWD" --json
```

## Troubleshooting

### Common Issues

1. **Service won't start**: Check systemd logs with `journalctl --user -u openclaw-ai-quant-trader-v8-paper1 -f`

2. **YAML changes not taking effect**: Verify mtime polling is working. Check engine logs for "Config reloaded" messages.

3. **Profile does not behave as expected**: Run `cargo run -p aiq-runtime -- pipeline --mode paper --profile <name> --json` and inspect `behaviours.*` before changing code.

4. **GPU build fails**: Ensure CUDA toolkit is installed. On WSL2, set `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.

5. **Unexpected entry filtering**: Check `entry_min_confidence` setting. Code default is `"high"`, which blocks most entries. Set to `"low"` in YAML to allow all confidence levels.

6. **Unexpected exit reason**: Inspect the emitted `behaviour_trace` from paper/live output to confirm which exit behaviour triggered or was disabled.

7. **OMS intent failures**: If live entries fail, inspect the live manifest/effective-config output and review runtime diagnostics for duplicate or blocked intents.

8. **Deployment blocked by replay gate**: Check `/tmp/openclaw-ai-quant/replay_gate/release_blocker.json`. Use `--ignore-replay-gate` only for emergency workflows.

## Best Practices

1. **Always read the code before suggesting changes**. Understand the existing implementation.

2. **Test configuration changes in paper mode first** before deploying to live trading.

3. **Validate backtester changes** using `dump-indicators` and comparison against Python `ta` library.

4. **Use parity profiles before forking logic**. Prefer `parity_baseline` or `parity_exit_isolation` over ad hoc debug edits.

5. **Preserve test coverage**. Add or extend Rust tests when changing config resolution, behaviour ordering, or diagnostics.

6. **Use YAML for strategy tuning**, not code edits, whenever the existing behaviour/config contract already supports the change.

7. **When in doubt, ask the user** before making risky changes (especially for live trading or automated parameter tuning).

8. **Version consistency**: When bumping versions, use `tools/release/set_version.sh` and verify with `tools/release/check_versions.sh`.
