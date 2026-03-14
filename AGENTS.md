# AI Agent Instructions for openclaw-crypto-ai-quant

This file is the short catalogue for AI coding agents working in this
repository. Keep it concise. Load deeper material on demand from the linked
docs and source files instead of expanding this file again.

## Non-Negotiable Guardrails

- The production worktree at `/home/fol2hk/openclaw-plugins/ai_quant` MUST stay
  on `master`.
- Never run branch-changing commands inside
  `/home/fol2hk/openclaw-plugins/ai_quant`.
- Never edit application code directly in the production worktree.
- Make every code change in a separate non-`master` worktree, for example
  `/home/fol2hk/openclaw-plugins/ai_quant_wt/<ticket-branch>`.
- Every change MUST land as one atomic PR to `master`: exactly one logical
  change per PR, with no unrelated batching.
- Do not commit ticket work directly on `master`.
- Mandatory PR flow for every successful code update:
  1. Create one atomic PR to `master`.
  2. Run a reviewer subagent for that PR.
  3. Be patient with PR reviewer subagents: let them finish, avoid duplicate
     reviewer launches, and do not interrupt them unless the review context has
     genuinely changed.
  4. Merge only after the review is acceptable.
  5. After merge, delete the PR branch locally and remotely if you created it.
  6. After merge, remove the PR worktree(s) if you created them.
  7. After merge, close any subagents opened specifically for that PR.
  8. Move to the next task only after the merge and cleanup are complete.
- Never delete branches, worktrees, or subagents owned by other concurrent
  agents or sessions.
- Never disable kill switches or weaken live-trading guardrails without explicit
  user approval.
- Never auto-tune strategy configuration through unattended scripts. Suggestion
  mode only.

## Active Stack

Active execution ownership lives in:

- `runtime/aiq-runtime` and `runtime/aiq-runtime-core`
- `backtester/`
- `ws_sidecar/`
- `hub/`

The repository is zero-Python in the active trust chain. Treat legacy
alternate-language runtime/tooling assumptions as obsolete unless the user asks
about archived history.

## Operational Defaults

- Config merge order:
  `Rust defaults <- global YAML <- symbols.<SYM> YAML <- live YAML`.
- `engine.interval` is not hot-reloadable; restart the affected service if it
  changes.
- Keep production on the `production` profile unless the user explicitly wants
  a parity/debug lane.
- `entry_min_confidence` defaults to `high`; set it explicitly to `low` in YAML
  if all confidence tiers should pass.
- Hyperliquid REST backfill is window-limited. Always query real DB coverage and
  keep backtests on identical date ranges across intervals.
- Rust indicator changes must match the external Python `ta` reference within
  `0.00005` absolute error.
- Use `behaviour_trace` plus `pipeline` inspection before changing runtime or
  parity logic.
- Use `parity_baseline` and `parity_exit_isolation` before ad hoc debug edits.
- Treat replay-alignment gates as fail-closed unless the user explicitly
  approves an emergency bypass.

## Load On Demand Catalogue

Load only the topic you need:

| Topic | Load on demand |
|---|---|
| Runtime ownership and surface map | `docs/current_authoritative_paths.md` |
| Operations and service commands | `docs/runbook.md` |
| Release/version process | `docs/release_process.md`, `VERSION`, `tools/release/*.sh` |
| Strategy lifecycle and promotion | `docs/strategy_lifecycle.md` |
| Success/risk metrics | `docs/success_metrics.md` |
| Repository architecture | `docs/ARCHITECTURE.md` |
| Canonical config example | `config/strategy_overrides.yaml.example` |
| Runtime CLI entrypoint | `runtime/aiq-runtime/src/main.rs` |
| Stage/behaviour plan resolution | `runtime/aiq-runtime-core/src/pipeline.rs` |
| Backtester engine | `backtester/crates/bt-core/src/engine.rs` |
| Decision kernel | `backtester/crates/bt-core/src/decision_kernel.rs` |
| Behaviour registry | `backtester/crates/bt-core/src/behaviour.rs`, `backtester/crates/bt-signals/src/behaviour.rs` |
| GPU sweep | `backtester/crates/bt-gpu/` |
| WS ingestion | `ws_sidecar/` |
| Dashboard and operator routes | `hub/` |
| Housekeeping/legacy notes | `docs/housekeeping/` |

## Quick Commands

Use these first-class commands before inventing alternatives:

```bash
cargo run -p aiq-runtime -- paper effective-config --lane paper1 --project-dir "$PWD" --json
cargo run -p aiq-runtime -- pipeline --mode paper --json
cargo run --manifest-path backtester/Cargo.toml -p bt-cli -- replay --candles-db candles_dbs/candles_1h.db
echo "close_only" > /tmp/ai-quant-kill
```

## Troubleshooting Order

1. Check the relevant systemd unit and logs.
2. Inspect effective config and resolved pipeline behaviour order.
3. Read emitted `behaviour_trace` before patching logic.
4. For indicator changes, run `dump-indicators` and compare against Python
   `ta`.
5. For release blockers, inspect
   `/tmp/openclaw-ai-quant/replay_gate/release_blocker.json`.

## Working Style

- Prefer existing YAML/config controls over code edits when the current
  contract already supports the change.
- Test risky changes in paper mode first.
- Preserve or extend tests when behaviour changes.
- Ask the user before risky live-trading actions or ambiguous operational
  decisions.
