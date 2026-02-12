# v8 Parallel Instance (major-v8)

This document describes how to run a fully isolated **v8 paper instance** in parallel with the production `master`
deployment, including a nightly **factory cycle** (sweep → validate → deploy).

The v8 instance is intentionally namespaced:

- systemd unit names end with `-v8`
- WS sidecar socket is `%t/openclaw-ai-quant-ws-v8.sock`
- monitor port is `61018`

## Safety Rules (Do Not Break These)

1. Do not switch branches inside the live production working directory.
2. Do not reuse production sockets/ports for v8.
3. Do not overwrite production binaries in `~/.local/bin/`.

## 1) Create A Dedicated Runtime Worktree

Use a long-lived worktree for running v8 so it can be modified by the factory/deploy loop without polluting your dev checkout.

Example:

```bash
git worktree add /home/fol2hk/openclaw-plugins/ai_quant_wt/major-v8-runtime origin/major-v8
cd /home/fol2hk/openclaw-plugins/ai_quant_wt/major-v8-runtime
uv sync --dev
```

## 2) Build v8 Binaries Without Touching Production

### WS sidecar

Build and install to a v8-scoped filename:

```bash
cd ws_sidecar
cargo build --release
install -m 0755 target/release/openclaw_ai_quant_ws_sidecar ~/.local/bin/openclaw-ai-quant-ws-sidecar-v8
```

### Backtester (GPU)

```bash
cd /home/fol2hk/openclaw-plugins/ai_quant_wt/major-v8-runtime
python3 tools/build_mei_backtester.py --gpu
```

This produces:

- `backtester/dist/mei-backtester-gpu`

## 3) Configure v8 Environment Files

Copy:

```bash
cp systemd/ai-quant-universe-v8.env.example ~/.config/openclaw/ai-quant-universe-v8.env
cp systemd/ai-quant-v8.env.example ~/.config/openclaw/ai-quant-v8.env
cp systemd/ai-quant-monitor-v8.env.example ~/.config/openclaw/ai-quant-monitor-v8.env
```

Then edit:

- `~/.config/openclaw/ai-quant-universe-v8.env` (symbols / intervals)
- `~/.config/openclaw/ai-quant-v8.env` (Discord channel, kill-switch file, retention)
- `~/.config/openclaw/ai-quant-monitor-v8.env` (bind + port)

## 4) Install v8 systemd User Units

Copy the example units and substitute `$PROJECT_DIR` with your runtime worktree path:

```bash
PROJECT_DIR=/home/fol2hk/openclaw-plugins/ai_quant_wt/major-v8-runtime
mkdir -p ~/.config/systemd/user

for f in systemd/openclaw-ai-quant-*-v8.*.example; do
  out=~/.config/systemd/user/$(basename "$f" .example)
  sed "s|\\$PROJECT_DIR|$PROJECT_DIR|g" "$f" > "$out"
done

systemctl --user daemon-reload
```

Create per-instance strategy YAML files (candidate lanes + proven lane):

```bash
cp "$PROJECT_DIR/config/strategy_overrides.yaml" "$PROJECT_DIR/config/strategy_overrides.paper1.yaml"
cp "$PROJECT_DIR/config/strategy_overrides.yaml" "$PROJECT_DIR/config/strategy_overrides.paper2.yaml"
cp "$PROJECT_DIR/config/strategy_overrides.yaml" "$PROJECT_DIR/config/strategy_overrides.paper3.yaml"
cp "$PROJECT_DIR/config/strategy_overrides.yaml" "$PROJECT_DIR/config/strategy_overrides.livepaper.yaml"
```

Set each trader service to its own YAML via `AI_QUANT_STRATEGY_YAML`:

- `openclaw-ai-quant-trader-v8-paper1` → `strategy_overrides.paper1.yaml`
- `openclaw-ai-quant-trader-v8-paper2` → `strategy_overrides.paper2.yaml`
- `openclaw-ai-quant-trader-v8-paper3` → `strategy_overrides.paper3.yaml`
- `openclaw-ai-quant-trader-v8-livepaper` → `strategy_overrides.livepaper.yaml`

The factory service should run with:

- `--candidate-count 3`
- `--candidate-services paper1,paper2,paper3`
- `--candidate-yaml-paths ...paper1.yaml,...paper2.yaml,...paper3.yaml`
- `--enable-livepaper-promotion --livepaper-service ... --livepaper-yaml-path ...livepaper.yaml`
- `--max-age-fail-hours` and `--funding-max-stale-symbols` (or matching `AI_QUANT_FUNDING_*` env vars) for isolated stale-funding tolerance.

Enable and start (v8 only):

```bash
systemctl --user enable --now openclaw-ai-quant-ws-sidecar-v8.service
systemctl --user enable --now openclaw-ai-quant-trader-v8-paper1.service
systemctl --user enable --now openclaw-ai-quant-monitor-v8.service

systemctl --user enable --now openclaw-ai-quant-funding-v8.timer
systemctl --user enable --now openclaw-ai-quant-factory-v8.timer
systemctl --user enable --now openclaw-ai-quant-prune-runtime-logs-v8.timer
```

## 5) Observe

Monitor UI:

- `http://127.0.0.1:61018/`

Logs:

```bash
journalctl --user -u openclaw-ai-quant-ws-sidecar-v8.service -f
journalctl --user -u openclaw-ai-quant-trader-v8-paper1.service -f
journalctl --user -u openclaw-ai-quant-factory-v8.service -f
```

Artifacts:

- `artifacts/YYYY-MM-DD/run_<run_id>/`
- `artifacts/registry/registry.sqlite`
- `artifacts/deployments/paper/.../deploy_event.json`
