---
name: Bug report
about: Report a bug in the Rust runtime, archival Python recovery path, backtester, or tooling
title: ''
labels: ''
assignees: ''
---

**Component**
Which surface is affected? Examples: `runtime/`, `engine/`, `strategy/`, `live/`, `exchange/`, `backtester/`, `tools/`.

**Mode**
Paper / Dry Live / Live / Backtest / Other.

**Describe the bug**
A clear and concise description of what happened and what you expected instead.

**Exact command or service path**
Include the precise command, script, or systemd unit you used.

```bash
./scripts/run_paper_lane.sh paper1
# or
./scripts/run_live.sh
# or
AI_QUANT_MODE=paper AI_QUANT_ALLOW_LEGACY_PYTHON_RUNTIME=1 python -m engine.daemon
```

If you used the archival Python recovery path, say so explicitly.

**To reproduce**
1. Set config or environment value(s):
2. Run the exact command above:
3. Observe the failure:

**Relevant logs**
Paste logs, tracebacks, or service output. Redact secrets.

```text
...
```

**Configuration**
Paste the relevant YAML fragment or environment overrides. Redact secrets.

```yaml
...
```

**Additional context**
Anything else that helps explain timing, market conditions, or whether this reproduced under the Rust runtime, the archival Python recovery path, or both.
