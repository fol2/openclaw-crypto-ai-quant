# Claude Code Instructions

See [AGENTS.md](AGENTS.md) for AI coding assistant instructions for this project.

In particular, follow the repository SDLC guardrails strictly:

- every change must land as one atomic PR to `master`
- after merge, remove the PR branch both locally and remotely if you created it
- after merge, remove the PR worktree(s) if you created them
- after merge, close any subagents opened specifically for that PR
