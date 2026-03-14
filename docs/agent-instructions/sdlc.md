# SDLC Instructions

Load this file when the task involves branching, worktrees, pull requests,
review flow, merge sequencing, or cleanup duties.

## Branching and Worktrees

- Production stays in `/home/fol2hk/openclaw-plugins/ai_quant` on `master`.
- Never change branches or edit application code in the production worktree.
- Make every code change in a separate non-`master` worktree.
- Never delete branches, worktrees, or subagents owned by another concurrent
  agent or session.

## Atomic PR Rule

- Every change must land as one atomic PR to `master`.
- Do not batch unrelated fixes, docs, refactors, or housekeeping into one PR
  unless they are inseparable parts of the same logical change.
- Do not commit ticket work directly on `master`.

## Mandatory PR Flow

1. Create one atomic PR to `master`.
2. Run a reviewer subagent for that PR.
3. Be patient with reviewer subagents. Let them finish, avoid duplicate reviewer
   launches, and do not interrupt them unless the review context has genuinely
   changed.
4. Merge only after the review is acceptable.
5. After merge, delete the PR branch locally and remotely if you created it.
6. After merge, remove the PR worktree(s) if you created them.
7. After merge, close any subagents opened specifically for that PR.
8. Move to the next task only after the merge and cleanup are complete.

## Safety Rules

- Never disable kill switches or weaken live guardrails without explicit user
  approval.
- Never auto-tune strategy configuration through unattended scripts. Use
  suggestion-only mode.
- Ask the user before risky live-trading actions or ambiguous operational
  decisions.
