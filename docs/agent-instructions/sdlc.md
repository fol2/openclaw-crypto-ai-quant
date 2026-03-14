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
2. Before PR review, run a documentation subagent for that PR.
3. Have the documentation subagent update the relevant repo docs, refresh the
   active plan doc when the task follows one, and refactor docs if coverage or
   structure needs improvement.
4. Run a reviewer subagent for that PR only after the documentation pass is
   complete.
5. Be patient with reviewer subagents. Let them finish, avoid duplicate reviewer
   launches, and do not interrupt them unless the review context has genuinely
   changed.
6. Merge only after the review is acceptable.
7. After merge, delete the PR branch locally and remotely if you created it.
8. After merge, remove the PR worktree(s) if you created them.
9. After merge, close any subagents opened specifically for that PR.
10. Move to the next task only after the merge and cleanup are complete.

## Documentation Sweep

- Treat documentation as part of the atomic PR, not follow-up housekeeping.
- The documentation subagent runs after implementation and validation are done,
  but before the reviewer subagent is launched.
- Its scope includes the active plan doc when the task follows one, the
  relevant repo docs, and any instruction or operator docs touched by the
  behavioural change.
- Ask it to make sure the docs cover what changed end to end and to refactor
  doc structure when the current layout hides or duplicates the new contract.
- Reviewers should receive the post-doc state so they assess the complete PR,
  not code plus stale docs.
- When an active plan is fully complete on `master`, close it out in a final
  docs PR or in the finishing PR itself: update its status/progress, move it to
  `plans/archived/` when appropriate, and refresh any active/archived plan
  indexes in the same atomic change.

## Long-Running Plan Hygiene

- When a plan spans many PRs or many hours, refresh from the latest `master`
  before starting the next atomic PR instead of assuming the earlier baseline is
  still valid.
- If concurrent merges changed auth, routing, or control-plane contracts, treat
  any older worktree as stale until you re-check the new `master` state.
- Prefer opening a fresh worktree from the latest `master` for the next PR over
  stacking more changes onto an old worktree when the baseline has shifted.
- If prototype work happened on a stale worktree, re-apply only the still-valid
  slices onto a fresh branch and discard the stale branch after the replacement
  PR lands.

## Safety Rules

- Never disable kill switches or weaken live guardrails without explicit user
  approval.
- Never auto-tune strategy configuration through unattended scripts. Use
  suggestion-only mode.
- Ask the user before risky live-trading actions or ambiguous operational
  decisions.
