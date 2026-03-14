# Agent Instructions Catalogue

Use this folder for load-on-demand agent instructions. Start with the topic that
matches the current task instead of reading every file.

## Design Rules

- Keep each instruction file topic-scoped and independently loadable.
- Put only repository-wide essentials in `AGENTS.md`; keep detailed procedures
  in this folder.
- When a topic grows, split it into another module instead of inflating an
  existing file.
- Every module should point to deeper primary docs or source files when more
  detail is needed.
- Prefer additive growth here so the catalogue stays modular, flexible, and
  easy to extend.

## Instruction Files

| Topic | File |
|---|---|
| SDLC, PR flow, cleanup, and reviewer patience | `sdlc.md` |
| Runtime ownership, service wrappers, and operator commands | `runtime-and-operations.md` |
| Config merge order, parity lanes, behaviour traces, and debugging | `configuration-and-debugging.md` |
| Backtester ownership, replay/sweep commands, and indicator parity | `backtester-and-parity.md` |
| Version governance and release actions | `release-and-versioning.md` |

## Primary Source Material

These instruction files summarise the most important rules and point onward to
the authoritative operational docs and source files when deeper detail is
needed.
