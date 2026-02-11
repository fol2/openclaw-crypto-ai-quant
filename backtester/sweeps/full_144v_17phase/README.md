# 17-Phase 144v Sweep Pack

This directory contains a phase-based grid sweep pack generated from
`backtester/sweeps/full_144v.yaml`.

## Why this pack exists

- `full_144v.yaml` gives broad parameter coverage, but the full cartesian grid is intractable.
- This pack keeps **all 143 axes** while reducing value density per axis for practical grid runs.
- Total scale stays near the legacy 17-phase workflow.

## Current scale

- Phases: **17**
- Axes covered: **143** (all `full_144v` axes, each exactly once)
- Total combos per interval: **17,280**

## Files

- `p01_144v.yaml` ... `p17_144v.yaml`: phase specs
- `manifest.yaml`: phase metadata, combo counts, and path assignment

## Regeneration

From repo root:

```bash
python3 backtester/sweeps/generate_17phase_144v.py
```

This rewrites all phase files and `manifest.yaml` deterministically.
