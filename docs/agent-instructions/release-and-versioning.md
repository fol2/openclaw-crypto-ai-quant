# Release and Versioning Instructions

## Version Governance

- `VERSION` is the single human-edited version source.
- Use `tools/release/set_version.sh` to change versions.
- Use `tools/release/check_versions.sh` to verify version consistency.

## Release Automation

- The repository keeps `release-on-tag.yml` as the automatic release workflow.
- Non-release GitHub Actions are manual-only unless the user explicitly asks to
  reintroduce automatic remote CI.

## Load On Demand

For the full release workflow, read
[`docs/release_process.md`](../release_process.md).
