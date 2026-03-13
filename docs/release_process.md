# Release Process

## Version Source

`VERSION` is the single human-edited version marker.

`tools/release/set_version.sh` updates:

- `VERSION`
- all tracked `Cargo.toml` crate versions

`tools/release/check_versions.sh` verifies:

- `VERSION` is valid SemVer
- every tracked `Cargo.toml` matches `VERSION`
- the current tag, when present, matches `VERSION`

## Typical Flow

```bash
tools/release/set_version.sh 0.1.1
tools/release/check_versions.sh
git commit -am "release: bump version to 0.1.1"
git tag v0.1.1
```
