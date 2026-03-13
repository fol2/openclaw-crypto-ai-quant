# Contributing

## Branching

- keep production `master` clean
- use a dedicated worktree and branch for each logical change
- deliver through reviewed, atomic PRs

## Expectations

- preserve Rust-only ownership across runtime, tooling, and documentation
- prefer small, verifiable changes
- update documentation when contracts or commands change

## Verification

Run the narrowest useful checks first, then broaden:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

For backtester-focused work:

```bash
cargo test --manifest-path backtester/Cargo.toml
```

## Documentation

- use UK English spelling
- keep operational commands current
- remove historical material once it no longer helps the active Rust stack
