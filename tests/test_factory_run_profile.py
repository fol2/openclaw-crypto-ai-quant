from __future__ import annotations

from factory_run import PROFILE_DEFAULTS, _parse_cli_args


def test_profile_smoke_sets_trials_and_candidate_counts() -> None:
    args = _parse_cli_args(["--run-id", "x", "--profile", "smoke"])
    assert args.tpe_trials == 2000
    assert args.shortlist_per_mode == 3
    assert args.shortlist_max_rank == 20
    assert args.num_candidates == 2


def test_profile_daily_sets_updated_values() -> None:
    args = _parse_cli_args(["--run-id", "x", "--profile", "daily"])
    assert args.tpe_trials == 200000
    assert args.shortlist_per_mode == 20
    assert args.shortlist_max_rank == 150
    assert args.num_candidates == 5


def test_profile_deep_backward_compat() -> None:
    args = _parse_cli_args(["--run-id", "x", "--profile", "deep"])
    assert args.tpe_trials == 2000000
    assert args.shortlist_per_mode == 30
    assert args.shortlist_max_rank == 400
    assert args.num_candidates == 10


def test_profile_weekly_values() -> None:
    args = _parse_cli_args(["--run-id", "x", "--profile", "weekly"])
    assert args.tpe_trials == 2000000
    assert args.shortlist_per_mode == 30
    assert args.shortlist_max_rank == 400
    assert args.num_candidates == 10


def test_profile_weekly_matches_deep() -> None:
    """weekly and deep must be identical (weekly is the canonical name)."""
    assert PROFILE_DEFAULTS["weekly"] == PROFILE_DEFAULTS["deep"]


def test_profile_does_not_override_explicit_values() -> None:
    args = _parse_cli_args(
        [
            "--run-id",
            "x",
            "--profile",
            "deep",
            "--tpe-trials",
            "123",
            "--shortlist-per-mode",
            "7",
            "--shortlist-max-rank",
            "77",
            "--num-candidates",
            "9",
        ]
    )
    assert args.tpe_trials == 123
    assert args.shortlist_per_mode == 7
    assert args.shortlist_max_rank == 77
    assert args.num_candidates == 9


def test_all_profiles_present_in_defaults() -> None:
    assert set(PROFILE_DEFAULTS.keys()) == {"smoke", "daily", "deep", "weekly"}
