"""End-to-end integration tests for the factory pipeline.

Validates profile resolution, candidate promotion logic (including edge cases),
and the weekly/deep alias equivalence — all without GPU or real candle data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from factory_run import (
    PROFILE_DEFAULTS,
    _parse_cli_args,
    _promote_candidates,
    _write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    *,
    config_id: str,
    total_pnl: float,
    max_drawdown_pct: float,
    profit_factor: float,
    win_rate: float = 0.55,
    total_trades: int = 120,
    tmp_path: Path,
) -> dict[str, Any]:
    """Create a candidate dict with a real YAML config file on disk."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{config_id}.yaml"
    cfg_path.write_text(f"# stub config for {config_id}\nglobal:\n  engine:\n    interval: 1h\n", encoding="utf-8")
    return {
        "config_id": config_id,
        "config_path": str(cfg_path),
        "total_pnl": total_pnl,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": total_trades,
    }


# ===========================================================================
# (a) Profile resolution tests
# ===========================================================================


class TestProfileResolution:
    """Verify _parse_cli_args with --profile produces correct defaults for ALL fields."""

    def test_daily_profile_full_defaults(self) -> None:
        args = _parse_cli_args(["--run-id", "test", "--profile", "daily"])
        assert args.tpe_trials == 1_000_000
        assert args.shortlist_per_mode == 20
        assert args.shortlist_max_rank == 200
        assert args.num_candidates == 5

        # Non-profile defaults should also be present.
        assert args.profile == "daily"
        assert args.interval == "1h"
        assert args.sort_by == "balanced"
        assert args.tpe_batch == 256
        assert args.tpe_seed == 42
        assert args.promote_count == 3
        assert args.promote_dir == "promoted_configs"
        assert args.score_min_trades == 30

    def test_smoke_profile_full_defaults(self) -> None:
        args = _parse_cli_args(["--run-id", "test", "--profile", "smoke"])
        assert args.tpe_trials == 2_000
        assert args.shortlist_per_mode == 3
        assert args.shortlist_max_rank == 20
        assert args.num_candidates == 2

    def test_deep_profile_full_defaults(self) -> None:
        args = _parse_cli_args(["--run-id", "test", "--profile", "deep"])
        assert args.tpe_trials == 5_000_000
        assert args.shortlist_per_mode == 40
        assert args.shortlist_max_rank == 500
        assert args.num_candidates == 10

    def test_weekly_profile_full_defaults(self) -> None:
        args = _parse_cli_args(["--run-id", "test", "--profile", "weekly"])
        assert args.tpe_trials == 5_000_000
        assert args.shortlist_per_mode == 40
        assert args.shortlist_max_rank == 500
        assert args.num_candidates == 10

    def test_explicit_cli_overrides_profile(self) -> None:
        args = _parse_cli_args([
            "--run-id", "test",
            "--profile", "daily",
            "--tpe-trials", "999",
            "--shortlist-per-mode", "7",
            "--shortlist-max-rank", "77",
            "--num-candidates", "2",
        ])
        assert args.tpe_trials == 999
        assert args.shortlist_per_mode == 7
        assert args.shortlist_max_rank == 77
        assert args.num_candidates == 2

    def test_unknown_profile_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_cli_args(["--run-id", "test", "--profile", "nonexistent"])


# ===========================================================================
# (b) Promotion integration tests
# ===========================================================================


class TestPromotionIntegration:
    """Test _promote_candidates() with realistic metrics."""

    def test_primary_has_best_balanced_score(self, tmp_path: Path) -> None:
        """Primary role should go to the candidate with the highest balanced score."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="high_balanced",
                total_pnl=500.0,
                max_drawdown_pct=0.10,
                profit_factor=2.5,
                win_rate=0.62,
                total_trades=200,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="medium_balanced",
                total_pnl=300.0,
                max_drawdown_pct=0.05,
                profit_factor=1.8,
                win_rate=0.58,
                total_trades=150,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="low_balanced",
                total_pnl=100.0,
                max_drawdown_pct=0.02,
                profit_factor=1.2,
                win_rate=0.52,
                total_trades=80,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(
            candidates,
            run_dir=run_dir,
            promote_dir="promoted_configs",
            promote_count=3,
        )

        assert not result["skipped"]
        assert result["promoted_count"] == 3

        primary = result["roles"]["primary"]
        assert primary["config_id"] == "high_balanced"
        assert (Path(primary["promoted_path"])).exists()

        # Verify primary.yaml file exists on disk
        assert (run_dir / "promoted_configs" / "primary.yaml").is_file()

    def test_fallback_has_lowest_dd_among_positive_pnl(self, tmp_path: Path) -> None:
        """Fallback should be the candidate with lowest DD among positive PnL."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="high_dd",
                total_pnl=500.0,
                max_drawdown_pct=0.25,
                profit_factor=2.5,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="low_dd",
                total_pnl=100.0,
                max_drawdown_pct=0.02,
                profit_factor=1.2,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="mid_dd",
                total_pnl=200.0,
                max_drawdown_pct=0.08,
                profit_factor=1.5,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        fallback = result["roles"]["fallback"]
        assert fallback["config_id"] == "low_dd"
        assert (run_dir / "promoted_configs" / "fallback.yaml").is_file()

    def test_conservative_has_absolute_lowest_dd(self, tmp_path: Path) -> None:
        """Conservative should have the absolute lowest max_drawdown_pct among positive PnL."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="big_dd",
                total_pnl=800.0,
                max_drawdown_pct=0.30,
                profit_factor=3.0,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="tiny_dd",
                total_pnl=50.0,
                max_drawdown_pct=0.01,
                profit_factor=1.1,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="medium_dd",
                total_pnl=200.0,
                max_drawdown_pct=0.12,
                profit_factor=1.7,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        conservative = result["roles"]["conservative"]
        assert conservative["config_id"] == "tiny_dd"
        assert (run_dir / "promoted_configs" / "conservative.yaml").is_file()

    def test_all_three_roles_written_and_metadata_populated(self, tmp_path: Path) -> None:
        """Verify all three YAML files exist and metadata has promotion entries."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="alpha",
                total_pnl=400.0,
                max_drawdown_pct=0.15,
                profit_factor=2.0,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="beta",
                total_pnl=200.0,
                max_drawdown_pct=0.05,
                profit_factor=1.5,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="gamma",
                total_pnl=100.0,
                max_drawdown_pct=0.03,
                profit_factor=1.1,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        # All three role YAML files exist
        for role in ("primary", "fallback", "conservative"):
            yaml_path = run_dir / "promoted_configs" / f"{role}.yaml"
            assert yaml_path.is_file(), f"{role}.yaml should exist"

        # Metadata structure
        assert not result["skipped"]
        assert result["promoted_count"] == 3
        for role in ("primary", "fallback", "conservative"):
            role_meta = result["roles"][role]
            assert "config_id" in role_meta
            assert "promoted_path" in role_meta
            assert "total_pnl" in role_meta
            assert "max_drawdown_pct" in role_meta
            assert "profit_factor" in role_meta
            assert "balanced_score" in role_meta

    def test_run_metadata_json_has_promotion_entries(self, tmp_path: Path) -> None:
        """Simulate writing to run_metadata.json as the pipeline does."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="cand1",
                total_pnl=300.0,
                max_drawdown_pct=0.10,
                profit_factor=1.8,
                tmp_path=tmp_path,
            ),
        ]

        meta: dict[str, Any] = {"run_id": "test_run", "steps": []}

        promotion_meta = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)
        meta["promotion"] = promotion_meta
        _write_json(run_dir / "run_metadata.json", meta)

        # Read back and verify
        loaded = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert "promotion" in loaded
        assert loaded["promotion"]["promoted_count"] >= 1
        assert "primary" in loaded["promotion"]["roles"]


# ===========================================================================
# (c) Edge cases
# ===========================================================================


class TestPromotionEdgeCases:
    """Test promotion edge cases for robustness."""

    def test_all_negative_pnl_no_promotion(self, tmp_path: Path) -> None:
        """All candidates have negative PnL → no promotion, graceful skip."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="loser_a",
                total_pnl=-100.0,
                max_drawdown_pct=0.30,
                profit_factor=0.5,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="loser_b",
                total_pnl=-50.0,
                max_drawdown_pct=0.20,
                profit_factor=0.8,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="loser_c",
                total_pnl=-200.0,
                max_drawdown_pct=0.40,
                profit_factor=0.3,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        assert result["skipped"] is True
        assert result["reason"] == "no_positive_pnl_candidates"
        assert not (run_dir / "promoted_configs").exists()

    def test_single_positive_pnl_gets_all_roles(self, tmp_path: Path) -> None:
        """Only 1 candidate with positive PnL → that candidate gets all 3 roles."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="winner",
                total_pnl=200.0,
                max_drawdown_pct=0.08,
                profit_factor=1.6,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="loser_a",
                total_pnl=-50.0,
                max_drawdown_pct=0.20,
                profit_factor=0.7,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="loser_b",
                total_pnl=-100.0,
                max_drawdown_pct=0.35,
                profit_factor=0.4,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        assert not result["skipped"]
        assert result["promoted_count"] == 3

        # All three roles point to the same winning candidate
        for role in ("primary", "fallback", "conservative"):
            assert result["roles"][role]["config_id"] == "winner"
            assert (run_dir / "promoted_configs" / f"{role}.yaml").is_file()

    def test_promote_count_zero_skips(self, tmp_path: Path) -> None:
        """promote_count=0 → no promotion directory created."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="any",
                total_pnl=100.0,
                max_drawdown_pct=0.05,
                profit_factor=1.5,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=0)

        assert result["skipped"] is True
        assert result["reason"] == "promote_count=0"
        assert not (run_dir / "promoted_configs").exists()

    def test_twenty_candidates_only_top_3_promoted(self, tmp_path: Path) -> None:
        """20 candidates → only top 3 promoted (verify selection logic)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = []
        for i in range(20):
            candidates.append(
                _make_candidate(
                    config_id=f"cand_{i:02d}",
                    total_pnl=float(50 + i * 25),     # 50, 75, ..., 525
                    max_drawdown_pct=0.02 + i * 0.015,  # 0.02, 0.035, ..., 0.305
                    profit_factor=1.0 + i * 0.1,        # 1.0, 1.1, ..., 2.9
                    win_rate=0.50 + i * 0.01,
                    total_trades=50 + i * 10,
                    tmp_path=tmp_path,
                )
            )

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        assert not result["skipped"]
        assert result["promoted_count"] == 3
        assert len(result["roles"]) == 3

        # Only 3 files written
        promoted_dir = run_dir / "promoted_configs"
        yaml_files = list(promoted_dir.glob("*.yaml"))
        assert len(yaml_files) == 3

        # Verify the selection logic:
        # - primary: highest balanced score (pnl * (1 - dd) * pf)
        # - fallback/conservative: lowest DD (cand_00 with dd=0.02)
        primary_id = result["roles"]["primary"]["config_id"]
        fallback_id = result["roles"]["fallback"]["config_id"]
        conservative_id = result["roles"]["conservative"]["config_id"]

        # Conservative & fallback should be the lowest-DD candidate (cand_00)
        assert conservative_id == "cand_00"
        assert fallback_id == "cand_00"

        # Primary should NOT be cand_00 (it has low PnL/PF, so balanced score is lower)
        # The balanced score = pnl * (1 - dd) * pf
        # Let's verify by computing the actual best balanced score candidate
        def balanced(c: dict[str, Any]) -> float:
            return float(c["total_pnl"]) * (1.0 - min(float(c["max_drawdown_pct"]), 1.0)) * max(float(c["profit_factor"]), 0.0)

        best_idx = max(range(len(candidates)), key=lambda i: balanced(candidates[i]))
        assert primary_id == f"cand_{best_idx:02d}"

    def test_zero_pnl_not_promoted(self, tmp_path: Path) -> None:
        """A candidate with exactly 0 PnL should not be promoted (positive PnL required)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="zero_pnl",
                total_pnl=0.0,
                max_drawdown_pct=0.01,
                profit_factor=1.0,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="negative",
                total_pnl=-50.0,
                max_drawdown_pct=0.10,
                profit_factor=0.5,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=3)

        assert result["skipped"] is True
        assert result["reason"] == "no_positive_pnl_candidates"

    def test_empty_candidate_list(self, tmp_path: Path) -> None:
        """Empty candidate list → no promotion, graceful skip."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = _promote_candidates([], run_dir=run_dir, promote_count=3)

        assert result["skipped"] is True
        assert result["reason"] == "no_positive_pnl_candidates"

    def test_promote_count_1_limits_output(self, tmp_path: Path) -> None:
        """promote_count=1 → only primary is written."""
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            _make_candidate(
                config_id="alpha",
                total_pnl=400.0,
                max_drawdown_pct=0.15,
                profit_factor=2.0,
                tmp_path=tmp_path,
            ),
            _make_candidate(
                config_id="beta",
                total_pnl=200.0,
                max_drawdown_pct=0.05,
                profit_factor=1.5,
                tmp_path=tmp_path,
            ),
        ]

        result = _promote_candidates(candidates, run_dir=run_dir, promote_count=1)

        assert not result["skipped"]
        assert result["promoted_count"] == 1
        assert "primary" in result["roles"]
        assert (run_dir / "promoted_configs" / "primary.yaml").is_file()

        # fallback and conservative should not be written
        yaml_files = list((run_dir / "promoted_configs").glob("*.yaml"))
        assert len(yaml_files) == 1


# ===========================================================================
# (d) Weekly/deep alias tests
# ===========================================================================


class TestWeeklyDeepAlias:
    """Verify that `weekly` and `deep` profiles produce identical args."""

    def test_profile_defaults_dict_identical(self) -> None:
        assert PROFILE_DEFAULTS["weekly"] == PROFILE_DEFAULTS["deep"]

    def test_parsed_args_identical(self) -> None:
        args_weekly = _parse_cli_args(["--run-id", "test", "--profile", "weekly"])
        args_deep = _parse_cli_args(["--run-id", "test", "--profile", "deep"])

        # Compare all profile-controlled fields
        for key in PROFILE_DEFAULTS["weekly"]:
            assert getattr(args_weekly, key) == getattr(args_deep, key), (
                f"Mismatch on {key}: weekly={getattr(args_weekly, key)} vs deep={getattr(args_deep, key)}"
            )

    def test_all_profile_keys_match_exactly(self) -> None:
        """Ensure the PROFILE_DEFAULTS keys for weekly and deep have the same set of keys."""
        assert set(PROFILE_DEFAULTS["weekly"].keys()) == set(PROFILE_DEFAULTS["deep"].keys())
        for key in PROFILE_DEFAULTS["weekly"]:
            assert PROFILE_DEFAULTS["weekly"][key] == PROFILE_DEFAULTS["deep"][key]

    def test_all_profiles_present(self) -> None:
        """All expected profiles exist in PROFILE_DEFAULTS."""
        assert set(PROFILE_DEFAULTS.keys()) == {"smoke", "daily", "deep", "weekly"}
