import pytest

from tools.sensitivity_check import _compute_aggregate


def test_compute_aggregate_sensitivity_v1_positive_rate_and_medians():
    variants = [
        {"exit_code": 0, "total_pnl": 10.0, "max_drawdown_pct": 0.10},
        {"exit_code": 0, "total_pnl": -5.0, "max_drawdown_pct": 0.20},
        # Non-zero exit codes are treated as skipped for aggregation.
        {"exit_code": 2, "total_pnl": 999.0, "max_drawdown_pct": 9.99},
    ]

    agg = _compute_aggregate(20.0, variants)

    assert agg["variants_total"] == 3
    assert agg["variants_ran"] == 2
    assert agg["variants_skipped"] == 1

    assert agg["positive_rate"] == 0.5
    assert agg["median_total_pnl"] == 2.5
    assert agg["median_drawdown_pct"] == pytest.approx(0.15)
