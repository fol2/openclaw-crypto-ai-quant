from __future__ import annotations

from engine.kernel_shadow_report import DEFAULT_SHADOW_MAX_COMPARISONS, ShadowReport


def test_shadow_report_comparisons_are_bounded_by_env_limit(monkeypatch):
    monkeypatch.setenv("AI_QUANT_SHADOW_MAX_COMPARISONS", "3")
    report = ShadowReport()

    for i in range(5):
        report.add_comparison(
            metric=f"m{i}",
            old_val=1.0,
            kernel_val=1.0,
            tolerance=0.01,
            timestamp=float(i),
        )

    assert report.comparisons.maxlen == 3
    assert len(report.comparisons) == 3
    assert [c.metric for c in report.comparisons] == ["m2", "m3", "m4"]


def test_shadow_report_invalid_env_uses_default_limit(monkeypatch):
    monkeypatch.setenv("AI_QUANT_SHADOW_MAX_COMPARISONS", "not-a-number")
    report = ShadowReport()
    assert report.comparisons.maxlen == DEFAULT_SHADOW_MAX_COMPARISONS
