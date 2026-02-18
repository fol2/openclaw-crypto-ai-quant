from __future__ import annotations

import logging

from engine.utils import deep_merge, file_mtime


def test_file_mtime_none_returns_none() -> None:
    assert file_mtime(None) is None


def test_deep_merge_warns_and_ignores_non_dict_override(caplog) -> None:
    base = {"trade": {"leverage": 3}}

    with caplog.at_level(logging.WARNING):
        out = deep_merge(base, "not-a-dict")

    assert out is base
    assert base == {"trade": {"leverage": 3}}
    assert any("deep_merge override ignored" in rec.message for rec in caplog.records)


def test_deep_merge_with_none_override_keeps_base_without_warning(caplog) -> None:
    base = {"filters": {"enable_market_breadth_filter": True}}

    with caplog.at_level(logging.WARNING):
        out = deep_merge(base, None)

    assert out is base
    assert base == {"filters": {"enable_market_breadth_filter": True}}
    assert not caplog.records
