from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from engine import daemon


class _FakeWs:
    def __init__(self, mid: float | None):
        self.mid = mid
        self.calls: list[tuple[str, float | None]] = []

    def get_mid(self, symbol: str, max_age_s: float | None = None) -> float | None:
        self.calls.append((symbol, max_age_s))
        return self.mid


class _FakeTrader:
    def __init__(self):
        self.positions = {"ETH": {"size": 1.0}}
        self.close_calls: list[tuple[str, float, float, str]] = []

    def close_position(self, symbol: str, price: float, ts_s: float, *, reason: str, meta: dict) -> None:
        self.close_calls.append((symbol, price, ts_s, reason))


def _build_plugin(*, ws_mid: float | None) -> tuple[daemon.LivePlugin, _FakeTrader, _FakeWs]:
    plugin = daemon.LivePlugin.__new__(daemon.LivePlugin)
    plugin._risk = SimpleNamespace(
        kill_mode="close_only",
        kill_reason="drawdown breach",
        drawdown_reduce_policy="close_all",
        kill_since_s=1234.0,
    )
    plugin._last_drawdown_reduce_kill_since_s = None
    trader = _FakeTrader()
    ws = _FakeWs(mid=ws_mid)
    plugin.trader = trader
    plugin._ws = ws
    return plugin, trader, ws


def test_drawdown_close_all_uses_bounded_mid_age():
    plugin, trader, ws = _build_plugin(ws_mid=150.0)

    plugin._maybe_reduce_risk_on_drawdown_kill()

    assert ws.calls == [("ETH", 10.0)]
    assert len(trader.close_calls) == 1
    symbol, px, ts_s, reason = trader.close_calls[0]
    assert symbol == "ETH"
    assert px == pytest.approx(150.0)
    assert ts_s <= time.time()
    assert "drawdown kill" in reason


def test_drawdown_close_all_skips_when_mid_stale(
    caplog: pytest.LogCaptureFixture,
):
    plugin, trader, ws = _build_plugin(ws_mid=None)

    with caplog.at_level(logging.WARNING):
        plugin._maybe_reduce_risk_on_drawdown_kill()

    assert ws.calls == [("ETH", 10.0)]
    assert trader.close_calls == []
    assert any("missing/stale WS mid" in rec.message for rec in caplog.records)
