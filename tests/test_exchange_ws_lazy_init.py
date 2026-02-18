from __future__ import annotations

import exchange.ws as ws_mod


class _DummyWS:
    def __init__(self) -> None:
        self.calls = 0

    def get_mid(self, _symbol: str, max_age_s: float | None = None):  # noqa: ARG002
        self.calls += 1
        return 123.45


def test_hl_ws_is_lazy_singleton(monkeypatch) -> None:
    calls = {"n": 0}
    dummy = _DummyWS()

    def _fake_make_default_ws():
        calls["n"] += 1
        return dummy

    monkeypatch.setattr(ws_mod, "_default_ws_singleton", None)
    monkeypatch.setattr(ws_mod, "_make_default_ws", _fake_make_default_ws)

    assert calls["n"] == 0
    assert ws_mod.hl_ws.get_mid("ETH", max_age_s=1.0) == 123.45
    assert calls["n"] == 1
    assert ws_mod.hl_ws.get_mid("BTC", max_age_s=1.0) == 123.45
    assert calls["n"] == 1
    assert dummy.calls == 2
