from __future__ import annotations

import itertools

import pytest

from tools import manual_trade


def _fake_fill(
    *,
    cloid: str,
    coin: str = "PUMP",
    oid: str = "339230023584",
    px: str = "0.002067",
    sz: str = "1423.0",
    side: str = "B",
    t_ms: int = 1772719461174,
    start_position: str = "-47460.0",
    dir_s: str = "Close Short",
    closed_pnl: str = "0.055497",
    fill_hash: str = "0xb8434527",
    tid: int = 892677432799448,
) -> dict[str, object]:
    return {
        "coin": coin,
        "px": px,
        "sz": sz,
        "side": side,
        "time": t_ms,
        "startPosition": start_position,
        "dir": dir_s,
        "closedPnl": closed_pnl,
        "hash": fill_hash,
        "oid": oid,
        "crossed": True,
        "fee": "0.00127",
        "tid": tid,
        "cloid": cloid,
        "feeToken": "USDC",
        "twapId": None,
    }


def test_dir_to_pos_action_marks_terminal_close_leg_as_close():
    assert manual_trade._dir_to_pos_action("Close Long", 49043.0, 49043.0) == ("LONG", "CLOSE")
    assert manual_trade._dir_to_pos_action("Close Short", -46037.0, 46037.0) == ("SHORT", "CLOSE")
    assert manual_trade._dir_to_pos_action("Close Short", -47460.0, 1423.0) == ("SHORT", "REDUCE")
    assert manual_trade._dir_to_pos_action("Open Long", 0.0, 1.0) == ("LONG", "OPEN")
    assert manual_trade._dir_to_pos_action("Open Long", 1.0, 0.5) == ("LONG", "ADD")


def test_poll_fills_collects_all_partial_fills(monkeypatch: pytest.MonkeyPatch):
    cloid = "0x6d616e5fed92369a17de46f495862bf7"
    fills_by_call = [
        [
            _fake_fill(cloid=cloid, sz="1423.0", start_position="-47460.0", fill_hash="0xb843", tid=1),
            _fake_fill(
                cloid=cloid, sz="46037.0", start_position="-46037.0", closed_pnl="1.795443", fill_hash="0xb843", tid=2
            ),
        ],
        [
            _fake_fill(cloid=cloid, sz="1423.0", start_position="-47460.0", fill_hash="0xb843", tid=1),
            _fake_fill(
                cloid=cloid, sz="46037.0", start_position="-46037.0", closed_pnl="1.795443", fill_hash="0xb843", tid=2
            ),
        ],
    ]

    class _FakeInfo:
        def __init__(self):
            self.calls = 0

        def user_fills_by_time(self, *_args, **_kwargs):
            idx = min(self.calls, len(fills_by_call) - 1)
            self.calls += 1
            return fills_by_call[idx]

    class _FakeExecutor:
        def __init__(self):
            self._fills = _FakeInfo()

        def user_fills_by_time(self, *_args, **_kwargs):
            return self._fills.user_fills_by_time(*_args, **_kwargs)

    clock = itertools.count()
    monkeypatch.setattr(manual_trade.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(manual_trade.time, "monotonic", lambda: next(clock) * 0.1)
    monkeypatch.setattr(manual_trade.time, "time", lambda: 1772719461.174)

    fills = manual_trade._poll_fills(
        _FakeExecutor(),
        cloid,
        "PUMP",
        1772719459000,
        exchange_order_id="339230023584",
        expected_size=47460.0,
        max_wait_s=1.0,
        poll_interval_s=0.01,
    )

    assert [int(f["tid"]) for f in fills] == [1, 2]
    summary = manual_trade._summarise_fills(fills)
    assert summary["filled_size"] == pytest.approx(47460.0)
    assert summary["filled_price"] == pytest.approx(0.002067)
    assert summary["closed_pnl"] == pytest.approx(1.85094)


def test_poll_fills_waits_for_delayed_sibling_fill_until_expected_size_is_reached(
    monkeypatch: pytest.MonkeyPatch,
):
    cloid = "0x6d616e5fed92369a17de46f495862bf7"
    first = _fake_fill(cloid=cloid, sz="1423.0", start_position="-47460.0", fill_hash="0xb843", tid=1)
    second = _fake_fill(
        cloid=cloid,
        sz="46037.0",
        start_position="-46037.0",
        closed_pnl="1.795443",
        fill_hash="0xb843",
        tid=2,
    )
    fills_by_call = [
        [first],
        [first],
        [first, second],
        [first, second],
    ]

    class _FakeInfo:
        def __init__(self):
            self.calls = 0

        def user_fills_by_time(self, *_args, **_kwargs):
            idx = min(self.calls, len(fills_by_call) - 1)
            self.calls += 1
            return fills_by_call[idx]

    class _FakeExecutor:
        def __init__(self):
            self._fills = _FakeInfo()

        def user_fills_by_time(self, *_args, **_kwargs):
            return self._fills.user_fills_by_time(*_args, **_kwargs)

    clock = itertools.count()
    monkeypatch.setattr(manual_trade.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(manual_trade.time, "monotonic", lambda: next(clock) * 0.1)
    monkeypatch.setattr(manual_trade.time, "time", lambda: 1772719461.174)

    fills = manual_trade._poll_fills(
        _FakeExecutor(),
        cloid,
        "PUMP",
        1772719459000,
        exchange_order_id="339230023584",
        expected_size=47460.0,
        max_wait_s=1.0,
        poll_interval_s=0.01,
    )

    assert [int(f["tid"]) for f in fills] == [1, 2]


def test_poll_fills_can_match_by_exchange_order_id_when_fill_lacks_cloid(
    monkeypatch: pytest.MonkeyPatch,
):
    cloid = "0x6d616e5fe864104fd0584da2beb92328"
    fill = _fake_fill(
        cloid="",
        coin="SOL",
        oid="339315249361",
        px="89.479",
        sz="0.39",
        start_position="0.72",
        dir_s="Open Long",
        closed_pnl="0.0",
        fill_hash="0x29e6",
        tid=198099954761078,
    )

    class _FakeInfo:
        def user_fills_by_time(self, *_args, **_kwargs):
            return [fill]

    class _FakeExecutor:
        def __init__(self):
            self._fills = _FakeInfo()

        def user_fills_by_time(self, *_args, **_kwargs):
            return self._fills.user_fills_by_time(*_args, **_kwargs)

    clock = itertools.count()
    monkeypatch.setattr(manual_trade.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(manual_trade.time, "monotonic", lambda: next(clock) * 0.1)
    monkeypatch.setattr(manual_trade.time, "time", lambda: 1772723531.060)

    fills = manual_trade._poll_fills(
        _FakeExecutor(),
        cloid,
        "SOL",
        1772723529000,
        exchange_order_id="339315249361",
        expected_size=0.39,
        max_wait_s=0.5,
        poll_interval_s=0.01,
    )

    assert len(fills) == 1
    assert fills[0]["oid"] == "339315249361"
