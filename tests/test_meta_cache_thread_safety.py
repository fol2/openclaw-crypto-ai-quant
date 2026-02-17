from __future__ import annotations

import threading
import time

import exchange.meta as meta


class _FakeInfo:
    def __init__(self, *_args, **_kwargs):
        pass

    def meta_and_asset_ctxs(self):
        return [
            {
                "universe": [
                    {"name": "BTC", "szDecimals": 3, "marginTableId": 7},
                ],
                "marginTables": [
                    [7, {"marginTiers": [{"lowerBound": 0, "maxLeverage": 25}]}],
                ],
            },
            [
                {"funding": "0.001", "dayNtlVlm": "12345", "dayBaseVlm": "100", "openInterest": "10"},
            ],
        ]


def test_get_sz_decimals_respects_cache_lock(monkeypatch):
    monkeypatch.setattr(
        meta,
        "_cached_instruments",
        {
            "BTC": meta.PerpInstrument(
                name="BTC",
                sz_decimals=3,
                margin_table_id=1,
                funding_rate=0.0,
                day_ntl_vlm=0.0,
                day_base_vlm=0.0,
                open_interest=0.0,
            )
        },
    )
    monkeypatch.setattr(meta, "_cached_at_s", time.time())
    monkeypatch.setattr(meta, "_next_refresh_allowed_s", None)

    started = threading.Event()
    finished = threading.Event()
    result: dict[str, int] = {}

    def _worker() -> None:
        started.set()
        result["value"] = meta.get_sz_decimals("BTC")
        finished.set()

    with meta._CACHE_LOCK:
        thread = threading.Thread(target=_worker)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)

    thread.join(timeout=1.0)
    assert finished.is_set()
    assert result["value"] == 3


def test_refresh_cache_respects_cache_lock(monkeypatch):
    monkeypatch.setattr(meta, "Info", _FakeInfo)
    monkeypatch.setattr(meta, "_cached_at_s", None)
    monkeypatch.setattr(meta, "_cached_instruments", {})
    monkeypatch.setattr(meta, "_cached_margin_tables", {})
    monkeypatch.setattr(meta, "_next_refresh_allowed_s", None)

    started = threading.Event()
    finished = threading.Event()

    def _worker() -> None:
        started.set()
        meta._refresh_cache()
        finished.set()

    with meta._CACHE_LOCK:
        thread = threading.Thread(target=_worker)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)

    thread.join(timeout=1.0)
    assert finished.is_set()
    assert meta.get_sz_decimals("BTC") == 3
    assert meta.max_leverage("BTC", 1000.0) == 25.0
