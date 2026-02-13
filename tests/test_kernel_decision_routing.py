from __future__ import annotations

import json
import types
from dataclasses import dataclass
import importlib

import pytest
from unittest.mock import Mock

from engine.core import (
    KernelDecision,
    UnifiedEngine,
    KernelDecisionRustBindingProvider,
    KernelDecisionFileProvider,
    NoopDecisionProvider,
    _build_default_decision_provider,
)
from live.trader import LiveTrader


class FakeStrategyManager:
    def __init__(self) -> None:
        self.snapshot = types.SimpleNamespace(version="tkt-004", overrides_sha1="sha1")
        self.reload_calls = 0

    def maybe_reload(self) -> None:
        self.reload_calls += 1

    def get_watchlist(self) -> list[str]:
        return ["ETH"]

    def get_config(self, scope: str) -> dict[str, object]:
        if str(scope) == "__GLOBAL__":
            return {
                "engine": {
                    "heartbeat_every_s": "0",
                    "entry_interval": "1m",
                }
            }
        return {}

    def analyze(self, *_args, **_kwargs) -> None:
        raise AssertionError("Kernel decision path should not call analyze")


class FakeMarket:
    def ensure(self, *, symbols, interval, candle_limit, user) -> None:
        return None

    def ws_health(self, symbols) -> object:
        class _S:
            mids_age_s = 0
            candle_age_s = 0
            bbo_age_s = 0

        return _S()

    def candles_ready(self, *, symbols, interval):
        return (True, [])

    def get_last_closed_candle_key(self, symbol, *, interval, grace_ms=2000) -> int:
        return 1700000000000

    def get_latest_candle_open_key(self, symbol, *, interval) -> int:
        return 1700000000000

    def get_candles_df(self, symbol, *, interval, min_rows) -> None:
        return None

    def get_mid_price(self, symbol, *, max_age_s=10.0, interval=None):
        return types.SimpleNamespace(price=123.0, source="test", age_s=0.0)

    def health(self, *, symbols, interval) -> dict[str, object]:
        return {"connected": True, "thread_alive": True}


class FakeDecisionProvider:
    def __init__(self, decisions: list[KernelDecision]) -> None:
        self.decisions = decisions

    def get_decisions(
        self,
        *,
        symbols: list[str],
        watchlist: list[str],
        open_symbols: list[str],
        market: object,
        interval: str,
        lookback_bars: int,
        mode: str,
        not_ready_symbols: set[str],
        strategy: object,
        now_ms: int,
    ) -> list[KernelDecision]:
        return self.decisions


class FakeTrader:
    def __init__(self) -> None:
        self.positions: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, str, float | None, str]] = []

    def execute_trade(
        self,
        symbol,
        signal,
        price,
        timestamp,
        confidence,
        atr=0.0,
        indicators=None,
        *,
        action: str | None = None,
        target_size: float | None = None,
        reason: str | None = None,
    ) -> None:
        self.calls.append((str(action or ""), str(signal), float(target_size) if target_size is not None else None, str(confidence)))

    def check_exit_conditions(self, *args, **kwargs) -> None:
        return None


def test_unified_engine_routes_kernel_decisions_to_trader(monkeypatch) -> None:
    monkeypatch.setattr("engine.core.time.sleep", lambda *_: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr("strategy.mei_alpha_v1.get_strategy_config", lambda _symbol: {})

    strategy = FakeStrategyManager()
    market = FakeMarket()
    trader = FakeTrader()
    decisions = [
        KernelDecision.from_raw(
            {
                "symbol": "ETH",
                "action": "OPEN",
                "signal": "BUY",
                "confidence": "high",
                "score": 10.0,
                "target_size": 12.5,
            }
        ),
        KernelDecision.from_raw(
            {
                "symbol": "ETH",
                "action": "REDUCE",
                "signal": "BUY",
                "confidence": "high",
                "score": 5.0,
                "target_size": 0.75,
            }
        ),
    ]

    provider = FakeDecisionProvider([d for d in decisions if d is not None])
    engine = UnifiedEngine(
        trader=trader,
        strategy=strategy,
        market=market,
        interval="1m",
        lookback_bars=50,
        mode="paper",
        mode_plugin=None,
        decision_provider=provider,
    )

    with pytest.raises(SystemExit):
        engine.run_forever()

    assert [c[0] for c in trader.calls] == ["OPEN", "REDUCE"]
    assert [c[2] for c in trader.calls] == [12.5, 0.75]


class DummyExecutor:
    def __init__(self) -> None:
        self.update_leverage_calls: list[tuple[str, float, bool]] = []
        self.market_open_calls: list[tuple] = []
        self.last_order_error = {}

    def account_snapshot(self, force: bool = False):  # pragma: no cover - tiny test stub
        from types import SimpleNamespace

        return SimpleNamespace(
            account_value_usd=100000.0,
            withdrawable_usd=100000.0,
            total_margin_used_usd=0.0,
        )

    def get_positions(self, force: bool = False):  # pragma: no cover - tiny test stub
        return {}

    def update_leverage(self, symbol, leverage, is_cross=True):
        self.update_leverage_calls.append((str(symbol).strip().upper(), float(leverage), bool(is_cross)))
        return True

    def market_open(
        self,
        symbol,
        is_buy,
        sz,
        px,
        slippage_pct,
        cloid=None,
    ):
        self.market_open_calls.append((str(symbol).strip().upper(), bool(is_buy), float(sz), float(px), float(slippage_pct), cloid))
        return True


@dataclass
class _KernelSizing:
    leverage: float
    desired_notional_usd: float


def test_live_execute_trade_action_routing(monkeypatch) -> None:
    monkeypatch.setattr("live.trader.log_live_signal", lambda **kwargs: None)
    trader = LiveTrader(executor=DummyExecutor())
    trader.positions = {
        "BTC": {
            "type": "LONG",
            "size": 1.0,
            "entry_price": 100.0,
            "entry_atr": 2.0,
            "tp1_taken": 0,
            "leverage": 2,
        }
    }

    mock_add = Mock(return_value=True)
    mock_close = Mock(return_value=True)
    mock_reduce = Mock(return_value=True)
    monkeypatch.setattr(trader, "add_to_position", mock_add)
    monkeypatch.setattr(trader, "close_position", mock_close)
    monkeypatch.setattr(trader, "reduce_position", mock_reduce)

    trader.execute_trade(
        "BTC",
        "BUY",
        101.0,
        1700000000000,
        "high",
        action="CLOSE",
        reason="kernel-close",
    )
    assert mock_close.call_count == 1
    assert mock_close.call_args.args == ("BTC", 101.0, 1700000000000)
    assert mock_close.call_args.kwargs.get("reason") == "kernel-close"
    assert mock_add.call_count == 0

    trader.execute_trade(
        "BTC",
        "SELL",
        101.0,
        1700000000000,
        "high",
        action="ADD",
        target_size=7.5,
        reason="kernel-add",
    )
    mock_add.assert_called_once_with(
        "BTC", 101.0, 1700000000000, "high", atr=0.0, indicators=None, target_size=7.5, reason="kernel-add"
    )
    assert mock_reduce.call_count == 0

    trader.execute_trade(
        "BTC",
        "BUY",
        101.0,
        1700000000000,
        "high",
        action="REDUCE",
        target_size=0.5,
        reason="kernel-reduce",
    )
    assert mock_reduce.call_count == 1
    assert mock_reduce.call_args.args == ("BTC", 0.5, 101.0, 1700000000000)
    assert mock_reduce.call_args.kwargs.get("reason") == "kernel-reduce"


def test_live_execute_trade_open_action_uses_open_kernel_path(monkeypatch) -> None:
    monkeypatch.setattr("live.trader.log_live_signal", lambda **kwargs: None)
    monkeypatch.setattr("live.trader.mei_alpha_v1.compute_entry_sizing", lambda *args, **kwargs: _KernelSizing(2.0, 200.0))
    monkeypatch.setattr(
        "live.trader.mei_alpha_v1._get_fill_price",
        lambda *a, **k: float((a[2] if len(a) > 2 else k.get("price", 0.0)) or 0.0),
    )
    monkeypatch.setattr("live.trader.mei_alpha_v1.get_strategy_config", lambda _symbol: {})
    monkeypatch.setattr("live.trader.mei_alpha_v1.log_audit_event", lambda *a, **k: None)
    monkeypatch.setattr("live.trader.hyperliquid_meta.max_leverage", lambda symbol, notional: 100.0)
    monkeypatch.setattr("live.trader.hyperliquid_meta.round_size", lambda symbol, sz: float(sz))
    monkeypatch.setattr("live.trader.hyperliquid_meta.min_size_for_notional", lambda symbol, min_notional, price: float(min_notional) / float(price))
    monkeypatch.setattr(
        "live.trader.LiveTrader._live_trade_cfg",
        lambda self, symbol: {
            "enable_ssf_filter": False,
            "enable_reef_filter": False,
            "max_notional_usd_per_order": 10000.0,
            "size_multiplier": 1.0,
            "max_open_positions": 1,
        },
    )
    monkeypatch.setattr("live.trader.live_entries_enabled", lambda: True)
    monkeypatch.setattr("live.trader.live_mode", lambda: "live")
    monkeypatch.setattr("live.trader.live_orders_enabled", lambda: True)

    trader = LiveTrader(executor=DummyExecutor())
    trader._account_value_usd = 100000.0
    trader.balance = 100000.0
    trader.positions = {}
    trader._entry_budget_remaining = 1
    trader._total_margin_used_usd = 0.0

    mock_add = Mock(return_value=False)
    mock_close = Mock(return_value=False)
    mock_reduce = Mock(return_value=False)
    monkeypatch.setattr(trader, "add_to_position", mock_add)
    monkeypatch.setattr(trader, "close_position", mock_close)
    monkeypatch.setattr(trader, "reduce_position", mock_reduce)

    trader.execute_trade("ETH", "BUY", 101.0, 1700000000000, "high", action="OPEN", target_size=120.0, reason="kernel-open")

    assert mock_add.call_count == 0
    assert mock_close.call_count == 0
    assert mock_reduce.call_count == 0
    assert len(trader.executor.market_open_calls) == 1


class _FakeKernelRuntime:
    def __init__(self) -> None:
        self.default_state = {
            "schema_version": 1,
            "timestamp_ms": 1700000000000,
            "step": 1,
            "cash_usd": 10_000.0,
            "positions": {},
        }
        self.default_params = {
            "schema_version": 1,
            "default_notional_usd": 10_000.0,
            "min_notional_usd": 10.0,
            "max_notional_usd": 100_000.0,
            "maker_fee_bps": 3.5,
            "taker_fee_bps": 3.5,
            "allow_pyramid": True,
            "allow_reverse": True,
        }

    def default_kernel_state_json(self, initial_cash_usd: float, timestamp_ms: int) -> str:
        state = dict(self.default_state)
        state["cash_usd"] = float(initial_cash_usd)
        state["timestamp_ms"] = int(timestamp_ms)
        return json.dumps(state)

    def default_kernel_params_json(self) -> str:
        return json.dumps(self.default_params)

    def step_decision(self, state_json: str, event_json: str, _params_json: str) -> str:
        event = json.loads(event_json)
        if not isinstance(event, dict):
            return json.dumps({"ok": False, "error": {"code": "INVALID_EVENT", "message": "invalid", "details": []}})

        symbol = str(event.get("symbol", "")).strip().upper()
        if not symbol:
            return json.dumps({"ok": False, "error": {"code": "INVALID_EVENT", "message": "missing symbol", "details": []}})

        price = float(event.get("price", 0.0))
        quantity = 0.25
        notional = price * quantity
        intent_id = 1001

        state = json.loads(state_json)
        if not isinstance(state, dict):
            state = dict(self.default_state)

        state["step"] = int(state.get("step", 0)) + 1
        state["timestamp_ms"] = int(event.get("timestamp_ms", 0))

        decision = {
            "schema_version": 1,
            "state": state,
            "intents": [
                {
                    "schema_version": 1,
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "kind": "open",
                    "side": "long",
                    "quantity": quantity,
                    "price": price,
                    "notional_usd": notional,
                    "fee_rate": 0.0,
                }
            ],
            "fills": [],
            "diagnostics": {
                "schema_version": 1,
                "errors": [],
                "warnings": [],
                "intent_count": 1,
                "fill_count": 0,
                "step": 2,
            },
        }
        return json.dumps({"ok": True, "decision": decision})


def test_rust_binding_provider_converts_kernel_intents(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "events.json"
    payload_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "symbol": "ETH",
                    "action": "OPEN",
                    "signal": "BUY",
                    "price": 100.0,
                    "timestamp_ms": 1700000000000,
                    "notional_hint_usd": 2500.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("engine.core._load_kernel_runtime_module", lambda *_args, **_kwargs: _FakeKernelRuntime())

    provider = KernelDecisionRustBindingProvider(path=str(payload_path))
    decisions = list(
        provider.get_decisions(
            symbols=["ETH"],
            watchlist=["ETH"],
            open_symbols=[],
            market=None,
            interval="1m",
            lookback_bars=50,
            mode="paper",
            not_ready_symbols=set(),
            strategy=None,
            now_ms=1700000000000,
        )
    )

    assert len(decisions) == 1
    assert decisions[0].action == "OPEN"
    assert decisions[0].symbol == "ETH"
    assert decisions[0].target_size == 0.25
    assert decisions[0].confidence == "N/A"


def test_bt_runtime_binding_returns_schema_mismatch_error_code() -> None:
    try:
        bt_runtime = importlib.import_module("bt_runtime")
    except Exception:
        pytest.skip("bt_runtime extension is not available in the test environment")

    state_json = bt_runtime.default_kernel_state_json(10_000.0, 1_700_000_000_000)
    event_json = json.dumps(
        {
            "schema_version": 2,
            "event_id": 1,
            "timestamp_ms": 1_700_000_000_100,
            "symbol": "ETH",
            "signal": "BUY",
            "price": 100.0,
            "notional_hint_usd": 1000.0,
        }
    )
    params_json = bt_runtime.default_kernel_params_json()

    response = json.loads(bt_runtime.step_decision(state_json, event_json, params_json))
    assert isinstance(response, dict)
    assert response.get("ok") is False
    error = response.get("error", {})
    assert error.get("code") == "SCHEMA_VERSION_MISMATCH"


def test_bt_runtime_binding_invalid_json_returns_explicit_error_code() -> None:
    try:
        bt_runtime = importlib.import_module("bt_runtime")
    except Exception:
        pytest.skip("bt_runtime extension is not available in the test environment")

    response = json.loads(bt_runtime.step_decision("{not-json", "{} {}", "{}"))
    assert isinstance(response, dict)
    assert response.get("ok") is False
    error = response.get("error", {})
    assert error.get("code") == "INVALID_JSON"


def test_build_default_decision_provider_prefers_rust_when_available(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "events.json"
    payload_path.write_text("[]", encoding="utf-8")
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_PROVIDER", raising=False)
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_FILE", raising=False)
    monkeypatch.setattr("engine.core._load_kernel_runtime_module", lambda *_args, **_kwargs: _FakeKernelRuntime())

    provider = _build_default_decision_provider()
    assert isinstance(provider, KernelDecisionRustBindingProvider)


def test_build_default_decision_provider_falls_back_to_file_if_rust_not_available(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "events.json"
    payload_path.write_text(
                json.dumps(
                [
                    {
                        "schema_version": 1,
                        "symbol": "ETH",
                        "action": "OPEN",
                        "signal": "BUY",
                        "price": 100.0,
                        "timestamp_ms": 1700000000000,
                        "notional_hint_usd": 2500.0,
                    }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_PROVIDER", raising=False)
    monkeypatch.setenv("AI_QUANT_KERNEL_DECISION_FILE", str(payload_path))
    monkeypatch.setattr(
        "engine.core._load_kernel_runtime_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("missing bt_runtime")),
    )

    provider = _build_default_decision_provider()
    assert isinstance(provider, KernelDecisionFileProvider)
    decisions = list(
        provider.get_decisions(
            symbols=["ETH"],
            watchlist=["ETH"],
            open_symbols=[],
            market=None,
            interval="1m",
            lookback_bars=50,
            mode="paper",
            not_ready_symbols=set(),
            strategy=None,
            now_ms=1700000000000,
        )
        )
    assert len(decisions) == 1
    assert decisions[0].action == "OPEN"


def test_build_default_decision_provider_needs_file_or_kernel_when_no_provider_available(monkeypatch) -> None:
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_PROVIDER", raising=False)
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_FILE", raising=False)
    monkeypatch.setattr(
        "engine.core._load_kernel_runtime_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("missing bt_runtime")),
    )

    with pytest.raises(RuntimeError, match="Decision provider auto-mode cannot start"):
        _build_default_decision_provider()


def test_build_default_decision_provider_respects_noop_mode(monkeypatch) -> None:
    monkeypatch.setenv("AI_QUANT_KERNEL_DECISION_PROVIDER", "none")
    monkeypatch.delenv("AI_QUANT_KERNEL_DECISION_FILE", raising=False)

    provider = _build_default_decision_provider()
    assert isinstance(provider, NoopDecisionProvider)
