from __future__ import annotations

import os
import sys
import types

import pytest

import engine.daemon as daemon
from engine.daemon import _build_run_fingerprint, StrategyModePolicy


def test_build_run_fingerprint_carries_effective_config_identity(monkeypatch) -> None:
    strategy = types.SimpleNamespace(
        snapshot=types.SimpleNamespace(
            yaml_path="/tmp/effective.yaml",
            yaml_mtime=123.0,
            overrides_sha1="s" * 64,
            version="v-test",
            config_id="c" * 64,
        ),
        get_watchlist=lambda: ["eth", "btc"],
    )

    monkeypatch.setenv("AI_QUANT_ACTIVE_STRATEGY_YAML", "/tmp/active.yaml")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_STRATEGY_YAML", "/tmp/effective.yaml")
    monkeypatch.setenv("AI_QUANT_PROMOTED_ROLE", "primary")
    monkeypatch.setenv("AI_QUANT_STRATEGY_MODE", "fallback")
    monkeypatch.setenv("AI_QUANT_STRATEGY_MODE_SOURCE", "file")

    _fingerprint, payload = _build_run_fingerprint(
        strategy=strategy,
        mode="paper",
        interval="5m",
        lookback_bars=400,
    )

    assert payload["config_id"] == "c" * 64
    assert payload["active_yaml_path"] == "/tmp/active.yaml"
    assert payload["effective_yaml_path"] == "/tmp/effective.yaml"
    assert payload["strategy_mode"] == "fallback"
    assert payload["promoted_role"] == "primary"


def test_mode_policy_refreshes_rust_effective_config_on_switch(monkeypatch) -> None:
    risk = types.SimpleNamespace(kill_mode="close_only", kill_reason="test", kill_since_s=1.0)
    manager = types.SimpleNamespace(replace_yaml_path=lambda path: setattr(manager, "yaml_path", path))

    monkeypatch.setenv("AI_QUANT_MODE_SWITCH_ENABLE", "1")
    monkeypatch.setenv("AI_QUANT_MODE", "paper")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_OWNER", "rust")
    monkeypatch.setenv("AI_QUANT_STRATEGY_MODE", "primary")
    monkeypatch.setattr(
        "engine.promoted_config.apply_paper_effective_config",
        lambda: types.SimpleNamespace(config_path="/tmp/fallback.yaml"),
    )
    monkeypatch.setattr("engine.daemon.StrategyManager.get", lambda: manager)
    monkeypatch.setattr("engine.daemon._atomic_write_text", lambda path, text: None)

    policy = StrategyModePolicy(risk=risk)
    policy.maybe_switch()

    assert manager.yaml_path == "/tmp/fallback.yaml"
    assert os.environ["AI_QUANT_STRATEGY_MODE"] == "fallback"


def test_mode_policy_refreshes_live_rust_effective_config_on_switch(monkeypatch) -> None:
    risk = types.SimpleNamespace(kill_mode="close_only", kill_reason="test", kill_since_s=1.0)
    manager = types.SimpleNamespace(replace_yaml_path=lambda path: setattr(manager, "yaml_path", path))

    monkeypatch.setenv("AI_QUANT_MODE_SWITCH_ENABLE", "1")
    monkeypatch.setenv("AI_QUANT_MODE", "live")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_OWNER", "rust")
    monkeypatch.setenv("AI_QUANT_STRATEGY_MODE", "primary")
    monkeypatch.setattr(
        "engine.promoted_config.apply_live_effective_config",
        lambda: types.SimpleNamespace(config_path="/tmp/live-fallback.yaml"),
    )
    monkeypatch.setattr("engine.daemon.StrategyManager.get", lambda: manager)
    monkeypatch.setattr("engine.daemon._atomic_write_text", lambda path, text: None)

    policy = StrategyModePolicy(risk=risk)
    policy.maybe_switch()

    assert manager.yaml_path == "/tmp/live-fallback.yaml"
    assert os.environ["AI_QUANT_STRATEGY_MODE"] == "fallback"


def test_main_resolves_rust_effective_config_before_strategy_manager(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class StopBootstrap(RuntimeError):
        pass

    monkeypatch.setenv("AI_QUANT_MODE", "paper")
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(tmp_path / "paper.db"))
    monkeypatch.setenv("AI_QUANT_LOCK_PATH", str(tmp_path / "paper.lock"))
    monkeypatch.setattr(daemon, "_enforce_v8_only_runtime", lambda mode: None)
    monkeypatch.setattr(daemon, "acquire_lock_or_exit", lambda lock_path: object())
    monkeypatch.setattr(daemon, "_register_lock_cleanup", lambda lock: None)
    monkeypatch.setattr(daemon, "_harden_db_permissions", lambda *paths: None)

    def _unexpected_legacy_merge() -> None:
        raise AssertionError("legacy Python merge path must not be used")

    def _resolve() -> object:
        events.append("resolve")
        os.environ["AI_QUANT_EFFECTIVE_CONFIG_OWNER"] = "rust"
        return types.SimpleNamespace(promoted_role="primary", config_path=str(tmp_path / "effective.yaml"))

    monkeypatch.setattr("engine.promoted_config.maybe_apply_promoted_config", _unexpected_legacy_merge)

    monkeypatch.setattr(
        "engine.promoted_config.apply_paper_effective_config",
        _resolve,
    )

    def _stop_after_get() -> object:
        events.append("get")
        raise StopBootstrap("stop after StrategyManager.get")

    monkeypatch.setattr(daemon.StrategyManager, "get", staticmethod(_stop_after_get))
    monkeypatch.setitem(
        sys.modules,
        "strategy.mei_alpha_v1",
        types.SimpleNamespace(DB_PATH=str(tmp_path / "paper.db"), INTERVAL="5m", LOOKBACK_HOURS=400),
    )

    with pytest.raises(StopBootstrap, match="stop after StrategyManager.get"):
        daemon.main()

    assert events == ["resolve", "get"]
    assert os.environ["AI_QUANT_EFFECTIVE_CONFIG_OWNER"] == "rust"


def test_main_resolves_live_rust_effective_config_before_strategy_manager(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class StopBootstrap(RuntimeError):
        pass

    monkeypatch.setenv("AI_QUANT_MODE", "live")
    monkeypatch.setenv("AI_QUANT_DB_PATH", str(tmp_path / "live.db"))
    monkeypatch.setenv("AI_QUANT_LOCK_PATH", str(tmp_path / "live.lock"))
    monkeypatch.setattr(daemon, "_enforce_v8_only_runtime", lambda mode: None)
    monkeypatch.setattr(daemon, "acquire_lock_or_exit", lambda lock_path: object())
    monkeypatch.setattr(daemon, "_register_lock_cleanup", lambda lock: None)
    monkeypatch.setattr(daemon, "_harden_db_permissions", lambda *paths: None)
    monkeypatch.setitem(
        sys.modules,
        "exchange.executor",
        types.SimpleNamespace(
            load_live_secrets=lambda path: types.SimpleNamespace(secret_key="secret", main_address="addr"),
            HyperliquidLiveExecutor=lambda **kwargs: types.SimpleNamespace(
                account_snapshot=lambda force=True: types.SimpleNamespace(withdrawable_usd=1234.5)
            ),
        ),
    )

    def _resolve() -> object:
        events.append("resolve")
        os.environ["AI_QUANT_EFFECTIVE_CONFIG_OWNER"] = "rust"
        return types.SimpleNamespace(promoted_role=None, config_path=str(tmp_path / "live-effective.yaml"))

    monkeypatch.setattr(
        "engine.promoted_config.apply_live_effective_config",
        _resolve,
    )

    def _stop_after_get() -> object:
        events.append("get")
        raise StopBootstrap("stop after StrategyManager.get")

    monkeypatch.setattr(daemon.StrategyManager, "get", staticmethod(_stop_after_get))
    monkeypatch.setitem(
        sys.modules,
        "strategy.mei_alpha_v1",
        types.SimpleNamespace(DB_PATH=str(tmp_path / "live.db"), INTERVAL="30m", LOOKBACK_HOURS=200),
    )

    with pytest.raises(StopBootstrap, match="stop after StrategyManager.get"):
        daemon.main()

    assert events == ["resolve", "get"]
    assert os.environ["AI_QUANT_EFFECTIVE_CONFIG_OWNER"] == "rust"


def test_main_aborts_when_rust_resolver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy_manager_called = False

    monkeypatch.setenv("AI_QUANT_MODE", "paper")
    monkeypatch.setattr(daemon, "_enforce_v8_only_runtime", lambda mode: None)
    monkeypatch.setattr(
        "engine.promoted_config.apply_paper_effective_config",
        lambda: (_ for _ in ()).throw(RuntimeError("resolver boom")),
    )

    def _mark_called() -> object:
        nonlocal strategy_manager_called
        strategy_manager_called = True
        return object()

    monkeypatch.setattr(daemon.StrategyManager, "get", staticmethod(_mark_called))

    with pytest.raises(RuntimeError, match="paper effective-config resolution failed before daemon bootstrap"):
        daemon.main()

    assert strategy_manager_called is False


def test_main_aborts_when_live_rust_resolver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy_manager_called = False

    monkeypatch.setenv("AI_QUANT_MODE", "live")
    monkeypatch.setattr(daemon, "_enforce_v8_only_runtime", lambda mode: None)
    monkeypatch.setattr(
        "engine.promoted_config.apply_live_effective_config",
        lambda: (_ for _ in ()).throw(RuntimeError("resolver boom")),
    )

    def _mark_called() -> object:
        nonlocal strategy_manager_called
        strategy_manager_called = True
        return object()

    monkeypatch.setattr(daemon.StrategyManager, "get", staticmethod(_mark_called))

    with pytest.raises(RuntimeError, match="live effective-config resolution failed before daemon bootstrap"):
        daemon.main()

    assert strategy_manager_called is False
