from __future__ import annotations

import os
import types

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
