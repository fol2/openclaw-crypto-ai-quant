from __future__ import annotations

import types

from engine.daemon import _build_run_fingerprint


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
