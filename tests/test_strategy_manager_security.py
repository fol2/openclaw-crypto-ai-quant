from __future__ import annotations

import os
import time

import pytest
import yaml

from engine.strategy_manager import StrategyManager
from engine.utils import sha1_json
from tools.config_id import config_id_from_yaml_text


def test_check_yaml_path_rejects_world_writable_file(tmp_path) -> None:
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text("global: {}\n", encoding="utf-8")
    yaml_path.chmod(0o666)

    assert StrategyManager._check_yaml_path(str(yaml_path)) is False


def test_check_yaml_path_accepts_owner_writable_file(tmp_path) -> None:
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text("global: {}\n", encoding="utf-8")
    yaml_path.chmod(0o600)

    assert StrategyManager._check_yaml_path(str(yaml_path)) is True


def test_world_writable_reload_keeps_last_known_good_config(tmp_path) -> None:
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text("global:\n  trade:\n    leverage: 7\n", encoding="utf-8")
    yaml_path.chmod(0o600)

    manager = StrategyManager.bootstrap(
        defaults={"trade": {"leverage": 1.0}, "indicators": {}, "filters": {}, "thresholds": {}},
        yaml_path=str(yaml_path),
        changelog_path=None,
    )
    assert float(manager.get_config("BTC").get("trade", {}).get("leverage", 0.0)) == 7.0

    # Unsafe permissions: reload must be skipped, preserving last-known-good overrides.
    yaml_path.write_text("global:\n  trade:\n    leverage: 9\n", encoding="utf-8")
    yaml_path.chmod(0o666)
    future_ts = time.time() + 1.0
    os.utime(yaml_path, (future_ts, future_ts))
    manager.maybe_reload()
    assert float(manager.get_config("BTC").get("trade", {}).get("leverage", 0.0)) == 7.0

    # After returning to safe permissions and changing content, reload should proceed.
    yaml_path.chmod(0o600)
    yaml_path.write_text("global:\n  trade:\n    leverage: 9\n", encoding="utf-8")
    future_ts = time.time() + 2.0
    os.utime(yaml_path, (future_ts, future_ts))
    manager.maybe_reload()
    assert float(manager.get_config("BTC").get("trade", {}).get("leverage", 0.0)) == 9.0


def test_snapshot_recomputes_resolver_owned_identity_from_loaded_yaml(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "effective.yaml"
    yaml_text = "global:\n  engine:\n    interval: 5m\n"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_OWNER", "rust")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_MATERIALISED", "1")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_ID", "a" * 64)
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_STRATEGY_SHA", "b" * 64)
    monkeypatch.setenv("AI_QUANT_STRATEGY_MODE", "primary")

    manager = StrategyManager.bootstrap(
        defaults={"trade": {}, "indicators": {}, "filters": {}, "thresholds": {}, "engine": {}},
        yaml_path=str(yaml_path),
        changelog_path=None,
    )

    snap = manager.snapshot
    assert snap.config_id == config_id_from_yaml_text(yaml_text)
    assert snap.overrides_sha1 == sha1_json(yaml.safe_load(yaml_text))
    assert (manager.get_config("BTC").get("engine") or {}).get("interval") == "5m"


def test_snapshot_identity_tracks_reloaded_yaml_contents(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = tmp_path / "effective.yaml"
    yaml_path.write_text("global:\n  engine:\n    interval: 30m\n", encoding="utf-8")

    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_OWNER", "rust")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_MATERIALISED", "1")
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_ID", "a" * 64)
    monkeypatch.setenv("AI_QUANT_EFFECTIVE_STRATEGY_SHA", "b" * 64)

    manager = StrategyManager.bootstrap(
        defaults={"trade": {}, "indicators": {}, "filters": {}, "thresholds": {}, "engine": {}},
        yaml_path=str(yaml_path),
        changelog_path=None,
    )
    first_config_id = manager.snapshot.config_id

    yaml_path.write_text("global:\n  engine:\n    interval: 5m\n", encoding="utf-8")
    future_ts = time.time() + 1.0
    os.utime(yaml_path, (future_ts, future_ts))
    manager.maybe_reload()

    assert (manager.get_config("BTC").get("engine") or {}).get("interval") == "5m"
    assert manager.snapshot.config_id != first_config_id
