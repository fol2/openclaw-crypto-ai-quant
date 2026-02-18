from __future__ import annotations

from engine.strategy_manager import StrategyManager


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
