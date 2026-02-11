from __future__ import annotations

from pathlib import Path

import factory_run


def test_resolve_path_for_backtester_resolves_repo_root_relative_paths() -> None:
    p = factory_run._resolve_path_for_backtester("config/strategy_overrides.yaml")
    assert p is not None
    pp = Path(p)
    assert pp.is_absolute()
    assert pp.exists()
    assert pp.name == "strategy_overrides.yaml"


def test_resolve_path_for_backtester_resolves_backtester_relative_sweep_specs() -> None:
    p = factory_run._resolve_path_for_backtester("sweeps/smoke.yaml")
    assert p is not None
    pp = Path(p)
    assert pp.is_absolute()
    assert pp.exists()
    assert pp.name == "smoke.yaml"

