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


def test_normalise_candles_db_dir_for_backtester_uses_candles_glob(tmp_path) -> None:
    (tmp_path / "candles_1m.db").write_text("", encoding="utf-8")
    (tmp_path / "funding_rates.db").write_text("", encoding="utf-8")

    out = factory_run._normalise_candles_db_arg_for_backtester(str(tmp_path))
    assert out is not None
    assert out.endswith("candles_*.db")
