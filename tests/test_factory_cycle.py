from __future__ import annotations

import pytest

from tools.factory_cycle import _apply_strategy_mode_overlay


def test_apply_strategy_mode_overlay_materialises_modes_into_global() -> None:
    base = {
        "global": {"engine": {"interval": "1h", "entry_interval": "3m", "exit_interval": "3m"}},
        "modes": {
            "primary": {
                "global": {"engine": {"interval": "30m", "entry_interval": "5m", "exit_interval": "5m"}},
            }
        },
    }

    eff = _apply_strategy_mode_overlay(base=base, strategy_mode="primary")
    assert eff["global"]["engine"]["interval"] == "30m"
    assert eff["global"]["engine"]["entry_interval"] == "5m"
    assert eff["global"]["engine"]["exit_interval"] == "5m"
    # Preserve the original modes section (useful for operators).
    assert "modes" in eff


def test_apply_strategy_mode_overlay_raises_on_unknown_mode() -> None:
    base = {"global": {"engine": {"interval": "1h"}}, "modes": {"primary": {"global": {"engine": {"interval": "30m"}}}}}
    with pytest.raises(KeyError):
        _apply_strategy_mode_overlay(base=base, strategy_mode="does_not_exist")

