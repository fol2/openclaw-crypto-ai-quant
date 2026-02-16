from __future__ import annotations


def test_drawdown_kill_switch_requires_explicit_resume(tmp_path, monkeypatch):
    cfg = tmp_path / "strategy_overrides.yaml"
    cfg.write_text(
        "global:\n"
        "  risk:\n"
        "    max_drawdown_pct: 5.0\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_QUANT_STRATEGY_YAML", str(cfg))

    from engine.strategy_manager import StrategyManager

    StrategyManager.bootstrap(
        defaults={"trade": {}, "indicators": {}, "filters": {}, "thresholds": {}},
        yaml_path=str(cfg),
        changelog_path=None,
    )

    from engine.risk import RiskManager

    class FakeTrader:
        def __init__(self, values):
            self._it = iter(values)

        def get_live_balance(self):
            return next(self._it)

    trader = FakeTrader([100.0, 94.0, 100.0, 100.0])
    risk = RiskManager()

    # Establish the equity peak first.
    risk.refresh(trader=trader)
    assert risk.kill_mode == "off"

    # 6% drawdown breaches the 5% threshold.
    risk.refresh(trader=trader)
    assert risk.kill_mode == "close_only"

    # Do not auto-clear when equity recovers.
    risk.refresh(trader=trader)
    assert risk.kill_mode == "close_only"

    # Explicit operator resume.
    monkeypatch.setenv("AI_QUANT_KILL_SWITCH", "clear")
    risk.refresh(trader=trader)
    assert risk.kill_mode == "off"


def test_drawdown_reduce_policy_is_loaded_from_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "strategy_overrides.yaml"
    cfg.write_text(
        "global:\n"
        "  risk:\n"
        "    max_drawdown_pct: 5.0\n"
        "    drawdown_reduce_policy: close_all\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_QUANT_STRATEGY_YAML", str(cfg))

    from engine.strategy_manager import StrategyManager

    StrategyManager.bootstrap(
        defaults={"trade": {}, "indicators": {}, "filters": {}, "thresholds": {}},
        yaml_path=str(cfg),
        changelog_path=None,
    )

    from engine.risk import RiskManager

    risk = RiskManager()
    risk.refresh(trader=None)
    assert risk.drawdown_reduce_policy == "close_all"
