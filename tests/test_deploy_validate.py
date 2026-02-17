from tools.deploy_validate import validate_yaml_text


def _valid_yaml(
    *,
    leverage: float = 3.0,
    leverage_low: float = 1.0,
    leverage_medium: float = 3.0,
    leverage_high: float = 5.0,
    leverage_max_cap: float = 10.0,
    max_entry_orders_per_loop: int = 6,
    ema_fast_window: int = 20,
    ema_slow_window: int = 50,
) -> str:
    return (
        "global:\n"
        "  trade:\n"
        "    allocation_pct: 0.2\n"
        f"    leverage: {leverage}\n"
        f"    leverage_low: {leverage_low}\n"
        f"    leverage_medium: {leverage_medium}\n"
        f"    leverage_high: {leverage_high}\n"
        f"    leverage_max_cap: {leverage_max_cap}\n"
        "    sl_atr_mult: 2.0\n"
        "    tp_atr_mult: 6.0\n"
        "    slippage_bps: 10.0\n"
        "    max_open_positions: 20\n"
        f"    max_entry_orders_per_loop: {max_entry_orders_per_loop}\n"
        "    max_total_margin_pct: 0.6\n"
        "    min_notional_usd: 10.0\n"
        "    min_atr_pct: 0.003\n"
        "    bump_to_min_notional: true\n"
        "  indicators:\n"
        "    adx_window: 14\n"
        f"    ema_fast_window: {ema_fast_window}\n"
        f"    ema_slow_window: {ema_slow_window}\n"
        "    bb_window: 20\n"
        "    atr_window: 14\n"
        "  thresholds:\n"
        "    entry:\n"
        "      min_adx: 22.0\n"
    )


def test_deploy_validate_rejects_missing_global():
    errs = validate_yaml_text("trade:\n  leverage: 3\n")
    assert errs
    assert any("global" in e for e in errs)


def test_deploy_validate_rejects_invalid_ema_invariant():
    y = _valid_yaml(ema_fast_window=50, ema_slow_window=20)
    errs = validate_yaml_text(y)
    assert any("ema_fast_window" in e for e in errs)


def test_deploy_validate_rejects_leverage_above_safe_cap():
    errs = validate_yaml_text(_valid_yaml(leverage=50.0))
    assert any("global.trade.leverage" in e for e in errs)


def test_deploy_validate_rejects_dynamic_leverage_above_safe_cap():
    errs = validate_yaml_text(_valid_yaml(leverage_high=25.0))
    assert any("global.trade.leverage_high" in e for e in errs)


def test_deploy_validate_rejects_max_entry_orders_per_loop_above_cap():
    errs = validate_yaml_text(_valid_yaml(max_entry_orders_per_loop=999))
    assert any("global.trade.max_entry_orders_per_loop" in e for e in errs)
