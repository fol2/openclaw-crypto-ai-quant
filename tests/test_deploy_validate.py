from tools.deploy_validate import validate_yaml_text


def test_deploy_validate_rejects_missing_global():
    errs = validate_yaml_text("trade:\n  leverage: 3\n")
    assert errs
    assert any("global" in e for e in errs)


def test_deploy_validate_rejects_invalid_ema_invariant():
    y = (
        "global:\n"
        "  trade:\n"
        "    allocation_pct: 0.2\n"
        "    leverage: 3.0\n"
        "    sl_atr_mult: 2.0\n"
        "    tp_atr_mult: 6.0\n"
        "    slippage_bps: 10.0\n"
        "    max_open_positions: 20\n"
        "    max_total_margin_pct: 0.6\n"
        "    min_notional_usd: 10.0\n"
        "    min_atr_pct: 0.003\n"
        "    bump_to_min_notional: true\n"
        "  indicators:\n"
        "    adx_window: 14\n"
        "    ema_fast_window: 50\n"
        "    ema_slow_window: 20\n"
        "    bb_window: 20\n"
        "    atr_window: 14\n"
        "  thresholds:\n"
        "    entry:\n"
        "      min_adx: 22.0\n"
    )
    errs = validate_yaml_text(y)
    assert any("ema_fast_window" in e for e in errs)

