import json

from tools.rollback_to_last_good import rollback_to_last_good


VALID_YAML = (
    "global:\n"
    "  trade:\n"
    "    allocation_pct: 0.20\n"
    "    leverage: 3.0\n"
    "    sl_atr_mult: 2.0\n"
    "    tp_atr_mult: 6.0\n"
    "    slippage_bps: 10.0\n"
    "    max_open_positions: 20\n"
    "    max_total_margin_pct: 0.60\n"
    "    min_notional_usd: 10.0\n"
    "    min_atr_pct: 0.003\n"
    "    bump_to_min_notional: true\n"
    "  indicators:\n"
    "    adx_window: 14\n"
    "    ema_fast_window: 20\n"
    "    ema_slow_window: 50\n"
    "    bb_window: 20\n"
    "    atr_window: 14\n"
    "  thresholds:\n"
    "    entry:\n"
    "      min_adx: 22.0\n"
    "  engine:\n"
    "    interval: 1h\n"
)


def test_rollback_to_last_good_uses_prev_config_yaml(tmp_path):
    artifacts = tmp_path / "artifacts"
    deploy_root = artifacts / "deployments" / "paper"
    deploy_root.mkdir(parents=True, exist_ok=True)

    latest = deploy_root / "20260210T100000Z_deadbeef"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "prev_config.yaml").write_text(VALID_YAML, encoding="utf-8")

    target_yaml = tmp_path / "strategy_overrides.yaml"
    target_yaml.write_text("global:\n  engine:\n    interval: 30m\n", encoding="utf-8")

    rb_dir = rollback_to_last_good(
        artifacts_dir=artifacts,
        yaml_path=target_yaml,
        steps=1,
        reason="unit test rollback",
        restart="never",
        service="does-not-matter",
        dry_run=False,
    )

    assert target_yaml.read_text(encoding="utf-8").strip() == VALID_YAML.strip()
    event = json.loads((rb_dir / "rollback_event.json").read_text(encoding="utf-8"))
    assert event["why"]["reason"] == "unit test rollback"


def test_rollback_to_last_good_falls_back_to_older_deployed_config(tmp_path):
    artifacts = tmp_path / "artifacts"
    deploy_root = artifacts / "deployments" / "paper"
    deploy_root.mkdir(parents=True, exist_ok=True)

    latest = deploy_root / "20260210T100000Z_deadbeef"
    older = deploy_root / "20260210T090000Z_cafebabe"
    latest.mkdir(parents=True, exist_ok=True)
    older.mkdir(parents=True, exist_ok=True)

    # Empty prev_config.yaml forces fallback.
    (latest / "prev_config.yaml").write_text("", encoding="utf-8")
    (older / "deployed_config.yaml").write_text(VALID_YAML, encoding="utf-8")

    target_yaml = tmp_path / "strategy_overrides.yaml"
    target_yaml.write_text("global:\n  engine:\n    interval: 30m\n", encoding="utf-8")

    rb_dir = rollback_to_last_good(
        artifacts_dir=artifacts,
        yaml_path=target_yaml,
        steps=1,
        reason="unit test fallback",
        restart="never",
        service="does-not-matter",
        dry_run=False,
    )

    assert target_yaml.read_text(encoding="utf-8").strip() == VALID_YAML.strip()
    assert (rb_dir / "restored_config.yaml").exists()
