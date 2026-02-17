import json
import sqlite3

from tools.config_id import config_id_from_yaml_text
from tools.paper_deploy import deploy_paper_config


def _init_registry_db(path, *, config_id, yaml_text):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS configs (config_id TEXT PRIMARY KEY, yaml_text TEXT NOT NULL)"
        )
        con.execute(
            "INSERT OR REPLACE INTO configs (config_id, yaml_text) VALUES (?, ?)",
            (str(config_id), str(yaml_text)),
        )
        con.commit()
    finally:
        con.close()


def test_paper_deploy_writes_yaml_and_deploy_event(tmp_path):
    yaml_text = (
        "global:\n"
        "  trade:\n"
        "    allocation_pct: 0.20\n"
        "    leverage: 3.0\n"
        "    leverage_low: 1.0\n"
        "    leverage_medium: 3.0\n"
        "    leverage_high: 5.0\n"
        "    leverage_max_cap: 10.0\n"
        "    sl_atr_mult: 2.0\n"
        "    tp_atr_mult: 6.0\n"
        "    slippage_bps: 10.0\n"
        "    max_open_positions: 20\n"
        "    max_entry_orders_per_loop: 6\n"
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
    cid = config_id_from_yaml_text(yaml_text)

    artifacts_dir = tmp_path / "artifacts"
    registry_db = artifacts_dir / "registry" / "registry.sqlite"
    _init_registry_db(registry_db, config_id=cid, yaml_text=yaml_text)

    target_yaml = tmp_path / "strategy_overrides.yaml"
    target_yaml.write_text("global:\\n  engine:\\n    interval: 30m\\n", encoding="utf-8")

    out_dir = tmp_path / "deploy_out"
    deploy_dir = deploy_paper_config(
        config_id=cid,
        artifacts_dir=artifacts_dir,
        yaml_path=target_yaml,
        out_dir=out_dir,
        reason="unit test",
        restart="never",
        service="does-not-matter",
        dry_run=False,
        validate=True,
    )

    assert deploy_dir.exists()
    assert target_yaml.read_text(encoding="utf-8") == yaml_text

    event_path = deploy_dir / "deploy_event.json"
    assert event_path.exists()
    obj = json.loads(event_path.read_text(encoding="utf-8"))
    assert obj["what"]["config_id"] == cid
    assert obj["why"]["reason"] == "unit test"


def test_paper_deploy_dry_run_does_not_modify_yaml(tmp_path):
    yaml_text = (
        "global:\n"
        "  trade:\n"
        "    allocation_pct: 0.20\n"
        "    leverage: 3.0\n"
        "    leverage_low: 1.0\n"
        "    leverage_medium: 3.0\n"
        "    leverage_high: 5.0\n"
        "    leverage_max_cap: 10.0\n"
        "    sl_atr_mult: 2.0\n"
        "    tp_atr_mult: 6.0\n"
        "    slippage_bps: 10.0\n"
        "    max_open_positions: 20\n"
        "    max_entry_orders_per_loop: 6\n"
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
    cid = config_id_from_yaml_text(yaml_text)

    artifacts_dir = tmp_path / "artifacts"
    registry_db = artifacts_dir / "registry" / "registry.sqlite"
    _init_registry_db(registry_db, config_id=cid, yaml_text=yaml_text)

    target_yaml = tmp_path / "strategy_overrides.yaml"
    original = "global:\\n  engine:\\n    interval: 30m\\n"
    target_yaml.write_text(original, encoding="utf-8")

    out_dir = tmp_path / "deploy_out"
    deploy_paper_config(
        config_id=cid,
        artifacts_dir=artifacts_dir,
        yaml_path=target_yaml,
        out_dir=out_dir,
        reason="dry run",
        restart="never",
        service="does-not-matter",
        dry_run=True,
        validate=True,
    )

    assert target_yaml.read_text(encoding="utf-8") == original
