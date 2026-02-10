import json
import sqlite3

import pytest

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
    yaml_text = "global:\\n  engine:\\n    interval: 1h\\n"
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
    )

    assert deploy_dir.exists()
    assert target_yaml.read_text(encoding="utf-8") == yaml_text

    event_path = deploy_dir / "deploy_event.json"
    assert event_path.exists()
    obj = json.loads(event_path.read_text(encoding="utf-8"))
    assert obj["what"]["config_id"] == cid
    assert obj["why"]["reason"] == "unit test"


def test_paper_deploy_dry_run_does_not_modify_yaml(tmp_path):
    yaml_text = "global:\\n  engine:\\n    interval: 1h\\n"
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
    )

    assert target_yaml.read_text(encoding="utf-8") == original

