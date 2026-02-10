import json

from tools.config_id import config_id_from_yaml_text


def test_event_logger_writes_jsonl_and_includes_config_id(tmp_path, monkeypatch):
    cfg = tmp_path / "strategy_overrides.yaml"
    yaml_text = "global:\n  hello: world\n"
    cfg.write_text(yaml_text, encoding="utf-8")

    out = tmp_path / "events.jsonl"

    monkeypatch.setenv("AI_QUANT_STRATEGY_YAML", str(cfg))
    monkeypatch.setenv("AI_QUANT_EVENT_LOG", "1")
    monkeypatch.setenv("AI_QUANT_EVENT_LOG_PATH", str(out))
    monkeypatch.setenv("AI_QUANT_RUN_ID", "run_123")
    monkeypatch.setenv("AI_QUANT_MODE", "paper")

    from engine.event_logger import _close_for_tests, emit_event

    emit_event(kind="unit_test", symbol="btc", data={"x": 1})
    _close_for_tests()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])

    assert evt["schema"] == "aiq_event_v1"
    assert evt["kind"] == "unit_test"
    assert evt["symbol"] == "BTC"
    assert evt["run_id"] == "run_123"
    assert evt["config_id"] == config_id_from_yaml_text(yaml_text)

