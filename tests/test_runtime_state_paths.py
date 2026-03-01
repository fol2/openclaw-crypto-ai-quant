import strategy.mei_alpha_v1 as mei_alpha_v1
from pathlib import Path


def test_instance_state_path_defaults_to_home_mei(monkeypatch):
    monkeypatch.delenv("AI_QUANT_KERNEL_STATE_DIR", raising=False)
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "v8-paper1")
    monkeypatch.setattr(mei_alpha_v1, "DB_PATH", "/tmp/trading_engine_v8_paper1.db", raising=False)

    got = mei_alpha_v1.PaperTrader._instance_state_path("kernel_state.json")
    assert got == str(Path("~/.mei/kernel_state_v8-paper1.json").expanduser())


def test_instance_state_path_honours_relative_state_dir(monkeypatch):
    monkeypatch.setenv("AI_QUANT_KERNEL_STATE_DIR", ".runtime_state")
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "v8-paper2")
    monkeypatch.setattr(mei_alpha_v1, "DB_PATH", "/tmp/trading_engine_v8_paper2.db", raising=False)

    got = mei_alpha_v1.PaperTrader._instance_state_path("kernel_shadow_report.json")
    assert got == "/tmp/.runtime_state/kernel_shadow_report_v8-paper2.json"


def test_instance_state_path_legacy_db_dir_fallback(monkeypatch):
    monkeypatch.setenv("AI_QUANT_KERNEL_STATE_DIR", "/var/lib/ai-quant/state")
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "v8-paper3")
    monkeypatch.setattr(mei_alpha_v1, "DB_PATH", "/tmp/trading_engine_v8_paper3.db", raising=False)

    got = mei_alpha_v1.PaperTrader._instance_state_path("kernel_state.json", legacy_db_dir=True)
    assert got == "/tmp/kernel_state_v8-paper3.json"


def test_kernel_bootstrap_balance_prefers_db_seed(monkeypatch):
    monkeypatch.setattr(mei_alpha_v1, "PAPER_BALANCE", 10000.0, raising=False)
    trader = object.__new__(mei_alpha_v1.PaperTrader)
    trader._seed_balance = 23456.78

    assert trader._kernel_bootstrap_balance() == 23456.78


def test_kernel_bootstrap_balance_falls_back_to_seed_env(monkeypatch):
    monkeypatch.setattr(mei_alpha_v1, "PAPER_BALANCE", 9876.54, raising=False)
    trader = object.__new__(mei_alpha_v1.PaperTrader)
    trader._seed_balance = "invalid"

    assert trader._kernel_bootstrap_balance() == 9876.54
