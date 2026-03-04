from __future__ import annotations

import pytest

from engine import daemon


def test_v8_only_guard_allows_v8_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_QUANT_ENFORCE_V8_ONLY", "1")
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "v8-paper1")
    monkeypatch.setenv("AI_QUANT_DB_PATH", "/tmp/trading_engine_v8_paper1.db")

    daemon._enforce_v8_only_runtime("paper")


def test_v8_only_guard_blocks_legacy_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_QUANT_ENFORCE_V8_ONLY", "1")
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "legacy-paper")
    monkeypatch.setenv("AI_QUANT_DB_PATH", "/tmp/trading_engine.db")

    with pytest.raises(SystemExit, match="Legacy engine runtime blocked by v8-only guard"):
        daemon._enforce_v8_only_runtime("paper")


def test_v8_only_guard_can_be_bypassed_for_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_QUANT_ENFORCE_V8_ONLY", "1")
    monkeypatch.setenv("AI_QUANT_ALLOW_LEGACY_ENGINE", "1")
    monkeypatch.setenv("AI_QUANT_INSTANCE_TAG", "legacy-paper")
    monkeypatch.setenv("AI_QUANT_DB_PATH", "/tmp/trading_engine.db")

    daemon._enforce_v8_only_runtime("paper")
