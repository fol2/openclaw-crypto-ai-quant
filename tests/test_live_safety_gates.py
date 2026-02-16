"""H8: Tests for live_orders_enabled() safety gates.

Covers all combinations of AI_QUANT_LIVE_ENABLE, AI_QUANT_LIVE_CONFIRM,
and AI_QUANT_HARD_KILL_SWITCH to verify fail-closed behaviour.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure a clean environment for each test."""
    for var in (
        "AI_QUANT_LIVE_ENABLE",
        "AI_QUANT_LIVE_CONFIRM",
        "AI_QUANT_HARD_KILL_SWITCH",
        "AI_QUANT_KILL_SWITCH",
        "AI_QUANT_MODE",
    ):
        monkeypatch.delenv(var, raising=False)


def _import_live_orders_enabled():
    from exchange.executor import live_orders_enabled
    return live_orders_enabled


def _import_live_entries_enabled():
    from exchange.executor import live_entries_enabled
    return live_entries_enabled


class TestLiveOrdersEnabled:
    """Test live_orders_enabled() safety gate."""

    def test_both_disabled(self, monkeypatch):
        """Both switches disabled → no orders."""
        fn = _import_live_orders_enabled()
        assert fn() is False

    def test_enable_without_confirm(self, monkeypatch):
        """ENABLE=1 without CONFIRM → no orders."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        fn = _import_live_orders_enabled()
        assert fn() is False

    def test_enable_with_wrong_confirm(self, monkeypatch):
        """ENABLE=1 with wrong CONFIRM → no orders."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "WRONG_STRING")
        fn = _import_live_orders_enabled()
        assert fn() is False

    def test_hard_kill_switch_overrides(self, monkeypatch):
        """HARD_KILL_SWITCH overrides everything."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        monkeypatch.setenv("AI_QUANT_HARD_KILL_SWITCH", "1")
        fn = _import_live_orders_enabled()
        assert fn() is False

    def test_correct_enable_and_confirm(self, monkeypatch):
        """Both set correctly → orders enabled."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        fn = _import_live_orders_enabled()
        assert fn() is True

    def test_confirm_without_enable(self, monkeypatch):
        """CONFIRM set but ENABLE not set → no orders."""
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        fn = _import_live_orders_enabled()
        assert fn() is False


class TestLiveEntriesEnabled:
    """Test live_entries_enabled() which also checks AI_QUANT_KILL_SWITCH."""

    def test_kill_switch_blocks_entries(self, monkeypatch):
        """KILL_SWITCH=1 blocks entries even when orders are enabled."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        monkeypatch.setenv("AI_QUANT_KILL_SWITCH", "1")
        fn = _import_live_entries_enabled()
        assert fn() is False

    def test_hard_kill_switch_blocks_entries(self, monkeypatch):
        """HARD_KILL_SWITCH=1 blocks entries (orders disabled entirely)."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        monkeypatch.setenv("AI_QUANT_HARD_KILL_SWITCH", "1")
        fn = _import_live_entries_enabled()
        assert fn() is False

    def test_entries_enabled_when_all_gates_pass(self, monkeypatch):
        """All gates pass → entries enabled."""
        monkeypatch.setenv("AI_QUANT_LIVE_ENABLE", "1")
        monkeypatch.setenv("AI_QUANT_LIVE_CONFIRM", "I_UNDERSTAND_THIS_CAN_LOSE_MONEY")
        fn = _import_live_entries_enabled()
        assert fn() is True
