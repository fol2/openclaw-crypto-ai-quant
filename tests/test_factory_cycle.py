from __future__ import annotations

import datetime
import json

import pytest

from tools.factory_cycle import _apply_strategy_mode_overlay, _parse_candidate, _select_deployable_candidates, _stable_promotion_since_s


def _iso(ts: datetime.datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("tz-aware required")
    return ts.isoformat().replace("+00:00", "Z")


def _write_deploy_event(*, root, stamp: str, service: str, config_id: str, ts_utc: str) -> None:
    ev_dir = root / "deployments" / "paper" / stamp
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev = {
        "ts_utc": ts_utc,
        "what": {"config_id": config_id},
        "restart": {"service": service},
    }
    (ev_dir / "deploy_event.json").write_text(json.dumps(ev), encoding="utf-8")


def test_apply_strategy_mode_overlay_materialises_modes_into_global() -> None:
    base = {
        "global": {"engine": {"interval": "1h", "entry_interval": "3m", "exit_interval": "3m"}},
        "modes": {
            "primary": {
                "global": {"engine": {"interval": "30m", "entry_interval": "5m", "exit_interval": "5m"}},
            }
        },
    }

    eff = _apply_strategy_mode_overlay(base=base, strategy_mode="primary")
    assert eff["global"]["engine"]["interval"] == "30m"
    assert eff["global"]["engine"]["entry_interval"] == "5m"
    assert eff["global"]["engine"]["exit_interval"] == "5m"
    # Preserve the original modes section (useful for operators).
    assert "modes" in eff


def test_apply_strategy_mode_overlay_raises_on_unknown_mode() -> None:
    base = {"global": {"engine": {"interval": "1h"}}, "modes": {"primary": {"global": {"engine": {"interval": "30m"}}}}}
    with pytest.raises(KeyError):
        _apply_strategy_mode_overlay(base=base, strategy_mode="does_not_exist")


def test_stable_promotion_since_tracks_oldest_timestamp_for_contiguous_config_segment(tmp_path) -> None:
    start = datetime.datetime(2026, 2, 10, 0, 0, tzinfo=datetime.timezone.utc)
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T000000Z_cfg",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T010000Z_cfg",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start + datetime.timedelta(hours=1)),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T020000Z_cfg",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start + datetime.timedelta(hours=2)),
    )

    since_s = _stable_promotion_since_s(artifacts_dir=tmp_path, service="svc-a", config_id="cfgA")
    assert since_s is not None
    assert since_s == pytest.approx(start.timestamp())


def test_stable_promotion_since_uses_latest_contiguous_segment_after_config_switch(tmp_path) -> None:
    start = datetime.datetime(2026, 2, 10, 0, 0, tzinfo=datetime.timezone.utc)
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T000000Z_cfgA",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T010000Z_cfgA",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start + datetime.timedelta(hours=1)),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T020000Z_cfgB",
        service="svc-a",
        config_id="cfgB",
        ts_utc=_iso(start + datetime.timedelta(hours=2)),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T030000Z_cfgA",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start + datetime.timedelta(hours=3)),
    )
    _write_deploy_event(
        root=tmp_path,
        stamp="20260210T040000Z_cfgA",
        service="svc-a",
        config_id="cfgA",
        ts_utc=_iso(start + datetime.timedelta(hours=4)),
    )

    since_a = _stable_promotion_since_s(artifacts_dir=tmp_path, service="svc-a", config_id="cfgA")
    since_b = _stable_promotion_since_s(artifacts_dir=tmp_path, service="svc-a", config_id="cfgB")
    assert since_a is not None
    assert since_b is not None
    assert since_a == pytest.approx((start + datetime.timedelta(hours=3)).timestamp())
    assert since_b == pytest.approx((start + datetime.timedelta(hours=2)).timestamp())


def test_parse_candidate_legacy_fields_are_backward_compatible() -> None:
    legacy = {
        "config_id": "abc",
        "config_path": "/tmp/candidate.yaml",
        "total_pnl": 1.5,
        "profit_factor": 1.2,
        "total_trades": 2,
    }
    cand = _parse_candidate(legacy)
    assert cand is not None
    assert cand.pipeline_stage == "legacy"
    assert cand.sweep_stage == "legacy"
    assert cand.replay_stage == ""
    assert cand.validation_gate == "replay_only"
    assert cand.canonical_cpu_verified is True
    assert cand.has_stage_metadata is False


def test_select_deployable_candidates_prefers_stage_and_cpu_verified_candidates() -> None:
    candidates = [
        {
            "config_id": "cand_1",
            "config_path": "/tmp/a.yaml",
            "total_pnl": 1.0,
            "total_trades": 1,
            "profit_factor": 1.0,
            "max_drawdown_pct": 0.1,
            "pipeline_stage": "candidate_generation",
            "sweep_stage": "cpu",
            "replay_stage": "",
            "validation_gate": "replay_only",
            "canonical_cpu_verified": False,
        },
        {
            "config_id": "cand_2",
            "config_path": "/tmp/b.yaml",
            "total_pnl": 0.5,
            "total_trades": 1,
            "profit_factor": 1.0,
            "max_drawdown_pct": 0.1,
            "pipeline_stage": "candidate_validation",
            "sweep_stage": "gpu",
            "replay_stage": "cpu_replay",
            "validation_gate": "replay_only",
            "canonical_cpu_verified": True,
        },
        {
            "config_id": "cand_3",
            "config_path": "/tmp/c.yaml",
            "total_pnl": 2.0,
            "total_trades": 1,
            "profit_factor": 1.0,
            "max_drawdown_pct": 0.1,
        },
    ]
    parsed = [_parse_candidate(c) for c in candidates]
    assert all(p is not None for p in parsed)
    selected = _select_deployable_candidates([p for p in parsed if p is not None], limit=1)
    assert len(selected) == 1
    assert selected[0].config_id == "cand_2"


def test_unverified_staged_candidates_are_not_deployable() -> None:
    candidates = [
        {
            "config_id": "cand_legacy",
            "config_path": "/tmp/a.yaml",
            "total_pnl": 1.0,
            "total_trades": 1,
            "profit_factor": 1.0,
            "max_drawdown_pct": 0.1,
        },
        {
            "config_id": "cand_unverified",
            "config_path": "/tmp/b.yaml",
            "total_pnl": 10.0,
            "total_trades": 1,
            "profit_factor": 3.0,
            "max_drawdown_pct": 0.1,
            "pipeline_stage": "candidate_validation",
            "sweep_stage": "gpu",
            "replay_stage": "cpu_replay",
            "validation_gate": "replay_only",
            "canonical_cpu_verified": False,
        },
    ]
    parsed = [_parse_candidate(c) for c in candidates]
    assert all(p is not None for p in parsed)
    selected = _select_deployable_candidates([p for p in parsed if p is not None], limit=2)
    assert len(selected) == 1
    assert selected[0].config_id == "cand_legacy"
