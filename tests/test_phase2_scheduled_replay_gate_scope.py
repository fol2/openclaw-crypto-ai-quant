from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import assert_replay_bundle_alignment as gate_tool
from tools import build_live_replay_bundle as replay_bundle_builder
from tools import run_paper_deterministic_replay as paper_harness
from tools import run_scheduled_replay_alignment_gate as scheduled_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_default_strategy_config_path_prefers_live_db_parent_then_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    live_root = tmp_path / "live_runtime"
    repo_cfg = repo_root / "config" / "strategy_overrides.yaml"
    live_cfg = live_root / "config" / "strategy_overrides.yaml"
    live_db = live_root / "trading_engine_live.db"

    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    live_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text("repo\n", encoding="utf-8")
    live_cfg.write_text("live\n", encoding="utf-8")
    live_db.write_text("", encoding="utf-8")

    assert scheduled_gate._default_strategy_config_path(
        repo_root=repo_root,
        live_db=live_db,
    ) == live_cfg.resolve()

    live_cfg.unlink()
    assert scheduled_gate._default_strategy_config_path(
        repo_root=repo_root,
        live_db=live_db,
    ) == repo_cfg.resolve()

    repo_cfg.unlink()
    assert (
        scheduled_gate._default_strategy_config_path(
            repo_root=repo_root,
            live_db=live_db,
        )
        is None
    )


def test_assert_gate_can_skip_trade_and_action_axes(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "bundle"
    output = tmp_path / "gate_report.json"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        bundle_dir / "replay_bundle_manifest.json",
        {
            "schema_version": 1,
            "artefacts": {
                "backtester_replay_report_json": "backtester_replay_report.json",
            },
            "live_run_fingerprint_provenance": {
                "run_fingerprint_distinct": 1,
                "rows_sampled": 0,
                "run_fingerprint_timeline": [],
            },
        },
    )
    _write_json(
        bundle_dir / "state_alignment_report.json",
        {
            "ok": True,
            "summary": {"diff_count": 0},
        },
    )
    _write_json(
        bundle_dir / "backtester_replay_report.json",
        {
            "config_fingerprint": "a" * 64,
        },
    )
    _write_json(
        bundle_dir / "trade_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": False},
            "accepted_residuals": [],
            "counts": {"mismatch_total": 3},
            "scope_contract": {"mismatch": False},
        },
    )
    _write_json(
        bundle_dir / "action_reconcile_report.json",
        {
            "status": {"strict_alignment_pass": False, "artefact_only_mismatch": False},
            "accepted_residuals": [],
            "counts": {
                "mismatch_total": 7,
                "live_simulatable_actions": 1,
                "paper_simulatable_actions": 1,
            },
            "scope_contract": {"mismatch": False},
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assert_replay_bundle_alignment.py",
            "--bundle-dir",
            str(bundle_dir),
            "--skip-candles-provenance-check",
            "--skip-trade-axis",
            "--skip-action-axis",
            "--output",
            str(output),
        ],
    )

    rc = gate_tool.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["ok"] is True
    assert report["checks"]["trade_axis_skipped"] is True
    assert report["checks"]["action_axis_skipped"] is True
    assert report["contract"]["axes"]["trade"]["required"] is False
    assert report["contract"]["axes"]["action"]["required"] is False
    assert report["contract"]["axes"]["trade"]["gate_ok"] is True
    assert report["contract"]["axes"]["action"]["gate_ok"] is True
    assert report["checks"]["action_ok"] is True


def test_build_bundle_alignment_script_wires_skip_axis_env_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    live_db = tmp_path / "trading_engine_live.db"
    paper_db = tmp_path / "paper.db"
    candles_db = tmp_path / "candles_30m.db"
    strategy_cfg = tmp_path / "strategy_overrides.yaml"
    bundle_dir = tmp_path / "bundle"

    for path in (live_db, paper_db, candles_db):
        path.write_bytes(b"")
    strategy_cfg.write_text(
        (
            "global:\n"
            "  engine:\n"
            "    interval: 30m\n"
            "    entry_interval: 30m\n"
            "    exit_interval: 30m\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay_bundle_builder,
        "_resolve_live_db_for_window",
        lambda **_: (
            live_db.resolve(),
            {
                "requested_live_db": str(live_db.resolve()),
                "resolved_live_db": str(live_db.resolve()),
                "resolution_reason": "requested_live_db_has_best_window_coverage",
                "auto_switched": False,
                "strict_live_db_path": False,
                "allow_empty_live_window": True,
                "window_from_ts": 0,
                "window_to_ts": 0,
                "window_live_to_ts": 1_799_999,
                "coverage_by_db": {
                    str(live_db.resolve()): {"has_any_rows": True, "live_baseline_trades": 0},
                },
            },
        ),
    )
    monkeypatch.setattr(replay_bundle_builder, "_load_live_baseline_trades", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(replay_bundle_builder, "_load_live_order_fail_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        replay_bundle_builder,
        "_load_runtime_strategy_provenance",
        lambda *_args, **_kwargs: {
            "window_from_ts": 0,
            "window_to_ts": 0,
            "runtime_rows_in_window": 0,
            "strategy_rows_sampled": 0,
            "strategy_sha1_distinct": 0,
            "strategy_version_distinct": 0,
            "strategy_sha1_timeline": [],
        },
    )
    monkeypatch.setattr(
        replay_bundle_builder,
        "_load_oms_strategy_provenance",
        lambda *_args, **_kwargs: {
            "window_from_ts": 0,
            "window_to_ts": 0,
            "oms_rows_in_window": 0,
            "oms_rows_sampled": 0,
            "strategy_sha1_distinct": 0,
            "strategy_version_distinct": 0,
            "strategy_sha1_timeline": [],
        },
    )
    monkeypatch.setattr(
        replay_bundle_builder,
        "build_candles_window_provenance",
        lambda *_args, **_kwargs: {
            "interval": "30m",
            "from_ts": 0,
            "to_ts": 0,
            "row_count": 0,
            "symbol_count": 0,
            "symbols": [],
            "window_hash_sha256": "0" * 64,
            "universe_hash_sha256": "1" * 64,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_live_replay_bundle.py",
            "--live-db",
            str(live_db),
            "--paper-db",
            str(paper_db),
            "--candles-db",
            str(candles_db),
            "--strategy-config",
            str(strategy_cfg),
            "--interval",
            "30m",
            "--from-ts",
            "0",
            "--to-ts",
            "0",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-empty-live-window",
        ],
    )

    rc = replay_bundle_builder.main()
    script = (bundle_dir / "run_08_assert_alignment.sh").read_text(encoding="utf-8")

    assert rc == 0
    assert 'if [ "${AQC_SKIP_TRADE_AXIS:-0}" = "1" ]; then' in script
    assert 'GATE_ARGS+=(--skip-trade-axis)' in script
    assert 'if [ "${AQC_SKIP_ACTION_AXIS:-0}" = "1" ]; then' in script
    assert 'GATE_ARGS+=(--skip-action-axis)' in script
    assert 'if [ "${AQC_SKIP_GPU_PARITY:-0}" != "1" ]; then' in script
    assert 'GATE_ARGS+=(--gpu-parity-report "$BUNDLE_DIR/gpu_smoke_parity_report.json" --require-gpu-parity)' in script


def test_paper_harness_skips_gpu_parity_when_env_flag_is_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    output = bundle_dir / "paper_deterministic_replay_run.json"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for script_name in (
        "run_01_export_and_seed.sh",
        "run_02_replay.sh",
        "run_03_audit.sh",
        "run_03b_mirror_live_window_to_paper.sh",
        "run_04_trade_reconcile.sh",
        "run_05_action_reconcile.sh",
        "run_06_live_paper_action_reconcile.sh",
        "run_07_live_paper_decision_trace_reconcile.sh",
        "run_07b_event_order_parity.sh",
        "run_08_assert_alignment.sh",
    ):
        path = bundle_dir / script_name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        path.chmod(0o755)

    skipped_gpu = bundle_dir / "run_07c_gpu_parity.sh"
    skipped_gpu.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    skipped_gpu.chmod(0o755)

    monkeypatch.setenv("AQC_SKIP_GPU_PARITY", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper_deterministic_replay.py",
            "--bundle-dir",
            str(bundle_dir),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    rc = paper_harness.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["ok"] is True
    assert "gpu_parity" not in report["planned_steps"]
    assert all(step["step"] != "gpu_parity" for step in report["steps"])
