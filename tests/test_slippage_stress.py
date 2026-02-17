from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.slippage_stress as slippage_stress


def test_parse_bps_list_dedupes_and_preserves_order() -> None:
    parsed = slippage_stress._parse_bps_list("10,20,10,30")
    assert parsed == [10.0, 20.0, 30.0]


def test_parse_bps_list_rejects_invalid_value() -> None:
    with pytest.raises(SystemExit, match="Invalid slippage bps value"):
        slippage_stress._parse_bps_list("10,bad,30")


def test_main_writes_summary_and_rejects_flip_sign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("global: {}\n", encoding="utf-8")
    out_dir = tmp_path / "slippage_out"
    summary_path = tmp_path / "summary.json"

    pnl_by_bps = {10.0: 100.0, 20.0: -5.0, 30.0: -25.0}
    dd_by_bps = {10.0: 0.10, 20.0: 0.20, 30.0: 0.30}

    def _fake_run_cmd(argv: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
        bps = float(argv[argv.index("--slippage-bps") + 1])
        replay_out = Path(argv[argv.index("--output") + 1])
        replay_out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config_id": "cfg",
            "initial_balance": 1000.0,
            "final_balance": 1000.0 + float(pnl_by_bps[bps]),
            "total_pnl": float(pnl_by_bps[bps]),
            "total_trades": 10,
            "win_rate": 0.5,
            "profit_factor": 1.1,
            "sharpe_ratio": 0.4,
            "max_drawdown_pct": float(dd_by_bps[bps]),
            "total_fees": 1.0,
            "slippage_bps": float(bps),
        }
        replay_out.write_text(json.dumps(report), encoding="utf-8")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(slippage_stress, "_resolve_backtester_cmd", lambda: ["mei-backtester"])
    monkeypatch.setattr(slippage_stress, "_run_cmd", _fake_run_cmd)

    rc = slippage_stress.main(
        [
            "--config",
            str(config_path),
            "--interval",
            "1h",
            "--slippage-bps",
            "10,20,30",
            "--reject-flip-bps",
            "20",
            "--out-dir",
            str(out_dir),
            "--output",
            str(summary_path),
        ]
    )
    assert rc == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agg = summary["aggregate"]
    assert agg["baseline_bps"] == 10.0
    assert agg["baseline_total_pnl"] == pytest.approx(100.0)
    assert agg["pnl_at_reject_bps"] == pytest.approx(-5.0)
    assert agg["pnl_drop_at_reject_bps"] == pytest.approx(105.0)
    assert agg["slippage_fragility"] == pytest.approx(0.105)
    assert agg["flip_sign_at_reject_bps"] is True
    assert agg["reject"] is True


def test_main_raises_when_replay_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("global: {}\n", encoding="utf-8")

    monkeypatch.setattr(slippage_stress, "_resolve_backtester_cmd", lambda: ["mei-backtester"])
    monkeypatch.setattr(slippage_stress, "_run_cmd", lambda *args, **kwargs: 9)

    with pytest.raises(SystemExit, match="Replay failed for slippage_bps=10"):
        slippage_stress.main(
            [
                "--config",
                str(config_path),
                "--slippage-bps",
                "10,20",
                "--reject-flip-bps",
                "20",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
