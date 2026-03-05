from __future__ import annotations

from collections import Counter
import json
import sqlite3
import stat
from pathlib import Path

import factory_run

_FIXED_TS_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z — stable test anchor


_CANDLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    t INTEGER NOT NULL,
    t_close INTEGER,
    o REAL,
    h REAL,
    l REAL,
    c REAL,
    v REAL,
    n INTEGER,
    updated_at TEXT,
    PRIMARY KEY (symbol, interval, t)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_t
ON candles(symbol, interval, t);
"""


_FUNDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    time INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    premium REAL,
    PRIMARY KEY (symbol, time)
);
"""


def _mk_candles_db(path: Path, *, symbol: str, interval: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.executescript(_CANDLES_SCHEMA)
        interval_ms = 3_600_000 if interval == "1h" else 60_000
        now_ms = _FIXED_TS_MS
        anchor = (now_ms // interval_ms) * interval_ms

        # Generate enough contiguous history so default factory data checks pass (200 bars).
        bars_needed = 240
        start = anchor - ((bars_needed - 1) * interval_ms)
        rows = []
        for t in range(int(start), int(anchor) + interval_ms, interval_ms):
            rows.append(
                (
                    symbol,
                    interval,
                    int(t),
                    int(t + interval_ms),
                    float(100 + (t - start) / interval_ms),
                    float(101 + (t - start) / interval_ms),
                    float(99 + (t - start) / interval_ms),
                    float(100 + (t - start) / interval_ms),
                    1.0,
                )
            )
        con.executemany(
            "INSERT OR REPLACE INTO candles (symbol, interval, t, t_close, o, h, l, c, v) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def _mk_funding_db(path: Path, *, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.executescript(_FUNDING_SCHEMA)
        now_ms = _FIXED_TS_MS
        h = 3_600_000
        rows = []
        # Generate >72h of hourly rows to satisfy the default lookback window.
        for i in range(0, 80):
            rows.append((symbol, int(now_ms - (79 - i) * h), 0.0001))
        con.executemany(
            "INSERT OR REPLACE INTO funding_rates (symbol, time, funding_rate, premium) VALUES (?, ?, ?, NULL)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def _mk_stub_backtester(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys

def _arg(flag: str):
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except Exception:
        return None

def main() -> int:
    if len(sys.argv) < 2:
        return 2
    cmd = sys.argv[1]
    if cmd == "sweep":
        out = _arg("--output")
        if not out:
            return 2
        # Keep total_trades >= 20 because tools/generate_config.py filters by --min-trades (default 20).
        rows = []
        for idx in range(30):
            n = idx + 1
            rows.append(
                {
                    "total_pnl": 10.0 - (idx * 0.15),
                    "final_balance": 10000.0 + 10.0 - (idx * 0.15),
                    "total_trades": 25,
                    "win_rate": 0.60 - (idx * 0.002),
                    "profit_factor": 1.20 - (idx * 0.005),
                    "max_drawdown_pct": 0.05 - (idx * 0.0005),
                    "sharpe_ratio": 0.10 - (idx * 0.001),
                    "overrides": [["trade.leverage", float(3.0 + idx)], ["indicators.ema_fast_window", float(21.0 + idx)]],
                }
            )
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\\n")
        return 0
    if cmd == "replay":
        out = _arg("--output")
        if not out:
            return 2
        rep = {
            "initial_balance": 10000.0,
            "final_balance": 10005.0,
            "total_pnl": 5.0,
            "total_trades": 25,
            "win_rate": 0.60,
            "profit_factor": 1.20,
            "max_drawdown_pct": 0.02,
            "total_fees": 1.0,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rep, f)
        return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    st = path.stat()
    path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_base_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "global:",
                "  trade:",
                "    allocation_pct: 0.03",
                "    leverage: 3.0",
                "    sl_atr_mult: 2.0",
                "    tp_atr_mult: 4.0",
                "    slippage_bps: 10.0",
                "    max_open_positions: 20",
                "    max_total_margin_pct: 0.6",
                "    min_notional_usd: 10.0",
                "    min_atr_pct: 0.003",
                "    bump_to_min_notional: false",
                "  indicators:",
                "    adx_window: 14",
                "    ema_fast_window: 20",
                "    ema_slow_window: 50",
                "    bb_window: 20",
                "    atr_window: 14",
                "  thresholds:",
                "    entry:",
                "      min_adx: 22.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_factory_inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    artifacts_root = tmp_path / "artifacts"
    cfg = tmp_path / "base.yaml"
    _write_base_config(cfg)

    candles_db = tmp_path / "candles_1h.db"
    _mk_candles_db(candles_db, symbol="BTC", interval="1h")

    funding_db = tmp_path / "funding_rates.db"
    _mk_funding_db(funding_db, symbol="BTC")

    bt_stub = tmp_path / "mei-backtester-stub.py"
    _mk_stub_backtester(bt_stub)
    monkeypatch.setenv("MEI_BACKTESTER_BIN", str(bt_stub))

    return artifacts_root, cfg, candles_db, funding_db


def _shortlist_stub_identity(mode: str, rank: int) -> str | None:
    if mode == "efficient" and 1 <= rank <= 10:
        return f"shared_{rank:02d}"
    if mode == "growth" and 1 <= rank <= 20:
        return f"shared_{rank:02d}"
    if mode == "conservative" and 1 <= rank <= 25:
        return f"shared_{rank:02d}"
    return None


def _make_shortlist_generate_config_stub(real_run_cmd, *, exhaustion_is_graceful: bool):
    def _fake_run_cmd(argv, **kwargs):
        argv_list = [str(part) for part in argv]
        if argv_list[:2] != ["python3", "tools/generate_config.py"]:
            return real_run_cmd(argv, **kwargs)

        mode = argv_list[argv_list.index("--sort-by") + 1]
        rank = int(argv_list[argv_list.index("--rank") + 1])
        out_path = Path(argv_list[argv_list.index("-o") + 1])
        stdout_path = Path(str(kwargs["stdout_path"]))
        stderr_path = Path(str(kwargs["stderr_path"]))
        cwd = str(kwargs["cwd"])

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        cfg_identity = _shortlist_stub_identity(mode, rank)
        if cfg_identity is None:
            if mode == "conservative" and rank == 26 and exhaustion_is_graceful:
                stderr_path.write_text(
                    "[generate] Loaded 25 results, sorted by conservative\n[generate] Rank 26 out of range (1..25).\n",
                    encoding="utf-8",
                )
                stdout_path.write_text("", encoding="utf-8")
                return factory_run.CmdResult(
                    argv=argv_list,
                    cwd=cwd,
                    exit_code=1,
                    elapsed_s=0.01,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                )

            stderr_path.write_text("ValueError: malformed shortlist row\n", encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
            return factory_run.CmdResult(
                argv=argv_list,
                cwd=cwd,
                exit_code=2,
                elapsed_s=0.01,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(
                [
                    "global:",
                    "  engine:",
                    "    interval: 1h",
                    f"  candidate_marker: {cfg_identity}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return factory_run.CmdResult(
            argv=argv_list,
            cwd=cwd,
            exit_code=0,
            elapsed_s=0.01,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    return _fake_run_cmd


def test_factory_smoke_run_produces_artifacts(tmp_path, monkeypatch) -> None:
    artifacts_root, cfg, candles_db, funding_db = _prepare_factory_inputs(tmp_path, monkeypatch)

    run_id = "test_smoke"
    rc = factory_run.main(
        [
            "--run-id",
            run_id,
            "--profile",
            "smoke",
            "--artifacts-dir",
            str(artifacts_root),
            "--config",
            str(cfg),
            "--interval",
            "1h",
            "--candles-db",
            str(candles_db),
            "--funding-db",
            str(funding_db),
            "--sweep-spec",
            str(tmp_path / "sweep.yaml"),
            "--shortlist-per-mode",
            "0",
            "--num-candidates",
            "2",
        ]
    )
    assert rc == 0

    metas = list(artifacts_root.rglob("run_metadata.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id

    run_dir = Path(meta["run_dir"])
    assert (run_dir / "reports" / "report.md").exists()
    assert (run_dir / "reports" / "report.json").exists()
    assert (run_dir / "reports" / "validation_report.md").exists()
    assert isinstance(meta.get("registry_db"), str)
    assert Path(str(meta.get("registry_db"))).exists()
    assert (artifacts_root / "registry" / "registry.sqlite").exists()

    candidates = meta.get("candidate_configs", [])
    assert isinstance(candidates, list)
    assert candidates, "candidate metadata should be recorded"
    assert candidates[0].get("sweep_stage") in {"cpu", "gpu", "gpu_tpe"}
    assert candidates[0].get("validation_gate") in {
        "replay_only",
        "score_v1+walk_forward",
        "score_v1+walk_forward+slippage",
    }
    assert "canonical_cpu_verified" in candidates[0]

    report = json.loads((run_dir / "reports" / "report.json").read_text(encoding="utf-8"))
    items = report.get("items", []) if isinstance(report, dict) else []
    assert items and isinstance(items[0], dict)
    assert "sweep_stage" in items[0]
    assert "replay_stage" in items[0]
    assert "canonical_cpu_verified" in items[0]


def test_factory_smoke_shortlist_out_of_range_exhaustion_is_graceful(tmp_path, monkeypatch) -> None:
    artifacts_root, cfg, candles_db, funding_db = _prepare_factory_inputs(tmp_path, monkeypatch)
    real_run_cmd = factory_run._run_cmd
    monkeypatch.setattr(
        factory_run,
        "_run_cmd",
        _make_shortlist_generate_config_stub(real_run_cmd, exhaustion_is_graceful=True),
    )

    run_id = "test_shortlist_exhaustion"
    rc = factory_run.main(
        [
            "--run-id",
            run_id,
            "--profile",
            "smoke",
            "--artifacts-dir",
            str(artifacts_root),
            "--config",
            str(cfg),
            "--interval",
            "1h",
            "--candles-db",
            str(candles_db),
            "--funding-db",
            str(funding_db),
            "--sweep-spec",
            str(tmp_path / "sweep.yaml"),
            "--shortlist-modes",
            "efficient,growth,conservative",
            "--shortlist-per-mode",
            "10",
            "--shortlist-max-rank",
            "1000",
            "--shortlist-top-pnl",
            "1000",
        ]
    )

    assert rc == 0

    metas = list(artifacts_root.rglob("run_metadata.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id

    selected = [entry for entry in meta.get("candidate_configs", []) if entry.get("selected")]
    assert Counter(entry["sort_by"] for entry in selected) == {
        "efficient": 10,
        "growth": 10,
        "conservative": 5,
    }

    run_dir = Path(meta["run_dir"])
    assert (run_dir / "reports" / "report.json").exists()


def test_factory_smoke_shortlist_non_exhaustion_generate_error_still_fails(tmp_path, monkeypatch) -> None:
    artifacts_root, cfg, candles_db, funding_db = _prepare_factory_inputs(tmp_path, monkeypatch)
    real_run_cmd = factory_run._run_cmd
    monkeypatch.setattr(
        factory_run,
        "_run_cmd",
        _make_shortlist_generate_config_stub(real_run_cmd, exhaustion_is_graceful=False),
    )

    rc = factory_run.main(
        [
            "--run-id",
            "test_shortlist_genuine_error",
            "--profile",
            "smoke",
            "--artifacts-dir",
            str(artifacts_root),
            "--config",
            str(cfg),
            "--interval",
            "1h",
            "--candles-db",
            str(candles_db),
            "--funding-db",
            str(funding_db),
            "--sweep-spec",
            str(tmp_path / "sweep.yaml"),
            "--shortlist-modes",
            "efficient,growth,conservative",
            "--shortlist-per-mode",
            "10",
            "--shortlist-max-rank",
            "1000",
            "--shortlist-top-pnl",
            "1000",
        ]
    )

    assert rc == 2

    meta_path = next(artifacts_root.rglob("run_metadata.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert any(
        step.get("name") == "generate_config_conservative_rank26" and int(step.get("exit_code", 0)) == 2
        for step in meta.get("steps", [])
    )
    assert "candidate_configs" not in meta
