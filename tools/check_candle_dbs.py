#!/usr/bin/env python3
"""Check candle DB freshness and gaps (per symbol, per interval).

This tool scans one or more per-interval candle SQLite DBs (typically:
`candles_dbs/candles_*.db`) and reports:

- Per symbol + interval: last candle timestamps (`t`, `t_close` where available)
- Data freshness: staleness of the latest *closed* candle relative to now
- Gap detection: missing candle opens within a recent lookback window
- Bar counts: number of bars observed in the lookback window vs expected

Output:
- Human summary is printed to stderr
- JSON report is printed to stdout

Exit codes:
- 0: all DBs are OK (or WARN)
- 1: at least one DB is WARN (only with --fail-on-warn)
- 2: at least one DB is FAIL
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


AIQ_ROOT = Path(__file__).resolve().parents[1]


class Status:
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


_STATUS_ORDER: dict[str, int] = {Status.OK: 0, Status.WARN: 1, Status.FAIL: 2}


def _severity_max(a: str, b: str) -> str:
    return a if _STATUS_ORDER.get(a, 0) >= _STATUS_ORDER.get(b, 0) else b


def _now_ms(*, override_now_ms: int | None) -> int:
    if override_now_ms is not None:
        return int(override_now_ms)
    return int(time.time() * 1000)


def _iso_utc_from_ms(ms: int | None) -> str | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _parse_interval_ms(interval: str) -> int | None:
    s = str(interval or "").strip().lower()
    if not s:
        return None
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() and not unit:
            num += ch
            continue
        unit += ch
    if not num or not unit:
        return None
    try:
        n = int(num)
    except Exception:
        return None
    if n <= 0:
        return None

    if unit == "s":
        return n * 1000
    if unit == "m":
        return n * 60 * 1000
    if unit == "h":
        return n * 60 * 60 * 1000
    if unit == "d":
        return n * 24 * 60 * 60 * 1000
    return None


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _csv_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").replace("\n", ",").split(","):
        sym = part.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


_INTERVAL_FROM_FILENAME = re.compile(r"^candles_(?P<interval>[^/\\\\]+)\\.db$", re.IGNORECASE)


def _interval_from_path(path: Path) -> str | None:
    m = _INTERVAL_FROM_FILENAME.match(path.name)
    if not m:
        return None
    iv = str(m.group("interval") or "").strip()
    return iv.lower() if iv else None


@dataclass(frozen=True)
class Thresholds:
    lookback_hours: float
    freshness_warn_mult: float
    freshness_fail_mult: float
    gap_warn_bars: int
    gap_fail_bars: int
    missing_warn_bars: int
    missing_fail_bars: int
    grace_seconds: float
    max_gap_ranges_per_symbol: int


def _status_for_value(*, value: float, warn_at: float, fail_at: float) -> str:
    if value >= float(fail_at):
        return Status.FAIL
    if value >= float(warn_at):
        return Status.WARN
    return Status.OK


def _status_for_int(*, value: int, warn_at: int, fail_at: int) -> str:
    if int(value) >= int(fail_at):
        return Status.FAIL
    if int(value) >= int(warn_at):
        return Status.WARN
    return Status.OK


def _last_times(
    cur: sqlite3.Cursor, *, symbol: str, interval: str, now_ms: int, grace_ms: int
) -> dict[str, int | None]:
    """Return last open/close times and last *closed* candle close time (best-effort)."""
    cur.execute(
        """
        SELECT t, t_close
        FROM candles
        WHERE symbol = ? AND interval = ?
        ORDER BY t DESC
        LIMIT 5
        """,
        (symbol, interval),
    )
    rows = cur.fetchall() or []

    last_t: int | None = None
    last_t_close: int | None = None
    last_closed_t_close: int | None = None

    for i, r in enumerate(rows):
        if i == 0:
            try:
                last_t = int(r["t"]) if r["t"] is not None else None
            except Exception:
                last_t = None
            try:
                last_t_close = int(r["t_close"]) if r["t_close"] is not None else None
            except Exception:
                last_t_close = None

        try:
            tc = r["t_close"]
        except Exception:
            tc = None
        if tc is None:
            continue
        try:
            tc_i = int(tc)
        except Exception:
            continue
        if tc_i <= int(now_ms) - int(grace_ms):
            last_closed_t_close = int(tc_i)
            break

    return {
        "last_t_ms": last_t,
        "last_t_close_ms": last_t_close,
        "last_closed_t_close_ms": last_closed_t_close,
    }


def _iter_t_values(cur: sqlite3.Cursor, *, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[int]:
    cur.execute(
        """
        SELECT t
        FROM candles
        WHERE symbol = ? AND interval = ? AND t >= ? AND t <= ?
        ORDER BY t ASC
        """,
        (symbol, interval, int(start_ms), int(end_ms)),
    )
    out: list[int] = []
    for r in cur:
        try:
            out.append(int(r[0]))
        except Exception:
            continue
    return out


def _detect_gaps(
    *,
    t_values_ms: list[int],
    interval_ms: int,
    expected_start_open_ms: int,
    expected_end_open_ms: int,
    max_ranges: int,
) -> dict:
    """Detect missing candle opens within [expected_start_open_ms, expected_end_open_ms]."""
    if interval_ms <= 0 or expected_end_open_ms < expected_start_open_ms:
        return {"missing_bars": 0, "max_gap_bars": 0, "gap_ranges": []}

    expected_bars = ((expected_end_open_ms - expected_start_open_ms) // interval_ms) + 1
    if expected_bars <= 0:
        return {"missing_bars": 0, "max_gap_bars": 0, "gap_ranges": []}

    missing_total = 0
    max_gap_bars = 0
    ranges: list[dict] = []

    if not t_values_ms:
        missing_total = int(expected_bars)
        max_gap_bars = int(expected_bars)
        ranges.append(
            {
                "start_ms": int(expected_start_open_ms),
                "end_ms": int(expected_end_open_ms),
                "missing_bars": int(expected_bars),
            }
        )
    else:
        first = int(t_values_ms[0])
        last = int(t_values_ms[-1])

        # Head gap
        if first > expected_start_open_ms:
            head_bars = (first - expected_start_open_ms) // interval_ms
            if head_bars > 0:
                missing_total += int(head_bars)
                max_gap_bars = max(max_gap_bars, int(head_bars))
                ranges.append(
                    {
                        "start_ms": int(expected_start_open_ms),
                        "end_ms": int(first - interval_ms),
                        "missing_bars": int(head_bars),
                    }
                )

        # Internal gaps
        prev = first
        for cur in t_values_ms[1:]:
            cur_i = int(cur)
            diff = cur_i - prev
            if diff > interval_ms:
                gap_bars = (diff // interval_ms) - 1
                if gap_bars > 0:
                    missing_total += int(gap_bars)
                    max_gap_bars = max(max_gap_bars, int(gap_bars))
                    ranges.append(
                        {
                            "start_ms": int(prev + interval_ms),
                            "end_ms": int(cur_i - interval_ms),
                            "missing_bars": int(gap_bars),
                        }
                    )
            prev = cur_i

        # Tail gap
        if last < expected_end_open_ms:
            tail_bars = (expected_end_open_ms - last) // interval_ms
            if tail_bars > 0:
                missing_total += int(tail_bars)
                max_gap_bars = max(max_gap_bars, int(tail_bars))
                ranges.append(
                    {
                        "start_ms": int(last + interval_ms),
                        "end_ms": int(expected_end_open_ms),
                        "missing_bars": int(tail_bars),
                    }
                )

    if ranges:
        ranges.sort(key=lambda r: int(r.get("missing_bars") or 0), reverse=True)
        ranges = ranges[: max(0, int(max_ranges))]

    for r in ranges:
        r["start_utc"] = _iso_utc_from_ms(r.get("start_ms"))
        r["end_utc"] = _iso_utc_from_ms(r.get("end_ms"))

    return {
        "missing_bars": int(missing_total),
        "max_gap_bars": int(max_gap_bars),
        "gap_ranges": ranges,
    }


def _fetch_intervals(cur: sqlite3.Cursor) -> list[str]:
    cur.execute("SELECT DISTINCT interval FROM candles")
    out: list[str] = []
    for r in cur.fetchall() or []:
        iv = str(r[0] or "").strip()
        if iv:
            out.append(iv)
    out.sort()
    return out


def _fetch_symbols(cur: sqlite3.Cursor, *, interval: str) -> list[str]:
    cur.execute("SELECT DISTINCT symbol FROM candles WHERE interval = ? ORDER BY symbol ASC", (interval,))
    out: list[str] = []
    for r in cur.fetchall() or []:
        s = str(r[0] or "").strip().upper()
        if s:
            out.append(s)
    return out


def check_db(
    *,
    db_path: Path,
    thresholds: Thresholds,
    now_ms: int,
    symbols_filter: list[str] | None,
) -> dict:
    res: dict = {"db_path": str(db_path), "status": Status.FAIL, "errors": [], "intervals": []}

    if not db_path.exists():
        res["errors"].append("db_missing")
        return res

    try:
        con = _connect_ro(db_path)
    except Exception as e:
        res["errors"].append(f"db_open_failed: {e}")
        return res

    try:
        cur = con.cursor()
        try:
            cur.execute("SELECT 1 FROM candles LIMIT 1")
            _ = cur.fetchone()
        except Exception as e:
            res["errors"].append(f"candles_table_missing_or_unreadable: {e}")
            return res

        intervals = _fetch_intervals(cur)
        if not intervals:
            iv = _interval_from_path(db_path)
            if iv:
                intervals = [iv]

        overall = Status.OK
        lookback_ms = int(float(thresholds.lookback_hours) * 3600.0 * 1000.0)
        grace_ms = int(float(thresholds.grace_seconds) * 1000.0)

        for iv in intervals:
            iv_ms = _parse_interval_ms(iv)
            iv_obj: dict = {
                "interval": iv,
                "interval_ms": iv_ms,
                "lookback": {
                    "hours": float(thresholds.lookback_hours),
                    "start_ms": int(now_ms - lookback_ms),
                    "start_utc": _iso_utc_from_ms(int(now_ms - lookback_ms)),
                },
                "symbols": [],
                "summary": {},
            }

            if iv_ms is not None and iv_ms > 0:
                start_open = _ceil_div(int(now_ms - lookback_ms), int(iv_ms)) * int(iv_ms)
                end_open = ((int(now_ms) - int(grace_ms) - int(iv_ms)) // int(iv_ms)) * int(iv_ms)
                if end_open < start_open:
                    expected_bars = 0
                else:
                    expected_bars = ((end_open - start_open) // int(iv_ms)) + 1
                iv_obj["lookback"].update(
                    {
                        "expected_start_open_ms": int(start_open),
                        "expected_start_open_utc": _iso_utc_from_ms(int(start_open)),
                        "expected_end_open_ms": int(end_open),
                        "expected_end_open_utc": _iso_utc_from_ms(int(end_open)),
                        "expected_bars": int(expected_bars),
                    }
                )
            else:
                start_open = None
                end_open = None
                expected_bars = None

            symbols = symbols_filter if symbols_filter is not None else _fetch_symbols(cur, interval=iv)
            if symbols_filter is not None:
                symbols = [s.upper() for s in symbols_filter]

            ok_n = warn_n = fail_n = 0
            worst_stale_ms: int | None = None
            worst_stale_sym: str | None = None
            worst_gap_bars: int | None = None
            worst_gap_sym: str | None = None

            for sym in symbols:
                sym_u = str(sym).upper()

                last = _last_times(cur, symbol=sym_u, interval=iv, now_ms=now_ms, grace_ms=grace_ms)
                last_t = last.get("last_t_ms")
                last_t_close = last.get("last_t_close_ms")
                last_closed_t_close = last.get("last_closed_t_close_ms")

                freshness_basis_ms = None
                if last_closed_t_close is not None:
                    freshness_basis_ms = int(last_closed_t_close)
                elif last_t is not None:
                    # Conservative: avoid using a future t_close from an open candle.
                    freshness_basis_ms = int(last_t)

                staleness_ms = None
                if freshness_basis_ms is not None:
                    staleness_ms = max(0, int(now_ms) - int(freshness_basis_ms))

                freshness_status = Status.OK
                if staleness_ms is None or iv_ms is None or iv_ms <= 0:
                    freshness_status = Status.WARN
                else:
                    warn_at = float(thresholds.freshness_warn_mult) * float(iv_ms)
                    fail_at = float(thresholds.freshness_fail_mult) * float(iv_ms)
                    freshness_status = _status_for_value(value=float(staleness_ms), warn_at=warn_at, fail_at=fail_at)

                # Gap detection
                gaps = {"missing_bars": 0, "max_gap_bars": 0, "gap_ranges": []}
                gap_status = Status.OK
                bars = 0

                if iv_ms is None or iv_ms <= 0 or start_open is None or end_open is None or expected_bars is None:
                    gap_status = Status.WARN
                elif int(expected_bars) <= 0:
                    gaps = {"missing_bars": 0, "max_gap_bars": 0, "gap_ranges": []}
                    gap_status = Status.OK
                else:
                    t_vals = _iter_t_values(
                        cur, symbol=sym_u, interval=iv, start_ms=int(start_open), end_ms=int(end_open)
                    )
                    bars = int(len(t_vals))
                    gaps = _detect_gaps(
                        t_values_ms=t_vals,
                        interval_ms=int(iv_ms),
                        expected_start_open_ms=int(start_open),
                        expected_end_open_ms=int(end_open),
                        max_ranges=int(thresholds.max_gap_ranges_per_symbol),
                    )
                    gap_status = _severity_max(
                        _status_for_int(
                            value=int(gaps["max_gap_bars"]),
                            warn_at=int(thresholds.gap_warn_bars),
                            fail_at=int(thresholds.gap_fail_bars),
                        ),
                        _status_for_int(
                            value=int(gaps["missing_bars"]),
                            warn_at=int(thresholds.missing_warn_bars),
                            fail_at=int(thresholds.missing_fail_bars),
                        ),
                    )

                status = _severity_max(freshness_status, gap_status)
                if status == Status.OK:
                    ok_n += 1
                elif status == Status.WARN:
                    warn_n += 1
                else:
                    fail_n += 1

                if staleness_ms is not None:
                    if worst_stale_ms is None or int(staleness_ms) > int(worst_stale_ms):
                        worst_stale_ms = int(staleness_ms)
                        worst_stale_sym = sym_u

                mgb = int(gaps.get("max_gap_bars") or 0)
                if worst_gap_bars is None or mgb > int(worst_gap_bars):
                    worst_gap_bars = mgb
                    worst_gap_sym = sym_u

                iv_obj["symbols"].append(
                    {
                        "symbol": sym_u,
                        "status": status,
                        "freshness": {
                            "status": freshness_status,
                            "staleness_ms": staleness_ms,
                            "staleness_s": (float(staleness_ms) / 1000.0) if staleness_ms is not None else None,
                            "last_t_ms": last_t,
                            "last_t_utc": _iso_utc_from_ms(last_t),
                            "last_t_close_ms": last_t_close,
                            "last_t_close_utc": _iso_utc_from_ms(last_t_close),
                            "last_closed_t_close_ms": last_closed_t_close,
                            "last_closed_t_close_utc": _iso_utc_from_ms(last_closed_t_close),
                        },
                        "lookback": {
                            "bars": int(bars),
                            "expected_bars": int(expected_bars) if expected_bars is not None else None,
                            "bars_ratio": (float(bars) / float(expected_bars)) if expected_bars else None,
                            "gaps": gaps,
                        },
                    }
                )

                overall = _severity_max(overall, status)

            iv_obj["summary"] = {
                "symbols": int(ok_n + warn_n + fail_n),
                "ok": int(ok_n),
                "warn": int(warn_n),
                "fail": int(fail_n),
                "worst_staleness_ms": worst_stale_ms,
                "worst_staleness_s": (float(worst_stale_ms) / 1000.0) if worst_stale_ms is not None else None,
                "worst_staleness_symbol": worst_stale_sym,
                "worst_gap_bars": worst_gap_bars,
                "worst_gap_symbol": worst_gap_sym,
            }
            res["intervals"].append(iv_obj)

        res["status"] = overall
        return res
    finally:
        try:
            con.close()
        except Exception:
            pass


def _default_db_glob() -> str:
    raw_dir = str(os.getenv("AI_QUANT_CANDLES_DB_DIR", "") or "").strip()
    if raw_dir:
        return str(Path(raw_dir) / "candles_*.db")
    return str(AIQ_ROOT / "candles_dbs" / "candles_*.db")


def _summarise_human(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"Generated: {report.get('generated_at_utc')}  Lookback: {report.get('lookback_hours')}h")
    lines.append(f"Overall: {report.get('overall_status')}")
    lines.append("")

    dbs = report.get("dbs") or []
    for db in dbs:
        status = str(db.get("status") or Status.FAIL)
        path = str(db.get("db_path") or "")
        errs = db.get("errors") or []
        if errs:
            lines.append(f"[{status}] {path}  errors={len(errs)}")
            for e in errs[:10]:
                lines.append(f"  - {e}")
            continue

        intervals = db.get("intervals") or []
        sym_total = 0
        sym_fail = 0
        worst_gap = None
        worst_gap_sym = None
        worst_stale_s = None
        worst_stale_sym = None
        for iv in intervals:
            summ = iv.get("summary") or {}
            try:
                sym_total += int(summ.get("symbols") or 0)
                sym_fail += int(summ.get("fail") or 0)
            except Exception:
                pass
            try:
                g = summ.get("worst_gap_bars")
                if g is not None:
                    g_i = int(g)
                    if worst_gap is None or g_i > int(worst_gap):
                        worst_gap = g_i
                        worst_gap_sym = summ.get("worst_gap_symbol")
            except Exception:
                pass
            try:
                s = summ.get("worst_staleness_s")
                if s is not None:
                    s_f = float(s)
                    if worst_stale_s is None or s_f > float(worst_stale_s):
                        worst_stale_s = s_f
                        worst_stale_sym = summ.get("worst_staleness_symbol")
            except Exception:
                pass

        iv_names = ",".join([str(iv.get("interval") or "") for iv in intervals]) or "?"
        extra = [f"intervals={iv_names}", f"symbols={sym_total}", f"fail_symbols={sym_fail}"]
        if worst_stale_s is not None:
            extra.append(f"worst_stale={worst_stale_s:.1f}s({worst_stale_sym})")
        if worst_gap is not None:
            extra.append(f"worst_gap={worst_gap}bars({worst_gap_sym})")
        lines.append(f"[{status}] {path}  " + " ".join(extra))

    return "\n".join(lines).rstrip() + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check candle DB freshness and gaps")
    p.add_argument(
        "--db-glob",
        type=str,
        default=_default_db_glob(),
        help="Glob for candle DB paths (default: derived from AI_QUANT_CANDLES_DB_DIR or ./candles_dbs/candles_*.db)",
    )
    p.add_argument(
        "--lookback-hours",
        type=float,
        default=24.0,
        help="Lookback window size for gap detection (default: 24)",
    )
    p.add_argument(
        "--freshness-warn-mult",
        type=float,
        default=2.0,
        help="Warn if staleness >= (this * interval) (default: 2.0)",
    )
    p.add_argument(
        "--freshness-fail-mult",
        type=float,
        default=5.0,
        help="Fail if staleness >= (this * interval) (default: 5.0)",
    )
    p.add_argument(
        "--gap-warn-bars",
        type=int,
        default=1,
        help="Warn if the maximum consecutive missing bars within lookback >= this (default: 1)",
    )
    p.add_argument(
        "--gap-fail-bars",
        type=int,
        default=3,
        help="Fail if the maximum consecutive missing bars within lookback >= this (default: 3)",
    )
    p.add_argument(
        "--missing-warn-bars",
        type=int,
        default=10,
        help="Warn if total missing bars within lookback >= this (default: 10)",
    )
    p.add_argument(
        "--missing-fail-bars",
        type=int,
        default=50,
        help="Fail if total missing bars within lookback >= this (default: 50)",
    )
    p.add_argument(
        "--grace-seconds",
        type=float,
        default=5.0,
        help="Grace period for considering a candle 'closed' (default: 5.0)",
    )
    p.add_argument(
        "--max-gap-ranges-per-symbol",
        type=int,
        default=5,
        help="Maximum gap ranges to include per symbol in JSON (default: 5)",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated symbol filter (default: all symbols found in each DB)",
    )
    p.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero when overall status is WARN",
    )
    p.add_argument(
        "--now-ms",
        type=int,
        default=None,
        help="Override current time in ms since epoch (testing / reproducibility)",
    )
    p.add_argument(
        "--json-indent",
        type=int,
        default=None,
        help="Pretty-print JSON with the given indent (default: compact)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if float(args.freshness_warn_mult) > float(args.freshness_fail_mult):
        raise SystemExit("--freshness-warn-mult must be <= --freshness-fail-mult")
    if int(args.gap_warn_bars) > int(args.gap_fail_bars):
        raise SystemExit("--gap-warn-bars must be <= --gap-fail-bars")
    if int(args.missing_warn_bars) > int(args.missing_fail_bars):
        raise SystemExit("--missing-warn-bars must be <= --missing-fail-bars")

    thresholds = Thresholds(
        lookback_hours=float(args.lookback_hours),
        freshness_warn_mult=float(args.freshness_warn_mult),
        freshness_fail_mult=float(args.freshness_fail_mult),
        gap_warn_bars=int(args.gap_warn_bars),
        gap_fail_bars=int(args.gap_fail_bars),
        missing_warn_bars=int(args.missing_warn_bars),
        missing_fail_bars=int(args.missing_fail_bars),
        grace_seconds=float(args.grace_seconds),
        max_gap_ranges_per_symbol=int(args.max_gap_ranges_per_symbol),
    )

    now_ms = _now_ms(override_now_ms=args.now_ms)
    symbols_filter = _csv_symbols(args.symbols) if str(args.symbols or "").strip() else None

    db_paths = sorted({Path(p) for p in glob.glob(str(args.db_glob), recursive=True)})

    report: dict = {
        "generated_at_ms": int(now_ms),
        "generated_at_utc": _iso_utc_from_ms(int(now_ms)),
        "db_glob": str(args.db_glob),
        "lookback_hours": float(thresholds.lookback_hours),
        "thresholds": {
            "freshness_warn_mult": float(thresholds.freshness_warn_mult),
            "freshness_fail_mult": float(thresholds.freshness_fail_mult),
            "gap_warn_bars": int(thresholds.gap_warn_bars),
            "gap_fail_bars": int(thresholds.gap_fail_bars),
            "missing_warn_bars": int(thresholds.missing_warn_bars),
            "missing_fail_bars": int(thresholds.missing_fail_bars),
            "grace_seconds": float(thresholds.grace_seconds),
        },
        "overall_status": Status.OK,
        "dbs": [],
        "summary": {},
    }

    if not db_paths:
        report["overall_status"] = Status.FAIL
        report["dbs"].append(
            {
                "db_path": None,
                "status": Status.FAIL,
                "errors": [f"no_dbs_matched_glob: {args.db_glob}"],
                "intervals": [],
            }
        )
    else:
        overall = Status.OK
        for pth in db_paths:
            db_res = check_db(db_path=pth, thresholds=thresholds, now_ms=now_ms, symbols_filter=symbols_filter)
            report["dbs"].append(db_res)
            overall = _severity_max(overall, str(db_res.get("status") or Status.FAIL))
        report["overall_status"] = overall

    dbs = report.get("dbs") or []
    report["summary"] = {
        "dbs_total": len(dbs),
        "dbs_ok": sum(1 for d in dbs if d.get("status") == Status.OK),
        "dbs_warn": sum(1 for d in dbs if d.get("status") == Status.WARN),
        "dbs_fail": sum(1 for d in dbs if d.get("status") == Status.FAIL),
    }

    try:
        sys.stderr.write(_summarise_human(report))
    except Exception:
        pass

    json.dump(report, sys.stdout, indent=args.json_indent, sort_keys=True)
    sys.stdout.write("\n")

    if report["overall_status"] == Status.FAIL:
        return 2
    if report["overall_status"] == Status.WARN and bool(args.fail_on_warn):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
