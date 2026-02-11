#!/usr/bin/env python3
"""Synthetic CPU/GPU signal-capture experiment harness.

This script creates deterministic synthetic candle databases (1h + 3m),
runs the same sweep on CPU and GPU, and checks whether each target parameter
is actually captured by both engines.

Focus parameters:
- trade.sl_atr_mult
- trade.tp_atr_mult
- trade.allocation_pct
- trade.trailing_start_atr
- trade.trailing_distance_atr
- trade.enable_partial_tp
- trade.tp_partial_pct
- trade.tp_partial_atr_mult
- trade.max_total_margin_pct
- trade.max_open_positions
- trade.leverage
- trade.smart_exit_adx_exhaustion_lt
- trade.smart_exit_adx_exhaustion_lt_low_conf
- filters.require_macro_alignment (explicit false/true gate validation)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sqlite3
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AXIS_ORDER: Sequence[str] = (
    "trade.sl_atr_mult",
    "trade.tp_atr_mult",
    "trade.allocation_pct",
    "trade.trailing_start_atr",
    "trade.trailing_distance_atr",
    "trade.enable_partial_tp",
    "trade.tp_partial_pct",
    "trade.tp_partial_atr_mult",
    "trade.max_total_margin_pct",
    "trade.max_open_positions",
    "trade.leverage",
    "trade.smart_exit_adx_exhaustion_lt",
    "trade.smart_exit_adx_exhaustion_lt_low_conf",
    "filters.require_macro_alignment",
)

AXIS_VALUES: Dict[str, Sequence[float]] = {
    "trade.sl_atr_mult": (1.5, 2.0, 2.5, 3.0),
    "trade.tp_atr_mult": (3.0, 4.0, 5.0, 6.0),
    "trade.allocation_pct": (0.10, 0.15, 0.20),
    "trade.trailing_start_atr": (1.5, 2.0, 2.5),
    "trade.trailing_distance_atr": (0.5, 1.0),
    "trade.enable_partial_tp": (0.0, 1.0),
    "trade.tp_partial_pct": (0.25, 0.50, 0.75),
    "trade.tp_partial_atr_mult": (0.5, 1.0, 2.0),
    "trade.max_total_margin_pct": (0.20, 0.40, 0.60, 0.80),
    "trade.max_open_positions": (1.0, 2.0, 4.0),
    "trade.leverage": (1.0, 2.0, 3.0, 5.0),
    "trade.smart_exit_adx_exhaustion_lt": (0.0, 20.0, 200.0),
    "trade.smart_exit_adx_exhaustion_lt_low_conf": (0.0, 20.0, 200.0),
    "filters.require_macro_alignment": (0.0, 1.0),  # false, true
}

SCHEMA_VERSION = 3

AXIS_SCENARIOS: Dict[str, str] = {
    "trade.sl_atr_mult": "sl_isolation",
    "trade.tp_atr_mult": "tp_isolation",
    "trade.allocation_pct": "allocation_dual",
    "trade.trailing_start_atr": "trailing_single",
    "trade.trailing_distance_atr": "trailing_single",
    "trade.enable_partial_tp": "partial_tp_toggle",
    "trade.tp_partial_pct": "partial_tp_pct",
    "trade.tp_partial_atr_mult": "partial_tp_atr_mult",
    "trade.max_total_margin_pct": "margin_cap_isolation",
    "trade.max_open_positions": "max_positions_isolation",
    "trade.leverage": "leverage_isolation",
    "trade.smart_exit_adx_exhaustion_lt": "smart_exit_high_conf",
    "trade.smart_exit_adx_exhaustion_lt_low_conf": "smart_exit_low_conf",
    "filters.require_macro_alignment": "macro_alignment_gate",
}


@dataclass(frozen=True)
class ExpectationSpec:
    metric: str
    relation: str
    note: str
    require_direction_match: bool = False
    require_triggered: bool = True


AXIS_EXPECTATIONS: Dict[str, ExpectationSpec] = {
    "trade.sl_atr_mult": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Changing SL multiplier should alter aggregate PnL.",
        require_direction_match=True,
    ),
    "trade.tp_atr_mult": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Changing TP multiplier should alter aggregate PnL.",
        require_direction_match=True,
    ),
    "trade.allocation_pct": ExpectationSpec(
        metric="total_pnl",
        relation="increase",
        note="Higher allocation should increase PnL under positive trend fixture.",
    ),
    "trade.trailing_start_atr": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Trailing activation threshold should change realised PnL.",
    ),
    "trade.trailing_distance_atr": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Trailing distance should change realised PnL.",
    ),
    "trade.enable_partial_tp": ExpectationSpec(
        metric="total_trades",
        relation="change",
        note="Enabling partial TP should change trade count in partial-only fixture.",
    ),
    "trade.tp_partial_pct": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Partial TP percentage should change realised PnL.",
    ),
    "trade.tp_partial_atr_mult": ExpectationSpec(
        metric="total_pnl",
        relation="change",
        note="Partial TP ATR trigger level should change realised PnL.",
    ),
    "trade.max_total_margin_pct": ExpectationSpec(
        metric="total_pnl",
        relation="increase",
        note="More margin headroom should increase realised PnL in profitable multi-symbol fixture.",
    ),
    "trade.max_open_positions": ExpectationSpec(
        metric="total_trades",
        relation="increase",
        note="Allowing more concurrent positions should increase executed trades.",
    ),
    "trade.leverage": ExpectationSpec(
        metric="total_pnl",
        relation="increase",
        note="Higher leverage should increase realised PnL in profitable fixture.",
    ),
    "trade.smart_exit_adx_exhaustion_lt": ExpectationSpec(
        metric="total_trades",
        relation="change",
        note="Raising ADX exhaustion threshold should alter exit churn.",
    ),
    "trade.smart_exit_adx_exhaustion_lt_low_conf": ExpectationSpec(
        metric="total_trades",
        relation="change",
        note="Low-confidence ADX exhaustion override should alter low-confidence exit churn.",
    ),
    "filters.require_macro_alignment": ExpectationSpec(
        metric="total_trades",
        relation="change",
        note="Macro alignment gate false/true should alter captured trades.",
    ),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    config_yaml: str
    anchors: Dict[str, float]
    main_rows: List[Tuple[object, ...]]
    sub_rows: List[Tuple[object, ...]]


@dataclass(frozen=True)
class ExperimentSpec:
    axis: str
    scenario_name: str
    expectation: ExpectationSpec


def _default_anchors() -> Dict[str, float]:
    return {
        "trade.sl_atr_mult": 1.5,
        "trade.tp_atr_mult": 3.0,
        "trade.allocation_pct": 0.10,
        "trade.trailing_start_atr": 1.5,
        "trade.trailing_distance_atr": 0.5,
        "trade.enable_partial_tp": 0.0,
        "trade.tp_partial_pct": 0.50,
        "trade.tp_partial_atr_mult": 1.0,
        "trade.max_total_margin_pct": 1.0,
        "trade.max_open_positions": 20.0,
        "trade.leverage": 1.0,
        "trade.smart_exit_adx_exhaustion_lt": 0.0,
        "trade.smart_exit_adx_exhaustion_lt_low_conf": 0.0,
        "filters.require_macro_alignment": 0.0,
    }


def _anchors(**overrides: float) -> Dict[str, float]:
    anchors = _default_anchors()
    anchors.update(overrides)
    return anchors


def _parse_axes(raw_axes: str) -> List[str]:
    if not raw_axes.strip():
        return list(AXIS_ORDER)

    seen = set()
    out: List[str] = []
    for axis in raw_axes.split(","):
        key = axis.strip()
        if not key:
            continue
        if key not in AXIS_ORDER:
            valid = ", ".join(AXIS_ORDER)
            raise SystemExit(f"Unknown axis '{key}'. Valid axes: {valid}")
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected fragment not found for replacement: {old!r}")
    return text.replace(old, new, 1)


def _create_candle_db(path: Path, rows: Iterable[Tuple[object, ...]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("DROP TABLE IF EXISTS candles")
        con.execute(
            """
            CREATE TABLE candles (
                symbol TEXT,
                interval TEXT,
                t INTEGER,
                t_close INTEGER,
                o REAL,
                h REAL,
                l REAL,
                c REAL,
                v REAL,
                n INTEGER
            )
            """
        )
        con.executemany(
            "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?)",
            list(rows),
        )
        con.execute("CREATE INDEX idx_candles_symbol_interval_t ON candles(symbol, interval, t)")
        con.commit()
    finally:
        con.close()


def _build_symbol_rows(
    symbol: str,
    closes: Sequence[float],
    main_pad: float,
    rel_path: Sequence[float],
    rel_high_pad: float,
    rel_low_pad: float,
    offset: float = 0.0,
    scale_alt: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[List[Tuple[object, ...]], List[Tuple[object, ...]]]:
    main_rows: List[Tuple[object, ...]] = []
    sub_rows: List[Tuple[object, ...]] = []

    for i, raw_c in enumerate(closes):
        c = raw_c + offset
        o = (closes[i - 1] + offset) if i > 0 else c
        t = i * 3_600_000
        h = max(o, c) + main_pad
        l = min(o, c) - main_pad
        main_rows.append((symbol, "1h", t, t + 3_600_000, o, h, l, c, 1000.0, 1))

    for i in range(1, len(closes)):
        base = closes[i - 1] + offset
        start = (i - 1) * 3_600_000
        scale = scale_alt[0] if i % 2 == 0 else scale_alt[1]
        for j, rel in enumerate(rel_path, 1):
            px = base + rel * scale
            t = start + j * 180_000
            sub_rows.append(
                (
                    symbol,
                    "3m",
                    t,
                    t + 180_000,
                    px,
                    px + rel_high_pad,
                    px - rel_low_pad,
                    px,
                    900.0,
                    1,
                )
            )

    return main_rows, sub_rows


def _portfolio_fixture_rows(
    symbols: Sequence[Tuple[str, float]],
) -> Tuple[List[Tuple[object, ...]], List[Tuple[object, ...]]]:
    closes = [100.0, 101.0, 102.0, 102.5, 102.0, 103.0, 102.7, 103.5, 103.2, 104.0]
    rel = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.35,
        0.55,
        0.85,
        1.15,
        0.90,
        0.70,
        0.55,
        0.45,
        0.35,
        0.25,
        0.10,
        -0.15,
        -0.35,
        -0.15,
        0.05,
        0.18,
    ]

    main_rows: List[Tuple[object, ...]] = []
    sub_rows: List[Tuple[object, ...]] = []
    for symbol, offset in symbols:
        symbol_main, symbol_sub = _build_symbol_rows(
            symbol=symbol,
            closes=closes,
            main_pad=0.20,
            rel_path=rel,
            rel_high_pad=0.05,
            rel_low_pad=0.05,
            offset=offset,
            scale_alt=(1.0, 1.25),
        )
        main_rows.extend(symbol_main)
        sub_rows.extend(symbol_sub)
    return main_rows, sub_rows


def _scenario_trailing() -> Scenario:
    cfg = textwrap.dedent(
        """
        global:
          trade:
            allocation_pct: 0.10
            sl_atr_mult: 1.5
            tp_atr_mult: 3.0
            leverage: 1.0
            max_open_positions: 20
            max_total_margin_pct: 1.0
            min_notional_usd: 1.0
            min_atr_pct: 0.0
            bump_to_min_notional: false
            enable_dynamic_sizing: false
            enable_dynamic_leverage: false
            enable_pyramiding: false
            enable_partial_tp: false
            enable_ssf_filter: false
            enable_reef_filter: false
            enable_breakeven_stop: false
            enable_rsi_overextension_exit: false
            enable_vol_buffered_trailing: false
            tsme_require_adx_slope_negative: false
            reverse_entry_signal: false
            block_exits_on_extreme_dev: false
            entry_min_confidence: low
            add_min_confidence: low
            trailing_start_atr: 1.5
            trailing_distance_atr: 0.5
            smart_exit_adx_exhaustion_lt: 0.0
            smart_exit_adx_exhaustion_lt_low_conf: 0.0
            slippage_bps: 0.0
          indicators:
            ema_fast_window: 2
            ema_slow_window: 3
            ema_macro_window: 3
            adx_window: 2
            atr_window: 2
            rsi_window: 2
            bb_window: 2
            bb_width_avg_window: 2
            vol_sma_window: 2
            vol_trend_window: 2
            stoch_rsi_window: 2
            stoch_rsi_smooth1: 1
            stoch_rsi_smooth2: 1
          filters:
            enable_ranging_filter: false
            enable_anomaly_filter: false
            enable_extension_filter: false
            require_adx_rising: false
            require_volume_confirmation: false
            use_stoch_rsi_filter: false
            require_btc_alignment: false
            require_macro_alignment: false
          thresholds:
            entry:
              min_adx: -1.0
              max_dist_ema_fast: 1.0
              macd_hist_entry_mode: none
              ave_enabled: false
              enable_pullback_entries: false
              enable_slow_drift_entries: false
            stoch_rsi:
              block_long_if_k_gt: 1.0
              block_short_if_k_lt: 0.0
        """
    ).strip() + "\n"

    closes = [100.0, 101.0, 102.0, 102.5, 102.0, 103.0, 102.7, 103.5, 103.2, 104.0]
    rel = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.35,
        0.55,
        0.85,
        1.15,
        0.90,
        0.70,
        0.55,
        0.45,
        0.35,
        0.25,
        0.10,
        -0.15,
        -0.35,
        -0.15,
        0.05,
        0.18,
    ]
    main_rows, sub_rows = _build_symbol_rows(
        symbol="BTC",
        closes=closes,
        main_pad=0.20,
        rel_path=rel,
        rel_high_pad=0.05,
        rel_low_pad=0.05,
        scale_alt=(1.0, 1.25),
    )

    anchors = _anchors()
    return Scenario(
        name="trailing_single",
        description="Single-symbol trend/retrace path to force trailing threshold decisions.",
        config_yaml=cfg,
        anchors=anchors,
        main_rows=main_rows,
        sub_rows=sub_rows,
    )


def _scenario_allocation_dual() -> Scenario:
    base = _scenario_trailing()
    cfg = base.config_yaml.replace("max_total_margin_pct: 1.0", "max_total_margin_pct: 0.50")
    main_rows, sub_rows = _portfolio_fixture_rows([("BTC", 0.0), ("ETH", 1.2)])
    return Scenario(
        name="allocation_dual",
        description="Dual-symbol synchronous path to force portfolio sizing effects.",
        config_yaml=cfg,
        anchors=_anchors(**{"trade.max_total_margin_pct": 0.50}),
        main_rows=main_rows,
        sub_rows=sub_rows,
    )


def _scenario_tp_isolation() -> Scenario:
    cfg = textwrap.dedent(
        """
        global:
          trade:
            allocation_pct: 0.10
            sl_atr_mult: 99.0
            tp_atr_mult: 4.0
            leverage: 1.0
            max_open_positions: 20
            max_total_margin_pct: 1.0
            min_notional_usd: 1.0
            min_atr_pct: 0.0
            bump_to_min_notional: false
            enable_dynamic_sizing: false
            enable_dynamic_leverage: false
            enable_pyramiding: false
            enable_partial_tp: false
            enable_ssf_filter: false
            enable_reef_filter: false
            enable_breakeven_stop: false
            enable_rsi_overextension_exit: false
            enable_vol_buffered_trailing: false
            tsme_require_adx_slope_negative: false
            reverse_entry_signal: false
            block_exits_on_extreme_dev: false
            entry_min_confidence: low
            add_min_confidence: low
            trailing_start_atr: 99.0
            trailing_distance_atr: 99.0
            smart_exit_adx_exhaustion_lt: 0.0
            smart_exit_adx_exhaustion_lt_low_conf: 0.0
            slippage_bps: 0.0
          indicators:
            ema_fast_window: 2
            ema_slow_window: 3
            ema_macro_window: 3
            adx_window: 2
            atr_window: 2
            rsi_window: 2
            bb_window: 2
            bb_width_avg_window: 2
            vol_sma_window: 2
            vol_trend_window: 2
            stoch_rsi_window: 2
            stoch_rsi_smooth1: 1
            stoch_rsi_smooth2: 1
          filters:
            enable_ranging_filter: false
            enable_anomaly_filter: false
            enable_extension_filter: false
            require_adx_rising: false
            require_volume_confirmation: false
            use_stoch_rsi_filter: false
            require_btc_alignment: false
            require_macro_alignment: false
          thresholds:
            entry:
              min_adx: -1.0
              max_dist_ema_fast: 1.0
              macd_hist_entry_mode: none
              ave_enabled: false
              enable_pullback_entries: false
              enable_slow_drift_entries: false
            stoch_rsi:
              block_long_if_k_gt: 1.0
              block_short_if_k_lt: 0.0
        """
    ).strip() + "\n"

    closes = [100.0, 100.7, 101.3, 101.9, 102.5, 103.1, 103.7, 104.3]
    rel = [
        0.05,
        0.10,
        0.20,
        0.35,
        0.55,
        0.80,
        1.10,
        1.45,
        1.75,
        2.05,
        2.20,
        2.35,
        2.45,
        2.50,
        2.35,
        2.10,
        1.85,
        1.60,
        1.35,
        1.20,
    ]
    main_rows, sub_rows = _build_symbol_rows(
        symbol="BTC",
        closes=closes,
        main_pad=0.12,
        rel_path=rel,
        rel_high_pad=0.03,
        rel_low_pad=0.03,
    )
    anchors = _anchors(
        **{
            "trade.sl_atr_mult": 99.0,
            "trade.tp_atr_mult": 3.0,
            "trade.trailing_start_atr": 99.0,
            "trade.trailing_distance_atr": 99.0,
        }
    )
    return Scenario(
        name="tp_isolation",
        description="High-upside spikes with trailing/SL effectively disabled to isolate TP multipliers.",
        config_yaml=cfg,
        anchors=anchors,
        main_rows=main_rows,
        sub_rows=sub_rows,
    )


def _scenario_sl_isolation() -> Scenario:
    cfg = textwrap.dedent(
        """
        global:
          trade:
            allocation_pct: 0.10
            sl_atr_mult: 2.0
            tp_atr_mult: 99.0
            leverage: 1.0
            max_open_positions: 20
            max_total_margin_pct: 1.0
            min_notional_usd: 1.0
            min_atr_pct: 0.0
            bump_to_min_notional: false
            enable_dynamic_sizing: false
            enable_dynamic_leverage: false
            enable_pyramiding: false
            enable_partial_tp: false
            enable_ssf_filter: false
            enable_reef_filter: false
            enable_breakeven_stop: false
            enable_rsi_overextension_exit: false
            enable_vol_buffered_trailing: false
            tsme_require_adx_slope_negative: false
            reverse_entry_signal: false
            block_exits_on_extreme_dev: false
            entry_min_confidence: low
            add_min_confidence: low
            trailing_start_atr: 99.0
            trailing_distance_atr: 99.0
            smart_exit_adx_exhaustion_lt: 0.0
            smart_exit_adx_exhaustion_lt_low_conf: 0.0
            slippage_bps: 0.0
          indicators:
            ema_fast_window: 2
            ema_slow_window: 3
            ema_macro_window: 3
            adx_window: 2
            atr_window: 2
            rsi_window: 2
            bb_window: 2
            bb_width_avg_window: 2
            vol_sma_window: 2
            vol_trend_window: 2
            stoch_rsi_window: 2
            stoch_rsi_smooth1: 1
            stoch_rsi_smooth2: 1
          filters:
            enable_ranging_filter: false
            enable_anomaly_filter: false
            enable_extension_filter: false
            require_adx_rising: false
            require_volume_confirmation: false
            use_stoch_rsi_filter: false
            require_btc_alignment: false
            require_macro_alignment: false
          thresholds:
            entry:
              min_adx: -1.0
              max_dist_ema_fast: 1.0
              macd_hist_entry_mode: none
              ave_enabled: false
              enable_pullback_entries: false
              enable_slow_drift_entries: false
            stoch_rsi:
              block_long_if_k_gt: 1.0
              block_short_if_k_lt: 0.0
        """
    ).strip() + "\n"

    closes = [100.0, 100.9, 101.6, 101.8, 101.4, 101.0, 100.7, 100.4, 100.2]
    rel = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.15,
        0.00,
        -0.20,
        -0.45,
        -0.75,
        -1.05,
        -1.35,
        -1.65,
        -1.90,
        -1.70,
        -1.40,
        -1.10,
        -0.80,
        -0.55,
        -0.35,
        -0.20,
    ]
    main_rows, sub_rows = _build_symbol_rows(
        symbol="BTC",
        closes=closes,
        main_pad=0.15,
        rel_path=rel,
        rel_high_pad=0.03,
        rel_low_pad=0.03,
    )
    anchors = _anchors(
        **{
            "trade.sl_atr_mult": 1.5,
            "trade.tp_atr_mult": 99.0,
            "trade.trailing_start_atr": 99.0,
            "trade.trailing_distance_atr": 99.0,
        }
    )
    return Scenario(
        name="sl_isolation",
        description="Deep downside spikes with TP/trailing effectively disabled to isolate SL multipliers.",
        config_yaml=cfg,
        anchors=anchors,
        main_rows=main_rows,
        sub_rows=sub_rows,
    )


def _scenario_macro_alignment_gate() -> Scenario:
    base = _scenario_tp_isolation()
    anchors = dict(base.anchors)
    anchors["filters.require_macro_alignment"] = 0.0
    return Scenario(
        name="macro_alignment_gate",
        description=(
            "Dedicated false/true macro-alignment gate toggle. "
            "Reuses tp_isolation fixture (ema_slow_window == ema_macro_window) so enabling "
            "the gate consistently suppresses directional alignment on both CPU and GPU."
        ),
        config_yaml=base.config_yaml,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_partial_tp_toggle() -> Scenario:
    base = _scenario_tp_isolation()
    cfg = base.config_yaml
    cfg = _replace_once(
        cfg,
        "enable_partial_tp: false",
        "enable_partial_tp: false\n"
        "    tp_partial_pct: 0.50\n"
        "    tp_partial_atr_mult: 1.0\n"
        "    tp_partial_min_notional_usd: 1.0",
    )
    anchors = _anchors(
        **{
            "trade.sl_atr_mult": 99.0,
            "trade.tp_atr_mult": 4.0,
            "trade.trailing_start_atr": 99.0,
            "trade.trailing_distance_atr": 99.0,
            "trade.enable_partial_tp": 0.0,
            "trade.tp_partial_pct": 0.50,
            "trade.tp_partial_atr_mult": 1.0,
        }
    )
    return Scenario(
        name="partial_tp_toggle",
        description=(
            "Partial-TP toggle fixture with full TP effectively disabled. "
            "When enabled, partial reduce can trigger; when disabled, the same path cannot emit a partial action."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_partial_tp_pct() -> Scenario:
    base = _scenario_partial_tp_toggle()
    cfg = _replace_once(base.config_yaml, "enable_partial_tp: false", "enable_partial_tp: true")
    anchors = dict(base.anchors)
    anchors["trade.enable_partial_tp"] = 1.0
    anchors["trade.tp_partial_pct"] = 0.25
    return Scenario(
        name="partial_tp_pct",
        description=(
            "Partial-TP percentage fixture with partial TP forced on and dedicated partial trigger enabled. "
            "Only tp_partial_pct changes closed fraction at the first TP event."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_partial_tp_atr_mult() -> Scenario:
    base = _scenario_partial_tp_pct()
    anchors = dict(base.anchors)
    anchors["trade.tp_partial_atr_mult"] = 0.5
    return Scenario(
        name="partial_tp_atr_mult",
        description=(
            "Partial-TP ATR-multiplier fixture with partial TP fixed on and constant partial fraction. "
            "Only tp_partial_atr_mult changes when partial reduction is triggered."
        ),
        config_yaml=base.config_yaml,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_margin_cap_isolation() -> Scenario:
    base = _scenario_allocation_dual()
    cfg = _replace_once(base.config_yaml, "allocation_pct: 0.10", "allocation_pct: 0.35")
    cfg = _replace_once(cfg, "max_total_margin_pct: 0.50", "max_total_margin_pct: 0.20")
    main_rows, sub_rows = _portfolio_fixture_rows([("BTC", 0.0), ("ETH", 1.2), ("SOL", 2.4)])
    anchors = _anchors(
        **{
            "trade.allocation_pct": 0.35,
            "trade.max_total_margin_pct": 0.20,
            "trade.max_open_positions": 20.0,
        }
    )
    return Scenario(
        name="margin_cap_isolation",
        description=(
            "Three-symbol profitable fixture with aggressive per-position allocation. "
            "Only max_total_margin_pct controls how much portfolio headroom is available."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=main_rows,
        sub_rows=sub_rows,
    )


def _scenario_max_positions_isolation() -> Scenario:
    base = _scenario_margin_cap_isolation()
    cfg = _replace_once(base.config_yaml, "allocation_pct: 0.35", "allocation_pct: 0.12")
    cfg = _replace_once(cfg, "max_total_margin_pct: 0.20", "max_total_margin_pct: 1.0")
    cfg = _replace_once(cfg, "max_open_positions: 20", "max_open_positions: 1")
    anchors = _anchors(
        **{
            "trade.allocation_pct": 0.12,
            "trade.max_total_margin_pct": 1.0,
            "trade.max_open_positions": 1.0,
        }
    )
    return Scenario(
        name="max_positions_isolation",
        description=(
            "Three-symbol synchronous-entry fixture with ample margin headroom. "
            "Only max_open_positions throttles how many concurrent entries can be opened."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_leverage_isolation() -> Scenario:
    base = _scenario_tp_isolation()
    cfg = _replace_once(base.config_yaml, "tp_atr_mult: 4.0", "tp_atr_mult: 3.0")
    anchors = _anchors(
        **{
            "trade.sl_atr_mult": 99.0,
            "trade.tp_atr_mult": 3.0,
            "trade.trailing_start_atr": 99.0,
            "trade.trailing_distance_atr": 99.0,
            "trade.leverage": 1.0,
        }
    )
    return Scenario(
        name="leverage_isolation",
        description=(
            "Single-symbol profitable TP fixture with dynamic leverage disabled. "
            "Only leverage rescales position notional and realised PnL."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_smart_exit_high_conf() -> Scenario:
    base = _scenario_trailing()
    cfg = _replace_once(base.config_yaml, "sl_atr_mult: 1.5", "sl_atr_mult: 99.0")
    cfg = _replace_once(cfg, "tp_atr_mult: 3.0", "tp_atr_mult: 99.0")
    cfg = _replace_once(cfg, "trailing_start_atr: 1.5", "trailing_start_atr: 99.0")
    cfg = _replace_once(cfg, "trailing_distance_atr: 0.5", "trailing_distance_atr: 99.0")
    anchors = _anchors(
        **{
            "trade.sl_atr_mult": 99.0,
            "trade.tp_atr_mult": 99.0,
            "trade.trailing_start_atr": 99.0,
            "trade.trailing_distance_atr": 99.0,
            "trade.smart_exit_adx_exhaustion_lt": 0.0,
            "trade.smart_exit_adx_exhaustion_lt_low_conf": 0.0,
        }
    )
    return Scenario(
        name="smart_exit_high_conf",
        description=(
            "High-confidence smart-exit fixture with SL/TP/trailing neutralised. "
            "Only smart_exit_adx_exhaustion_lt controls ADX-based early exit pressure."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _scenario_smart_exit_low_conf() -> Scenario:
    base = _scenario_smart_exit_high_conf()
    cfg = _replace_once(base.config_yaml, "min_adx: -1.0", "min_adx: 200.0")
    cfg = _replace_once(
        cfg,
        "enable_slow_drift_entries: false",
        "enable_slow_drift_entries: true\n"
        "      slow_drift_slope_window: 2\n"
        "      slow_drift_min_slope_pct: 0.0\n"
        "      slow_drift_min_adx: -1.0\n"
        "      slow_drift_rsi_long_min: 0.0\n"
        "      slow_drift_rsi_short_max: 100.0\n"
        "      slow_drift_require_macd_sign: false",
    )
    anchors = dict(base.anchors)
    anchors["trade.smart_exit_adx_exhaustion_lt"] = 0.0
    anchors["trade.smart_exit_adx_exhaustion_lt_low_conf"] = 0.0
    return Scenario(
        name="smart_exit_low_conf",
        description=(
            "Low-confidence smart-exit fixture that blocks standard entries and forces slow-drift entries. "
            "Only smart_exit_adx_exhaustion_lt_low_conf should influence ADX-exhaustion exits."
        ),
        config_yaml=cfg,
        anchors=anchors,
        main_rows=base.main_rows,
        sub_rows=base.sub_rows,
    )


def _canonical_overrides(overrides: object) -> Dict[str, float]:
    if isinstance(overrides, dict):
        items = overrides.items()
    else:
        items = overrides  # GPU list[[k, v], ...]
    return {str(k): float(v) for k, v in items}


def _run_sweep(
    repo_root: Path,
    bin_path: Path,
    config_path: Path,
    main_db: Path,
    sub_db: Path,
    spec_path: Path,
    out_path: Path,
    gpu: bool,
) -> List[Dict[str, object]]:
    cmd = [
        str(bin_path),
        "sweep",
        "--sweep-spec",
        str(spec_path),
        "--config",
        str(config_path),
        "--candles-db",
        str(main_db),
        "--interval",
        "1h",
        "--entry-candles-db",
        str(sub_db),
        "--entry-interval",
        "3m",
        "--exit-candles-db",
        str(sub_db),
        "--exit-interval",
        "3m",
        "--output",
        str(out_path),
    ]
    env = os.environ.copy()
    if gpu:
        cmd.insert(2, "--gpu")
        env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        side = "GPU" if gpu else "CPU"
        raise RuntimeError(
            f"{side} sweep failed (rc={proc.returncode}).\n"
            f"STDERR tail:\n{proc.stderr[-1200:]}"
        )

    rows: List[Dict[str, object]] = []
    for line in out_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_spec_yaml(anchors: Dict[str, float], axis: str, axis_values: Sequence[float]) -> str:
    lines: List[str] = ["axes:"]
    for path in AXIS_ORDER:
        anchor_value = float(anchors.get(path, AXIS_VALUES[path][0]))
        vals = axis_values if path == axis else [anchor_value]
        lines.append(f"  - path: {path}")
        lines.append(f"    values: {list(vals)}")
    lines.append("lookback: 0")
    lines.append("initial_balance: 1000")
    lines.append("")
    return "\n".join(lines)


def _summarise_rows(rows: List[Dict[str, object]], axis: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        ov = _canonical_overrides(row["overrides"])
        key = f"{ov[axis]:.10g}"
        out[key] = {
            "total_pnl": float(row["total_pnl"]),
            "total_trades": float(row["total_trades"]),
            "win_rate": float(row["win_rate"]),
            "max_drawdown_pct": float(row["max_drawdown_pct"]),
        }
    return out


def _triggered(summary: Dict[str, Dict[str, float]]) -> bool:
    sig = {(v["total_pnl"], v["total_trades"]) for v in summary.values()}
    return len(sig) > 1


def _sign(delta: float, tol: float = 1e-9) -> int:
    if delta > tol:
        return 1
    if delta < -tol:
        return -1
    return 0


def _sign_label(value: int) -> str:
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "flat"


def _direction_agreement(
    cpu_summary: Dict[str, Dict[str, float]],
    gpu_summary: Dict[str, Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    values = sorted(set(cpu_summary) & set(gpu_summary), key=lambda v: float(v))
    comparisons = 0
    matches = 0
    for prev, cur in zip(values, values[1:]):
        cpu_delta = cpu_summary[cur][metric] - cpu_summary[prev][metric]
        gpu_delta = gpu_summary[cur][metric] - gpu_summary[prev][metric]
        if _sign(cpu_delta) == _sign(gpu_delta):
            matches += 1
        comparisons += 1
    mismatches = comparisons - matches
    ratio = 1.0 if comparisons == 0 else matches / comparisons
    return {
        "matches": float(matches),
        "mismatches": float(mismatches),
        "comparisons": float(comparisons),
        "ratio": float(ratio),
    }


def _agreement_metrics(
    cpu_summary: Dict[str, Dict[str, float]],
    gpu_summary: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    return {
        "pnl_direction": _direction_agreement(cpu_summary, gpu_summary, "total_pnl"),
        "trades_direction": _direction_agreement(cpu_summary, gpu_summary, "total_trades"),
    }


def _value_key(value: float) -> str:
    return f"{float(value):.10g}"


def _relation_holds(delta: float, relation: str, tol: float = 1e-9) -> bool:
    if relation == "change":
        return _sign(delta, tol) != 0
    if relation == "increase":
        return delta > tol
    if relation == "decrease":
        return delta < -tol
    if relation == "nondecrease":
        return delta >= -tol
    if relation == "nonincrease":
        return delta <= tol
    raise ValueError(f"Unsupported relation: {relation}")


def _evaluate_expectation(
    axis: str,
    axis_values: Sequence[float],
    expectation: ExpectationSpec,
    cpu_summary: Dict[str, Dict[str, float]],
    gpu_summary: Dict[str, Dict[str, float]],
    cpu_triggered: bool,
    gpu_triggered: bool,
) -> Dict[str, object]:
    first_key = _value_key(axis_values[0])
    last_key = _value_key(axis_values[-1])
    if first_key not in cpu_summary or last_key not in cpu_summary:
        raise RuntimeError(f"CPU summary missing expected axis keys for {axis}: {first_key}, {last_key}")
    if first_key not in gpu_summary or last_key not in gpu_summary:
        raise RuntimeError(f"GPU summary missing expected axis keys for {axis}: {first_key}, {last_key}")

    metric = expectation.metric
    cpu_delta = float(cpu_summary[last_key][metric] - cpu_summary[first_key][metric])
    gpu_delta = float(gpu_summary[last_key][metric] - gpu_summary[first_key][metric])
    cpu_relation_ok = _relation_holds(cpu_delta, expectation.relation)
    gpu_relation_ok = _relation_holds(gpu_delta, expectation.relation)
    cpu_delta_sign = _sign(cpu_delta)
    gpu_delta_sign = _sign(gpu_delta)
    direction_match = cpu_delta_sign == gpu_delta_sign
    triggered_ok = (not expectation.require_triggered) or (cpu_triggered and gpu_triggered)
    direction_ok = (not expectation.require_direction_match) or direction_match
    passed = cpu_relation_ok and gpu_relation_ok and direction_ok and triggered_ok

    return {
        "metric": metric,
        "relation": expectation.relation,
        "note": expectation.note,
        "first_value_key": first_key,
        "last_value_key": last_key,
        "cpu_delta_last_minus_first": cpu_delta,
        "gpu_delta_last_minus_first": gpu_delta,
        "cpu_delta_sign": cpu_delta_sign,
        "gpu_delta_sign": gpu_delta_sign,
        "cpu_relation_ok": cpu_relation_ok,
        "gpu_relation_ok": gpu_relation_ok,
        "direction_match": direction_match,
        "direction_check_required": expectation.require_direction_match,
        "triggered_check_required": expectation.require_triggered,
        "triggered_ok": triggered_ok,
        "passed": passed,
    }


def _macro_alignment_validation(
    cpu_summary: Dict[str, Dict[str, float]],
    gpu_summary: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    false_key = _value_key(0.0)
    true_key = _value_key(1.0)

    cpu_trade_delta = float(cpu_summary[true_key]["total_trades"] - cpu_summary[false_key]["total_trades"])
    gpu_trade_delta = float(gpu_summary[true_key]["total_trades"] - gpu_summary[false_key]["total_trades"])
    cpu_pnl_delta = float(cpu_summary[true_key]["total_pnl"] - cpu_summary[false_key]["total_pnl"])
    gpu_pnl_delta = float(gpu_summary[true_key]["total_pnl"] - gpu_summary[false_key]["total_pnl"])

    trade_direction_match = _sign(cpu_trade_delta) == _sign(gpu_trade_delta)
    trade_changed_both = _sign(cpu_trade_delta) != 0 and _sign(gpu_trade_delta) != 0
    consistent_trade_direction = trade_direction_match and trade_changed_both

    return {
        "false_value_key": false_key,
        "true_value_key": true_key,
        "cpu_trade_delta_true_minus_false": cpu_trade_delta,
        "gpu_trade_delta_true_minus_false": gpu_trade_delta,
        "cpu_pnl_delta_true_minus_false": cpu_pnl_delta,
        "gpu_pnl_delta_true_minus_false": gpu_pnl_delta,
        "trade_direction_match": trade_direction_match,
        "trade_changed_both_engines": trade_changed_both,
        "consistent_trade_direction": consistent_trade_direction,
    }


def _format_axis_value(axis: str, value_key: str) -> str:
    if axis == "filters.require_macro_alignment":
        val = float(value_key)
        if val == 0.0:
            return "false"
        if val == 1.0:
            return "true"
    return value_key


def _experiment_specs_for_axes(axes: Sequence[str]) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    for axis in axes:
        scenario_name = AXIS_SCENARIOS.get(axis)
        expectation = AXIS_EXPECTATIONS.get(axis)
        if scenario_name is None:
            raise RuntimeError(f"Axis {axis} has no scenario mapping in AXIS_SCENARIOS.")
        if expectation is None:
            raise RuntimeError(f"Axis {axis} has no expectation mapping in AXIS_EXPECTATIONS.")
        specs.append(ExperimentSpec(axis=axis, scenario_name=scenario_name, expectation=expectation))
    return specs


def _evaluate_experiment(
    axis: str,
    cpu_triggered: bool,
    gpu_triggered: bool,
    agreement: Dict[str, Dict[str, float]],
    expectation_result: Dict[str, object],
    macro_validation: Optional[Dict[str, object]],
    expectation: ExpectationSpec,
) -> Tuple[str, bool, List[str]]:
    notes: List[str] = [f"Expectation: {expectation.note}"]
    passed = bool(expectation_result.get("passed", False))

    notes.append(
        "Expectation check: "
        f"{expectation_result['metric']} {expectation_result['relation']} from "
        f"{expectation_result['first_value_key']} -> {expectation_result['last_value_key']}; "
        f"CPU delta={float(expectation_result['cpu_delta_last_minus_first']):.6f}, "
        f"GPU delta={float(expectation_result['gpu_delta_last_minus_first']):.6f}."
    )

    if not bool(expectation_result["cpu_relation_ok"]):
        notes.append("CPU first->last relation check failed.")
    if not bool(expectation_result["gpu_relation_ok"]):
        notes.append("GPU first->last relation check failed.")
    if expectation.require_direction_match and not bool(expectation_result["direction_match"]):
        notes.append(
            "Direction mismatch: CPU/GPU first->last delta directions do not match "
            f"(CPU={_sign_label(int(expectation_result['cpu_delta_sign']))}, "
            f"GPU={_sign_label(int(expectation_result['gpu_delta_sign']))})."
        )
    if expectation.require_triggered and not bool(expectation_result["triggered_ok"]):
        if not cpu_triggered:
            notes.append("CPU did not react to axis changes.")
        if not gpu_triggered:
            notes.append("GPU did not react to axis changes.")
    elif cpu_triggered and gpu_triggered:
        notes.append("CPU and GPU both reacted to axis changes.")

    pnl_ratio = float(agreement["pnl_direction"]["ratio"])
    trades_ratio = float(agreement["trades_direction"]["ratio"])
    if pnl_ratio < 0.5:
        notes.append(
            f"Low CPU/GPU PnL-direction agreement ({pnl_ratio * 100.0:.1f}%, "
            f"mismatches={int(agreement['pnl_direction']['mismatches'])})."
        )
    if trades_ratio < 0.5:
        notes.append(
            f"Low CPU/GPU trade-direction agreement ({trades_ratio * 100.0:.1f}%, "
            f"mismatches={int(agreement['trades_direction']['mismatches'])})."
        )

    if axis == "filters.require_macro_alignment":
        if macro_validation is None:
            passed = False
            notes.append("Macro gate validation output missing.")
        elif not bool(macro_validation["consistent_trade_direction"]):
            passed = False
            notes.append("Macro gate false->true did not produce a consistent trade-direction change.")
        else:
            notes.append("Macro gate false->true trade-direction change is consistent across CPU and GPU.")

    status = "pass" if passed else "fail"
    return status, passed, notes


def _build_coverage_rows(
    experiments: Sequence[Dict[str, object]],
    selected_axes: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    selected_set = set(selected_axes)
    exp_by_axis = {str(exp["axis"]): exp for exp in experiments}

    coverage: List[Dict[str, object]] = []
    pass_count = 0
    fail_count = 0
    pending_count = 0

    for axis in AXIS_ORDER:
        exp = exp_by_axis.get(axis)
        if exp is not None:
            passed = bool(exp["pass"])
            status = str(exp["status"])
            notes = [str(note) for note in exp["notes"]]
            if passed:
                pass_count += 1
            else:
                fail_count += 1
            coverage.append(
                {
                    "axis": axis,
                    "scenario": exp["scenario"],
                    "status": status,
                    "pass": passed,
                    "notes": notes,
                }
            )
            continue

        pending_count += 1
        if axis not in selected_set:
            notes = ["Axis not selected for this run."]
        elif axis not in AXIS_SCENARIOS:
            notes = ["No synthetic scenario mapping configured for this axis yet."]
        else:
            notes = ["Axis selected but no experiment result was produced."]
        coverage.append(
            {
                "axis": axis,
                "scenario": AXIS_SCENARIOS.get(axis),
                "status": "pending",
                "pass": None,
                "notes": notes,
            }
        )

    summary = {
        "total_axes": len(AXIS_ORDER),
        "selected_axes": len(selected_axes),
        "executed_axes": len(experiments),
        "pass_axes": pass_count,
        "fail_axes": fail_count,
        "pending_axes": pending_count,
    }
    return coverage, summary


def run_experiments(repo_root: Path, bin_path: Path, selected_axes: Sequence[str]) -> Dict[str, object]:
    scenarios = {
        "trailing_single": _scenario_trailing(),
        "allocation_dual": _scenario_allocation_dual(),
        "tp_isolation": _scenario_tp_isolation(),
        "sl_isolation": _scenario_sl_isolation(),
        "macro_alignment_gate": _scenario_macro_alignment_gate(),
        "partial_tp_toggle": _scenario_partial_tp_toggle(),
        "partial_tp_pct": _scenario_partial_tp_pct(),
        "partial_tp_atr_mult": _scenario_partial_tp_atr_mult(),
        "margin_cap_isolation": _scenario_margin_cap_isolation(),
        "max_positions_isolation": _scenario_max_positions_isolation(),
        "leverage_isolation": _scenario_leverage_isolation(),
        "smart_exit_high_conf": _scenario_smart_exit_high_conf(),
        "smart_exit_low_conf": _scenario_smart_exit_low_conf(),
    }
    specs = _experiment_specs_for_axes(selected_axes)

    with tempfile.TemporaryDirectory(prefix="gpu_signal_capture_") as tmp:
        td = Path(tmp)
        out: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "axis_order": list(AXIS_ORDER),
            "selected_axes": list(selected_axes),
            "experiments": [],
            "per_axis_agreement": {},
        }
        for spec in specs:
            scenario = scenarios[spec.scenario_name]
            main_db = td / f"{scenario.name}_{spec.axis}_main.db"
            sub_db = td / f"{scenario.name}_{spec.axis}_sub.db"
            cfg_path = td / f"{scenario.name}_{spec.axis}.yaml"
            sweep_path = td / f"{scenario.name}_{spec.axis}_sweep.yaml"
            cpu_out = td / f"{scenario.name}_{spec.axis}_cpu.jsonl"
            gpu_out = td / f"{scenario.name}_{spec.axis}_gpu.jsonl"

            _create_candle_db(main_db, scenario.main_rows)
            _create_candle_db(sub_db, scenario.sub_rows)
            cfg_path.write_text(scenario.config_yaml)

            # Baseline: all axes fixed to scenario anchors.
            baseline_anchor = float(scenario.anchors.get(spec.axis, AXIS_VALUES[spec.axis][0]))
            baseline_yaml = _build_spec_yaml(
                anchors=scenario.anchors,
                axis=spec.axis,
                axis_values=[baseline_anchor],
            )
            sweep_path.write_text(baseline_yaml)
            cpu_base = _run_sweep(repo_root, bin_path, cfg_path, main_db, sub_db, sweep_path, cpu_out, gpu=False)[
                0
            ]
            gpu_base = _run_sweep(repo_root, bin_path, cfg_path, main_db, sub_db, sweep_path, gpu_out, gpu=True)[
                0
            ]

            # Axis experiment: vary only target axis.
            exp_yaml = _build_spec_yaml(
                anchors=scenario.anchors,
                axis=spec.axis,
                axis_values=AXIS_VALUES[spec.axis],
            )
            sweep_path.write_text(exp_yaml)
            cpu_rows = _run_sweep(repo_root, bin_path, cfg_path, main_db, sub_db, sweep_path, cpu_out, gpu=False)
            gpu_rows = _run_sweep(repo_root, bin_path, cfg_path, main_db, sub_db, sweep_path, gpu_out, gpu=True)

            cpu_summary = _summarise_rows(cpu_rows, spec.axis)
            gpu_summary = _summarise_rows(gpu_rows, spec.axis)
            agreement = _agreement_metrics(cpu_summary, gpu_summary)
            cpu_triggered = _triggered(cpu_summary)
            gpu_triggered = _triggered(gpu_summary)
            expectation_result = _evaluate_expectation(
                axis=spec.axis,
                axis_values=AXIS_VALUES[spec.axis],
                expectation=spec.expectation,
                cpu_summary=cpu_summary,
                gpu_summary=gpu_summary,
                cpu_triggered=cpu_triggered,
                gpu_triggered=gpu_triggered,
            )

            exp_out: Dict[str, object] = {
                "axis": spec.axis,
                "scenario": scenario.name,
                "description": scenario.description,
                "anchors": scenario.anchors,
                "axis_values": [float(v) for v in AXIS_VALUES[spec.axis]],
                "baseline": {
                    "cpu_total_pnl": float(cpu_base["total_pnl"]),
                    "cpu_total_trades": int(cpu_base["total_trades"]),
                    "gpu_total_pnl": float(gpu_base["total_pnl"]),
                    "gpu_total_trades": int(gpu_base["total_trades"]),
                },
                "cpu": cpu_summary,
                "gpu": gpu_summary,
                "cpu_triggered": cpu_triggered,
                "gpu_triggered": gpu_triggered,
                "agreement": agreement,
                "expectation": expectation_result,
            }
            macro_validation: Optional[Dict[str, object]] = None
            if spec.axis == "filters.require_macro_alignment":
                macro_validation = _macro_alignment_validation(cpu_summary, gpu_summary)
                exp_out["macro_alignment_validation"] = macro_validation

            status, passed, notes = _evaluate_experiment(
                axis=spec.axis,
                cpu_triggered=cpu_triggered,
                gpu_triggered=gpu_triggered,
                agreement=agreement,
                expectation_result=expectation_result,
                macro_validation=macro_validation,
                expectation=spec.expectation,
            )
            exp_out["status"] = status
            exp_out["pass"] = passed
            exp_out["notes"] = notes

            out["per_axis_agreement"][spec.axis] = {
                "scenario": scenario.name,
                "pnl_direction": agreement["pnl_direction"],
                "trades_direction": agreement["trades_direction"],
                "pnl_direction_mismatches": int(agreement["pnl_direction"]["mismatches"]),
                "trades_direction_mismatches": int(agreement["trades_direction"]["mismatches"]),
                "status": status,
                "pass": passed,
            }

            out["experiments"].append(exp_out)

        coverage, summary = _build_coverage_rows(out["experiments"], selected_axes)
        out["coverage"] = coverage
        out["summary"] = summary
        return out


def _format_markdown(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Synthetic Signal Capture Progress")
    lines.append("")
    lines.append("Deterministic synthetic fixtures used to verify whether CPU and GPU both react to each target parameter.")
    lines.append("")
    summary = result["summary"]
    lines.append(f"- Schema version: {result.get('schema_version', SCHEMA_VERSION)}")
    lines.append(f"- Total target axes: {summary['total_axes']}")
    lines.append(f"- Selected in this run: {summary['selected_axes']}")
    lines.append(f"- Executed: {summary['executed_axes']}")
    lines.append(f"- Pass: {summary['pass_axes']}")
    lines.append(f"- Fail: {summary['fail_axes']}")
    lines.append(f"- Pending: {summary['pending_axes']}")
    lines.append("")
    lines.append("## Coverage Matrix")
    lines.append("")
    lines.append("| Axis | Scenario | Status | Pass | Notes |")
    lines.append("|---|---|---|---:|---|")
    for row in result["coverage"]:
        pass_txt = "yes" if row["pass"] is True else "no" if row["pass"] is False else "-"
        notes_txt = "<br>".join(row["notes"])
        scenario = row["scenario"] if row["scenario"] is not None else "-"
        lines.append(
            f"| `{row['axis']}` | `{scenario}` | `{row['status']}` | {pass_txt} | {notes_txt} |"
        )
    lines.append("")
    lines.append("## Executed Experiment Details")
    lines.append("")
    for exp in result["experiments"]:
        lines.append(f"### `{exp['axis']}` on `{exp['scenario']}`")
        lines.append("")
        lines.append(exp["description"])
        lines.append("")
        lines.append(f"- Status: `{exp['status']}`")
        lines.append(f"- Pass: {'yes' if exp['pass'] else 'no'}")
        lines.append(f"- Notes: {'; '.join(exp['notes'])}")
        expectation = exp["expectation"]
        lines.append(
            "- Expectation first->last delta: "
            f"{expectation['metric']} {expectation['relation']} "
            f"(CPU {expectation['cpu_delta_last_minus_first']:.6f}, "
            f"GPU {expectation['gpu_delta_last_minus_first']:.6f}, "
            f"pass={'yes' if expectation['passed'] else 'no'})"
        )
        base = exp["baseline"]
        lines.append(
            f"- Baseline: CPU trades={base['cpu_total_trades']}, pnl={base['cpu_total_pnl']:.4f}; "
            f"GPU trades={base['gpu_total_trades']}, pnl={base['gpu_total_pnl']:.4f}"
        )
        lines.append(f"- CPU triggered: {'yes' if exp['cpu_triggered'] else 'no'}")
        lines.append(f"- GPU triggered: {'yes' if exp['gpu_triggered'] else 'no'}")
        pnl_agree = exp["agreement"]["pnl_direction"]
        trade_agree = exp["agreement"]["trades_direction"]
        lines.append(
            f"- CPU/GPU PnL direction agreement across axis steps: "
            f"{int(pnl_agree['matches'])}/{int(pnl_agree['comparisons'])} "
            f"({pnl_agree['ratio'] * 100.0:.1f}%, mismatches={int(pnl_agree['mismatches'])})"
        )
        lines.append(
            f"- CPU/GPU trade-count direction agreement across axis steps: "
            f"{int(trade_agree['matches'])}/{int(trade_agree['comparisons'])} "
            f"({trade_agree['ratio'] * 100.0:.1f}%, mismatches={int(trade_agree['mismatches'])})"
        )
        if "macro_alignment_validation" in exp:
            val = exp["macro_alignment_validation"]
            lines.append(
                "- Macro gate false->true trade delta (CPU/GPU): "
                f"{val['cpu_trade_delta_true_minus_false']:.1f}/{val['gpu_trade_delta_true_minus_false']:.1f}"
            )
            lines.append(
                f"- Macro gate consistent trade-direction change: "
                f"{'yes' if val['consistent_trade_direction'] else 'no'}"
            )
        lines.append("")
        lines.append("| Value | CPU PnL | CPU Trades | GPU PnL | GPU Trades |")
        lines.append("|---:|---:|---:|---:|---:|")
        for val in sorted(exp["cpu"].keys(), key=lambda v: float(v)):
            cpu = exp["cpu"][val]
            gpu = exp["gpu"][val]
            lines.append(
                f"| {_format_axis_value(exp['axis'], val)} | {cpu['total_pnl']:.4f} | {int(cpu['total_trades'])} | "
                f"{gpu['total_pnl']:.4f} | {int(gpu['total_trades'])} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic CPU/GPU signal-capture experiments.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repository root (default: current directory).",
    )
    parser.add_argument(
        "--bin",
        default="backtester/target/release/mei-backtester",
        help="Path to prebuilt mei-backtester binary (default: backtester/target/release/mei-backtester).",
    )
    parser.add_argument(
        "--json-out",
        default="/tmp/v71_synthetic_signal_capture.json",
        help="Output JSON path (default: /tmp/v71_synthetic_signal_capture.json).",
    )
    parser.add_argument(
        "--md-out",
        default="/tmp/v71_synthetic_signal_capture.md",
        help="Output Markdown path (default: /tmp/v71_synthetic_signal_capture.md).",
    )
    parser.add_argument(
        "--axes",
        default="",
        help=(
            "Comma-separated axis list to run (default: all axes in AXIS_ORDER). "
            "Example: trade.sl_atr_mult,filters.require_macro_alignment"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bin_path = (repo_root / args.bin).resolve()
    if not bin_path.exists():
        raise SystemExit(
            f"Binary not found at {bin_path}. Build first: "
            f"`cargo build --release --manifest-path backtester/Cargo.toml --features gpu`."
        )

    selected_axes = _parse_axes(args.axes)
    result = run_experiments(repo_root=repo_root, bin_path=bin_path, selected_axes=selected_axes)
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.write_text(json.dumps(result, indent=2))
    md_out.write_text(_format_markdown(result))

    print(f"Wrote JSON: {json_out}")
    print(f"Wrote Markdown: {md_out}")
    print(
        "Summary: "
        f"executed={result['summary']['executed_axes']} "
        f"pass={result['summary']['pass_axes']} "
        f"fail={result['summary']['fail_axes']} "
        f"pending={result['summary']['pending_axes']}"
    )
    for exp in result["experiments"]:
        print(
            f"{exp['axis']}: status={exp['status']} pass={exp['pass']} "
            f"CPU triggered={exp['cpu_triggered']} GPU triggered={exp['gpu_triggered']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
