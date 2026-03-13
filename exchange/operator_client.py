"""Hyperliquid operator client for non-runtime tooling.

This module exists for operator and emergency workflows that still need direct
exchange access without depending on the legacy runtime adapter in
`exchange.executor`.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants, types

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _normalise_limit_px_for_wire(px: float | None, *, is_buy: bool) -> float | None:
    """Normalise limit price to Hyperliquid wire precision."""
    if px is None:
        return None
    try:
        dec = Decimal(str(float(px)))
    except Exception:
        return None
    if not dec.is_finite() or dec <= 0:
        return None

    quantum = Decimal("0.00000001")
    rounding = ROUND_UP if bool(is_buy) else ROUND_DOWN
    try:
        out = dec.quantize(quantum, rounding=rounding)
    except (InvalidOperation, ValueError):
        return None
    try:
        out_f = float(out)
    except Exception:
        return None
    if out_f <= 0 or not math.isfinite(out_f):
        return None
    return out_f


def _local_slippage_limit_px(*, px: float | None, is_buy: bool, slippage_pct: float) -> float | None:
    """Compute a local IOC limit-price fallback when SDK helpers are unavailable."""
    if px is None:
        return None
    try:
        base_px = float(px)
    except Exception:
        return None
    if base_px <= 0 or not math.isfinite(base_px):
        return None
    try:
        slip = float(slippage_pct)
    except Exception:
        slip = 0.0
    slip = max(0.0, slip)
    mult = (1.0 + slip) if bool(is_buy) else max(0.0, 1.0 - slip)
    out = base_px * mult
    return _normalise_limit_px_for_wire(out, is_buy=bool(is_buy))


def is_ok_response(res) -> bool:
    try:
        if not isinstance(res, dict):
            return False
        return str(res.get("status", "")).lower() == "ok"
    except Exception:
        return False


@dataclass(frozen=True)
class LiveSecrets:
    secret_key: str
    main_address: str


def load_live_secrets(path: str) -> LiveSecrets:
    path = os.path.expanduser(str(path or "").strip())
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.name != "nt":
            st = os.fstat(fd)
            if (int(st.st_mode) & 0o077) != 0:
                raise ValueError(
                    f"Secrets file permissions too open: {path} (expected no group/other permissions; suggested: chmod 600)"
                )
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            fd = -1
            data = json.load(f)
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(data, dict):
        raise ValueError(f"Secrets file must contain a JSON object, got {type(data).__name__}: {path}")

    secret_key = str(data.get("secret_key") or "").strip()
    main_address = str(data.get("main_address") or "").strip()
    if not secret_key:
        raise ValueError(f"Missing 'secret_key' in {path}")
    if not re.match(r"^(0x)?[0-9a-fA-F]{64}$", secret_key):
        raise ValueError(
            f"Invalid 'secret_key' format in {path}: expected 64-char hex string (with optional 0x prefix)"
        )
    if not main_address:
        raise ValueError(f"Missing 'main_address' in {path}")
    if not re.match(r"^0x[0-9a-fA-F]{40}$", main_address):
        raise ValueError(
            f"Invalid 'main_address' format in {path}: expected 0x-prefixed 40-char hex string (42 chars total)"
        )
    return LiveSecrets(secret_key=secret_key, main_address=main_address)


@dataclass(frozen=True)
class OperatorAccountSnapshot:
    account_value_usd: float
    withdrawable_usd: float
    total_margin_used_usd: float
    ts: float


class HyperliquidOperatorClient:
    """Thin Hyperliquid client for operator tooling and recovery workflows."""

    _MAX_LEVERAGE = 50

    def __init__(
        self,
        *,
        secret_key: str,
        main_address: str,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ):
        self.main_address = str(main_address).strip()
        if not self.main_address:
            raise ValueError("main_address is required")
        self._wallet = Account.from_key(secret_key)
        self._base_url = base_url or constants.MAINNET_API_URL
        self._timeout_s = timeout_s
        self._info = Info(self._base_url, skip_ws=True, timeout=timeout_s)
        self._exchange = Exchange(
            self._wallet,
            self._base_url,
            account_address=self.main_address,
            timeout=timeout_s,
        )
        self._state_cache: dict | None = None
        self._state_cache_at: float | None = None
        try:
            self._state_ttl_s = float(os.getenv("HL_LIVE_USER_STATE_TTL_S", "3.0"))
        except Exception:
            self._state_ttl_s = 3.0
        self.last_order_error: dict | None = None

    def user_state(self, *, force: bool = False) -> dict:
        now = time.time()
        if not force and self._state_cache is not None and self._state_cache_at is not None:
            if (now - self._state_cache_at) <= max(0.0, self._state_ttl_s):
                return self._state_cache
        st = self._info.user_state(self.main_address) or {}
        self._state_cache = st
        self._state_cache_at = now
        return st

    def account_snapshot(self, *, force: bool = False) -> OperatorAccountSnapshot:
        st = self.user_state(force=force)
        margin = st.get("marginSummary") or {}
        return OperatorAccountSnapshot(
            account_value_usd=_safe_float(margin.get("accountValue"), 0.0),
            withdrawable_usd=_safe_float(st.get("withdrawable"), 0.0),
            total_margin_used_usd=_safe_float(margin.get("totalMarginUsed"), 0.0),
            ts=time.time(),
        )

    def get_positions(self, *, force: bool = False) -> dict[str, dict]:
        st = self.user_state(force=force)
        out: dict[str, dict] = {}
        for ap in st.get("assetPositions") or []:
            pos = (ap or {}).get("position") or {}
            coin = str(pos.get("coin") or "").strip().upper()
            if not coin:
                continue
            szi = _safe_float(pos.get("szi"), 0.0)
            if abs(szi) <= 0:
                continue
            pos_type = "LONG" if szi > 0 else "SHORT"
            size = abs(szi)
            lev_val = 1.0
            lev = pos.get("leverage") or {}
            try:
                lev_val = float(lev.get("value") or 1.0)
            except Exception:
                lev_val = 1.0
            lev_val = max(1.0, lev_val)
            out[coin] = {
                "type": pos_type,
                "size": float(size),
                "entry_price": _safe_float(pos.get("entryPx"), 0.0),
                "leverage": float(lev_val),
                "margin_used": _safe_float(pos.get("marginUsed"), 0.0),
                "raw": pos,
            }
        return out

    def all_mids(self) -> dict[str, str]:
        mids = self._info.all_mids() or {}
        return mids if isinstance(mids, dict) else {}

    def user_fills_by_time(self, start_ms: int, end_ms: int) -> list[dict]:
        fills = self._info.user_fills_by_time(
            self.main_address,
            start_ms,
            end_ms,
            aggregate_by_time=False,
        )
        return fills or []

    def open_orders(self) -> list[dict]:
        raw = self._info.open_orders(self.main_address) or []
        return raw if isinstance(raw, list) else []

    def update_leverage(self, symbol: str, leverage: float, *, is_cross: bool = True) -> bool:
        try:
            lev_f = float(leverage)
        except Exception:
            lev_f = 1.0
        if not math.isfinite(lev_f) or lev_f <= 0:
            return False
        lev_i = max(1, int(round(lev_f)))
        if lev_i > self._MAX_LEVERAGE:
            return False
        sym = str(symbol or "").strip().upper()
        if not sym:
            return False
        try:
            res = self._exchange.update_leverage(lev_i, sym, is_cross=is_cross)
            return is_ok_response(res)
        except Exception as exc:
            logger.warning("failed to update leverage for %s -> %dx: %s", sym, lev_i, exc)
            return False

    def _make_cloid_obj(self, cloid: str | None):
        if not cloid:
            return None
        try:
            return types.Cloid(str(cloid))
        except Exception:
            return None

    def market_open(
        self,
        symbol: str,
        *,
        is_buy: bool,
        sz: float,
        px: float | None = None,
        slippage_pct: float,
        cloid: str | None = None,
    ) -> dict | None:
        sym = str(symbol or "").strip().upper()
        try:
            sz_f = float(sz)
        except Exception:
            return None
        if not sym or sz_f <= 0 or not math.isfinite(sz_f):
            logger.warning("[preflight] market_open blocked: invalid sz=%r for %s", sz, sym)
            return None
        try:
            slip = float(slippage_pct)
        except Exception:
            slip = 0.01
        slip = max(0.0005, min(0.10, slip))
        px_f = None
        if px is not None:
            try:
                px_f = float(px)
            except Exception:
                px_f = None
        cloid_obj = self._make_cloid_obj(cloid)
        try:
            res = self._exchange.market_open(
                sym,
                is_buy=bool(is_buy),
                sz=sz_f,
                px=px_f,
                slippage=slip,
                cloid=cloid_obj,
            )
            if not is_ok_response(res):
                self.last_order_error = {"kind": "rejected", "op": "market_open", "response": res}
                return None
            self.last_order_error = None
            return res
        except Exception as exc:
            kind = "timeout" if "timed out" in str(exc).lower() else "exception"
            self.last_order_error = {"kind": kind, "op": "market_open", "error": repr(exc)}
            logger.warning("market_open failed (%s, is_buy=%s, sz=%s): %s", sym, is_buy, sz_f, exc)
            return None

    def limit_order(
        self,
        symbol: str,
        *,
        is_buy: bool,
        sz: float,
        limit_px: float,
        tif: str,
        reduce_only: bool,
        cloid: str | None = None,
    ) -> dict | None:
        sym = str(symbol or "").strip().upper()
        try:
            sz_f = float(sz)
            px_f = float(limit_px)
        except Exception:
            return None
        if not sym or sz_f <= 0 or not math.isfinite(sz_f) or px_f <= 0 or not math.isfinite(px_f):
            return None
        tif_norm = str(tif or "").strip().title()
        if tif_norm not in {"Ioc", "Gtc"}:
            raise ValueError(f"unsupported tif: {tif}")
        cloid_obj = self._make_cloid_obj(cloid)
        try:
            res = self._exchange.order(
                sym,
                is_buy=bool(is_buy),
                sz=sz_f,
                limit_px=px_f,
                order_type={"limit": {"tif": tif_norm}},
                reduce_only=bool(reduce_only),
                cloid=cloid_obj,
            )
            if not is_ok_response(res):
                self.last_order_error = {
                    "kind": "rejected",
                    "op": "limit_order",
                    "response": res,
                }
                return None
            self.last_order_error = None
            return res
        except Exception as exc:
            kind = "timeout" if "timed out" in str(exc).lower() else "exception"
            self.last_order_error = {"kind": kind, "op": "limit_order", "error": repr(exc)}
            logger.warning("limit_order failed (%s, is_buy=%s, sz=%s): %s", sym, is_buy, sz_f, exc)
            return None

    def market_close(
        self,
        symbol: str,
        *,
        is_buy: bool,
        sz: float,
        px: float | None = None,
        slippage_pct: float,
        cloid: str | None = None,
    ) -> dict | None:
        sym = str(symbol or "").strip().upper()
        try:
            sz_f = float(sz)
        except Exception:
            return None
        if not sym or sz_f <= 0 or not math.isfinite(sz_f):
            logger.warning("[preflight] market_close blocked: invalid sz=%r for %s", sz, sym)
            return None
        try:
            slip = float(slippage_pct)
        except Exception:
            slip = 0.01
        slip = max(0.0005, min(0.10, slip))
        px_f = None
        if px is not None:
            try:
                px_f = float(px)
            except Exception:
                px_f = None
            if px_f is not None and (px_f <= 0 or not math.isfinite(px_f)):
                px_f = None
        cloid_obj = self._make_cloid_obj(cloid)
        try:
            limit_px = self._exchange._slippage_price(sym, bool(is_buy), slip, px=px_f)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "_slippage_price unavailable for %s; falling back to local slippage calc: %s",
                sym,
                exc,
            )
            limit_px = None
        if limit_px is None:
            logger.warning("_slippage_price returned None for %s; attempting local slippage fallback", sym)
            if px_f is None:
                mids = self.all_mids()
                mid_px = _safe_float(mids.get(sym), 0.0)
                if mid_px > 0 and math.isfinite(mid_px):
                    px_f = mid_px
                    logger.warning("market_close using all_mids reference px for %s: %.8f", sym, px_f)
        limit_px_f = _safe_float(_normalise_limit_px_for_wire(limit_px, is_buy=bool(is_buy)), 0.0)
        if limit_px_f <= 0 or not math.isfinite(limit_px_f):
            if px_f is not None:
                limit_px_f = _safe_float(
                    _local_slippage_limit_px(px=px_f, is_buy=bool(is_buy), slippage_pct=slip),
                    0.0,
                )
                if limit_px_f > 0 and math.isfinite(limit_px_f):
                    logger.warning("market_close using local slippage fallback for %s", sym)
        if limit_px_f <= 0 or not math.isfinite(limit_px_f):
            self.last_order_error = {
                "kind": "preflight",
                "op": "market_close",
                "error": "unable to derive limit price for market_close",
            }
            logger.warning("[preflight] market_close blocked: unable to derive limit price for %s", sym)
            return None
        try:
            res = self._exchange.order(
                sym,
                is_buy=bool(is_buy),
                sz=sz_f,
                limit_px=float(limit_px_f),
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,
                cloid=cloid_obj,
            )
            if not is_ok_response(res):
                self.last_order_error = {"kind": "rejected", "op": "market_close", "response": res}
                return None
            self.last_order_error = None
            return res
        except Exception as exc:
            kind = "timeout" if "timed out" in str(exc).lower() else "exception"
            self.last_order_error = {"kind": kind, "op": "market_close", "error": repr(exc)}
            logger.warning("market_close failed (%s, sz=%s): %s", sym, sz_f, exc)
            return None

    def cancel_order(self, *, symbol: str, oid: str | None = None, cloid: str | None = None) -> dict | None:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return None
        cloid_obj = self._make_cloid_obj(cloid)
        if cloid_obj is not None:
            try:
                res = self._exchange.cancel_by_cloid(sym, cloid_obj)
                return res if is_ok_response(res) else None
            except Exception as exc:
                logger.warning("cancel_by_cloid failed (%s, cloid=%s): %s", sym, cloid, exc)
        if oid is None:
            return None
        try:
            oid_i = int(float(str(oid).strip()))
        except Exception:
            return None
        try:
            res = self._exchange.cancel(sym, oid_i)
            return res if is_ok_response(res) else None
        except Exception as exc:
            logger.warning("cancel failed (%s, oid=%s): %s", sym, oid_i, exc)
            return None


def utc_iso(ms: int | None = None) -> str:
    if ms is None:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    return datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).isoformat()


def live_mode() -> str:
    return str(os.getenv("AI_QUANT_MODE", "paper") or "paper").strip().lower()


def live_orders_enabled() -> bool:
    if _env_bool("AI_QUANT_HARD_KILL_SWITCH", False):
        return False
    if not _env_bool("AI_QUANT_LIVE_ENABLE", False):
        return False
    confirm = os.getenv("AI_QUANT_LIVE_CONFIRM", "")
    return confirm.strip() == "I_UNDERSTAND_THIS_CAN_LOSE_MONEY"


def live_entries_enabled() -> bool:
    if _env_bool("AI_QUANT_KILL_SWITCH", False):
        return False
    return live_orders_enabled()


def live_trading_enabled() -> bool:
    return live_entries_enabled()
