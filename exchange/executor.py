import datetime
import json
import os
import re
import time
from dataclasses import dataclass

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants, types


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


def _is_ok_response(res) -> bool:
    """
    Hyperliquid SDK actions typically return:
      {"status":"ok","response":...}
    or:
      {"status":"err","response":"..."}
    """
    if isinstance(res, dict) and "status" in res:
        ok = str(res.get("status") or "").strip().lower() == "ok"
        if not ok:
            return False

        # HL can return {"status":"ok"} while the embedded per-order status contains an error,
        # e.g. {"response":{"data":{"statuses":[{"error":"Order must have minimum value of $10. ..."}]}}}.
        try:
            resp = res.get("response") or {}
            data = resp.get("data") or {}
            statuses = data.get("statuses") or []
        except Exception:
            statuses = []
        if isinstance(statuses, list):
            for st in statuses:
                if isinstance(st, dict) and st.get("error"):
                    return False
        return True

    return bool(res)


@dataclass(frozen=True)
class LiveSecrets:
    secret_key: str
    main_address: str


def load_live_secrets(path: str) -> LiveSecrets:
    path = os.path.expanduser(str(path or "").strip())
    # Refuse to use secrets files that are group/world-readable on Unix.
    # Private keys must be treated as production credentials.
    st = None
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise
    except Exception:
        st = None
    if st is not None and os.name != "nt":
        if (int(st.st_mode) & 0o077) != 0:
            raise ValueError(
                f"Secrets file permissions too open: {path} (expected no group/other permissions; suggested: chmod 600)"
            )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Secrets file must contain a JSON object, got {type(data).__name__}: {path}")

    secret_key = str(data.get("secret_key") or "").strip()
    main_address = str(data.get("main_address") or "").strip()

    if not secret_key:
        raise ValueError(f"Missing 'secret_key' in {path}")
    # Accept 64 hex chars or 66 with 0x prefix (standard Ethereum private key).
    _HEX_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
    if not _HEX_KEY_RE.match(secret_key):
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
class LiveAccountSnapshot:
    account_value_usd: float
    withdrawable_usd: float
    total_margin_used_usd: float
    ts: float


class HyperliquidLiveExecutor:
    """
    Thin wrapper around the Hyperliquid SDK (REST for account state + signed REST for orders).

    - Market data and user fills/funding are handled separately via `hyperliquid_ws.py`.
    - This class rate-limits `user_state()` to avoid hammering the API.
    - NOTE(L10): SSL/TLS verification is handled by the Hyperliquid Python SDK (requests library).
      The SDK uses the system CA bundle by default. No custom SSL configuration is needed here.
    """

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
        # IMPORTANT: account_address ensures an approved agent wallet can trade for the main address.
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

        self._spot_cache: dict | None = None
        self._spot_cache_at: float | None = None
        try:
            self._spot_ttl_s = float(os.getenv("HL_LIVE_SPOT_STATE_TTL_S", "10.0"))
        except Exception:
            self._spot_ttl_s = 10.0

        # Last order submit outcome (best-effort). Used by the trader to distinguish hard rejections
        # from transient transport errors (for example timeouts) so OMS intents can be kept matchable.
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

    def spot_user_state(self, *, force: bool = False) -> dict:
        now = time.time()
        if not force and self._spot_cache is not None and self._spot_cache_at is not None:
            if (now - self._spot_cache_at) <= max(0.0, self._spot_ttl_s):
                return self._spot_cache

        st = self._info.spot_user_state(self.main_address) or {}
        self._spot_cache = st
        self._spot_cache_at = now
        return st

    def spot_usdc_total(self, *, force: bool = False) -> float:
        st = self.spot_user_state(force=force)
        for bal in st.get("balances") or []:
            if not isinstance(bal, dict):
                continue
            if str(bal.get("coin") or "").strip().upper() != "USDC":
                continue
            return _safe_float(bal.get("total"), 0.0)
        return 0.0

    def account_snapshot(self, *, force: bool = False) -> LiveAccountSnapshot:
        st = self.user_state(force=force)
        margin = st.get("marginSummary") or {}
        account_value = _safe_float(margin.get("accountValue"), 0.0)
        total_margin_used = _safe_float(margin.get("totalMarginUsed"), 0.0)
        withdrawable = _safe_float(st.get("withdrawable"), 0.0)
        return LiveAccountSnapshot(
            account_value_usd=account_value,
            withdrawable_usd=withdrawable,
            total_margin_used_usd=total_margin_used,
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
            entry_px = _safe_float(pos.get("entryPx"), 0.0)

            lev_val = 1.0
            lev = pos.get("leverage") or {}
            try:
                lev_val = float(lev.get("value") or 1.0)
            except Exception:
                lev_val = 1.0
            lev_val = max(1.0, lev_val)

            margin_used = _safe_float(pos.get("marginUsed"), 0.0)
            out[coin] = {
                "type": pos_type,
                "size": float(size),
                "entry_price": float(entry_px),
                "leverage": float(lev_val),
                "margin_used": float(margin_used),
                "raw": pos,
            }
        return out

    def update_leverage(self, symbol: str, leverage: float, *, is_cross: bool = True) -> bool:
        try:
            lev_i = int(round(float(leverage)))
        except Exception:
            lev_i = 1
        lev_i = max(1, lev_i)

        sym = str(symbol or "").strip().upper()
        if not sym:
            return False

        try:
            res = self._exchange.update_leverage(lev_i, sym, is_cross=is_cross)
            if not _is_ok_response(res):
                print(f"⚠️ update_leverage rejected for {sym} -> {lev_i}x: {res}")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Failed to update leverage for {sym} -> {lev_i}x: {e}")
            return False

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
        if not sym:
            return None
        try:
            sz_f = float(sz)
        except Exception:
            return None
        if sz_f <= 0:
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
            if px_f is not None and px_f <= 0:
                px_f = None

        cloid_obj = None
        if cloid:
            try:
                cloid_obj = types.Cloid(str(cloid))
            except Exception:
                cloid_obj = None

        try:
            res = self._exchange.market_open(
                sym,
                is_buy=bool(is_buy),
                sz=sz_f,
                px=px_f,
                slippage=slip,
                cloid=cloid_obj,
            )
            if not _is_ok_response(res):
                self.last_order_error = {
                    "kind": "rejected",
                    "op": "market_open",
                    "symbol": sym,
                    "is_buy": bool(is_buy),
                    "sz": float(sz_f),
                    "response": res,
                }
                print(f"⚠️ market_open rejected ({sym}, is_buy={is_buy}, sz={sz_f}): {res}")
                return None
            self.last_order_error = None
            return res
        except Exception as e:
            msg = str(e)
            kind = "timeout" if ("timed out" in msg.lower()) else "exception"
            self.last_order_error = {
                "kind": kind,
                "op": "market_open",
                "symbol": sym,
                "is_buy": bool(is_buy),
                "sz": float(sz_f),
                "error": repr(e),
            }
            print(f"⚠️ market_open failed ({sym}, is_buy={is_buy}, sz={sz_f}): {e}")
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
        if not sym:
            return None

        try:
            sz_f = float(sz)
        except Exception:
            return None
        if sz_f <= 0:
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
            if px_f is not None and px_f <= 0:
                px_f = None

        cloid_obj = None
        if cloid:
            try:
                cloid_obj = types.Cloid(str(cloid))
            except Exception:
                cloid_obj = None

        try:
            # Avoid Exchange.market_close(): it performs an extra Info.user_state() REST call.
            # We already know side + size in the strategy, so submit a reduce-only IOC limit.
            try:
                limit_px = self._exchange._slippage_price(sym, bool(is_buy), slip, px=px_f)  # type: ignore[attr-defined]
            except Exception:
                limit_px = None
            if limit_px is None:
                return None

            res = self._exchange.order(
                sym,
                is_buy=bool(is_buy),
                sz=sz_f,
                limit_px=float(limit_px),
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,
                cloid=cloid_obj,
            )
            if not _is_ok_response(res):
                self.last_order_error = {
                    "kind": "rejected",
                    "op": "market_close",
                    "symbol": sym,
                    "is_buy": bool(is_buy),
                    "sz": float(sz_f),
                    "response": res,
                }
                print(f"⚠️ market_close rejected ({sym}, sz={sz_f}): {res}")
                return None
            self.last_order_error = None
            return res
        except Exception as e:
            msg = str(e)
            kind = "timeout" if ("timed out" in msg.lower()) else "exception"
            self.last_order_error = {
                "kind": kind,
                "op": "market_close",
                "symbol": sym,
                "is_buy": bool(is_buy),
                "sz": float(sz_f),
                "error": repr(e),
            }
            print(f"⚠️ market_close failed ({sym}, sz={sz_f}): {e}")
            return None

    def cancel_order(self, *args, **kwargs) -> dict | None:
        """Best-effort cancel wrapper used by OMS reconciler.

        Supports multiple call shapes:
          - cancel_order(oid)
          - cancel_order(symbol, oid)
          - cancel_order(symbol=..., oid=...)
          - cancel_order(coin=..., exchange_order_id=...)
          - cancel_order(cloid=...)  (requires symbol/coin)
        """
        symbol = kwargs.get("symbol") or kwargs.get("coin") or kwargs.get("name")
        oid = kwargs.get("oid") or kwargs.get("exchange_order_id") or kwargs.get("order_id")
        cloid = kwargs.get("cloid") or kwargs.get("client_order_id")

        if len(args) == 2 and not symbol:
            symbol = args[0]
            if oid is None:
                oid = args[1]
        elif len(args) == 1 and oid is None and cloid is None:
            # Heuristic: numeric -> oid else cloid.
            s = str(args[0] or "").strip()
            if s.isdigit():
                oid = s
            else:
                cloid = s

        sym = str(symbol or "").strip().upper()
        if not sym:
            return None

        # Prefer cloid cancel when available.
        if cloid:
            cloid_obj = None
            try:
                cloid_obj = types.Cloid(str(cloid))
            except Exception:
                cloid_obj = None
            if cloid_obj is not None:
                try:
                    res = self._exchange.cancel_by_cloid(sym, cloid_obj)
                    if not _is_ok_response(res):
                        print(f"⚠️ cancel_by_cloid rejected ({sym}, cloid={cloid}): {res}")
                        return None
                    return res
                except Exception as e:
                    print(f"⚠️ cancel_by_cloid failed ({sym}, cloid={cloid}): {e}")
                    return None

        if oid is None:
            return None
        try:
            oid_i = int(float(str(oid).strip()))
        except Exception:
            oid_i = None
        if oid_i is None:
            return None

        try:
            res = self._exchange.cancel(sym, oid_i)
            if not _is_ok_response(res):
                print(f"⚠️ cancel rejected ({sym}, oid={oid_i}): {res}")
                return None
            return res
        except Exception as e:
            print(f"⚠️ cancel failed ({sym}, oid={oid_i}): {e}")
            return None

    def cancel(self, *args, **kwargs) -> dict | None:
        # Alias to match best-effort reconciler call shapes.
        return self.cancel_order(*args, **kwargs)


def live_mode() -> str:
    return str(os.getenv("AI_QUANT_MODE", "paper") or "paper").strip().lower()


def live_orders_enabled() -> bool:
    # Hard stop (halts ALL orders, including exits)
    if _env_bool("AI_QUANT_HARD_KILL_SWITCH", False):
        return False

    if not _env_bool("AI_QUANT_LIVE_ENABLE", False):
        return False

    confirm = os.getenv("AI_QUANT_LIVE_CONFIRM", "")
    return confirm.strip() == "I_UNDERSTAND_THIS_CAN_LOSE_MONEY"


def live_entries_enabled() -> bool:
    # Close-only mode: no new entries/adds, but allow exits/reduces.
    if _env_bool("AI_QUANT_KILL_SWITCH", False):
        return False
    return live_orders_enabled()


def live_trading_enabled() -> bool:
    # Backwards-compat alias: "trading" == "entries enabled".
    return live_entries_enabled()


def utc_iso(ms: int | None = None) -> str:
    if ms is None:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    return datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).isoformat()
