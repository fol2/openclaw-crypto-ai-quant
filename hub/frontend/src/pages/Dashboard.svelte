<script lang="ts">
  import { appState } from '../lib/stores.svelte';
  import { getSnapshot, getCandles, getMarks, postFlashDebug } from '../lib/api';
  import { hubWs } from '../lib/ws';

  const INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'] as const;
  const BAR_COUNTS = [50, 100, 200, 400] as const;
  const CHART_PREF_INTERVAL_COOKIE = 'aiq_dash_iv';
  const CHART_PREF_BARS_COOKIE = 'aiq_dash_bars';

  let snap: any = $state(null);
  let focusSym = $state('');
  let candles: any[] = $state([]);
  let marks: any = $state(null);
  let selectedInterval = $state('1h');
  let selectedBars: number = $state(200);
  let pollTimer: any = null;
  let error = $state('');
  let mobileTab: 'symbols' | 'detail' | 'feed' = $state('symbols');
  let detailTab: 'detail' | 'trades' | 'oms' | 'audit' = $state('detail');
  let detailExpanded = $state(false);
  let chartHeight = $state(240);
  let chartDragging = $state(false);
  const CHART_MIN = 120;
  const CHART_MAX = 600;

  function onChartSplitterDown(e: PointerEvent) {
    e.preventDefault();
    chartDragging = true;
    const startY = e.clientY;
    const startH = chartHeight;
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);

    function onMove(ev: PointerEvent) {
      const h = startH + (ev.clientY - startY);
      chartHeight = Math.max(CHART_MIN, Math.min(CHART_MAX, h));
    }
    function onUp() {
      chartDragging = false;
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
    }
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
  }

  function readCookie(name: string): string | null {
    if (typeof document === 'undefined') return null;
    const needle = `${encodeURIComponent(name)}=`;
    const parts = String(document.cookie || '').split(';');
    for (const raw of parts) {
      const p = raw.trim();
      if (p.startsWith(needle)) return decodeURIComponent(p.slice(needle.length));
    }
    return null;
  }

  function writeCookie(name: string, value: string, maxAgeS = 31_536_000) {
    if (typeof document === 'undefined') return;
    document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}; Max-Age=${maxAgeS}; Path=/; SameSite=Lax`;
  }

  function isValidInterval(iv: string): iv is (typeof INTERVALS)[number] {
    return (INTERVALS as readonly string[]).includes(iv);
  }

  function isValidBars(n: number): n is (typeof BAR_COUNTS)[number] {
    return (BAR_COUNTS as readonly number[]).includes(n);
  }

  function loadChartPrefsFromCookie() {
    const ivRaw = String(readCookie(CHART_PREF_INTERVAL_COOKIE) || '').trim().toLowerCase();
    if (ivRaw && isValidInterval(ivRaw)) selectedInterval = ivRaw;

    const barsRaw = Number(readCookie(CHART_PREF_BARS_COOKIE));
    if (Number.isFinite(barsRaw) && isValidBars(barsRaw)) selectedBars = barsRaw;
  }

  type FlashDebugEvent = {
    symbol: string;
    prev: number;
    mid: number;
    direction: 'up' | 'down';
    phase: 'a' | 'b';
    source: 'table' | 'detail';
    tone: 'table' | 'accent';
    at_ms: number;
  };

  const flashDebugEnabled = resolveFlashDebugEnabled();
  let flashDebugQueue: FlashDebugEvent[] = [];
  let flashDebugTimer: any = null;
  const flashDebugFlushMs = 250;
  const flashDebugBatchMax = 300;

  function resolveFlashDebugEnabled(): boolean {
    if (typeof window === 'undefined') return false;
    try {
      const query = new URLSearchParams(window.location.search).get('flash_debug')?.toLowerCase();
      if (query === '1' || query === 'true') {
        window.localStorage.setItem('aiq_flash_debug', '1');
        return true;
      }
      if (query === '0' || query === 'false') {
        window.localStorage.removeItem('aiq_flash_debug');
        return false;
      }
      return window.localStorage.getItem('aiq_flash_debug') === '1';
    } catch {
      return false;
    }
  }

  function scheduleFlashDebugFlush() {
    if (flashDebugTimer != null) return;
    flashDebugTimer = setTimeout(() => {
      flashDebugTimer = null;
      void flushFlashDebug();
    }, flashDebugFlushMs);
  }

  function onMidFlashTrigger(event: CustomEvent<any>, source: 'table' | 'detail') {
    if (!flashDebugEnabled) return;
    const detail = event?.detail;
    if (!detail || typeof detail.symbol !== 'string' || detail.symbol.length === 0) return;
    if (!Number.isFinite(detail.prev) || !Number.isFinite(detail.mid)) return;
    if (detail.direction !== 'up' && detail.direction !== 'down') return;
    if (detail.phase !== 'a' && detail.phase !== 'b') return;
    if (detail.tone !== 'table' && detail.tone !== 'accent') return;

    flashDebugQueue.push({
      symbol: detail.symbol,
      prev: detail.prev,
      mid: detail.mid,
      direction: detail.direction,
      phase: detail.phase,
      source,
      tone: detail.tone,
      at_ms: Number.isFinite(detail.at_ms) ? detail.at_ms : Date.now(),
    });

    if (flashDebugQueue.length >= flashDebugBatchMax) {
      if (flashDebugTimer != null) {
        clearTimeout(flashDebugTimer);
        flashDebugTimer = null;
      }
      void flushFlashDebug();
      return;
    }
    scheduleFlashDebugFlush();
  }

  async function flushFlashDebug() {
    if (!flashDebugEnabled || flashDebugQueue.length === 0) return;
    const batch = flashDebugQueue.splice(0, flashDebugBatchMax);
    try {
      await postFlashDebug(batch);
    } catch {
      // Keep the debug pipeline best-effort and avoid affecting UI flow.
      flashDebugQueue = batch.concat(flashDebugQueue).slice(0, 2000);
    }
    if (flashDebugQueue.length > 0) {
      scheduleFlashDebugFlush();
    }
  }

  function pnlPct(pos: any): number | null {
    if (!pos || pos.entry_price == null || !pos.size || pos.unreal_pnl_est == null) return null;
    const notional = pos.entry_price * Math.abs(pos.size);
    if (notional === 0) return null;
    return (pos.unreal_pnl_est / notional) * 100;
  }

  function fmtNum(v: number | null | undefined, dp = 2): string {
    if (v === null || v === undefined || !Number.isFinite(v)) return '\u2014';
    return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }
  function fmtAge(s: number | null | undefined): string {
    if (s === null || s === undefined) return '\u2014';
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.round(s / 60)}m`;
    return `${(s / 3600).toFixed(1)}h`;
  }
  function sigAge(ts: string | null | undefined): string {
    if (!ts) return '';
    const d = new Date(ts);
    if (isNaN(d.getTime())) return '';
    return fmtAge((Date.now() - d.getTime()) / 1000);
  }
  function isFreshSig(ts: string | null | undefined): boolean {
    if (!ts) return false;
    const d = new Date(ts);
    if (isNaN(d.getTime())) return false;
    return (Date.now() - d.getTime()) < 3_600_000; // 1 hour
  }
  function pnlClass(v: number | null | undefined): string {
    if (v === null || v === undefined) return '';
    return v >= 0 ? 'green' : 'red';
  }

  async function refresh() {
    try {
      appState.loading = true;
      const data = await getSnapshot(appState.mode);
      updateServerClockOffset(data?.now_ts_ms);
      // Preserve live WS mid prices — don't let stale REST prices overwrite them.
      // WS owns price; REST owns everything else (positions, signals, heartbeat, balances).
      if (snap?.symbols && data?.symbols) {
        const liveMids: Record<string, number> = {};
        for (const s of snap.symbols) {
          if (s.mid != null) liveMids[s.symbol] = s.mid;
        }
        for (const s of data.symbols) {
          if (liveMids[s.symbol] !== undefined) s.mid = liveMids[s.symbol];
        }
      }
      snap = data;
      appState.snapshot = data;
      error = '';
    } catch (e: any) {
      error = e.message || 'fetch failed';
    } finally {
      appState.loading = false;
    }
  }

  // Returns candle limit sized to give roughly 1-7 days of data per interval.
  function candleLimit(iv: string): number {
    switch (iv) {
      case '1m':  return 400;   // ~6.7 h
      case '3m':  return 300;   // ~15 h
      case '5m':  return 288;   // ~24 h
      case '15m': return 240;   // ~60 h
      case '30m': return 200;   // ~4 days
      case '1h':  return 168;   // ~7 days
      case '4h':  return 180;   // ~30 days
      case '1d':  return 200;   // ~200 days
      default:    return 200;
    }
  }

  function intervalToMs(iv: string): number {
    const m = /^([0-9]+)([mhd])$/i.exec(String(iv || '').trim());
    if (!m) return 60_000;
    const n = Number(m[1]);
    if (!Number.isFinite(n) || n <= 0) return 60_000;
    const unit = String(m[2] || '').toLowerCase();
    if (unit === 'm') return n * 60_000;
    if (unit === 'h') return n * 60 * 60_000;
    if (unit === 'd') return n * 24 * 60 * 60_000;
    return 60_000;
  }

  function newestCandleIndex(rows: any[]): number {
    let idx = 0;
    for (let i = 1; i < rows.length; i++) {
      if (Number(rows[i]?.t || 0) > Number(rows[idx]?.t || 0)) idx = i;
    }
    return idx;
  }

  // Track previous symbol so we can clear candles on symbol switch
  // (plain let — not reactive, no effect re-run on change)
  let _prevFocusSym = '';
  // Client/server clock offset for candle-boundary alignment.
  let _serverNowOffsetMs = 0;
  // Timestamp of last live-candle paint; used to cap redraws at ~15fps
  let _liveUpdateMs = 0;
  // Candle reconciliation controls.
  const CANDLE_ROLLOVER_RECONCILE_DELAY_MS = 1600;
  const CANDLE_PERIODIC_RECONCILE_MS = 25_000;
  let _candlesFetchInFlight = false;
  let _candlesFetchQueued = false;
  let _candlesRolloverReconcileTimer: ReturnType<typeof setTimeout> | null = null;

  function updateServerClockOffset(serverNowMs: unknown) {
    const ts = Number(serverNowMs);
    if (!Number.isFinite(ts) || ts <= 0) return;
    _serverNowOffsetMs = ts - Date.now();
  }

  function serverNowMs(): number {
    return Date.now() + _serverNowOffsetMs;
  }

  function clearRolloverReconcileTimer() {
    if (_candlesRolloverReconcileTimer == null) return;
    clearTimeout(_candlesRolloverReconcileTimer);
    _candlesRolloverReconcileTimer = null;
  }

  async function reconcileCandlesForCurrentView(sym = focusSym, iv = selectedInterval, bars = selectedBars): Promise<void> {
    if (!sym) return;

    if (_candlesFetchInFlight) {
      _candlesFetchQueued = true;
      return;
    }
    _candlesFetchInFlight = true;
    try {
      const res = await getCandles(sym, iv, bars);
      // Ignore stale responses if focus context changed mid-flight.
      if (focusSym !== sym || selectedInterval !== iv || selectedBars !== bars) return;
      candles = res.candles || [];
    } catch {
      // Keep rendering the current candles on transient API failures.
    } finally {
      _candlesFetchInFlight = false;
      if (_candlesFetchQueued) {
        _candlesFetchQueued = false;
        setTimeout(() => { void reconcileCandlesForCurrentView(); }, 0);
      }
    }
  }

  function scheduleRolloverReconcile() {
    clearRolloverReconcileTimer();
    _candlesRolloverReconcileTimer = setTimeout(() => {
      _candlesRolloverReconcileTimer = null;
      void reconcileCandlesForCurrentView();
    }, CANDLE_ROLLOVER_RECONCILE_DELAY_MS);
  }

  async function setFocus(sym: string) {
    focusSym = sym;
    appState.focus = sym;
    candles = [];
    marks = null;
    _candlesFetchQueued = false;
    clearRolloverReconcileTimer();
    if (!sym) return;
    detailTab = 'detail';
    mobileTab = 'detail';
    try {
      marks = await getMarks(sym, appState.mode);
    } catch { /* ignore */ }
  }

  // Re-fetch candles when symbol or interval changes.
  // Only clears the chart when the symbol changes (not on interval switch)
  // so old data stays visible during the brief network round-trip.
  $effect(() => {
    const sym  = focusSym;
    const iv = selectedInterval;
    const bars = selectedBars;
    if (!sym) { candles = []; _prevFocusSym = ''; return; }
    if (sym !== _prevFocusSym) { candles = []; _prevFocusSym = sym; }
    void reconcileCandlesForCurrentView(sym, iv, bars);
  });

  // Low-frequency official-candle reconcile while detail view is open.
  $effect(() => {
    const sym = focusSym;
    const tab = detailTab;
    const iv = selectedInterval;
    const bars = selectedBars;
    if (!sym || tab !== 'detail') return;
    const id = setInterval(() => { void reconcileCandlesForCurrentView(sym, iv, bars); }, CANDLE_PERIODIC_RECONCILE_MS);
    return () => clearInterval(id);
  });

  // Live candle update: mutate current developing candle, and roll over to a
  // new synthetic candle once the exchange interval boundary is crossed.
  // Rate-limited to ~15fps (66ms) to avoid triggering JSON.stringify + full
  // canvas redraw at the raw WS cadence (~100ms / 10Hz).
  $effect(() => {
    const sym = focusSym;
    if (!sym) return;
    const handler = (data: any) => {
      updateServerClockOffset(data?.server_ts_ms);
      const localNow = Date.now();
      if (localNow - _liveUpdateMs < 66) return; // ~15fps cap
      const mid = Number(data?.mids?.[sym]);
      if (!Number.isFinite(mid) || mid <= 0 || candles.length === 0) return;

      const wsServerNow = Number(data?.server_ts_ms);
      const now = Number.isFinite(wsServerNow) && wsServerNow > 0 ? wsServerNow : serverNowMs();
      const msPerBar = intervalToMs(selectedInterval);
      const barStart = Math.floor(now / msPerBar) * msPerBar;
      const barClose = barStart + msPerBar - 1;

      const liveIdx = newestCandleIndex(candles);
      const c = candles[liveIdx];
      const candleStart = Number(c?.t || 0);
      if (!Number.isFinite(candleStart) || candleStart <= 0) return;

      if (candleStart < barStart) {
        const prevClose = Number.isFinite(Number(c.c)) ? Number(c.c) : mid;
        const o = prevClose;
        candles.push({
          t: barStart,
          t_close: barClose,
          o,
          h: Math.max(o, mid),
          l: Math.min(o, mid),
          c: mid,
          v: 0,
          n: 0,
        });
        if (candles.length > selectedBars) {
          candles.splice(0, candles.length - selectedBars);
        }
        // Replace synthetic rollover candle with exchange-written candles shortly after boundary.
        scheduleRolloverReconcile();
        _liveUpdateMs = localNow;
        return;
      }

      // Ignore out-of-order future candles caused by temporary clock skew.
      if (candleStart > barStart) return;

      const closeMs = Number(c.t_close || 0);
      if (!Number.isFinite(closeMs) || closeMs <= 0 || closeMs < barClose) {
        c.t_close = barClose;
      }

      const prevHigh = Number.isFinite(Number(c.h)) ? Number(c.h) : mid;
      const prevLow = Number.isFinite(Number(c.l)) ? Number(c.l) : mid;
      _liveUpdateMs = localNow;
      c.c = mid;
      c.h = Math.max(prevHigh, mid);
      c.l = Math.min(prevLow, mid);
    };
    hubWs.subscribe('mids', handler);
    return () => hubWs.unsubscribe('mids', handler);
  });

  function setMode(m: string) {
    appState.mode = m;
    focusSym = '';
    refresh();
  }

  function setFeed(f: 'trades' | 'oms' | 'audit') {
    detailTab = f;
  }

  // Persist last chart controls for the next visit.
  $effect(() => {
    writeCookie(CHART_PREF_INTERVAL_COOKIE, selectedInterval);
    writeCookie(CHART_PREF_BARS_COOKIE, String(selectedBars));
  });

  $effect(() => {
    loadChartPrefsFromCookie();
    refresh();
    pollTimer = setInterval(refresh, 5000);
    hubWs.connect();
    return () => {
      clearInterval(pollTimer);
      clearRolloverReconcileTimer();
      if (flashDebugTimer != null) {
        clearTimeout(flashDebugTimer);
        flashDebugTimer = null;
      }
      void flushFlashDebug();
    };
  });

  // Subscribe to real-time mid price updates over WebSocket (~100ms).
  // WS owns all price data; REST owns positions/signals/heartbeat/balances.
  $effect(() => {
    const midsHandler = (data: any) => {
      if (!snap?.symbols || !data?.mids) return;
      const newMids = data.mids as Record<string, number>;

      // Mutate mid prices + recalculate unrealized PnL in-place.
      // Svelte 5 deep proxies track these writes automatically.
      for (const s of snap.symbols) {
        const p = newMids[s.symbol];
        if (p === undefined) continue;
        s.mid = p;
        if (s.position?.entry_price != null && s.position?.size != null) {
          s.position.unreal_pnl_est =
            s.position.type === 'LONG'
              ? (p - s.position.entry_price) * s.position.size
              : (s.position.entry_price - p) * s.position.size;
        }
      }
    };
    hubWs.subscribe('mids', midsHandler);
    return () => hubWs.unsubscribe('mids', midsHandler);
  });

  // ── Resizable columns ──────────────────────────────────────────────────────
  const SYM_MIN = 200;
  let symWidth = $state(455);
  let dragging = $state(false);

  function onSplitterDown(e: PointerEvent) {
    e.preventDefault();
    dragging = true;
    const startX = e.clientX;
    const startW = symWidth;
    const gridEl = (e.currentTarget as HTMLElement).closest('.dashboard-grid') as HTMLElement | null;
    const symMax = Math.max(SYM_MIN + 100, (gridEl ? gridEl.clientWidth : window.innerWidth - 66) - 280);
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);

    function onMove(ev: PointerEvent) {
      const w = startW + (ev.clientX - startX);
      symWidth = Math.max(SYM_MIN, Math.min(symMax, w));
    }
    function onUp() {
      dragging = false;
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
    }
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
  }

  let symbols = $derived(snap?.symbols || []);
  let filteredSymbols = $derived.by(() => {
    const q = appState.search.trim().toUpperCase();
    if (!q) return symbols;
    return symbols.filter((s: any) => String(s.symbol).includes(q));
  });
  let health = $derived(snap?.health || {});
  let balances = $derived(snap?.balances || {});
  let daily = $derived(snap?.daily || {});
  let recent = $derived(snap?.recent || {});
  let openPositions = $derived(snap?.open_positions || []);

  // ── Range selector for PnL / DD ───────────────────────────────────────
  let metricsRange = $state<'today' | 'since' | 'all'>('today');
  let rangeMenuOpen = $state(false);

  let activePnl = $derived(
    metricsRange === 'today' ? daily.pnl_usd
    : metricsRange === 'since' ? snap?.since_config?.pnl_usd
    : snap?.all_time?.pnl_usd
  );
  let activeDd = $derived(
    metricsRange === 'today' ? daily.drawdown_pct
    : metricsRange === 'since' ? snap?.since_config?.drawdown_pct
    : snap?.all_time?.drawdown_pct
  );
  let pnlLabel = $derived(
    metricsRange === 'today' ? 'PnL'
    : metricsRange === 'since' ? 'PnL\u2219cfg'
    : 'PnL\u2219all'
  );
  let ddLabel = $derived(
    metricsRange === 'today' ? 'DD'
    : metricsRange === 'since' ? 'DD\u2219cfg'
    : 'DD\u2219all'
  );
  let sinceLabel = $derived(snap?.since_config?.label ?? 'Since cfg');

  function selectRange(r: 'today' | 'since' | 'all') {
    metricsRange = r;
    rangeMenuOpen = false;
  }

  function onRangeClickOutside(e: MouseEvent) {
    const target = e.target as HTMLElement;
    if (!target.closest('.range-dropdown-wrap')) {
      rangeMenuOpen = false;
    }
  }

  $effect(() => {
    if (rangeMenuOpen) {
      document.addEventListener('click', onRangeClickOutside, true);
      return () => document.removeEventListener('click', onRangeClickOutside, true);
    }
  });

  const gateReasonMap: Record<string, { label: string; desc: string }> = {
    disabled:            { label: 'Disabled',        desc: 'Gate feature is off — new entries always allowed.' },
    trend_ok:            { label: 'Trend OK',         desc: 'Market breadth is trending and BTC ADX + ATR% pass thresholds. Gate is open.' },
    breadth_chop:        { label: 'Breadth chop',     desc: 'Market breadth is inside the chop zone. New entries are blocked.' },
    btc_adx_low:         { label: 'BTC ADX weak',     desc: 'BTC ADX is below the minimum threshold (weak trend). New entries are blocked.' },
    btc_atr_low:         { label: 'BTC ATR low',      desc: 'BTC ATR% is below the minimum threshold (low volatility). New entries are blocked.' },
    breadth_missing:     { label: 'No breadth data',  desc: 'Market breadth data is unavailable. Gate state depends on fail-open setting.' },
    btc_metrics_missing: { label: 'No BTC metrics',   desc: 'BTC ADX/ATR could not be computed. Gate state depends on fail-open setting.' },
  };
  let gateInfo = $derived(
    gateReasonMap[(health.regime_reason ?? '').toLowerCase()] ??
    { label: (health.regime_reason ?? '—'), desc: '' }
  );
</script>

<!-- Mode selector + metrics -->
<div class="topbar">
  <div class="topbar-row">
    <div class="mode-tabs">
      {#each ['live', 'paper1', 'paper2', 'paper3'] as m}
        <button
          class="mode-btn"
          class:active={appState.mode === m || (appState.mode === 'paper' && m === 'paper1')}
          class:is-live={m === 'live'}
          onclick={() => setMode(m)}
        >{m.toUpperCase()}</button>
      {/each}
    </div>

    <div class="status-chip" class:ok={health.ok} class:bad={!health.ok}>
      <span class="status-dot" class:alive={health.ok}></span>
      {health.ok ? 'ENGINE' : 'NO HB'}
    </div>
  </div>

  <div class="metrics-bar">
    {#if health.kill_mode && health.kill_mode !== 'off'}
      <span class="metric-pill danger">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        KILL: {health.kill_mode}
      </span>
    {/if}
    {#if health.regime_gate !== undefined && health.regime_gate !== null}
      <span class="metric-pill gate-pill" class:gate-on={health.regime_gate} class:gate-off={!health.regime_gate}>
        GATE {health.regime_gate ? 'ON' : 'OFF'}
        <span class="gate-tooltip">
          <span class="gt-title" class:gt-on={health.regime_gate} class:gt-off={!health.regime_gate}>
            GATE {health.regime_gate ? 'ON' : 'OFF'} — {gateInfo.label}
          </span>
          {#if gateInfo.desc}
            <span class="gt-desc">{gateInfo.desc}</span>
          {/if}
          <span class="gt-note">Gate OFF blocks new entries. Exits always continue.</span>
        </span>
      </span>
    {/if}
    <span class="metric-pill">
      <span class="metric-label">BAL</span>
      <span class="metric-value">${fmtNum(balances.realised_usd)}</span>
    </span>
    <span class="metric-pill">
      <span class="metric-label">EQ</span>
      <span class="metric-value">${fmtNum(balances.equity_est_usd)}</span>
    </span>
    <span class="range-dropdown-wrap">
      <button class="metric-pill range-pill {pnlClass(activePnl)}" onclick={() => rangeMenuOpen = !rangeMenuOpen}>
        <span class="metric-label">{pnlLabel}<svg class="range-caret" class:open={rangeMenuOpen} width="8" height="8" viewBox="0 0 8 8"><path d="M1.5 3L4 5.5L6.5 3" fill="none" stroke="currentColor" stroke-width="1.2"/></svg></span>
        <span class="metric-value">${fmtNum(activePnl)}</span>
      </button>
      {#if rangeMenuOpen}
        <div class="range-menu">
          <button class="range-opt" class:active={metricsRange === 'today'} onclick={() => selectRange('today')}>
            <span class="range-dot"></span>Today
          </button>
          <button class="range-opt" class:active={metricsRange === 'since'} onclick={() => selectRange('since')}>
            <span class="range-dot"></span>{sinceLabel}
          </button>
          <button class="range-opt" class:active={metricsRange === 'all'} onclick={() => selectRange('all')}>
            <span class="range-dot"></span>All-time
          </button>
        </div>
      {/if}
    </span>
    <span class="metric-pill">
      <span class="metric-label">{ddLabel}</span>
      <span class="metric-value">{fmtNum(activeDd, 1)}%</span>
    </span>
    <span class="metric-pill">
      <span class="metric-label">POS</span>
      <span class="metric-value">{openPositions.length}</span>
    </span>
  </div>

  {#if error}
    <div class="error-banner">{error}</div>
  {/if}
</div>

<!-- Mobile tab switcher -->
<div class="mobile-tabs">
  <button class="m-tab" class:active={mobileTab === 'symbols'} onclick={() => mobileTab = 'symbols'}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 6h16M4 10h16M4 14h16M4 18h16"/></svg>
    Symbols
  </button>
  <button class="m-tab" class:active={mobileTab === 'detail'} onclick={() => { mobileTab = 'detail'; detailTab = 'detail'; }}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
    Detail
  </button>
  <button class="m-tab" class:active={mobileTab === 'feed'} onclick={() => { mobileTab = 'feed'; if (detailTab === 'detail') detailTab = 'trades'; }}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
    Feed
  </button>
</div>

<div class="dashboard-grid" class:is-dragging={dragging || chartDragging} class:drag-col={dragging} class:drag-row={chartDragging}>
  <!-- Symbol table -->
  <div class="panel symbols-panel" class:mobile-visible={mobileTab === 'symbols'} class:expanded-hidden={detailExpanded} style="width:{symWidth}px;min-width:{symWidth}px">
    <div class="panel-header">
      <div class="search-wrap">
        <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input
          type="text"
          class="search-input"
          placeholder="Filter..."
          bind:value={appState.search}
        />
      </div>
      <span class="sym-count">{filteredSymbols.length}</span>
    </div>
    <div class="sym-table-wrap">
      <table class="sym-table">
        <thead>
          <tr>
            <th>SYM</th>
            <th class="col-mid">MID</th>
            <th>SIG</th>
            <th>POS</th>
          </tr>
        </thead>
        <tbody>
          {#each filteredSymbols as s (s.symbol)}
            <tr
              class:is-focus={focusSym === s.symbol}
              class:row-long={s.position?.type === 'LONG'}
              class:row-short={s.position?.type === 'SHORT'}
              onclick={() => setFocus(s.symbol)}
            >
              <td class="sym-name">{s.symbol}</td>
              <td class="num col-mid">
                <mid-price
                  symbol={s.symbol}
                  value={s.mid != null ? String(s.mid) : ''}
                  decimals={6}
                  onmid-flash-trigger={(e) => onMidFlashTrigger(e as CustomEvent, 'table')}
                ></mid-price>
              </td>
              <td>
                {#if isFreshSig(s.last_signal?.timestamp)}
                  {#if s.last_signal?.signal === 'BUY'}
                    <span class="sig-badge buy">BUY</span>
                  {:else if s.last_signal?.signal === 'SELL'}
                    <span class="sig-badge sell">SELL</span>
                  {/if}
                  <span class="sig-age">{sigAge(s.last_signal.timestamp)}</span>
                {:else}
                  <span class="sig-badge none">\u2014</span>
                {/if}
              </td>
              <td>
                {#if s.position}
                  <span class="pos-badge" class:long={s.position.type === 'LONG'} class:short={s.position.type === 'SHORT'}>
                    {s.position.type}
                  </span>
                  {#if s.position.unreal_pnl_est != null}
                    {@const pct = pnlPct(s.position)}
                    <span class="pos-pnl {pnlClass(s.position.unreal_pnl_est)}">
                      {s.position.unreal_pnl_est >= 0 ? '+' : ''}{fmtNum(s.position.unreal_pnl_est)}{pct != null ? ` ${pct >= 0 ? '+' : ''}${fmtNum(pct, 1)}%` : ''}
                    </span>
                  {/if}
                  {#if s.position.open_timestamp}
                    <span class="pos-age">{sigAge(s.position.open_timestamp)}</span>
                  {/if}
                {:else}
                  <span class="flat-label">flat</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Drag splitter -->
  <div class="splitter" class:active={dragging} class:expanded-hidden={detailExpanded} role="separator" aria-orientation="vertical" onpointerdown={onSplitterDown}></div>

  <!-- Detail + Feed panel (merged, tabbed) -->
  <div class="panel detail-panel" class:mobile-visible={mobileTab === 'detail' || mobileTab === 'feed'}>
    <div class="panel-header detail-header">
      {#if focusSym}
        <div class="focus-sym">
          <h3>{focusSym}</h3>
          {#each symbols.filter((s: any) => s.symbol === focusSym).slice(0, 1) as sym}
            <mid-price
              symbol={sym.symbol}
              tone="accent"
              value={sym.mid != null ? String(sym.mid) : ''}
              decimals={6}
              onmid-flash-trigger={(e) => onMidFlashTrigger(e as CustomEvent, 'detail')}
            ></mid-price>
          {/each}
        </div>
      {/if}
      <div class="detail-tabs">
        <button class="tab" class:is-on={detailTab === 'detail'} onclick={() => detailTab = 'detail'}>DETAIL</button>
        <button class="tab" class:is-on={detailTab === 'trades'} onclick={() => setFeed('trades')}>TRADES</button>
        <button class="tab" class:is-on={detailTab === 'oms'} onclick={() => setFeed('oms')}>OMS</button>
        <button class="tab" class:is-on={detailTab === 'audit'} onclick={() => setFeed('audit')}>AUDIT</button>
      </div>
      <button class="expand-btn" aria-label={detailExpanded ? 'Collapse' : 'Expand'} onclick={() => detailExpanded = !detailExpanded}>
        {#if detailExpanded}
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7"/></svg>
        {:else}
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
        {/if}
      </button>
      {#if focusSym}
        <button class="close-focus" aria-label="Close" onclick={() => { focusSym = ''; mobileTab = 'symbols'; }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      {/if}
    </div>

    {#if detailTab === 'detail'}
      {#if focusSym}
        <!-- Interval + bar count selector -->
        <div class="iv-bar">
          {#each INTERVALS as iv}
            <button
              class="iv-tab"
              class:is-on={selectedInterval === iv}
              onclick={() => { selectedInterval = iv; }}
            >{iv.toUpperCase()}</button>
          {/each}
          <span class="iv-sep"></span>
          {#each BAR_COUNTS as bc}
            <button
              class="iv-tab"
              class:is-on={selectedBars === bc}
              onclick={() => { selectedBars = bc; }}
            >{bc}</button>
          {/each}
        </div>
        <div class="chart-wrap" style="height:{chartHeight}px">
          <candle-chart
            candles={JSON.stringify(candles)}
            entries={JSON.stringify(marks?.entries || [])}
            entryPrice={marks?.position?.entry_price ?? 0}
            postype={marks?.position?.type ?? ''}
            symbol={focusSym}
            interval={selectedInterval}
          ></candle-chart>
        </div>
        <div class="chart-splitter" class:active={chartDragging} role="separator" aria-orientation="horizontal" onpointerdown={onChartSplitterDown}></div>

        {#if marks?.position}
          {@const p = marks.position}
          {@const livePos = snap?.symbols?.find((s: any) => s.symbol === focusSym)?.position}
          <div class="kv-section">
            <h4>Position</h4>
            <div class="kv"><span class="k">Type</span><span class="v">{p.pos_type || p.type}</span></div>
            <div class="kv"><span class="k">Size</span><span class="v mono">{fmtNum(p.size, 6)}</span></div>
            <div class="kv"><span class="k">Entry</span><span class="v mono">{fmtNum(p.entry_price, 6)}</span></div>
            <div class="kv"><span class="k">uPnL</span><span class="v {pnlClass(livePos?.unreal_pnl_est)}">{fmtNum(livePos?.unreal_pnl_est)}</span></div>
            <div class="kv"><span class="k">Leverage</span><span class="v">{fmtNum(p.leverage, 1)}x</span></div>
          </div>
          {#if marks?.entries?.length}
            <div class="kv-section">
              <h4>Entries</h4>
              {#each marks.entries as e}
                <div class="kv">
                  <span class="k">{e.action}</span>
                  <span class="v mono">@ {fmtNum(e.price, 6)} &times; {fmtNum(e.size, 4)}</span>
                </div>
              {/each}
            </div>
          {/if}
        {:else}
          <div class="empty-state">
            <span class="empty-label">Flat</span>
            <span class="empty-sub">No open position</span>
          </div>
        {/if}
      {:else}
        <div class="empty-focus">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" stroke-width="1"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
          <p>Select a symbol</p>
        </div>
      {/if}
    {:else}
      <!-- Feed content (trades / oms / audit) -->
      <div class="feed-content">
        {#if detailTab === 'trades'}
          {#each (recent.trades || []).slice(0, 40) as t}
            <div class="feed-item">
              <div class="feed-row">
                <span class="feed-l">
                  <span class="feed-sym">{t.symbol}</span>
                  <span class="feed-action">{t.action}</span>
                  <span class="feed-type">{t.type}</span>
                </span>
                <span class="feed-r">{t.timestamp?.slice(11, 19) || ''}</span>
              </div>
              <div class="feed-sub">
                px <span class="green mono">{fmtNum(t.price, 6)}</span> size {fmtNum(t.size, 6)}
                pnl <span class="{pnlClass(t.pnl)} mono">{t.pnl != null ? fmtNum(t.pnl) : '\u2014'}</span>
              </div>
            </div>
          {/each}
        {:else if detailTab === 'oms'}
          {#each (recent.oms_intents || []).slice(0, 25) as i}
            <div class="feed-item">
              <div class="feed-row">
                <span class="feed-l">
                  <span class="feed-sym">{i.symbol}</span>
                  {i.action} {i.side}
                  <span class:green={i.status === 'FILLED'} class:red={i.status === 'REJECTED'} class:yellow={i.status !== 'FILLED' && i.status !== 'REJECTED'}>{i.status}</span>
                </span>
                <span class="feed-r">{i.created_ts_ms ? new Date(i.created_ts_ms).toISOString().slice(11, 19) : ''}Z</span>
              </div>
              <div class="feed-sub">reason: {i.reason || '\u2014'} conf: {i.confidence || '\u2014'}</div>
            </div>
          {/each}
          {#each (recent.oms_fills || []).slice(0, 15) as f}
            <div class="feed-item">
              <div class="feed-row">
                <span class="feed-l">FILL {f.symbol} {f.side} {fmtNum(f.size, 6)}</span>
                <span class="feed-r">{f.ts_ms ? new Date(f.ts_ms).toISOString().slice(11, 19) : ''}Z</span>
              </div>
              <div class="feed-sub">px {fmtNum(f.price, 6)} pnl <span class="{pnlClass(f.pnl_usd)}">{f.pnl_usd != null ? fmtNum(f.pnl_usd) : '\u2014'}</span></div>
            </div>
          {/each}
        {:else if detailTab === 'audit'}
          {#each (recent.audit_events || []).slice(0, 40) as a}
            <div class="feed-item">
              <div class="feed-row">
                <span class="feed-l">{a.event || '\u2014'} {a.symbol || ''}</span>
                <span class="feed-r">{a.timestamp?.slice(11, 19) || ''}</span>
              </div>
              <div class="feed-sub">{a.level || 'info'}</div>
            </div>
          {/each}
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* ─── Topbar ─── */
  .topbar {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    animation: slideUp 0.3s ease;
  }
  .topbar-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .mode-tabs {
    display: flex;
    gap: 3px;
    background: var(--bg);
    border-radius: var(--radius-md);
    padding: 3px;
    border: 1px solid var(--border);
  }
  .mode-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 5px 14px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-family: 'IBM Plex Mono', monospace;
    transition: all var(--t-fast);
  }
  .mode-btn:hover {
    color: var(--text);
    background: var(--surface-hover);
  }
  .mode-btn.active {
    background: var(--accent);
    color: var(--bg);
    box-shadow: 0 1px 4px rgba(77,171,247,0.3);
  }
  .mode-btn.active.is-live {
    background: var(--red);
    box-shadow: 0 1px 4px rgba(255,107,107,0.3);
  }

  .status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--border);
  }
  .status-chip.ok {
    border-color: rgba(81,207,102,0.3);
    color: var(--green);
  }
  .status-chip.bad {
    border-color: rgba(255,107,107,0.3);
    color: var(--red);
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--red);
  }
  .status-dot.alive {
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2s ease-in-out infinite;
  }

  .metrics-bar {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
  }
  .metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-pill.danger {
    border-color: rgba(255,107,107,0.3);
    color: var(--red);
    background: var(--red-bg);
  }
  .metric-pill.gate-on {
    border-color: rgba(81,207,102,0.3);
    color: var(--green);
  }
  .metric-pill.gate-off {
    border-color: rgba(255,107,107,0.3);
    color: var(--red);
  }
  .gate-pill {
    position: relative;
    cursor: help;
  }
  .gate-tooltip {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    width: 250px;
    background: #111118;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 8px 10px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    z-index: 200;
    flex-direction: column;
    gap: 4px;
    white-space: normal;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  }
  .gate-pill:hover .gate-tooltip {
    display: flex;
  }
  .gt-title {
    font-weight: 600;
    font-size: 11px;
  }
  .gt-title.gt-on  { color: var(--green); }
  .gt-title.gt-off { color: var(--red); }
  .gt-desc {
    color: var(--text);
    font-size: 10px;
    margin-top: 2px;
    line-height: 1.4;
  }
  .gt-note {
    color: var(--text-muted);
    font-size: 10px;
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px solid var(--border);
    line-height: 1.4;
  }
  .metric-label {
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 400;
  }
  .metric-value {
    font-weight: 600;
  }
  .metric-pill.green .metric-value { color: var(--green); }
  .metric-pill.red .metric-value { color: var(--red); }

  /* ─── Range dropdown ─── */
  .range-dropdown-wrap {
    position: relative;
    display: inline-flex;
  }
  .range-pill {
    cursor: pointer;
    user-select: none;
  }
  .range-caret {
    display: inline-block;
    margin-left: 2px;
    vertical-align: middle;
    transition: transform var(--t-fast);
  }
  .range-caret.open {
    transform: rotate(180deg);
  }
  .range-menu {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    min-width: 130px;
    background: #111118;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 4px 0;
    z-index: 200;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .range-opt {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 5px 10px;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 11px;
    font-family: inherit;
    text-align: left;
    transition: background var(--t-fast), color var(--t-fast);
  }
  .range-opt:hover {
    background: rgba(255,255,255,0.04);
    color: var(--text);
  }
  .range-opt.active {
    color: var(--accent);
  }
  .range-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    border: 1px solid var(--border);
    flex-shrink: 0;
  }
  .range-opt.active .range-dot {
    background: var(--accent);
    border-color: var(--accent);
  }

  .error-banner {
    padding: 6px 12px;
    border-radius: var(--radius-md);
    font-size: 12px;
    background: var(--red-bg);
    color: var(--red);
    border: 1px solid rgba(255,107,107,0.2);
  }

  /* ─── Mobile tabs ─── */
  .mobile-tabs {
    display: none;
    gap: 2px;
    margin-bottom: 8px;
    background: var(--bg);
    border-radius: var(--radius-md);
    padding: 3px;
    border: 1px solid var(--border);
  }
  .m-tab {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 8px 4px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }
  .m-tab.active {
    background: var(--surface);
    color: var(--text);
  }

  /* ─── Grid layout ─── */
  .dashboard-grid {
    display: flex;
    gap: 0;
    height: calc(100vh - 140px);
    height: calc(100dvh - 140px);
    min-height: 400px;
  }
  .dashboard-grid.is-dragging {
    user-select: none;
  }
  .dashboard-grid.drag-col {
    cursor: col-resize;
  }
  .dashboard-grid.drag-row {
    cursor: row-resize;
  }

  .splitter {
    width: 10px;
    cursor: col-resize;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .splitter::after {
    content: '';
    width: 3px;
    height: 32px;
    border-radius: 2px;
    background: var(--border);
    transition: background var(--t-fast);
  }
  .splitter:hover::after,
  .splitter.active::after {
    background: var(--accent);
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    gap: 8px;
  }
  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  /* ─── Expanded detail (hide symbol panel + splitter) ─── */
  .expanded-hidden {
    display: none !important;
  }

  /* ─── Symbol table ─── */
  .symbols-panel { flex-shrink: 0; }

  .search-wrap {
    position: relative;
    flex: 1;
  }
  .search-icon {
    position: absolute;
    left: 8px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-dim);
    pointer-events: none;
  }
  .search-input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 8px 6px 28px;
    border-radius: var(--radius-md);
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    transition: border-color var(--t-fast);
  }
  .search-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .sym-count {
    font-size: 11px;
    color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
    flex-shrink: 0;
  }
  .sym-table-wrap {
    overflow-y: auto;
    flex: 1;
  }
  .sym-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .sym-table th {
    text-align: left;
    padding: 6px 10px;
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--surface);
    z-index: 1;
  }
  .sym-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .sym-table tr {
    cursor: pointer;
    transition: background var(--t-fast);
  }
  .sym-table tr:hover {
    background: rgba(77,171,247,0.04);
  }
  .sym-table tr.is-focus {
    background: var(--accent-bg);
  }
  .sym-table tr.row-long {
    border-left: 2px solid var(--green);
  }
  .sym-table tr.row-short {
    border-left: 2px solid var(--red);
  }

  /* Price flash animation styles live in Lit web component: wc/mid-price.ts */

  .sym-name {
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.02em;
  }

  .num {
    font-variant-numeric: tabular-nums;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text);
  }

  .sig-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .sig-badge.buy {
    background: var(--green-bg);
    color: var(--green);
  }
  .sig-badge.sell {
    background: var(--red-bg);
    color: var(--red);
  }
  .sig-badge.none {
    color: var(--text-dim);
  }

  .pos-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .pos-badge.long {
    color: var(--green);
    background: var(--green-bg);
  }
  .pos-badge.short {
    color: var(--red);
    background: var(--red-bg);
  }
  .flat-label {
    color: var(--text-dim);
    font-size: 11px;
  }
  .pos-pnl {
    display: inline;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    margin-left: 4px;
    letter-spacing: 0.01em;
  }
  .sig-age {
    display: inline;
    font-size: 9px;
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 3px;
  }
  .pos-pnl.green { color: var(--green); }
  .pos-pnl.red   { color: var(--red); }
  .pos-age {
    display: inline;
    font-size: 9px;
    color: var(--text);
    margin-left: 3px;
  }

  /* ─── Detail panel ─── */
  .detail-panel { overflow-y: auto; flex: 1; min-width: 0; }
  .detail-tabs {
    display: flex;
    gap: 2px;
    background: var(--bg);
    border-radius: 5px;
    padding: 2px;
    flex-shrink: 0;
  }

  /* ─── Interval selector ─── */
  .iv-bar {
    display: flex;
    gap: 1px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-subtle);
    flex-shrink: 0;
    background: var(--surface);
  }
  .iv-tab {
    background: transparent;
    border: none;
    color: var(--text-dim);
    padding: 3px 7px;
    cursor: pointer;
    font-size: 10px;
    font-weight: 600;
    border-radius: 3px;
    letter-spacing: 0.05em;
    font-family: 'IBM Plex Mono', monospace;
    transition: all var(--t-fast);
  }
  .iv-tab:hover { color: var(--text); background: rgba(255,255,255,0.04); }
  .iv-tab.is-on { color: var(--accent); background: var(--accent-bg); }
  .iv-sep {
    width: 1px;
    height: 14px;
    background: var(--border);
    margin: 0 4px;
    align-self: center;
    flex-shrink: 0;
  }

  /* ─── Candle chart container ─── */
  .chart-wrap {
    flex-shrink: 0;
    overflow: hidden;
  }
  .chart-splitter {
    height: 8px;
    cursor: row-resize;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    border-bottom: 1px solid var(--border-subtle);
  }
  .chart-splitter::after {
    content: '';
    width: 32px;
    height: 3px;
    border-radius: 2px;
    background: var(--border);
    transition: background var(--t-fast);
  }
  .chart-splitter:hover::after,
  .chart-splitter.active::after {
    background: var(--accent);
  }
  @media (max-width: 768px) {
    .chart-wrap { height: 200px !important; }
    .chart-splitter { display: none; }
  }
  .detail-header {
    gap: 8px;
  }
  .focus-sym {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex: 1;
  }
  .focus-sym h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.01em;
  }
  .expand-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    cursor: pointer;
    flex-shrink: 0;
  }
  .expand-btn:hover {
    background: var(--surface-hover);
    color: var(--text);
  }
  .close-focus {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    cursor: pointer;
    flex-shrink: 0;
  }
  .close-focus:hover {
    background: var(--surface-hover);
    color: var(--text);
  }

  .kv-section {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-subtle);
    animation: slideUp 0.2s ease;
  }
  .kv-section h4 {
    margin: 0 0 8px;
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
  }
  .kv {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 12px;
  }
  .kv .k { color: var(--text-muted); }
  .kv .v { font-weight: 500; }

  .empty-focus {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 8px;
    color: var(--text-dim);
  }
  .empty-focus p {
    font-size: 13px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    gap: 2px;
  }
  .empty-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
  }
  .empty-sub {
    font-size: 11px;
    color: var(--text-dim);
  }

  /* ─── Feed content ─── */
  .feed-tabs {
    display: flex;
    gap: 2px;
    background: var(--bg);
    border-radius: 5px;
    padding: 2px;
  }
  .tab {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 4px 10px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.04em;
    font-family: 'IBM Plex Mono', monospace;
    transition: all var(--t-fast);
  }
  .tab:hover {
    color: var(--text);
  }
  .tab.is-on {
    color: var(--accent);
    background: var(--accent-bg);
  }
  .feed-content {
    overflow-y: auto;
    flex: 1;
  }
  .feed-item {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
    transition: background var(--t-fast);
  }
  .feed-item:hover {
    background: rgba(255,255,255,0.015);
  }
  .feed-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .feed-l { font-weight: 500; display: flex; gap: 6px; align-items: center; }
  .feed-sym {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 11px;
  }
  .feed-action { font-size: 11px; }
  .feed-type { font-size: 11px; color: var(--text-muted); }
  .feed-r {
    color: var(--text-dim);
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .feed-sub {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 3px;
  }
  .green { color: var(--green); }
  .red { color: var(--red); }
  .yellow { color: var(--yellow); }

  /* ─── Mobile ─── */
  @media (max-width: 768px) {
    .topbar-row {
      flex-wrap: wrap;
    }
    .metrics-bar {
      gap: 4px;
    }
    .metric-pill {
      font-size: 10px;
      padding: 2px 7px;
    }

    .mobile-tabs {
      display: flex;
    }

    .dashboard-grid {
      flex-direction: column;
      height: calc(100dvh - 240px);
    }

    .splitter { display: none; }
    .expand-btn { display: none; }

    .symbols-panel {
      width: auto !important;
      min-width: 0 !important;
    }

    .panel {
      display: none;
    }
    .panel.mobile-visible {
      display: flex;
    }

    /* Column widths for ~375px: SYM 24 / MID 12 / SIG 22 / POS 42 */
    .sym-table {
      table-layout: fixed;
    }
    .sym-table th:nth-child(1), .sym-table td:nth-child(1) { width: 15%; }
    .sym-table th:nth-child(2), .sym-table td:nth-child(2) { width: 20%; }
    .sym-table th:nth-child(3), .sym-table td:nth-child(3) { width: 25%; }
    .sym-table th:nth-child(4), .sym-table td:nth-child(4) { width: 40%; }

    .col-mid {
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sym-table td {
      padding: 10px 6px;
    }
    .sym-table tr {
      min-height: 44px;
    }
  }

  @media (max-width: 480px) {
    .mode-btn {
      padding: 5px 10px;
      font-size: 10px;
    }
    .metric-pill {
      font-size: 9px;
      padding: 2px 5px;
    }
    .metric-label {
      display: none;
    }
  }
</style>
