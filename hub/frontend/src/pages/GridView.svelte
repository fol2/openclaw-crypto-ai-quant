<script lang="ts">
  import { getSnapshot, getMids, getTrendCloses } from '../lib/api';
  import { hubWs } from '../lib/ws';
  import { CANDIDATE_FAMILY_ORDER, getModeLabel, LIVE_MODE } from '../lib/mode-labels';

  let mode = $state('_pending_');     // set after cookie helpers
  let gridSize = $state(3);          // set after cookie helpers
  let symbols: any[] = $state([]);
  let mids: Record<string, number> = $state({});
  let trendCloses: Record<string, number[]> = $state({});
  let loading = $state(true);
  let filter = $state('');
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let trendTimer: ReturnType<typeof setInterval> | null = null;
  let midsSeeded = false;

  async function refreshTrend() {
    try {
      const res = await getTrendCloses(trendInterval, trendBars);
      trendCloses = res.closes || {};
    } catch {}
  }

  async function refresh() {
    try {
      const snap = await getSnapshot(mode);
      symbols = snap.symbols || [];
      // Seed mids from REST only on first load; WS takes over after that.
      if (!midsSeeded) {
        const m = await getMids();
        mids = m.mids || {};
        midsSeeded = true;
      }
      await refreshTrend();
    } catch {}
    loading = false;
  }

  let filteredSymbols = $derived.by(() => {
    const q = filter.trim().toUpperCase();
    let syms = symbols;
    if (q) syms = syms.filter((s: any) => String(s.symbol).includes(q));
    return syms;
  });

  // ── Cookie helpers ───────────────────────────────────────────────
  function getCookie(key: string, fallback: string): string {
    const m = document.cookie.match(new RegExp(`(?:^|; )${key}=([^;]*)`));
    return m ? decodeURIComponent(m[1]) : fallback;
  }
  function setCookie(key: string, val: string) {
    document.cookie = `${key}=${encodeURIComponent(val)};path=/;max-age=31536000;SameSite=Lax`;
  }
  function getNumCookie(key: string, fallback: number): number {
    const v = Number(getCookie(key, String(fallback)));
    return Number.isFinite(v) ? v : fallback;
  }

  // ── Persisted view settings ─────────────────────────────────────
  mode = getCookie('gridMode', 'paper1');
  gridSize = getNumCookie('gridSize', 3);

  // ── Trend debug controls (persisted via cookies) ───────────────
  let debugTrend = $state(getCookie('trendDebug', '') === '1' || new URLSearchParams(window.location.search).has('trend_debug'));
  let trendFullPct = $state(getNumCookie('trendFullPct', 1.0));
  let trendCurve   = $state(getNumCookie('trendCurve', 1.0));
  let trendWindow  = $state(getNumCookie('trendWindow', 0));
  let trendInterval = $state(getCookie('trendInterval', '5m'));
  let trendBars    = $state(getNumCookie('trendBars', 60));
  let trendStrengthMul = $state(getNumCookie('trendStrengthMul', 1.0));
  let showCardDbg = $state(getCookie('showCardDbg', '1') === '1');

  // Persist debug settings to cookies on change
  $effect(() => {
    setCookie('trendDebug', debugTrend ? '1' : '0');
    setCookie('trendFullPct', String(trendFullPct));
    setCookie('trendCurve', String(trendCurve));
    setCookie('trendWindow', String(trendWindow));
    setCookie('trendInterval', trendInterval);
    setCookie('trendBars', String(trendBars));
    setCookie('trendStrengthMul', String(trendStrengthMul));
    setCookie('showCardDbg', showCardDbg ? '1' : '0');
    setCookie('gridMode', mode);
    setCookie('gridSize', String(gridSize));
  });

  // ── Trend computation (OLS linear regression) ───────────────────
  function linregTrend(pts: number[], win: number): number {
    if (!pts || pts.length < 2) return 0;
    const src = win > 0 && pts.length > win ? pts.slice(-win) : pts;
    const n = src.length;
    if (n < 2) return 0;
    let sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (let i = 0; i < n; i++) {
      sx += i; sy += src[i]; sxy += i * src[i]; sx2 += i * i;
    }
    const meanY = sy / n;
    if (meanY <= 0) return 0;
    const slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx);
    return (slope * (n - 1)) / meanY;   // fractional change over window
  }

  function trendStrength(pts: number[]): number {
    const raw = linregTrend(pts, trendWindow);
    const scaled = raw / (trendFullPct / 100);
    const clamped = Math.max(-1, Math.min(1, scaled));
    const abs = Math.abs(clamped);
    const curved = Math.pow(abs, trendCurve);
    return clamped >= 0 ? curved : -curved;
  }

  // Build inline CSS custom properties for OKLCH trend color.
  function trendVars(strength: number): string {
    const abs = Math.abs(strength);
    if (abs < 0.05) return '';
    const mul = trendStrengthMul;
    const h = strength > 0 ? 145 : 25;
    const l  = 13 + abs * 8 * mul;          // 13→21%  (surface ≈ 13%)
    const c  = abs * 0.045 * mul;           // 0→0.045
    const gl = l + 8 * mul;                 // glow is brighter
    const gc = abs * 0.08 * mul;
    const ga = abs * 0.12 * mul;
    const ba = 0.15 + abs * 0.25 * mul;     // border alpha
    return `--trend-bg:oklch(${l.toFixed(1)}% ${c.toFixed(3)} ${h});--trend-glow:oklch(${gl.toFixed(1)}% ${gc.toFixed(3)} ${h} / ${ga.toFixed(2)});--trend-border:oklch(50% ${(abs * 0.06 * mul).toFixed(3)} ${h} / ${ba.toFixed(2)})`;
  }

  function adaptiveDecimals(n: any): number {
    const v = Number(n);
    if (!Number.isFinite(v) || v <= 0) return 6;
    if (v >= 1000) return 2;
    if (v >= 1) return 4;
    return 6;
  }

  $effect(() => {
    refresh();
    pollTimer = setInterval(refresh, 10000);
    // Refresh trend data every 30s (candle DB updates less frequently than WS)
    trendTimer = setInterval(refreshTrend, 30000);
    hubWs.connect();
    return () => {
      if (pollTimer) clearInterval(pollTimer);
      if (trendTimer) clearInterval(trendTimer);
    };
  });

  // Real-time mid-price updates via WS (mids + bbo)
  $effect(() => {
    function finitePositive(v: unknown): number | null {
      const n = Number(v);
      return Number.isFinite(n) && n > 0 ? n : null;
    }
    function quoteMid(quote: any): number | null {
      const explicit = finitePositive(quote?.mid);
      if (explicit != null) return explicit;
      const bid = finitePositive(quote?.bid);
      const ask = finitePositive(quote?.ask);
      if (bid == null || ask == null) return null;
      return (bid + ask) / 2;
    }

    const handler = (data: any) => {
      const midsRaw = data?.mids;
      if (midsRaw && typeof midsRaw === 'object') {
        for (const [sym, raw] of Object.entries(midsRaw)) {
          const px = finitePositive(raw);
          if (px != null) mids[String(sym).toUpperCase()] = px;
        }
      }
      const bboRaw = data?.bbo;
      if (bboRaw && typeof bboRaw === 'object') {
        for (const [sym, raw] of Object.entries(bboRaw)) {
          const px = quoteMid(raw);
          if (px != null) mids[String(sym).toUpperCase()] = px;
        }
      }
    };

    hubWs.subscribe('mids', handler);
    hubWs.subscribe('bbo', handler);
    return () => {
      hubWs.unsubscribe('mids', handler);
      hubWs.unsubscribe('bbo', handler);
    };
  });
</script>

<div class="page">
  <div class="page-header">
    <h1>Grid View</h1>
    <div class="controls">
      <select bind:value={mode} onchange={refresh}>
        <optgroup label="Live Engine">
          <option value={LIVE_MODE}>{getModeLabel(LIVE_MODE)}</option>
        </optgroup>
        <optgroup label="Candidate Family">
          {#each CANDIDATE_FAMILY_ORDER as m}
            <option value={m}>{getModeLabel(m)}</option>
          {/each}
        </optgroup>
      </select>
      <div class="search-wrap">
        <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input type="text" bind:value={filter} placeholder="Filter..." />
      </div>
      <select bind:value={gridSize}>
        <option value={2}>2x2</option>
        <option value={3}>3x3</option>
        <option value={4}>4x4</option>
        <option value={5}>5x5</option>
      </select>
      <button class="debug-toggle" class:active={debugTrend} onclick={() => debugTrend = !debugTrend} title="Trend debug">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
      </button>
    </div>
  </div>

  {#if debugTrend}
    <div class="debug-bar">
      <label>
        <span class="db-label" title="Candle interval for trend calculation (e.g. 5m = 5-minute candles)">Interval</span>
        <select title="Candle interval for trend calculation (e.g. 5m = 5-minute candles)" bind:value={trendInterval} onchange={refreshTrend}>
          <option value="1m">1m</option>
          <option value="3m">3m</option>
          <option value="5m">5m</option>
          <option value="15m">15m</option>
          <option value="30m">30m</option>
          <option value="1h">1h</option>
        </select>
      </label>
      <label>
        <span class="db-label" title="Number of candle bars in the trend regression. More bars = longer lookback">Bars</span>
        <input title="Number of candle bars in the trend regression. More bars = longer lookback" type="range" min="10" max="200" step="10" bind:value={trendBars} onchange={refreshTrend} />
        <span class="db-val">{trendBars}</span>
      </label>
      <label>
        <span class="db-label" title="Price change (%) that maps to full intensity. Lower = more sensitive">Full&nbsp;%</span>
        <input title="Price change (%) that maps to full intensity. Lower = more sensitive" type="range" min="0.1" max="5" step="0.1" bind:value={trendFullPct} />
        <span class="db-val">{trendFullPct.toFixed(1)}</span>
      </label>
      <label>
        <span class="db-label" title="Exponent curve. <1 = compress (subtle diffs visible), >1 = amplify (only strong trends)">Curve</span>
        <input title="Exponent curve. <1 = compress (subtle diffs visible), >1 = amplify (only strong trends)" type="range" min="0.2" max="3" step="0.1" bind:value={trendCurve} />
        <span class="db-val">{trendCurve.toFixed(1)}</span>
      </label>
      <label>
        <span class="db-label" title="Use only the last N bars for regression. 0 = use all bars">Window</span>
        <select title="Use only the last N bars for regression. 0 = use all bars" bind:value={trendWindow}>
          <option value={0}>All</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </label>
      <label>
        <span class="db-label" title="OKLCH color intensity multiplier. Higher = more vivid background">Strength</span>
        <input title="OKLCH color intensity multiplier. Higher = more vivid background" type="range" min="0.5" max="5" step="0.1" bind:value={trendStrengthMul} />
        <span class="db-val">{trendStrengthMul.toFixed(1)}x</span>
      </label>
      <label class="db-checkbox">
        <input title="Show per-card debug text (data points, raw %, trend strength)" type="checkbox" bind:checked={showCardDbg} />
        <span class="db-label" title="Show per-card debug text (data points, raw %, trend strength)">Card text</span>
      </label>
    </div>
  {/if}

  {#if loading}
    <div class="empty-state">Loading...</div>
  {:else}
    <div class="symbol-grid" class:debug-active={debugTrend} style="grid-template-columns: repeat({gridSize}, 1fr);">
      {#each filteredSymbols as s (s.symbol)}
        {@const hist = trendCloses[s.symbol] || []}
        {@const trend = trendStrength(hist)}
        <div class="grid-cell" class:has-trend={Math.abs(trend) >= 0.05} style={trendVars(trend)}>
          {#if debugTrend && showCardDbg}
            <span class="trend-dbg">{hist.length}pt {(linregTrend(hist, trendWindow) * 100).toFixed(3)}%→{trend.toFixed(2)}</span>
          {/if}
          <div class="cell-header">
            <span class="cell-symbol">{s.symbol}</span>
            {#if s.position_side && s.position_side !== 'NONE'}
              <span class="pos-badge" class:long={s.position_side === 'LONG'} class:short={s.position_side === 'SHORT'}>
                {s.position_side}
              </span>
            {/if}
          </div>
          <div class="cell-price">
            <mid-price
              symbol={s.symbol}
              value={(mids[s.symbol] ?? s.mid) != null ? String(mids[s.symbol] ?? s.mid) : ''}
              decimals={adaptiveDecimals(mids[s.symbol] ?? s.mid)}
              tone="grid"
            ></mid-price>
          </div>
          {#if s.signal}
            <div class="cell-signal">
              <span class="signal-badge">{s.signal}</span>
            </div>
          {/if}
          <div class="cell-sparkline">
            <sparkline-chart
              points={JSON.stringify(s.recent_mids || [])}
              width="160"
              height="40"
              color={s.position_side === 'LONG' ? '#51cf66' : s.position_side === 'SHORT' ? '#ff6b6b' : '#3d4f63'}
            ></sparkline-chart>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .page { max-width: 1600px; animation: slideUp 0.3s ease; }

  .page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--sp-md);
    gap: var(--sp-md);
  }
  .page-header h1 {
    font-size: 20px; font-weight: 700; margin: 0; letter-spacing: -0.02em;
  }

  .controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .controls select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .search-wrap {
    position: relative;
  }
  .search-icon {
    position: absolute;
    left: 8px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-dim);
    pointer-events: none;
  }
  .controls input {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px 8px 28px;
    border-radius: var(--radius-md);
    font-size: 13px;
    width: 140px;
  }
  .controls input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .symbol-grid {
    display: grid;
    gap: 10px;
  }

  .grid-cell {
    background-color: var(--trend-bg, var(--surface));
    border: 1px solid var(--trend-border, var(--border));
    border-radius: var(--radius-lg);
    padding: 14px;
    min-height: 100px;
    transition: background-color 1.5s ease, border-color 1.5s ease;
  }
  .grid-cell:hover {
    border-color: var(--text-dim);
    transition: border-color 0.15s ease;
  }
  .grid-cell.has-trend {
    animation: trendBreath 4s ease-in-out infinite;
  }
  @keyframes trendBreath {
    0%, 100% { box-shadow: inset 0 0 20px var(--trend-glow, transparent); }
    50%      { box-shadow: inset 0 0 40px var(--trend-glow, transparent); }
  }

  .cell-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  .cell-symbol {
    font-weight: 700;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
  }

  .cell-price {
    font-size: 18px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
  }

  .pos-badge {
    font-size: 9px;
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-weight: 700;
    letter-spacing: 0.06em;
  }
  .pos-badge.long { background: var(--green-bg); color: var(--green); }
  .pos-badge.short { background: var(--red-bg); color: var(--red); }

  .signal-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    background: var(--yellow-bg);
    color: var(--yellow);
    font-weight: 600;
  }

  .cell-sparkline { margin-top: 6px; }

  /* ── Debug toolbar ─────────────────────────────────────────────── */
  .debug-toggle {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 6px 8px;
    border-radius: var(--radius-md);
    cursor: pointer;
    line-height: 0;
  }
  .debug-toggle.active {
    border-color: var(--accent);
    color: var(--accent);
  }

  .debug-bar {
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 8px 14px;
    margin-bottom: var(--sp-sm);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-dim);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .debug-bar::-webkit-scrollbar { display: none; }
  .debug-bar label {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .db-label {
    color: var(--text-dim);
    min-width: 42px;
  }
  .db-val {
    color: var(--accent);
    min-width: 28px;
    text-align: right;
  }
  .db-checkbox input[type="checkbox"] {
    accent-color: var(--accent);
    margin: 0;
  }
  .debug-bar input[type="range"] {
    width: 100px;
    accent-color: var(--accent);
    height: 4px;
  }
  .debug-bar select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-size: 11px;
    font-family: inherit;
  }

  .trend-dbg {
    position: absolute;
    top: 3px;
    right: 5px;
    font-size: 8px;
    font-family: 'IBM Plex Mono', monospace;
    color: #fff;
  }

  .debug-active .grid-cell {
    position: relative;
    transition-duration: 0.15s;
  }

  .empty-state {
    color: var(--text-dim);
    padding: 40px 0;
    text-align: center;
    font-size: 13px;
  }

  @media (max-width: 768px) {
    .page-header {
      flex-direction: column;
      align-items: stretch;
    }
    .controls {
      flex-wrap: wrap;
    }
    .symbol-grid {
      grid-template-columns: repeat(2, 1fr) !important;
      gap: 8px;
    }
    .cell-price { font-size: 15px; }
    .grid-cell { padding: 10px; }
  }

  @media (max-width: 480px) {
    .symbol-grid {
      grid-template-columns: 1fr !important;
    }
  }
</style>
