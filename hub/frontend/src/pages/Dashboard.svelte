<script lang="ts">
  import { appState } from '../lib/stores.svelte';
  import { getSnapshot, getCandles, getMarks } from '../lib/api';
  import { hubWs } from '../lib/ws';

  let snap: any = $state(null);
  let focusSym = $state('');
  let candles: any[] = $state([]);
  let marks: any = $state(null);
  let pollTimer: any = null;
  let error = $state('');
  let mobileTab: 'symbols' | 'detail' | 'feed' = $state('symbols');

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
  function pnlClass(v: number | null | undefined): string {
    if (v === null || v === undefined) return '';
    return v >= 0 ? 'green' : 'red';
  }

  async function refresh() {
    try {
      appState.loading = true;
      const data = await getSnapshot(appState.mode);
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

  async function setFocus(sym: string) {
    focusSym = sym;
    appState.focus = sym;
    candles = [];
    marks = null;
    if (!sym) return;
    mobileTab = 'detail';
    try {
      const [c, m] = await Promise.all([
        getCandles(sym, undefined, 200),
        getMarks(sym, appState.mode),
      ]);
      candles = c.candles || [];
      marks = m;
    } catch { /* ignore */ }
  }

  function setMode(m: string) {
    appState.mode = m;
    focusSym = '';
    refresh();
  }

  function setFeed(f: string) {
    appState.feed = f;
  }

  $effect(() => {
    refresh();
    pollTimer = setInterval(refresh, 5000);
    hubWs.connect();
    return () => {
      clearInterval(pollTimer);
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
      <span class="metric-pill" class:gate-on={health.regime_gate} class:gate-off={!health.regime_gate}>
        GATE {health.regime_gate ? 'ON' : 'OFF'}
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
    <span class="metric-pill {pnlClass(daily.pnl_usd)}">
      <span class="metric-label">PnL</span>
      <span class="metric-value">${fmtNum(daily.pnl_usd)}</span>
    </span>
    <span class="metric-pill">
      <span class="metric-label">DD</span>
      <span class="metric-value">{fmtNum(daily.drawdown_pct, 1)}%</span>
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
  <button class="m-tab" class:active={mobileTab === 'detail'} onclick={() => mobileTab = 'detail'}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
    Detail
  </button>
  <button class="m-tab" class:active={mobileTab === 'feed'} onclick={() => mobileTab = 'feed'}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
    Feed
  </button>
</div>

<div class="dashboard-grid">
  <!-- Symbol table -->
  <div class="panel symbols-panel" class:mobile-visible={mobileTab === 'symbols'}>
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
                  value={s.mid != null ? String(s.mid) : ''}
                  decimals={6}
                ></mid-price>
              </td>
              <td>
                {#if s.last_signal?.signal === 'BUY'}
                  <span class="sig-badge buy">BUY</span>
                {:else if s.last_signal?.signal === 'SELL'}
                  <span class="sig-badge sell">SELL</span>
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

  <!-- Focus detail panel -->
  <div class="panel detail-panel" class:mobile-visible={mobileTab === 'detail'}>
    {#if focusSym}
      <div class="panel-header detail-header">
        <div class="focus-sym">
          <h3>{focusSym}</h3>
          {#each symbols.filter((s: any) => s.symbol === focusSym).slice(0, 1) as sym}
            <mid-price
              tone="accent"
              value={sym.mid != null ? String(sym.mid) : ''}
              decimals={6}
            ></mid-price>
          {/each}
        </div>
        <button class="close-focus" aria-label="Close" onclick={() => { focusSym = ''; mobileTab = 'symbols'; }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      {#if marks?.position}
        {@const p = marks.position}
        <div class="kv-section">
          <h4>Position</h4>
          <div class="kv"><span class="k">Type</span><span class="v">{p.pos_type || p.type}</span></div>
          <div class="kv"><span class="k">Size</span><span class="v mono">{fmtNum(p.size, 6)}</span></div>
          <div class="kv"><span class="k">Entry</span><span class="v mono">{fmtNum(p.entry_price, 6)}</span></div>
          <div class="kv"><span class="k">uPnL</span><span class="v {pnlClass(p.unreal_pnl_est)}">{fmtNum(p.unreal_pnl_est)}</span></div>
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

      {#if candles.length > 0}
        <div class="kv-section">
          <h4>Candles ({candles.length})</h4>
          <div class="kv"><span class="k">Last close</span><span class="v mono">{fmtNum(candles[candles.length - 1]?.c, 6)}</span></div>
          <div class="kv"><span class="k">Last high</span><span class="v mono">{fmtNum(candles[candles.length - 1]?.h, 6)}</span></div>
          <div class="kv"><span class="k">Last low</span><span class="v mono">{fmtNum(candles[candles.length - 1]?.l, 6)}</span></div>
        </div>
      {/if}

    {:else}
      <div class="empty-focus">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" stroke-width="1"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
        <p>Select a symbol</p>
      </div>
    {/if}
  </div>

  <!-- Activity feeds -->
  <div class="panel feed-panel" class:mobile-visible={mobileTab === 'feed'}>
    <div class="panel-header">
      <div class="feed-tabs">
        {#each ['trades', 'oms', 'audit'] as f}
          <button
            class="tab"
            class:is-on={appState.feed === f}
            onclick={() => setFeed(f)}
          >{f.toUpperCase()}</button>
        {/each}
      </div>
    </div>
    <div class="feed-content">
      {#if appState.feed === 'trades'}
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
      {:else if appState.feed === 'oms'}
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
      {:else if appState.feed === 'audit'}
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
    display: grid;
    grid-template-columns: 300px 1fr 340px;
    grid-template-rows: 1fr;
    gap: 10px;
    height: calc(100vh - 140px);
    height: calc(100dvh - 140px);
    min-height: 400px;
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

  /* ─── Symbol table ─── */
  .symbols-panel { min-width: 0; }

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
    display: block;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    margin-top: 2px;
    letter-spacing: 0.01em;
  }
  .pos-pnl.green { color: var(--green); }
  .pos-pnl.red   { color: var(--red); }

  /* ─── Detail panel ─── */
  .detail-panel { overflow-y: auto; }
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

  /* ─── Feed panel ─── */
  .feed-panel { min-width: 0; }
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

  /* ─── Tablet ─── */
  @media (max-width: 1200px) {
    .dashboard-grid {
      grid-template-columns: 260px 1fr 300px;
    }
  }

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
      grid-template-columns: 1fr;
      grid-template-rows: 1fr;
      height: calc(100dvh - 240px);
    }

    .panel {
      display: none;
    }
    .panel.mobile-visible {
      display: flex;
    }

    /* Column widths for ~375px: SYM/MID/SIG/POS — POS wider for PnL sub-line */
    .sym-table {
      table-layout: fixed;
    }
    .sym-table th:nth-child(1), .sym-table td:nth-child(1) { width: 28%; }
    .sym-table th:nth-child(2), .sym-table td:nth-child(2) { width: 26%; }
    .sym-table th:nth-child(3), .sym-table td:nth-child(3) { width: 14%; }
    .sym-table th:nth-child(4), .sym-table td:nth-child(4) { width: 32%; }

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
