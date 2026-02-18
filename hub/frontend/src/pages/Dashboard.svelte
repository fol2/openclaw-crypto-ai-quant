<script lang="ts">
  import { appState } from '../lib/stores';
  import { getSnapshot, getCandles, getMarks } from '../lib/api';
  import { hubWs } from '../lib/ws';

  let snap: any = $state(null);
  let focusSym = $state('');
  let candles: any[] = $state([]);
  let marks: any = $state(null);
  let pollTimer: any = null;
  let error = $state('');

  // Formatting helpers
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

  // Poll every 5s
  $effect(() => {
    refresh();
    pollTimer = setInterval(refresh, 5000);
    hubWs.connect();
    return () => {
      clearInterval(pollTimer);
    };
  });

  // Derived values
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

<!-- Mode selector -->
<div class="topbar">
  <div class="mode-tabs">
    {#each ['live', 'paper1', 'paper2', 'paper3'] as m}
      <button
        class="mode-btn"
        class:active={appState.mode === m || (appState.mode === 'paper' && m === 'paper1')}
        onclick={() => setMode(m)}
      >{m.toUpperCase()}</button>
    {/each}
  </div>

  <div class="metrics-bar">
    <span class="pill" class:ok={health.ok} class:bad={!health.ok}>
      <span class="dot" class:green={health.ok} class:red={!health.ok}></span>
      {health.ok ? 'ENGINE' : 'NO HB'}
    </span>
    {#if health.kill_mode && health.kill_mode !== 'off'}
      <span class="pill paused">KILL: {health.kill_mode}</span>
    {/if}
    {#if health.regime_gate !== undefined && health.regime_gate !== null}
      <span class="pill" class:ok={health.regime_gate} class:bad={!health.regime_gate}>
        GATE {health.regime_gate ? 'ON' : 'OFF'}
      </span>
    {/if}
    <span class="pill">BAL ${fmtNum(balances.realised_usd)}</span>
    <span class="pill">EQ ${fmtNum(balances.equity_est_usd)}</span>
    <span class="pill {pnlClass(daily.pnl_usd)}">PnL ${fmtNum(daily.pnl_usd)}</span>
    <span class="pill">DD {fmtNum(daily.drawdown_pct, 1)}%</span>
    <span class="pill">POS {openPositions.length}</span>
  </div>

  {#if error}
    <span class="pill bad">{error}</span>
  {/if}
</div>

<div class="dashboard-grid">
  <!-- Symbol table -->
  <div class="panel symbols-panel">
    <div class="panel-header">
      <input
        type="text"
        class="search-input"
        placeholder="Search symbols..."
        bind:value={appState.search}
      />
      <span class="sym-count">{filteredSymbols.length} symbols</span>
    </div>
    <div class="sym-table-wrap">
      <table class="sym-table">
        <thead>
          <tr>
            <th>SYM</th>
            <th>MID</th>
            <th>SIGNAL</th>
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
              <td><span class="badge">{s.symbol}</span></td>
              <td class="num">{s.mid != null ? fmtNum(s.mid, 6) : '\u2014'}</td>
              <td>
                <span class="badge" class:sig-buy={s.last_signal?.signal === 'BUY'} class:sig-sell={s.last_signal?.signal === 'SELL'}>
                  {s.last_signal?.signal || '\u2014'}
                </span>
              </td>
              <td>
                {#if s.position}
                  <span class="badge" class:pos-long={s.position.type === 'LONG'} class:pos-short={s.position.type === 'SHORT'}>
                    {s.position.type} {fmtNum(s.position.size, 4)}
                  </span>
                {:else}
                  <span class="muted">flat</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Focus detail panel -->
  <div class="panel detail-panel">
    {#if focusSym}
      <div class="panel-header">
        <h3>Focus: {focusSym}</h3>
        {#each symbols.filter((s: any) => s.symbol === focusSym).slice(0, 1) as sym}
          <span class="mid-price">MID {sym.mid != null ? fmtNum(sym.mid, 6) : '\u2014'}</span>
        {/each}
      </div>

      <!-- Position info -->
      {#if marks?.position}
        {@const p = marks.position}
        <div class="kv-section">
          <h4>Position</h4>
          <div class="kv"><span class="k">type</span><span class="v">{p.pos_type || p.type}</span></div>
          <div class="kv"><span class="k">size</span><span class="v">{fmtNum(p.size, 6)}</span></div>
          <div class="kv"><span class="k">entry</span><span class="v">{fmtNum(p.entry_price, 6)}</span></div>
          <div class="kv"><span class="k">uPnL</span><span class="v {pnlClass(p.unreal_pnl_est)}">{fmtNum(p.unreal_pnl_est)}</span></div>
          <div class="kv"><span class="k">leverage</span><span class="v">{fmtNum(p.leverage, 1)}x</span></div>
        </div>
        {#if marks?.entries?.length}
          <div class="kv-section">
            <h4>Entries</h4>
            {#each marks.entries as e}
              <div class="kv">
                <span class="k">{e.action}</span>
                <span class="v">@ {fmtNum(e.price, 6)} size {fmtNum(e.size, 4)}</span>
              </div>
            {/each}
          </div>
        {/if}
      {:else}
        <div class="muted" style="padding:12px">Flat (no position)</div>
      {/if}

      <!-- Candle summary -->
      {#if candles.length > 0}
        <div class="kv-section">
          <h4>Candles ({candles.length})</h4>
          <div class="kv"><span class="k">last close</span><span class="v">{fmtNum(candles[candles.length - 1]?.c, 6)}</span></div>
          <div class="kv"><span class="k">last high</span><span class="v">{fmtNum(candles[candles.length - 1]?.h, 6)}</span></div>
          <div class="kv"><span class="k">last low</span><span class="v">{fmtNum(candles[candles.length - 1]?.l, 6)}</span></div>
        </div>
      {/if}

    {:else}
      <div class="empty-focus">
        <p>Click a symbol to see details</p>
      </div>
    {/if}
  </div>

  <!-- Activity feeds -->
  <div class="panel feed-panel">
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
              <span class="feed-l">{t.symbol} {t.action} {t.type}</span>
              <span class="feed-r">{t.timestamp?.slice(11, 19) || ''}</span>
            </div>
            <div class="feed-sub">
              px <span class="green">{fmtNum(t.price, 6)}</span> size {fmtNum(t.size, 6)}
              pnl <span class="{pnlClass(t.pnl)}">{t.pnl != null ? fmtNum(t.pnl) : '\u2014'}</span>
            </div>
          </div>
        {/each}
      {:else if appState.feed === 'oms'}
        {#each (recent.oms_intents || []).slice(0, 25) as i}
          <div class="feed-item">
            <div class="feed-row">
              <span class="feed-l">{i.symbol} {i.action} {i.side}
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
  .topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .mode-tabs {
    display: flex;
    gap: 4px;
  }
  .mode-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
  }
  .mode-btn.active {
    background: var(--accent);
    color: var(--bg);
    border-color: var(--accent);
  }
  .metrics-bar {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
  }
  .pill.ok { border-color: var(--green); }
  .pill.bad { border-color: var(--red); color: var(--red); }
  .pill.paused { border-color: var(--red); color: var(--red); }
  .dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
  }
  .dot.green { background: var(--green); }
  .dot.red { background: var(--red); }

  .dashboard-grid {
    display: grid;
    grid-template-columns: 320px 1fr 360px;
    grid-template-rows: 1fr;
    gap: 12px;
    height: calc(100vh - 110px);
    min-height: 400px;
  }

  .panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .panel-header h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  /* Symbol table */
  .symbols-panel { min-width: 0; }
  .search-input {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    width: 140px;
  }
  .sym-count {
    font-size: 11px;
    color: var(--text-muted);
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
    padding: 4px 8px;
    color: var(--text-muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg-card);
  }
  .sym-table td {
    padding: 4px 8px;
    border-bottom: 1px solid var(--border);
  }
  .sym-table tr { cursor: pointer; }
  .sym-table tr:hover { background: rgba(255,255,255,0.03); }
  .sym-table tr.is-focus { background: rgba(59, 130, 246, 0.12); }
  .sym-table tr.row-long { border-left: 2px solid var(--green); }
  .sym-table tr.row-short { border-left: 2px solid var(--red); }
  .num { font-variant-numeric: tabular-nums; text-align: right; }
  .badge {
    display: inline-block;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 11px;
  }
  .sig-buy { background: rgba(34, 197, 94, 0.2); color: var(--green); }
  .sig-sell { background: rgba(239, 68, 68, 0.2); color: var(--red); }
  .pos-long { color: var(--green); }
  .pos-short { color: var(--red); }
  .muted { color: var(--text-muted); }

  /* Detail panel */
  .detail-panel { overflow-y: auto; }
  .mid-price { font-size: 12px; color: var(--accent); }
  .kv-section {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .kv-section h4 {
    margin: 0 0 6px;
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .kv {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 12px;
  }
  .kv .k { color: var(--text-muted); }
  .kv .v { font-weight: 500; }
  .empty-focus {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
  }

  /* Feed panel */
  .feed-panel { min-width: 0; }
  .feed-tabs {
    display: flex;
    gap: 4px;
  }
  .tab {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 4px 8px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    border-bottom: 2px solid transparent;
  }
  .tab.is-on {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .feed-content {
    overflow-y: auto;
    flex: 1;
  }
  .feed-item {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .feed-row {
    display: flex;
    justify-content: space-between;
  }
  .feed-l { font-weight: 500; }
  .feed-r { color: var(--text-muted); font-size: 11px; }
  .feed-sub { color: var(--text-muted); font-size: 11px; margin-top: 2px; }
  .green { color: var(--green); }
  .red { color: var(--red); }
  .yellow { color: var(--yellow); }

  @media (max-width: 1000px) {
    .dashboard-grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto auto;
      height: auto;
    }
    .symbols-panel { max-height: 300px; }
    .detail-panel { max-height: 400px; }
    .feed-panel { max-height: 400px; }
  }
</style>
