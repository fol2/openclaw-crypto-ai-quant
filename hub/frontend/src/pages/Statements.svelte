<script lang="ts">
  import { appState } from '../lib/stores.svelte';
  import { getJourneys, getTrades, getConfigHistory, getConfigDiffPrivileged, getConfigFiles } from '../lib/api';
  import { CANDIDATE_FAMILY_ORDER, getModeLabel, LIVE_MODE } from '../lib/mode-labels';

  type ConfigFileEntry = {
    variant: string;
    filename?: string;
    exists?: boolean;
    modified?: string;
    size?: number;
  };

  type ConfigHistoryEntry = {
    filename: string;
    modified?: string | null;
    size?: number | null;
  };

  type CandidateMode = (typeof CANDIDATE_FAMILY_ORDER)[number];

  function isCandidateMode(value: string): value is CandidateMode {
    return (CANDIDATE_FAMILY_ORDER as readonly string[]).includes(value);
  }

  function normaliseMode(value: string | null | undefined): string {
    if (value === LIVE_MODE) return LIVE_MODE;
    if (value && isCandidateMode(value)) return value;
    return CANDIDATE_FAMILY_ORDER[0];
  }

  // ── UI state ─────────────────────────────────────────────────────
  let tab: 'transactions' | 'config' = $state('transactions');
  let txView: 'journeys' | 'trades' = $state('journeys');
  let mode = $state(normaliseMode(appState.mode));

  // ── Journey state ────────────────────────────────────────────────
  let journeys: any[] = $state([]);
  let journeyOffset = $state(0);
  let journeyHasMore = $state(true);
  let journeyLoading = $state(false);
  let journeySymFilter = $state('');

  // ── Flat trades state ────────────────────────────────────────────
  let trades: any[] = $state([]);
  let tradeOffset = $state(0);
  let tradeHasMore = $state(true);
  let tradeLoading = $state(false);
  let tradeSummary = $state({ total: 0, pnl: 0, fees: 0 });
  let filterSymbol = $state('');
  let filterAction = $state('');
  let filterFrom = $state('');
  let filterTo = $state('');

  // ── Config state ─────────────────────────────────────────────────
  let configFiles: ConfigFileEntry[] = $state([]);
  let selectedConfigFile = $state('main');
  let configHistory: ConfigHistoryEntry[] = $state([]);
  let configLoading = $state(false);
  let diffA = $state('');
  let diffB = $state('current');
  let diffResult: string[] = $state([]);
  let diffLoading = $state(false);

  const PAGE_SIZE = 50;
  const TRADE_PAGE_SIZE = 100;

  // ── Helpers ──────────────────────────────────────────────────────
  function fmtNum(v: number | null | undefined, dp = 2): string {
    if (v == null || !isFinite(v)) return '-';
    return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }

  function fmtDuration(ms: number): string {
    if (ms < 0) ms = 0;
    const mins = Math.floor(ms / 60_000);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    const rm = mins % 60;
    if (hrs < 24) return rm > 0 ? `${hrs}h ${rm}m` : `${hrs}h`;
    const days = Math.floor(hrs / 24);
    const rh = hrs % 24;
    return rh > 0 ? `${days}d ${rh}h` : `${days}d`;
  }

  function pnlClass(v: number | null | undefined): string {
    if (v == null) return '';
    return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : '';
  }

  function fmtTs(ts: string | null | undefined): string {
    if (!ts) return '-';
    return ts.replace('T', ' ').slice(0, 19);
  }

  function journeyDuration(j: any): string {
    const openTs = Date.parse((j.open_ts || '').replace(' ', 'T'));
    const closeTs = j.close_ts ? Date.parse((j.close_ts || '').replace(' ', 'T')) : Date.now();
    if (!isFinite(openTs)) return '-';
    return fmtDuration(closeTs - openTs);
  }

  // ── Mode switching ───────────────────────────────────────────────
  function setMode(m: string) {
    const nextMode = normaliseMode(m);
    mode = nextMode;
    appState.mode = nextMode;
    resetAll();
    loadCurrentView();
  }

  function resetConfigDiff() {
    diffA = '';
    diffB = 'current';
    diffResult = [];
  }

  function resetAll() {
    journeys = []; journeyOffset = 0; journeyHasMore = true;
    trades = []; tradeOffset = 0; tradeHasMore = true;
    tradeSummary = { total: 0, pnl: 0, fees: 0 };
    configHistory = [];
    resetConfigDiff();
  }

  // ── Data loading ─────────────────────────────────────────────────
  async function fetchJourneys(reset = false) {
    if (journeyLoading) return;
    journeyLoading = true;
    try {
      const off = reset ? 0 : journeyOffset;
      const sym = journeySymFilter.trim().toUpperCase() || undefined;
      const res = await getJourneys(mode, PAGE_SIZE, off, sym);
      const batch = res.journeys || [];
      if (reset) {
        journeys = batch;
        journeyOffset = batch.length;
      } else {
        journeys = [...journeys, ...batch];
        journeyOffset += batch.length;
      }
      journeyHasMore = batch.length >= PAGE_SIZE;
    } catch (e) { console.error('fetchJourneys:', e); }
    journeyLoading = false;
  }

  async function fetchTrades(reset = false) {
    if (tradeLoading) return;
    tradeLoading = true;
    try {
      const off = reset ? 0 : tradeOffset;
      const sym = filterSymbol.trim().toUpperCase() || undefined;
      const act = filterAction || undefined;
      const from = filterFrom || undefined;
      const to = filterTo ? filterTo + 'T23:59:59' : undefined;
      const res = await getTrades(mode, TRADE_PAGE_SIZE, off, sym, act, from, to);
      const batch = res.trades || [];
      if (reset) {
        trades = batch;
        tradeOffset = batch.length;
      } else {
        trades = [...trades, ...batch];
        tradeOffset += batch.length;
      }
      tradeHasMore = batch.length >= TRADE_PAGE_SIZE;
      tradeSummary = { total: res.total ?? 0, pnl: res.summary_pnl ?? 0, fees: res.summary_fees ?? 0 };
    } catch (e) { console.error('fetchTrades:', e); }
    tradeLoading = false;
  }

  async function loadConfigFiles() {
    try {
      const res = await getConfigFiles();
      configFiles = Array.isArray(res) ? res : [];
    } catch (e) { console.error('loadConfigFiles:', e); }
  }

  async function loadConfigHistory() {
    configLoading = true;
    const file = selectedConfigFile;
    try {
      const res = await getConfigHistory(file);
      if (file === selectedConfigFile) {
        configHistory = Array.isArray(res) ? res : [];
      }
    } catch (e) {
      console.error('loadConfigHistory:', e);
      if (file === selectedConfigFile) {
        configHistory = [];
      }
    }
    finally { configLoading = false; }
  }

  async function loadDiff() {
    if (!diffA || !diffB || diffA === diffB) {
      diffResult = [];
      return;
    }
    diffLoading = true;
    diffResult = [];
    try {
      const res = await getConfigDiffPrivileged(diffA, diffB, selectedConfigFile);
      diffResult = res.diff || [];
    } catch (e) {
      console.error('loadDiff:', e);
      diffResult = [e instanceof Error ? `Error: ${e.message}` : 'Error: Failed to load diff'];
    }
    diffLoading = false;
  }

  function loadCurrentView() {
    if (tab === 'transactions') {
      if (txView === 'journeys') fetchJourneys(true);
      else fetchTrades(true);
    } else {
      loadConfigFiles();
      loadConfigHistory();
    }
  }

  function applyTradeFilters() { fetchTrades(true); }
  function clearTradeFilters() {
    filterSymbol = ''; filterAction = ''; filterFrom = ''; filterTo = '';
    fetchTrades(true);
  }

  function applyJourneyFilter() { fetchJourneys(true); }
  function clearJourneyFilter() { journeySymFilter = ''; fetchJourneys(true); }

  // ── Init ─────────────────────────────────────────────────────────
  loadCurrentView();
</script>

<!-- ── Mode selector ──────────────────────────────────────────────── -->
<div class="topbar">
  <div class="topbar-row">
    <h2 class="page-title">Statements</h2>
    <div class="mode-tabs">
      <button
        class="mode-btn mode-btn-live"
        class:active={mode === LIVE_MODE}
        onclick={() => setMode(LIVE_MODE)}
      >{getModeLabel(LIVE_MODE)}</button>
      <span class="mode-divider" aria-hidden="true"></span>
      <div class="family-tabs">
        {#each CANDIDATE_FAMILY_ORDER as m}
          <button
            class="mode-btn"
            class:active={mode === m}
            onclick={() => setMode(m)}
          >{getModeLabel(m)}</button>
        {/each}
      </div>
    </div>
  </div>
</div>

<!-- ── Top tabs ───────────────────────────────────────────────────── -->
<div class="top-tabs">
  <button class="top-tab" class:active={tab === 'transactions'} onclick={() => { tab = 'transactions'; loadCurrentView(); }}>Transactions</button>
  <button class="top-tab" class:active={tab === 'config'} onclick={() => { tab = 'config'; loadCurrentView(); }}>Config Changes</button>
</div>

{#if tab === 'transactions'}
  <!-- ── Sub tabs: Journeys / All Trades ──────────────────────────── -->
  <div class="sub-tabs">
    <button class="sub-tab" class:active={txView === 'journeys'} onclick={() => { txView = 'journeys'; fetchJourneys(true); }}>Journeys</button>
    <button class="sub-tab" class:active={txView === 'trades'} onclick={() => { txView = 'trades'; fetchTrades(true); }}>All Trades</button>
  </div>

  {#if txView === 'journeys'}
    <!-- ── Journey filter ─────────────────────────────────────────── -->
    <div class="filter-bar">
      <input class="filter-input" type="text" placeholder="Symbol" bind:value={journeySymFilter} onkeydown={(e) => e.key === 'Enter' && applyJourneyFilter()} />
      <button class="btn-sm" onclick={applyJourneyFilter}>Filter</button>
      {#if journeySymFilter}
        <button class="btn-sm btn-dim" onclick={clearJourneyFilter}>Clear</button>
      {/if}
    </div>

    <!-- ── Journey table ──────────────────────────────────────────── -->
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Type</th>
            <th>Open</th>
            <th>Close</th>
            <th>Duration</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Size</th>
            <th class="num">PnL</th>
            <th class="num">Fees</th>
            <th class="num">Net</th>
            <th>Legs</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {#each journeys as j}
            <tr>
              <td class="mono">{j.symbol}</td>
              <td><span class="badge" class:long={j.pos_type === 'LONG'} class:short={j.pos_type === 'SHORT'}>{j.pos_type}</span></td>
              <td class="ts">{fmtTs(j.open_ts)}</td>
              <td class="ts">{j.close_ts ? fmtTs(j.close_ts) : '-'}</td>
              <td>{journeyDuration(j)}</td>
              <td class="num">{fmtNum(j.entry_price, 4)}</td>
              <td class="num">{j.exit_price != null ? fmtNum(j.exit_price, 4) : '-'}</td>
              <td class="num">{fmtNum(j.peak_size, 4)}</td>
              <td class="num {pnlClass(j.total_pnl)}">{fmtNum(j.total_pnl)}</td>
              <td class="num">{fmtNum(j.total_fees)}</td>
              <td class="num {pnlClass(j.total_pnl - j.total_fees)}">{fmtNum(j.total_pnl - j.total_fees)}</td>
              <td>{j.legs?.length ?? 0}</td>
              <td><span class="badge" class:open-badge={j.is_open} class:closed-badge={!j.is_open}>{j.is_open ? 'OPEN' : 'CLOSED'}</span></td>
            </tr>
          {/each}
          {#if journeys.length === 0 && !journeyLoading}
            <tr><td colspan="13" class="empty">No journeys found</td></tr>
          {/if}
        </tbody>
      </table>
    </div>

    {#if journeyHasMore}
      <button class="btn-load-more" onclick={() => fetchJourneys(false)} disabled={journeyLoading}>
        {journeyLoading ? 'Loading...' : 'Load more'}
      </button>
    {/if}

  {:else}
    <!-- ── Trade filters ──────────────────────────────────────────── -->
    <div class="filter-bar">
      <input class="filter-input" type="text" placeholder="Symbol" bind:value={filterSymbol} />
      <select class="filter-select" bind:value={filterAction}>
        <option value="">All actions</option>
        <option value="OPEN">OPEN</option>
        <option value="ADD">ADD</option>
        <option value="REDUCE">REDUCE</option>
        <option value="CLOSE">CLOSE</option>
      </select>
      <input class="filter-input date-input" type="date" bind:value={filterFrom} />
      <input class="filter-input date-input" type="date" bind:value={filterTo} />
      <button class="btn-sm" onclick={applyTradeFilters}>Filter</button>
      {#if filterSymbol || filterAction || filterFrom || filterTo}
        <button class="btn-sm btn-dim" onclick={clearTradeFilters}>Clear</button>
      {/if}
    </div>

    <!-- ── Summary pills ──────────────────────────────────────────── -->
    <div class="summary-pills">
      <span class="pill">Trades: <strong>{tradeSummary.total.toLocaleString()}</strong></span>
      <span class="pill {pnlClass(tradeSummary.pnl)}">PnL: <strong>{fmtNum(tradeSummary.pnl)}</strong></span>
      <span class="pill">Fees: <strong>{fmtNum(tradeSummary.fees)}</strong></span>
      <span class="pill {pnlClass(tradeSummary.pnl - tradeSummary.fees)}">Net: <strong>{fmtNum(tradeSummary.pnl - tradeSummary.fees)}</strong></span>
    </div>

    <!-- ── Trade table ────────────────────────────────────────────── -->
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Type</th>
            <th>Action</th>
            <th class="num">Price</th>
            <th class="num">Size</th>
            <th class="num">Notional</th>
            <th class="num">PnL</th>
            <th class="num">Fee</th>
            <th class="num">Balance</th>
            <th>Reason</th>
            <th>Conf</th>
          </tr>
        </thead>
        <tbody>
          {#each trades as t}
            <tr>
              <td class="mono dim">{t.id}</td>
              <td class="ts">{fmtTs(t.timestamp)}</td>
              <td class="mono">{t.symbol}</td>
              <td><span class="badge" class:long={t.type === 'LONG'} class:short={t.type === 'SHORT'}>{t.type ?? '-'}</span></td>
              <td>{t.action ?? '-'}</td>
              <td class="num">{fmtNum(t.price, 4)}</td>
              <td class="num">{fmtNum(t.size, 4)}</td>
              <td class="num">{fmtNum(t.notional)}</td>
              <td class="num {pnlClass(t.pnl)}">{fmtNum(t.pnl)}</td>
              <td class="num">{fmtNum(t.fee_usd)}</td>
              <td class="num">{fmtNum(t.balance)}</td>
              <td class="reason">{t.reason ?? '-'}</td>
              <td>{t.confidence ?? '-'}</td>
            </tr>
          {/each}
          {#if trades.length === 0 && !tradeLoading}
            <tr><td colspan="13" class="empty">No trades found</td></tr>
          {/if}
        </tbody>
      </table>
    </div>

    {#if tradeHasMore}
      <button class="btn-load-more" onclick={() => fetchTrades(false)} disabled={tradeLoading}>
        {tradeLoading ? 'Loading...' : 'Load more'}
      </button>
    {/if}
  {/if}

{:else}
  <!-- ── Config Changes tab ───────────────────────────────────────── -->
  <div class="config-section">
    <div class="filter-bar">
      <label class="filter-label" for="config-file-select">Config file:</label>
      <select
        id="config-file-select"
        class="filter-select"
        bind:value={selectedConfigFile}
        onchange={() => {
          resetConfigDiff();
          loadConfigHistory();
        }}
      >
        {#each configFiles as f}
          <option value={f.variant}>{f.variant}{f.exists === false ? ' (missing)' : ''}</option>
        {/each}
        {#if configFiles.length === 0}
          <option value="main">main</option>
        {/if}
      </select>
    </div>

    <!-- ── Backup history ───────────────────────────────────────────── -->
    <h3 class="section-title">Backup History</h3>
    {#if configLoading}
      <p class="dim">Loading...</p>
    {:else if configHistory.length === 0}
      <p class="dim">No backup history found for this config file.</p>
    {:else}
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Modified</th>
              <th>Size</th>
            </tr>
          </thead>
          <tbody>
            {#each configHistory as h}
              <tr>
                <td class="mono">{h.filename}</td>
                <td class="ts">{h.modified ?? '-'}</td>
                <td>{h.size != null ? `${(h.size / 1024).toFixed(1)} KB` : '-'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <!-- ── Diff viewer ──────────────────────────────────────────────── -->
    <h3 class="section-title">Compare Versions</h3>
    <div class="diff-controls">
      <select class="filter-select" bind:value={diffA}>
        <option value="">Version A</option>
        {#each configHistory as h}
          <option value={h.filename}>{h.filename}</option>
        {/each}
      </select>
      <span class="diff-arrow">vs</span>
      <select class="filter-select" bind:value={diffB}>
        <option value="current">Current</option>
        {#each configHistory as h}
          <option value={h.filename}>{h.filename}</option>
        {/each}
      </select>
      <button class="btn-sm" onclick={loadDiff} disabled={!diffA || !diffB || diffA === diffB || diffLoading}>
        {diffLoading ? 'Loading...' : 'Compare'}
      </button>
    </div>

    {#if diffResult.length > 0}
      <div class="diff-output">
        {#each diffResult as line}
          <div class="diff-line" class:diff-add={line.startsWith('+')} class:diff-del={line.startsWith('-')} class:diff-hunk={line.startsWith('@@')}>{line}</div>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<style>
  /* ─── Topbar ─── */
  .topbar {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .topbar-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .page-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    white-space: nowrap;
  }
  .mode-tabs {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .family-tabs {
    display: flex;
    gap: 4px;
  }
  .mode-divider {
    width: 1px;
    height: 18px;
    background: var(--border);
    margin: 0 2px;
    align-self: stretch;
    opacity: 0.9;
  }
  .mode-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    letter-spacing: 0.02em;
    transition: all var(--t-fast);
    white-space: nowrap;
  }
  .mode-btn:hover { color: var(--text); background: var(--surface-hover); }
  .mode-btn.active { background: var(--accent); color: var(--bg); box-shadow: 0 1px 4px rgba(77,171,247,0.3); }
  .mode-btn-live { color: rgba(255,107,107,0.95); border: 1px solid rgba(255,107,107,0.35); flex-shrink: 0; }
  .mode-btn-live:hover { background: rgba(255,107,107,0.12); color: #ffc9c9; }
  .mode-btn-live.active { background: var(--red); box-shadow: 0 1px 4px rgba(255,107,107,0.3); color: var(--bg); }

  /* ─── Top tabs ─── */
  .top-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .top-tab {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all var(--t-fast);
  }
  .top-tab:hover { color: var(--text); }
  .top-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ─── Sub tabs ─── */
  .sub-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 10px;
  }
  .sub-tab {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--t-fast);
  }
  .sub-tab:hover { color: var(--text); background: var(--surface-hover); }
  .sub-tab.active { background: var(--accent-bg); color: var(--accent); border-color: var(--accent); }

  /* ─── Filter bar ─── */
  .filter-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .filter-label {
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 500;
  }
  .filter-input {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 4px 8px;
    width: 100px;
  }
  .filter-input:focus { outline: none; border-color: var(--accent); }
  .date-input { width: 130px; }
  .filter-select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-size: 12px;
    padding: 4px 8px;
  }
  .btn-sm {
    background: var(--accent-bg);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--t-fast);
  }
  .btn-sm:hover { background: var(--accent); color: var(--bg); }
  .btn-dim { border-color: var(--border); color: var(--text-muted); background: transparent; }
  .btn-dim:hover { background: var(--surface-hover); color: var(--text); }

  /* ─── Summary pills ─── */
  .summary-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 3px 10px;
    font-size: 12px;
    color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
  }
  .pill strong { color: var(--text); }
  .pill.pnl-pos strong { color: var(--green); }
  .pill.pnl-neg strong { color: var(--red); }

  /* ─── Table ─── */
  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
  }
  .data-table th {
    background: var(--surface);
    color: var(--text-dim);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 1;
  }
  .data-table td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }
  .data-table tbody tr:hover { background: var(--surface-hover); }
  .data-table th.num, .data-table td.num { text-align: right; font-family: 'IBM Plex Mono', monospace; }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  .dim { color: var(--text-dim); }
  .ts { font-size: 11px; color: var(--text-dim); font-family: 'IBM Plex Mono', monospace; }
  .reason { max-width: 150px; overflow: hidden; text-overflow: ellipsis; }
  .empty { text-align: center; color: var(--text-dim); padding: 24px 10px !important; }

  /* ─── Badges ─── */
  .badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 1px 6px;
    border-radius: 3px;
    background: var(--surface);
    color: var(--text-dim);
  }
  .badge.long { background: rgba(81,207,102,0.15); color: var(--green); }
  .badge.short { background: rgba(255,107,107,0.15); color: var(--red); }
  .badge.open-badge { background: rgba(77,171,247,0.15); color: var(--accent); }
  .badge.closed-badge { background: var(--surface); color: var(--text-dim); }

  /* ─── PnL colors ─── */
  .pnl-pos { color: var(--green) !important; }
  .pnl-neg { color: var(--red) !important; }

  /* ─── Load more ─── */
  .btn-load-more {
    display: block;
    width: 100%;
    margin-top: 8px;
    padding: 8px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--t-fast);
  }
  .btn-load-more:hover { background: var(--surface-hover); color: var(--text); }
  .btn-load-more:disabled { opacity: 0.5; cursor: default; }

  /* ─── Config section ─── */
  .config-section { padding-top: 4px; }
  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin: 16px 0 8px;
  }

  /* ─── Diff viewer ─── */
  .diff-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .diff-arrow {
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 600;
  }
  .diff-output {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    overflow-x: auto;
    max-height: 500px;
    overflow-y: auto;
  }
  .diff-line {
    white-space: pre;
    line-height: 1.5;
    color: var(--text-dim);
  }
  .diff-add { color: var(--green); background: rgba(81,207,102,0.08); }
  .diff-del { color: var(--red); background: rgba(255,107,107,0.08); }
  .diff-hunk { color: var(--accent); font-weight: 600; }

  /* ─── Mobile ─── */
  @media (max-width: 768px) {
    .topbar-row { flex-wrap: wrap; }
    .mode-tabs { width: 100%; overflow-x: auto; }
    .filter-bar { flex-direction: column; align-items: stretch; }
    .filter-input, .filter-select { width: 100%; }
    .date-input { width: 100%; }
    .summary-pills { flex-direction: column; }
    .diff-controls { flex-direction: column; align-items: stretch; }
  }
</style>
