<script lang="ts">
  import { getSnapshot, getMids } from '../lib/api';
  import { CANDIDATE_FAMILY_ORDER, getModeLabel, LIVE_MODE } from '../lib/mode-labels';

  let mode = $state('paper1');
  let gridSize = $state(3);
  let symbols: any[] = $state([]);
  let mids: Record<string, number> = $state({});
  let loading = $state(true);
  let filter = $state('');
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      const snap = await getSnapshot(mode);
      symbols = snap.symbols || [];
      const m = await getMids();
      mids = m.mids || {};
    } catch {}
    loading = false;
  }

  let filteredSymbols = $derived.by(() => {
    const q = filter.trim().toUpperCase();
    let syms = symbols;
    if (q) syms = syms.filter((s: any) => String(s.symbol).includes(q));
    return syms;
  });

  function fmtPrice(n: any): string {
    if (n == null) return '—';
    const v = Number(n);
    if (v >= 1000) return v.toFixed(2);
    if (v >= 1) return v.toFixed(4);
    return v.toFixed(6);
  }

  $effect(() => {
    refresh();
    pollTimer = setInterval(refresh, 10000);
    return () => { if (pollTimer) clearInterval(pollTimer); };
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
    </div>
  </div>

  {#if loading}
    <div class="empty-state">Loading...</div>
  {:else}
    <div class="symbol-grid" style="grid-template-columns: repeat({gridSize}, 1fr);">
      {#each filteredSymbols as s (s.symbol)}
        <div class="grid-cell" class:has-long={s.position_side === 'LONG'} class:has-short={s.position_side === 'SHORT'}>
          <div class="cell-header">
            <span class="cell-symbol">{s.symbol}</span>
            {#if s.position_side && s.position_side !== 'NONE'}
              <span class="pos-badge" class:long={s.position_side === 'LONG'} class:short={s.position_side === 'SHORT'}>
                {s.position_side}
              </span>
            {/if}
          </div>
          <div class="cell-price">{fmtPrice(mids[s.symbol] ?? s.mid)}</div>
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
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 14px;
    min-height: 100px;
    transition: all var(--t-fast);
  }
  .grid-cell:hover {
    border-color: var(--text-dim);
  }
  .grid-cell.has-long {
    border-color: rgba(81,207,102,0.3);
    background: linear-gradient(180deg, rgba(81,207,102,0.03) 0%, var(--surface) 100%);
  }
  .grid-cell.has-short {
    border-color: rgba(255,107,107,0.3);
    background: linear-gradient(180deg, rgba(255,107,107,0.03) 0%, var(--surface) 100%);
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
