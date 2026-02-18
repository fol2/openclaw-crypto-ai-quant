<script lang="ts">
  import { getSnapshot, getMids } from '../lib/api';

  let mode = $state('paper1');
  let gridSize = $state(3); // NxN grid
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

<div class="grid-page">
  <div class="grid-header">
    <h1>Grid View</h1>
    <div class="controls">
      <select bind:value={mode} onchange={refresh}>
        {#each ['live', 'paper1', 'paper2', 'paper3'] as m}
          <option value={m}>{m}</option>
        {/each}
      </select>
      <input type="text" bind:value={filter} placeholder="Filter symbols..." />
      <select bind:value={gridSize}>
        <option value={2}>2x2</option>
        <option value={3}>3x3</option>
        <option value={4}>4x4</option>
        <option value={5}>5x5</option>
      </select>
    </div>
  </div>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="symbol-grid" style="grid-template-columns: repeat({gridSize}, 1fr);">
      {#each filteredSymbols as s (s.symbol)}
        <div class="grid-cell" class:has-position={s.position_side && s.position_side !== 'NONE'}>
          <div class="cell-header">
            <span class="cell-symbol">{s.symbol}</span>
            {#if s.position_side && s.position_side !== 'NONE'}
              <span class="pos-badge {s.position_side === 'LONG' ? 'long' : 'short'}">
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
              color={s.position_side === 'LONG' ? '#4ade80' : s.position_side === 'SHORT' ? '#f87171' : '#666'}
            ></sparkline-chart>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .grid-page { max-width: 1600px; }

  .grid-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .grid-header h1 { font-size: 20px; font-weight: 600; margin: 0; }

  .controls { display: flex; gap: 8px; }
  .controls select, .controls input {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
  }

  .symbol-grid {
    display: grid;
    gap: 8px;
  }

  .grid-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    min-height: 100px;
  }

  .grid-cell.has-position { border-color: var(--accent, #3b82f6); }

  .cell-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .cell-symbol { font-weight: 600; font-size: 13px; }

  .cell-price { font-size: 16px; font-weight: 500; margin-bottom: 4px; }

  .pos-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
  }
  .pos-badge.long { background: rgba(34,197,94,0.2); color: #4ade80; }
  .pos-badge.short { background: rgba(239,68,68,0.2); color: #f87171; }

  .signal-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(234,179,8,0.2);
    color: #eab308;
  }

  .cell-sparkline { margin-top: 4px; }

  .empty { color: var(--text-muted); font-style: italic; padding: 20px 0; }
</style>
