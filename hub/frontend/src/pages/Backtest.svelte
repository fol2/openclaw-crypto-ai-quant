<script lang="ts">
  import { runBacktest, getBacktestJobs, getBacktestStatus, getBacktestResult, cancelBacktest } from '../lib/api';

  let config = $state('main');
  let balance = $state(10000);
  let symbol = $state('');
  let launching = $state(false);
  let error = $state('');

  let jobs: any[] = $state([]);
  let activeJob: string | null = $state(null);
  let activeStatus: any = $state(null);
  let activeResult: any = $state(null);
  let stderrLines: string[] = $state([]);

  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function launch() {
    launching = true;
    error = '';
    try {
      const res = await runBacktest({
        config,
        initial_balance: balance,
        symbol: symbol.trim() || undefined,
      });
      activeJob = res.job_id;
      activeResult = null;
      stderrLines = [];
      startPolling();
      await refreshJobs();
    } catch (e: any) {
      error = e.message || 'Failed to launch';
    }
    launching = false;
  }

  async function refreshJobs() {
    try {
      jobs = await getBacktestJobs();
    } catch {}
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      if (!activeJob) return;
      try {
        const s = await getBacktestStatus(activeJob);
        activeStatus = s;
        stderrLines = s.stderr_tail || [];
        if (s.status !== 'running') {
          stopPolling();
          if (s.status === 'done') {
            try {
              activeResult = await getBacktestResult(activeJob);
            } catch {}
          }
          await refreshJobs();
        }
      } catch {}
    }, 2000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function cancel() {
    if (!activeJob) return;
    try {
      await cancelBacktest(activeJob);
      stopPolling();
      await refreshJobs();
    } catch {}
  }

  function selectJob(id: string) {
    activeJob = id;
    activeResult = null;
    stderrLines = [];
    const job = jobs.find((j: any) => j.id === id);
    if (job?.status === 'running') {
      startPolling();
    } else if (job?.status === 'done') {
      getBacktestResult(id).then(r => { activeResult = r; }).catch(() => {});
      getBacktestStatus(id).then(s => { stderrLines = s.stderr_tail || []; }).catch(() => {});
    }
  }

  function fmtNum(n: any, d = 2): string {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(d);
  }

  $effect(() => {
    refreshJobs();
    return () => stopPolling();
  });
</script>

<div class="bt-page">
  <h1>Backtest</h1>

  <div class="launcher">
    <div class="form-row">
      <label>Config <select bind:value={config}>
        <option value="main">main</option>
        <option value="live">live</option>
        <option value="paper1">paper1</option>
        <option value="paper2">paper2</option>
        <option value="paper3">paper3</option>
      </select></label>
      <label>Balance <input type="number" bind:value={balance} min="100" step="100" /></label>
      <label>Symbol <input type="text" bind:value={symbol} placeholder="All symbols" /></label>
      <button class="btn btn-primary" onclick={launch} disabled={launching}>
        {launching ? 'Launching...' : 'Run Backtest'}
      </button>
    </div>
    {#if error}
      <div class="alert alert-error">{error}</div>
    {/if}
  </div>

  <div class="content-grid">
    <!-- Jobs list -->
    <div class="jobs-panel">
      <h2>Jobs</h2>
      {#if jobs.length === 0}
        <div class="empty">No backtest jobs yet</div>
      {:else}
        {#each jobs as j (j.id)}
          <button class="job-card" class:active={activeJob === j.id} onclick={() => selectJob(j.id)}>
            <div class="job-id">{j.id.slice(0, 8)}</div>
            <div class="job-meta">
              <span class="status-pill {j.status}">{j.status}</span>
              <span class="job-time">{j.created_at ? new Date(j.created_at).toLocaleTimeString() : ''}</span>
            </div>
          </button>
        {/each}
      {/if}
    </div>

    <!-- Result panel -->
    <div class="result-panel">
      {#if activeJob && activeStatus?.status === 'running'}
        <div class="progress-section">
          <div class="progress-header">
            <span>Running...</span>
            <button class="btn btn-secondary" onclick={cancel}>Cancel</button>
          </div>
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        </div>
      {:else if activeResult}
        <div class="result-section">
          <h2>Result</h2>
          <div class="stats-grid">
            <div class="stat"><span class="stat-label">Trades</span><span class="stat-value">{activeResult.total_trades ?? '—'}</span></div>
            <div class="stat"><span class="stat-label">Net PnL</span><span class="stat-value">${fmtNum(activeResult.net_pnl)}</span></div>
            <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value">{fmtNum((activeResult.win_rate ?? 0) * 100, 1)}%</span></div>
            <div class="stat"><span class="stat-label">Max DD</span><span class="stat-value">{fmtNum(activeResult.max_drawdown_pct, 1)}%</span></div>
            <div class="stat"><span class="stat-label">Sharpe</span><span class="stat-value">{fmtNum(activeResult.sharpe_ratio)}</span></div>
            <div class="stat"><span class="stat-label">Profit Factor</span><span class="stat-value">{fmtNum(activeResult.profit_factor)}</span></div>
            <div class="stat"><span class="stat-label">Final Balance</span><span class="stat-value">${fmtNum(activeResult.final_balance)}</span></div>
            <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">{activeResult.duration_days ?? '—'} days</span></div>
          </div>

          {#if activeResult.equity_curve?.length > 0}
            <h3>Equity Curve</h3>
            <div class="equity-chart">
              <sparkline-chart
                points={JSON.stringify(activeResult.equity_curve.map((e: any) => e[1] ?? e.balance ?? e))}
                width="800"
                height="200"
                color="#4ade80"
              ></sparkline-chart>
            </div>
          {/if}

          {#if activeResult.per_symbol}
            <h3>Per Symbol</h3>
            <table class="symbol-table">
              <thead><tr><th>Symbol</th><th>Trades</th><th>PnL</th><th>Win%</th></tr></thead>
              <tbody>
                {#each Object.entries(activeResult.per_symbol) as [sym, data]}
                  <tr>
                    <td>{sym}</td>
                    <td>{(data as any).trades ?? '—'}</td>
                    <td>${fmtNum((data as any).pnl)}</td>
                    <td>{fmtNum(((data as any).win_rate ?? 0) * 100, 1)}%</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      {:else if activeStatus?.error}
        <div class="alert alert-error">{activeStatus.error}</div>
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
      {:else}
        <div class="empty">Select a job or run a new backtest</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .bt-page { max-width: 1400px; }

  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  h3 { font-size: 14px; font-weight: 500; margin: 16px 0 8px; color: var(--text-muted); }

  .launcher { margin-bottom: 20px; }

  .form-row {
    display: flex;
    gap: 12px;
    align-items: end;
    flex-wrap: wrap;
  }

  .form-row label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-row input, .form-row select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
  }

  .btn {
    padding: 6px 14px;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn-primary { background: var(--accent, #3b82f6); color: #fff; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }

  .alert { padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-top: 8px; }
  .alert-error { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

  .content-grid { display: grid; grid-template-columns: 240px 1fr; gap: 16px; }

  .jobs-panel { border-right: 1px solid var(--border); padding-right: 16px; }

  .job-card {
    display: block;
    width: 100%;
    text-align: left;
    padding: 8px 10px;
    background: none;
    border: 1px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 4px;
    color: var(--text);
  }
  .job-card:hover { background: var(--surface); }
  .job-card.active { border-color: var(--accent, #3b82f6); background: var(--surface); }

  .job-id { font-family: monospace; font-size: 12px; }
  .job-meta { display: flex; gap: 8px; align-items: center; margin-top: 4px; }
  .job-time { font-size: 11px; color: var(--text-muted); }

  .status-pill {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
  }
  .status-pill.running { background: rgba(59,130,246,0.2); color: #60a5fa; }
  .status-pill.done { background: rgba(34,197,94,0.2); color: #4ade80; }
  .status-pill.failed { background: rgba(239,68,68,0.2); color: #f87171; }
  .status-pill.cancelled { background: rgba(234,179,8,0.2); color: #eab308; }

  .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }

  .stderr-log {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.5;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: var(--text-muted);
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .stat-label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
  .stat-value { font-size: 16px; font-weight: 600; }

  .equity-chart { margin-bottom: 16px; }

  .symbol-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .symbol-table th { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; }
  .symbol-table td { padding: 6px 10px; border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.05)); }

  .empty { color: var(--text-muted); font-style: italic; padding: 20px 0; }
</style>
