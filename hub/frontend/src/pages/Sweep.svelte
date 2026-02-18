<script lang="ts">
  import { runSweep, getSweepJobs, getSweepStatus, getSweepResults, cancelSweep } from '../lib/api';

  let config = $state('main');
  let sweepSpec = $state('');
  let balance = $state(10000);
  let launching = $state(false);
  let error = $state('');

  let jobs: any[] = $state([]);
  let activeJob: string | null = $state(null);
  let activeStatus: any = $state(null);
  let activeResults: any = $state(null);
  let stderrLines: string[] = $state([]);

  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function launch() {
    if (!sweepSpec.trim()) { error = 'Sweep spec path required'; return; }
    launching = true;
    error = '';
    try {
      const res = await runSweep({
        config,
        sweep_spec: sweepSpec.trim(),
        initial_balance: balance,
      });
      activeJob = res.job_id;
      activeResults = null;
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
      jobs = await getSweepJobs();
    } catch {}
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      if (!activeJob) return;
      try {
        const s = await getSweepStatus(activeJob);
        activeStatus = s;
        stderrLines = s.stderr_tail || [];
        if (s.status !== 'running') {
          stopPolling();
          if (s.status === 'done') {
            try {
              activeResults = await getSweepResults(activeJob);
            } catch {}
          }
          await refreshJobs();
        }
      } catch {}
    }, 3000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function cancel() {
    if (!activeJob) return;
    try { await cancelSweep(activeJob); stopPolling(); await refreshJobs(); } catch {}
  }

  function selectJob(id: string) {
    activeJob = id;
    activeResults = null;
    stderrLines = [];
    const job = jobs.find((j: any) => j.id === id);
    if (job?.status === 'running') {
      startPolling();
    } else if (job?.status === 'done') {
      getSweepResults(id).then(r => { activeResults = r; }).catch(() => {});
      getSweepStatus(id).then(s => { stderrLines = s.stderr_tail || []; }).catch(() => {});
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

<div class="sweep-page">
  <h1>Parameter Sweep</h1>

  <div class="launcher">
    <div class="form-row">
      <label>Config <select bind:value={config}>
        <option value="main">main</option>
        <option value="live">live</option>
        <option value="paper1">paper1</option>
      </select></label>
      <label>Balance <input type="number" bind:value={balance} min="100" step="100" /></label>
      <label class="wide">Sweep Spec <input type="text" bind:value={sweepSpec} placeholder="config/sweep_spec.yaml" /></label>
      <button class="btn btn-primary" onclick={launch} disabled={launching || !sweepSpec.trim()}>
        {launching ? 'Launching...' : 'Run Sweep'}
      </button>
    </div>
    {#if error}
      <div class="alert alert-error">{error}</div>
    {/if}
  </div>

  <div class="content-grid">
    <div class="jobs-panel">
      <h2>Jobs</h2>
      {#if jobs.length === 0}
        <div class="empty">No sweep jobs yet</div>
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

    <div class="result-panel">
      {#if activeJob && activeStatus?.status === 'running'}
        <div class="progress-section">
          <div class="progress-header">
            <span>Running sweep...</span>
            <button class="btn btn-secondary" onclick={cancel}>Cancel</button>
          </div>
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        </div>
      {:else if activeResults}
        <div class="result-section">
          <h2>Sweep Results</h2>
          {#if Array.isArray(activeResults)}
            <table class="results-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Config ID</th>
                  <th>Net PnL</th>
                  <th>Win Rate</th>
                  <th>Max DD</th>
                  <th>Sharpe</th>
                  <th>Trades</th>
                </tr>
              </thead>
              <tbody>
                {#each activeResults as r, i}
                  <tr>
                    <td>{i + 1}</td>
                    <td class="mono">{r.config_id ?? r.id ?? '—'}</td>
                    <td>${fmtNum(r.report?.net_pnl ?? r.net_pnl)}</td>
                    <td>{fmtNum(((r.report?.win_rate ?? r.win_rate ?? 0) * 100), 1)}%</td>
                    <td>{fmtNum(r.report?.max_drawdown_pct ?? r.max_drawdown_pct, 1)}%</td>
                    <td>{fmtNum(r.report?.sharpe_ratio ?? r.sharpe_ratio)}</td>
                    <td>{r.report?.total_trades ?? r.total_trades ?? '—'}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <pre class="result-json">{JSON.stringify(activeResults, null, 2)}</pre>
          {/if}
        </div>
      {:else if activeStatus?.error}
        <div class="alert alert-error">{activeStatus.error}</div>
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
      {:else}
        <div class="empty">Select a job or run a new sweep</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .sweep-page { max-width: 1400px; }

  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }

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

  .form-row label.wide { flex: 1; min-width: 200px; }

  .form-row input, .form-row select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
  }

  .btn { padding: 6px 14px; border: none; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn-primary { background: var(--accent, #3b82f6); color: #fff; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }

  .alert { padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-top: 8px; }
  .alert-error { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

  .content-grid { display: grid; grid-template-columns: 240px 1fr; gap: 16px; }

  .jobs-panel { border-right: 1px solid var(--border); padding-right: 16px; }

  .job-card {
    display: block; width: 100%; text-align: left;
    padding: 8px 10px; background: none; border: 1px solid transparent;
    border-radius: 6px; cursor: pointer; margin-bottom: 4px; color: var(--text);
  }
  .job-card:hover { background: var(--surface); }
  .job-card.active { border-color: var(--accent, #3b82f6); background: var(--surface); }

  .job-id { font-family: monospace; font-size: 12px; }
  .job-meta { display: flex; gap: 8px; align-items: center; margin-top: 4px; }
  .job-time { font-size: 11px; color: var(--text-muted); }

  .status-pill { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 500; }
  .status-pill.running { background: rgba(59,130,246,0.2); color: #60a5fa; }
  .status-pill.done { background: rgba(34,197,94,0.2); color: #4ade80; }
  .status-pill.failed { background: rgba(239,68,68,0.2); color: #f87171; }
  .status-pill.cancelled { background: rgba(234,179,8,0.2); color: #eab308; }

  .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }

  .stderr-log {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px; font-size: 11px; line-height: 1.5;
    max-height: 400px; overflow-y: auto; white-space: pre-wrap; color: var(--text-muted);
  }

  .results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .results-table th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; }
  .results-table td { padding: 8px 10px; border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.05)); }

  .mono { font-family: monospace; font-size: 12px; }

  .result-json {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; font-size: 12px;
    max-height: 600px; overflow: auto; white-space: pre-wrap;
  }

  .empty { color: var(--text-muted); font-style: italic; padding: 20px 0; }
</style>
