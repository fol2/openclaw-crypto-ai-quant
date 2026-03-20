<script lang="ts">
  import { runBacktest, getBacktestJobs, getBacktestStatus, getBacktestResult, cancelBacktest } from '../lib/api';
  import { CANDIDATE_FAMILY_ORDER, getConfigLabel, LIVE_MODE } from '../lib/mode-labels';

  const candidateConfigs = CANDIDATE_FAMILY_ORDER;
  const BACKTEST_CONFIG_LABELS: Record<string, string> = {
    main: 'Main config',
    live: 'Live engine',
    paper1: 'Paper 1 (Efficient)',
    paper2: 'Paper 2 (Growth)',
    paper3: 'Paper 3 (Conservative)',
  };

  let config = $state('main');
  let balance = $state(10000);
  let symbol = $state('');
  let launching = $state(false);
  let error = $state('');
  let jobsLoadError = $state('');

  let jobs: any[] = $state([]);
  let activeJob: string | null = $state(null);
  let activeStatus: any = $state(null);
  let activeResult: any = $state(null);
  let stderrLines: string[] = $state([]);
  let loadingJob = $state(false);

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
      activeStatus = {
        id: res.job_id,
        status: res.status ?? 'running',
        stderr_tail: [],
        error: null,
      };
      stderrLines = [];
      loadingJob = false;
      startPolling();
      await refreshJobs();
    } catch (e: any) {
      error = getErrorMessage(e, 'Failed to launch');
    }
    launching = false;
  }

  async function refreshJobs() {
    jobsLoadError = '';
    try {
      jobs = await getBacktestJobs();
    } catch (e: any) {
      jobsLoadError = getErrorMessage(e, 'Failed to load backtest jobs');
    }
  }

  async function loadJobResult(id: string, status: string, fallbackError: string) {
    try {
      const result = await getBacktestResult(id);
      if (activeJob === id) {
        activeResult = result;
      }
    } catch (e: any) {
      if (activeJob !== id) return;
      activeResult = null;
      if (status === 'done') {
        activeStatus = {
          ...(activeStatus ?? {}),
          id,
          status,
          stderr_tail: stderrLines,
          error: getErrorMessage(e, fallbackError),
        };
      }
    }
  }

  async function loadSelectedJob(id: string) {
    try {
      const status = await getBacktestStatus(id);
      if (activeJob !== id) return;

      activeStatus = status;
      stderrLines = status.stderr_tail || [];
      activeResult = null;

      if (status.status === 'running') {
        loadingJob = false;
        startPolling();
        return;
      }

      stopPolling();

      if (status.status === 'done' || status.status === 'failed') {
        await loadJobResult(
          id,
          status.status,
          'Backtest completed, but no structured result payload was available.'
        );
      }
    } catch (e: any) {
      if (activeJob === id) {
        activeStatus = {
          ...(activeStatus ?? {}),
          id,
          status: activeStatus?.status ?? 'failed',
          stderr_tail: stderrLines,
          error: getErrorMessage(e, 'Failed to load backtest job details'),
        };
      }
    } finally {
      if (activeJob === id) {
        loadingJob = false;
      }
    }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      const jobId = activeJob;
      if (!jobId) return;
      try {
        const s = await getBacktestStatus(jobId);
        if (activeJob !== jobId) return;
        activeStatus = s;
        stderrLines = s.stderr_tail || [];
        if (s.status !== 'running') {
          stopPolling();
          if (s.status === 'done' || s.status === 'failed') {
            await loadJobResult(
              jobId,
              s.status,
              'Backtest completed, but no structured result payload was available.'
            );
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
    const jobId = activeJob;
    try {
      await cancelBacktest(jobId);
      stopPolling();
      loadingJob = false;
      await refreshJobs();
      if (activeJob === jobId) {
        activeStatus = {
          ...(activeStatus ?? {}),
          id: jobId,
          status: 'cancelled',
          stderr_tail: stderrLines,
          error: null,
        };
      }
      void loadSelectedJob(jobId);
    } catch (e: any) {
      error = getErrorMessage(e, 'Failed to cancel');
    }
  }

  function selectJob(id: string) {
    const selected = jobs.find((job) => job.id === id);
    stopPolling();
    activeJob = id;
    activeResult = null;
    activeStatus = selected
      ? {
          ...selected,
          stderr_tail: [],
        }
      : null;
    stderrLines = [];
    loadingJob = true;
    void loadSelectedJob(id);
  }

  function getBacktestConfigLabel(configName: string): string {
    return BACKTEST_CONFIG_LABELS[configName] ?? getConfigLabel(configName);
  }

  function formatStatusLabel(status: string | null | undefined): string {
    if (!status) return 'Unknown';
    return status
      .split('_')
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ');
  }

  function fmtNum(n: any, d = 2): string {
    if (n == null || isNaN(n)) return '—';
    const numeric = Number(n);
    if (numeric === Number.POSITIVE_INFINITY) return '∞';
    if (numeric === Number.NEGATIVE_INFINITY) return '-∞';
    return numeric.toFixed(d);
  }

  function getNetPnl(result: any): number | null {
    const value = result?.total_pnl ?? result?.net_pnl;
    return value == null || isNaN(value) ? null : Number(value);
  }

  function getMaxDrawdownPercent(result: any): number | null {
    const displayValue = result?.max_drawdown_percent;
    if (displayValue != null && !isNaN(displayValue)) {
      return Number(displayValue);
    }

    const rawValue = result?.max_drawdown_pct;
    return rawValue == null || isNaN(rawValue) ? null : Number(rawValue) * 100;
  }

  function getPerSymbolPnl(data: any): number | null {
    const value = data?.net_pnl_usd ?? data?.pnl ?? data?.realised_pnl_usd;
    return value == null || isNaN(value) ? null : Number(value);
  }

  function getErrorMessage(value: unknown, fallback: string): string {
    if (value instanceof Error && value.message.trim()) return value.message;
    if (typeof value === 'string' && value.trim()) return value;
    return fallback;
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
        <option value="main">{getBacktestConfigLabel('main')}</option>
        <optgroup label="Live Engine">
          <option value={LIVE_MODE}>{getBacktestConfigLabel(LIVE_MODE)}</option>
        </optgroup>
        <optgroup label="Candidate Family">
          {#each candidateConfigs as option}
            <option value={option}>{getBacktestConfigLabel(option)}</option>
          {/each}
        </optgroup>
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
      {#if jobsLoadError}
        <div class="alert alert-error">{jobsLoadError}</div>
      {:else if jobs.length === 0}
        <div class="empty">No backtest jobs yet</div>
      {:else}
        {#each jobs as j (j.id)}
          <button class="job-card" class:active={activeJob === j.id} onclick={() => selectJob(j.id)}>
            <div class="job-id">{j.id.slice(0, 8)}</div>
            <div class="job-meta">
              <span class="status-pill {j.status}">{formatStatusLabel(j.status)}</span>
              <span class="job-time">{j.created_at ? new Date(j.created_at).toLocaleTimeString() : ''}</span>
            </div>
          </button>
        {/each}
      {/if}
    </div>

    <!-- Result panel -->
    <div class="result-panel">
      {#if activeJob && loadingJob}
        <div class="empty">Loading backtest job details...</div>
      {:else if activeJob && activeStatus?.status === 'running'}
        <div class="progress-section">
          <div class="progress-header">
            <span>Running...</span>
            <button class="btn btn-secondary" onclick={cancel}>Cancel</button>
          </div>
          <pre class="stderr-log">{stderrLines.length > 0 ? stderrLines.join('\n') : 'Waiting for progress output...'}</pre>
        </div>
      {:else if activeResult}
        <div class="result-section">
          <h2>Result</h2>
          {#if activeStatus?.error}
            <div class="alert alert-error">{activeStatus.error}</div>
          {/if}
          <div class="stats-grid">
            <div class="stat"><span class="stat-label">Trades</span><span class="stat-value">{activeResult.total_trades ?? '—'}</span></div>
            <div class="stat"><span class="stat-label">Total PnL</span><span class="stat-value">${fmtNum(getNetPnl(activeResult))}</span></div>
            <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value">{fmtNum((activeResult.win_rate ?? 0) * 100, 1)}%</span></div>
            <div class="stat"><span class="stat-label">Max DD</span><span class="stat-value">{fmtNum(getMaxDrawdownPercent(activeResult), 1)}%</span></div>
            <div class="stat"><span class="stat-label">Sharpe</span><span class="stat-value">{fmtNum(activeResult.sharpe_ratio)}</span></div>
            <div class="stat"><span class="stat-label">Profit Factor</span><span class="stat-value">{fmtNum(activeResult.profit_factor)}</span></div>
            <div class="stat"><span class="stat-label">Final Balance</span><span class="stat-value">${fmtNum(activeResult.final_balance)}</span></div>
            <div class="stat"><span class="stat-label">Total Fees</span><span class="stat-value">${fmtNum(activeResult.total_fees)}</span></div>
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
              <thead><tr><th>Symbol</th><th>Trades</th><th>Net PnL</th><th>Win%</th></tr></thead>
              <tbody>
                {#each Object.entries(activeResult.per_symbol) as [sym, data]}
                  <tr>
                    <td>{sym}</td>
                    <td>{(data as any).trades ?? '—'}</td>
                    <td>${fmtNum(getPerSymbolPnl(data))}</td>
                    <td>{fmtNum(((data as any).win_rate ?? 0) * 100, 1)}%</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}

          {#if stderrLines.length > 0}
            <h3>Job Log</h3>
            <pre class="stderr-log">{stderrLines.join('\n')}</pre>
          {/if}
        </div>
      {:else if activeJob && activeStatus?.status === 'done'}
        <div class="empty">Backtest finished, but no structured result payload was available.</div>
        {#if activeStatus?.error}
          <div class="alert alert-error">{activeStatus.error}</div>
        {/if}
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
      {:else if activeStatus?.status === 'failed'}
        <div class="alert alert-error">{activeStatus.error || 'Backtest failed'}</div>
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
      {:else if activeStatus?.status === 'cancelled'}
        <div class="empty">Backtest cancelled</div>
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
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
  /* ─── Page entry ─── */
  .bt-page {
    max-width: 1400px;
    animation: slideUp 0.3s ease;
  }

  /* ─── Typography ─── */
  h1 {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: var(--sp-md);
    letter-spacing: -0.01em;
  }
  h2 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: var(--sp-sm);
  }
  h3 {
    font-size: 14px;
    font-weight: 500;
    margin: var(--sp-md) 0 var(--sp-sm);
    color: var(--text-muted);
  }

  /* ─── Launcher ─── */
  .launcher {
    margin-bottom: var(--sp-lg);
  }

  .form-row {
    display: flex;
    gap: var(--sp-md);
    align-items: end;
    flex-wrap: wrap;
  }

  .form-row label {
    display: flex;
    flex-direction: column;
    gap: var(--sp-xs);
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.02em;
  }

  .form-row input,
  .form-row select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: var(--radius-md);
    font-size: 13px;
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }

  .form-row input:focus,
  .form-row select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-bg);
  }

  /* ─── Buttons ─── */
  .btn {
    padding: 8px 16px;
    border: none;
    border-radius: var(--radius-md);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background var(--t-fast) var(--ease-out),
                transform var(--t-fast) var(--ease-out);
    letter-spacing: 0.02em;
  }
  .btn:hover {
    transform: translateY(-1px);
  }
  .btn:active {
    transform: scale(0.97);
  }
  .btn:disabled {
    opacity: 0.35;
    cursor: default;
    pointer-events: none;
    transform: none;
  }
  .btn-primary {
    background: var(--accent);
    color: #fff;
  }
  .btn-secondary {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover {
    background: var(--surface-hover);
  }

  /* ─── Alerts ─── */
  .alert {
    padding: 10px var(--sp-md);
    border-radius: var(--radius-md);
    font-size: 13px;
    margin-top: var(--sp-sm);
  }
  .alert-error {
    background: var(--red-bg);
    color: var(--red);
    border: 1px solid rgba(255, 107, 107, 0.25);
  }

  /* ─── Content grid ─── */
  .content-grid {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: var(--sp-md);
  }

  /* ─── Jobs panel ─── */
  .jobs-panel {
    border-right: 1px solid var(--border);
    padding-right: var(--sp-md);
  }

  .job-card {
    display: block;
    width: 100%;
    text-align: left;
    padding: 10px 12px;
    background: none;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    cursor: pointer;
    margin-bottom: var(--sp-xs);
    color: var(--text);
    transition: background var(--t-fast) var(--ease-out),
                border-color var(--t-fast) var(--ease-out);
  }
  .job-card:hover {
    background: var(--surface);
  }
  .job-card.active {
    border-color: var(--accent);
    background: var(--accent-bg);
  }

  .job-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: -0.01em;
  }
  .job-meta {
    display: flex;
    gap: var(--sp-sm);
    align-items: center;
    margin-top: var(--sp-xs);
  }
  .job-time {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* ─── Status pills ─── */
  .status-pill {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: var(--radius-sm, 4px);
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  .status-pill.running {
    background: var(--accent-bg);
    color: var(--accent);
  }
  .status-pill.done {
    background: var(--green-bg);
    color: var(--green);
  }
  .status-pill.failed {
    background: var(--red-bg);
    color: var(--red);
  }
  .status-pill.cancelled {
    background: var(--yellow-bg);
    color: var(--yellow);
  }

  /* ─── Progress ─── */
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--sp-sm);
  }

  .stderr-log {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-md);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    line-height: 1.6;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: var(--text-muted);
  }

  /* ─── Stats grid ─── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--sp-md);
    margin-bottom: var(--sp-md);
  }

  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-md);
    transition: border-color var(--t-fast) var(--ease-out);
  }
  .stat:hover {
    border-color: var(--border-subtle);
  }

  .stat-label {
    display: block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: var(--sp-xs);
  }
  .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  /* ─── Equity chart ─── */
  .equity-chart {
    margin-bottom: var(--sp-md);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-md);
  }

  /* ─── Per-symbol table ─── */
  .symbol-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .symbol-table th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .symbol-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }
  .symbol-table tbody tr {
    transition: background var(--t-fast);
  }
  .symbol-table tbody tr:hover {
    background: var(--surface);
  }

  /* ─── Empty state ─── */
  .empty {
    color: var(--text-muted);
    font-style: italic;
    padding: var(--sp-lg) 0;
  }

  /* ─── Mobile ─── */
  @media (max-width: 768px) {
    .content-grid {
      grid-template-columns: 1fr;
    }

    .jobs-panel {
      border-right: none;
      border-bottom: 1px solid var(--border);
      padding-right: 0;
      padding-bottom: var(--sp-md);
    }

    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
      gap: var(--sp-sm);
    }

    .form-row {
      flex-direction: column;
      gap: var(--sp-sm);
    }

    .form-row label {
      width: 100%;
    }

    .form-row input,
    .form-row select {
      width: 100%;
    }

    .btn {
      padding: 12px 16px;
      width: 100%;
      text-align: center;
    }

    .symbol-table {
      font-size: 12px;
    }
    .symbol-table th,
    .symbol-table td {
      padding: 6px 8px;
    }
  }
</style>
