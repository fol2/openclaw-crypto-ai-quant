<script lang="ts">
  import { runSweep, getSweepJobs, getSweepStatus, getSweepResults, cancelSweep } from '../lib/api';
  import { CANDIDATE_FAMILY_ORDER, getConfigLabel, LIVE_MODE } from '../lib/mode-labels';

  type SweepJob = {
    id: string;
    status: string;
    created_at?: string | null;
    finished_at?: string | null;
    error?: string | null;
    stderr_tail?: string[];
  };

  type SweepResultRow = Record<string, any> & {
    config_id?: string;
    id?: string;
    output_mode?: string;
    candidate_mode?: boolean;
  };

  const candidateConfigs = CANDIDATE_FAMILY_ORDER;
  const sweepSpecPlaceholder = 'backtester/sweeps/smoke.yaml';

  let config = $state('main');
  let sweepSpec = $state('');
  let balance = $state(10000);
  let launching = $state(false);
  let error = $state('');

  let jobs: SweepJob[] = $state([]);
  let activeJob: string | null = $state(null);
  let activeStatus: SweepJob | null = $state(null);
  let activeResults: unknown = $state(null);
  let stderrLines: string[] = $state([]);

  let pollTimer: ReturnType<typeof setInterval> | null = null;

  function isRecord(value: unknown): value is Record<string, any> {
    return value != null && typeof value === 'object' && !Array.isArray(value);
  }

  function normaliseResults(value: unknown): SweepResultRow[] {
    if (Array.isArray(value)) {
      return value.filter(isRecord) as SweepResultRow[];
    }
    if (!isRecord(value)) {
      return [];
    }
    if ('config_id' in value || 'total_pnl' in value || 'net_pnl' in value || 'overrides' in value) {
      return [value as SweepResultRow];
    }
    const nested = value.results ?? value.rows ?? value.data;
    return Array.isArray(nested) ? nested.filter(isRecord) as SweepResultRow[] : [];
  }

  function resultRows() {
    return normaliseResults(activeResults);
  }

  function readMetric(row: SweepResultRow, key: string) {
    if (row[key] != null) {
      return row[key];
    }
    if (isRecord(row.report) && row.report[key] != null) {
      return row.report[key];
    }
    return undefined;
  }

  function getResultConfigId(row: SweepResultRow): string {
    return String(readMetric(row, 'config_id') ?? row.id ?? '—');
  }

  function getResultOutputMode(row: SweepResultRow): string {
    return String(readMetric(row, 'output_mode') ?? (row.candidate_mode ? 'candidate' : 'full'));
  }

  function getTotalPnl(row: SweepResultRow) {
    return readMetric(row, 'total_pnl') ?? readMetric(row, 'net_pnl');
  }

  function getWinRate(row: SweepResultRow) {
    return readMetric(row, 'win_rate');
  }

  function getMaxDrawdown(row: SweepResultRow) {
    return readMetric(row, 'max_drawdown_pct');
  }

  function getSharpe(row: SweepResultRow) {
    return readMetric(row, 'sharpe_ratio');
  }

  function getTrades(row: SweepResultRow) {
    return readMetric(row, 'total_trades');
  }

  async function launch() {
    const trimmedSweepSpec = sweepSpec.trim();
    if (!trimmedSweepSpec) { error = 'Sweep spec path required'; return; }
    launching = true;
    error = '';
    clearActiveState();
    try {
      const res = await runSweep({
        config,
        sweep_spec: trimmedSweepSpec,
        initial_balance: balance,
      });
      activeJob = res.job_id;
      activeStatus = { id: res.job_id, status: res.status, stderr_tail: [] };
      startPolling();
      await refreshJobs();
    } catch (e: any) {
      error = e.message || 'Failed to launch';
    }
    launching = false;
  }

  function clearActiveState() {
    activeStatus = null;
    activeResults = null;
    stderrLines = [];
  }

  async function refreshJobs() {
    try {
      jobs = await getSweepJobs();
      if (activeJob) {
        const summary = jobs.find((job) => job.id === activeJob);
        if (summary && activeStatus?.status !== 'running') {
          activeStatus = { ...summary, stderr_tail: activeStatus?.stderr_tail ?? [] };
        }
      }
    } catch {}
  }

  async function pollActiveJob() {
    if (!activeJob) return;
    const jobId = activeJob;
    try {
      const s = await getSweepStatus(jobId);
      if (activeJob !== jobId) return;
      activeStatus = s;
      stderrLines = s.stderr_tail || [];
      if (s.status !== 'running') {
        stopPolling();
        if (s.status === 'done') {
          try {
            const results = await getSweepResults(jobId);
            if (activeJob === jobId) {
              activeResults = results;
            }
          } catch (e: any) {
            if (activeJob === jobId) {
              activeStatus = { ...s, error: e.message || 'Failed to load sweep results' };
            }
          }
        }
        await refreshJobs();
      }
    } catch {}
  }

  function startPolling() {
    stopPolling();
    void pollActiveJob();
    pollTimer = setInterval(() => {
      void pollActiveJob();
    }, 3000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function cancel() {
    if (!activeJob) return;
    const jobId = activeJob;
    try {
      await cancelSweep(jobId);
      activeStatus = {
        ...(activeStatus ?? { id: jobId }),
        status: 'cancelled',
        error: null,
        finished_at: new Date().toISOString(),
      };
      stopPolling();
      await refreshJobs();
    } catch {}
  }

  async function selectJob(id: string) {
    activeJob = id;
    clearActiveState();
    stopPolling();
    try {
      const status = await getSweepStatus(id);
      if (activeJob !== id) return;
      activeStatus = status;
      stderrLines = status.stderr_tail || [];

      if (status.status === 'running') {
        startPolling();
        return;
      }

      if (status.status === 'done') {
        try {
          const results = await getSweepResults(id);
          if (activeJob === id) {
            activeResults = results;
          }
        } catch (e: any) {
          if (activeJob === id) {
            activeStatus = {
              ...status,
              error: e.message || 'Failed to load sweep results',
            };
          }
        }
      }
    } catch (e: any) {
      if (activeJob === id) {
        activeStatus = {
          id,
          status: 'failed',
          error: e.message || 'Failed to load sweep status',
          stderr_tail: [],
        };
      }
    }
  }

  function formatJobTime(timestamp?: string | null) {
    return timestamp ? new Date(timestamp).toLocaleTimeString() : '';
  }

  function hasActiveResultRows() {
    return resultRows().length > 0;
  }

  function fmtNum(n: any, d = 2): string {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(d);
  }

  function fmtPctFraction(n: any, d = 1): string {
    if (n == null || isNaN(n)) return '—';
    return `${(Number(n) * 100).toFixed(d)}%`;
  }

  $effect(() => {
    void refreshJobs();
    return () => stopPolling();
  });
</script>

<div class="sweep-page">
  <h1>Parameter Sweep</h1>

  <div class="launcher">
    <p class="launcher-copy">
      Runs the Rust backtester sweep and reads the per-job JSONL artifact. Use a repo-relative spec such as <span class="mono">{sweepSpecPlaceholder}</span>.
    </p>
    <div class="form-row">
      <label>Config <select bind:value={config}>
        <option value="main">{getConfigLabel('main')}</option>
        <optgroup label="Live Engine">
          <option value={LIVE_MODE}>{getConfigLabel(LIVE_MODE)}</option>
        </optgroup>
        <optgroup label="Candidate Family">
          {#each candidateConfigs as option}
            <option value={option}>{getConfigLabel(option)}</option>
          {/each}
        </optgroup>
      </select></label>
      <label>Balance <input type="number" bind:value={balance} min="100" step="100" /></label>
      <label class="wide">Sweep Spec <input type="text" bind:value={sweepSpec} placeholder={sweepSpecPlaceholder} /></label>
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
          <button class="job-card" class:active={activeJob === j.id} onclick={() => void selectJob(j.id)}>
            <div class="job-id">{j.id.slice(0, 8)}</div>
            <div class="job-meta">
              <span class="status-pill {j.status}">{j.status}</span>
              <span class="job-time">{formatJobTime(j.created_at)}</span>
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
      {:else if hasActiveResultRows()}
        <div class="result-section">
          <h2>Sweep Results</h2>
          {#if activeStatus}
            <div class="status-summary">
              <span class="status-pill {activeStatus.status}">{activeStatus.status}</span>
              <span class="status-meta">{formatJobTime(activeStatus.finished_at ?? activeStatus.created_at)}</span>
            </div>
          {/if}
          <table class="results-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Config ID</th>
                <th>Output</th>
                <th>Total PnL</th>
                <th>Win Rate</th>
                <th>Max DD</th>
                <th>Sharpe</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              {#each resultRows() as r, i}
                <tr>
                  <td>{i + 1}</td>
                  <td class="mono">{getResultConfigId(r)}</td>
                  <td>{getResultOutputMode(r)}</td>
                  <td>${fmtNum(getTotalPnl(r))}</td>
                  <td>{fmtPctFraction(getWinRate(r))}</td>
                  <td>{fmtPctFraction(getMaxDrawdown(r))}</td>
                  <td>{fmtNum(getSharpe(r))}</td>
                  <td>{fmtNum(getTrades(r), 0)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
          {#if stderrLines.length > 0}
            <pre class="stderr-log">{stderrLines.join('\n')}</pre>
          {/if}
        </div>
      {:else if activeStatus?.status === 'cancelled'}
        <div class="result-section">
          <div class="status-summary">
            <span class="status-pill cancelled">cancelled</span>
            <span class="status-meta">{formatJobTime(activeStatus.finished_at)}</span>
          </div>
          <div class="empty">This sweep was cancelled before results were produced.</div>
          {#if stderrLines.length > 0}
            <pre class="stderr-log">{stderrLines.join('\n')}</pre>
          {/if}
        </div>
      {:else if activeStatus?.error}
        <div class="alert alert-error">{activeStatus.error}</div>
        {#if stderrLines.length > 0}
          <pre class="stderr-log">{stderrLines.join('\n')}</pre>
        {/if}
      {:else if activeStatus?.status === 'done'}
        <div class="result-section">
          <div class="status-summary">
            <span class="status-pill done">done</span>
            <span class="status-meta">{formatJobTime(activeStatus.finished_at)}</span>
          </div>
          <div class="empty">This sweep completed without any result rows.</div>
          {#if activeResults}
            <pre class="result-json">{JSON.stringify(activeResults, null, 2)}</pre>
          {/if}
          {#if stderrLines.length > 0}
            <pre class="stderr-log">{stderrLines.join('\n')}</pre>
          {/if}
        </div>
      {:else if activeResults}
        <pre class="result-json">{JSON.stringify(activeResults, null, 2)}</pre>
      {:else}
        <div class="empty">Select a job or run a new sweep</div>
      {/if}
    </div>
  </div>
</div>

<style>
  /* ─── Page entry ─── */
  .sweep-page {
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

  /* ─── Launcher ─── */
  .launcher {
    margin-bottom: var(--sp-lg);
  }

  .launcher-copy {
    margin: 0 0 var(--sp-sm);
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.5;
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

  .form-row label.wide {
    flex: 1;
    min-width: 200px;
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

  .status-summary {
    display: flex;
    gap: var(--sp-sm);
    align-items: center;
    margin-bottom: var(--sp-sm);
  }

  .status-meta {
    color: var(--text-muted);
    font-size: 12px;
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
    margin-top: var(--sp-sm);
  }

  /* ─── Results table ─── */
  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .results-table th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .results-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }
  .results-table tbody tr {
    transition: background var(--t-fast);
  }
  .results-table tbody tr:hover {
    background: var(--surface);
  }

  .mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: -0.01em;
  }

  /* ─── Result JSON fallback ─── */
  .result-json {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-md);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    max-height: 600px;
    overflow: auto;
    white-space: pre-wrap;
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

    .form-row {
      flex-direction: column;
      gap: var(--sp-sm);
    }

    .form-row label,
    .form-row label.wide {
      width: 100%;
      flex: unset;
      min-width: unset;
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

    .results-table {
      font-size: 12px;
    }
    .results-table th,
    .results-table td {
      padding: 6px 8px;
    }

    /* Horizontal scroll for wide tables */
    .result-section {
      overflow-x: auto;
    }
  }
</style>
