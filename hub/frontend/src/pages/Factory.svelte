<script lang="ts">
  import { getFactoryRuns, getFactoryRun, getFactoryReport, getFactoryCandidates } from '../lib/api';

  let runs: any[] = $state([]);
  let loading = $state(true);
  let error = $state('');

  let selectedRun: any = $state(null);
  let runDetail: any = $state(null);
  let report: any = $state(null);
  let candidates: any[] = $state([]);
  let loadingDetail = $state(false);

  async function loadRuns() {
    loading = true;
    try {
      runs = await getFactoryRuns();
    } catch (e: any) {
      error = e.message || 'Failed to load runs';
    }
    loading = false;
  }

  async function selectRun(run: any) {
    selectedRun = run;
    loadingDetail = true;
    report = null;
    candidates = [];
    runDetail = null;
    try {
      const [detail, rep, cands] = await Promise.all([
        getFactoryRun(run.date, run.run_id),
        run.has_report ? getFactoryReport(run.date, run.run_id).catch(() => null) : Promise.resolve(null),
        getFactoryCandidates(run.date, run.run_id).catch(() => []),
      ]);
      runDetail = detail;
      report = rep;
      candidates = cands;
    } catch (e: any) {
      error = e.message;
    }
    loadingDetail = false;
  }

  function fmtNum(n: any, d = 2): string {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(d);
  }

  $effect(() => { loadRuns(); });
</script>

<div class="factory-page">
  <h1>Factory Candidates</h1>

  {#if error}
    <div class="alert alert-error">{error}</div>
  {/if}

  <div class="content-grid">
    <!-- Runs list -->
    <div class="runs-panel">
      <h2>Runs</h2>
      {#if loading}
        <div class="empty">Loading...</div>
      {:else if runs.length === 0}
        <div class="empty">No factory runs found</div>
      {:else}
        {#each runs as r (r.date + '/' + r.run_id)}
          <button
            class="run-card"
            class:active={selectedRun?.run_id === r.run_id && selectedRun?.date === r.date}
            onclick={() => selectRun(r)}
          >
            <div class="run-date">{r.date}</div>
            <div class="run-id">{r.run_id}</div>
            <div class="run-meta">
              <span class="profile-pill">{r.profile}</span>
              {#if r.has_report}
                <span class="report-badge">report</span>
              {/if}
            </div>
          </button>
        {/each}
      {/if}
    </div>

    <!-- Detail panel -->
    <div class="detail-panel">
      {#if loadingDetail}
        <div class="empty">Loading run detail...</div>
      {:else if selectedRun && runDetail}
        <h2>{selectedRun.run_id}</h2>
        <div class="meta-section">
          <div class="meta-row"><span class="meta-label">Date</span><span>{selectedRun.date}</span></div>
          <div class="meta-row"><span class="meta-label">Profile</span><span>{runDetail.metadata?.args?.profile ?? '—'}</span></div>
          <div class="meta-row"><span class="meta-label">Candidates</span><span>{runDetail.metadata?.args?.num_candidates ?? '—'}</span></div>
          <div class="meta-row"><span class="meta-label">Subdirs</span><span>{runDetail.subdirs?.join(', ') ?? '—'}</span></div>
        </div>

        {#if report}
          <h3>Report</h3>
          {#if report.candidates && Array.isArray(report.candidates)}
            <table class="candidates-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Config</th>
                  <th>Score</th>
                  <th>PnL</th>
                  <th>Trades</th>
                  <th>Win%</th>
                  <th>DD%</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {#each report.candidates as c, i}
                  <tr>
                    <td>{i + 1}</td>
                    <td class="mono">{c.config_id ?? c.id ?? '—'}</td>
                    <td>{fmtNum(c.score_v1 ?? c.score)}</td>
                    <td>${fmtNum(c.pnl ?? c.net_pnl)}</td>
                    <td>{c.trades ?? c.total_trades ?? '—'}</td>
                    <td>{fmtNum((c.win_rate ?? 0) * 100, 1)}%</td>
                    <td>{fmtNum(c.max_drawdown_pct ?? c.drawdown_pct, 1)}%</td>
                    <td>
                      <span class="verdict {c.verdict === 'PASS' || c.verdict === 'pass' ? 'pass' : 'fail'}">
                        {c.verdict ?? '—'}
                      </span>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <pre class="report-json">{JSON.stringify(report, null, 2)}</pre>
          {/if}
        {/if}

        {#if candidates.length > 0}
          <h3>Config Files</h3>
          <div class="config-list">
            {#each candidates as c}
              <div class="config-item">
                <span class="mono">{c.filename}</span>
                <span class="config-size">{(c.size / 1024).toFixed(1)} KB</span>
              </div>
            {/each}
          </div>
        {/if}
      {:else}
        <div class="empty">Select a factory run to view details</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .factory-page { max-width: 1400px; }

  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  h3 { font-size: 14px; font-weight: 500; margin: 16px 0 8px; color: var(--text-muted); }

  .alert { padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; }
  .alert-error { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

  .content-grid { display: grid; grid-template-columns: 280px 1fr; gap: 16px; }

  .runs-panel { border-right: 1px solid var(--border); padding-right: 16px; max-height: 80vh; overflow-y: auto; }

  .run-card {
    display: block; width: 100%; text-align: left;
    padding: 8px 10px; background: none; border: 1px solid transparent;
    border-radius: 6px; cursor: pointer; margin-bottom: 4px; color: var(--text);
  }
  .run-card:hover { background: var(--surface); }
  .run-card.active { border-color: var(--accent, #3b82f6); background: var(--surface); }

  .run-date { font-size: 11px; color: var(--text-muted); }
  .run-id { font-family: monospace; font-size: 12px; margin-top: 2px; }
  .run-meta { display: flex; gap: 6px; margin-top: 4px; }

  .profile-pill { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: rgba(59,130,246,0.2); color: #60a5fa; }
  .report-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: rgba(34,197,94,0.2); color: #4ade80; }

  .meta-section { margin-bottom: 16px; }
  .meta-row { display: flex; gap: 12px; padding: 4px 0; font-size: 13px; }
  .meta-label { color: var(--text-muted); min-width: 100px; }

  .candidates-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .candidates-table th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; }
  .candidates-table td { padding: 8px 10px; border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.05)); }

  .mono { font-family: monospace; font-size: 12px; }

  .verdict { font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: 500; }
  .verdict.pass { background: rgba(34,197,94,0.2); color: #4ade80; }
  .verdict.fail { background: rgba(239,68,68,0.2); color: #f87171; }

  .report-json {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; font-size: 12px;
    max-height: 500px; overflow: auto; white-space: pre-wrap;
  }

  .config-list { display: flex; flex-direction: column; gap: 4px; }
  .config-item { display: flex; justify-content: space-between; padding: 4px 8px; font-size: 13px; }
  .config-size { color: var(--text-muted); font-size: 12px; }

  .empty { color: var(--text-muted); font-style: italic; padding: 20px 0; }
</style>
