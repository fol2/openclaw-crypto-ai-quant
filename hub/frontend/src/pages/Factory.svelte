<script lang="ts">
  import {
    getFactoryCapability,
    getFactoryRuns,
    getFactoryRun,
    getFactoryReport,
    getFactoryCandidates,
    getFactorySettings,
    getFactoryTimer,
  } from '../lib/api';

  let loading = $state(true);
  let error = $state('');
  let capability: any = $state(null);
  let runs: any[] = $state([]);
  let settings: any = $state(null);
  let timerInfo: any = $state(null);
  let selectedRun: any = $state(null);
  let selectedReport: any = $state(null);
  let selectedCandidates: any[] = $state([]);
  let selecting = $state(false);

  async function refresh() {
    loading = true;
    error = '';
    try {
      const [cap, runRows, savedSettings, timers] = await Promise.all([
        getFactoryCapability(),
        getFactoryRuns(),
        getFactorySettings().catch(() => ({})),
        getFactoryTimer().catch(() => ({ timers: [] })),
      ]);
      capability = cap;
      runs = Array.isArray(runRows) ? runRows : [];
      settings = savedSettings;
      timerInfo = timers;
    } catch (e: any) {
      error = e.message || 'Failed to load factory state';
    }
    loading = false;
  }

  async function openRun(date: string, runId: string) {
    selecting = true;
    try {
      const [detail, report, candidates] = await Promise.all([
        getFactoryRun(date, runId),
        getFactoryReport(date, runId).catch(() => null),
        getFactoryCandidates(date, runId).catch(() => []),
      ]);
      selectedRun = detail;
      selectedReport = report;
      selectedCandidates = Array.isArray(candidates) ? candidates : [];
    } catch (e: any) {
      error = e.message || 'Failed to load factory artefacts';
    }
    selecting = false;
  }

  function fmtJson(value: any): string {
    return JSON.stringify(value ?? {}, null, 2);
  }

  $effect(() => {
    refresh();
  });
</script>

<div class="factory-page">
  <div class="header">
    <div>
      <h1>Factory</h1>
      <p class="subhead">Dormant contract preserved for future reactivation, with read-only artefact access available now.</p>
    </div>
    {#if capability}
      <span class="mode-pill" class:enabled={capability.execution_enabled}>
        {capability.mode}
      </span>
    {/if}
  </div>

  {#if error}
    <div class="alert alert-error">{error}</div>
  {/if}

  {#if loading}
    <div class="empty">Loading factory state…</div>
  {:else}
    <div class="hero-grid">
      <section class="card">
        <h2>Capability</h2>
        {#if capability}
          <dl class="keyvals">
            <div><dt>Compiled</dt><dd>{capability.compiled ? 'yes' : 'no'}</dd></div>
            <div><dt>Policy gate</dt><dd>{capability.policy_enabled ? 'enabled' : 'disabled'}</dd></div>
            <div><dt>Execution</dt><dd>{capability.execution_enabled ? 'enabled' : 'dormant'}</dd></div>
            <div><dt>Enable switch</dt><dd class="mono">{capability.enable_env}=1</dd></div>
          </dl>
          <p class="reason">{capability.reason}</p>
        {/if}
      </section>

      <section class="card">
        <h2>Timer Contract</h2>
        {#if timerInfo?.timers?.length}
          <div class="timer-list">
            {#each timerInfo.timers as timer}
              <div class="timer-row">
                <span class="mono">{timer.timer}</span>
                <span class="timer-state">{timer.active || timer.load || 'unknown'}</span>
              </div>
            {/each}
          </div>
        {:else}
          <div class="empty-inline">No factory timers are active in this Hub context.</div>
        {/if}
      </section>

      <section class="card">
        <h2>Settings Contract</h2>
        <pre class="code-block">{fmtJson(settings)}</pre>
      </section>
    </div>

    <section class="card action-card">
      <div>
        <h2>Execution Surface</h2>
        <p>Run, cancel, save settings, and timer actions stay fail-closed until a `factory` build is deployed and policy explicitly enables it.</p>
      </div>
      <div class="actions">
        <button class="btn btn-primary" disabled={!capability || !capability.execution_enabled}>Run Factory</button>
        <button class="btn" disabled={!capability || !capability.execution_enabled}>Save Settings</button>
        <button class="btn" disabled={!capability || !capability.execution_enabled}>Enable Timer</button>
      </div>
    </section>

    <div class="content-grid">
      <section class="card">
        <div class="section-head">
          <h2>Run Archive</h2>
          <button class="btn" onclick={refresh}>Refresh</button>
        </div>
        {#if runs.length === 0}
          <div class="empty-inline">No factory artefacts found.</div>
        {:else}
          <div class="run-list">
            {#each runs as run}
              <button class="run-row" onclick={() => openRun(run.date, run.run_id)}>
                <div>
                  <div class="run-id mono">{run.run_id}</div>
                  <div class="run-meta">{run.date} · profile {run.profile || 'unknown'}</div>
                </div>
                <span class="candidate-count">{run.num_candidates ?? 0} candidates</span>
              </button>
            {/each}
          </div>
        {/if}
      </section>

      <section class="card detail-card">
        <h2>Selected Run</h2>
        {#if selecting}
          <div class="empty-inline">Loading artefacts…</div>
        {:else if selectedRun}
          <div class="detail-grid">
            <div>
              <h3>Metadata</h3>
              <pre class="code-block">{fmtJson(selectedRun.metadata)}</pre>
            </div>
            <div>
              <h3>Candidates</h3>
              {#if selectedCandidates.length === 0}
                <div class="empty-inline">No candidate files recorded.</div>
              {:else}
                <ul class="candidate-list">
                  {#each selectedCandidates as candidate}
                    <li class="mono">{candidate.filename}</li>
                  {/each}
                </ul>
              {/if}
            </div>
          </div>
          {#if selectedReport}
            <h3>Report</h3>
            <pre class="code-block">{fmtJson(selectedReport)}</pre>
          {/if}
        {:else}
          <div class="empty-inline">Select a run to inspect stored artefacts.</div>
        {/if}
      </section>
    </div>
  {/if}
</div>

<style>
  .factory-page {
    max-width: 1360px;
    animation: slideUp 0.3s ease;
  }

  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--sp-md);
    margin-bottom: var(--sp-md);
  }

  h1 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
  }

  h3 {
    font-size: 12px;
    font-weight: 600;
    margin: 0 0 8px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .subhead {
    max-width: 720px;
    color: var(--text-muted);
    font-size: 13px;
  }

  .mode-pill {
    border-radius: 999px;
    padding: 8px 12px;
    background: var(--amber-bg);
    color: var(--yellow);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .mode-pill.enabled {
    background: var(--green-bg);
    color: var(--green);
  }

  .hero-grid,
  .content-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  }

  .content-grid {
    grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px;
  }

  .action-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 12px;
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .btn {
    padding: 8px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--text);
    font-size: 12px;
    font-weight: 500;
  }

  .btn-primary {
    background: var(--accent-bg);
    color: var(--accent);
    border-color: transparent;
  }

  .btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .alert {
    padding: 10px 14px;
    border-radius: var(--radius-md);
    font-size: 13px;
    margin-bottom: 12px;
  }

  .alert-error {
    background: var(--red-bg);
    color: var(--red);
    border: 1px solid rgba(255, 107, 107, 0.2);
  }

  .keyvals {
    display: grid;
    gap: 8px;
    margin-bottom: 10px;
  }

  .keyvals div {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    font-size: 12px;
  }

  dt {
    color: var(--text-muted);
  }

  dd {
    margin: 0;
    color: var(--text);
    font-weight: 500;
  }

  .reason,
  .empty-inline,
  .empty {
    color: var(--text-muted);
    font-size: 12px;
  }

  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }

  .run-list,
  .timer-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .run-row,
  .timer-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
  }

  .run-row {
    width: 100%;
    text-align: left;
    color: inherit;
  }

  .run-id,
  .candidate-list,
  .code-block,
  .mono {
    font-family: 'IBM Plex Mono', monospace;
  }

  .run-meta,
  .candidate-count,
  .timer-state {
    color: var(--text-muted);
    font-size: 11px;
  }

  .detail-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);
    gap: 12px;
    margin-bottom: 12px;
  }

  .code-block {
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px;
    font-size: 11px;
    line-height: 1.5;
    max-height: 320px;
    overflow: auto;
  }

  .candidate-list {
    display: grid;
    gap: 6px;
    padding-left: 18px;
    font-size: 11px;
  }

  @media (max-width: 980px) {
    .hero-grid,
    .content-grid,
    .detail-grid,
    .action-card {
      grid-template-columns: 1fr;
      display: grid;
    }
  }
</style>
