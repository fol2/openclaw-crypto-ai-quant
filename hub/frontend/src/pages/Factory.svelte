<script lang="ts">
  import { onMount } from 'svelte';
  import {
    factoryTimerAction,
    getFactoryCapability,
    getFactoryCandidates,
    getFactoryReport,
    getFactoryRun,
    getFactoryRuns,
    getFactorySettings,
    getFactoryTimer,
    putFactorySettings,
    runFactory,
  } from '../lib/api';

  let capability = $state<any | null>(null);
  let runs: any[] = $state([]);
  let selectedRun: any = $state(null);
  let selectedReport: any = $state(null);
  let selectedCandidates: any[] = $state([]);
  let settingsText = $state('{}');
  let timers: any[] = $state([]);
  let loading = $state(true);
  let loadingRun = $state(false);
  let saving = $state(false);
  let actionError = $state('');
  let actionSuccess = $state('');

  async function refreshOverview() {
    loading = true;
    actionError = '';
    try {
      const [cap, runList, settings, timerState] = await Promise.all([
        getFactoryCapability(),
        getFactoryRuns().catch(() => []),
        getFactorySettings().catch(() => ({})),
        getFactoryTimer().catch(() => ({ timers: [] })),
      ]);
      capability = cap;
      runs = runList;
      settingsText = JSON.stringify(settings, null, 2);
      timers = timerState.timers ?? [];

      if (!selectedRun && runs.length > 0) {
        await selectRun(runs[0].date, runs[0].run_id);
      }
    } catch (e: any) {
      actionError = e.message || 'Failed to load factory state';
    }
    loading = false;
  }

  async function selectRun(date: string, runId: string) {
    loadingRun = true;
    actionError = '';
    try {
      const [detail, report, candidates] = await Promise.all([
        getFactoryRun(date, runId),
        getFactoryReport(date, runId).catch(() => null),
        getFactoryCandidates(date, runId).catch(() => []),
      ]);
      selectedRun = detail;
      selectedReport = report;
      selectedCandidates = candidates;
    } catch (e: any) {
      actionError = e.message || 'Failed to load run';
    }
    loadingRun = false;
  }

  async function launchFactory() {
    if (!capability?.execution_enabled) return;
    actionError = '';
    actionSuccess = '';
    try {
      const result = await runFactory({});
      actionSuccess = `Factory job started: ${result.job_id ?? 'submitted'}`;
      await refreshOverview();
    } catch (e: any) {
      actionError = e.message || 'Factory launch failed';
    }
  }

  async function saveSettings() {
    if (!capability?.execution_enabled) return;
    saving = true;
    actionError = '';
    actionSuccess = '';
    try {
      await putFactorySettings(JSON.parse(settingsText));
      actionSuccess = 'Factory settings saved';
      await refreshOverview();
    } catch (e: any) {
      actionError = e.message || 'Invalid settings JSON';
    }
    saving = false;
  }

  async function applyTimer(action: 'enable' | 'disable') {
    if (!capability?.execution_enabled) return;
    actionError = '';
    actionSuccess = '';
    try {
      await factoryTimerAction(action);
      actionSuccess = `Factory timers ${action}d`;
      await refreshOverview();
    } catch (e: any) {
      actionError = e.message || `Timer ${action} failed`;
    }
  }

  onMount(async () => {
    await refreshOverview();
  });
</script>

<div class="factory-page">
  <div class="page-header">
    <div>
      <h1>Factory</h1>
      <p class="subtitle">Dormant contract for future factory reactivation. Historical artefacts stay readable even when execution is disabled.</p>
    </div>
    <div class="header-actions">
      <button class="btn btn-primary" onclick={launchFactory} disabled={!capability?.execution_enabled}>
        Run Factory
      </button>
      <button class="btn btn-secondary" onclick={() => applyTimer('enable')} disabled={!capability?.execution_enabled}>
        Enable Timers
      </button>
      <button class="btn btn-secondary" onclick={() => applyTimer('disable')} disabled={!capability?.execution_enabled}>
        Disable Timers
      </button>
    </div>
  </div>

  {#if actionError}
    <div class="alert alert-error">{actionError}</div>
  {/if}
  {#if actionSuccess}
    <div class="alert alert-success">{actionSuccess}</div>
  {/if}

  {#if loading}
    <div class="empty">Loading factory capability…</div>
  {:else}
    <div class="capability-card">
      <div class="capability-meta">
        <span class="status-chip" class:enabled={capability?.execution_enabled} class:dormant={!capability?.execution_enabled}>
          {capability?.mode ?? 'unknown'}
        </span>
        <span class="mono">feature: {capability?.compiled ? 'factory' : 'not compiled'}</span>
        <span class="mono">policy: {capability?.policy_enabled ? 'enabled' : 'disabled'}</span>
      </div>
      <p>{capability?.reason}</p>
      <div class="capability-grid">
        <div>
          <div class="field-label">Enable gate</div>
          <code>{capability?.enable_env ?? 'AI_QUANT_FACTORY_ENABLE'}</code>
        </div>
        <div>
          <div class="field-label">Settings path</div>
          <code>{capability?.settings_path ?? 'config/factory_defaults.yaml'}</code>
        </div>
        <div>
          <div class="field-label">Service units</div>
          <code>{(capability?.service_units ?? []).join(', ') || 'none'}</code>
        </div>
      </div>
    </div>

    <div class="content-grid">
      <section class="panel">
        <div class="panel-header">
          <h2>Historical Runs</h2>
          <span class="count">{runs.length}</span>
        </div>
        {#if runs.length === 0}
          <div class="empty">No factory artefacts found under `artifacts/`.</div>
        {:else}
          <div class="runs-list">
            {#each runs as run (run.date + run.run_id)}
              <button
                class="run-card"
                class:selected={selectedRun?.run_id === run.run_id && selectedRun?.date === run.date}
                onclick={() => selectRun(run.date, run.run_id)}
              >
                <div class="run-id">{run.run_id}</div>
                <div class="run-meta">
                  <span>{run.date}</span>
                  <span>{run.profile ?? 'unknown'}</span>
                </div>
              </button>
            {/each}
          </div>
        {/if}
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>Run Detail</h2>
          {#if loadingRun}<span class="count">Loading…</span>{/if}
        </div>
        {#if selectedRun}
          <div class="detail-grid">
            <div>
              <div class="field-label">Run</div>
              <div class="mono">{selectedRun.date}/{selectedRun.run_id}</div>
            </div>
            <div>
              <div class="field-label">Subdirectories</div>
              <div class="mono">{(selectedRun.subdirs ?? []).join(', ') || '—'}</div>
            </div>
          </div>

          <div class="stack">
            <div>
              <div class="field-label">Metadata</div>
              <pre>{JSON.stringify(selectedRun.metadata ?? {}, null, 2)}</pre>
            </div>
            <div>
              <div class="field-label">Report</div>
              <pre>{JSON.stringify(selectedReport ?? {}, null, 2)}</pre>
            </div>
            <div>
              <div class="field-label">Candidates</div>
              <pre>{JSON.stringify(selectedCandidates ?? [], null, 2)}</pre>
            </div>
          </div>
        {:else}
          <div class="empty">Select a historical run to inspect its metadata, report, and candidates.</div>
        {/if}
      </section>
    </div>

    <div class="content-grid secondary">
      <section class="panel">
        <div class="panel-header">
          <h2>Settings</h2>
          <button class="btn btn-secondary" onclick={saveSettings} disabled={!capability?.execution_enabled || saving}>
            {saving ? 'Saving…' : 'Save Defaults'}
          </button>
        </div>
        <textarea bind:value={settingsText} rows="18" disabled={!capability?.execution_enabled}></textarea>
        {#if !capability?.execution_enabled}
          <p class="hint">Settings stay visible while the factory contract is dormant. Mutations are blocked until a `factory` build is deployed with `AI_QUANT_FACTORY_ENABLE=1`.</p>
        {/if}
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>Timer State</h2>
          <span class="count">{timers.length}</span>
        </div>
        {#if timers.length === 0}
          <div class="empty">No factory timers reported.</div>
        {:else}
          <div class="timer-list">
            {#each timers as timer}
              <div class="timer-card">
                <div class="timer-name mono">{timer.unit ?? timer.name}</div>
                <div class="timer-meta">
                  <span>active: <strong>{timer.active || 'unknown'}</strong></span>
                  <span>enabled: <strong>{timer.enabled ? 'yes' : 'no'}</strong></span>
                  <span>mode: <strong>{timer.mode || capability?.mode || 'unknown'}</strong></span>
                </div>
                <div class="timer-next mono">{timer.next_trigger || 'No scheduled trigger recorded'}</div>
              </div>
            {/each}
          </div>
        {/if}
      </section>
    </div>
  {/if}
</div>

<style>
  .factory-page {
    max-width: 1440px;
    animation: slideUp 0.3s ease;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 16px;
  }

  h1 {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }

  h2 {
    font-size: 15px;
    font-weight: 600;
  }

  .subtitle {
    font-size: 13px;
    color: var(--text-muted);
    max-width: 760px;
    line-height: 1.5;
  }

  .header-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
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

  .alert-success {
    background: var(--green-bg);
    color: var(--green);
    border: 1px solid rgba(81, 207, 102, 0.2);
  }

  .capability-card,
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px;
  }

  .capability-card {
    margin-bottom: 16px;
  }

  .capability-meta,
  .panel-header,
  .timer-meta,
  .run-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
  }

  .capability-meta {
    margin-bottom: 10px;
  }

  .status-chip {
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .status-chip.enabled {
    background: var(--green-bg);
    color: var(--green);
  }

  .status-chip.dormant {
    background: var(--accent-bg);
    color: var(--accent);
  }

  .capability-grid,
  .detail-grid,
  .content-grid {
    display: grid;
    gap: 12px;
  }

  .capability-grid {
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin-top: 12px;
  }

  .content-grid {
    grid-template-columns: minmax(280px, 360px) 1fr;
    margin-bottom: 16px;
  }

  .content-grid.secondary {
    grid-template-columns: 1fr 1fr;
  }

  .runs-list,
  .timer-list,
  .stack {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .run-card,
  .timer-card {
    width: 100%;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px;
    text-align: left;
    transition: border-color var(--t-fast), background var(--t-fast);
  }

  .run-card.selected {
    border-color: rgba(58, 134, 255, 0.25);
    background: var(--accent-bg);
  }

  .run-id,
  .mono,
  code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }

  .run-id {
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
    word-break: break-all;
  }

  .run-meta,
  .timer-meta,
  .hint,
  .field-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .field-label {
    display: block;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  pre,
  textarea {
    width: 100%;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px;
    color: var(--text);
    font-size: 12px;
    line-height: 1.5;
    font-family: 'IBM Plex Mono', monospace;
    overflow: auto;
  }

  textarea {
    resize: vertical;
    min-height: 280px;
  }

  .empty {
    padding: 18px;
    border: 1px dashed var(--border);
    border-radius: var(--radius-md);
    font-size: 13px;
    color: var(--text-muted);
    text-align: center;
  }

  .count {
    font-size: 11px;
    color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
  }

  .btn {
    padding: 8px 14px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    transition: all var(--t-fast);
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--accent);
    color: white;
    border-color: transparent;
  }

  .btn-secondary {
    background: var(--bg-secondary);
    color: var(--text);
  }

  @media (max-width: 960px) {
    .page-header,
    .content-grid,
    .content-grid.secondary {
      grid-template-columns: 1fr;
    }

    .header-actions {
      width: 100%;
      justify-content: flex-start;
    }
  }
</style>
