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

  type JsonMap = Record<string, any>;

  interface FactoryCapability extends JsonMap {
    compiled?: boolean;
    policy_enabled?: boolean;
    executor_wired?: boolean;
    execution_enabled?: boolean;
    mode?: string;
    reason?: string;
    enable_env?: string;
    settings_path?: string;
    service_units?: string[];
  }

  interface FactoryRunSummary extends JsonMap {
    date: string;
    run_id: string;
    directory_name?: string;
    has_report?: boolean;
    profile?: string;
    candidate_count?: number | null;
    role_candidate_count?: number | null;
    selected_count?: number | null;
    selection_stage?: string | null;
    deploy_stage?: string | null;
  }

  interface FactorySelectionSummary extends JsonMap {
    selection_stage?: string;
    deploy_stage?: string;
    promotion_stage?: string;
    step5_gate_status?: string;
    deployed?: boolean;
    role_candidate_count?: number;
    selected_count?: number;
    challenge_count?: number;
    deployment_count?: number;
    selected_targets?: JsonMap[];
  }

  interface FactoryRunDetail extends JsonMap {
    date: string;
    run_id: string;
    directory_name?: string;
    metadata?: JsonMap;
    subdirs?: string[];
    report_available?: boolean;
    candidate_count?: number | null;
    selection_summary?: FactorySelectionSummary | null;
  }

  interface FactoryTimer extends JsonMap {
    unit?: string;
    active?: string;
    load?: string;
    enabled?: boolean;
    available?: boolean;
    unit_file_state?: string;
    mode?: string;
    next_trigger?: string;
  }

  type FactoryRunRef = Pick<FactoryRunSummary, 'date' | 'run_id'>;

  let capability = $state<FactoryCapability | null>(null);
  let runs = $state<FactoryRunSummary[]>([]);
  let selectedRunRef = $state<FactoryRunRef | null>(null);
  let selectedRun = $state<FactoryRunDetail | null>(null);
  let selectedReport = $state<JsonMap | null>(null);
  let selectedCandidates = $state<JsonMap[]>([]);
  let settingsText = $state('{}');
  let timers = $state<FactoryTimer[]>([]);
  let loading = $state(true);
  let loadingRun = $state(false);
  let saving = $state(false);
  let actionError = $state('');
  let actionSuccess = $state('');
  let runsError = $state('');
  let settingsError = $state('');
  let timerError = $state('');
  let selectedRunRequest = 0;

  function sameRun(a: FactoryRunRef | null, b: FactoryRunRef | null) {
    return !!a && !!b && a.date === b.date && a.run_id === b.run_id;
  }

  function clearSelectedRun() {
    selectedRunRef = null;
    selectedRun = null;
    selectedReport = null;
    selectedCandidates = [];
  }

  function canMutate() {
    return capability?.execution_enabled === true;
  }

  function errorMessage(error: unknown, fallback: string) {
    return error instanceof Error ? error.message : fallback;
  }

  function basename(value?: string | null) {
    if (!value) return '';
    const parts = value.split(/[\\/]/).filter(Boolean);
    return parts.at(-1) ?? value;
  }

  function formatNumber(value: unknown, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  function formatPercent(value: unknown, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
  }

  function formatCount(value: unknown) {
    return typeof value === 'number' && Number.isFinite(value) ? String(value) : '—';
  }

  function humanise(value?: string | null) {
    return value ? value.replaceAll('_', ' ') : '—';
  }

  function runStage(run: FactoryRunSummary) {
    return humanise(run.selection_stage || run.deploy_stage || (run.has_report ? 'report_ready' : 'metadata_only'));
  }

  function candidateLabel(candidate: JsonMap) {
    return candidate.config_id
      || basename(candidate.config_path)
      || basename(candidate.source_config_path)
      || candidate.filename
      || 'candidate';
  }

  function candidateStatus(candidate: JsonMap) {
    if (candidate.rejected) {
      return 'rejected';
    }
    return humanise(candidate.validation_gate || candidate.pipeline_stage || candidate.shortlist_mode || 'candidate');
  }

  function timerStatus(timer: FactoryTimer) {
    if (timer.available === false) return 'not loaded';
    return timer.active || 'unknown';
  }

  async function refreshOverview(options: { preferLatest?: boolean } = {}) {
    loading = true;
    actionError = '';
    runsError = '';
    settingsError = '';
    timerError = '';

    try {
      const [capabilityResult, runsResult, settingsResult, timerResult] = await Promise.allSettled([
        getFactoryCapability(),
        getFactoryRuns(),
        getFactorySettings(),
        getFactoryTimer(),
      ]);

      if (capabilityResult.status === 'rejected') {
        throw capabilityResult.reason;
      }

      capability = capabilityResult.value ?? null;

      if (runsResult.status === 'fulfilled') {
        runs = Array.isArray(runsResult.value) ? runsResult.value : [];
      } else {
        runs = [];
        runsError = errorMessage(runsResult.reason, 'Failed to load historical runs');
      }

      if (settingsResult.status === 'fulfilled') {
        settingsText = JSON.stringify(settingsResult.value ?? {}, null, 2);
      } else {
        settingsText = '{}';
        settingsError = errorMessage(settingsResult.reason, 'Failed to load factory settings');
      }

      if (timerResult.status === 'fulfilled') {
        timers = Array.isArray(timerResult.value?.timers) ? timerResult.value.timers : [];
      } else {
        timers = [];
        timerError = errorMessage(timerResult.reason, 'Failed to load factory timer state');
      }

      if (runsResult.status === 'fulfilled') {
        const selectedRunKey = selectedRunRef ?? selectedRun;
        const nextRun = options.preferLatest
          ? runs[0]
          : selectedRunKey
            ? runs.find((run) => sameRun(run, selectedRunKey)) ?? runs[0]
            : runs[0]
          ;

        if (nextRun) {
          await selectRun(nextRun.date, nextRun.run_id);
        } else {
          clearSelectedRun();
        }
      }
    } catch (error) {
      clearSelectedRun();
      actionError = errorMessage(error, 'Failed to load factory state');
    } finally {
      loading = false;
    }
  }

  async function selectRun(date: string, runId: string) {
    const requestedRun: FactoryRunRef = { date, run_id: runId };
    const switchingRuns = !sameRun(selectedRunRef, requestedRun);
    const requestId = ++selectedRunRequest;
    selectedRunRef = requestedRun;
    loadingRun = true;
    actionError = '';
    if (switchingRuns) {
      selectedRun = null;
      selectedReport = null;
      selectedCandidates = [];
    }
    try {
      const [detail, report, candidates] = await Promise.all([
        getFactoryRun(date, runId),
        getFactoryReport(date, runId).catch(() => null),
        getFactoryCandidates(date, runId).catch(() => []),
      ]);
      if (requestId !== selectedRunRequest || !sameRun(selectedRunRef, requestedRun)) {
        return;
      }
      selectedRun = detail;
      selectedReport = report;
      selectedCandidates = Array.isArray(candidates) ? candidates : [];
    } catch (error) {
      if (requestId !== selectedRunRequest || !sameRun(selectedRunRef, requestedRun)) {
        return;
      }
      actionError = errorMessage(error, 'Failed to load run');
    } finally {
      if (requestId === selectedRunRequest) {
        loadingRun = false;
      }
    }
  }

  async function launchFactory() {
    if (!canMutate()) return;
    actionError = '';
    actionSuccess = '';
    try {
      const result = await runFactory({});
      actionSuccess = `Factory ${result.profile ?? 'daily'} job started: ${result.job_id ?? 'submitted'}`;
      await refreshOverview({ preferLatest: true });
    } catch (error) {
      actionError = errorMessage(error, 'Factory launch failed');
    }
  }

  async function saveSettings() {
    if (!canMutate()) return;
    saving = true;
    actionError = '';
    actionSuccess = '';

    let parsed: JsonMap;
    try {
      parsed = JSON.parse(settingsText);
    } catch {
      actionError = 'Factory settings must be valid JSON before they can be saved.';
      saving = false;
      return;
    }

    try {
      await putFactorySettings(parsed);
      actionSuccess = 'Factory settings saved';
      await refreshOverview();
    } catch (error) {
      actionError = errorMessage(error, 'Failed to save factory settings');
    } finally {
      saving = false;
    }
  }

  async function applyTimer(action: 'enable' | 'disable') {
    if (!canMutate()) return;
    actionError = '';
    actionSuccess = '';
    try {
      const result = await factoryTimerAction(action);
      const failed = Array.isArray(result?.results)
        ? result.results.filter((entry: JsonMap) => entry?.ok !== true)
        : [];
      if (result?.ok === false || failed.length > 0) {
        const failedUnits = failed.map((entry: JsonMap) => entry?.unit).filter(Boolean).join(', ');
        actionError = failedUnits
          ? `Factory timer ${action} failed for ${failedUnits}.`
          : `Factory timer ${action} failed.`;
        return;
      }

      actionSuccess = `Factory timers ${action}d`;
      await refreshOverview();
    } catch (error) {
      actionError = errorMessage(error, `Timer ${action} failed`);
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
      <p class="subtitle">Rust factory control plane and artefact inspector. Reads stay available while execution is dormant, but run, timer, and settings mutations remain capability-gated.</p>
    </div>
    <div class="header-actions">
      <button class="btn btn-primary" onclick={launchFactory} disabled={!canMutate()}>
        Run Factory
      </button>
      <button class="btn btn-secondary" onclick={() => applyTimer('enable')} disabled={!canMutate()}>
        Enable Timers
      </button>
      <button class="btn btn-secondary" onclick={() => applyTimer('disable')} disabled={!canMutate()}>
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
        <span class="mono">feature: {capability?.compiled ? 'compiled' : 'not compiled'}</span>
        <span class="mono">policy: {capability?.policy_enabled ? 'enabled' : 'disabled'}</span>
        <span class="mono">executor: {capability?.executor_wired ? 'wired' : 'missing'}</span>
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
        {#if runsError}
          <p class="section-error">{runsError}</p>
        {/if}
        {#if runs.length === 0}
          <div class="empty">No Rust factory artefacts were found under `artifacts/`.</div>
        {:else}
          <div class="runs-list">
            {#each runs as run (run.date + run.run_id)}
              <button
                class="run-card"
                class:selected={sameRun(run, selectedRunRef)}
                onclick={() => selectRun(run.date, run.run_id)}
              >
                <div class="run-id">{run.run_id}</div>
                <div class="run-meta">
                  <span>{run.date}</span>
                  <span>{run.profile ?? 'unknown'}</span>
                  <span>{runStage(run)}</span>
                </div>
                <div class="run-meta">
                  <span>candidates: <strong>{formatCount(run.candidate_count)}</strong></span>
                  <span>role candidates: <strong>{formatCount(run.role_candidate_count)}</strong></span>
                  <span>selected: <strong>{formatCount(run.selected_count)}</strong></span>
                  <span>report: <strong>{run.has_report ? 'yes' : 'no'}</strong></span>
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
              <div class="field-label">Directory</div>
              <div class="mono">{selectedRun.directory_name ?? '—'}</div>
            </div>
            <div>
              <div class="field-label">Candidates</div>
              <div class="mono">{formatCount(selectedRun.candidate_count ?? selectedCandidates.length)}</div>
            </div>
            <div>
              <div class="field-label">Report</div>
              <div class="mono">{selectedRun.report_available ? 'report.json present' : 'metadata only'}</div>
            </div>
          </div>

          {#if selectedRun.selection_summary}
            <div class="detail-grid detail-grid-compact">
              <div>
                <div class="field-label">Selection stage</div>
                <div class="mono">{humanise(selectedRun.selection_summary.selection_stage)}</div>
              </div>
              <div>
                <div class="field-label">Deploy stage</div>
                <div class="mono">{humanise(selectedRun.selection_summary.deploy_stage)}</div>
              </div>
              <div>
                <div class="field-label">Promotion stage</div>
                <div class="mono">{humanise(selectedRun.selection_summary.promotion_stage)}</div>
              </div>
              <div>
                <div class="field-label">Step 5 gate</div>
                <div class="mono">{humanise(selectedRun.selection_summary.step5_gate_status)}</div>
              </div>
              <div>
                <div class="field-label">Role candidates</div>
                <div class="mono">{formatCount(selectedRun.selection_summary.role_candidate_count)}</div>
              </div>
              <div>
                <div class="field-label">Selected</div>
                <div class="mono">{formatCount(selectedRun.selection_summary.selected_count)}</div>
              </div>
            </div>
          {/if}

          <div class="stack">
            <div>
              <div class="field-label">Metadata</div>
              <div class="hint">Available subdirectories: {(selectedRun.subdirs ?? []).join(', ') || 'none'}</div>
              <pre>{JSON.stringify(selectedRun.metadata ?? {}, null, 2)}</pre>
            </div>

            <div>
              <div class="field-label">Gate Report</div>
              {#if selectedReport}
                <div class="detail-grid detail-grid-compact">
                  <div>
                    <div class="field-label">Candidate count</div>
                    <div class="mono">{formatCount(selectedReport.candidate_count)}</div>
                  </div>
                  <div>
                    <div class="field-label">Deployable</div>
                    <div class="mono">{formatCount(selectedReport.deployable_count)}</div>
                  </div>
                  <div>
                    <div class="field-label">Selected</div>
                    <div class="mono">{formatCount(selectedReport.selected_count)}</div>
                  </div>
                  <div>
                    <div class="field-label">Blocked</div>
                    <div class="mono">{selectedReport.blocked ? 'yes' : 'no'}</div>
                  </div>
                </div>
                <pre>{JSON.stringify(selectedReport, null, 2)}</pre>
              {:else}
                <div class="empty">No `reports/report.json` was exported for this run.</div>
              {/if}
            </div>

            <div>
              <div class="field-label">Candidate Evidence</div>
              {#if selectedCandidates.length === 0}
                <div class="empty">No candidate evidence was exported for this run.</div>
              {:else}
                <div class="candidate-list">
                  {#each selectedCandidates as candidate, index (candidate.config_id ?? candidate.filename ?? candidate.config_path ?? index)}
                    <div class="candidate-card" class:rejected={candidate.rejected}>
                      <div class="candidate-header">
                        <div class="run-id">{candidateLabel(candidate)}</div>
                        <span class="status-chip" class:enabled={!candidate.rejected} class:dormant={candidate.rejected}>
                          {candidateStatus(candidate)}
                        </span>
                      </div>
                      <div class="candidate-meta">
                        <span>mode: <strong>{candidate.shortlist_mode ?? candidate.sort_by ?? '—'}</strong></span>
                        <span>trades: <strong>{formatCount(candidate.total_trades)}</strong></span>
                        <span>PnL: <strong>{formatNumber(candidate.total_pnl)}</strong></span>
                        <span>PF: <strong>{formatNumber(candidate.profit_factor)}</strong></span>
                        <span>holdout: <strong>{formatPercent(candidate.holdout_median_daily_return, 4)}</strong></span>
                        <span>parity: <strong>{candidate.step4_parity?.status ?? '—'}</strong></span>
                      </div>
                      {#if candidate.rejected && candidate.reject_reason}
                        <div class="hint">{candidate.reject_reason}</div>
                      {/if}
                    </div>
                  {/each}
                </div>
                <pre>{JSON.stringify(selectedCandidates, null, 2)}</pre>
              {/if}
            </div>
          </div>
        {:else}
          <div class="empty">Select a run to inspect Rust metadata, gate output, and candidate evidence.</div>
        {/if}
      </section>
    </div>

    <div class="content-grid secondary">
      <section class="panel">
        <div class="panel-header">
          <h2>Settings</h2>
          <button class="btn btn-secondary" onclick={saveSettings} disabled={!canMutate() || saving}>
            {saving ? 'Saving…' : 'Save Defaults'}
          </button>
        </div>
        {#if settingsError}
          <p class="section-error">{settingsError}</p>
        {/if}
        <textarea bind:value={settingsText} rows="18" readonly={!canMutate()} spellcheck="false"></textarea>
        {#if !canMutate()}
          <p class="hint">Settings stay readable while Factory is dormant. Save is unlocked only when the Hub build includes the `factory` feature, the Rust executor binary is wired, and `AI_QUANT_FACTORY_ENABLE=1` is set.</p>
        {/if}
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>Timer State</h2>
          <span class="count">{timers.length}</span>
        </div>
        {#if timerError}
          <p class="section-error">{timerError}</p>
        {/if}
        {#if timers.length === 0}
          <div class="empty">No factory timers were reported.</div>
        {:else}
          <div class="timer-list">
            {#each timers as timer}
              <div class="timer-card">
                <div class="timer-name mono">{timer.unit ?? timer.name}</div>
                <div class="timer-meta">
                  <span>active: <strong>{timerStatus(timer)}</strong></span>
                  <span>enabled: <strong>{timer.enabled ? 'yes' : 'no'}</strong></span>
                  <span>unit file: <strong>{timer.unit_file_state || 'unknown'}</strong></span>
                  <span>load: <strong>{timer.load || 'unknown'}</strong></span>
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
  .run-meta,
  .candidate-meta,
  .candidate-header {
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

  .detail-grid {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }

  .detail-grid-compact {
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
  .stack,
  .candidate-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .run-card,
  .timer-card,
  .candidate-card {
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

  .candidate-card.rejected {
    border-color: rgba(255, 107, 107, 0.2);
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
  .candidate-meta,
  .hint,
  .field-label,
  .section-error {
    font-size: 12px;
    color: var(--text-muted);
  }

  .section-error {
    margin: 8px 0 0;
    color: var(--red);
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
      justify-content: flex-start;
    }
  }
</style>
