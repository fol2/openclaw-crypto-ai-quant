<script lang="ts">
  import {
    getFactoryRuns, getFactoryRun, getFactoryReport, getFactoryCandidates,
    runFactory, getFactoryJobs, getFactoryJobStatus, cancelFactory,
    getFactorySettings, putFactorySettings, getFactoryTimer, factoryTimerAction,
  } from '../lib/api';

  // ── Tab state ────────────────────────────────────────────
  let tab: 'run' | 'history' | 'settings' = $state('run');

  // ── Run tab state ────────────────────────────────────────
  let profile = $state('daily');
  let strategyMode = $state('');
  let candidateCount = $state(3);
  let gpu = $state(true);
  let tpe = $state(true);
  let walkForward = $state(true);
  let slippageStress = $state(true);
  let concentrationChecks = $state(true);
  let sensitivityChecks = $state(true);
  let dryRun = $state(false);
  let noDeploy = $state(false);
  let livepaperPromotion = $state(true);

  let launching = $state(false);
  let error = $state('');

  // Active job tracking
  let jobs: any[] = $state([]);
  let activeJobId: string | null = $state(null);
  let activeStatus: any = $state(null);
  let stderrLines: string[] = $state([]);
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  // Timer state
  let timerData: any = $state(null);
  let timerLoading = $state(false);

  // ── History tab state ────────────────────────────────────
  let runs: any[] = $state([]);
  let loadingRuns = $state(true);
  let selectedRun: any = $state(null);
  let runDetail: any = $state(null);
  let report: any = $state(null);
  let candidates: any[] = $state([]);
  let loadingDetail = $state(false);
  let historyFilter: 'all' | 'running' = $state('all');

  // ── Settings tab state ───────────────────────────────────
  let settingsProfile = $state('daily');
  let settingsStrategyMode = $state('');
  let settingsCandidateCount = $state(3);
  let settingsGpu = $state(true);
  let settingsTpe = $state(true);
  let settingsWalkForward = $state(true);
  let settingsSlippageStress = $state(true);
  let settingsConcentrationChecks = $state(true);
  let settingsSensitivityChecks = $state(true);
  let settingsDryRun = $state(false);
  let settingsNoDeploy = $state(false);
  let settingsLivepaperPromotion = $state(true);
  let savingSettings = $state(false);
  let settingsMsg = $state('');

  // ── Helpers ──────────────────────────────────────────────
  function fmtNum(n: any, d = 2): string {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(d);
  }

  function elapsed(created: string): string {
    const ms = Date.now() - new Date(created).getTime();
    const secs = Math.floor(ms / 1000);
    const m = Math.floor(secs / 60);
    const h = Math.floor(m / 60);
    if (h > 0) return `${h}h ${m % 60}m`;
    if (m > 0) return `${m}m ${secs % 60}s`;
    return `${secs}s`;
  }

  // ── Run tab actions ──────────────────────────────────────
  async function launch() {
    launching = true;
    error = '';
    try {
      const res = await runFactory({
        profile,
        strategy_mode: strategyMode || undefined,
        candidate_count: candidateCount,
        gpu, tpe, walk_forward: walkForward,
        slippage_stress: slippageStress,
        concentration_checks: concentrationChecks,
        sensitivity_checks: sensitivityChecks,
        dry_run: dryRun, no_deploy: noDeploy,
        enable_livepaper_promotion: livepaperPromotion,
      });
      activeJobId = res.job_id;
      activeStatus = null;
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
      jobs = await getFactoryJobs();
    } catch {}
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      if (!activeJobId) return;
      try {
        const s = await getFactoryJobStatus(activeJobId);
        activeStatus = s;
        stderrLines = s.stderr_tail || [];
        if (s.status !== 'running') {
          stopPolling();
          await refreshJobs();
        }
      } catch {}
    }, 5000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function cancel() {
    if (!activeJobId) return;
    try {
      await cancelFactory(activeJobId);
      stopPolling();
      await refreshJobs();
    } catch {}
  }

  function selectJob(id: string) {
    activeJobId = id;
    activeStatus = null;
    stderrLines = [];
    const job = jobs.find((j: any) => j.id === id);
    if (job?.status === 'running') {
      startPolling();
    } else {
      getFactoryJobStatus(id).then(s => {
        activeStatus = s;
        stderrLines = s.stderr_tail || [];
      }).catch(() => {});
    }
  }

  // Timer actions
  async function loadTimer() {
    timerLoading = true;
    try {
      timerData = await getFactoryTimer();
    } catch {}
    timerLoading = false;
  }

  async function toggleTimer(action: string) {
    try {
      await factoryTimerAction(action);
      await loadTimer();
    } catch (e: any) {
      error = e.message;
    }
  }

  // ── History tab actions ──────────────────────────────────
  async function loadRuns() {
    loadingRuns = true;
    try {
      runs = await getFactoryRuns();
    } catch (e: any) {
      error = e.message || 'Failed to load runs';
    }
    loadingRuns = false;
  }

  async function selectHistoryRun(run: any) {
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

  // ── Settings tab actions ─────────────────────────────────
  async function loadSettings() {
    try {
      const s = await getFactorySettings();
      if (s && typeof s === 'object') {
        settingsProfile = s.profile || 'daily';
        settingsStrategyMode = s.strategy_mode || '';
        settingsCandidateCount = s.candidate_count ?? 3;
        settingsGpu = s.gpu ?? true;
        settingsTpe = s.tpe ?? true;
        settingsWalkForward = s.walk_forward ?? true;
        settingsSlippageStress = s.slippage_stress ?? true;
        settingsConcentrationChecks = s.concentration_checks ?? true;
        settingsSensitivityChecks = s.sensitivity_checks ?? true;
        settingsDryRun = s.dry_run ?? false;
        settingsNoDeploy = s.no_deploy ?? false;
        settingsLivepaperPromotion = s.enable_livepaper_promotion ?? true;
      }
    } catch {}
  }

  async function saveSettings() {
    savingSettings = true;
    settingsMsg = '';
    try {
      await putFactorySettings({
        profile: settingsProfile,
        strategy_mode: settingsStrategyMode || undefined,
        candidate_count: settingsCandidateCount,
        gpu: settingsGpu,
        tpe: settingsTpe,
        walk_forward: settingsWalkForward,
        slippage_stress: settingsSlippageStress,
        concentration_checks: settingsConcentrationChecks,
        sensitivity_checks: settingsSensitivityChecks,
        dry_run: settingsDryRun,
        no_deploy: settingsNoDeploy,
        enable_livepaper_promotion: settingsLivepaperPromotion,
      });
      settingsMsg = 'Settings saved';
      setTimeout(() => { settingsMsg = ''; }, 3000);
    } catch (e: any) {
      settingsMsg = `Error: ${e.message}`;
    }
    savingSettings = false;
  }

  function resetSettings() {
    settingsProfile = 'daily';
    settingsStrategyMode = '';
    settingsCandidateCount = 3;
    settingsGpu = true;
    settingsTpe = true;
    settingsWalkForward = true;
    settingsSlippageStress = true;
    settingsConcentrationChecks = true;
    settingsSensitivityChecks = true;
    settingsDryRun = false;
    settingsNoDeploy = false;
    settingsLivepaperPromotion = true;
  }

  // ── Filtered jobs for run panel ──────────────────────────
  let runningJob = $derived(jobs.find((j: any) => j.status === 'running'));
  let isRunning = $derived(!!runningJob);

  // ── Init ─────────────────────────────────────────────────
  $effect(() => {
    refreshJobs();
    loadTimer();
    loadRuns();
    loadSettings();
    return () => stopPolling();
  });

  // Auto-track running job on load
  $effect(() => {
    if (runningJob && !activeJobId) {
      activeJobId = runningJob.id;
      startPolling();
    }
  });
</script>

<div class="page">
  <h1>Factory</h1>

  {#if error}
    <div class="alert alert-error">{error}
      <button class="alert-dismiss" onclick={() => error = ''}>×</button>
    </div>
  {/if}

  <div class="tabs">
    <button class="tab" class:active={tab === 'run'} onclick={() => tab = 'run'}>Run</button>
    <button class="tab" class:active={tab === 'history'} onclick={() => tab = 'history'}>History</button>
    <button class="tab" class:active={tab === 'settings'} onclick={() => tab = 'settings'}>Settings</button>
  </div>

  <!-- ═══════════════ RUN TAB ═══════════════ -->
  {#if tab === 'run'}
    <div class="run-layout">
      <!-- Launch form -->
      <div class="launch-section">
        <div class="form-grid">
          <label class="form-field">
            <span class="field-label">Profile</span>
            <select bind:value={profile}>
              <option value="smoke">smoke</option>
              <option value="daily">daily</option>
              <option value="deep">deep</option>
              <option value="weekly">weekly</option>
            </select>
          </label>

          <label class="form-field">
            <span class="field-label">Strategy Mode</span>
            <select bind:value={strategyMode}>
              <option value="">(default)</option>
              <option value="primary">primary</option>
              <option value="fallback">fallback</option>
            </select>
          </label>

          <label class="form-field">
            <span class="field-label">Candidate Count</span>
            <input type="number" bind:value={candidateCount} min="1" max="10" />
          </label>
        </div>

        <div class="toggle-grid">
          <label class="toggle-item">
            <input type="checkbox" bind:checked={gpu} />
            <span>GPU</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={tpe} />
            <span>TPE</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={walkForward} />
            <span>Walk Forward</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={slippageStress} />
            <span>Slippage Stress</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={concentrationChecks} />
            <span>Concentration</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={sensitivityChecks} />
            <span>Sensitivity</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={dryRun} />
            <span>Dry Run</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={noDeploy} />
            <span>No Deploy</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" bind:checked={livepaperPromotion} />
            <span>Livepaper Promo</span>
          </label>
        </div>

        <div class="launch-actions">
          {#if isRunning}
            <button class="btn btn-danger" onclick={cancel}>Cancel</button>
          {:else}
            <button class="btn btn-primary" onclick={launch} disabled={launching}>
              {launching ? 'Launching...' : 'Run Factory'}
            </button>
          {/if}
        </div>
      </div>

      <!-- Active job progress -->
      {#if activeJobId && activeStatus}
        <div class="progress-section">
          <div class="progress-header">
            <div class="progress-meta">
              <span class="status-pill {activeStatus.status}">{activeStatus.status}</span>
              <span class="elapsed">{elapsed(activeStatus.created_at)}</span>
            </div>
            <span class="job-id-label mono">{activeJobId.slice(0, 8)}</span>
          </div>
          <pre class="stderr-log">{stderrLines.join('\n') || 'Waiting for output...'}</pre>
        </div>
      {:else if activeJobId}
        <div class="progress-section">
          <div class="progress-header">
            <span class="status-pill running">starting</span>
          </div>
          <pre class="stderr-log">Waiting for output...</pre>
        </div>
      {/if}

      <!-- Jobs list -->
      {#if jobs.length > 0}
        <div class="jobs-compact">
          <h3>Recent Jobs</h3>
          {#each jobs as j (j.id)}
            <button class="job-row" class:active={activeJobId === j.id} onclick={() => selectJob(j.id)}>
              <span class="mono">{j.id.slice(0, 8)}</span>
              <span class="status-pill {j.status}">{j.status}</span>
              <span class="job-time">{j.created_at ? new Date(j.created_at).toLocaleString() : ''}</span>
            </button>
          {/each}
        </div>
      {/if}

      <!-- Timer status -->
      <div class="timer-card">
        <h3>Timer</h3>
        {#if timerLoading}
          <span class="dim">Loading...</span>
        {:else if timerData?.timers}
          {#each timerData.timers as t}
            <div class="timer-row">
              <span class="mono timer-unit">{t.unit}</span>
              <span class="status-chip" class:enabled={t.enabled} class:disabled={!t.enabled}>
                {t.enabled ? 'enabled' : 'disabled'}
              </span>
              {#if t.next_trigger && t.enabled}
                <span class="dim">Next: {t.next_trigger}</span>
              {/if}
            </div>
          {/each}
          <div class="timer-actions">
            <button class="btn-sm btn-start" onclick={() => toggleTimer('enable')}>Enable</button>
            <button class="btn-sm btn-stop" onclick={() => toggleTimer('disable')}>Disable</button>
          </div>
        {:else}
          <span class="dim">Timer data unavailable</span>
        {/if}
      </div>
    </div>

  <!-- ═══════════════ HISTORY TAB ═══════════════ -->
  {:else if tab === 'history'}
    <div class="content-grid">
      <div class="runs-panel">
        <div class="runs-header">
          <h2>Runs</h2>
          <div class="filter-pills">
            <button class="fpill" class:active={historyFilter === 'all'} onclick={() => historyFilter = 'all'}>All</button>
            <button class="fpill" class:active={historyFilter === 'running'} onclick={() => historyFilter = 'running'}>Running</button>
          </div>
        </div>

        <!-- Hub-triggered jobs at top -->
        {#each jobs.filter(j => historyFilter === 'all' || j.status === 'running') as j (j.id)}
          <button
            class="run-card hub-job"
            class:active={activeJobId === j.id && !selectedRun}
            onclick={() => { selectedRun = null; selectJob(j.id); }}
          >
            <div class="run-date">
              {#if j.status === 'running'}
                <span class="pulse-dot"></span>
              {/if}
              Hub Job
            </div>
            <div class="run-id">{j.id.slice(0, 12)}</div>
            <div class="run-meta">
              <span class="status-pill {j.status}">{j.status}</span>
              <span class="job-time">{j.created_at ? new Date(j.created_at).toLocaleTimeString() : ''}</span>
            </div>
          </button>
        {/each}

        {#if loadingRuns}
          <div class="empty-state">Loading...</div>
        {:else if runs.length === 0}
          <div class="empty-state">No factory runs found</div>
        {:else}
          {#each runs as r (r.date + '/' + r.run_id)}
            <button
              class="run-card"
              class:active={selectedRun?.run_id === r.run_id && selectedRun?.date === r.date}
              onclick={() => { activeJobId = null; selectHistoryRun(r); }}
            >
              <div class="run-date">{r.date}</div>
              <div class="run-id">{r.run_id}</div>
              <div class="run-meta">
                <span class="pill-blue">{r.profile}</span>
                {#if r.has_report}
                  <span class="pill-green">report</span>
                {/if}
              </div>
            </button>
          {/each}
        {/if}
      </div>

      <div class="detail-panel">
        {#if activeJobId && !selectedRun && activeStatus}
          <!-- Show hub job detail -->
          <h2>Hub Job {activeJobId.slice(0, 8)}</h2>
          <div class="meta-section">
            <div class="meta-row"><span class="meta-label">Status</span><span class="status-pill {activeStatus.status}">{activeStatus.status}</span></div>
            <div class="meta-row"><span class="meta-label">Created</span><span>{new Date(activeStatus.created_at).toLocaleString()}</span></div>
            {#if activeStatus.finished_at}
              <div class="meta-row"><span class="meta-label">Finished</span><span>{new Date(activeStatus.finished_at).toLocaleString()}</span></div>
            {/if}
            {#if activeStatus.error}
              <div class="meta-row"><span class="meta-label">Error</span><span class="text-red">{activeStatus.error}</span></div>
            {/if}
          </div>
          {#if stderrLines.length > 0}
            <h3>Output</h3>
            <pre class="stderr-log">{stderrLines.join('\n')}</pre>
          {/if}

        {:else if loadingDetail}
          <div class="empty-state">Loading run detail...</div>
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
              <div class="table-wrap">
                <table class="data-table">
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
                          <span class="verdict" class:pass={c.verdict === 'PASS' || c.verdict === 'pass'} class:fail={c.verdict !== 'PASS' && c.verdict !== 'pass'}>
                            {c.verdict ?? '—'}
                          </span>
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else}
              <pre class="code-block">{JSON.stringify(report, null, 2)}</pre>
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
          <div class="empty-state">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" stroke-width="1"><path d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
            <p>Select a factory run</p>
          </div>
        {/if}
      </div>
    </div>

  <!-- ═══════════════ SETTINGS TAB ═══════════════ -->
  {:else if tab === 'settings'}
    <div class="settings-section">
      {#if settingsMsg}
        <div class="alert" class:alert-success={!settingsMsg.startsWith('Error')} class:alert-error={settingsMsg.startsWith('Error')}>
          {settingsMsg}
        </div>
      {/if}

      <div class="form-grid">
        <label class="form-field">
          <span class="field-label">Profile</span>
          <select bind:value={settingsProfile}>
            <option value="smoke">smoke</option>
            <option value="daily">daily</option>
            <option value="deep">deep</option>
            <option value="weekly">weekly</option>
          </select>
        </label>

        <label class="form-field">
          <span class="field-label">Strategy Mode</span>
          <select bind:value={settingsStrategyMode}>
            <option value="">(default)</option>
            <option value="primary">primary</option>
            <option value="fallback">fallback</option>
          </select>
        </label>

        <label class="form-field">
          <span class="field-label">Candidate Count</span>
          <input type="number" bind:value={settingsCandidateCount} min="1" max="10" />
        </label>
      </div>

      <div class="toggle-grid">
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsGpu} />
          <span>GPU</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsTpe} />
          <span>TPE</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsWalkForward} />
          <span>Walk Forward</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsSlippageStress} />
          <span>Slippage Stress</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsConcentrationChecks} />
          <span>Concentration</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsSensitivityChecks} />
          <span>Sensitivity</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsDryRun} />
          <span>Dry Run</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsNoDeploy} />
          <span>No Deploy</span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" bind:checked={settingsLivepaperPromotion} />
          <span>Livepaper Promo</span>
        </label>
      </div>

      <div class="settings-actions">
        <button class="btn btn-primary" onclick={saveSettings} disabled={savingSettings}>
          {savingSettings ? 'Saving...' : 'Save'}
        </button>
        <button class="btn btn-secondary" onclick={resetSettings}>Reset to Defaults</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .page { max-width: 1400px; animation: slideUp 0.3s ease; }

  h1 { font-size: 20px; font-weight: 700; margin-bottom: var(--sp-md); letter-spacing: -0.02em; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: var(--sp-sm); }
  h3 {
    font-size: 10px; font-weight: 600; margin: var(--sp-md) 0 var(--sp-sm);
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em;
  }

  .alert {
    padding: 10px 14px; border-radius: var(--radius-md);
    font-size: 13px; margin-bottom: 12px; animation: slideUp 0.2s ease;
    display: flex; justify-content: space-between; align-items: center;
  }
  .alert-error { background: var(--red-bg); color: var(--red); border: 1px solid rgba(255,107,107,0.2); }
  .alert-success { background: var(--green-bg); color: var(--green); border: 1px solid rgba(81,207,102,0.2); }
  .alert-dismiss {
    background: none; border: none; color: inherit;
    font-size: 18px; cursor: pointer; padding: 0 4px; opacity: 0.7;
  }
  .alert-dismiss:hover { opacity: 1; }

  /* ─── Tabs ─── */
  .tabs {
    display: flex; gap: 2px; margin-bottom: var(--sp-md);
    background: var(--bg); border-radius: var(--radius-md);
    padding: 3px; border: 1px solid var(--border); width: fit-content;
  }
  .tab {
    padding: 8px 16px; background: transparent; border: none;
    color: var(--text-muted); font-size: 13px; font-weight: 500;
    cursor: pointer; border-radius: 5px; transition: all var(--t-fast);
  }
  .tab.active { color: var(--accent); background: var(--accent-bg); }

  /* ─── Form ─── */
  .form-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: var(--sp-md); margin-bottom: var(--sp-md);
  }
  .form-field {
    display: flex; flex-direction: column; gap: var(--sp-xs);
  }
  .field-label {
    font-size: 11px; color: var(--text-dim); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .form-field select, .form-field input {
    background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 8px 12px; border-radius: var(--radius-md);
    font-size: 13px; transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .form-field select:focus, .form-field input:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-bg);
  }

  /* ─── Toggle grid ─── */
  .toggle-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: var(--sp-md);
  }
  .toggle-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius-md);
    font-size: 13px; cursor: pointer; transition: border-color var(--t-fast);
  }
  .toggle-item:hover { border-color: var(--accent); }
  .toggle-item input[type="checkbox"] {
    accent-color: var(--accent); width: 16px; height: 16px; cursor: pointer;
  }

  /* ─── Buttons ─── */
  .btn {
    padding: 8px 20px; border: none; border-radius: var(--radius-md);
    font-size: 12px; font-weight: 600; cursor: pointer;
    transition: background var(--t-fast), transform var(--t-fast);
    letter-spacing: 0.02em;
  }
  .btn:hover { transform: translateY(-1px); }
  .btn:active { transform: scale(0.97); }
  .btn:disabled { opacity: 0.35; cursor: default; pointer-events: none; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .btn-secondary:hover { background: var(--surface-hover); }
  .btn-danger { background: var(--red); color: #fff; }

  .btn-sm {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 6px 12px; border: 1px solid var(--border); border-radius: var(--radius-md);
    background: var(--surface); color: var(--text); font-size: 11px;
    font-weight: 500; cursor: pointer; transition: all var(--t-fast);
  }
  .btn-sm:hover { background: var(--surface-hover); }
  .btn-sm:active { transform: scale(0.97); }
  .btn-start { color: var(--green); border-color: rgba(81,207,102,0.3); }
  .btn-stop { color: var(--red); border-color: rgba(255,107,107,0.3); }

  .launch-actions { margin-bottom: var(--sp-md); }
  .settings-actions { display: flex; gap: var(--sp-sm); }

  /* ─── Progress ─── */
  .progress-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: var(--sp-md); margin-bottom: var(--sp-md);
  }
  .progress-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: var(--sp-sm);
  }
  .progress-meta { display: flex; gap: var(--sp-sm); align-items: center; }
  .elapsed { font-size: 12px; color: var(--text-dim); font-family: 'IBM Plex Mono', monospace; }
  .job-id-label { font-size: 11px; color: var(--text-dim); }

  .stderr-log {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: var(--sp-md);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    line-height: 1.6; max-height: 400px; overflow-y: auto;
    white-space: pre-wrap; color: var(--text-muted);
  }

  /* ─── Status pills ─── */
  .status-pill {
    font-size: 10px; padding: 2px 8px; border-radius: var(--radius-sm, 4px);
    font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
  }
  .status-pill.running { background: var(--accent-bg); color: var(--accent); }
  .status-pill.starting { background: var(--accent-bg); color: var(--accent); }
  .status-pill.done { background: var(--green-bg); color: var(--green); }
  .status-pill.failed { background: var(--red-bg); color: var(--red); }
  .status-pill.cancelled { background: var(--yellow-bg); color: var(--yellow); }

  /* ─── Jobs compact list ─── */
  .jobs-compact { margin-bottom: var(--sp-md); }
  .job-row {
    display: flex; gap: var(--sp-md); align-items: center;
    width: 100%; text-align: left; padding: 8px 12px;
    background: none; border: 1px solid transparent;
    border-radius: var(--radius-md); cursor: pointer;
    color: var(--text); transition: all var(--t-fast);
    margin-bottom: 2px;
  }
  .job-row:hover { background: var(--surface); }
  .job-row.active { border-color: var(--accent); background: var(--accent-bg); }
  .job-time { font-size: 11px; color: var(--text-muted); margin-left: auto; }

  /* ─── Timer card ─── */
  .timer-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: var(--sp-md);
  }
  .timer-row {
    display: flex; gap: var(--sp-md); align-items: center;
    padding: 6px 0; font-size: 12px;
  }
  .timer-unit { font-size: 11px; }
  .timer-actions { margin-top: var(--sp-sm); display: flex; gap: 6px; }
  .status-chip {
    font-size: 10px; padding: 2px 8px; border-radius: var(--radius-sm);
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .status-chip.enabled { background: var(--green-bg); color: var(--green); }
  .status-chip.disabled { background: var(--red-bg); color: var(--red); }

  /* ─── History tab ─── */
  .content-grid { display: grid; grid-template-columns: 280px 1fr; gap: var(--sp-md); }

  .runs-panel {
    border-right: 1px solid var(--border);
    padding-right: var(--sp-md);
    max-height: 80vh; overflow-y: auto;
  }
  .runs-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-sm); }
  .filter-pills { display: flex; gap: 2px; }
  .fpill {
    font-size: 10px; padding: 3px 10px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    cursor: pointer; color: var(--text-muted); transition: all var(--t-fast);
  }
  .fpill.active { color: var(--accent); border-color: var(--accent); background: var(--accent-bg); }

  .run-card {
    display: block; width: 100%; text-align: left;
    padding: 10px 12px; background: none; border: 1px solid transparent;
    border-radius: var(--radius-md); cursor: pointer; margin-bottom: 4px;
    color: var(--text); transition: all var(--t-fast);
  }
  .run-card:hover { background: var(--surface); }
  .run-card.active { border-color: var(--accent); background: var(--accent-bg); }
  .run-card.hub-job { border-left: 3px solid var(--accent); }

  .run-date {
    font-size: 11px; color: var(--text-dim);
    font-family: 'IBM Plex Mono', monospace;
    display: flex; align-items: center; gap: 6px;
  }
  .run-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; margin-top: 2px; font-weight: 500;
  }
  .run-meta { display: flex; gap: 6px; margin-top: 6px; align-items: center; }

  .pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); display: inline-block;
    animation: pulse 2s ease-in-out infinite;
  }

  .pill-blue {
    font-size: 10px; padding: 2px 8px; border-radius: var(--radius-sm);
    background: var(--accent-bg); color: var(--accent); font-weight: 600; letter-spacing: 0.02em;
  }
  .pill-green {
    font-size: 10px; padding: 2px 8px; border-radius: var(--radius-sm);
    background: var(--green-bg); color: var(--green); font-weight: 600; letter-spacing: 0.02em;
  }

  .detail-panel { min-height: 300px; }

  .meta-section { margin-bottom: var(--sp-md); }
  .meta-row {
    display: flex; gap: 12px; padding: 5px 0; font-size: 13px;
    border-bottom: 1px solid var(--border-subtle); align-items: center;
  }
  .meta-label { color: var(--text-dim); min-width: 100px; font-weight: 500; font-size: 12px; }
  .text-red { color: var(--red); }

  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }

  .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .data-table th {
    text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border);
    color: var(--text-dim); font-weight: 600; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border-subtle); }

  .verdict {
    font-size: 10px; padding: 2px 8px; border-radius: var(--radius-sm);
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .verdict.pass { background: var(--green-bg); color: var(--green); }
  .verdict.fail { background: var(--red-bg); color: var(--red); }

  .code-block {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    max-height: 500px; overflow: auto; white-space: pre-wrap;
  }

  .config-list { display: flex; flex-direction: column; gap: 4px; }
  .config-item {
    display: flex; justify-content: space-between; padding: 8px 12px;
    font-size: 13px; background: var(--surface); border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
  }
  .config-size { color: var(--text-dim); font-size: 12px; font-family: 'IBM Plex Mono', monospace; }

  .empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 8px; padding: 40px 0;
    color: var(--text-dim); font-size: 13px;
  }

  /* ─── Settings ─── */
  .settings-section { max-width: 700px; }

  /* ─── Utilities ─── */
  .mono { font-family: 'IBM Plex Mono', monospace; }
  .dim { color: var(--text-dim); font-size: 12px; }

  @media (max-width: 768px) {
    .content-grid { grid-template-columns: 1fr; }
    .runs-panel {
      border-right: none; border-bottom: 1px solid var(--border);
      padding-right: 0; padding-bottom: var(--sp-md); max-height: 40vh;
    }
    .form-grid { grid-template-columns: 1fr; }
    .toggle-grid { grid-template-columns: repeat(2, 1fr); }
    .tabs { width: 100%; }
    .tab { flex: 1; text-align: center; }
    .btn { padding: 12px 16px; }
    .timer-row { flex-wrap: wrap; gap: 8px; }
  }
</style>
