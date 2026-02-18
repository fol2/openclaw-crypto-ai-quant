<script lang="ts">
  import { getSystemServices, serviceAction, getDbStats, getDiskUsage, getServiceLogs } from '../lib/api';

  let services: any[] = $state([]);
  let dbStats: any[] = $state([]);
  let disk: any[] = $state([]);
  let loading = $state(true);
  let actionError = $state('');
  let actionSuccess = $state('');

  let logService = $state('');
  let logLines: string = $state('');
  let loadingLogs = $state(false);

  let tab: 'services' | 'databases' | 'logs' | 'disk' = $state('services');

  async function refresh() {
    loading = true;
    try {
      const [svcs, stats, d] = await Promise.all([
        getSystemServices(),
        getDbStats(),
        getDiskUsage(),
      ]);
      services = svcs;
      dbStats = stats;
      disk = d;
    } catch {}
    loading = false;
  }

  async function doAction(name: string, action: string) {
    actionError = '';
    actionSuccess = '';
    try {
      const res = await serviceAction(name, action);
      if (res.ok) {
        actionSuccess = `${action} ${name}: OK`;
      } else {
        actionError = `${action} ${name}: ${res.stderr || 'failed'}`;
      }
      setTimeout(() => { actionSuccess = ''; }, 5000);
      await refresh();
    } catch (e: any) {
      actionError = e.message;
    }
  }

  async function loadLogs() {
    if (!logService) return;
    loadingLogs = true;
    try {
      const res = await getServiceLogs(logService, 100);
      logLines = res.log || '';
    } catch (e: any) {
      logLines = `Error: ${e.message}`;
    }
    loadingLogs = false;
  }

  $effect(() => { refresh(); });
</script>

<div class="system-page">
  <h1>System</h1>

  {#if actionError}
    <div class="alert alert-error">{actionError}</div>
  {/if}
  {#if actionSuccess}
    <div class="alert alert-success">{actionSuccess}</div>
  {/if}

  <div class="tabs">
    <button class="tab" class:active={tab === 'services'} onclick={() => tab = 'services'}>Services</button>
    <button class="tab" class:active={tab === 'databases'} onclick={() => tab = 'databases'}>Databases</button>
    <button class="tab" class:active={tab === 'logs'} onclick={() => tab = 'logs'}>Logs</button>
    <button class="tab" class:active={tab === 'disk'} onclick={() => tab = 'disk'}>Disk</button>
  </div>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if tab === 'services'}
    <div class="services-grid">
      {#each services as svc (svc.name)}
        <div class="service-card">
          <div class="svc-header">
            <status-badge status={svc.status} label={svc.active}></status-badge>
            <span class="svc-name">{svc.name}</span>
          </div>
          <div class="svc-meta">
            <span>PID: {svc.pid || '—'}</span>
            <span>State: {svc.sub}</span>
          </div>
          <div class="svc-actions">
            <button class="btn-sm" onclick={() => doAction(svc.name, 'restart')}>Restart</button>
            {#if svc.active !== 'active'}
              <button class="btn-sm btn-start" onclick={() => doAction(svc.name, 'start')}>Start</button>
            {:else}
              <button class="btn-sm btn-stop" onclick={() => doAction(svc.name, 'stop')}>Stop</button>
            {/if}
          </div>
        </div>
      {/each}
    </div>

  {:else if tab === 'databases'}
    <table class="db-table">
      <thead>
        <tr>
          <th>Database</th>
          <th>Size</th>
          <th>Modified</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {#each dbStats as db}
          <tr>
            <td class="mono">{db.label}</td>
            <td>{db.size_mb} MB</td>
            <td>{db.modified ? new Date(db.modified).toLocaleString() : '—'}</td>
            <td>
              <status-badge status={db.exists ? 'ok' : 'bad'} label={db.exists ? 'exists' : 'missing'}></status-badge>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>

  {:else if tab === 'logs'}
    <div class="logs-section">
      <div class="logs-controls">
        <select bind:value={logService}>
          <option value="">— select service —</option>
          {#each services as svc}
            <option value={svc.name}>{svc.name}</option>
          {/each}
        </select>
        <button class="btn btn-primary" onclick={loadLogs} disabled={!logService || loadingLogs}>
          {loadingLogs ? 'Loading...' : 'Load Logs'}
        </button>
      </div>
      {#if logLines}
        <pre class="log-output">{logLines}</pre>
      {/if}
    </div>

  {:else if tab === 'disk'}
    <div class="disk-section">
      {#each disk as d}
        <div class="disk-row">
          <span class="disk-label">{d.label}</span>
          <span class="disk-size">{d.size}</span>
          <span class="disk-path mono">{d.path}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .system-page { max-width: 1200px; }

  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }

  .alert { padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; }
  .alert-error { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .alert-success { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }

  .tabs {
    display: flex; gap: 2px; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .tab {
    padding: 8px 16px; background: none; border: none;
    color: var(--text-muted); font-size: 13px; cursor: pointer;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .tab.active { color: var(--accent, #60a5fa); border-bottom-color: var(--accent, #60a5fa); }

  .services-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }

  .service-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }

  .svc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .svc-name { font-size: 12px; font-weight: 500; }
  .svc-meta { font-size: 11px; color: var(--text-muted); display: flex; gap: 12px; margin-bottom: 8px; }
  .svc-actions { display: flex; gap: 6px; }

  .btn-sm {
    padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface); color: var(--text); font-size: 11px; cursor: pointer;
  }
  .btn-sm:hover { background: var(--border); }
  .btn-start { color: #4ade80; border-color: rgba(34,197,94,0.3); }
  .btn-stop { color: #f87171; border-color: rgba(239,68,68,0.3); }

  .db-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .db-table th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; }
  .db-table td { padding: 8px 10px; border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.05)); }

  .mono { font-family: monospace; font-size: 12px; }

  .logs-controls { display: flex; gap: 12px; align-items: end; margin-bottom: 12px; }
  .logs-controls select {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 6px 10px; border-radius: 6px; font-size: 12px;
  }

  .btn { padding: 6px 14px; border: none; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn-primary { background: var(--accent, #3b82f6); color: #fff; }

  .log-output {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; font-size: 11px; line-height: 1.5;
    max-height: 600px; overflow-y: auto; white-space: pre-wrap;
  }

  .disk-section { display: flex; flex-direction: column; gap: 8px; }
  .disk-row {
    display: flex; gap: 16px; align-items: center; padding: 8px 12px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  }
  .disk-label { font-weight: 500; font-size: 13px; min-width: 120px; }
  .disk-size { font-size: 16px; font-weight: 600; min-width: 80px; }
  .disk-path { color: var(--text-muted); font-size: 12px; }

  .empty { color: var(--text-muted); font-style: italic; padding: 20px 0; }
</style>
