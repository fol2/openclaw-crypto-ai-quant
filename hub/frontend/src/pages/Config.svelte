<script lang="ts">
  import { getConfigRaw, putConfig, reloadConfig, getConfigHistory, getConfigDiff, getConfigFiles } from '../lib/api';

  // State
  let files: any[] = $state([]);
  let selectedFile = $state('main');
  let yamlText = $state('');
  let originalText = $state('');
  let dirty = $derived(yamlText !== originalText);
  let saving = $state(false);
  let loading = $state(false);
  let error = $state('');
  let success = $state('');
  let tab: 'editor' | 'history' | 'diff' = $state('editor');

  // History state
  let history: any[] = $state([]);
  let loadingHistory = $state(false);

  // Diff state
  let diffA = $state('');
  let diffB = $state('current');
  let diffResult: string[] = $state([]);
  let loadingDiff = $state(false);

  async function loadFiles() {
    try {
      files = await getConfigFiles();
    } catch {}
  }

  async function loadConfig() {
    loading = true;
    error = '';
    try {
      const raw = await getConfigRaw(selectedFile);
      yamlText = raw;
      originalText = raw;
    } catch (e: any) {
      error = e.message || 'Failed to load config';
    } finally {
      loading = false;
    }
  }

  async function save() {
    saving = true;
    error = '';
    success = '';
    try {
      const res = await putConfig(yamlText, selectedFile);
      originalText = yamlText;
      success = `Saved! Backup: ${res.backup || 'none'}`;
      setTimeout(() => { success = ''; }, 5000);
    } catch (e: any) {
      error = e.message || 'Failed to save';
    } finally {
      saving = false;
    }
  }

  async function saveAndReload() {
    await save();
    if (!error) {
      try {
        await reloadConfig(selectedFile);
        success = 'Saved & hot-reload triggered!';
        setTimeout(() => { success = ''; }, 5000);
      } catch (e: any) {
        error = `Saved but reload failed: ${e.message}`;
      }
    }
  }

  async function loadHistory() {
    loadingHistory = true;
    try {
      history = await getConfigHistory(selectedFile);
    } catch {}
    loadingHistory = false;
  }

  async function loadDiff() {
    if (!diffA) return;
    loadingDiff = true;
    try {
      const res = await getConfigDiff(diffA, diffB, selectedFile);
      diffResult = res.diff || [];
    } catch (e: any) {
      diffResult = [`Error: ${e.message}`];
    }
    loadingDiff = false;
  }

  function colorDiffLine(line: string): string {
    const escaped = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    if (line.startsWith('+')) return `<span class="diff-add">${escaped}</span>`;
    if (line.startsWith('-')) return `<span class="diff-del">${escaped}</span>`;
    return `<span class="diff-ctx">${escaped}</span>`;
  }

  $effect(() => {
    loadFiles();
    loadConfig();
  });

  function onFileChange(e: Event) {
    selectedFile = (e.target as HTMLSelectElement).value;
    loadConfig();
    if (tab === 'history') loadHistory();
  }
</script>

<div class="config-page">
  <div class="config-header">
    <h1>Config Management</h1>
    <div class="header-controls">
      <select class="file-select" onchange={onFileChange} value={selectedFile}>
        {#each files as f}
          <option value={f.variant} disabled={!f.exists}>
            {f.variant}{f.exists ? '' : ' (missing)'}
          </option>
        {/each}
      </select>
    </div>
  </div>

  <div class="tabs">
    <button class="tab" class:active={tab === 'editor'} onclick={() => tab = 'editor'}>Editor</button>
    <button class="tab" class:active={tab === 'history'} onclick={() => { tab = 'history'; loadHistory(); }}>History</button>
    <button class="tab" class:active={tab === 'diff'} onclick={() => { tab = 'diff'; loadHistory(); }}>Diff</button>
  </div>

  {#if error}
    <div class="alert alert-error">{error}</div>
  {/if}
  {#if success}
    <div class="alert alert-success">{success}</div>
  {/if}

  {#if tab === 'editor'}
    <div class="editor-section">
      <div class="editor-toolbar">
        <span class="file-label">{selectedFile}.yaml</span>
        {#if dirty}
          <span class="dirty-badge">unsaved changes</span>
        {/if}
        <div class="toolbar-actions">
          <button class="btn btn-secondary" onclick={() => { yamlText = originalText; }} disabled={!dirty}>
            Revert
          </button>
          <button class="btn btn-primary" onclick={save} disabled={!dirty || saving}>
            {saving ? 'Saving...' : 'Save'}
          </button>
          <button class="btn btn-accent" onclick={saveAndReload} disabled={!dirty || saving}>
            Save & Reload
          </button>
        </div>
      </div>
      {#if loading}
        <div class="loading">Loading config...</div>
      {:else}
        <textarea
          class="yaml-editor"
          bind:value={yamlText}
          spellcheck="false"
          wrap="off"
        ></textarea>
      {/if}
    </div>

  {:else if tab === 'history'}
    <div class="history-section">
      {#if loadingHistory}
        <div class="loading">Loading history...</div>
      {:else if history.length === 0}
        <div class="empty">No backups yet for {selectedFile}</div>
      {:else}
        <table class="history-table">
          <thead>
            <tr>
              <th>Backup File</th>
              <th>Modified</th>
              <th>Size</th>
            </tr>
          </thead>
          <tbody>
            {#each history as h}
              <tr>
                <td class="mono">{h.filename}</td>
                <td>{h.modified ? new Date(h.modified).toLocaleString() : '—'}</td>
                <td>{(h.size / 1024).toFixed(1)} KB</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>

  {:else if tab === 'diff'}
    <div class="diff-section">
      <div class="diff-controls">
        <label>
          Version A:
          <select bind:value={diffA}>
            <option value="">— select backup —</option>
            {#each history as h}
              <option value={h.filename}>{h.filename}</option>
            {/each}
          </select>
        </label>
        <label>
          Version B:
          <select bind:value={diffB}>
            <option value="current">current</option>
            {#each history as h}
              <option value={h.filename}>{h.filename}</option>
            {/each}
          </select>
        </label>
        <button class="btn btn-primary" onclick={loadDiff} disabled={!diffA || loadingDiff}>
          {loadingDiff ? 'Loading...' : 'Compare'}
        </button>
      </div>
      {#if diffResult.length > 0}
        <pre class="diff-view">{#each diffResult as line}{@html colorDiffLine(line)}
{/each}</pre>
      {/if}
    </div>
  {/if}
</div>

<style>
  .config-page {
    max-width: 1200px;
  }

  .config-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .config-header h1 {
    font-size: 20px;
    font-weight: 600;
    margin: 0;
  }

  .file-select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
  }

  .tabs {
    display: flex;
    gap: 2px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .tab {
    padding: 8px 16px;
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }

  .tab.active {
    color: var(--accent, #60a5fa);
    border-bottom-color: var(--accent, #60a5fa);
  }

  .alert {
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    margin-bottom: 12px;
  }

  .alert-error {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .alert-success {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }

  .editor-toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .file-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
  }

  .dirty-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(234, 179, 8, 0.2);
    color: #eab308;
  }

  .toolbar-actions {
    margin-left: auto;
    display: flex;
    gap: 8px;
  }

  .btn {
    padding: 6px 14px;
    border: none;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }

  .btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .btn-primary {
    background: var(--accent, #3b82f6);
    color: #fff;
  }

  .btn-secondary {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
  }

  .btn-accent {
    background: #22c55e;
    color: #fff;
  }

  .yaml-editor {
    width: 100%;
    min-height: 600px;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.5;
    resize: vertical;
    tab-size: 2;
  }

  .loading, .empty {
    color: var(--text-muted);
    font-style: italic;
    padding: 20px 0;
  }

  .history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .history-table th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 500;
  }

  .history-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle, rgba(255,255,255,0.05));
  }

  .mono {
    font-family: monospace;
    font-size: 12px;
  }

  .diff-controls {
    display: flex;
    gap: 16px;
    align-items: end;
    margin-bottom: 12px;
  }

  .diff-controls label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .diff-controls select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
  }

  .diff-view {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    font-size: 12px;
    line-height: 1.6;
    overflow-x: auto;
    white-space: pre;
  }

  :global(.diff-add) { color: #4ade80; }
  :global(.diff-del) { color: #f87171; }
  :global(.diff-ctx) { color: var(--text-muted); }
</style>
