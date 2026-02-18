<script lang="ts">
  let { currentPage }: { currentPage: string } = $props();

  const links = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'config', label: 'Config', icon: '⚙️' },
    { id: 'backtest', label: 'Backtest', icon: '🔬' },
    { id: 'sweep', label: 'Sweep', icon: '🔍' },
    { id: 'factory', label: 'Factory', icon: '🏭' },
    { id: 'grid', label: 'Grid View', icon: '📈' },
    { id: 'system', label: 'System', icon: '🖥️' },
  ];
</script>

<nav class="sidebar">
  <div class="logo">
    <strong>AQC Hub</strong>
    <span class="version">v0.1</span>
  </div>

  <ul>
    {#each links as link}
      <li>
        <a
          href="#{link.id}"
          class:active={currentPage === link.id}
        >
          <span class="icon">{link.icon}</span>
          {link.label}
        </a>
      </li>
    {/each}
  </ul>

  <div class="sidebar-footer">
    <div class="connection-status" id="ws-status">
      <span class="dot"></span>
      Connecting…
    </div>
  </div>
</nav>

<style>
  .sidebar {
    width: var(--sidebar-width);
    min-width: var(--sidebar-width);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .logo {
    padding: 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .version {
    font-size: 11px;
    color: var(--text-muted);
  }

  ul {
    list-style: none;
    flex: 1;
    padding: 8px 0;
  }

  li a {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    color: var(--text-muted);
    font-size: 13px;
    transition: background 0.1s;
  }

  li a:hover {
    background: var(--bg-tertiary);
    color: var(--text);
  }

  li a.active {
    background: var(--bg-tertiary);
    color: var(--accent);
    border-right: 2px solid var(--accent);
  }

  .icon {
    font-size: 16px;
    width: 20px;
    text-align: center;
  }

  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
  }

  .connection-status {
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--yellow);
  }
</style>
