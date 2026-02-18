<script lang="ts">
  import './app.css';
  import Sidebar from './components/Sidebar.svelte';
  import Dashboard from './pages/Dashboard.svelte';

  // Simple hash-based routing for SPA.
  let currentPage = $state(window.location.hash.slice(1) || 'dashboard');

  function handleHashChange() {
    currentPage = window.location.hash.slice(1) || 'dashboard';
  }

  $effect(() => {
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  });
</script>

<Sidebar {currentPage} />

<main class="main-content">
  {#if currentPage === 'dashboard'}
    <Dashboard />
  {:else if currentPage === 'config'}
    <h1>Config</h1>
    <p class="placeholder">YAML editor — coming in Phase 3</p>
  {:else if currentPage === 'backtest'}
    <h1>Backtest</h1>
    <p class="placeholder">Backtest runner — coming in Phase 4</p>
  {:else if currentPage === 'sweep'}
    <h1>Sweep</h1>
    <p class="placeholder">Sweep manager — coming in Phase 4</p>
  {:else if currentPage === 'factory'}
    <h1>Factory</h1>
    <p class="placeholder">Candidate review — coming in Phase 5</p>
  {:else if currentPage === 'grid'}
    <h1>Grid View</h1>
    <p class="placeholder">Multi-symbol grid — coming in Phase 6</p>
  {:else if currentPage === 'system'}
    <h1>System</h1>
    <p class="placeholder">Housekeeping — coming in Phase 7</p>
  {:else}
    <h1>Not Found</h1>
    <p>Unknown page: {currentPage}</p>
  {/if}
</main>

<style>
  .main-content {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  h1 {
    font-size: 20px;
    margin-bottom: 12px;
    font-weight: 600;
  }

  .placeholder {
    color: var(--text-muted);
    font-style: italic;
  }
</style>
