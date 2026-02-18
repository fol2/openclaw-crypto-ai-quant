<script lang="ts">
  import './app.css';
  import Sidebar from './components/Sidebar.svelte';
  import Dashboard from './pages/Dashboard.svelte';
  import Config from './pages/Config.svelte';
  import Backtest from './pages/Backtest.svelte';
  import Sweep from './pages/Sweep.svelte';
  import Factory from './pages/Factory.svelte';

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
    <Config />
  {:else if currentPage === 'backtest'}
    <Backtest />
  {:else if currentPage === 'sweep'}
    <Sweep />
  {:else if currentPage === 'factory'}
    <Factory />
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
