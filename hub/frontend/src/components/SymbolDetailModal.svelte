<script lang="ts">
  import SymbolDetailPanel from './SymbolDetailPanel.svelte';

  let { symbol, mode, snap, mids, onclose }: {
    symbol: string;
    mode: string;
    snap: any;
    mids: Record<string, number>;
    onclose: () => void;
  } = $props();

  // ── Derived state (from parent props) ───────────────────────────────
  let symbolData = $derived(snap?.symbols?.find((s: any) => s.symbol === symbol));
  let liveMid = $derived(mids[symbol] ?? symbolData?.mid ?? 0);
  let recent = $derived(snap?.recent || {});

  // ── Mobile swipe-down state ───────────────────────────────────────────
  let sheetTranslateY = $state(0);
  let sheetSwiping = $state(false);

  function onDragHandleDown(e: PointerEvent) {
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    sheetSwiping = true;
    const startY = e.clientY;
    function onMove(ev: PointerEvent) {
      const dy = ev.clientY - startY;
      sheetTranslateY = Math.max(0, dy);
    }
    function onUp() {
      sheetSwiping = false;
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
      if (sheetTranslateY > 120) { onclose(); }
      sheetTranslateY = 0;
    }
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
  }

  // ── Lifecycle: Escape key ─────────────────────────────────────────────
  $effect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onclose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  // ── Lifecycle: Body scroll lock ───────────────────────────────────────
  $effect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="modal-backdrop" onclick={onclose}></div>

<div
  class="modal-container"
  style={sheetTranslateY > 0 ? `transform: translateY(${sheetTranslateY}px)` : ''}
>
  <!-- Mobile drag handle -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="drag-handle" onpointerdown={onDragHandleDown}>
    <div class="drag-pill"></div>
  </div>

  <!-- Header -->
  <div class="modal-header">
    <div class="header-left">
      <h3>{symbol}</h3>
      <mid-price
        symbol={symbol}
        tone="accent"
        value={liveMid ? String(liveMid) : ''}
        decimals={6}
      ></mid-price>
    </div>
    <button class="close-btn" aria-label="Close" onclick={onclose}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
    </button>
  </div>

  <!-- Content -->
  <div class="modal-body">
    <SymbolDetailPanel
      {symbol}
      {mode}
      {symbolData}
      {liveMid}
      {recent}
      isLiveMode={mode === 'live'}
    />
  </div>
</div>

<style>
  /* ── Backdrop ──────────────────────────────────────────────────────── */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 90;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(4px);
    animation: fadeIn 0.2s ease;
  }

  /* ── Container: Desktop ────────────────────────────────────────────── */
  .modal-container {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 95;
    width: 90vw;
    max-width: 720px;
    max-height: 85vh;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: modalSlideUp 0.25s ease-out;
  }

  /* ── Drag handle (mobile only) ─────────────────────────────────────── */
  .drag-handle {
    display: none;
    justify-content: center;
    padding: 8px 0 4px;
    touch-action: none;
    cursor: grab;
    flex-shrink: 0;
  }
  .drag-pill {
    width: 32px;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
  }

  /* ── Header ────────────────────────────────────────────────────────── */
  .modal-header {
    display: flex;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    gap: 8px;
    flex-shrink: 0;
  }
  .header-left {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex: 1;
    min-width: 0;
  }
  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.01em;
  }
  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    cursor: pointer;
    flex-shrink: 0;
  }
  .close-btn:hover {
    background: var(--surface-hover);
    color: var(--text);
  }

  /* ── Body ───────────────────────────────────────────────────────────── */
  .modal-body {
    overflow-y: auto;
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  /* ── Animations ────────────────────────────────────────────────────── */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  @keyframes modalSlideUp {
    from { opacity: 0; transform: translate(-50%, -48%); }
    to { opacity: 1; transform: translate(-50%, -50%); }
  }
  @keyframes sheetSlideUp {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
  }

  /* ── Mobile: bottom sheet ──────────────────────────────────────────── */
  @media (max-width: 768px) {
    .modal-container {
      top: auto;
      left: 0;
      right: 0;
      bottom: 0;
      transform: none;
      width: 100%;
      max-width: 100%;
      max-height: 95dvh;
      border-radius: var(--radius-lg) var(--radius-lg) 0 0;
      animation: sheetSlideUp 0.3s ease-out;
    }

    .drag-handle {
      display: flex;
    }

    .modal-header {
      flex-wrap: wrap;
    }
  }
</style>
