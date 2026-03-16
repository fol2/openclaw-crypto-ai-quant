import { DEFAULT_PAPER_MODE } from './mode-labels';

/**
 * Simple reactive stores for shared state.
 *
 * Uses Svelte 5 $state runes pattern — these are plain objects that can be
 * imported and mutated from any component.
 */

export const appState = $state({
  mode: DEFAULT_PAPER_MODE as string,
  connected: false,
  mids: {} as Record<string, number>,
  snapshot: null as any,
  loading: false,
  focus: '' as string,
  feed: 'trades' as string,
  search: '' as string,
  paused: false,
});
