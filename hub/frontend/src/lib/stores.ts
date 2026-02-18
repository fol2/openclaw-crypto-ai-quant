/**
 * Simple reactive stores for shared state.
 *
 * Uses Svelte 5 $state runes pattern — these are plain objects that can be
 * imported and mutated from any component.
 */

export const appState = $state({
  mode: 'paper' as string,
  connected: false,
  mids: {} as Record<string, number>,
  snapshot: null as any,
  loading: false,
});
