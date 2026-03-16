export const LIVE_MODE = 'live' as const;
export const DEFAULT_PAPER_MODE = 'paper1' as const;
export const CANDIDATE_FAMILY_ORDER = [DEFAULT_PAPER_MODE, 'paper2', 'paper3'] as const;
export const MODE_ORDER = [LIVE_MODE, ...CANDIDATE_FAMILY_ORDER] as const;
const CANONICAL_MODES = new Set<string>(MODE_ORDER);

const MODE_LABELS: Record<string, string> = {
  live: 'Live',
  paper: 'Efficient',
  paper1: 'Efficient',
  paper2: 'Growth',
  paper3: 'Conservative',
};

const CONFIG_LABELS: Record<string, string> = {
  main: 'Main',
  live: MODE_LABELS.live,
  paper1: MODE_LABELS.paper1,
  paper2: MODE_LABELS.paper2,
  paper3: MODE_LABELS.paper3,
};

export function getModeLabel(mode: string): string {
  const key = String(mode || '').trim().toLowerCase();
  if (key === 'paper') return MODE_LABELS[DEFAULT_PAPER_MODE];
  return MODE_LABELS[key] ?? mode;
}

export function getConfigLabel(config: string): string {
  return CONFIG_LABELS[config] ?? config;
}

export function normaliseMode(mode: string | null | undefined): string {
  const key = String(mode || '').trim().toLowerCase();
  if (!key || key === 'paper') return DEFAULT_PAPER_MODE;
  return CANONICAL_MODES.has(key) ? key : DEFAULT_PAPER_MODE;
}
