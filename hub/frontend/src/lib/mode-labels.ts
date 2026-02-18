export const LIVE_MODE = 'live' as const;
export const CANDIDATE_FAMILY_ORDER = ['paper1', 'paper2', 'paper3'] as const;
export const MODE_ORDER = [LIVE_MODE, ...CANDIDATE_FAMILY_ORDER] as const;

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
  return MODE_LABELS[mode] ?? mode;
}

export function getConfigLabel(config: string): string {
  return CONFIG_LABELS[config] ?? config;
}
