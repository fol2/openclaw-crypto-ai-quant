type PositionLike = {
  type?: string | null;
  pos_type?: string | null;
  open_timestamp?: string | null;
} | null | undefined;

type TunnelPointLike = {
  ts_ms?: number | null;
  pos_type?: string | null;
};

function normalisePosType(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase();
}

export function tunnelFromTsForPosition(position: PositionLike): number | undefined {
  const raw = position?.open_timestamp;
  if (!raw) return undefined;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function filterTunnelPointsForPosition<T extends TunnelPointLike>(
  points: T[],
  position: PositionLike,
): T[] {
  if (!Array.isArray(points) || points.length === 0) return [];

  let filtered = [...points];
  const fromTs = tunnelFromTsForPosition(position);
  if (fromTs != null) {
    filtered = filtered.filter((point) => Number(point?.ts_ms) >= fromTs);
  }

  const currentPosType = normalisePosType(position?.pos_type || position?.type);
  if (!currentPosType || filtered.length === 0) return filtered;

  const suffix: T[] = [];
  let seenCurrentSide = false;
  for (let index = filtered.length - 1; index >= 0; index -= 1) {
    const point = filtered[index];
    if (normalisePosType(point?.pos_type) !== currentPosType) {
      if (seenCurrentSide) break;
      continue;
    }
    seenCurrentSide = true;
    suffix.push(point);
  }
  if (suffix.length > 0) return suffix.reverse();

  return filtered.filter(
    (point) => normalisePosType(point?.pos_type) === currentPosType,
  );
}
