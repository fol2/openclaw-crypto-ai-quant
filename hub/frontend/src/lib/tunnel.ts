type PositionLike = {
  type?: string | null;
  pos_type?: string | null;
  open_timestamp?: string | null;
} | null | undefined;

type JourneyLike = {
  type?: string | null;
  pos_type?: string | null;
  open_ts?: string | null;
  close_ts?: string | null;
} | null | undefined;

type TunnelPointLike = {
  ts_ms?: number | null;
  pos_type?: string | null;
  open_time_ms?: number | null;
};

function normalisePosType(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase();
}

function parseIsoTsMs(raw: string | null | undefined): number | undefined {
  if (!raw) return undefined;
  const parsed = Date.parse(String(raw).replace(' ', 'T'));
  return Number.isFinite(parsed) ? parsed : undefined;
}

function pointOpenTimeMs(point: TunnelPointLike | null | undefined): number | undefined {
  const value = Number(point?.open_time_ms);
  if (!Number.isFinite(value) || value <= 0) return undefined;
  return value;
}

function filterPointsBySide<T extends TunnelPointLike>(
  points: T[],
  posType: string,
): T[] {
  if (!posType) return points;
  return points.filter((point) => normalisePosType(point?.pos_type) === posType);
}

function latestOpenTimeMs<T extends TunnelPointLike>(points: T[]): number | undefined {
  let latest: number | undefined;
  for (const point of points) {
    const value = pointOpenTimeMs(point);
    if (value == null) continue;
    latest = latest == null ? value : Math.max(latest, value);
  }
  return latest;
}

export function tunnelFromTsForPosition(position: PositionLike): number | undefined {
  return parseIsoTsMs(position?.open_timestamp);
}

export function tunnelOpenTimeMsForJourney(journey: JourneyLike): number | undefined {
  return parseIsoTsMs(journey?.open_ts);
}

function tunnelCloseTimeMsForJourney(journey: JourneyLike): number | undefined {
  return parseIsoTsMs(journey?.close_ts);
}

export function filterTunnelPointsForPosition<T extends TunnelPointLike>(
  points: T[],
  position: PositionLike,
): T[] {
  if (!Array.isArray(points) || points.length === 0) return [];

  let filtered = [...points];
  const currentPosType = normalisePosType(position?.pos_type || position?.type);
  const currentOpenTimeMs = tunnelFromTsForPosition(position);

  if (currentOpenTimeMs != null) {
    const exact = filtered.filter(
      (point) => pointOpenTimeMs(point) === currentOpenTimeMs,
    );
    if (exact.length > 0) {
      filtered = exact;
    } else {
      filtered = filtered.filter((point) => Number(point?.ts_ms) >= currentOpenTimeMs);
    }
  }

  filtered = filterPointsBySide(filtered, currentPosType);
  if (filtered.length === 0) return [];

  if (currentOpenTimeMs != null) return filtered;

  const latestGroupOpenTimeMs = latestOpenTimeMs(filtered);
  if (latestGroupOpenTimeMs != null) {
    const latestGroup = filtered.filter(
      (point) => pointOpenTimeMs(point) === latestGroupOpenTimeMs,
    );
    if (latestGroup.length > 0) return latestGroup;
  }

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
  return suffix.length > 0 ? suffix.reverse() : filtered;
}

export function filterTunnelPointsForJourney<T extends TunnelPointLike>(
  points: T[],
  journey: JourneyLike,
): T[] {
  if (!Array.isArray(points) || points.length === 0) return [];

  const openTimeMs = tunnelOpenTimeMsForJourney(journey);
  const closeTimeMs = tunnelCloseTimeMsForJourney(journey);
  const posType = normalisePosType(journey?.pos_type || journey?.type);
  let filtered = [...points];

  if (openTimeMs != null) {
    const exact = filtered.filter((point) => pointOpenTimeMs(point) === openTimeMs);
    if (exact.length > 0) {
      filtered = exact;
    } else {
      filtered = filtered.filter((point) => Number(point?.ts_ms) >= openTimeMs);
    }
  }

  if (closeTimeMs != null) {
    filtered = filtered.filter((point) => Number(point?.ts_ms) <= closeTimeMs);
  }

  filtered = filterPointsBySide(filtered, posType);
  if (filtered.length === 0) return [];

  if (openTimeMs == null) return filtered;

  const exact = filtered.filter((point) => pointOpenTimeMs(point) === openTimeMs);
  return exact.length > 0 ? exact : filtered;
}
