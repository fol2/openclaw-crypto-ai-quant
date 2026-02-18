/**
 * REST API fetch wrapper with auth token support.
 */

let authToken: string | null = null;

export function setAuthToken(token: string) {
  authToken = token;
}

async function apiFetch<T = any>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init?.headers as Record<string, string> || {}),
  };

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const resp = await fetch(path, {
    ...init,
    headers,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`API ${resp.status}: ${text}`);
  }

  return resp.json();
}

export async function getSnapshot(mode = 'paper') {
  return apiFetch(`/api/snapshot?mode=${encodeURIComponent(mode)}`);
}

export async function getMids() {
  return apiFetch('/api/mids');
}

export async function getCandles(symbol: string, interval?: string, limit = 200) {
  const params = new URLSearchParams({ symbol, limit: String(limit) });
  if (interval) params.set('interval', interval);
  return apiFetch(`/api/candles?${params}`);
}

export async function getHealth() {
  return apiFetch('/api/health');
}

export async function getMetrics(mode = 'paper') {
  return apiFetch(`/api/metrics?mode=${encodeURIComponent(mode)}`);
}

export async function getMarks(symbol: string, mode = 'paper') {
  return apiFetch(`/api/marks?symbol=${encodeURIComponent(symbol)}&mode=${encodeURIComponent(mode)}`);
}

export async function getDecisions(mode = 'paper', params: Record<string, string> = {}) {
  const qs = new URLSearchParams({ mode, ...params });
  return apiFetch(`/api/v2/decisions?${qs}`);
}
