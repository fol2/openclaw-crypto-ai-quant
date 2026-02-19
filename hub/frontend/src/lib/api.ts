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

export async function getTrendCloses(interval = '5m', limit = 60): Promise<{ closes: Record<string, number[]> }> {
  return apiFetch(`/api/trend-closes?interval=${interval}&limit=${limit}`);
}

export interface CandleData {
  t: number;
  t_close: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  n: number;
}

export async function getTrendCandles(interval = '30m', limit = 30): Promise<{ candles: Record<string, CandleData[]> }> {
  return apiFetch(`/api/trend-candles?interval=${interval}&limit=${limit}`);
}

export async function getVolumes(): Promise<{ volumes: Record<string, number> }> {
  return apiFetch('/api/volumes');
}

export async function postFlashDebug(events: Array<Record<string, any>>) {
  return apiFetch('/api/flash-debug', {
    method: 'POST',
    body: JSON.stringify({ events }),
  });
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

export async function getJourneys(mode = 'paper', limit = 50, offset = 0, symbol?: string) {
  const params = new URLSearchParams({ mode, limit: String(limit), offset: String(offset) });
  if (symbol) params.set('symbol', symbol);
  return apiFetch(`/api/journeys?${params}`);
}

export async function getCandlesRange(symbol: string, interval: string, fromTs?: number, toTs?: number, limit = 500) {
  const params = new URLSearchParams({ symbol, interval, limit: String(limit) });
  if (fromTs != null) params.set('from_ts', String(fromTs));
  if (toTs != null) params.set('to_ts', String(toTs));
  return apiFetch(`/api/candles/range?${params}`);
}

export async function getMarks(symbol: string, mode = 'paper') {
  return apiFetch(`/api/marks?symbol=${encodeURIComponent(symbol)}&mode=${encodeURIComponent(mode)}`);
}

export async function getDecisions(mode = 'paper', params: Record<string, string> = {}) {
  const qs = new URLSearchParams({ mode, ...params });
  return apiFetch(`/api/v2/decisions?${qs}`);
}

// ── Config API ──────────────────────────────────────────────────────

export async function getConfig(file = 'main') {
  return apiFetch(`/api/config?file=${encodeURIComponent(file)}`);
}

export async function getConfigRaw(file = 'main'): Promise<string> {
  const headers: Record<string, string> = {};
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  const resp = await fetch(`/api/config/raw?file=${encodeURIComponent(file)}`, { headers });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  return resp.text();
}

export async function putConfig(yaml: string, file = 'main') {
  return apiFetch(`/api/config?file=${encodeURIComponent(file)}`, {
    method: 'PUT',
    body: JSON.stringify({ yaml }),
  });
}

export async function reloadConfig(file = 'main') {
  return apiFetch(`/api/config/reload?file=${encodeURIComponent(file)}`, { method: 'POST' });
}

export async function getConfigHistory(file = 'main') {
  return apiFetch(`/api/config/history?file=${encodeURIComponent(file)}`);
}

export async function getConfigDiff(a: string, b: string, file = 'main') {
  const qs = new URLSearchParams({ a, b, file });
  return apiFetch(`/api/config/diff?${qs}`);
}

export async function getConfigFiles() {
  return apiFetch('/api/config/files');
}

// ── Backtest API ────────────────────────────────────────────────────

export async function runBacktest(opts: { config?: string; initial_balance?: number; symbol?: string }) {
  return apiFetch('/api/backtest/run', { method: 'POST', body: JSON.stringify(opts) });
}

export async function getBacktestJobs() {
  return apiFetch('/api/backtest/jobs');
}

export async function getBacktestStatus(id: string) {
  return apiFetch(`/api/backtest/${encodeURIComponent(id)}/status`);
}

export async function getBacktestResult(id: string) {
  return apiFetch(`/api/backtest/${encodeURIComponent(id)}/result`);
}

export async function cancelBacktest(id: string) {
  return apiFetch(`/api/backtest/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

// ── Sweep API ───────────────────────────────────────────────────────

export async function runSweep(opts: { config?: string; sweep_spec: string; initial_balance?: number }) {
  return apiFetch('/api/sweep/run', { method: 'POST', body: JSON.stringify(opts) });
}

export async function getSweepJobs() {
  return apiFetch('/api/sweep/jobs');
}

export async function getSweepStatus(id: string) {
  return apiFetch(`/api/sweep/${encodeURIComponent(id)}/status`);
}

export async function getSweepResults(id: string) {
  return apiFetch(`/api/sweep/${encodeURIComponent(id)}/results`);
}

export async function cancelSweep(id: string) {
  return apiFetch(`/api/sweep/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

// ── Factory API ─────────────────────────────────────────────────────

export async function getFactoryRuns() {
  return apiFetch('/api/factory/runs');
}

export async function getFactoryRun(date: string, runId: string) {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}`);
}

export async function getFactoryReport(date: string, runId: string) {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}/report`);
}

export async function getFactoryCandidates(date: string, runId: string) {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}/candidates`);
}

// ── System API ──────────────────────────────────────────────────────

export async function getSystemServices() {
  return apiFetch('/api/system/services');
}

export async function serviceAction(name: string, action: string) {
  return apiFetch(`/api/system/services/${encodeURIComponent(name)}/${encodeURIComponent(action)}`, { method: 'POST' });
}

export async function getDbStats() {
  return apiFetch('/api/system/db-stats');
}

export async function getDiskUsage() {
  return apiFetch('/api/system/disk');
}

export async function getServiceLogs(service: string, lines = 50) {
  return apiFetch(`/api/system/logs?service=${encodeURIComponent(service)}&lines=${lines}`);
}

// ── Trade API ────────────────────────────────────────────────────────

export async function tradePreview(body: Record<string, any>) {
  return apiFetch('/api/trade/preview', { method: 'POST', body: JSON.stringify(body) });
}

export async function tradeExecute(body: Record<string, any>) {
  return apiFetch('/api/trade/execute', { method: 'POST', body: JSON.stringify(body) });
}

export async function tradeClose(body: Record<string, any>) {
  return apiFetch('/api/trade/close', { method: 'POST', body: JSON.stringify(body) });
}

export async function tradeCancel(body: Record<string, any>) {
  return apiFetch('/api/trade/cancel', { method: 'POST', body: JSON.stringify(body) });
}

export async function tradeOpenOrders(symbol: string) {
  return apiFetch(`/api/trade/open-orders/${encodeURIComponent(symbol)}`);
}

export async function tradeJobResult(jobId: string) {
  return apiFetch(`/api/trade/${encodeURIComponent(jobId)}/result`);
}

export async function tradeEnabled() {
  return apiFetch('/api/trade/enabled');
}
