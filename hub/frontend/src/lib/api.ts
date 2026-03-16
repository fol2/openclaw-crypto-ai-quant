/**
 * REST API fetch wrapper with auth token support.
 */

import { normaliseMode } from './mode-labels';
import { appState } from './stores.svelte';

const AUTH_TOKEN_STORAGE_KEY = 'aiq_hub_auth_token';

let authToken: string | null = null;

if (typeof window !== 'undefined') {
  authToken = readStoredAuthToken();
}

function readStoredAuthToken(): string | null {
  try {
    const value = window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)?.trim() || '';
    return value || null;
  } catch {
    return null;
  }
}

function persistAuthToken(token: string | null) {
  if (typeof window === 'undefined') return;
  try {
    if (token) {
      window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    } else {
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    }
  } catch {
    // Ignore localStorage failures.
  }
}

export function setAuthToken(token: string) {
  const trimmed = token.trim();
  authToken = trimmed || null;
  persistAuthToken(authToken);
}

export function clearAuthToken() {
  authToken = null;
  persistAuthToken(null);
}

export function getAuthToken() {
  return authToken;
}

export function bootstrapAuthTokenFromLocation() {
  if (typeof window === 'undefined') return authToken;
  const url = new URL(window.location.href);
  const queryToken = url.searchParams.get('token')?.trim() || '';
  if (queryToken) {
    setAuthToken(queryToken);
    url.searchParams.delete('token');
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
  }
  return authToken;
}

export const normaliseHubMode = normaliseMode;

function resolveHubMode(mode?: string | null): string {
  return normaliseMode(mode || appState.mode);
}

function resolveExplicitHubMode(mode?: string | null): string | null {
  if (mode == null) return null;
  return normaliseMode(mode);
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

export async function getSnapshot(mode?: string) {
  return apiFetch(`/api/snapshot?mode=${encodeURIComponent(resolveHubMode(mode))}`);
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

export async function getCandles(symbol: string, interval?: string, limit = 200, mode?: string) {
  const params = new URLSearchParams({
    symbol,
    limit: String(limit),
  });
  const explicitMode = resolveExplicitHubMode(mode);
  if (explicitMode) params.set('mode', explicitMode);
  if (interval) params.set('interval', interval);
  return apiFetch(`/api/candles?${params}`);
}

export async function getHealth() {
  return apiFetch('/api/health');
}

export async function getMetrics(mode?: string) {
  return apiFetch(`/api/metrics?mode=${encodeURIComponent(resolveHubMode(mode))}`);
}

export async function getJourneys(mode?: string, limit = 50, offset = 0, symbol?: string) {
  const params = new URLSearchParams({
    mode: resolveHubMode(mode),
    limit: String(limit),
    offset: String(offset),
  });
  if (symbol) params.set('symbol', symbol);
  return apiFetch(`/api/journeys?${params}`);
}

export async function getTrades(mode?: string, limit = 100, offset = 0, symbol?: string, action?: string, fromTs?: string, toTs?: string) {
  const params = new URLSearchParams({
    mode: resolveHubMode(mode),
    limit: String(limit),
    offset: String(offset),
  });
  if (symbol) params.set('symbol', symbol);
  if (action) params.set('action', action);
  if (fromTs) params.set('from_ts', fromTs);
  if (toTs) params.set('to_ts', toTs);
  return apiFetch(`/api/trades?${params}`);
}

export async function getCandlesRange(symbol: string, interval: string, fromTs?: number, toTs?: number, limit = 500, mode?: string) {
  const params = new URLSearchParams({
    symbol,
    interval,
    limit: String(limit),
  });
  const explicitMode = resolveExplicitHubMode(mode);
  if (explicitMode) params.set('mode', explicitMode);
  if (fromTs != null) params.set('from_ts', String(fromTs));
  if (toTs != null) params.set('to_ts', String(toTs));
  return apiFetch(`/api/candles/range?${params}`);
}

export async function getMarks(symbol: string, mode?: string) {
  return apiFetch(`/api/marks?symbol=${encodeURIComponent(symbol)}&mode=${encodeURIComponent(resolveHubMode(mode))}`);
}

export async function getDecisions(mode?: string, params: Record<string, string> = {}) {
  const qs = new URLSearchParams({ mode: resolveHubMode(mode), ...params });
  return apiFetch(`/api/v2/decisions?${qs}`);
}

// ── Tunnel API ──────────────────────────────────────────

export async function getTunnel(
  symbol: string,
  mode?: string,
  fromTs?: number,
  toTs?: number,
  limit = 2000,
  openTimeMs?: number,
) {
  const params = new URLSearchParams({
    symbol,
    mode: resolveHubMode(mode),
    limit: String(limit),
  });
  if (fromTs != null) params.set('from_ts', String(fromTs));
  if (toTs != null) params.set('to_ts', String(toTs));
  if (openTimeMs != null) params.set('open_time_ms', String(openTimeMs));
  return apiFetch(`/api/tunnel?${params}`);
}

// ── Config API ──────────────────────────────────────────────────────

function normaliseListResponse<T>(value: unknown, key: string): T[] {
  if (Array.isArray(value)) return value as T[];
  if (value && typeof value === 'object') {
    const candidate = (value as Record<string, unknown>)[key];
    if (Array.isArray(candidate)) return candidate as T[];
  }
  return [];
}

export async function getConfig(file = 'main') {
  return apiFetch(`/api/config?file=${encodeURIComponent(file)}`);
}

export async function getConfigRaw(file = 'main'): Promise<{
  raw: string;
  lockId: string | null;
  runtimeConfigId: string | null;
}> {
  const headers: Record<string, string> = {};
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  const resp = await fetch(`/api/config/raw?file=${encodeURIComponent(file)}`, { headers });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const lockId = resp.headers.get('x-aiq-config-lock-id')
    || resp.headers.get('etag')?.replace(/^W\//, '').replace(/^"|"$/g, '')
    || null;
  return {
    raw: await resp.text(),
    lockId,
    runtimeConfigId: resp.headers.get('x-aiq-config-id'),
  };
}

export async function getConfigRawPrivileged(file = 'main'): Promise<{
  raw: string;
  lockId: string | null;
  runtimeConfigId: string | null;
}> {
  const headers: Record<string, string> = {};
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  const resp = await fetch(`/api/config/raw/privileged?file=${encodeURIComponent(file)}`, { headers });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const lockId = resp.headers.get('x-aiq-config-lock-id')
    || resp.headers.get('etag')?.replace(/^W\//, '').replace(/^"|"$/g, '')
    || null;
  return {
    raw: await resp.text(),
    lockId,
    runtimeConfigId: resp.headers.get('x-aiq-config-id'),
  };
}

export async function putConfig(yaml: string, file = 'main', expectedConfigId?: string | null) {
  return apiFetch(`/api/config?file=${encodeURIComponent(file)}`, {
    method: 'PUT',
    headers: expectedConfigId ? { 'If-Match': expectedConfigId } : undefined,
    body: JSON.stringify({ yaml }),
  });
}

export async function getConfigHistory(file = 'main') {
  const response = await apiFetch(`/api/config/history?file=${encodeURIComponent(file)}`);
  return normaliseListResponse(response, 'history');
}

export async function getConfigDiff(a: string, b: string, file = 'main') {
  const qs = new URLSearchParams({ a, b, file });
  return apiFetch(`/api/config/diff?${qs}`);
}

export async function getConfigDiffPrivileged(a: string, b: string, file = 'main') {
  const qs = new URLSearchParams({ a, b, file });
  return apiFetch(`/api/config/diff/privileged?${qs}`);
}

export async function getConfigFiles() {
  const response = await apiFetch('/api/config/files');
  return normaliseListResponse(response, 'files');
}

export interface PromoteLiveRequest {
  paper_mode?: 'paper' | 'paper1' | 'paper2' | 'paper3';
  config_id?: string;
  dry_run?: boolean;
}

export interface RollbackLiveRequest {
  steps?: number;
  reason?: string;
  restart?: 'auto' | 'always' | 'never';
  dry_run?: boolean;
}

export interface ApplyLiveRequest {
  yaml: string;
  reason?: string;
  restart?: 'auto' | 'always' | 'never';
  dry_run?: boolean;
}

export interface LiveApprovalDecision {
  reason?: string;
}

export async function promoteLiveConfig(body: PromoteLiveRequest = {}) {
  return apiFetch('/api/config/actions/promote-live', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function applyLiveConfig(body: ApplyLiveRequest, expectedConfigId?: string | null) {
  return apiFetch('/api/config/actions/apply-live', {
    method: 'POST',
    headers: expectedConfigId ? { 'If-Match': expectedConfigId } : undefined,
    body: JSON.stringify(body),
  });
}

export async function requestLiveApplyConfig(body: ApplyLiveRequest, expectedConfigId?: string | null) {
  return apiFetch('/api/config/actions/apply-live/request', {
    method: 'POST',
    headers: expectedConfigId ? { 'If-Match': expectedConfigId } : undefined,
    body: JSON.stringify(body),
  });
}

export async function rollbackLiveConfig(body: RollbackLiveRequest = {}) {
  return apiFetch('/api/config/actions/rollback-live', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function requestLiveRollbackConfig(body: RollbackLiveRequest = {}) {
  return apiFetch('/api/config/actions/rollback-live/request', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function getPendingConfigApprovals() {
  return apiFetch('/api/config/approvals?status=pending');
}

export async function approveConfigApproval(requestId: string, body: LiveApprovalDecision = {}) {
  return apiFetch(`/api/config/approvals/${encodeURIComponent(requestId)}/approve`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function rejectConfigApproval(requestId: string, body: LiveApprovalDecision = {}) {
  return apiFetch(`/api/config/approvals/${encodeURIComponent(requestId)}/reject`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
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

export interface FactoryCapability {
  compiled: boolean;
  policy_enabled: boolean;
  executor_wired: boolean;
  execution_enabled: boolean;
  mode: string;
  reason: string;
  enable_env: string;
  settings_path: string;
  service_units: string[];
}

export interface FactoryRunSummary {
  date: string;
  run_id: string;
  directory_name?: string;
  has_report: boolean;
  profile: string;
  candidate_count?: number | null;
  selected_count?: number | null;
  selection_stage?: string | null;
  deploy_stage?: string | null;
  generated_at_ms?: number | null;
}

export interface FactoryRunDetail {
  date: string;
  run_id: string;
  directory_name?: string | null;
  metadata: Record<string, any>;
  subdirs: string[];
  report_available?: boolean;
  candidate_count?: number | null;
  selection_summary?: Record<string, any> | null;
}

export interface FactoryTimerStatus {
  unit?: string;
  name?: string;
  active?: string;
  load?: string;
  enabled?: boolean;
  available?: boolean;
  unit_file_state?: string;
  mode?: string;
  next_trigger?: string;
}

export interface FactoryTimerResponse {
  capability?: FactoryCapability;
  timers: FactoryTimerStatus[];
}

export async function getFactoryCapability(): Promise<FactoryCapability> {
  return apiFetch('/api/factory/capability');
}

export async function getFactoryRuns(): Promise<FactoryRunSummary[]> {
  return apiFetch('/api/factory/runs');
}

export async function getFactoryRun(date: string, runId: string): Promise<FactoryRunDetail> {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}`);
}

export async function getFactoryReport(date: string, runId: string) {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}/report`);
}

export async function getFactoryCandidates(date: string, runId: string) {
  return apiFetch(`/api/factory/runs/${encodeURIComponent(date)}/${encodeURIComponent(runId)}/candidates`);
}

export async function runFactory(opts: Record<string, any>) {
  return apiFetch('/api/factory/run', { method: 'POST', body: JSON.stringify(opts) });
}

export async function getFactoryJobs() {
  return apiFetch('/api/factory/jobs');
}

export async function getFactoryJobStatus(id: string) {
  return apiFetch(`/api/factory/jobs/${encodeURIComponent(id)}/status`);
}

export async function cancelFactory(id: string) {
  return apiFetch(`/api/factory/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function getFactorySettings() {
  return apiFetch('/api/factory/settings');
}

export async function putFactorySettings(settings: Record<string, any>) {
  return apiFetch('/api/factory/settings', { method: 'PUT', body: JSON.stringify(settings) });
}

export async function getFactoryTimer(): Promise<FactoryTimerResponse> {
  return apiFetch('/api/factory/timer');
}

export async function factoryTimerAction(action: string) {
  return apiFetch(`/api/factory/timer/${encodeURIComponent(action)}`, { method: 'POST' });
}

// ── System API ──────────────────────────────────────────────────────

export interface SystemServiceSummary {
  name: string;
  active: string;
  sub: string;
  pid: string;
  load: string;
  status: 'ok' | 'bad' | 'unknown' | 'dormant';
  dormant: boolean;
}

export interface SystemDbStat {
  label: string;
  path_redacted: boolean;
  exists: boolean;
  size_bytes: number;
  size_mb: string;
  modified: string;
}

export interface SystemDiskUsage {
  label: string;
  path_redacted: boolean;
  size: string;
}

export interface SystemLogsResponse {
  service: string;
  lines?: number;
  log?: string;
  redacted?: boolean;
  message?: string;
}

function hasApiStatus(error: unknown, statuses: number[]) {
  return error instanceof Error && statuses.some((status) => error.message.includes(`API ${status}`));
}

export async function getSystemServices(): Promise<SystemServiceSummary[]> {
  return apiFetch('/api/system/services');
}

export async function serviceAction(name: string, action: string) {
  return apiFetch(`/api/system/services/${encodeURIComponent(name)}/${encodeURIComponent(action)}`, { method: 'POST' });
}

export async function getDbStats(): Promise<SystemDbStat[]> {
  return apiFetch('/api/system/db-stats');
}

export async function getDiskUsage(): Promise<SystemDiskUsage[]> {
  return apiFetch('/api/system/disk');
}

export async function getServiceLogs(service: string, lines = 50): Promise<SystemLogsResponse> {
  return apiFetch(`/api/system/logs?service=${encodeURIComponent(service)}&lines=${lines}`);
}

export async function getServiceLogsPrivileged(service: string, lines = 50): Promise<SystemLogsResponse> {
  return apiFetch(`/api/system/logs/raw?service=${encodeURIComponent(service)}&lines=${lines}`);
}

export async function getSystemPageServiceLogs(service: string, lines = 50): Promise<SystemLogsResponse> {
  if (!authToken) {
    return getServiceLogs(service, lines);
  }

  try {
    return await getServiceLogsPrivileged(service, lines);
  } catch (error) {
    if (hasApiStatus(error, [401, 403])) {
      return getServiceLogs(service, lines);
    }
    throw error;
  }
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
