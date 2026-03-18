// Shared formatting utilities used by Dashboard, SymbolDetailPanel, and Modal.

export function fmtNum(v: number | null | undefined, dp = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '\u2014';
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

export function fmtAge(s: number | null | undefined): string {
  if (s === null || s === undefined) return '\u2014';
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86_400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86_400).toFixed(1)}d`;
}

export function sigAge(ts: string | null | undefined, nowMs?: number): string {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '';
  const now = nowMs ?? Date.now();
  return fmtAge((now - d.getTime()) / 1000);
}

export function pnlPct(pos: any): number | null {
  if (!pos || pos.entry_price == null || !pos.size || pos.unreal_pnl_est == null) return null;
  const notional = pos.entry_price * Math.abs(pos.size);
  if (notional === 0) return null;
  return (pos.unreal_pnl_est / notional) * 100;
}

export function pnlClass(v: number | null | undefined): string {
  if (v === null || v === undefined) return '';
  return v >= 0 ? 'green' : 'red';
}

export function entryActionLabel(action: string | null | undefined): string {
  const value = String(action || '').toUpperCase();
  if (value === 'REDUCE') return 'PARTIAL CLOSE';
  if (value === 'CLOSE') return 'FULL CLOSE';
  return value || '\u2014';
}

export function journeySourceLabel(source: string | null | undefined): string {
  const value = String(source || '').toLowerCase();
  if (value === 'manual') return 'MANUAL';
  if (value === 'mixed') return 'MANUAL+AUTO';
  return '';
}

export function tradeTimeLabel(ts: string | null | undefined): string {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return String(ts).slice(11, 19);
  return `${d.toISOString().slice(11, 19)}Z`;
}

export function intervalToMs(iv: string): number {
  const m = /^([0-9]+)([mhd])$/i.exec(String(iv || '').trim());
  if (!m) return 60_000;
  const n = Number(m[1]);
  if (!Number.isFinite(n) || n <= 0) return 60_000;
  const unit = String(m[2] || '').toLowerCase();
  if (unit === 'm') return n * 60_000;
  if (unit === 'h') return n * 60 * 60_000;
  if (unit === 'd') return n * 24 * 60 * 60_000;
  return 60_000;
}

export function fmtDuration(ms: number): string {
  if (ms < 0) ms = 0;
  const mins = Math.floor(ms / 60_000);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  const rm = mins % 60;
  if (hrs < 24) return rm > 0 ? `${hrs}h ${rm}m` : `${hrs}h`;
  const days = Math.floor(hrs / 24);
  const rh = hrs % 24;
  return rh > 0 ? `${days}d ${rh}h` : `${days}d`;
}

export function pickJourneyInterval(durationMs: number): string {
  if (durationMs < 30 * 60_000)       return '1m';
  if (durationMs < 2 * 60 * 60_000)   return '3m';
  if (durationMs < 6 * 60 * 60_000)   return '5m';
  if (durationMs < 24 * 60 * 60_000)  return '15m';
  return '1h';
}

export function journeyTimeRange(j: any): { openTs: number; closeTs: number; dur: number; fromTs: number; toTs: number } | null {
  const openTs = Date.parse((j.open_ts || '').replace(' ', 'T'));
  const closeTs = j.close_ts ? Date.parse((j.close_ts || '').replace(' ', 'T')) : Date.now();
  if (!isFinite(openTs)) return null;
  const dur = closeTs - openTs;
  const pad = Math.max(dur * 0.1, 60_000);
  return { openTs, closeTs, dur, fromTs: Math.floor(openTs - pad), toTs: Math.ceil(closeTs + pad) };
}
