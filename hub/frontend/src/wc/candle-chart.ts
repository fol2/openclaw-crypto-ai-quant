import { LitElement, html, css, type PropertyValues } from 'lit';
import { customElement, property } from 'lit/decorators.js';

export interface CandleData {
  t: number;
  t_close?: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v?: number;
  n?: number;
}

export interface EntryMark {
  price: number;
  action: string;
  size?: number;
}

// ─── Colour palette ───────────────────────────────────────────────────────────
const C = {
  bg:         '#0d0d14',
  gridLine:   'rgba(255,255,255,0.05)',
  gridMid:    'rgba(255,255,255,0.09)',
  sep:        'rgba(255,255,255,0.10)',
  upBody:     '#22c55e',
  dnBody:     '#ef4444',
  volUp:      'rgba(34,197,94,0.38)',
  volDn:      'rgba(239,68,68,0.38)',
  axis:       'rgba(255,255,255,0.32)',
  lastLine:   'rgba(255,255,255,0.24)',
  lastBg:     '#1e293b',
  lastFg:     '#f8fafc',
  entryAvg:   '#a78bfa',  // avg entry position — solid violet line
  entryLong:  '#3b82f6',
  entryShort: '#f59e0b',
  xhairV:     'rgba(255,255,255,0.20)',
  xhairH:     'rgba(255,255,255,0.10)',
  ttBg:       'rgba(9,9,16,0.95)',
  ttBorder:   'rgba(255,255,255,0.10)',
  ttText:     '#e2e8f0',
  ttDim:      'rgba(255,255,255,0.42)',
  noData:     'rgba(255,255,255,0.22)',
} as const;

// ─── Layout constants (logical px) ────────────────────────────────────────────
const RIGHT   = 68;  // right margin: price-axis labels
const TOP     = 6;   // top padding
const XAXIS_H = 20;  // x-axis row
const SEP_H   = 1;   // separator bar height

const FONT    = '10px "SF Mono","Fira Code",Consolas,monospace';
const FONT_XS = '9px "SF Mono","Fira Code",Consolas,monospace';
const FONT_TT = '11px "SF Mono","Fira Code",Consolas,monospace';
const FONT_B  = 'bold 10px "SF Mono","Fira Code",Consolas,monospace';

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmtPrice(p: number): string {
  if (!isFinite(p)) return '—';
  const a = Math.abs(p);
  if (a === 0)    return '0';
  if (a >= 10000) return p.toFixed(1);
  if (a >= 100)   return p.toFixed(2);
  if (a >= 1)     return p.toFixed(4);
  if (a >= 0.01)  return p.toFixed(6);
  return p.toFixed(8);
}

function fmtVol(v: number): string {
  if (!isFinite(v) || v <= 0) return '—';
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(0);
}

function fmtAxisTime(ts: number, iv: string): string {
  const d  = new Date(ts);
  const hh = d.getUTCHours().toString().padStart(2, '0');
  const mm = d.getUTCMinutes().toString().padStart(2, '0');
  const dd = d.getUTCDate().toString().padStart(2, '0');
  const mo = (d.getUTCMonth() + 1).toString().padStart(2, '0');
  if (iv === '1d')                    return `${mo}/${dd}`;
  if (iv === '4h' || iv === '1h')     return `${dd} ${hh}:00`;
  return `${hh}:${mm}`;
}

function fmtTooltipTime(ts: number): string {
  const d = new Date(ts);
  const p = (n: number) => n.toString().padStart(2, '0');
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}  ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
}

// ─── Component ────────────────────────────────────────────────────────────────
@customElement('candle-chart')
export class CandleChart extends LitElement {
  @property({ type: Array })  candles:    CandleData[] = [];
  @property({ type: Array })  entries:    EntryMark[]  = [];
  @property({ type: Number }) entryPrice: number       = 0;   // avg position entry (solid line)
  @property({ type: String }) symbol   = '';
  @property({ type: String }) interval = '';

  private _ro?: ResizeObserver;
  private _hoverIdx: number | null = null;
  private _hoverY = 0;
  private _listening = false;
  private _rafId: number | null = null;

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 160px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this._ro = new ResizeObserver(() => this._draw());
    this._ro.observe(this);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._ro?.disconnect();
    if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
  }

  protected willUpdate(changed: PropertyValues<this>) {
    if (changed.has('candles') && typeof this.candles === 'string') {
      try { this.candles = JSON.parse(this.candles as any); } catch { this.candles = []; }
    }
    if (changed.has('entries') && typeof this.entries === 'string') {
      try { this.entries = JSON.parse(this.entries as any); } catch { this.entries = []; }
    }
  }

  updated() {
    this._ensureListeners();
    this._draw();
  }

  private _ensureListeners() {
    if (this._listening) return;
    const cv = this.shadowRoot?.querySelector('canvas');
    if (!cv) return;
    cv.addEventListener('mousemove', this._onMove);
    cv.addEventListener('mouseleave', this._onLeave);
    this._listening = true;
  }

  private _onMove = (e: MouseEvent) => {
    const cv   = e.target as HTMLCanvasElement;
    const rect = cv.getBoundingClientRect();
    const n    = this.candles.length;
    if (n === 0) return;
    const W    = this.offsetWidth || 400;
    const slot = (W - RIGHT) / n;
    const idx  = Math.max(0, Math.min(n - 1, Math.floor((e.clientX - rect.left) / slot)));
    this._hoverIdx = idx;
    this._hoverY   = e.clientY - rect.top;
    // RAF throttle: skip if a draw is already scheduled
    if (this._rafId !== null) return;
    this._rafId = requestAnimationFrame(() => { this._rafId = null; this._draw(); });
  };

  private _onLeave = () => {
    if (this._hoverIdx === null) return;
    this._hoverIdx = null;
    if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    this._draw();
  };

  private _draw() {
    const cv = this.shadowRoot?.querySelector('canvas');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W   = this.offsetWidth  || 400;
    const H   = this.offsetHeight || 280;
    // H < 65: minimum to keep PRICE_H positive (layout uses 64px for non-price panels)
    if (W < 20 || H < 65) return;

    cv.width  = W * dpr;
    cv.height = H * dpr;
    cv.style.width  = `${W}px`;
    cv.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    // ── Background ────────────────────────────────────────────────────────────
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Sort oldest → newest (API returns newest-first)
    const data: CandleData[] = [...this.candles].sort((a, b) => a.t - b.t);
    const n = data.length;

    if (n < 2) {
      ctx.fillStyle    = C.noData;
      ctx.font         = FONT;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(n === 0 ? 'No data' : 'Loading…', (W - RIGHT) / 2, H / 2);
      return;
    }

    // ── Layout ────────────────────────────────────────────────────────────────
    const VOL_H   = Math.min(Math.max(Math.floor(H * 0.18), 36), 68);
    const PRICE_H = H - TOP - SEP_H - VOL_H - SEP_H - XAXIS_H;
    const volY0   = TOP + PRICE_H + SEP_H;
    const chartW  = W - RIGHT;

    // ── Price range ───────────────────────────────────────────────────────────
    let minP = Infinity, maxP = -Infinity;
    for (const c of data) {
      if (c.l < minP) minP = c.l;
      if (c.h > maxP) maxP = c.h;
    }
    for (const e of this.entries) {
      if (e.price > 0) {
        if (e.price < minP) minP = e.price;
        if (e.price > maxP) maxP = e.price;
      }
    }
    const pad    = (maxP - minP) * 0.06 || maxP * 0.01 || 1;
    minP -= pad; maxP += pad;
    const pRange = maxP - minP;
    const pToY   = (p: number) => TOP + PRICE_H - ((p - minP) / pRange) * PRICE_H;

    // ── Bar geometry ──────────────────────────────────────────────────────────
    const slot = chartW / n;
    const barW = Math.max(1, Math.min(slot * 0.72, 11));
    const xOf  = (i: number) => (i + 0.5) * slot;

    // ── Price grid ────────────────────────────────────────────────────────────
    ctx.lineWidth = 0.5;
    ctx.setLineDash([]);
    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
      const p = minP + (pRange * i) / gridSteps;
      const y = pToY(p);
      ctx.strokeStyle = i === Math.floor(gridSteps / 2) ? C.gridMid : C.gridLine;
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(chartW, y);
      ctx.stroke();
      ctx.fillStyle    = C.axis;
      ctx.font         = FONT;
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(fmtPrice(p), chartW + 4, y);
    }

    // ── Volume panel separator ────────────────────────────────────────────────
    ctx.strokeStyle = C.sep;
    ctx.lineWidth   = SEP_H;
    ctx.beginPath();
    ctx.moveTo(0, volY0 - 1); ctx.lineTo(W, volY0 - 1);
    ctx.stroke();

    // ── Volume bars (97th-percentile cap prevents outliers dominating scale) ──
    const rawVols = data.map(c => Math.max(0, c.v ?? 0));
    const sorted  = [...rawVols].filter(v => v > 0).sort((a, b) => a - b);
    let volMax = sorted[sorted.length - 1] || 1;
    if (sorted.length >= 10) {
      const p97 = sorted[Math.floor((sorted.length - 1) * 0.97)];
      if (p97 > 0 && volMax > p97 * 3) volMax = p97 * 3;
    }
    for (let i = 0; i < n; i++) {
      const vol = Math.min(rawVols[i], volMax);
      const vh  = (vol / volMax) * (VOL_H - 2);
      ctx.fillStyle = data[i].c >= data[i].o ? C.volUp : C.volDn;
      ctx.fillRect(xOf(i) - barW / 2, volY0 + VOL_H - vh, barW, Math.max(1, vh));
    }
    // Volume scale max label (top-left of volume panel)
    if (sorted.length > 0) {
      ctx.fillStyle    = C.ttDim;
      ctx.font         = FONT_XS;
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(fmtVol(volMax), 4, volY0 + 2);
    }

    // ── Wicks ─────────────────────────────────────────────────────────────────
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    for (let i = 0; i < n; i++) {
      const c = data[i];
      ctx.strokeStyle = c.c >= c.o ? C.upBody : C.dnBody;
      ctx.beginPath();
      ctx.moveTo(xOf(i), pToY(c.h));
      ctx.lineTo(xOf(i), pToY(c.l));
      ctx.stroke();
    }

    // ── Bodies ────────────────────────────────────────────────────────────────
    for (let i = 0; i < n; i++) {
      const c   = data[i];
      const top = pToY(Math.max(c.o, c.c));
      const bot = pToY(Math.min(c.o, c.c));
      ctx.fillStyle   = c.c >= c.o ? C.upBody : C.dnBody;
      // Last candle is still forming — render semi-transparent
      ctx.globalAlpha = i === n - 1 ? 0.55 : 1.0;
      ctx.fillRect(xOf(i) - barW / 2, top, barW, Math.max(1, bot - top));
    }
    ctx.globalAlpha = 1.0;

    // ── Entry price overlay lines ─────────────────────────────────────────────
    // Individual entries: dashed, max 5 (avoid cluttering chart with DCA stacks)
    ctx.lineWidth = 1;
    const visEntries = this.entries.slice(0, 5);
    for (const e of visEntries) {
      const y = pToY(e.price);
      if (y < TOP - 6 || y > TOP + PRICE_H + 6) continue;
      const col = /buy|long/i.test(e.action) ? C.entryLong : C.entryShort;
      ctx.strokeStyle = col;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(chartW, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle    = col;
      ctx.font         = FONT_XS;
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${e.action} ${fmtPrice(e.price)}`, chartW + 4, y - 1);
    }
    ctx.setLineDash([]);

    // Average entry: solid violet line — the position's true cost basis
    if (this.entryPrice > 0) {
      const y = pToY(this.entryPrice);
      if (y >= TOP - 6 && y <= TOP + PRICE_H + 6) {
        ctx.strokeStyle = C.entryAvg;
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(chartW, y);
        ctx.stroke();
        ctx.fillStyle    = C.entryAvg;
        ctx.font         = FONT_XS;
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`AVG ${fmtPrice(this.entryPrice)}`, chartW + 4, y - 1);
      }
    }

    // ── Last close line with label box ────────────────────────────────────────
    const last = data[n - 1];
    if (last) {
      const y = pToY(last.c);
      ctx.strokeStyle = C.lastLine;
      ctx.lineWidth   = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(chartW, y);
      ctx.stroke();
      ctx.setLineDash([]);
      const lbl = fmtPrice(last.c);
      ctx.font  = FONT_B;
      const tw  = ctx.measureText(lbl).width;
      const lx  = chartW + 2;
      const ly  = Math.max(TOP + 1, Math.min(y - 8, TOP + PRICE_H - 16));
      ctx.fillStyle = C.lastBg;
      ctx.fillRect(lx, ly, tw + 10, 16);
      ctx.fillStyle    = C.lastFg;
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(lbl, lx + 5, ly + 3);
    }

    // ── X-axis time labels ────────────────────────────────────────────────────
    const xAxisMid = volY0 + VOL_H + SEP_H + XAXIS_H / 2;
    const step     = Math.max(1, Math.ceil(n / 8));
    ctx.fillStyle    = C.axis;
    ctx.font         = FONT_XS;
    ctx.textBaseline = 'middle';
    ctx.textAlign    = 'center';
    for (let i = 0; i < n; i += step) {
      const cx = xOf(i);
      if (cx > chartW - 4) continue;
      ctx.fillText(fmtAxisTime(data[i].t, this.interval), cx, xAxisMid);
    }

    // ── Crosshair + Tooltip ───────────────────────────────────────────────────
    const hi = this._hoverIdx;
    if (hi !== null && hi >= 0 && hi < n) {
      const hc = data[hi];
      const hx = xOf(hi);
      const hy = this._hoverY;

      // Vertical crosshair spans price + volume areas
      ctx.strokeStyle = C.xhairV;
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, TOP); ctx.lineTo(hx, volY0 + VOL_H);
      ctx.stroke();

      // Horizontal crosshair (price area only)
      if (hy >= TOP && hy <= TOP + PRICE_H) {
        ctx.strokeStyle = C.xhairH;
        ctx.beginPath();
        ctx.moveTo(0, hy); ctx.lineTo(chartW, hy);
        ctx.stroke();
        // Price label on right axis
        ctx.setLineDash([]);
        const hp   = minP + ((TOP + PRICE_H - hy) / PRICE_H) * pRange;
        const hlbl = fmtPrice(hp);
        ctx.font   = FONT;
        const htw  = ctx.measureText(hlbl).width;
        const hly  = Math.max(TOP + 1, Math.min(hy - 8, TOP + PRICE_H - 16));
        ctx.fillStyle = 'rgba(40,50,68,0.94)';
        ctx.fillRect(chartW + 2, hly, htw + 10, 16);
        ctx.fillStyle    = C.ttText;
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(hlbl, chartW + 7, hly + 3);
      }

      ctx.setLineDash([]);
      this._drawTooltip(ctx, hc, hx, hy, W, TOP + PRICE_H);
    }
  }

  private _drawTooltip(
    ctx: CanvasRenderingContext2D,
    c: CandleData,
    hx: number, hy: number,
    W: number, maxY: number,
  ) {
    const up  = c.c >= c.o;
    const chg = c.o !== 0 ? ((c.c - c.o) / c.o * 100) : 0;
    const dir = up ? '▲' : '▼';

    const lines: Array<{ text: string; color?: string }> = [
      { text: fmtTooltipTime(c.t),                                            color: C.ttDim },
      { text: `O  ${fmtPrice(c.o)}` },
      { text: `H  ${fmtPrice(c.h)}` },
      { text: `L  ${fmtPrice(c.l)}` },
      { text: `C  ${fmtPrice(c.c)}  ${dir}${Math.abs(chg).toFixed(2)}%`,     color: up ? '#22c55e' : '#ef4444' },
    ];
    if (c.v != null && c.v > 0) lines.push({ text: `Vol  ${fmtVol(c.v)}`,    color: C.ttDim });
    if (c.n != null && c.n > 0) lines.push({ text: `${c.n.toLocaleString()} trades`, color: C.ttDim });

    const PAD    = 8;
    const LH     = 16;
    const TW     = 196;
    const TH     = lines.length * LH + PAD * 2;
    const chartW = W - RIGHT;

    let tx = hx + 14;
    let ty = hy - TH / 2;
    if (tx + TW > chartW - 2) tx = hx - TW - 14;
    if (tx < 2)               tx = 2;
    if (ty < TOP + 2)         ty = TOP + 2;
    if (ty + TH > maxY - 2)   ty = maxY - TH - 2;

    // Background box
    ctx.fillStyle   = C.ttBg;
    ctx.strokeStyle = C.ttBorder;
    ctx.lineWidth   = 1;
    if (typeof (ctx as any).roundRect === 'function') {
      ctx.beginPath();
      (ctx as any).roundRect(tx, ty, TW, TH, 4);
      ctx.fill(); ctx.stroke();
    } else {
      ctx.fillRect(tx, ty, TW, TH);
      ctx.strokeRect(tx, ty, TW, TH);
    }

    // Lines
    ctx.font         = FONT_TT;
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'top';
    for (let i = 0; i < lines.length; i++) {
      ctx.fillStyle = lines[i].color ?? C.ttText;
      ctx.fillText(lines[i].text, tx + PAD, ty + PAD + i * LH);
    }
  }

  render() {
    return html`<canvas></canvas>`;
  }
}
