import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v?: number;
}

@customElement('candle-chart')
export class CandleChart extends LitElement {
  @property({ type: Array }) candles: Candle[] = [];
  @property({ type: Number }) width = 600;
  @property({ type: Number }) height = 300;
  @property({ type: String }) symbol = '';

  // Marks (entry lines)
  @property({ type: Array }) entries: { price: number; action: string }[] = [];
  @property({ type: Number }) entryPrice: number | null = null;

  static styles = css`
    :host { display: block; }
    canvas { width: 100%; height: 100%; }
  `;

  updated() {
    this.draw();
  }

  private draw() {
    const canvas = this.shadowRoot?.querySelector('canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = this.width;
    const h = this.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    const data = this.candles;
    if (!data || data.length < 2) {
      ctx.fillStyle = '#666';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('No candle data', w / 2, h / 2);
      return;
    }

    // Price range
    let minP = Infinity, maxP = -Infinity;
    for (const c of data) {
      if (c.l < minP) minP = c.l;
      if (c.h > maxP) maxP = c.h;
    }
    const pad = (maxP - minP) * 0.05 || 1;
    minP -= pad;
    maxP += pad;
    const range = maxP - minP;

    const margin = { top: 10, right: 50, bottom: 20, left: 10 };
    const cw = w - margin.left - margin.right;
    const ch = h - margin.top - margin.bottom;

    const barW = Math.max(1, (cw / data.length) * 0.7);
    const gap = cw / data.length;

    const priceToY = (p: number) => margin.top + ch - ((p - minP) / range) * ch;

    // Draw candles
    for (let i = 0; i < data.length; i++) {
      const c = data[i];
      const x = margin.left + i * gap + gap / 2;
      const isGreen = c.c >= c.o;
      const color = isGreen ? '#22c55e' : '#ef4444';

      // Wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, priceToY(c.h));
      ctx.lineTo(x, priceToY(c.l));
      ctx.stroke();

      // Body
      const bodyTop = priceToY(Math.max(c.o, c.c));
      const bodyBot = priceToY(Math.min(c.o, c.c));
      const bodyH = Math.max(1, bodyBot - bodyTop);
      ctx.fillStyle = color;
      ctx.fillRect(x - barW / 2, bodyTop, barW, bodyH);
    }

    // Entry price line
    if (this.entryPrice && this.entryPrice >= minP && this.entryPrice <= maxP) {
      const y = priceToY(this.entryPrice);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(w - margin.right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#3b82f6';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`ENTRY ${this.entryPrice.toFixed(2)}`, w - margin.right + 4, y + 3);
    }

    // Price labels
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
      const p = minP + (range * i) / steps;
      const y = priceToY(p);
      ctx.fillText(p.toFixed(2), w - margin.right + 4, y + 3);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(w - margin.right, y);
      ctx.stroke();
    }
  }

  render() {
    return html`<canvas></canvas>`;
  }
}
