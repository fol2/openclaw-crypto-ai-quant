import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';

/**
 * <price-tick value={number} dp={number}>
 *
 * Displays a formatted price and flashes green/red when the value changes.
 * Each instance is self-contained: it tracks its own previous value and
 * drives the Web Animations API directly — no external flash state needed.
 *
 * At 100ms WS intervals a symbol may change 10×/sec. Each incoming change
 * cancels the running animation and restarts it immediately, so the flash
 * always reflects the latest tick regardless of animation progress.
 *
 * No shadow DOM — inherits color/font from the parent element.
 * Resting color is read lazily from getComputedStyle on first update.
 */
@customElement('price-tick')
export class PriceTick extends LitElement {
  @property({ type: Number }) value: number | null = null;
  @property({ type: Number }) dp = 6;

  #prevValue: number | null = null;
  #restColor = '';

  // No shadow DOM — fully inherits parent styles (color, font, size).
  override createRenderRoot() { return this; }

  override updated(changed: Map<string, unknown>) {
    if (!changed.has('value')) return;

    const prev = this.#prevValue;
    this.#prevValue = this.value;

    if (prev === null || this.value === null || prev === this.value) return;

    // Lazily capture resting color on first animation — styles are resolved by now.
    if (!this.#restColor) this.#restColor = getComputedStyle(this).color;

    const dir = this.value > prev ? 'up' : 'down';
    const flashColor = dir === 'up' ? 'var(--green)' : 'var(--red)';

    this.getAnimations().forEach(a => a.cancel());
    this.animate(
      [
        { color: flashColor },
        { color: flashColor, offset: 0.6 },
        { color: this.#restColor },
      ],
      { duration: 500, easing: 'ease-out' }
    );
  }

  override render() {
    if (this.value == null) return html`&mdash;`;
    return html`${this.value.toLocaleString('en-US', {
      minimumFractionDigits: this.dp,
      maximumFractionDigits: this.dp,
    })}`;
  }
}
