/**
 * WebSocket client with auto-reconnect and topic subscriptions.
 */

import { getAuthToken } from './api';

type MessageHandler = (data: any) => void;

export class HubWS {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private reconnectMs = 1000;
  private maxReconnectMs = 30000;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private pendingSubs: Set<string> = new Set();
  private connected = false;

  constructor(url?: string) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.baseUrl = url || `${proto}//${window.location.host}/ws`;
  }

  connect() {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    try {
      this.ws = new WebSocket(this.buildSocketUrl());
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectMs = 1000;

      // Re-subscribe to all pending topics.
      for (const topic of this.pendingSubs) {
        this.sendSubscribe(topic);
      }

      this.updateStatusIndicator(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (!msg || typeof msg !== 'object') return;

        const topic = this.extractTopic(msg);
        const payload = this.extractPayload(msg);
        if (topic) {
          this.dispatch(topic, payload);
          return;
        }

        if (payload && typeof payload === 'object') {
          if ('mids' in payload) this.dispatch('mids', payload);
          if ('bbo' in payload) this.dispatch('bbo', payload);
        }
      } catch {
        // Ignore parse errors.
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.updateStatusIndicator(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  subscribe(topic: string, handler: MessageHandler) {
    if (!this.handlers.has(topic)) {
      this.handlers.set(topic, new Set());
    }
    this.handlers.get(topic)!.add(handler);
    this.pendingSubs.add(topic);

    if (this.connected) {
      this.sendSubscribe(topic);
    }
  }

  unsubscribe(topic: string, handler: MessageHandler) {
    const set = this.handlers.get(topic);
    if (set) {
      set.delete(handler);
      if (set.size === 0) {
        this.handlers.delete(topic);
        this.pendingSubs.delete(topic);
      }
    }
  }

  private sendSubscribe(topic: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe', topic }));
    }
  }

  private dispatch(topic: string, payload: any) {
    const handlers = this.handlers.get(topic);
    if (!handlers) return;
    for (const handler of handlers) {
      handler(payload);
    }
  }

  private extractTopic(msg: Record<string, any>): string | null {
    const candidates = [msg.topic, msg.channel, msg.stream, msg.type];
    for (const raw of candidates) {
      const topic = typeof raw === 'string' ? raw.trim() : '';
      if (topic && this.handlers.has(topic)) return topic;
    }
    return null;
  }

  private extractPayload(msg: Record<string, any>) {
    for (const key of ['data', 'payload', 'body', 'message']) {
      if (Object.prototype.hasOwnProperty.call(msg, key) && msg[key] !== undefined) {
        return msg[key];
      }
    }
    return msg;
  }

  private buildSocketUrl() {
    const url = new URL(this.baseUrl, window.location.href);
    const token = getAuthToken();
    if (token) {
      url.searchParams.set('token', token);
    } else {
      url.searchParams.delete('token');
    }
    return url.toString();
  }

  private scheduleReconnect() {
    setTimeout(() => this.connect(), this.reconnectMs);
    this.reconnectMs = Math.min(this.reconnectMs * 1.5, this.maxReconnectMs);
  }

  private updateStatusIndicator(ok: boolean) {
    const el = document.getElementById('ws-status');
    if (el) {
      const dot = el.querySelector('.dot') as HTMLElement;
      if (dot) {
        dot.style.background = ok ? 'var(--green)' : 'var(--yellow)';
        dot.style.boxShadow = ok ? '0 0 6px var(--green)' : '0 0 6px var(--yellow)';
        dot.style.animation = ok ? 'none' : 'pulse 2s ease-in-out infinite';
      }
      const label = el.querySelector('.conn-label') as HTMLElement;
      if (label) {
        label.textContent = ok ? 'Connected' : 'Reconnecting';
      }
    }
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}

// Singleton instance.
export const hubWs = new HubWS();
