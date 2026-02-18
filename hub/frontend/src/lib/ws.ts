/**
 * WebSocket client with auto-reconnect and topic subscriptions.
 */

type MessageHandler = (data: any) => void;

export class HubWS {
  private url: string;
  private ws: WebSocket | null = null;
  private reconnectMs = 1000;
  private maxReconnectMs = 30000;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private pendingSubs: Set<string> = new Set();
  private connected = false;

  constructor(url?: string) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.url = url || `${proto}//${window.location.host}/ws`;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);
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
        const type = msg.type as string;
        if (type && this.handlers.has(type)) {
          for (const handler of this.handlers.get(type)!) {
            handler(msg.data ?? msg);
          }
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

  private scheduleReconnect() {
    setTimeout(() => this.connect(), this.reconnectMs);
    this.reconnectMs = Math.min(this.reconnectMs * 1.5, this.maxReconnectMs);
  }

  private updateStatusIndicator(ok: boolean) {
    const el = document.getElementById('ws-status');
    if (el) {
      const dot = el.querySelector('.dot') as HTMLElement;
      if (dot) {
        dot.style.background = ok
          ? 'var(--green)'
          : 'var(--yellow)';
      }
      const textNode = el.childNodes[el.childNodes.length - 1];
      if (textNode) {
        textNode.textContent = ok ? ' Connected' : ' Reconnecting…';
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
