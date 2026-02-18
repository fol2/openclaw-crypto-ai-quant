use std::collections::HashMap;
use std::sync::RwLock;
use tokio::sync::broadcast;

const CHANNEL_CAPACITY: usize = 64;

/// Central broadcast hub.  Each topic has a `broadcast::Sender`.
///
/// Topics are created lazily on first subscribe or publish.
pub struct BroadcastHub {
    channels: RwLock<HashMap<String, broadcast::Sender<String>>>,
}

impl BroadcastHub {
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
        }
    }

    /// Subscribe to a topic.  Creates the channel if it doesn't exist.
    pub fn subscribe(&self, topic: &str) -> broadcast::Receiver<String> {
        {
            let channels = self.channels.read().unwrap();
            if let Some(tx) = channels.get(topic) {
                return tx.subscribe();
            }
        }

        let mut channels = self.channels.write().unwrap();
        let entry = channels
            .entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(CHANNEL_CAPACITY).0);
        entry.subscribe()
    }

    /// Publish a message to a topic.  No-op if no one is subscribed.
    pub fn publish(&self, topic: &str, message: String) {
        let channels = self.channels.read().unwrap();
        if let Some(tx) = channels.get(topic) {
            let _ = tx.send(message);
        }
    }
}

impl Default for BroadcastHub {
    fn default() -> Self {
        Self::new()
    }
}
