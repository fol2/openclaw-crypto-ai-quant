pub mod broadcast;
pub mod topics;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::stream::StreamExt;
use futures::SinkExt;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::state::AppState;

/// WebSocket upgrade handler.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

#[derive(Debug, Deserialize)]
struct WsClientMsg {
    #[serde(rename = "type")]
    msg_type: String,
    topic: Option<String>,
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let hub = &state.broadcast;

    let mut subscriptions: HashMap<String, tokio::task::JoinHandle<()>> = HashMap::new();

    // Spawn a task that forwards broadcast messages to the client.
    let (tx_to_client, mut rx_to_client) = tokio::sync::mpsc::channel::<String>(64);

    let forward_task = tokio::spawn(async move {
        while let Some(msg) = rx_to_client.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Process incoming messages from the client (subscribe/unsubscribe).
    loop {
        tokio::select! {
            // Client message
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(parsed) = serde_json::from_str::<WsClientMsg>(&text) {
                            match parsed.msg_type.as_str() {
                                "subscribe" => {
                                    if let Some(topic) = &parsed.topic {
                                        if !subscriptions.contains_key(topic) {
                                            let tx = tx_to_client.clone();
                                            let mut sub_rx = hub.subscribe(topic);
                                            let handle = tokio::spawn(async move {
                                                while let Ok(msg) = sub_rx.recv().await {
                                                    if tx.send(msg).await.is_err() {
                                                        break;
                                                    }
                                                }
                                            });
                                            subscriptions.insert(topic.clone(), handle);
                                        }
                                    }
                                }
                                "unsubscribe" => {
                                    if let Some(topic) = &parsed.topic {
                                        if let Some(handle) = subscriptions.remove(topic) {
                                            handle.abort();
                                        }
                                    }
                                }
                                "ping" => {
                                    let _ = tx_to_client.send(r#"{"type":"pong"}"#.to_string()).await;
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    forward_task.abort();
    for handle in subscriptions.into_values() {
        handle.abort();
    }
}
