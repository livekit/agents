//! The `/agent` WebSocket transport implemented with `tokio-tungstenite`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest};

use super::{Connected, WorkerSink, WorkerStream, WorkerTransport};
use super::{TransportError, TransportEvent};
use crate::protocol::{self, WorkerMessage};

type Ws =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

pub struct WsTransport {
    pub heartbeat: Duration,
}

#[async_trait]
impl WorkerTransport for WsTransport {
    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        let heartbeat = self.heartbeat;
        if heartbeat.is_zero() {
            return Err(TransportError::Io("heartbeat must be positive".into()));
        }
        let url = super::agent_url(url, worker_token, false);
        let mut req = url
            .into_client_request()
            .map_err(|e| TransportError::Io(format!("bad worker url: {e}")))?;

        for (k, v) in headers {
            let name: tokio_tungstenite::tungstenite::http::HeaderName =
                k.parse().map_err(|_| TransportError::Io(format!("bad header name: {k}")))?;
            let val =
                v.parse().map_err(|_| TransportError::Io(format!("bad header value for {k}")))?;
            req.headers_mut().insert(name, val);
        }

        let (ws, _resp) = tokio_tungstenite::connect_async_with_config(
            req,
            Some(
                tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
                    .max_message_size(Some(super::MAX_CONTROL_MESSAGE)),
            ),
            false,
        )
        .await
        .map_err(|e| TransportError::Io(e.to_string()))?;
        let (write, read) = ws.split();
        let write = Arc::new(Mutex::new(write));

        // tungstenite originates no pings, so without this a half-open connection is invisible
        // and every offered job times out.
        let pinger = write.clone();
        let pinger = tokio::spawn(async move {
            let mut tick = tokio::time::interval(heartbeat);
            tick.tick().await; // the first tick is immediate
            loop {
                tick.tick().await;
                if pinger.lock().await.send(Message::Ping(Vec::new().into())).await.is_err() {
                    return;
                }
            }
        });

        Ok(Connected {
            sink: Arc::new(WsSink { write, pinger }),
            stream: Box::new(WsStream {
                read,
                idle: heartbeat.saturating_mul(2),
                idle_deadline: tokio::time::Instant::now() + heartbeat.saturating_mul(2),
            }),
        })
    }
}

struct WsSink {
    pinger: tokio::task::JoinHandle<()>,
    write: Arc<Mutex<futures_util::stream::SplitSink<Ws, Message>>>,
}

impl Drop for WsSink {
    fn drop(&mut self) {
        self.pinger.abort();
    }
}

#[async_trait]
impl WorkerSink for WsSink {
    async fn send(&self, message: WorkerMessage) -> Result<(), TransportError> {
        let bytes = protocol::encode(&message);
        if bytes.len() > super::MAX_CONTROL_MESSAGE {
            return Err(TransportError::Protocol("control message exceeds 1 MiB".into()));
        }
        self.write
            .lock()
            .await
            .send(Message::Binary(bytes.into()))
            .await
            .map_err(|e| TransportError::Io(e.to_string()))
    }

    async fn close(&self) {
        self.pinger.abort();
        let _ = self.write.lock().await.close().await;
    }
}

struct WsStream {
    read: futures_util::stream::SplitStream<Ws>,
    idle: Duration,
    idle_deadline: tokio::time::Instant,
}

#[async_trait]
impl WorkerStream for WsStream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        loop {
            let next = match tokio::time::timeout_at(self.idle_deadline, self.read.next()).await {
                Ok(n) => n,
                // No frame and no pong for two heartbeats: report a close so the caller reconnects.
                Err(_) => {
                    return Err(TransportError::Closed("websocket heartbeat timed out".into()));
                }
            };
            if matches!(next, Some(Ok(_))) {
                self.idle_deadline = tokio::time::Instant::now() + self.idle;
            }
            match next {
                None => return Err(TransportError::Closed("websocket ended".into())),
                Some(Err(e)) => return Err(TransportError::Io(e.to_string())),
                Some(Ok(Message::Binary(b))) => {
                    return protocol::decode(&b)
                        .map(TransportEvent::ServerMessage)
                        .map_err(|e| TransportError::Protocol(e.to_string()));
                }
                Some(Ok(Message::Close(c))) => {
                    let reason = c
                        .map(|c| format!("websocket code {}: {}", c.code, c.reason))
                        .unwrap_or_else(|| "websocket closed without a status code".into());
                    return Err(TransportError::Closed(reason));
                }
                Some(Ok(Message::Text(t))) => {
                    log::warn!(target: crate::LOG, ws_data = truncate_ws_data(&t).as_str();
                        "unexpected message type");
                }
                // Tungstenite handles ping/pong internally.
                Some(Ok(_)) => {}
            }
        }
    }
}

/// Truncated to 128 characters for logging.
fn truncate_ws_data(data: &str) -> String {
    let n = data.chars().count();
    if n <= 128 {
        return data.to_owned();
    }
    let head: String = data.chars().take(128).collect();
    format!("{head}...(+{} more)", n - 128)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Catches: truncating by bytes; `&s[..128]` panics mid-codepoint on a non-ASCII payload.
    #[test]
    fn the_unexpected_frame_warning_truncates_by_characters_not_bytes() {
        let s = "é".repeat(200);
        let out = truncate_ws_data(&s);
        assert!(out.starts_with(&"é".repeat(128)));
        assert!(out.ends_with("...(+72 more)"));
        assert_eq!(truncate_ws_data("short"), "short");
        assert_eq!(truncate_ws_data(&"x".repeat(128)), "x".repeat(128));
    }
    #[tokio::test]
    async fn control_messages_hide_websocket_frames_and_preserve_close_details() {
        use tokio_tungstenite::tungstenite::protocol::{CloseFrame, frame::coding::CloseCode};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            assert_eq!(ws.next().await.unwrap().unwrap(), Message::Binary(Vec::new().into()));
            ws.send(Message::Text("hello".into())).await.unwrap();
            ws.send(Message::Ping(b"ping".to_vec().into())).await.unwrap();
            ws.send(Message::Binary(Vec::new().into())).await.unwrap();
            ws.send(Message::Binary(Vec::new().into())).await.unwrap();
            ws.send(Message::Close(Some(CloseFrame {
                code: CloseCode::Away,
                reason: "draining".into(),
            })))
            .await
            .unwrap();
        });
        let Connected { sink, mut stream, .. } = WsTransport { heartbeat: Duration::from_secs(30) }
            .connect(&url, None, vec![])
            .await
            .unwrap();
        sink.send(WorkerMessage::default()).await.unwrap();
        let received = tokio::time::timeout(Duration::from_secs(5), async {
            assert!(matches!(stream.recv().await.unwrap(), TransportEvent::ServerMessage(m) if m.message.is_none()));
            assert!(matches!(stream.recv().await.unwrap(), TransportEvent::ServerMessage(m) if m.message.is_none()));
            let err = stream.recv().await.unwrap_err();
            assert!(matches!(&err, TransportError::Closed(reason) if reason.contains("1001") && reason.contains("draining")), "{err:?}");
        }).await;
        received.expect("control receive must skip text and ping without stalling");
        sink.close().await;
        server.await.unwrap();
    }
    #[tokio::test(start_paused = true)]
    async fn cancelling_receive_does_not_restart_the_liveness_deadline() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let (release, hold) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let _ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let _ = hold.await;
        });
        let Connected { sink, mut stream } = WsTransport { heartbeat: Duration::from_secs(1) }
            .connect(&url, None, vec![])
            .await
            .unwrap();
        for _ in 0..7 {
            assert!(tokio::time::timeout(Duration::from_millis(250), stream.recv()).await.is_err());
        }
        let result = tokio::time::timeout(Duration::from_millis(500), stream.recv()).await.unwrap();
        assert!(
            matches!(result, Err(TransportError::Closed(reason)) if reason.contains("heartbeat"))
        );
        sink.close().await;
        let _ = release.send(());
        server.await.unwrap();
    }
}
