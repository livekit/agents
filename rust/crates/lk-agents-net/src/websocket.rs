//! The real WebSocket transport, over `tokio-tungstenite`.

use crate::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;

fn response_reason(status: i64) -> String {
    lk_agents_abi::errors::http_reason(status).unwrap_or("HTTP error").to_owned()
}

pub struct TungsteniteTransport;

#[async_trait]
impl WsTransport for TungsteniteTransport {
    /// Unbounded: the clients wrap this in `lk_agents_abi::timeout` on their session clock.
    async fn connect(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;

        let mut request =
            url.into_client_request().map_err(|e| TransportError::Connect(e.to_string()))?;
        for (name, value) in headers {
            let name =
                tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|e| TransportError::Connect(e.to_string()))?;
            let value = tokio_tungstenite::tungstenite::http::HeaderValue::from_str(&value)
                .map_err(|e| TransportError::Connect(e.to_string()))?;
            request.headers_mut().insert(name, value);
        }

        let (stream, _response) =
            tokio_tungstenite::connect_async(request).await.map_err(classify_tungstenite)?;
        let (sink, source) = stream.split();
        Ok((
            Arc::new(TungsteniteSender {
                sink: tokio::sync::Mutex::new(sink),
                closed: std::sync::atomic::AtomicBool::new(false),
            }),
            Box::new(TungsteniteReceiver { source, close_code: None }),
        ))
    }
}

/// A rejected handshake's HTTP status (401 not retryable, 503 retryable). Shared with
/// `eot::connector`.
pub fn classify_tungstenite(err: tokio_tungstenite::tungstenite::Error) -> TransportError {
    use tokio_tungstenite::tungstenite::Error;
    match err {
        Error::Http(response) => {
            let status = i64::from(response.status().as_u16());
            let headers: Vec<(String, String)> = response
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_owned(), v.to_owned())))
                .collect();
            let body = response.body().as_ref().map(|b| String::from_utf8_lossy(b).into_owned());
            TransportError::Status {
                status,
                message: body.clone().unwrap_or_else(|| response_reason(status)),
                headers,
                body,
            }
        }
        Error::ConnectionClosed | Error::AlreadyClosed => TransportError::Closed,
        Error::Io(e) => TransportError::Connect(e.to_string()),
        other => TransportError::Io(other.to_string()),
    }
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct TungsteniteSender {
    /// A `tokio` mutex: held across the `send` await, and `&self` is required.
    sink: tokio::sync::Mutex<
        futures_util::stream::SplitSink<WsStream, tokio_tungstenite::tungstenite::Message>,
    >,
    closed: std::sync::atomic::AtomicBool,
}

#[async_trait]
impl WsSender for TungsteniteSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(TransportError::Closed);
        }
        self.sink
            .lock()
            .await
            .send(tokio_tungstenite::tungstenite::Message::text(text))
            .await
            .map_err(classify_tungstenite)
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(TransportError::Closed);
        }
        self.sink
            .lock()
            .await
            .send(tokio_tungstenite::tungstenite::Message::binary(data))
            .await
            .map_err(classify_tungstenite)
    }

    fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = self.sink.lock().await.close().await;
    }
}

struct TungsteniteReceiver {
    source: futures_util::stream::SplitStream<WsStream>,
    close_code: Option<u16>,
}

#[async_trait]
impl WsReceiver for TungsteniteReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        use tokio_tungstenite::tungstenite::Message;
        Ok(match self.source.next().await {
            Some(Ok(Message::Text(text))) => WsMessage::Text(text.to_string()),
            Some(Ok(Message::Close(frame))) => {
                self.close_code = frame.map(|f| u16::from(f.code));
                WsMessage::Closed
            }
            None => WsMessage::Closed,
            // Ping/pong are answered by tungstenite; anything else is `Other` for the caller to
            // log.
            Some(Ok(_)) => WsMessage::Other,
            Some(Err(err)) => return Err(classify_tungstenite(err)),
        })
    }
}
