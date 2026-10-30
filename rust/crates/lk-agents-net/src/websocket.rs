//! The real WebSocket transport, over `tokio-tungstenite`.

use crate::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;

pub use tokio_tungstenite::tungstenite::{self, Message};

/// A socket for protocols that need binary frames and explicit ping/pong handling.
pub type RawWebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect with an optional protocol message limit; `None` preserves the transport default.
/// The caller owns its connection deadline and the returned socket's lifecycle.
pub async fn connect_with_limit(
    url: &str,
    headers: Vec<(String, String)>,
    max_message_size: Option<usize>,
) -> Result<RawWebSocket, TransportError> {
    use tungstenite::client::IntoClientRequest;

    let mut request =
        url.into_client_request().map_err(|error| TransportError::Connect(error.to_string()))?;
    for (name, value) in headers {
        let name = tungstenite::http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| TransportError::Connect(error.to_string()))?;
        let value = tungstenite::http::HeaderValue::from_str(&value)
            .map_err(|error| TransportError::Connect(error.to_string()))?;
        request.headers_mut().insert(name, value);
    }
    let config = max_message_size.map(|limit| {
        tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(limit))
            .max_frame_size(Some(limit))
    });
    let uri = request.uri();
    let host = uri.host().ok_or_else(|| TransportError::Connect(format!("no host in {url}")))?;
    let host = host.trim_start_matches('[').trim_end_matches(']').to_owned();
    let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("wss") { 443 } else { 80 });
    let stream = crate::dns::connect((host, port))
        .await
        .map_err(|error| TransportError::Connect(error.to_string()))?;
    let (socket, _response) =
        tokio_tungstenite::client_async_tls_with_config(request, stream, config, None)
            .await
            .map_err(classify_tungstenite)?;
    Ok(socket)
}

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
        let stream = connect_with_limit(url, headers, None).await?;
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

/// A rejected handshake's HTTP status (401 not retryable, 503 retryable).
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

struct TungsteniteSender {
    /// A `tokio` mutex: held across the `send` await, and `&self` is required.
    sink: tokio::sync::Mutex<futures_util::stream::SplitSink<RawWebSocket, Message>>,
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
    source: futures_util::stream::SplitStream<RawWebSocket>,
    close_code: Option<u16>,
}

#[async_trait]
impl WsReceiver for TungsteniteReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        use tokio_tungstenite::tungstenite::Message;
        Ok(match self.source.next().await {
            Some(Ok(Message::Text(text))) => WsMessage::Text(text.to_string()),
            Some(Ok(Message::Binary(data))) => WsMessage::Binary(data.to_vec()),
            Some(Ok(Message::Close(frame))) => {
                self.close_code = frame.map(|f| u16::from(f.code));
                WsMessage::Closed
            }
            None => WsMessage::Closed,
            // Ping/pong are answered by tungstenite.
            Some(Ok(_)) => WsMessage::Other,
            Some(Err(err)) => return Err(classify_tungstenite(err)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn connection_message_limit_refuses_an_oversized_prefix_without_waiting_for_payload() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let (release, hold) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            // A binary message advertising 129 bytes, with all payload withheld.
            socket.get_mut().write_all(&[0x82, 126, 0, 129]).await.unwrap();
            let _ = hold.await;
        });
        let mut socket = connect_with_limit(&url, vec![], Some(128)).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(1), socket.next()).await;
        let _ = release.send(());
        server.await.unwrap();
        assert!(matches!(
            result.expect("the size limit must be checked before waiting for payload"),
            Some(Err(tungstenite::Error::Capacity(_)))
        ));
    }
}
