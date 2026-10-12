//! The real transport: `reqwest` for HTTPS/SSE and `tokio-tungstenite` for the WebSockets.

use crate::transport::{
    HttpRequest, HttpStream, HttpTransport, LineStream, TransportError, WsMessage, WsReceiver,
    WsSender, WsTransport,
};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use std::time::Duration;

/// The reference's connection limits and 15 s `connect_timeout`; no read or total timeout, since
/// [`crate::llm`] bounds each wait on the session clock.
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub fn new() -> Result<Self, TransportError> {
        crate::transport::install_tls_provider();
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(50)
            .pool_idle_timeout(Duration::from_secs(120))
            .connect_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| TransportError::Connect(e.to_string()))?;
        Ok(ReqwestTransport { client })
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn post_stream(&self, req: HttpRequest) -> Result<HttpStream, TransportError> {
        let mut builder = self.client.post(&req.url).body(req.body);
        for (name, value) in &req.headers {
            builder = builder.header(name, value);
        }
        let response = builder.send().await.map_err(classify_reqwest)?;

        let status = i64::from(response.status().as_u16());
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_owned(), v.to_owned())))
            .collect();

        if !response.status().is_success() {
            // The body is read here only; the reference's `APIStatusError` carries it.
            let body = response.text().await.ok();
            // The body first, the status phrase only as a last resort (a plain-text
            // `404 page not found` from a doubled `/v1` must survive).
            let message = body
                .as_deref()
                .and_then(gateway_error_message)
                .or_else(|| body.as_deref().map(truncate_body).filter(|b| !b.is_empty()))
                .unwrap_or_else(|| response_reason(status));
            return Err(TransportError::Status { status, message, headers, body });
        }

        Ok(HttpStream { status, headers, body: Box::new(SseStream::new(response)) })
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        ReqwestTransport::new().expect("a default reqwest client")
    }
}

fn response_reason(status: i64) -> String {
    lk_agents_abi::errors::http_reason(status).unwrap_or("HTTP error").to_owned()
}

/// The message from a rejected request's body: OpenAI's `{"error": {"message": ...}}`, or plain
/// text.
fn gateway_error_message(body: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(body).ok()?;
    parsed
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .map(str::to_owned)
        .or_else(|| parsed.get("message").and_then(|m| m.as_str()).map(str::to_owned))
}

fn truncate_body(body: &str) -> String {
    const MAX: usize = 512;
    let body = body.trim();
    match body.char_indices().nth(MAX) {
        Some((idx, _)) => format!("{}…", &body[..idx]),
        None => body.to_owned(),
    }
}

fn classify_reqwest(err: reqwest::Error) -> TransportError {
    if err.is_timeout() {
        TransportError::Timeout
    } else if err.is_connect() {
        TransportError::Connect(err.to_string())
    } else {
        TransportError::Io(err.to_string())
    }
}

/// Server-sent events over a chunked body; payload decoding is `crate::llm::parse`.
struct SseStream {
    inner:
        std::pin::Pin<Box<dyn futures_util::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buf: String,
    done: bool,
}

impl SseStream {
    fn new(response: reqwest::Response) -> Self {
        SseStream { inner: Box::pin(response.bytes_stream()), buf: String::new(), done: false }
    }

    fn take_event(&mut self) -> Option<String> {
        loop {
            let idx = self.buf.find('\n')?;
            let line: String = self.buf.drain(..=idx).collect();
            let line = line.trim_end_matches(['\r', '\n']);
            // Blank lines separate events; `:` opens a comment (the keep-alive).
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            let Some(payload) = line.strip_prefix("data:") else { continue };
            let payload = payload.trim_start();
            if payload == "[DONE]" {
                self.done = true;
                return None;
            }
            return Some(payload.to_owned());
        }
    }
}

#[async_trait]
impl LineStream for SseStream {
    async fn next_event(&mut self) -> Result<Option<String>, TransportError> {
        loop {
            if self.done {
                return Ok(None);
            }
            if let Some(event) = self.take_event() {
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }
            match self.inner.next().await {
                Some(Ok(chunk)) => {
                    self.buf.push_str(&String::from_utf8_lossy(&chunk));
                }
                Some(Err(err)) => return Err(classify_reqwest(err)),
                None => {
                    self.done = true;
                    // A body ending without `[DONE]` ends the response, as in the reference.
                    return Ok(self.take_event());
                }
            }
        }
    }
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
pub(crate) fn classify_tungstenite(err: tokio_tungstenite::tungstenite::Error) -> TransportError {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(chunks: &[&str]) -> SseStream {
        // Constructed by hand: the framing is what is under test, not reqwest.
        let mut stream = SseStream {
            inner: Box::pin(futures_util::stream::empty()),
            buf: String::new(),
            done: false,
        };
        for chunk in chunks {
            stream.buf.push_str(chunk);
        }
        stream
    }

    // Catches: treating an SSE keep-alive comment as a data payload.
    #[test]
    fn sse_comments_and_blank_lines_are_not_events() {
        let mut stream = sse(&[": keep-alive\n", "\n", "data: {\"id\":\"1\"}\n", "\n"]);
        assert_eq!(stream.take_event().as_deref(), Some(r#"{"id":"1"}"#));
        assert_eq!(stream.take_event(), None);
    }

    // Catches: emitting `[DONE]` as a payload, or failing to stop on it.
    #[test]
    fn the_done_sentinel_ends_the_stream_and_is_not_emitted() {
        let mut stream =
            sse(&["data: {\"id\":\"1\"}\n", "data: [DONE]\n", "data: {\"id\":\"2\"}\n"]);
        assert_eq!(stream.take_event().as_deref(), Some(r#"{"id":"1"}"#));
        assert_eq!(stream.take_event(), None);
        assert!(stream.done);
    }

    // Catches: a payload split across two TCP chunks parsed as two events.
    #[test]
    fn a_payload_split_across_chunks_is_reassembled() {
        let mut stream = sse(&["data: {\"id\":\"", "abc\"}\n"]);
        assert_eq!(stream.take_event().as_deref(), Some(r#"{"id":"abc"}"#));
    }

    // Catches: an overlong error body reaching a log line whole.
    #[test]
    fn a_long_error_body_is_truncated() {
        let long = "x".repeat(2000);
        let got = truncate_body(&long);
        assert_eq!(got.chars().count(), 513, "512 plus the ellipsis");
        assert_eq!(truncate_body("  short  "), "short");
    }

    // Catches: reporting a gateway error body as a bare status phrase.
    #[test]
    fn a_json_error_body_becomes_the_message() {
        assert_eq!(
            gateway_error_message(r#"{"error": {"message": "model not found"}}"#).as_deref(),
            Some("model not found")
        );
        assert_eq!(
            gateway_error_message(r#"{"message": "quota exceeded"}"#).as_deref(),
            Some("quota exceeded")
        );
        assert_eq!(gateway_error_message("404 page not found"), None);
    }
}
