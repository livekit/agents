//! The socket seam. [`WsTransport::connect`] returns a `Sync` sender and a receiver so neither half
//! blocks the other. No timeouts here: the client bounds every wait on the session clock.

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError, TimeoutPhase};
use lk_agents_abi::{ApiConnectOptions, Ts};
use std::sync::Arc;

/// Make `ring` the process TLS provider, once: `reqwest` uses `rustls-no-provider` and panics without one.
pub fn install_tls_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _already_installed = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Retry policy depends on the call site; see [`connect_error`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransportError {
    #[error("timed out")]
    Timeout,
    #[error("connection failed: {0}")]
    Connect(String),
    /// Including a rejected WebSocket handshake (`aiohttp.ClientResponseError`).
    #[error("http {status}: {message}")]
    Status {
        status: i64,
        message: String,
        /// Response headers, lower-cased; a gateway 429 stamps the quota telemetry here.
        headers: Vec<(String, String)>,
        body: Option<String>,
    },
    #[error("connection closed")]
    Closed,
    #[error("{0}")]
    Io(String),
}

/// The `ws_connect` error mapping STT and TTS share: HTTP status via [`ApiError::from_http`],
/// timeout as `APITimeoutError("... timed out.")`, anything else as `APIConnectionError`.
pub fn connect_error(what: &str, err: TransportError) -> ApiError {
    match err {
        TransportError::Status { status, message, .. } => {
            ApiError::from_http(&message, status, None)
        }
        TransportError::Timeout => ApiError {
            phase: Some(TimeoutPhase::Connect),
            ..ApiError::timeout(Some(format!("{what} connection timed out.")))
        },
        other => ApiError::connection(Some(format!("failed to connect to {what}: {other}"))),
    }
}

/// A transport wait bounded on the session clock; the lapse is [`TransportError::Timeout`].
pub async fn within<T>(
    clock: &dyn lk_agents_abi::Clock,
    deadline: Option<Ts>,
    fut: impl Future<Output = Result<T, TransportError>>,
) -> Result<T, TransportError> {
    lk_agents_abi::before(clock, deadline, fut).await.unwrap_or(Err(TransportError::Timeout))
}

/// STT's and TTS's opening: connect within `conn_options.connect_timeout` (a stalled handshake is
/// a retryable connect-phase `APITimeoutError`), then send the session-create frame. Failures map
/// through [`connect_error`].
pub async fn open_ws(
    ws: &dyn WsTransport,
    url: &str,
    headers: Vec<(String, String)>,
    clock: &dyn lk_agents_abi::Clock,
    conn_options: &ApiConnectOptions,
    what: &str,
    create_frame: &serde_json::Value,
) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), ApiError> {
    let deadline = conn_options.connect_deadline(clock.now());
    let (sender, receiver) = within(clock, deadline, ws.connect(url, headers))
        .await
        .map_err(|e| connect_error(what, e))?;
    if let Err(err) = sender.send_text(create_frame.to_string()).await {
        sender.close().await;
        return Err(connect_error(what, err));
    }
    Ok((sender, receiver))
}

/// A send on a stream whose session already ended.
pub fn stream_closed(what: &str) -> ProviderError {
    ProviderError::Local(ApiError::connection(Some(format!("{what} stream is closed"))))
}

/// One HTTP exchange as the runtime sends it.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    Text(String),
    Binary(Vec<u8>),
    /// `WSMsgType.{CLOSED, CLOSE, CLOSING}` collapsed into one, as the reference's tuple test does.
    Closed,
    /// Ping, pong — skipped.
    Other,
}

#[async_trait]
pub trait WsSender: Send + Sync {
    async fn send_text(&self, text: String) -> Result<(), TransportError>;
    /// A binary frame (`ws.send_bytes`): the interruption wire's audio.
    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError>;
    fn is_closed(&self) -> bool;
    async fn close(&self);
}

#[async_trait]
pub trait WsReceiver: Send {
    /// Unbounded; the caller bounds it where the protocol owes a message. Cancel-safe: a caller
    /// may race it and call again.
    async fn recv(&mut self) -> Result<WsMessage, TransportError>;

    /// `ws.close_code` once [`WsMessage::Closed`] was read: the peer's close frame's code, `None`
    /// when it sent none.
    fn close_code(&self) -> Option<u16> {
        None
    }
}

#[async_trait]
pub trait WsTransport: Send + Sync {
    /// Unbounded here.
    async fn connect(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError>;
}

/// `ws.close_code or -1`: an unexpected close's status, `-1` when the peer sent no code or `0`.
pub fn close_status(code: Option<u16>) -> i64 {
    code.filter(|&c| c != 0).map_or(-1, i64::from)
}

/// `base_url.replace("http", "ws", 1)` when it starts with `http(s)://`, as a prefix swap.
pub fn to_ws_url(base_url: &str) -> String {
    if let Some(rest) = base_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base_url.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::ApiErrorKind;

    // Catches: a global `replace("http", "ws")` corrupting a later `http` in the URL.
    #[test]
    fn only_the_scheme_becomes_a_websocket_scheme() {
        assert_eq!(to_ws_url("https://gw.livekit.cloud/v1"), "wss://gw.livekit.cloud/v1");
        assert_eq!(to_ws_url("http://localhost:8080/v1"), "ws://localhost:8080/v1");
        assert_eq!(to_ws_url("https://gw/v1?u=http://x"), "wss://gw/v1?u=http://x");
        // already a ws URL, or something unrecognised: untouched, as the guard implies
        assert_eq!(to_ws_url("wss://gw/v1"), "wss://gw/v1");
    }

    // Catches: `close_code == 0` passed through as a status (R2b).
    #[test]
    fn a_zero_or_absent_close_code_becomes_minus_one() {
        assert_eq!(close_status(Some(0)), -1);
        assert_eq!(close_status(None), -1);
        assert_eq!(close_status(Some(1006)), 1006);
    }

    // Catches: collapsing the three connect failures into one APIConnectionError.
    #[test]
    fn the_three_connect_failures_map_to_three_taxonomy_classes() {
        let status = TransportError::Status {
            status: 401,
            message: "Unauthorized".into(),
            headers: vec![],
            body: None,
        };
        let err = connect_error("LiveKit Inference STT", status);
        assert_eq!(err.kind, ApiErrorKind::Status);
        assert!(!err.retryable, "a 401 must not be retried");

        let err = connect_error("LiveKit Inference STT", TransportError::Timeout);
        assert_eq!(err.kind, ApiErrorKind::Timeout);
        assert_eq!(err.phase, Some(TimeoutPhase::Connect));
        assert!(err.retryable);

        let err = connect_error("LiveKit Inference STT", TransportError::Connect("dns".into()));
        assert_eq!(err.kind, ApiErrorKind::Connection);
        assert!(err.retryable);
    }
}
