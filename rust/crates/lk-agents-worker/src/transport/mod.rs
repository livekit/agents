//! WebSocket and WebTransport connections to the worker endpoint.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::protocol::{ServerMessage, WorkerMessage};

pub(crate) const MAX_CONTROL_MESSAGE: usize = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("{0}")]
    Io(String),
    #[error("connection closed: {0}")]
    Closed(String),
    #[error("invalid control message: {0}")]
    Protocol(String),
}

#[derive(Debug)]
pub enum TransportEvent {
    ServerMessage(ServerMessage),
    EndpointRequest(EndpointStream),
}

pub enum TransportConfig {
    WebSocket { heartbeat: Duration },
    WebTransport { insecure: bool },
}

impl TransportConfig {
    pub fn build(self) -> Box<dyn WorkerTransport> {
        match self {
            Self::WebSocket { heartbeat } => Box::new(WsTransport { heartbeat }),
            Self::WebTransport { insecure } => Box::new(WtTransport { insecure }),
        }
    }
}

pub mod webtransport;
mod ws;

pub use webtransport::{
    AgentEndpointKind, EndpointReader, EndpointStream, EndpointWriter, HttpStreamResetCode,
    StreamPreamble, WtTransport,
};
pub use ws::WsTransport;

#[async_trait]
pub trait WorkerSink: Send + Sync {
    async fn send(&self, message: WorkerMessage) -> Result<(), TransportError>;
    async fn close(&self);
}

#[async_trait]
pub trait WorkerStream: Send {
    /// Closure is an error (the worker must reconnect). Cancellation must preserve unread data.
    async fn recv(&mut self) -> Result<TransportEvent, TransportError>;
}

pub struct Connected {
    pub sink: Arc<dyn WorkerSink>,
    pub stream: Box<dyn WorkerStream>,
}

#[async_trait]
pub trait WorkerTransport: Send + Sync {
    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError>;
}

#[async_trait]
impl WorkerTransport for Box<dyn WorkerTransport> {
    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        (**self).connect(url, worker_token, headers).await
    }
}

/// Scheme normalised, trailing `/` stripped, `/agent` appended, `worker_token` as a query param.
pub fn agent_url(ws_url: &str, worker_token: Option<&str>, webtransport: bool) -> String {
    let (scheme, rest) = match ws_url.split_once("://") {
        // WebTransport is QUIC and QUIC is always TLS: every spelling is `https`.
        Some((_, r)) if webtransport => ("https", r),
        None if webtransport => ("https", ws_url),
        Some(("https", r)) | Some(("wss", r)) => ("wss", r),
        Some(("http", r)) | Some(("ws", r)) => ("ws", r),
        Some((s, r)) => (s, r),
        None => ("ws", ws_url),
    };
    let mut url = format!("{scheme}://{}", rest.trim_end_matches('/'));
    url.push_str("/agent");
    if let Some(tok) = worker_token.filter(|t| !t.is_empty()) {
        url.push('?');
        url.push_str(
            &url::form_urlencoded::Serializer::new(String::new())
                .append_pair("worker_token", tok)
                .finish(),
        );
    }
    url
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a double slash in the `/agent` URL, an http(s) scheme not swapped to ws(s), a lost
    // path prefix (proxied gateways), WebTransport not forced to `https`, or the worker token
    // interpolated without percent-encoding.
    #[test]
    fn the_agent_url() {
        for (base, token, wt, want) in [
            ("wss://gw.livekit.cloud", None, false, "wss://gw.livekit.cloud/agent"),
            ("wss://gw.livekit.cloud///", None, false, "wss://gw.livekit.cloud/agent"),
            ("https://gw.livekit.cloud", None, false, "wss://gw.livekit.cloud/agent"),
            ("http://localhost:7880", None, false, "ws://localhost:7880/agent"),
            ("wss://host/lk/", None, false, "wss://host/lk/agent"),
            ("wss://gw.livekit.cloud", None, true, "https://gw.livekit.cloud/agent"),
            ("http://localhost:7882", None, true, "https://localhost:7882/agent"),
            ("wss://h", Some("a+b/c"), false, "wss://h/agent?worker_token=a%2Bb%2Fc"),
            ("wss://h", Some(""), false, "wss://h/agent"),
        ] {
            assert_eq!(agent_url(base, token, wt), want, "{base} {token:?} {wt}");
        }
    }
}
