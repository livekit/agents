//! An endpoint exchange served with an ASGI application: one stream after its preamble is either
//! one HTTP/1.1 exchange or, after a `101`, a WebSocket. The codecs and the outcome reporting are
//! here; the application is behind [`AsgiApp`] ([`host`] calls the host's).
//!
//! Every outcome is reported the way the data plane expects: a reset code before any response
//! byte, `x-lk-completion`/`x-lk-error` trailers after (chunked responses only), FIN on success.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::LOG;
use crate::transport::HttpStreamResetCode as Reset;
use lk_agents_abi::worker::endpoint::EndpointLimits;

mod admission;
pub mod budget;
pub mod host;

mod request;
use request::{HeadError, parse_head, read_head};

mod http;
mod websocket;
use http::{HttpExchange, serve_http};
use websocket::{WsExchange, serve_ws};

const READ_SIZE: usize = 64 * 1024;
// Graceful close shares the bounded IPC flush budget; frame scheduling is a separate contract.
const CLOSE_FLUSH_TIMEOUT: Duration = crate::ipc::FLUSH_TIMEOUT;

pub use lk_agents_abi::worker::endpoint_host::Headers;

/// One exchange stream: what the worker hands over after the preamble.
#[async_trait]
pub trait StreamIo: Send + Sync {
    /// `b""` once the peer half-closed; `Err` when it reset the stream.
    async fn read(&self, max: usize) -> Result<Vec<u8>, String>;
    async fn write(&self, data: &[u8]) -> Result<(), String>;
    /// FIN.
    async fn finish(&self);
    /// RESET_STREAM and STOP_SENDING with `code`.
    async fn reset(&self, code: Reset);
    /// STOP_SENDING only.
    async fn stop(&self, code: Reset);
}

pub use lk_agents_abi::worker::endpoint::EndpointRequest as RequestInfo;

/// The ASGI scope (without `state`, which the host adds) and the messages of `receive()` and
/// `send()`: declared records, so the host shapes them as ASGI dicts itself.
pub use lk_agents_abi::worker::endpoint_host::{
    AsgiReceived as Received, AsgiScope as Scope, AsgiSent as Sent,
};

/// The application raised: `"{Type}: {message}"` and its traceback.
#[derive(Debug, Clone)]
pub struct AppError {
    pub summary: String,
    pub traceback: String,
}

/// The connection an application call talks over.
#[derive(Clone)]
pub enum Conn {
    Http(Arc<HttpExchange>),
    Ws(Arc<WsExchange>),
}

impl Conn {
    pub async fn receive(&self) -> Received {
        match self {
            Conn::Http(h) => h.receive().await,
            Conn::Ws(w) => w.receive().await,
        }
    }

    /// `Err` is the `RuntimeError` the application sees.
    pub async fn send(&self, msg: Sent) -> Result<(), String> {
        match self {
            Conn::Http(h) => h.send(msg).await,
            Conn::Ws(w) => w.send(msg).await,
        }
    }
}

#[async_trait]
pub trait AsgiApp: Send + Sync {
    /// Runs the application to completion; dropping the future cancels it.
    async fn call(&self, scope: Scope, conn: Conn) -> Result<(), AppError>;
}

pub use admission::Registry;

/// Serve one exchange stream to completion.
pub async fn serve(
    io: Arc<dyn StreamIo>,
    info: RequestInfo,
    app: &dyn AsgiApp,
    registry: &Registry,
) {
    let limits = registry.admission.limits;
    let Some(exchange) = registry.admission.try_exchange(limits.request_head_bytes + READ_SIZE)
    else {
        log::warn!(target: LOG, requestId = info.request_id.as_str();
            "endpoint worker admission exhausted");
        io.reset(Reset::HsrRefused).await;
        return;
    };
    let buffered = match read_head(io.as_ref(), limits.request_head_bytes).await {
        Ok(b) => b,
        Err(HeadError::Protocol(e)) => {
            log::warn!(target: LOG, requestId = info.request_id.as_str(), error = e.as_str();
                "endpoint request head is not HTTP/1.1");
            io.reset(Reset::HsrProtocol).await;
            return;
        }
        // the peer went away before a head arrived; nothing to answer
        Err(HeadError::Gone) => {
            io.reset(Reset::HsrAbort).await;
            return;
        }
    };
    let head = match parse_head(&buffered) {
        Ok(h) => h,
        Err(e) => {
            log::warn!(target: LOG, requestId = info.request_id.as_str(), error = e.as_str();
                "endpoint request is not valid HTTP/1.1");
            io.reset(Reset::HsrProtocol).await;
            return;
        }
    };
    if head.is_websocket_upgrade() {
        serve_ws(io, info, head, app, registry, exchange).await;
    } else {
        serve_http(io, info, head, app, exchange, limits).await;
    }
}

// ----------------------------------------------------------------------------- the head

/// `wire.sanitizeReason`: bounded, no control characters that would break trailer framing.
fn sanitize(reason: &str) -> String {
    reason
        .chars()
        .map(|c| if (c as u32) < 0x20 || c as u32 == 0x7f { ' ' } else { c })
        .take(256)
        .collect()
}

fn valid_header(name: &[u8], value: &[u8]) -> bool {
    !name.is_empty()
        && name.iter().all(|b| b.is_ascii_graphic() && *b != b':')
        && value.iter().all(|b| *b != b'\r' && *b != b'\n' && *b != 0)
}

/// `h11.Response`: an empty reason phrase, the headers as given.
fn response_head(status: u16, headers: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status} \r\n").into_bytes();
    for (k, v) in headers {
        out.extend_from_slice(k);
        out.extend_from_slice(b": ");
        out.extend_from_slice(v);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
    out
}

fn chunk(data: &[u8]) -> Vec<u8> {
    let mut out = format!("{:x}\r\n", data.len()).into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    out
}

/// The ASGI event type of what the app sent.
fn kind_of(msg: &Sent) -> &str {
    match msg {
        Sent::ResponseStart { .. } => "http.response.start",
        Sent::ResponseBody { .. } => "http.response.body",
        Sent::ResponseTrailers => "http.response.trailers",
        Sent::WsAccept { .. } => "websocket.accept",
        Sent::WsText(_) | Sent::WsBytes(_) => "websocket.send",
        Sent::WsClose { .. } => "websocket.close",
        Sent::WsHttpResponseStart { .. } => "websocket.http.response.start",
        Sent::WsHttpResponseBody { .. } => "websocket.http.response.body",
        Sent::Unknown(k) => k,
    }
}

/// A drop that runs `f` on the runtime when the guarded future was cancelled.
struct OnCancel(Option<Box<dyn FnOnce() + Send>>);

impl OnCancel {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for OnCancel {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod profile;
