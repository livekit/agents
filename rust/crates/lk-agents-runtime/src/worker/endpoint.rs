//! An endpoint exchange served with an ASGI application: one stream after its preamble is either
//! one HTTP/1.1 exchange or, after a `101`, a WebSocket. The codecs and the outcome reporting are
//! here; the application is behind [`AsgiApp`] (`python.rs` calls the Python one).
//!
//! Every outcome is reported the way the data plane expects: a reset code before any response
//! byte, `x-lk-completion`/`x-lk-error` trailers after (chunked responses only), FIN on success.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use tokio::sync::{Mutex, mpsc, watch};
use tokio_tungstenite::tungstenite::{
    self, Message,
    protocol::{CloseFrame, Role, WebSocket, WebSocketConfig, frame::coding::CloseCode},
};

use crate::worker::LOG;
use crate::worker::transport::HttpStreamResetCode as Reset;
use lk_agents_abi::worker::endpoint::EndpointLimits;

#[path = "endpoint_admission.rs"]
mod admission;
use admission::{EndpointAdmission, EndpointReservation, MessageAdmission, MessageReservation};

#[path = "endpoint_request.rs"]
mod request;
use request::{Head, HeadError, find, parse_head, read_head, scope_base};

const READ_SIZE: usize = 64 * 1024;
// Graceful close shares the bounded IPC flush budget; frame scheduling is a separate contract.
const CLOSE_FLUSH_TIMEOUT: Duration = crate::worker::ipc::FLUSH_TIMEOUT;

pub type Headers = Vec<(Vec<u8>, Vec<u8>)>;

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

/// The ASGI scope, without `state` (the host adds `request.state.lk_request`).
#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    /// `"http"` or `"websocket"`.
    pub kind: &'static str,
    /// HTTP only.
    pub method: String,
    pub scheme: String,
    pub path: String,
    pub raw_path: Vec<u8>,
    pub query_string: Vec<u8>,
    pub headers: Headers,
    pub client: Option<(String, u16)>,
    /// WebSocket only.
    pub subprotocols: Vec<String>,
}

/// What `receive()` returns.
#[derive(Debug, Clone, PartialEq)]
pub enum Received {
    Request { body: Vec<u8>, more_body: bool },
    Disconnect,
    WsConnect,
    WsText(String),
    WsBytes(Vec<u8>),
    WsDisconnect { code: u16 },
}

/// What `send()` takes.
#[derive(Debug, Clone, PartialEq)]
pub enum Sent {
    ResponseStart {
        status: u16,
        headers: Headers,
    },
    ResponseBody {
        body: Vec<u8>,
        more_body: bool,
    },
    ResponseTrailers,
    WsAccept {
        subprotocol: Option<String>,
        headers: Headers,
    },
    WsText(String),
    WsBytes(Vec<u8>),
    WsClose {
        code: u16,
        reason: String,
    },
    WsHttpResponseStart {
        status: u16,
        headers: Headers,
    },
    WsHttpResponseBody {
        body: Vec<u8>,
        more_body: bool,
    },
    /// Any other `type`: a `RuntimeError` for the application.
    Unknown(String),
}

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

// ----------------------------------------------------------------------------- HTTP

enum Body {
    Length(u64),
    Chunked(ChunkState),
    /// HTTP/1.0 without a length: none.
    Done,
}

#[derive(Clone, Copy)]
enum ChunkState {
    Size,
    Data(u64),
    DataEnd,
    Trailers,
}

struct RequestSide {
    buf: Vec<u8>,
    body: Body,
    eof: bool,
    done: bool,
}

#[derive(Default)]
struct ResponseSide {
    started: bool,
    ended: bool,
    chunked: bool,
    /// Declared `content-length` left to send.
    remaining: Option<u64>,
}

/// One HTTP exchange: request bytes in, ASGI `http` events to the app, response bytes out.
pub struct HttpExchange {
    _payload: EndpointReservation,
    limits: EndpointLimits,
    io: Arc<dyn StreamIo>,
    is_head: bool,
    http10: bool,
    req: Mutex<RequestSide>,
    resp: Mutex<ResponseSide>,
    head_sent: watch::Sender<bool>,
    done: watch::Sender<bool>,
    peer_gone: AtomicBool,
}

impl HttpExchange {
    fn new(
        io: Arc<dyn StreamIo>,
        head: &Head,
        payload: EndpointReservation,
        limits: EndpointLimits,
    ) -> Result<Self, String> {
        let te = head.tokens(b"transfer-encoding");
        let body = if !te.is_empty() {
            if te.last().map(Vec::as_slice) != Some(b"chunked") {
                return Err("Transfer-Encoding must end with chunked".into());
            }
            Body::Chunked(ChunkState::Size)
        } else {
            let lengths: Vec<&[u8]> = head.values(b"content-length").collect();
            match lengths.first() {
                None => Body::Done,
                Some(first) => {
                    if lengths.iter().any(|l| l != first) {
                        return Err("conflicting Content-Length headers".into());
                    }
                    let s = std::str::from_utf8(first).map_err(|e| e.to_string())?.trim();
                    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                        return Err("bad Content-Length".into());
                    }
                    Body::Length(s.parse().map_err(|_| "bad Content-Length".to_owned())?)
                }
            }
        };
        let body = match body {
            Body::Length(0) => Body::Done,
            b => b,
        };
        Ok(HttpExchange {
            _payload: payload,
            limits,
            io,
            is_head: head.method == "HEAD",
            http10: head.http10,
            req: Mutex::new(RequestSide { buf: head.rest.clone(), body, eof: false, done: false }),
            resp: Mutex::new(ResponseSide::default()),
            head_sent: watch::channel(false).0,
            done: watch::channel(false).0,
            peer_gone: AtomicBool::new(false),
        })
    }

    pub async fn receive(&self) -> Received {
        let mut r = self.req.lock().await;
        if r.done {
            drop(r);
            // The node half-closes once the request is written; that is the end of the body, not
            // a disconnect. A client that goes away shows as a failed write, so an app listening
            // for a disconnect waits here until the exchange ends.
            let mut done = self.done.subscribe();
            let _ = done.wait_for(|d| *d).await;
            return Received::Disconnect;
        }
        match self.next_body(&mut r).await {
            Ok(Some(body)) => Received::Request { body, more_body: true },
            Ok(None) => {
                r.done = true;
                Received::Request { body: Vec::new(), more_body: false }
            }
            Err(()) => {
                r.done = true;
                self.done.send_replace(true);
                self.io.reset(Reset::HsrProtocol).await;
                Received::Disconnect
            }
        }
    }

    /// The next piece of the body, `None` at its end, `Err` when the stream ended early or broke.
    async fn next_body(&self, r: &mut RequestSide) -> Result<Option<Vec<u8>>, ()> {
        loop {
            match &mut r.body {
                Body::Done => return Ok(None),
                Body::Length(left) => {
                    if !r.buf.is_empty() {
                        let n = (*left).min(r.buf.len() as u64) as usize;
                        let data: Vec<u8> = r.buf.drain(..n).collect();
                        *left -= n as u64;
                        if *left == 0 {
                            r.body = Body::Done;
                        }
                        return Ok(Some(data));
                    }
                }
                Body::Chunked(state) => match *state {
                    ChunkState::Size => {
                        let end = find(&r.buf, b"\r\n");
                        if end.map_or(r.buf.len() >= self.limits.chunk_line_bytes, |i| {
                            i + 2 > self.limits.chunk_line_bytes
                        }) {
                            return Err(());
                        }
                        if let Some(i) = end {
                            let line: Vec<u8> = r.buf.drain(..i + 2).collect();
                            let size = line[..i].split(|b| *b == b';').next().unwrap_or_default();
                            let size = std::str::from_utf8(size).map_err(|_| ())?.trim();
                            let size = u64::from_str_radix(size, 16).map_err(|_| ())?;
                            *state = if size == 0 {
                                ChunkState::Trailers
                            } else {
                                ChunkState::Data(size)
                            };
                            continue;
                        }
                    }
                    ChunkState::Data(left) => {
                        if !r.buf.is_empty() {
                            let n = left.min(r.buf.len() as u64) as usize;
                            let data: Vec<u8> = r.buf.drain(..n).collect();
                            *state = if left == n as u64 {
                                ChunkState::DataEnd
                            } else {
                                ChunkState::Data(left - n as u64)
                            };
                            return Ok(Some(data));
                        }
                    }
                    ChunkState::DataEnd => {
                        if r.buf.len() >= 2 {
                            if &r.buf[..2] != b"\r\n" {
                                return Err(());
                            }
                            r.buf.drain(..2);
                            *state = ChunkState::Size;
                            continue;
                        }
                    }
                    ChunkState::Trailers => {
                        if r.buf.starts_with(b"\r\n") {
                            r.buf.drain(..2);
                            r.body = Body::Done;
                            continue;
                        }
                        let end = find(&r.buf, b"\r\n\r\n");
                        if end.map_or(r.buf.len() >= self.limits.trailers_bytes, |i| {
                            i + 4 > self.limits.trailers_bytes
                        }) {
                            return Err(());
                        }
                        if let Some(i) = end {
                            r.buf.drain(..i + 4);
                            r.body = Body::Done;
                            continue;
                        }
                    }
                },
            }
            if r.eof {
                return Err(());
            }
            let chunk = self.io.read(READ_SIZE).await.map_err(|_| ())?;
            if chunk.is_empty() {
                r.eof = true;
            }
            r.buf.extend_from_slice(&chunk);
        }
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        self.io.write(data).await.inspect_err(|_| {
            // the peer reset the stream or stopped reading: the client is gone
            self.peer_gone.store(true, Ordering::Release);
        })
    }

    pub async fn send(&self, msg: Sent) -> Result<(), String> {
        if *self.done.borrow() {
            return Err("endpoint exchange is closed".into());
        }
        match msg {
            Sent::ResponseStart { status, mut headers } => {
                let mut resp = self.resp.lock().await;
                if resp.started {
                    return Err("response already started".into());
                }
                if let Some((k, v)) = headers.iter().find(|(k, v)| !valid_header(k, v)) {
                    return Err(format!(
                        "illegal header: {:?}: {:?}",
                        String::from_utf8_lossy(k),
                        String::from_utf8_lossy(v)
                    ));
                }
                let has = |name: &[u8]| headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
                let length = headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(b"content-length"))
                    .and_then(|(_, v)| std::str::from_utf8(v).ok()?.trim().parse::<u64>().ok());
                let bodiless = matches!(status, 204 | 304) || self.is_head;
                // h11 frames an HTTP/1.1 response without a length as chunked, the one framing
                // that can carry a completion trailer, so a dying stream reads as truncation
                let unframed = !has(b"content-length") && !has(b"transfer-encoding");
                if unframed && !matches!(status, 204 | 304) {
                    if self.http10 {
                        headers.push((b"connection".to_vec(), b"close".to_vec()));
                    } else {
                        headers.push((b"transfer-encoding".to_vec(), b"chunked".to_vec()));
                    }
                }
                resp.chunked = unframed && !self.http10 && !bodiless;
                resp.remaining = if bodiless { None } else { length };
                resp.started = true;
                self.head_sent.send_replace(true);
                self.write(&response_head(status, &headers)).await
            }
            Sent::ResponseBody { body, more_body } => {
                let mut resp = self.resp.lock().await;
                if !resp.started {
                    return Err("response body before its head".into());
                }
                if resp.ended {
                    return Ok(());
                }
                if !body.is_empty() && !self.is_head {
                    if let Some(left) = resp.remaining {
                        if body.len() as u64 > left {
                            return Err("Too much data for declared Content-Length".into());
                        }
                        resp.remaining = Some(left - body.len() as u64);
                    }
                    let data = if resp.chunked { chunk(&body) } else { body };
                    self.write(&data).await?;
                }
                if !more_body {
                    if resp.remaining.is_some_and(|l| l > 0) {
                        return Err("Too little data for declared Content-Length".into());
                    }
                    if resp.chunked {
                        self.write(b"0\r\n\r\n").await?;
                    }
                    resp.ended = true;
                    self.io.finish().await;
                    // nothing more of the request is wanted
                    self.io.stop(Reset::HsrAbort).await;
                    self.done.send_replace(true);
                }
                Ok(())
            }
            // the x-lk- namespace is ours; applications do not send trailers here
            Sent::ResponseTrailers => Ok(()),
            other => Err(format!("unexpected ASGI message {:?} on an http scope", kind_of(&other))),
        }
    }

    /// Report an outcome other than success in the only sound place for it.
    async fn fail(&self, completion: &str, reason: &str, code: Reset) {
        if *self.done.borrow() {
            return;
        }
        let mut resp = self.resp.lock().await;
        if resp.ended {
            return;
        }
        if !resp.started {
            self.io.reset(code).await;
            return;
        }
        if resp.chunked {
            // in band, where it cannot race the body bytes
            let trailers = format!(
                "0\r\nx-lk-completion: {completion}\r\nx-lk-error: {}\r\n\r\n",
                sanitize(reason)
            );
            if self.write(trailers.as_bytes()).await.is_ok() {
                self.io.finish().await;
            }
            resp.ended = true;
            return;
        }
        // a declared length cannot carry trailers: the short body is the signal
        self.io.reset(Reset::HsrAbort).await;
    }
}

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

async fn serve_http(
    io: Arc<dyn StreamIo>,
    info: RequestInfo,
    head: Head,
    app: &dyn AsgiApp,
    payload: EndpointReservation,
    limits: EndpointLimits,
) {
    let ex = match HttpExchange::new(io.clone(), &head, payload, limits) {
        Ok(ex) => Arc::new(ex),
        Err(e) => {
            log::warn!(target: LOG, requestId = info.request_id.as_str(), error = e.as_str();
                "endpoint request is not valid HTTP/1.1");
            io.reset(Reset::HsrProtocol).await;
            return;
        }
    };
    let mut scope = scope_base(&info, "http", head.headers.clone(), &head.target);
    scope.method = head.method.clone();
    scope.scheme = if info.scheme.is_empty() { "http".into() } else { info.scheme.clone() };

    // the worker is stopping or reconnecting: the stream goes with it
    let guard = {
        let ex = ex.clone();
        OnCancel(Some(Box::new(move || {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(
                    async move { ex.fail("canceled", "worker stopped", Reset::HsrAbort).await },
                );
            }
        })))
    };
    let mut call = app.call(scope, Conn::Http(ex.clone()));
    let result = match (info.timeout_ms > 0).then(|| Duration::from_millis(info.timeout_ms)) {
        None => Some(call.await),
        // the deadline bounds the wait for a response head only; bodies are unbounded
        Some(timeout) => {
            let mut head_sent = ex.head_sent.subscribe();
            let first = tokio::select! {
                r = &mut call => Some(Some(r)),
                _ = head_sent.wait_for(|s| *s) => Some(None),
                () = tokio::time::sleep(timeout) => None,
            };
            match first {
                Some(Some(r)) => Some(r),
                Some(None) => Some(call.await),
                None => {
                    drop(call);
                    None
                }
            }
        }
    };
    guard.disarm();
    let rid = info.request_id.as_str();
    let route = info.route.as_str();
    match result {
        None => {
            log::warn!(target: LOG, requestId = rid, route = route;
                "endpoint request timed out before a response head");
            ex.fail("timeout", "no response head before the deadline", Reset::HsrTimeout).await;
        }
        Some(Err(_)) if ex.peer_gone.load(Ordering::Acquire) => {
            log::debug!(target: LOG, requestId = rid, route = route; "endpoint client went away");
        }
        Some(Err(e)) => {
            log::error!(target: LOG, requestId = rid, route = route;
                "endpoint request failed\n{}{}", e.traceback, e.summary);
            ex.fail("internal", &e.summary, Reset::HsrInternal).await;
        }
        Some(Ok(())) => {
            let (started, ended) = {
                let r = ex.resp.lock().await;
                (r.started, r.ended)
            };
            if !started {
                log::error!(target: LOG, requestId = rid, route = route;
                    "ASGI app returned without a response");
                ex.fail("internal", "no response", Reset::HsrInternal).await;
            } else if !ended {
                ex.fail("internal", "response body left open", Reset::HsrInternal).await;
            }
        }
    }
    ex.done.send_replace(true);
}

// ----------------------------------------------------------------------------- WebSocket

/// tungstenite over memory: frames are fed in and taken out; the stream I/O is ours.
#[derive(Default)]
struct Pipe {
    inbound: VecDeque<u8>,
    outbound: Vec<u8>,
}

impl Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.inbound.is_empty() {
            return Err(std::io::ErrorKind::WouldBlock.into());
        }
        let n = buf.len().min(self.inbound.len());
        for (dst, src) in buf.iter_mut().zip(self.inbound.drain(..n)) {
            *dst = src;
        }
        Ok(n)
    }
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.outbound.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// After the upgrade request the stream is a byte pipe: frames in, ASGI `websocket` events to the
/// app, frames out. The `101` is written here.
pub struct WsExchange {
    _payload: EndpointReservation,
    _parser: EndpointReservation,
    io: Arc<dyn StreamIo>,
    key: Vec<u8>,
    subprotocols: Vec<String>,
    ws: std::sync::Mutex<WebSocket<Pipe>>,

    /// Holds the order of writes: bytes are taken from the pipe and written under it.
    write_lock: Mutex<()>,
    events: mpsc::Sender<InboxEvent>,
    inbox: Mutex<mpsc::Receiver<InboxEvent>>,
    admission: EndpointAdmission,
    inbox_admission: MessageAdmission,
    write_admission: MessageAdmission,
    disconnect: watch::Sender<Option<u16>>,
    accept_ready: watch::Sender<bool>,
    accepted: AtomicBool,
    closed: AtomicBool,
}

struct InboxEvent {
    event: Received,
    _payload: Option<MessageReservation>,
}

impl WsExchange {
    fn ws(&self) -> std::sync::MutexGuard<'_, WebSocket<Pipe>> {
        self.ws.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub async fn receive(&self) -> Received {
        let mut inbox = self.inbox.lock().await;
        if let Ok(event) = inbox.try_recv() {
            return event.event;
        }
        let mut disconnect = self.disconnect.subscribe();
        if let Some(code) = *disconnect.borrow() {
            return Received::WsDisconnect { code };
        }
        let next = {
            let event = Box::pin(inbox.recv());
            let closed = Box::pin(disconnect.changed());
            match futures_util::future::select(event, closed).await {
                futures_util::future::Either::Left((event, closed)) => {
                    drop(closed);
                    event
                }
                futures_util::future::Either::Right((_, event)) => {
                    drop(event);
                    None
                }
            }
        };
        next.map(|event| event.event)
            .unwrap_or(Received::WsDisconnect { code: (*disconnect.borrow()).unwrap_or(1006) })
    }

    async fn write_raw(&self, data: &[u8]) -> Result<(), String> {
        let _w = self.write_lock.lock().await;
        if data.is_empty() { Ok(()) } else { self.io.write(data).await }
    }

    /// Whatever tungstenite queued (a frame, a pong, a close reply), in order.
    async fn flush(&self) -> Result<(), String> {
        let _w = self.write_lock.lock().await;
        let out = {
            let mut ws = self.ws();
            let _ = ws.flush();
            std::mem::take(&mut ws.get_mut().outbound)
        };
        if out.is_empty() { Ok(()) } else { self.io.write(&out).await }
    }

    async fn send_frame(&self, msg: Message) -> Result<(), String> {
        if msg.len() > self.admission.limits.websocket_message_bytes {
            return Err("websocket response exceeds message limit".into());
        }
        let _payload = self
            .write_admission
            .try_message(&self.admission, msg.len().max(READ_SIZE))
            .ok_or("websocket response capacity exhausted")?;
        {
            let _w = self.write_lock.lock().await;
            let out = {
                let mut ws = self.ws();
                ws.send(msg).map_err(|e| e.to_string())?;
                std::mem::take(&mut ws.get_mut().outbound)
            };
            if !out.is_empty() {
                self.io.write(&out).await?;
            }
        }
        Ok(())
    }

    fn enqueue(&self, message: Message) -> Result<(), &'static str> {
        let bytes = message.len();
        let payload = self
            .inbox_admission
            .try_message(&self.admission, bytes)
            .ok_or("websocket inbox byte limit exceeded")?;
        let event = match message {
            Message::Text(text) => Received::WsText(text.to_string()),
            Message::Binary(bytes) => Received::WsBytes(bytes.to_vec()),
            _ => unreachable!("only data messages enter the inbox"),
        };
        self.events
            .try_send(InboxEvent { event, _payload: Some(payload) })
            .map_err(|_| "websocket inbox item limit exceeded")
    }

    /// Frames from the peer, once accepted. `false` when the peer closed or broke the protocol.
    async fn process(&self) -> bool {
        if !self.accepted.load(Ordering::Acquire) {
            return true;
        }
        let mut parsed = 0;
        loop {
            parsed += 1;
            if parsed == 16 {
                tokio::task::yield_now().await;
                parsed = 0;
            }
            let next = self.ws().read();
            match next {
                Ok(message @ (Message::Text(_) | Message::Binary(_))) => {
                    if let Err(reason) = self.enqueue(message) {
                        self.close(1009, reason).await;
                        return false;
                    }
                }
                // the reply is queued; `flush` below sends it
                Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => {}
                Ok(Message::Close(frame)) => {
                    let _ = self.flush().await;
                    if !self.closed.swap(true, Ordering::AcqRel) {
                        let code = frame.map_or(1005, |f| u16::from(f.code));
                        self.disconnect.send_replace(Some(code));
                    }
                    self.io.finish().await;
                    return false;
                }
                Err(tungstenite::Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    let _ = self.flush().await;
                    return true;
                }
                Err(tungstenite::Error::Capacity(_)) => {
                    self.close(1009, "websocket frame or message limit exceeded").await;
                    return false;
                }
                Err(_) => {
                    self.close(1002, "invalid websocket frame").await;
                    return false;
                }
            }
        }
    }

    async fn read_loop(self: Arc<Self>) {
        let mut accepted = self.accept_ready.subscribe();
        let _ = accepted.wait_for(|ready| *ready).await;
        while !self.closed.load(Ordering::Acquire) {
            let Ok(chunk) = self.io.read(READ_SIZE).await else { break };
            if chunk.is_empty() {
                break;
            }
            self.ws().get_mut().inbound.extend(chunk);
            if !self.process().await {
                break;
            }
        }
        // the peer is gone (FIN, reset or a protocol error): tell the app once
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.disconnect.send_replace(Some(1006));
            self.io.finish().await;
        }
    }

    pub async fn send(&self, msg: Sent) -> Result<(), String> {
        match msg {
            Sent::WsAccept { subprotocol, headers } => {
                if self.accepted.swap(true, Ordering::AcqRel) {
                    return Err("websocket already accepted".into());
                }
                let accept = tungstenite::handshake::derive_accept_key(&self.key);
                let mut out = vec![
                    (b"Upgrade".to_vec(), b"WebSocket".to_vec()),
                    (b"Connection".to_vec(), b"Upgrade".to_vec()),
                    (b"Sec-WebSocket-Accept".to_vec(), accept.into_bytes()),
                ];
                if let Some(p) = subprotocol {
                    if !self.subprotocols.contains(&p) {
                        return Err(format!("unexpected subprotocol {p}"));
                    }
                    out.push((b"Sec-WebSocket-Protocol".to_vec(), p.into_bytes()));
                }
                out.extend(headers);
                self.write_raw(&response_head(101, &out)).await?;
                // frames the peer sent ahead of the 101
                if !self.process().await && !self.closed.load(Ordering::Acquire) {
                    self.close(1002, "").await;
                }
                self.accept_ready.send_replace(true);
                Ok(())
            }
            Sent::WsText(_) | Sent::WsBytes(_) => {
                if !self.accepted.load(Ordering::Acquire) {
                    return Err("websocket.send before accept".into());
                }
                if self.closed.load(Ordering::Acquire) {
                    return Ok(());
                }
                self.send_frame(match msg {
                    Sent::WsText(t) => Message::text(t),
                    Sent::WsBytes(b) => Message::binary(b),
                    _ => unreachable!(),
                })
                .await
            }
            Sent::WsClose { code, reason } => {
                if self.accepted.load(Ordering::Acquire) {
                    self.close(code, &reason).await;
                } else {
                    self.reject(403, Vec::new()).await;
                }
                Ok(())
            }
            // a rejection with a body, as the ASGI extension spells it
            Sent::WsHttpResponseStart { status, headers } => {
                self.reject(status, headers).await;
                Ok(())
            }
            Sent::WsHttpResponseBody { body, more_body } => {
                if !body.is_empty() {
                    self.write_raw(&body).await?;
                }
                if !more_body {
                    self.io.finish().await;
                }
                Ok(())
            }
            other => {
                Err(format!("unexpected ASGI message {:?} on a websocket scope", kind_of(&other)))
            }
        }
    }

    /// A declined upgrade is an ordinary HTTP response, then FIN.
    async fn reject(&self, status: u16, mut headers: Headers) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        headers.push((b"content-length".to_vec(), b"0".to_vec()));
        if self.write_raw(&response_head(status, &headers)).await.is_ok() {
            self.io.finish().await;
        }
    }

    /// A close frame to the client, then FIN. Idempotent.
    pub async fn close(&self, code: u16, reason: &str) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let frame = CloseFrame { code: CloseCode::from(code), reason: reason.to_owned().into() };
        let _ = self.ws().close(Some(frame));
        if tokio::time::timeout(CLOSE_FLUSH_TIMEOUT, self.flush()).await.is_err() {
            self.io.reset(Reset::HsrAbort).await;
        } else {
            self.io.finish().await;
        }
        self.disconnect.send_replace(Some(code));
    }
}

async fn serve_ws(
    io: Arc<dyn StreamIo>,
    info: RequestInfo,
    mut head: Head,
    app: &dyn AsgiApp,
    registry: &Registry,
    mut payload: EndpointReservation,
) {
    // wsproto's checks on the upgrade request
    let version = head.values(b"sec-websocket-version").next();
    let key = head.values(b"sec-websocket-key").next().map(<[u8]>::to_vec);
    let host = head.values(b"host").next().map(<[u8]>::to_vec);
    let (Some(key), Some(host), Some(b"13")) = (key, host, version) else {
        log::warn!(target: LOG, requestId = info.request_id.as_str();
            "endpoint websocket upgrade is incomplete");
        io.reset(Reset::HsrProtocol).await;
        return;
    };
    let skipped: [&[u8]; 3] = [b"host", b"sec-websocket-extensions", b"sec-websocket-protocol"];
    let mut headers: Headers =
        head.headers.iter().filter(|(k, _)| !skipped.contains(&k.as_slice())).cloned().collect();
    headers.extend([
        (b"host".to_vec(), host),
        (b"upgrade".to_vec(), b"websocket".to_vec()),
        (b"connection".to_vec(), b"upgrade".to_vec()),
    ]);
    let subprotocols: Vec<String> = head
        .values(b"sec-websocket-protocol")
        .flat_map(|v| v.split(|b| *b == b','))
        .map(|t| String::from_utf8_lossy(t.trim_ascii()).into_owned())
        .filter(|t| !t.is_empty())
        .collect();
    let mut scope = scope_base(&info, "websocket", headers, &head.target);
    scope.scheme = if info.scheme == "https" { "wss" } else { "ws" }.into();
    scope.subprotocols = subprotocols.clone();

    let limits = registry.admission.limits;
    let Some(parser) = registry.admission.try_payload(limits.websocket_message_bytes + READ_SIZE)
    else {
        io.reset(Reset::HsrRefused).await;
        return;
    };
    let config = WebSocketConfig::default()
        .max_message_size(Some(limits.websocket_message_bytes))
        .max_frame_size(Some(limits.websocket_frame_bytes));
    let mut pipe = Pipe::default();
    pipe.inbound.extend(std::mem::take(&mut head.rest));
    drop(head);
    payload.release_payload();
    let (events, inbox) = mpsc::channel(limits.websocket_inbox);
    let _ = events.try_send(InboxEvent { event: Received::WsConnect, _payload: None });
    let ex = Arc::new(WsExchange {
        _payload: payload,
        _parser: parser,
        io,
        key,
        subprotocols,
        ws: std::sync::Mutex::new(WebSocket::from_raw_socket(pipe, Role::Server, Some(config))),
        write_lock: Mutex::new(()),
        events,
        inbox: Mutex::new(inbox),
        admission: registry.admission.clone(),
        inbox_admission: MessageAdmission::new(limits.websocket_inbox_bytes),
        write_admission: MessageAdmission::new(limits.websocket_outbox_bytes),
        disconnect: watch::channel(None).0,
        accept_ready: watch::channel(false).0,
        accepted: AtomicBool::new(false),
        closed: AtomicBool::new(false),
    });
    let id = registry.next.fetch_add(1, Ordering::Relaxed);
    registry.lock().insert(id, Arc::downgrade(&ex));
    let reader = tokio::spawn(ex.clone().read_loop());

    let guard = {
        let (ex, reader) = (ex.clone(), reader.abort_handle());
        OnCancel(Some(Box::new(move || {
            reader.abort();
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move { ex.close(1001, "worker stopped").await });
            }
        })))
    };
    let result = app.call(scope, Conn::Ws(ex.clone())).await;
    guard.disarm();
    let undecided = || !ex.accepted.load(Ordering::Acquire) && !ex.closed.load(Ordering::Acquire);
    match result {
        // the app returned without deciding: ASGI servers answer 403
        Ok(()) if undecided() => ex.reject(403, Vec::new()).await,
        Ok(()) => {}
        Err(e) => {
            log::error!(target: LOG, requestId = info.request_id.as_str(), route = info.route.as_str();
                "websocket endpoint failed\n{}{}", e.traceback, e.summary);
            if undecided() {
                ex.reject(500, Vec::new()).await;
            } else {
                ex.close(1011, &e.summary).await;
            }
        }
    }
    registry.lock().remove(&id);
    reader.abort();
    let _ = reader.await;
    ex.close(1000, "").await;
}

#[cfg(test)]
#[path = "endpoint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "endpoint_profile.rs"]
mod profile;
