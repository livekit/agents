//! Endpoint streams: the worker serves one HTTP/1.1 exchange, or one WebSocket connection, on each
//! stream and hands the binding the parsed exchange. Parsing, framing, limits, deadlines, completion
//! trailers and closing are decided here; the binding only adapts the exchange to its framework.

mod exchange;
mod websocket;

#[cfg(test)]
mod exchange_tests;
#[cfg(test)]
mod hostile_input_tests;

use std::collections::HashMap;
use std::io::Cursor;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::{StreamPreamble, WorkerHost};
use hyper::body::Bytes;
use lk_agents_abi::worker::http::{
    EndpointHandler, EndpointKind, EndpointRequest as RequestMetadata, HttpScheme,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{Mutex, watch};
use tokio::time::Sleep;

use crate::transport::{
    AgentEndpointKind, EndpointControl, EndpointReader, EndpointStream as TransportStream,
    EndpointWriter, HttpStreamResetCode, TransportError,
};

pub use exchange::Exchange;
pub(crate) use exchange::{Origin, decode_path};

/// The largest single read from an endpoint stream.
pub(crate) const READ_SIZE: usize = 64 << 10;

/// Resource limits of the endpoint server.
#[derive(Debug, Clone)]
pub struct HttpServerLimits {
    pub request_head_bytes: usize,
    /// One chunk-size line, or all of a chunked body's trailers.
    pub body_metadata_bytes: usize,
    pub head_timeout: Duration,
    /// How long a request body read or a response write may make no progress: a peer that
    /// stalls its body or stopped reading.
    pub idle_timeout: Duration,
    /// How long a closing WebSocket waits to flush its close frame and for the peer's reply.
    pub close_flush_timeout: Duration,
    pub websocket_frame_bytes: usize,
    pub websocket_message_bytes: usize,
    pub websocket_inbox: usize,
    pub websocket_inbox_bytes: usize,
    /// A WebSocket idle this long is pinged; one that answers nothing for another
    /// `websocket_ping_timeout` is gone.
    pub websocket_ping_interval: Duration,
    pub websocket_ping_timeout: Duration,
    /// How long `stop` waits for cancelled exchanges to report their end.
    pub stop_grace: Duration,
    /// Connections served at once, HTTP and WebSocket alike; the node opens at most 256
    /// streams per worker by default.
    pub max_connections: usize,
}

impl Default for HttpServerLimits {
    fn default() -> Self {
        Self {
            request_head_bytes: 1 << 20,
            body_metadata_bytes: 64 << 10,
            head_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(30),
            close_flush_timeout: Duration::from_secs(1),
            websocket_frame_bytes: 1 << 20,
            websocket_message_bytes: 4 << 20,
            websocket_inbox: 16,
            websocket_inbox_bytes: 8 << 20,
            websocket_ping_interval: Duration::from_secs(20),
            websocket_ping_timeout: Duration::from_secs(20),
            stop_grace: Duration::from_secs(2),
            max_connections: 256,
        }
    }
}

/// Open connections, bounded; watched so a drain can wait for the last one.
#[derive(Debug)]
pub(crate) struct Connections {
    max: usize,
    open: watch::Sender<usize>,
}

/// One admitted connection, released when dropped.
#[derive(Debug)]
pub(crate) struct Admission(Arc<Connections>);

impl Connections {
    fn new(max: usize) -> Arc<Self> {
        Arc::new(Self { max, open: watch::Sender::new(0) })
    }

    pub(crate) fn admit(self: &Arc<Self>) -> Option<Admission> {
        let admitted = self.open.send_if_modified(|open| {
            let room = *open < self.max;
            if room {
                *open += 1;
            }
            room
        });
        admitted.then(|| Admission(self.clone()))
    }

    /// Until no connection is open.
    async fn idle(&self) {
        let _ = self.open.subscribe().wait_for(|open| *open == 0).await;
    }

    #[cfg(test)]
    pub(crate) fn open(&self) -> usize {
        *self.open.borrow()
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        self.0.open.send_modify(|open| *open -= 1);
    }
}

/// What an exchange does to its stream besides moving bytes.
pub(crate) trait StreamEnd: Send + Sync {
    /// FIN: the response is complete.
    fn finish(&self);
    /// RESET_STREAM and STOP_SENDING.
    fn reset(&self, code: HttpStreamResetCode);
    /// STOP_SENDING only: the request is no longer read.
    fn stop(&self, code: HttpStreamResetCode);
}

pub(crate) type ReadHalf = Pin<Box<dyn AsyncRead + Send>>;
pub(crate) type WriteHalf = Pin<Box<dyn AsyncWrite + Send>>;

/// One exchange stream after its preamble.
pub(crate) struct Wire {
    pub read: ReadHalf,
    pub write: WriteHalf,
    pub end: Arc<dyn StreamEnd>,
}

/// The agent's endpoint server: the binding's handler, its limits and its open WebSockets.
pub struct EndpointServer {
    handler: Arc<dyn EndpointHandler>,
    limits: HttpServerLimits,
    connections: Arc<Connections>,
    /// Changed only with `websockets` locked, so a drain and a registration never race.
    state: watch::Sender<ServerState>,
    websockets: parking_lot::Mutex<WebSockets>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ServerState {
    Serving,
    /// No new exchange is admitted; those in flight finish.
    Draining,
    /// Exchanges still running are cancelled.
    Stopping,
}

#[derive(Default)]
struct WebSockets {
    next: u64,
    open: HashMap<u64, Arc<Exchange>>,
}

/// Unregisters an open WebSocket however its exchange ends.
struct Registered<'a> {
    server: &'a EndpointServer,
    id: u64,
}

impl Drop for Registered<'_> {
    fn drop(&mut self) {
        self.server.websockets.lock().open.remove(&self.id);
    }
}

impl EndpointServer {
    pub fn new(handler: Arc<dyn EndpointHandler>, limits: HttpServerLimits) -> Self {
        let connections = Connections::new(limits.max_connections);
        let state = watch::Sender::new(ServerState::Serving);
        Self { handler, limits, connections, state, websockets: Default::default() }
    }

    /// Admit no new exchange, send every open WebSocket a close frame so its client reconnects
    /// to another worker, and wait for the exchanges in flight to finish. The caller bounds the
    /// wait; `stop` ends what remains.
    pub async fn drain(&self) {
        let open = self.enter(ServerState::Draining);
        futures_util::future::join_all(
            open.iter().map(|exchange| exchange.close(websocket::GOING_AWAY, "worker draining")),
        )
        .await;
        self.connections.idle().await;
    }

    /// Cancel every exchange still running: a started HTTP response ends with a `canceled`
    /// completion while the stream can still carry it, a WebSocket closes with `1001`. Waits up
    /// to `stop_grace` for them to settle.
    pub async fn stop(&self) {
        self.enter(ServerState::Stopping);
        let _ = tokio::time::timeout(self.limits.stop_grace, self.connections.idle()).await;
    }

    /// Move to `state`, never back; the WebSockets open at that moment.
    fn enter(&self, state: ServerState) -> Vec<Arc<Exchange>> {
        let websockets = self.websockets.lock();
        self.state.send_if_modified(|current| {
            let advanced = *current < state;
            if advanced {
                *current = state;
            }
            advanced
        });
        websockets.open.values().cloned().collect()
    }

    fn admitting(&self) -> bool {
        *self.state.borrow() == ServerState::Serving
    }

    /// Track an open WebSocket for drain; `None` once draining.
    fn register(&self, exchange: &Arc<Exchange>) -> Option<Registered<'_>> {
        let mut websockets = self.websockets.lock();
        if *self.state.borrow() != ServerState::Serving {
            return None;
        }
        let id = websockets.next;
        websockets.next += 1;
        websockets.open.insert(id, exchange.clone());
        Some(Registered { server: self, id })
    }

    /// Serve one exchange stream to completion.
    pub(crate) async fn serve(&self, wire: Wire, origin: Origin) {
        let write = Box::pin(WriteIdle::new(wire.write, self.limits.idle_timeout));
        exchange::serve(self, Wire { write, ..wire }, origin).await;
    }
}

/// Fails a write that makes no progress for its limit: a peer that stopped reading.
struct WriteIdle {
    inner: WriteHalf,
    limit: Duration,
    stall: Pin<Box<Sleep>>,
    stalled: bool,
}

impl WriteIdle {
    fn new(inner: WriteHalf, limit: Duration) -> Self {
        Self { inner, limit, stall: Box::pin(tokio::time::sleep(limit)), stalled: false }
    }

    fn progress<T>(
        &mut self,
        cx: &mut Context<'_>,
        poll: Poll<std::io::Result<T>>,
    ) -> Poll<std::io::Result<T>> {
        if poll.is_ready() {
            self.stalled = false;
            return poll;
        }
        if !self.stalled {
            self.stalled = true;
            let deadline = tokio::time::Instant::now() + self.limit;
            self.stall.as_mut().reset(deadline);
        }
        match self.stall.as_mut().poll(cx) {
            Poll::Ready(()) => Poll::Ready(Err(std::io::ErrorKind::TimedOut.into())),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncWrite for WriteIdle {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let poll = self.inner.as_mut().poll_write(cx, buf);
        self.progress(cx, poll)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let poll = self.inner.as_mut().poll_flush(cx);
        self.progress(cx, poll)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let poll = self.inner.as_mut().poll_shutdown(cx);
        self.progress(cx, poll)
    }
}

/// One node stream, or one public listener connection, after its preamble. Reads and writes are
/// independent; dropping it without `finish` resets with `HSR_ABORT`.
pub struct HttpStream {
    reader: Mutex<EndpointReader>,
    writer: Mutex<EndpointWriter>,
    control: EndpointControl,
}

impl HttpStream {
    pub fn new(reader: EndpointReader, writer: EndpointWriter) -> Self {
        let control = writer.control();
        Self { reader: Mutex::new(reader), writer: Mutex::new(writer), control }
    }

    /// Up to `max_bytes`; empty once the peer half-closed.
    pub async fn read(&self, max_bytes: usize) -> Result<Vec<u8>, TransportError> {
        let mut reader = self.reader.lock().await;
        let mut buf = vec![0u8; max_bytes.max(1)];
        match reader.read(&mut buf).await? {
            Some(n) => {
                buf.truncate(n);
                Ok(buf)
            }
            None => Ok(Vec::new()),
        }
    }

    pub async fn write(&self, data: &[u8]) -> Result<(), TransportError> {
        self.writer.lock().await.write_all(data).await
    }

    /// Control is admitted independently of pending data I/O: applied now when the half is
    /// free, else by the I/O holding it.
    fn apply_writer(&self) {
        if let Ok(mut writer) = self.writer.try_lock() {
            let _ = writer.apply_control();
        }
    }

    fn apply_reader(&self) {
        if let Ok(mut reader) = self.reader.try_lock() {
            reader.apply_control();
        }
    }

    /// The stream as async byte halves the exchange reads and writes.
    pub(crate) fn into_wire(self) -> Wire {
        let stream = Arc::new(self);
        let chunks = futures_util::stream::unfold(Some(stream.clone()), |stream| async move {
            let stream = stream?;
            match stream.read(READ_SIZE).await {
                Ok(chunk) if chunk.is_empty() => None,
                Ok(chunk) => Some((Ok(Cursor::new(chunk)), Some(stream))),
                Err(error) => Some((Err(std::io::Error::other(error.to_string())), None)),
            }
        });
        let sink = futures_util::sink::unfold(stream.clone(), |stream, data: Bytes| async move {
            stream.write(&data).await.map_err(|error| std::io::Error::other(error.to_string()))?;
            Ok::<_, std::io::Error>(stream)
        });
        Wire {
            read: Box::pin(tokio_util::io::StreamReader::new(chunks)),
            write: Box::pin(tokio_util::io::SinkWriter::new(tokio_util::io::CopyToBytes::new(
                Box::pin(sink),
            ))),
            end: stream,
        }
    }
}

impl StreamEnd for HttpStream {
    fn finish(&self) {
        self.control.finish();
        self.apply_writer();
    }

    fn reset(&self, code: HttpStreamResetCode) {
        self.control.reset(code);
        self.apply_writer();
        self.stop(code);
    }

    fn stop(&self, code: HttpStreamResetCode) {
        self.control.stop(code);
        self.apply_reader();
    }
}

/// `wire.MaxPreambleSize` is small, so a peer that opens a stream and sends nothing is the only
/// way to wait here; the node writes the preamble in the same call as the request head.
const PREAMBLE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn accept<P: WorkerHost + ?Sized>(host: Arc<P>, mut stream: TransportStream) {
    let preamble = match tokio::time::timeout(PREAMBLE_TIMEOUT, stream.read_preamble()).await {
        Ok(Ok(preamble)) => preamble,
        Ok(Err(error)) => {
            host.logger().emit(
                log::Level::Warn,
                format!("endpoint stream without a valid preamble error={error}"),
            );
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
        Err(_) => {
            host.logger().emit(log::Level::Warn, "endpoint stream sent no preamble in time".into());
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
    };
    host.serve_endpoint(IncomingRequest { preamble, stream }).await;
}

/// The node's metadata and the stream positioned at the first application byte.
#[derive(Debug)]
pub struct IncomingRequest {
    pub preamble: StreamPreamble,
    pub stream: TransportStream,
}

/// Serve a node's request with the mounted endpoint server.
pub(crate) async fn dispatch(request: IncomingRequest, server: Option<&EndpointServer>) {
    let IncomingRequest { preamble, stream } = request;
    let Some(server) = server else {
        return stream.refuse();
    };
    let info = match metadata(preamble) {
        Ok(info) => info,
        Err(error) => {
            log::warn!(target: crate::LOG, "invalid HTTP stream metadata: {error}");
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
    };
    let (writer, reader) = stream.split();
    server.serve(HttpStream::new(reader, writer).into_wire(), Origin::Node(info)).await;
}

fn metadata(preamble: StreamPreamble) -> Result<RequestMetadata, TransportError> {
    let kind = match AgentEndpointKind::try_from(preamble.kind) {
        Ok(AgentEndpointKind::AekHttp) => EndpointKind::Http,
        Ok(AgentEndpointKind::AekText) => EndpointKind::Text,
        _ => {
            return Err(TransportError::Protocol(format!(
                "unknown endpoint kind {}",
                preamble.kind
            )));
        }
    };
    let scheme = match preamble.scheme.as_str() {
        "http" => HttpScheme::Http,
        "https" => HttpScheme::Https,
        scheme => {
            return Err(TransportError::Protocol(format!("unknown client scheme {scheme:?}")));
        }
    };
    Ok(RequestMetadata {
        request_id: preamble.request_id,
        authorized: preamble.authorized,
        route: preamble.route,
        kind,
        client_addr: preamble.client_addr,
        scheme,
        timeout_ms: u64::from(preamble.timeout_ms),
        unauthenticated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_preserves_the_nodes_values_and_uses_typed_kinds_and_schemes() {
        for (wire, kind) in [
            (AgentEndpointKind::AekHttp, EndpointKind::Http),
            (AgentEndpointKind::AekText, EndpointKind::Text),
        ] {
            let info = metadata(StreamPreamble {
                request_id: "AER_test".into(),
                route: "/items/{id}".into(),
                kind: wire as i32,
                scheme: "https".into(),
                client_addr: "2001:db8::1".into(),
                authorized: true,
                timeout_ms: 1234,
            })
            .unwrap();
            assert_eq!(info.request_id, "AER_test");
            assert_eq!(info.route, "/items/{id}");
            assert_eq!(info.kind, kind);
            assert_eq!(info.scheme, HttpScheme::Https);
            assert_eq!(info.client_addr, "2001:db8::1");
            assert!(info.authorized);
            assert_eq!(info.timeout_ms, 1234);
        }
    }

    #[test]
    fn unknown_metadata_is_rejected_instead_of_becoming_http() {
        for (kind, scheme) in [
            (i32::MAX, "https"),
            (AgentEndpointKind::AekHttp as i32, ""),
            (AgentEndpointKind::AekHttp as i32, "ftp"),
        ] {
            assert!(matches!(
                metadata(StreamPreamble { kind, scheme: scheme.into(), ..Default::default() }),
                Err(TransportError::Protocol(_))
            ));
        }
    }

    #[test]
    fn admissions_are_bounded_and_released_on_drop() {
        let connections = Connections::new(1);
        let admitted = connections.admit().unwrap();
        assert!(connections.admit().is_none(), "one connection at a time");
        drop(admitted);
        assert_eq!(connections.open(), 0);
        assert!(connections.admit().is_some());
    }
}
