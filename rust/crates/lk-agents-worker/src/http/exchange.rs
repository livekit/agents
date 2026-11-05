//! One HTTP/1.1 exchange on an endpoint stream: the request head and body in, the response out.
//! Each stream carries exactly one exchange. A failure after the response head is reported in
//! the chunked body's completion trailers; before it, by resetting the stream.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::telemetry;
use lk_agents_abi::worker::http::{
    EndpointExchange, EndpointRequest, ExchangeError, ExchangeEvent, ExchangeKind, ExchangeMessage,
    HttpExchange, RequestHead,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, watch};

use super::websocket::Connection;
use super::{EndpointServer, READ_SIZE, ReadHalf, ServerState, StreamEnd, Wire, WriteHalf};
use crate::server::PublicRoutes;
use crate::transport::HttpStreamResetCode as Code;

const MAX_HEADERS: usize = 128;
const REASON_LIMIT: usize = 256;
/// The node's completion trailer: its absence, or an empty value, is success.
const COMPLETION: &[u8] = b"x-lk-completion";
const COMPLETION_ERROR: &[u8] = b"x-lk-error";

/// How a response that already started ended, as its completion trailer names it.
#[derive(Debug, Clone, Copy)]
enum Completion {
    /// The application failed.
    Internal,
    /// The worker stopped while the response streamed.
    Canceled,
}

impl Completion {
    fn token(self) -> &'static [u8] {
        match self {
            Completion::Internal => b"internal",
            Completion::Canceled => b"canceled",
        }
    }
}

/// One exchange as the binding sees it.
pub struct Exchange {
    head: RequestHead,
    kind: Kind,
}

enum Kind {
    Http(Box<Http>),
    WebSocket(Connection),
}

impl EndpointExchange for Exchange {}

impl HttpExchange for Exchange {
    fn head(&self) -> RequestHead {
        self.head.clone()
    }

    async fn receive(&self) -> Result<ExchangeEvent, ExchangeError> {
        match &self.kind {
            Kind::Http(http) => http.receive().await,
            Kind::WebSocket(connection) => connection.receive().await,
        }
    }

    async fn send(&self, message: ExchangeMessage) -> Result<(), ExchangeError> {
        match &self.kind {
            Kind::Http(http) => http.send(message).await,
            Kind::WebSocket(connection) => connection.send(message).await,
        }
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        if let Kind::WebSocket(connection) = &self.kind {
            connection.shutdown();
        }
    }
}

impl Exchange {
    fn end_refused(&self) {
        if let Kind::WebSocket(connection) = &self.kind {
            connection.refuse();
        }
    }

    /// Close an open WebSocket; an HTTP exchange runs to its own end.
    pub(super) async fn close(&self, code: u16, reason: &str) {
        if let Kind::WebSocket(connection) = &self.kind {
            connection.close(code, reason).await;
        }
    }
}

/// The head limit of the worker's own listener, which faces clients directly.
const LISTENER_HEAD_BYTES: usize = 64 << 10;

/// Where an exchange came from.
pub(crate) enum Origin {
    /// A node's stream: the node routed it and vouches for the caller.
    Node(EndpointRequest),
    /// The worker's own listener, which routes the parsed head itself.
    Listener { routes: Arc<PublicRoutes>, peer: String },
}

/// Why a request head was not accepted.
enum HeadError {
    Malformed(&'static str),
    TooLarge,
    TimedOut,
}

impl HeadError {
    fn reason(&self) -> &'static str {
        match self {
            HeadError::Malformed(reason) => reason,
            HeadError::TooLarge => "the request head exceeds its limit",
            HeadError::TimedOut => "no request head in time",
        }
    }

    /// The status the worker's own listener answers with.
    fn status(&self) -> u16 {
        match self {
            HeadError::Malformed(_) => 400,
            HeadError::TooLarge => 431,
            HeadError::TimedOut => 408,
        }
    }
}

pub(super) async fn serve(server: &EndpointServer, wire: Wire, origin: Origin) {
    let limits = &server.limits;
    let head_limit = match origin {
        Origin::Node(_) => limits.request_head_bytes,
        Origin::Listener { .. } => LISTENER_HEAD_BYTES,
    };
    // A node retries a refused stream on another worker.
    let draining = !server.admitting();
    if draining && matches!(origin, Origin::Node(_)) {
        return wire.end.reset(Code::HsrRefused);
    }
    let Some(_admitted) = server.connections.admit() else {
        wire.end.reset(Code::HsrRefused);
        return;
    };
    let Wire { read, write, end } = wire;
    let mut read = BufReader::with_capacity(READ_SIZE, read);
    let parsed = tokio::time::timeout(limits.head_timeout, parse(&mut read, head_limit))
        .await
        .unwrap_or(Err(HeadError::TimedOut));
    let (parsed, request) = match (parsed, origin) {
        (Ok(parsed), Origin::Node(request)) => (parsed, request),
        (Ok(_), Origin::Listener { .. }) if draining => {
            return refuse(write, end, 503).await;
        }
        (Ok(mut parsed), Origin::Listener { routes, peer }) => {
            match routes.resolve(&mut parsed.head, &peer) {
                Ok(request) => (parsed, request),
                Err(status) => return refuse(write, end, status).await,
            }
        }
        (Err(error), Origin::Node(_)) => {
            log::debug!(target: crate::LOG, "endpoint request rejected: {}", error.reason());
            return end.reset(Code::HsrProtocol);
        }
        (Err(error), Origin::Listener { .. }) => return refuse(write, end, error.status()).await,
    };
    let deadline = (request.timeout_ms > 0).then(|| Duration::from_millis(request.timeout_ms));
    let response = Response::new(write, parsed.version_11, parsed.head_method);
    match parsed.websocket_key {
        None => {
            let request_side = Request {
                read,
                body: parsed.body,
                metadata_bytes: limits.body_metadata_bytes,
                idle_timeout: limits.idle_timeout,
                expect_continue: parsed.expect_continue,
                disconnected: false,
            };
            let http = Http {
                request: Mutex::new(request_side),
                response: Mutex::new(response),
                phase: watch::Sender::new(Phase::Waiting),
                end,
            };
            let exchange =
                Arc::new(Exchange { head: parsed.head, kind: Kind::Http(Box::new(http)) });
            let native = exchange.clone() as Arc<dyn EndpointExchange>;
            // An A2A route is the worker's own: its door answers it.
            let handled = match server.text.as_ref().and_then(|text| text.route(&request)) {
                Some((door, serving)) => door.serve(native, serving),
                None => server.handler.handle(&telemetry::Cx::NONE, native, request),
            };
            let Kind::Http(http) = &exchange.kind else { unreachable!("built as HTTP") };
            http.run(handled, deadline, server.state.subscribe()).await;
        }
        Some(key) => {
            let connection = Connection::new(read, response, end, &key, limits.clone());
            let exchange =
                Arc::new(Exchange { head: parsed.head, kind: Kind::WebSocket(connection) });
            let Some(_registered) = server.register(&exchange) else {
                // Drain began while the upgrade was read.
                exchange.end_refused();
                return;
            };
            let handled = server.handler.handle(
                &telemetry::Cx::NONE,
                exchange.clone() as Arc<dyn EndpointExchange>,
                request,
            );
            let Kind::WebSocket(connection) = &exchange.kind else {
                unreachable!("built as a WebSocket")
            };
            // An upgrade is answered within the node's deadline, else the head timeout.
            connection
                .run(handled, deadline.unwrap_or(limits.head_timeout), server.state.subscribe())
                .await;
        }
    }
}

/// Answer a request the application never sees with a bodiless status.
async fn refuse(write: WriteHalf, end: Arc<dyn StreamEnd>, status: u16) {
    let mut response = Response::new(write, true, false);
    let headers = [(b"content-length".to_vec(), b"0".to_vec())];
    let answered = async {
        response.start(status, &headers).await?;
        response.body(b"", false).await
    };
    match answered.await {
        Ok(_) => end.finish(),
        Err(_) => end.reset(Code::HsrAbort),
    }
}

/// A request path, percent-decoded for the application.
pub(crate) fn decode_path(raw: &str) -> String {
    percent_encoding::percent_decode_str(raw).decode_utf8_lossy().into_owned()
}

/// What the request head says about the exchange.
struct Parsed {
    head: RequestHead,
    body: Body,
    version_11: bool,
    head_method: bool,
    expect_continue: bool,
    /// The client's `sec-websocket-key`, for an upgrade.
    websocket_key: Option<String>,
}

async fn parse(read: &mut BufReader<ReadHalf>, head_limit: usize) -> Result<Parsed, HeadError> {
    let raw = read_head(read, head_limit).await?;
    let mut fields = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut parsed = httparse::Request::new(&mut fields);
    match parsed.parse(&raw) {
        Ok(httparse::Status::Complete(n)) if n == raw.len() => {}
        _ => return Err(HeadError::Malformed("malformed request head")),
    }
    let (Some(method), Some(target), Some(version)) = (parsed.method, parsed.path, parsed.version)
    else {
        return Err(HeadError::Malformed("malformed request line"));
    };
    if !target.starts_with('/') {
        return Err(HeadError::Malformed("the request target is not a path"));
    }
    let version_11 = version == 1;
    let headers: Vec<(Vec<u8>, Vec<u8>)> = parsed
        .headers
        .iter()
        .map(|field| (field.name.to_ascii_lowercase().into_bytes(), field.value.to_vec()))
        .collect();
    if version_11 && values(&headers, b"host").count() != 1 {
        return Err(HeadError::Malformed("an HTTP/1.1 request names exactly one host"));
    }
    let body = body_framing(&headers, version_11).map_err(HeadError::Malformed)?;

    let upgrade = tokens(&headers, b"upgrade").any(|token| token.eq_ignore_ascii_case("websocket"))
        && tokens(&headers, b"connection").any(|token| token.eq_ignore_ascii_case("upgrade"));
    let websocket_key = if upgrade && method == "GET" && version_11 {
        let mut keys = values(&headers, b"sec-websocket-key");
        let (Some(key), None) = (keys.next(), keys.next()) else {
            return Err(HeadError::Malformed("a WebSocket upgrade carries one key"));
        };
        if !tokens(&headers, b"sec-websocket-version").eq(["13"]) {
            return Err(HeadError::Malformed("unsupported WebSocket version"));
        }
        if !matches!(body, Body::Length(0)) {
            return Err(HeadError::Malformed("a WebSocket upgrade has no body"));
        }
        Some(String::from_utf8_lossy(key).trim().to_owned())
    } else {
        None
    };

    let (raw_path, query) = match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    };
    let expect_continue =
        tokens(&headers, b"expect").any(|token| token.eq_ignore_ascii_case("100-continue"));
    let subprotocols = tokens(&headers, b"sec-websocket-protocol").map(str::to_owned).collect();
    Ok(Parsed {
        head: RequestHead {
            kind: if websocket_key.is_some() {
                ExchangeKind::Websocket
            } else {
                ExchangeKind::Http
            },
            method: method.to_owned(),
            http_version: if version_11 { "1.1" } else { "1.0" }.to_owned(),
            path: decode_path(raw_path),
            raw_path: raw_path.as_bytes().to_vec(),
            query_string: query.as_bytes().to_vec(),
            headers,
            subprotocols,
        },
        body,
        version_11,
        head_method: method == "HEAD",
        expect_continue,
        websocket_key,
    })
}

/// The request's body framing. Both `transfer-encoding` and `content-length`, a coding other
/// than `chunked`, or disagreeing lengths are rejected rather than guessed (request smuggling).
fn body_framing(headers: &[(Vec<u8>, Vec<u8>)], version_11: bool) -> Result<Body, &'static str> {
    let lengths = values(headers, b"content-length").count();
    if values(headers, b"transfer-encoding").next().is_some() {
        if !version_11 || lengths > 0 {
            return Err("ambiguous request body framing");
        }
        if !tokens(headers, b"transfer-encoding").eq(["chunked"]) {
            return Err("unsupported transfer-coding");
        }
        return Ok(Body::ChunkSize);
    }
    let mut length = None;
    for value in tokens(headers, b"content-length") {
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) || value.len() > 19 {
            return Err("invalid content-length");
        }
        let value: u64 = value.parse().map_err(|_| "invalid content-length")?;
        if length.is_some_and(|length| length != value) {
            return Err("conflicting content-length values");
        }
        length = Some(value);
    }
    if lengths > 0 && length.is_none() {
        return Err("invalid content-length");
    }
    Ok(Body::Length(length.unwrap_or(0)))
}

fn values<'a>(
    headers: &'a [(Vec<u8>, Vec<u8>)],
    name: &'a [u8],
) -> impl Iterator<Item = &'a [u8]> + 'a {
    headers.iter().filter(move |(key, _)| key == name).map(|(_, value)| value.as_slice())
}

/// The comma-separated elements of every `name` field, trimmed.
fn tokens<'a>(headers: &'a [(Vec<u8>, Vec<u8>)], name: &'a [u8]) -> impl Iterator<Item = &'a str> {
    values(headers, name)
        .filter_map(|value| std::str::from_utf8(value).ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

/// The bytes through the blank line that ends the head; what follows stays buffered.
async fn read_head(read: &mut BufReader<ReadHalf>, limit: usize) -> Result<Vec<u8>, HeadError> {
    let mut head = Vec::new();
    loop {
        let buf = read
            .fill_buf()
            .await
            .map_err(|_| HeadError::Malformed("the stream failed before the request head"))?;
        if buf.is_empty() {
            return Err(HeadError::Malformed("the stream ended before the request head"));
        }
        let from = head.len().saturating_sub(3);
        let take = buf.len().min(limit - head.len());
        head.extend_from_slice(&buf[..take]);
        if let Some(at) = head[from..].windows(4).position(|window| window == b"\r\n\r\n") {
            let end = from + at + 4;
            read.consume(take - (head.len() - end));
            head.truncate(end);
            return Ok(head);
        }
        read.consume(take);
        if head.len() >= limit {
            return Err(HeadError::TooLarge);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    /// The bytes left of a `content-length` body.
    Length(u64),
    /// At a chunk-size line.
    ChunkSize,
    /// The bytes left of the current chunk.
    ChunkData(u64),
    /// The last piece was delivered.
    Complete,
}

#[derive(Debug)]
enum BodyError {
    /// Bad framing, or the stream ended inside the body.
    Malformed,
    /// The node reported that the client's body was cut short.
    Truncated,
}

struct Request {
    read: BufReader<ReadHalf>,
    body: Body,
    metadata_bytes: usize,
    /// A body that makes no progress for this long is abandoned.
    idle_timeout: Duration,
    expect_continue: bool,
    disconnected: bool,
}

impl Request {
    /// The next piece of the body, and whether more follows.
    async fn next(&mut self) -> Result<(Vec<u8>, bool), BodyError> {
        loop {
            match self.body {
                Body::Length(0) => {
                    self.body = Body::Complete;
                    return Ok((Vec::new(), false));
                }
                Body::Length(left) => {
                    let data = self.read_up_to(left).await?;
                    let left = left - data.len() as u64;
                    self.body = if left == 0 { Body::Complete } else { Body::Length(left) };
                    return Ok((data, left > 0));
                }
                Body::ChunkSize => {
                    let line = self.read_line(self.metadata_bytes).await?;
                    match chunk_size(&line)? {
                        0 => {
                            self.read_trailers().await?;
                            self.body = Body::Complete;
                            return Ok((Vec::new(), false));
                        }
                        size => self.body = Body::ChunkData(size),
                    }
                }
                Body::ChunkData(left) => {
                    let data = self.read_up_to(left).await?;
                    let left = left - data.len() as u64;
                    self.body = if left == 0 {
                        if !self.read_line(2).await?.is_empty() {
                            return Err(BodyError::Malformed);
                        }
                        Body::ChunkSize
                    } else {
                        Body::ChunkData(left)
                    };
                    return Ok((data, true));
                }
                Body::Complete => return Ok((Vec::new(), false)),
            }
        }
    }

    /// Buffered bytes, at least one; the end of the stream inside a body is malformed.
    async fn fill(&mut self) -> Result<&[u8], BodyError> {
        match tokio::time::timeout(self.idle_timeout, self.read.fill_buf()).await {
            Ok(Ok(buf)) if !buf.is_empty() => Ok(buf),
            _ => Err(BodyError::Malformed),
        }
    }

    async fn read_up_to(&mut self, left: u64) -> Result<Vec<u8>, BodyError> {
        let buf = self.fill().await?;
        let take = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
        let data = buf[..take].to_vec();
        self.read.consume(take);
        Ok(data)
    }

    /// One CRLF-terminated line, without its CRLF; bare LF is malformed.
    async fn read_line(&mut self, limit: usize) -> Result<Vec<u8>, BodyError> {
        let mut line = Vec::new();
        loop {
            let buf = self.fill().await?;
            let (take, done) = match buf.iter().position(|b| *b == b'\n') {
                Some(at) => (at + 1, true),
                None => (buf.len(), false),
            };
            if line.len() + take > limit {
                return Err(BodyError::Malformed);
            }
            line.extend_from_slice(&buf[..take]);
            self.read.consume(take);
            if done {
                return line.strip_suffix(b"\r\n").map(<[u8]>::to_vec).ok_or(BodyError::Malformed);
            }
        }
    }

    /// The trailer section, bounded as a whole; a completion other than success is a truncated
    /// body.
    async fn read_trailers(&mut self) -> Result<(), BodyError> {
        let mut budget = self.metadata_bytes;
        let mut truncated = false;
        loop {
            let line = self.read_line(budget).await?;
            if line.is_empty() {
                return if truncated { Err(BodyError::Truncated) } else { Ok(()) };
            }
            budget = budget.saturating_sub(line.len() + 2);
            let (name, value) = line
                .split_at_checked(line.iter().position(|b| *b == b':').ok_or(BodyError::Malformed)?)
                .ok_or(BodyError::Malformed)?;
            if name.eq_ignore_ascii_case(COMPLETION) && !value[1..].trim_ascii().is_empty() {
                truncated = true;
            }
        }
    }
}

/// A chunk-size line: hex digits, then optional extensions, which are ignored.
fn chunk_size(line: &[u8]) -> Result<u64, BodyError> {
    let digits = line.split(|b| *b == b';').next().unwrap_or_default().trim_ascii();
    if digits.is_empty() || digits.len() > 15 || !digits.iter().all(u8::is_ascii_hexdigit) {
        return Err(BodyError::Malformed);
    }
    let digits = std::str::from_utf8(digits).map_err(|_| BodyError::Malformed)?;
    u64::from_str_radix(digits, 16).map_err(|_| BodyError::Malformed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Waiting,
    Started,
    Done,
}

struct Http {
    request: Mutex<Request>,
    response: Mutex<Response>,
    phase: watch::Sender<Phase>,
    end: Arc<dyn StreamEnd>,
}

impl Http {
    /// Run the application to its end, a missed response-head deadline, or the server stopping;
    /// dropping the handler cancels the application.
    async fn run(
        &self,
        handled: BoxFuture<'_, Result<(), HostError>>,
        deadline: Option<Duration>,
        mut state: watch::Receiver<ServerState>,
    ) {
        let mut handled = handled;
        let stopped = async {
            let _ = state.wait_for(|state| *state == ServerState::Stopping).await;
        };
        let head_missed = async {
            let Some(limit) = deadline else { return std::future::pending().await };
            let mut phase = self.phase.subscribe();
            let started = phase.wait_for(|p| *p != Phase::Waiting);
            if tokio::time::timeout(limit, started).await.is_ok() {
                std::future::pending::<()>().await;
            }
        };
        let result = tokio::select! {
            biased;
            () = stopped => {
                drop(handled);
                let completion = Completion::Canceled;
                return self.fail(completion, "worker stopped", Code::HsrAbort).await;
            }
            () = head_missed => {
                drop(handled);
                return self.abort(Code::HsrTimeout);
            }
            result = &mut handled => result,
        };
        let reason = match &result {
            Ok(()) => "application returned with no complete response".to_owned(),
            Err(error) => error.to_string(),
        };
        self.fail(Completion::Internal, &reason, Code::HsrInternal).await;
    }

    async fn receive(&self) -> Result<ExchangeEvent, ExchangeError> {
        let mut request = self.request.lock().await;
        if request.disconnected {
            return Ok(ExchangeEvent::Disconnected);
        }
        if request.body == Body::Complete {
            drop(request);
            // The end of the request body does not disconnect a response still streaming.
            let _ = self.phase.subscribe().wait_for(|p| *p == Phase::Done).await;
            return Ok(ExchangeEvent::Disconnected);
        }
        if std::mem::take(&mut request.expect_continue) {
            self.response.lock().await.interim_continue().await;
        }
        match request.next().await {
            Ok((data, more)) => Ok(ExchangeEvent::Body { data, more }),
            Err(BodyError::Truncated) => {
                request.disconnected = true;
                Ok(ExchangeEvent::Disconnected)
            }
            Err(BodyError::Malformed) => {
                request.disconnected = true;
                self.abort(Code::HsrProtocol);
                Ok(ExchangeEvent::Disconnected)
            }
        }
    }

    async fn send(&self, message: ExchangeMessage) -> Result<(), ExchangeError> {
        if *self.phase.borrow() == Phase::Done {
            return Err(ExchangeError::Closed("the exchange is closed".into()));
        }
        let mut response = self.response.lock().await;
        match message {
            ExchangeMessage::Start { status, headers } => {
                response.start(status, &headers).await?;
                self.phase.send_replace(Phase::Started);
            }
            ExchangeMessage::Body { data, more } => {
                if response.body(&data, more).await? {
                    self.end.finish();
                    // The response is whole: what remains of the request is not read.
                    self.end.stop(Code::HsrAbort);
                    self.phase.send_replace(Phase::Done);
                }
            }
            _ => {
                return Err(ExchangeError::Protocol(
                    "a WebSocket message on an HTTP exchange".into(),
                ));
            }
        }
        Ok(())
    }

    /// End the exchange without its response; no-op once it ended.
    fn abort(&self, code: Code) {
        if self.phase.send_replace(Phase::Done) != Phase::Done {
            self.end.reset(code);
        }
    }

    async fn fail(&self, completion: Completion, reason: &str, code: Code) {
        if *self.phase.borrow() == Phase::Done {
            return;
        }
        let ended = self.response.lock().await.fail(completion, reason).await;
        if self.phase.send_replace(Phase::Done) == Phase::Done {
            return;
        }
        match ended {
            Failure::Unstarted => self.end.reset(code),
            Failure::Reported => {
                self.end.finish();
                self.end.stop(Code::HsrAbort);
            }
            Failure::Unframed => self.end.reset(Code::HsrAbort),
            Failure::Ended => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    Chunked,
    Length(u64),
    /// HTTP/1.0 without a length: the body ends with the stream.
    Close,
    /// HEAD, 204 and 304 answers carry no body.
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Waiting,
    Body(Framing),
    Done,
}

/// How a response that could not complete ended.
enum Failure {
    /// Nothing was written: reset with the failure's own code.
    Unstarted,
    /// The completion trailers went out.
    Reported,
    /// The framing cannot carry trailers.
    Unframed,
    /// The response had already ended.
    Ended,
}

/// The response side: one response, framed by the worker.
pub(super) struct Response {
    write: WriteHalf,
    version_11: bool,
    head_method: bool,
    state: State,
    /// A write was cancelled part-way, so the framing can no longer be trusted.
    interrupted: bool,
}

impl Response {
    fn new(write: WriteHalf, version_11: bool, head_method: bool) -> Self {
        Self { write, version_11, head_method, state: State::Waiting, interrupted: false }
    }

    async fn write_all(&mut self, parts: &[&[u8]]) -> Result<(), ExchangeError> {
        if self.interrupted {
            return Err(ExchangeError::Closed("an earlier response write was interrupted".into()));
        }
        self.interrupted = true;
        for part in parts {
            self.write.write_all(part).await.map_err(closed)?;
        }
        self.write.flush().await.map_err(closed)?;
        self.interrupted = false;
        Ok(())
    }

    pub(super) async fn start(
        &mut self,
        status: u16,
        headers: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ExchangeError> {
        if self.state != State::Waiting {
            return Err(ExchangeError::Protocol("the response already started".into()));
        }
        if !(200..=999).contains(&status) {
            return Err(ExchangeError::Protocol(format!("invalid response status {status}")));
        }
        let reason = hyper::StatusCode::from_u16(status)
            .ok()
            .and_then(|status| status.canonical_reason())
            .unwrap_or("");
        let mut out = format!("HTTP/1.1 {status} {reason}\r\n").into_bytes();
        let mut length = None;
        for (name, value) in headers {
            check_field(name, value)?;
            let name_lower = name.to_ascii_lowercase();
            // The worker frames the response and ends the stream after it.
            if name_lower == b"transfer-encoding" || name_lower == b"connection" {
                continue;
            }
            if name_lower == b"content-length" {
                let value = std::str::from_utf8(value)
                    .ok()
                    .and_then(|value| value.trim().parse::<u64>().ok())
                    .ok_or_else(|| ExchangeError::Protocol("invalid content-length".into()))?;
                if length.replace(value).is_some_and(|previous| previous != value) {
                    return Err(ExchangeError::Protocol("conflicting content-length".into()));
                }
            }
            push_field(&mut out, name, value);
        }
        let framing = if self.head_method || status == 204 || status == 304 {
            Framing::Empty
        } else if let Some(length) = length {
            Framing::Length(length)
        } else if self.version_11 {
            push_field(&mut out, b"transfer-encoding", b"chunked");
            Framing::Chunked
        } else {
            Framing::Close
        };
        push_field(&mut out, b"connection", b"close");
        out.extend_from_slice(b"\r\n");
        self.write_all(&[&out]).await?;
        self.state = State::Body(framing);
        Ok(())
    }

    /// Whether the response is now complete.
    pub(super) async fn body(&mut self, data: &[u8], more: bool) -> Result<bool, ExchangeError> {
        let framing = match self.state {
            State::Body(framing) => framing,
            State::Waiting => {
                return Err(ExchangeError::Protocol("a response body before its start".into()));
            }
            State::Done => return Err(ExchangeError::Closed("the response is complete".into())),
        };
        match framing {
            Framing::Empty => {}
            Framing::Length(left) => {
                let left = left.checked_sub(data.len() as u64).ok_or_else(|| {
                    ExchangeError::Protocol("the response body exceeds its content-length".into())
                })?;
                if !more && left > 0 {
                    return Err(ExchangeError::Protocol(
                        "the response body is shorter than its content-length".into(),
                    ));
                }
                self.write_all(&[data]).await?;
                self.state = State::Body(Framing::Length(left));
            }
            Framing::Chunked => {
                let size = format!("{:x}\r\n", data.len());
                let last: &[u8] = if more { b"" } else { b"0\r\n\r\n" };
                if data.is_empty() {
                    self.write_all(&[last]).await?;
                } else {
                    self.write_all(&[size.as_bytes(), data, b"\r\n", last]).await?;
                }
            }
            Framing::Close => self.write_all(&[data]).await?,
        }
        if !more {
            self.state = State::Done;
        }
        Ok(!more)
    }

    async fn interim_continue(&mut self) {
        if self.state == State::Waiting && self.version_11 {
            let _ = self.write_all(&[b"HTTP/1.1 100 Continue\r\n\r\n"]).await;
        }
    }

    async fn fail(&mut self, completion: Completion, reason: &str) -> Failure {
        let state = std::mem::replace(&mut self.state, State::Done);
        match state {
            State::Done => Failure::Ended,
            State::Waiting => Failure::Unstarted,
            State::Body(Framing::Chunked) if !self.interrupted => {
                let reason: String = reason
                    .chars()
                    .map(|c| if c.is_control() { ' ' } else { c })
                    .take(REASON_LIMIT)
                    .collect();
                let mut out = b"0\r\n".to_vec();
                push_field(&mut out, COMPLETION, completion.token());
                push_field(&mut out, COMPLETION_ERROR, reason.as_bytes());
                out.extend_from_slice(b"\r\n");
                match self.write_all(&[&out]).await {
                    Ok(()) => Failure::Reported,
                    Err(_) => Failure::Unframed,
                }
            }
            State::Body(_) => Failure::Unframed,
        }
    }

    /// Answer a WebSocket upgrade with `101`; the stream then carries frames.
    pub(super) async fn switch(
        &mut self,
        accept: &str,
        subprotocol: Option<&str>,
        headers: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ExchangeError> {
        if self.state != State::Waiting {
            return Err(ExchangeError::Protocol("the upgrade was already answered".into()));
        }
        let mut out = b"HTTP/1.1 101 Switching Protocols\r\n".to_vec();
        push_field(&mut out, b"upgrade", b"websocket");
        push_field(&mut out, b"connection", b"Upgrade");
        push_field(&mut out, b"sec-websocket-accept", accept.as_bytes());
        if let Some(subprotocol) = subprotocol {
            check_field(b"sec-websocket-protocol", subprotocol.as_bytes())?;
            push_field(&mut out, b"sec-websocket-protocol", subprotocol.as_bytes());
        }
        for (name, value) in headers {
            check_field(name, value)?;
            let name_lower = name.to_ascii_lowercase();
            let reserved: [&[u8]; 4] =
                [b"upgrade", b"connection", b"sec-websocket-accept", b"sec-websocket-protocol"];
            if !reserved.contains(&name_lower.as_slice()) {
                push_field(&mut out, name, value);
            }
        }
        out.extend_from_slice(b"\r\n");
        self.write_all(&[&out]).await?;
        self.state = State::Done;
        Ok(())
    }

    pub(super) fn into_write(self) -> WriteHalf {
        self.write
    }
}

fn check_field(name: &[u8], value: &[u8]) -> Result<(), ExchangeError> {
    let token = |b: &u8| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(b);
    if name.is_empty() || !name.iter().all(token) {
        return Err(ExchangeError::Protocol("invalid header name".into()));
    }
    if value.iter().any(|b| matches!(b, b'\r' | b'\n' | b'\0')) {
        return Err(ExchangeError::Protocol("invalid header value".into()));
    }
    Ok(())
}

fn push_field(out: &mut Vec<u8>, name: &[u8], value: &[u8]) {
    out.extend_from_slice(name);
    out.extend_from_slice(b": ");
    out.extend_from_slice(value);
    out.extend_from_slice(b"\r\n");
}

fn closed(error: std::io::Error) -> ExchangeError {
    ExchangeError::Closed(error.to_string())
}
