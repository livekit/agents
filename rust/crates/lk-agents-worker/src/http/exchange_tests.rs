//! The endpoint server over an in-memory stream: the test is the node, a Rust fake is the
//! binding's application.

use std::any::Any;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::telemetry;
use lk_agents_abi::worker::http::{
    EndpointExchange, EndpointHandler, EndpointRequest, ExchangeEvent, ExchangeKind,
    ExchangeMessage, HttpExchange,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

use super::{EndpointServer, Exchange, HttpServerLimits, Origin, StreamEnd, Wire};
use crate::transport::HttpStreamResetCode as Code;

#[derive(Default)]
pub(super) struct FakeEnd {
    finished: AtomicBool,
    reset: parking_lot::Mutex<Option<Code>>,
    stopped: parking_lot::Mutex<Option<Code>>,
}

impl StreamEnd for FakeEnd {
    fn finish(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }
    fn reset(&self, code: Code) {
        self.reset.lock().get_or_insert(code);
    }
    fn stop(&self, code: Code) {
        self.stopped.lock().get_or_insert(code);
    }
}

impl FakeEnd {
    pub(super) fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
    pub(super) fn reset(&self) -> Option<Code> {
        *self.reset.lock()
    }
}

type AppFn = dyn Fn(Arc<Exchange>) -> BoxFuture<'static, Result<(), HostError>> + Send + Sync;

struct App(Box<AppFn>);

impl EndpointHandler for App {
    fn handle(
        &self,
        _cx: &telemetry::Cx,
        exchange: Arc<dyn EndpointExchange>,
        _request: EndpointRequest,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        let exchange: Arc<dyn Any + Send + Sync> = exchange;
        (self.0)(exchange.downcast::<Exchange>().expect("the worker's exchange"))
    }
}

pub(super) fn server<F, Fut>(limits: HttpServerLimits, app: F) -> Arc<EndpointServer>
where
    F: Fn(Arc<Exchange>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), HostError>> + Send + 'static,
{
    let app = App(Box::new(move |exchange| Box::pin(app(exchange))));
    Arc::new(EndpointServer::new(Arc::new(app), limits))
}

pub(super) fn default_server<F, Fut>(app: F) -> Arc<EndpointServer>
where
    F: Fn(Arc<Exchange>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), HostError>> + Send + 'static,
{
    server(HttpServerLimits::default(), app)
}

pub(super) fn preamble() -> EndpointRequest {
    EndpointRequest {
        request_id: "AER_test".into(),
        authorized: true,
        route: "/".into(),
        ..Default::default()
    }
}

pub(super) fn wire() -> (Wire, DuplexStream, Arc<FakeEnd>) {
    wire_sized(8 << 20)
}

/// A stream whose buffer holds `capacity` bytes each way: a small one is a slow peer.
fn wire_sized(capacity: usize) -> (Wire, DuplexStream, Arc<FakeEnd>) {
    let (client, server) = tokio::io::duplex(capacity);
    let (read, write) = tokio::io::split(server);
    let end = Arc::new(FakeEnd::default());
    (Wire { read: Box::pin(read), write: Box::pin(write), end: end.clone() }, client, end)
}

/// Serve `raw` as a whole request, half-closed after it, and return the response bytes.
pub(super) async fn exchange_with(
    server: &EndpointServer,
    raw: &[u8],
    request: EndpointRequest,
) -> (Vec<u8>, Arc<FakeEnd>) {
    let (wire, mut client, end) = wire();
    client.write_all(raw).await.unwrap();
    client.shutdown().await.unwrap();
    let read = async {
        let mut out = Vec::new();
        client.read_to_end(&mut out).await.unwrap();
        out
    };
    let ((), out) = tokio::time::timeout(
        Duration::from_secs(5),
        futures_util::future::join(server.serve(wire, Origin::Node(request)), read),
    )
    .await
    .expect("the exchange settles");
    (out, end)
}

pub(super) async fn exchange(server: &EndpointServer, raw: &[u8]) -> (Vec<u8>, Arc<FakeEnd>) {
    exchange_with(server, raw, preamble()).await
}

pub(super) struct Parsed {
    pub(super) status: u16,
    pub(super) headers: Vec<(String, String)>,
    pub(super) body: Vec<u8>,
    pub(super) trailers: Vec<(String, String)>,
}

impl Parsed {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    fn trailer(&self, name: &str) -> Option<&str> {
        self.trailers.iter().find(|(key, _)| key == name).map(|(_, v)| v.as_str())
    }
}

fn fields(lines: &str) -> Vec<(String, String)> {
    lines
        .split("\r\n")
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (name, value) = line.split_once(':').unwrap();
            (name.trim().to_ascii_lowercase(), value.trim().to_owned())
        })
        .collect()
}

pub(super) fn parse(raw: &[u8]) -> Parsed {
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut response = httparse::Response::new(&mut headers);
    let httparse::Status::Complete(at) = response.parse(raw).unwrap() else {
        panic!("incomplete response head: {:?}", String::from_utf8_lossy(raw));
    };
    let status = response.code.unwrap();
    let headers: Vec<(String, String)> = response
        .headers
        .iter()
        .map(|h| (h.name.to_ascii_lowercase(), String::from_utf8_lossy(h.value).into_owned()))
        .collect();
    let mut rest = &raw[at..];
    let chunked = headers.iter().any(|(k, v)| k == "transfer-encoding" && v == "chunked");
    let mut body = Vec::new();
    let mut trailers = Vec::new();
    if chunked {
        loop {
            let line_end = rest.windows(2).position(|w| w == b"\r\n").expect("a chunk size");
            let size =
                usize::from_str_radix(std::str::from_utf8(&rest[..line_end]).unwrap(), 16).unwrap();
            rest = &rest[line_end + 2..];
            if size == 0 {
                let end = rest.windows(2).position(|w| w == b"\r\n").expect("trailers end");
                if end > 0 {
                    let all = rest.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                    trailers = fields(std::str::from_utf8(&rest[..all]).unwrap());
                }
                break;
            }
            body.extend_from_slice(&rest[..size]);
            assert_eq!(&rest[size..size + 2], b"\r\n");
            rest = &rest[size + 2..];
        }
    } else {
        body = rest.to_vec();
    }
    Parsed { status, headers, body, trailers }
}

pub(super) fn get(target: &str) -> Vec<u8> {
    format!("GET {target} HTTP/1.1\r\nhost: agents.test\r\n\r\n").into_bytes()
}

pub(super) fn post(target: &str, body: &[u8]) -> Vec<u8> {
    let mut raw = format!(
        "POST {target} HTTP/1.1\r\nhost: agents.test\r\ncontent-length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(body);
    raw
}

pub(super) fn start(status: u16, headers: &[(&str, &str)]) -> ExchangeMessage {
    ExchangeMessage::Start {
        status,
        headers: headers
            .iter()
            .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
            .collect(),
    }
}

pub(super) fn body(data: &[u8], more: bool) -> ExchangeMessage {
    ExchangeMessage::Body { data: data.to_vec(), more }
}

pub(super) async fn read_body(exchange: &Exchange) -> Option<Vec<u8>> {
    let mut all = Vec::new();
    loop {
        match exchange.receive().await.unwrap() {
            ExchangeEvent::Body { data, more } => {
                all.extend_from_slice(&data);
                if !more {
                    return Some(all);
                }
            }
            ExchangeEvent::Disconnected => return None,
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// Answer with the request body's length, or `-` when the client went away. After a malformed
/// body the exchange is already reset, so the answer fails.
pub(super) fn echo_size() -> Arc<EndpointServer> {
    default_server(|exchange| async move {
        let size = read_body(&exchange).await.map_or("-".to_owned(), |b| b.len().to_string());
        let length = size.len().to_string();
        let answered = async {
            exchange.send(start(200, &[("content-length", &length)])).await?;
            exchange.send(body(size.as_bytes(), false)).await
        };
        answered.await.map_err(|error| HostError::new(error.to_string()))
    })
}

#[tokio::test]
async fn a_small_exchange_round_trips_with_a_parsed_head() {
    let server = default_server(|exchange| async move {
        let head = exchange.head();
        assert_eq!(head.kind, ExchangeKind::Http);
        assert_eq!((head.method.as_str(), head.http_version.as_str()), ("GET", "1.1"));
        assert_eq!(head.path, "/a b/\u{20ac}");
        assert_eq!(head.raw_path, b"/a%20b/%E2%82%AC");
        assert_eq!(head.query_string, b"q=%20");
        assert!(head.headers.contains(&(b"x-custom".to_vec(), b"Value".to_vec())));
        assert_eq!(read_body(&exchange).await.unwrap(), b"");
        exchange.send(start(200, &[("content-type", "text/plain")])).await.unwrap();
        exchange.send(body(b"hello", false)).await.unwrap();
        Ok(())
    });
    let raw = b"GET /a%20b/%E2%82%AC?q=%20 HTTP/1.1\r\nHost: x\r\nX-Custom: Value\r\n\r\n";
    let (out, end) = exchange(&server, raw).await;
    let response = parse(&out);
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"hello");
    assert!(response.trailers.is_empty(), "success announces nothing in trailers");
    assert_eq!(response.header("connection"), Some("close"));
    assert!(end.finished() && end.reset().is_none());
    assert_eq!(*end.stopped.lock(), Some(Code::HsrAbort), "the request is not read further");
    assert_eq!(server.connections.open(), 0);
}

#[tokio::test]
async fn the_end_of_the_request_does_not_disconnect_a_streaming_response() {
    let server = default_server(|exchange| async move {
        assert_eq!(read_body(&exchange).await.unwrap(), b"");
        exchange.send(start(200, &[])).await.unwrap();
        let listener = exchange.clone();
        let disconnected = tokio::spawn(async move { listener.receive().await.unwrap() });
        for i in 0..3 {
            tokio::task::yield_now().await;
            assert!(!disconnected.is_finished());
            exchange.send(body(format!("chunk {i}\n").as_bytes(), true)).await.unwrap();
        }
        exchange.send(body(b"", false)).await.unwrap();
        assert_eq!(disconnected.await.unwrap(), ExchangeEvent::Disconnected);
        Ok(())
    });
    let (out, end) = exchange(&server, &get("/stream")).await;
    let response = parse(&out);
    assert_eq!(response.header("transfer-encoding"), Some("chunked"));
    assert_eq!(response.body, b"chunk 0\nchunk 1\nchunk 2\n");
    assert!(response.trailers.is_empty() && end.finished() && end.reset().is_none());
}

#[tokio::test]
async fn request_bodies_stream_in_by_length_or_chunks() {
    let server = echo_size();
    let large = vec![7u8; 4 << 20];
    let (out, _) = exchange(&server, &post("/echo", &large)).await;
    assert_eq!(parse(&out).body, (4 << 20).to_string().as_bytes());

    let chunked = b"POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n\
        3;ext=1\r\nabc\r\n2\r\nde\r\n0\r\nx-other: 1\r\n\r\n";
    let (out, _) = exchange(&server, chunked).await;
    assert_eq!(parse(&out).body, b"5");
}

#[tokio::test]
async fn a_body_the_node_reports_cut_short_is_a_disconnect() {
    for completion in ["truncated", "peer_gone"] {
        let raw = format!(
            "POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n\
             3\r\nabc\r\n0\r\nX-LK-Completion: {completion}\r\nx-lk-error: gone\r\n\r\n"
        );
        let (out, _) = exchange(&echo_size(), raw.as_bytes()).await;
        assert_eq!(parse(&out).body, b"-", "{completion}");
    }
}

#[tokio::test]
async fn bad_framing_is_a_protocol_reset_before_the_application_runs() {
    let called = Arc::new(AtomicBool::new(false));
    let seen = called.clone();
    let server = default_server(move |_| {
        seen.store(true, Ordering::SeqCst);
        async { Ok(()) }
    });
    let cases: [&[u8]; 8] = [
        b"this is not http\r\n\r\n",
        b"GET / HTTP/1.1\r\n\r\n",
        b"GET http://x/ HTTP/1.1\r\nHost: x\r\n\r\n",
        b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n",
        b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\n",
        b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n",
        b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip, chunked\r\n\r\n",
        b"GET / HTTP/1.1\r\nHost: x\r\n",
    ];
    for raw in cases {
        let (out, end) = exchange(&server, raw).await;
        assert!(out.is_empty(), "{:?}", String::from_utf8_lossy(raw));
        assert_eq!(end.reset(), Some(Code::HsrProtocol), "{:?}", String::from_utf8_lossy(raw));
    }
    let mut huge = b"GET / HTTP/1.1\r\nHost: x\r\nX: ".to_vec();
    huge.extend(std::iter::repeat_n(b'x', 1 << 20));
    huge.extend_from_slice(b"\r\n\r\n");
    let (_, end) = exchange(&server, &huge).await;
    assert_eq!(end.reset(), Some(Code::HsrProtocol));
    assert!(!called.load(Ordering::SeqCst));
}

#[tokio::test]
async fn oversized_chunk_metadata_is_a_protocol_reset() {
    let mut size_line = b"1;".to_vec();
    size_line.extend(std::iter::repeat_n(b'x', 1 << 20));
    let mut trailer = b"0\r\nX: ".to_vec();
    trailer.extend(std::iter::repeat_n(b'x', 1 << 20));
    for tail in [size_line, trailer, b"5\r\nabc".to_vec()] {
        let mut raw = b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        raw.extend_from_slice(&tail);
        let (out, end) = exchange(&echo_size(), &raw).await;
        assert!(out.is_empty());
        assert_eq!(end.reset(), Some(Code::HsrProtocol));
    }
}

#[tokio::test]
async fn a_failure_before_the_head_is_a_reset_and_after_it_a_completion_trailer() {
    let server = default_server(|exchange| async move {
        match exchange.head().path.as_str() {
            "/early" => Err(HostError::new("before any head")),
            "/silent" => Ok(()),
            _ => {
                exchange.send(start(200, &[])).await.unwrap();
                exchange.send(body(b"first chunk\n", true)).await.unwrap();
                Err(HostError::new("RuntimeError: the source died\r\nx-injected: 1"))
            }
        }
    });
    for target in ["/early", "/silent"] {
        let (out, end) = exchange(&server, &get(target)).await;
        assert!(out.is_empty() && end.reset() == Some(Code::HsrInternal), "{target}");
    }
    let (out, end) = exchange(&server, &get("/late")).await;
    let response = parse(&out);
    assert_eq!((response.status, response.body.as_slice()), (200, &b"first chunk\n"[..]));
    assert_eq!(response.trailer("x-lk-completion"), Some("internal"));
    assert!(response.trailer("x-lk-error").unwrap().contains("the source died"));
    assert!(response.trailer("x-injected").is_none(), "control characters are neutralized");
    assert!(end.finished() && end.reset().is_none());
}

#[tokio::test]
async fn a_length_framed_failure_cannot_carry_trailers_and_aborts() {
    let server = default_server(|exchange| async move {
        exchange.send(start(200, &[("content-length", "10")])).await.unwrap();
        exchange.send(body(b"short", true)).await.unwrap();
        assert!(exchange.send(body(b"", false)).await.is_err(), "fewer bytes than declared");
        Err(HostError::new("short body"))
    });
    let (_, end) = exchange(&server, &get("/")).await;
    assert_eq!(end.reset(), Some(Code::HsrAbort));
}

#[tokio::test]
async fn the_deadline_bounds_the_response_head_only() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let server = default_server(move |exchange| {
        let flag = flag.clone();
        async move {
            struct Dropped(Arc<AtomicBool>);
            impl Drop for Dropped {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _guard = Dropped(flag);
            if exchange.head().path == "/slow" {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            exchange.send(start(200, &[])).await.unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
            exchange.send(body(b"late body", false)).await.unwrap();
            Ok(())
        }
    });
    let deadline = EndpointRequest { timeout_ms: 10, ..preamble() };
    let (out, end) = exchange_with(&server, &get("/slow"), deadline.clone()).await;
    assert!(out.is_empty() && end.reset() == Some(Code::HsrTimeout));
    assert!(dropped.load(Ordering::SeqCst), "a missed deadline cancels the application");

    let (out, end) = exchange_with(&server, &get("/fast"), deadline).await;
    assert_eq!(parse(&out).body, b"late body");
    assert!(end.reset().is_none());
}

#[tokio::test]
async fn head_requests_and_bodiless_statuses_send_no_body() {
    let server = default_server(|exchange| async move {
        let status = if exchange.head().path == "/none" { 204 } else { 200 };
        exchange.send(start(status, &[("content-length", "5")])).await.unwrap();
        exchange.send(body(b"hello", false)).await.unwrap();
        Ok(())
    });
    let (out, _) = exchange(&server, b"HEAD / HTTP/1.1\r\nHost: x\r\n\r\n").await;
    let response = parse(&out);
    assert_eq!((response.header("content-length"), response.body.len()), (Some("5"), 0));
    let (out, _) = exchange(&server, &get("/none")).await;
    assert_eq!((parse(&out).status, parse(&out).body.len()), (204, 0));
}

#[tokio::test]
async fn an_http_10_response_is_delimited_by_the_end_of_the_stream() {
    let server = default_server(|exchange| async move {
        exchange.send(start(200, &[])).await.unwrap();
        exchange.send(body(b"whole", false)).await.unwrap();
        Ok(())
    });
    let (out, end) = exchange(&server, b"GET / HTTP/1.0\r\n\r\n").await;
    let response = parse(&out);
    assert!(response.header("transfer-encoding").is_none());
    assert_eq!(response.body, b"whole");
    assert!(end.finished());
}

#[tokio::test]
async fn expect_continue_is_answered_when_the_body_is_read() {
    let (out, _) = exchange(
        &echo_size(),
        b"POST / HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi",
    )
    .await;
    let rest = out.strip_prefix(b"HTTP/1.1 100 Continue\r\n\r\n".as_slice()).unwrap();
    assert_eq!(parse(rest).body, b"2");
}

#[tokio::test]
async fn invalid_response_heads_are_the_applications_error() {
    let server = default_server(|exchange| async move {
        for bad in
            [start(99, &[]), start(200, &[("bad name", "x")]), start(200, &[("x", "a\r\nb")])]
        {
            assert!(exchange.send(bad).await.is_err());
        }
        assert!(exchange.send(body(b"", false)).await.is_err(), "a body before its start");
        exchange.send(start(200, &[("content-length", "0")])).await.unwrap();
        exchange.send(body(b"", false)).await.unwrap();
        assert!(exchange.send(body(b"", false)).await.is_err(), "the response is complete");
        Ok(())
    });
    let (out, end) = exchange(&server, &get("/")).await;
    assert_eq!(parse(&out).status, 200);
    assert!(end.reset().is_none());
}

#[tokio::test]
async fn admission_is_bounded_and_released_when_an_exchange_is_dropped() {
    let limits = HttpServerLimits { max_connections: 1, ..Default::default() };
    let server = server(limits, |_| std::future::pending());
    let (stalled, _client, _) = wire();
    let task = tokio::spawn({
        let server = server.clone();
        async move { server.serve(stalled, Origin::Node(preamble())).await }
    });
    while server.connections.open() == 0 {
        tokio::task::yield_now().await;
    }
    let (_, end) = exchange(&server, &get("/")).await;
    assert_eq!(end.reset(), Some(Code::HsrRefused));
    task.abort();
    let _ = task.await;
    assert_eq!(server.connections.open(), 0);
}

/// Accept, echo text and reverse binary, and report the close code.
fn echo_websocket(closed: Arc<parking_lot::Mutex<Option<u16>>>) -> Arc<EndpointServer> {
    default_server(move |exchange| {
        let closed = closed.clone();
        async move {
            assert_eq!(exchange.head().kind, ExchangeKind::Websocket);
            assert_eq!(exchange.head().subprotocols, ["chat", "v2"]);
            let accept =
                ExchangeMessage::Accept { subprotocol: Some("chat".into()), headers: vec![] };
            exchange.send(accept).await.unwrap();
            loop {
                let reply = match exchange.receive().await.unwrap() {
                    ExchangeEvent::Text { text } => {
                        ExchangeMessage::Text { text: format!("echo:{text}") }
                    }
                    ExchangeEvent::Binary { mut data } => {
                        data.reverse();
                        ExchangeMessage::Binary { data }
                    }
                    ExchangeEvent::Closed { code } => {
                        *closed.lock() = Some(code);
                        return Ok(());
                    }
                    other => panic!("unexpected {other:?}"),
                };
                exchange.send(reply).await.unwrap();
            }
        }
    })
}

fn upgrade() -> tokio_tungstenite::tungstenite::handshake::client::Request {
    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(
            "ws://agents.test/ws",
        )
        .unwrap();
    request.headers_mut().insert("sec-websocket-protocol", "chat, v2".parse().unwrap());
    request
}

#[tokio::test]
async fn a_websocket_upgrade_exchanges_frames_and_the_peers_close() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed.clone());
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, response) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    assert_eq!(response.status(), 101);
    assert_eq!(response.headers()["sec-websocket-protocol"], "chat");

    socket.send(Message::text("hi")).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::text("echo:hi"));
    socket.send(Message::binary(b"abc".to_vec())).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::binary(b"cba".to_vec()));
    socket.send(Message::Ping(b"p".to_vec().into())).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::Pong(b"p".to_vec().into()));

    socket.close(None).await.unwrap();
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Status)));
    assert!(end.finished() && end.reset().is_none());
    assert_eq!(server.connections.open(), 0);
}

#[tokio::test]
async fn drain_closes_open_websockets_with_going_away() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed.clone());
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    let draining = tokio::spawn({
        let server = server.clone();
        async move { server.drain().await }
    });
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!(frame.code, CloseCode::Away);
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), draining).await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Away)));
    assert!(end.finished());
    assert!(server.websockets.lock().open.is_empty());
}

pub(super) const HANDSHAKE: &[u8] = b"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n\
    Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
    Sec-WebSocket-Version: 13\r\n\r\n";

/// The close code of the last close frame the server sent after its 101.
pub(super) fn close_code(out: &[u8]) -> u16 {
    let head = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    assert!(out.starts_with(b"HTTP/1.1 101"));
    assert!(String::from_utf8_lossy(&out[..head]).contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    let frame = &out[head..];
    assert_eq!(frame[0], 0x88, "a close frame");
    u16::from_be_bytes([frame[2], frame[3]])
}

#[tokio::test]
async fn websocket_limits_close_with_message_too_big_and_release_payloads() {
    // A declared 9-byte frame exceeds the limit before any payload arrives; two messages
    // overflow a one-message inbox the application does not drain.
    let oversized: &[u8] = b"\x82\x89\x00\x00\x00\x00";
    let flood: &[u8] = b"\x82\x81\x00\x00\x00\x00a\x82\x81\x00\x00\x00\x00a";
    for frames in [oversized, flood] {
        let limits = HttpServerLimits {
            websocket_inbox: 1,
            websocket_frame_bytes: 8,
            close_flush_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let server = server(limits, |exchange| async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            while !matches!(exchange.receive().await.unwrap(), ExchangeEvent::Closed { .. }) {}
            Ok(())
        });
        let mut raw = HANDSHAKE.to_vec();
        raw.extend_from_slice(frames);
        let (out, end) = exchange(&server, &raw).await;
        assert_eq!(close_code(&out), u16::from(CloseCode::Size));
        assert!(end.finished());
        assert_eq!(server.connections.open(), 0);
    }
}

#[tokio::test]
async fn an_unanswered_or_refused_upgrade_is_an_http_response() {
    let server = default_server(|exchange| async move {
        if exchange.head().path == "/ws" {
            let close = ExchangeMessage::Close { code: 1000, reason: String::new() };
            exchange.send(close).await.unwrap();
        } else {
            exchange.send(start(401, &[("content-length", "4")])).await.unwrap();
            exchange.send(body(b"nope", false)).await.unwrap();
        }
        Ok(())
    });
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert_eq!(parse(&out).status, 403);
    assert!(end.finished() && end.reset().is_none());

    let denied = String::from_utf8_lossy(HANDSHAKE).replace("/ws", "/denied");
    let (out, end) = exchange(&server, denied.as_bytes()).await;
    let response = parse(&out);
    assert_eq!((response.status, response.body.as_slice()), (401, &b"nope"[..]));
    assert!(end.finished() && end.reset().is_none());
}

fn listener() -> Origin {
    use livekit_protocol::agent_http::{AgentEndpoint, AgentEndpointKind};
    let routes = crate::server::PublicRoutes {
        prefix: "/agents/clinic/_".into(),
        routes: vec![AgentEndpoint {
            path: "/items/{id}".into(),
            methods: vec!["GET".into()],
            kind: AgentEndpointKind::AekHttp as i32,
            public: true,
        }],
        unauthenticated: true,
        proxy_headers: false,
    };
    Origin::Listener { routes: Arc::new(routes), peer: "203.0.113.1".into() }
}

async fn listener_exchange(server: &EndpointServer, raw: &[u8], close: bool) -> Vec<u8> {
    let (wire, mut client, end) = wire();
    client.write_all(raw).await.unwrap();
    if close {
        client.shutdown().await.unwrap();
    }
    let read = async {
        let mut out = Vec::new();
        client.read_to_end(&mut out).await.unwrap();
        out
    };
    let serve = server.serve(wire, listener());
    let ((), out) =
        tokio::time::timeout(Duration::from_secs(5), futures_util::future::join(serve, read))
            .await
            .expect("the exchange settles");
    assert!(end.finished() && end.reset().is_none());
    out
}

#[tokio::test]
async fn the_listener_routes_parsed_heads_and_answers_what_it_refuses() {
    let limits = HttpServerLimits { head_timeout: Duration::from_millis(50), ..Default::default() };
    let server = server(limits, |exchange| async move {
        assert_eq!(exchange.head().path, "/items/a b");
        assert_eq!(exchange.head().raw_path, b"/items/a%20b");
        exchange.send(start(200, &[("content-length", "2")])).await.unwrap();
        exchange.send(body(b"ok", false)).await.unwrap();
        Ok(())
    });
    let out = listener_exchange(&server, &get("/agents/clinic/_/items/a%20b?q=1"), true).await;
    assert_eq!((parse(&out).status, parse(&out).body.as_slice()), (200, &b"ok"[..]));
    let status = |out: Vec<u8>| parse(&out).status;
    assert_eq!(
        status(listener_exchange(&server, &get("/agents/other/_/items/1"), true).await),
        404
    );
    assert_eq!(status(listener_exchange(&server, &get("/agents/clinic/_/nope"), true).await), 404);
    let delete = b"DELETE /agents/clinic/_/items/1 HTTP/1.1\r\nHost: x\r\n\r\n";
    assert_eq!(status(listener_exchange(&server, delete, true).await), 405);
    assert_eq!(status(listener_exchange(&server, b"not http\r\n\r\n", true).await), 400);
    let mut huge = b"GET /agents/clinic/_/items/1 HTTP/1.1\r\nHost: x\r\nX: ".to_vec();
    huge.extend(std::iter::repeat_n(b'x', 70 << 10));
    assert_eq!(status(listener_exchange(&server, &huge, true).await), 431);
    let partial = b"GET /agents/clinic/_/items/1 HTTP/1.1\r\nHost: x\r\n";
    assert_eq!(status(listener_exchange(&server, partial, false).await), 408);
    assert_eq!(server.connections.open(), 0);
}

#[tokio::test]
async fn an_unanswered_upgrade_is_reset_and_its_application_cancelled() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let limits = HttpServerLimits { head_timeout: Duration::from_millis(50), ..Default::default() };
    let server = server(limits, move |_| {
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let guard = Dropped(flag.clone());
        async move {
            let _guard = guard;
            std::future::pending().await
        }
    });
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrTimeout));
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(server.connections.open(), 0);
    assert!(server.websockets.lock().open.is_empty());
}

fn quick_idle() -> HttpServerLimits {
    HttpServerLimits {
        idle_timeout: Duration::from_millis(50),
        close_flush_timeout: Duration::from_millis(50),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_stalled_request_body_is_a_disconnect() {
    let server = server(quick_idle(), |exchange| async move {
        assert_eq!(read_body(&exchange).await, None);
        Ok(())
    });
    let (wire, mut client, end) = wire();
    client.write_all(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nab").await.unwrap();
    // The client stays connected and sends nothing more.
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("the stall is noticed");
    assert_eq!(end.reset(), Some(Code::HsrProtocol));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}

#[tokio::test]
async fn a_client_that_stops_reading_fails_the_write_instead_of_holding_it() {
    let failed = Arc::new(AtomicBool::new(false));
    let flag = failed.clone();
    let server = server(quick_idle(), move |exchange| {
        let flag = flag.clone();
        async move {
            exchange.send(start(200, &[])).await.unwrap();
            let sent = exchange.send(body(&vec![0u8; 1 << 20], false)).await;
            flag.store(sent.is_err(), Ordering::SeqCst);
            Err(HostError::new("the client stopped reading"))
        }
    });
    let (wire, mut client, end) = wire_sized(1024);
    client.write_all(&get("/")).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("the stall is noticed");
    assert!(failed.load(Ordering::SeqCst));
    assert_eq!(end.reset(), Some(Code::HsrAbort));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}

#[tokio::test]
async fn a_websocket_peer_that_stops_reading_is_dropped() {
    let failed = Arc::new(AtomicBool::new(false));
    let flag = failed.clone();
    let server = server(quick_idle(), move |exchange| {
        let flag = flag.clone();
        async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            loop {
                let message = ExchangeMessage::Binary { data: vec![0u8; 64 << 10] };
                if exchange.send(message).await.is_err() {
                    flag.store(true, Ordering::SeqCst);
                    return Ok(());
                }
            }
        }
    });
    let (wire, mut client, end) = wire_sized(1024);
    client.write_all(HANDSHAKE).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("the stall is noticed");
    assert!(failed.load(Ordering::SeqCst));
    assert_eq!(end.reset(), Some(Code::HsrAbort));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}

#[tokio::test]
async fn upgrades_after_drain_are_refused() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed);
    server.drain().await;
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrRefused));
    let (out, _) = exchange(&echo_size(), &get("/")).await;
    assert_eq!(parse(&out).status, 200, "plain requests still run while draining");
}

#[tokio::test]
async fn close_codes_are_validated_and_reasons_fit_a_close_frame() {
    let server = default_server(|exchange| async move {
        exchange
            .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
            .await
            .unwrap();
        for code in [1005, 1006, 1015, 999, 5000] {
            let close = ExchangeMessage::Close { code, reason: String::new() };
            assert!(exchange.send(close).await.is_err(), "{code} is refused");
        }
        let reason = "\u{e9}".repeat(100);
        exchange.send(ExchangeMessage::Close { code: 4001, reason }).await.unwrap();
        Ok(())
    });
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let request = tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(
        "ws://agents.test/ws",
    )
    .unwrap();
    let (mut socket, _) = tokio_tungstenite::client_async(request, client).await.unwrap();
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!(u16::from(frame.code), 4001);
    assert_eq!(frame.reason.len(), 122, "123 bytes, cut at a character boundary");
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert!(end.finished());
}

/// Stream chunks until cancelled, recording that the application was dropped.
fn endless(dropped: Arc<AtomicBool>) -> Arc<EndpointServer> {
    default_server(move |exchange| {
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let guard = Dropped(dropped.clone());
        async move {
            let _guard = guard;
            if exchange.head().path == "/silent" {
                std::future::pending::<()>().await;
            }
            exchange.send(start(200, &[])).await.unwrap();
            loop {
                exchange.send(body(b"tick\n", true)).await.unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
    })
}

#[tokio::test]
async fn stopping_ends_a_streaming_response_with_a_canceled_completion() {
    let dropped = Arc::new(AtomicBool::new(false));
    let server = endless(dropped.clone());
    let stopping = tokio::spawn({
        let server = server.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            server.stop().await;
        }
    });
    let (out, end) = exchange(&server, &get("/stream")).await;
    stopping.await.unwrap();
    let response = parse(&out);
    assert!(response.body.starts_with(b"tick\n"));
    assert_eq!(response.trailer("x-lk-completion"), Some("canceled"));
    assert_eq!(response.trailer("x-lk-error"), Some("worker stopped"));
    assert!(end.finished() && end.reset().is_none());
    assert!(dropped.load(Ordering::SeqCst), "the application is cancelled");
    assert_eq!(server.connections.open(), 0);
}

#[tokio::test]
async fn stopping_before_the_response_head_is_an_abort() {
    let dropped = Arc::new(AtomicBool::new(false));
    let server = endless(dropped.clone());
    let stopping = tokio::spawn({
        let server = server.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            server.stop().await;
        }
    });
    let (out, end) = exchange(&server, &get("/silent")).await;
    stopping.await.unwrap();
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrAbort), "no claim about whether it was applied");
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn drain_lets_exchanges_in_flight_finish_and_refuses_new_ones() {
    let server = default_server(|exchange| async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        exchange.send(start(200, &[("content-length", "4")])).await.unwrap();
        exchange.send(body(b"done", false)).await.unwrap();
        Ok(())
    });
    let in_flight = tokio::spawn({
        let server = server.clone();
        async move { exchange(&server, &get("/")).await }
    });
    while server.connections.open() == 0 {
        tokio::task::yield_now().await;
    }
    let draining = tokio::spawn({
        let server = server.clone();
        async move { server.drain().await }
    });
    tokio::task::yield_now().await;
    let (out, end) = exchange(&server, &get("/")).await;
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrRefused), "a node retries it on another worker");
    let out = listener_exchange(&server, &get("/agents/clinic/_/items/1"), true).await;
    assert_eq!(parse(&out).status, 503);
    assert!(!draining.is_finished(), "drain waits for the exchange in flight");
    let (out, _) = in_flight.await.unwrap();
    assert_eq!(parse(&out).body, b"done");
    tokio::time::timeout(Duration::from_secs(5), draining).await.unwrap().unwrap();
}

#[tokio::test]
async fn stopping_closes_open_websockets_with_going_away() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed);
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    let stopping = tokio::spawn({
        let server = server.clone();
        async move { server.stop().await }
    });
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!((frame.code, frame.reason.as_str()), (CloseCode::Away, "worker stopped"));
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    stopping.await.unwrap();
    assert!(end.finished());
}

fn quick_pings() -> HttpServerLimits {
    HttpServerLimits {
        websocket_ping_interval: Duration::from_millis(20),
        websocket_ping_timeout: Duration::from_millis(20),
        close_flush_timeout: Duration::from_millis(50),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_quiet_peer_is_pinged_and_one_that_answers_stays_connected() {
    let server = server(quick_pings(), |exchange| async move {
        let accept = ExchangeMessage::Accept { subprotocol: Some("chat".into()), headers: vec![] };
        exchange.send(accept).await.unwrap();
        loop {
            match exchange.receive().await.unwrap() {
                ExchangeEvent::Text { text } => {
                    exchange.send(ExchangeMessage::Text { text }).await.unwrap();
                }
                ExchangeEvent::Closed { .. } => return Ok(()),
                _ => {}
            }
        }
    });
    let (wire, client, _) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    // Reading answers each ping with a pong, as every WebSocket client does.
    let mut pings = 0;
    let quiet = tokio::time::sleep(Duration::from_millis(200));
    tokio::pin!(quiet);
    loop {
        tokio::select! {
            biased;
            () = &mut quiet => break,
            message = socket.next() => match message.unwrap().unwrap() {
                Message::Ping(_) => pings += 1,
                other => panic!("unexpected {other:?}"),
            },
        }
    }
    assert!(pings >= 3, "a quiet connection is pinged: {pings}");
    socket.send(Message::text("still here")).await.unwrap();
    loop {
        match socket.next().await.unwrap().unwrap() {
            Message::Text(text) => break assert_eq!(text.as_str(), "still here"),
            Message::Ping(_) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    socket.close(None).await.unwrap();
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
}

#[tokio::test]
async fn a_peer_that_answers_no_ping_is_gone() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let code = closed.clone();
    let server = server(quick_pings(), move |exchange| {
        let code = code.clone();
        async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            loop {
                if let ExchangeEvent::Closed { code: closed } = exchange.receive().await.unwrap() {
                    *code.lock() = Some(closed);
                    return Ok(());
                }
            }
        }
    });
    // The client sends its handshake, then never reads or answers again.
    let (wire, mut client, end) = wire();
    client.write_all(HANDSHAKE).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("an unresponsive peer is noticed");
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Abnormal)));
    assert_eq!(end.reset(), Some(Code::HsrAbort));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}
