use super::*;

/// The node's side of one stream, in memory.
#[derive(Default)]
struct Fake {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    inbox: std::sync::Mutex<VecDeque<Vec<u8>>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    out: std::sync::Mutex<Vec<u8>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    finished: AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    reset: std::sync::Mutex<Option<Reset>>,
}

impl Fake {
    fn with(request: &[u8]) -> Arc<Fake> {
        let f = Fake::default();
        f.inbox.lock().unwrap().extend([request.to_vec(), Vec::new()]);
        Arc::new(f)
    }

    fn out(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).into_owned()
    }
}

#[async_trait]
impl StreamIo for Fake {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        let next = {
            let mut inbox = self.inbox.lock().unwrap();
            inbox.pop_front().map(|mut chunk| {
                if chunk.len() > max {
                    let rest = chunk.split_off(max);
                    inbox.push_front(rest);
                }
                chunk
            })
        };
        match next {
            Some(c) => Ok(c),
            None => std::future::pending().await,
        }
    }
    async fn write(&self, data: &[u8]) -> Result<(), String> {
        self.out.lock().unwrap().extend_from_slice(data);
        Ok(())
    }
    async fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }
    async fn reset(&self, code: Reset) {
        *self.reset.lock().unwrap() = Some(code);
    }
    async fn stop(&self, _code: Reset) {}
}

/// Echoes the request body's length; `sleep` holds the head back.
struct Echo {
    sleep: Duration,
    fail_after_head: bool,
}

#[async_trait]
impl AsgiApp for Echo {
    #[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
    async fn call(&self, scope: Scope, conn: Conn) -> Result<(), AppError> {
        assert_eq!(scope.kind, "http");
        let mut n = 0;
        loop {
            match conn.receive().await {
                Received::Request { body, more_body } => {
                    n += body.len();
                    if !more_body {
                        break;
                    }
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        tokio::time::sleep(self.sleep).await;
        let start = Sent::ResponseStart { status: 200, headers: vec![] };
        conn.send(start).await.unwrap();
        let body = Sent::ResponseBody { body: n.to_string().into_bytes(), more_body: true };
        conn.send(body).await.unwrap();
        if self.fail_after_head {
            return Err(AppError {
                summary: "RuntimeError: the source\ndied".into(),
                traceback: String::new(),
            });
        }
        conn.send(Sent::ResponseBody { body: vec![], more_body: false }).await.unwrap();
        Ok(())
    }
}

const ECHO: Echo = Echo { sleep: Duration::ZERO, fail_after_head: false };

fn info(timeout_ms: u64) -> RequestInfo {
    RequestInfo { request_id: "AER_test".into(), timeout_ms, ..Default::default() }
}

// Catches: a chunked request body read as the raw chunk framing, or its last chunk lost.
#[tokio::test]
async fn a_chunked_request_body_reaches_the_app_unframed_and_the_reply_is_chunked() {
    let io = Fake::with(
        b"POST /x HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n3;x=1\r\nabc\r\n0\r\n\r\n",
    );
    serve(io.clone(), info(0), &ECHO, &Registry::default()).await;
    assert_eq!(io.out(), "HTTP/1.1 200 \r\ntransfer-encoding: chunked\r\n\r\n1\r\n8\r\n0\r\n\r\n");
    assert!(io.finished.load(Ordering::Acquire));
    assert_eq!(*io.reset.lock().unwrap(), None);
}

// Catches: a failure after the head reported as success (a clean `0\r\n\r\n`) or as a reset that
// races the body; and a reason with a newline breaking the trailer framing.
#[tokio::test]
async fn a_failure_after_the_head_is_a_sanitized_completion_trailer() {
    let io = Fake::with(b"GET /x HTTP/1.1\r\nHost: a\r\n\r\n");
    let app = Echo { sleep: Duration::ZERO, fail_after_head: true };
    serve(io.clone(), info(0), &app, &Registry::default()).await;
    assert!(
        io.out().ends_with(
            "0\r\nx-lk-completion: internal\r\nx-lk-error: RuntimeError: the source died\r\n\r\n"
        ),
        "{}",
        io.out()
    );
}

// Catches: junk accepted as a request, and a missed head deadline answered with bytes.
#[tokio::test(start_paused = true)]
async fn junk_is_a_protocol_reset_and_a_missed_deadline_a_timeout_reset() {
    let io = Fake::with(b"this is not http\r\n\r\n");
    serve(io.clone(), info(0), &ECHO, &Registry::default()).await;
    assert_eq!(*io.reset.lock().unwrap(), Some(Reset::HsrProtocol));
    assert!(io.out().is_empty());

    let io = Fake::with(b"GET /slow HTTP/1.1\r\nHost: a\r\n\r\n");
    let slow = Echo { sleep: Duration::from_secs(5), fail_after_head: false };
    serve(io.clone(), info(50), &slow, &Registry::default()).await;
    assert_eq!(*io.reset.lock().unwrap(), Some(Reset::HsrTimeout));
    assert!(io.out().is_empty());
}

// Catches: the path left percent-encoded, or the query kept in it.
#[test]
fn the_scope_path_is_unquoted_and_split_from_the_query() {
    let s = scope_base(&info(0), "http", vec![], b"/a%20b/%E2%82%AC?q=%20");
    assert_eq!(s.path, "/a b/\u{20ac}");
    assert_eq!(s.raw_path, b"/a%20b/%E2%82%AC");
    assert_eq!(s.query_string, b"q=%20");
}

struct RefuseInvalidBody;

#[async_trait]
impl AsgiApp for RefuseInvalidBody {
    async fn call(&self, _: Scope, conn: Conn) -> Result<(), AppError> {
        assert_eq!(conn.receive().await, Received::Disconnect);
        Ok(())
    }
}

#[tokio::test]
async fn endpoint_limit_vectors_reject_header_overflow() {
    let mut head = b"GET / HTTP/1.1\r\nHost: a\r\nX-Large: ".to_vec();
    head.extend(vec![b'x'; 1 << 20]);
    head.extend(b"\r\n\r\n");
    let io = Fake::with(&head);
    serve(io.clone(), info(0), &RefuseInvalidBody, &Registry::default()).await;
    assert_eq!(*io.reset.lock().unwrap(), Some(Reset::HsrProtocol));
    assert!(io.out().is_empty());
}

#[tokio::test]
async fn endpoint_limit_vectors_reject_chunk_line_and_trailer_overflow() {
    for prefix in [b"1;".as_slice(), b"0\r\nX-Large: ".as_slice()] {
        let mut request =
            b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        request.extend(prefix);
        request.extend(vec![b'x'; 1 << 20]);
        let io = Fake::with(&request);
        serve(io.clone(), info(0), &RefuseInvalidBody, &Registry::default()).await;
        assert_eq!(*io.reset.lock().unwrap(), Some(Reset::HsrProtocol));
    }
}

struct WaitForWebsocketLimit;

#[async_trait]
impl AsgiApp for WaitForWebsocketLimit {
    async fn call(&self, _: Scope, conn: Conn) -> Result<(), AppError> {
        assert_eq!(conn.receive().await, Received::WsConnect);
        conn.send(Sent::WsAccept { subprotocol: None, headers: vec![] }).await.unwrap();
        let mut messages = 0;
        loop {
            match conn.receive().await {
                Received::WsDisconnect { code } => {
                    assert_eq!(code, 1009);
                    assert!(messages <= 16);
                    return Ok(());
                }
                Received::WsBytes(_) => messages += 1,
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}

#[tokio::test]
async fn endpoint_limit_vectors_refuse_oversized_websocket_prefix_before_payload() {
    let mut request = b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n".to_vec();
    request.extend([0x82, 0xff]);
    request.extend(((1_u64 << 20) + 1).to_be_bytes());
    request.extend([0_u8; 4]);
    let io = Fake::with(&request);
    serve(io.clone(), info(0), &WaitForWebsocketLimit, &Registry::default()).await;
    assert!(io.finished.load(Ordering::Acquire));
}

#[tokio::test]
async fn endpoint_limit_vectors_refuse_websocket_message_and_inbox_overflow() {
    let upgrade = b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    let mut flood = upgrade.to_vec();
    for _ in 0..17 {
        flood.extend([0x82, 0x81, 0, 0, 0, 0, b'x']);
    }
    let mut fragmented = upgrade.to_vec();
    for n in 0..5 {
        fragmented.extend([
            if n == 0 {
                0x02
            } else if n == 4 {
                0x80
            } else {
                0x00
            },
            0xff,
        ]);
        fragmented.extend((1_u64 << 20).to_be_bytes());
        fragmented.extend([0_u8; 4]);
        fragmented.extend(vec![b'x'; 1 << 20]);
    }
    for request in [flood, fragmented] {
        let io = Fake::with(&request);
        serve(io.clone(), info(0), &WaitForWebsocketLimit, &Registry::default()).await;
        assert!(io.finished.load(Ordering::Acquire));
    }
}
