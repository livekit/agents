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
    async fn read(&self, _max: usize) -> Result<Vec<u8>, String> {
        let next = self.inbox.lock().unwrap().pop_front();
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
