use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::{ApiErrorKind, Timeout};
use lk_agents_abi::{ChatContext, Clock, Closed};
use lk_agents_inference::llm::gateway::{GatewayLlm, LlmChunks};
use lk_agents_inference::llm::machine::Policy;
use lk_agents_inference::llm::{Auth, LlmOptions};
use lk_agents_net::clock::RealClock;
use lk_agents_net::http::GatewayHttp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Clone)]
enum Part {
    Bytes(&'static [u8]),
    Pause(u64),
    /// Stop writing and wait for the client to go away.
    Hang,
}

#[derive(Clone)]
enum Reply {
    Sse(Vec<Part>),
    /// Hang before the response head.
    Silent,
    Status(u16, &'static str),
    StatusHeaders(u16, &'static str, &'static str),
}

struct Peer {
    url: String,
    requests: Arc<AtomicUsize>,
    /// Each request's `X-LiveKit-Session-ID`, in arrival order.
    sessions: Arc<parking_lot::Mutex<Vec<String>>>,
    /// Connections the client closed while the peer was still serving them.
    released: Arc<AtomicUsize>,
}

impl Peer {
    async fn start(replies: Vec<Reply>) -> Peer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let replies = Arc::new(Mutex::new(replies.into_iter()));
        let (requests, released) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let sessions = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (req, rel, ses) = (requests.clone(), released.clone(), sessions.clone());
        tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let (replies, req, rel) = (replies.clone(), req.clone(), rel.clone());
                let ses = ses.clone();
                tokio::spawn(async move { serve(socket, replies, req, rel, ses).await });
            }
        });
        Peer { url, requests, sessions, released }
    }
}

async fn serve(
    mut socket: TcpStream,
    replies: Arc<Mutex<std::vec::IntoIter<Reply>>>,
    requests: Arc<AtomicUsize>,
    released: Arc<AtomicUsize>,
    sessions: Arc<parking_lot::Mutex<Vec<String>>>,
) {
    // Requests on a kept-alive connection are served in turn.
    loop {
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        while !request.ends_with(b"\r\n0\r\n\r\n") {
            match socket.read(&mut buf).await {
                Ok(0) | Err(_) => return,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        if request.starts_with(b"GET") {
            continue;
        }
        requests.fetch_add(1, Ordering::SeqCst);
        let head = String::from_utf8_lossy(&request).into_owned();
        let session = head.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("x-livekit-session-id").then(|| value.trim().to_owned())
        });
        sessions.lock().extend(session);
        let Some(reply) = replies.lock().await.next() else { return };
        let hang = |mut socket: TcpStream, released: Arc<AtomicUsize>| async move {
            let mut buf = [0u8; 64];
            while let Ok(n) = socket.read(&mut buf).await {
                if n == 0 {
                    break;
                }
            }
            released.fetch_add(1, Ordering::SeqCst);
        };
        match reply {
            Reply::Silent => return hang(socket, released).await,
            Reply::Status(status, body) | Reply::StatusHeaders(status, body, _) => {
                let headers = match reply {
                    Reply::StatusHeaders(_, _, headers) => headers,
                    _ => "",
                };
                let head = format!(
                    "HTTP/1.1 {status} Nope\r\ncontent-length: {}\r\n{headers}\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            }
            Reply::Sse(parts) => {
                let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nX-LiveKit-Request-Id: stream-id\r\ntransfer-encoding: chunked\r\n\r\n";
                let _ = socket.write_all(head.as_bytes()).await;
                for part in parts {
                    match part {
                        Part::Bytes(bytes) => {
                            let chunk =
                                [format!("{:x}\r\n", bytes.len()).as_bytes(), bytes, b"\r\n"]
                                    .concat();
                            let _ = socket.write_all(&chunk).await;
                            let _ = socket.flush().await;
                        }
                        Part::Pause(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
                        Part::Hang => return hang(socket, released).await,
                    }
                }
                let _ = socket.write_all(b"0\r\n\r\n").await;
            }
        }
    }
}

fn policy(max_retry: u32, read_ms: u64) -> Policy {
    let limit = Some(Duration::from_secs(5));
    Policy {
        connect_timeout: limit,
        write_timeout: limit,
        read_timeout: Some(Duration::from_millis(read_ms)),
        total_timeout: None,
        max_retry,
        retry_interval: Duration::from_millis(50),
    }
}

fn llm(peer: &Peer) -> GatewayLlm {
    let opts = LlmOptions {
        model: "openai/gpt-4o".into(),
        base_url: peer.url.clone(),
        auth: Auth::Gateway { api_key: "key".into(), api_secret: "secret".into() },
        provider: None,
        inference_class: None,
        strict_tools: true,
        body: Default::default(),
        label: "llm".into(),
        extra_kwargs: Default::default(),
    };
    let http = Arc::new(GatewayHttp::new().unwrap());
    GatewayLlm::new(opts, http, tokio::runtime::Handle::current(), Arc::new(RealClock::new()))
}

fn open(llm: &GatewayLlm, policy: Policy) -> LlmChunks {
    llm.chat(&ChatContext::new(Vec::new()), &[], Default::default(), policy, None).unwrap()
}

async fn text(chunks: &mut LlmChunks) -> Result<String, lk_agents_abi::errors::ApiError> {
    let mut text = String::new();
    while let Some(chunk) = chunks.next().await? {
        if let Some(content) = chunk.delta.and_then(|d| d.content) {
            text.push_str(&content);
        }
    }
    Ok(text)
}

fn deadline(ms: u64) -> lk_agents_abi::Ts {
    RealClock::new().now().saturating_add(Duration::from_millis(ms))
}

async fn eventually(what: &str, check: impl Fn() -> bool) {
    for _ in 0..200 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{what}");
}

// Catches: framing by network chunk: a character, a CRLF or a field split across chunks, a
// chunk boundary inside `[DONE]`, or events delivered before their blank line.
#[tokio::test(flavor = "multi_thread")]
async fn a_completion_cut_at_hostile_boundaries_streams_whole() {
    let peer = Peer::start(vec![Reply::Sse(vec![
        Part::Bytes(b": ping\r"),
        Part::Bytes(b"\ndata: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"Hel"),
        Part::Pause(20),
        Part::Bytes(b"l\xc3"),
        Part::Bytes(b"\xb3\"}}]}\r\n\r"),
        Part::Bytes(
            b"\ndata: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\" ok\"}}]}\n\nda",
        ),
        Part::Bytes(b"ta: [DO"),
        Part::Bytes(b"NE]\n\n"),
    ])])
    .await;
    let llm = llm(&peer);
    let mut chunks = open(&llm, policy(0, 2_000));
    assert_eq!(text(&mut chunks).await.unwrap(), "Hell\u{f3} ok");
    assert_eq!(chunks.close(deadline(1_000)).await, Closed::Clean);
}

// Catches: the retry loop living anywhere but the request core: a 503 not retried, retried
// without the response being read, or retried more than `max_retry` times.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_attempt_is_retried_and_the_retry_answers() {
    let ok = Reply::Sse(vec![Part::Bytes(
        b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
    )]);
    let peer =
        Peer::start(vec![Reply::Status(503, "{\"error\":{\"message\":\"busy\"}}"), ok]).await;
    let llm = llm(&peer);
    let mut chunks = open(&llm, policy(1, 2_000));
    assert_eq!(text(&mut chunks).await.unwrap(), "hi");
    assert_eq!(peer.requests.load(Ordering::SeqCst), 2);

    let peer = Peer::start(vec![Reply::Status(401, "{\"error\":{\"message\":\"bad key\"}}")]).await;
    let llm = self::llm(&peer);
    let err = text(&mut open(&llm, policy(3, 2_000))).await.unwrap_err();
    assert_eq!(
        (err.kind, err.status_code, err.retryable),
        (ApiErrorKind::Status, Some(401), false)
    );
    assert!(err.message.contains("bad key"), "{}", err.message);
    assert_eq!(peer.requests.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_terminal_quota_rejection_is_not_retried_and_explains_the_credit_limit() {
    let peer = Peer::start(vec![
        Reply::StatusHeaders(429,
            r#"{"type":"inference_quota_exceeded","error":"Project out of credits","hint":"Add credits to continue."}"#,
            "X-Should-Retry: false\r\nX-LiveKit-Request-Id: quota-id\r\n"),
        Reply::Sse(vec![Part::Bytes(b"data: [DONE]\n\n")]),
    ]).await;
    let llm = llm(&peer);
    let error = text(&mut open(&llm, policy(3, 2_000))).await.unwrap_err();
    assert_eq!(
        (error.kind, error.status_code, error.retryable),
        (ApiErrorKind::Status, Some(429), false)
    );
    assert!(error.message.contains("Project out of credits"), "{error}");
    assert!(error.message.contains("Add credits to continue."), "{error}");
    assert_eq!(error.request_id.as_deref(), Some("quota-id"));
    assert_eq!(peer.requests.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_inband_gateway_error_surfaces_its_status_and_message() {
    for (frame, status, message) in [
        (b"data: {\"error\":{\"message\":\"Service Unavailable\",\"type\":\"server_error\",\"code\":503}}\n\n".as_slice(), 503, "Service Unavailable"),
        (b"data: {\"error\":{\"message\":\"Gateway Timeout\",\"type\":\"server_error\",\"code\":504}}\n\n".as_slice(), 504, "Gateway Timeout"),
    ] {
        let peer = Peer::start(vec![Reply::Sse(vec![
            Part::Bytes(b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"),
            Part::Bytes(frame),
        ])]).await;
        let llm = llm(&peer);
        let mut chunks = open(&llm, policy(3, 2_000));
        assert_eq!(chunks.next().await.unwrap().unwrap().delta.unwrap().content.as_deref(), Some("Hello"));
        let error = text(&mut chunks).await.unwrap_err();
        assert_eq!((error.kind, error.status_code, error.retryable), (ApiErrorKind::Status, Some(status), false));
        assert!(error.message.contains(message), "{error}");
        assert_eq!(error.request_id.as_deref(), Some("stream-id"));
        assert_eq!(peer.requests.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_gateway_rejection_uses_the_livekit_request_id() {
    let peer = Peer::start(vec![Reply::StatusHeaders(
        401,
        r#"{"error":{"message":"bad key"}}"#,
        "X-LiveKit-Request-Id: livekit-id\r\nX-Request-Id: unrelated-id\r\n",
    )])
    .await;
    let llm = llm(&peer);
    let error = text(&mut open(&llm, policy(0, 2_000))).await.unwrap_err();
    assert_eq!(error.request_id.as_deref(), Some("livekit-id"));
    assert!(error.message.contains("bad key"));
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_omit_the_unused_gateway_session_header() {
    let busy = Reply::Status(503, "{\"error\":{\"message\":\"busy\"}}");
    let ok = Reply::Sse(vec![Part::Bytes(
        b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
    )]);
    let peer = Peer::start(vec![busy.clone(), busy, ok]).await;
    let llm = llm(&peer);
    let job: lk_agents_inference::headers::Headers =
        [("X-LiveKit-Session-ID".to_owned(), "inference_job".to_owned())].into_iter().collect();
    let mut chunks = llm
        .chat(&ChatContext::new(Vec::new()), &[], Default::default(), policy(2, 2_000), Some(&job))
        .unwrap();
    assert_eq!(text(&mut chunks).await.unwrap(), "hi");
    let sessions = peer.sessions.lock().clone();
    assert!(sessions.is_empty(), "unused session header leaked: {sessions:?}");
    assert_eq!(peer.requests.load(Ordering::SeqCst), 3);
}

// Catches: a stalled stream waited on forever, a read timeout reported without naming its limit or as
// retryable after visible output, or the stalled connection left open.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_stalled_after_output_times_out_on_read_and_releases_its_connection() {
    let peer = Peer::start(vec![Reply::Sse(vec![
        Part::Bytes(b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"),
        Part::Hang,
    ])])
    .await;
    let llm = llm(&peer);
    let mut chunks = open(&llm, policy(3, 300));
    let err = text(&mut chunks).await.unwrap_err();
    assert_eq!(
        (err.kind, err.timeout, err.retryable),
        (ApiErrorKind::Timeout, Some(Timeout::Read), false)
    );
    assert_eq!(chunks.close(deadline(1_000)).await, Closed::Clean);
    eventually("the stalled connection was never released", || {
        peer.released.load(Ordering::SeqCst) == 1
    })
    .await;
    assert_eq!(peer.requests.load(Ordering::SeqCst), 1, "output was delivered: no replay");
}

// Catches: an undecodable chunk skipped (silently losing part of the answer) instead of failing.
#[tokio::test(flavor = "multi_thread")]
async fn a_malformed_stream_fails_the_request() {
    let peer = Peer::start(vec![Reply::Sse(vec![Part::Bytes(b"data: {not json\n\n")])]).await;
    let llm = llm(&peer);
    let err = text(&mut open(&llm, policy(0, 2_000))).await.unwrap_err();
    assert_eq!(err.kind, ApiErrorKind::Connection);
    assert!(err.message.contains("malformed"), "{}", err.message);
}

// Catches: a body cut before `[DONE]` read as a complete answer (the provider contract ends every
// completion with the sentinel).
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_ending_without_done_fails_the_request() {
    let peer = Peer::start(vec![Reply::Sse(vec![Part::Bytes(
        b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"hal\"}}]}\n\n",
    )])])
    .await;
    let llm = llm(&peer);
    let err = text(&mut open(&llm, policy(0, 2_000))).await.unwrap_err();
    assert_eq!(err.kind, ApiErrorKind::Connection);
    assert!(err.message.contains("[DONE]"), "{}", err.message);
}

// Catches: a consumer that stops waiting (its pull dropped) or closes the stream at any await
// point leaving the attempt's connection open, close reporting Clean before the transport is
// released, or a dropped pull losing what was already received.
#[tokio::test(flavor = "multi_thread")]
async fn closing_at_every_await_point_releases_the_attempt() {
    let first = b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n";
    let cases: Vec<(&str, Reply, usize)> = vec![
        ("waiting for the response head", Reply::Silent, 0),
        ("waiting for the first event", Reply::Sse(vec![Part::Hang]), 0),
        ("waiting between events", Reply::Sse(vec![Part::Bytes(first), Part::Hang]), 1),
    ];
    for (point, reply, delivered) in cases {
        let peer = Peer::start(vec![reply]).await;
        let llm = llm(&peer);
        let mut chunks = open(&llm, policy(0, 10_000));
        for _ in 0..delivered {
            assert!(chunks.next().await.unwrap().is_some(), "{point}");
        }
        // The consumer stops waiting: the pull is dropped mid-await.
        let pulled = tokio::time::timeout(Duration::from_millis(150), chunks.next()).await;
        assert!(pulled.is_err(), "{point}: the peer sent nothing more");
        assert_eq!(chunks.close(deadline(1_000)).await, Closed::Clean, "{point}");
        eventually(&format!("{point}: connection never released"), || {
            peer.released.load(Ordering::SeqCst) == 1
        })
        .await;
    }

    // During the retry backoff: no attempt is live, and none launches after close.
    let peer = Peer::start(vec![Reply::Status(503, "busy"), Reply::Status(503, "busy")]).await;
    let llm = llm(&peer);
    let mut chunks =
        open(&llm, Policy { retry_interval: Duration::from_secs(5), ..policy(3, 10_000) });
    let _ = tokio::time::timeout(Duration::from_millis(400), chunks.next()).await;
    assert_eq!(
        peer.requests.load(Ordering::SeqCst),
        2,
        "the 100 ms first retry ran, the next waits 5 s"
    );
    assert_eq!(chunks.close(deadline(1_000)).await, Closed::Clean);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(peer.requests.load(Ordering::SeqCst), 2, "nothing launches after close");
}

// Catches: `chat()` sending the request before anyone reads the stream (a stream opened and
// dropped must cost nothing).
#[tokio::test(flavor = "multi_thread")]
async fn nothing_is_sent_before_the_first_pull() {
    let peer = Peer::start(vec![Reply::Silent]).await;
    let llm = llm(&peer);
    let chunks = open(&llm, policy(0, 10_000));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(peer.requests.load(Ordering::SeqCst), 0);
    assert_eq!(chunks.close(deadline(1_000)).await, Closed::Clean);
}

// Catches: a consumer's own pause (downstream backpressure) counted as remote silence: the read
// limit fired on a stream that kept sending while nobody pulled.
#[tokio::test(flavor = "multi_thread")]
async fn a_consumer_pausing_longer_than_the_read_limit_is_not_a_timeout() {
    let event = b"data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
    let mut parts = Vec::new();
    for _ in 0..80 {
        parts.push(Part::Bytes(event));
        parts.push(Part::Pause(5));
    }
    parts.push(Part::Bytes(b"data: [DONE]\n\n"));
    let peer = Peer::start(vec![Reply::Sse(parts)]).await;
    let llm = llm(&peer);
    let mut chunks = open(&llm, policy(0, 200));
    assert!(chunks.next().await.unwrap().is_some());
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(text(&mut chunks).await.unwrap().len(), 79);
}

mod live {

    use std::sync::Arc;
    use std::time::Duration;

    use lk_agents_abi::{ChatContext, Role};
    use lk_agents_inference::llm::gateway::GatewayLlm;
    use lk_agents_inference::llm::machine::Policy;
    use lk_agents_inference::llm::{Auth, LlmOptions};
    use lk_agents_net::clock::RealClock;
    use lk_agents_net::http::GatewayHttp;

    fn llm(model: &str) -> Option<GatewayLlm> {
        let api_key = lk_agents_inference::credentials::resolve_api_key(
            None,
            std::env::var("LIVEKIT_INFERENCE_API_KEY").ok().as_deref(),
            std::env::var("LIVEKIT_API_KEY").ok().as_deref(),
        )
        .ok()?;
        let api_secret = lk_agents_inference::credentials::resolve_api_secret(
            None,
            std::env::var("LIVEKIT_INFERENCE_API_SECRET").ok().as_deref(),
            std::env::var("LIVEKIT_API_SECRET").ok().as_deref(),
        )
        .ok()?;
        let opts = LlmOptions {
            model: model.to_owned(),
            provider: None,
            base_url: lk_agents_inference::headers::default_inference_url(
                std::env::var("LIVEKIT_INFERENCE_URL").ok().as_deref(),
                std::env::var("LIVEKIT_URL").ok().as_deref(),
            ),
            auth: Auth::Gateway { api_key, api_secret },
            inference_class: None,
            strict_tools: true,
            body: Default::default(),
            label: "llm".into(),
            extra_kwargs: [("max_completion_tokens".to_owned(), serde_json::json!(16))]
                .into_iter()
                .collect(),
        };
        let http = Arc::new(GatewayHttp::new().unwrap());
        Some(GatewayLlm::new(
            opts,
            http,
            tokio::runtime::Handle::current(),
            Arc::new(RealClock::new()),
        ))
    }

    fn policy() -> Policy {
        let limit = Some(Duration::from_secs(20));
        Policy {
            connect_timeout: limit,
            write_timeout: limit,
            read_timeout: limit,
            total_timeout: None,
            max_retry: 0,
            retry_interval: Duration::ZERO,
        }
    }

    fn chat_ctx(text: &str) -> ChatContext {
        ChatContext::new(vec![
            lk_agents_abi::ChatMessage::text("m1", Role::User, text, std::time::SystemTime::now())
                .into(),
        ])
    }

    // Catches: a claim set or auth scheme the gateway rejects, a wrong path, `include_usage`
    // refused, or the framing losing a token.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs LiveKit Inference credentials and the network"]
    async fn a_streamed_completion_round_trips() {
        let Some(llm) = llm("openai/gpt-4o-mini") else {
            return eprintln!("skipped: no credentials");
        };
        let mut chunks = llm
            .chat(&chat_ctx("Say the single word: pong."), &[], Default::default(), policy(), None)
            .unwrap();
        let (mut text, mut usage) = (String::new(), None);
        while let Some(chunk) = chunks.next().await.expect("the gateway answered") {
            if let Some(content) = chunk.delta.as_ref().and_then(|d| d.content.as_ref()) {
                text.push_str(content);
            }
            usage = chunk.usage.or(usage);
        }
        assert!(!text.trim().is_empty(), "no text");
        let usage = usage.expect("include_usage produces a usage chunk");
        assert!(usage.input_tokens > 0 && usage.output_tokens > 0, "{usage:?}");
        eprintln!("live: {text:?} usage={usage:?}");
    }

    // Catches: a 4xx mapped to the wrong class, retried, or reported without the gateway's body.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "needs LiveKit Inference credentials and the network"]
    async fn an_unknown_model_is_a_non_retryable_status_error() {
        let Some(llm) = llm("openai/definitely-not-a-model") else {
            return eprintln!("skipped: no credentials");
        };
        let mut chunks =
            llm.chat(&chat_ctx("hi"), &[], Default::default(), policy(), None).unwrap();
        let err = chunks.next().await.expect_err("an unknown model fails");
        eprintln!("live 4xx: {err:?}");
        assert!(err.status_code.is_some_and(|s| (400..500).contains(&s)), "{err:?}");
        assert!(!err.retryable && !err.message.is_empty());
    }
}
