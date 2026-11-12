use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::{ApiErrorKind, Timeout};
use lk_agents_abi::{ChatContext, Clock, Closed};
use lk_agents_inference::llm::gateway::Policy;
use lk_agents_inference::llm::gateway::{GatewayLlm, LlmChunks};
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

#[derive(Default)]
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
    use lk_agents_inference::llm::gateway::Policy;
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

/// The request's decisions, stepped by hand at chosen instants. Its attempts are spawned but never
/// polled (the test never yields), so their reports never arrive.
mod stepped {
    use std::time::Duration;

    use super::super::{Event, LlmChunks, Phase};
    use crate::llm::parse::WireChunk;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, Timeout};
    use lk_agents_abi::{HOST_TEARDOWN_TIMEOUT, LlmChunk, Ts};
    use lk_agents_core::request::{AttemptFailure, Policy};

    fn at(ms: u64) -> Ts {
        Ts::from_nanos(ms * 1_000_000)
    }

    fn policy() -> Policy {
        Policy {
            connect_timeout: Some(Duration::from_secs(10)),
            write_timeout: Some(Duration::from_secs(10)),
            read_timeout: Some(Duration::from_secs(10)),
            total_timeout: None,
            max_retry: 3,
            retry_interval: Duration::from_secs(2),
        }
    }

    fn start(policy: Policy) -> LlmChunks {
        let peer = super::Peer { url: "http://127.0.0.1:9/v1".into(), ..Default::default() };
        let mut chunks = super::open(&super::llm(&peer), policy);
        chunks.accept(policy, at(0));
        chunks
    }

    fn launched(chunks: &LlmChunks) -> Vec<u32> {
        chunks.tasks.keys().copied().collect()
    }

    fn settled(chunks: &LlmChunks) -> bool {
        matches!(&chunks.phase, Phase::Accepted(a) if a.attempts.is_settled())
    }

    fn outcome(chunks: &mut LlmChunks) -> Option<Result<(), ApiError>> {
        chunks.outcome.take()
    }

    fn error(chunks: &mut LlmChunks) -> ApiError {
        outcome(chunks).expect("an outcome").unwrap_err()
    }

    fn delivered(chunks: &mut LlmChunks) -> Vec<LlmChunk> {
        chunks.ready.drain(..).collect()
    }

    fn text(content: &str) -> WireChunk {
        serde_json::from_value(
            serde_json::json!({"id": "c", "choices": [{"delta": {"content": content}}]}),
        )
        .unwrap()
    }

    fn failure() -> AttemptFailure {
        AttemptFailure::Connection("reset".into())
    }

    /// Brings attempt `n` to the point where a response is expected, with the consumer waiting.
    fn reading(chunks: &mut LlmChunks, attempt: u32, now: Ts) {
        chunks.step(Event::DemandStarted, now);
        chunks.step(Event::Connected { attempt }, now);
        chunks.step(Event::RequestWritten { attempt }, now);
    }

    // Catches: retrying without backoff, a first retry on `retry_interval` instead of 100 ms, or
    // exceeding `max_retry` additional attempts.
    #[tokio::test]
    async fn failures_before_output_are_retried_max_retry_times_with_backoff() {
        let mut chunks = start(policy());
        let mut now = at(0);
        for attempt in 0..=3u32 {
            chunks.step(Event::Failed { attempt, failure: failure() }, now);
            chunks.step(Event::AttemptClosed { attempt }, now);
            if attempt == 3 {
                let err = error(&mut chunks);
                assert_eq!(err.kind, ApiErrorKind::Connection);
                assert!(err.retryable, "nothing was delivered: a fallback may still try");
                assert!(settled(&chunks));
                return;
            }
            assert_eq!(outcome(&mut chunks), None);
            let backoff = if attempt == 0 { 100 } else { 2_000 };
            let due = now.saturating_add(Duration::from_millis(backoff));
            chunks.step(Event::Wake, due.saturating_sub(Duration::from_millis(1)));
            assert_eq!(launched(&chunks), Vec::<u32>::new());
            chunks.step(Event::Wake, due);
            assert_eq!(launched(&chunks), vec![attempt + 1]);
            now = due;
        }
    }

    // Catches: replaying a request whose answer already reached the consumer.
    #[tokio::test]
    async fn a_failure_after_visible_output_ends_the_request_unretryable() {
        let mut chunks = start(policy());
        reading(&mut chunks, 0, at(0));
        chunks.step(Event::Chunk { attempt: 0, chunk: text("Hel") }, at(5));
        assert!(matches!(delivered(&mut chunks).as_slice(), [chunk] if chunk.has_response()));
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(6));
        assert!(!error(&mut chunks).retryable);
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(7));
        chunks.step(Event::Wake, at(60_000));
        assert_eq!(launched(&chunks), Vec::<u32>::new());
    }

    // Catches: reasoning counted as output: the person never sees it, so an attempt that only
    // reasoned before failing is retried, and the first said word ends retries.
    #[tokio::test]
    async fn an_attempt_that_only_reasoned_is_retried() {
        let mut chunks = start(policy());
        reading(&mut chunks, 0, at(0));
        chunks.step(Event::Chunk { attempt: 0, chunk: text("<think>the user wants") }, at(5));
        chunks.step(Event::Chunk { attempt: 0, chunk: text(" X</thi") }, at(6));
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(7));
        assert_eq!(outcome(&mut chunks), None, "nothing was said: retried");
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(8));
        let due = chunks.wake_at().unwrap();
        chunks.step(Event::Wake, due);
        assert_eq!(launched(&chunks), vec![1]);
        // the retry starts its own reading: a said word now makes a failure final
        reading(&mut chunks, 1, due);
        chunks.step(Event::Chunk { attempt: 1, chunk: text("<think>x</think>Hi") }, due);
        chunks.step(Event::Failed { attempt: 1, failure: failure() }, due);
        assert!(!error(&mut chunks).retryable);
    }

    // Catches: replay guarded on any chunk instead of on output the consumer can act on: the
    // gateway's metadata-only leading delta or a tool call whose arguments are still streaming
    // made a failure final, while a completed tool call must never be issued twice.
    #[tokio::test]
    async fn only_actionable_output_blocks_a_retry() {
        let wire = |delta: serde_json::Value, finish: Option<&str>| -> WireChunk {
            serde_json::from_value(serde_json::json!({
                "id": "c",
                "choices": [{"delta": delta, "finish_reason": finish}],
            }))
            .unwrap()
        };
        let metadata = wire(
            serde_json::json!({"extra_content": {"livekit": {"inference_tier_billed": "s"}}}),
            None,
        );
        let call = wire(
            serde_json::json!({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "lookup"}}]}),
            None,
        );
        let args = wire(
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
            None,
        );
        let done = wire(serde_json::json!({}), Some("tool_calls"));
        let attempts: [&[WireChunk]; 3] = [
            std::slice::from_ref(&metadata),
            &[metadata.clone(), call.clone(), args.clone()],
            &[call, args, done],
        ];
        let mut chunks = start(policy());
        let mut now = at(0);
        for (attempt, wires) in (0u32..).zip(attempts) {
            reading(&mut chunks, attempt, now);
            for chunk in wires {
                chunks.step(Event::Chunk { attempt, chunk: chunk.clone() }, now);
            }
            chunks.step(Event::Failed { attempt, failure: failure() }, now);
            if attempt < 2 {
                assert_eq!(outcome(&mut chunks), None, "attempt {attempt} is retried");
                chunks.step(Event::AttemptClosed { attempt }, now);
                now = chunks.wake_at().unwrap();
                chunks.step(Event::Wake, now);
                assert_eq!(launched(&chunks), vec![attempt + 1]);
            } else {
                assert!(!error(&mut chunks).retryable);
            }
        }
    }

    // Catches: status retry rules lost in the request: a 401 retried, a 429 not.
    #[tokio::test]
    async fn a_rejection_is_retried_only_when_its_status_is_transient() {
        let reject = |status| AttemptFailure::Rejected {
            status,
            message: "no".into(),
            request_id: None,
            retryable: None,
        };
        let mut chunks = start(policy());
        chunks.step(Event::Failed { attempt: 0, failure: reject(429) }, at(0));
        assert_eq!(outcome(&mut chunks), None);
        chunks.step(Event::Wake, at(100));
        chunks.step(Event::Failed { attempt: 1, failure: reject(401) }, at(100));
        let err = error(&mut chunks);
        assert_eq!(
            (err.kind, err.status_code, err.retryable),
            (ApiErrorKind::Status, Some(401), false)
        );
    }

    // Catches: a read timer that counts the consumer's own pauses (backpressure) as remote
    // silence, or one not restarted by progress.
    #[tokio::test]
    async fn the_read_limit_counts_remote_silence_only_while_the_consumer_waits() {
        let mut chunks = start(Policy { max_retry: 0, ..policy() });
        reading(&mut chunks, 0, at(0));
        chunks.step(Event::Chunk { attempt: 0, chunk: text("a") }, at(9_000));
        assert_eq!(chunks.wake_at(), Some(at(19_000)), "progress restarts the limit");
        // The consumer stops pulling for a minute: no read timer runs meanwhile.
        chunks.step(Event::DemandEnded, at(9_001));
        assert_eq!(chunks.wake_at(), None);
        chunks.step(Event::Wake, at(60_000));
        assert_eq!(outcome(&mut chunks), None);
        chunks.step(Event::DemandStarted, at(70_000));
        assert_eq!(chunks.wake_at(), Some(at(80_000)));
        chunks.step(Event::Wake, at(80_000));
        let err = error(&mut chunks);
        assert_eq!((err.kind, err.timeout), (ApiErrorKind::Timeout, Some(Timeout::Read)));
    }

    // Catches: one timer for the whole exchange, or a timeout reported without its phase.
    #[tokio::test]
    async fn connect_and_write_limits_report_their_own_phase() {
        let limited =
            Policy { write_timeout: Some(Duration::from_secs(3)), max_retry: 0, ..policy() };
        let mut chunks = start(limited);
        chunks.step(Event::Wake, at(10_000));
        assert_eq!(error(&mut chunks).timeout, Some(Timeout::Connect));
        let mut chunks = start(limited);
        chunks.step(Event::Connected { attempt: 0 }, at(1_000));
        assert_eq!(chunks.wake_at(), Some(at(4_000)));
        chunks.step(Event::Wake, at(4_000));
        assert_eq!(error(&mut chunks).timeout, Some(Timeout::Write));
    }

    // Catches: a total deadline reset by retries or progress, or an attempt launched after it.
    #[tokio::test]
    async fn the_total_limit_spans_every_attempt_and_stops_the_request() {
        let mut chunks = start(Policy { total_timeout: Some(Duration::from_secs(3)), ..policy() });
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(0));
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(0));
        chunks.step(Event::Wake, at(100));
        reading(&mut chunks, 1, at(100));
        chunks.step(Event::Progress { attempt: 1 }, at(2_900));
        assert_eq!(chunks.wake_at(), Some(at(3_000)));
        chunks.step(Event::Wake, at(3_000));
        let err = error(&mut chunks);
        assert_eq!((err.timeout, err.retryable), (Some(Timeout::Total), false));
        chunks.step(Event::Failed { attempt: 1, failure: failure() }, at(3_001));
        chunks.step(Event::AttemptClosed { attempt: 1 }, at(3_001));
        chunks.step(Event::Wake, at(9_000));
        assert_eq!(launched(&chunks), Vec::<u32>::new());
        assert_eq!(outcome(&mut chunks), None);
    }

    // Catches: a cancelled request that still retries, reports an outcome, or forgets the cleanup
    // of the attempt it stopped.
    #[tokio::test]
    async fn cancel_stops_the_attempt_and_waits_for_its_cleanup() {
        let mut chunks = start(policy());
        chunks.step(Event::Cancel, at(10));
        assert!(!settled(&chunks), "the attempt still owes its release");
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(11));
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(12));
        assert!(settled(&chunks));
        assert_eq!(chunks.wake_at(), None);
        assert_eq!(outcome(&mut chunks), None);
        assert_eq!(launched(&chunks), Vec::<u32>::new());
    }

    // Catches: a late chunk from a superseded attempt delivered into the retry's answer.
    #[tokio::test]
    async fn a_superseded_attempts_events_are_ignored() {
        let mut chunks = start(policy());
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(0));
        chunks.step(Event::Wake, at(100));
        chunks.step(Event::Chunk { attempt: 0, chunk: text("stale") }, at(101));
        chunks.step(Event::Ended { attempt: 0 }, at(102));
        assert!(delivered(&mut chunks).is_empty());
        assert_eq!(outcome(&mut chunks), None);
        chunks.step(Event::Chunk { attempt: 1, chunk: text("fresh") }, at(103));
        assert_eq!(delivered(&mut chunks).len(), 1);
        chunks.step(Event::Ended { attempt: 1 }, at(104));
        assert_eq!(outcome(&mut chunks), Some(Ok(())));
    }

    // Catches: an attempt that never settles reported as clean termination, or reported forever
    // (one typed observation, ownership stays with the runtime).
    #[tokio::test]
    async fn a_missed_cleanup_deadline_is_reported_once() {
        let mut chunks = start(policy());
        chunks.step(Event::Cancel, at(0));
        let due = at(0).saturating_add(HOST_TEARDOWN_TIMEOUT);
        assert_eq!(chunks.wake_at(), Some(due));
        chunks.step(Event::Wake, due);
        assert!(chunks.cleanup_incomplete);
        assert_eq!(chunks.wake_at(), None);
    }

    // Catches: a transport that closes an attempt without a terminal report leaving the consumer
    // waiting forever.
    #[tokio::test]
    async fn an_attempt_closed_without_an_outcome_fails() {
        let mut chunks = start(Policy { max_retry: 0, ..policy() });
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(5));
        assert_eq!(error(&mut chunks).kind, ApiErrorKind::Connection);
        assert!(settled(&chunks));
    }

    #[tokio::test]
    async fn an_expired_acceptance_budget_never_authorizes_the_first_exchange() {
        let chunks = &mut start(Policy { total_timeout: Some(Duration::ZERO), ..policy() });
        assert_eq!(launched(chunks), Vec::<u32>::new());
        assert_eq!(error(chunks).timeout, Some(Timeout::Total));
    }

    // Catches: the time to first token latched on the failed attempt's metadata-only chunk, or
    // measured from the retry instead of from the request: the caller waited through the failure
    // and the backoff before the first token.
    #[tokio::test]
    async fn time_to_first_token_spans_the_retry() {
        use lk_agents_core::metrics::provider::LlmMetricState;

        let metadata: WireChunk = serde_json::from_value(serde_json::json!({
            "id": "c",
            "choices": [{"delta": {"extra_content": {"livekit": {"inference_tier_billed": "s"}}}}],
        }))
        .unwrap();
        let mut fold = LlmMetricState::new(0.0, String::new(), None);
        let mut observe = |chunks: Vec<LlmChunk>, now: Ts| {
            let secs = now.saturating_duration_since(at(0)).as_secs_f64();
            for chunk in chunks {
                fold.observe(&chunk, || secs);
            }
        };
        let mut chunks = start(policy());
        reading(&mut chunks, 0, at(0));
        chunks.step(Event::Chunk { attempt: 0, chunk: metadata }, at(10));
        observe(delivered(&mut chunks), at(10));
        chunks.step(Event::Failed { attempt: 0, failure: failure() }, at(20));
        chunks.step(Event::AttemptClosed { attempt: 0 }, at(20));
        let retry = chunks.wake_at().unwrap();
        chunks.step(Event::Wake, retry);
        assert_eq!(launched(&chunks), vec![1]);
        reading(&mut chunks, 1, retry);
        let first = retry.saturating_add(Duration::from_millis(30));
        chunks.step(Event::Chunk { attempt: 1, chunk: text("hi") }, first);
        observe(delivered(&mut chunks), first);
        let ttft = fold.ttft().expect("the retry answered");
        assert!((ttft - 0.15).abs() < 1e-9, "ttft {ttft}");
    }
}
