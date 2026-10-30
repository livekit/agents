//! The gateway LLM through its real HTTP/SSE adapter against a controlled peer: a local HTTP/1.1
//! server that writes each response in chunks cut where the test says, pauses, hangs, rejects or
//! resets, and records whether the client released the connection.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::{ApiErrorKind, TimeoutPhase};
use lk_agents_abi::{ChatContext, Clock, Closed};
use lk_agents_inference::llm::LlmOptions;
use lk_agents_inference::llm::gateway::{GatewayLlm, LlmChunks};
use lk_agents_inference::llm::lifecycle::RequestPolicy;
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
}

struct Peer {
    url: String,
    requests: Arc<AtomicUsize>,
    /// Each request's `X-LiveKit-Session-ID`, in arrival order.
    sessions: Arc<lk_agents_net::lock::Mutex<Vec<String>>>,
    /// Connections the client closed while the peer was still serving them.
    released: Arc<AtomicUsize>,
}

impl Peer {
    async fn start(replies: Vec<Reply>) -> Peer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let replies = Arc::new(Mutex::new(replies.into_iter()));
        let (requests, released) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let sessions = Arc::new(lk_agents_net::lock::Mutex::new(Vec::new()));
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
    sessions: Arc<lk_agents_net::lock::Mutex<Vec<String>>>,
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
            Reply::Status(status, body) => {
                let head =
                    format!("HTTP/1.1 {status} Nope\r\ncontent-length: {}\r\n\r\n", body.len());
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            }
            Reply::Sse(parts) => {
                let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
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

fn policy(max_retry: u32, read_ms: u64) -> RequestPolicy {
    let limit = Some(Duration::from_secs(5));
    RequestPolicy {
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
        api_key: "key".into(),
        api_secret: "secret".into(),
        provider: None,
        inference_class: None,
        extra_kwargs: Default::default(),
    };
    let http = Arc::new(GatewayHttp::new().unwrap());
    GatewayLlm::new(opts, http, tokio::runtime::Handle::current(), Arc::new(RealClock::new()))
}

fn open(llm: &GatewayLlm, policy: RequestPolicy) -> LlmChunks {
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

// Catches: a retry replaying the failed attempt's gateway session: each attempt is its own
// session, so the gateway never joins a retry to the request it already failed.
#[tokio::test(flavor = "multi_thread")]
async fn each_attempt_opens_a_fresh_gateway_session() {
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
    assert_eq!(sessions.len(), 3, "{sessions:?}");
    assert!(sessions.iter().all(|s| s.starts_with("inference_")), "{sessions:?}");
    let distinct: std::collections::BTreeSet<_> = sessions.iter().collect();
    assert_eq!(distinct.len(), 3, "one session per attempt: {sessions:?}");
}

// Catches: a stalled stream waited on forever, a read timeout reported without its phase or as
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
        (err.kind, err.phase, err.retryable),
        (ApiErrorKind::Timeout, Some(TimeoutPhase::Read), false)
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
        open(&llm, RequestPolicy { retry_interval: Duration::from_secs(5), ..policy(3, 10_000) });
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
