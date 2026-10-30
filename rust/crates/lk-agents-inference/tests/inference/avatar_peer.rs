//! The gateway avatar through its real HTTP adapter against a controlled peer: a local HTTP/1.1
//! server answering each request with the next scripted reply, after a delay or never, and
//! recording what it received.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::Closed;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::inference::avatar_session::AvatarError;
use lk_agents_core::attempts::RequestPolicy;
use lk_agents_inference::avatar::gateway::GatewayAvatar;
use lk_agents_inference::avatar::{AvatarOptions, AvatarRoom};
use lk_agents_inference::headers::Headers;
use lk_agents_net::clock::RealClock;
use lk_agents_net::http::GatewayHttp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Clone, Copy)]
enum Reply {
    Answer {
        delay_ms: u64,
        status: u16,
        body: &'static str,
    },
    /// Never answer.
    Silent,
}

fn ok(body: &'static str) -> Reply {
    Reply::Answer { delay_ms: 0, status: 200, body }
}

const CREATED: &str =
    r#"{"session_id":"s","provider_session_id":"p","terminate_token":"t","sample_rate":24000}"#;

/// Each received request: its head and its body.
type Received = Arc<Mutex<Vec<(String, String)>>>;

async fn peer(replies: Vec<Reply>) -> (String, Received) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let replies = Arc::new(Mutex::new(replies.into_iter()));
    let received = Received::default();
    let log = received.clone();
    tokio::spawn(async move {
        loop {
            let (socket, _) = listener.accept().await.unwrap();
            let (replies, log) = (replies.clone(), log.clone());
            tokio::spawn(async move { serve(socket, replies, log).await });
        }
    });
    (url, received)
}

async fn serve(
    mut socket: TcpStream,
    replies: Arc<Mutex<std::vec::IntoIter<Reply>>>,
    received: Received,
) {
    loop {
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        // The body is streamed, so it is chunked.
        while !request.ends_with(b"\r\n0\r\n\r\n") {
            match socket.read(&mut buf).await {
                Ok(0) | Err(_) => return,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        let text = String::from_utf8_lossy(&request).to_string();
        let (head, chunked) = text.split_once("\r\n\r\n").unwrap();
        let body = chunked.split("\r\n").nth(1).unwrap_or_default().to_owned();
        received.lock().await.push((head.to_owned(), body));
        let Some(reply) = replies.lock().await.next() else { return };
        match reply {
            Reply::Silent => {
                while let Ok(n) = socket.read(&mut buf).await {
                    if n == 0 {
                        return;
                    }
                }
                return;
            }
            Reply::Answer { delay_ms, status, body } => {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                let head = format!("HTTP/1.1 {status} X\r\ncontent-length: {}\r\n\r\n", body.len());
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            }
        }
    }
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().find_map(|l| {
        let (k, v) = l.split_once(": ")?;
        k.eq_ignore_ascii_case(name).then_some(v)
    })
}

fn avatar(url: String, read_ms: u64) -> GatewayAvatar {
    let options = serde_json::json!({"prompt": "hi", "custom": 1}).as_object().unwrap().clone();
    let opts = AvatarOptions::new("lemonslice/agent_1", None, None, options, url).unwrap();
    let policy = RequestPolicy {
        connect_timeout: Some(Duration::from_secs(10)),
        write_timeout: Some(Duration::from_secs(10)),
        read_timeout: Some(Duration::from_millis(read_ms)),
        total_timeout: None,
        max_retry: 3,
        retry_interval: Duration::from_millis(10),
    };
    let http = Arc::new(GatewayHttp::new().unwrap());
    let runtime = tokio::runtime::Handle::current();
    GatewayAvatar::new(
        opts,
        ("key".into(), "secret".into()),
        policy,
        http,
        &runtime,
        Arc::new(RealClock::new()),
    )
}

fn room() -> AvatarRoom {
    AvatarRoom {
        livekit_url: "wss://lk".into(),
        room_name: "room".into(),
        room_sid: "RM_1".into(),
        agent_identity: "agent".into(),
    }
}

/// Waits (bounded) until the peer received `n` requests.
async fn requests(received: &Received, n: usize) -> Vec<(String, String)> {
    for _ in 0..500 {
        if received.lock().await.len() >= n {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    received.lock().await.clone()
}

// Catches: a retry minting a second idempotency key (a second billable session), the gateway's
// rate or ids lost, or close not terminating with the retained identity.
#[tokio::test]
async fn a_retried_create_keeps_one_idempotency_key_and_close_terminates_it() {
    let (url, received) =
        peer(vec![Reply::Answer { delay_ms: 0, status: 503, body: "busy" }, ok(CREATED), ok("")])
            .await;
    let avatar = avatar(url, 5_000);
    assert_eq!(avatar.create(room(), Headers::new()).await.unwrap(), 24_000);
    use lk_agents_abi::inference::avatar_session::InferenceAvatar as _;
    assert_eq!(
        (avatar.session_id(), avatar.provider_session_id()),
        (Some("s".into()), Some("p".into()))
    );
    assert_eq!(avatar.close(Headers::new()).await, Closed::Clean);
    assert_eq!(avatar.provider_session_id(), None);
    let seen = requests(&received, 3).await;
    let (first, second, terminate) = (&seen[0].0, &seen[1].0, &seen[2]);
    assert!(first.starts_with("POST /v1/avatar/sessions "), "{first}");
    assert_eq!(header(first, "idempotency-key"), header(second, "idempotency-key"));
    assert_eq!(header(first, "idempotency-key").unwrap().len(), 32);
    assert_eq!(header(first, "x-livekit-inference-provider"), Some("lemonslice"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&seen[1].1).unwrap(),
        serde_json::json!({"provider":"lemonslice","livekit_url":"wss://lk","room_name":"room","room_sid":"RM_1",
            "avatar_identity":"lemonslice-inference-avatar","avatar_name":"lemonslice-inference-avatar",
            "agent_identity":"agent","avatar_id":"agent_1","prompt":"hi","extra_kwargs":{"custom":1}})
    );
    assert!(terminate.0.starts_with("POST /v1/avatar/sessions/terminate "));
    assert_eq!(header(&terminate.0, "idempotency-key"), None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&terminate.1).unwrap(),
        serde_json::json!({"provider":"lemonslice","provider_session_id":"p","terminate_token":"t"})
    );
}

// Catches (regression: the old check refused only completed sessions): two concurrent starts
// both creating a billable session, or the refusal disturbing the first.
#[tokio::test]
async fn a_concurrent_second_start_is_refused_and_the_first_completes() {
    let (url, received) =
        peer(vec![Reply::Answer { delay_ms: 200, status: 200, body: CREATED }]).await;
    let avatar = avatar(url, 5_000);
    let (first, second) = tokio::join!(avatar.create(room(), Headers::new()), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        avatar.create(room(), Headers::new()).await
    });
    assert!(matches!(second, Err(AvatarError::AlreadyStarted)), "{second:?}");
    assert_eq!(first.unwrap(), 24_000);
    assert_eq!(received.lock().await.len(), 1);
}

// Catches: a client error retried, or the reservation kept after a failure.
#[tokio::test]
async fn a_rejected_create_is_raised_unretried_and_the_avatar_can_start_again() {
    let (url, received) =
        peer(vec![Reply::Answer { delay_ms: 0, status: 400, body: "bad image" }, ok(CREATED)])
            .await;
    let avatar = avatar(url, 5_000);
    let err = avatar.create(room(), Headers::new()).await.unwrap_err();
    let AvatarError::Api(ProviderError::Local(err)) = err else { panic!("{err:?}") };
    assert_eq!(err.status_code, Some(400));
    assert_eq!(received.lock().await.len(), 1);
    assert_eq!(avatar.create(room(), Headers::new()).await.unwrap(), 24_000);
}

// Catches: a start cancelled while the gateway is creating leaking the session the gateway
// still answers with, or adopting it.
#[tokio::test]
async fn a_session_created_after_its_start_was_cancelled_is_terminated() {
    let (url, received) =
        peer(vec![Reply::Answer { delay_ms: 300, status: 200, body: CREATED }, ok("")]).await;
    let avatar = avatar(url, 5_000);
    let cancelled =
        tokio::time::timeout(Duration::from_millis(50), avatar.create(room(), Headers::new()))
            .await;
    assert!(cancelled.is_err());
    let seen = requests(&received, 2).await;
    assert_eq!(seen.len(), 2, "the late session is terminated");
    assert!(seen[1].0.starts_with("POST /v1/avatar/sessions/terminate "));
    use lk_agents_abi::inference::avatar_session::InferenceAvatar as _;
    assert_eq!(avatar.session_id(), None, "never adopted");
    assert_eq!(avatar.close(Headers::new()).await, Closed::Clean);
}

// Catches (regression, DST avatar_sessions_are_never_leaked): a create the gateway received,
// cut by its read timeout after close, abandoned while the gateway still makes the session.
#[tokio::test]
async fn close_during_a_received_create_asks_again_under_its_key_and_terminates() {
    let (url, received) = peer(vec![Reply::Silent, ok(CREATED), ok("")]).await;
    let avatar = Arc::new(avatar(url, 200));
    let creating = tokio::spawn({
        let avatar = avatar.clone();
        async move { avatar.create(room(), Headers::new()).await }
    });
    requests(&received, 1).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(avatar.close(Headers::new()).await, Closed::Clean);
    assert!(matches!(creating.await.unwrap(), Err(AvatarError::Closed)));
    let seen = requests(&received, 3).await;
    assert_eq!(header(&seen[0].0, "idempotency-key"), header(&seen[1].0, "idempotency-key"));
    assert!(seen[1].0.starts_with("POST /v1/avatar/sessions "), "{}", seen[1].0);
    assert!(seen[2].0.starts_with("POST /v1/avatar/sessions/terminate "), "{}", seen[2].0);
}

// Catches: close waiting forever on a received creation that never answers, or
// reporting a clean close while it may still create a billable session.
#[tokio::test(start_paused = true)]
async fn close_reports_incomplete_cleanup_when_a_creation_never_answers() {
    let (url, received) = peer(vec![Reply::Silent]).await;
    let avatar = Arc::new(avatar(url, 60_000));
    let creating = tokio::spawn({
        let avatar = avatar.clone();
        async move { avatar.create(room(), Headers::new()).await }
    });
    requests(&received, 1).await;
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(avatar.close(Headers::new()).await, Closed::TimedOut);
    assert!(matches!(creating.await.unwrap(), Err(AvatarError::Closed)));
}
