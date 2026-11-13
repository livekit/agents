use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::Closed;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::inference::avatar_session::AvatarError;
use lk_agents_abi::telemetry::Cx;
use lk_agents_core::request::Policy;
use lk_agents_inference::avatar::gateway::GatewayAvatar;
use lk_agents_inference::avatar::{AvatarOptions, AvatarRoom};
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
    let policy = Policy {
        connect_timeout: Some(Duration::from_secs(10)),
        write_timeout: Some(Duration::from_secs(10)),
        read_timeout: Some(Duration::from_millis(read_ms)),
        total_timeout: None,
        max_retry: 3,
        retry_interval: Duration::from_millis(10),
    };
    let http = Arc::new(GatewayHttp::new().unwrap());
    let background = lk_agents_net::owned::Background::new(
        Arc::default(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        Arc::new(RealClock::new()),
    );
    GatewayAvatar::new(opts, ("key".into(), "secret".into()), policy, http, &background)
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
    assert_eq!(avatar.create(room(), Cx::NONE).await.unwrap(), 24_000);
    use lk_agents_abi::inference::avatar_session::InferenceAvatar as _;
    assert_eq!(
        (avatar.session_id(), avatar.provider_session_id()),
        (Some("s".into()), Some("p".into()))
    );
    assert_eq!(avatar.close(Cx::NONE).await, Closed::Clean);
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
    let (first, second) = tokio::join!(avatar.create(room(), Cx::NONE), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        avatar.create(room(), Cx::NONE).await
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
    let err = avatar.create(room(), Cx::NONE).await.unwrap_err();
    let AvatarError::Api(ProviderError::Local(err)) = err else { panic!("{err:?}") };
    assert_eq!(err.status_code, Some(400));
    assert_eq!(received.lock().await.len(), 1);
    assert_eq!(avatar.create(room(), Cx::NONE).await.unwrap(), 24_000);
}

// Catches: a start cancelled while the gateway is creating leaking the session the gateway
// still answers with, or adopting it.
#[tokio::test]
async fn a_session_created_after_its_start_was_cancelled_is_terminated() {
    let (url, received) =
        peer(vec![Reply::Answer { delay_ms: 300, status: 200, body: CREATED }, ok("")]).await;
    let avatar = avatar(url, 5_000);
    let cancelled =
        tokio::time::timeout(Duration::from_millis(50), avatar.create(room(), Cx::NONE)).await;
    assert!(cancelled.is_err());
    let seen = requests(&received, 2).await;
    assert_eq!(seen.len(), 2, "the late session is terminated");
    assert!(seen[1].0.starts_with("POST /v1/avatar/sessions/terminate "));
    use lk_agents_abi::inference::avatar_session::InferenceAvatar as _;
    assert_eq!(avatar.session_id(), None, "never adopted");
    assert_eq!(avatar.close(Cx::NONE).await, Closed::Clean);
}

// Catches (regression, DST avatar_sessions_are_never_leaked): a create the gateway received,
// cut by its read timeout after close, abandoned while the gateway still makes the session.
#[tokio::test]
async fn close_during_a_received_create_asks_again_under_its_key_and_terminates() {
    let (url, received) = peer(vec![Reply::Silent, ok(CREATED), ok("")]).await;
    let avatar = Arc::new(avatar(url, 200));
    let creating = tokio::spawn({
        let avatar = avatar.clone();
        async move { avatar.create(room(), Cx::NONE).await }
    });
    requests(&received, 1).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(avatar.close(Cx::NONE).await, Closed::Clean);
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
        async move { avatar.create(room(), Cx::NONE).await }
    });
    requests(&received, 1).await;
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(avatar.close(Cx::NONE).await, Closed::TimedOut);
    assert!(matches!(creating.await.unwrap(), Err(AvatarError::Closed)));
}

/// The session lifecycle, stepped by hand at chosen instants. Attempts are spawned but never
/// polled (the test never yields), so only the events a test steps arrive.
mod stepped {
    use std::sync::Arc;
    use std::time::Duration;

    use super::super::{Driver, Event, Input, RETIREMENT_TIMEOUT, RequestId, View};
    use lk_agents_abi::inference::avatar_session::AvatarError;
    use lk_agents_abi::telemetry::Cx;
    use lk_agents_abi::{Closed, HOST_TEARDOWN_TIMEOUT, Ts, VirtualClock};
    use lk_agents_core::request::{AttemptFailure, Policy};
    use lk_agents_inference::avatar::{AvatarOptions, ProviderSession};
    use lk_agents_net::http::GatewayHttp;
    use lk_agents_net::owned::Background;
    use tokio::sync::oneshot;

    type Answer = oneshot::Receiver<Result<u32, AvatarError>>;

    fn at(ms: u64) -> Ts {
        Ts::from_nanos(ms * 1_000_000)
    }

    fn policy() -> Policy {
        Policy {
            connect_timeout: Some(Duration::from_secs(10)),
            write_timeout: Some(Duration::from_secs(10)),
            read_timeout: Some(Duration::from_secs(10)),
            total_timeout: None,
            max_retry: 2,
            retry_interval: Duration::from_secs(1),
        }
    }

    fn driver(policy: Policy) -> Driver {
        let opts = AvatarOptions::new(
            "lemonslice/a",
            None,
            None,
            Default::default(),
            "http://127.0.0.1:9".into(),
        )
        .unwrap();
        let http = Arc::new(GatewayHttp::new().unwrap());
        let view = Arc::new(parking_lot::Mutex::new(View::default()));
        let background = Background::new(
            Arc::default(),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
            Arc::new(VirtualClock::new()),
        );
        let credentials = ("key".into(), "secret".into());
        let (driver, observed) =
            Driver::new(Arc::new(opts), credentials, policy, http, &background, view);
        // Reports are never read: the attempts are never polled.
        std::mem::forget(observed);
        driver
    }

    fn session(id: &str) -> ProviderSession {
        ProviderSession {
            session_id: Some(id.into()),
            provider_session_id: Some(format!("p-{id}")),
            terminate_token: Some("t".into()),
            sample_rate: 24_000,
        }
    }

    fn create(d: &mut Driver, caller: u64, now: Ts) -> Answer {
        let (reply, answer) = oneshot::channel();
        let input = Input::Create { caller, room: super::room(), cx: Cx::NONE, reply };
        d.input(input, now);
        answer
    }

    fn close(d: &mut Driver, now: Ts) -> oneshot::Receiver<Closed> {
        let (reply, settled) = oneshot::channel();
        d.input(Input::Close { cx: Cx::NONE, reply }, now);
        settled
    }

    fn cancel(d: &mut Driver, caller: u64, now: Ts) {
        d.input(Input::Cancel { caller }, now);
    }

    /// The attempts launched and not yet closed, as (request, attempt).
    fn launched(d: &Driver) -> Vec<(u64, u32)> {
        d.tasks.keys().map(|(r, a)| (r.0, *a)).collect()
    }

    /// The session request `r` terminates, if it is a running terminate.
    fn terminating(d: &Driver, r: u64) -> Option<ProviderSession> {
        d.terminating.get(&RequestId(r)).cloned()
    }

    fn created(d: &mut Driver, request: u64, attempt: u32, s: &ProviderSession, now: Ts) {
        let request = RequestId(request);
        d.step(Event::Created { request, attempt, session: s.clone() }, now);
    }

    fn failed(d: &mut Driver, request: u64, attempt: u32, failure: AttemptFailure, now: Ts) {
        d.step(Event::Failed { request: RequestId(request), attempt, failure }, now);
    }

    fn closed(d: &mut Driver, request: u64, attempt: u32, now: Ts) {
        d.step(Event::AttemptClosed { request: RequestId(request), attempt }, now);
    }

    fn written(d: &mut Driver, request: u64, attempt: u32, now: Ts) {
        let request = RequestId(request);
        d.step(Event::Connected { request, attempt }, now);
        d.step(Event::RequestWritten { request, attempt }, now);
    }

    fn reset() -> AttemptFailure {
        AttemptFailure::Connection("reset".into())
    }

    fn rejected(status: i64) -> AttemptFailure {
        AttemptFailure::Rejected { status, message: "no".into(), request_id: None, retryable: None }
    }

    /// Releases every attempt that is no longer its request's current one, as the runtime does
    /// once a stopped or failed attempt's task ends.
    fn release_stopped(d: &mut Driver, now: Ts) {
        let stale: Vec<(u64, u32)> = d
            .tasks
            .keys()
            .filter(|(r, a)| d.requests.get(r).and_then(|r| r.attempts.current()) != Some(*a))
            .map(|(r, a)| (r.0, *a))
            .collect();
        for (r, a) in stale {
            closed(d, r, a, now);
        }
    }

    // Catches: a second start launching a second billable create, the refusal disturbing the
    // first, or a started instance starting again.
    #[tokio::test]
    async fn a_second_caller_is_refused_and_the_first_creation_is_untouched() {
        let mut d = driver(policy());
        let mut first = create(&mut d, 1, at(0));
        assert_eq!(launched(&d), vec![(1, 0)], "reserved and launched at once");
        let mut second = create(&mut d, 2, at(1));
        assert!(matches!(second.try_recv(), Ok(Err(AvatarError::AlreadyStarted))));
        assert_eq!(launched(&d), vec![(1, 0)]);
        let s = session("a");
        created(&mut d, 1, 0, &s, at(2));
        assert!(matches!(first.try_recv(), Ok(Ok(24_000))));
        assert_eq!(d.held, vec![s]);
        closed(&mut d, 1, 0, at(4));
        assert!(d.is_settled());
        let mut third = create(&mut d, 3, at(5));
        assert!(
            matches!(third.try_recv(), Ok(Err(AvatarError::AlreadyStarted))),
            "one instance is one avatar"
        );
    }

    // Catches: a failed creation keeping its reservation, so the instance can never start.
    #[tokio::test]
    async fn a_failed_creation_releases_its_reservation() {
        let mut d = driver(policy());
        let mut answer = create(&mut d, 1, at(0));
        failed(&mut d, 1, 0, rejected(400), at(1));
        assert!(matches!(
            answer.try_recv(),
            Ok(Err(AvatarError::Api(lk_agents_abi::errors::ProviderError::Local(e)))) if e.status_code == Some(400)
        ));
        let _again = create(&mut d, 2, at(2));
        assert!(launched(&d).contains(&(2, 0)));
    }

    // Catches: a retry minting a new operation (a second provider session), or a stale attempt's
    // answer of the same operation treated as a stray session and terminated under the caller.
    #[tokio::test]
    async fn retries_keep_the_operation_and_any_attempts_answer_settles_it() {
        let mut d = driver(policy());
        let mut answer = create(&mut d, 1, at(0));
        failed(&mut d, 1, 0, reset(), at(0));
        assert!(answer.try_recv().is_err(), "retried, not answered");
        d.step(Event::Wake, at(100));
        assert_eq!(launched(&d), vec![(1, 0), (1, 1)], "same operation, next attempt");
        // Attempt 0's answer was already queued when it failed: the gateway replays one result per
        // operation, so it is the operation's answer.
        let s = session("a");
        created(&mut d, 1, 0, &s, at(101));
        assert!(matches!(answer.try_recv(), Ok(Ok(24_000))));
        created(&mut d, 1, 1, &s, at(103));
        assert_eq!(d.held, vec![s], "the replay is the held session");
        assert!(d.terminating.is_empty());
    }

    // Catches: a creation whose caller left launching more attempts, or its late success adopted
    // or leaked instead of terminated.
    #[tokio::test]
    async fn a_late_success_after_the_caller_left_is_retired() {
        let mut d = driver(policy());
        let mut answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(2));
        cancel(&mut d, 1, at(5));
        assert_eq!(launched(&d), vec![(1, 0)], "the sent attempt is left to answer");
        let s = session("late");
        created(&mut d, 1, 0, &s, at(900));
        assert!(matches!(answer.try_recv(), Err(oneshot::error::TryRecvError::Closed)));
        assert_eq!(terminating(&d, 2), Some(s.clone()));
        assert!(launched(&d).contains(&(2, 0)));
        d.step(Event::Terminated { request: RequestId(2), attempt: 0 }, at(950));
        assert!(d.terminating.is_empty());
        closed(&mut d, 1, 0, at(951));
        closed(&mut d, 2, 0, at(952));
        assert!(d.is_settled());
        let _again = create(&mut d, 2, at(953));
        assert_eq!(launched(&d), vec![(3, 0)], "nothing was adopted: it may start again");
    }

    // Catches: a caller cancelled during backoff still getting a retry launched.
    #[tokio::test]
    async fn a_caller_leaving_during_backoff_ends_the_creation() {
        let mut d = driver(policy());
        let _answer = create(&mut d, 1, at(0));
        failed(&mut d, 1, 0, reset(), at(0));
        cancel(&mut d, 1, at(50));
        d.step(Event::Wake, at(100));
        assert_eq!(launched(&d), vec![(1, 0)]);
        closed(&mut d, 1, 0, at(101));
        assert!(d.is_settled());
    }

    // Catches: a session delivered to a caller that is gone left unowned.
    #[tokio::test]
    async fn a_refused_delivery_keeps_cleanup_with_the_sender() {
        let mut d = driver(policy());
        drop(create(&mut d, 1, at(0)));
        let s = session("a");
        created(&mut d, 1, 0, &s, at(1));
        assert_eq!(terminating(&d, 2), Some(s));
        assert!(launched(&d).contains(&(2, 0)));
    }

    // Catches: close during creation leaving the caller waiting, or the session created after it
    // kept alive.
    #[tokio::test]
    async fn close_during_creation_settles_the_caller_and_retires_the_result() {
        let mut d = driver(policy());
        let mut answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(0));
        let _settled = close(&mut d, at(1));
        assert!(matches!(answer.try_recv(), Ok(Err(AvatarError::Closed))));
        assert!(!d.is_settled(), "the running attempt still owes its answer");
        let s = session("a");
        created(&mut d, 1, 0, &s, at(2));
        assert_eq!(terminating(&d, 2), Some(s));
        let mut again = create(&mut d, 2, at(3));
        assert!(matches!(again.try_recv(), Ok(Err(AvatarError::AlreadyStarted))));
    }

    // Catches: an unsent create kept running after its caller left, or a stopped attempt that
    // never releases reported as clean or waited on forever (one report, the runtime keeps the
    // attempt).
    #[tokio::test]
    async fn a_missed_cleanup_deadline_is_reported_once() {
        let mut d = driver(policy());
        let _answer = create(&mut d, 1, at(0));
        cancel(&mut d, 1, at(0));
        let due = at(0).saturating_add(HOST_TEARDOWN_TIMEOUT);
        assert_eq!(
            d.wake_at(),
            Some(due),
            "the gateway cannot have received it: nothing to retire"
        );
        let (reply, mut settled) = oneshot::channel();
        d.closers.push(reply);
        d.step(Event::Wake, due);
        assert_eq!(settled.try_recv(), Ok(Closed::TimedOut));
        assert!(d.is_settled());
        assert_eq!(d.wake_at(), None);
        assert_eq!(launched(&d), vec![(1, 0)], "the runtime still owns the attempt");
        // A session the stopped attempt reports anyway is still terminated.
        let s = session("very-late");
        created(&mut d, 1, 0, &s, at(9_000));
        assert_eq!(terminating(&d, 2), Some(s));
    }

    // Catches: a failed terminate forgetting the session (it bills on), or a repeat close not
    // retrying it.
    #[tokio::test]
    async fn close_terminates_the_held_session_and_a_failure_keeps_it_held() {
        let mut d = driver(policy());
        let _answer = create(&mut d, 1, at(0));
        let s = session("a");
        created(&mut d, 1, 0, &s, at(1));
        closed(&mut d, 1, 0, at(1));
        let _settled = close(&mut d, at(2));
        assert_eq!(terminating(&d, 2), Some(s.clone()));
        failed(&mut d, 2, 0, rejected(403), at(3));
        assert_eq!(d.held, vec![s.clone()]);
        closed(&mut d, 2, 0, at(4));
        assert!(d.is_settled());
        let _settled = close(&mut d, at(5));
        assert_eq!(terminating(&d, 3), Some(s));
        assert_eq!(d.view.lock().provider_session_id.as_deref(), Some("p-a"));
        d.step(Event::Terminated { request: RequestId(3), attempt: 0 }, at(6));
        assert!(d.terminating.is_empty() && d.held.is_empty());
        assert_eq!(d.view.lock().provider_session_id, None);
    }

    // Catches: a terminate request sent without the identity it needs.
    #[tokio::test]
    async fn a_session_without_terminate_identity_is_released() {
        let mut d = driver(policy());
        drop(create(&mut d, 1, at(0)));
        let s = ProviderSession { terminate_token: None, ..session("a") };
        created(&mut d, 1, 0, &s, at(1));
        assert_eq!(launched(&d), vec![(1, 0)]);
        assert!(d.terminating.is_empty() && d.held.is_empty());
    }

    // Catches: a create the gateway received abandoned when its read timeout fires after close:
    // the gateway still makes the session and nobody ends it (DST avatar_sessions_are_never_leaked,
    // seed 0).
    #[tokio::test]
    async fn a_received_create_cut_by_its_timeout_after_close_is_asked_again_and_retired() {
        let read = Policy { read_timeout: Some(Duration::from_secs(2)), max_retry: 0, ..policy() };
        let mut d = driver(read);
        let mut answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(10));
        let _settled = close(&mut d, at(20));
        assert!(matches!(answer.try_recv(), Ok(Err(AvatarError::Closed))));
        d.step(Event::Wake, d.wake_at().unwrap());
        assert_eq!(launched(&d), vec![(1, 0)], "the timed-out attempt is stopped, then retried");
        d.step(Event::Wake, d.wake_at().unwrap());
        assert_eq!(launched(&d), vec![(1, 0), (1, 1)], "same operation identity");
        let s = session("a");
        created(&mut d, 1, 1, &s, at(3_000));
        assert_eq!(terminating(&d, 2), Some(s));
    }

    // Catches: a create whose last attempt lost its answer after the gateway received it failing
    // its caller and forgetting the operation, so the session the gateway made bills on.
    #[tokio::test]
    async fn a_create_failed_after_it_was_received_is_retired_once_its_caller_has_the_error() {
        let mut d = driver(Policy { max_retry: 0, ..policy() });
        let mut answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(10));
        failed(&mut d, 1, 0, reset(), at(500));
        assert!(matches!(answer.try_recv(), Ok(Err(AvatarError::Api(_)))));
        d.step(Event::Wake, d.wake_at().unwrap());
        assert_eq!(launched(&d), vec![(1, 0), (1, 1)], "asked again under its identity");
        let s = session("a");
        created(&mut d, 1, 1, &s, at(2_000));
        assert_eq!(terminating(&d, 2), Some(s));
    }

    // Catches: a retirement that cannot learn the operation's outcome waited on forever, or ended
    // silently (it is reported; a later answer is still terminated).
    #[tokio::test]
    async fn a_retirement_unanswered_by_its_deadline_is_reported() {
        let mut d = driver(policy());
        let _answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(10));
        let mut settled = close(&mut d, at(20));
        let deadline = at(20).saturating_add(RETIREMENT_TIMEOUT);
        while let Some(due) = d.wake_at().filter(|due| *due <= deadline) {
            d.step(Event::Wake, due);
            release_stopped(&mut d, due);
        }
        assert!(d.is_settled());
        assert_eq!(settled.try_recv(), Ok(Closed::TimedOut));
        let s = session("very-late");
        created(&mut d, 1, 0, &s, deadline.saturating_add(Duration::from_secs(1)));
        assert_eq!(terminating(&d, 2), Some(s));
    }

    // Catches: a retirement replay that the gateway leaves silent waited on until the retirement
    // deadline when the caller set no phase limits, so the replay is never asked again and the
    // session the gateway made leaks (DST avatar_sessions_are_never_leaked, seed 428).
    #[tokio::test]
    async fn a_silent_retirement_replay_without_caller_limits_is_asked_again_within_the_window() {
        let unbounded = Policy {
            connect_timeout: None,
            write_timeout: None,
            read_timeout: None,
            max_retry: 0,
            ..policy()
        };
        let mut d = driver(unbounded);
        let _answer = create(&mut d, 1, at(0));
        written(&mut d, 1, 0, at(10));
        let _settled = close(&mut d, at(20));
        let deadline = at(20).saturating_add(RETIREMENT_TIMEOUT);
        let mut replays = Vec::new();
        while let Some(due) = d.wake_at().filter(|due| *due < deadline) {
            d.step(Event::Wake, due);
            assert!(d.requests.contains_key(&RequestId(1)), "retirement gave up inside its window");
            release_stopped(&mut d, due);
            for (r, a) in launched(&d) {
                if r == 1 && a > 0 && !replays.contains(&a) {
                    replays.push(a);
                    written(&mut d, 1, a, due);
                }
            }
            if replays.len() == 2 {
                break;
            }
        }
        assert_eq!(replays.len(), 2, "a silent replay is abandoned and asked again: {replays:?}");
        let s = session("a");
        let now = d.wake_at().unwrap().min(deadline);
        created(&mut d, 1, replays[1], &s, now);
        assert_eq!(terminating(&d, 2), Some(s));
    }
}
