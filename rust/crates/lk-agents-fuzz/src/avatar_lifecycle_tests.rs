//! An agent's `AvatarSession` lifecycle over FakeRoom: the join wait and its timeout, the join and
//! playback-latency metrics, the close on the job's shutdown and the participant's removal.
use lk_agents_abi::errors::{HostError, RefRegistry};
use lk_agents_abi::metrics::base::AvatarMetrics;
use lk_agents_abi::voice::avatar::{AvatarHost, AvatarJoinError, AvatarLifecycle as _};
use lk_agents_abi::voice::rtc::{ConnectionState, ParticipantKind, TrackSource};
use lk_agents_abi::voice::session::SessionOptions as _;
use lk_agents_abi::voice::session_options::{SessionKwargs, TurnHandlingKwargs};
use lk_agents_abi::voice::session_owner::SessionOwner as _;
use lk_agents_abi::worker::job::SessionJob;
use lk_agents_abi::{ChatItem, ChatMessage, HostRef, Maybe, Role};
use lk_agents_fuzz::fakes::room::{FakeRoom, FakeRtc};
use lk_agents_runtime::avatar::lifecycle::{AvatarLifecycle, item_added};
use lk_agents_runtime::net::env::ProcessEnv;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::rtc::events::RoomEventBus;
use lk_agents_runtime::voice::session::SessionOptions;
use lk_agents_runtime::voice::session::owner::SessionOwner;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Registry;

impl RefRegistry for Registry {
    fn release(&self, _id: u64) {}
}

#[derive(Default)]
struct Avatar(Mutex<Vec<AvatarMetrics>>);

impl AvatarHost for Avatar {
    fn metrics_collected(&self, metrics: AvatarMetrics) {
        self.0.lock().unwrap().push(metrics);
    }
    fn close(&self) -> Result<HostRef, HostError> {
        let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
        Ok(HostRef::new(7, &registry))
    }
}

#[derive(Default)]
struct Job {
    shutdown: Mutex<Vec<u64>>,
    env: BTreeMap<String, String>,
}

impl SessionJob for Job {
    fn connect(
        &self,
        _: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        unreachable!("an avatar does not connect its job")
    }
    fn enable_recording(&self) -> Result<bool, HostError> {
        unreachable!("an avatar does not record")
    }
    fn claim_primary(&self, _: HostRef) -> Result<bool, HostError> {
        unreachable!("an avatar claims no session")
    }
    fn recording_dir(&self) -> Result<String, HostError> {
        unreachable!("an avatar does not record")
    }
    fn init_recording(&self, _: BTreeMap<String, bool>) -> Result<(), HostError> {
        unreachable!("an avatar does not record")
    }
    fn on_shutdown(&self, _: HostRef) -> Result<(), HostError> {
        unreachable!("an avatar closes no session")
    }
    fn on_avatar_shutdown(&self, close: HostRef) -> Result<(), HostError> {
        self.shutdown.lock().unwrap().push(close.id());
        Ok(())
    }
    fn server_env(&self) -> Result<BTreeMap<String, String>, HostError> {
        Ok(self.env.clone())
    }
}

fn owner() -> Arc<SessionOwner> {
    let kwargs = SessionKwargs {
        turn_handling: Some(TurnHandlingKwargs::default()),
        max_tool_steps: 3,
        use_tts_aligned_transcript: Maybe::NotGiven,
        tts_text_transforms: Maybe::NotGiven,
        min_consecutive_speech_delay: 0.0,
        expressive: Default::default(),
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: false,
        user_away_timeout: None,
        transcription_timeout: None,
        session_close_transcript_timeout: 2.0,
        conn_options: Default::default(),
    };
    Arc::new(SessionOwner::new(&ProcessEnv::current_runtime(), SessionOptions::new(kwargs)))
}

struct Started {
    lifecycle: Arc<AvatarLifecycle>,
    avatar: Arc<Avatar>,
    owner: Arc<SessionOwner>,
    room: Arc<FakeRoom>,
    clock: Arc<lk_agents_abi::VirtualClock>,
}

fn start(name: &str, job: Option<Arc<Job>>) -> Started {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room(name);
    let (owner, avatar) = (owner(), Arc::new(Avatar::default()));
    let lifecycle = AvatarLifecycle::open(
        owner.clone(),
        room.clone(),
        "avatar".into(),
        "tavus".into(),
        avatar.clone(),
        job.map(|job| job as Arc<dyn SessionJob>),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    )
    .unwrap();
    Started { lifecycle: Arc::new(lifecycle), avatar, owner, room, clock }
}

fn assistant(latency: Option<f64>, role: Role) -> ChatItem {
    let mut message = ChatMessage::new("item", role, vec![], std::time::SystemTime::UNIX_EPOCH);
    message.metrics.playback_latency = latency;
    ChatItem::Message(message)
}

async fn settle() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

// Catches: a join wait that is not the lifecycle's own, a join metric without its times or
// provider, or a waiter that does not see a join that happened before it waited.
#[tokio::test]
async fn the_join_is_reported_and_seen_by_every_waiter() {
    let s = start("joins", None);
    let waiting = {
        let lifecycle = s.lifecycle.clone();
        tokio::spawn(async move { lifecycle.wait_for_join(None).await })
    };
    let avatar = s.room.add_participant("avatar", ParticipantKind::Agent);
    settle().await;
    s.clock.advance(Duration::from_secs(2));
    avatar.publish_track("TR_cam", TrackSource::Camera);
    tokio::time::timeout(Duration::from_secs(5), waiting).await.unwrap().unwrap().unwrap();
    s.lifecycle.wait_for_join(Some(0.0)).await.unwrap();
    let metrics = s.avatar.0.lock().unwrap().clone();
    assert_eq!(metrics.len(), 1, "{metrics:?}");
    let (started, joined) =
        (metrics[0].session_started_time.unwrap(), metrics[0].avatar_joined_time.unwrap());
    assert!((joined - started - 2.0).abs() < 0.01, "{started} {joined}");
    assert_eq!(metrics[0].timestamp, joined);
    assert_eq!(metrics[0].metadata.as_ref().unwrap().model_provider.as_deref(), Some("tavus"));
    s.lifecycle.aclose().await;
}

// Catches: a timed-out waiter that cancels the join wait, or a waiter left pending by a close.
#[tokio::test]
async fn a_timeout_leaves_the_wait_running_and_a_close_ends_it() {
    let s = start("times-out", None);
    let timed_out = {
        let lifecycle = s.lifecycle.clone();
        tokio::spawn(async move { lifecycle.wait_for_join(Some(1.5)).await })
    };
    settle().await;
    s.clock.advance(Duration::from_secs(2));
    let outcome = tokio::time::timeout(Duration::from_secs(5), timed_out).await.unwrap().unwrap();
    assert_eq!(outcome, Err(AvatarJoinError::Timeout(1.5)));
    let pending = {
        let lifecycle = s.lifecycle.clone();
        tokio::spawn(async move { lifecycle.wait_for_join(None).await })
    };
    settle().await;
    assert!(!pending.is_finished(), "the join wait goes on after a waiter timed out");
    s.lifecycle.aclose().await;
    let closed = tokio::time::timeout(Duration::from_secs(5), pending).await.unwrap().unwrap();
    assert_eq!(closed, Err(AvatarJoinError::Closed));
    assert_eq!(s.lifecycle.wait_for_join(None).await, Err(AvatarJoinError::Closed));
    // The avatar joining after the close is not reported.
    s.room
        .add_participant("avatar", ParticipantKind::Agent)
        .publish_track("TR", TrackSource::Camera);
    settle().await;
    assert!(s.avatar.0.lock().unwrap().is_empty());
}

// Catches: a playback latency reported for a user message, without one, or after the close.
#[tokio::test]
async fn an_assistant_message_with_playback_latency_is_reported_until_the_close() {
    let s = start("playback", None);
    item_added(&s.owner, &assistant(Some(0.25), Role::Assistant), 10.0);
    item_added(&s.owner, &assistant(None, Role::Assistant), 11.0);
    item_added(&s.owner, &assistant(Some(0.5), Role::User), 12.0);
    {
        let metrics = s.avatar.0.lock().unwrap();
        assert_eq!(metrics.len(), 1, "{metrics:?}");
        assert_eq!((metrics[0].timestamp, metrics[0].playback_latency), (10.0, 0.25));
        assert_eq!(metrics[0].avatar_joined_time, None);
    }
    s.lifecycle.aclose().await;
    item_added(&s.owner, &assistant(Some(0.25), Role::Assistant), 13.0);
    assert_eq!(s.avatar.0.lock().unwrap().len(), 1);
}

/// A RoomService answering every request with `status` and `body`; the paths it was sent.
async fn room_service(
    status: &'static str,
    body: &'static str,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = paths.clone();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut buf).await.unwrap();
                request.extend_from_slice(&buf[..n]);
            }
            let head = String::from_utf8_lossy(&request).into_owned();
            seen.lock().unwrap().push(head.split_whitespace().nth(1).unwrap().to_owned());
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, paths)
}

fn job(url: &str) -> Arc<Job> {
    let env = [
        ("LIVEKIT_URL", url),
        ("LIVEKIT_API_KEY", "key"),
        ("LIVEKIT_API_SECRET", "a-secret-long-enough-for-signing-000000"),
    ];
    let env = env.into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    Arc::new(Job { env, ..Default::default() })
}

// Catches: an avatar whose close the job's shutdown does not run, a close that leaves the avatar
// in the room, a participant already gone failing the close, or a second close removing again.
#[tokio::test]
async fn the_close_removes_the_participant_once_and_skips_one_already_gone() {
    let (url, paths) =
        room_service("404 Not Found", r#"{"code":"not_found","msg":"participant not found"}"#)
            .await;
    let job = job(&url);
    let s = start("removes", Some(job.clone()));
    assert_eq!(*job.shutdown.lock().unwrap(), vec![7], "the job's shutdown runs aclose");
    let (first, second) = tokio::join!(s.lifecycle.aclose(), s.lifecycle.aclose());
    let ((), ()) = (first, second);
    s.lifecycle.aclose().await;
    assert_eq!(
        *paths.lock().unwrap(),
        vec!["/twirp/livekit.RoomService/RemoveParticipant".to_owned()]
    );
}

// Catches: a removal attempted from a disconnected room.
#[tokio::test]
async fn a_disconnected_room_removes_nobody() {
    let (url, paths) = room_service("200 OK", "").await;
    let s = start("disconnected", Some(job(&url)));
    s.room.set_connection_state(ConnectionState::Disconnected);
    s.lifecycle.aclose().await;
    assert!(paths.lock().unwrap().is_empty());
}
