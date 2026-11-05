//! An agent's `AvatarSession` (`_core_AvatarLifecycle`): the avatar's join wait, its metrics, its
//! close on the job's shutdown and its participant's removal.

use crate::voice::session::owner::SessionOwner;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::metrics::base::{AvatarMetrics, Metadata};
use lk_agents_abi::voice::avatar::{self, AvatarHost, AvatarJoinError};
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_abi::voice::session_owner::SessionOwner as _;
use lk_agents_abi::worker::job::SessionJob;
use lk_agents_abi::{ChatItem, Clock, Role, Spawner};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::room_service::{Removal, RoomService};
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OnceCell, watch};

/// The avatars attached to a session, each told of the session's assistant messages' playback
/// latency.
#[derive(Default)]
pub struct Avatars(Mutex<Attachments>);

#[derive(Default)]
struct Attachments {
    next: u64,
    attached: Vec<(u64, Attached)>,
}

#[derive(Clone)]
struct Attached {
    provider: String,
    host: Arc<dyn AvatarHost>,
}

impl Avatars {
    fn attach(&self, attached: Attached) -> u64 {
        let mut avatars = self.0.lock();
        avatars.next += 1;
        let id = avatars.next;
        avatars.attached.push((id, attached));
        id
    }

    fn detach(&self, id: u64) {
        let detached = {
            let mut avatars = self.0.lock();
            let at = avatars.attached.iter().position(|(attached, _)| *attached == id);
            at.map(|at| avatars.attached.remove(at))
        };
        drop(detached);
    }
}

/// `item` added to the session's history: an assistant message with a `playback_latency` is each
/// attached avatar's `AvatarMetrics(playback_latency=..)`, stamped `at` (wall seconds).
pub fn item_added(owner: &SessionOwner, item: &ChatItem, at: f64) {
    let ChatItem::Message(message) = item else { return };
    let Some(latency) = message.metrics.playback_latency else { return };
    if message.role != Role::Assistant {
        return;
    }
    let attached: Vec<Attached> =
        owner.avatars().0.lock().attached.iter().map(|(_, a)| a.clone()).collect();
    for avatar in attached {
        let metrics = AvatarMetrics {
            timestamp: at,
            playback_latency: latency,
            session_started_time: None,
            avatar_joined_time: None,
            metadata: Some(metadata(&avatar.provider)),
        };
        report(owner, avatar.host.as_ref(), metrics);
    }
}

fn metadata(provider: &str) -> Metadata {
    Metadata { model_name: None, model_provider: Some(provider.to_owned()) }
}

/// The avatar's `metrics_collected`, and the session's metrics once it runs.
fn report(owner: &SessionOwner, host: &dyn AvatarHost, metrics: AvatarMetrics) {
    if let Some(session) = owner.session() {
        // A closed session records no more metrics.
        let _ = session.report_avatar_metrics(metrics.clone());
    }
    host.metrics_collected(metrics);
}

pub struct AvatarLifecycle(Arc<Inner>);

struct Inner {
    owner: Arc<SessionOwner>,
    room: Arc<dyn RtcRoom>,
    identity: String,
    clock: Arc<dyn Clock>,
    job: Option<Arc<dyn SessionJob>>,
    attached: u64,
    /// The join wait's outcome: `None` while it waits.
    joined: watch::Sender<Option<Result<(), AvatarJoinError>>>,
    waiting: Arc<TaskSet>,
    closed: OnceCell<()>,
}

impl avatar::AvatarLifecycle for AvatarLifecycle {
    type Env = ProcessEnv;
    type Owner = SessionOwner;

    fn new(
        env: &ProcessEnv,
        owner: Arc<SessionOwner>,
        room: Arc<dyn RtcRoom>,
        identity: String,
        provider: String,
        avatar: Arc<dyn AvatarHost>,
        job: Option<Arc<dyn SessionJob>>,
    ) -> Result<Self, HostError> {
        let spawner = (env.spawner)().map_err(HostError::new)?;
        let clock = env.clock.clone();
        Self::open(owner, room, identity, provider, avatar, job, clock, spawner)
    }

    async fn wait_for_join(&self, timeout: Option<f64>) -> Result<(), AvatarJoinError> {
        let mut joined = self.0.joined.subscribe();
        let outcome = async move {
            match joined.wait_for(Option::is_some).await {
                Ok(outcome) => outcome.clone().unwrap_or(Err(AvatarJoinError::Closed)),
                Err(_) => Err(AvatarJoinError::Closed),
            }
        };
        let Some(limit) = timeout else { return outcome.await };
        let wait = Duration::try_from_secs_f64(limit).unwrap_or(Duration::ZERO);
        lk_agents_abi::timeout(self.0.clock.as_ref(), wait, outcome)
            .await
            .unwrap_or(Err(AvatarJoinError::Timeout(limit)))
    }

    async fn aclose(&self) {
        let inner = &self.0;
        inner
            .closed
            .get_or_init(|| async {
                inner.owner.avatars().detach(inner.attached);
                inner.waiting.cancel_and_wait(inner.clock.as_ref()).await;
                inner.settle(Err(AvatarJoinError::Closed));
                inner.remove_participant().await;
            })
            .await;
    }
}

impl AvatarLifecycle {
    /// `new`, on `clock` and `spawner`.
    #[allow(clippy::too_many_arguments, reason = "the class's arguments, and its executor")]
    pub fn open(
        owner: Arc<SessionOwner>,
        room: Arc<dyn RtcRoom>,
        identity: String,
        provider: String,
        avatar: Arc<dyn AvatarHost>,
        job: Option<Arc<dyn SessionJob>>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Result<Self, HostError> {
        match &job {
            Some(job) => job.on_avatar_shutdown(avatar.close()?)?,
            None => log::debug!(
                target: "livekit.agents",
                "AvatarSession started outside a job context; call aclose() manually to release \
                 resources when the job shuts down"
            ),
        }
        let attached =
            owner.avatars().attach(Attached { provider: provider.clone(), host: avatar.clone() });
        let inner = Arc::new(Inner {
            owner,
            room,
            identity,
            clock,
            job,
            attached,
            joined: watch::channel(None).0,
            waiting: Arc::default(),
            closed: OnceCell::new(),
        });
        let waiter = inner.clone();
        let wait = async move {
            let (room, identity, clock) =
                (waiter.room.clone(), waiter.identity.clone(), waiter.clock.clone());
            let joined =
                crate::avatar::receiver::wait_avatar_join_room(room, identity, clock).await;
            let outcome = joined.map(|(started, joined)| {
                let metrics = AvatarMetrics {
                    timestamp: joined,
                    playback_latency: 0.0,
                    session_started_time: Some(started),
                    avatar_joined_time: Some(joined),
                    metadata: Some(metadata(&provider)),
                };
                report(&waiter.owner, avatar.as_ref(), metrics);
            });
            waiter.settle(outcome.map_err(AvatarJoinError::from));
        };
        inner.waiting.spawn(spawner.as_ref(), "avatar join", Box::pin(wait));
        Ok(Self(inner))
    }
}

impl Inner {
    /// The join wait's first outcome is its outcome.
    fn settle(&self, outcome: Result<(), AvatarJoinError>) {
        self.joined.send_if_modified(|joined| {
            let unset = joined.is_none();
            if unset {
                *joined = Some(outcome);
            }
            unset
        });
    }

    /// Inside a job, while the room is connected: the avatar participant leaves the room.
    async fn remove_participant(&self) {
        let Some(job) = &self.job else { return };
        if !self.room.connection_state().is_ok_and(|state| state.is_connected()) {
            return;
        }
        let identity = self.identity.as_str();
        let removed = async {
            let room = self.room.name().map_err(|error| error.to_string())?;
            let env = job.server_env().map_err(|error| error.to_string())?;
            let setting = |name: &str| env.get(name).map_or("", String::as_str).to_owned();
            let service = RoomService::new(
                &setting("LIVEKIT_URL"),
                &setting("LIVEKIT_API_KEY"),
                &setting("LIVEKIT_API_SECRET"),
            );
            service.remove_participant(&room, identity).await.map_err(|error| error.to_string())
        };
        match removed.await {
            Ok(Removal::Removed) => {}
            Ok(Removal::NotInRoom) => log::debug!(
                target: "livekit.agents", identity = identity;
                "avatar participant not in room, skipping removal"
            ),
            Err(error) => log::warn!(
                target: "livekit.agents", identity = identity, error = error.as_str();
                "failed to remove avatar participant"
            ),
        }
    }
}
