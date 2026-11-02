//! `RealtimeModelFallbackAdapter`: the realtime fallback on the shared chain machine. Unlike the
//! LLM/STT/TTS chains, failure is a pushed event, recovery is a cooldown, a healthy primary is not
//! preempted, and availability is per session. `capabilities` merges the models' capabilities,
//! `session` replaces a failed model's session and replays the conversation, and `backlog` bounds
//! the requests held while it does.

mod backlog;
mod capabilities;
mod session;
#[cfg(test)]
#[path = "../realtime_tests.rs"]
mod tests;

pub use backlog::{
    BacklogOverload, FALLBACK_PENDING_BYTES, FALLBACK_PENDING_REPLIES, FallbackBacklog,
};
pub use capabilities::{CapabilityMismatch, merge_capabilities};
pub use session::FallbackSession;
use session::SessionState;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ProviderError, TimeoutPhase};
use lk_agents_abi::llm::fallback::AvailabilityChanged;
use lk_agents_abi::llm::realtime::{
    Event, GenerationDone, GenerationStatus, HostRealtime, MessageDone, MessageModalities,
    Modalities, RealtimeModelError,
};
use lk_agents_abi::{ChatContext, Realtime, RealtimeCommand, RealtimeSession, WallAnchor};
use parking_lot::Mutex;

use crate::fallback::machine::{ChainEvent, ChainRequest};
use crate::fallback::{ChainObserver, ChainPolicy, FallbackPolicy, backend_resource};
use crate::realtime::host::HostRealtimeModel;
use lk_agents_abi::llm::realtime::RealtimeCapabilities;
use lk_agents_abi::resources::ResourceId;

/// One model of the chain: the core's own, or a plugin's seen through its host view. A host model
/// is wrapped afresh at each open: `HostRealtimeModel` keeps the session it opened.
#[derive(Clone)]
pub enum ChainModel {
    Native(Arc<dyn Realtime>),
    /// With the spawner its leg's tasks run on.
    Host(Arc<dyn HostRealtime>, Arc<dyn lk_agents_abi::Spawner>),
}

impl ChainModel {
    fn resource_id(&self) -> ResourceId {
        match self {
            Self::Native(model) => backend_resource(model.as_ref(), model.instance_id()),
            Self::Host(model, _) => backend_resource(model.as_ref(), model.instance_id()),
        }
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        match self {
            ChainModel::Native(m) => m.capabilities(),
            ChainModel::Host(h, _) => h.capabilities(),
        }
    }

    fn label(&self) -> String {
        match self {
            ChainModel::Native(m) => m.label(),
            ChainModel::Host(h, _) => h.label(),
        }
    }

    /// A session on this model; an open that outlasts `attempt_timeout` fails as a connect
    /// timeout.
    async fn open(
        &self,
        turn_detection_disabled: bool,
        anchor: WallAnchor,
        clock: &dyn lk_agents_abi::Clock,
        attempt_timeout: Option<Duration>,
    ) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let open = async {
            match self {
                ChainModel::Native(m) => m.session(turn_detection_disabled).await,
                // Given its anchor and spawner, it needs no environment.
                ChainModel::Host(h, spawner) => {
                    <HostRealtimeModel>::new(Arc::clone(h), anchor, Arc::clone(spawner))
                        .session(turn_detection_disabled)
                        .await
                }
            }
        };
        let Some(limit) = attempt_timeout else { return open.await };
        let deadline = clock.now().saturating_add(limit);
        tokio::select! {
            biased;
            opened = open => opened,
            () = clock.wait_until(deadline) => Err(ApiError::timed_out(TimeoutPhase::Connect).into()),
        }
    }
}

/// `RealtimeModelFallbackAdapter`: falls back between realtime models (or restarts one),
/// preserving the conversation. A swap is reported as `SessionSwapped`: the driver cuts what was
/// playing, gives the new model its frame (instructions, the conversation the caller heard,
/// tools), and asks again for a reply the swap cut (`_swap`).
pub struct RealtimeFallbackModel {
    children: Vec<ChainModel>,
    caps: RealtimeCapabilities,
    cooldown: f64,
    /// Caps each model's session open.
    attempt_timeout: Option<Duration>,
    regenerate_on_swap: bool,
    observer: Arc<dyn ChainObserver>,
    anchor: WallAnchor,
    clock: Arc<dyn lk_agents_abi::Clock>,
    sessions: Mutex<Vec<std::sync::Weak<FallbackSession>>>,
    /// The model serving the newest session, for the metrics label.
    active: Arc<std::sync::atomic::AtomicUsize>,
}

impl lk_agents_abi::llm::fallback::RealtimeFallbackModel for RealtimeFallbackModel {
    type Env = lk_agents_net::env::ProcessEnv;

    fn new(
        env: &Self::Env,
        models: Vec<lk_agents_abi::llm::fallback::RealtimeFallbackMember>,
        cooldown: f64,
        attempt_timeout: Option<f64>,
        regenerate_on_swap: bool,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, lk_agents_abi::llm::realtime::RealtimeConfigurationError> {
        use lk_agents_abi::llm::fallback::RealtimeFallbackMember;
        use lk_agents_abi::llm::realtime::RealtimeConfigurationError;
        let children = models
            .into_iter()
            .map(|model| {
                Ok(match model {
                    RealtimeFallbackMember::Native(model) => ChainModel::Native(model),
                    RealtimeFallbackMember::Host(model) => ChainModel::Host(
                        model,
                        (env.spawner)().map_err(lk_agents_abi::errors::HostError::new)?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, RealtimeConfigurationError>>()?;
        Self::new(
            children,
            cooldown,
            attempt_timeout,
            regenerate_on_swap,
            on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents)),
            env.clock.clone(),
        )
        .map_err(RealtimeConfigurationError::Invalid)
    }

    fn active_index(&self) -> usize {
        self.active_index()
    }
    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(self)
    }
    async fn restart(&self, switch_model: bool) {
        self.restart(switch_model).await;
    }
}

impl RealtimeFallbackModel {
    pub fn new(
        children: Vec<ChainModel>,
        cooldown: f64,
        attempt_timeout: Option<f64>,
        regenerate_on_swap: bool,
        observer: Arc<dyn ChainObserver>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Result<Self, String> {
        if children.is_empty() {
            return Err("at least one RealtimeModel instance must be provided.".to_owned());
        }
        let duration = |name: &str, seconds: f64| {
            Duration::try_from_secs_f64(seconds).map_err(|_| {
                format!("{name} must be a finite, nonnegative duration within the supported range")
            })
        };
        duration("cooldown", cooldown)?;
        let attempt_timeout =
            attempt_timeout.map(|seconds| duration("attempt_timeout", seconds)).transpose()?;
        let caps =
            merge_capabilities(&children.iter().map(ChainModel::capabilities).collect::<Vec<_>>())
                .map_err(|e| e.to_string())?;
        Ok(RealtimeFallbackModel {
            children,
            caps,
            cooldown,
            attempt_timeout,
            regenerate_on_swap,
            observer,
            anchor: WallAnchor::of(&*clock),
            clock,
            sessions: Mutex::new(Vec::new()),
            active: Arc::default(),
        })
    }

    pub fn active_index(&self) -> usize {
        self.active.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `restart_session(switch_model=)`: a fresh session on every live one, preserving the
    /// conversation.
    pub async fn restart(&self, switch_model: bool) {
        let live: Vec<_> =
            self.sessions.lock().iter().filter_map(std::sync::Weak::upgrade).collect();
        for s in live {
            s.request_restart(switch_model);
        }
    }
}

impl Realtime for RealtimeFallbackModel {
    fn label(&self) -> String {
        "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter".to_owned()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// A fresh session always starts on the primary, even after an earlier failover.
    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let first = self.children[0]
                .open(turn_detection_disabled, self.anchor, &*self.clock, self.attempt_timeout)
                .await?;
            let (restart_tx, restart_rx) = tokio::sync::watch::channel(false);
            self.active.store(0, std::sync::atomic::Ordering::Relaxed);
            let session = Arc::new(FallbackSession {
                children: self.children.clone(),
                priority: self.children.iter().map(ChainModel::resource_id).collect(),
                cooldown: Duration::from_secs_f64(self.cooldown),
                attempt_timeout: self.attempt_timeout,
                caps: self.caps,
                regenerate_on_swap: self.regenerate_on_swap,
                observer: Arc::clone(&self.observer),
                anchor: self.anchor,
                clock: Arc::clone(&self.clock),
                turn_detection_disabled,
                model_active: Arc::clone(&self.active),
                state: Mutex::new(SessionState {
                    active: first,
                    policy: FallbackPolicy::default(),
                    active_index: 0,
                    swapping: false,
                    fresh: false,
                    held: FallbackBacklog::default(),
                    events: VecDeque::new(),
                    generation: None,
                    closed: false,
                }),
                restart_tx,
                changed: tokio::sync::Notify::new(),
                restart_rx: tokio::sync::Mutex::new(restart_rx),
            });
            let mut sessions = self.sessions.lock();
            sessions.retain(|w| w.strong_count() > 0);
            sessions.push(Arc::downgrade(&session));
            Ok(session as Arc<dyn RealtimeSession>)
        })
    }
}
