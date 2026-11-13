mod backlog;
mod capabilities;
mod session;
#[cfg(test)]
#[path = "realtime_tests.rs"]
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

use lk_agents_abi::errors::{ApiError, ProviderError, Timeout};
use lk_agents_abi::llm::fallback::AvailabilityChanged;
use lk_agents_abi::llm::realtime::{
    Event, GenerationDone, GenerationStatus, HostRealtime, MessageDone, MessageModalities,
    Modalities, RealtimeConfigurationError, RealtimeModelError,
};
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::{ChatContext, Realtime, RealtimeCommand, RealtimeSession, WallAnchor};
use lk_agents_net::env::ProcessError;
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
    /// With the spawner its leg's tasks run on, resolved when it opens.
    Host(Arc<dyn HostRealtime>, fn() -> Result<Arc<dyn lk_agents_abi::Spawner>, ProcessError>),
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
        cx: lk_agents_abi::telemetry::Cx,
        turn_detection_disabled: bool,
        tools: Option<lk_agents_abi::HostRef>,
        anchor: WallAnchor,
        clock: &dyn lk_agents_abi::Clock,
        attempt_timeout: Option<Duration>,
    ) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let open = async {
            match self {
                ChainModel::Native(m) => m.session(cx, turn_detection_disabled, tools).await,
                // Given its anchor and spawner, it needs no environment.
                ChainModel::Host(h, spawner) => {
                    let spawner = spawner().map_err(ApiError::from)?;
                    <HostRealtimeModel>::new(Arc::clone(h), anchor, spawner)
                        .session(cx, turn_detection_disabled, tools)
                        .await
                }
            }
        };
        let Some(limit) = attempt_timeout else { return open.await };
        let deadline = clock.now().saturating_add(limit);
        tokio::select! {
            biased;
            opened = open => opened,
            () = clock.wait_until(deadline) => Err(ApiError::timed_out(Timeout::Connect).into()),
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
    /// The name logs give it: its host class's when its host built it.
    label: String,
}

impl lk_agents_abi::llm::fallback::RealtimeFallbackModel for RealtimeFallbackModel {
    type Env = lk_agents_net::env::ProcessEnv;

    fn new(
        env: &Self::Env,
        label: String,
        observer: Arc<dyn ChainObserver>,
        models: Vec<lk_agents_abi::llm::fallback::RealtimeLeg>,
        cooldown: f64,
        attempt_timeout: Option<f64>,
        regenerate_on_swap: bool,
    ) -> Result<Self, RealtimeConfigurationError> {
        use lk_agents_abi::llm::fallback::RealtimeLeg;
        let children = models
            .into_iter()
            .map(|model| match model {
                RealtimeLeg::Native(model) => ChainModel::Native(model),
                RealtimeLeg::Host(model) => ChainModel::Host(model, env.spawner),
            })
            .collect();
        let clock = env.clock.clone();
        let chain =
            Self::new(children, cooldown, attempt_timeout, regenerate_on_swap, observer, clock)?;
        Ok(RealtimeFallbackModel { label, ..chain })
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
    ) -> Result<Self, RealtimeConfigurationError> {
        let invalid = RealtimeConfigurationError::Invalid;
        if children.is_empty() {
            return Err(invalid(
                "at least one RealtimeModel instance must be provided.".to_owned(),
            ));
        }
        let duration = |name: &str, seconds: f64| {
            Duration::try_from_secs_f64(seconds).map_err(|_| {
                invalid(format!(
                    "{name} must be a finite, nonnegative duration within the supported range"
                ))
            })
        };
        duration("cooldown", cooldown)?;
        let attempt_timeout =
            attempt_timeout.map(|seconds| duration("attempt_timeout", seconds)).transpose()?;
        let caps =
            merge_capabilities(&children.iter().map(ChainModel::capabilities).collect::<Vec<_>>())
                .map_err(|e| invalid(e.to_string()))?;
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
            label: NativeLabel::RealtimeFallback.as_str().to_owned(),
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
        self.label.clone()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// A fresh session always starts on the primary, even after an earlier failover.
    fn session(
        &self,
        cx: lk_agents_abi::telemetry::Cx,
        turn_detection_disabled: bool,
        tools: Option<lk_agents_abi::HostRef>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let first = self.children[0]
                .open(
                    cx.clone(),
                    turn_detection_disabled,
                    tools.clone(),
                    self.anchor,
                    &*self.clock,
                    self.attempt_timeout,
                )
                .await?;
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
                tools,
                cx,
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
                    restart: None,
                }),
                changed: lk_agents_net::signal::Shared::new(0),
                consumer: tokio::sync::Mutex::new(()),
            });
            let mut sessions = self.sessions.lock();
            sessions.retain(|w| w.strong_count() > 0);
            sessions.push(Arc::downgrade(&session));
            Ok(session as Arc<dyn RealtimeSession>)
        })
    }
}
