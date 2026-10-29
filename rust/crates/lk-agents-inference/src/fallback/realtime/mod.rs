//! `RealtimeModelFallbackAdapter`: the realtime fallback on the shared chain machine. Unlike the
//! LLM/STT/TTS chains, failure is a pushed event, recovery is a cooldown, a healthy primary is not
//! preempted, and availability is per session. `capabilities` merges the models' capabilities,
//! `session` replaces a failed model's session and replays the conversation, and `backlog` bounds
//! the requests held while it does.

mod backlog;
mod capabilities;
mod session;
#[cfg(test)]
mod tests;

pub use backlog::{
    BacklogOverload, FALLBACK_PENDING_BYTES, FALLBACK_PENDING_REPLIES, FallbackBacklog,
};
pub use capabilities::{CapabilityMismatch, merge_capabilities};
pub use session::FallbackSession;
use session::SessionState;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::AvailabilityChanged;
use lk_agents_abi::llm::realtime::{
    Event, GenerationDone, GenerationStatus, HostRealtime, MessageDone, MessageModalities,
    Modalities, RealtimeModelError, TokenUsage,
};
use lk_agents_abi::{ChatContext, Realtime, RealtimeCommand, RealtimeSession, WallAnchor};

use crate::fallback::machine::{ChainEffect, ChainEvent, ChainRequest, Modality};
use crate::fallback::{ChainObserver, ChainPolicy, FallbackPolicy, backend_resource};
use crate::realtime::host::HostRealtimeModel;
use lk_agents_abi::llm::realtime::RealtimeCapabilities;
use lk_agents_abi::resources::ResourceId;

/// One model of the chain: the core's own, or a plugin's seen through its host view. A host model
/// is wrapped afresh at each open: `HostRealtimeModel` keeps the session it opened.
#[derive(Clone)]
pub enum RealtimeChild {
    Native(Arc<dyn Realtime>),
    /// With the spawner its leg's tasks run on.
    Host(Arc<dyn HostRealtime>, Arc<dyn lk_agents_abi::Spawner>),
}

impl RealtimeChild {
    fn resource_id(&self) -> ResourceId {
        match self {
            Self::Native(model) => backend_resource(model.as_ref(), model.instance_id()),
            Self::Host(model, _) => backend_resource(model.as_ref(), model.instance_id()),
        }
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        match self {
            RealtimeChild::Native(m) => m.capabilities(),
            RealtimeChild::Host(h, _) => h.capabilities(),
        }
    }

    fn label(&self) -> String {
        match self {
            RealtimeChild::Native(m) => m.label(),
            RealtimeChild::Host(h, _) => h.label(),
        }
    }

    async fn open(
        &self,
        turn_detection_disabled: bool,
        anchor: WallAnchor,
    ) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        match self {
            RealtimeChild::Native(m) => m.session(turn_detection_disabled).await,
            // Given its anchor and spawner, it needs no environment.
            RealtimeChild::Host(h, spawner) => {
                <HostRealtimeModel>::new(Arc::clone(h), anchor, Arc::clone(spawner))
                    .session(turn_detection_disabled)
                    .await
            }
        }
    }
}

/// `RealtimeModelFallbackAdapter`: falls back between realtime models (or restarts one),
/// preserving the conversation. A swap is reported as `SessionSwapped`: the driver cuts what was
/// playing, gives the new model its frame (instructions, the conversation the caller heard,
/// tools), and asks again for a reply the swap cut (`_swap`).
pub struct RealtimeFallbackModel {
    children: Vec<RealtimeChild>,
    caps: RealtimeCapabilities,
    cooldown: f64,
    regenerate_on_swap: bool,
    observer: Arc<dyn ChainObserver>,
    anchor: WallAnchor,
    clock: Arc<dyn lk_agents_abi::Clock>,
    sessions: Mutex<Vec<std::sync::Weak<FallbackSession>>>,
    /// The model serving the newest session, for the metrics label.
    active: Arc<std::sync::atomic::AtomicUsize>,
}

impl RealtimeFallbackModel {
    pub fn new(
        children: Vec<RealtimeChild>,
        cooldown: f64,
        regenerate_on_swap: bool,
        observer: Arc<dyn ChainObserver>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Result<Self, String> {
        if children.is_empty() {
            return Err("at least one RealtimeModel instance must be provided.".to_owned());
        }
        if std::time::Duration::try_from_secs_f64(cooldown).is_err() {
            return Err(
                "cooldown must be a finite, nonnegative duration within the supported range"
                    .to_owned(),
            );
        }
        let caps = merge_capabilities(
            &children.iter().map(RealtimeChild::capabilities).collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        Ok(RealtimeFallbackModel {
            children,
            caps,
            cooldown,
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
        let live: Vec<_> = self
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .collect();
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
            let first = self.children[0].open(turn_detection_disabled, self.anchor).await?;
            let (restart_tx, restart_rx) = tokio::sync::watch::channel(false);
            self.active.store(0, std::sync::atomic::Ordering::Relaxed);
            let session = Arc::new(FallbackSession {
                children: self.children.clone(),
                priority: self.children.iter().map(RealtimeChild::resource_id).collect(),
                cooldown: std::time::Duration::from_secs_f64(self.cooldown),
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
            let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
            sessions.retain(|w| w.strong_count() > 0);
            sessions.push(Arc::downgrade(&session));
            Ok(session as Arc<dyn RealtimeSession>)
        })
    }
}
