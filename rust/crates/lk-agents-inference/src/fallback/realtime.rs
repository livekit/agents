//! `llm/realtime_fallback_adapter.py`'s capability merge. The adapter's decisions are
//! [`super::realtime_policy`]; the host owns the model objects and the swap task.

/// The provider record, from the crate that declares it.
pub use lk_agents_abi::llm::realtime::RealtimeCapabilities;

/// Which hard capability disagreed, and the two values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityMismatch {
    pub name: &'static str,
    pub first: bool,
    pub other: bool,
}

impl std::fmt::Display for CapabilityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "all realtime models must agree on `{}` to be used in a RealtimeModelFallbackAdapter, got {} and {}",
            self.name, self.first, self.other
        )
    }
}
impl std::error::Error for CapabilityMismatch {}

/// `_merge_capabilities` (`:70-86`). `audio_output` and `turn_detection` must agree (they shape
/// the pipeline at activity start); the rest are ANDed. Raises `ValueError` naming the capability
/// and both values, as the reference does.
#[lk_agents_abi::host(export(name = "merge_realtime_capabilities"))]
pub fn merge_capabilities_of(caps: Vec<RealtimeCapabilities>) -> Result<RealtimeCapabilities, CapabilityMismatch> {
    merge_capabilities(&caps)
}

/// [`merge_capabilities_of`] over a borrowed list.
pub fn merge_capabilities(
    models: &[RealtimeCapabilities],
) -> Result<RealtimeCapabilities, CapabilityMismatch> {
    let Some(first) = models.first().copied() else {
        return Ok(RealtimeCapabilities::default());
    };
    for c in &models[1..] {
        if c.audio_output != first.audio_output {
            return Err(CapabilityMismatch {
                name: "audio_output",
                first: first.audio_output,
                other: c.audio_output,
            });
        }
        if c.turn_detection != first.turn_detection {
            return Err(CapabilityMismatch {
                name: "turn_detection",
                first: first.turn_detection,
                other: c.turn_detection,
            });
        }
    }
    let all = |f: fn(&RealtimeCapabilities) -> bool| models.iter().all(f);
    Ok(RealtimeCapabilities {
        audio_output: first.audio_output,
        turn_detection: first.turn_detection,
        message_truncation: all(|c| c.message_truncation),
        user_transcription: all(|c| c.user_transcription),
        manual_function_calls: all(|c| c.manual_function_calls),
        auto_tool_reply_generation: all(|c| c.auto_tool_reply_generation),
        mutable_chat_context: all(|c| c.mutable_chat_context),
        mutable_instructions: all(|c| c.mutable_instructions),
        mutable_tools: all(|c| c.mutable_tools),
        per_response_tool_choice: all(|c| c.per_response_tool_choice),
        supports_say: all(|c| c.supports_say),
        can_disable_turn_detection: all(|c| c.can_disable_turn_detection),
        auto_turn_reply_generation: all(|c| c.auto_turn_reply_generation),
    })
}


// ---------------------------------------------------------------------------------------------
// The adapter's session: `_FallbackRealtimeSession`
// ---------------------------------------------------------------------------------------------

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::AvailabilityChanged;
use lk_agents_abi::llm::realtime::{
    Event, GenerateReply, GenerationDone, GenerationStatus, HostRealtime, HostRealtimeModel,
    MessageDone, MessageModalities, Modalities, RealtimeModelError, SessionReconnected, TokenUsage,
};
use lk_agents_abi::{ChatContext, Realtime, RealtimeCommand, RealtimeSession, WallAnchor};

use crate::fallback::{ChainObserver, RealtimePolicy};

/// One model of the chain: the core's own, or a plugin's seen through its host view. A host model
/// is wrapped afresh at each open: `HostRealtimeModel` keeps the session it opened.
#[derive(Clone)]
pub enum RealtimeChild {
    Native(Arc<dyn Realtime>),
    /// With the spawner its leg's tasks run on.
    Host(Arc<dyn HostRealtime>, Arc<dyn lk_agents_abi::Spawner>),
}

impl RealtimeChild {
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
            RealtimeChild::Host(h, spawner) => {
                HostRealtimeModel::new(Arc::clone(h), anchor, Arc::clone(spawner))
                    .session(turn_detection_disabled)
                    .await
            }
        }
    }
}

/// `RealtimeModelFallbackAdapter`: falls back between realtime models (or restarts one),
/// preserving the conversation. A swap is reported as `SessionReconnected`, so the session replays
/// its frame onto the new model; a reply cut by the swap is asked for again.
pub struct RealtimeFallbackModel {
    children: Vec<RealtimeChild>,
    caps: RealtimeCapabilities,
    cooldown: f64,
    regenerate_on_swap: bool,
    observer: Arc<dyn ChainObserver>,
    anchor: WallAnchor,
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
        anchor: WallAnchor,
    ) -> Result<Self, String> {
        if children.is_empty() {
            return Err("at least one RealtimeModel instance must be provided.".to_owned());
        }
        let caps = merge_capabilities(&children.iter().map(RealtimeChild::capabilities).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?;
        Ok(RealtimeFallbackModel {
            children,
            caps,
            cooldown,
            regenerate_on_swap,
            observer,
            anchor,
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

#[async_trait::async_trait]
impl Realtime for RealtimeFallbackModel {
    fn label(&self) -> String {
        "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter".to_owned()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// A fresh session always starts on the primary, even after an earlier failover.
    async fn session(&self, turn_detection_disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let first = self.children[0].open(turn_detection_disabled, self.anchor).await?;
        let (restart_tx, restart_rx) = tokio::sync::mpsc::unbounded_channel();
        self.active.store(0, std::sync::atomic::Ordering::Relaxed);
        let session = Arc::new(FallbackSession {
            children: self.children.clone(),
            caps: self.caps,
            regenerate_on_swap: self.regenerate_on_swap,
            observer: Arc::clone(&self.observer),
            anchor: self.anchor,
            turn_detection_disabled,
            model_active: Arc::clone(&self.active),
            state: Mutex::new(SessionState {
                active: first,
                policy: RealtimePolicy::new(self.children.len(), std::time::Duration::from_secs_f64(self.cooldown.max(0.0))),
                swapping: false,
                queued: Vec::new(),
                events: VecDeque::new(),
                generation: None,
                own: HashSet::new(),
                regenerate: false,
                closed: false,
            }),
            restart_tx,
            restart_rx: tokio::sync::Mutex::new(restart_rx),
        });
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        sessions.retain(|w| w.strong_count() > 0);
        sessions.push(Arc::downgrade(&session));
        Ok(session as Arc<dyn RealtimeSession>)
    }
}

/// The generation in flight: its messages, and which settled their modalities.
struct OpenGeneration {
    response_id: String,
    messages: Vec<(String, bool)>,
}

struct SessionState {
    active: Arc<dyn RealtimeSession>,
    policy: RealtimePolicy,
    /// Audio and video are dropped during a swap: replaying them would lag the model.
    swapping: bool,
    /// Other commands sent during a swap, for the new session.
    queued: Vec<RealtimeCommand>,
    events: VecDeque<Event>,
    generation: Option<OpenGeneration>,
    /// The replies this adapter asked for itself: the session sees them as the server's.
    own: HashSet<String>,
    /// Ask for a reply once the session has replayed its frame onto the new model.
    regenerate: bool,
    closed: bool,
}

pub struct FallbackSession {
    children: Vec<RealtimeChild>,
    caps: RealtimeCapabilities,
    regenerate_on_swap: bool,
    observer: Arc<dyn ChainObserver>,
    anchor: WallAnchor,
    turn_detection_disabled: bool,
    model_active: Arc<std::sync::atomic::AtomicUsize>,
    state: Mutex<SessionState>,
    restart_tx: tokio::sync::mpsc::UnboundedSender<bool>,
    /// Polled by `next_event` only (one consumer).
    restart_rx: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<bool>>,
}

/// `time.time()`: the policy's cooldowns compare these with each other only.
fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl FallbackSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, SessionState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn request_restart(&self, switch_model: bool) {
        let _ = self.restart_tx.send(switch_model);
    }

    fn announce(&self, changes: Vec<crate::fallback::AvailabilityChange>) {
        for c in changes {
            let label = self.children[c.index].label();
            self.observer.availability_changed(AvailabilityChanged { index: c.index, label, available: c.available });
        }
    }

    /// The generation a dead session leaves open, closed as the child would on its close.
    fn close_generation(&self, st: &mut SessionState) -> bool {
        let Some(g) = st.generation.take() else { return false };
        let modalities = if self.caps.audio_output { Modalities::AudioAndText } else { Modalities::TextOnly };
        let response_id = Some(g.response_id.clone());
        for (message_id, settled) in g.messages {
            if !settled {
                st.events.push_back(Event::MessageModalities(MessageModalities {
                    message_id: message_id.clone(),
                    response_id: response_id.clone(),
                    modalities,
                }));
            }
            st.events.push_back(Event::MessageDone(MessageDone { message_id, response_id: response_id.clone() }));
        }
        st.events.push_back(Event::GenerationDone(GenerationDone {
            response_id: g.response_id,
            status: GenerationStatus::Cancelled,
            usage: TokenUsage::default(),
        }));
        true
    }

    /// `_swap`: close the active child, bring up `target` (cascading to the next available model
    /// on failure), hand it what was sent meanwhile, and report the reconnect.
    async fn swap(&self, mut target: usize) {
        let (old, was_generating) = {
            let mut st = self.lock();
            st.swapping = true;
            let was = self.close_generation(&mut st);
            (Arc::clone(&st.active), was)
        };
        old.aclose().await;
        let opened = loop {
            match self.children[target].open(self.turn_detection_disabled, self.anchor).await {
                Ok(s) => break Ok(s),
                Err(e) => {
                    log::error!(target: "livekit.agents", "failed to start realtime model on swap, trying next: {e}");
                    let (next, changes) = {
                        let mut st = self.lock();
                        let mut changes: Vec<_> = st.policy.set_available(target, false, now()).into_iter().collect();
                        let (next, swept) = st.policy.next_available_index(now(), false);
                        changes.extend(swept);
                        (next, changes)
                    };
                    self.announce(changes);
                    match next {
                        Some(n) => target = n,
                        None => break Err(e),
                    }
                }
            }
        };
        let mut st = self.lock();
        st.swapping = false;
        match opened {
            Ok(session) => {
                for cmd in std::mem::take(&mut st.queued) {
                    session.send(cmd);
                }
                st.active = session;
                st.policy.set_active_index(target);
                self.model_active.store(target, std::sync::atomic::Ordering::Relaxed);
                st.events.push_back(Event::SessionReconnected(SessionReconnected {}));
                st.regenerate = was_generating && self.regenerate_on_swap;
            }
            // Every model failed to start: escalate so the session closes.
            Err(e) => {
                st.events.push_back(Event::Error(RealtimeModelError {
                    timestamp: lk_agents_abi::Ts::ZERO,
                    label: "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter".to_owned(),
                    message: e.to_string(),
                    recoverable: false,
                    req_id: None,
                    code: None,
                }));
            }
        }
    }

    /// Track the generation in flight, and present this adapter's own replies as the server's.
    fn observe(&self, st: &mut SessionState, ev: Event) -> Option<Event> {
        match ev {
            Event::GenerationCreated(mut g) => {
                st.generation = Some(OpenGeneration { response_id: g.response_id.clone(), messages: Vec::new() });
                if g.req_id.as_ref().is_some_and(|id| st.own.remove(id)) {
                    g.req_id = None;
                }
                Some(Event::GenerationCreated(g))
            }
            Event::GenerateReplyFailed(f) if st.own.remove(&f.req_id) => None,
            Event::MessageStarted(m) => {
                if let Some(g) = st.generation.as_mut() {
                    g.messages.push((m.message_id.clone(), false));
                }
                Some(Event::MessageStarted(m))
            }
            Event::MessageModalities(m) => {
                if let Some(g) = st.generation.as_mut()
                    && let Some(entry) = g.messages.iter_mut().find(|(id, _)| *id == m.message_id)
                {
                    entry.1 = true;
                }
                Some(Event::MessageModalities(m))
            }
            Event::MessageDone(m) => {
                if let Some(g) = st.generation.as_mut() {
                    g.messages.retain(|(id, _)| *id != m.message_id);
                }
                Some(Event::MessageDone(m))
            }
            Event::GenerationDone(d) => {
                st.generation = None;
                Some(Event::GenerationDone(d))
            }
            other => Some(other),
        }
    }
}

#[async_trait::async_trait]
impl RealtimeSession for FallbackSession {
    /// The adapter's merged capabilities: the session's pipeline was shaped by them.
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        let mut st = self.lock();
        if st.closed {
            return;
        }
        if st.swapping {
            match cmd {
                RealtimeCommand::PushAudio(_) | RealtimeCommand::PushItem(_) => {}
                // Dropped: the swap replays the conversation afterwards. Answered, so the
                // session does not wait out its timeout.
                RealtimeCommand::UpdateChatCtx(c) => {
                    st.events.push_back(Event::ChatCtxUpdated { req_id: c.req_id, rejected: vec![] });
                }
                other => st.queued.push(other),
            }
            return;
        }
        let replayed_tools = matches!(cmd, RealtimeCommand::UpdateTools(_));
        st.active.send(cmd);
        // The replay ends with the tools: re-issue the reply the swap cut.
        if replayed_tools && std::mem::take(&mut st.regenerate) {
            let req_id = crate::ids::shortuuid("response_create_");
            st.own.insert(req_id.clone());
            st.active.send(RealtimeCommand::GenerateReply(GenerateReply { req_id, instructions: None, tool_choice: None }));
        }
    }

    async fn next_event(&self) -> Option<Event> {
        loop {
            let active = {
                let mut st = self.lock();
                if let Some(ev) = st.events.pop_front() {
                    return Some(ev);
                }
                if st.closed {
                    return None;
                }
                Arc::clone(&st.active)
            };
            let mut restart_rx = self.restart_rx.lock().await;
            let ev = tokio::select! {
                ev = active.next_event() => ev,
                Some(switch_model) = restart_rx.recv() => {
                    let (target, changes) = self.lock().policy.restart_target(switch_model, now());
                    self.announce(changes);
                    self.swap(target).await;
                    continue;
                }
            };
            match ev {
                None => {
                    if !Arc::ptr_eq(&active, &self.lock().active) {
                        continue;
                    }
                    return None;
                }
                Some(Event::Error(e)) if !e.recoverable => {
                    let plan = self.lock().policy.on_child_error(false, false, now());
                    self.announce(plan.changes);
                    match plan.swap_to {
                        Some(target) => {
                            // While a fallback remains the error is recoverable.
                            self.lock().events.push_back(Event::Error(RealtimeModelError { recoverable: true, ..e }));
                            self.swap(target).await;
                        }
                        None => return Some(Event::Error(e)),
                    }
                }
                Some(ev) => {
                    if let Some(ev) = self.observe(&mut self.lock(), ev) {
                        return Some(ev);
                    }
                }
            }
        }
    }

    async fn chat_ctx(&self) -> ChatContext {
        let active = Arc::clone(&self.lock().active);
        active.chat_ctx().await
    }

    async fn aclose(&self) {
        let active = {
            let mut st = self.lock();
            st.closed = true;
            Arc::clone(&st.active)
        };
        active.aclose().await;
    }
}

/// `_core.RealtimeFallbackModel`: `RealtimeModelFallbackAdapter`'s engine, the session's realtime
/// model when handed to it. Hand-written: its children are either kind of model.
#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use lk_agents_abi::llm::realtime::HostRealtime;
    use lk_agents_abi::python::{HostProxy, check_pid, native};
    use lk_agents_abi::{Realtime, WallAnchor};
    use pyo3::prelude::*;

    use super::{RealtimeChild, RealtimeFallbackModel};
    use crate::fallback::ChainObserver;
    use crate::python::register_class;

    #[pyclass(frozen, name = "RealtimeFallbackModel", module = "livekit.agents._core")]
    pub struct PyRealtimeFallbackModel {
        inner: Arc<RealtimeFallbackModel>,
    }

    register_class!(PyRealtimeFallbackModel);

    lk_agents_abi::python::inventory::submit! {
        lk_agents_abi::python::HostNative {
            trait_id: || std::any::TypeId::of::<dyn Realtime>(),
            read: |obj| {
                let n = obj.extract::<PyRef<'_, PyRealtimeFallbackModel>>().ok()?;
                let v: Arc<dyn Realtime> = n.inner.clone();
                Some(Box::new(v) as Box<dyn std::any::Any>)
            },
        }
    }

    #[pymethods]
    impl PyRealtimeFallbackModel {
        /// `models` in order: each the core's own model (`_core.NativeRealtimeModel`) or a host
        /// realtime model's view. `on_availability._core_availability_changed(ev)` hears each
        /// change. Raises `ValueError` when the models disagree on a hard capability.
        #[new]
        #[pyo3(signature = (models, *, cooldown = 10.0, regenerate_on_swap = true, on_availability = None))]
        fn new(
            models: Vec<Bound<'_, PyAny>>,
            cooldown: f64,
            regenerate_on_swap: bool,
            on_availability: Option<Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            check_pid()?;
            let children = models
                .iter()
                .map(|m| match native::<dyn Realtime>(m) {
                    Some(n) => Ok(RealtimeChild::Native(n)),
                    None => <dyn HostRealtime as HostProxy>::from_host(m, None).map(|h| {
                        let spawner = <Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient();
                        RealtimeChild::Host(h, spawner)
                    }),
                })
                .collect::<PyResult<Vec<_>>>()?;
            let observer: Arc<dyn ChainObserver> = match on_availability {
                Some(o) => <dyn ChainObserver as HostProxy>::from_host(&o, None)?,
                None => Arc::new(crate::fallback::IgnoreEvents),
            };
            let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
            let inner = RealtimeFallbackModel::new(children, cooldown, regenerate_on_swap, observer, WallAnchor::of(&*clock))
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok(PyRealtimeFallbackModel { inner: Arc::new(inner) })
        }

        /// The model serving the newest session.
        #[getter]
        fn active_index(&self) -> usize {
            self.inner.active_index()
        }

        /// The merged capabilities.
        fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            Realtime::capabilities(&*self.inner).into_pyobject(py).map(Bound::into_any)
        }

        /// `restart_session(switch_model=)` on every live session.
        #[pyo3(signature = (*, switch_model = false))]
        fn restart<'py>(&self, py: Python<'py>, switch_model: bool) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            lk_agents_abi::python::future_into_py(py, async move {
                inner.restart(switch_model).await;
                Ok(())
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(audio_output: bool, supports_say: bool) -> RealtimeCapabilities {
        RealtimeCapabilities { audio_output, turn_detection: true, supports_say, ..Default::default() }
    }

    // Catches: ANDing a hard capability instead of rejecting a mismatch, a mismatch that does not
    // name the capability and both values, or a soft capability ORed.
    #[test]
    fn hard_capabilities_must_agree_and_soft_ones_are_anded() {
        let merged = merge_capabilities(&[caps(true, true), caps(true, false)]).expect("hard caps agree");
        assert!(merged.audio_output);
        assert!(!merged.supports_say, "soft caps are the conservative AND");

        let err = merge_capabilities(&[caps(true, true), caps(false, true)])
            .expect_err("audio_output differs");
        assert_eq!(
            err.to_string(),
            "all realtime models must agree on `audio_output` to be used in a RealtimeModelFallbackAdapter, got true and false"
        );

        let td = RealtimeCapabilities { turn_detection: false, ..caps(true, true) };
        let err = merge_capabilities(&[caps(true, true), td]).expect_err("turn_detection differs");
        assert_eq!(
            err.to_string(),
            "all realtime models must agree on `turn_detection` to be used in a RealtimeModelFallbackAdapter, got true and false"
        );
    }

    // Catches: a field `merge_capabilities` forgets to carry (a one-model merge must be the identity).
    #[test]
    fn merging_one_model_carries_every_field() {
        let all = RealtimeCapabilities {
            message_truncation: true,
            turn_detection: true,
            user_transcription: true,
            auto_tool_reply_generation: true,
            audio_output: true,
            manual_function_calls: true,
            can_disable_turn_detection: true,
            mutable_chat_context: true,
            mutable_instructions: true,
            mutable_tools: true,
            per_response_tool_choice: true,
            supports_say: true,
            auto_turn_reply_generation: true,
        };
        assert_eq!(merge_capabilities(&[all]).expect("one model cannot disagree"), all);
    }
}

#[cfg(test)]
mod fallback_session_tests {
    use super::*;
    use lk_agents_abi::llm::realtime::{GenerationCreated, MessageStarted};

    /// A scripted session: events to hand out, then parked; every command recorded.
    struct Scripted {
        events: Mutex<VecDeque<Event>>,
        sent: Mutex<Vec<String>>,
        closed: std::sync::atomic::AtomicBool,
    }

    #[async_trait::async_trait]
    impl RealtimeSession for Scripted {
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        fn send(&self, cmd: RealtimeCommand) {
            let label = match &cmd {
                RealtimeCommand::GenerateReply(g) => format!("generate_reply:{}", g.req_id),
                other => format!("{other:?}").split('(').next().unwrap_or_default().to_owned(),
            };
            self.sent.lock().unwrap().push(label);
        }
        async fn next_event(&self) -> Option<Event> {
            let next = self.events.lock().unwrap().pop_front();
            match next {
                Some(ev) => Some(ev),
                None => std::future::pending().await,
            }
        }
        async fn chat_ctx(&self) -> ChatContext {
            ChatContext::default()
        }
        async fn aclose(&self) {
            self.closed.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    struct Model(Arc<Scripted>);

    #[async_trait::async_trait]
    impl Realtime for Model {
        fn label(&self) -> String {
            "scripted".into()
        }
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        async fn session(&self, _td: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
            Ok(Arc::clone(&self.0) as Arc<dyn RealtimeSession>)
        }
    }

    fn scripted(events: Vec<Event>) -> Arc<Scripted> {
        Arc::new(Scripted { events: Mutex::new(events.into()), sent: Mutex::default(), closed: Default::default() })
    }

    fn fatal() -> Event {
        Event::Error(RealtimeModelError {
            timestamp: lk_agents_abi::Ts::ZERO,
            label: "scripted".into(),
            message: "boom".into(),
            recoverable: false,
            req_id: None,
            code: None,
        })
    }

    fn kind(ev: &Event) -> String {
        format!("{ev:?}").split('(').next().unwrap_or_default().split(' ').next().unwrap_or_default().to_owned()
    }

    // Catches: a fatal child error passed on as fatal while a fallback exists, the dead child's
    // open generation left open, no reconnect for the session to replay onto, or the cut reply
    // not asked for again after the replay (and surfacing as an answer to an unknown request).
    #[tokio::test]
    async fn a_fatal_error_mid_reply_swaps_to_the_next_model_and_asks_for_the_reply_again() {
        let primary = scripted(vec![
            Event::GenerationCreated(GenerationCreated { response_id: "r1".into(), req_id: None }),
            Event::MessageStarted(MessageStarted { message_id: "m1".into(), response_id: Some("r1".into()) }),
            fatal(),
        ]);
        let backup = scripted(vec![]);
        let changes = Arc::new(Mutex::new(Vec::new()));
        struct Record(Arc<Mutex<Vec<(usize, bool)>>>);
        impl ChainObserver for Record {
            fn availability_changed(&self, ev: AvailabilityChanged) {
                self.0.lock().unwrap().push((ev.index, ev.available));
            }
        }
        let model = RealtimeFallbackModel::new(
            vec![
                RealtimeChild::Native(Arc::new(Model(Arc::clone(&primary)))),
                RealtimeChild::Native(Arc::new(Model(Arc::clone(&backup)))),
            ],
            10.0,
            true,
            Arc::new(Record(Arc::clone(&changes))),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();

        let mut kinds = Vec::new();
        for _ in 0..7 {
            let ev = session.next_event().await.unwrap();
            if let Event::Error(e) = &ev {
                assert!(e.recoverable, "a fallback remains: the error is rewritten recoverable");
            }
            kinds.push(kind(&ev));
        }
        assert_eq!(
            kinds,
            [
                "GenerationCreated",
                "MessageStarted",
                "Error",
                "MessageModalities",
                "MessageDone",
                "GenerationDone",
                "SessionReconnected"
            ]
        );
        assert!(primary.closed.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(*changes.lock().unwrap(), [(0, false)]);
        assert_eq!(model.active_index(), 1);

        // The session replays its frame; the tools end it, and the reply goes again.
        session.send(RealtimeCommand::UpdateInstructions(lk_agents_abi::UpdateInstructions { instructions: "hi".into() }));
        session.send(RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools { tools: vec![] }));
        let sent = backup.sent.lock().unwrap().clone();
        assert_eq!(sent[..2], ["UpdateInstructions".to_owned(), "UpdateTools".to_owned()]);
        let req_id = sent[2].strip_prefix("generate_reply:").expect("the reply is asked for again").to_owned();

        backup.events.lock().unwrap().push_back(Event::GenerationCreated(GenerationCreated {
            response_id: "r2".into(),
            req_id: Some(req_id),
        }));
        match session.next_event().await.unwrap() {
            Event::GenerationCreated(g) => assert_eq!(g.req_id, None, "the session sees the server's reply"),
            other => panic!("{other:?}"),
        }
    }
}
