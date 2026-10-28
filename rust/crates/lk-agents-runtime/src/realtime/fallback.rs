//! Realtime fallback model resources and asynchronous session replacement.

// ---------------------------------------------------------------------------------------------
// The adapter's session: `_FallbackRealtimeSession`
// ---------------------------------------------------------------------------------------------

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::AvailabilityChanged;
use lk_agents_abi::llm::realtime::{
    Event, GenerationDone, GenerationStatus, HostRealtime, HostRealtimeModel, MessageDone,
    MessageModalities, Modalities, RealtimeModelError, TokenUsage,
};
use lk_agents_abi::{ChatContext, Realtime, RealtimeCommand, RealtimeSession, WallAnchor};

use lk_agents_inference::fallback::realtime::{
    FallbackBacklog, RealtimeCapabilities, merge_capabilities,
};
use lk_agents_inference::fallback::{ChainObserver, RealtimePolicy};

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
    async fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let first = self.children[0].open(turn_detection_disabled, self.anchor).await?;
        let (restart_tx, restart_rx) = tokio::sync::watch::channel(false);
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
                policy: RealtimePolicy::new(
                    self.children.len(),
                    std::time::Duration::from_secs_f64(self.cooldown),
                ),
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
    /// The old session is closing and the new one opening: everything sent names the old one, or
    /// is superseded by the frame the driver gives the new one, and is dropped (audio would lag).
    swapping: bool,
    /// The new session waits for its frame, which ends with the tools: the requests asked
    /// meanwhile are held until then, so none reaches a model without its conversation.
    fresh: bool,
    held: FallbackBacklog,
    events: VecDeque<Event>,
    generation: Option<OpenGeneration>,
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
    restart_tx: tokio::sync::watch::Sender<bool>,
    changed: tokio::sync::Notify,
    /// Polled by `next_event` only (one consumer).
    restart_rx: tokio::sync::Mutex<tokio::sync::watch::Receiver<bool>>,
}

// `time.time()`: the policy's cooldowns compare these with each other only.
use lk_agents_abi::unix_now as now;

impl FallbackSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, SessionState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn request_restart(&self, switch_model: bool) {
        let _ = self.restart_tx.send(switch_model);
    }

    fn announce(&self, changes: Vec<lk_agents_inference::fallback::AvailabilityChange>) {
        for c in changes {
            let label = self.children[c.index].label();
            self.observer.availability_changed(AvailabilityChanged {
                index: c.index,
                label,
                available: c.available,
            });
        }
    }

    /// The generation a dead session leaves open, closed as the child would on its close.
    fn close_generation(&self, st: &mut SessionState) {
        let Some(g) = st.generation.take() else { return };
        let modalities =
            if self.caps.audio_output { Modalities::AudioAndText } else { Modalities::TextOnly };
        let response_id = Some(g.response_id.clone());
        for (message_id, settled) in g.messages {
            if !settled {
                st.events.push_back(Event::MessageModalities(MessageModalities {
                    message_id: message_id.clone(),
                    response_id: response_id.clone(),
                    modalities,
                }));
            }
            st.events.push_back(Event::MessageDone(MessageDone {
                message_id,
                response_id: response_id.clone(),
            }));
        }
        st.events.push_back(Event::GenerationDone(GenerationDone {
            response_id: g.response_id,
            status: GenerationStatus::Cancelled,
            usage: TokenUsage::default(),
        }));
    }

    /// `_swap`: close the active child, bring up `target` (cascading to the next available model
    /// on failure), hand it what was sent meanwhile, and report the reconnect.
    async fn swap(&self, mut target: usize) {
        let old = {
            let mut st = self.lock();
            st.swapping = true;
            st.fresh = true;
            st.held.clear();
            self.close_generation(&mut st);
            Arc::clone(&st.active)
        };
        old.aclose().await;
        let opened = loop {
            match self.children[target].open(self.turn_detection_disabled, self.anchor).await {
                Ok(s) => break Ok(s),
                Err(e) => {
                    log::error!(target: "livekit.agents", "failed to start realtime model on swap, trying next: {e}");
                    let (next, changes) = {
                        let mut st = self.lock();
                        let mut changes: Vec<_> =
                            st.policy.set_available(target, false, now()).into_iter().collect();
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
        // Closed while the new model opened: it is closed too, never adopted.
        if let Ok(session) = &opened
            && self.lock().closed
        {
            session.aclose().await;
            return;
        }
        let mut st = self.lock();
        st.swapping = false;
        match opened {
            Ok(session) => {
                st.active = session;
                st.policy.set_active_index(target);
                self.model_active.store(target, std::sync::atomic::Ordering::Relaxed);
                st.events.push_back(Event::SessionSwapped { regenerate: self.regenerate_on_swap });
            }
            // Every model failed to start: escalate so the session closes.
            Err(e) => {
                st.events.push_back(Event::Error(RealtimeModelError {
                    timestamp: lk_agents_abi::Ts::ZERO,
                    label:
                        "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter"
                            .to_owned(),
                    message: e.to_string(),
                    recoverable: false,
                    req_id: None,
                    code: None,
                    provider_error: None,
                }));
            }
        }
    }

    /// Track the generation in flight, so a swap can close it.
    fn observe(&self, st: &mut SessionState, ev: Event) -> Event {
        match ev {
            Event::GenerationCreated(g) => {
                st.generation = Some(OpenGeneration {
                    response_id: g.response_id.clone(),
                    messages: Vec::new(),
                });
                Event::GenerationCreated(g)
            }
            Event::MessageStarted(m) => {
                if let Some(g) = st.generation.as_mut() {
                    g.messages.push((m.message_id.clone(), false));
                }
                Event::MessageStarted(m)
            }
            Event::MessageModalities(m) => {
                if let Some(g) = st.generation.as_mut()
                    && let Some(entry) = g.messages.iter_mut().find(|(id, _)| *id == m.message_id)
                {
                    entry.1 = true;
                }
                Event::MessageModalities(m)
            }
            Event::MessageDone(m) => {
                if let Some(g) = st.generation.as_mut() {
                    g.messages.retain(|(id, _)| *id != m.message_id);
                }
                Event::MessageDone(m)
            }
            Event::GenerationDone(d) => {
                st.generation = None;
                Event::GenerationDone(d)
            }
            other => other,
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
        let mut queued_event = false;
        let dispatch = {
            let mut st = self.lock();
            if st.closed {
                return;
            }
            if !st.swapping && !st.fresh {
                Some((Arc::clone(&st.active), vec![cmd]))
            } else {
                match cmd {
                    // The frame, onto the new session; the tools end it, and what was held goes after.
                    RealtimeCommand::UpdateInstructions(_) | RealtimeCommand::UpdateChatCtx(_)
                        if !st.swapping =>
                    {
                        Some((Arc::clone(&st.active), vec![cmd]))
                    }
                    RealtimeCommand::UpdateTools(_) if !st.swapping => {
                        st.fresh = false;
                        let mut commands = vec![cmd];
                        commands.extend(st.held.take());
                        Some((Arc::clone(&st.active), commands))
                    }
                    RealtimeCommand::GenerateReply(_) | RealtimeCommand::Say(_) => {
                        if let Err((reason, rejected)) = st.held.hold(cmd) {
                            queued_event = true;
                            if st.events.len() < 256 {
                                st.events.push_back(Event::GenerateReplyFailed(
                                    lk_agents_abi::llm::realtime::GenerateReplyFailed {
                                        req_id: rejected.req_id().unwrap_or_default().to_owned(),
                                        error:
                                            lk_agents_abi::llm::realtime::RealtimeError::with_code(
                                                reason.to_string(),
                                                Some("realtime_fallback_backlog_exceeded".into()),
                                            ),
                                    },
                                ));
                            } else {
                                st.closed = true;
                                st.held.clear();
                                st.events.clear();
                                st.events.push_back(Event::Error(RealtimeModelError {
                                    timestamp: lk_agents_abi::Ts::ZERO,
                                    label: "RealtimeModelFallbackAdapter".into(),
                                    message: "realtime fallback event backlog exceeded".into(),
                                    recoverable: false,
                                    req_id: None,
                                    code: Some("realtime_fallback_event_backlog_exceeded".into()),
                                    provider_error: None,
                                }));
                            }
                        }
                        None
                    }
                    RealtimeCommand::CancelReply(c) => {
                        st.held.cancel(&c.req_id);
                        None
                    }
                    // The old session's (an interrupt, a truncation, its audio), or superseded by the
                    // frame (a pushed item is in the conversation the driver gives).
                    _ => None,
                }
            }
        };
        if queued_event {
            self.changed.notify_one();
        }
        if let Some((active, commands)) = dispatch {
            for command in commands {
                active.send(command);
            }
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
                () = self.changed.notified() => continue,
                ev = active.next_event() => ev,
                changed = restart_rx.changed() => {
                    if changed.is_err() { continue; }
                    let switch_model = *restart_rx.borrow_and_update();
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
                            self.lock().events.push_back(Event::Error(RealtimeModelError {
                                recoverable: true,
                                ..e
                            }));
                            self.swap(target).await;
                        }
                        None => return Some(Event::Error(e)),
                    }
                }
                Some(ev) => return Some(self.observe(&mut self.lock(), ev)),
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
            st.held.clear();
            Arc::clone(&st.active)
        };
        self.changed.notify_one();
        active.aclose().await;
    }
}

/// Conversion of opaque provider instances uses the same native registry and host bridge as
/// the provider boundary; the generated facade owns the resulting native model.
#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use lk_agents_abi::llm::realtime::HostRealtime;
    use lk_agents_abi::llm::realtime_native::RealtimeConfigurationError;
    use lk_agents_abi::python::{HostProxy, native};
    use lk_agents_abi::{HostRef, Realtime, WallAnchor};
    use pyo3::prelude::*;

    use super::{RealtimeChild, RealtimeFallbackModel};
    use lk_agents_inference::fallback::ChainObserver;

    #[async_trait::async_trait]
    impl lk_agents_abi::llm::realtime_native::RealtimeFallbackModel for RealtimeFallbackModel {
        fn new(
            models: Vec<HostRef>,
            cooldown: f64,
            regenerate_on_swap: bool,
            on_availability: Option<HostRef>,
        ) -> Result<Self, RealtimeConfigurationError> {
            let (children, observer) = Python::attach(|py| -> PyResult<_> {
                let children = models.into_iter().map(|model| {
                    let model = model.into_pyobject(py)?;
                    match native::<dyn Realtime>(&model) {
                        Some(native) => Ok(RealtimeChild::Native(native)),
                        None => <dyn HostRealtime as HostProxy>::from_host(&model, None).map(|host| {
                            let spawner = <Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient();
                            RealtimeChild::Host(host, spawner)
                        }),
                    }
                }).collect::<PyResult<Vec<_>>>()?;
                let observer: Arc<dyn ChainObserver> = match on_availability {
                    Some(observer) => <dyn ChainObserver as HostProxy>::from_host(&observer.into_pyobject(py)?, None)?,
                    None => Arc::new(lk_agents_inference::fallback::IgnoreEvents),
                };
                Ok((children, observer))
            }).map_err(|error| Python::attach(|py| lk_agents_abi::python::capture_host(py, &error)))?;
            let clock = crate::process::host_clock();
            RealtimeFallbackModel::new(
                children,
                cooldown,
                regenerate_on_swap,
                observer,
                WallAnchor::of(&*clock),
            )
            .map_err(RealtimeConfigurationError::Invalid)
        }

        fn active_index(&self) -> usize {
            self.active_index()
        }
        fn capabilities(&self) -> lk_agents_abi::llm::realtime::RealtimeCapabilities {
            Realtime::capabilities(self)
        }
        async fn restart(&self, switch_model: bool) {
            self.restart(switch_model).await;
        }
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
        Arc::new(Scripted {
            events: Mutex::new(events.into()),
            sent: Mutex::default(),
            closed: Default::default(),
        })
    }

    fn fatal() -> Event {
        Event::Error(RealtimeModelError {
            timestamp: lk_agents_abi::Ts::ZERO,
            label: "scripted".into(),
            message: "boom".into(),
            recoverable: false,
            req_id: None,
            code: None,
            provider_error: None,
        })
    }

    fn kind(ev: &Event) -> String {
        format!("{ev:?}")
            .split('(')
            .next()
            .unwrap_or_default()
            .split(' ')
            .next()
            .unwrap_or_default()
            .to_owned()
    }

    // Catches: a fatal child error passed on as fatal while a fallback exists, the dead child's
    // open generation left open, a swap reported as a reconnect (the driver then gives the new
    // model nothing), or a request reaching the new model before its frame.
    #[tokio::test]
    async fn a_fatal_error_mid_reply_swaps_to_the_next_model_and_asks_for_the_reply_again() {
        let primary = scripted(vec![
            Event::GenerationCreated(GenerationCreated { response_id: "r1".into(), req_id: None }),
            Event::MessageStarted(MessageStarted {
                message_id: "m1".into(),
                response_id: Some("r1".into()),
            }),
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
                "SessionSwapped"
            ]
        );
        assert!(primary.closed.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(*changes.lock().unwrap(), [(0, false)]);
        assert_eq!(model.active_index(), 1);

        // A reply asked before the driver gave the frame waits for it; the tools end the frame.
        let ask = |id: &str| {
            RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
                req_id: id.into(),
                instructions: None,
                tool_choice: None,
            })
        };
        session.send(ask("early"));
        session.send(RealtimeCommand::Interrupt(lk_agents_abi::llm::realtime::Interrupt {
            response_id: Some("r1".into()),
        }));
        assert!(
            backup.sent.lock().unwrap().is_empty(),
            "nothing reaches the new model before its frame"
        );
        session.send(RealtimeCommand::UpdateInstructions(lk_agents_abi::UpdateInstructions {
            instructions: "hi".into(),
        }));
        session.send(RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools { tools: vec![] }));
        session.send(ask("late"));
        let sent = backup.sent.lock().unwrap().clone();
        assert_eq!(
            sent,
            ["UpdateInstructions", "UpdateTools", "generate_reply:early", "generate_reply:late"]
        );
    }
    #[tokio::test]
    async fn realtime_fallback_backlog_refuses_excess_requests_before_forwarding() {
        let primary = scripted(vec![fatal()]);
        let backup = scripted(vec![]);
        let model = RealtimeFallbackModel::new(
            vec![
                RealtimeChild::Native(Arc::new(Model(primary))),
                RealtimeChild::Native(Arc::new(Model(backup.clone()))),
            ],
            10.0,
            false,
            Arc::new(lk_agents_inference::fallback::IgnoreEvents),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();
        assert!(matches!(session.next_event().await, Some(Event::Error(_))));
        assert!(matches!(session.next_event().await, Some(Event::SessionSwapped { .. })));
        let (polling, polled) = tokio::sync::oneshot::channel();
        let rejected_reader = tokio::spawn({
            let session = session.clone();
            async move {
                let _ = polling.send(());
                session.next_event().await
            }
        });
        polled.await.unwrap();
        for index in 0..33 {
            session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
                req_id: format!("queued_{index}"),
                instructions: None,
                tool_choice: None,
            }));
        }
        session.send(RealtimeCommand::CancelReply(lk_agents_abi::CancelReply {
            req_id: "queued_0".into(),
        }));
        session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
            req_id: "oversized".into(),
            instructions: Some("x".repeat(1024 * 1024)),
            tool_choice: None,
        }));
        session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
            req_id: "replacement".into(),
            instructions: None,
            tool_choice: None,
        }));
        session.send(RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools { tools: vec![] }));
        assert!(
            !backup
                .sent
                .lock()
                .unwrap()
                .iter()
                .any(|command| command == "generate_reply:queued_32"),
            "a request beyond the bounded replay backlog must never reach the replacement model"
        );
        let rejected = tokio::time::timeout(std::time::Duration::from_secs(1), rejected_reader)
            .await
            .expect("a held-backlog rejection must wake an already parked reader")
            .unwrap();
        assert!(matches!(rejected, Some(Event::GenerateReplyFailed(failed))
            if failed.req_id == "queued_32" && failed.error.code.as_deref() == Some("realtime_fallback_backlog_exceeded")));
        assert!(
            backup
                .sent
                .lock()
                .unwrap()
                .iter()
                .any(|command| command == "generate_reply:replacement"),
            "cancellation releases admission"
        );
        assert!(
            !backup
                .sent
                .lock()
                .unwrap()
                .iter()
                .any(|command| command == "generate_reply:oversized")
        );
        let rejected =
            tokio::time::timeout(std::time::Duration::from_secs(1), session.next_event())
                .await
                .unwrap();
        assert!(
            matches!(rejected, Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "oversized" && failed.error.code.as_deref() == Some("realtime_fallback_backlog_exceeded"))
        );
        session.aclose().await;
    }
}
