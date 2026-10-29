//! Realtime fallback model resources and asynchronous session replacement.

// ---------------------------------------------------------------------------------------------
// The adapter's session: `_FallbackRealtimeSession`
// ---------------------------------------------------------------------------------------------

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
use crate::fallback::realtime::{FallbackBacklog, RealtimeCapabilities, merge_capabilities};
use crate::fallback::{ChainObserver, ChainPolicy, FallbackPolicy};
use crate::realtime::host::HostRealtimeModel;
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
        let instance = match self {
            Self::Native(model) => model.instance_id(),
            Self::Host(model, _) => model.instance_id(),
        };
        ResourceId(instance as u64)
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
            clock: Arc::new(lk_agents_abi::RealClock::new()),
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

/// The generation in flight: its messages, and which settled their modalities.
struct OpenGeneration {
    response_id: String,
    messages: Vec<(String, bool)>,
}

struct SessionState {
    active: Arc<dyn RealtimeSession>,
    policy: FallbackPolicy,
    active_index: usize,
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
    priority: Vec<ResourceId>,
    cooldown: std::time::Duration,
    caps: RealtimeCapabilities,
    regenerate_on_swap: bool,
    observer: Arc<dyn ChainObserver>,
    anchor: WallAnchor,
    clock: Arc<dyn lk_agents_abi::Clock>,
    turn_detection_disabled: bool,
    model_active: Arc<std::sync::atomic::AtomicUsize>,
    state: Mutex<SessionState>,
    restart_tx: tokio::sync::watch::Sender<bool>,
    changed: tokio::sync::Notify,
    /// Polled by `next_event` only (one consumer).
    restart_rx: tokio::sync::Mutex<tokio::sync::watch::Receiver<bool>>,
}

impl FallbackSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, SessionState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn request_restart(&self, switch_model: bool) {
        let _ = self.restart_tx.send(switch_model);
    }

    fn announce(&self, changes: Vec<ResourceId>, available: bool) {
        for backend in changes {
            let index = self.priority.iter().position(|id| *id == backend).expect("known backend");
            self.observer.availability_changed(AvailabilityChanged {
                index,
                label: self.children[index].label(),
                available,
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
    async fn swap(&self, target: usize) {
        let old = {
            let mut st = self.lock();
            st.swapping = true;
            st.fresh = true;
            st.held.clear();
            self.close_generation(&mut st);
            Arc::clone(&st.active)
        };
        old.aclose().await;
        let mut request = {
            let st = self.lock();
            let mut order: Vec<_> = (0..self.children.len()).collect();
            order.remove(target);
            order.insert(0, target);
            let entries = order
                .into_iter()
                .map(|index| (index, self.priority[index], self.children[index].label()))
                .collect();
            ChainRequest::new(
                Modality::Realtime,
                entries,
                ChainPolicy {
                    max_retry_per_backend: 0,
                    attempt_timeout: None,
                    retry_interval: None,
                    sticky: false,
                },
                None,
                &st.policy,
            )
        };
        let mut effects: VecDeque<_> = request.handle(ChainEvent::Wake, self.clock.now()).into();
        let mut opened: Option<(u64, Arc<dyn RealtimeSession>)> = None;
        let mut target = target;
        let outcome = loop {
            let Some(effect) = effects.pop_front() else { unreachable!("fallback must settle") };
            match effect {
                ChainEffect::RealtimeSessionOpen { attempt, index, .. } => {
                    if self.lock().closed {
                        effects.extend(request.handle(ChainEvent::Cancel, self.clock.now()));
                        continue;
                    }
                    let event = match self.children[index]
                        .open(self.turn_detection_disabled, self.anchor)
                        .await
                    {
                        Ok(session) => {
                            target = index;
                            opened = Some((attempt, session));
                            if self.lock().closed {
                                ChainEvent::Cancel
                            } else {
                                ChainEvent::Succeeded { attempt }
                            }
                        }
                        Err(error) => {
                            let backend = self.priority[index];
                            let (closed, changed) = {
                                let mut st = self.lock();
                                let closed = st.closed;
                                let changed = if closed {
                                    None
                                } else {
                                    st.policy.failed(
                                        backend,
                                        error.api_kind(),
                                        Some(self.clock.now().saturating_add(self.cooldown)),
                                    )
                                };
                                (closed, changed)
                            };
                            if closed {
                                effects
                                    .extend(request.handle(ChainEvent::Cancel, self.clock.now()));
                                continue;
                            }
                            if changed == Some(true) {
                                self.announce(vec![backend], false);
                            }
                            ChainEvent::Failed { attempt, error, produced_output: false }
                        }
                    };
                    effects.extend(request.handle(event, self.clock.now()));
                }
                ChainEffect::WakeAt(at) => {
                    self.clock.wait_until(at).await;
                    effects.extend(request.handle(ChainEvent::Wake, self.clock.now()));
                }
                ChainEffect::Finished(outcome) => break outcome,
                ChainEffect::Cancel { attempt } => {
                    if opened.as_ref().is_some_and(|(id, _)| *id == attempt) {
                        let (_, session) = opened.take().expect("matching opened session");
                        session.aclose().await;
                    }
                }
                _ => unreachable!("realtime fallback only opens realtime sessions"),
            }
        };
        // Adoption and close share the state lock. A late session is released outside it.
        let late = {
            let mut st = self.lock();
            if st.closed {
                opened.take().map(|(_, session)| session)
            } else {
                st.swapping = false;
                match outcome {
                    Ok(()) => {
                        st.active = opened.take().expect("successful realtime open").1;
                        st.active_index = target;
                        self.model_active.store(target, std::sync::atomic::Ordering::Relaxed);
                        st.events.push_back(Event::SessionSwapped {
                            regenerate: self.regenerate_on_swap,
                        });
                    }
                    Err(error) => {
                        st.events.push_back(Event::Error(RealtimeModelError {
                            timestamp: lk_agents_abi::Ts::ZERO,
                            label: "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter".to_owned(),
                            message: error.to_string(),
                            recoverable: false,
                            req_id: None,
                            code: None,
                            provider_error: Some(error.attempt_error()),
                        }));
                    }
                }
                None
            }
        };
        if let Some(session) = late {
            session.aclose().await;
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

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move {
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
                        let (target, changes) = {
                            let mut st = self.lock();
                            let changes = st.policy.refresh(&self.priority, self.clock.now());
                            let target = if switch_model {
                                st.policy.candidates(&self.priority, false).into_iter()
                                    .find(|&index| self.priority[index] != self.priority[st.active_index]).unwrap_or(st.active_index)
                            } else { st.active_index };
                            (target, changes)
                        };
                        self.announce(changes, true);
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
                        let (target, recovered, failed) = {
                            let mut st = self.lock();
                            let now = self.clock.now();
                            let backend = self.priority[st.active_index];
                            let Some(failed) = st.policy.failed(
                                backend,
                                e.provider_error.as_ref().map(|error| error.kind),
                                Some(now.saturating_add(self.cooldown)),
                            ) else {
                                return Some(Event::Error(e));
                            };
                            // A zero cooldown applies to the next pass, not the failure just observed.
                            let recovery_priority: Vec<_> =
                                self.priority.iter().copied().filter(|id| *id != backend).collect();
                            let recovered = st.policy.refresh(&recovery_priority, now);
                            (
                                st.policy.candidates(&self.priority, false).first().copied(),
                                recovered,
                                failed.then_some(backend),
                            )
                        };
                        self.announce(recovered, true);
                        self.announce(failed.into_iter().collect(), false);
                        match target {
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
        })
    }

    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
        Box::pin(async move {
            let active = Arc::clone(&self.lock().active);
            active.chat_ctx().await
        })
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            let active = {
                let mut st = self.lock();
                st.closed = true;
                st.held.clear();
                Arc::clone(&st.active)
            };
            self.changed.notify_one();
            active.aclose().await;
        })
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
        fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
            Box::pin(async move {
                let next = self.events.lock().unwrap().pop_front();
                match next {
                    Some(ev) => Some(ev),
                    None => std::future::pending().await,
                }
            })
        }
        fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
            Box::pin(async move { ChatContext::default() })
        }
        fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
            Box::pin(async move {
                self.closed.store(true, std::sync::atomic::Ordering::Release);
            })
        }
    }

    struct Model(Arc<Scripted>);

    impl Realtime for Model {
        fn label(&self) -> String {
            "scripted".into()
        }
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        fn session(
            &self,
            _td: bool,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
            Box::pin(async move { Ok(Arc::clone(&self.0) as Arc<dyn RealtimeSession>) })
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
    #[derive(Default)]
    struct AvailabilityLog(Mutex<Vec<(usize, bool)>>);
    impl ChainObserver for AvailabilityLog {
        fn availability_changed(&self, ev: AvailabilityChanged) {
            self.0.lock().unwrap().push((ev.index, ev.available));
        }
    }

    async fn cancellation_case(failed_open: bool) {
        struct CancelledOpen;
        impl Realtime for CancelledOpen {
            fn label(&self) -> String {
                "cancelled-open".into()
            }
            fn capabilities(&self) -> RealtimeCapabilities {
                RealtimeCapabilities { audio_output: true, ..Default::default() }
            }
            fn session(
                &self,
                _: bool,
            ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>>
            {
                Box::pin(async {
                    Err(lk_agents_abi::errors::ApiError::status(
                        "provider cancelled",
                        499,
                        None,
                        None,
                    )
                    .into())
                })
            }
        }
        let mut error = fatal();
        if !failed_open && let Event::Error(e) = &mut error {
            e.provider_error = Some(
                lk_agents_abi::errors::ApiError::status("provider cancelled", 499, None, None)
                    .into(),
            );
        }
        let primary = scripted(vec![error]);
        let backup: Arc<dyn Realtime> =
            if failed_open { Arc::new(CancelledOpen) } else { Arc::new(Model(scripted(vec![]))) };
        let changes = Arc::new(AvailabilityLog::default());
        let model = RealtimeFallbackModel::new(
            vec![
                RealtimeChild::Native(Arc::new(Model(primary.clone()))),
                RealtimeChild::Native(backup),
            ],
            10.0,
            false,
            changes.clone(),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();
        let first = session.next_event().await.unwrap();
        let terminal = if failed_open {
            assert!(matches!(first, Event::Error(e) if e.recoverable));
            session.next_event().await.unwrap()
        } else {
            first
        };
        assert_eq!(
            *changes.0.lock().unwrap(),
            if failed_open { vec![(0, false)] } else { vec![] },
            "cancellation must not change the cancelled backend's health"
        );
        let Event::Error(error) = terminal else { panic!("cancellation was retried") };
        assert!(!error.recoverable);
        assert!(
            error
                .provider_error
                .as_ref()
                .is_some_and(|e| e.kind == lk_agents_abi::errors::ApiErrorKind::Cancelled),
            "the cancellation's class and diagnostic cause must survive fallback"
        );
        assert_eq!(model.active_index(), 0, "cancellation must not swap providers");
        if !failed_open {
            assert!(!primary.closed.load(std::sync::atomic::Ordering::Acquire));
        }
        session.aclose().await;
    }

    #[tokio::test]
    async fn a_cancelled_realtime_request_does_not_swap_or_penalize_its_provider() {
        cancellation_case(false).await;
    }

    #[tokio::test]
    async fn a_cancelled_realtime_open_preserves_its_cause_without_a_health_penalty() {
        cancellation_case(true).await;
    }

    #[tokio::test]
    async fn closing_during_a_failed_swap_does_not_publish_a_fatal_cancellation() {
        struct BlockedOpen(Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>);
        impl Realtime for BlockedOpen {
            fn label(&self) -> String {
                "blocked-open".into()
            }
            fn capabilities(&self) -> RealtimeCapabilities {
                RealtimeCapabilities { audio_output: true, ..Default::default() }
            }
            fn session(
                &self,
                _: bool,
            ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>>
            {
                Box::pin(async move {
                    self.0.notify_one();
                    self.1.notified().await;
                    Err(lk_agents_abi::errors::ApiError::connection(None).into())
                })
            }
        }
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let changes = Arc::new(AvailabilityLog::default());
        let model = RealtimeFallbackModel::new(
            vec![
                RealtimeChild::Native(Arc::new(Model(scripted(vec![fatal()])))),
                RealtimeChild::Native(Arc::new(BlockedOpen(entered.clone(), release.clone()))),
                RealtimeChild::Native(Arc::new(Model(scripted(vec![])))),
            ],
            10.0,
            false,
            changes.clone(),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();
        let reading = tokio::spawn({
            let session = session.clone();
            async move { session.next_event().await }
        });
        entered.notified().await;
        session.aclose().await;
        release.notify_one();
        let _ = reading.await.unwrap();
        assert_eq!(
            *changes.0.lock().unwrap(),
            vec![(0, false)],
            "a late failure after close must not announce a health penalty"
        );
        assert!(
            session.next_event().await.is_none(),
            "session close fabricated a fatal cancellation error"
        );
    }

    #[tokio::test]
    async fn repeating_the_same_backend_does_not_create_a_second_fallback_identity() {
        let child: Arc<dyn Realtime> = Arc::new(Model(scripted(vec![fatal()])));
        let model = RealtimeFallbackModel::new(
            vec![RealtimeChild::Native(child.clone()), RealtimeChild::Native(child)],
            10.0,
            false,
            Arc::new(crate::fallback::IgnoreEvents),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();
        assert!(
            matches!(session.next_event().await, Some(Event::Error(e)) if !e.recoverable),
            "one failed backend cannot fall back to another occurrence of itself"
        );
        session.aclose().await;
    }

    // A failed open is attempted once in one failover, even when it is immediately eligible
    // for a later request. The cap turns an accidental infinite swap loop into a test failure.
    #[tokio::test]
    async fn zero_cooldown_exhausts_one_fallback_pass_without_reopening_failed_children() {
        struct FailsToOpen(std::sync::atomic::AtomicUsize);
        impl Realtime for FailsToOpen {
            fn label(&self) -> String {
                "failed-open".into()
            }
            fn capabilities(&self) -> RealtimeCapabilities {
                RealtimeCapabilities { audio_output: true, ..Default::default() }
            }
            fn session(
                &self,
                _: bool,
            ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>>
            {
                Box::pin(async move {
                    assert_eq!(
                        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                        0,
                        "the same failover must not reopen a failed child"
                    );
                    Err(lk_agents_abi::errors::ApiError::connection(Some("open failed".into()))
                        .into())
                })
            }
        }
        let primary = scripted(vec![fatal()]);
        let backup = Arc::new(FailsToOpen(Default::default()));
        let model = RealtimeFallbackModel::new(
            vec![
                RealtimeChild::Native(Arc::new(Model(primary))),
                RealtimeChild::Native(backup.clone()),
            ],
            0.0,
            false,
            Arc::new(crate::fallback::IgnoreEvents),
            WallAnchor::of(&lk_agents_abi::RealClock::default()),
        )
        .unwrap();
        let session = model.session(false).await.unwrap();
        assert!(matches!(session.next_event().await, Some(Event::Error(e)) if e.recoverable));
        assert!(matches!(session.next_event().await, Some(Event::Error(e)) if !e.recoverable));
        assert_eq!(backup.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        session.aclose().await;
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
            Arc::new(crate::fallback::IgnoreEvents),
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
