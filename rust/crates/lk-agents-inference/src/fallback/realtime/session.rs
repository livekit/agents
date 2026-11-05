//! `_FallbackRealtimeSession`: one session over the active model, swapped for the next one on
//! failure with the conversation replayed.

use super::*;
use crate::effects::FallbackChainEffect;

/// The generation in flight: its messages, and which settled their modalities.
pub(super) struct OpenGeneration {
    response_id: String,
    messages: Vec<(String, bool)>,
}

/// The events a swap may queue for the consumer before the session fails instead.
const EVENT_BACKLOG: usize = 256;

/// The error code of a reply the held backlog refused during a swap.
const BACKLOG_EXCEEDED_CODE: &str = "realtime_fallback_backlog_exceeded";

/// The error code of a session failed because its event queue filled during a swap.
const EVENT_BACKLOG_EXCEEDED_CODE: &str = "realtime_fallback_event_backlog_exceeded";

/// Where `send` takes a command.
enum Routed {
    /// To this session, in this order.
    Dispatch(Arc<dyn RealtimeSession>, Vec<RealtimeCommand>),
    /// Held, cancelled or dropped here; an event for the consumer was queued.
    EventQueued,
    /// Held, cancelled or dropped here, with nothing to tell the consumer.
    Kept,
}

/// What a fatal child error changed: the model to swap to, the models that recovered meanwhile,
/// and the failed one when the failure changed its availability.
struct Failover {
    target: Option<usize>,
    recovered: Vec<ResourceId>,
    failed: Option<ResourceId>,
}

pub(super) struct SessionState {
    pub(super) active: Arc<dyn RealtimeSession>,
    pub(super) policy: FallbackPolicy,
    pub(super) active_index: usize,
    /// The old session is closing and the new one opening: everything sent names the old one, or
    /// is superseded by the frame the driver gives the new one, and is dropped (audio would lag).
    pub(super) swapping: bool,
    /// The new session waits for its frame, which ends with the tools: the requests asked
    /// meanwhile are held until then, so none reaches a model without its conversation.
    pub(super) fresh: bool,
    pub(super) held: FallbackBacklog,
    pub(super) events: VecDeque<Event>,
    pub(super) generation: Option<OpenGeneration>,
    pub(super) closed: bool,
}

pub struct FallbackSession {
    pub(super) children: Vec<ChainModel>,
    /// What every child session works under, the one swapped in included.
    pub(super) cx: lk_agents_abi::telemetry::Cx,
    pub(super) priority: Vec<ResourceId>,
    pub(super) cooldown: std::time::Duration,
    pub(super) attempt_timeout: Option<std::time::Duration>,
    pub(super) caps: RealtimeCapabilities,
    pub(super) regenerate_on_swap: bool,
    pub(super) observer: Arc<dyn ChainObserver>,
    pub(super) anchor: WallAnchor,
    pub(super) clock: Arc<dyn lk_agents_abi::Clock>,
    pub(super) turn_detection_disabled: bool,
    pub(super) model_active: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) state: Mutex<SessionState>,
    pub(super) restart_tx: tokio::sync::watch::Sender<bool>,
    pub(super) changed: tokio::sync::Notify,
    /// Polled by `next_event` only (one consumer).
    pub(super) restart_rx: tokio::sync::Mutex<tokio::sync::watch::Receiver<bool>>,
}

impl FallbackSession {
    fn lock(&self) -> parking_lot::MutexGuard<'_, SessionState> {
        self.state.lock()
    }

    pub(super) fn request_restart(&self, switch_model: bool) {
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
            usage: Default::default(),
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
        let mut request = self.swap_request(target);
        let mut effects: VecDeque<_> = request.handle(ChainEvent::Wake, self.clock.now()).into();
        let mut opened: Option<(u64, Arc<dyn RealtimeSession>)> = None;
        let mut target = target;
        let outcome = loop {
            let Some(effect) = effects.pop_front() else { unreachable!("fallback must settle") };
            let event = match effect {
                FallbackChainEffect::Start { attempt, index, .. } => {
                    self.open_child(attempt, index, &mut opened, &mut target).await
                }
                FallbackChainEffect::WakeAt(at) => {
                    self.clock.wait_until(at).await;
                    ChainEvent::Wake
                }
                FallbackChainEffect::Finished(outcome) => break outcome,
                FallbackChainEffect::Cancel { attempt } => {
                    if let Some((_, session)) = opened.take_if(|(id, _)| *id == attempt) {
                        session.aclose().await;
                    }
                    continue;
                }
            };
            effects.extend(request.handle(event, self.clock.now()));
        };
        if let Some(late) = self.adopt(outcome, opened, target) {
            late.aclose().await;
        }
    }

    /// The swap's machine: `target` first, then the others in priority order, each tried once.
    fn swap_request(&self, target: usize) -> ChainRequest {
        let st = self.lock();
        let mut order: Vec<_> = (0..self.children.len()).collect();
        order.remove(target);
        order.insert(0, target);
        let entries = order
            .into_iter()
            .map(|index| (index, self.priority[index], self.children[index].label()))
            .collect();
        ChainRequest::new(
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
    }

    /// Opens child `index`; a session that opens becomes the swap's target. Once the adapter has
    /// closed, the attempt is cancelled instead.
    async fn open_child(
        &self,
        attempt: u64,
        index: usize,
        opened: &mut Option<(u64, Arc<dyn RealtimeSession>)>,
        target: &mut usize,
    ) -> ChainEvent {
        if self.lock().closed {
            return ChainEvent::Cancel;
        }
        let child = &self.children[index];
        let open = child.open(
            self.cx.clone(),
            self.turn_detection_disabled,
            self.anchor,
            &*self.clock,
            self.attempt_timeout,
        );
        let error = match open.await {
            Ok(session) => {
                *target = index;
                *opened = Some((attempt, session));
                let closed = self.lock().closed;
                return if closed { ChainEvent::Cancel } else { ChainEvent::Succeeded { attempt } };
            }
            Err(error) => error,
        };
        let backend = self.priority[index];
        let changed = {
            let mut st = self.lock();
            if st.closed {
                return ChainEvent::Cancel;
            }
            let until = Some(self.clock.now().saturating_add(self.cooldown));
            st.policy.failed(backend, error.api_kind(), until)
        };
        if changed == Some(true) {
            self.announce(vec![backend], false);
        }
        ChainEvent::Failed { attempt, error, produced_output: false }
    }

    /// Adoption and close share the state lock: a session that opened after the close is handed
    /// back, to be released outside it.
    fn adopt(
        &self,
        outcome: Result<(), ProviderError>,
        opened: Option<(u64, Arc<dyn RealtimeSession>)>,
        target: usize,
    ) -> Option<Arc<dyn RealtimeSession>> {
        let mut st = self.lock();
        if st.closed {
            return opened.map(|(_, session)| session);
        }
        st.swapping = false;
        match outcome {
            Ok(()) => {
                st.active = opened.expect("successful realtime open").1;
                st.active_index = target;
                self.model_active.store(target, std::sync::atomic::Ordering::Relaxed);
                st.events.push_back(Event::SessionSwapped { regenerate: self.regenerate_on_swap });
            }
            Err(error) => st.events.push_back(Event::Error(RealtimeModelError {
                timestamp: lk_agents_abi::Ts::ZERO,
                label: "livekit.agents.llm.realtime.RealtimeModelFallbackAdapter".to_owned(),
                message: error.to_string(),
                recoverable: false,
                req_id: None,
                code: None,
                provider_error: Some(error.attempt_error()),
            })),
        }
        None
    }

    /// Where a command goes. Outside a swap it goes to the serving session; during one only the
    /// new session's frame does, and replies are held until the tools end the frame.
    fn route(st: &mut SessionState, cmd: RealtimeCommand) -> Routed {
        if !st.swapping && !st.fresh {
            return Routed::Dispatch(Arc::clone(&st.active), vec![cmd]);
        }
        match cmd {
            // The frame, onto the new session; the tools end it, and what was held goes after.
            RealtimeCommand::UpdateInstructions(_) | RealtimeCommand::UpdateChatCtx(_)
                if !st.swapping =>
            {
                Routed::Dispatch(Arc::clone(&st.active), vec![cmd])
            }
            RealtimeCommand::UpdateTools(_) if !st.swapping => {
                st.fresh = false;
                let mut commands = vec![cmd];
                commands.extend(st.held.take());
                Routed::Dispatch(Arc::clone(&st.active), commands)
            }
            RealtimeCommand::GenerateReply(_) | RealtimeCommand::Say(_) => {
                FallbackSession::hold_reply(st, cmd)
            }
            RealtimeCommand::CancelReply(c) => {
                st.held.cancel(&c.req_id);
                Routed::Kept
            }
            // The old session's (an interrupt, a truncation, its audio), or superseded by the
            // frame (a pushed item is in the conversation the driver gives).
            _ => Routed::Kept,
        }
    }

    /// Holds a reply for the new session. One the backlog refuses is answered with its failure;
    /// when the event queue is full too, the session fails.
    fn hold_reply(st: &mut SessionState, cmd: RealtimeCommand) -> Routed {
        let Err((reason, rejected)) = st.held.hold(cmd) else { return Routed::Kept };
        if st.events.len() < EVENT_BACKLOG {
            st.events.push_back(Event::GenerateReplyFailed(
                lk_agents_abi::llm::realtime::GenerateReplyFailed {
                    req_id: rejected.req_id().unwrap_or_default().to_owned(),
                    error: lk_agents_abi::llm::realtime::RealtimeError::with_code(
                        reason.to_string(),
                        Some(BACKLOG_EXCEEDED_CODE.into()),
                    ),
                },
            ));
            return Routed::EventQueued;
        }
        st.closed = true;
        st.held.clear();
        st.events.clear();
        st.events.push_back(Event::Error(RealtimeModelError {
            timestamp: lk_agents_abi::Ts::ZERO,
            label: "RealtimeModelFallbackAdapter".into(),
            message: "realtime fallback event backlog exceeded".into(),
            recoverable: false,
            req_id: None,
            code: Some(EVENT_BACKLOG_EXCEEDED_CODE.into()),
            provider_error: None,
        }));
        Routed::EventQueued
    }

    /// A restart the driver asked for: a swap to the serving model, or with `switch_model` to the
    /// first other available one.
    async fn restart(
        &self,
        changed: Result<(), tokio::sync::watch::error::RecvError>,
        restart_rx: &mut tokio::sync::watch::Receiver<bool>,
    ) {
        if changed.is_err() {
            return;
        }
        let switch_model = *restart_rx.borrow_and_update();
        let (target, changes) = {
            let mut st = self.lock();
            let changes = st.policy.refresh(&self.priority, self.clock.now());
            let serving = st.active_index;
            let target = if switch_model {
                (st.policy.candidates(&self.priority, false).into_iter())
                    .find(|&index| self.priority[index] != self.priority[serving])
                    .unwrap_or(serving)
            } else {
                serving
            };
            (target, changes)
        };
        self.announce(changes, true);
        self.swap(target).await;
    }

    /// What the consumer gets for a child's event; `None` loops again, after a superseded
    /// child's end or a swap away from a failed one.
    async fn on_child_event(
        &self,
        active: &Arc<dyn RealtimeSession>,
        ev: Option<Event>,
    ) -> Option<Option<Event>> {
        match ev {
            None if !Arc::ptr_eq(active, &self.lock().active) => None,
            None => Some(None),
            Some(Event::Error(e)) if !e.recoverable => self.fail_over(e).await.map(Some),
            Some(ev) => Some(Some(self.observe(&mut self.lock(), ev))),
        }
    }

    /// A fatal child error marks its model down and swaps to the best remaining one, the error
    /// passed on as recoverable; `None` once swapped. With no model left the error stays fatal.
    async fn fail_over(&self, e: RealtimeModelError) -> Option<Event> {
        let Some(failover) = self.mark_failed(&e) else { return Some(Event::Error(e)) };
        self.announce(failover.recovered, true);
        self.announce(failover.failed.into_iter().collect(), false);
        let Some(target) = failover.target else { return Some(Event::Error(e)) };
        // While a fallback remains the error is recoverable.
        self.lock().events.push_back(Event::Error(RealtimeModelError { recoverable: true, ..e }));
        self.swap(target).await;
        None
    }

    /// Marks the serving model failed; `None` when the policy ignores the failure.
    fn mark_failed(&self, e: &RealtimeModelError) -> Option<Failover> {
        let mut st = self.lock();
        let now = self.clock.now();
        let backend = self.priority[st.active_index];
        let failed = st.policy.failed(
            backend,
            e.provider_error.as_ref().map(|error| error.kind),
            Some(now.saturating_add(self.cooldown)),
        )?;
        // A zero cooldown applies to the next pass, not the failure just observed.
        let recovery_priority: Vec<_> =
            self.priority.iter().copied().filter(|id| *id != backend).collect();
        let recovered = st.policy.refresh(&recovery_priority, now);
        let target = st.policy.candidates(&self.priority, false).first().copied();
        Some(Failover { target, recovered, failed: failed.then_some(backend) })
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
        let routed = {
            let mut st = self.lock();
            if st.closed {
                return;
            }
            FallbackSession::route(&mut st, cmd)
        };
        match routed {
            Routed::Dispatch(active, commands) => commands.into_iter().for_each(|c| active.send(c)),
            Routed::EventQueued => self.changed.notify_one(),
            Routed::Kept => {}
        }
    }

    /// Where `send` would take its audio: the serving session, outside a swap.
    fn head(&self, frame: lk_agents_abi::AudioFrame) {
        let active = {
            let st = self.lock();
            if st.closed || st.swapping || st.fresh {
                return;
            }
            Arc::clone(&st.active)
        };
        active.head(frame);
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
                // The consumer's lock on the restart signal stays held across a restart's swap.
                let ev = tokio::select! {
                    biased;
                    () = self.changed.notified() => continue,
                    changed = restart_rx.changed() => {
                        self.restart(changed, &mut restart_rx).await;
                        continue;
                    }
                    ev = active.next_event() => ev,
                };
                if let Some(out) = self.on_child_event(&active, ev).await {
                    return out;
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
