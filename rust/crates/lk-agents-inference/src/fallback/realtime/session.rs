//! `_FallbackRealtimeSession`: one session over the active model, swapped for the next one on
//! failure with the conversation replayed.

use super::*;

/// The generation in flight: its messages, and which settled their modalities.
pub(super) struct OpenGeneration {
    response_id: String,
    messages: Vec<(String, bool)>,
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
    pub(super) children: Vec<RealtimeChild>,
    pub(super) priority: Vec<ResourceId>,
    pub(super) cooldown: std::time::Duration,
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
    fn lock(&self) -> std::sync::MutexGuard<'_, SessionState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
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
                    biased;
                    () = self.changed.notified() => continue,
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
                    ev = active.next_event() => ev,
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
