//! The realtime fallback adapter (`RealtimeModelFallbackAdapter`) over two simulated models, as
//! the driver sees it at the `Realtime` boundary.
//!
//! A child's terminal error swaps to the other model while it is available: the error reaches
//! the driver as recoverable, the generation in flight is closed as cancelled, the old session
//! closes and the new one opens, and `SessionSwapped` asks the driver for the new session's
//! frame. Until the frame's tools arrive, what names the old session is dropped and replies are
//! held; past the adapter's bound a reply is refused. A failed model is out for the adapter's
//! cooldown, on the run's wall clock; with no other model available the error escalates as is.
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use lk_agents_abi::{
    ChatContext, Clock, Realtime, RealtimeCommand, RealtimeSession,
    errors::ProviderError,
    llm::realtime::{
        Event, GenerateReplyFailed, GenerationDone, GenerationStatus, MessageDone,
        MessageModalities, Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError,
        TokenUsage,
    },
};

/// The adapter's cooldown before a failed model is tried again.
const COOLDOWN: Duration = Duration::from_secs(10);
/// The replies the adapter holds for a new model before it refuses one.
const HELD_REPLIES: usize = 32;

pub struct FallbackRealtime {
    children: Vec<Arc<dyn Realtime>>,
    caps: RealtimeCapabilities,
    clock: Arc<dyn Clock>,
}

impl FallbackRealtime {
    /// Children of one setup: their capabilities agree, so the adapter's are theirs.
    pub fn new(children: Vec<Arc<dyn Realtime>>, clock: Arc<dyn Clock>) -> Arc<Self> {
        let caps = children[0].capabilities();
        assert!(children.iter().all(|c| c.capabilities() == caps), "children that agree");
        Arc::new(FallbackRealtime { children, caps, clock })
    }
}

#[async_trait::async_trait]
impl Realtime for FallbackRealtime {
    fn label(&self) -> String {
        "livekit.agents.llm.realtime_fallback_adapter.RealtimeModelFallbackAdapter".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// A fresh session starts on the primary.
    async fn session(&self, disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let active = self.children[0].session(disabled).await?;
        Ok(Arc::new(FallbackSession {
            children: self.children.clone(),
            caps: self.caps,
            clock: self.clock.clone(),
            disabled,
            state: Mutex::new(State {
                active,
                active_index: 0,
                down_until: vec![None; self.children.len()],
                swapping: false,
                fresh: false,
                held: Vec::new(),
                events: Default::default(),
                generation: None,
                closed: false,
            }),
            changed: tokio::sync::Notify::new(),
        }))
    }
}

/// The generation in flight: its messages, and whether each settled its modalities.
struct Open {
    response_id: String,
    messages: Vec<(String, bool)>,
}

struct State {
    active: Arc<dyn RealtimeSession>,
    active_index: usize,
    /// Each model's cooldown after its terminal error: out until then.
    down_until: Vec<Option<SystemTime>>,
    /// The old session is closing and the new one opening.
    swapping: bool,
    /// The new session waits for its frame, which ends with the tools.
    fresh: bool,
    /// The replies asked while the new model waits for its frame, in order.
    held: Vec<RealtimeCommand>,
    events: std::collections::VecDeque<Event>,
    generation: Option<Open>,
    closed: bool,
}

struct FallbackSession {
    children: Vec<Arc<dyn Realtime>>,
    caps: RealtimeCapabilities,
    clock: Arc<dyn Clock>,
    disabled: bool,
    state: Mutex<State>,
    changed: tokio::sync::Notify,
}

impl FallbackSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    /// The active model failed for good: it is out for the cooldown, and the model to swap to is
    /// the first available one, the primary preferred.
    fn swap_target(&self, st: &mut State) -> Option<usize> {
        let now = self.clock.wall();
        st.down_until[st.active_index] = Some(now + COOLDOWN);
        let up = |until: &Option<SystemTime>| until.is_none_or(|at| at <= now);
        st.down_until.iter().position(up)
    }

    /// The generation a dead session leaves open, closed as the child would on its close.
    fn close_generation(&self, st: &mut State) {
        let Some(open) = st.generation.take() else { return };
        let modalities =
            if self.caps.audio_output { Modalities::AudioAndText } else { Modalities::TextOnly };
        let response_id = Some(open.response_id.clone());
        for (message_id, settled) in open.messages {
            if !settled {
                st.events.push_back(Event::MessageModalities(MessageModalities {
                    message_id: message_id.clone(),
                    response_id: response_id.clone(),
                    modalities,
                }));
            }
            let response_id = response_id.clone();
            st.events.push_back(Event::MessageDone(MessageDone { message_id, response_id }));
        }
        st.events.push_back(Event::GenerationDone(GenerationDone {
            response_id: open.response_id,
            status: GenerationStatus::Cancelled,
            usage: TokenUsage::default(),
        }));
    }

    async fn swap(&self, target: usize) {
        let old = {
            let mut st = self.lock();
            st.swapping = true;
            st.fresh = true;
            st.held.clear();
            self.close_generation(&mut st);
            st.active.clone()
        };
        old.aclose().await;
        let opened = self.children[target].session(self.disabled).await;
        let Ok(session) = opened else { unreachable!("a simulated model always opens") };
        if self.lock().closed {
            return session.aclose().await;
        }
        let mut st = self.lock();
        st.swapping = false;
        st.active = session;
        st.active_index = target;
        st.events.push_back(Event::SessionSwapped { regenerate: true });
    }

    /// Track the generation in flight, so a swap can close it.
    fn observe(st: &mut State, ev: &Event) {
        match ev {
            Event::GenerationCreated(g) => {
                st.generation = Some(Open { response_id: g.response_id.clone(), messages: vec![] });
            }
            Event::MessageStarted(m) => {
                if let Some(open) = st.generation.as_mut() {
                    open.messages.push((m.message_id.clone(), false));
                }
            }
            Event::MessageModalities(m) => {
                let open = st.generation.as_mut();
                if let Some(entry) =
                    open.and_then(|o| o.messages.iter_mut().find(|e| e.0 == m.message_id))
                {
                    entry.1 = true;
                }
            }
            Event::MessageDone(m) => {
                if let Some(open) = st.generation.as_mut() {
                    open.messages.retain(|(id, _)| *id != m.message_id);
                }
            }
            Event::GenerationDone(_) => st.generation = None,
            _ => {}
        }
    }
}

#[async_trait::async_trait]
impl RealtimeSession for FallbackSession {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        let dispatch = {
            let mut st = self.lock();
            if st.closed {
                return;
            }
            if !st.swapping && !st.fresh {
                Some((st.active.clone(), vec![cmd]))
            } else {
                match cmd {
                    // The frame, onto the new session; its tools end it, and what was held goes
                    // after them.
                    RealtimeCommand::UpdateInstructions(_) | RealtimeCommand::UpdateChatCtx(_)
                        if !st.swapping =>
                    {
                        Some((st.active.clone(), vec![cmd]))
                    }
                    RealtimeCommand::UpdateTools(_) if !st.swapping => {
                        st.fresh = false;
                        let mut commands = vec![cmd];
                        commands.append(&mut st.held);
                        Some((st.active.clone(), commands))
                    }
                    RealtimeCommand::GenerateReply(_) | RealtimeCommand::Say(_) => {
                        if st.held.len() < HELD_REPLIES {
                            st.held.push(cmd);
                        } else {
                            st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed {
                                req_id: cmd.req_id().unwrap_or_default().to_owned(),
                                error: RealtimeError::with_code(
                                    "realtime fallback pending reply limit exceeded",
                                    Some("realtime_fallback_backlog_exceeded".into()),
                                ),
                            }));
                            self.changed.notify_one();
                        }
                        None
                    }
                    RealtimeCommand::CancelReply(c) => {
                        st.held.retain(|held| held.req_id() != Some(c.req_id.as_str()));
                        None
                    }
                    // The old session's (an interrupt, a truncation, its audio), or superseded by
                    // the frame.
                    _ => None,
                }
            }
        };
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
                st.active.clone()
            };
            let ev = tokio::select! {
                () = self.changed.notified() => continue,
                ev = active.next_event() => ev,
            };
            match ev {
                // A swapped-away session's end is not the adapter's.
                None if !Arc::ptr_eq(&active, &self.lock().active) => continue,
                None => return None,
                Some(Event::Error(e)) if !e.recoverable => {
                    let target = {
                        let mut st = self.lock();
                        self.swap_target(&mut st)
                    };
                    let Some(target) = target else { return Some(Event::Error(e)) };
                    // While a fallback remains the error is recoverable.
                    let e = RealtimeModelError { recoverable: true, ..e };
                    self.lock().events.push_back(Event::Error(e));
                    self.swap(target).await;
                }
                Some(ev) => {
                    Self::observe(&mut self.lock(), &ev);
                    return Some(ev);
                }
            }
        }
    }

    async fn chat_ctx(&self) -> ChatContext {
        let active = self.lock().active.clone();
        active.chat_ctx().await
    }

    async fn aclose(&self) {
        let active = {
            let mut st = self.lock();
            st.closed = true;
            st.held.clear();
            st.active.clone()
        };
        self.changed.notify_one();
        active.aclose().await;
    }
}
