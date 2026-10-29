//! The duplex model run as a realtime model: its sessions, their event delivery and host integration.

use crate::realtime::native::queue::{self, Delivery, EventReceiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex as AsyncMutex, Notify};

pub use crate::duplex::{
    AdaptiveNoiseGate, AudioGate, DuplexCapabilities, DuplexCommand, DuplexEvent, FixedGate,
    Fragment, Segmenter, Step,
};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use lk_agents_abi::{
    ChatContext, ChatItem, Clock, Realtime, RealtimeCommand, RealtimeSession, WallAnchor,
};

/// A plugin's `DuplexModel`.
#[async_trait::async_trait]
pub trait DuplexHost: Send + Sync {
    fn label(&self) -> String;
    fn capabilities(&self) -> DuplexCapabilities;
    /// A new plugin session, and the gate the adapter or the model declares (`None`: the adaptive
    /// gate).
    async fn open(
        &self,
    ) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError>;
}

/// A plugin's `DuplexSession`.
#[async_trait::async_trait]
pub trait DuplexHostSession: Send + Sync {
    /// Queued on the host, in order.
    fn send(&self, cmd: DuplexCommand);
    /// `None` once the session is closed.
    async fn next_event(&self) -> Option<DuplexEvent>;
    async fn aclose(&self);
}

/// `DuplexRealtimeAdapter`'s engine: the session's realtime model, its sessions on the clock
/// and executor it was built with.
pub struct NativeDuplexModel {
    host: Arc<dyn DuplexHost>,
    label: String,
    caps: RealtimeCapabilities,
    audio_timeout: Duration,
    clock: Arc<dyn Clock>,
    anchor: WallAnchor,
}

impl NativeDuplexModel {
    pub fn new(host: Arc<dyn DuplexHost>, audio_timeout: Duration, clock: Arc<dyn Clock>) -> Self {
        let anchor = WallAnchor::of(&*clock);
        NativeDuplexModel {
            label: host.label(),
            caps: crate::duplex::realtime_capabilities(host.capabilities()),
            host,
            audio_timeout,
            clock,
            anchor,
        }
    }
}

impl Realtime for NativeDuplexModel {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    /// Turn detection is inherent to a duplex model, so it is never asked to be off.
    fn session(
        &self,
        _turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let (host, gate) = self.host.open().await?;
            let gate = gate.unwrap_or_else(|| Box::new(AdaptiveNoiseGate::default()));
            let segmenter = Segmenter::new(
                self.label.clone(),
                self.audio_timeout,
                format!("{}_", crate::ids::shortuuid("item_")),
            );
            let (delivery, events) = queue::events();
            Ok(Arc::new(DuplexSession {
                host,
                label: self.label.clone(),
                delivery,
                events: AsyncMutex::new(events),
                changed: Notify::new(),
                gate: Mutex::new(gate),
                caps: self.caps,
                clock: Arc::clone(&self.clock),
                anchor: self.anchor,
                state: Mutex::new(State {
                    seg: segmenter,
                    closed: false,
                    config: Some(Config::default()),
                }),
            }) as Arc<dyn RealtimeSession>)
        })
    }
}

/// The startup configuration, collected until the session's replay ends with its tools.
#[derive(Default)]
struct Config {
    instructions: Option<String>,
    items: Option<Vec<ChatItem>>,
}

struct State {
    seg: Segmenter,
    closed: bool,
    config: Option<Config>,
}

pub struct DuplexSession {
    host: Arc<dyn DuplexHostSession>,
    label: String,
    delivery: Delivery,
    events: AsyncMutex<EventReceiver>,
    changed: Notify,
    gate: Mutex<Box<dyn AudioGate>>,
    caps: RealtimeCapabilities,
    clock: Arc<dyn Clock>,
    anchor: WallAnchor,
    state: Mutex<State>,
}

impl DuplexSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn publish(&self, events: Vec<Event>) -> bool {
        let mut terminal = false;
        for event in events {
            terminal |= matches!(&event, Event::Error(error) if !error.recoverable);
            if let Err(error) = self.delivery.send(event) {
                self.delivery.fail(self.clock.now(), &self.label, error);
                let tail = {
                    let mut state = self.lock();
                    state.closed = true;
                    state.seg.close()
                };
                if tail.deactivate_gate {
                    self.gate.lock().unwrap_or_else(|p| p.into_inner()).deactivate();
                }
                self.changed.notify_one();
                return false;
            }
        }
        if terminal {
            self.close_core();
        }
        true
    }

    fn apply(&self, step: Step) {
        if step.deactivate_gate {
            self.gate.lock().unwrap_or_else(|p| p.into_inner()).deactivate();
        }
        self.publish(step.events);
    }

    fn close_core(&self) {
        let step = {
            let mut state = self.lock();
            if state.closed {
                return;
            }
            state.closed = true;
            state.seg.close()
        };
        self.apply(step);
        self.delivery.finish();
        self.changed.notify_one();
    }

    fn on_host_event(&self, ev: DuplexEvent) {
        let now = self.clock.now();
        let wall = self.clock.wall();
        let gate_open = match &ev {
            DuplexEvent::Audio { frame, .. } => {
                Some(self.gate.lock().unwrap_or_else(|p| p.into_inner()).update(frame))
            }
            _ => None,
        };
        let step = {
            let mut st = self.lock();
            if st.closed {
                return;
            }
            match ev {
                DuplexEvent::Audio { frame, start_ms } => st.seg.on_audio(
                    now,
                    wall,
                    frame,
                    start_ms,
                    gate_open.expect("audio gate evaluated"),
                ),
                DuplexEvent::Transcript(f) => st.seg.on_transcript(now, wall, f),
                DuplexEvent::OutputEnded { error } => st.seg.on_output_ended(now, wall, error),
                DuplexEvent::Session(Event::FunctionCall(ready)) => {
                    st.seg.on_function_call(*ready.call).into()
                }
                DuplexEvent::Session(Event::SessionReconnected(_)) => st.seg.on_reconnected(),
                DuplexEvent::Session(Event::InputTranscription(t)) => {
                    let started_at = t
                        .turn_started_at
                        .filter(|s| s.is_finite())
                        .map(|s| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(s.max(0.0)))
                        .unwrap_or(wall);
                    st.seg.on_input_transcription(
                        &t.item_id,
                        &t.transcript,
                        t.is_final,
                        t.confidence,
                        started_at,
                    );
                    vec![Event::InputTranscription(t)].into()
                }
                DuplexEvent::Session(Event::GenerateReplyFailed(f)) => {
                    st.seg.forget_reply(&f.req_id);
                    vec![Event::GenerateReplyFailed(f)].into()
                }
                DuplexEvent::Session(ev) => vec![ev.anchored(Some(self.anchor))].into(),
            }
        };
        self.apply(step);
    }
}

impl RealtimeSession for DuplexSession {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        let mut commands = Vec::new();
        let mut events = Vec::new();
        let mut st = self.lock();
        if st.closed {
            return;
        }
        match cmd {
            RealtimeCommand::PushAudio(p) => commands.push(DuplexCommand::PushAudio(p.frame)),
            RealtimeCommand::UpdateInstructions(u) => match st.config.as_mut() {
                Some(c) => c.instructions = Some(u.instructions),
                None => commands.push(DuplexCommand::UpdateInstructions(u.instructions)),
            },
            RealtimeCommand::UpdateChatCtx(u) => {
                if st.config.is_some() {
                    let items = st.seg.seed(&u.chat_ctx);
                    st.config.as_mut().expect("checked").items = Some(items);
                } else {
                    let items = st.seg.update_chat_ctx(&u.chat_ctx);
                    if !items.is_empty() {
                        commands.push(DuplexCommand::AppendItems(items));
                    }
                }
            }
            RealtimeCommand::UpdateTools(u) => {
                let tools = u.tools.iter().map(|t| t.name.clone()).collect();
                match st.config.take() {
                    // the replay ends with the tools: the configuration goes as one unit
                    Some(c) => commands.push(DuplexCommand::Configure {
                        instructions: c.instructions,
                        items: c.items,
                        tools,
                    }),
                    None => commands.push(DuplexCommand::UpdateTools(tools)),
                }
            }
            RealtimeCommand::PushItem(p) => {
                st.seg.push_item((*p.item).clone());
                commands.push(DuplexCommand::AppendItems(vec![*p.item]));
            }
            RealtimeCommand::GenerateReply(g) => {
                events.extend(st.seg.generate_reply(self.clock.now(), g.req_id.clone()));
                commands.push(DuplexCommand::GenerateReply {
                    req_id: g.req_id,
                    instructions: g.instructions,
                    tool_choice: g.tool_choice,
                });
            }
            RealtimeCommand::CancelReply(c) => st.seg.forget_reply(&c.req_id),
            // input is consumed continuously and barge-in is the model's own; the model owns its
            // output timeline and decides itself what to say
            RealtimeCommand::Say(_)
            | RealtimeCommand::CommitAudio(_)
            | RealtimeCommand::ClearAudio(_)
            | RealtimeCommand::Interrupt(_)
            | RealtimeCommand::Truncate(_) => {}
        }
        drop(st);
        if !self.publish(events) {
            return;
        }
        self.changed.notify_one();
        for command in commands {
            self.host.send(command);
        }
    }

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move {
            let mut events = self.events.lock().await;
            loop {
                let deadline = self.lock().seg.next_deadline();
                let timer = async {
                    match deadline {
                        Some(at) => self.clock.wait_until(at).await,
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    biased;
                    event = events.next() => return event,
                    () = self.changed.notified() => {},
                    () = timer => {
                        let events = self.lock().seg.tick(self.clock.now());
                        self.apply(events);
                    },
                    ev = self.host.next_event() => match ev {
                        Some(ev) => self.on_host_event(ev),
                        None => {
                            self.on_host_event(DuplexEvent::OutputEnded { error: None });
                            self.close_core();
                        },
                    },
                }
            }
        })
    }

    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
        Box::pin(async move { self.lock().seg.chat_ctx().clone() })
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            self.close_core();
            self.host.aclose().await;
        })
    }
}

#[cfg(test)]
mod tests;
