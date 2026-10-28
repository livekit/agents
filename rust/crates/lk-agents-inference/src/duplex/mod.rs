//! `DuplexRealtimeAdapter` (`llm/duplex_adapter.py`): a full-duplex speech model (it listens and
//! speaks at once, handling interruptions itself) run as the session's realtime model. The model's
//! continuous output is gated, cut into generations and given its transcript here; the plugin's
//! session is reached through [`DuplexHost`].

pub mod gate;
#[cfg(feature = "python")]
mod python;
pub mod segmenter;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use lk_agents_abi::{
    AudioFrame, ChatContext, ChatItem, Clock, Realtime, RealtimeCommand, RealtimeSession, ToolChoice,
    WallAnchor,
};

pub use gate::{AdaptiveNoiseGate, AudioGate, FixedGate};
pub use segmenter::{AUDIO_TIMEOUT, Fragment, Segmenter};

/// `DuplexCapabilities`: what varies between duplex providers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DuplexCapabilities {
    pub user_transcription: bool,
    pub auto_tool_reply_generation: bool,
    pub mutable_chat_context: bool,
    pub mutable_instructions: bool,
    pub mutable_tools: bool,
}

impl DuplexCapabilities {
    /// The realtime capabilities a duplex model runs under: it detects turns and answers them
    /// itself, speaks only audio, and owns its output timeline.
    pub fn realtime(self) -> RealtimeCapabilities {
        RealtimeCapabilities {
            message_truncation: false,
            turn_detection: true,
            auto_turn_reply_generation: true,
            user_transcription: self.user_transcription,
            auto_tool_reply_generation: self.auto_tool_reply_generation,
            audio_output: true,
            manual_function_calls: false,
            can_disable_turn_detection: false,
            mutable_chat_context: self.mutable_chat_context,
            mutable_instructions: self.mutable_instructions,
            mutable_tools: self.mutable_tools,
            per_response_tool_choice: false,
            supports_say: false,
        }
    }
}

/// What the adapter tells the plugin's session.
#[derive(Debug, Clone)]
pub enum DuplexCommand {
    PushAudio(AudioFrame),
    UpdateInstructions(String),
    /// New chat items only: the adapter owns the context.
    AppendItems(Vec<ChatItem>),
    /// Tool names: the host owns the tool objects.
    UpdateTools(Vec<String>),
    GenerateReply { req_id: String, instructions: Option<String>, tool_choice: Option<ToolChoice> },
    /// The whole startup configuration at once (`_update_session`): a model whose configuration
    /// is immutable once started waits on it before connecting.
    Configure { instructions: Option<String>, items: Option<Vec<ChatItem>>, tools: Vec<String> },
}

/// What the plugin's session reports.
#[derive(Debug, Clone)]
pub enum DuplexEvent {
    /// One frame of the model's output, its own silence included.
    Audio { frame: AudioFrame, start_ms: Option<i64> },
    Transcript(Fragment),
    /// The output stream ended, or failed with `error`; it cannot be resumed.
    OutputEnded { error: Option<String> },
    /// Everything else, as a host realtime session reports it (the caller's turns, transcriptions,
    /// function calls, reconnects, errors, a refused reply).
    Session(Event),
}

/// A plugin's `DuplexModel`.
#[async_trait::async_trait]
pub trait DuplexHost: Send + Sync {
    fn label(&self) -> String;
    fn capabilities(&self) -> DuplexCapabilities;
    /// A new plugin session, and the gate the adapter or the model declares (`None`: the adaptive
    /// gate).
    async fn open(&self) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError>;
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

/// `DuplexRealtimeAdapter`'s engine: the session's realtime model.
pub struct DuplexModel {
    host: Arc<dyn DuplexHost>,
    label: String,
    caps: RealtimeCapabilities,
    audio_timeout: Duration,
    clock: Arc<dyn Clock>,
    anchor: WallAnchor,
}

impl DuplexModel {
    pub fn new(host: Arc<dyn DuplexHost>, audio_timeout: Duration, clock: Arc<dyn Clock>) -> Self {
        let anchor = WallAnchor::of(&*clock);
        DuplexModel { label: host.label(), caps: host.capabilities().realtime(), host, audio_timeout, clock, anchor }
    }
}

#[async_trait::async_trait]
impl Realtime for DuplexModel {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Turn detection is inherent to a duplex model, so it is never asked to be off.
    async fn session(&self, _turn_detection_disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let (host, gate) = self.host.open().await?;
        let gate = gate.unwrap_or_else(|| Box::new(AdaptiveNoiseGate::default()));
        let segmenter = Segmenter::new(gate, self.label.clone(), self.audio_timeout, Box::new(|| crate::ids::shortuuid("item_")));
        Ok(Arc::new(DuplexSession {
            host,
            caps: self.caps,
            clock: Arc::clone(&self.clock),
            anchor: self.anchor,
            state: Mutex::new(State { seg: segmenter, events: VecDeque::new(), config: Some(Config::default()) }),
        }))
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
    events: VecDeque<Event>,
    config: Option<Config>,
}

pub struct DuplexSession {
    host: Arc<dyn DuplexHostSession>,
    caps: RealtimeCapabilities,
    clock: Arc<dyn Clock>,
    anchor: WallAnchor,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<State>,
}

impl DuplexSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    fn on_host_event(&self, st: &mut State, ev: DuplexEvent) {
        let now = self.clock.now();
        let events = match ev {
            DuplexEvent::Audio { frame, start_ms } => st.seg.on_audio(now, frame, start_ms),
            DuplexEvent::Transcript(f) => st.seg.on_transcript(f),
            DuplexEvent::OutputEnded { error } => st.seg.on_output_ended(now, error),
            DuplexEvent::Session(Event::FunctionCall(ready)) => st.seg.on_function_call(*ready.call),
            DuplexEvent::Session(Event::SessionReconnected(_)) => st.seg.on_reconnected(),
            DuplexEvent::Session(Event::InputTranscription(t)) => {
                let started_at = t
                    .turn_started_at
                    .map(|s| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(s.max(0.0)))
                    .unwrap_or_else(SystemTime::now);
                st.seg.on_input_transcription(&t.item_id, &t.transcript, t.is_final, t.confidence, started_at);
                vec![Event::InputTranscription(t)]
            }
            DuplexEvent::Session(Event::GenerateReplyFailed(f)) => {
                st.seg.forget_reply(&f.req_id);
                vec![Event::GenerateReplyFailed(f)]
            }
            DuplexEvent::Session(ev) => vec![ev.anchored(Some(self.anchor))],
        };
        st.events.extend(events);
    }
}

#[async_trait::async_trait]
impl RealtimeSession for DuplexSession {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        let mut st = self.lock();
        match cmd {
            RealtimeCommand::PushAudio(p) => self.host.send(DuplexCommand::PushAudio(p.frame)),
            RealtimeCommand::UpdateInstructions(u) => match st.config.as_mut() {
                Some(c) => c.instructions = Some(u.instructions),
                None => self.host.send(DuplexCommand::UpdateInstructions(u.instructions)),
            },
            RealtimeCommand::UpdateChatCtx(u) => {
                if st.config.is_some() {
                    let items = st.seg.seed(&u.chat_ctx);
                    st.config.as_mut().expect("checked").items = Some(items);
                } else {
                    let items = st.seg.update_chat_ctx(&u.chat_ctx);
                    if !items.is_empty() {
                        self.host.send(DuplexCommand::AppendItems(items));
                    }
                }
            }
            RealtimeCommand::UpdateTools(u) => {
                let tools = u.tools.iter().map(|t| t.name.clone()).collect();
                match st.config.take() {
                    // the replay ends with the tools: the configuration goes as one unit
                    Some(c) => self.host.send(DuplexCommand::Configure { instructions: c.instructions, items: c.items, tools }),
                    None => self.host.send(DuplexCommand::UpdateTools(tools)),
                }
            }
            RealtimeCommand::PushItem(p) => {
                st.seg.push_item((*p.item).clone());
                self.host.send(DuplexCommand::AppendItems(vec![*p.item]));
            }
            RealtimeCommand::GenerateReply(g) => {
                let events = st.seg.generate_reply(self.clock.now(), g.req_id.clone());
                st.events.extend(events);
                self.host.send(DuplexCommand::GenerateReply {
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
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    async fn next_event(&self) -> Option<Event> {
        loop {
            let deadline = {
                let mut st = self.lock();
                if let Some(ev) = st.events.pop_front() {
                    return Some(ev);
                }
                st.seg.next_deadline()
            };
            let timer = async {
                match deadline {
                    Some(at) => self.clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                ev = self.host.next_event() => match ev {
                    Some(ev) => {
                        let mut st = self.lock();
                        self.on_host_event(&mut st, ev);
                    }
                    None => {
                        let mut st = self.lock();
                        let tail = st.seg.close();
                        st.events.extend(tail);
                        if st.events.is_empty() {
                            return None;
                        }
                    }
                },
                () = timer => {
                    let mut st = self.lock();
                    let events = st.seg.tick(self.clock.now());
                    st.events.extend(events);
                }
            }
        }
    }

    async fn chat_ctx(&self) -> ChatContext {
        self.lock().seg.chat_ctx().clone()
    }

    async fn aclose(&self) {
        let _ = self.lock().seg.close();
        self.host.aclose().await;
    }
}
