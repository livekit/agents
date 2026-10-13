pub mod driver;

use async_trait::async_trait;
use lk_agents_abi::errors::HostError;
pub use lk_agents_abi::metrics::usage::AgentSessionUsage;
use lk_agents_abi::metrics::usage::ModelUsage;
pub use lk_agents_abi::voice::amd::AmdCategory;
pub use lk_agents_abi::voice::events::AgentState;
pub use lk_agents_abi::worker::remote_session::{
    AmdPredictionWire, DebugMessageWire, ErrorWire, FunctionCallOutputRef, FunctionCallRef,
    FunctionToolsExecutedWire, FunctionToolsStartedWire, OverlappingSpeechWire,
    SessionUsageUpdatedWire, ToolCallStartedWire, ToolExecutionUpdatedWire, ToolUpdateWire,
    WireEvent,
};
use lk_agents_abi::{ChatContent, ChatMessage, Extra, MetricsReport, Role};
pub use lk_agents_abi::{ChatItem, WebSocketSink, WebSocketStream};
use prost::Message as _;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The option keys a [`SessionState`] row's `options` carries, in the reference's order, and whether each is
/// a record the host expands first. Only the key set is pinned: values are Python `str()` output,
/// which stays with the host rather than a second `repr` implementation here.
#[lk_agents_abi::host(export(name = "session_state_option_fields"))]
pub const STATE_OPTION_FIELDS: &[(&str, bool)] = &[
    ("endpointing", true),
    ("interruption", true),
    ("max_tool_steps", false),
    ("user_away_timeout", false),
    ("transcription_timeout", false),
    ("preemptive_generation", true),
    ("min_consecutive_speech_delay", false),
    ("use_tts_aligned_transcript", false),
    ("ivr_detection", false),
];

/// `_TCP_HEADER_SIZE` (`remote_session.py:184`): big-endian `>I`.
pub const HEADER_SIZE: usize = 4;

/// `_TCP_MAX_MESSAGE_SIZE`: 1 MiB. A receive bound only, as in the reference.
pub const MAX_MESSAGE_SIZE: usize = 1 << 20;

/// `TOPIC_SESSION_MESSAGES` (`remote_session.py:57`): the byte-stream topic both halves register on.
#[lk_agents_abi::host(export(name = "session_messages_topic", pid = false))]
pub const TOPIC_SESSION_MESSAGES: &str = "lk.agent.session";

pub use lk_agents_abi::voice::events::UserState;

fn encode_agent_state(s: AgentState) -> proto::AgentState {
    match s {
        AgentState::Initializing => proto::AgentState::AsInitializing,
        AgentState::Idle => proto::AgentState::AsIdle,
        AgentState::Listening => proto::AgentState::AsListening,
        AgentState::Thinking => proto::AgentState::AsThinking,
        AgentState::Speaking => proto::AgentState::AsSpeaking,
    }
}

fn encode_user_state(s: UserState) -> proto::UserState {
    match s {
        UserState::Speaking => proto::UserState::UsSpeaking,
        UserState::Listening => proto::UserState::UsListening,
        UserState::Away => proto::UserState::UsAway,
    }
}

fn encode_amd_category(c: AmdCategory) -> proto::AmdCategory {
    match c {
        AmdCategory::Human => proto::AmdCategory::AmdHuman,
        AmdCategory::MachineIvr => proto::AmdCategory::AmdMachineIvr,
        AmdCategory::MachineVm => proto::AmdCategory::AmdMachineVm,
        AmdCategory::MachineUnavailable => proto::AmdCategory::AmdMachineUnavailable,
        AmdCategory::Uncertain => proto::AmdCategory::AmdUncertain,
    }
}

/// `_send_event`'s mapping (`remote_session.py:459`) from the facts the host reports, unstamped.
/// `Ok(None)` for a tool status outside `_TOOL_CALL_STATUS_MAP` / `_TOOL_REPLY_STATUS_MAP`
/// (`remote_session.py:295-301`), which the reference drops; a malformed debug payload is refused.
/// `language` and the hand-built tool calls' `id`/`created_at` go out unset, as in the reference.
pub fn encode_wire(ev: WireEvent) -> Result<Option<proto::agent_session_event::Event>, String> {
    use proto::agent_session_event::{self as pb, Event, tool_execution_updated as teu};
    let call = |c: FunctionCallRef| proto::FunctionCall {
        name: c.name,
        arguments: c.arguments,
        call_id: c.call_id,
        ..Default::default()
    };
    Ok(Some(match ev {
        WireEvent::AgentStateChanged(v) => Event::AgentStateChanged(pb::AgentStateChanged {
            old_state: encode_agent_state(v.old_state) as i32,
            new_state: encode_agent_state(v.new_state) as i32,
        }),
        WireEvent::UserStateChanged(v) => Event::UserStateChanged(pb::UserStateChanged {
            old_state: encode_user_state(v.old_state) as i32,
            new_state: encode_user_state(v.new_state) as i32,
        }),
        WireEvent::ConversationItemAdded(v) => {
            Event::ConversationItemAdded(pb::ConversationItemAdded {
                item: Some(encode_chat_item(&v.item)),
            })
        }
        WireEvent::UserInputTranscribed(v) => {
            Event::UserInputTranscribed(pb::UserInputTranscribed {
                transcript: v.transcript,
                is_final: v.is_final,
                language: v.language,
            })
        }
        WireEvent::FunctionToolsExecuted(v) => {
            Event::FunctionToolsExecuted(pb::FunctionToolsExecuted {
                function_calls: v.function_calls.into_iter().map(call).collect(),
                function_call_outputs: v
                    .function_call_outputs
                    .into_iter()
                    .map(|o| proto::FunctionCallOutput {
                        call_id: o.call_id,
                        name: o.name,
                        output: o.output,
                        is_error: o.is_error,
                        ..Default::default()
                    })
                    .collect(),
            })
        }
        WireEvent::Error(v) => Event::Error(pb::Error { message: v.message }),
        WireEvent::OverlappingSpeech(v) => Event::OverlappingSpeech(pb::OverlappingSpeech {
            is_interruption: v.is_interruption,
            overlap_started_at: v.overlap_started_at.map(encode_epoch_secs),
            detection_delay: v.detection_delay,
            detected_at: Some(encode_epoch_secs(v.detected_at)),
        }),
        WireEvent::SessionUsageUpdated(v) => Event::SessionUsageUpdated(pb::SessionUsageUpdated {
            usage: Some(encode_session_usage(&v.usage)),
        }),
        // Not `AgentSession` events: the AMD and EOT back-channels (`amd/detector.py:564`,
        // `agent_activity.py:2393`).
        WireEvent::AmdPrediction(v) => Event::AmdPrediction(pb::AmdPrediction {
            speech_duration: Some(encode_duration_secs(v.speech_duration)),
            category: encode_amd_category(v.category) as i32,
            reason: v.reason,
            transcript: v.transcript,
            delay: Some(encode_duration_secs(v.delay)),
        }),
        WireEvent::EotPrediction(v) => Event::EotPrediction(pb::EotPrediction {
            // `as f32`, as pyo3's own `f32` extraction does.
            probability: v.probability as f32,
            threshold: v.threshold as f32,
            inference_duration: Some(encode_duration_secs(v.inference_duration)),
            delay: Some(encode_duration_secs(v.delay)),
        }),
        // A dead arm in the reference, kept because a peer may still send it.
        WireEvent::FunctionToolsStarted(v) => {
            Event::FunctionToolsStarted(pb::FunctionToolsStarted {
                function_calls: v.function_calls.into_iter().map(call).collect(),
            })
        }
        WireEvent::DebugMessage(v) => Event::DebugMessage(proto::DebugMessage {
            payload: match v.payload {
                None => None,
                Some(raw) => {
                    decode_debug_message(&raw).map_err(|e| format!("not a DebugMessage: {e}"))?
                }
            },
        }),
        WireEvent::ToolExecutionUpdated(ToolExecutionUpdatedWire { update }) => {
            let update = match update {
                ToolUpdateWire::Started(v) => teu::Update::Started(teu::Started {
                    function_call: Some(proto::FunctionCall {
                        id: v.id,
                        call_id: v.call_id,
                        name: v.name,
                        arguments: v.arguments,
                        created_at: None,
                        extra: Default::default(),
                    }),
                }),
                ToolUpdateWire::CallUpdated(v) => teu::Update::CallUpdated(teu::CallUpdated {
                    id: v.id,
                    call_id: v.call_id,
                    message: v.message,
                }),
                ToolUpdateWire::Ended(v) => {
                    let status = match v.status.as_str() {
                        "done" => proto::ToolCallStatus::TcDone,
                        "error" => proto::ToolCallStatus::TcError,
                        "cancelled" => proto::ToolCallStatus::TcCancelled,
                        _ => return Ok(None),
                    };
                    // `message` unset means cancelled or an empty return.
                    teu::Update::Ended(teu::Ended {
                        id: v.id,
                        call_id: v.call_id,
                        message: v.message,
                        status: status as i32,
                    })
                }
                ToolUpdateWire::ReplyUpdated(v) => {
                    let status = match v.status.as_str() {
                        "scheduled" => proto::ToolReplyStatus::TrScheduled,
                        "completed" => proto::ToolReplyStatus::TrCompleted,
                        "interrupted" => proto::ToolReplyStatus::TrInterrupted,
                        "skipped" => proto::ToolReplyStatus::TrSkipped,
                        _ => return Ok(None),
                    };
                    teu::Update::ReplyUpdated(teu::ReplyUpdated {
                        update_ids: v.update_ids,
                        status: status as i32,
                        speech_id: v.speech_id,
                    })
                }
            };
            Event::ToolExecutionUpdated(pb::ToolExecutionUpdated { update: Some(update) })
        }
        WireEvent::AgentFalseInterruption(v) => {
            Event::AgentFalseInterruption(pb::AgentFalseInterruption { resumed: v.resumed })
        }
    }))
}

/// The name a client emits an arriving event under: `msg.event.WhichOneof("event")`
/// (`remote_session.py:1018`). Five of these are not in `RemoteSessionEventTypes`.
pub fn event_name(event: &proto::agent_session_event::Event) -> &'static str {
    use proto::agent_session_event::Event;
    match event {
        Event::AgentStateChanged(_) => "agent_state_changed",
        Event::UserStateChanged(_) => "user_state_changed",
        Event::ConversationItemAdded(_) => "conversation_item_added",
        Event::UserInputTranscribed(_) => "user_input_transcribed",
        Event::FunctionToolsExecuted(_) => "function_tools_executed",
        Event::Error(_) => "error",
        Event::OverlappingSpeech(_) => "overlapping_speech",
        Event::SessionUsageUpdated(_) => "session_usage_updated",
        Event::AmdPrediction(_) => "amd_prediction",
        Event::EotPrediction(_) => "eot_prediction",
        Event::FunctionToolsStarted(_) => "function_tools_started",
        Event::DebugMessage(_) => "debug_message",
        Event::ToolExecutionUpdated(_) => "tool_execution_updated",
        Event::AgentFalseInterruption(_) => "agent_false_interruption",
    }
}

/// Decode a [`DebugMessageWire`]'s serialized `DebugMessage` to its `Struct`; `Err` is refused.
pub fn decode_debug_message(raw: &[u8]) -> Result<Option<proto::Struct>, prost::DecodeError> {
    proto::DebugMessage::decode(raw).map(|m| m.payload)
}

/// What `SessionHost` asks of the session it hosts: the port the reference's reach-ins into
/// `AgentSession` become (`voice/remote_session.py`'s `_SessionPort`). An `Err` is a host failure
/// (answered `"internal error"`); a `(value, error)` row's error becomes the response's `error`
/// verbatim. The rows are the host's tuples, so each crosses as it is.
#[lk_agents_abi::host(proxy(
    host = "livekit.agents.voice.remote_session._SessionPort",
    cancel_on_drop
))]
#[async_trait]
pub trait SessionPort: Send + Sync {
    /// `session.history.items`.
    async fn chat_history(&self) -> Result<Vec<ChatItem>, HostError>;

    /// `session.current_agent`, flattened; instructions rendered with `modality="audio"`.
    async fn agent_info(&self) -> Result<AgentInfo, HostError>;

    /// `interrupt(force=True)`, then `await session.run(user_input=text)` and its items. An empty
    /// list is a silent turn, not a failure.
    async fn run_input(&self, text: &str) -> Result<RunOutcome, HostError>;

    async fn session_state(&self) -> Result<SessionState, HostError>;

    /// `session._room_io.room.get_rtc_stats()`; `None` without room IO.
    async fn rtc_stats(
        &self,
    ) -> Result<Option<(Vec<serde_json::Value>, Vec<serde_json::Value>)>, HostError>;

    /// `session.usage.model_usage`.
    async fn session_usage(&self) -> Result<Vec<ModelUsage>, HostError>;

    /// `("python", __version__)` on the reference. Read once.
    fn framework_info(&self) -> (String, String);

    /// Only the `Some` channels are applied.
    async fn update_io(&self, update: IoUpdate) -> Result<(), HostError>;

    /// Hand the agent the simulator's provisional verdict; return the agent's own
    /// (`on_simulation_end`). `(None, None)`: no simulation or no verdict.
    async fn finalize_simulation(
        &self,
        provisional: SimulationVerdict,
    ) -> Result<VerdictOutcome, HostError>;
}

/// `(id, instructions, tool names, chat_ctx items)`. `None` instructions differ from `""`; tools are
/// flattened across `Toolset`s, names only.
pub type AgentInfo = (String, Option<String>, Vec<String>, Vec<ChatItem>);

/// `(items, None)` or `(None, error)`.
pub type RunOutcome = (Option<Vec<ChatItem>>, Option<String>);

/// `(agent_state, user_state, agent_id, options, started_at)` as the host holds them: an unknown
/// state word answers `AS_IDLE` / `US_LISTENING`; `options` keys are [`STATE_OPTION_FIELDS`];
/// `started_at` is epoch seconds, `None` or `0` stamped with the answer time.
pub type SessionState = (String, String, String, HashMap<String, String>, Option<f64>);

/// `(publisher stats, subscriber stats)`, each `MessageToDict` of one stat.
pub type RtcStats = (Vec<serde_json::Value>, Vec<serde_json::Value>);

/// `(input audio, input video, output audio, output video, output transcription)`; `None` leaves
/// the channel alone.
pub type IoUpdate = (Option<bool>, Option<bool>, Option<bool>, Option<bool>, Option<bool>);

/// Absent inner field == absent submessage, as the reference checks `HasField("input")` first.
pub fn io_update(io: &proto::session_request::UpdateIo) -> IoUpdate {
    let (i, o) = (io.input.as_ref(), io.output.as_ref());
    (
        i.and_then(|i| i.audio_enabled),
        i.and_then(|i| i.video_enabled),
        o.and_then(|o| o.audio_enabled),
        o.and_then(|o| o.video_enabled),
        o.and_then(|o| o.transcription_enabled),
    )
}

/// `(success, reason)`.
pub type SimulationVerdict = (bool, String);

/// `(verdict, None)` or `(None, error)`.
pub type VerdictOutcome = (Option<SimulationVerdict>, Option<String>);

/// `request.WhichOneof("request")`, used in logs and client errors.
pub fn request_name(request: &proto::session_request::Request) -> &'static str {
    use proto::session_request::Request as R;
    match request {
        R::Ping(_) => "ping",
        R::GetChatHistory(_) => "get_chat_history",
        R::GetAgentInfo(_) => "get_agent_info",
        R::RunInput(_) => "run_input",
        R::GetSessionState(_) => "get_session_state",
        R::GetRtcStats(_) => "get_rtc_stats",
        R::GetSessionUsage(_) => "get_session_usage",
        R::GetFrameworkInfo(_) => "get_framework_info",
        R::UpdateIo(_) => "update_io",
        R::FinalizeSimulation(_) => "finalize_simulation",
    }
}

pub struct SessionHost {
    port: Arc<dyn SessionPort>,
    sdk: (String, String),
}

impl SessionHost {
    pub fn new(port: Arc<dyn SessionPort>) -> Self {
        let sdk = port.framework_info();
        SessionHost { port, sdk }
    }

    /// `_handle_request` (`remote_session.py:715`). `Err` is a host failure. `Ok(None)` for an
    /// unset or unknown arm: the reference sends no response (the client times out); kept for parity.
    pub async fn handle_request(
        &self,
        req: &proto::SessionRequest,
    ) -> Result<Option<proto::AgentSessionMessage>, HostError> {
        use proto::session_request::Request;
        use proto::session_response::Response;

        let request_id = req.request_id.clone();
        let Some(request) = req.request.as_ref() else { return Ok(None) };
        let (error, response) = match request {
            Request::Ping(_) => (None, Response::Pong(proto::session_response::Pong {})),

            Request::GetChatHistory(_) => (
                None,
                Response::GetChatHistory(proto::session_response::GetChatHistoryResponse {
                    items: self.port.chat_history().await?.iter().map(encode_chat_item).collect(),
                }),
            ),

            Request::GetAgentInfo(_) => {
                let (id, instructions, tools, chat_ctx) = self.port.agent_info().await?;
                (
                    None,
                    Response::GetAgentInfo(proto::session_response::GetAgentInfoResponse {
                        id,
                        instructions,
                        tools,
                        chat_ctx: chat_ctx.iter().map(encode_chat_item).collect(),
                    }),
                )
            }

            Request::RunInput(r) => {
                // Checked before the interrupt, so an empty request doesn't disturb the agent.
                let (error, items) = if r.text.is_empty() {
                    (Some("empty run_input text".to_owned()), Vec::new())
                } else {
                    match self.port.run_input(&r.text).await? {
                        (_, Some(e)) => (Some(e), Vec::new()),
                        (items, None) => (None, items.unwrap_or_default()),
                    }
                };
                // Both `error` and the empty `run_input` are set, as the reference does.
                (
                    error,
                    Response::RunInput(proto::session_response::RunInputResponse {
                        items: items.iter().map(encode_chat_item).collect(),
                    }),
                )
            }

            Request::GetSessionState(_) => {
                let (agent_state, user_state, agent_id, options, started_at) =
                    self.port.session_state().await?;
                let agent_state = AgentState::from_wire(&agent_state).unwrap_or(AgentState::Idle);
                let user_state = UserState::from_wire(&user_state).unwrap_or(UserState::Listening);
                // `session._started_at or time.time()`
                let started_at = started_at.filter(|t| *t != 0.0).unwrap_or_else(epoch_secs_now);
                (
                    None,
                    Response::GetSessionState(proto::session_response::GetSessionStateResponse {
                        agent_state: encode_agent_state(agent_state) as i32,
                        user_state: encode_user_state(user_state) as i32,
                        agent_id,
                        options: options.into_iter().collect(),
                        created_at: Some(encode_epoch_secs(started_at)),
                    }),
                )
            }

            Request::GetRtcStats(_) => {
                let (publisher, subscriber) = self.port.rtc_stats().await?.unwrap_or_default();
                let structs =
                    |v: &[serde_json::Value]| v.iter().filter_map(json_to_struct).collect();
                (
                    None,
                    Response::GetRtcStats(proto::session_response::GetRtcStatsResponse {
                        publisher_stats: structs(&publisher),
                        subscriber_stats: structs(&subscriber),
                    }),
                )
            }

            Request::GetSessionUsage(_) => (
                None,
                Response::GetSessionUsage(proto::session_response::GetSessionUsageResponse {
                    usage: Some(encode_session_usage(&AgentSessionUsage {
                        model_usage: self.port.session_usage().await?,
                    })),
                    // `time.time()` at answer time, not the session's start.
                    created_at: Some(encode_timestamp(SystemTime::now())),
                }),
            ),

            Request::GetFrameworkInfo(_) => (
                None,
                Response::GetFrameworkInfo(proto::session_response::GetFrameworkInfoResponse {
                    sdk: self.sdk.0.clone(),
                    sdk_version: self.sdk.1.clone(),
                }),
            ),

            Request::UpdateIo(io) => {
                self.port.update_io(io_update(io)).await?;
                (None, Response::UpdateIo(proto::session_response::UpdateIoResponse {}))
            }

            Request::FinalizeSimulation(f) => {
                let provisional = (f.provisional_success, f.provisional_reason.clone());
                let (error, verdict) = match self.port.finalize_simulation(provisional).await? {
                    (_, Some(e)) => (Some(e), None),
                    (v, None) => (None, v),
                };
                (
                    error,
                    Response::FinalizeSimulation(proto::session_response::FinalizeSimulationResponse {
                        user_verdict: verdict.map(|(success, reason)| {
                            proto::session_response::finalize_simulation_response::SimulationVerdict {
                                success,
                                reason,
                            }
                        }),
                    }),
                )
            }
        };

        Ok(Some(proto::AgentSessionMessage {
            message: Some(proto::agent_session_message::Message::Response(
                proto::SessionResponse { request_id, error, response: Some(response) },
            )),
        }))
    }

    /// `_handle_request_safe`'s catch arm: the opaque `"internal error"`, never the exception text.
    pub fn internal_error_response(request_id: &str) -> proto::AgentSessionMessage {
        proto::AgentSessionMessage {
            message: Some(proto::agent_session_message::Message::Response(
                proto::SessionResponse {
                    request_id: request_id.to_owned(),
                    error: Some("internal error".to_owned()),
                    response: None,
                },
            )),
        }
    }
}

fn epoch_secs_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Stamp an encoded event and wrap it in a message. A non-finite stamp is refused.
pub fn stamp_event(
    mut event: proto::AgentSessionEvent,
    created_at: f64,
) -> Result<proto::AgentSessionMessage, NonFiniteStamp> {
    if !created_at.is_finite() {
        return Err(NonFiniteStamp);
    }
    event.created_at = Some(encode_epoch_secs(created_at));
    Ok(proto::AgentSessionMessage {
        message: Some(proto::agent_session_message::Message::Event(event)),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("created_at must be a finite epoch-seconds value")]
pub struct NonFiniteStamp;

/// `SessionTransport` (`remote_session.py:78`), as `voice/remote_session.py`'s `_CoreTransport`
/// sees it. A host failure keeps the original exception.
#[lk_agents_abi::host(proxy(
    host = "livekit.agents.voice.remote_session._CoreTransport",
    cancel_on_drop
))]
#[async_trait]
pub trait SessionTransport: Send + Sync {
    /// `start()`; idempotent.
    async fn start(&self) -> Result<(), HostError>;

    /// `send_message`. Failures are raised: only the caller knows whether anyone waits on it.
    async fn send_message(&self, msg: &proto::AgentSessionMessage) -> Result<(), HostError>;

    /// `send_message_threadsafe`: enqueue and forget, dropped when closed. Not ordered against
    /// `send_message`.
    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage);

    /// The next message (`__anext__`); `Ok(None)` is `StopAsyncIteration`, `Err` ends the loop.
    /// Only the receive loop reads.
    async fn recv(&self) -> Result<Option<proto::AgentSessionMessage>, HostError>;

    /// `close()`; idempotent, never fails.
    async fn close(&self);
}

/// The console's local legs (`TcpAudioInput.push_frame`, `TcpAudioOutput.notify_playout_finished`),
/// on `SessionHost`.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.remote_session.SessionHost"))]
pub trait ConsoleSink: Send + Sync {
    fn push_input_frame(&self, frame: &proto::agent_session_message::console_io::AudioFrame);

    fn playout_finished(&self);
}

/// Where a client's arriving events go: `RemoteSession.emit(name, event)`.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.remote_session.RemoteSession"))]
pub trait ClientSink: Send + Sync {
    fn event(&self, name: &str, event: &proto::AgentSessionEvent);
}

/// `struct.pack(">I", len(data)) + data`, as one buffer so no sender can splice between header and body.
#[lk_agents_abi::host(export)]
pub fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_SIZE + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Why the reader gave up (all `StopAsyncIteration` in the reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// `logger.error("TCP message too large: %d bytes", length)`.
    #[error("frame of {0} bytes exceeds the {MAX_MESSAGE_SIZE} byte limit")]
    Oversize(u32),
}

/// Reassembles length-prefixed frames from a byte stream.
#[lk_agents_abi::class(mutable)]
#[derive(Debug, Default)]
pub struct FrameDecoder {
    buf: Vec<u8>,
}

#[lk_agents_abi::host(object)]
impl FrameDecoder {
    #[host(new)]
    pub fn new() -> Self {
        FrameDecoder { buf: Vec::new() }
    }

    /// Append any chunk, including one spanning several frames.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// The next complete frame, or `Ok(None)`. The length is checked on the header alone.
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        if self.buf.len() < HEADER_SIZE {
            return Ok(None);
        }
        let len = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]);
        if len as usize > MAX_MESSAGE_SIZE {
            return Err(FrameError::Oversize(len));
        }
        let total = HEADER_SIZE + len as usize;
        if self.buf.len() < total {
            return Ok(None);
        }
        let body = self.buf[HEADER_SIZE..total].to_vec();
        self.buf.drain(..total);
        Ok(Some(body))
    }

    /// Bytes held but not yet a frame; non-zero at end of stream is a truncated frame.
    #[host(getter)]
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Push `chunk` and return every complete frame. An oversize frame is always an error.
    #[host(method)]
    pub fn decode_frames(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, FrameError> {
        self.push(chunk);
        let mut out = Vec::new();
        while let Some(f) = self.next_frame()? {
            out.push(f);
        }
        Ok(out)
    }
}

pub mod proto {
    pub use livekit_protocol::agent::*;
    pub use pbjson_types::{Duration, Struct, Timestamp};
}

/// `encode_timestamp` (`_proto.py:47`). Pre-epoch clamps to the epoch.
pub fn encode_timestamp(t: SystemTime) -> proto::Timestamp {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    proto::Timestamp { seconds: d.as_secs() as i64, nanos: d.subsec_nanos() as i32 }
}

pub fn encode_epoch_secs(t: f64) -> proto::Timestamp {
    let nanos = (t * 1e9) as i128;
    proto::Timestamp {
        seconds: (nanos.div_euclid(1_000_000_000)) as i64,
        nanos: (nanos.rem_euclid(1_000_000_000)) as i32,
    }
}

/// `Duration.FromNanoseconds(int(x * 1e9))`. Unlike `Timestamp`, a proto `Duration`'s `nanos` must
/// share the sign of `seconds`, so -0.25 s is `(0, -250_000_000)`, as Python renormalises.
pub fn encode_duration_secs(secs: f64) -> proto::Duration {
    let nanos = (secs * 1e9) as i128;
    let mut s = nanos.div_euclid(1_000_000_000);
    let mut n = nanos.rem_euclid(1_000_000_000);
    if s < 0 && n > 0 {
        s += 1;
        n -= 1_000_000_000;
    }
    proto::Duration { seconds: s as i64, nanos: n as i32 }
}

/// `ROLES` (`_proto.py:30`). Total, so there is no default arm to get wrong.
pub fn encode_role(role: Role) -> proto::ChatRole {
    match role {
        Role::Developer => proto::ChatRole::Developer,
        Role::System => proto::ChatRole::System,
        Role::User => proto::ChatRole::User,
        Role::Assistant => proto::ChatRole::Assistant,
    }
}

/// `encode_metrics` (`_proto.py:76`). Ten of fifteen fields survive.
pub fn encode_metrics(m: &MetricsReport) -> proto::MetricsReport {
    proto::MetricsReport {
        started_speaking_at: m.started_speaking_at.map(encode_epoch_secs),
        stopped_speaking_at: m.stopped_speaking_at.map(encode_epoch_secs),
        transcription_delay: m.transcription_delay,
        end_of_turn_delay: m.end_of_turn_delay,
        on_user_turn_completed_delay: m.on_user_turn_completed_delay,
        llm_node_ttft: m.llm_node_ttft,
        tts_node_ttfb: m.tts_node_ttfb,
        e2e_latency: m.e2e_latency,
        llm_node_tps: m.llm_node_tps,
        llm_node_ttfs: m.llm_node_ttfs,
        // dropped: the proto declares none of the metadata fields.
    }
}

/// `{k: str(v) ...}` (`_proto.py:88`), but compact JSON rather than Python `str(dict)`. `extra` does
/// not round-trip in either implementation.
fn encode_extra(extra: &Extra) -> std::collections::HashMap<String, String> {
    extra
        .iter()
        .map(|(k, v)| {
            let s = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (k.clone(), s)
        })
        .collect()
}

/// `encode_chat_item` (`_proto.py:80`).
pub fn encode_chat_item(item: &ChatItem) -> proto::chat_context::ChatItem {
    use proto::chat_context::chat_item::Item;
    let inner = match item {
        ChatItem::Message(m) => Item::Message(encode_chat_message(m)),
        ChatItem::FunctionCall(f) => Item::FunctionCall(proto::FunctionCall {
            id: f.id.clone(),
            call_id: f.call_id.clone(),
            arguments: f.arguments.clone(),
            name: f.name.clone(),
            created_at: Some(encode_timestamp(f.created_at)),
            extra: encode_extra(&f.extra),
            // dropped: group_id
        }),
        ChatItem::FunctionCallOutput(f) => Item::FunctionCallOutput(proto::FunctionCallOutput {
            id: f.id.clone(),
            name: f.name.clone(),
            call_id: f.call_id.clone(),
            output: f.output.clone(),
            is_error: f.is_error,
            created_at: Some(encode_timestamp(f.created_at)),
            // dropped: reply_required
        }),
        ChatItem::AgentHandoff(h) => Item::AgentHandoff(proto::AgentHandoff {
            id: h.id.clone(),
            old_agent_id: h.old_agent_id.clone(),
            new_agent_id: h.new_agent_id.clone(),
            created_at: Some(encode_timestamp(h.created_at)),
        }),
        // An empty instructions string is absent, as in the reference (proto3 would tell them apart).
        ChatItem::AgentConfigUpdate(c) => Item::AgentConfigUpdate(proto::AgentConfigUpdate {
            id: c.id.clone(),
            instructions: c.instructions.clone().filter(|s| !s.is_empty()),
            tools_added: c.tools_added.clone().unwrap_or_default(),
            tools_removed: c.tools_removed.clone().unwrap_or_default(),
            created_at: Some(encode_timestamp(c.created_at)),
        }),
    };
    proto::chat_context::ChatItem { item: Some(inner) }
}

fn encode_chat_message(m: &ChatMessage) -> proto::ChatMessage {
    use proto::chat_message::{ChatContent as PbContent, chat_content::Payload};
    proto::ChatMessage {
        id: m.id.clone(),
        role: encode_role(m.role) as i32,
        // Image and audio content are filtered out, not rendered.
        content: m
            .content
            .iter()
            .filter_map(|c| match c {
                ChatContent::Text(t) => Some(PbContent { payload: Some(Payload::Text(t.clone())) }),
                ChatContent::Image(_) | ChatContent::Audio(_) => None,
            })
            .collect(),
        interrupted: m.interrupted,
        transcript_confidence: m.transcript_confidence,
        extra: encode_extra(&m.extra),
        metrics: Some(encode_metrics(&m.metrics)),
        created_at: Some(encode_timestamp(m.created_at)),
    }
}

/// `encode_session_usage` (`_proto.py:113`).
pub fn encode_session_usage(usage: &AgentSessionUsage) -> proto::AgentSessionUsage {
    use proto::model_usage::Usage;
    proto::AgentSessionUsage {
        model_usage: usage
            .model_usage
            .iter()
            .map(|mu| proto::ModelUsage {
                usage: Some(match mu {
                    ModelUsage::Llm(u) => Usage::Llm(proto::LlmModelUsage {
                        provider: u.provider.clone(),
                        model: u.model.clone(),
                        input_tokens: u.input_tokens as i32,
                        input_cached_tokens: u.input_cached_tokens as i32,
                        input_audio_tokens: u.input_audio_tokens as i32,
                        input_cached_audio_tokens: u.input_cached_audio_tokens as i32,
                        input_text_tokens: u.input_text_tokens as i32,
                        input_cached_text_tokens: u.input_cached_text_tokens as i32,
                        input_image_tokens: u.input_image_tokens as i32,
                        input_cached_image_tokens: u.input_cached_image_tokens as i32,
                        output_tokens: u.output_tokens as i32,
                        output_audio_tokens: u.output_audio_tokens as i32,
                        output_text_tokens: u.output_text_tokens as i32,
                        session_duration: u.session_duration,
                        // dropped: input_cache_creation_tokens, which the wire cannot report.
                    }),
                    ModelUsage::Tts(u) => Usage::Tts(proto::TtsModelUsage {
                        provider: u.provider.clone(),
                        model: u.model.clone(),
                        input_tokens: u.input_tokens as i32,
                        output_tokens: u.output_tokens as i32,
                        characters_count: u.characters_count as i32,
                        audio_duration: u.audio_duration,
                    }),
                    ModelUsage::Stt(u) => Usage::Stt(proto::SttModelUsage {
                        provider: u.provider.clone(),
                        model: u.model.clone(),
                        input_tokens: u.input_tokens as i32,
                        output_tokens: u.output_tokens as i32,
                        audio_duration: u.audio_duration,
                    }),
                    ModelUsage::Interruption(u) => {
                        Usage::Interruption(proto::InterruptionModelUsage {
                            provider: u.provider.clone(),
                            model: u.model.clone(),
                            total_requests: u.total_requests as i32,
                        })
                    }
                    ModelUsage::Eot(u) => Usage::Eot(proto::EotModelUsage {
                        provider: u.provider.clone(),
                        model: u.model.clone(),
                        total_requests: u.total_requests as i32,
                    }),
                }),
            })
            .collect(),
    }
}

/// `MessageToDict` then `Struct().update(d)` (`remote_session.py:833-841`). Only an object becomes
/// a `Struct`; anything else is `None`.
pub fn json_to_struct(v: &serde_json::Value) -> Option<proto::Struct> {
    match v {
        serde_json::Value::Object(map) => Some(proto::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect(),
        }),
        _ => None,
    }
}

fn json_to_value(v: &serde_json::Value) -> pbjson_types::Value {
    use pbjson_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        // Precision loss past 2^53 is shared with the Python and TS implementations.
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(a) => Kind::ListValue(pbjson_types::ListValue {
            values: a.iter().map(json_to_value).collect(),
        }),
        serde_json::Value::Object(m) => Kind::StructValue(pbjson_types::Struct {
            fields: m.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect(),
        }),
    };
    pbjson_types::Value { kind: Some(kind) }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    // Catches: a little-endian length header (the reader would stall on a real peer).
    #[test]
    fn the_frame_header_is_four_big_endian_bytes() {
        assert_eq!(frame(b"ab"), [0, 0, 0, 2, b'a', b'b']);
    }

    // Catches: a stateless `decode_frames` dropping frames that span reads.
    #[test]
    fn frames_reassemble_across_read_boundaries() {
        let mut d = FrameDecoder::new();
        let wire = frame(b"hello");
        let (a, b) = wire.split_at(3);
        assert!(d.decode_frames(a).unwrap().is_empty(), "a partial header yields nothing");
        assert_eq!(d.pending(), 3);
        assert_eq!(d.decode_frames(b).unwrap(), [b"hello".to_vec()]);
        assert_eq!(d.pending(), 0, "a truncated tail at EOS is what pending detects");
    }
}
