pub use lk_agents_abi::llm::chat_context::protocol::{
    encode_chat_item, encode_epoch_secs, encode_metrics, encode_role, encode_timestamp,
};
pub mod console;

use lk_agents_abi::ChatItem;
use lk_agents_abi::metrics::usage::{AgentSessionUsage, ModelUsage};
use lk_agents_abi::voice::amd::AmdCategory;
use lk_agents_abi::voice::events::AgentState;
pub use lk_agents_abi::worker::remote_session::{
    AgentInfo, IoUpdate, RtcStats, RunOutcome, SessionPort, SessionState, SimulationVerdict,
    VerdictOutcome,
};
use lk_agents_abi::worker::remote_session::{
    FunctionCallRef, ToolExecutionUpdatedWire, ToolUpdateWire, WireEvent,
};
use prost::Message as _;

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
            // `as f32`, as the reference narrows a float to the proto's `float`.
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

/// The session events `SessionHost` forwards to the client (`register_session`'s handlers, and the
/// AMD and end-of-turn back-channels), in their wire shapes, each with the stamp it keeps: a user
/// state change is back-dated to the speech boundary, every other event is sent stamped now. The
/// conversation items are the ones `conversation_item_added` is emitted for (messages and
/// handoffs); the core's other session events have no wire form here.
pub fn session_event(
    wire: &lk_agents_abi::voice::events::HostWire,
) -> Option<(WireEvent, Option<f64>)> {
    use lk_agents_abi::voice::events::{HostWire as W, ToolUpdate};
    use lk_agents_abi::worker::remote_session::{
        AmdPredictionWire, ErrorWire, OverlappingSpeechWire, SessionUsageUpdatedWire,
        ToolCallStartedWire,
    };
    let event = match wire {
        W::AgentStateChanged(ev) => WireEvent::AgentStateChanged(ev.clone()),
        W::UserStateChanged(ev) => {
            return Some((WireEvent::UserStateChanged(ev.clone()), Some(ev.created_at)));
        }
        W::UserTranscript(ev) => WireEvent::UserInputTranscribed(ev.clone()),
        W::EotPrediction(ev) => WireEvent::EotPrediction(ev.clone()),
        W::AmdResult(p) => WireEvent::AmdPrediction(AmdPredictionWire {
            speech_duration: p.speech_duration.as_secs_f64(),
            category: p.category,
            reason: p.reason.as_str().to_owned(),
            transcript: p.transcript.clone(),
            delay: p.delay.as_secs_f64(),
        }),
        W::AgentFalseInterruption(ev) => WireEvent::AgentFalseInterruption(ev.clone()),
        W::ItemAdded(ev) if matches!(ev.item, ChatItem::Message(_) | ChatItem::AgentHandoff(_)) => {
            WireEvent::ConversationItemAdded(ev.clone())
        }
        W::Error(ev) => WireEvent::Error(ErrorWire { message: ev.error.to_string() }),
        W::ToolExecutionUpdated(ev) => WireEvent::ToolExecutionUpdated(ToolExecutionUpdatedWire {
            update: match &ev.update {
                ToolUpdate::Started(started) => {
                    let call = &started.function_call;
                    ToolUpdateWire::Started(ToolCallStartedWire {
                        id: call.id.clone(),
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    })
                }
                ToolUpdate::Updated(updated) => ToolUpdateWire::CallUpdated(updated.clone()),
                ToolUpdate::Ended(ended) => ToolUpdateWire::Ended(ended.clone()),
                ToolUpdate::Reply(reply) => ToolUpdateWire::ReplyUpdated(reply.clone()),
            },
        }),
        W::FunctionToolsExecuted(ev) => tools_executed(ev),
        W::SessionUsageUpdated(ev) => WireEvent::SessionUsageUpdated(SessionUsageUpdatedWire {
            usage: AgentSessionUsage { model_usage: ev.model_usage.clone() },
        }),
        W::OverlappingSpeech(ev) => WireEvent::OverlappingSpeech(OverlappingSpeechWire {
            is_interruption: ev.event.is_interruption,
            overlap_started_at: ev.event.overlap_started_at,
            detection_delay: ev.event.detection_delay,
            detected_at: ev.event.detected_at,
        }),
        _ => return None,
    };
    Some((event, None))
}

/// `function_tools_executed`, as `SessionHost` forwards it: names, arguments and outputs only.
pub fn tools_executed(ev: &lk_agents_abi::voice::events::ToolsExecutedEvent) -> WireEvent {
    use lk_agents_abi::worker::remote_session::{FunctionCallOutputRef, FunctionToolsExecutedWire};
    WireEvent::FunctionToolsExecuted(FunctionToolsExecutedWire {
        function_calls: ev
            .function_calls
            .iter()
            .map(|c| FunctionCallRef {
                name: c.name.clone(),
                arguments: c.arguments.clone(),
                call_id: c.call_id.clone(),
            })
            .collect(),
        function_call_outputs: ev
            .function_call_outputs
            .iter()
            .map(|o| FunctionCallOutputRef {
                call_id: o.call_id.clone(),
                name: o.name.clone(),
                output: o.output.clone(),
                is_error: o.is_error,
            })
            .collect(),
    })
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

/// The result of one remote session operation, before its protocol encoding.
pub enum SessionReply {
    Pong,
    ChatHistory(Vec<ChatItem>),
    AgentInfo(AgentInfo),
    RunInput(RunOutcome),
    SessionState(SessionState),
    RtcStats(Option<RtcStats>),
    SessionUsage(Vec<ModelUsage>),
    FrameworkInfo(String, String),
    UpdateIo,
    FinalizeSimulation(VerdictOutcome),
}

/// One remote request over the supplied session port, before transport encoding.
/// Empty input is refused before entering the session, so it cannot interrupt the caller.
pub fn reply<'a>(
    port: &'a dyn SessionPort,
    request: &'a proto::SessionRequest,
) -> lk_agents_abi::BoxFuture<'a, Result<Option<SessionReply>, lk_agents_abi::errors::HostError>> {
    Box::pin(async move {
        use proto::session_request::Request;
        let Some(operation) = &request.request else { return Ok(None) };
        let reply = match operation {
            Request::Ping(_) => SessionReply::Pong,
            Request::GetChatHistory(_) => SessionReply::ChatHistory(port.chat_history().await?),
            Request::GetAgentInfo(_) => SessionReply::AgentInfo(port.agent_info().await?),
            Request::RunInput(input) if input.text.is_empty() => {
                SessionReply::RunInput((None, Some("empty run_input text".to_owned())))
            }
            Request::RunInput(input) => SessionReply::RunInput(port.run_input(&input.text).await?),
            Request::GetSessionState(_) => SessionReply::SessionState(port.session_state().await?),
            Request::GetRtcStats(_) => SessionReply::RtcStats(port.rtc_stats().await?),
            Request::GetSessionUsage(_) => SessionReply::SessionUsage(port.session_usage().await?),
            Request::GetFrameworkInfo(_) => {
                let (sdk, version) = port.framework_info();
                SessionReply::FrameworkInfo(sdk, version)
            }
            Request::UpdateIo(io) => {
                port.update_io(io_update(io)).await?;
                SessionReply::UpdateIo
            }
            Request::FinalizeSimulation(input) => SessionReply::FinalizeSimulation(
                port.finalize_simulation((
                    input.provisional_success,
                    input.provisional_reason.clone(),
                ))
                .await?,
            ),
        };
        Ok(Some(reply))
    })
}

/// Encode an observed operation result with the answer's supplied wall time.
pub fn encode_session_reply(
    request_id: String,
    reply: SessionReply,
    now: f64,
) -> proto::AgentSessionMessage {
    use proto::session_response::Response;
    let (error, response) = match reply {
        SessionReply::Pong => (None, Response::Pong(proto::session_response::Pong {})),
        SessionReply::ChatHistory(items) => (
            None,
            Response::GetChatHistory(proto::session_response::GetChatHistoryResponse {
                items: items.iter().map(encode_chat_item).collect(),
            }),
        ),
        SessionReply::AgentInfo((id, instructions, tools, chat_ctx)) => (
            None,
            Response::GetAgentInfo(proto::session_response::GetAgentInfoResponse {
                id,
                instructions,
                tools,
                chat_ctx: chat_ctx.iter().map(encode_chat_item).collect(),
            }),
        ),
        SessionReply::RunInput((items, error)) => {
            let items = if error.is_some() { Vec::new() } else { items.unwrap_or_default() };
            (
                error,
                Response::RunInput(proto::session_response::RunInputResponse {
                    items: items.iter().map(encode_chat_item).collect(),
                }),
            )
        }
        SessionReply::SessionState((agent_state, user_state, agent_id, options, started_at)) => {
            let agent_state = AgentState::from_wire(&agent_state).unwrap_or(AgentState::Idle);
            let user_state = UserState::from_wire(&user_state).unwrap_or(UserState::Listening);
            let started_at = started_at.filter(|t| *t != 0.0).unwrap_or(now);
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
        SessionReply::RtcStats(stats) => {
            let (publisher, subscriber) = stats.unwrap_or_default();
            let structs = |v: &[serde_json::Value]| v.iter().filter_map(json_to_struct).collect();
            (
                None,
                Response::GetRtcStats(proto::session_response::GetRtcStatsResponse {
                    publisher_stats: structs(&publisher),
                    subscriber_stats: structs(&subscriber),
                }),
            )
        }
        SessionReply::SessionUsage(model_usage) => (
            None,
            Response::GetSessionUsage(proto::session_response::GetSessionUsageResponse {
                usage: Some(encode_session_usage(&AgentSessionUsage { model_usage })),
                created_at: Some(encode_epoch_secs(now)),
            }),
        ),
        SessionReply::FrameworkInfo(sdk, sdk_version) => (
            None,
            Response::GetFrameworkInfo(proto::session_response::GetFrameworkInfoResponse {
                sdk,
                sdk_version,
            }),
        ),
        SessionReply::UpdateIo => {
            (None, Response::UpdateIo(proto::session_response::UpdateIoResponse {}))
        }
        SessionReply::FinalizeSimulation((verdict, error)) => {
            let verdict = if error.is_some() { None } else { verdict };
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
    proto::AgentSessionMessage {
        message: Some(proto::agent_session_message::Message::Response(proto::SessionResponse {
            request_id,
            error,
            response: Some(response),
        })),
    }
}

/// The opaque response to an operation failure; host exception details stay local.
pub fn internal_error_response(request_id: &str) -> proto::AgentSessionMessage {
    proto::AgentSessionMessage {
        message: Some(proto::agent_session_message::Message::Response(proto::SessionResponse {
            request_id: request_id.to_owned(),
            error: Some("internal error".to_owned()),
            response: None,
        })),
    }
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

pub use lk_agents_abi::voice::console::{ConsoleSink, SessionTransport};

/// Where a client's arriving events go: `RemoteSession.emit(name, event)`.
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
}

pub mod proto {
    pub use livekit_protocol::agent::*;
    pub use pbjson_types::{Duration, Struct, Timestamp};
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

    // Catches: a stateless decoder dropping frames that span reads.
    #[test]
    fn frames_reassemble_across_read_boundaries() {
        let mut d = FrameDecoder::new();
        let wire = frame(b"hello");
        let (a, b) = wire.split_at(3);
        d.push(a);
        assert_eq!(d.next_frame().unwrap(), None, "a partial header yields nothing");
        d.push(b);
        assert_eq!(d.next_frame().unwrap(), Some(b"hello".to_vec()));
    }
}

#[cfg(test)]
mod tests;
