//! The voice session: its lifecycle, agent activities, RPC, AMD and keyterm detection.

use super::agent_turn::AgentTurn;
use super::common::{GenAiOperation, ParticipantAttrs};
use super::job::JobEntrypoint;
use crate::telemetry::decl::{Attrs, Content, Current, Seconds, SpanValue, event, span};
use crate::telemetry::semconv::{otel, protocol};
use crate::voice::events::{AgentState, UserState};
use crate::voice::{amd, rtc};

/// The agent session, `session.start()` to the end of close.
#[span(
    name = "agent_session",
    parent = JobEntrypoint | Current,
    end = AgentSessionEnd,
    events = [
        ParticipantLinked, AgentStateChanged, UserStateChanged, ConnectionStateChanged,
        ParticipantDisconnected, EventLoopBlockedEvent
    ],
    handle = Manual
)]
#[derive(Attrs)]
pub struct AgentSession {
    /// Label of the agent running the activity.
    #[attr("lk.agent_label")]
    pub agent_label: String,
    /// `invoke_workflow`.
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    /// `agent_session`.
    #[attr(otel::GEN_AI_WORKFLOW_NAME)]
    pub workflow: String,
    #[attr("lk.dispatch_id")]
    pub dispatch_id: Option<String>,
    #[attr("lk.job.worker_id")]
    pub worker_id: Option<String>,
    #[attr("lk.job.agent_id")]
    pub agent_id: Option<String>,
}

#[derive(Attrs)]
pub struct AgentSessionEnd {
    /// Event-loop stalls detected in the session.
    #[attr("lk.blocking.count")]
    pub blocking_count: Option<u32>,
    /// Seconds the event loop was stalled in total.
    #[attr("lk.blocking.total_duration")]
    pub blocking_total: Option<Seconds>,
    /// The longest stall, in seconds.
    #[attr("lk.blocking.max_duration")]
    pub blocking_max: Option<Seconds>,
    /// Phone number of a linked SIP participant.
    #[attr("lk.pii.sip.phoneNumber")]
    pub sip_phone_number: Option<Content<String>>,
}

/// The session linked to its participant.
#[event(name = "participant_linked")]
#[derive(Attrs)]
pub struct ParticipantLinked {
    #[attrs]
    pub participant: ParticipantAttrs,
}

/// The agent's state changed.
#[event(name = "agent_state_changed")]
#[derive(Attrs)]
pub struct AgentStateChanged {
    /// State before the transition.
    #[attr("lk.old_state")]
    pub old: ConversationState,
    /// State after the transition.
    #[attr("lk.new_state")]
    pub new: ConversationState,
}

/// The user's state changed.
#[event(name = "user_state_changed")]
#[derive(Attrs)]
pub struct UserStateChanged {
    /// The change's instant, never after the step that records it.
    #[at]
    pub at: crate::Ts,
    #[attr("lk.old_state")]
    pub old: ConversationState,
    #[attr("lk.new_state")]
    pub new: ConversationState,
}

/// The agent's or the user's state.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    Initializing,
    Idle,
    Listening,
    Thinking,
    Speaking,
    Away,
}

impl From<AgentState> for ConversationState {
    fn from(state: AgentState) -> Self {
        match state {
            AgentState::Initializing => Self::Initializing,
            AgentState::Idle => Self::Idle,
            AgentState::Listening => Self::Listening,
            AgentState::Thinking => Self::Thinking,
            AgentState::Speaking => Self::Speaking,
        }
    }
}

impl From<UserState> for ConversationState {
    fn from(state: UserState) -> Self {
        match state {
            UserState::Speaking => Self::Speaking,
            UserState::Listening => Self::Listening,
            UserState::Away => Self::Away,
        }
    }
}

/// The room connection state changed.
#[event(name = "connection_state_changed")]
#[derive(Attrs)]
pub struct ConnectionStateChanged {
    /// Room connection state.
    #[attr("lk.connection_state")]
    pub state: ConnectionState,
}

/// A room connection state, as the protocol enum names it.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    #[wire = "CONN_DISCONNECTED"]
    Disconnected,
    #[wire = "CONN_CONNECTED"]
    Connected,
    #[wire = "CONN_RECONNECTING"]
    Reconnecting,
}

impl From<rtc::ConnectionState> for ConnectionState {
    fn from(state: rtc::ConnectionState) -> Self {
        match state {
            rtc::ConnectionState::Disconnected => Self::Disconnected,
            rtc::ConnectionState::Connected => Self::Connected,
            rtc::ConnectionState::Reconnecting => Self::Reconnecting,
        }
    }
}

/// The linked participant left the room.
#[event(name = "participant_disconnected")]
#[derive(Attrs)]
pub struct ParticipantDisconnected {
    #[attrs]
    pub participant: ParticipantAttrs,
    #[attr(protocol::LK_DISCONNECT_REASON)]
    pub reason: String,
}

/// An event-loop stall, mirroring the `event_loop_blocked` span.
#[event(name = "event_loop_blocked")]
#[derive(Attrs)]
pub struct EventLoopBlockedEvent {
    /// Seconds the event loop was blocked.
    #[attr("lk.blocking.duration")]
    pub duration: Seconds,
}

/// `start()` to the first agent's `on_enter`.
#[span(name = "session_start", parent = AgentSession, handle = Manual)]
#[derive(Attrs)]
pub struct SessionStart;

/// Closing the session.
#[span(name = "session_close", parent = AgentSession, handle = Manual)]
#[derive(Attrs)]
pub struct SessionClose {
    /// Why the session closed.
    #[attr("lk.close_reason")]
    pub reason: CloseReason,
    /// The close drains the current speech first.
    #[attr("lk.close.drain")]
    pub drain: bool,
    /// The close error's type, when closed by an error.
    #[attr(otel::EXCEPTION_TYPE)]
    pub error: Option<String>,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    Error,
    JobShutdown,
    ParticipantDisconnected,
    UserInitiated,
    TaskCompleted,
}

/// A handoff: the old agent's exit to the new one's entry.
#[span(name = "update_agent", parent = AgentSession, handle = Manual)]
#[derive(Attrs)]
pub struct UpdateAgent {
    /// Label of the agent handed off from.
    #[attr("lk.previous_agent_label")]
    pub previous_agent_label: String,
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// Starting an agent's activity.
#[span(name = "start_agent_activity", parent = SessionStart | UpdateAgent, handle = Manual)]
#[derive(Attrs)]
pub struct StartAgentActivity {
    #[attr("lk.agent_label")]
    pub agent_label: String,
    /// `create_agent`.
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_AGENT_NAME)]
    pub agent_name: String,
    /// The agent's LLM provider.
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: Option<String>,
    /// The agent's LLM.
    #[attr(otel::GEN_AI_REQUEST_MODEL)]
    pub model: Option<String>,
}

/// Setting up the agent's toolsets.
#[span(name = "setup_toolsets", parent = StartAgentActivity)]
#[derive(Attrs)]
pub struct SetupToolsets;

/// The agent's `on_enter` hook.
#[span(
    name = "on_enter",
    parent = StartAgentActivity,
    outlives_parent = "on_enter runs as a task the activity does not wait for",
    handle = Manual
)]
#[derive(Attrs)]
pub struct OnEnter {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// Draining an agent's activity before it leaves.
#[span(name = "drain_agent_activity", parent = UpdateAgent | SessionClose, handle = Manual)]
#[derive(Attrs)]
pub struct DrainAgentActivity {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// The agent's `on_exit` hook.
#[span(name = "on_exit", parent = DrainAgentActivity, handle = Manual)]
#[derive(Attrs)]
pub struct OnExit {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// Pausing the agent's activity during a handoff.
#[span(name = "pause_agent_activity", parent = UpdateAgent, handle = Manual)]
#[derive(Attrs)]
pub struct PauseAgentActivity {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// Resuming the agent's activity during a handoff.
#[span(name = "resume_agent_activity", parent = UpdateAgent, handle = Manual)]
#[derive(Attrs)]
pub struct ResumeAgentActivity {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// A synchronous stall of the event loop, under the blocked task's span. Error status at the
/// `error` severity; at most 30 per minute.
#[span(name = "event_loop_blocked", parent = Current)]
#[derive(Attrs)]
pub struct EventLoopBlocked {
    #[attr("lk.blocking.duration")]
    pub duration: Seconds,
    /// The stall duration that triggers a report, in seconds.
    #[attr("lk.blocking.threshold")]
    pub threshold: Seconds,
    /// Seconds of the stall spent in garbage collection.
    #[attr("lk.blocking.gc_time")]
    pub gc_time: Option<Seconds>,
    /// CPU seconds the stall consumed.
    #[attr("lk.blocking.cpu_time")]
    pub cpu_time: Option<Seconds>,
    /// How severe the stall was.
    #[attr("lk.blocking.severity")]
    pub severity: BlockingSeverity,
    /// The blocked task.
    #[attr("lk.blocking.task")]
    pub task: Option<String>,
    /// The blocked task's stack.
    #[attr("lk.blocking.stack")]
    pub stack: Option<String>,
    /// The module being imported during the stall.
    #[attr("lk.blocking.import")]
    pub import: Option<String>,
    /// Stalls not reported since the last report, over the rate limit.
    #[attr("lk.blocking.suppressed")]
    pub suppressed: Option<u32>,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockingSeverity {
    Warning,
    Error,
}

/// An RPC method handled for a remote participant.
#[span(name = "rpc_handler", parent = AgentSession, kind = Server, end = RpcEnd)]
#[derive(Attrs)]
pub struct RpcHandler {
    #[attr(otel::RPC_METHOD)]
    pub method: String,
    /// The RPC request id.
    #[attr("lk.rpc.request_id")]
    pub request_id: String,
    /// Identity of the caller.
    #[attr("lk.rpc.caller_identity")]
    pub caller_identity: String,
    /// Seconds the caller waits for the response.
    #[attr("lk.rpc.response_timeout")]
    pub response_timeout: Seconds,
    /// A handler is registered for the method.
    #[attr("lk.rpc.handler_registered")]
    pub handler_registered: bool,
    /// Payload size, in bytes.
    #[attr("lk.rpc.payload_size")]
    pub payload_size: u64,
    /// The payload's first 1024 characters.
    #[attr("lk.pii.rpc.payload")]
    pub payload: Content<String>,
}

/// An RPC call to a remote participant, from wherever it is made.
#[span(name = "rpc_call", parent = Current, kind = Client, end = RpcEnd)]
#[derive(Attrs)]
pub struct RpcCall {
    #[attr(otel::RPC_METHOD)]
    pub method: String,
    /// Identity of the participant called.
    #[attr("lk.rpc.destination_identity")]
    pub destination_identity: String,
    #[attr("lk.rpc.payload_size")]
    pub payload_size: u64,
    #[attr("lk.pii.rpc.payload")]
    pub payload: Content<String>,
    #[attr("lk.rpc.response_timeout")]
    pub response_timeout: Seconds,
}

#[derive(Attrs)]
pub struct RpcEnd {
    /// The RPC error code, on error.
    #[attr("lk.rpc.error_code")]
    pub error_code: Option<i64>,
    /// Response size, in bytes, on success.
    #[attr("lk.rpc.response_size")]
    pub response_size: Option<u64>,
    /// The response's first 1024 characters, on success.
    #[attr("lk.pii.rpc.response")]
    pub response: Option<Content<String>>,
}

/// Answering-machine detection, from the detector's attach to its verdict.
#[span(name = "amd", parent = AgentSession, end = Option<AmdVerdict>, handle = Manual)]
#[derive(Attrs)]
pub struct Amd;

#[derive(Attrs)]
pub struct AmdVerdict {
    /// AMD classification.
    #[attr("lk.amd.category")]
    pub category: AmdCategory,
    /// Why the detector decided.
    #[attr("lk.amd.reason")]
    pub reason: String,
    /// Seconds of speech the detector heard.
    #[attr("lk.amd.speech_duration")]
    pub speech_duration: Seconds,
    /// Seconds from the call start to the verdict.
    #[attr("lk.amd.delay")]
    pub delay: Seconds,
    /// The transcript the verdict was made on.
    #[attr("lk.pii.amd.transcript")]
    pub transcript: Content<String>,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdCategory {
    Human,
    #[wire = "machine-ivr"]
    MachineIvr,
    #[wire = "machine-vm"]
    MachineVm,
    #[wire = "machine-unavailable"]
    MachineUnavailable,
    Uncertain,
}

impl From<amd::AmdCategory> for AmdCategory {
    fn from(category: amd::AmdCategory) -> Self {
        match category {
            amd::AmdCategory::Human => Self::Human,
            amd::AmdCategory::MachineIvr => Self::MachineIvr,
            amd::AmdCategory::MachineVm => Self::MachineVm,
            amd::AmdCategory::MachineUnavailable => Self::MachineUnavailable,
            amd::AmdCategory::Uncertain => Self::Uncertain,
        }
    }
}

/// Keyterm detection over the conversation, under the replying turn or the session.
#[span(
    name = "keyterm_detection",
    parent = AgentTurn | AgentSession,
    end = KeytermDetectionEnd,
    outlives_parent = "the pass runs alongside the reply and can outlast a short or interrupted turn",
    handle = Manual
)]
#[derive(Attrs)]
pub struct KeytermDetection {
    #[attr(otel::GEN_AI_REQUEST_MODEL)]
    pub model: String,
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: String,
}

#[derive(Attrs)]
pub struct KeytermDetectionEnd {
    /// Keyterms after the pass.
    #[attr("lk.keyterms.count")]
    pub count: u32,
    /// Keyterms the pass added.
    #[attr("lk.keyterms.added")]
    pub added: u32,
    /// Keyterms the pass removed.
    #[attr("lk.keyterms.removed")]
    pub removed: u32,
}
