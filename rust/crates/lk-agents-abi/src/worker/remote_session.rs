use crate::ChatItem;
use crate::voice::amd::AmdCategory;
use crate::voice::events::{
    AgentFalseInterruption, AgentStateChanged, EotPrediction, ItemAddedWire, ToolCallEnded,
    ToolCallUpdated, ToolReplyUpdated, UserInputTranscribed, UserStateChanged,
};

#[crate::host(file(module = "remote_session_wire"))]
const _: () = ();

/// The option keys a [`SessionState`] row's `options` carries, in the reference's order. Only the
/// key set is pinned: values are the host's `str()` of each option.
#[crate::host(constant)]
pub const STATE_OPTION_FIELDS: &[&str] = &[
    "endpointing",
    "interruption",
    "max_tool_steps",
    "user_away_timeout",
    "transcription_timeout",
    "preemptive_generation",
    "min_consecutive_speech_delay",
    "use_tts_aligned_transcript",
    "ivr_detection",
];

/// `id` and `created_at` are left unset, as in the reference (`remote_session.py:533`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallRef {
    pub name: String,
    pub arguments: String,
    pub call_id: String,
}

/// `id` and `created_at` are dropped by the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallOutputRef {
    pub call_id: String,
    pub name: String,
    pub output: String,
    pub is_error: bool,
}

/// `error` (tag 15). The whole error becomes one string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorWire {
    pub message: String,
}

/// `overlapping_speech` (tag 16).
#[derive(Debug, Clone, PartialEq)]
pub struct OverlappingSpeechWire {
    pub is_interruption: bool,
    /// Absent when the overlap has no recorded start, which the proto allows.
    pub overlap_started_at: Option<f64>,
    pub detection_delay: f64,
    pub detected_at: f64,
}

/// `amd_prediction` (tag 18). Back-channel: never passes through `session.on(..)`.
#[derive(Debug, Clone, PartialEq)]
pub struct AmdPredictionWire {
    pub speech_duration: f64,
    /// The detector's verdict.
    pub category: AmdCategory,
    pub reason: String,
    pub transcript: String,
    pub delay: f64,
}

/// `function_tools_executed` (tag 14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionToolsExecutedWire {
    pub function_calls: Vec<FunctionCallRef>,
    pub function_call_outputs: Vec<FunctionCallOutputRef>,
}

/// `function_tools_started` (tag 20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionToolsStartedWire {
    pub function_calls: Vec<FunctionCallRef>,
}

/// `debug_message` (tag 21): already a proto on both sides, so the serialized bytes cross.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugMessageWire {
    /// A serialized `livekit.agent.DebugMessage`, decoded by `remote_session::decode_debug_message`.
    pub payload: Option<Vec<u8>>,
}

// Tool lifecycle payloads travel directly in Rust from the voice driver to `SessionHost`.

/// `tool_call_started`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallStartedWire {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// `session_usage_updated` (tag 17).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionUsageUpdatedWire {
    /// The session's latest usage snapshot.
    pub usage: crate::usage::Usage,
}

/// `tool_execution_updated` (tag 22).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecutionUpdatedWire {
    pub update: ToolUpdateWire,
}

/// One tool lifecycle update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolUpdateWire {
    Started(ToolCallStartedWire),
    CallUpdated(ToolCallUpdated),
    Ended(ToolCallEnded),
    ReplyUpdated(ToolReplyUpdated),
}

/// An event the voice driver forwards to `SessionHost` for protocol encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum WireEvent {
    AgentStateChanged(AgentStateChanged),
    UserStateChanged(UserStateChanged),
    UserInputTranscribed(UserInputTranscribed),
    Error(ErrorWire),
    OverlappingSpeech(OverlappingSpeechWire),
    EotPrediction(EotPrediction),
    AmdPrediction(AmdPredictionWire),
    FunctionToolsExecuted(FunctionToolsExecutedWire),
    FunctionToolsStarted(FunctionToolsStartedWire),
    DebugMessage(DebugMessageWire),
    ConversationItemAdded(Box<ItemAddedWire>),
    SessionUsageUpdated(SessionUsageUpdatedWire),
    ToolExecutionUpdated(ToolExecutionUpdatedWire),
    AgentFalseInterruption(AgentFalseInterruption),
}

/// What `SessionHost` asks of the session it hosts: the port the reference's reach-ins into
/// `AgentSession` become (`voice/remote_session.py`'s `RemoteSessionAccess`). An `Err` is a host
/// failure (answered `"internal error"`), except input and simulation errors, which become the
/// response's `error` verbatim. Its `session` is the `VoiceSession`, read on the session's loop.
#[crate::host(proxy(host = "livekit.agents.voice._session_bindings.RemoteSessionAccess"))]
pub trait SessionPort: Send + Sync {
    /// `session.history.items`.
    #[host(await, getter = "session.history.items")]
    fn chat_history(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Vec<ChatItem>, crate::errors::HostError>>;

    /// `session.current_agent`, flattened; instructions rendered with `modality="audio"`.
    #[host(await, name = "_core_agent_info")]
    fn agent_info(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<AgentInfo, crate::errors::HostError>>;

    /// `interrupt(force=True)`, then `await session.run(user_input=text)` and its items. An empty
    /// list is a silent turn, not a failure.
    #[host(await, name = "_core_run_input")]
    fn run_input<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        text: &'a str,
    ) -> crate::BoxFuture<'a, Result<Vec<ChatItem>, crate::errors::HostError>>;

    #[host(await, name = "_core_session_state")]
    fn session_state(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<RemoteSessionState, crate::errors::HostError>>;

    /// `session._room_io.room.get_rtc_stats()`; `None` without room IO.
    #[host(await, name = "_core_rtc_stats")]
    fn rtc_stats(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<RtcStats>, crate::errors::HostError>>;

    /// `session.usage`.
    #[host(await, getter = "session.usage")]
    fn session_usage(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<crate::usage::Usage, crate::errors::HostError>>;

    /// `FrameworkInfo(sdk="python", sdk_version=__version__)` on the reference. Read once.
    #[host(read_once, name = "_core_framework_info")]
    fn framework_info(&self) -> FrameworkInfo;

    /// Only the `Some` channels are applied.
    #[host(await, name = "_core_update_io")]
    fn update_io(
        &self,
        cx: &crate::telemetry::Cx,
        update: IoUpdate,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;

    /// Hand the agent the simulator's provisional verdict; return the agent's own
    /// (`on_simulation_end`). `None`: no simulation or no verdict.
    #[host(await, name = "_core_finalize_simulation")]
    fn finalize_simulation(
        &self,
        cx: &crate::telemetry::Cx,
        provisional: crate::worker::SimulationVerdict,
    ) -> crate::BoxFuture<
        '_,
        Result<Option<crate::worker::SimulationVerdict>, crate::errors::HostError>,
    >;
}

/// The current agent: `None` instructions differ from `""`; tools are flattened across
/// `Toolset`s, names only.
#[crate::host(record(module = "remote_session", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentInfo {
    pub id: String,
    pub instructions: Option<String>,
    pub tools: Vec<String>,
    pub chat_ctx: Vec<ChatItem>,
}

/// The session's state as the host holds it: an unknown state word answers `AS_IDLE` /
/// `US_LISTENING`; `options` keys are `STATE_OPTION_FIELDS`; `started_at` is epoch seconds,
/// `None` or `0` stamped with the answer time.
#[crate::host(record(module = "remote_session", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteSessionState {
    pub agent_state: String,
    pub user_state: String,
    pub agent_id: String,
    pub options: std::collections::BTreeMap<String, String>,
    pub started_at: Option<f64>,
}

/// Each stat is `MessageToDict` of one SDK stat.
#[crate::host(record(module = "remote_session", extract))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RtcStats {
    pub publisher_stats: Vec<serde_json::Value>,
    pub subscriber_stats: Vec<serde_json::Value>,
}

/// The SDK the session runs on and its version.
#[crate::host(record(module = "remote_session", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct FrameworkInfo {
    pub sdk: String,
    pub sdk_version: String,
}

/// The channels to switch; `None` leaves a channel alone.
#[crate::host(record(module = "remote_session", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IoUpdate {
    pub input_audio: Option<bool>,
    pub input_video: Option<bool>,
    pub output_audio: Option<bool>,
    pub output_video: Option<bool>,
    pub output_transcription: Option<bool>,
}

/// The runtime-owned remote host attached to one live voice session.
pub trait SessionHosting: Send + Sync {
    fn forward(&self, event: &crate::voice::events::HostWire);
    /// The session's usage moved: `session_usage_updated` for the remote client only.
    fn forward_usage(&self, usage: &crate::usage::Usage);
    fn forward_error(&self, message: String);
    fn forward_tools_executed(&self, event: &crate::voice::events::ToolsExecutedEvent);
    /// Start once; a host already closing cannot reopen its transport.
    fn start(&self) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
    /// Drain and close once. Repeated or cancelled callers join the same cleanup.
    fn aclose(&self) -> crate::BoxFuture<'_, ()>;
}

/// The name a remote session event arrives under: its `AgentSessionEvent.event` oneof arm.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteSessionEventKind {
    AgentStateChanged,
    UserStateChanged,
    ConversationItemAdded,
    UserInputTranscribed,
    FunctionToolsExecuted,
    Error,
    OverlappingSpeech,
    SessionUsageUpdated,
    AmdPrediction,
    EotPrediction,
    FunctionToolsStarted,
    DebugMessage,
    ToolExecutionUpdated,
    AgentFalseInterruption,
}

/// A remote session request's `SessionRequest.request` oneof arm, used in logs and client errors.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteSessionRequestKind {
    Ping,
    GetChatHistory,
    GetAgentInfo,
    RunInput,
    GetSessionState,
    GetRtcStats,
    GetSessionUsage,
    GetFrameworkInfo,
    UpdateIo,
    FinalizeSimulation,
}
