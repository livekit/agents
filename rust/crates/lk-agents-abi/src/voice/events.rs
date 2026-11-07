//! `voice/events.py`'s generated events, and the session-event wire the host routes by record
//! class (`_generated/session_dispatch.py`). The public events are dataclasses
//! (`_generated/events.py`); the wire-only records are `_generated/session_wire.py`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::voice::amd::AmdPrediction;
use crate::voice::room_io::{
    DeleteRoomRequested, ParticipantLinked, ParticipantUnlinked, TextInput,
};

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "events", path = "livekit.agents.voice.events"))]
const _: () = ();

// --- the two state vocabularies ------------------------------------------------------------------

/// Whether the user is speaking, listening, or has gone away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[crate::host(record(module = "events", emit, extract))]
#[crate::host(wire_strs(msg = format!("unknown user state {s:?}")))]
#[serde(rename_all = "snake_case")]
pub enum UserState {
    #[host(rename = "speaking")]
    Speaking,
    #[host(rename = "listening")]
    Listening,
    #[host(rename = "away")]
    Away,
}

/// The agent's lifecycle label.
#[crate::host(record(module = "events", emit, extract))]
#[crate::host(wire_strs(msg = format!("unknown agent state {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    #[host(rename = "initializing")]
    Initializing,
    #[host(rename = "idle")]
    Idle,
    #[host(rename = "listening")]
    Listening,
    #[host(rename = "thinking")]
    Thinking,
    #[host(rename = "speaking")]
    Speaking,
}

/// A leg of the session's input: an audio channel a provider hears, a detector the session asks,
/// or the room input a reader pulls.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Leg {
    Stt,
    Vad,
    Realtime,
    Eot,
    Interruption,
    Amd,
    Keyterms,
    AudioIn,
    VideoIn,
}

/// The host code whose failure a session reports: a hook it asked for, or the host's own setup
/// and close steps.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookName {
    OnEnter,
    OnExit,
    OnUserTurnCompleted,
    OnUserTurnExceeded,
    FillerSource,
    FunctionToolsExecuted,
    /// Resolving the agent's tools before it starts.
    SetupTools,
    /// The host's close of the session.
    Closing,
}

// --- the session events --------------------------------------------------------------------------

#[crate::host(record(
    module = "events",
    name = "UserStateChangedEvent",
    emit,
    constant(type = "user_state_changed")
))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserStateChanged {
    pub old_state: UserState,
    pub new_state: UserState,
    // Not skipped: the driver back-dates this to the speech boundary it reports, and the binding
    // converts that instant through the session's wall anchor.
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "events",
    name = "AgentStateChangedEvent",
    emit,
    constant(type = "agent_state_changed")
))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentStateChanged {
    pub old_state: AgentState,
    pub new_state: AgentState,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "events",
    name = "UserInputTranscribedEvent",
    emit,
    constant(type = "user_input_transcribed")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct UserInputTranscribed {
    pub transcript: String,
    pub is_final: bool,
    /// Provider-specific ID for the transcribed input item, when available.
    #[host(default = None)]
    pub item_id: Option<String>,
    #[host(default = None)]
    pub speaker_id: Option<String>,
    #[host(default = None)]
    pub language: Option<String>,
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "events",
    name = "UserTranscriptionTimeoutEvent",
    emit,
    constant(type = "user_transcription_timeout")
))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserTranscriptionTimeout {
    /// Total VAD-detected speech (s) in the turn that produced no transcript.
    pub speech_duration: f64,
    /// When VAD first detected speech for this (untranscribed) turn.
    pub vad_speech_started_at: f64,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "events",
    name = "EotPredictionEvent",
    emit,
    constant(type = "eot_prediction")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct EotPrediction {
    pub probability: f64,
    pub threshold: f64,
    /// Server-side model inference time.
    pub inference_duration: f64,
    /// End of user speech → prediction received latency (s), anchored on the
    /// VAD-backdated last_speaking_time.
    pub delay: f64,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

/// Internal: a window in which the agent could backchannel (a short
/// acknowledgment such as "mm-hmm"), as predicted by the turn detector. Passed to
/// ``AgentActivity`` only — not surfaced as a public ``AgentSession`` event yet.
///
/// ``AgentActivity`` owns the decision of what to do with it. The end-of-turn margin
/// (``end_of_turn_threshold - end_of_turn_probability``) gives a progressive risk axis:
/// a large positive margin means the user is clearly still going, so riskier
/// backchannels (yeah/okay/right) are safe; a small margin (or a negative one, where
/// ``end_of_turn_probability >= end_of_turn_threshold`` and a reply is imminent) calls
/// for safe, less ambiguous ones (hmm/uh-huh) that won't collide with the reply.
#[crate::host(record(
    module = "events",
    name = "_AgentBackchannelOpportunityEvent",
    emit,
    constant(type = "agent_backchannel_opportunity")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentBackchannelOpportunityEvent {
    pub probability: f64,
    pub threshold: f64,
    pub end_of_turn_probability: f64,
    pub end_of_turn_threshold: f64,
    #[host(default = None)]
    pub language: Option<String>,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "events",
    name = "AgentFalseInterruptionEvent",
    emit,
    constant(type = "agent_false_interruption")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentFalseInterruption {
    /// Whether the false interruption was resumed automatically.
    pub resumed: bool,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

/// How a speech handle was created: only `say()` is `say`; every reply is `generate_reply`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechSource {
    Say,
    GenerateReply,
}

/// `speech_created`'s payload: the host wraps the `_core.SpeechHandle` in its `SpeechHandle` for
/// the public `SpeechCreatedEvent` (`voice/events.py`).
#[crate::host(record(module = "session_wire", name = "SpeechCreatedWire", emit))]
#[derive(Clone)]
pub struct SpeechCreated {
    pub user_initiated: bool,
    pub source: SpeechSource,
    pub speech_handle: std::sync::Arc<dyn crate::voice::speech_handle::Speech>,
}

// --- the tool lifecycle ---------------------------------------------------------------------------
//
// `ToolExecutionUpdatedEvent.update`'s members. A started call crosses as `ToolCallStartedWire`;
// the host builds the public `ToolCallStarted` (`voice/events.py`) with its `llm.FunctionCall`.

/// A function tool call was dispatched: the call's fields, which the host validates into its
/// `llm.FunctionCall`.
#[crate::host(record(module = "session_wire", name = "ToolCallStartedWire", emit))]
#[derive(Clone)]
pub struct ToolCallStarted {
    pub function_call: crate::FunctionCall,
}

/// A progress update emitted via ``ctx.update()`` while a tool call runs.
#[crate::host(record(
    module = "events",
    name = "ToolCallUpdated",
    emit,
    constant(type = "tool_call_updated")
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallUpdated {
    /// Entry id: ``call_id`` inline, ``{call_id}_update_N`` when deferred.
    pub id: String,
    pub call_id: String,
    pub message: String,
}

/// A tool call's single terminal entry.
#[crate::host(record(
    module = "events",
    name = "ToolCallEnded",
    emit,
    constant(type = "tool_call_ended")
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallEnded {
    /// Entry id: ``call_id`` inline, ``{call_id}_final`` when deferred.
    pub id: String,
    pub call_id: String,
    /// Result or error text; None when there is nothing to voice.
    #[host(default = None)]
    pub message: Option<String>,
    /// `"done"`, `"error"` or `"cancelled"`.
    pub status: ToolEndStatus,
}

/// How a tool call ended: [`ToolCallEnded::status`].
#[crate::host(record(module = "events", emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEndStatus {
    Done,
    Error,
    Cancelled,
}

/// Where the deferred tool reply is: [`ToolReplyUpdated::status`].
#[crate::host(record(module = "events", emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolReplyStatus {
    Scheduled,
    Completed,
    Interrupted,
    /// The model judged the results already covered and said nothing.
    Skipped,
}

/// Lifecycle of the deferred reply that voices buffered tool updates: ``scheduled``
/// when queued, then ``completed`` / ``interrupted`` / ``skipped``. One reply may cover
/// several calls; an inline first update never gets one.
#[crate::host(record(
    module = "events",
    name = "ToolReplyUpdated",
    emit,
    constant(type = "tool_reply_updated")
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolReplyUpdated {
    /// ``ToolCallUpdated.id`` values this reply covers.
    pub update_ids: Vec<String>,
    /// `"scheduled"`, `"completed"`, `"interrupted"` or `"skipped"`.
    pub status: ToolReplyStatus,
    /// Id of the reply speech; ``speech_created`` carries its handle.
    pub speech_id: String,
}

/// The four tool-lifecycle updates, each crossing as its own record.
#[crate::host(record(module = "session_wire", name = "ToolUpdateWire", emit, transparent))]
#[derive(Clone)]
pub enum ToolUpdate {
    Started(ToolCallStarted),
    Updated(ToolCallUpdated),
    Ended(ToolCallEnded),
    Reply(ToolReplyUpdated),
}

/// `tool_execution_updated`'s payload; the host emits `ToolExecutionUpdatedEvent` from it.
#[crate::host(record(module = "session_wire", name = "ToolExecutionUpdatedWire", emit))]
pub struct ToolExecutionUpdated {
    pub update: ToolUpdate,
}

// --- limits, errors and the close ------------------------------------------------------------------

// The user turn passed a configured word or duration limit. No `///`: it would become a docstring.
#[crate::host(record(
    module = "events",
    name = "UserTurnExceededEvent",
    emit,
    constant(type = "user_turn_exceeded")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct UserTurnExceededEvent {
    /// Transcript from the current (uncommitted) user turn only.
    /// Previous turns in the accumulation window are already in the chat context.
    pub transcript: String,
    /// Full transcript since the start of user speaking.
    // Everything since the agent last spoke (`_UserTurnTracker`), a longer span than this turn.
    pub accumulated_transcript: String,
    /// Total word count since the start of user speaking.
    pub accumulated_word_count: usize,
    /// Duration of the user turn in seconds.
    pub duration: std::time::Duration,
    // The host stamps this; the core has no wall clock.
    #[host(skip, default_factory = "time.time")]
    pub created_at: f64,
}

// `//`, not `///`: the reference has no docstring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[crate::host(record(module = "events", emit, extract, enum))]
#[crate::host(wire_strs(msg = format!("unknown close reason {s:?}")))]
pub enum CloseReason {
    #[host(rename = "error")]
    Error,
    #[host(rename = "job_shutdown")]
    JobShutdown,
    #[host(rename = "participant_disconnected")]
    ParticipantDisconnected,
    #[host(rename = "user_initiated")]
    UserInitiated,
    #[host(rename = "task_completed")]
    TaskCompleted,
}

// --- the session-event wire payloads --------------------------------------------------------------
//
// What `VoiceSession._core_event` routes by class; no public model.

/// The user started speaking (`on_start_of_speech`): the record is the whole message.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartOfSpeechEvent {
    /// `speech_start_time`: back-dated by VAD, or the provider's onset stamp.
    #[host(skip)]
    pub at: crate::Ts,
    #[host(skip)]
    pub speech_duration: std::time::Duration,
    /// `false` for the STT-driven turn detection path, where the reference passes `ev=None`.
    #[host(skip)]
    pub from_vad: bool,
}

/// A transcript stable enough to start generating against (`_PreemptiveGenerationInfo`): the record
/// is the whole message.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct PreemptiveGenerationEvent {
    #[host(skip)]
    pub new_transcript: String,
    #[host(skip)]
    pub transcript_confidence: f64,
    #[host(skip)]
    pub started_speaking_at: Option<crate::Ts>,
}

/// A backchannel was confirmed: the record is the whole message.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackchannelConfirmedWire {}

/// The overlap was a real interruption (`on_interruption`): the record is the whole message.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapInterruptionEvent {
    /// `ev.overlap_started_at or ev.detected_at`, resolved (a `0.0` falls through).
    #[host(skip)]
    pub ignore_user_transcript_until: crate::Ts,
}

/// `user_turn_completed`'s payload.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct UserTurnCompletedWire {
    /// The turn's id: what `hook_done(turn, ..)` answers for.
    pub turn: u64,
    pub new_transcript: String,
    pub transcript_confidence: f64,
}

/// The user stopped speaking (`on_end_of_speech`).
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndOfSpeechEvent {
    #[host(skip)]
    pub at: crate::Ts,
    #[host(skip)]
    pub from_vad: bool,
    /// Silence already elapsed when the event was raised, in seconds (`0.0` on the STT path). `at`
    /// includes it, but a host needs it: AMD's post-speech window runs from when speech ended.
    pub silence_duration: f64,
}

/// Which of the host's projections take an item: carried on `ItemAdded` so the host's
/// projections decide nothing.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Vocabulary)]
pub enum Visibility {
    /// The agent's context only: a host-supplied item, not in `session.history`.
    Model,
    /// `session.history` only: the handoff record.
    History,
    /// The agent's context and `session.history`: everything said, and the config records.
    Both,
}

impl Visibility {
    pub fn model(self) -> bool {
        matches!(self, Visibility::Model | Visibility::Both)
    }

    pub fn history(self) -> bool {
        matches!(self, Visibility::History | Visibility::Both)
    }
}

/// `conversation_item_added`'s payload.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ItemAddedWire {
    /// The item; the host reads `host_item`.
    #[host(skip)]
    pub item: crate::ChatItem,
    /// The item as the session's history holds it: the host's `llm.ChatItem`, the same object
    /// `session.history` returns. `None` only for a host that made no object for it.
    #[host(annotation = "livekit.agents.llm.ChatItem | None")]
    pub host_item: Option<crate::HostRef>,
    /// Which agent's frame produced it, as the opaque id the host gave the core.
    pub agent: String,
    pub visibility: Visibility,
}

/// `function_tools_executed`, as the driver emits it; calls and outputs are parallel lists. The
/// host ignores it: its `FunctionToolsExecutedEvent` comes from the `function_tools_executed`
/// hook, and the remote session forwards these lists.
#[crate::host(record(module = "session_wire", name = "FunctionToolsExecutedWire", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolsExecutedEvent {
    #[host(skip)]
    pub function_calls: Vec<crate::FunctionCall>,
    #[host(skip)]
    pub function_call_outputs: Vec<crate::FunctionCallOutput>,
    /// `_handoff_required`; false when two tools returned an agent.
    #[host(skip)]
    pub handoff_required: bool,
}

impl ToolsExecutedEvent {
    /// `FunctionToolsExecutedEvent.has_tool_reply`.
    pub fn has_tool_reply(&self) -> bool {
        self.function_call_outputs.iter().any(|o| o.reply_required)
    }
}

/// A diagnostic's level: the name the host's `logging` module takes.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Info,
    Warning,
    Error,
}

/// A sentence the core decided and the host logs. `fields` is the reference's `extra={...}`, so
/// `lk.pii.*` entries go through the host's redaction filter.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct DiagnosticWire {
    pub level: DiagnosticLevel,
    pub message: String,
    pub fields: BTreeMap<String, String>,
}

// The run: host-only events; the run settles as a text run's last update does.

/// `run_started`'s payload.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct RunStartedWire {
    /// The host's object for the run (the future its `_run()` awaits); `None` when the host
    /// passed none.
    pub run: Option<crate::HostRef>,
}

/// Why a run opened nothing (`run_refused`).
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Vocabulary)]
pub enum RunRefusal {
    /// A run is open: `RuntimeError("nested runs are not supported")`.
    Nested,
    Closing,
}

/// `run_refused`'s payload.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct RunRefusedWire {
    /// The host's object for the refused run (the future its `_run()` awaits).
    pub run: Option<crate::HostRef>,
    pub cause: RunRefusal,
}

/// `run_finished`'s payload: the run settled with a result, as a text run's `Finished` update.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct RunFinishedWire {
    /// The host's object for the run (the future its `_run()` awaits).
    pub run: Option<crate::HostRef>,
    pub settled: crate::agent::run::RunSettled,
}

/// `run_failed`'s payload: the run settled without a result, as a text run's `Failed` update.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct RunFailedWire {
    /// The host's object for the run (the future its `_run()` awaits).
    pub run: Option<crate::HostRef>,
    pub error: crate::agent::run::RunError,
    #[host(default_factory = "list")]
    pub input_requirements: Vec<crate::agent::run::RunInputRequirement>,
    /// The provider failure the run's last speech ended on, raised as itself.
    #[host(default = None)]
    pub provider_error: Option<crate::errors::ProviderError>,
}

/// How a speech ended.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Vocabulary)]
pub enum SpeechOutcome {
    Completed,
    /// Cut short: an interruption, `interrupt()`, or the interruption watchdog.
    Interrupted,
    /// A provider failed it; the handle's `exception()` carries the error.
    Failed,
}

/// `speech_done`'s payload.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct SpeechDoneWire {
    /// `speech_{n}`, the handle's id.
    pub speech_id: String,
    pub outcome: SpeechOutcome,
}

/// `floor_changed`'s payload. `speech_id` is `None` when the floor is free.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct FloorChangedWire {
    pub speech_id: Option<String>,
}

/// `context_carried`'s payload: item ids moved from `from_agent`'s context to `to_agent`'s.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct ContextCarriedWire {
    pub from_agent: String,
    pub to_agent: String,
    pub items: Vec<String>,
}

/// `close`'s wire payload: `error` is the unrecoverable `error` event that closed the session, the
/// key absent when none did.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct CloseWire {
    pub reason: CloseReason,
    pub error: Option<SessionErrorWire>,
}

/// `metrics_collected`'s wire payload: the record plus the driver's `speech_id` correlation.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct MetricsCollectedWire {
    /// The `AgentMetrics` record as the dict `TypeAdapter(AgentMetrics)` validates.
    pub metrics: crate::metrics::base::AgentMetrics,
    /// `speech_{n}` of the speech the record belongs to; `None` for a leg that has none.
    pub speech_id: Option<String>,
    /// The session leg whose provider emits the record as its own `metrics_collected`: `"vad"`
    /// for the VAD's, which the session folds; `None` for one a provider metered itself.
    pub leg: Option<MetricsLeg>,
}

/// [`MetricsCollectedWire::leg`]: the session leg whose provider a record belongs to.
#[crate::host(record(module = "session_wire", emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsLeg {
    /// The session's VAD.
    Vad,
}

/// `overlapping_speech`'s wire payload: the interruption detector's verdict, as it reported it.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct OverlappingSpeechWire {
    pub event: OverlappingSpeechEvent,
}

/// The leg an `ErrorEvent` names: `LLMError.type` and its siblings, which are public API. An
/// interruption detector failure degrades to VAD silently and the EOT model's `error` is never
/// subscribed.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, crate::Vocabulary)]
pub enum ErrorSource {
    #[host(rename = "stt_error")]
    Stt,
    #[host(rename = "llm_error")]
    Llm,
    #[host(rename = "tts_error")]
    Tts,
    #[host(rename = "realtime_model_error")]
    Realtime,
    /// An input audio processor; always recoverable.
    #[host(rename = "audio_processing_error")]
    AudioProcessing,
}

/// Where a session error was captured.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorOrigin {
    /// Off a plugin object: the host's own exception.
    Host,
    Engine,
}

/// `error`'s wire payload: the ingredients of `ErrorEvent`; the host wraps `error` in the leg's
/// class chosen by `source`.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone)]
pub struct SessionErrorWire {
    pub source: ErrorSource,
    pub recoverable: bool,
    pub label: ErrorOrigin,
    /// The host's own exception when the provider raised one, else an `APIError`.
    pub error: crate::errors::ProviderError,
    pub speech_id: Option<String>,
    /// An `audio_processing_error`'s processor (`ErrorEvent.source`) and the exception it raised.
    pub processor_failure: Option<crate::voice::rtc::ProcessorFailure>,
}

// --- the interruption detector's verdict and its wire ----------------------------------------

/// `OverlappingSpeechEvent` (`inference/interruption.py:120`), the detector's only public verdict.
#[crate::host(record(
    module = "overlap",
    name = "OverlappingSpeechEvent",
    path = "livekit.agents.inference",
    pydantic,
    arbitrary_types,
    serializer = "overlapping_speech_serialize",
    constant(type = "overlapping_speech"),
    boundary,
    emit
))]
#[derive(Debug, Clone, PartialEq)]
pub struct OverlappingSpeechEvent {
    /// Wall clock seconds at construction.
    #[host(default_factory = "time.time")]
    pub created_at: f64,
    /// Wall clock seconds at which the overlap was resolved.
    #[host(default_factory = "time.time")]
    pub detected_at: f64,
    /// The verdict. `false` is only a *backchannel* verdict when `agent_ended` is also false.
    #[host(default = false)]
    pub is_interruption: bool,
    /// The overlap ended because the agent stopped talking; `is_interruption == false` is then
    /// inconclusive.
    #[host(default = false)]
    pub agent_ended: bool,
    #[host(default = 0.0)]
    pub total_duration: f64,
    #[host(default = 0.0)]
    pub prediction_duration: f64,
    #[host(default = 0.0)]
    pub detection_delay: f64,
    /// Wall clock seconds at which the overlap began.
    #[host(default = None)]
    pub overlap_started_at: Option<f64>,
    #[host(emit_with = "py_values::numpy_i16", omit_none, default = None, annotation = "numpy.typing.NDArray[numpy.int16] | None")]
    pub speech_input: Option<Vec<i16>>,
    #[host(read_as = Option<Vec<f64>>, emit_with = "py_values::numpy_f32", omit_none, default = None, annotation = "numpy.typing.NDArray[numpy.float32] | None")]
    pub probabilities: Option<Vec<f32>>,
    #[host(default = 0.0)]
    pub probability: f64,
    /// Requests sent for this overlap. Reset to zero as it is read.
    #[host(default = 0)]
    pub num_requests: u32,
}

impl OverlappingSpeechEvent {
    pub const TYPE: &'static str = "overlapping_speech";
}

#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryReloadedWire {
    /// The host reads the reloaded history from the event's state.
    #[host(skip)]
    pub items: Vec<crate::ChatItem>,
}

// --- the host event stream, as one union ------------------------------------------------------------

/// Every event the session hands its host, as the record the host reads, routed by its class.
///
/// `#[host(event)]` members are emitted as public session events and `#[host(ignored)]` ones are
/// dropped, both via `_generated/session_dispatch.py`; the rest have host behaviour.
/// `#[host(emits = "Model")]` names the public event that behaviour emits, so the generated
/// `EventTypes` and `AgentEvent` list every public session event.
#[crate::host(record(module = "session_wire", emit, transparent, dispatch = "session_dispatch"))]
pub enum HostWire {
    #[host(ignored)]
    HistoryReloaded(HistoryReloadedWire),
    #[host(event)]
    UserTranscript(UserInputTranscribed),
    StartOfSpeech(StartOfSpeechEvent),
    EndOfSpeech(EndOfSpeechEvent),
    #[host(emits = "SpeechCreatedEvent")]
    SpeechCreated(SpeechCreated),
    Diagnostic(DiagnosticWire),
    // The host emits its event from the `function_tools_executed` callback.
    #[host(ignored, emits = "FunctionToolsExecutedEvent")]
    FunctionToolsExecuted(ToolsExecutedEvent),
    #[host(ignored)]
    UserTurnCompleted(UserTurnCompletedWire),
    EotPrediction(EotPrediction),
    #[host(ignored)]
    AgentBackchannelOpportunity(AgentBackchannelOpportunityEvent),
    #[host(ignored)]
    PreemptiveGeneration(PreemptiveGenerationEvent),
    #[host(event)]
    UserTurnExceeded(UserTurnExceededEvent),
    #[host(event)]
    TranscriptionTimeout(UserTranscriptionTimeout),
    #[host(ignored)]
    BackchannelConfirmed(BackchannelConfirmedWire),
    #[host(ignored)]
    OverlapInterruption(OverlapInterruptionEvent),
    #[host(event)]
    AgentStateChanged(AgentStateChanged),
    #[host(event)]
    UserStateChanged(UserStateChanged),
    #[host(event)]
    AgentFalseInterruption(AgentFalseInterruption),
    AmdResult(AmdPrediction),
    #[host(emits = "ConversationItemAddedEvent")]
    ItemAdded(ItemAddedWire),
    ContextCarried(ContextCarriedWire),
    RunStarted(RunStartedWire),
    RunRefused(RunRefusedWire),
    RunFinished(RunFinishedWire),
    RunFailed(RunFailedWire),
    // A speech's handle learns its end natively (`SpeechHandle.wait_reported`).
    #[host(ignored)]
    SpeechDone(SpeechDoneWire),
    FloorChanged(FloorChangedWire),
    #[host(emits = "CloseEvent")]
    Close(CloseWire),
    MetricsCollected(MetricsCollectedWire),
    #[host(emits = "OverlappingSpeechEvent")]
    OverlappingSpeech(OverlappingSpeechWire),
    #[host(emits = "ErrorEvent")]
    Error(SessionErrorWire),
    #[host(emits = "ToolExecutionUpdatedEvent")]
    ToolExecutionUpdated(ToolExecutionUpdated),
    // The session acts on the link's edges itself (the away timer, the close).
    #[host(ignored)]
    ParticipantLinked(ParticipantLinked),
    #[host(ignored)]
    ParticipantUnlinked(ParticipantUnlinked),
    TextInput(TextInput),
    DeleteRoomRequested(DeleteRoomRequested),
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    dataclass,
    frozen,
    kw_only,
    slots,
    arbitrary_types,
    constant(type = "speech_created")
))]
pub struct SpeechCreatedEvent {
    /// True if the speech was created using public methods like `say` or `generate_reply`
    pub user_initiated: bool,
    /// Source indicating how the speech handle was created
    pub source: SpeechSource,
    /// The speech handle that was created
    #[host(
        hide_repr,
        annotation = "typing.Annotated[livekit.agents.voice.io.SpeechHandle, pydantic.Field(exclude=True)]"
    )]
    pub speech_handle: crate::HostRef,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

/// A function tool call was dispatched.
#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    dataclass,
    frozen,
    kw_only,
    slots,
    constant(type = "tool_call_started"),
    name = "ToolCallStarted"
))]
pub struct ToolCallStartedEvent {
    pub function_call: crate::FunctionCall,
}

/// The public tool updates retain validated calls and their typed lifecycle records.
#[crate::host(record(module = "session_events", name = "ToolUpdate", transparent))]
pub enum ToolLifecycleUpdate {
    Started(ToolCallStartedEvent),
    Updated(ToolCallUpdated),
    Ended(ToolCallEnded),
    Reply(ToolReplyUpdated),
}

/// One flat tool-lifecycle update. Discriminate on ``update.type``: ``tool_call_started``
///     → ``tool_call_updated`` → ``tool_call_ended`` → ``tool_reply_updated``.
///
#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    dataclass,
    frozen,
    kw_only,
    slots,
    constant(type = "tool_execution_updated")
))]
pub struct ToolExecutionUpdatedEvent {
    pub update: ToolLifecycleUpdate,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    dataclass,
    frozen,
    kw_only,
    slots,
    name = "_TypeDiscriminator",
    constant(type = "unknown")
))]
pub struct UnknownConversationItem {}

#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    pydantic,
    constant(type = "conversation_item_added")
))]
pub struct ConversationItemAddedEvent {
    #[host(
        annotation = ".chat_items.ChatMessage | livekit.agents.llm.AgentHandoff | _TypeDiscriminator"
    )]
    pub item: crate::HostRef,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    pydantic,
    constant(type = "close")
))]
pub struct CloseEvent {
    #[host(default = None, annotation = ".llm.LLMError | livekit.agents.stt.STTError | livekit.agents.tts.TTSError | livekit.agents.llm.RealtimeModelError | livekit.agents.inference.InterruptionDetectionError | None")]
    pub error: Option<crate::HostRef>,
    pub reason: CloseReason,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

// Public data contracts. Python behavior remains in its adapters.

/// Emitted after a batch of function tools finishes executing.
///
///     ``function_calls`` and ``function_call_outputs`` are parallel lists: the
///     output at a given index belongs to the call at the same index and carries
///     the same ``call_id``. Every call has one output, even one whose tool raised
///     ``StopResponse``; such an output asks for no reply with ``reply_required``.
///
#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    pydantic,
    constant(type = "function_tools_executed"),
    private(_handoff_required = "bool"),
    validator = "verify_lists_length"
))]
#[python(method(
    name = "zipped",
    ret = "list[tuple[.chat_items.FunctionCall, .chat_items.FunctionCallOutput]]",
    hook = "function_tools_zipped"
))]
#[python(method(
    name = "cancel_tool_reply",
    ret = "None",
    hook = "function_tools_cancel_tool_reply"
))]
#[python(method(
    name = "cancel_agent_handoff",
    ret = "None",
    hook = "function_tools_cancel_agent_handoff"
))]
#[python(property(name = "has_tool_reply", ret = "bool", hook = "function_tools_has_tool_reply"))]
#[python(property(name = "has_agent_handoff", ret = "bool", getter = "_handoff_required"))]
pub struct FunctionToolsExecutedEvent {
    pub function_calls: Vec<crate::FunctionCall>,
    pub function_call_outputs: Vec<crate::FunctionCallOutput>,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}

#[crate::host(record(
    module = "session_events",
    path = "livekit.agents.voice.events",
    pydantic,
    arbitrary_types,
    constant(type = "error"),
    serializer = "session_error_serialize"
))]
pub struct ErrorEvent {
    #[host(
        annotation = ".llm.LLMError | livekit.agents.stt.STTError | livekit.agents.tts.TTSError | livekit.agents.llm.RealtimeModelError | livekit.agents.inference.InterruptionDetectionError | livekit.agents.voice.room_io.AudioProcessingError | typing.Any"
    )]
    pub error: crate::HostRef,
    #[host(
        annotation = ".llm.LLM | livekit.agents.stt.STT | livekit.agents.tts.TTS | livekit.agents.llm.RealtimeModel | livekit.agents.inference.AdaptiveInterruptionDetector | typing.Any"
    )]
    pub source: crate::HostRef,
    #[host(default_factory = "time.time")]
    pub created_at: f64,
}
