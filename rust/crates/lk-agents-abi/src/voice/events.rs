//! `voice/events.py`. Field types we do not own are spelled with `#[python(ty = .., import = ..)]`:
//! the generator cannot see `ChatMessage` or `AgentMetrics` and must not invent a name for them.

use std::collections::HashMap;

use lk_agents_host_derive::Host;

use crate::voice::amd::AmdPrediction;
use crate::voice::room_io::{DeleteRoomRequested, ParticipantLinked, ParticipantUnlinked, TextInput};
use crate::voice::session_effects::Span;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "events", path = "livekit.agents.voice.events"))]
const _: () = ();

// --- the two state vocabularies ------------------------------------------------------------------

/// Whether the user is speaking, listening, or has gone away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Host)]
#[python(literal, name = "UserState")]
#[node(enum)]
#[host(wire(msg = format!("unknown user state {s:?}")))]
pub enum UserState {
    #[host(rename = "speaking")]
    Speaking,
    #[host(rename = "listening")]
    Listening,
    #[host(rename = "away")]
    Away,
}

/// The agent's lifecycle label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Host)]
#[python(literal, name = "AgentState")]
#[node(enum)]
#[host(wire(msg = format!("unknown agent state {s:?}")))]
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

// --- the session events --------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "UserStateChangedEvent",
    tag = "type",
    tag_value = "user_state_changed"
)]
#[node(interface)]
// Also read back from the remote session's `kind`-keyed dict (`remote_session.py:_send_wire`).
#[binding(emit, extract, from_items, read_tag = "kind")]
pub struct UserStateChanged {
    pub old_state: UserState,
    pub new_state: UserState,
    // Not skipped: the driver back-dates this to the speech boundary it reports, and the binding
    // converts that instant through the session's wall anchor.
    #[binding(extract_skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "AgentStateChangedEvent",
    tag = "type",
    tag_value = "agent_state_changed"
)]
#[node(interface)]
#[binding(emit, extract, from_items, read_tag = "kind")]
pub struct AgentStateChanged {
    pub old_state: AgentState,
    pub new_state: AgentState,
    // The host stamps this; the core has no wall clock.
    #[binding(skip, extract_skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "UserInputTranscribedEvent",
    tag = "type",
    tag_value = "user_input_transcribed"
)]
#[node(interface)]
// The binding's dict is a subset of the model: `_on_core_event` reads `transcript` and `is_final`
// and the `#[binding(skip)]` fields take the model's defaults. The remote session reads it back
// (`language` too, which its host never sends).
#[binding(emit, extract, from_items, read_tag = "kind")]
pub struct UserInputTranscribed {
    pub transcript: String,
    pub is_final: bool,
    /// Provider-specific ID for the transcribed input item, when available.
    #[binding(skip, extract_skip)]
    #[python(default = "None")]
    pub item_id: Option<String>,
    #[binding(skip, extract_skip)]
    #[python(default = "None")]
    pub speaker_id: Option<String>,
    #[binding(skip, default)]
    #[python(ty = "LanguageCode | None", import = "..language:LanguageCode", default = "None")]
    #[node(ty = "string | null")]
    pub language: Option<String>,
    #[binding(skip, extract_skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Host)]
#[python(
    pydantic,
    name = "UserTranscriptionTimeoutEvent",
    tag = "type",
    tag_value = "user_transcription_timeout"
)]
#[node(interface)]
// The public discriminant is `user_transcription_timeout`; the wire kind the binding sends is
// `transcription_timeout`. They differ on purpose.
#[binding(emit, tag = "type", tag_value = "transcription_timeout")]
pub struct UserTranscriptionTimeout {
    /// Total VAD-detected speech (s) in the turn that produced no transcript.
    pub speech_duration: f64,
    /// When VAD first detected speech for this (untranscribed) turn.
    pub vad_speech_started_at: f64,
    // The host stamps this; the core has no wall clock.
    #[binding(skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "EotPredictionEvent",
    tag = "type",
    tag_value = "eot_prediction"
)]
#[node(interface)]
// Read back by the remote session, whose proto narrows the two probabilities to `f32`.
#[binding(emit, extract, from_items, read_tag = "kind")]
pub struct EotPrediction {
    pub probability: f64,
    pub threshold: f64,
    /// Server-side model inference time.
    pub inference_duration: f64,
    /// End of user speech → prediction received latency (s), anchored on the
    /// VAD-backdated last_speaking_time.
    pub delay: f64,
    // The host stamps this; the core has no wall clock.
    #[binding(skip, extract_skip)]
    #[python(default_factory = "time.time")]
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
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "_AgentBackchannelOpportunityEvent",
    tag = "type",
    tag_value = "agent_backchannel_opportunity",
    // The host stamps this; the core has no wall clock.
    host_field(name = "created_at", ty = "float", default = "Field(default_factory=time.time)")
)]
#[binding(emit)]
pub struct AgentBackchannelOpportunityEvent {
    pub probability: f64,
    pub threshold: f64,
    pub end_of_turn_probability: f64,
    pub end_of_turn_threshold: f64,
    #[python(default = "None")]
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "AgentFalseInterruptionEvent",
    tag = "type",
    tag_value = "agent_false_interruption"
)]
#[binding(emit, extract, from_items, read_tag = "kind")]
pub struct AgentFalseInterruption {
    /// Whether the false interruption was resumed automatically.
    pub resumed: bool,
    // The host stamps this; the core has no wall clock.
    #[binding(skip, extract_skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
    // Deprecated, always `None` here; `#[python(deprecated)]` emits the warning shim.
    #[binding(skip, extract_skip)]
    #[python(
        ty = "ChatMessage | None",
        import = "..llm:ChatMessage",
        default = "None",
        deprecated
    )]
    pub message: (),
    #[binding(skip, extract_skip)]
    #[python(default = "None", deprecated)]
    pub extra_instructions: Option<String>,
}

/// Emitted after a batch of function tools finishes executing.
///
/// ``function_calls`` and ``function_call_outputs`` are parallel lists: the
/// output at a given index belongs to the call at the same index and carries
/// the same ``call_id``. Every call has one output, even one whose tool raised
/// ``StopResponse``; such an output asks for no reply with ``reply_required``.
#[derive(Host)]
#[python(
    pydantic,
    name = "FunctionToolsExecutedEvent",
    tag = "type",
    tag_value = "function_tools_executed",
    validator = "verify_lists_length",
    method(
        name = "zipped",
        ret = "list[tuple[FunctionCall, FunctionCallOutput]]",
        doc = "Return calls paired with outputs by list position."
    ),
    method(name = "cancel_tool_reply", ret = "None"),
    method(name = "cancel_agent_handoff", ret = "None"),
    property(name = "has_tool_reply", ret = "bool"),
    property(name = "has_agent_handoff", ret = "bool")
)]
pub struct FunctionToolsExecuted {
    #[python(ty = "list[FunctionCall]", import = "..llm:FunctionCall")]
    pub function_calls: (),
    #[python(ty = "list[FunctionCallOutput]", import = "..llm:FunctionCallOutput")]
    pub function_call_outputs: (),
    #[python(default_factory = "time.time")]
    pub created_at: f64,
    #[host(rename = "_handoff_required")]
    #[python(private, default = "False")]
    pub handoff_required: bool,
}

/// Deprecated: use session_usage_updated for usage tracking.
/// Per-turn latency metrics are available on ChatMessage.metrics.
#[derive(Host)]
#[python(
    pydantic,
    name = "MetricsCollectedEvent",
    tag = "type",
    tag_value = "metrics_collected"
)]
pub struct MetricsCollected {
    #[python(ty = "AgentMetrics", import = "..metrics:AgentMetrics")]
    pub metrics: (),
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Host)]
#[python(
    pydantic,
    name = "SessionUsageUpdatedEvent",
    tag = "type",
    tag_value = "session_usage_updated"
)]
pub struct SessionUsageUpdated {
    #[python(ty = "AgentSessionUsage", import = "..metrics:AgentSessionUsage")]
    pub usage: (),
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Host)]
#[python(
    pydantic,
    name = "_TypeDiscriminator",
    tag = "type",
    tag_value = "unknown"
)]
pub struct TypeDiscriminator {}

#[derive(Host)]
#[python(
    pydantic,
    name = "ConversationItemAddedEvent",
    tag = "type",
    tag_value = "conversation_item_added"
)]
pub struct ConversationItemAdded {
    #[python(
        ty = "ChatMessage | AgentHandoff | _TypeDiscriminator",
        import = "..llm:ChatMessage,AgentHandoff"
    )]
    pub item: (),
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

#[derive(Host)]
#[python(
    tag = "type",
    tag_value = "speech_created",
    pydantic,
    name = "SpeechCreatedEvent",
    config = "arbitrary_types_allowed=True"
)]
// The two `#[binding(py)]` fields are the call site's: the live `_core.SpeechHandle`, and `source`
// from `SpeechKind::as_source_str`.
#[binding(emit)]
pub struct SpeechCreated {
    /// True if the speech was created using public methods like `say` or `generate_reply`
    pub user_initiated: bool,
    /// Source indicating how the speech handle was created
    #[python(ty = "Literal[\"say\", \"generate_reply\"]")]
    pub source: String,
    /// The speech handle that was created
    #[python(ty = "SpeechHandle", type_import = "..voice.speech_handle:SpeechHandle", exclude)]
    pub speech_handle: crate::CoreObject,
    #[binding(skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

// --- the tool lifecycle ---------------------------------------------------------------------------
//
// The `update` dict's discriminators, not session event kinds: each carries a `TYPE` const, and
// they generate into `tool_updates` so `session_events`' `KINDS` stays the dispatcher's list.

/// A function tool call was dispatched.
#[derive(Clone, Host)]
#[python(
    pydantic,
    name = "ToolCallStarted",
    tag = "type",
    tag_const = TYPE
)]
pub struct ToolCallStarted {
    // On the wire, the `_core.ChatItem` for the call; the host rebuilds its `llm.FunctionCall`.
    #[python(ty = "FunctionCall", import = "..llm:FunctionCall")]
    pub function_call: crate::FunctionCall,
}

impl ToolCallStarted {
    pub const TYPE: &'static str = "tool_call_started";
}

/// A function call crosses as the host's `_core.ChatItem`, not as its fields.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &ToolCallStarted {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDictMethods as _;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("type", ToolCallStarted::TYPE)?;
        d.set_item("function_call", crate::ChatItem::FunctionCall(self.function_call.clone()))?;
        Ok(d)
    }
}

/// A progress update emitted via ``ctx.update()`` while a tool call runs.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "ToolCallUpdated",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct ToolCallUpdated {
    /// Entry id: ``call_id`` inline, ``{call_id}_update_N`` when deferred.
    pub id: String,
    pub call_id: String,
    pub message: String,
}

impl ToolCallUpdated {
    pub const TYPE: &'static str = "tool_call_updated";
}

/// A tool call's single terminal entry.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "ToolCallEnded",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct ToolCallEnded {
    /// Entry id: ``call_id`` inline, ``{call_id}_final`` when deferred.
    pub id: String,
    pub call_id: String,
    /// Result or error text; None when there is nothing to voice.
    #[python(default = "None")]
    pub message: Option<String>,
    #[python(ty = "Literal[\"done\", \"error\", \"cancelled\"]")]
    pub status: String,
}

impl ToolCallEnded {
    pub const TYPE: &'static str = "tool_call_ended";
}

/// Lifecycle of the deferred reply that voices buffered tool updates: ``scheduled``
/// when queued, then ``completed`` / ``interrupted`` / ``skipped``. One reply may cover
/// several calls; an inline first update never gets one.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "ToolReplyUpdated",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct ToolReplyUpdated {
    /// ``ToolCallUpdated.id`` values this reply covers.
    pub update_ids: Vec<String>,
    #[python(ty = "Literal[\"scheduled\", \"completed\", \"interrupted\", \"skipped\"]")]
    pub status: String,
    /// Id of the reply speech; ``speech_created`` carries its handle.
    pub speech_id: String,
}

impl ToolReplyUpdated {
    pub const TYPE: &'static str = "tool_reply_updated";
}

/// The four tool-lifecycle updates, as one discriminated union.
#[derive(Clone, Host)]
#[python(
    union,
    inline,
    discriminator = "type",
    name = "ToolUpdate"
)]
#[binding(emit)]
pub enum ToolUpdate {
    Started(ToolCallStarted),
    Updated(ToolCallUpdated),
    Ended(ToolCallEnded),
    Reply(ToolReplyUpdated),
}

/// One flat tool-lifecycle update. Discriminate on ``update.type``: ``tool_call_started``
/// → ``tool_call_updated`` → ``tool_call_ended`` → ``tool_reply_updated``.
#[derive(Host)]
#[python(
    pydantic,
    name = "ToolExecutionUpdatedEvent",
    tag = "type",
    tag_value = "tool_execution_updated"
)]
#[binding(emit)]
pub struct ToolExecutionUpdated {
    pub update: ToolUpdate,
    #[binding(skip)]
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

// --- limits, errors and the close ------------------------------------------------------------------

// The user turn passed a configured word or duration limit. No `///`: it would become a docstring.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "UserTurnExceededEvent",
    tag = "type",
    tag_value = "user_turn_exceeded",
    // The host stamps this; the core has no wall clock.
    host_field(name = "created_at", ty = "float", default = "Field(default_factory=time.time)")
)]
#[binding(emit)]
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
    #[python(ty = "float")]
    pub duration: std::time::Duration,
}

#[derive(Host)]
#[python(
    tag = "type",
    tag_value = "error",
    pydantic,
    name = "ErrorEvent",
    config = "arbitrary_types_allowed=True"
)]
pub struct Error {
    #[python(
        ty = "LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | Any",
        import = "..llm:LLMError,RealtimeModelError",
        import = "..stt:STTError",
        import = "..tts:TTSError",
        import = "..inference.interruption:InterruptionDetectionError",
        serializer = "serialize_error"
    )]
    pub error: (),
    #[python(
        ty = "LLM | STT | TTS | RealtimeModel | AdaptiveInterruptionDetector | Any",
        import = "..llm:LLM,RealtimeModel",
        import = "..stt:STT",
        import = "..tts:TTS",
        import = "..inference.interruption:AdaptiveInterruptionDetector",
        serializer = "serialize_source"
    )]
    pub source: (),
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

// `//`, not `///`: the reference has no docstring, and one would change `model_json_schema()`,
// which the gate digests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Host)]
#[python(enum, name = "CloseReason")]
#[node(enum)]
#[host(wire(msg = format!("unknown close reason {s:?}")))]
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

#[derive(Host)]
#[python(
    pydantic,
    name = "CloseEvent",
    tag = "type",
    tag_value = "close"
)]
pub struct Close {
    #[python(
        ty = "LLMError | STTError | TTSError | RealtimeModelError | InterruptionDetectionError | None",
        import = "..llm:LLMError,RealtimeModelError",
        import = "..stt:STTError",
        import = "..tts:TTSError",
        import = "..inference.interruption:InterruptionDetectionError",
        default = "None"
    )]
    pub error: (),
    pub reason: CloseReason,
    #[python(default_factory = "time.time")]
    pub created_at: f64,
}

// --- the union --------------------------------------------------------------------------------

/// `AgentEvent` -- everything an `AgentSession` surfaces, discriminated on `type`.
#[derive(Host)]
#[python(
    union,
    discriminator = "type",
    name = "AgentEvent"
)]
pub enum AgentEvent {
    UserInputTranscribed(UserInputTranscribed),
    UserTranscriptionTimeout(UserTranscriptionTimeout),
    UserStateChanged(UserStateChanged),
    AgentStateChanged(AgentStateChanged),
    AgentFalseInterruption(AgentFalseInterruption),
    MetricsCollected(MetricsCollected),
    SessionUsageUpdated(SessionUsageUpdated),
    ConversationItemAdded(ConversationItemAdded),
    FunctionToolsExecuted(FunctionToolsExecuted),
    SpeechCreated(SpeechCreated),
    ToolExecutionUpdated(ToolExecutionUpdated),
    Error(Error),
    Close(Close),
    /// Named, not declared: the real class (`inference/interruption.py`) carries numpy fields.
    OverlappingSpeech(
        #[python(
            ty = "OverlappingSpeechEvent",
            import = "..inference.interruption:OverlappingSpeechEvent"
        )]
        (),
    ),
}

// --- the session-event wire payloads --------------------------------------------------------------
//
// The wire shape `_on_core_event` reads: no public model, so no `python` target. Tag-only kinds are
// declared too, so each appears in the generated `KINDS` list the host tests its dispatcher against.

/// The user started speaking (`on_start_of_speech`). On the wire (`user_started_speaking`) the kind
/// is the whole message.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "user_started_speaking")]
pub struct StartOfSpeechEvent {
    /// `speech_start_time`: back-dated by VAD, or the provider's onset stamp.
    #[binding(skip)]
    pub at: crate::Ts,
    #[binding(skip)]
    pub speech_duration: std::time::Duration,
    /// `false` for the STT-driven turn detection path, where the reference passes `ev=None`.
    #[binding(skip)]
    pub from_vad: bool,
}

/// A transcript stable enough to start generating against (`_PreemptiveGenerationInfo`). On the wire
/// (`preemptive_generation`) the kind is the whole message.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "preemptive_generation")]
pub struct PreemptiveGenerationEvent {
    #[binding(skip)]
    pub new_transcript: String,
    #[binding(skip)]
    pub transcript_confidence: f64,
    #[binding(skip)]
    pub started_speaking_at: Option<crate::Ts>,
}

/// `backchannel_confirmed`: the kind is the whole message.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "backchannel_confirmed")]
pub struct BackchannelConfirmedWire {}

/// The overlap was a real interruption (`on_interruption`). On the wire (`overlap_interruption`) the
/// kind is the whole message.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "overlap_interruption")]
pub struct OverlapInterruptionEvent {
    /// `ev.overlap_started_at or ev.detected_at`, resolved (a `0.0` falls through).
    #[binding(skip)]
    pub ignore_user_transcript_until: crate::Ts,
}

/// `stream_closed`: the drain's last word; releases a host still awaiting a `speech_done`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "stream_closed")]
pub struct StreamClosedWire {}

/// `user_turn_completed`'s payload.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "user_turn_completed")]
pub struct UserTurnCompletedWire {
    /// The turn's id: what `hook_done(turn, ..)` answers for.
    pub turn: u64,
    pub new_transcript: String,
    pub transcript_confidence: f64,
}

/// The user stopped speaking (`on_end_of_speech`); `user_stopped_speaking` on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "user_stopped_speaking")]
pub struct EndOfSpeechEvent {
    #[binding(skip)]
    pub at: crate::Ts,
    #[binding(skip)]
    pub from_vad: bool,
    /// Silence already elapsed when the event was raised, in seconds (`0.0` on the STT path). `at`
    /// includes it, but a host needs it: AMD's post-speech window runs from when speech ended.
    pub silence_duration: f64,
}

/// `conversation_item_added`'s payload, read back by the remote session (its `item` only).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, extract, from_items, read_tag = "kind", tag = "type", tag_value = "conversation_item_added")]
pub struct ItemAddedWire {
    /// The `_core.ChatItem`; the host rebuilds its `llm.ChatMessage` with `ChatContext._from_core`.
    pub item: crate::ChatItem,
    /// Which agent's frame produced it, as the opaque id the host gave the core.
    #[binding(extract_skip)]
    pub agent: String,
    /// Which host projections take the item: `"both"` (`chat_ctx` and `session.history`) or
    /// `"history"` (the handoff record).
    #[binding(extract_skip)]
    pub visibility: String,
    /// The open run's id when an owned speech produced it or it is the handoff record, else `None`.
    #[binding(extract_skip)]
    pub run: Option<u64>,
}

/// `function_tools_executed`, as the driver emits it; calls and outputs are parallel lists. A twin
/// of [`FunctionToolsExecuted`]: this dict keys `handoff_required`, the model `_handoff_required`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolsExecutedEvent {
    pub function_calls: Vec<crate::FunctionCall>,
    pub function_call_outputs: Vec<crate::FunctionCallOutput>,
    /// `_handoff_required`; false when two tools returned an agent.
    pub handoff_required: bool,
}

impl ToolsExecutedEvent {
    pub const TYPE: &'static str = "function_tools_executed";

    /// `FunctionToolsExecutedEvent.has_tool_reply`.
    pub fn has_tool_reply(&self) -> bool {
        self.function_call_outputs.iter().any(|o| o.reply_required)
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &ToolsExecutedEvent {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDictMethods as _;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("type", ToolsExecutedEvent::TYPE)?;
        let calls: Vec<crate::ChatItem> = self.function_calls.iter().cloned().map(crate::ChatItem::FunctionCall).collect();
        let outputs: Vec<crate::ChatItem> = self.function_call_outputs.iter().cloned().map(crate::ChatItem::FunctionCallOutput).collect();
        d.set_item("function_calls", calls)?;
        d.set_item("function_call_outputs", outputs)?;
        d.set_item("handoff_required", self.handoff_required)?;
        Ok(d)
    }
}

#[cfg(feature = "python")]
impl crate::python::HostEmitTag for ToolsExecutedEvent {
    const TAG: Option<&'static str> = Some(ToolsExecutedEvent::TYPE);
}

/// A sentence the core decided and the host logs. `fields` is the reference's `extra={...}`, so
/// `lk.pii.*` entries go through the host's redaction filter.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "diagnostic")]
pub struct DiagnosticWire {
    /// `"debug"`, `"info"` or `"warning"` -- the name `logging` takes.
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

// The run: three host-only events `RunResult` is projected from; the call site fills each shape.

/// `run_item_added`'s payload: an item that is the run's before it is committed (a started call).
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "run_item_added")]
pub struct RunItemAddedWire {
    /// The run's id (`RunId`).
    pub run: u64,
    pub item: crate::ChatItem,
}

/// `run_started`'s payload.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "run_started")]
pub struct RunStartedWire {
    /// The run's id (`RunId`).
    pub run: u64,
}

/// `run_refused`'s payload.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "run_refused")]
pub struct RunRefusedWire {
    /// `RunRefusal::as_str`: `"nested"` or `"closing"`.
    pub cause: String,
}

/// `run_settled`'s payload: the outcome as `RunResult._mark_done` reads it. The `#[binding(py)]`
/// keys are always written, `None` when absent.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "run_settled")]
pub struct RunSettledWire {
    pub run: u64,
    /// `RunOutcome::as_str`: `"completed"`, `"failed"` or `"closed"`.
    pub outcome: String,
    /// The interned object a tool handed `AgentTask.complete()`, resolved; `None` otherwise.
    pub final_output: Option<crate::HostRef>,
    /// The exception a tool completed with, resolved; `None` otherwise.
    pub error: Option<crate::HostRef>,
    /// The last owned speech's `_core.SpeechHandle`; `None` when the run owned none.
    pub speech_handle: Option<crate::CoreObject>,
}

/// `speech_done`'s payload.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "speech_done")]
pub struct SpeechDoneWire {
    /// `speech_{n}`, the handle's id.
    pub speech_id: String,
    /// `SpeechOutcome::as_str`: `"completed"`, `"interrupted"` or `"failed"`.
    pub outcome: String,
}

/// `floor_changed`'s payload. `speech_id` is `None` when the floor is free.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "floor_changed")]
pub struct FloorChangedWire {
    pub speech_id: Option<String>,
}

/// `span_started`'s payload, emitted right before its state change. `start_time`: wall seconds.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "span_started")]
pub struct SpanStartedWire {
    pub span: Span,
    pub start_time: Option<f64>,
}

/// `span_ended`'s payload. `end_time` is wall seconds, `None` for now.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "span_ended")]
pub struct SpanEndedWire {
    pub span: Span,
    pub end_time: Option<f64>,
}

/// `context_carried`'s payload: item ids moved from `from_agent`'s context to `to_agent`'s.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "context_carried")]
pub struct ContextCarriedWire {
    pub from_agent: String,
    pub to_agent: String,
    pub items: Vec<String>,
}

/// `close`'s wire payload. `error` is written only when the driver holds one: the host attaches
/// its own error when the key is absent, so absent and `None` differ.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "close")]
pub struct CloseWire {
    pub reason: CloseReason,
    pub has_error: bool,
    /// The driver's own error as a host exception; the key is absent when there is none.
    #[binding(skip_if_none)]
    pub error: Option<crate::errors::ProviderError>,
}

/// `metrics_collected`'s wire payload: the record plus the driver's `speech_id` correlation.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "metrics_collected")]
pub struct MetricsCollectedWire {
    /// The `AgentMetrics` record as the dict `TypeAdapter(AgentMetrics)` validates.
    pub metrics: crate::metrics::base::AgentMetrics,
    /// `speech_{n}` of the speech the record belongs to; `None` for a leg that has none.
    pub speech_id: Option<String>,
}

/// `error`'s wire payload: the ingredients of `ErrorEvent`; the host wraps `error` in the leg's
/// class chosen by `source`.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "error")]
pub struct SessionErrorWire {
    /// `ErrorSource::as_str`.
    pub source: String,
    pub recoverable: bool,
    /// `"host"` when the error was captured off a plugin object, else `"engine"`.
    pub label: String,
    /// The host's own exception when the provider raised one, else an `APIError`.
    pub error: crate::errors::ProviderError,
    pub speech_id: Option<String>,
}

// --- the hook calls ------------------------------------------------------------------------------
//
// One record per `HookCall` arm, all tagged `hook_call`: the host dispatches on `hook` and answers
// with `id`.

/// `on_enter`: `Agent.on_enter()` on the frame just installed.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct OnEnterHookWire {
    pub hook: String,
    pub id: u64,
    /// The frame's agent id.
    pub agent: String,
}

/// `on_exit`: `Agent.on_exit()` on the frame being left.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct OnExitHookWire {
    pub hook: String,
    pub id: u64,
    pub agent: String,
}

/// `on_user_turn_completed`: `Agent.on_user_turn_completed(turn_ctx, new_message)` for the turn.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct OnUserTurnCompletedHookWire {
    pub hook: String,
    pub id: u64,
    /// The turn's id (`TurnId`).
    pub turn: u64,
    pub new_transcript: String,
    pub transcript_confidence: f64,
}

/// `filler_source`: a filler with a callable source fired.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct FillerSourceHookWire {
    pub hook: String,
    pub id: u64,
    /// The host's number for the `with_filler` scope (`FillerId`).
    pub filler: u64,
    pub step: u32,
}

/// `render_tool_reply`: the instructions of a deferred tool reply, rendered by the host.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct RenderToolReplyHookWire {
    pub hook: String,
    pub id: u64,
    pub agent: String,
    pub call_ids: Vec<String>,
    pub at_tail: bool,
}

/// `function_tools_executed`: one step's tool results. The host emits the event, then answers
/// `hook_tools_executed` with what its handlers left: each output's `reply_required` and the
/// event's `handoff_required` (`cancel_tool_reply()`, `cancel_agent_handoff()`).
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct ToolsExecutedHookWire {
    pub hook: String,
    pub id: u64,
    pub function_calls: Vec<crate::ChatItem>,
    pub function_call_outputs: Vec<crate::ChatItem>,
    pub handoff_required: bool,
}

/// `describe_agent`: a tool returned an `Agent` the driver cannot read.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "hook_call")]
pub struct DescribeAgentHookWire {
    pub hook: String,
    pub id: u64,
    /// `speech_{n}` of the speech whose tool returned the agent.
    pub speech_id: String,
    /// The `Agent` object, or `None` when the handle outlived its entry (answered `hook_failed`).
    pub agent: crate::HostRef,
}

// --- the interruption detector's verdict and its wire ----------------------------------------

/// `OverlappingSpeechEvent` (`inference/interruption.py:120`), the detector's only public verdict.
#[derive(Debug, Clone, PartialEq, Host)]
// The dict `_on_core_event` reads for `overlapping_speech`, field for field with the model. Also
// read back for `interruption_metrics`; the `extract_skip` fields are the ones it never reads.
#[binding(emit, extract)]
pub struct OverlappingSpeechEvent {
    /// Wall clock seconds at construction.
    #[binding(extract_skip)]
    pub created_at: f64,
    /// Wall clock seconds at which the overlap was resolved.
    #[binding(extract_skip)]
    pub detected_at: f64,
    /// The verdict. `false` is only a *backchannel* verdict when `agent_ended` is also false.
    pub is_interruption: bool,
    /// The overlap ended because the agent stopped talking; `is_interruption == false` is then
    /// inconclusive.
    pub agent_ended: bool,
    pub total_duration: f64,
    pub prediction_duration: f64,
    pub detection_delay: f64,
    /// Wall clock seconds at which the overlap began.
    #[binding(extract_skip)]
    pub overlap_started_at: Option<f64>,
    #[binding(extract_skip)]
    pub speech_input: Option<Vec<i16>>,
    #[binding(extract_skip)]
    pub probabilities: Option<Vec<f32>>,
    #[binding(extract_skip)]
    pub probability: f64,
    /// Requests sent for this overlap. Reset to zero as it is read.
    pub num_requests: u32,
}

impl OverlappingSpeechEvent {
    pub const TYPE: &'static str = "overlapping_speech";
}

/// `{"type": "overlapping_speech", "event": ..}`: the detector-side copy, which hosts skip.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "overlapping_speech")]
pub struct DetectorOverlapWire {
    /// The generated `OverlappingSpeechEventPy` dict.
    #[binding(py)]
    pub event: (),
}

/// `{"type": "error", "message": .., "recoverable": ..}`: the error rendered as a string.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "type", tag_value = "error")]
pub struct DetectorErrorWire {
    pub message: String,
    /// `true` while the detector still intends to reconnect; `false` makes the session raise.
    pub recoverable: bool,
}

// --- the host event stream, as one union ------------------------------------------------------------

/// Every event the session hands its host, as the record the host reads.
///
/// `#[python(event)]` members are emitted as public session events and `#[python(ignored)]` ones
/// are dropped, both via `_generated/session_dispatch.py`; the rest have host behaviour. Its
/// `EVENT_KINDS` also holds `stream_closed`, which the stream carries outside this union.
#[derive(Host)]
#[binding(emit)]
#[python(dispatch = "session_dispatch", dispatch_also = "StreamClosedWire")]
pub enum HostWire {
    UserTranscript(UserInputTranscribed),
    StartOfSpeech(StartOfSpeechEvent),
    EndOfSpeech(EndOfSpeechEvent),
    SpeechCreated(SpeechCreated),
    Diagnostic(DiagnosticWire),
    #[python(kind = "function_tools_executed")]
    ToolsExecuted(ToolsExecutedEvent),
    #[python(ignored)]
    UserTurnCompleted(UserTurnCompletedWire),
    EotPrediction(EotPrediction),
    #[python(ignored)]
    AgentBackchannelOpportunity(AgentBackchannelOpportunityEvent),
    #[python(ignored)]
    PreemptiveGeneration(PreemptiveGenerationEvent),
    #[python(event)]
    UserTurnExceeded(UserTurnExceededEvent),
    #[python(event)]
    TranscriptionTimeout(UserTranscriptionTimeout),
    #[python(ignored)]
    BackchannelConfirmed(BackchannelConfirmedWire),
    #[python(ignored)]
    OverlapInterruption(OverlapInterruptionEvent),
    SpanStarted(SpanStartedWire),
    SpanEnded(SpanEndedWire),
    AgentStateChanged(AgentStateChanged),
    UserStateChanged(UserStateChanged),
    #[python(event)]
    AgentFalseInterruption(AgentFalseInterruption),
    AmdResult(AmdPrediction),
    ItemAdded(ItemAddedWire),
    ContextCarried(ContextCarriedWire),
    RunStarted(RunStartedWire),
    RunItemAdded(RunItemAddedWire),
    RunRefused(RunRefusedWire),
    RunSettled(RunSettledWire),
    SpeechDone(SpeechDoneWire),
    FloorChanged(FloorChangedWire),
    Close(CloseWire),
    MetricsCollected(MetricsCollectedWire),
    Error(SessionErrorWire),
    ToolExecutionUpdated(ToolExecutionUpdated),
    HookOnEnter(OnEnterHookWire),
    HookOnExit(OnExitHookWire),
    HookOnUserTurnCompleted(OnUserTurnCompletedHookWire),
    HookFillerSource(FillerSourceHookWire),
    HookRenderToolReply(RenderToolReplyHookWire),
    HookDescribeAgent(DescribeAgentHookWire),
    HookToolsExecuted(ToolsExecutedHookWire),
    ParticipantLinked(ParticipantLinked),
    ParticipantUnlinked(ParticipantUnlinked),
    TextInput(TextInput),
    DeleteRoomRequested(DeleteRoomRequested),
}
