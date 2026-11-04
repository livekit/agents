//! `llm/realtime.py`'s data half. The reference's stream-of-streams is flattened into [`Event`]s the
//! host reassembles; `generate_reply`'s future becomes a `req_id` correlated with the answer event.

use crate::errors::{ApiError, HostError};
use crate::{ApiConnectOptions, AudioFrame, Maybe, ToolChoice, ToolSpec, Ts};
use crate::{ChatItem, FunctionCall};

#[crate::host(record(
    module = "realtime",
    path = "livekit.agents.llm.realtime",
    name = "RealtimeCapabilities",
    dataclass,
    // The host's alone: the core neither holds the model's speech back nor cuts it by itself.
    host_fields(supports_overlapping_speech(
        ty = "bool",
        default = "False",
        doc = "Whether the model and the caller may speak at once, the model deciding when to yield"
    )),
    extract,
    emit,
))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RealtimeCapabilities {
    /// Whether generated assistant messages can be truncated after interruption
    pub message_truncation: bool,
    /// Whether the model emits server-side speech start and stop events for turn taking
    pub turn_detection: bool,
    /// Whether the model emits user audio transcription events
    pub user_transcription: bool,
    /// Whether the model automatically generates a reply after receiving tool results
    pub auto_tool_reply_generation: bool,
    /// Whether the model can produce audio output directly
    pub audio_output: bool,
    /// Whether function call items already in the chat context can be resumed
    pub manual_function_calls: bool,
    /// Whether server-side turn detection can be disabled so the client drives turn-taking.
    #[host(default = false)]
    pub can_disable_turn_detection: bool,
    /// Whether the chat context can be updated mid-session
    #[host(default = false)]
    pub mutable_chat_context: bool,
    /// Whether the instructions can be updated mid-session
    #[host(default = false)]
    pub mutable_instructions: bool,
    /// Whether the tools can be updated mid-session
    #[host(default = false)]
    pub mutable_tools: bool,
    /// Whether the tool and tool choice can be specified per response
    #[host(default = false)]
    pub per_response_tool_choice: bool,
    /// Whether session.say() can use the realtime session directly, without TTS.
    ///
    /// When used through a RealtimeModel, add_to_chat_ctx=False is ignored and the
    /// message is still added to the chat context.
    #[host(default = false)]
    pub supports_say: bool,
    /// Whether the server generates the reply by itself when its turn detection ends a user turn.
    // Last: the reference has no such field, and positional arguments must keep its order.
    #[host(default = false)]
    pub auto_turn_reply_generation: bool,
}

/// `RealtimeError` (`llm/realtime.py:88`): the failure a request reports.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
#[error("{message}")]
pub struct RealtimeError {
    pub message: String,
    /// The provider's error code, when it gave one (callers branch on it).
    pub code: Option<String>,
}

impl RealtimeError {
    pub fn new(message: impl Into<String>) -> Self {
        RealtimeError { message: message.into(), code: None }
    }

    pub fn with_code(message: impl Into<String>, code: Option<String>) -> Self {
        RealtimeError { message: message.into(), code }
    }
}

/// `RealtimeModelError` (`llm/realtime.py:48`), the payload of the `"error"` event. A host session
/// reports it as a [`HostRealtimeError`], whose wall-clock stamp the leg places on the session's.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RealtimeModelError {
    /// When the host raised it, on the session's clock.
    pub timestamp: Ts,
    pub label: String,
    pub message: String,
    pub recoverable: bool,
    /// The request this error answers, when the server echoed its `client_event_id`; `None` for a
    /// session-level error.
    pub req_id: Option<String>,
    /// The provider's error code, when it has one -- the same reason [`RealtimeError::code`] exists.
    pub code: Option<String>,
    /// Native transport failure, emitted with its category and timeout phase. Host failures
    /// already carry their captured exception through the host-error boundary.
    pub provider_error: Option<crate::errors::ApiError>,
}

// --- the events a host session reports and the commands it is sent (`_generated/realtime_wire.py`)

/// The mirror gained an item the server created.
#[crate::host(record(
    module = "realtime_wire",
    path = "livekit.agents.llm.realtime",
    name = "RemoteItemAddedEvent",
    dataclass,
    extract
))]
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteItemAdded {
    /// The item this one was inserted after; `None` means "appended".
    pub previous_item_id: Option<String>,
    pub item: Box<ChatItem>,
}

/// What the model measured (each response's `RealtimeModelMetrics`, the connection's, the input
/// transcription's `STTMetrics`), for the session's metrics and usage.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeMetrics {
    pub metrics: Box<crate::metrics::base::AgentMetrics>,
}

/// A `response.create` will never be answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateReplyFailed {
    pub req_id: String,
    pub error: RealtimeError,
}

/// A function call the model made.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCallReady {
    /// The plugin's `llm.FunctionCall`, not the legacy model its derive generates.
    #[host(host_type = "livekit.agents.llm.chat_context.FunctionCall")]
    pub call: Box<FunctionCall>,
}

/// One chunk of assistant audio.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct AudioDelta {
    pub message_id: String,
    /// The generation this frame belongs to, when the plugin says.
    #[host(default = None)]
    pub response_id: Option<String>,
    pub frame: AudioFrame,
}

/// What a message turned out to contain.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageModalities {
    pub message_id: String,
    /// The generation this message belongs to; see `AudioDelta::response_id`.
    #[host(default = None)]
    pub response_id: Option<String>,
    pub modalities: Modalities,
}

/// The turn is over. Carries what `response.done` reported.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationDone {
    pub response_id: String,
    /// Not reported by a host session (a plugin's generation has no status): `completed`.
    #[host(skip)]
    pub status: GenerationStatus,
    /// Not reported by a host session: it reports usage through its own metrics events.
    #[host(skip)]
    pub usage: Box<crate::usage::TokenUsage>,
}

/// A generation the server started by itself (`generation_created` with `user_initiated=False`):
/// the host leg drains it like a requested one.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerGeneration {
    /// The plugin's `GenerationCreatedEvent`, handed back to `_core_drain`.
    pub generation: crate::HostRef,
}

/// A turn has begun.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationCreated {
    pub response_id: String,
    /// `Some` exactly when this answers a [`RealtimeCommand::GenerateReply`].
    #[host(default = None)]
    pub req_id: Option<String>,
}

/// A transcript generation whose audio belonged to an already completed speech generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptAttribution {
    pub response_id: String,
    pub source_response_id: String,
}

/// A new assistant message inside the current generation.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageStarted {
    pub message_id: String,
    /// The generation this message belongs to; lets the driver drop deltas of a refused one.
    #[host(default = None)]
    pub response_id: Option<String>,
}

/// A transcript or text delta.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextDelta {
    pub message_id: String,
    /// The generation this delta belongs to; see `AudioDelta::response_id`.
    #[host(default = None)]
    pub response_id: Option<String>,
    pub delta: String,
    /// The `io.TranscriptChunk` stamp when the provider sends one; never synthesised.
    #[host(default = None)]
    pub start_time: Option<f64>,
}

/// The message is complete.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDone {
    pub message_id: String,
    /// The generation this message belongs to; see `AudioDelta::response_id`.
    #[host(default = None)]
    pub response_id: Option<String>,
}

/// `InputSpeechStoppedEvent` (`llm/realtime.py:53`).
#[crate::host(record(
    module = "realtime_wire",
    path = "livekit.agents.llm.realtime",
    name = "InputSpeechStoppedEvent",
    dataclass,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpeechStopped {
    pub user_transcription_enabled: bool,
}

/// `InputTranscriptionCompleted` (`llm/realtime.py:152`); also emitted for partials (`is_final`).
#[crate::host(record(
    module = "realtime_wire",
    path = "livekit.agents.llm.realtime",
    name = "InputTranscriptionCompleted",
    dataclass,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputTranscription {
    pub item_id: String,
    pub transcript: String,
    pub is_final: bool,
    /// Geometric mean of the per-token probabilities, when the provider returns logprobs.
    #[host(default = None)]
    pub confidence: Option<f64>,
    /// When the turn began, as `time.time()`: final transcripts only, when server VAD reported an
    /// onset for `item_id`. Wall seconds, whatever clock the session runs on: it may precede the
    /// session (the message is placed at it, `agent_activity.py:2040`).
    #[host(default = None)]
    pub turn_started_at: Option<f64>,
}

/// Server-side VAD heard the caller start speaking.
#[crate::host(record(
    module = "realtime_wire",
    path = "livekit.agents.llm.realtime",
    name = "InputSpeechStartedEvent",
    dataclass,
    extract
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSpeechStarted {}

/// The session reconnected to the server.
#[crate::host(record(
    module = "realtime_wire",
    path = "livekit.agents.llm.realtime",
    name = "RealtimeSessionReconnectedEvent",
    dataclass,
    extract
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionReconnected {}

/// `error` as a host session reports it, `timestamp` the host's `time.time()`.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct HostRealtimeError {
    pub label: String,
    pub message: String,
    pub recoverable: bool,
    /// With no anchor yet it lands on `Ts::ZERO`, so the error still forwards.
    pub timestamp: f64,
    #[host(default = None)]
    pub req_id: Option<String>,
    #[host(default = None)]
    pub code: Option<String>,
}

/// What a message contains: audio always comes with its transcript.
#[crate::host(record(module = "realtime_wire", extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Modalities {
    #[default]
    #[host(rename = "text")]
    TextOnly,
    #[host(rename = "audio")]
    AudioAndText,
}

/// `response.done`'s `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationStatus {
    #[default]
    Completed,
    Cancelled,
    Incomplete,
    Failed,
}

/// `MessageStarted` through `GenerationDone` are the inside of the host's one `"generation_created"`.
/// A host session reports one event at a time, the reference's stream of streams flattened by the
/// generated `RealtimeSession._core_drain`. The native variants are not on that wire.
#[crate::host(record(module = "realtime_wire", extract, transparent))]
#[derive(Debug, Clone, PartialEq, lk_agents_host_derive::Vocabulary)]
pub enum Event {
    /// Server-side VAD; also what drives interruption.
    InputSpeechStarted(InputSpeechStarted),
    InputSpeechStopped(InputSpeechStopped),
    InputTranscription(InputTranscription),

    GenerationCreated(GenerationCreated),
    /// A `response.create` will never be answered: rejected, timed out, cancelled, or closed.
    #[host(native)]
    GenerateReplyFailed(GenerateReplyFailed),
    /// Consumed by inference’s `HostRealtimeLeg`, which drains it; never reaches the driver.
    ServerGeneration(ServerGeneration),

    MessageStarted(MessageStarted),
    TextDelta(TextDelta),
    #[host(native)]
    TranscriptAttribution(TranscriptAttribution),
    AudioDelta(AudioDelta),
    /// Settled once per message, as early as the provider allows.
    MessageModalities(MessageModalities),
    MessageDone(MessageDone),
    /// Boxed inside to keep `Event` small: `AudioDelta` shares the channel at ~10/s.
    FunctionCall(FunctionCallReady),
    GenerationDone(GenerationDone),

    /// The mirror gained an item the *server* created.
    RemoteItemAdded(RemoteItemAdded),
    SessionReconnected(SessionReconnected),
    /// A new session replaced the one that failed or was restarted (the fallback adapter's
    /// swap): it holds no instructions, conversation or tools until the driver gives it its frame,
    /// and takes no request before. `regenerate`: ask again for a reply the swap cut.
    #[host(native)]
    SessionSwapped {
        regenerate: bool,
    },
    /// The host session's `metrics_collected`, and the native session's measurements.
    Metrics(RealtimeMetrics),
    /// A host session reports it as [`Event::HostError`].
    #[host(native)]
    Error(RealtimeModelError),

    /// The wall-clock twin of `Error`, as a host session reports it. inference’s `HostRealtimeLeg` places its
    /// stamp on the session's clock ([`Event::anchored`]).
    HostError(HostRealtimeError),
}

impl Event {
    /// A host session's error, its stamp placed by `anchor` on the session's clock.
    pub fn anchored(self, anchor: Option<crate::WallAnchor>) -> Event {
        match self {
            // Forwarded, not dropped: the host's `error` subscription owns the recoverable policy.
            Event::HostError(HostRealtimeError {
                label,
                message,
                recoverable,
                timestamp,
                req_id,
                code,
            }) => Event::Error(RealtimeModelError {
                timestamp: anchor.map(|a| a.ts(timestamp)).unwrap_or(Ts::ZERO),
                label,
                message,
                recoverable,
                req_id,
                code,
                provider_error: None,
            }),
            ev => ev,
        }
    }
}

/// A session is a socket that outlives every turn, so the driver opens one per agent frame.
/// `async fn session` because the reference's session starts running in its constructor.
#[crate::host(native(refused = "expected OpenAIRealtimeModel"))]
pub trait Realtime: Send + Sync + 'static {
    /// The host model's identity; `None` for a native model.
    fn instance_id(&self) -> Option<usize> {
        None
    }

    fn label(&self) -> String;

    /// The model and provider its responses are counted and labelled against.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    fn capabilities(&self) -> RealtimeCapabilities;

    /// `session(turn_detection_disabled=..)`: the session's choice, not the model's.
    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> crate::BoxFuture<
        '_,
        Result<std::sync::Arc<dyn RealtimeSession>, crate::errors::ProviderError>,
    >;
}

/// Commands, not trait methods: `&self` on a queue send cannot deadlock the audio pusher against
/// the event drain the way a `&mut self` duplex trait can.
pub trait RealtimeSession: Send + Sync {
    fn capabilities(&self) -> RealtimeCapabilities;

    /// Never blocks; a push after close is dropped, as the reference suppresses `ChanClosed`.
    fn send(&self, cmd: RealtimeCommand);

    /// `None` once the session has stopped. One consumer only.
    fn next_event(&self) -> crate::BoxFuture<'_, Option<Event>>;

    /// The server's mirror of history; may lag the agent's chat context by a turn.
    fn chat_ctx(&self) -> crate::BoxFuture<'_, crate::ChatContext>;

    fn aclose(&self) -> crate::BoxFuture<'_, ()>;
}

/// What the driver sends a session. A host session is sent the ones it applies at once
/// (`HostRealtimeSession::send`) and the reply requests; the leg answers the native ones itself or
/// through the session's update calls.
#[crate::host(record(module = "realtime_wire", emit, transparent))]
#[derive(Debug, Clone, lk_agents_host_derive::Vocabulary)]
#[vocabulary(kind = RealtimeCommandKind)]
pub enum RealtimeCommand {
    /// The handoff's three: a new agent's frame goes over the same socket, keeping the mirror.
    #[host(native)]
    UpdateInstructions(UpdateInstructions),
    #[host(native)]
    UpdateChatCtx(UpdateChatCtx),
    #[host(native)]
    UpdateTools(UpdateTools),

    PushAudio(PushAudio),
    #[host(native)]
    PushItem(PushItem),
    GenerateReply(GenerateReply),
    /// `say()`: speak these exact words. Only when the capabilities report `supports_say`.
    Say(Say),
    #[host(native)]
    CancelReply(CancelReply),
    /// End the user's turn explicitly. Only meaningful with server turn detection off.
    CommitAudio(CommitAudio),
    ClearAudio(ClearAudio),
    /// A host session's `interrupt()` ignores the id.
    Interrupt(Interrupt),
    /// Tell the server how much of a message the user heard. Ignored without `message_truncation`.
    Truncate(Truncate),
}

impl RealtimeCommand {
    pub fn req_id(&self) -> Option<&str> {
        match self {
            RealtimeCommand::GenerateReply(c) => Some(&c.req_id),
            RealtimeCommand::Say(c) => Some(&c.req_id),
            RealtimeCommand::CancelReply(c) => Some(&c.req_id),
            RealtimeCommand::UpdateInstructions(_)
            | RealtimeCommand::UpdateChatCtx(_)
            | RealtimeCommand::UpdateTools(_)
            | RealtimeCommand::PushAudio(_)
            | RealtimeCommand::PushItem(_)
            | RealtimeCommand::CommitAudio(_)
            | RealtimeCommand::ClearAudio(_)
            | RealtimeCommand::Interrupt(_)
            | RealtimeCommand::Truncate(_) => None,
        }
    }
}

/// The new agent's instructions. The whole system prompt, not a delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateInstructions {
    pub instructions: String,
}

#[derive(Debug, Clone)]
pub struct UpdateChatCtx {
    pub chat_ctx: crate::ChatContext,
}

#[derive(Debug, Clone)]
pub struct UpdateTools {
    pub tools: Vec<ToolSpec>,
}

/// One frame of microphone audio.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone)]
pub struct PushAudio {
    pub frame: AudioFrame,
}

/// A whole item to append to the mirror.
#[derive(Debug, Clone)]
pub struct PushItem {
    /// Boxed: the one large variant.
    pub item: Box<ChatItem>,
    /// Whether a generation is expected to follow this item (R-26); `false` for a mirror update.
    pub reply_required: bool,
}

/// Ask for a turn.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone)]
pub struct GenerateReply {
    pub req_id: String,
    pub instructions: Option<String>,
    /// Sent only for sessions reporting `per_response_tool_choice`; `None` when absent.
    pub tool_choice: Option<ToolChoice>,
}

/// Speak these exact words.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Say {
    pub req_id: String,
    pub text: String,
    /// `say(add_to_chat_ctx=)`; the reference's realtime model ignores `False` (R-10).
    pub add_to_chat_ctx: bool,
}

/// End the user's turn explicitly.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitAudio {}

/// Discard audio the server has buffered but not yet turned into a turn.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClearAudio {}

/// Cancel the open generation. See [`RealtimeCommand::Interrupt`].
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interrupt {
    /// The generation being cancelled; `None` is "whatever is in progress".
    pub response_id: Option<String>,
}

/// The caller dropped the reply it was waiting for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelReply {
    pub req_id: String,
}

/// How much of a message the user actually heard.
#[crate::host(record(module = "realtime_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncate {
    pub message_id: String,
    pub has_audio: bool,
    pub audio_end_ms: u64,
    pub audio_transcript: Option<String>,
}

// A host `RealtimeModel`, as a session binds it to its tools: the model half, through the
// `_core_*` surface of its hand-written base class (`realtime.py`).
/// A realtime model: speech in, speech out, over one live session.
///
/// Plugins implement `session()`, returning a `RealtimeSession`.
#[crate::host(proxy(host = "livekit.agents.llm.realtime.RealtimeModel", base = "realtime.py"))]
pub trait HostRealtime: Send + Sync {
    /// The host object's identity, independent of proxy instances and bound connection options;
    /// `None` for a native provider.
    #[host(read_once, name = "_core_instance_id")]
    fn instance_id(&self) -> Option<usize> {
        None
    }

    /// The name logs and metrics give this instance: its class path.
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    // Read once: the audio path never takes the GIL to ask.
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> RealtimeCapabilities;

    /// The model and provider labelling this instance's turn metrics.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Create a new session, optionally with server-side turn detection disabled.
    ///
    /// The model itself is left unchanged and reusable.
    ///
    /// Args:
    /// - turn_detection_disabled: Hand turn-taking to the framework; honored only by plugins
    ///   reporting `can_disable_turn_detection`.
    // Open the plugin session with the driver's `turn_detection_disabled`. The session's events and
    // commands cross through `HostRealtimeSession`, which the session class answers.
    #[host(await, name = "_core_session")]
    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> crate::BoxFuture<
        '_,
        Result<std::sync::Arc<dyn HostRealtimeSession>, crate::errors::ProviderError>,
    >;
}

/// A host `RealtimeSession`, flattened: the leg orders the commands, the session's callbacks
/// report one event at a time.
// A dropped call cancels its host coroutine.
#[crate::host(proxy(host = "livekit.agents.llm.realtime.RealtimeSession", base = "realtime.py"))]
pub trait HostRealtimeSession: Send + Sync {
    /// A command that applies at once (`PushAudio`, `Interrupt`, `CommitAudio`, `ClearAudio`,
    /// `Truncate`), queued on the host loop in order.
    #[host(notify, name = "_core_send")]
    fn send(&self, cmd: RealtimeCommand);

    /// The new agent's instructions; the leg awaits each update before the next.
    #[host(await, name = "_core_update_instructions")]
    fn update_instructions(
        &self,
        instructions: String,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;

    /// The conversation, reconciled into the plugin's `llm.ChatContext`.
    #[host(await, name = "_core_update_chat_ctx")]
    fn update_chat_ctx(
        &self,
        items: Vec<ChatItem>,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;

    /// One item appended to the plugin's chat context.
    #[host(await, name = "_core_push_item")]
    fn push_item(
        &self,
        item: ChatItem,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;

    /// The tools the session offers, by name: the host owns the `FunctionTool` objects.
    #[host(await, name = "_core_update_tools")]
    fn update_tools(
        &self,
        names: Vec<String>,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;

    /// Send a reply request (`GenerateReply`, `Say`): its pending future, as the session returned
    /// it.
    #[host(await, name = "_core_request")]
    fn request(
        &self,
        cmd: RealtimeCommand,
    ) -> crate::BoxFuture<'_, Result<crate::HostRef, RealtimeError>>;

    /// The generation a request's future resolves to.
    #[host(await, name = "_core_generation")]
    fn generation(
        &self,
        request: crate::HostRef,
    ) -> crate::BoxFuture<'_, Result<crate::HostRef, RealtimeError>>;

    /// Drop a request still waiting for its generation (`future.cancel()`).
    #[host(await, name = "_core_cancel")]
    fn cancel(&self, request: crate::HostRef) -> crate::BoxFuture<'_, ()>;

    /// Report a generation's messages and function calls as events; `req_id` names the request it
    /// answers, `None` for one the server started.
    #[host(await, name = "_core_drain")]
    fn drain(&self, generation: crate::HostRef, req_id: Option<String>)
    -> crate::BoxFuture<'_, ()>;

    /// The next event; `None` once the session is closed. An `Err` is one event the host could
    /// not report, not the end.
    #[host(await, name = "_core_next_event")]
    fn next_event(&self) -> crate::BoxFuture<'_, Result<Option<Event>, crate::errors::HostError>>;

    #[host(await, name = "_core_aclose")]
    fn aclose(&self) -> crate::BoxFuture<'_, ()>;
}

/// A plugin's realtime model as the core's own: its session opens when the driver first asks and
/// is the same when asked again.
#[crate::host(class(name = "HostRealtimeModel", native = crate::llm::realtime::Realtime))]
pub trait HostRealtimeModel: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        model: std::sync::Arc<dyn HostRealtime>,
    ) -> Result<Self, crate::errors::HostError>;
}

// The OpenAI-protocol realtime client (OpenAI/Azure, xAI, LiveKit Inference) and its
// presence-aware option boundary.

/// A session update: an omitted option (`NOT_GIVEN`) keeps its setting, an explicit `None`
/// clears it.
#[crate::host(record(module = "realtime_options", extract))]
#[derive(Clone, Default)]
pub struct RealtimeOptions {
    pub input_audio_transcription: Maybe<Option<serde_json::Value>>,
    pub input_audio_noise_reduction: Maybe<Option<serde_json::Value>>,
    pub turn_detection: Maybe<Option<serde_json::Value>>,
    pub tool_choice: Maybe<Option<ToolChoice>>,
    pub speed: Maybe<f64>,
    pub tracing: Maybe<Option<serde_json::Value>>,
    pub truncation: Maybe<Option<serde_json::Value>>,
    pub reasoning: Maybe<Option<serde_json::Value>>,
    pub max_session_duration: Maybe<Option<f64>>,
    /// A voice name, or the provider's voice object.
    pub voice: Maybe<serde_json::Value>,
    pub max_response_output_tokens: Maybe<Option<serde_json::Value>>,
}

#[crate::host(record(module = "realtime_options", extract))]
#[derive(Clone)]
pub struct OpenAIRealtimeConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub conn_options: ApiConnectOptions,
    pub options: RealtimeOptions,
    #[host(default = None)]
    pub dialect: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = None)]
    pub voice: Option<serde_json::Value>,
    #[host(default = None)]
    pub modalities: Option<Vec<String>>,
    #[host(default = None)]
    pub azure_deployment: Option<String>,
    #[host(default = None)]
    pub entra_token: Option<String>,
    #[host(default = false)]
    pub azure_defaults: bool,
    #[host(default = None)]
    pub api_secret: Option<String>,
    #[host(default = None)]
    pub inference_provider: Option<String>,
    #[host(default = None)]
    pub inference_class: Option<String>,
    #[host(default = None)]
    pub label: Option<String>,
}

#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum RealtimeConfigurationError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    #[host(delegate)]
    Api(#[from] ApiError),
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

#[crate::host(class(name = "OpenAIRealtimeModel", native = Realtime))]
pub trait OpenAIRealtimeModel: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        config: OpenAIRealtimeConfig,
    ) -> Result<Self, RealtimeConfigurationError>;
    #[host(getter)]
    fn provider(&self) -> String;
    #[host(getter)]
    fn model(&self) -> String;
    #[host(getter)]
    fn label(&self) -> String;
    #[host(call)]
    fn capabilities(&self) -> RealtimeCapabilities;
    #[host(call)]
    fn update_options(&self, options: RealtimeOptions);
    #[host(call)]
    fn bind_identification(&self, headers: Vec<(String, String)>);
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "realtime_wire", path = "livekit.agents.llm.realtime", dataclass))]
pub struct MessageGeneration {
    pub message_id: String,
    #[host(annotation = "collections.abc.AsyncIterable[str | .transcript.TranscriptChunk]")]
    pub text_stream: crate::HostRef,
    #[host(annotation = "collections.abc.AsyncIterable[rtc.AudioFrame]")]
    pub audio_stream: crate::HostRef,
    #[host(annotation = "collections.abc.Awaitable[list[typing.Literal['text', 'audio']]]")]
    pub modalities: crate::HostRef,
}

#[crate::host(record(module = "realtime_wire", path = "livekit.agents.llm.realtime", dataclass))]
pub struct GenerationCreatedEvent {
    #[host(annotation = "collections.abc.AsyncIterable[MessageGeneration]")]
    pub message_stream: crate::HostRef,
    #[host(annotation = "collections.abc.AsyncIterable[.chat_items.FunctionCall]")]
    pub function_stream: crate::HostRef,
    /// True if the message was generated by the user using generate_reply()
    pub user_initiated: bool,
    /// The response ID associated with this generation, used for metrics attribution
    #[host(default = None)]
    pub response_id: Option<String>,
}

#[crate::host(exceptions(module = "realtime_errors", path = "livekit.agents.llm.realtime", classes(
    RealtimeError(base = "Exception", doc = "A realtime request failed, with an optional provider error code.",
        fields(message(ty = "str", positional, inherited), code(ty = "str | None", default = None)), base_args(message))
)))]
const _: () = ();

/// The public provider error event; its Python exception is retained but never serialized.
#[crate::host(record(
    module = "realtime",
    name = "RealtimeModelError",
    path = "livekit.agents.llm",
    pydantic,
    arbitrary_types,
    constant(type = "realtime_model_error")
))]
pub struct RealtimeErrorEvent {
    pub timestamp: f64,
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    pub recoverable: bool,
}
