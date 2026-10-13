//! `llm/realtime.py`'s data half. The reference's stream-of-streams is flattened into [`Event`]s the
//! host reassembles; `generate_reply`'s future becomes a `req_id` correlated with the answer event.

use crate::Ts;
use crate::{AudioFrame, ToolChoice, ToolSpec};
use crate::{ChatItem, FunctionCall};
use lk_agents_host_derive::Host;

// The reader is strict (no `#[binding(default)]`): a missing key is a rename, and should be loud.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "realtime",
    path = "livekit.agents.llm.realtime",
    name = "RealtimeCapabilities"
)]
#[binding(extract, emit, from_items)]
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
    #[python(default = "False")]
    pub can_disable_turn_detection: bool,
    /// Whether the chat context can be updated mid-session
    #[python(default = "False")]
    pub mutable_chat_context: bool,
    /// Whether the instructions can be updated mid-session
    #[python(default = "False")]
    pub mutable_instructions: bool,
    /// Whether the tools can be updated mid-session
    #[python(default = "False")]
    pub mutable_tools: bool,
    /// Whether the tool and tool choice can be specified per response
    #[python(default = "False")]
    pub per_response_tool_choice: bool,
    /// Whether session.say() can use the realtime session directly, without TTS.
    ///
    /// When used through a RealtimeModel, add_to_chat_ctx=False is ignored and the
    /// message is still added to the chat context.
    #[python(default = "False")]
    pub supports_say: bool,
    /// Whether the server generates the reply by itself when its turn detection ends a user turn.
    /// Last: the reference has no such field, and positional arguments must keep its order.
    #[python(default = "False")]
    pub auto_turn_reply_generation: bool,
}

/// `RealtimeError` (`llm/realtime.py:88`): the failure a request reports.
///
/// `fields_only`: nothing crosses as this payload; `generate_reply_failed` spreads its keys flat.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Host)]
#[binding(fields_only)]
#[error("{message}")]
pub struct RealtimeError {
    pub message: String,
    /// The provider's error code, when it gave one (callers branch on it).
    #[binding(default)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[binding(fields_only)]
pub struct RealtimeModelError {
    /// When the host raised it, on the session's clock.
    #[binding(extract_skip)]
    pub timestamp: Ts,
    pub label: String,
    pub message: String,
    pub recoverable: bool,
    /// The request this error answers, when the server echoed its `client_event_id`; `None` for a
    /// session-level error.
    #[binding(default)]
    pub req_id: Option<String>,
    /// The provider's error code, when it has one -- the same reason [`RealtimeError::code`] exists.
    #[binding(default)]
    pub code: Option<String>,
}

/// The wire contract of `remote_item_added`: the mirror gained an item the server created.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "remote_item_added")]
pub struct RemoteItemAdded {
    /// The item this one was inserted after; absent and `None` both mean "appended".
    #[binding(default)]
    pub previous_item_id: Option<String>,
    /// A host `_core.ChatItem`.
    pub item: Box<ChatItem>,
}

/// A `response.create` will never be answered. `generate_reply_failed`, with the error flattened.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "generate_reply_failed")]
pub struct GenerateReplyFailed {
    pub req_id: String,
    #[binding(flatten)]
    pub error: RealtimeError,
}

/// The wire contract of `function_call`: the key `call` and the event name.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "function_call")]
pub struct FunctionCallReady {
    /// A host `ChatItem` that must be a `FunctionCall`; `FunctionCall`'s reader checks it.
    #[binding(skip)]
    pub call: Box<FunctionCall>,
}

/// One chunk of assistant audio. `audio_delta`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "audio_delta")]
pub struct AudioDelta {
    pub message_id: String,
    /// The generation this frame belongs to, when the plugin says.
    #[binding(default)]
    pub response_id: Option<String>,
    /// An `rtc.AudioFrame`, read by the same twin every other frame on this seam goes through.
    pub frame: AudioFrame,
}

/// What a message turned out to contain. `message_modalities`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "message_modalities")]
pub struct MessageModalities {
    pub message_id: String,
    /// The generation this message belongs to; see `AudioDelta::response_id`.
    #[binding(default)]
    pub response_id: Option<String>,
    /// `"audio"` when the host's `list[str]` has it, else `"text"`: the only question asked of it.
    pub modalities: Modalities,
}

/// The turn is over. Carries what `response.done` reported. `generation_done`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "generation_done")]
pub struct GenerationDone {
    pub response_id: String,
    /// Absent or unrecognised means `completed`, as in the reference (`GenerationStatus`' reader).
    #[binding(default)]
    pub status: GenerationStatus,
    /// Not on this wire: the host-backed path reports usage through its own metrics events.
    #[binding(extract_skip)]
    pub usage: TokenUsage,
}

/// A turn has begun. `generation_created`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "generation_created")]
pub struct GenerationCreated {
    pub response_id: String,
    /// `Some` exactly when this answers a [`RealtimeCommand::GenerateReply`].
    #[binding(default)]
    pub req_id: Option<String>,
}

/// A new assistant message inside the current generation. `message_started`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "message_started")]
pub struct MessageStarted {
    pub message_id: String,
    /// The generation this message belongs to; lets the driver drop deltas of a refused one.
    #[binding(default)]
    pub response_id: Option<String>,
}

/// A transcript or text delta. `text_delta`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "text_delta")]
pub struct TextDelta {
    pub message_id: String,
    /// The generation this delta belongs to; see `AudioDelta::response_id`.
    #[binding(default)]
    pub response_id: Option<String>,
    pub delta: String,
    /// The `io.TimedString` stamp when the provider sends one; never synthesised.
    #[binding(default)]
    pub start_time: Option<f64>,
}

/// The message is complete. `message_done`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "message_done")]
pub struct MessageDone {
    pub message_id: String,
    /// The generation this message belongs to; see `AudioDelta::response_id`.
    #[binding(default)]
    pub response_id: Option<String>,
}

/// `InputSpeechStoppedEvent` (`llm/realtime.py:53`). `user_transcription_enabled` is required.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "input_speech_stopped")]
pub struct InputSpeechStopped {
    pub user_transcription_enabled: bool,
}

/// `InputTranscriptionCompleted` (`llm/realtime.py:152`); also emitted for partials (`is_final`).
/// A host session reports it as a [`HostInputTranscription`], its onset in wall seconds.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[binding(fields_only)]
pub struct InputTranscription {
    pub item_id: String,
    pub transcript: String,
    pub is_final: bool,
    /// Geometric mean of the per-token probabilities, when the provider returns logprobs.
    #[binding(default)]
    pub confidence: Option<f64>,
    /// When the turn began: final transcripts only, when server VAD reported an onset for `item_id`.
    #[binding(extract_skip)]
    pub turn_started_at: Option<Ts>,
}

/// `input_speech_started`: the kind is the whole event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "input_speech_started")]
pub struct InputSpeechStarted {}

/// `session_reconnected`: the kind is the whole event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "session_reconnected")]
pub struct SessionReconnected {}

/// `input_transcription` as a host session reports it: the transcription, and `turn_started_at` as
/// the host's `time.time()`, which the driver uses as given.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "input_transcription")]
pub struct HostInputTranscription {
    #[binding(flatten)]
    pub transcription: InputTranscription,
    /// Absent or `None` reads as no turn start; a wrongly typed value raises.
    #[binding(default)]
    pub turn_started_at: Option<f64>,
}

/// `error` as a host session reports it: the error, and `timestamp` as the host's `time.time()`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "error")]
pub struct HostRealtimeError {
    #[binding(flatten)]
    pub error: RealtimeModelError,
    /// Required; with no anchor yet it lands on `Ts::ZERO`, so the error still forwards.
    pub timestamp: f64,
}

#[crate::host(wire_strs(msg = format!("unknown modalities {s:?}: expected 'audio' or 'text'")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Modalities {
    #[default]
    #[host(rename = "text")]
    TextOnly,
    /// `["audio", "text"]`. Audio always comes with its transcript.
    #[host(rename = "audio")]
    AudioAndText,
}

#[cfg(feature = "python")]
crate::python::from_host_str!(Modalities);

/// `response.done`'s `status`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationStatus {
    #[default]
    Completed,
    Cancelled,
    Incomplete,
    Failed,
}

/// Lenient: anything that is not a status this build knows (a newer plugin's word, a non-string)
/// reads as `completed`, as the reference's does.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for GenerationStatus {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(obj.extract::<String>().map(|s| GenerationStatus::parse(&s)).unwrap_or_default())
    }
}

impl GenerationStatus {
    /// The reference logs "Unknown response status" and treats it as complete.
    pub fn parse(s: &str) -> GenerationStatus {
        Self::from_wire(s).unwrap_or_default()
    }
}

/// Only the fields `response.done` fills; the host stamps the rest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_audio_tokens: u64,
    pub input_text_tokens: u64,
    pub input_cached_tokens: u64,
    pub input_image_tokens: u64,
    pub cached_text_tokens: u64,
    pub cached_audio_tokens: u64,
    pub cached_image_tokens: u64,
    pub output_audio_tokens: u64,
    pub output_text_tokens: u64,
    pub output_image_tokens: u64,
}

/// `MessageStarted` through `GenerationDone` are the inside of the host's one `"generation_created"`.
/// A host session reports one flat event at a time, picked by its `kind`: the reference's stream of
/// streams, flattened by `llm/realtime.py`'s `RealtimeSession._core_next_event`. The native-only
/// variants are not on that wire.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract)]
#[derive(lk_agents_host_derive::Vocabulary)]
pub enum Event {
    /// Server-side VAD; also what drives interruption.
    InputSpeechStarted(InputSpeechStarted),
    InputSpeechStopped(InputSpeechStopped),
    /// A host session reports it as [`Event::HostInputTranscription`].
    #[binding(skip)]
    InputTranscription(InputTranscription),

    GenerationCreated(GenerationCreated),
    /// A `response.create` will never be answered: rejected, timed out, cancelled, or closed.
    GenerateReplyFailed(GenerateReplyFailed),

    MessageStarted(MessageStarted),
    TextDelta(TextDelta),
    AudioDelta(AudioDelta),
    /// Settled once per message, as early as the provider allows.
    MessageModalities(MessageModalities),
    MessageDone(MessageDone),
    /// Boxed inside to keep `Event` small: `AudioDelta` shares the channel at ~10/s.
    FunctionCall(FunctionCallReady),
    GenerationDone(GenerationDone),

    /// The mirror gained an item the *server* created.
    RemoteItemAdded(RemoteItemAdded),
    /// The native session's `UpdateChatCtx` ack.
    #[binding(skip)]
    ChatCtxUpdated {
        req_id: String,
        rejected: Vec<String>,
    },
    /// The far side closed the session. After this, `next_event()`'s `None` means "closed by us".
    #[binding(skip)]
    Disconnected {
        reason: String,
    },

    SessionReconnected(SessionReconnected),
    /// `metrics_collected` with zero usage: `_report_connection_acquired`.
    #[binding(skip)]
    ConnectionAcquired {
        acquire_time: std::time::Duration,
    },
    /// A host session reports it as [`Event::HostError`].
    #[binding(skip)]
    Error(RealtimeModelError),

    /// The wall-clock twins of `InputTranscription` and `Error`, as a host session reports them.
    /// [`HostRealtimeLeg`] places an error's stamp on the session's clock ([`Event::anchored`]); a
    /// transcription's `turn_started_at` stays the host's wall seconds, which may precede the
    /// session (the message is placed at it, `agent_activity.py:2040`).
    HostInputTranscription(HostInputTranscription),
    HostError(HostRealtimeError),
}

impl Event {
    /// A host session's error, its stamp placed by `anchor` on the session's clock; a host
    /// transcription keeps its wall-clock onset.
    pub fn anchored(self, anchor: Option<crate::WallAnchor>) -> Event {
        match self {
            // Forwarded, not dropped: the host's `error` subscription owns the recoverable policy.
            Event::HostError(HostRealtimeError { mut error, timestamp }) => {
                error.timestamp = anchor.map(|a| a.ts(timestamp)).unwrap_or(Ts::ZERO);
                Event::Error(error)
            }
            ev => ev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: an unknown `response.done` status mapped to `Failed` instead of `Completed`.
    #[test]
    fn an_unknown_response_status_is_not_a_failure() {
        assert_eq!(GenerationStatus::parse("completed"), GenerationStatus::Completed);
        assert_eq!(GenerationStatus::parse("failed"), GenerationStatus::Failed);
        assert_eq!(GenerationStatus::parse("cancelled"), GenerationStatus::Cancelled);
        assert_eq!(GenerationStatus::parse("incomplete"), GenerationStatus::Incomplete);
        assert_eq!(GenerationStatus::parse("in_progress"), GenerationStatus::Completed);
    }

    /// Every `tag_value` an extract record (an event the host sends) declares in this file, in
    /// source order.
    fn declared_event_names() -> Vec<&'static str> {
        include_str!("realtime.rs")
            .split("#[binding(")
            .skip(1)
            .map(|block| block.split(")]").next().unwrap_or(""))
            .filter(|attrs| attrs.split(',').any(|w| w.trim() == "extract"))
            .filter_map(|attrs| attrs.split("tag_value = \"").nth(1)?.split('"').next())
            .collect()
    }

    // Catches: two declared events with the same wire name.
    #[test]
    fn every_declared_event_name_is_distinct() {
        let names = declared_event_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(names.len() > 1, "the scan found no events");
        assert_eq!(sorted.len(), names.len(), "two events share a wire name: {names:?}");
    }
}

/// A session is a socket that outlives every turn, so the driver opens one per agent frame.
/// `async fn session` because the reference's session starts running in its constructor.
#[async_trait::async_trait]
pub trait Realtime: Send + Sync + 'static {
    fn label(&self) -> String;

    fn capabilities(&self) -> RealtimeCapabilities;

    /// `session(turn_detection_disabled=..)`: the session's choice, not the model's.
    async fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> Result<std::sync::Arc<dyn RealtimeSession>, crate::errors::ProviderError>;
}

/// Commands, not trait methods: `&self` on a queue send cannot deadlock the audio pusher against
/// the event drain the way a `&mut self` duplex trait can.
#[async_trait::async_trait]
pub trait RealtimeSession: Send + Sync {
    fn capabilities(&self) -> RealtimeCapabilities;

    /// Never blocks; a push after close is dropped, as the reference suppresses `ChanClosed`.
    fn send(&self, cmd: RealtimeCommand);

    /// `None` once the session has stopped. One consumer only.
    async fn next_event(&self) -> Option<Event>;

    /// The server's mirror of history; may lag the agent's chat context by a turn.
    async fn chat_ctx(&self) -> crate::ChatContext;

    async fn aclose(&self);
}

/// **The `tag_value` strings below are the wire.** A host session reads each as its member's
/// dict, `kind` included.
#[derive(Debug, Clone, Host)]
#[binding(emit)]
#[derive(lk_agents_host_derive::Vocabulary)]
#[vocabulary(kind = RealtimeCommandKind)]
pub enum RealtimeCommand {
    /// The handoff's three: a new agent's frame goes over the same socket, keeping the mirror.
    UpdateInstructions(UpdateInstructions),
    UpdateChatCtx(UpdateChatCtx),
    UpdateTools(UpdateTools),

    PushAudio(PushAudio),
    PushItem(PushItem),
    GenerateReply(GenerateReply),
    /// `say()`: speak these exact words. Only when the capabilities report `supports_say`.
    Say(Say),
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
            RealtimeCommand::UpdateChatCtx(c) => Some(&c.req_id),
            RealtimeCommand::GenerateReply(c) => Some(&c.req_id),
            RealtimeCommand::Say(c) => Some(&c.req_id),
            RealtimeCommand::CancelReply(c) => Some(&c.req_id),
            RealtimeCommand::UpdateInstructions(_)
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
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "update_instructions")]
pub struct UpdateInstructions {
    pub instructions: String,
}

#[derive(Debug, Clone)]
pub struct UpdateChatCtx {
    pub req_id: String,
    /// The chat context's items; the host reconciles them into its plugin's `llm.ChatContext`.
    pub chat_ctx: crate::ChatContext,
}

impl UpdateChatCtx {
    pub const KIND: &'static str = "update_chat_ctx";
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &UpdateChatCtx {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDictMethods as _;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("kind", UpdateChatCtx::KIND)?;
        d.set_item("req_id", &self.req_id)?;
        d.set_item("items", &self.chat_ctx.items)?;
        Ok(d)
    }
}

#[cfg(feature = "python")]
impl crate::python::HostEmitTag for UpdateChatCtx {
    const TAG: Option<&'static str> = Some(UpdateChatCtx::KIND);
}

#[derive(Debug, Clone)]
pub struct UpdateTools {
    /// Tool names only: the host owns the `FunctionTool` objects.
    pub tools: Vec<ToolSpec>,
}

impl UpdateTools {
    pub const KIND: &'static str = "update_tools";
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &UpdateTools {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDictMethods as _;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("kind", UpdateTools::KIND)?;
        d.set_item("names", self.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>())?;
        Ok(d)
    }
}

#[cfg(feature = "python")]
impl crate::python::HostEmitTag for UpdateTools {
    const TAG: Option<&'static str> = Some(UpdateTools::KIND);
}

/// One frame of microphone audio.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "kind", tag_value = "push_audio")]
pub struct PushAudio {
    pub frame: AudioFrame,
}

/// A whole item to append to the mirror.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "kind", tag_value = "push_item")]
pub struct PushItem {
    /// Boxed: the one large variant.
    pub item: Box<ChatItem>,
    /// Whether a generation is expected to follow this item (R-26); `false` for a mirror update.
    pub reply_required: bool,
}

/// Ask for a turn.
#[derive(Debug, Clone, Host)]
#[binding(emit, tag = "kind", tag_value = "generate_reply")]
pub struct GenerateReply {
    pub req_id: String,
    pub instructions: Option<String>,
    /// Sent only for sessions reporting `per_response_tool_choice`; `None` when absent.
    pub tool_choice: Option<ToolChoice>,
}

/// Speak these exact words.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "say")]
pub struct Say {
    pub req_id: String,
    pub text: String,
    /// `say(add_to_chat_ctx=)`; the reference's realtime model ignores `False` (R-10).
    pub add_to_chat_ctx: bool,
}

/// `commit_audio`: end the user's turn explicitly. The kind is the whole message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "commit_audio")]
pub struct CommitAudio {}

/// `clear_audio`: discard audio the server has buffered but not yet turned into a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "clear_audio")]
pub struct ClearAudio {}

/// Cancel the open generation. See [`RealtimeCommand::Interrupt`].
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "interrupt")]
pub struct Interrupt {
    /// The generation being cancelled; `None` is "whatever is in progress". Written as explicit `None`.
    pub response_id: Option<String>,
}

/// The caller dropped the reply it was waiting for.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "cancel_reply")]
pub struct CancelReply {
    pub req_id: String,
}

/// How much of a message the user actually heard.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "truncate")]
pub struct Truncate {
    pub message_id: String,
    pub has_audio: bool,
    pub audio_end_ms: u64,
    /// Written as an explicit `None`, not an absent key: the host reads it unconditionally.
    pub audio_transcript: Option<String>,
}

/// A host `RealtimeModel` (`llm/realtime.py`'s base class, or the session's `RealtimeBridge` over
/// one): the model half, through its `_core_*` surface.
#[crate::host(proxy(host = "livekit.agents.llm.realtime.RealtimeModel"))]
#[async_trait::async_trait]
pub trait HostRealtime: Send + Sync {
    fn label(&self) -> String;

    /// Read once: the audio path never takes the GIL to ask.
    fn capabilities(&self) -> RealtimeCapabilities;

    /// Open the plugin session with the driver's `turn_detection_disabled`.
    async fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> Result<std::sync::Arc<dyn HostRealtimeSession>, crate::errors::ProviderError>;
}

/// A host `RealtimeSession`, flattened: commands in, one event at a time out.
#[crate::host(proxy(host = "livekit.agents.llm.realtime.RealtimeSession"))]
#[async_trait::async_trait]
pub trait HostRealtimeSession: Send + Sync {
    /// Queued on the host loop, in order; a command after close is dropped there.
    fn send(&self, cmd: RealtimeCommand);

    /// The next event; `None` once the session is closed. An `Err` is one event the host could
    /// not report, not the end.
    async fn next_event(&self) -> Result<Option<Event>, crate::errors::HostError>;

    async fn aclose(&self);
}

/// The plugin session opens only when the driver calls `session()`; asked again, it is the same.
pub struct HostRealtimeModel {
    host: std::sync::Arc<dyn HostRealtime>,
    label: String,
    caps: RealtimeCapabilities,
    anchor: crate::WallAnchor,
    session: tokio::sync::Mutex<Option<std::sync::Arc<HostRealtimeLeg>>>,
}

impl HostRealtimeModel {
    pub fn new(host: std::sync::Arc<dyn HostRealtime>, anchor: crate::WallAnchor) -> Self {
        HostRealtimeModel { label: host.label(), caps: host.capabilities(), host, anchor, session: Default::default() }
    }
}

#[async_trait::async_trait]
impl Realtime for HostRealtimeModel {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    async fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> Result<std::sync::Arc<dyn RealtimeSession>, crate::errors::ProviderError> {
        let mut slot = self.session.lock().await;
        if let Some(leg) = slot.as_ref() {
            return Ok(std::sync::Arc::clone(leg) as std::sync::Arc<dyn RealtimeSession>);
        }
        // The plugin's `session()` raised: the driver's start fails with its message, as the
        // reference's activity start does.
        let host = self.host.session(turn_detection_disabled).await.map_err(|e| {
            crate::errors::ProviderError::Local(crate::errors::ApiError::new(format!(
                "realtime session could not be opened: {}",
                e.message()
            )))
        })?;
        let leg = std::sync::Arc::new(HostRealtimeLeg { host, caps: self.caps, anchor: self.anchor });
        *slot = Some(std::sync::Arc::clone(&leg));
        Ok(leg as std::sync::Arc<dyn RealtimeSession>)
    }
}

pub struct HostRealtimeLeg {
    host: std::sync::Arc<dyn HostRealtimeSession>,
    caps: RealtimeCapabilities,
    anchor: crate::WallAnchor,
}

#[async_trait::async_trait]
impl RealtimeSession for HostRealtimeLeg {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        self.host.send(cmd);
    }

    async fn next_event(&self) -> Option<Event> {
        loop {
            match self.host.next_event().await {
                Ok(ev) => return ev.map(|ev| ev.anchored(Some(self.anchor))),
                // A malformed event is the plugin's bug; the session goes on.
                Err(e) => log::warn!(target: "livekit.agents", error = e.message.as_str(); "realtime event refused"),
            }
        }
    }

    async fn chat_ctx(&self) -> crate::ChatContext {
        // Only read when no agent context is bound, which never happens for a host session.
        crate::ChatContext::default()
    }

    async fn aclose(&self) {
        self.host.aclose().await;
    }
}

/// Only a native model crosses as `realtime`: a plugin's `RealtimeModel` is its `realtime_host`.
#[cfg(feature = "python")]
impl crate::python::HostProxy for dyn Realtime {
    fn from_host(
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        _clock: Option<std::sync::Arc<dyn crate::Clock>>,
    ) -> pyo3::PyResult<std::sync::Arc<Self>> {
        crate::python::native::<dyn Realtime>(obj)
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected NativeRealtime"))
    }

    fn view(_handle: crate::python::HostHandle, _clock: Option<std::sync::Arc<dyn crate::Clock>>) -> pyo3::PyResult<Box<Self>> {
        Err(pyo3::exceptions::PyTypeError::new_err("expected NativeRealtime"))
    }
}
