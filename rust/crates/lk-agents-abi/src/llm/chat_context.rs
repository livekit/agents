//! `llm/chat_context.py` as plain data. The reference's `copy()` is shallow and the pipeline relies
//! on the aliasing; Rust clones, so every mutation through an alias is a `DiffOp::Replace`.

pub mod protocol;

use std::time::SystemTime;

/// JSON, not an opaque ref: the reference branches on it.
pub type Extra = serde_json::Map<String, serde_json::Value>;

use crate::{HostRef, MediaRef};

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "chat_items", path = "livekit.agents.llm.chat_context"))]
const _: () = ();

// `ChatRole`. Declaration order matches the reference's; the wire strings
// are the exact `Literal`s.
#[crate::host(record(module = "chat_items", name = "ChatRole", extract, emit))]
#[crate::host(wire_strs)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
}

/// `speech_handle.py:19`. The wire strings are public API: users compare
/// `details.modality == "audio"`. The default is `voice::speech::DEFAULT_INPUT_MODALITY`.
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral { kind: "modality", value: s.to_owned() }))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Audio,
}

/// The `FromStr` error of every literal enum a binding reads back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLiteral {
    pub kind: &'static str,
    pub value: String,
}

impl std::fmt::Display for UnknownLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown {} {:?}", self.kind, self.value)
    }
}

impl std::error::Error for UnknownLiteral {}

impl Role {
    /// `None` for anything else (no case folding).
    pub fn from_literal(s: &str) -> Option<Role> {
        s.parse().ok()
    }
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Message,
    FunctionCall,
    FunctionCallOutput,
    AgentHandoff,
    AgentConfigUpdate,
}

impl ItemKind {
    pub fn is_function_call_pair(self) -> bool {
        matches!(self, ItemKind::FunctionCall | ItemKind::FunctionCallOutput)
    }
}

// `ImageContent.inference_detail`. OpenAI-only today. An unknown value is
// refused, never defaulted: it reaches OpenAI's request body verbatim.
#[crate::host(record(module = "chat_items", extract, emit))]
#[crate::host(wire_strs)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceDetail {
    #[default]
    Auto,
    High,
    Low,
}

/// A URL passes through as-is; a frame is encoded first, and the resize fields apply only to it.
/// The frame crosses as its side-table key, handed back as the host's own frame (callers compare
/// by identity).
#[crate::host(record(module = "chat_items", extract, emit, transparent))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", serde(tag = "kind", content = "value", rename_all = "snake_case"))]
pub enum ImageSource {
    Url(String),
    #[host(host_type = "livekit.rtc.VideoFrame")]
    Frame(MediaRef),
}

// `bare_urls`: the `///` below is a Python docstring, and the reference writes the URL bare.
#[allow(rustdoc::bare_urls)]
/// ImageContent is used to input images into the ChatContext on supported LLM providers / plugins.
///
/// You may need to consult your LLM provider's documentation on supported URL types.
///
/// ```python
/// # Pass a VideoFrame directly, which will be automatically converted to a JPEG data URL internally
/// async for event in rtc.VideoStream(video_track):
///     chat_image = ImageContent(image=event.frame)
///     # this instance is now available for your ChatContext
///
/// # Encode your VideoFrame yourself for more control, and pass the result as a data URL (see EncodeOptions for more details)
/// from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
///
/// image_bytes = await encode(
///     event.frame,
///     EncodeOptions(
///         format="PNG",
///         resize_options=ResizeOptions(width=512, height=512, strategy="scale_aspect_fit"),
///     ),
/// )
/// chat_image = ImageContent(
///     image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
/// )
///
/// # With an external URL
/// chat_image = ImageContent(image="https://example.com/image.jpg")
/// ```
// Field order (`type` after `id`) is `model_dump()` key order and the `repr`'s.
#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "image_content"),
    constants_after = "id",
    private(_cache = "dict[object, object]"),
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageContent {
    /// Unique identifier for the image. Use a new id when replacing inline image data
    /// or a video frame; input-delta telemetry uses this id instead of hashing media bytes.
    #[host(default_factory = "..utils.shortuuid(\"img_\")")]
    pub id: String,
    /// Either a string URL or a VideoFrame object
    pub image: ImageSource,
    /// Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    #[host(default = None)]
    pub inference_width: Option<u32>,
    /// Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    #[host(default = None)]
    pub inference_height: Option<u32>,
    /// Detail parameter for LLM provider, if supported.
    ///
    /// Currently only supported by OpenAI (see https://platform.openai.com/docs/guides/vision?lang=node#low-or-high-fidelity-image-understanding)
    #[host(default = "auto")]
    pub inference_detail: InferenceDetail,
    /// MIME type of the image
    #[host(default = None)]
    pub mime_type: Option<String>,
}

// `type` comes first, and pydantic field order is `model_dump()` key order.
#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "audio_content"),
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioContent {
    // One side-table entry for the whole list.
    #[host(annotation = "list[rtc.AudioFrame]")]
    pub frame: MediaRef,
    #[host(default = None)]
    pub transcript: Option<String>,
}

// One element of `ChatMessage.content`, in the reference's union order.
#[crate::host(record(module = "chat_items", transparent, extract, emit))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", serde(tag = "kind", content = "value", rename_all = "snake_case"))]
pub enum ChatContent {
    Image(ImageContent),
    Audio(AudioContent),
    Text(String),
}

// A field with no value contributes no key.
#[crate::host(record(module = "chat_items", typed_dict, extract, emit, model_dump))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetricsMetadata {
    #[host(default = None)]
    pub model_name: Option<String>,
    #[host(default = None)]
    pub model_provider: Option<String>,
}

// `total=False`: absent is distinct from null (and an absent list from an
// empty one), so every field is absent when `None`.
#[crate::host(record(module = "chat_items", typed_dict, extract, emit, model_dump))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsReport {
    #[host(default = None)]
    pub started_speaking_at: Option<f64>,
    #[host(default = None)]
    pub stopped_speaking_at: Option<f64>,

    /// Time taken to obtain the transcript after the end of the user's speech
    ///
    /// User `ChatMessage` only
    #[host(default = None)]
    pub transcription_delay: Option<f64>,
    /// Amount of time between the end of speech and the decision to end the user's turn
    ///
    /// User `ChatMessage` only
    #[host(default = None)]
    pub end_of_turn_delay: Option<f64>,
    /// Time taken to invoke the developer's `Agent.on_user_turn_completed` callback.
    ///
    /// User `ChatMessage` only
    #[host(default = None)]
    pub on_user_turn_completed_delay: Option<f64>,

    /// Time taken for the `llm_node` to return the first token
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub llm_node_ttft: Option<f64>,
    /// LLM output tokens per second for this turn, measured at the `llm_node` over the
    /// streaming window (first to last text chunk). Absent for a reply that arrived in a
    /// single chunk, which has no measurable rate
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub llm_node_tps: Option<f64>,
    /// Time from LLM generation start until the first sentence reached the TTS provider, as
    /// segmented by that TTS. Absent when no audio came from a LiveKit TTS this turn: no TTS,
    /// an interruption before the first frame, or a `tts_node` synthesizing audio on its own
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub llm_node_ttfs: Option<f64>,
    /// Time taken for the `tts_node` to return the first chunk of audio (after the first text token has been sent)
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub tts_node_ttfb: Option<f64>,
    /// Delay between forwarding the first audio frame and the `AudioOutput` reporting
    /// playback started. Near-zero for the default room output (self-reported when the frame
    /// is pushed to the track, so it doesn't account for network delivery to the client);
    /// meaningful when a remote avatar worker is in the chain and reports playback via
    /// the `lk.playback_started` RPC.
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub playback_latency: Option<f64>,
    /// Time from when the user finished speaking to when the agent began responding
    ///
    /// Assistant `ChatMessage` only
    #[host(default = None)]
    pub e2e_latency: Option<f64>,
    /// Provider-known request or response IDs associated with this turn.
    ///
    /// Assistant `ChatMessage` only. These IDs can be used to correlate a turn with
    /// provider-side logs.
    #[host(default = None)]
    pub provider_request_ids: Option<Vec<String>>,

    #[host(default = None)]
    pub llm_metadata: Option<MetricsMetadata>,
    #[host(default = None)]
    pub tts_metadata: Option<MetricsMetadata>,
    #[host(default = None)]
    pub stt_metadata: Option<MetricsMetadata>,
}

// The text properties keep their bodies in `_bindings` (`text_content`
// imports `strip_expr_markup` lazily: a cycle).
#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "message"),
    constants_after = "id",
    extract,
    emit,
    model_dump,
))]
#[python(
    property(
        name = "text_content",
        ret = "str | None",
        hook = "chat_message_text_content",
        doc = "Returns a string of all text content in the message, with LiveKit's\nexpressive ``<expr/>`` tags removed from assistant messages.\n\nMultiple text content items will be joined by a newline.\nUse :attr:`raw_text_content` for the exact model-facing content."
    ),
    property(
        name = "raw_text_content",
        ret = "str | None",
        hook = "chat_message_raw_text_content",
        doc = "Returns a string of all text content in the message, exactly as generated\n(assistant messages may contain expressive ``<expr/>`` tags).\n\nMultiple text content items will be joined by a newline."
    )
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    #[host(default_factory = "..utils.shortuuid(\"item_\")")]
    pub id: String,
    pub role: Role,
    #[host(dump_with = crate::llm::to_dict::content_dump)]
    pub content: Vec<ChatContent>,
    #[host(default = false)]
    pub interrupted: bool,
    #[host(default = None)]
    pub transcript_confidence: Option<f64>,
    // `None` is the reference's empty dict.
    #[host(default_factory = "dict")]
    pub extra: Extra,
    #[host(default_factory = "MetricsReport()")]
    pub metrics: MetricsReport,
    // Wall clock, not `Ts`.
    #[host(default_factory = "time.time")]
    pub created_at: SystemTime,
}

impl ChatMessage {
    /// Every text part joined by `"\n"`, `None` if there are none.
    pub fn raw_text_content(&self) -> Option<String> {
        let mut parts = self.content.iter().filter_map(|c| match c {
            ChatContent::Text(t) => Some(t.as_str()),
            _ => None,
        });
        let first = parts.next()?;
        let mut out = first.to_string();
        for part in parts {
            out.push('\n');
            out.push_str(part);
        }
        Some(out)
    }
}

fn text_message(parts: Vec<String>, assistant: bool) -> ChatMessage {
    ChatMessage::new(
        "",
        if assistant { Role::Assistant } else { Role::User },
        parts.into_iter().map(ChatContent::Text).collect(),
        std::time::UNIX_EPOCH,
    )
}

/// `ChatMessage.raw_text_content` over the message's text parts.
pub fn chat_message_raw_text_content(parts: Vec<String>) -> Option<String> {
    text_message(parts, false).raw_text_content()
}

impl ChatMessage {
    /// `id` and `created_at` are parameters: the pure tier can mint neither.
    pub fn new(
        id: impl Into<String>,
        role: Role,
        content: Vec<ChatContent>,
        created_at: SystemTime,
    ) -> Self {
        ChatMessage {
            id: id.into(),
            role,
            content,
            interrupted: false,
            transcript_confidence: None,
            extra: Extra::new(),
            metrics: MetricsReport::default(),
            created_at,
        }
    }

    pub fn text(
        id: impl Into<String>,
        role: Role,
        text: impl Into<String>,
        created_at: SystemTime,
    ) -> Self {
        ChatMessage::new(id, role, vec![ChatContent::Text(text.into())], created_at)
    }
}

#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "function_call"),
    constants_after = "id",
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    #[host(default_factory = "..utils.shortuuid(\"item_\")")]
    pub id: String,
    pub call_id: String,
    pub arguments: String,
    pub name: String,
    #[host(default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Extra data for this function call. Can include provider-specific data
    /// (e.g., extra["google"] for thought signatures).
    // Thought signatures actually travel as a separate `thought_signatures` argument
    // (`_provider_format/google.py:23`), so this stays JSON.
    #[host(default_factory = "dict")]
    pub extra: Extra,
    /// Optional group ID for parallel function calls. When multiple function calls
    /// should be grouped together (e.g., parallel tool calls from a single API response),
    /// set this to a shared value. If not set, falls back to using id for grouping.
    #[host(default = None)]
    pub group_id: Option<String>,
}

#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "function_call_output"),
    constants_after = "id",
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallOutput {
    #[host(default_factory = "..utils.shortuuid(\"item_\")")]
    pub id: String,
    // Defaults to `""` in the reference, not `None`.
    #[host(default = "")]
    pub name: String,
    pub call_id: String,
    pub output: String,
    pub is_error: bool,
    #[host(default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Whether the model should answer once it receives this output.
    ///
    /// VoiceSession uses it to decide whether to generate a follow-up reply.
    /// Realtime models can also use it to schedule their response.
    #[host(default = true)]
    pub reply_required: bool,
}

impl FunctionCallOutput {
    /// Defaults matching the model: `name = ""` and `reply_required = true`.
    pub fn new(
        id: impl Into<String>,
        call_id: impl Into<String>,
        output: impl Into<String>,
        is_error: bool,
        created_at: SystemTime,
    ) -> Self {
        FunctionCallOutput {
            id: id.into(),
            name: String::new(),
            call_id: call_id.into(),
            output: output.into(),
            is_error,
            created_at,
            reply_required: true,
        }
    }
}

// `...Item` to stay apart from `AgentHandoffEvent`. The model
// writes `id` then `type`, and field order is user-visible (`model_dump()` and `repr`).
#[crate::host(record(
    module = "chat_items",
    name = "AgentHandoff",
    pydantic,
    constant(type = "agent_handoff"),
    constants_after = "id",
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentHandoffItem {
    #[host(default_factory = "..utils.shortuuid(\"item_\")")]
    pub id: String,
    // `None` on the first handoff of a session.
    #[host(default = None)]
    pub old_agent_id: Option<String>,
    pub new_agent_id: String,
    #[host(default_factory = "time.time")]
    pub created_at: SystemTime,
}

#[crate::host(record(
    module = "chat_items",
    pydantic,
    constant(type = "agent_config_update"),
    constants_after = "id",
    extract,
    emit,
    model_dump,
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfigUpdate {
    #[host(default_factory = "..utils.shortuuid(\"item_\")")]
    pub id: String,
    #[host(default = None)]
    pub instructions: Option<String>,
    #[host(default = None)]
    pub tools_added: Option<Vec<String>>,
    #[host(default = None)]
    pub tools_removed: Option<Vec<String>>,
    #[host(default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Full tool definitions (in-memory only, not serialized).
    // The live callables, interned whole: `agent_activity.py:599` mutates the same list later.
    #[host(private, default_factory = "list", annotation = "list[..llm.tool_context.Tool]")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub tools: Option<HostRef>,
}

// Not boxed: boxing `ChatMessage` would cost an allocation per message on the warm read path.
#[allow(clippy::large_enum_variant)]
#[crate::host(record(module = "chat_items", transparent, discriminator = "type", extract, emit))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", serde(tag = "kind", content = "value", rename_all = "snake_case"))]
pub enum ChatItem {
    Message(ChatMessage),
    FunctionCall(FunctionCall),
    FunctionCallOutput(FunctionCallOutput),
    AgentHandoff(AgentHandoffItem),
    AgentConfigUpdate(AgentConfigUpdate),
}

impl ChatItem {
    pub fn id(&self) -> &str {
        match self {
            ChatItem::Message(m) => &m.id,
            ChatItem::FunctionCall(f) => &f.id,
            ChatItem::FunctionCallOutput(f) => &f.id,
            ChatItem::AgentHandoff(h) => &h.id,
            ChatItem::AgentConfigUpdate(c) => &c.id,
        }
    }

    /// Wall clock, not monotonic: items minted on both sides of the boundary are ordered against
    /// each other by this. Mint from [`Clock::wall`](crate::Clock::wall), strictly increasing.
    pub fn created_at(&self) -> SystemTime {
        match self {
            ChatItem::Message(m) => m.created_at,
            ChatItem::FunctionCall(f) => f.created_at,
            ChatItem::FunctionCallOutput(f) => f.created_at,
            ChatItem::AgentHandoff(h) => h.created_at,
            ChatItem::AgentConfigUpdate(c) => c.created_at,
        }
    }

    pub fn kind(&self) -> ItemKind {
        match self {
            ChatItem::Message(_) => ItemKind::Message,
            ChatItem::FunctionCall(_) => ItemKind::FunctionCall,
            ChatItem::FunctionCallOutput(_) => ItemKind::FunctionCallOutput,
            ChatItem::AgentHandoff(_) => ItemKind::AgentHandoff,
            ChatItem::AgentConfigUpdate(_) => ItemKind::AgentConfigUpdate,
        }
    }

    /// A system or developer message (`"developer"` included: #4845).
    pub fn is_instruction(&self) -> bool {
        matches!(self, ChatItem::Message(m) if matches!(m.role, Role::System | Role::Developer))
    }

    /// The `name` of a function call or its output; `None` for every other kind.
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            ChatItem::FunctionCall(f) => Some(&f.name),
            ChatItem::FunctionCallOutput(f) => Some(&f.name),
            _ => None,
        }
    }

    pub fn as_message(&self) -> Option<&ChatMessage> {
        match self {
            ChatItem::Message(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_message_mut(&mut self) -> Option<&mut ChatMessage> {
        match self {
            ChatItem::Message(m) => Some(m),
            _ => None,
        }
    }
}

macro_rules! chat_item_from {
    ($($ty:ty => $variant:ident),* $(,)?) => {$(
        impl From<$ty> for ChatItem {
            fn from(v: $ty) -> Self {
                ChatItem::$variant(v)
            }
        }
    )*};
}

chat_item_from!(
    ChatMessage => Message,
    FunctionCall => FunctionCall,
    FunctionCallOutput => FunctionCallOutput,
    AgentHandoffItem => AgentHandoff,
    AgentConfigUpdate => AgentConfigUpdate,
);

mod plans;
pub use plans::{ChatContextPlans, ChatFactsError, HostFacts};

/// A conversation, ordered by [`ChatItem::created_at`]. Nothing assumes `items` is sorted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatContext {
    pub items: Vec<ChatItem>,
}

/// Filters for conversation selection (`ChatContext.copy`).
#[derive(Debug, Clone, Default)]
pub struct CopyOptions {
    pub exclude_function_call: bool,
    pub exclude_instructions: bool,
    /// Drops messages whose `content` list is empty -- not messages whose text is empty.
    pub exclude_empty_message: bool,
    pub exclude_handoff: bool,
    pub exclude_config_update: bool,
    /// `None` skips the filter; `Some(&[])` drops every call.
    pub tools: Option<Vec<String>>,
}

/// Smaller than [`CopyOptions`] on purpose: merging an `AgentTask` back in must keep its handoff
/// rows (#2731).
#[derive(Debug, Clone, Default)]
pub struct MergeOptions {
    pub exclude_function_call: bool,
    pub exclude_instructions: bool,
    pub exclude_config_update: bool,
}

impl ChatContext {
    pub fn new(items: Vec<ChatItem>) -> Self {
        ChatContext { items }
    }

    pub fn empty() -> Self {
        ChatContext { items: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn messages(&self) -> impl Iterator<Item = &ChatMessage> {
        self.items.iter().filter_map(ChatItem::as_message)
    }

    pub fn get_by_id(&self, id: &str) -> Option<&ChatItem> {
        self.items.iter().find(|it| it.id() == id)
    }

    pub fn index_by_id(&self, id: &str) -> Option<usize> {
        self.items.iter().position(|it| it.id() == id)
    }

    /// Append, without consulting `created_at`: `add_message` with no `created_at`
    /// (`ChatContext.add_message`). Not the same as [`insert`](Self::insert) on an unsorted list.
    pub fn push(&mut self, item: impl Into<ChatItem>) {
        self.items.push(item.into());
    }

    /// `ChatContext.insert`.
    pub fn insert(&mut self, item: impl Into<ChatItem>) {
        let item = item.into();
        let idx = self.find_insertion_index(item.created_at());
        self.items.insert(idx, item);
    }

    /// Insert several, each placed against the list as it stands after the previous one.
    pub fn insert_all(&mut self, items: impl IntoIterator<Item = ChatItem>) {
        for item in items {
            self.insert(item);
        }
    }

    /// After the last item with `created_at <=` it (`ChatContext.find_insertion_index`). A reverse scan, not a
    /// bisect: the list may be unsorted, and ties keep arrival order.
    pub fn find_insertion_index(&self, created_at: SystemTime) -> usize {
        for (i, item) in self.items.iter().enumerate().rev() {
            if item.created_at() <= created_at {
                return i + 1;
            }
        }
        0
    }
}

impl From<Vec<ChatItem>> for ChatContext {
    fn from(items: Vec<ChatItem>) -> Self {
        ChatContext { items }
    }
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "chat_diff", path = "livekit.agents.llm.utils", dataclass))]
pub struct DiffOps {
    pub to_remove: Vec<String>,
    pub to_create: Vec<(Option<String>, String)>,
    pub to_update: Vec<(Option<String>, String)>,
}

#[cfg(test)]
mod item_tests {
    //! The chat items on their own: text content, instruction and tool-name classification, stamps.

    use crate::{Clock, VirtualClock};

    use super::*;
    use crate::testkit::{t, text};

    fn assistant(content: Vec<ChatContent>) -> ChatMessage {
        ChatMessage::new("item_1", Role::Assistant, content, t(1))
    }

    // Catches: joining the text parts with anything but a newline, or `Some("")` for a message
    // with no text parts (diffs would report it updated each time).
    #[test]
    fn raw_text_content_joins_parts_with_a_newline() {
        let m = assistant(vec![text("one"), text("two"), text("three")]);
        assert_eq!(m.raw_text_content().as_deref(), Some("one\ntwo\nthree"));
        assert_eq!(assistant(vec![]).raw_text_content(), None);
        assert_eq!(assistant(vec![text("")]).raw_text_content().as_deref(), Some(""));
    }

    // Catches: the #4845 bug class -- the instruction test written as `role == System`.
    #[test]
    fn is_instruction_covers_developer_as_well_as_system() {
        for role in [Role::System, Role::Developer] {
            assert!(ChatItem::from(ChatMessage::text("i", role, "x", t(0))).is_instruction());
        }
        for role in [Role::User, Role::Assistant] {
            assert!(!ChatItem::from(ChatMessage::text("i", role, "x", t(0))).is_instruction());
        }
    }

    // Catches: stamping `created_at` from the monotonic clock, or a `wall()` that repeats.
    #[test]
    fn two_items_minted_in_one_tick_get_distinct_ordered_stamps() {
        let clock = VirtualClock::new();
        let first = ChatMessage::text("item_a", Role::User, "hi", clock.wall());
        let second = ChatMessage::text("item_b", Role::Assistant, "hello", clock.wall());
        assert!(
            second.created_at > first.created_at,
            "{:?} !> {:?}",
            second.created_at,
            first.created_at
        );
    }
}
