//! `llm/chat_context.py` as plain data. The reference's `copy()` is shallow and the pipeline relies
//! on the aliasing; Rust clones, so every mutation through an alias is a `DiffOp::Replace`.

use std::time::SystemTime;

/// JSON, not an opaque ref: the reference branches on it (`chat_context.py:814`).
pub type Extra = serde_json::Map<String, serde_json::Value>;

use crate::{HostRef, MediaRef};
use lk_agents_host_derive::Host;

/// `ChatRole` (`chat_context.py:222`). Declaration order matches the reference's; the wire strings
/// are the exact `Literal`s, the `to_core`/`from_core` contract.
#[crate::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
}

/// `speech_handle.py:19`. Defaults to `Audio`, like `SpeechHandle`. The wire strings are public
/// API: users compare `details.modality == "audio"`.
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral { kind: "modality", value: s.to_owned() }))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Modality {
    Text,
    #[default]
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

/// `ImageContent.inference_detail` (`chat_context.py:203`). OpenAI-only today. An unknown value is
/// refused, never defaulted: it reaches OpenAI's request body verbatim.
#[crate::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceDetail {
    #[default]
    Auto,
    High,
    Low,
}

/// A URL passes through as-is; a frame is encoded first, and the resize fields apply only to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    Url(String),
    Frame(MediaRef),
}

/// The host's own frame back out of the side table, not a copy: callers compare by identity.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for ImageSource {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        match self {
            ImageSource::Url(url) => Ok(pyo3::types::PyString::new(py, &url).into_any()),
            ImageSource::Frame(r) => r.into_pyobject(py),
        }
    }
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
/// image_bytes = encode(
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
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "ImageContent",
    tag = "type",
    tag_value = "image_content",
    tag_after = "id",
    host_field(name = "_cache", ty = "dict[Any, Any]", private = "yes", default_factory = "dict")
)]
#[binding(emit, model_dump)]
pub struct ImageContent {
    /// Unique identifier for the image
    #[python(default_factory = "lambda: utils.shortuuid(\"img_\")", import = "..:utils")]
    pub id: String,
    /// Either a string URL or a VideoFrame object
    #[python(ty = "str | rtc.VideoFrame", import = "livekit:rtc")]
    pub image: ImageSource,
    /// Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    #[python(ty = "int | None", default = "None")]
    pub inference_width: Option<u32>,
    /// Resizing parameter for rtc.VideoFrame inputs (ignored for URL images)
    #[python(ty = "int | None", default = "None")]
    pub inference_height: Option<u32>,
    /// Detail parameter for LLM provider, if supported.
    ///
    /// Currently only supported by OpenAI (see https://platform.openai.com/docs/guides/vision?lang=node#low-or-high-fidelity-image-understanding)
    #[python(ty = "Literal[\"auto\", \"high\", \"low\"]", default = "\"auto\"")]
    pub inference_detail: InferenceDetail,
    /// MIME type of the image
    #[python(ty = "str | None", default = "None")]
    pub mime_type: Option<String>,
}

// `chat_context.py:216`. No `tag_after`: `type` comes first, and pydantic field order is
// `model_dump()` key order.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "AudioContent",
    tag = "type",
    tag_value = "audio_content"
)]
#[binding(emit, model_dump)]
pub struct AudioContent {
    // One side-table entry for the whole `list[rtc.AudioFrame]`; the reference field is `frame`.
    #[host(rename = "frame")]
    #[python(ty = "list[rtc.AudioFrame]", import = "livekit:rtc")]
    pub frames: MediaRef,
    #[python(default = "None")]
    pub transcript: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatContent {
    Text(String),
    Image(ImageContent),
    Audio(AudioContent),
}

/// One element of `ChatMessage.content`: `str | ImageContent | AudioContent`, built by the host one
/// constructor per arm so Rust never reads attributes or dispatches on `.type`.
#[crate::host(object(
    cell = "value",
    pid = false,
    pyclass(eq, from_py_object),
    derive(Clone, PartialEq),
    stub = "@property",
    stub = "def value(self) -> Any: ..."
))]
impl ChatContent {
    /// A plain `str` content part.
    #[host(method)]
    pub fn text(text: String) -> Self {
        ChatContent::Text(text)
    }

    /// `ImageContent` whose `image` is a URL or a `data:` URI, passed to the provider untouched.
    #[host(method(kw_only, defaults(inference_detail = "auto")))]
    pub fn image_url(
        id: String,
        url: String,
        inference_width: Option<u32>,
        inference_height: Option<u32>,
        inference_detail: InferenceDetail,
        mime_type: Option<String>,
    ) -> Self {
        let image = ImageSource::Url(url);
        ChatContent::Image(ImageContent { id, image, inference_width, inference_height, inference_detail, mime_type })
    }

    /// `ImageContent` whose `image` is an `rtc.VideoFrame`; only its side-table key crosses.
    #[host(method(kw_only, defaults(inference_detail = "auto")))]
    pub fn image_frame(
        id: String,
        frame: MediaRef,
        inference_width: Option<u32>,
        inference_height: Option<u32>,
        inference_detail: InferenceDetail,
        mime_type: Option<String>,
    ) -> Self {
        let image = ImageSource::Frame(frame);
        ChatContent::Image(ImageContent { id, image, inference_width, inference_height, inference_detail, mime_type })
    }

    /// `AudioContent`: one side-table entry for the whole `list[rtc.AudioFrame]`.
    #[host(method(kw_only))]
    pub fn audio(frames: MediaRef, transcript: Option<String>) -> Self {
        ChatContent::Audio(AudioContent { frames, transcript })
    }

    /// `"text"`, or the pydantic model's own `type` literal, so the host switches on one table.
    #[host(getter)]
    pub fn kind(&self) -> &'static str {
        match self {
            ChatContent::Text(_) => "text",
            ChatContent::Image(_) => "image_content",
            ChatContent::Audio(_) => "audio_content",
        }
    }
}

// `MetricsMetadata` (`chat_context.py:226`). A field with no value contributes no key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    name = "MetricsMetadata"
)]
#[binding(emit, extract, from_items, model_dump)]
pub struct MetricsMetadata {
    #[binding(skip_if_none)]
    #[python(ty = "str")]
    pub model_name: Option<String>,
    #[binding(skip_if_none)]
    #[python(ty = "str")]
    pub model_provider: Option<String>,
}

// `chat_context.py:231`, `total=False`: absent is distinct from null (and an absent list from an
// empty one), so every field is `skip_if_none`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    name = "MetricsReport"
)]
#[binding(emit, extract, from_items, skip_if_none, model_dump)]
pub struct MetricsReport {
    #[python(ty = "float")]
    pub started_speaking_at: Option<f64>,
    #[python(ty = "float")]
    pub stopped_speaking_at: Option<f64>,

    /// Time taken to obtain the transcript after the end of the user's speech
    ///
    /// User `ChatMessage` only
    #[python(ty = "float")]
    pub transcription_delay: Option<f64>,
    /// Amount of time between the end of speech and the decision to end the user's turn
    ///
    /// User `ChatMessage` only
    #[python(ty = "float")]
    pub end_of_turn_delay: Option<f64>,
    /// Time taken to invoke the developer's `Agent.on_user_turn_completed` callback.
    ///
    /// User `ChatMessage` only
    #[python(ty = "float")]
    pub on_user_turn_completed_delay: Option<f64>,

    /// Time taken for the `llm_node` to return the first token
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub llm_node_ttft: Option<f64>,
    /// LLM output tokens per second for this turn, measured at the `llm_node` over the
    /// streaming window (first to last text chunk). Absent for a reply that arrived in a
    /// single chunk, which has no measurable rate
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub llm_node_tps: Option<f64>,
    /// Time from LLM generation start until the first sentence reached the TTS provider, as
    /// segmented by that TTS. Absent when no audio came from a LiveKit TTS this turn: no TTS,
    /// an interruption before the first frame, or a `tts_node` synthesizing audio on its own
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub llm_node_ttfs: Option<f64>,
    /// Time taken for the `tts_node` to return the first chunk of audio (after the first text token has been sent)
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub tts_node_ttfb: Option<f64>,
    /// Delay between forwarding the first audio frame and the `AudioOutput` reporting
    /// playback started. Near-zero for the default room output (self-reported when the frame
    /// is pushed to the track, so it doesn't account for network delivery to the client);
    /// meaningful when a remote avatar worker is in the chain and reports playback via
    /// the `lk.playback_started` RPC.
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub playback_latency: Option<f64>,
    /// Time from when the user finished speaking to when the agent began responding
    ///
    /// Assistant `ChatMessage` only
    #[python(ty = "float")]
    pub e2e_latency: Option<f64>,
    /// Provider-known request or response IDs associated with this turn.
    ///
    /// Assistant `ChatMessage` only. These IDs can be used to correlate a turn with
    /// provider-side logs.
    #[python(ty = "list[str]")]
    pub provider_request_ids: Option<Vec<String>>,

    #[python(ty = "MetricsMetadata")]
    pub llm_metadata: Option<MetricsMetadata>,
    #[python(ty = "MetricsMetadata")]
    pub tts_metadata: Option<MetricsMetadata>,
    #[python(ty = "MetricsMetadata")]
    pub stt_metadata: Option<MetricsMetadata>,
}

// `chat_context.py:310`. The deprecated `hash` has no reader: a `host_field`. The text properties
// keep their bodies in `_bindings` (`text_content` imports `strip_expr_markup` lazily: a cycle).
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "ChatMessage",
    tag = "type",
    tag_value = "message",
    tag_after = "id",
    host_field(
        name = "hash",
        ty = "bytes | None",
        default = "None",
        deprecated_msg = "hash is deprecated"
    ),
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
#[binding(emit, model_dump)]
pub struct ChatMessage {
    #[python(default_factory = "lambda: utils.shortuuid(\"item_\")", import = "..:utils")]
    pub id: String,
    // `type_import`: `chat_context.py` imports this generated module, so a runtime import is a cycle.
    #[python(ty = "ChatRole", type_import = "..llm.chat_context:ChatRole")]
    pub role: Role,
    #[python(ty = "list[ChatContent]", type_import = "..llm.chat_context:ChatContent")]
    #[binding(dump_with = crate::llm::to_dict::content_dump)]
    pub content: Vec<ChatContent>,
    #[python(default = "False")]
    pub interrupted: bool,
    #[python(default = "None")]
    pub transcript_confidence: Option<f64>,
    // `None` is the reference's empty dict.
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub extra: Extra,
    #[python(ty = "MetricsReport", default_factory = "lambda: MetricsReport()")]
    pub metrics: MetricsReport,
    // Wall clock, not `Ts`.
    #[python(ty = "float", default_factory = "time.time")]
    pub created_at: SystemTime,
}

impl ChatMessage {
    /// Every text part joined by `"\n"`, `None` if there are none (`chat_context.py:340`).
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

    /// Display text strips expressive markup only from assistant messages.
    pub fn text_content(&self) -> Option<String> {
        self.raw_text_content().map(|text| {
            if self.role == Role::Assistant { crate::tts::markup::strip_expr_markup(&text) } else { text }
        })
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
#[crate::host(export)]
pub fn chat_message_raw_text_content(parts: Vec<String>) -> Option<String> {
    text_message(parts, false).raw_text_content()
}

/// `ChatMessage.text_content` over the message's text parts: an assistant's loses its markup.
#[crate::host(export)]
pub fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
    text_message(parts, assistant).text_content()
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

// `chat_context.py:361`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "FunctionCall",
    tag = "type",
    tag_value = "function_call",
    tag_after = "id"
)]
#[binding(emit, model_dump)]
pub struct FunctionCall {
    #[python(default_factory = "lambda: utils.shortuuid(\"item_\")", import = "..:utils")]
    pub id: String,
    pub call_id: String,
    pub arguments: String,
    pub name: String,
    #[python(ty = "float", default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Extra data for this function call. Can include provider-specific data
    /// (e.g., extra["google"] for thought signatures).
    // Thought signatures actually travel as a separate `thought_signatures` argument
    // (`_provider_format/google.py:23`), so this stays JSON.
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub extra: Extra,
    /// Optional group ID for parallel function calls. When multiple function calls
    /// should be grouped together (e.g., parallel tool calls from a single API response),
    /// set this to a shared value. If not set, falls back to using id for grouping.
    #[python(default = "None")]
    pub group_id: Option<String>,
}

/// Anything else is a plugin bug that would otherwise become a tool call that never runs.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for FunctionCall {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        match obj.extract::<ChatItem>()? {
            ChatItem::FunctionCall(call) => Ok(call),
            _ => Err(pyo3::exceptions::PyValueError::new_err("`call` is not a FunctionCall")),
        }
    }
}

// `chat_context.py:374`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "FunctionCallOutput",
    tag = "type",
    tag_value = "function_call_output",
    tag_after = "id"
)]
#[binding(emit, model_dump)]
pub struct FunctionCallOutput {
    #[python(default_factory = "lambda: utils.shortuuid(\"item_\")", import = "..:utils")]
    pub id: String,
    // Defaults to `""` in the reference, not `None`.
    #[python(default = "\"\"")]
    pub name: String,
    pub call_id: String,
    pub output: String,
    pub is_error: bool,
    #[python(ty = "float", default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Whether the model should answer once it receives this output (read by realtime models only).
    #[python(default = "True")]
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

// `chat_context.py:389`; `...Item` to stay apart from `AgentHandoffEvent`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
// `tag_after = "id"`: `chat_context.py` writes `id` then `type`, and field order is user-visible
// (`model_dump()` and `repr`).
#[python(
    pydantic,
    name = "AgentHandoff",
    tag = "type",
    tag_value = "agent_handoff",
    tag_after = "id"
)]
#[binding(emit, model_dump)]
pub struct AgentHandoffItem {
    #[python(default_factory = "lambda: utils.shortuuid(\"item_\")", import = "..:utils")]
    pub id: String,
    // `None` on the first handoff of a session.
    #[python(default = "None")]
    pub old_agent_id: Option<String>,
    pub new_agent_id: String,
    #[python(ty = "float", default_factory = "time.time")]
    pub created_at: SystemTime,
}

// `chat_context.py:397`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "AgentConfigUpdate",
    tag = "type",
    tag_value = "agent_config_update",
    tag_after = "id"
)]
#[binding(emit, model_dump)]
pub struct AgentConfigUpdate {
    #[python(default_factory = "lambda: utils.shortuuid(\"item_\")", import = "..:utils")]
    pub id: String,
    pub instructions: Option<String>,
    pub tools_added: Option<Vec<String>>,
    pub tools_removed: Option<Vec<String>>,
    #[python(ty = "float", default_factory = "time.time")]
    pub created_at: SystemTime,
    /// Full tool definitions (in-memory only, not serialized).
    // `_tools: list[Tool]`, a `PrivateAttr` of live callables (`chat_context.py:405`).
    #[host(rename = "_tools")]
    #[python(
        private,
        ty = "list[Tool]",
        type_import = ".tool_context:Tool",
        default_factory = "list"
    )]
    #[python(required)]
    pub tools: Option<HostRef>,
}

/// Not boxed: boxing `ChatMessage` would cost an allocation per message on the warm `to_core` path.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatItem {
    Message(ChatMessage),
    FunctionCall(FunctionCall),
    FunctionCallOutput(FunctionCallOutput),
    AgentHandoff(AgentHandoffItem),
    AgentConfigUpdate(AgentConfigUpdate),
}

/// One row of conversation history: the discriminated union of `chat_context.py:407`, built by the
/// host one constructor per variant. `created_at` is a wall clock; ids are always host-supplied.
#[crate::host(object(
    cell = "value",
    pid = false,
    pyclass(eq, from_py_object),
    derive(Clone, PartialEq),
    stub = "def fields(self) -> dict[str, Any]: ..."
))]
impl ChatItem {
    #[allow(clippy::too_many_arguments)]
    #[host(method(kw_only, defaults(interrupted = false)))]
    pub fn message(
        id: String,
        role: Role,
        content: Vec<ChatContent>,
        created_at: SystemTime,
        interrupted: bool,
        transcript_confidence: Option<f64>,
        extra: Option<Extra>,
        metrics: Option<MetricsReport>,
    ) -> Self {
        ChatItem::Message(ChatMessage {
            id,
            role,
            content,
            interrupted,
            transcript_confidence,
            extra: extra.unwrap_or_default(),
            metrics: metrics.unwrap_or_default(),
            created_at,
        })
    }

    #[host(method(kw_only))]
    pub fn function_call(
        id: String,
        call_id: String,
        arguments: String,
        name: String,
        created_at: SystemTime,
        extra: Option<Extra>,
        group_id: Option<String>,
    ) -> Self {
        let extra = extra.unwrap_or_default();
        ChatItem::FunctionCall(FunctionCall { id, call_id, arguments, name, created_at, extra, group_id })
    }

    #[allow(clippy::too_many_arguments)]
    #[host(method(kw_only, defaults(reply_required = true)))]
    pub fn function_call_output(
        id: String,
        call_id: String,
        output: String,
        is_error: bool,
        created_at: SystemTime,
        name: Option<String>,
        reply_required: bool,
    ) -> Self {
        let name = name.unwrap_or_default();
        ChatItem::FunctionCallOutput(FunctionCallOutput { id, name, call_id, output, is_error, created_at, reply_required })
    }

    #[host(method(kw_only))]
    pub fn agent_handoff(id: String, new_agent_id: String, created_at: SystemTime, old_agent_id: Option<String>) -> Self {
        ChatItem::AgentHandoff(AgentHandoffItem { id, old_agent_id, new_agent_id, created_at })
    }

    /// `tools` is interned whole, even when empty: `agent_activity.py:599` mutates the same list
    /// object in place later.
    #[host(method(kw_only))]
    pub fn agent_config_update(
        id: String,
        created_at: SystemTime,
        instructions: Option<String>,
        tools_added: Option<Vec<String>>,
        tools_removed: Option<Vec<String>>,
        tools: Option<HostRef>,
    ) -> Self {
        ChatItem::AgentConfigUpdate(AgentConfigUpdate { id, instructions, tools_added, tools_removed, created_at, tools })
    }

    /// The pydantic `type` literal, so `from_core` can index a `{type: model}` table.
    #[host(getter(name = "type"))]
    pub fn type_literal(&self) -> &'static str {
        self.kind().as_str()
    }

    /// `AgentConfigUpdate._tools`, the original list, or `None`. Not in `fields` because a
    /// `PrivateAttr` is not a constructor kwarg.
    #[host(getter(name = "tools"))]
    pub fn host_tools(&self) -> Option<&HostRef> {
        match self {
            ChatItem::AgentConfigUpdate(c) => c.tools.as_ref(),
            _ => None,
        }
    }

    #[host(getter)]
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
    #[host(getter)]
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

use std::collections::HashSet;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "chat_items", path = "livekit.agents.llm.chat_context"))]
const _: () = ();

/// A conversation, ordered by [`ChatItem::created_at`]. Nothing assumes `items` is sorted.
#[crate::class(name = "TurnContext")]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatContext {
    pub items: Vec<ChatItem>,
}

/// Filters for [`ChatContext::copy`] (`chat_context.py:495`).
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
    /// (`chat_context.py:465`). Not the same as [`insert`](Self::insert) on an unsorted list.
    pub fn push(&mut self, item: impl Into<ChatItem>) {
        self.items.push(item.into());
    }

    /// `chat_context.py:472`.
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

    /// After the last item with `created_at <=` it (`chat_context.py:749`). A reverse scan, not a
    /// bisect: the list may be unsorted, and ties keep arrival order.
    pub fn find_insertion_index(&self, created_at: SystemTime) -> usize {
        for (i, item) in self.items.iter().enumerate().rev() {
            if item.created_at() <= created_at {
                return i + 1;
            }
        }
        0
    }

    /// A filtered copy (`chat_context.py:495`); deep, where the reference shares items.
    pub fn copy(&self, opts: &CopyOptions) -> ChatContext {
        let plan = copy_plan(&self.facts(), opts);
        ChatContext { items: plan.into_iter().map(|i| self.items[i].clone()).collect() }
    }

    fn facts(&self) -> Vec<ItemFacts<'_>> {
        self.items.iter().map(ItemFacts::of).collect()
    }
}

impl From<Vec<ChatItem>> for ChatContext {
    fn from(items: Vec<ChatItem>) -> Self {
        ChatContext { items }
    }
}

/// What a host that owns its items reports; the rules over them stay here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemFacts<'a> {
    pub kind: ItemKind,
    pub role: Option<Role>,
    /// `len(item.content)`: the reference tests the list, so one empty string survives.
    pub content_len: usize,
    pub tool_name: Option<&'a str>,
    pub id: &'a str,
    pub created_at: SystemTime,
}

impl<'a> ItemFacts<'a> {
    pub fn of(item: &'a ChatItem) -> Self {
        ItemFacts {
            kind: item.kind(),
            role: item.as_message().map(|m| m.role),
            content_len: item.as_message().map_or(0, |m| m.content.len()),
            tool_name: item.tool_name(),
            id: item.id(),
            created_at: item.created_at(),
        }
    }

    fn is_instruction(&self) -> bool {
        self.kind == ItemKind::Message && matches!(self.role, Some(Role::System | Role::Developer))
    }

    fn is_function_call_pair(&self) -> bool {
        self.kind.is_function_call_pair()
    }
}

pub fn copy_excludes_facts(f: &ItemFacts<'_>, opts: &CopyOptions) -> bool {
    let is_message = f.kind == ItemKind::Message;

    if opts.exclude_function_call && f.is_function_call_pair() {
        return true;
    }
    // `role in ("system", "developer")`, and a *message* only -- the other kinds have no role.
    if opts.exclude_instructions && f.is_instruction() {
        return true;
    }
    // `not item.content` -- an empty content *list*. A message holding one empty string survives.
    if opts.exclude_empty_message && is_message && f.content_len == 0 {
        return true;
    }
    if opts.exclude_handoff && f.kind == ItemKind::AgentHandoff {
        return true;
    }
    if opts.exclude_config_update && f.kind == ItemKind::AgentConfigUpdate {
        return true;
    }
    if let Some(valid) = &opts.tools
        && let Some(name) = f.tool_name
        && !valid.iter().any(|t| t == name)
    {
        return true;
    }
    false
}

fn merge_excludes_facts(f: &ItemFacts<'_>, opts: &MergeOptions) -> bool {
    (opts.exclude_function_call && f.is_function_call_pair())
        || (opts.exclude_instructions && f.is_instruction())
        || (opts.exclude_config_update && f.kind == ItemKind::AgentConfigUpdate)
}

// The operations as index plans over the facts, applied by whoever owns the items. Positions, not
// ids, so duplicate ids pick the right object.

pub fn copy_plan(facts: &[ItemFacts<'_>], opts: &CopyOptions) -> Vec<usize> {
    (0..facts.len()).filter(|&i| !copy_excludes_facts(&facts[i], opts)).collect()
}

/// `None` when under the limit. Leading function calls are dropped; the first instruction message
/// comes back unless the window holds its id (#4845). Sliced like `self._items[-max_items:]`: `0`
/// keeps everything, a negative `n` drops the first `|n|`.
pub fn truncate_plan(facts: &[ItemFacts<'_>], max_items: i64) -> Option<Vec<usize>> {
    let n = facts.len();
    if (n as i64) <= max_items {
        return None;
    }
    let instructions = facts.iter().position(ItemFacts::is_instruction);
    let start = if max_items > 0 { n - max_items as usize } else { max_items.unsigned_abs().min(n as u64) as usize };
    let mut kept: Vec<usize> = (start..n).collect();
    let lead = kept.iter().take_while(|&&i| facts[i].is_function_call_pair()).count();
    kept.drain(..lead);
    if let Some(ins) = instructions
        && !kept.iter().any(|&i| facts[i].id == facts[ins].id)
    {
        kept.insert(0, ins);
    }
    Some(kept)
}

/// `merge`: `(index, j)` pairs -- insert `theirs[j]` at `index` of the growing list, in order.
/// De-duplicated by id; ties go after.
pub fn merge_plan(
    mine: &[ItemFacts<'_>],
    theirs: &[ItemFacts<'_>],
    opts: &MergeOptions,
) -> Vec<(usize, usize)> {
    let mut existing: HashSet<&str> = mine.iter().map(|f| f.id).collect();
    let mut stamps: Vec<SystemTime> = mine.iter().map(|f| f.created_at).collect();
    let mut plan = Vec::new();
    for (j, f) in theirs.iter().enumerate() {
        if merge_excludes_facts(f, opts) || !existing.insert(f.id) {
            continue;
        }
        let idx = stamps.iter().rposition(|&at| at <= f.created_at).map_or(0, |i| i + 1);
        stamps.insert(idx, f.created_at);
        plan.push((idx, j));
    }
    plan
}

/// `(type, id, role, len(content), name, created_at)`.
pub type HostFacts = (String, String, Option<String>, usize, Option<String>, f64);

fn host_facts(facts: &[HostFacts]) -> Result<Vec<ItemFacts<'_>>, String> {
    facts
        .iter()
        .map(|(kind, id, role, content_len, name, created_at)| {
            Ok(ItemFacts {
                kind: kind.parse()?,
                role: role.as_deref().map(str::parse).transpose()?,
                content_len: *content_len,
                tool_name: name.as_deref(),
                id,
                created_at: crate::wall_from_secs(*created_at)
                    .ok_or_else(|| format!("created_at={created_at} is not a usable wall clock"))?,
            })
        })
        .collect()
}

/// `ChatContext.copy` as an index plan over the host's own items.
#[allow(clippy::too_many_arguments)]
#[crate::host(export(pid = false, name = "chat_ctx_copy_plan", kw_only = "exclude_function_call", defaults(exclude_function_call = false, exclude_instructions = false, exclude_empty_message = false, exclude_handoff = false, exclude_config_update = false)))]
pub fn copy_plan_of(
    facts: Vec<HostFacts>,
    exclude_function_call: bool,
    exclude_instructions: bool,
    exclude_empty_message: bool,
    exclude_handoff: bool,
    exclude_config_update: bool,
    tools: Option<Vec<String>>,
) -> Result<Vec<usize>, String> {
    let opts = CopyOptions { exclude_function_call, exclude_instructions, exclude_empty_message, exclude_handoff, exclude_config_update, tools };
    Ok(copy_plan(&host_facts(&facts)?, &opts))
}

/// `ChatContext.truncate` as an index plan; `None` when under the limit.
#[crate::host(export(pid = false, name = "chat_ctx_truncate_plan"))]
pub fn truncate_plan_of(facts: Vec<HostFacts>, max_items: i64) -> Result<Option<Vec<usize>>, String> {
    Ok(truncate_plan(&host_facts(&facts)?, max_items))
}

/// `ChatContext.merge` as `(index, j)` insertions.
#[crate::host(export(pid = false, name = "chat_ctx_merge_plan", kw_only = "exclude_function_call", defaults(exclude_function_call = false, exclude_instructions = false, exclude_config_update = false)))]
pub fn merge_plan_of(
    mine: Vec<HostFacts>,
    theirs: Vec<HostFacts>,
    exclude_function_call: bool,
    exclude_instructions: bool,
    exclude_config_update: bool,
) -> Result<Vec<(usize, usize)>, String> {
    let opts = MergeOptions { exclude_function_call, exclude_instructions, exclude_config_update };
    Ok(merge_plan(&host_facts(&mine)?, &host_facts(&theirs)?, &opts))
}

// The two methods no declaration expresses: each builds the host's view of a variant.
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyChatContent {
    /// The `str` for a text part; the model's constructor kwargs for the other two, with media
    /// resolved back to the original host object.
    #[getter]
    fn value<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::IntoPyObjectExt as _;
        match &self.inner {
            ChatContent::Text(t) => t.into_bound_py_any(py),
            ChatContent::Image(img) => img.into_bound_py_any(py),
            ChatContent::Audio(audio) => audio.into_bound_py_any(py),
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyChatItem {
    /// Every field of this variant keyed by its pydantic name (without `type`), so `from_core` is
    /// `Model(**item.fields())` once `content` is mapped.
    fn fields<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::IntoPyObject as _;
        match &self.inner {
            ChatItem::Message(m) => m.into_pyobject(py),
            ChatItem::FunctionCall(f) => f.into_pyobject(py),
            ChatItem::FunctionCallOutput(f) => f.into_pyobject(py),
            ChatItem::AgentHandoff(h) => h.into_pyobject(py),
            ChatItem::AgentConfigUpdate(c) => c.into_pyobject(py),
        }
    }
}

#[cfg(test)]
mod item_tests;
#[cfg(test)]
mod context_tests;
