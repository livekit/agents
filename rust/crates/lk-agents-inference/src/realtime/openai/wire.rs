//! The OpenAI Realtime protocol: every frame this crate sends and every message it understands.
//! Stateless: builders return `serde_json::Value`s; [`decode_at`] validates an [`Incoming`].
//!
//! Explicit nulls are load-bearing: the reference dumps with `exclude_unset=True`, so a `None`
//! passed to the constructor is on the wire as `null` ("turn it off"), while an absent key means
//! "leave it". Azure and LiveKit Inference reshape frames on the socket ([`WireShape`]).

use base64::Engine as _;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role};
use serde_json::{Map, Value, json};
use std::time::SystemTime;

use crate::realtime::dialect::{NUM_CHANNELS, SAMPLE_RATE};

/// Azure's deployment, for the URL.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Azure {
    pub deployment: Option<String>,
}

/// `process_base_url`: `http(s)` -> `ws(s)` as a prefix swap; `/realtime` appended only to an empty
/// or recognised base path (xAI's `/v1/realtime` is untouched). Not Azure: `model` is added as a
/// query parameter unless present. Azure: the deployment as `model`.
pub fn process_base_url(url: &str, model: &str, azure: Option<&Azure>) -> String {
    let url = if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        url.to_owned()
    };

    let (base, query) = match url.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (url.as_str(), None),
    };

    // Everything up to the first `/` after `://` is the authority.
    let (authority, path) = match base.find("://") {
        Some(i) => match base[i + 3..].find('/') {
            Some(j) => base.split_at(i + 3 + j),
            None => (base, ""),
        },
        None => (base, ""),
    };

    let stripped = path.trim_end_matches('/');
    let path = match azure {
        Some(_) => match stripped {
            "" | "/openai" => format!("{stripped}/v1/realtime"),
            "/openai/v1" => "/openai/v1/realtime".to_owned(),
            _ => path.to_owned(),
        },
        None if path.is_empty() || matches!(stripped, "" | "/v1" | "/openai" | "/openai/v1") => {
            format!("{stripped}/realtime")
        }
        None => path.to_owned(),
    };

    let mut params: Vec<(String, String)> = query
        .into_iter()
        .flat_map(|q| q.split('&'))
        .filter(|kv| !kv.is_empty())
        .map(|kv| match kv.split_once('=') {
            Some((k, v)) => (k.to_owned(), v.to_owned()),
            None => (kv.to_owned(), String::new()),
        })
        .collect();
    let has = |params: &[(String, String)], k: &str| params.iter().any(|(p, _)| p == k);
    match azure {
        Some(Azure { deployment: Some(d) }) if !has(&params, "model") => {
            params.push(("model".into(), d.clone()));
        }
        Some(_) => {}
        None if !has(&params, "model") => params.push(("model".into(), model.to_owned())),
        None => {}
    }
    let query: Vec<String> = params
        .into_iter()
        .map(|(k, v)| if v.is_empty() { k } else { format!("{k}={v}") })
        .collect();
    if query.is_empty() {
        format!("{authority}{path}")
    } else {
        format!("{authority}{path}?{}", query.join("&"))
    }
}

/// What a deployment changes about the frames on the socket, applied by the driver to every frame
/// in each direction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WireShape {
    /// Azure: the beta event names come in, and are read as the GA ones.
    pub azure: bool,
    /// LiveKit Inference picks the model from the URL; `session.model` is dropped.
    pub drop_model: bool,
}

/// `_AZURE_EVENT_MAPPING`.
const AZURE_EVENT_MAPPING: &[(&str, &str)] = &[
    ("response.text.delta", "response.output_text.delta"),
    ("response.text.done", "response.output_text.done"),
    ("response.audio_transcript.delta", "response.output_audio_transcript.delta"),
    ("response.audio_transcript.done", "response.output_audio_transcript.done"),
    ("response.audio.delta", "response.output_audio.delta"),
    ("response.audio.done", "response.output_audio.done"),
    ("conversation.item.created", "conversation.item.added"),
];

impl WireShape {
    /// A frame about to be sent.
    pub fn outbound(&self, frame: &mut Value) {
        if frame.get("type").and_then(Value::as_str) == Some("session.update")
            && self.drop_model
            && let Some(Value::Object(session)) = frame.get_mut("session")
        {
            session.remove("model");
        }
    }

    /// A frame just received.
    pub fn inbound(&self, frame: &mut Value) {
        if !self.azure {
            return;
        }
        let ty = frame.get("type").and_then(Value::as_str).unwrap_or("");
        if let Some((_, ga)) = AZURE_EVENT_MAPPING.iter().find(|(beta, _)| *beta == ty) {
            frame["type"] = json!(ga);
        }
    }
}

/// Where `utils.shortuuid(prefix)` comes from; a port so wire tests can predict `event_id`s.
pub trait IdGen: Send + Sync {
    fn short(&self, prefix: &str) -> String;
}

/// Production: `utils.shortuuid`, six random bytes hex-encoded behind the prefix.
pub struct RandomIds;

impl IdGen for RandomIds {
    fn short(&self, prefix: &str) -> String {
        crate::shortuuid(prefix)
    }
}

pub struct SeqIds(std::sync::atomic::AtomicU64);

impl Default for SeqIds {
    fn default() -> Self {
        SeqIds(std::sync::atomic::AtomicU64::new(0))
    }
}

impl IdGen for SeqIds {
    fn short(&self, prefix: &str) -> String {
        let n = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        format!("{prefix}{n}")
    }
}

/// Options that reach a `session.update`. Absent from the map means "not given", `Value::Null`
/// means "explicitly off".
#[derive(Debug, Clone, Default)]
pub struct SessionPatch {
    /// Top-level `session` keys.
    pub top: Map<String, Value>,
    pub audio_in: Map<String, Value>,
    pub audio_out: Map<String, Value>,
}

impl SessionPatch {
    /// Assemble the `session` object. `audio` appears only when a half was touched.
    pub fn into_session(self) -> Map<String, Value> {
        let mut session = Map::new();
        session.insert("type".into(), json!("realtime"));
        for (k, v) in self.top {
            session.insert(k, v);
        }
        if !self.audio_in.is_empty() || !self.audio_out.is_empty() {
            let mut audio = Map::new();
            audio.insert("input".into(), Value::Object(self.audio_in));
            audio.insert("output".into(), Value::Object(self.audio_out));
            session.insert("audio".into(), Value::Object(audio));
        }
        session
    }
}

/// `{"rate": 24000, "type": "audio/pcm"}` -- the same block on both halves.
pub fn audio_format() -> Value {
    json!({"rate": SAMPLE_RATE, "type": "audio/pcm"})
}

/// `{"type": "session.update", "session": ..., "event_id": ...}`, in the reference's key order.
pub fn session_update(event_id: &str, session: Map<String, Value>) -> Value {
    json!({"type": "session.update", "session": Value::Object(session), "event_id": event_id})
}

pub fn conversation_item_create(
    event_id: &str,
    item: Value,
    previous_item_id: Option<&str>,
) -> Value {
    let mut ev = Map::new();
    ev.insert("type".into(), json!("conversation.item.create"));
    ev.insert("item".into(), item);
    // `"root"` for "first in the conversation"; `null` here means "append at the tail".
    if let Some(prev) = previous_item_id {
        ev.insert("previous_item_id".into(), json!(prev));
    }
    ev.insert("event_id".into(), json!(event_id));
    Value::Object(ev)
}

/// xAI's `force_message`: not a conversation item, and no `previous_item_id`.
pub fn force_message(event_id: &str, text: &str) -> Value {
    json!({
        "type": "conversation.item.create",
        "event_id": event_id,
        "item": {
            "type": "force_message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    })
}

pub fn conversation_item_delete(event_id: &str, item_id: &str) -> Value {
    json!({"type": "conversation.item.delete", "item_id": item_id, "event_id": event_id})
}

pub fn conversation_item_truncate(item_id: &str, audio_end_ms: u64) -> Value {
    json!({
        "type": "conversation.item.truncate",
        "content_index": 0,
        "item_id": item_id,
        "audio_end_ms": audio_end_ms,
    })
}

pub fn input_audio_buffer_append(pcm: &[u8]) -> Value {
    json!({
        "type": "input_audio_buffer.append",
        "audio": base64::engine::general_purpose::STANDARD.encode(pcm),
    })
}

pub fn input_audio_buffer_commit() -> Value {
    json!({"type": "input_audio_buffer.commit"})
}

pub fn input_audio_buffer_clear() -> Value {
    json!({"type": "input_audio_buffer.clear"})
}

/// `response.create`. The server echoes `metadata.client_event_id` on `response.created`; that is
/// how a reply is matched to its request.
pub fn response_create(
    event_id: &str,
    instructions: Option<&str>,
    tool_choice: Option<Value>,
) -> Value {
    let mut params = Map::new();
    // Always present, even as null (`instructions=instructions or None`).
    params.insert("instructions".into(), instructions.map_or(Value::Null, |s| json!(s)));
    params.insert("metadata".into(), json!({"client_event_id": event_id}));
    if let Some(tc) = tool_choice {
        params.insert("tool_choice".into(), tc);
    }
    json!({"type": "response.create", "event_id": event_id, "response": Value::Object(params)})
}

/// `response.cancel`. `None` cancels whatever is in flight; `Some` cancels one by id.
pub fn response_cancel(response_id: Option<&str>) -> Value {
    match response_id {
        Some(id) => json!({"type": "response.cancel", "response_id": id}),
        None => json!({"type": "response.cancel"}),
    }
}

/// `to_oai_tool_choice`; `None` and anything unrecognised become `"auto"`.
pub fn tool_choice(tc: Option<&lk_agents_abi::ToolChoice>) -> Value {
    use lk_agents_abi::ToolChoice;
    match tc {
        Some(ToolChoice::Auto) | None => json!("auto"),
        Some(ToolChoice::Required) => json!("required"),
        Some(ToolChoice::Forbidden) => json!("none"),
        Some(ToolChoice::Named(name)) => json!({"name": name, "type": "function"}),
    }
}

/// `_convert_tools_to_oai`: the internally-tagged function schema.
pub fn tool_spec(t: &lk_agents_abi::ToolSpec) -> Value {
    if let Some(provider_tool) = &t.provider_tool {
        return provider_tool.payload.clone();
    }
    json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.json_schema,
    })
}

const MAX_CALL_ID_LEN: usize = 32;

/// `_shorten_call_id`: OpenAI caps `call_id` at 32 characters. FNV-1a as 32 hex characters, not
/// the reference's SHA-256 prefix; only determinism matters, since both ends of a pair go
/// through this function.
pub fn shorten_call_id(call_id: &str) -> String {
    if call_id.len() <= MAX_CALL_ID_LEN {
        return call_id.to_owned();
    }
    let mut a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b: u64 = 0x9e37_79b9_7f4a_7c15;
    for (i, byte) in call_id.bytes().enumerate() {
        a = (a ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        b = (b ^ ((byte as u64) << (i % 8))).wrapping_mul(0x100_0000_01b3);
    }
    format!("{a:016x}{b:016x}")
}

/// `livekit_item_to_openai_item`. Content-part type is `output_text` for assistant, `input_text`
/// otherwise; `input_text` on an assistant turn is read as new user input after a reconnect.
pub fn to_wire_item(item: &ChatItem, images: &dyn ImageResolver) -> Result<Value, String> {
    let mut out = Map::new();
    match item {
        ChatItem::FunctionCall(fc) => {
            out.insert("id".into(), json!(fc.id));
            out.insert("type".into(), json!("function_call"));
            out.insert("call_id".into(), json!(shorten_call_id(&fc.call_id)));
            out.insert("name".into(), json!(fc.name));
            out.insert("arguments".into(), json!(fc.arguments));
        }
        ChatItem::FunctionCallOutput(fco) => {
            out.insert("id".into(), json!(fco.id));
            out.insert("type".into(), json!("function_call_output"));
            out.insert("call_id".into(), json!(shorten_call_id(&fco.call_id)));
            out.insert("output".into(), json!(fco.output));
        }
        ChatItem::Message(m) => {
            let (role, text_type) = match m.role {
                // `role == "system" or role == "developer"` both become `"system"` on this wire.
                Role::System | Role::Developer => ("system", "input_text"),
                Role::Assistant => ("assistant", "output_text"),
                Role::User => ("user", "input_text"),
            };
            let mut content = Vec::new();
            for c in &m.content {
                match c {
                    ChatContent::Text(t) => {
                        content.push(json!({"type": text_type, "text": t}));
                    }
                    // Images and audio are user-only in the reference; other roles drop them
                    // silently.
                    ChatContent::Image(img) if m.role == Role::User => {
                        let s = images.resolve(img)?;
                        if s.external_url.is_some() {
                            return Err("realtime images require an inline data URL; external URLs are unsupported".into());
                        }
                        let Some(bytes) = s.data_bytes else { continue };
                        let mime = s.mime_type.as_deref().unwrap_or("image/jpeg");
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        content.push(json!({
                            "type": "input_image",
                            "image_url": format!("data:{mime};base64,{b64}"),
                        }));
                    }
                    // `AudioContent` needs the host's `rtc.combine_audio_frames` (§1.11); only a
                    // context replayed from another provider carries one.
                    ChatContent::Audio(_) | ChatContent::Image(_) => continue,
                }
            }
            out.insert("id".into(), json!(m.id));
            out.insert("type".into(), json!("message"));
            out.insert("role".into(), json!(role));
            out.insert("content".into(), Value::Array(content));
        }
        // The reference falls into `UnboundLocalError` here; every caller filters these out.
        other => return Err(format!("unsupported item type: {}", other.kind())),
    }
    Ok(Value::Object(out))
}

/// `openai_item_to_livekit_item`. `created_at` is a parameter: the pure tier holds no clock.
pub fn from_wire_item(item: &Value, created_at: SystemTime) -> Result<ChatItem, String> {
    let id = item.get("id").and_then(Value::as_str).ok_or("id is None")?.to_owned();
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => Ok(ChatItem::FunctionCall(FunctionCall {
            id,
            call_id: required_string(item, "call_id")?,
            arguments: required_string(item, "arguments")?,
            name: required_string(item, "name")?,
            created_at,
            extra: Default::default(),
            group_id: None,
        })),
        Some("function_call_output") => Ok(ChatItem::FunctionCallOutput(FunctionCallOutput::new(
            id,
            required_string(item, "call_id")?,
            required_string(item, "output")?,
            false,
            created_at,
        ))),
        Some("message") => {
            let role = match item.get("role").and_then(Value::as_str) {
                Some("system") => Role::System,
                Some("developer") => Role::Developer,
                Some("assistant") => Role::Assistant,
                Some("user") => Role::User,
                other => return Err(format!("unsupported role: {other:?}")),
            };
            let mut content = Vec::new();
            for part in item.get("content").and_then(Value::as_array).into_iter().flatten() {
                let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
                match ty {
                    // `input_audio` contributes its transcript; the frames stay server-side.
                    "input_audio" => {
                        if let Some(t) = part.get("transcript").and_then(Value::as_str) {
                            content.push(ChatContent::Text(t.to_owned()));
                        }
                    }
                    "input_image" => {
                        if let Some(url) = part.get("image_url").and_then(Value::as_str) {
                            content.push(ChatContent::Image(lk_agents_abi::ImageContent {
                                id: format!("img_{id}_{}", content.len()),
                                image: lk_agents_abi::ImageSource::Url(url.to_owned()),
                                inference_width: None,
                                inference_height: None,
                                inference_detail: Default::default(),
                                mime_type: None,
                            }));
                        }
                    }
                    // Everything else contributes a non-empty `text`, as the reference's three
                    // role branches do.
                    _ => {
                        if let Some(t) = part.get("text").and_then(Value::as_str)
                            && !t.is_empty()
                        {
                            content.push(ChatContent::Text(t.to_owned()));
                        }
                    }
                }
            }
            Ok(ChatItem::Message(ChatMessage::new(id, role, content, created_at)))
        }
        other => Err(format!("unsupported item type: {other:?}")),
    }
}

pub use crate::realtime::protocol::{
    Incoming, ResponseDone, ServerError, TranscriptionCompleted, TranscriptionUsage,
};

fn transcription_usage(v: Option<&Value>) -> Result<Option<TranscriptionUsage>, String> {
    let Some(v) = v.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let usage = match required_string(v, "type")?.as_str() {
        "tokens" => TranscriptionUsage::Tokens(TokenUsage {
            input_tokens: u(v, &["input_tokens"])?,
            output_tokens: u(v, &["output_tokens"])?,
            input_audio_tokens: opt_u(v, &["input_token_details", "audio_tokens"])?,
            ..TokenUsage::default()
        }),
        "duration" => TranscriptionUsage::Duration {
            seconds: v
                .get("seconds")
                .and_then(Value::as_f64)
                .filter(|seconds| *seconds >= 0.0)
                .ok_or_else(|| {
                    "transcription usage seconds must be a non-negative number".to_owned()
                })?,
        },
        _ => return Ok(None),
    };
    Ok(Some(usage))
}

/// `calculate_confidence_from_logprobs`: the geometric mean `exp(sum(logprob) / n)`, not
/// `mean(exp(logprob))`.
pub fn confidence_from_logprobs(logprobs: Option<&Vec<Value>>) -> Option<f64> {
    let lps = logprobs?;
    if lps.is_empty() {
        return None;
    }
    let total: f64 = lps.iter().filter_map(|l| l.get("logprob").and_then(Value::as_f64)).sum();
    Some((total / lps.len() as f64).exp())
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

fn u(v: &Value, path: &[&str]) -> Result<u64, String> {
    Ok(opt_u(v, path)?.unwrap_or(0))
}

/// `None` when the provider did not report the count (usage.md: absent is not zero).
fn opt_u(v: &Value, path: &[&str]) -> Result<Option<u64>, String> {
    let mut cur = v;
    for part in path {
        match cur.get(part) {
            Some(next) => cur = next,
            None if cur.is_null() || cur.is_object() => return Ok(None),
            None => return Err(format!("{} must be an unsigned integer", path.join("."))),
        }
    }
    match cur {
        Value::Null => Ok(None),
        _ => cur
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{} must be an unsigned integer", path.join("."))),
    }
}

/// `response.done`'s usage. OpenAI's input total includes its cached tokens.
fn response_usage(r: &Value) -> Result<TokenUsage, String> {
    let at = |path: &[&str]| opt_u(r, &[&["usage"], path].concat());
    let input = |k: &str| at(&["input_token_details", k]);
    let cached = |k: &str| at(&["input_token_details", "cached_tokens_details", k]);
    let output = |k: &str| at(&["output_token_details", k]);
    Ok(TokenUsage {
        input_tokens: at(&["input_tokens"])?.unwrap_or(0),
        output_tokens: at(&["output_tokens"])?.unwrap_or(0),
        cache_read_tokens: input("cached_tokens")?,
        input_text_tokens: input("text_tokens")?,
        input_audio_tokens: input("audio_tokens")?,
        input_image_tokens: input("image_tokens")?,
        output_text_tokens: output("text_tokens")?,
        output_audio_tokens: output("audio_tokens")?,
        output_image_tokens: output("image_tokens")?,
        cache_read_text_tokens: cached("text_tokens")?,
        cache_read_audio_tokens: cached("audio_tokens")?,
        cache_read_image_tokens: cached("image_tokens")?,
        ..TokenUsage::default()
    })
}

/// A malformed provider frame retains its operation and correlated event identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid realtime {operation} ({event_id:?}): {cause}")]
pub struct DecodeError {
    pub operation: String,
    pub event_id: Option<String>,
    pub cause: String,
}

impl DecodeError {
    fn at(ev: &Value, cause: String) -> Self {
        Self {
            operation: ev.get("type").and_then(Value::as_str).unwrap_or("message").to_owned(),
            event_id: ev.get("event_id").and_then(Value::as_str).map(str::to_owned),
            cause,
        }
    }
}

/// Parse and validate one socket frame before any core transition.
pub fn decode_text(
    text: &str,
    shape: &WireShape,
    created_at: SystemTime,
) -> Result<Incoming, DecodeError> {
    let mut value: Value = serde_json::from_str(text).map_err(|error| DecodeError {
        operation: "message".into(),
        event_id: None,
        cause: error.to_string(),
    })?;
    shape.inbound(&mut value);
    decode_at(&value, created_at)
}

fn required_string(value: &Value, field: &str) -> Result<String, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{field} must be a string"))
}

#[cfg(test)]
pub fn decode(ev: &Value) -> Incoming {
    decode_at(ev, SystemTime::UNIX_EPOCH).expect("valid fixture")
}

/// The `if/elif` chain of `_recv_task`, as a total function.
pub fn decode_at(ev: &Value, created_at: SystemTime) -> Result<Incoming, DecodeError> {
    let ty = required_string(ev, "type").map_err(|cause| DecodeError::at(ev, cause))?;

    (|| -> Result<Incoming, String> {
        Ok(match ty.as_str() {
            "input_audio_buffer.speech_started" => {
                Incoming::SpeechStarted { item_id: s(ev, "item_id") }
            }
            "input_audio_buffer.speech_stopped" => Incoming::SpeechStopped,
            "response.created" => {
                let response = ev.get("response").unwrap_or(&Value::Null);
                Incoming::ResponseCreated {
                    response_id: required_string(response, "id")?,
                    client_event_id: response
                        .get("metadata")
                        .and_then(|m| m.get("client_event_id"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                }
            }
            "response.output_item.added" => {
                let item = ev.get("item").unwrap_or(&Value::Null);
                Incoming::OutputItemAdded {
                    item_id: required_string(item, "id")?,
                    item_type: required_string(item, "type")?,
                }
            }
            "response.content_part.added" => Incoming::ContentPartAdded {
                item_id: required_string(ev, "item_id")?,
                part_type: required_string(ev.get("part").unwrap_or(&Value::Null), "type")?,
            },
            "conversation.item.added" => Incoming::ItemAdded {
                previous_item_id: s(ev, "previous_item_id"),
                item: from_wire_item(ev.get("item").unwrap_or(&Value::Null), created_at)?,
            },
            "conversation.item.deleted" => {
                Incoming::ItemDeleted { item_id: required_string(ev, "item_id")? }
            }
            "conversation.item.input_audio_transcription.delta" => Incoming::TranscriptionDelta {
                item_id: required_string(ev, "item_id")?,
                content_index: u(ev, &["content_index"])? as usize,
                delta: required_string(ev, "delta")?,
            },
            "conversation.item.input_audio_transcription.completed" => {
                Incoming::TranscriptionCompleted(TranscriptionCompleted {
                    item_id: required_string(ev, "item_id")?,
                    content_index: u(ev, &["content_index"])? as usize,
                    transcript: required_string(ev, "transcript")?,
                    confidence: confidence_from_logprobs(
                        ev.get("logprobs").and_then(Value::as_array),
                    ),
                    in_progress: ev.get("status").and_then(Value::as_str) == Some("in_progress"),
                    event_id: s(ev, "event_id").unwrap_or_default(),
                    usage: transcription_usage(ev.get("usage"))?,
                })
            }
            "conversation.item.input_audio_transcription.failed" => Incoming::TranscriptionFailed {
                item_id: required_string(ev, "item_id")?,
                content_index: u(ev, &["content_index"])? as usize,
            },
            "conversation.item.input_audio_transcription.updated" => {
                Incoming::TranscriptionUpdated {
                    item_id: required_string(ev, "item_id")?,
                    transcript: required_string(ev, "transcript")?,
                }
            }
            "response.output_text.delta" => Incoming::TextDelta {
                item_id: required_string(ev, "item_id")?,
                delta: required_string(ev, "delta")?,
            },
            "response.output_audio_transcript.delta" => Incoming::AudioTranscriptDelta {
                item_id: required_string(ev, "item_id")?,
                delta: required_string(ev, "delta")?,
                start_time: ev.get("start_time").and_then(Value::as_f64),
            },
            "response.output_audio.delta" => Incoming::AudioDelta {
                item_id: required_string(ev, "item_id")?,
                frame: {
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(required_string(ev, "delta")?)
                        .map_err(|error| format!("invalid PCM base64: {error}"))?;
                    if data.len() % 2 != 0 {
                        return Err("PCM data ends in a partial i16 sample".into());
                    }
                    audio_frame(data)
                },
            },
            "response.output_item.done" => Incoming::OutputItemDone {
                item: from_wire_item(ev.get("item").unwrap_or(&Value::Null), created_at)?,
            },
            "response.done" => {
                let r = ev.get("response").unwrap_or(&Value::Null);
                let details = r.get("status_details");
                let error = details.and_then(|d| d.get("error"));
                Incoming::ResponseDone(ResponseDone {
                    response_id: required_string(r, "id")?,
                    status: match required_string(r, "status")?.as_str() {
                        "completed" => lk_agents_abi::llm::realtime::GenerationStatus::Completed,
                        "cancelled" => lk_agents_abi::llm::realtime::GenerationStatus::Cancelled,
                        "incomplete" => lk_agents_abi::llm::realtime::GenerationStatus::Incomplete,
                        "failed" => lk_agents_abi::llm::realtime::GenerationStatus::Failed,
                        other => return Err(format!("unsupported response status: {other}")),
                    },
                    error_type: error.and_then(|e| s(e, "type")),
                    error_code: error.and_then(|e| s(e, "code")),
                    usage: response_usage(r)?,
                })
            }
            "error" => {
                let e = ev.get("error").unwrap_or(&Value::Null);
                Incoming::Error(ServerError {
                    event_id: s(e, "event_id"),
                    code: s(e, "code"),
                    type_: s(e, "type"),
                    message: s(e, "message").unwrap_or_default(),
                })
            }
            "input_audio_buffer.timeout_triggered" => Incoming::IdleTimeoutTriggered,
            "session.created" => {
                Incoming::SessionCreated { model: ev.get("session").and_then(|s_| s(s_, "model")) }
            }
            other => Incoming::Unhandled { event_type: other.to_owned() },
        })
    })()
    .map_err(|cause| DecodeError::at(ev, cause))
}

/// An `AudioDelta`'s PCM as an [`AudioFrame`](lk_agents_abi::AudioFrame) at the session rate.
pub fn audio_frame(pcm: Vec<u8>) -> lk_agents_abi::AudioFrame {
    let samples = (pcm.len() / 2) as u32;
    lk_agents_abi::AudioFrame {
        data: pcm.into(),
        sample_rate: SAMPLE_RATE,
        num_channels: NUM_CHANNELS,
        samples_per_channel: samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::openai::testfix::{TIME, msg};
    use crate::testkit::NoImages;

    #[test]
    fn the_base_url_becomes_a_realtime_websocket_url() {
        for (base, want) in [
            ("wss://api.x.ai/v1/realtime", "wss://api.x.ai/v1/realtime?model=m"),
            ("https://api.openai.com/v1", "wss://api.openai.com/v1/realtime?model=m"),
            ("https://gw.example", "wss://gw.example/realtime?model=m"),
            ("https://gw/v1?next=http://x", "wss://gw/v1/realtime?next=http://x&model=m"),
            ("wss://gw/v1?model=pinned", "wss://gw/v1/realtime?model=pinned"),
        ] {
            assert_eq!(process_base_url(base, "m", None), want);
        }
    }

    // Catches: an Azure URL built the OpenAI way: Azure takes `/openai/v1/realtime` and the
    // deployment as `model`.
    #[test]
    fn azure_urls_take_the_deployment_as_model() {
        let ga = Azure { deployment: Some("dep".into()) };
        assert_eq!(
            process_base_url("https://r.azure.com/openai", "m", Some(&ga)),
            "wss://r.azure.com/openai/v1/realtime?model=dep"
        );
        assert_eq!(
            process_base_url("https://r.azure.com/openai/v1", "m", Some(&ga)),
            "wss://r.azure.com/openai/v1/realtime?model=dep"
        );
    }

    // Catches: Azure's beta event names unmapped.
    #[test]
    fn azure_beta_event_names_read_as_the_ga_ones() {
        let shape = WireShape { azure: true, drop_model: false };
        let mut inc = json!({"type": "response.audio.delta"});
        shape.inbound(&mut inc);
        assert_eq!(inc["type"], json!("response.output_audio.delta"));
    }

    // Catches: dropping explicit nulls from session.update (a patch could not clear a setting).
    #[test]
    fn an_explicit_null_survives_into_the_session_object() {
        let mut p = SessionPatch::default();
        p.audio_in.insert("noise_reduction".into(), Value::Null);
        p.audio_in.insert("transcription".into(), json!({"model": "gpt-4o-mini-transcribe"}));
        let session = p.into_session();
        assert_eq!(session["audio"]["input"]["noise_reduction"], Value::Null);
        assert!(session["audio"]["input"].as_object().unwrap().contains_key("noise_reduction"));
    }

    // Catches: emitting an empty `audio` block for a patch that touched neither half.
    #[test]
    fn a_patch_with_no_audio_fields_ships_no_audio_key() {
        let mut p = SessionPatch::default();
        p.top.insert("tool_choice".into(), json!("auto"));
        let session = p.into_session();
        assert!(!session.contains_key("audio"));
        assert_eq!(Value::Object(session), json!({"type": "realtime", "tool_choice": "auto"}));
    }

    // Catches: `response.create` omitting `metadata.client_event_id`.
    #[test]
    fn response_create_carries_the_correlation_id_in_metadata() {
        let ev = response_create("response_create_1", None, None);
        assert_eq!(ev["type"], json!("response.create"));
        assert_eq!(ev["event_id"], json!("response_create_1"));
        assert_eq!(ev["response"]["metadata"]["client_event_id"], json!("response_create_1"));
        assert_eq!(ev["response"]["instructions"], Value::Null);
        assert!(ev["response"].get("tool_choice").is_none(), "not given means absent");
    }

    // Catches: writing an assistant message's content parts as `input_text`.
    #[test]
    fn the_content_part_type_follows_the_role() {
        let user = to_wire_item(&msg("u1", "hello"), &NoImages).unwrap();
        assert_eq!(user["content"][0]["type"], json!("input_text"));
        assert_eq!(user["role"], json!("user"));

        let mut assistant = msg("a1", "hi there");
        if let ChatItem::Message(m) = &mut assistant {
            m.role = Role::Assistant;
        }
        let assistant = to_wire_item(&assistant, &NoImages).unwrap();
        assert_eq!(assistant["content"][0]["type"], json!("output_text"));
        assert_eq!(assistant["role"], json!("assistant"));

        let mut developer = msg("d1", "be brief");
        if let ChatItem::Message(m) = &mut developer {
            m.role = Role::Developer;
        }
        let developer = to_wire_item(&developer, &NoImages).unwrap();
        // "developer" collapses to "system" on this wire; there is no developer role here.
        assert_eq!(developer["role"], json!("system"));
        assert_eq!(developer["content"][0]["type"], json!("input_text"));
    }

    // Catches: shortening one side of a function-call pair, or non-deterministically.
    #[test]
    fn an_over_long_call_id_shortens_to_the_same_value_on_both_sides() {
        let long = "a".repeat(64);
        let short = shorten_call_id(&long);
        assert_eq!(short.len(), MAX_CALL_ID_LEN);
        assert_eq!(short, shorten_call_id(&long), "deterministic");
        assert_ne!(short, shorten_call_id(&"b".repeat(64)));
        // under the cap, untouched -- the common case must not be hashed
        assert_eq!(shorten_call_id("call_abc"), "call_abc");
    }

    // Catches: the geometric mean written as `mean(exp(x))`.
    #[test]
    fn confidence_is_the_geometric_mean_not_the_arithmetic_one() {
        let lps = vec![json!({"logprob": -0.1}), json!({"logprob": -2.0})];
        let got = confidence_from_logprobs(Some(&lps)).unwrap();
        let geometric = ((-0.1f64 + -2.0) / 2.0).exp();
        let arithmetic = ((-0.1f64).exp() + (-2.0f64).exp()) / 2.0;
        assert!((got - geometric).abs() < 1e-12);
        assert!((got - arithmetic).abs() > 0.05, "the two must not be confused");
        assert_eq!(confidence_from_logprobs(None), None);
        assert_eq!(confidence_from_logprobs(Some(&vec![])), None);
    }

    // Catches: decoding `response.created` without reaching into `response.metadata`.
    #[test]
    fn response_created_is_decoded_with_its_nested_correlation_id() {
        let ev = json!({
            "type": "response.created",
            "response": {"id": "resp_1", "metadata": {"client_event_id": "response_create_7"}},
        });
        assert_eq!(
            decode(&ev),
            Incoming::ResponseCreated {
                response_id: "resp_1".into(),
                client_event_id: Some("response_create_7".into()),
            }
        );
        // force_message replies carry no metadata at all -- the case xAI's FIFO tag exists for
        let bare = json!({"type": "response.created", "response": {"id": "resp_2"}});
        assert_eq!(
            decode(&bare),
            Incoming::ResponseCreated { response_id: "resp_2".into(), client_event_id: None }
        );
    }

    // Catches: treating xAI's `status: "in_progress"` transcript as final.
    #[test]
    fn an_in_progress_transcript_is_flagged_as_such() {
        let ev = json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "transcript": "hello", "status": "in_progress",
        });
        let Incoming::TranscriptionCompleted(t) = decode(&ev) else { panic!("wrong variant") };
        assert!(t.in_progress);

        let ev = json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "transcript": "hello there",
        });
        let Incoming::TranscriptionCompleted(t) = decode(&ev) else { panic!("wrong variant") };
        assert!(!t.in_progress, "OpenAI sends no status; absent must mean final");
    }

    // Catches: an unknown event type panicking or being dropped; `decode` is total.
    #[test]
    fn an_unknown_event_type_decodes_rather_than_panics() {
        assert_eq!(
            decode(&json!({"type": "response.mcp_call.completed"})),
            Incoming::Unhandled { event_type: "response.mcp_call.completed".into() }
        );
        assert!(decode_at(&json!({}), TIME).is_err());
    }

    // Catches: reading token usage off the flat `usage` object only (cached tokens are nested).
    #[test]
    fn nested_token_details_are_read_at_every_depth() {
        let ev = json!({"type": "response.done", "response": {
            "id": "resp_1", "status": "completed",
            "usage": {
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                "input_token_details": {
                    "audio_tokens": 80, "text_tokens": 20, "cached_tokens": 64,
                    "cached_tokens_details": {"audio_tokens": 60, "text_tokens": 4},
                },
                "output_token_details": {"audio_tokens": 18, "text_tokens": 2},
            },
        }});
        let Incoming::ResponseDone(d) = decode(&ev) else { panic!("wrong variant") };
        assert_eq!(d.usage.input_tokens, 100);
        assert_eq!(d.usage.cache_read_tokens, Some(64));
        assert_eq!(d.usage.cache_read_audio_tokens, Some(60));
        assert_eq!(d.usage.cache_read_image_tokens, None, "not reported");
        assert_eq!(d.usage.output_audio_tokens, Some(18));
        assert_eq!(d.status, lk_agents_abi::llm::realtime::GenerationStatus::Completed);
    }

    // Catches: losing the error `type` when only `code` is set, or vice versa.
    #[test]
    fn both_error_discriminants_survive_decoding() {
        let ev = json!({"type": "error", "error": {
            "type": "invalid_request_error", "code": "item_create_duplicate_item_id",
            "message": "already exists", "event_id": "chat_ctx_create_3",
        }});
        assert_eq!(
            decode(&ev),
            Incoming::Error(ServerError {
                event_id: Some("chat_ctx_create_3".into()),
                code: Some("item_create_duplicate_item_id".into()),
                type_: Some("invalid_request_error".into()),
                message: "already exists".into(),
            })
        );
    }

    // Catches: an `input_audio` part contributing nothing on the way back in.
    #[test]
    fn a_committed_audio_turn_round_trips_as_its_transcript() {
        let item = json!({
            "id": "item_9", "type": "message", "role": "user",
            "content": [{"type": "input_audio", "transcript": "what time is it"}],
        });
        let got = from_wire_item(&item, TIME).unwrap();
        assert_eq!(
            got.as_message().unwrap().raw_text_content().as_deref(),
            Some("what time is it")
        );
        assert_eq!(got.as_message().unwrap().role, Role::User);
    }
}
