//! The LLM pipeline events (`llm/llm.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::llm::chat_context::{Extra, Role};
use lk_agents_host_derive::Host;

// `llm/llm.py:35`. Field `///`s are the reference's, verbatim.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(pydantic, module = "llm", path = "livekit.agents.llm.llm", name = "CompletionUsage")]
#[binding(emit, extract)]
pub struct CompletionUsage {
    /// The number of tokens in the completion.
    pub completion_tokens: u32,
    /// The number of input tokens used (includes cached tokens).
    pub prompt_tokens: u32,
    /// The number of cached input tokens used.
    #[python(default = "0")]
    pub prompt_cached_tokens: u32,
    /// The number of tokens used to create the cache.
    #[python(default = "0")]
    pub cache_creation_tokens: u32,
    /// The number of tokens read from the cache.
    #[python(default = "0")]
    pub cache_read_tokens: u32,
    /// The total number of tokens used (completion + prompt tokens).
    // Reported by the provider, not derived: providers disagree on cache reads.
    pub total_tokens: u32,
    /// The service tier used for processing the request (e.g. 'default', 'priority', 'flex').
    /// Returned by providers that support tiered processing (e.g. OpenAI).
    #[python(default = "None")]
    pub service_tier: Option<String>,
}

// `llm/llm.py:53`. `arguments` is the raw, possibly partial JSON text; validated after the stream.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    pydantic,
    module = "llm",
    path = "livekit.agents.llm.llm",
    name = "FunctionToolCall",
    tag = "type",
    tag_value = "function"
)]
#[binding(emit, extract)]
pub struct FunctionToolCall {
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    /// Provider-specific extra data (e.g., Google thought signatures).
    // Read but not emitted: a Google thought signature must survive; the binding dict takes three
    // fields. The Python class still declares the key.
    #[binding(skip)]
    #[python(ty = "dict[str, Any] | None", import = "typing:Any", default = "None")]
    pub extra: Option<Extra>,
}

// `llm/llm.py:71`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(pydantic, module = "llm", path = "livekit.agents.llm.llm", name = "ChoiceDelta")]
// Read by hand (`FromPyObject` below): an unknown role reads as absent.
#[binding(emit)]
pub struct ChoiceDelta {
    // An unknown role reads as absent, as the reference's `delta.get("role") or "assistant"`.
    #[python(ty = "ChatRole | None", import = "..llm.chat_context:ChatRole", default = "None")]
    pub role: Option<Role>,
    #[python(default = "None")]
    pub content: Option<String>,
    #[python(default_factory = "list")]
    pub tool_calls: Vec<FunctionToolCall>,
    /// Provider-specific extra data (e.g., Google thought signatures).
    // Not emitted (`_chunk_from_core` takes three fields); read, as `FunctionToolCall::extra`.
    #[binding(skip)]
    #[python(ty = "dict[str, Any] | None", import = "typing:Any", default = "None")]
    pub extra: Option<Extra>,
}

/// A `role` this build does not know reads as `None`.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ChoiceDelta {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr, attr_with, extra_from_py};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "ChoiceDeltaFromPy";
        let obj = &*obj;
        Ok(ChoiceDelta {
            role: attr_with(obj, OWNER, "role", |v| Ok(v.extract::<String>().ok().and_then(|s| s.parse().ok())))?,
            content: attr(obj, OWNER, "content")?,
            tool_calls: attr(obj, OWNER, "tool_calls")?,
            extra: attr_with(obj, OWNER, "extra", |v| {
                if v.is_none() { Ok(None) } else { extra_from_py(v.cast::<pyo3::types::PyDict>()?).map(Some) }
            })?,
        })
    }
}

// `llm/llm.py:79`. `has_response` is a host method hook: a core call per token would be a GIL
// crossing per token.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    module = "llm",
    path = "livekit.agents.llm.llm",
    name = "ChatChunk",
    method(
        name = "has_response",
        ret = "bool",
        hook = "chat_chunk_has_response",
        doc = "Whether this chunk delivered generation the caller can see.\n\nToken counts and provider metadata (a gateway deployment stamp, a thought\nsignature) reach the caller without being output: they neither start the\nclock on time-to-first-token nor give a retry anything to duplicate."
    )
)]
#[binding(emit, extract)]
pub struct LlmChunk {
    pub id: String,
    // Emitted as `None` rather than omitted; `_chunk_from_core` treats both alike.
    pub delta: Option<ChoiceDelta>,
    pub usage: Option<CompletionUsage>,
    /// `FlushSentinel`: finish the current TTS segment at this point in the stream.
    ///
    /// Only an `Agent.llm_node` override produces one. Read on the provider path with `default`
    /// (a `ChatChunk` has no `flush` attribute); never emitted on the inference path.
    // Excluded from the Python class: the reference has no `flush` field.
    #[binding(skip, default)]
    #[python(skip)]
    pub flush: bool,
}

/// `ChatChunk`'s host model asks whether a chunk delivered visible generation.
#[crate::host(export)]
pub fn chat_chunk_has_response(chunk: LlmChunk) -> bool {
    chunk.has_response()
}

impl LlmChunk {
    /// Truthiness, as the reference: `{"content": ""}` (common on a final chunk) has no response.
    pub fn has_response(&self) -> bool {
        self.delta.as_ref().is_some_and(|delta| {
            delta.content.as_deref().is_some_and(|text| !text.is_empty())
                || !delta.tool_calls.is_empty()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: reading `content: Some("")` or an empty tool_calls list as a response.
    #[test]
    fn an_empty_delta_is_not_a_response() {
        let mut chunk = LlmChunk { id: "c1".into(), ..Default::default() };
        assert!(!chunk.has_response(), "no delta at all");

        chunk.delta = Some(ChoiceDelta::default());
        assert!(!chunk.has_response(), "empty delta");

        chunk.delta = Some(ChoiceDelta { content: Some(String::new()), ..Default::default() });
        assert!(!chunk.has_response(), "content is the empty string, which Python reads as false");

        chunk.delta = Some(ChoiceDelta { role: Some(Role::Assistant), ..Default::default() });
        assert!(!chunk.has_response(), "a role-only opening chunk carries nothing to show");

        let usage_only = LlmChunk {
            id: "c1".into(),
            usage: Some(CompletionUsage { total_tokens: 42, ..Default::default() }),
            ..Default::default()
        };
        assert!(!usage_only.has_response());

        // Content or tool calls each count on their own (`or`, not `and`).
        chunk.delta = Some(ChoiceDelta { content: Some("hi".into()), ..Default::default() });
        assert!(chunk.has_response());
        chunk.delta = Some(ChoiceDelta {
            tool_calls: vec![FunctionToolCall { name: "lookup".into(), call_id: "1".into(), ..Default::default() }],
            ..Default::default()
        });
        assert!(chunk.has_response());
    }
}
