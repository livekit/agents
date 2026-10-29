//! The LLM pipeline events (`llm/llm.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::llm::chat_context::{Extra, Role};

// The Python `module` and `path` of every record here.
#[crate::host(file(module = "llm", path = "livekit.agents.llm"))]
const _: () = ();

// `llm/llm.py:35`. Field `///`s are the reference's, verbatim.
#[crate::host(record(module = "llm", pydantic, emit, extract))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletionUsage {
    /// The number of tokens in the completion.
    pub completion_tokens: u32,
    /// The number of input tokens used (includes cached tokens).
    pub prompt_tokens: u32,
    /// The number of cached input tokens used.
    #[host(default = 0)]
    pub prompt_cached_tokens: u32,
    /// The number of tokens used to create the cache.
    #[host(default = 0)]
    pub cache_creation_tokens: u32,
    /// The number of tokens read from the cache.
    #[host(default = 0)]
    pub cache_read_tokens: u32,
    /// The number of completion tokens spent on hidden reasoning.
    /// Already counted in ``completion_tokens``; do not add it to totals. Only reported by
    /// providers that break reasoning out separately (e.g. OpenAI-compatible responses
    /// carrying ``completion_tokens_details.reasoning_tokens``), 0 everywhere else.
    #[host(default = 0)]
    pub reasoning_tokens: u32,
    /// The total number of tokens used (completion + prompt tokens).
    // Reported by the provider, not derived: providers disagree on cache reads.
    pub total_tokens: u32,
    /// The service tier used for processing the request (e.g. 'default', 'priority', 'flex').
    /// Returned by providers that support tiered processing (e.g. OpenAI).
    #[host(default = None)]
    pub service_tier: Option<String>,
}

// `llm/llm.py:53`. `arguments` is the raw, possibly partial JSON text; validated after the stream.
#[crate::host(record(module = "llm", pydantic, constant(type = "function"), emit, extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FunctionToolCall {
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    /// Provider-specific extra data (e.g., Google thought signatures).
    #[host(default = None)]
    pub extra: Option<Extra>,
}

// `llm/llm.py:71`.
#[crate::host(record(module = "llm", pydantic, emit, extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChoiceDelta {
    #[host(default = None, host_type = "..llm.chat_context.ChatRole")]
    pub role: Option<Role>,
    #[host(default = None)]
    pub content: Option<String>,
    #[host(default_factory = "list")]
    pub tool_calls: Vec<FunctionToolCall>,
    /// Provider-specific extra data (e.g., Google thought signatures).
    #[host(default = None)]
    pub extra: Option<Extra>,
}

// What `LLMStream.collect()` gathers: built by the host, the generated stream's `collect()`.
// The reference has no class docstring.
#[crate::host(record(module = "llm", pydantic))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CollectedResponse {
    #[host(default = "")]
    pub text: String,
    #[host(default_factory = "list")]
    pub tool_calls: Vec<FunctionToolCall>,
    #[host(default = None)]
    pub usage: Option<CompletionUsage>,
    /// Provider-specific extra data accumulated across chunks
    /// (e.g., xAI encrypted reasoning, Google thought signatures).
    #[host(default_factory = "dict")]
    pub extra: Extra,
}

// `llm/llm.py:79`. `has_response` is a host method hook: a core call per token would be a GIL
// crossing per token.
#[crate::host(record(module = "llm", name = "ChatChunk", pydantic, emit, extract))]
#[python(
    method(
        name = "has_response",
        ret = "bool",
        hook = "chat_chunk_has_response",
        doc = "Whether this chunk delivered generation the caller can see.\n\nToken counts and provider metadata (a gateway deployment stamp, a thought\nsignature) reach the caller without being output: they neither start the\nclock on time-to-first-token nor give a retry anything to duplicate."
    )
)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmChunk {
    pub id: String,
    #[host(default = None)]
    pub delta: Option<ChoiceDelta>,
    #[host(default = None)]
    pub usage: Option<CompletionUsage>,
    /// `FlushSentinel`: finish the current TTS segment at this point in the stream.
    ///
    /// Only an `Agent.llm_node` override produces one; never crosses as a field.
    #[host(skip)]
    pub flush: bool,
    /// Text an `Agent.llm_node` answered itself (a `str`), not a model's: spoken, never metered as
    /// a model call (the reference meters the `LLMStream`, and such a node opened none).
    #[host(skip)]
    pub node_text: bool,
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
            tool_calls: vec![FunctionToolCall {
                name: "lookup".into(),
                call_id: "1".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert!(chunk.has_response());
    }
}
