use super::thinking::{FilterState, strip_thinking_tokens};
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{ChoiceDelta, Extra, FunctionToolCall, LlmChunk, Role};
use serde::Deserialize;

// Several branches are truthiness tests on strings providers routinely send empty.

/// `#[serde(default)]` covers only an absent key; providers send `"tool_calls": null` (and the STT
/// gateway `"words": null`), which would fail the whole message.
pub(crate) fn null_as_empty<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(d)?.unwrap_or_default())
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub choices: Vec<WireChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChoice {
    /// Azure content filtering sends a choice with no delta (livekit/agents#688).
    #[serde(default)]
    pub delta: Option<WireDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub tool_calls: Vec<WireToolCall>,
    /// Google thought signatures on text parts, via the gateway's OpenAI-compatible shim.
    #[serde(default)]
    pub extra_content: Option<Extra>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireToolCall {
    #[serde(default)]
    pub index: Option<i64>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<WireFunction>,
    #[serde(default)]
    pub extra_content: Option<Extra>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireUsage {
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Option<WirePromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<WireCompletionTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WirePromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireCompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

/// [`strip_thinking_tokens`] over one stream: the tags and the state carried across
/// SSE events. `None` means nothing visible yet; `Some("")` only for an empty input.
#[derive(Debug, Clone)]
pub struct TagFilter {
    start_tag: &'static str,
    end_tag: &'static str,
    state: FilterState,
}

impl TagFilter {
    pub fn new(start_tag: &'static str, end_tag: &'static str) -> Self {
        TagFilter { start_tag, end_tag, state: FilterState::default() }
    }

    pub fn for_model(model: &str) -> Self {
        let (start, end) = crate::llm::models::think_tags(model);
        TagFilter::new(start, end)
    }

    pub fn strip(&mut self, content: Option<String>, is_final: bool) -> Option<String> {
        let state = std::mem::take(&mut self.state);
        let (start, end) = (self.start_tag, self.end_tag);
        let out = strip_thinking_tokens(content.as_deref(), start, end, state, is_final);
        self.state = out.state;
        out.visible
    }
}

/// The five fields `_run` resets per attempt, and the logic that folds choices into them.
#[derive(Debug, Default)]
pub struct ChoiceAccumulator {
    tool_call_id: Option<String>,
    fnc_name: Option<String>,
    fnc_raw_arguments: Option<String>,
    tool_extra: Option<Extra>,
    tool_index: Option<i64>,
}

impl ChoiceAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// `_parse_choice`. `None` means "this choice produced nothing to forward".
    pub fn parse_choice(
        &mut self,
        id: &str,
        choice: &WireChoice,
        filter: &mut TagFilter,
    ) -> Option<LlmChunk> {
        // livekit/agents#688: Azure content filtering sends a choice with no delta.
        let delta = choice.delta.as_ref()?;

        let content = filter.strip(delta.content.clone(), choice.finish_reason.is_some());

        // Every call the delta closes, in order: a provider may send several calls in one delta.
        let mut closed = Vec::new();
        for tool in &delta.tool_calls {
            // `if not tool.function: continue` (some providers send only `{"index": 0}`).
            let Some(function) = tool.function.as_ref() else { continue };

            // An empty id does not open a new call: providers send `"id": ""` on continuations.
            let boundary = self.tool_call_id.as_deref().is_some_and(|s| !s.is_empty())
                && tool.id.as_deref().is_some_and(|s| !s.is_empty())
                && tool.index != self.tool_index;

            if boundary {
                closed.push(self.take_call());
            }

            // `if tool.function.name:` -- an empty name is a continuation.
            if function.name.as_deref().is_some_and(|s| !s.is_empty()) {
                self.tool_index = tool.index;
                self.tool_call_id = tool.id.clone();
                self.fnc_name = function.name.clone();
                // `tool.function.arguments or ""`: start the accumulator at "".
                self.fnc_raw_arguments = Some(function.arguments.clone().unwrap_or_default());
                self.tool_extra = tool.extra_content.clone();
            } else if function.arguments.as_deref().is_some_and(|s| !s.is_empty()) {
                // `elif tool.function.arguments:` -- an empty fragment is skipped.
                self.fnc_raw_arguments
                    .get_or_insert_with(String::new)
                    .push_str(function.arguments.as_deref().unwrap_or_default());
            }
        }

        let finished_with_call = matches!(choice.finish_reason.as_deref(), Some("tool_calls") | Some("stop"))
            // An empty accumulated id means there is no call to close.
            && self.tool_call_id.as_deref().is_some_and(|s| !s.is_empty());
        if finished_with_call {
            closed.push(self.take_call());
        }
        if !closed.is_empty() {
            return Some(LlmChunk {
                id: id.to_owned(),
                delta: Some(ChoiceDelta {
                    role: Some(Role::Assistant),
                    content,
                    tool_calls: closed,
                    extra: delta.extra_content.clone(),
                }),
                ..Default::default()
            });
        }

        // `if not delta.content and not delta_extra`: an empty delta would start the TTFT clock.
        let has_content = content.as_deref().is_some_and(|s| !s.is_empty());
        let has_extra = delta.extra_content.as_ref().is_some_and(|e| !e.is_empty());
        if !has_content && !has_extra {
            return None;
        }

        Some(LlmChunk {
            id: id.to_owned(),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                content,
                tool_calls: Vec::new(),
                extra: delta.extra_content.clone(),
            }),
            ..Default::default()
        })
    }

    /// `tool_index` is not cleared: the next boundary compares against it.
    fn take_call(&mut self) -> FunctionToolCall {
        FunctionToolCall {
            name: self.fnc_name.take().unwrap_or_default(),
            arguments: self.fnc_raw_arguments.take().unwrap_or_default(),
            call_id: self.tool_call_id.take().unwrap_or_default(),
            extra: self.tool_extra.take(),
        }
    }
}

/// The usage chunk, present because the request set `stream_options={"include_usage": True}`.
pub fn usage_chunk(chunk: &WireChunk) -> Option<LlmChunk> {
    let usage = chunk.usage.as_ref()?;
    let n = |n: Option<u32>| n.map(u64::from);
    Some(LlmChunk {
        id: chunk.id.clone(),
        delta: None,
        // OpenAI's prompt tokens include the cached ones; its schema reports no cache writes.
        usage: Some(Box::new(TokenUsage {
            input_tokens: n(usage.prompt_tokens).unwrap_or(0),
            output_tokens: n(usage.completion_tokens).unwrap_or(0),
            cache_read_tokens: n(usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)),
            output_reasoning_tokens: n(usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)),
            ..TokenUsage::default()
        })),
        ..Default::default()
    })
}

/// `_core.LlmChoiceParser`: one completion's accumulator and its model's thinking-tag filter.
pub struct ChoiceParser {
    accumulator: ChoiceAccumulator,
    filter: TagFilter,
}

impl lk_agents_abi::llm::inference::ChoiceParser for ChoiceParser {
    fn new(model: String) -> Self {
        ChoiceParser { accumulator: ChoiceAccumulator::new(), filter: TagFilter::for_model(&model) }
    }

    fn parse(
        &mut self,
        id: String,
        choice: serde_json::Value,
    ) -> Result<Option<LlmChunk>, Invalid> {
        let choice: WireChoice =
            serde_json::from_value(choice).map_err(|e| Invalid(format!("choice: {e}")))?;
        Ok(self.accumulator.parse_choice(&id, &choice, &mut self.filter))
    }
}

/// `_core`'s completion wire functions.
pub struct CompletionWire;

impl lk_agents_abi::llm::inference::CompletionWire for CompletionWire {
    fn llm_usage_chunk(chunk: serde_json::Value) -> Result<Option<LlmChunk>, Invalid> {
        let chunk: WireChunk =
            serde_json::from_value(chunk).map_err(|e| Invalid(format!("chunk: {e}")))?;
        Ok(usage_chunk(&chunk))
    }
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
