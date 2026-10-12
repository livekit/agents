//! `LLMStream._parse_choice` and the usage chunk, as a pure accumulator over SSE chunks. Several
//! branches are truthiness tests on strings providers routinely send empty.

use lk_agents_abi::tokenize::{FilterState, strip_thinking_tokens};
use lk_agents_abi::{ChoiceDelta, Extra, FunctionToolCall, LlmChunk, Role};
use serde::Deserialize;

// Only the fields the reference reads, `#[serde(default)]` everywhere.

/// `#[serde(default)]` covers only an absent key; some providers send `"tool_calls": null`, which
/// would fail the whole chunk.
fn null_as_empty<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
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
    /// `getattr(chunk, "service_tier", None)` -- read off the chunk, not off `usage`.
    #[serde(default)]
    pub service_tier: Option<String>,
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
    pub total_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Option<WirePromptTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WirePromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

/// `llm/utils.py`'s `strip_thinking_tokens` over one stream: the tags and the state carried across
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

        for tool in &delta.tool_calls {
            // `if not tool.function: continue` (some providers send only `{"index": 0}`).
            let Some(function) = tool.function.as_ref() else { continue };

            // An empty id does not open a new call: providers send `"id": ""` on continuations.
            let boundary = self.tool_call_id.as_deref().is_some_and(|s| !s.is_empty())
                && tool.id.as_deref().is_some_and(|s| !s.is_empty())
                && tool.index != self.tool_index;

            let call_chunk = boundary
                .then(|| self.take_call_chunk(id, content.clone(), delta.extra_content.clone()));

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

            if let Some(chunk) = call_chunk {
                // The reference returns from inside the loop; the next delta continues the call.
                return Some(chunk);
            }
        }

        let finished_with_call = matches!(choice.finish_reason.as_deref(), Some("tool_calls") | Some("stop"))
            // An empty accumulated id means there is no call to close.
            && self.tool_call_id.as_deref().is_some_and(|s| !s.is_empty());
        if finished_with_call {
            return Some(self.take_call_chunk(id, content, delta.extra_content.clone()));
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
            usage: None,
            flush: false,
        })
    }

    /// `tool_index` is not cleared: the next boundary compares against it.
    fn take_call_chunk(
        &mut self,
        id: &str,
        content: Option<String>,
        extra: Option<Extra>,
    ) -> LlmChunk {
        let call = FunctionToolCall {
            name: self.fnc_name.take().unwrap_or_default(),
            arguments: self.fnc_raw_arguments.take().unwrap_or_default(),
            call_id: self.tool_call_id.take().unwrap_or_default(),
            extra: self.tool_extra.take(),
        };
        LlmChunk {
            id: id.to_owned(),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                content,
                tool_calls: vec![call],
                extra,
            }),
            ..Default::default()
        }
    }
}

/// The usage chunk, present because the request set `stream_options={"include_usage": True}`.
pub fn usage_chunk(chunk: &WireChunk) -> Option<LlmChunk> {
    let usage = chunk.usage.as_ref()?;
    let cached = usage.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens).unwrap_or(0);
    Some(LlmChunk {
        id: chunk.id.clone(),
        delta: None,
        usage: Some(lk_agents_abi::CompletionUsage {
            completion_tokens: usage.completion_tokens.unwrap_or(0),
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            prompt_cached_tokens: cached,
            // OpenAI's schema has no cache-creation/read equivalent.
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            total_tokens: usage.total_tokens.unwrap_or(0),
            service_tier: chunk.service_tier.clone(),
        }),
        flush: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk(v: serde_json::Value) -> WireChunk {
        serde_json::from_value(v).unwrap()
    }

    fn parse(acc: &mut ChoiceAccumulator, v: serde_json::Value) -> Option<LlmChunk> {
        let c = chunk(v);
        let choice = c.choices.first().cloned().unwrap_or_default();
        acc.parse_choice(&c.id, &choice, &mut TagFilter::new("<think>", "</think>"))
    }

    fn text_of(chunk: &LlmChunk) -> Option<&str> {
        chunk.delta.as_ref()?.content.as_deref()
    }

    fn calls(chunk: &LlmChunk) -> &[FunctionToolCall] {
        chunk.delta.as_ref().map(|d| d.tool_calls.as_slice()).unwrap_or_default()
    }

    // Catches: `"tool_calls": null` failing the whole chunk (verbatim gateway frame, 08-30), which
    // silently dropped every content delta.
    #[test]
    fn a_null_tool_calls_list_decodes_as_empty() {
        let c: WireChunk = serde_json::from_str(
            r#"{"service_tier":"default","id":"e01351cd","object":"chat.completion.chunk",
                "created":1788150173,"model":"livekit/gemma-4-31b-it",
                "choices":[{"index":0,"delta":{"role":null,"content":"Hi",
                "reasoning_content":null,"tool_calls":null},"logprobs":null,
                "finish_reason":null}]}"#,
        )
        .expect("a null tool_calls must not fail the chunk");
        let mut acc = ChoiceAccumulator::new();
        let got = acc
            .parse_choice(&c.id, c.choices.first().unwrap(), &mut TagFilter::new("<think>", "</think>"))
            .expect("the content delta must survive");
        assert_eq!(text_of(&got), Some("Hi"));
        // The same for `choices`.
        let c: WireChunk = serde_json::from_str(r#"{"id":"c","choices":null}"#).unwrap();
        assert!(c.choices.is_empty());
    }

    // Catches: forwarding `{"content": ""}` as a chunk.
    #[test]
    fn an_empty_content_delta_produces_no_chunk() {
        let mut acc = ChoiceAccumulator::new();
        assert!(
            parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"content": ""}}]})).is_none()
        );
        assert!(parse(&mut acc, json!({"id": "c", "choices": [{"delta": {}}]})).is_none());
        let got = parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"content": "hi"}}]}));
        assert_eq!(text_of(got.as_ref().unwrap()), Some("hi"));
    }

    // Catches: dropping the Azure `delta: null` guard (livekit/agents#688).
    #[test]
    fn a_choice_with_no_delta_is_skipped() {
        let mut acc = ChoiceAccumulator::new();
        let got = parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": null, "finish_reason": "content_filter"}]}),
        );
        assert!(got.is_none());
    }

    // Catches: reading `"id": ""` as a new tool call, splitting one call into truncated fragments;
    // the renumbered fragment makes the empty-id check the only guard.
    #[test]
    fn empty_ids_on_continuation_deltas_do_not_open_a_new_tool_call() {
        let mut acc = ChoiceAccumulator::new();
        assert!(
            parse(
                &mut acc,
                json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
                ]}}]})
            )
            .is_none()
        );
        for (index, frag) in [(0, "{\"city\""), (1, ":\"Paris\"}")] {
            assert!(
                parse(
                    &mut acc,
                    json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                        {"index": index, "id": "", "function": {"name": "", "arguments": frag}}
                    ]}}]})
                )
                .is_none(),
                "a continuation fragment must not emit"
            );
        }
        let done = parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        )
        .expect("the finish reason closes the call");
        assert_eq!(calls(&done).len(), 1);
        assert_eq!(calls(&done)[0].name, "get_weather");
        assert_eq!(calls(&done)[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(calls(&done)[0].call_id, "call_1");
    }

    // Catches: flushing the previous call on `index` alone, or not flushing at all.
    #[test]
    fn a_new_index_with_a_real_id_flushes_the_previous_call() {
        let mut acc = ChoiceAccumulator::new();
        parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "function": {"name": "a", "arguments": "{}"}}
            ]}}]}),
        );
        let flushed = parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "call_b", "function": {"name": "b", "arguments": "{"}}
            ]}}]}),
        )
        .expect("call_a is emitted when call_b opens");
        assert_eq!(calls(&flushed)[0].call_id, "call_a");
        assert_eq!(calls(&flushed)[0].arguments, "{}");

        let done = parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}]}),
        )
        .unwrap();
        assert_eq!(calls(&done)[0].call_id, "call_b", "call_b kept accumulating");
        assert_eq!(calls(&done)[0].arguments, "{");
    }

    // Catches: closing an accumulated call only on `finish_reason == "tool_calls"`.
    #[test]
    fn both_stop_and_tool_calls_close_an_accumulated_call() {
        for reason in ["stop", "tool_calls"] {
            let mut acc = ChoiceAccumulator::new();
            parse(
                &mut acc,
                json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "x", "function": {"name": "f", "arguments": "{}"}}
                ]}}]}),
            );
            let done = parse(
                &mut acc,
                json!({"id": "c", "choices": [{"delta": {}, "finish_reason": reason}]}),
            );
            assert_eq!(calls(done.as_ref().unwrap())[0].name, "f", "reason={reason}");
        }
        // ...and a finish reason with nothing accumulated emits nothing.
        let mut acc = ChoiceAccumulator::new();
        assert!(
            parse(
                &mut acc,
                json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}]})
            )
            .is_none()
        );
    }

    // Catches: dropping `extra_content` (Google thought signatures) on a content-less delta.
    #[test]
    fn a_delta_with_only_extra_content_is_still_forwarded() {
        let mut acc = ChoiceAccumulator::new();
        let got = parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {"content": "", "extra_content": {"sig": "abc"}}}]}),
        )
        .expect("an extra alone is a chunk");
        assert_eq!(got.delta.as_ref().unwrap().extra.as_ref().unwrap()["sig"], json!("abc"));
        // ...but an *empty* extra dict is falsy and is not (R2b).
        let mut acc = ChoiceAccumulator::new();
        assert!(
            parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"extra_content": {}}}]}))
                .is_none()
        );
    }

    // Catches: service_tier read off `usage`, or missing cached-token details not read as 0.
    #[test]
    fn usage_tolerates_a_provider_that_omits_the_token_details() {
        let c = chunk(json!({
            "id": "c1",
            "service_tier": "flex",
            "usage": {"completion_tokens": 7, "prompt_tokens": 11, "total_tokens": 18}
        }));
        let u = usage_chunk(&c).unwrap().usage.unwrap();
        assert_eq!((u.completion_tokens, u.prompt_tokens, u.total_tokens), (7, 11, 18));
        assert_eq!(u.prompt_cached_tokens, 0);
        assert_eq!(u.service_tier.as_deref(), Some("flex"));

        assert!(usage_chunk(&chunk(json!({"id": "c1"}))).is_none(), "no usage, no chunk");
    }

    // Catches: emitting the usage chunk with a delta (it would read as a response and block
    // retry).
    #[test]
    fn a_usage_chunk_carries_no_delta() {
        let c = chunk(json!({"id": "c1", "usage": {"total_tokens": 3}}));
        let got = usage_chunk(&c).unwrap();
        assert!(got.delta.is_none());
        assert!(!got.has_response());
    }

    // ------------------------------------------------------------------ the thinking filter

    fn strip_all(filter: &mut TagFilter, chunks: &[&str]) -> String {
        let mut out = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let is_final = i + 1 == chunks.len();
            if let Some(visible) = filter.strip(Some((*chunk).to_owned()), is_final) {
                out.push_str(&visible);
            }
        }
        out
    }

    // Catches, per row: the reasoning block emitted; a tag matched per chunk rather than across
    // them; a partial marker held forever when it is not a tag (or when the stream ends on it); a
    // stream ending mid-block emitting what it held; the text between two blocks lost; a panic on a
    // multi-byte boundary.
    #[test]
    fn the_thinking_filter_hides_exactly_the_blocks() {
        for (chunks, want) in [
            (&["<think>I should be brief</think>Hello there"][..], "Hello there"),
            (&["<thi", "nk>hidden", " more hidden</thi", "nk>", "visible"], "visible"),
            (&["if a < b then", " done"], "if a < b then done"),
            (&["trailing <thi"], "trailing <thi"),
            (&["visible<think>cut off here"], "visible"),
            (&["<think>one</think>middle<think>two</think>end"], "middleend"),
            (&["héllo 🎉 wörld"], "héllo 🎉 wörld"),
        ] {
            assert_eq!(strip_all(&mut TagFilter::new("<think>", "</think>"), chunks), want, "{chunks:?}");
        }
    }

    // Catches: `Some("")` collapsing to `None`.
    #[test]
    fn an_empty_delta_stays_an_empty_delta_and_a_hidden_one_stays_none() {
        let mut f = TagFilter::new("<think>", "</think>");
        assert_eq!(f.strip(Some(String::new()), false), Some(String::new()));
        // Inside a block: content arrived, nothing is visible.
        assert_eq!(f.strip(Some("<think>hidden".to_owned()), false), None);
        assert_eq!(f.strip(None, false), None);
    }

    // Catches: the tag table being ignored (Gemma's delimiters are not `<think>`).
    #[test]
    fn for_model_takes_the_tags_from_the_table() {
        let mut f = TagFilter::for_model("google/gemma-4-31b-it");
        let (start, end) = crate::llm::models::think_tags("google/gemma-4-31b-it");
        let got = strip_all(&mut f, &[&format!("{start}hidden{end}shown")]);
        assert_eq!(got, "shown", "the model's own tags were not used");
    }
}
