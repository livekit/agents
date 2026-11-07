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
    assert!(parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"content": ""}}]})).is_none());
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

    let done =
        parse(&mut acc, json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}]}))
            .unwrap();
    assert_eq!(calls(&done)[0].call_id, "call_b", "call_b kept accumulating");
    assert_eq!(calls(&done)[0].arguments, "{");
}

// Catches: returning at the first closed call inside a delta, which lost every call after the
// second one a provider sends in a single delta.
#[test]
fn every_call_in_one_delta_is_assembled() {
    let mut acc = ChoiceAccumulator::new();
    let done = parse(
        &mut acc,
        json!({"id": "c", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "function": {"name": "a", "arguments": "{}"}},
                {"index": 1, "id": "call_b", "function": {"name": "b", "arguments": "{}"}},
                {"index": 2, "id": "call_c", "function": {"name": "c", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}]}),
    )
    .expect("the delta closes its calls");
    let ids: Vec<_> = calls(&done).iter().map(|call| call.call_id.as_str()).collect();
    assert_eq!(ids, ["call_a", "call_b", "call_c"]);
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
        parse(&mut acc, json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}]}))
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

// Catches: details a provider omits read as reported zeros (usage.md: absent stays `None`).
#[test]
fn usage_tolerates_a_provider_that_omits_the_token_details() {
    let c = chunk(json!({
        "id": "c1",
        "usage": {"completion_tokens": 7, "prompt_tokens": 11, "total_tokens": 18}
    }));
    let u = usage_chunk(&c).unwrap().usage.unwrap();
    assert_eq!((u.output_tokens, u.input_tokens, u.total_tokens()), (7, 11, 18));
    assert_eq!((u.cache_read_tokens, u.output_reasoning_tokens), (None, None));

    let c = chunk(json!({
        "id": "c1",
        "usage": {
            "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 4}
        }
    }));
    let u = usage_chunk(&c).unwrap().usage.unwrap();
    assert_eq!((u.cache_read_tokens, u.output_reasoning_tokens), (Some(0), Some(4)));

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
        assert_eq!(
            strip_all(&mut TagFilter::new("<think>", "</think>"), chunks),
            want,
            "{chunks:?}"
        );
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
