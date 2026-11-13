use super::*;
use serde_json::json;

fn chunk(v: serde_json::Value) -> WireChunk {
    serde_json::from_value(v).unwrap()
}

fn parse(acc: &mut ChoiceAccumulator, v: serde_json::Value) -> Option<LlmChunk> {
    let c = chunk(v);
    let choice = c.choices.first().cloned().unwrap_or_default();
    acc.parse_choice(&c.id, &choice)
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
        .parse_choice(&c.id, c.choices.first().unwrap())
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

/// Argument fragments before the tool name and null name/argument fragments still deliver each
/// call once with its exact name and arguments. https://github.com/livekit/agents/issues/8248
/// https://github.com/livekit/agents/issues/864
#[test]
fn arguments_before_tool_name_are_preserved() {
    let mut acc = ChoiceAccumulator::new();
    let mut emitted = Vec::new();
    for delta in [
        json!({"index": 0, "function": {"arguments": "{\"stray\""}}),
        json!({"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": null}}),
        json!({"index": 0, "function": {"name": null, "arguments": null}}),
        json!({"index": 0, "function": {"arguments": "{\"city\""}}),
        json!({"index": 0, "function": {"name": "", "arguments": ":\"Paris\"}"}}),
        json!({"index": 1, "id": "call_2", "function": {"name": "get_time", "arguments": null}}),
        json!({"index": 1, "function": {"arguments": "{}"}}),
    ] {
        let got =
            parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"tool_calls": [delta]}}]}));
        emitted.extend(got.iter().flat_map(|c| calls(c).to_vec()));
    }
    let done = parse(
        &mut acc,
        json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    );
    emitted.extend(done.iter().flat_map(|c| calls(c).to_vec()));
    let got: Vec<_> = emitted
        .iter()
        .map(|c| (c.call_id.as_str(), c.name.as_str(), c.arguments.as_str()))
        .collect();
    assert_eq!(
        got,
        [("call_1", "get_weather", r#"{"city":"Paris"}"#), ("call_2", "get_time", "{}")]
    );
}

/// List-shaped content parts (`[{"type": "text", "text": ...}]`) decode like scalar chunks.
/// https://github.com/livekit/agents/issues/6323
#[test]
#[ignore = "https://github.com/livekit/agents/issues/6323 reproduces: list-shaped delta content fails to decode the chunk"]
fn structured_text_parts_decode_like_scalar_content() {
    let mut acc = ChoiceAccumulator::new();
    let mut visible = String::new();
    for parts in [
        json!([{"type": "text", "text": "<thi"}]),
        json!([{"type": "text", "text": "nk>plan"}, {"type": "text", "text": "</think>Hel"}]),
        json!([{"type": "text", "text": "lo"}]),
    ] {
        let c: WireChunk =
            serde_json::from_value(json!({"id": "c", "choices": [{"delta": {"content": parts}}]}))
                .expect("list-shaped content decodes");
        let choice = c.choices.first().cloned().unwrap_or_default();
        if let Some(chunk) = acc.parse_choice(&c.id, &choice) {
            visible.push_str(text_of(&chunk).unwrap_or_default());
        }
    }
    assert_eq!(visible, "<think>plan</think>Hello");
}

/// A tool call after a text preamble is visible before `finish_reason`, so the preamble's speech
/// is not held until the arguments finish. https://github.com/livekit/agents/issues/5826
#[test]
#[ignore = "https://github.com/livekit/agents/issues/5826 reproduces: tool-only deltas emit nothing until finish_reason"]
fn tool_boundary_is_visible_before_completion() {
    let mut acc = ChoiceAccumulator::new();
    let preamble =
        parse(&mut acc, json!({"id": "c", "choices": [{"delta": {"content": "One moment."}}]}));
    assert_eq!(preamble.as_ref().and_then(text_of), Some("One moment."));
    let mut before_finish = Vec::new();
    for delta in [
        json!({"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": ""}}),
        json!({"index": 0, "function": {"arguments": "{\"q\":"}}),
        json!({"index": 0, "function": {"arguments": "\"x\"}"}}),
    ] {
        before_finish.extend(parse(
            &mut acc,
            json!({"id": "c", "choices": [{"delta": {"tool_calls": [delta]}}]}),
        ));
    }
    assert!(!before_finish.is_empty(), "nothing marks the tool call until the stream finishes");
    let done = parse(
        &mut acc,
        json!({"id": "c", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    )
    .expect("the finish reason closes the call");
    assert_eq!(calls(&done).len(), 1);
    assert_eq!(calls(&done)[0].arguments, r#"{"q":"x"}"#);
}

// Catches: a stream that ends without a `finish_reason` dropping the call it was streaming.
#[test]
fn a_call_still_open_when_the_stream_ends_is_closed() {
    let mut acc = ChoiceAccumulator::new();
    parse(
        &mut acc,
        json!({"id": "c", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}
        ]}}]}),
    );
    let done = acc.finish().expect("the open call closes at the end");
    assert_eq!(calls(&done)[0].call_id, "call_1");
    assert!(acc.finish().is_none(), "closed once");
}
