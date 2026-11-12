//! Regression tests for how the OpenAI realtime machine ends, splits and correlates responses.

use super::testfix::{RealtimeMachine, Step, machine, msg};
use super::wire;
use crate::realtime::dialect::{OPENAI, XAI};
use crate::realtime::machine::OptionsPatch;
use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::llm::realtime::*;
use lk_agents_abi::{ChatContext, ChatItem, Maybe, ToolChoice, Ts};
use serde_json::{Value, json};

fn at(secs: u64) -> Ts {
    Ts::from_nanos(secs * 1_000_000_000)
}

fn recv(m: &mut RealtimeMachine, now: Ts, ev: Value) -> Step {
    m.incoming(now, wire::decode(&ev))
}

fn reply(req_id: &str) -> Command {
    Command::GenerateReply(GenerateReply {
        req_id: req_id.into(),
        instructions: None,
        tool_choice: None,
    })
}

fn created(id: &str, req_id: Option<&str>) -> Value {
    let mut response = json!({"id": id});
    if let Some(req_id) = req_id {
        response["metadata"] = json!({"client_event_id": req_id});
    }
    json!({"type": "response.created", "response": response})
}

fn message_added(id: &str) -> Value {
    json!({"type": "response.output_item.added", "item": {"id": id, "type": "message"}})
}

fn done(id: &str, status: &str) -> Value {
    json!({"type": "response.done", "response": {"id": id, "status": status}})
}

fn types(frames: &[Value]) -> Vec<&str> {
    frames.iter().map(|f| f["type"].as_str().unwrap_or("?")).collect()
}

fn generation_done(step: &Step) -> Vec<GenerationStatus> {
    step.events
        .iter()
        .filter_map(|e| match e {
            Event::GenerationDone(done) => Some(done.status),
            _ => None,
        })
        .collect()
}

/// A failed, cancelled or incomplete `response.done` ends the generation with its status, finite
/// metrics and the provider's error. https://github.com/livekit/agents/issues/4488
/// https://github.com/livekit/agents/issues/5808
#[test]
fn response_status_variants_end_the_generation_safely() {
    use lk_agents_abi::metrics::base::AgentMetrics;
    let cases = [
        (
            json!({"id": "r", "status": "failed", "status_details": {"type": "failed",
                "error": {"type": "server_error", "code": "upstream"}}}),
            GenerationStatus::Failed,
            Some("server_error"),
        ),
        (json!({"id": "r", "status": "failed"}), GenerationStatus::Failed, Some("unknown")),
        (
            json!({"id": "r", "status": "cancelled",
                "status_details": {"type": "cancelled", "reason": "client_cancelled"}}),
            GenerationStatus::Cancelled,
            None,
        ),
        (
            json!({"id": "r", "status": "incomplete",
                "status_details": {"type": "incomplete", "reason": "max_output_tokens"}}),
            GenerationStatus::Incomplete,
            None,
        ),
    ];
    for (response, status, error_type) in cases {
        let mut m = machine(OPENAI).with_provider("api.openai.com".into());
        m.open(at(0));
        m.command(at(1), reply("response_create_1"));
        recv(&mut m, at(1), created("r", Some("response_create_1")));
        // Done in the same instant it was created: a zero duration.
        let step = recv(&mut m, at(1), json!({"type": "response.done", "response": response}));
        assert_eq!(generation_done(&step), [status], "{response}");
        assert!(!m.has_active_generation(), "{response}");
        for event in &step.events {
            if let Event::Metrics(RealtimeMetrics { metrics }) = event
                && let AgentMetrics::RealtimeModel(r) = &**metrics
            {
                assert!(r.duration.is_finite() && r.tokens_per_second.is_finite(), "{r:?}");
            }
        }
        let errors: Vec<_> = step
            .events
            .iter()
            .filter_map(|e| match e {
                Event::Error(error) => Some(error.message.clone()),
                _ => None,
            })
            .collect();
        match error_type {
            Some(kind) => {
                assert_eq!(errors.len(), 1, "{response}: {errors:?}");
                assert!(errors[0].ends_with(kind), "{response}: {errors:?}");
            }
            None => assert!(errors.is_empty(), "{response}: {errors:?}"),
        }
        // The next reply goes out.
        let next = m.command(at(2), reply("response_create_2"));
        assert_eq!(types(&next.frames), ["response.create"], "{response}");
    }
}

/// An OpenAI-compatible server (xAI) that sends `status_details` as a bare string still ends the
/// generation instead of failing the frame. https://github.com/livekit/agents/issues/5993
#[test]
#[ignore = "https://github.com/livekit/agents/issues/5993 reproduces: a string status_details fails decoding of response.done"]
fn string_status_details_still_end_the_generation() {
    let event = json!({"type": "response.done",
        "response": {"id": "r", "status": "incomplete", "status_details": "incomplete"}});
    let decoded = wire::decode_at(&event, std::time::SystemTime::UNIX_EPOCH);
    assert!(decoded.is_ok(), "{:?}", decoded.err());
    let mut m = machine(XAI);
    m.open(at(0));
    recv(&mut m, at(1), created("r", None));
    let step = m.incoming(at(2), decoded.expect("decoded"));
    assert_eq!(generation_done(&step), [GenerationStatus::Incomplete]);
}

/// A text-only response with a preamble then the answer surfaces each message with its own text
/// and ends the generation once, after the last. https://github.com/livekit/agents/issues/1844
/// https://github.com/livekit/agents/issues/5684
#[test]
fn text_and_multi_message_responses_preserve_each_message() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));
    let mut steps = vec![recv(&mut m, at(1), created("r", Some("response_create_1")))];
    for (id, text) in [("pre", "One moment."), ("ans", "It is noon.")] {
        steps.push(recv(&mut m, at(1), message_added(id)));
        steps.push(recv(
            &mut m,
            at(1),
            json!({"type": "response.content_part.added", "item_id": id,
                "part": {"type": "output_text"}}),
        ));
        steps.push(recv(
            &mut m,
            at(1),
            json!({"type": "response.output_text.delta", "item_id": id, "delta": text}),
        ));
        let closed = recv(
            &mut m,
            at(1),
            json!({"type": "response.output_item.done", "item": {
                "id": id, "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}]}}),
        );
        assert!(generation_done(&closed).is_empty(), "the generation is still open after {id}");
        steps.push(closed);
    }
    steps.push(recv(&mut m, at(2), done("r", "completed")));

    let lifecycle: Vec<(&str, &str)> = steps
        .iter()
        .flat_map(|s| &s.events)
        .filter_map(|e| match e {
            Event::MessageStarted(m) => Some(("started", m.message_id.as_str())),
            Event::MessageDone(m) => Some(("done", m.message_id.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        lifecycle,
        [("started", "pre"), ("done", "pre"), ("started", "ans"), ("done", "ans")]
    );
    let text: Vec<(String, String)> = steps
        .iter()
        .flat_map(|s| &s.events)
        .filter_map(|e| match e {
            Event::TextDelta(t) => Some((t.message_id.clone(), t.delta.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(text, [("pre".into(), "One moment.".into()), ("ans".into(), "It is noon.".into())]);
    let modalities: Vec<_> = steps
        .iter()
        .flat_map(|s| &s.events)
        .filter_map(|e| match e {
            Event::MessageModalities(m) => Some((m.message_id.as_str(), m.modalities)),
            _ => None,
        })
        .collect();
    assert_eq!(modalities, [("pre", Modalities::TextOnly), ("ans", Modalities::TextOnly)]);
    let ends: Vec<_> = steps.iter().flat_map(generation_done).collect();
    assert_eq!(ends, [GenerationStatus::Completed]);
}

/// Each assistant turn lands in the conversation mirror once with its transcript, and earlier
/// turns stay. https://github.com/livekit/agents/issues/4652 https://github.com/livekit/agents/issues/3208
#[test]
fn assistant_turns_reach_history_once_and_are_retained() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let mut steps = Vec::new();
    let mut previous = Value::Null;
    for (turn, (user, assistant, text)) in
        [("u1", "a1", "Hello there."), ("u2", "a2", "It is noon.")].into_iter().enumerate()
    {
        recv(
            &mut m,
            at(turn as u64),
            json!({"type": "conversation.item.added", "previous_item_id": previous, "item": {
                "id": user, "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hi"}]}}),
        );
        let response = format!("r{turn}");
        steps.push(recv(&mut m, at(turn as u64), created(&response, None)));
        steps.push(recv(&mut m, at(turn as u64), message_added(assistant)));
        recv(
            &mut m,
            at(turn as u64),
            json!({"type": "conversation.item.added", "previous_item_id": user, "item": {
                "id": assistant, "type": "message", "role": "assistant", "content": []}}),
        );
        steps.push(recv(
            &mut m,
            at(turn as u64),
            json!({"type": "response.output_audio_transcript.delta", "item_id": assistant,
                "delta": text}),
        ));
        steps.push(recv(
            &mut m,
            at(turn as u64),
            json!({"type": "response.output_item.done", "item": {
                "id": assistant, "type": "message", "role": "assistant",
                "content": [{"type": "output_audio", "transcript": text}]}}),
        ));
        steps.push(recv(&mut m, at(turn as u64), done(&response, "completed")));
        steps.push(recv(&mut m, at(turn as u64), done(&response, "completed")));
        previous = json!(assistant);
    }

    let lifecycle: Vec<(&str, &str)> = steps
        .iter()
        .flat_map(|s| &s.events)
        .filter_map(|e| match e {
            Event::MessageStarted(m) => Some(("started", m.message_id.as_str())),
            Event::MessageDone(m) => Some(("done", m.message_id.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(lifecycle, [("started", "a1"), ("done", "a1"), ("started", "a2"), ("done", "a2")]);
    assert_eq!(steps.iter().flat_map(generation_done).count(), 2);
    let ctx = m.chat_ctx();
    assert_eq!(ctx.items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["u1", "a1", "u2", "a2"]);
    for (id, text) in [("a1", "Hello there."), ("a2", "It is noon.")] {
        let message = ctx.get_by_id(id).and_then(ChatItem::as_message).expect("assistant message");
        assert_eq!(message.content.len(), 1, "{id}: {:?}", message.content);
        assert_eq!(message.raw_text_content().as_deref(), Some(text));
    }
}

/// An interruption before the first audio delta never sends `conversation.item.truncate` for audio
/// the server never produced, and the next reply goes out.
/// https://github.com/livekit/agents/issues/6157
#[test]
fn interrupt_before_first_audio_never_truncates_uncreated_audio() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));
    recv(&mut m, at(1), created("r", Some("response_create_1")));
    recv(&mut m, at(1), message_added("a1"));
    let mut frames = m.command(at(1), Command::Interrupt(Interrupt { response_id: None })).frames;
    for has_audio in [true, false] {
        frames.extend(
            m.command(
                at(1),
                Command::Truncate(Truncate {
                    message_id: "a1".into(),
                    has_audio,
                    audio_end_ms: 0,
                    audio_transcript: Some(String::new()),
                }),
            )
            .frames,
        );
    }
    assert!(
        !types(&frames).contains(&"conversation.item.truncate"),
        "no audio exists to truncate: {:?}",
        types(&frames)
    );
    recv(&mut m, at(2), done("r", "cancelled"));
    assert_eq!(types(&m.command(at(3), reply("response_create_2")).frames), ["response.create"]);
}

/// A context create the server refuses settles the update instead of stranding it.
/// https://github.com/livekit/agents/issues/7073
#[test]
fn a_refused_context_create_settles_the_update() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let update = m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(vec![msg("u1", "hello")]),
        }),
    );
    let event_id = update.frames[0]["event_id"].as_str().expect("event id").to_owned();

    recv(
        &mut m,
        at(2),
        json!({"type": "error", "error": {"event_id": event_id,
            "code": "invalid_value", "message": "bad item"}}),
    );

    assert!(!m.context_work_pending());
}

/// A reconnect mid-response ends the generation and its open message, and the next reply goes
/// out. https://github.com/livekit/agents/pull/5276
#[test]
fn a_reconnect_mid_response_ends_the_generation_and_its_message() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));
    recv(&mut m, at(1), created("r1", Some("response_create_1")));
    recv(&mut m, at(1), message_added("a1"));

    let step = m.reconnect(at(2));

    assert!(
        step.events.iter().any(|e| matches!(e, Event::MessageDone(d) if d.message_id == "a1")),
        "{:?}",
        step.events
    );
    assert!(!generation_done(&step).is_empty(), "{:?}", step.events);
    assert!(!m.has_active_generation());
    let next = m.command(at(3), reply("response_create_2"));
    assert!(types(&next.frames).contains(&"response.create"), "{:?}", types(&next.frames));
}

/// A partial options update sends only what changed, leaving the session's instructions, tools,
/// voice and turn detection in place. https://github.com/livekit/agents/issues/1400
/// https://github.com/livekit/agents/issues/3440 https://github.com/livekit/agents/issues/3445
#[test]
fn a_partial_options_update_sends_only_what_changed() {
    let mut m = machine(OPENAI);
    m.open(at(0));

    let step = m.update_options(OptionsPatch { speed: Maybe::Given(1.2), ..Default::default() });

    assert_eq!(types(&step.frames), ["session.update"]);
    let patch = &step.frames[0]["session"];
    for unset in ["instructions", "tools", "tool_choice"] {
        assert!(patch.get(unset).is_none(), "{unset} must be omitted: {patch}");
    }
    assert!(patch["audio"]["input"].get("turn_detection").is_none(), "{patch}");
    assert!(patch["audio"]["output"].get("voice").is_none(), "{patch}");
}

/// A per-response `tool_choice` rides on that `response.create` only.
/// https://github.com/livekit/agents/pull/5211
#[test]
fn a_per_response_tool_choice_applies_to_that_response_only() {
    let mut m = machine(OPENAI);
    m.open(at(0));

    let forbidden = m.command(
        at(1),
        Command::GenerateReply(GenerateReply {
            req_id: "response_create_1".into(),
            instructions: None,
            tool_choice: Some(ToolChoice::Forbidden),
        }),
    );
    recv(&mut m, at(2), created("r1", Some("response_create_1")));
    recv(&mut m, at(2), done("r1", "completed"));
    let next = m.command(at(3), reply("response_create_2"));

    assert_eq!(forbidden.frames[0]["response"]["tool_choice"], json!("none"));
    assert!(next.frames[0]["response"].get("tool_choice").is_none(), "{}", next.frames[0]);
}
