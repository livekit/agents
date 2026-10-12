//! The machine against scripted server frames, on both dialects.

use super::*;
use crate::realtime::dialect::{OPENAI, XAI};
use crate::realtime::testfix::{machine, msg};
use lk_agents_abi::llm::realtime::{
    GenerateReply, GenerationCreated, Interrupt, PushAudio, Say, UpdateChatCtx, UpdateInstructions,
};
use lk_agents_abi::{Clock, VirtualClock};
use serde_json::json;

fn at(secs: u64) -> Ts {
    Ts::from_nanos(secs * 1_000_000_000)
}

/// Decode a raw server frame the way the driver does.
fn recv(m: &mut RealtimeMachine, now: Ts, ev: Value) -> Step {
    m.incoming(now, wire::decode(&ev))
}

fn response_created(id: &str, client_event_id: Option<&str>) -> Value {
    let mut response = json!({"id": id});
    if let Some(c) = client_event_id {
        response["metadata"] = json!({"client_event_id": c});
    }
    json!({"type": "response.created", "response": response})
}

fn item_added(prev: Option<&str>, id: &str, text: &str) -> Value {
    json!({
        "type": "conversation.item.added",
        "previous_item_id": prev,
        "item": {
            "id": id, "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    })
}

fn reply(req_id: &str) -> Command {
    Command::GenerateReply(GenerateReply { req_id: req_id.into(), instructions: None, tool_choice: None })
}

fn types(frames: &[Value]) -> Vec<&str> {
    frames.iter().map(|f| f["type"].as_str().unwrap_or("?")).collect()
}

// -----------------------------------------------------------------------------------------
// generate_reply correlation -- the member that needed a design
// -----------------------------------------------------------------------------------------

// Catches: resolving `generate_reply` off whichever `response.created` arrives first instead
// of by `metadata.client_event_id` (server VAD creates untagged responses in between).
#[test]
fn generate_reply_correlation_survives_an_out_of_order_response() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));

    // the server's own VAD answers the user first
    let step = recv(&mut m, at(1), response_created("resp_vad", None));
    assert_eq!(
        step.events,
        [Event::GenerationCreated(GenerationCreated {
            response_id: "resp_vad".into(),
            req_id: None
        })],
        "an untagged turn must not claim the pending request"
    );

    // and only now the reply we asked for
    let step = recv(&mut m, at(2), response_created("resp_mine", Some("response_create_1")));
    let created: Vec<_> = step
        .events
        .iter()
        .filter(|e| matches!(e, Event::GenerationCreated(GenerationCreated { .. })))
        .collect();
    assert_eq!(
        created,
        [&Event::GenerationCreated(GenerationCreated {
            response_id: "resp_mine".into(),
            req_id: Some("response_create_1".into()),
        })]
    );
}

// Catches: forgetting the request id after its timeout (a late turn surfaces as spontaneous),
// or an untargeted `response.cancel`.
#[test]
fn a_reply_abandoned_at_its_deadline_is_cancelled_by_id_when_it_arrives_anyway() {
    let clock = VirtualClock::new();
    let mut m = machine(OPENAI);
    m.open(clock.now());
    m.command(clock.now(), reply("response_create_1"));

    assert_eq!(m.next_deadline(), Some(at(10)), "the 10 s generate_reply timeout");
    clock.advance_to(at(10));
    let step = m.tick(clock.now());
    assert_eq!(
        step.events,
        [Event::GenerateReplyFailed (GenerateReplyFailed {
            req_id: "response_create_1".into(),
            error: RealtimeError::new("generate_reply timed out."),
        })]
    );

    clock.advance_to(at(11));
    let step =
        recv(&mut m, clock.now(), response_created("resp_late", Some("response_create_1")));
    assert_eq!(step.frames, [json!({"type": "response.cancel", "response_id": "resp_late"})]);
    assert!(
        !step
            .events
            .iter()
            .any(|e| matches!(e, Event::GenerationCreated(GenerationCreated { .. }))),
        "a turn the caller gave up on must not be surfaced"
    );

    // and its trailing deltas are skipped rather than delivered to nobody
    let step = recv(
        &mut m,
        clock.now(),
        json!({"type": "response.output_audio.delta", "item_id": "i1", "delta": "AAAA"}),
    );
    assert!(step.events.is_empty(), "the discarded generation swallows its own deltas");
}

// Catches: leaving a rejected `response.create` to time out instead of failing it with the
// provider code.
#[test]
fn a_rejected_response_create_fails_its_request_at_once_and_keeps_the_code() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));

    let step = recv(
        &mut m,
        at(1),
        json!({"type": "error", "error": {
            "event_id": "response_create_1",
            "code": "conversation_already_has_active_response",
            "message": "Conversation already has an active response",
        }}),
    );
    assert!(step.events.contains(&Event::GenerateReplyFailed (GenerateReplyFailed {
        req_id: "response_create_1".into(),
        error: RealtimeError::with_code(
            "Conversation already has an active response",
            Some("conversation_already_has_active_response".into()),
        ),
    })));
    assert_eq!(
        m.next_deadline(),
        Some(at(20 * 60)),
        "the request's deadline goes with it; only the session recycle timer is left"
    );
}

// Catches: `interrupt()` being a no-op between `response.create` and `response.created`.
#[test]
fn interrupt_reaches_a_reply_that_has_been_asked_for_but_not_created() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    assert!(m.command(at(0), Command::Interrupt(Interrupt { response_id: None })).frames.is_empty(), "nothing to interrupt");

    m.command(at(0), reply("response_create_1"));
    assert!(m.has_active_generation());
    assert_eq!(
        m.command(at(1), Command::Interrupt(Interrupt { response_id: None })).frames,
        [json!({"type": "response.cancel"})]
    );
}

// Catches: sending per-response `instructions` alone, dropping the session's system prompt.
#[test]
fn per_response_instructions_are_concatenated_onto_the_session_prompt() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), Command::UpdateInstructions(UpdateInstructions { instructions: "You are terse.".into() }));
    let step = m.command(
        at(0),
        Command::GenerateReply(GenerateReply {
            req_id: "response_create_2".into(),
            instructions: Some("Greet the caller.".into()),
            tool_choice: None,
        }),
    );
    assert_eq!(
        step.frames[0]["response"]["instructions"],
        json!("You are terse.\nGreet the caller.")
    );
}

// -----------------------------------------------------------------------------------------
// remote chat context <-> diff ops
// -----------------------------------------------------------------------------------------

// Catches: wrong anchors, a missing `"root"` literal, or an op order the server cannot apply
// in `_create_update_chat_ctx_events`.
#[test]
fn a_context_update_becomes_the_delete_and_create_ops_its_diff_implies() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    recv(&mut m, at(0), item_added(Some("a"), "b", "second"));
    assert_eq!(m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a", "b"]);

    // drop "b", add "c" after "a"
    let step = m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx {
            req_id: "u1".into(),
            chat_ctx: ChatContext::new(vec![msg("a", "first"), msg("c", "third")]),
        }),
    );
    assert_eq!(types(&step.frames), ["conversation.item.delete", "conversation.item.create"]);
    assert_eq!(step.frames[0]["item_id"], json!("b"));
    assert_eq!(step.frames[1]["item"]["id"], json!("c"));
    assert_eq!(
        step.frames[1]["previous_item_id"],
        json!("a"),
        "the create anchors to the surviving predecessor, not to the deleted one"
    );
    assert!(step.events.is_empty(), "not settled until the server acks both");

    // the acks, in the order the server sends them
    let step =
        recv(&mut m, at(2), json!({"type": "conversation.item.deleted", "item_id": "b"}));
    assert!(step.events.is_empty(), "one of two");
    let step = recv(&mut m, at(2), item_added(Some("a"), "c", "third"));
    assert!(
        step.events.contains(&Event::ChatCtxUpdated { req_id: "u1".into(), rejected: vec![] })
    );
    assert_eq!(m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a", "c"]);
}

// Catches: an unacked update hanging forever instead of the reference's 5 s bound.
#[test]
fn an_unacknowledged_context_update_gives_up_after_five_seconds() {
    let clock = VirtualClock::new();
    let mut m = machine(OPENAI);
    m.open(clock.now());
    recv(&mut m, clock.now(), item_added(None, "a", "first"));
    m.command(
        clock.now(),
        Command::UpdateChatCtx(UpdateChatCtx {
            req_id: "u1".into(),
            chat_ctx: ChatContext::new(vec![msg("a", "first"), msg("b", "new")]),
        }),
    );

    assert_eq!(m.next_deadline(), Some(at(5)), "5 s, ahead of the 20 min recycle timer");
    clock.advance_to(at(5));
    let step = m.tick(clock.now());
    assert_eq!(
        step.events,
        [Event::ChatCtxUpdated {
            req_id: "u1".into(),
            rejected: vec!["update_chat_ctx timed out.".into()],
        }]
    );
    assert_eq!(m.next_deadline(), Some(at(20 * 60)), "only the recycle timer is left");
}

// Catches: treating `item_create_duplicate_item_id` as a rejection.
#[test]
fn a_duplicate_item_id_rejection_counts_as_success() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    m.command(
        at(0),
        Command::UpdateChatCtx(UpdateChatCtx {
            req_id: "u1".into(),
            chat_ctx: ChatContext::new(vec![msg("a", "first"), msg("b", "new")]),
        }),
    );

    let step = recv(
        &mut m,
        at(1),
        json!({"type": "error", "error": {
            "event_id": "chat_ctx_create_2",
            "code": "item_create_duplicate_item_id",
            "message": "Item with that id already exists",
        }}),
    );
    assert_eq!(
        step.events,
        [Event::ChatCtxUpdated { req_id: "u1".into(), rejected: vec![] }],
        "already-there is not a rejection"
    );
}

// Catches: dropping or head-inserting an item anchored to one just deleted (the reference
// falls back to the tail).
#[test]
fn an_item_anchored_to_a_vanished_one_lands_at_the_tail() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    recv(&mut m, at(0), item_added(Some("a"), "b", "second"));
    recv(&mut m, at(0), item_added(Some("ghost"), "c", "third"));
    assert_eq!(
        m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
}

// -----------------------------------------------------------------------------------------
// The xAI dialect's actual deltas
// -----------------------------------------------------------------------------------------

// Catches: expecting `client_event_id` back from xAI's `force_message`, or consuming the FIFO
// say tag more than once.
#[test]
fn xai_matches_a_metadata_less_response_to_the_oldest_say_and_only_once() {
    let mut m = machine(XAI);
    m.open(at(0));
    let step = m.command(at(0), Command::Say(Say { req_id: "say_1".into(), text: "hello".into(), add_to_chat_ctx: true }));
    assert_eq!(step.frames[0]["item"]["type"], json!("force_message"));
    assert_eq!(step.frames[0]["item"]["content"][0]["text"], json!("hello"));

    let step = recv(&mut m, at(1), response_created("resp_1", None));
    assert!(step.events.contains(&Event::GenerationCreated(GenerationCreated {
        response_id: "resp_1".into(),
        req_id: Some("say_1".into()),
    })));

    let step = recv(&mut m, at(2), response_created("resp_2", None));
    let created: Vec<_> = step
        .events
        .iter()
        .filter(|e| matches!(e, Event::GenerationCreated(GenerationCreated { .. })))
        .collect();
    assert_eq!(
        created,
        [&Event::GenerationCreated(GenerationCreated {
            response_id: "resp_2".into(),
            req_id: None
        })],
        "the tag was consumed; a later turn is not another say"
    );
}

// Catches: `say()` silently doing nothing on a dialect without force_message.
#[test]
fn say_on_openai_fails_rather_than_disappearing() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = m.command(at(0), Command::Say(Say { req_id: "say_1".into(), text: "hi".into(), add_to_chat_ctx: true }));
    assert!(step.frames.is_empty());
    assert!(matches!(step.events[0], Event::GenerateReplyFailed (GenerateReplyFailed { .. })));
}

// Catches: finalising xAI's first `transcription.completed`, which is only a prefix.
#[test]
fn xai_holds_a_completed_transcript_until_the_agent_starts_answering() {
    let mut m = machine(XAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "item_1", ""));

    let completed = |t: &str| {
        json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "transcript": t,
        })
    };
    let step = recv(&mut m, at(1), completed("what is"));
    assert_eq!(
        step.events,
        [Event::InputTranscription(InputTranscription {
            item_id: "item_1".into(),
            transcript: "what is".into(),
            is_final: false,
            confidence: None,
            turn_started_at: None,
        })],
        "never final inline on xAI"
    );

    // the revision
    recv(&mut m, at(2), completed("what is the time"));

    // the agent starts speaking: this is the last moment the transcript can change
    let mut turn = recv(&mut m, at(3), response_created("resp_1", None));
    turn.merge(recv(
        &mut m,
        at(3),
        json!({"type": "response.output_audio.delta", "item_id": "msg_1", "delta": "AAAA"}),
    ));
    let finals: Vec<_> = turn
        .events
        .iter()
        .filter_map(|e| match e {
            Event::InputTranscription(t) if t.is_final => Some(t.transcript.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(finals, ["what is the time"], "one final, and it is the revised one");
    assert_eq!(
        m.chat_ctx().get_by_id("item_1").unwrap().as_message().unwrap().raw_text_content(),
        Some("what is the time".to_owned()),
        "and the mirror holds exactly one copy of it"
    );
}

// Catches: appending each xAI revision onto the mirror instead of replacing it.
#[test]
fn xai_replaces_rather_than_appends_the_transcript_it_already_wrote() {
    let mut m = machine(XAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "item_1", ""));
    let completed = |t: &str| {
        json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "transcript": t,
        })
    };
    let audio = json!({
        "type": "response.output_audio.delta", "item_id": "msg_1", "delta": "AAAA",
    });

    recv(&mut m, at(1), completed("hel"));
    recv(&mut m, at(2), response_created("resp_1", None));
    recv(&mut m, at(2), audio.clone());
    recv(&mut m, at(3), completed("hello there"));
    recv(&mut m, at(4), response_created("resp_2", None));
    recv(&mut m, at(4), audio);

    assert_eq!(
        m.chat_ctx().get_by_id("item_1").unwrap().as_message().unwrap().content.len(),
        1,
        "one transcript, not one per flush"
    );
    assert_eq!(
        m.chat_ctx().get_by_id("item_1").unwrap().as_message().unwrap().raw_text_content(),
        Some("hello there".to_owned())
    );
}

// Catches: taking xAI's `conversation.item.deleted` with `item_id: ""` at face value.
#[test]
fn xai_resolves_the_empty_item_id_on_a_delete_ack() {
    let mut m = machine(XAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    recv(&mut m, at(0), item_added(Some("a"), "b", "second"));
    m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx {
            req_id: "u1".into(),
            chat_ctx: ChatContext::new(vec![msg("a", "first")]),
        }),
    );

    let step = recv(&mut m, at(2), json!({"type": "conversation.item.deleted", "item_id": ""}));
    assert_eq!(
        step.events,
        [Event::ChatCtxUpdated { req_id: "u1".into(), rejected: vec![] }],
        "the empty ack answered the delete of `b`"
    );
    assert_eq!(m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a"]);

    // and OpenAI must NOT do this: an empty id there is a bug, not a convention
    let mut m = machine(OPENAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx { req_id: "u1".into(), chat_ctx: ChatContext::empty() }),
    );
    let step = recv(&mut m, at(2), json!({"type": "conversation.item.deleted", "item_id": ""}));
    assert!(step.events.is_empty());
    assert_eq!(m.chat_ctx().items.len(), 1, "nothing was deleted");
}

// Catches: applying OpenAI's root convention (null anchor = head) to xAI (null = tail).
#[test]
fn a_null_anchor_means_the_head_on_openai_and_the_tail_on_xai() {
    let mut oai = machine(OPENAI);
    oai.open(at(0));
    recv(&mut oai, at(0), item_added(None, "a", "first"));
    recv(&mut oai, at(0), item_added(None, "b", "second"));
    assert_eq!(oai.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["b", "a"]);

    let mut xai = machine(XAI);
    xai.open(at(0));
    recv(&mut xai, at(0), item_added(None, "a", "first"));
    recv(&mut xai, at(0), item_added(None, "b", "second"));
    assert_eq!(xai.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a", "b"]);
}

// Catches: surfacing xAI's server-side tool calls as local function calls (OpenAI must keep
// surfacing them).
#[test]
fn xai_ignores_a_call_naming_no_local_tool_and_openai_does_not() {
    let done = json!({"type": "response.output_item.done", "item": {
        "id": "fc_1", "type": "function_call", "call_id": "call_1",
        "name": "web_search", "arguments": "{}",
    }});

    let mut xai = machine(XAI);
    xai.open(at(0));
    recv(&mut xai, at(0), response_created("resp_1", None));
    assert!(recv(&mut xai, at(1), done.clone()).events.is_empty());

    let mut oai = machine(OPENAI);
    oai.open(at(0));
    recv(&mut oai, at(0), response_created("resp_1", None));
    let step = recv(&mut oai, at(1), done);
    assert!(matches!(step.events[0], Event::FunctionCall(_)));
}

// Catches: an abandoned xAI response holding the generation slot and dropping the next turn.
#[test]
fn xai_discards_a_response_the_server_left_silent() {
    let mut m = machine(XAI);
    m.open(at(0));
    recv(&mut m, at(0), response_created("resp_silent", None));
    assert!(m.has_active_generation());

    let step = recv(&mut m, at(1), response_created("resp_real", None));
    assert!(step.events.contains(&Event::GenerationCreated(GenerationCreated {
        response_id: "resp_real".into(),
        req_id: None,
    })));
    // the silent one was closed out, not left dangling
    assert!(step.events.iter().any(|e| matches!(
        e,
        Event::GenerationDone (GenerationDone { status: GenerationStatus::Cancelled, .. })
    )));
}

// -----------------------------------------------------------------------------------------
// Session lifecycle
// -----------------------------------------------------------------------------------------

// Catches: recycling the socket mid-sentence; xAI must not recycle at all.
#[test]
fn the_socket_is_recycled_between_turns_and_never_during_one() {
    let clock = VirtualClock::new();
    let mut m = machine(OPENAI);
    m.open(clock.now());
    assert_eq!(m.next_deadline(), Some(at(20 * 60)));

    recv(&mut m, at(1), response_created("resp_1", None));
    clock.advance_to(at(20 * 60));
    assert!(!m.tick(clock.now()).recycle, "a turn is in flight");

    recv(
        &mut m,
        clock.now(),
        json!({
            "type": "response.done",
            "response": {"id": "resp_1", "status": "completed"},
        }),
    );
    assert!(m.tick(clock.now()).recycle, "now it is safe");

    let mut xai = machine(XAI);
    xai.open(at(0));
    assert_eq!(xai.next_deadline(), None, "xAI never recycles");
}

// Catches: a reconnect leaving pending replies hanging, or not rebuilding the mirror from
// scratch.
#[test]
fn a_reconnect_replays_the_conversation_and_fails_what_the_old_socket_owed() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    recv(&mut m, at(0), item_added(None, "a", "first"));
    recv(&mut m, at(0), item_added(Some("a"), "b", "second"));
    m.command(at(1), reply("response_create_2"));

    let step = m.reconnect(at(2));
    assert_eq!(
        types(&step.frames),
        ["session.update", "conversation.item.create", "conversation.item.create"],
    );
    assert_eq!(step.frames[1]["previous_item_id"], json!("root"));
    assert_eq!(step.frames[2]["previous_item_id"], json!("a"));
    assert!(step.events.contains(&Event::GenerateReplyFailed (GenerateReplyFailed {
        req_id: "response_create_2".into(),
        error: RealtimeError::new("pending response discarded due to session reconnection"),
    })));
    assert_eq!(step.events.last(), Some(&Event::SessionReconnected(lk_agents_abi::llm::realtime::SessionReconnected {})));

    // The mirror stays empty until the new server acks the creates: it is what the server
    // believes, and optimistic repopulation would hide a rejected create.
    assert!(m.chat_ctx().items.is_empty(), "the new server has an empty conversation");
    recv(&mut m, at(3), item_added(None, "a", "first"));
    recv(&mut m, at(3), item_added(Some("a"), "b", "second"));
    assert_eq!(m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a", "b"]);
}

// Catches: reconnecting forever on a dead API key instead of recognising the fatal codes.
#[test]
fn a_fatal_error_code_stops_the_reconnect_loop() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = recv(
        &mut m,
        at(1),
        json!({"type": "error", "error": {
            "type": "invalid_request_error", "code": "insufficient_quota",
            "message": "You exceeded your current quota",
        }}),
    );
    assert!(step.fatal.is_some(), "the driver must not reconnect");
    assert!(matches!(&step.events[0], Event::Error(e) if !e.recoverable));

    // an ordinary server error is recoverable and does NOT stop the loop
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = recv(
        &mut m,
        at(1),
        json!({"type": "error", "error": {
            "type": "server_error", "message": "try again",
        }}),
    );
    assert!(step.fatal.is_none());
    assert!(matches!(&step.events[0], Event::Error(e) if e.recoverable));
}

// Catches: committing a buffer of exactly 100 ms (OpenAI needs strictly more).
#[test]
fn a_commit_needs_strictly_more_than_one_hundred_milliseconds() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let frame = |ms: usize| lk_agents_abi::AudioFrame {
        data: vec![0u8; SAMPLE_RATE as usize / 1000 * ms * 2].into(),
        sample_rate: SAMPLE_RATE,
        num_channels: NUM_CHANNELS,
        samples_per_channel: (SAMPLE_RATE as usize / 1000 * ms) as u32,
    };

    let step = m.command(at(0), Command::PushAudio(PushAudio { frame: frame(100) }));
    assert_eq!(types(&step.frames), ["input_audio_buffer.append"]);
    assert!(m.command(at(0), Command::CommitAudio(lk_agents_abi::llm::realtime::CommitAudio {})).frames.is_empty(), "exactly 100 ms");

    m.command(at(0), Command::PushAudio(PushAudio { frame: frame(100) }));
    assert_eq!(
        types(&m.command(at(0), Command::CommitAudio(lk_agents_abi::llm::realtime::CommitAudio {})).frames),
        ["input_audio_buffer.commit"]
    );
    // and the counter resets, so the next commit needs its own 100 ms
    assert!(m.command(at(0), Command::CommitAudio(lk_agents_abi::llm::realtime::CommitAudio {})).frames.is_empty());
}

// Catches: overwriting a turn's speech onset when xAI re-announces it.
#[test]
fn a_re_announced_speech_onset_keeps_the_original_timestamp() {
    let mut m = machine(XAI);
    m.open(at(0));
    let started = json!({"type": "input_audio_buffer.speech_started", "item_id": "item_1"});
    recv(&mut m, at(3), started.clone());
    recv(&mut m, at(7), started);
    recv(
        &mut m,
        at(8),
        json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "transcript": "hello",
        }),
    );
    // flush it
    recv(&mut m, at(9), response_created("resp_1", None));
    let step = recv(
        &mut m,
        at(9),
        json!({
            "type": "response.output_audio.delta", "item_id": "msg_1", "delta": "AAAA",
        }),
    );
    let turn_started = step.events.iter().find_map(|e| match e {
        Event::InputTranscription(t) if t.is_final => Some(t.turn_started_at),
        _ => None,
    });
    assert_eq!(turn_started, Some(Some(at(3))), "the first onset, not the second");
}

// Catches: `aclose()` leaving a pending reply or a half-finished message's consumer hanging.
#[test]
fn closing_releases_the_in_flight_turn_and_every_waiting_caller() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    m.command(at(0), reply("response_create_1"));
    recv(&mut m, at(1), response_created("resp_1", Some("response_create_1")));
    recv(
        &mut m,
        at(1),
        json!({"type": "response.output_item.added", "item": {
            "id": "msg_1", "type": "message",
        }}),
    );
    m.command(at(1), reply("response_create_2"));

    let step = m.close(at(2));
    assert!(
        step.events.iter().any(|e| matches!(e, Event::MessageDone(MessageDone { message_id, .. }) if message_id == "msg_1"))
    );
    assert!(step.events.iter().any(|e| matches!(
        e,
        Event::GenerationDone (GenerationDone { status: GenerationStatus::Cancelled, .. })
    )));
    assert!(step.events.contains(&Event::GenerateReplyFailed (GenerateReplyFailed {
        req_id: "response_create_2".into(),
        error: RealtimeError::new("realtime session closed"),
    })));
}
