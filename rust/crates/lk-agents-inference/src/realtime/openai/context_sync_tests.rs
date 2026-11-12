//! Regression tests for what a chat context update puts on, and takes off, the OpenAI realtime
//! conversation.

use super::testfix::{RealtimeMachine, Step, TIME, machine, msg};
use super::wire;
use crate::realtime::dialect::OPENAI;
use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::llm::realtime::UpdateChatCtx;
use lk_agents_abi::{
    ChatContent, ChatContext, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role, Ts,
};
use serde_json::{Value, json};

fn at(secs: u64) -> Ts {
    Ts::from_nanos(secs * 1_000_000_000)
}

fn recv(m: &mut RealtimeMachine, now: Ts, ev: Value) -> Step {
    m.incoming(now, wire::decode(&ev))
}

fn update(m: &mut RealtimeMachine, now: Ts, items: Vec<ChatItem>) -> Step {
    m.command(now, Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::new(items) }))
}

fn creates(frames: &[Value]) -> Vec<&Value> {
    frames.iter().filter(|f| f["type"] == "conversation.item.create").collect()
}

/// Empty messages create nothing, and every `previous_item_id` names a created item or `root`.
/// https://github.com/livekit/agents/issues/1244 https://github.com/livekit/agents/issues/5951
#[test]
fn empty_local_items_never_become_remote_anchors() {
    let empty = |id: &str, role| ChatItem::Message(ChatMessage::new(id, role, Vec::new(), TIME));
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = update(
        &mut m,
        at(1),
        vec![
            empty("e0", Role::User),
            msg("a", "first"),
            empty("e1", Role::Assistant),
            empty("e2", Role::User),
            msg("b", "second"),
            empty("e3", Role::User),
        ],
    );
    let created = creates(&step.frames);
    let ids: Vec<_> = created.iter().map(|f| f["item"]["id"].as_str().unwrap_or("?")).collect();
    assert_eq!(ids, ["a", "b"]);
    let mut known = vec!["root".to_owned()];
    for frame in &created {
        assert!(
            frame["item"]["content"].as_array().is_some_and(|c| !c.is_empty()),
            "an empty content array goes out: {frame}"
        );
        let anchor = frame["previous_item_id"].as_str().unwrap_or("root").to_owned();
        assert!(known.contains(&anchor), "{anchor} was never created: {frame}");
        known.push(frame["item"]["id"].as_str().unwrap_or_default().to_owned());
    }
}

/// A context update keeps system and developer messages (sent as `system`) and does not recreate
/// them on the next update. https://github.com/livekit/agents/issues/4875
#[test]
fn updates_preserve_system_and_developer_messages() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = update(
        &mut m,
        at(1),
        vec![
            ChatItem::Message(ChatMessage::text("sys", Role::System, "Speak French.", TIME)),
            ChatItem::Message(ChatMessage::text("dev", Role::Developer, "Be brief.", TIME)),
            msg("u", "bonjour"),
        ],
    );
    let created = creates(&step.frames);
    let items: Vec<(&str, &str)> = created
        .iter()
        .map(|f| {
            (f["item"]["id"].as_str().unwrap_or("?"), f["item"]["role"].as_str().unwrap_or("?"))
        })
        .collect();
    assert_eq!(items, [("sys", "system"), ("dev", "system"), ("u", "user")]);
    assert_eq!(created[0]["item"]["content"][0]["text"], json!("Speak French."));
    assert_eq!(created[1]["item"]["content"][0]["text"], json!("Be brief."));

    // A later update that keeps them deletes and recreates nothing.
    for frame in creates(&step.frames) {
        let ack = json!({"type": "conversation.item.added",
            "previous_item_id": frame["previous_item_id"], "item": frame["item"]});
        recv(&mut m, at(2), ack);
    }
    let again = update(
        &mut m,
        at(3),
        vec![
            ChatItem::Message(ChatMessage::text("sys", Role::System, "Speak French.", TIME)),
            ChatItem::Message(ChatMessage::text("dev", Role::Developer, "Be brief.", TIME)),
            msg("u", "bonjour"),
            msg("u2", "merci"),
        ],
    );
    let ids: Vec<_> = again.frames.iter().map(|f| f["type"].as_str().unwrap_or("?")).collect();
    assert_eq!(ids, ["conversation.item.create"], "{:?}", again.frames);
}

/// Ids over OpenAI's 32-character limit go out within it and map back to the caller's ids.
/// https://github.com/livekit/agents/issues/1717 https://github.com/livekit/agents/pull/7958
#[test]
#[ignore = "https://github.com/livekit/agents/pull/7958 reproduces: item ids over 32 characters reach the wire unchanged"]
fn arbitrary_item_ids_round_trip_within_provider_limits() {
    let long = |n: &str| format!("item_{n}_0123456789abcdef-0123456789:abcdef");
    let call_id = "call_0123456789abcdef0123456789abcdef";
    let call = ChatItem::FunctionCall(FunctionCall {
        id: long("call"),
        call_id: call_id.into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: TIME,
        extra: Default::default(),
        group_id: None,
    });
    let output = ChatItem::FunctionCallOutput(FunctionCallOutput::new(
        long("out"),
        call_id,
        "42",
        false,
        TIME,
    ));
    let items = vec![msg(&long("user"), "hi"), call, output];
    let mut m = machine(OPENAI);
    m.open(at(0));
    let step = update(&mut m, at(1), items.clone());
    let created = creates(&step.frames);
    assert_eq!(created.len(), 3);
    for frame in &created {
        let id = frame["item"]["id"].as_str().unwrap_or_default();
        assert!(id.len() <= 32, "item id {id} exceeds 32 characters");
        if let Some(anchor) = frame["previous_item_id"].as_str() {
            assert!(anchor.len() <= 32, "anchor {anchor} exceeds 32 characters");
        }
    }
    assert_eq!(created[1]["item"]["call_id"], created[2]["item"]["call_id"]);

    for frame in creates(&step.frames) {
        let ack = json!({"type": "conversation.item.added",
            "previous_item_id": frame["previous_item_id"], "item": frame["item"]});
        recv(&mut m, at(2), ack);
    }
    let ids: Vec<String> = m.chat_ctx().items.iter().map(|i| i.id().to_owned()).collect();
    assert_eq!(ids, items.iter().map(|i| i.id().to_owned()).collect::<Vec<_>>());

    // Deleting by the caller's id reaches the server under the wire id.
    let step = update(&mut m, at(3), Vec::new());
    for frame in step.frames.iter().filter(|f| f["type"] == "conversation.item.delete") {
        assert!(frame["item_id"].as_str().unwrap_or_default().len() <= 32, "{frame}");
    }
}

/// Untranscribed input audio items are still deleted when the context is trimmed.
/// https://github.com/livekit/agents/issues/5886
#[test]
fn untranscribed_remote_audio_is_deleted_when_trimmed() {
    let mut m = machine(OPENAI);
    m.open(at(0));
    let mut previous = Value::Null;
    for id in ["aud1", "aud2"] {
        recv(
            &mut m,
            at(0),
            json!({"type": "conversation.item.added", "previous_item_id": previous, "item": {
                "id": id, "type": "message", "role": "user",
                "content": [{"type": "input_audio", "transcript": null}]}}),
        );
        previous = json!(id);
    }
    assert_eq!(m.chat_ctx().items.len(), 2);
    assert!(m.chat_ctx().items.iter().all(|i| {
        i.as_message().is_some_and(|m| !m.content.iter().any(|c| matches!(c, ChatContent::Text(_))))
    }));

    let step = update(&mut m, at(1), Vec::new());
    let mut deleted: Vec<&str> = step
        .frames
        .iter()
        .filter(|f| f["type"] == "conversation.item.delete")
        .filter_map(|f| f["item_id"].as_str())
        .collect();
    deleted.sort_unstable();
    assert_eq!(deleted, ["aud1", "aud2"]);
    for id in ["aud1", "aud2"] {
        recv(&mut m, at(2), json!({"type": "conversation.item.deleted", "item_id": id}));
    }
    assert!(!m.context_work_pending(), "the update settled");
    assert!(m.chat_ctx().items.is_empty());
}
