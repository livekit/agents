use super::*;
use lk_agents_abi::worker::remote_session::{DebugMessageWire, FunctionToolsStartedWire};
use lk_agents_abi::{Extra, Role};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn at(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

// Catches: a function call's provider extras dropped or not sent as their JSON text (decision 117,
// Q29: a string goes quoted, so it reads back a string), a call output losing its error flag, or an agent update sending an empty instructions string the console
// shows as "instructions changed to nothing".
#[test]
fn tool_items_and_agent_updates_encode_for_the_console() {
    use proto::chat_context::chat_item::Item;
    let mut extra = Extra::new();
    extra.insert("google".into(), serde_json::json!("sig"));
    extra.insert("n".into(), serde_json::json!(3));
    let call = ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
        id: "item_1".into(),
        call_id: "call_1".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: at(5),
        extra,
        group_id: None,
    });
    let Some(Item::FunctionCall(encoded)) = encode_chat_item(&call).item else { panic!() };
    assert_eq!(encoded.extra.get("google").map(String::as_str), Some("\"sig\""));
    assert_eq!(encoded.extra.get("n").map(String::as_str), Some("3"));
    assert_eq!(encoded.created_at.map(|t| t.seconds), Some(5));

    let output = lk_agents_abi::FunctionCallOutput::new("item_2", "call_1", "boom", true, at(6));
    let Some(Item::FunctionCallOutput(encoded)) =
        encode_chat_item(&ChatItem::FunctionCallOutput(output)).item
    else {
        panic!()
    };
    assert!(encoded.is_error);

    let update = ChatItem::AgentConfigUpdate(lk_agents_abi::AgentConfigUpdate {
        id: "item_3".into(),
        instructions: Some(String::new()),
        tools_added: Some(vec!["lookup".into()]),
        tools_removed: None,
        created_at: at(7),
        tools: None,
    });
    let Some(Item::AgentConfigUpdate(encoded)) = encode_chat_item(&update).item else { panic!() };
    assert_eq!(encoded.instructions, None);
    assert_eq!(encoded.tools_added, ["lookup"]);

    assert_eq!(encode_role(Role::Developer), proto::ChatRole::Developer);
    assert_eq!(encode_role(Role::System), proto::ChatRole::System);
}

// Catches: a negative duration whose nanos disagree in sign with its seconds (an invalid proto
// Duration the console rejects), or a pre-epoch timestamp with negative nanos.
#[test]
fn durations_and_timestamps_keep_their_sign_rules() {
    let d = encode_duration_secs(-0.25);
    assert_eq!((d.seconds, d.nanos), (0, -250_000_000));
    let d = encode_duration_secs(-1.5);
    assert_eq!((d.seconds, d.nanos), (-1, -500_000_000));
    let d = encode_duration_secs(1.5);
    assert_eq!((d.seconds, d.nanos), (1, 500_000_000));
    let t = encode_epoch_secs(-0.5);
    assert_eq!((t.seconds, t.nanos), (-1, 500_000_000));
}

// Catches: the interruption and end-of-turn models missing from the usage the console shows.
#[test]
fn every_model_kind_is_in_the_usage() {
    use lk_agents_abi::usage::{EotUsage, InterruptionUsage};
    use proto::model_usage::Usage as Wire;
    let usage = Usage {
        models: vec![
            ModelUsage::Interruption(InterruptionUsage {
                provider: Some("livekit".into()),
                model: Some("adaptive".into()),
                requests: 4,
                ..Default::default()
            }),
            ModelUsage::Eot(EotUsage {
                provider: Some("livekit".into()),
                model: Some("eot".into()),
                requests: 2,
                ..Default::default()
            }),
        ],
        ..Default::default()
    };
    let encoded = encode_session_usage(&usage);
    assert!(matches!(
        encoded.model_usage[0].usage,
        Some(Wire::Interruption(proto::InterruptionModelUsage { total_requests: 4, .. }))
    ));
    assert!(matches!(
        encoded.model_usage[1].usage,
        Some(Wire::Eot(proto::EotModelUsage { total_requests: 2, .. }))
    ));
}

// Catches: a nested JSON value flattened or dropped on its way into a proto `Struct`, and a
// non-object accepted as one.
#[test]
fn json_objects_become_structs_whole() {
    use pbjson_types::value::Kind;
    let value = serde_json::json!({
        "ok": true, "n": 1.5, "s": "x", "none": null, "list": [1, "a"], "nested": {"k": "v"}
    });
    let encoded = json_to_struct(&value).unwrap();
    let kind = |k: &str| encoded.fields[k].kind.clone().unwrap();
    assert_eq!(kind("ok"), Kind::BoolValue(true));
    assert_eq!(kind("n"), Kind::NumberValue(1.5));
    assert_eq!(kind("none"), Kind::NullValue(0));
    let Kind::ListValue(list) = kind("list") else { panic!() };
    assert_eq!(list.values.len(), 2);
    let Kind::StructValue(nested) = kind("nested") else { panic!() };
    assert_eq!(nested.fields["k"].kind, Some(Kind::StringValue("v".into())));
    assert_eq!(json_to_struct(&serde_json::json!([1])), None);
}

// Catches: a started-tools event or a debug message lost on the way to the console, a malformed
// debug payload sent as an empty one, or an oversize frame buffered instead of refused.
#[test]
fn tool_starts_and_debug_messages_reach_the_console() {
    use proto::agent_session_event::Event;
    let started = WireEvent::FunctionToolsStarted(FunctionToolsStartedWire {
        function_calls: vec![FunctionCallRef {
            name: "lookup".into(),
            arguments: "{}".into(),
            call_id: "call_1".into(),
        }],
    });
    let Ok(Some(Event::FunctionToolsStarted(ev))) = encode_wire(started) else { panic!() };
    assert_eq!(ev.function_calls[0].call_id, "call_1");
    assert_eq!(event_name(&Event::FunctionToolsStarted(ev)), "function_tools_started");

    let payload = json_to_struct(&serde_json::json!({"step": "plan"}));
    let raw = proto::DebugMessage { payload: payload.clone() }.encode_to_vec();
    let debug = WireEvent::DebugMessage(DebugMessageWire { payload: Some(raw) });
    let Ok(Some(Event::DebugMessage(ev))) = encode_wire(debug) else { panic!() };
    assert_eq!(ev.payload, payload);
    let garbage = WireEvent::DebugMessage(DebugMessageWire { payload: Some(vec![0xff, 0xff]) });
    assert!(encode_wire(garbage).is_err());
    let empty = WireEvent::DebugMessage(DebugMessageWire { payload: None });
    assert!(matches!(
        encode_wire(empty),
        Ok(Some(Event::DebugMessage(proto::DebugMessage { payload: None })))
    ));

    let mut decoder = FrameDecoder::new();
    decoder.push(&((MAX_MESSAGE_SIZE as u32) + 1).to_be_bytes());
    assert!(matches!(decoder.next_frame(), Err(FrameError::Oversize(_))));
}
