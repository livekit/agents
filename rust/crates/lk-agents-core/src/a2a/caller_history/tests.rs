use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use serde_json::json;

fn wire() -> Value {
    json!({"items": [
        {"type": "message", "id": "m", "role": "user", "content": ["Check coverage"],
         "created_at": 12.5},
        {"type": "function_call", "id": "c", "call_id": "lookup-1", "name": "lookup",
         "arguments": "{\"payer\":\"acme\"}", "created_at": 13},
        {"type": "function_call_output", "id": "o", "call_id": "lookup-1", "name": "lookup",
         "output": "covered", "is_error": false, "created_at": 14}
    ]})
}

#[test]
fn caller_history_schema_rejects_missing_fields_bad_variants_and_unpaired_calls() {
    let history = CallerHistory::decode(wire()).unwrap();
    assert_eq!(history.to_chat_items().len(), 3);
    let mut malformed = Vec::new();
    let mut missing_id = wire();
    missing_id["items"][0].as_object_mut().unwrap().remove("id");
    malformed.push(missing_id);
    let mut missing_time = wire();
    missing_time["items"][1].as_object_mut().unwrap().remove("created_at");
    malformed.push(missing_time);
    let mut wrong_time = wire();
    wrong_time["items"][0]["created_at"] = json!("today");
    malformed.push(wrong_time);
    let mut bad_type = wire();
    bad_type["items"][0]["type"] = json!("agent_handoff");
    malformed.push(bad_type);
    let mut media = wire();
    media["items"][0]["content"] = json!([{"type": "image", "image": "private"}]);
    malformed.push(media);
    let mut wrong_order = wire();
    wrong_order["items"].as_array_mut().unwrap().swap(1, 2);
    malformed.push(wrong_order);
    let mut missing_output = wire();
    missing_output["items"].as_array_mut().unwrap().pop();
    malformed.push(missing_output);
    let mut mismatched_name = wire();
    mismatched_name["items"][2]["name"] = json!("other");
    malformed.push(mismatched_name);
    let mut bad_arguments = wire();
    bad_arguments["items"][1]["arguments"] = json!("[]");
    malformed.push(bad_arguments);
    let mut extra_field = wire();
    extra_field["items"][0]["metrics"] = json!({});
    malformed.push(extra_field);
    let mut duplicate = wire();
    duplicate["items"][2]["id"] = json!("m");
    malformed.push(duplicate);
    for payload in malformed {
        assert!(CallerHistory::decode(payload.clone()).is_err(), "accepted {payload}");
    }
}

#[test]
fn caller_history_ingress_rejects_malformed_and_duplicate_parts_before_handler_use() {
    let make_message = |parts: Value| -> a2a::Message {
        serde_json::from_value(json!({"messageId": "m", "role": "ROLE_USER", "parts": parts}))
            .unwrap()
    };
    let history_part = json!({"data": wire(), "metadata": {KIND_KEY: "chat_ctx"}});
    let message = make_message(json!([{"text": "request"}, history_part]));
    assert_eq!(CallerHistory::from_message(&message).unwrap().unwrap().to_chat_items().len(), 3);
    let duplicate = make_message(json!([history_part, history_part]));
    assert!(CallerHistory::from_message(&duplicate).is_err());
    for malformed in [
        json!({"data": ["wrong"], "metadata": {KIND_KEY: "chat_ctx"}}),
        json!({"data": {"items": [{"type": "message"}]}, "metadata": {KIND_KEY: "chat_ctx"}}),
        json!({"text": "not a data part", "metadata": {KIND_KEY: "chat_ctx"}}),
    ] {
        assert!(CallerHistory::from_message(&make_message(json!([malformed]))).is_err());
    }
    let untagged = make_message(json!([{"data": wire()}]));
    assert!(CallerHistory::from_message(&untagged).unwrap().is_none());
}

#[test]
fn caller_history_keeps_parallel_complete_pairs_and_excludes_the_delegation_call() {
    let timestamp = UNIX_EPOCH + Duration::from_secs(5);
    let call = |id: &str| {
        ChatItem::FunctionCall(FunctionCall {
            id: format!("item-{id}"),
            call_id: id.into(),
            name: "lookup".into(),
            arguments: "{}".into(),
            created_at: timestamp,
            extra: Map::new(),
            group_id: Some("g".into()),
        })
    };
    let output = |id: &str| {
        ChatItem::FunctionCallOutput(FunctionCallOutput::new(
            format!("output-{id}"),
            id,
            "done",
            false,
            timestamp,
        ))
    };
    let items = vec![
        ChatItem::Message(ChatMessage::text("m", Role::User, "request", timestamp)),
        call("a"),
        call("b"),
        output("b"),
        output("a"),
        call("unrelated-unfinished"),
        call("delegate"),
        output("delegate"),
    ];
    let history =
        CallerHistory::from_chat_items(&items, Some("delegate"), &HistorySharing::default())
            .unwrap();
    let selected = history.to_chat_items();
    assert_eq!(selected, items[..5]);
    let encoded = history.to_value().unwrap();
    assert!(!encoded.to_string().contains("delegate"));
    assert!(!encoded.to_string().contains("unfinished"));
    assert_eq!(CallerHistory::decode(encoded).unwrap().to_chat_items(), selected);
}

#[test]
fn caller_history_shares_only_selected_public_extra_and_never_internal_nested_data() {
    let timestamp = UNIX_EPOCH + Duration::from_secs(5);
    let mut message = ChatMessage::text("m", Role::User, "request", timestamp);
    message.extra = serde_json::from_value(json!({
        "public_note": {"label": "coverage"},
        "unselected": "private app context",
        "lk.approval": "granted", "lk.delegation.task_id": "AST_private",
        "google": {"thought_signature": "private-provider-data"},
        "selected_but_sensitive": {"details": [{"api_key": "private-key"}]}
    }))
    .unwrap();
    let sharing = HistorySharing {
        extra_keys: [
            "public_note",
            "lk.approval",
            "lk.delegation.task_id",
            "google",
            "selected_but_sensitive",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        ..HistorySharing::default()
    };
    let mut call_extra = message.extra.clone();
    call_extra.insert("lk.approval_identity".into(), json!("private-identity"));
    let call = FunctionCall {
        id: "c".into(),
        call_id: "lookup".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: timestamp,
        extra: call_extra,
        group_id: None,
    };
    let history = CallerHistory::from_chat_items(
        &[
            ChatItem::Message(message),
            ChatItem::FunctionCall(call),
            ChatItem::FunctionCallOutput(FunctionCallOutput::new(
                "o", "lookup", "done", false, timestamp,
            )),
        ],
        None,
        &sharing,
    )
    .unwrap();
    let encoded = history.to_value().unwrap();
    assert_eq!(encoded["items"][0]["extra"], json!({"public_note": {"label": "coverage"}}));
    assert_eq!(encoded["items"][1]["extra"], json!({"public_note": {"label": "coverage"}}));
    assert!(!encoded.to_string().contains("private"));
    assert!(!encoded.to_string().contains("metrics"));
    assert!(CallerHistory::decode(encoded).is_ok());
    for extra in [json!({"lk.approval": "granted"}), json!({"public": {"credentials": "bad"}})] {
        let mut incoming = wire();
        incoming["items"][0]["extra"] = extra;
        assert!(CallerHistory::decode(incoming).is_err());
    }
}
