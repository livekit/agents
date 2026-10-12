//! A chat item between its in-memory form and its stored one. In memory it is `item_to_payload`'s
//! JSON (the pydantic dump less what storage drops); stored, `items.payload`, it is
//! `livekit.agent.ChatContext.ChatItem` as protobuf JSON, which the threads service reads.
//! Storage drops image and audio content, `group_id`, `reply_required` and `extra` values that
//! are not strings.

use livekit_protocol::agent::chat_context::{ChatItem, chat_item::Item as Pb};
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{
    AgentConfigUpdate, AgentHandoff, ChatMessage, ChatRole, FunctionCall, FunctionCallOutput,
};
use pbjson_types::Timestamp;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

use crate::StorageError;

/// The in-memory item as protobuf JSON.
pub fn encode(item: &Value) -> Result<Value, StorageError> {
    let text = |k: &str| item.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
    let opt = |k: &str| item.get(k).and_then(Value::as_str).map(str::to_owned);
    let strings = |k: &str| -> Vec<String> {
        item.get(k)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_owned).collect())
            .unwrap_or_default()
    };
    let extra: HashMap<String, String> = item
        .get("extra")
        .and_then(Value::as_object)
        .map(|o| o.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_owned()))).collect())
        .unwrap_or_default();
    let created_at = item.get("created_at").and_then(Value::as_f64).map(timestamp);
    let inner = match item.get("type").and_then(Value::as_str) {
        Some("message") => Pb::Message(ChatMessage {
            id: text("id"),
            role: ChatRole::from_str_name(&text("role").to_uppercase()).ok_or_else(|| {
                StorageError::Invalid(format!("chat message role {:?}", item.get("role")))
            })? as i32,
            content: strings("content")
                .into_iter()
                .map(|t| ChatContent { payload: Some(Payload::Text(t)) })
                .collect(),
            interrupted: item.get("interrupted").and_then(Value::as_bool).unwrap_or(false),
            transcript_confidence: item.get("transcript_confidence").and_then(Value::as_f64),
            extra,
            metrics: None,
            created_at,
        }),
        Some("function_call") => Pb::FunctionCall(FunctionCall {
            id: text("id"),
            call_id: text("call_id"),
            arguments: text("arguments"),
            name: text("name"),
            created_at,
            extra,
        }),
        Some("function_call_output") => Pb::FunctionCallOutput(FunctionCallOutput {
            id: text("id"),
            name: text("name"),
            call_id: text("call_id"),
            output: text("output"),
            is_error: item.get("is_error").and_then(Value::as_bool).unwrap_or(false),
            created_at,
        }),
        Some("agent_handoff") => Pb::AgentHandoff(AgentHandoff {
            id: text("id"),
            old_agent_id: opt("old_agent_id"),
            new_agent_id: text("new_agent_id"),
            created_at,
        }),
        Some("agent_config_update") => Pb::AgentConfigUpdate(AgentConfigUpdate {
            id: text("id"),
            instructions: opt("instructions"),
            tools_added: strings("tools_added"),
            tools_removed: strings("tools_removed"),
            created_at,
        }),
        other => return Err(StorageError::Invalid(format!("chat item of type {other:?}"))),
    };
    serde_json::to_value(ChatItem { item: Some(inner) })
        .map_err(|e| StorageError::Invalid(format!("chat item: {e}")))
}

/// The stored protobuf JSON as the in-memory item.
pub fn decode(stored: &Value) -> Result<Value, StorageError> {
    let item: ChatItem = serde_json::from_value(stored.clone())
        .map_err(|e| StorageError::Decode(format!("chat item: {e}")))?;
    let extra = |m: HashMap<String, String>| {
        Value::Object(m.into_iter().map(|(k, v)| (k, Value::String(v))).collect::<Map<_, _>>())
    };
    let (mut out, created_at) = match item.item {
        Some(Pb::Message(m)) => {
            let role = ChatRole::try_from(m.role).unwrap_or(ChatRole::User);
            let content: Vec<String> =
                m.content.into_iter().filter_map(|c| c.payload.map(|Payload::Text(t)| t)).collect();
            let mut out = json!({
                "id": m.id, "type": "message", "role": role.as_str_name().to_lowercase(),
                "content": content, "interrupted": m.interrupted, "extra": extra(m.extra),
            });
            if let Some(c) = m.transcript_confidence {
                out["transcript_confidence"] = json!(c);
            }
            (out, m.created_at)
        }
        Some(Pb::FunctionCall(f)) => (
            json!({
                "id": f.id, "type": "function_call", "call_id": f.call_id,
                "arguments": f.arguments, "name": f.name, "extra": extra(f.extra),
            }),
            f.created_at,
        ),
        Some(Pb::FunctionCallOutput(f)) => (
            json!({
                "id": f.id, "type": "function_call_output", "name": f.name, "call_id": f.call_id,
                "output": f.output, "is_error": f.is_error,
            }),
            f.created_at,
        ),
        Some(Pb::AgentHandoff(h)) => {
            let mut out =
                json!({"id": h.id, "type": "agent_handoff", "new_agent_id": h.new_agent_id});
            if let Some(old) = h.old_agent_id {
                out["old_agent_id"] = json!(old);
            }
            (out, h.created_at)
        }
        Some(Pb::AgentConfigUpdate(c)) => {
            let mut out = json!({"id": c.id, "type": "agent_config_update"});
            if let Some(i) = c.instructions {
                out["instructions"] = json!(i);
            }
            if !c.tools_added.is_empty() {
                out["tools_added"] = json!(c.tools_added);
            }
            if !c.tools_removed.is_empty() {
                out["tools_removed"] = json!(c.tools_removed);
            }
            (out, c.created_at)
        }
        None => return Err(StorageError::Decode(format!("chat item without an item: {stored}"))),
    };
    if let Some(t) = created_at {
        out["created_at"] = json!(t.seconds as f64 + f64::from(t.nanos) / 1e9);
    }
    Ok(out)
}

/// Unix seconds as a `Timestamp`, to the nearest nanosecond, so the float reads back unchanged.
fn timestamp(t: f64) -> Timestamp {
    let seconds = t.floor();
    let nanos = ((t - seconds) * 1e9).round() as i32;
    if nanos >= 1_000_000_000 {
        Timestamp { seconds: seconds as i64 + 1, nanos: 0 }
    } else {
        Timestamp { seconds: seconds as i64, nanos }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_round_trip_through_protobuf_json() {
        let items = [
            json!({"id": "i1", "type": "message", "role": "user", "content": ["hi", "there"],
                   "interrupted": false, "extra": {"k": "v"}, "created_at": 1791760302.4035509}),
            json!({"id": "i2", "type": "function_call", "call_id": "c", "arguments": "{}",
                   "name": "n", "extra": {"lk.delegation.task_id": "t1"}, "created_at": 3.5}),
            json!({"id": "i3", "type": "function_call_output", "name": "n", "call_id": "c",
                   "output": "o", "is_error": false, "created_at": 1791760302.403571}),
            json!({"id": "i4", "type": "agent_handoff", "old_agent_id": "a",
                   "new_agent_id": "b", "created_at": 0.25}),
        ];
        for item in items {
            assert_eq!(decode(&encode(&item).unwrap()).unwrap(), item);
        }
    }

    #[test]
    fn the_stored_form_is_protobuf_json_less_what_storage_drops() {
        let item = json!({"id": "i1", "type": "message", "role": "user",
            "content": ["hi", {"type": "image_content"}], "interrupted": false,
            "extra": {"message_ids": ["m1"]}, "created_at": 1.5});
        let stored = encode(&item).unwrap();
        assert_eq!(
            stored,
            json!({"message": {"id": "i1", "role": "USER", "content": [{"text": "hi"}],
                "createdAt": "1970-01-01T00:00:01.500+00:00"}})
        );
        let call = json!({"id": "i2", "type": "function_call", "call_id": "c", "arguments": "",
            "name": "n", "group_id": "g", "extra": {"lk.delegation.agent": "x"}});
        assert_eq!(
            encode(&call).unwrap(),
            json!({"functionCall": {"id": "i2", "callId": "c", "name": "n",
                "extra": {"lk.delegation.agent": "x"}}})
        );
        assert!(decode(&json!({"id": "i", "type": "message"})).is_err());
    }
}
