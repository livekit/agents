//! The canonical thread/remote-session chat protocol mapping.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod proto {
    pub use livekit_protocol::agent::*;
    pub use pbjson_types::Timestamp;
}
use crate::{ChatContent, ChatItem, ChatMessage, Extra, MetricsReport, Role};
use serde_json::Value;

pub fn encode_timestamp(t: SystemTime) -> proto::Timestamp {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    proto::Timestamp { seconds: d.as_secs() as i64, nanos: d.subsec_nanos() as i32 }
}

pub fn encode_epoch_secs(t: f64) -> proto::Timestamp {
    let nanos = (t * 1e9) as i128;
    proto::Timestamp {
        seconds: (nanos.div_euclid(1_000_000_000)) as i64,
        nanos: (nanos.rem_euclid(1_000_000_000)) as i32,
    }
}

/// `ROLES` (`_proto.py:30`). Total, so there is no default arm to get wrong.
pub fn encode_role(role: Role) -> proto::ChatRole {
    match role {
        Role::Developer => proto::ChatRole::Developer,
        Role::System => proto::ChatRole::System,
        Role::User => proto::ChatRole::User,
        Role::Assistant => proto::ChatRole::Assistant,
    }
}

/// `encode_metrics` (`_proto.py:76`). Ten of fifteen fields survive.
pub fn encode_metrics(m: &MetricsReport) -> proto::MetricsReport {
    proto::MetricsReport {
        started_speaking_at: m.started_speaking_at.map(encode_epoch_secs),
        stopped_speaking_at: m.stopped_speaking_at.map(encode_epoch_secs),
        transcription_delay: m.transcription_delay,
        end_of_turn_delay: m.end_of_turn_delay,
        on_user_turn_completed_delay: m.on_user_turn_completed_delay,
        llm_node_ttft: m.llm_node_ttft,
        tts_node_ttfb: m.tts_node_ttfb,
        e2e_latency: m.e2e_latency,
        llm_node_tps: m.llm_node_tps,
        llm_node_ttfs: m.llm_node_ttfs,
        // dropped: the proto declares none of the metadata fields.
    }
}

/// Each value as its JSON text, a string included (`"text"` in quotes), so a string that looks
/// like JSON reads back a string.
fn encode_extra(extra: &Extra) -> impl Iterator<Item = (String, String)> + '_ {
    extra.iter().map(|(k, v)| (k.clone(), v.to_string()))
}

/// `encode_chat_item` (`_proto.py:80`).
pub fn encode_chat_item(item: &ChatItem) -> proto::chat_context::ChatItem {
    use proto::chat_context::chat_item::Item;
    let inner = match item {
        ChatItem::Message(m) => Item::Message(encode_chat_message(m)),
        ChatItem::FunctionCall(f) => Item::FunctionCall(proto::FunctionCall {
            id: f.id.clone(),
            call_id: f.call_id.clone(),
            arguments: f.arguments.clone(),
            name: f.name.clone(),
            created_at: Some(encode_timestamp(f.created_at)),
            extra: encode_extra(&f.extra).collect(),
            // dropped: group_id
        }),
        ChatItem::FunctionCallOutput(f) => Item::FunctionCallOutput(proto::FunctionCallOutput {
            id: f.id.clone(),
            name: f.name.clone(),
            call_id: f.call_id.clone(),
            output: f.output.clone(),
            is_error: f.is_error,
            created_at: Some(encode_timestamp(f.created_at)),
            // dropped: reply_required
        }),
        ChatItem::AgentHandoff(h) => Item::AgentHandoff(proto::AgentHandoff {
            id: h.id.clone(),
            old_agent_id: h.old_agent_id.clone(),
            new_agent_id: h.new_agent_id.clone(),
            created_at: Some(encode_timestamp(h.created_at)),
        }),
        // An empty instructions string is absent, as in the reference (proto3 would tell them apart).
        ChatItem::AgentConfigUpdate(c) => Item::AgentConfigUpdate(proto::AgentConfigUpdate {
            id: c.id.clone(),
            instructions: c.instructions.clone().filter(|s| !s.is_empty()),
            tools_added: c.tools_added.clone().unwrap_or_default(),
            tools_removed: c.tools_removed.clone().unwrap_or_default(),
            created_at: Some(encode_timestamp(c.created_at)),
        }),
    };
    proto::chat_context::ChatItem { item: Some(inner) }
}

fn encode_chat_message(m: &ChatMessage) -> proto::ChatMessage {
    use proto::chat_message::{ChatContent as PbContent, chat_content::Payload};
    proto::ChatMessage {
        id: m.id.clone(),
        role: encode_role(m.role) as i32,
        // Image and audio content are filtered out, not rendered.
        content: m
            .content
            .iter()
            .filter_map(|c| match c {
                ChatContent::Text(t) => Some(PbContent { payload: Some(Payload::Text(t.clone())) }),
                ChatContent::Image(_) | ChatContent::Audio(_) => None,
            })
            .collect(),
        interrupted: m.interrupted,
        transcript_confidence: m.transcript_confidence,
        extra: encode_extra(&m.extra).collect(),
        metrics: Some(encode_metrics(&m.metrics)),
        created_at: Some(encode_timestamp(m.created_at)),
    }
}

fn time(at: Option<pbjson_types::Timestamp>) -> Result<SystemTime, String> {
    let at = at.unwrap_or_default();
    if !(0..1_000_000_000).contains(&at.nanos) {
        return Err("invalid chat timestamp nanoseconds".into());
    }
    let base = if at.seconds >= 0 {
        UNIX_EPOCH.checked_add(Duration::from_secs(at.seconds as u64))
    } else {
        UNIX_EPOCH.checked_sub(Duration::from_secs(at.seconds.unsigned_abs()))
    };
    base.and_then(|base| base.checked_add(Duration::from_nanos(at.nanos as u64)))
        .ok_or_else(|| "chat timestamp out of range".into())
}

/// [`encode_extra`]'s JSON text; a value that is not JSON is a string, as rows written before
/// it stored a string raw.
fn native_extra(
    values: impl IntoIterator<Item = (String, String)>,
) -> serde_json::Map<String, Value> {
    values
        .into_iter()
        .map(|(key, value)| {
            let decoded = serde_json::from_str(&value).unwrap_or(Value::String(value));
            (key, decoded)
        })
        .collect()
}

fn metric_seconds(at: Option<pbjson_types::Timestamp>) -> Option<f64> {
    at.and_then(|at| time(Some(at)).ok())
        .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
        .map(|at| at.as_secs_f64())
}

pub fn decode_chat_item(item: proto::chat_context::ChatItem) -> Result<ChatItem, String> {
    use proto::chat_context::chat_item::Item;
    Ok(match item.item.ok_or_else(|| "empty chat item".to_owned())? {
        Item::Message(message) => {
            let role = match proto::ChatRole::try_from(message.role) {
                Ok(proto::ChatRole::Developer) => Role::Developer,
                Ok(proto::ChatRole::System) => Role::System,
                Ok(proto::ChatRole::User) => Role::User,
                Ok(proto::ChatRole::Assistant) => Role::Assistant,
                _ => return Err("unknown chat role".into()),
            };
            let content = message
                .content
                .into_iter()
                .filter_map(|content| content.payload)
                .map(|content| match content {
                    proto::chat_message::chat_content::Payload::Text(text) => {
                        ChatContent::Text(text)
                    }
                })
                .collect();
            use crate::MetricsReport;
            let metrics = message.metrics.unwrap_or_default();
            ChatItem::Message(ChatMessage {
                id: message.id,
                role,
                content,
                interrupted: message.interrupted,
                transcript_confidence: message.transcript_confidence,
                extra: native_extra(message.extra),
                created_at: time(message.created_at)?,
                metrics: MetricsReport {
                    started_speaking_at: metric_seconds(metrics.started_speaking_at),
                    stopped_speaking_at: metric_seconds(metrics.stopped_speaking_at),
                    transcription_delay: metrics.transcription_delay,
                    end_of_turn_delay: metrics.end_of_turn_delay,
                    on_user_turn_completed_delay: metrics.on_user_turn_completed_delay,
                    llm_node_ttft: metrics.llm_node_ttft,
                    tts_node_ttfb: metrics.tts_node_ttfb,
                    e2e_latency: metrics.e2e_latency,
                    llm_node_tps: metrics.llm_node_tps,
                    llm_node_ttfs: metrics.llm_node_ttfs,
                    ..Default::default()
                },
            })
        }
        Item::FunctionCall(call) => ChatItem::FunctionCall(crate::FunctionCall {
            id: call.id,
            call_id: call.call_id,
            arguments: call.arguments,
            name: call.name,
            created_at: time(call.created_at)?,
            extra: native_extra(call.extra),
            group_id: None,
        }),
        Item::FunctionCallOutput(output) => {
            ChatItem::FunctionCallOutput(crate::FunctionCallOutput {
                id: output.id,
                name: output.name,
                call_id: output.call_id,
                output: output.output,
                is_error: output.is_error,
                created_at: time(output.created_at)?,
                reply_required: true,
            })
        }
        Item::AgentHandoff(handoff) => ChatItem::AgentHandoff(crate::AgentHandoffItem {
            id: handoff.id,
            old_agent_id: handoff.old_agent_id,
            new_agent_id: handoff.new_agent_id,
            created_at: time(handoff.created_at)?,
        }),
        Item::AgentConfigUpdate(update) => ChatItem::AgentConfigUpdate(crate::AgentConfigUpdate {
            id: update.id,
            instructions: update.instructions,
            tools_added: Some(update.tools_added),
            tools_removed: Some(update.tools_removed),
            created_at: time(update.created_at)?,
            tools: None,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decision 117 (Q29): `extra` crosses the proto losslessly; a raw string an earlier row
    /// stored still reads as it did.
    #[test]
    fn extra_keeps_a_string_that_looks_like_json_a_string() {
        let extra: Extra = serde_json::from_value(serde_json::json!({
            "json_like": "{\"a\": 1}", "number_like": "42", "text": "asked",
            "object": {"a": 1}, "number": 42,
        }))
        .unwrap();
        let encoded: Vec<(String, String)> = encode_extra(&extra).collect();
        assert_eq!(native_extra(encoded), extra);
        let old = [("text".to_owned(), "asked".to_owned()), ("n".to_owned(), "42".to_owned())];
        assert_eq!(
            native_extra(old),
            serde_json::json!({"text": "asked", "n": 42}).as_object().unwrap().clone()
        );
    }
}
