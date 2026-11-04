//! A chat item in the `livekit.agent.ChatContext.ChatItem` protobuf JSON form, as 1.8.10's
//! `MessageToDict(encode_chat_item(item), preserving_proto_field_name=True)` wrote it: proto field
//! names, default values omitted, enums by name, timestamps in RFC 3339, `extra` values as
//! Python's `str(v)`. Fields follow `chat_context::protocol::encode_chat_item`.

use std::time::{SystemTime, UNIX_EPOCH};

use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Extra, MetricsReport, Role};
use serde_json::{Map, Value};

pub(super) fn chat_item(item: &ChatItem) -> Map<String, Value> {
    let mut m = Map::new();
    let (key, body) = match item {
        ChatItem::Message(msg) => ("message", message(msg)),
        ChatItem::FunctionCall(f) => {
            string(&mut m, "id", &f.id);
            string(&mut m, "call_id", &f.call_id);
            string(&mut m, "arguments", &f.arguments);
            string(&mut m, "name", &f.name);
            m.insert("created_at".into(), timestamp(f.created_at));
            extra(&mut m, &f.extra);
            ("function_call", std::mem::take(&mut m))
        }
        ChatItem::FunctionCallOutput(f) => {
            string(&mut m, "id", &f.id);
            string(&mut m, "name", &f.name);
            string(&mut m, "call_id", &f.call_id);
            string(&mut m, "output", &f.output);
            if f.is_error {
                m.insert("is_error".into(), true.into());
            }
            m.insert("created_at".into(), timestamp(f.created_at));
            ("function_call_output", std::mem::take(&mut m))
        }
        ChatItem::AgentHandoff(h) => {
            string(&mut m, "id", &h.id);
            if let Some(old) = &h.old_agent_id {
                m.insert("old_agent_id".into(), old.clone().into());
            }
            string(&mut m, "new_agent_id", &h.new_agent_id);
            m.insert("created_at".into(), timestamp(h.created_at));
            ("agent_handoff", std::mem::take(&mut m))
        }
        ChatItem::AgentConfigUpdate(c) => {
            string(&mut m, "id", &c.id);
            if let Some(instructions) = c.instructions.as_ref().filter(|s| !s.is_empty()) {
                m.insert("instructions".into(), instructions.clone().into());
            }
            strings(&mut m, "tools_added", c.tools_added.as_deref());
            strings(&mut m, "tools_removed", c.tools_removed.as_deref());
            m.insert("created_at".into(), timestamp(c.created_at));
            ("agent_config_update", std::mem::take(&mut m))
        }
    };
    Map::from_iter([(key.to_owned(), Value::Object(body))])
}

fn message(msg: &ChatMessage) -> Map<String, Value> {
    let mut m = Map::new();
    string(&mut m, "id", &msg.id);
    // `DEVELOPER` is the enum's zero, which the JSON form omits.
    let role = match msg.role {
        Role::Developer => None,
        Role::System => Some("SYSTEM"),
        Role::User => Some("USER"),
        Role::Assistant => Some("ASSISTANT"),
    };
    if let Some(role) = role {
        m.insert("role".into(), role.into());
    }
    let content: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|c| match c {
            ChatContent::Text(t) => {
                Some(Value::Object(Map::from_iter([("text".to_owned(), Value::String(t.clone()))])))
            }
            ChatContent::Image(_) | ChatContent::Audio(_) => None,
        })
        .collect();
    if !content.is_empty() {
        m.insert("content".into(), content.into());
    }
    if msg.interrupted {
        m.insert("interrupted".into(), true.into());
    }
    if let Some(confidence) = msg.transcript_confidence {
        m.insert("transcript_confidence".into(), confidence.into());
    }
    extra(&mut m, &msg.extra);
    m.insert("metrics".into(), Value::Object(metrics(&msg.metrics)));
    m.insert("created_at".into(), timestamp(msg.created_at));
    m
}

fn metrics(r: &MetricsReport) -> Map<String, Value> {
    let mut m = Map::new();
    for (key, at) in [
        ("started_speaking_at", r.started_speaking_at),
        ("stopped_speaking_at", r.stopped_speaking_at),
    ] {
        if let Some(at) = at {
            m.insert(key.into(), rfc3339_secs(at).into());
        }
    }
    for (key, v) in [
        ("transcription_delay", r.transcription_delay),
        ("end_of_turn_delay", r.end_of_turn_delay),
        ("on_user_turn_completed_delay", r.on_user_turn_completed_delay),
        ("llm_node_ttft", r.llm_node_ttft),
        ("tts_node_ttfb", r.tts_node_ttfb),
        ("e2e_latency", r.e2e_latency),
        ("llm_node_tps", r.llm_node_tps),
        ("llm_node_ttfs", r.llm_node_ttfs),
    ] {
        if let Some(v) = v {
            m.insert(key.into(), v.into());
        }
    }
    m
}

fn string(m: &mut Map<String, Value>, key: &str, value: &str) {
    if !value.is_empty() {
        m.insert(key.into(), value.into());
    }
}

fn strings(m: &mut Map<String, Value>, key: &str, values: Option<&[String]>) {
    if let Some(values) = values.filter(|v| !v.is_empty()) {
        m.insert(key.into(), values.iter().cloned().map(Value::String).collect());
    }
}

fn extra(m: &mut Map<String, Value>, extra: &Extra) {
    if !extra.is_empty() {
        let values = extra.iter().map(|(k, v)| (k.clone(), Value::String(py_str(v)))).collect();
        m.insert("extra".into(), Value::Object(values));
    }
}

fn timestamp(t: SystemTime) -> Value {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    Value::String(rfc3339(d.as_secs() as i64, d.subsec_nanos()))
}

fn rfc3339_secs(secs: f64) -> String {
    let nanos = (secs * 1e9) as i128;
    rfc3339(nanos.div_euclid(1_000_000_000) as i64, nanos.rem_euclid(1_000_000_000) as u32)
}

/// Protobuf JSON's `Timestamp`: UTC, with 0, 3, 6 or 9 fractional digits.
pub(super) fn rfc3339(secs: i64, nanos: u32) -> String {
    chrono::DateTime::from_timestamp(secs, nanos)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
}

/// Python's `str(v)` of a JSON value.
pub(super) fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => py_repr(v),
    }
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
            let mut out = String::with_capacity(s.len() + 2);
            out.push(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        Value::Array(items) => {
            format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", "))
        }
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
