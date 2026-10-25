//! The incoming A2A message and the policy that handles it (plan 3.3.4, 7.3): `Message`,
//! `channel_of`, `resolve_policy`, `resolve_debounce` and the batch as one user chat message, the
//! port of the top of `storage/policies.py`.

use std::collections::HashMap;
use std::time::Duration;

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::StorageError;
use crate::run::shortuuid;
use crate::run::{CHANNEL_KEY, INTERRUPT_KEY, KIND_KEY};

/// A message during a run: one of the policies of plan 3.3.4, and `live` (docs/delegation.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "lowercase")]
#[python(literal, module = "storage_turn")]
pub enum Policy {
    #[host(rename = "batch")]
    Batch,
    #[host(rename = "interrupt")]
    Interrupt,
    #[host(rename = "queue")]
    Queue,
    #[host(rename = "concurrent")]
    Concurrent,
    #[host(rename = "reject")]
    Reject,
    /// Waits only while the floor is busy: a reply streams or a blocking tool runs. Messages
    /// that waited together run as one turn; one that interrupts stops the reply.
    #[host(rename = "live")]
    Live,
}

impl Policy {
    /// The Python `inbox` string; `None` is the `ValueError` of `resolve_policy`.
    pub fn parse(s: &str) -> Option<Policy> {
        Policy::from_wire(s)
    }
}

/// `inbox` as a door declares it: nothing (the default map), one policy for every
/// channel, or a map that overrides the default map channel by channel. On the wire `null`, a
/// policy string, or an object of them by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PolicySpec {
    #[default]
    Default,
    All(Policy),
    ByChannel(HashMap<String, Policy>),
}

/// `PolicySpec` as JSON carries it: the policy strings, read by name so an unknown one is named
/// in the error rather than reported as "no variant matched".
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PolicyWire {
    All(String),
    ByChannel(HashMap<String, String>),
}

fn policy_named(s: &str) -> Result<Policy, String> {
    Policy::parse(s).ok_or_else(|| format!("unknown policy {s:?}"))
}

impl Serialize for PolicySpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            PolicySpec::Default => None,
            PolicySpec::All(p) => Some(PolicyWire::All(p.as_str().to_owned())),
            PolicySpec::ByChannel(m) => Some(PolicyWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_str().to_owned())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for PolicySpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let named = |s: &str| policy_named(s).map_err(serde::de::Error::custom);
        Ok(match Option::<PolicyWire>::deserialize(d)? {
            None => PolicySpec::Default,
            Some(PolicyWire::All(s)) => PolicySpec::All(named(&s)?),
            Some(PolicyWire::ByChannel(m)) => PolicySpec::ByChannel(
                m.into_iter().map(|(k, v)| named(&v).map(|p| (k, p))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
/// map per channel. On the wire `null`, seconds, or an object of seconds by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DebounceSpec {
    #[default]
    Default,
    All(Duration),
    ByChannel(HashMap<String, Duration>),
}

/// `DebounceSpec` as JSON carries it: seconds as a number.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum DebounceWire {
    All(f64),
    ByChannel(HashMap<String, f64>),
}

/// Seconds as a `Duration`; negative, infinite or NaN is the caller's error.
pub(crate) fn duration_from_secs(secs: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(secs).map_err(|e| e.to_string())
}

impl Serialize for DebounceSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            DebounceSpec::Default => None,
            DebounceSpec::All(d) => Some(DebounceWire::All(d.as_secs_f64())),
            DebounceSpec::ByChannel(m) => Some(DebounceWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_secs_f64())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for DebounceSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let secs = |s: f64| duration_from_secs(s).map_err(serde::de::Error::custom);
        Ok(match Option::<DebounceWire>::deserialize(d)? {
            None => DebounceSpec::Default,
            Some(DebounceWire::All(s)) => DebounceSpec::All(secs(s)?),
            Some(DebounceWire::ByChannel(m)) => DebounceSpec::ByChannel(
                m.into_iter().map(|(k, v)| secs(v).map(|d| (k, d))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// Plan 7.3 and docs/delegation.md: the default map, keyed by `livekit/channel`.
pub const DEFAULT_POLICY_MAP: [(&str, Policy); 5] = [
    ("sms", Policy::Batch),
    ("whatsapp", Policy::Batch),
    ("email", Policy::Batch),
    ("a2a", Policy::Live),
    ("delegation", Policy::Live),
];

/// Plan 7.3: two seconds for the inbox channels, zero for `a2a`; starting points.
pub const DEFAULT_DEBOUNCE: [(&str, Duration); 5] = [
    ("sms", Duration::from_secs(2)),
    ("whatsapp", Duration::from_secs(2)),
    ("email", Duration::from_secs(2)),
    ("a2a", Duration::ZERO),
    ("delegation", Duration::ZERO),
];

pub const INTERRUPTION_NEW_MESSAGE: &str = "new-message";
pub const A2A_CHANNEL: &str = "a2a";
pub const DELEGATION_CHANNEL: &str = "delegation";

/// `livekit/channel` of an A2A message; `livekit/kind = delegation` counts as `delegation`, and a
/// message with no channel counts as `a2a`.
pub fn channel_of(message: &Value) -> String {
    let metadata = message.get("metadata").and_then(Value::as_object);
    if metadata.and_then(|m| m.get(KIND_KEY)).and_then(Value::as_str) == Some(DELEGATION_CHANNEL) {
        return DELEGATION_CHANNEL.to_owned();
    }
    match metadata.and_then(|m| m.get(CHANNEL_KEY)) {
        Some(c) if truthy(c) => match c {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
        _ => A2A_CHANNEL.to_owned(),
    }
}

/// One policy for a channel: a value applies to every channel, a map overrides the default map;
/// a channel in neither gets the `a2a` policy.
pub fn resolve_policy(policy: &PolicySpec, channel: &str) -> Policy {
    if let PolicySpec::All(p) = policy {
        return *p;
    }
    let mut table: HashMap<&str, Policy> = DEFAULT_POLICY_MAP.into_iter().collect();
    if let PolicySpec::ByChannel(map) = policy {
        for (k, v) in map {
            table.insert(k.as_str(), *v);
        }
    }
    table.get(channel).or_else(|| table.get(A2A_CHANNEL)).copied().unwrap_or(Policy::Live)
}

/// The wait before a turn starts so a burst has landed; per channel, or one value for all. A
/// channel the defaults do not name waits the `sms` default under `batch` and nothing otherwise.
pub fn resolve_debounce(debounce: &DebounceSpec, channel: &str, policy: Policy) -> Duration {
    match debounce {
        DebounceSpec::All(d) => return *d,
        DebounceSpec::ByChannel(map) => {
            if let Some(d) = map.get(channel) {
                return *d;
            }
        }
        DebounceSpec::Default => {}
    }
    let defaults: HashMap<&str, Duration> = DEFAULT_DEBOUNCE.into_iter().collect();
    if let Some(d) = defaults.get(channel) {
        return *d;
    }
    if policy == Policy::Batch { defaults["sms"] } else { Duration::ZERO }
}

/// One incoming A2A message, as the handler sees it in `ctx.messages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct Message {
    pub message_id: String,
    pub parts: Vec<Value>,
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub metadata: Map<String, Value>,
    pub channel: String,
    /// Unix seconds.
    pub received_at: f64,
    pub task_id: Option<String>,
    pub context_id: Option<String>,
    /// The A2A message as received.
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub raw: Map<String, Value>,
}

impl Message {
    /// `Message.from_a2a`: a message without a `messageId` is invalid.
    pub fn from_a2a(
        message: &Value,
        received_at: Option<f64>,
        task_id: Option<&str>,
    ) -> Result<Message, StorageError> {
        let raw = message.as_object().cloned().unwrap_or_default();
        let message_id = match raw.get("messageId") {
            Some(v) if truthy(v) => text_of(v),
            _ => return Err(StorageError::Invalid("an A2A message needs a messageId".into())),
        };
        let metadata = raw.get("metadata").and_then(Value::as_object).cloned().unwrap_or_default();
        // a delegation's `chat_ctx` part is the caller's conversation, read by `caller_history_of`
        // before the message is parsed; it is not part of the turn's text (`strip_profile_parts`)
        let parts: Vec<Value> = raw
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| parts.iter().filter(|p| !is_chat_ctx_part(p)).cloned().collect())
            .unwrap_or_default();
        let mut raw = raw;
        if raw.contains_key("parts") {
            raw.insert("parts".into(), Value::Array(parts.clone()));
        }
        let field = |key: &str| raw.get(key).filter(|v| truthy(v)).map(text_of);
        Ok(Message {
            message_id,
            parts,
            metadata,
            channel: channel_of(message),
            received_at: received_at.unwrap_or_else(now_secs),
            task_id: task_id.map(str::to_owned).or_else(|| field("taskId")),
            context_id: field("contextId"),
            raw,
        })
    }

    /// Whether the message stops the reply the lane is producing (`livekit/interrupt`, `true`
    /// when absent). Only the `live` policy reads it; the others decide by themselves.
    pub fn interrupts(&self) -> bool {
        self.metadata.get(INTERRUPT_KEY) != Some(&Value::Bool(false))
    }

    /// The text parts, joined by newlines.
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter(|p| p.get("kind").and_then(Value::as_str) == Some("text"))
            .map(|p| p.get("text").map(text_of).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The first data part's object.
    pub fn data(&self) -> Option<Map<String, Value>> {
        self.parts
            .iter()
            .filter(|p| p.get("kind").and_then(Value::as_str) == Some("data"))
            .find_map(|p| p.get("data").and_then(Value::as_object).cloned())
    }

    /// The parts as chat content: text as is, data and files as compact JSON.
    pub fn content(&self) -> Vec<String> {
        let mut out = Vec::new();
        for p in &self.parts {
            match p.get("kind").and_then(Value::as_str) {
                Some("text") => out.push(p.get("text").map(text_of).unwrap_or_default()),
                Some("data") => out.push(p.get("data").unwrap_or(&Value::Null).to_string()),
                Some("file") => {
                    out.push(serde_json::json!({"file": p.get("file")}).to_string());
                }
                _ => {}
            }
        }
        out
    }

    /// The message as received, with `messageId`, `role: "user"` and `parts` filled in when the
    /// wire left them out.
    pub fn to_a2a(&self) -> Value {
        let mut out = self.raw.clone();
        out.entry("messageId").or_insert_with(|| Value::String(self.message_id.clone()));
        out.entry("role").or_insert_with(|| Value::String("user".into()));
        out.entry("parts").or_insert_with(|| Value::Array(self.parts.clone()));
        Value::Object(out)
    }
}

/// The batch as one user chat message with every part in order (plan 3.3.4), in the stored item
/// form of `item_to_payload`: `{id, type, role, content, interrupted, extra, created_at}`.
pub fn batch_to_chat_message(messages: &[Message]) -> Value {
    let mut content: Vec<Value> = Vec::new();
    for m in messages {
        content.extend(m.content().into_iter().map(Value::String));
    }
    if content.is_empty() {
        content.push(Value::String(String::new()));
    }
    let created_at = messages.last().map_or_else(now_secs, |m| m.received_at);
    let message_ids: Vec<Value> =
        messages.iter().map(|m| Value::String(m.message_id.clone())).collect();
    serde_json::json!({
        "id": shortuuid("item_"),
        "type": "message",
        "role": "user",
        "content": content,
        "interrupted": false,
        "extra": {"message_ids": message_ids},
        "created_at": created_at,
    })
}

/// Whether a part is a delegation's `chat_ctx` data part: `livekit/kind = chat_ctx` on the part.
fn is_chat_ctx_part(part: &Value) -> bool {
    part.get("kind").and_then(Value::as_str) == Some("data")
        && part.get("metadata").and_then(|m| m.get(KIND_KEY)).and_then(Value::as_str)
            == Some("chat_ctx")
}

/// The `chat_ctx` data part of a delegation (plan 7.5), as the part's `data`: the caller's
/// conversation, read from the A2A message before `Message::from_a2a` strips the part.
pub fn caller_history_of(message: &Value) -> Option<Value> {
    message
        .get("parts")
        .and_then(Value::as_array)?
        .iter()
        .filter(|p| is_chat_ctx_part(p))
        .find_map(|p| p.get("data").filter(|d| d.is_object()).cloned())
}

/// Unix seconds as the Python `time.time()`.
pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `str(value)` for the JSON values a message field may hold.
fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn policy_and_debounce_resolution() {
        let none = PolicySpec::Default;
        assert_eq!(resolve_policy(&none, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&none, "a2a"), Policy::Live);
        assert_eq!(resolve_policy(&none, "delegation"), Policy::Live);
        assert_eq!(resolve_policy(&none, "carrier-pigeon"), Policy::Live);
        let map = PolicySpec::ByChannel([("a2a".to_owned(), Policy::Queue)].into_iter().collect());
        assert_eq!(resolve_policy(&map, "a2a"), Policy::Queue);
        assert_eq!(resolve_policy(&map, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&PolicySpec::All(Policy::Reject), "sms"), Policy::Reject);
        assert_eq!(Policy::parse("later"), None);
        assert_eq!(Policy::parse("batch"), Some(Policy::Batch));
        let d = DebounceSpec::Default;
        assert_eq!(resolve_debounce(&d, "sms", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "a2a", Policy::Live), Duration::ZERO);
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Queue), Duration::ZERO);
        let one = DebounceSpec::All(Duration::from_secs(1));
        assert_eq!(resolve_debounce(&one, "a2a", Policy::Interrupt), Duration::from_secs(1));
        let per = DebounceSpec::ByChannel(
            [("sms".to_owned(), Duration::from_secs(5))].into_iter().collect(),
        );
        assert_eq!(resolve_debounce(&per, "sms", Policy::Batch), Duration::from_secs(5));
        assert_eq!(resolve_debounce(&per, "email", Policy::Batch), Duration::from_secs(2));
    }

    #[test]
    fn a_message_reads_its_channel_text_data_and_round_trips() {
        let raw = json!({
            "messageId": "m1",
            "role": "user",
            "parts": [
                {"kind": "text", "text": "hi"},
                {"kind": "data", "data": {"b": 1, "a": 2}},
                {"kind": "file", "file": {"uri": "x"}},
                {"kind": "text", "text": "there"},
            ],
            "metadata": {"livekit/channel": "sms"},
            "taskId": "AST_1",
            "contextId": "AT_1",
        });
        let m = Message::from_a2a(&raw, Some(5.0), None).unwrap();
        assert_eq!((m.message_id.as_str(), m.channel.as_str()), ("m1", "sms"));
        assert_eq!(m.task_id.as_deref(), Some("AST_1"));
        assert_eq!(m.context_id.as_deref(), Some("AT_1"));
        assert_eq!(m.text(), "hi\nthere");
        assert_eq!(m.data(), json!({"b": 1, "a": 2}).as_object().cloned());
        assert_eq!(m.content(), vec!["hi", r#"{"b":1,"a":2}"#, r#"{"file":{"uri":"x"}}"#, "there"]);
        assert_eq!(m.to_a2a(), raw, "the raw message comes back as it was");
        let given = Message::from_a2a(&raw, None, Some("AST_9")).unwrap();
        assert_eq!(given.task_id.as_deref(), Some("AST_9"));
        assert!(given.received_at > 1.0e9);

        let bare = Message::from_a2a(&json!({"messageId": "m2"}), Some(1.0), None).unwrap();
        assert_eq!(bare.channel, "a2a");
        assert_eq!(
            bare.to_a2a().to_string(),
            r#"{"messageId":"m2","role":"user","parts":[]}"#,
            "the defaults are filled in after the raw keys"
        );
        assert!(Message::from_a2a(&json!({"parts": []}), None, None).is_err());
        let delegated = json!({"messageId": "x", "parts": [],
            "metadata": {"livekit/kind": "delegation", "livekit/interrupt": false}});
        let delegated = Message::from_a2a(&delegated, None, None).unwrap();
        assert_eq!(delegated.channel, "delegation");
        assert!(!delegated.interrupts(), "a delegation says it does not interrupt");
        assert!(m.interrupts(), "absent is an interrupt");
        let with_history = json!({"messageId": "y", "parts": [
            {"kind": "text", "text": "fare?"},
            {"kind": "data", "data": {"items": []}, "metadata": {"livekit/kind": "chat_ctx"}},
        ]});
        assert_eq!(caller_history_of(&with_history), Some(json!({"items": []})));
        let stripped = Message::from_a2a(&with_history, Some(1.0), None).unwrap();
        assert_eq!(stripped.parts, vec![json!({"kind": "text", "text": "fare?"})]);
        assert_eq!(stripped.to_a2a()["parts"], json!([{"kind": "text", "text": "fare?"}]));
        assert_eq!(channel_of(&json!({"metadata": {"livekit/channel": ""}})), "a2a");
    }

    #[test]
    fn the_batch_is_one_user_message_with_every_part_in_order() {
        let a = Message::from_a2a(
            &json!({"messageId": "m1", "parts": [{"kind": "text", "text": "where"}]}),
            Some(1.0),
            None,
        )
        .unwrap();
        let b = Message::from_a2a(
            &json!({"messageId": "m2", "parts": [{"kind": "text", "text": "are you"}]}),
            Some(2.0),
            None,
        )
        .unwrap();
        let mut item = batch_to_chat_message(&[a, b]);
        let id = item["id"].as_str().unwrap().to_owned();
        assert!(id.starts_with("item_") && id.len() == 17);
        item["id"] = json!("item_X");
        assert_eq!(
            item.to_string(),
            r#"{"id":"item_X","type":"message","role":"user","content":["where","are you"],"interrupted":false,"extra":{"message_ids":["m1","m2"]},"created_at":2.0}"#
        );
        let empty = batch_to_chat_message(&[]);
        assert_eq!(empty["content"], json!([""]));
    }
}
