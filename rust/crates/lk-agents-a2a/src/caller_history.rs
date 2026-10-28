//! The delegation conversation part: a deliberately narrower schema than a stored chat context.
//! Both outgoing selection and incoming validation use this schema; decoding never stores history.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lk_agents_abi::llm::chat_context::{
    ChatContent, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role,
};
use lk_agents_storage::run::KIND_KEY;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Explicit application selection; no internal or provider extra is shared by default.
#[derive(Debug, Default)]
pub struct HistorySharing {
    pub extra_keys: BTreeSet<String>,
    pub include_instructions: bool,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("invalid caller history: {0}")]
pub struct HistoryError(pub String);

/// Validated, ordered text-only history. Construction always checks complete tool-call pairs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CallerHistory {
    items: Vec<Item>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Item {
    Message {
        id: String,
        #[serde(with = "role")]
        role: Role,
        content: Vec<String>,
        created_at: f64,
        #[serde(default)]
        interrupted: bool,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        extra: Map<String, Value>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        created_at: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group_id: Option<String>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        extra: Map<String, Value>,
    },
    FunctionCallOutput {
        id: String,
        call_id: String,
        output: String,
        is_error: bool,
        created_at: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default = "reply_required")]
        reply_required: bool,
    },
}

fn reply_required() -> bool {
    true
}

mod role {
    use super::Role;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Role, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value.as_str())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Role, D::Error> {
        let value = String::deserialize(deserializer)?;
        Role::from_literal(&value).ok_or_else(|| serde::de::Error::custom("invalid message role"))
    }
}

impl CallerHistory {
    /// Extract the one explicitly tagged conversation part. Malformed or duplicate parts fail
    /// instead of silently becoming an absent history or choosing one conflicting conversation.
    pub fn from_message(message: &a2a::Message) -> Result<Option<Self>, HistoryError> {
        let mut history = None;
        for part in &message.parts {
            if part
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get(KIND_KEY))
                .and_then(Value::as_str)
                != Some("chat_ctx")
            {
                continue;
            }
            if history.is_some() {
                return Err(HistoryError(
                    "a message may contain only one conversation part".into(),
                ));
            }
            let a2a::PartContent::Data(value) = &part.content else {
                return Err(HistoryError("a conversation part must contain data".into()));
            };
            history = Some(Self::decode(value.clone())?);
        }
        Ok(history)
    }

    /// Validate the dedicated wire schema, including cross-item references, once at ingress.
    pub fn decode(value: Value) -> Result<Self, HistoryError> {
        let payload: Payload =
            serde_json::from_value(value).map_err(|error| HistoryError(error.to_string()))?;
        let history = Self { items: payload.items };
        history.validate()?;
        Ok(history)
    }

    /// Select public text and complete tool-call pairs, retaining their original order and IDs.
    /// The invoking delegation is excluded even if its history contains a waiting placeholder.
    pub fn from_chat_items(
        items: &[ChatItem],
        unresolved_call_id: Option<&str>,
        sharing: &HistorySharing,
    ) -> Result<Self, HistoryError> {
        let mut calls = BTreeMap::new();
        let mut complete = BTreeSet::new();
        for (index, item) in items.iter().enumerate() {
            match item {
                ChatItem::FunctionCall(call) => {
                    if calls.insert(call.call_id.as_str(), index).is_some() {
                        return Err(HistoryError(format!("duplicate call {}", call.call_id)));
                    }
                }
                ChatItem::FunctionCallOutput(output)
                    if calls.contains_key(output.call_id.as_str()) =>
                {
                    complete.insert(output.call_id.as_str());
                }
                _ => {}
            }
        }
        if let Some(call_id) = unresolved_call_id {
            complete.remove(call_id);
        }
        let mut selected = Vec::new();
        for item in items {
            let item = match item {
                ChatItem::Message(message)
                    if sharing.include_instructions
                        || !matches!(message.role, Role::System | Role::Developer) =>
                {
                    Item::Message {
                        id: message.id.clone(),
                        role: message.role,
                        content: message
                            .content
                            .iter()
                            .filter_map(|content| match content {
                                ChatContent::Text(text) => Some(text.clone()),
                                _ => None,
                            })
                            .collect(),
                        created_at: seconds(message.created_at),
                        interrupted: message.interrupted,
                        extra: selected_extra(&message.extra, sharing),
                    }
                }
                ChatItem::FunctionCall(call) if complete.contains(call.call_id.as_str()) => {
                    Item::FunctionCall {
                        id: call.id.clone(),
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                        created_at: seconds(call.created_at),
                        group_id: call.group_id.clone(),
                        extra: selected_extra(&call.extra, sharing),
                    }
                }
                ChatItem::FunctionCallOutput(output)
                    if complete.contains(output.call_id.as_str()) =>
                {
                    Item::FunctionCallOutput {
                        id: output.id.clone(),
                        call_id: output.call_id.clone(),
                        output: output.output.clone(),
                        is_error: output.is_error,
                        created_at: seconds(output.created_at),
                        name: (!output.name.is_empty()).then(|| output.name.clone()),
                        reply_required: output.reply_required,
                    }
                }
                _ => continue,
            };
            selected.push(item);
        }
        let history = Self { items: selected };
        history.validate()?;
        Ok(history)
    }

    pub fn to_value(&self) -> Result<Value, HistoryError> {
        serde_json::to_value(self).map_err(|error| HistoryError(error.to_string()))
    }

    /// The single peer-facing representation used by clients and remote tools.
    pub fn to_part(&self) -> Result<a2a::Part, HistoryError> {
        let mut part = a2a::Part::data(self.to_value()?);
        part.metadata =
            Some([(lk_agents_storage::run::KIND_KEY.to_owned(), "chat_ctx".into())].into());
        Ok(part)
    }

    /// Fresh model-history values; the caller decides where to use them, never persistence.
    pub fn to_chat_items(&self) -> Vec<ChatItem> {
        self.items
            .iter()
            .map(|item| match item {
                Item::Message { id, role, content, created_at, interrupted, extra } => {
                    let mut message = ChatMessage::new(
                        id.clone(),
                        *role,
                        content.iter().cloned().map(ChatContent::Text).collect(),
                        time(*created_at).expect("validated timestamp"),
                    );
                    message.interrupted = *interrupted;
                    message.extra = extra.clone();
                    ChatItem::Message(message)
                }
                Item::FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                    created_at,
                    group_id,
                    extra,
                } => ChatItem::FunctionCall(FunctionCall {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    created_at: time(*created_at).expect("validated timestamp"),
                    group_id: group_id.clone(),
                    extra: extra.clone(),
                }),
                Item::FunctionCallOutput {
                    id,
                    call_id,
                    output,
                    is_error,
                    created_at,
                    name,
                    reply_required,
                } => {
                    let mut output = FunctionCallOutput::new(
                        id.clone(),
                        call_id.clone(),
                        output.clone(),
                        *is_error,
                        time(*created_at).expect("validated timestamp"),
                    );
                    output.name = name.clone().unwrap_or_default();
                    output.reply_required = *reply_required;
                    ChatItem::FunctionCallOutput(output)
                }
            })
            .collect()
    }

    fn validate(&self) -> Result<(), HistoryError> {
        let mut ids = BTreeSet::new();
        let mut calls = BTreeMap::new();
        let mut pending = BTreeSet::new();
        for item in &self.items {
            let (id, created_at) = match item {
                Item::Message { id, created_at, extra, .. } => {
                    validate_extra(extra)?;
                    if !pending.is_empty() {
                        return Err(HistoryError("a message splits a call/output group".into()));
                    }
                    (id, *created_at)
                }
                Item::FunctionCall { id, call_id, name, arguments, created_at, extra, .. } => {
                    validate_extra(extra)?;
                    if call_id.is_empty() || name.is_empty() {
                        return Err(HistoryError("a call needs call_id and name".into()));
                    }
                    let arguments: Value = serde_json::from_str(arguments)
                        .map_err(|error| HistoryError(format!("call arguments: {error}")))?;
                    if !arguments.is_object() {
                        return Err(HistoryError(
                            "call arguments must contain a JSON object".into(),
                        ));
                    }
                    if calls.insert(call_id.as_str(), name.as_str()).is_some() {
                        return Err(HistoryError(format!("duplicate call {call_id}")));
                    }
                    pending.insert(call_id.as_str());
                    (id, *created_at)
                }
                Item::FunctionCallOutput { id, call_id, created_at, name, .. } => {
                    if !pending.remove(call_id.as_str()) {
                        return Err(HistoryError(format!(
                            "output has no preceding pending call {call_id}"
                        )));
                    }
                    if name
                        .as_deref()
                        .is_some_and(|name| calls.get(call_id.as_str()) != Some(&name))
                    {
                        return Err(HistoryError(format!(
                            "output name does not match call {call_id}"
                        )));
                    }
                    (id, *created_at)
                }
            };
            if id.is_empty() || !ids.insert(id.as_str()) {
                return Err(HistoryError("item IDs must be nonempty and unique".into()));
            }
            if time(created_at).is_none() {
                return Err(HistoryError("created_at must be finite Unix seconds".into()));
            }
        }
        if !pending.is_empty() {
            return Err(HistoryError("a function call is missing its output".into()));
        }
        Ok(())
    }
}

fn seconds(value: SystemTime) -> f64 {
    match value.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64(),
        Err(error) => -error.duration().as_secs_f64(),
    }
}

fn time(seconds: f64) -> Option<SystemTime> {
    let duration = Duration::try_from_secs_f64(seconds.abs()).ok()?;
    if seconds.is_sign_negative() {
        UNIX_EPOCH.checked_sub(duration)
    } else {
        UNIX_EPOCH.checked_add(duration)
    }
}

fn selected_extra(extra: &Map<String, Value>, sharing: &HistorySharing) -> Map<String, Value> {
    extra
        .iter()
        .filter(|(key, value)| {
            sharing.extra_keys.contains(*key) && public_key(key) && public_value(value)
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn validate_extra(extra: &Map<String, Value>) -> Result<(), HistoryError> {
    if extra.iter().all(|(key, value)| public_key(key) && public_value(value)) {
        Ok(())
    } else {
        Err(HistoryError("extra contains private framework, credential or provider data".into()))
    }
}

fn public_value(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.iter().all(|(key, value)| public_key(key) && public_value(value))
        }
        Value::Array(array) => array.iter().all(public_value),
        _ => true,
    }
}

fn public_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    !key.starts_with(['_', '$'])
        && !key.starts_with("lk.")
        && !key.starts_with("livekit/")
        && !key.starts_with("https://livekit.io/")
        && !matches!(
            key.as_str(),
            "approval"
                | "delegation"
                | "authorization"
                | "credential"
                | "credentials"
                | "password"
                | "secret"
                | "api_key"
                | "api_secret"
                | "token"
                | "access_token"
                | "refresh_token"
                | "provider"
                | "provider_data"
                | "provider_payload"
                | "thought_signature"
                | "thought_signatures"
                | "openai"
                | "anthropic"
                | "google"
                | "aws"
                | "azure"
        )
}

#[cfg(test)]
mod tests;
