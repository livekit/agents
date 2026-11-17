//! Immutable history revisions referenced by compact session routing snapshots.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::sql::{Bind, FromCell, SqlValue};
use crate::thread::schema::SelectSessionItems;
use crate::{StorageError, ThreadDbConnection};

#[derive(Clone, Default)]
pub(crate) struct SessionItems {
    /// Canonical encoded value -> immutable journal key; shared fork prefixes reuse entries.
    values: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(crate) struct SessionItemWrite {
    pub key: String,
    pub value: Value,
}

fn invalid(message: impl Into<String>) -> StorageError {
    StorageError::Invalid(message.into())
}

enum SessionContent<'a> {
    Sequence(&'a mut Value),
    Value(&'a mut Value),
}

fn content(state: &mut Value) -> Option<Vec<SessionContent<'_>>> {
    if state.get("kind")?.as_str()? != "conversation" {
        return None;
    }
    let mut content = Vec::new();
    let envelope = state.as_object_mut()?;
    for (field, value) in envelope {
        if field == "pending_user_turns" {
            content.push(SessionContent::Sequence(value));
            continue;
        }
        if field != "state" {
            continue;
        }
        let data = value.as_object_mut()?;
        for (field, value) in data {
            match field.as_str() {
                "chat_history" => content.push(SessionContent::Sequence(value)),
                "frames" => {
                    for frame in value.as_object_mut()?.values_mut() {
                        for (field, value) in frame.as_object_mut()? {
                            if matches!(field.as_str(), "history" | "visible_dialogue") {
                                content.push(SessionContent::Sequence(value));
                            } else if field == "input_requests" {
                                for input in value.as_object_mut()?.values_mut() {
                                    for (field, value) in input.as_object_mut()? {
                                        if matches!(
                                            field.as_str(),
                                            "source_message" | "response_message"
                                        ) && !value.is_null()
                                        {
                                            content.push(SessionContent::Value(value));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                "remote_agents" => {
                    for remote in value.as_object_mut()?.values_mut() {
                        for (field, value) in remote.as_object_mut()? {
                            match field.as_str() {
                                "events" => content.push(SessionContent::Sequence(value)),
                                "messages" => {
                                    for message in value.as_object_mut()?.values_mut() {
                                        content.push(SessionContent::Value(message));
                                    }
                                }
                                "tasks" => {
                                    for task in value.as_object_mut()?.values_mut() {
                                        content.push(SessionContent::Value(task.get_mut("task")?));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Some(content)
}

impl SessionItems {
    fn reference(
        &mut self,
        value: &Value,
        operation: &str,
        writes: &mut Vec<SessionItemWrite>,
        referenced: &mut BTreeMap<String, String>,
    ) -> Result<String, StorageError> {
        let encoded = serde_json::to_string(value).map_err(|error| invalid(error.to_string()))?;
        let key = match self.values.get(&encoded) {
            Some(key) => key.clone(),
            None => {
                let key = format!("{operation}/{}", writes.len());
                writes.push(SessionItemWrite { key: key.clone(), value: value.clone() });
                self.values.insert(encoded.clone(), key.clone());
                key
            }
        };
        referenced.insert(encoded, key.clone());
        Ok(key)
    }

    /// Payload revisions and ordered item-reference lists are immutable. Routing stores one
    /// list reference per frame, so its size does not grow with the conversation length.
    pub(crate) fn pack(
        &mut self,
        state: &Value,
        operation: &str,
    ) -> Result<(Value, Vec<SessionItemWrite>), StorageError> {
        let mut packed = state.clone();
        let Some(content) = content(&mut packed) else { return Ok((packed, Vec::new())) };
        let mut writes = Vec::new();
        let mut referenced = BTreeMap::new();
        for value in content {
            let history = match value {
                SessionContent::Value(value) => {
                    let key = self.reference(value, operation, &mut writes, &mut referenced)?;
                    *value = Value::String(key);
                    continue;
                }
                SessionContent::Sequence(value) => value,
            };
            let values = history.as_array().ok_or_else(|| invalid("invalid session history"))?;
            let mut references = Vec::with_capacity(values.len());
            for value in values {
                references.push(Value::String(self.reference(
                    value,
                    operation,
                    &mut writes,
                    &mut referenced,
                )?));
            }
            let key =
                self.reference(&Value::Array(references), operation, &mut writes, &mut referenced)?;
            *history = Value::String(key);
        }
        self.values = referenced;
        packed
            .as_object_mut()
            .ok_or_else(|| invalid("session state must be an object"))?
            .insert("storage_version".into(), Value::from(1));
        Ok((packed, writes))
    }
}

async fn read_values(
    conn: &dyn ThreadDbConnection,
    partition_key: &str,
    keys: &BTreeSet<String>,
) -> Result<BTreeMap<String, Value>, StorageError> {
    let keys = serde_json::to_value(keys).map_err(|error| invalid(error.to_string()))?;
    let rows = conn
        .query(SelectSessionItems { partition_key, keys: SqlValue::json(&keys) }.bind())
        .await?;
    rows.into_iter()
        .map(|row| Ok((String::from_cell(&row[0])?, Value::from_cell(&row[1])?)))
        .collect()
}

fn reference_list(value: &Value) -> Result<Vec<String>, StorageError> {
    value
        .as_array()
        .ok_or_else(|| invalid("invalid session history list"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("invalid session item reference"))
        })
        .collect()
}

/// Referenced lists and payloads can be read after routing: their meanings cannot change,
/// and the checkpoint that published the references inserted them in the same transaction.
pub(crate) async fn hydrate(
    conn: &dyn ThreadDbConnection,
    partition_key: &str,
    state: Option<Value>,
) -> Result<(Option<Value>, SessionItems), StorageError> {
    let Some(mut state) = state else { return Ok((None, SessionItems::default())) };
    if state.get("storage_version").is_none() {
        return Ok((Some(state), SessionItems::default()));
    }
    if state.get("storage_version") != Some(&Value::from(1)) {
        return Err(invalid("unsupported stored session version"));
    }
    let content = content(&mut state).ok_or_else(|| invalid("invalid stored session content"))?;
    let mut keys = BTreeSet::new();
    for value in &content {
        let (SessionContent::Sequence(history) | SessionContent::Value(history)) = value;
        keys.insert(
            history
                .as_str()
                .ok_or_else(|| invalid("invalid session history reference"))?
                .to_owned(),
        );
    }
    let lists = read_values(conn, partition_key, &keys).await?;
    let mut item_keys = BTreeSet::new();
    let mut cache = SessionItems::default();
    for value in &content {
        let (SessionContent::Sequence(value) | SessionContent::Value(value)) = value;
        let key = value.as_str().ok_or_else(|| invalid("invalid session content reference"))?;
        let stored = lists
            .get(key)
            .ok_or_else(|| invalid("session checkpoint references missing content"))?;
        cache.values.insert(
            serde_json::to_string(stored).map_err(|error| invalid(error.to_string()))?,
            key.to_owned(),
        );
    }
    for value in &content {
        if let SessionContent::Sequence(value) = value {
            let key = value.as_str().ok_or_else(|| invalid("invalid session list reference"))?;
            item_keys.extend(reference_list(&lists[key])?);
        }
    }
    let values = read_values(conn, partition_key, &item_keys).await?;
    for (key, value) in &values {
        cache.values.insert(
            serde_json::to_string(value).map_err(|error| invalid(error.to_string()))?,
            key.clone(),
        );
    }
    for value in content {
        let history = match value {
            SessionContent::Value(value) => {
                let key =
                    value.as_str().ok_or_else(|| invalid("invalid session content reference"))?;
                *value = lists[key].clone();
                continue;
            }
            SessionContent::Sequence(value) => value,
        };
        let key = history.as_str().ok_or_else(|| invalid("invalid session history reference"))?;
        let list = lists
            .get(key)
            .ok_or_else(|| invalid("session checkpoint references missing history"))?;
        let items = reference_list(list)?
            .iter()
            .map(|key| {
                values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid("session checkpoint references missing item"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        *history = Value::Array(items);
    }
    state
        .as_object_mut()
        .ok_or_else(|| invalid("invalid stored session state"))?
        .remove("storage_version");
    Ok((Some(state), cache))
}

/// Context staging accounts for routing/refs; history payloads are budgeted as item writes,
/// rather than counted a second time against the application's context-state limit.
pub(crate) fn routing_budget(state: &Value) -> Value {
    let mut routing = state.clone();
    if let Some(content) = content(&mut routing) {
        for value in content {
            match value {
                SessionContent::Value(value) => *value = Value::Null,
                SessionContent::Sequence(value) => {
                    if let Some(values) = value.as_array_mut() {
                        values.clear();
                    }
                }
            }
        }
    }
    routing
}

#[cfg(test)]
#[path = "session_snapshot_tests.rs"]
mod tests;
