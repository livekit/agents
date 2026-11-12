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

fn frames(state: &mut Value) -> Option<&mut serde_json::Map<String, Value>> {
    if state.get("kind")?.as_str()? != "text" {
        return None;
    }
    state.get_mut("state")?.get_mut("frames")?.as_object_mut()
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
        let Some(frames) = frames(&mut packed) else { return Ok((packed, Vec::new())) };
        let mut writes = Vec::new();
        let mut referenced = BTreeMap::new();
        for frame in frames.values_mut() {
            for field in ["history", "visible_dialogue"] {
                let values = frame
                    .get(field)
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid(format!("session frame has no {field}")))?;
                let mut references = Vec::with_capacity(values.len());
                for value in values {
                    references.push(Value::String(self.reference(
                        value,
                        operation,
                        &mut writes,
                        &mut referenced,
                    )?));
                }
                let key = self.reference(
                    &Value::Array(references),
                    operation,
                    &mut writes,
                    &mut referenced,
                )?;
                frame[field] = Value::String(key);
            }
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
    let snapshot_frames =
        frames(&mut state).ok_or_else(|| invalid("invalid stored session frames"))?;
    let mut keys = BTreeSet::new();
    for frame in snapshot_frames.values() {
        for field in ["history", "visible_dialogue"] {
            keys.insert(
                frame
                    .get(field)
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("invalid session history reference"))?
                    .to_owned(),
            );
        }
    }
    let lists = read_values(conn, partition_key, &keys).await?;
    let mut item_keys = BTreeSet::new();
    let mut cache = SessionItems::default();
    for key in &keys {
        let list = lists
            .get(key)
            .ok_or_else(|| invalid("session checkpoint references missing history"))?;
        item_keys.extend(reference_list(list)?);
        cache.values.insert(
            serde_json::to_string(list).map_err(|error| invalid(error.to_string()))?,
            key.clone(),
        );
    }
    let values = read_values(conn, partition_key, &item_keys).await?;
    for (key, value) in &values {
        cache.values.insert(
            serde_json::to_string(value).map_err(|error| invalid(error.to_string()))?,
            key.clone(),
        );
    }
    for frame in snapshot_frames.values_mut() {
        for field in ["history", "visible_dialogue"] {
            let key = frame
                .get(field)
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("invalid session history reference"))?;
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
            frame[field] = Value::Array(items);
        }
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
    if let Some(frames) = frames(&mut routing) {
        for frame in frames.values_mut() {
            for field in ["history", "visible_dialogue"] {
                if let Some(values) = frame.get_mut(field).and_then(Value::as_array_mut) {
                    values.clear();
                }
            }
        }
    }
    routing
}

#[cfg(test)]
#[path = "session_snapshot_tests.rs"]
mod tests;
