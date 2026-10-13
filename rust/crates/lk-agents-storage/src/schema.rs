//! The thread schema. Its DDL is livekit/protocol `livekit/agent/thread_schema_v1.sql`, run as is
//! ([`DDL`]). Each table is a record with `#[derive(Host)]` and `#[table]`, from which
//! `lk-agents-codegen` writes the pydantic row models Python reads rows as (`ChatItemRow`,
//! `EventRow`, `LaneRow`, `InboxRow`, `TaskRow`), and the typed decoders below turn a wire row into
//! the record.
//!
//! JSONB columns are written with `jsonb(?)` and read with `json(col)` (each record's `SELECT`),
//! since the wire `Value` has no JSON type. `items.seq` and `tasks.seq` within a lane are assigned
//! by the commit, `COALESCE(MAX(seq), 0) + 1` per row; `events.id` and `inbox.id` are the rowid.

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::StorageError;
use crate::wire::{Row, SqlValue};

/// Version one stores exactly these tables. The SDK applies the DDL on a session's first write and
/// migrations on open when the version is behind; agent-db never does.
pub const SCHEMA_VERSION: i64 = 1;

/// One history item of a lane: what the model sees, owned by the developer, editable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table]
pub struct ChatItemRow {
    /// The upsert key. `pos`, the rowid, is the platform's: its page token's sort key.
    pub id: String,
    pub lane: String,
    /// Assigned by the commit, per lane.
    pub seq: i64,
    /// The A2A `messageId` of the message this item came from, the idempotency key.
    pub message_id: Option<String>,
    /// The chat item in its in-memory form (`chat_item`); the column holds its protobuf JSON.
    pub payload: Value,
}

/// One A2A task of a lane (plan 5.5): a turn or a submitted `AgentTask`. The A2A `Task` is
/// `id = task_id`, `contextId = the thread id`, `status = (status, status_message, updated_at)`,
/// plus the fields of `data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table]
pub struct TaskRow {
    /// The upsert key. `pos`, the rowid, is the platform's: its page token's sort key.
    pub task_id: String,
    pub lane: String,
    /// Submission order; the lane runs its runnable tasks in it.
    pub seq: i64,
    /// The `AgentTask` class; `None` for a turn.
    pub name: Option<String>,
    pub message_id: Option<String>,
    /// `submitted`, `working`, `input-required`, `completed`, `failed`, `canceled`, `rejected`.
    pub status: String,
    /// `Task.status.message`.
    pub status_message: Option<Value>,
    pub updated_at: i64,
    /// The A2A `Task` fields under their A2A names: `history`, `artifacts`, `metadata`.
    pub data: Value,
    /// The framework's own: the `AgentTask` arguments, the pending `ask`, `deadline`, `parent`.
    pub internal: Option<Value>,
}

/// The `data` column: the A2A `Task` fields a row keeps, under their A2A names; an absent field is
/// left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct TaskData {
    /// `Task.history`: A2A messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[python(default_factory = "list")]
    pub history: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub artifacts: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub metadata: Map<String, Value>,
}

/// The `internal` column: what the framework keeps to resume an `AgentTask`; an absent field is left
/// out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct TaskInternal {
    /// The `AgentTask` constructor arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub args: Option<Value>,
    /// The pending transition: the tool that takes the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub ask: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub deadline: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub parent: Option<String>,
}

impl TaskInternal {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// One lane: a registered agent's history, state and agent stack, with its lease (plan 3.3.2) and
/// its inbox cursor (3.3.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table]
pub struct LaneRow {
    pub lane: String,
    /// The fencing token: a run loaded at `v` commits only `WHERE version = v`.
    pub version: i64,
    /// The application's typed state, one snapshot per run.
    pub state: Option<Value>,
    /// A list of `Ref {name, args}`: who is speaking.
    pub agent_stack: Option<Value>,
    pub lease_holder: Option<String>,
    pub lease_token: Option<String>,
    /// Unix seconds on agent-db's clock; `0` is released.
    pub lease_until: i64,
    /// A worker that failed the lease under `interrupt` and is polling for it.
    pub waiter: Option<String>,
    /// The id of the last inbox row a drain consumed.
    pub drained_id: i64,
}

/// One journal entry: every `AgentSessionEvent`, plus `run_started` and `run_ended`, tagged with
/// the lane. Append-only, owned by the framework.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table]
pub struct EventRow {
    /// Insertion order.
    pub id: i64,
    pub ts_ms: i64,
    pub lane: String,
    pub run_id: String,
    #[host(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

/// One message received while the lane was running, under the `inbox` policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table]
pub struct InboxRow {
    /// Arrival order.
    pub id: i64,
    pub lane: String,
    /// The task created for the message; also its answer stream.
    pub task_id: String,
    pub message_id: String,
    /// The A2A message.
    pub message: Value,
    pub received_at: i64,
}

/// The schema, every statement of it: livekit/protocol `livekit/agent/thread_schema_v1.sql`.
pub const DDL: &str = include_str!("../../../../protocol/livekit/agent/thread_schema_v1.sql");

/// [`DDL`]'s statements, in order, one per `Batch` statement.
pub fn ddl() -> impl Iterator<Item = &'static str> {
    DDL.split(';').map(str::trim).filter(|s| !s.is_empty())
}

/// `schema_version`'s one row, written with the DDL.
pub const INSERT_SCHEMA_VERSION: &str = "INSERT INTO schema_version (version) VALUES (?);";

// ---------------------------------------------------------------------------------------------
// decoding
// ---------------------------------------------------------------------------------------------

/// Typed reads of one row's cells, positional, each failure naming the column.
pub struct Cells<'a> {
    row: &'a Row,
    columns: &'static [&'static str],
}

impl<'a> Cells<'a> {
    pub fn new(row: &'a Row, columns: &'static [&'static str]) -> Result<Self, StorageError> {
        if row.len() != columns.len() {
            return Err(StorageError::Decode(format!(
                "expected {} columns, got {}",
                columns.len(),
                row.len()
            )));
        }
        Ok(Self { row, columns })
    }

    fn cell(&self, i: usize) -> &'a SqlValue {
        &self.row[i]
    }

    fn name(&self, i: usize) -> &'static str {
        self.columns[i]
    }

    pub fn int(&self, i: usize) -> Result<i64, StorageError> {
        self.cell(i)
            .as_i64()
            .ok_or_else(|| StorageError::Decode(format!("{}: expected an integer", self.name(i))))
    }

    pub fn opt_int(&self, i: usize) -> Result<Option<i64>, StorageError> {
        if self.cell(i).is_null() { Ok(None) } else { self.int(i).map(Some) }
    }

    pub fn text(&self, i: usize) -> Result<String, StorageError> {
        self.cell(i)
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| StorageError::Decode(format!("{}: expected text", self.name(i))))
    }

    pub fn opt_text(&self, i: usize) -> Result<Option<String>, StorageError> {
        if self.cell(i).is_null() { Ok(None) } else { self.text(i).map(Some) }
    }

    pub fn json(&self, i: usize) -> Result<Value, StorageError> {
        let text = self
            .cell(i)
            .as_str()
            .ok_or_else(|| StorageError::Decode(format!("{}: expected JSON text", self.name(i))))?;
        serde_json::from_str(text)
            .map_err(|e| StorageError::Decode(format!("{}: {e}", self.name(i))))
    }

    pub fn opt_json(&self, i: usize) -> Result<Option<Value>, StorageError> {
        if self.cell(i).is_null() { Ok(None) } else { self.json(i).map(Some) }
    }
}

impl ChatItemRow {
    pub fn from_row(row: &Row) -> Result<Self, StorageError> {
        let c = Cells::new(row, Self::COLUMNS)?;
        Ok(Self {
            id: c.text(0)?,
            lane: c.text(1)?,
            seq: c.int(2)?,
            message_id: c.opt_text(3)?,
            payload: crate::chat_item::decode(&c.json(4)?)?,
        })
    }
}

impl EventRow {
    pub fn from_row(row: &Row) -> Result<Self, StorageError> {
        let c = Cells::new(row, Self::COLUMNS)?;
        Ok(Self {
            id: c.int(0)?,
            ts_ms: c.int(1)?,
            lane: c.text(2)?,
            run_id: c.text(3)?,
            r#type: c.text(4)?,
            payload: c.json(5)?,
        })
    }
}

impl LaneRow {
    pub fn from_row(row: &Row) -> Result<Self, StorageError> {
        let c = Cells::new(row, Self::COLUMNS)?;
        Ok(Self {
            lane: c.text(0)?,
            version: c.int(1)?,
            state: c.opt_json(2)?,
            agent_stack: c.opt_json(3)?,
            lease_holder: c.opt_text(4)?,
            lease_token: c.opt_text(5)?,
            lease_until: c.int(6)?,
            waiter: c.opt_text(7)?,
            drained_id: c.int(8)?,
        })
    }
}

impl InboxRow {
    pub fn from_row(row: &Row) -> Result<Self, StorageError> {
        let c = Cells::new(row, Self::COLUMNS)?;
        Ok(Self {
            id: c.int(0)?,
            lane: c.text(1)?,
            task_id: c.text(2)?,
            message_id: c.text(3)?,
            message: c.json(4)?,
            received_at: c.int(5)?,
        })
    }
}

impl TaskRow {
    pub fn from_row(row: &Row) -> Result<Self, StorageError> {
        let c = Cells::new(row, Self::COLUMNS)?;
        Ok(Self {
            task_id: c.text(0)?,
            lane: c.text(1)?,
            seq: c.int(2)?,
            name: c.opt_text(3)?,
            message_id: c.opt_text(4)?,
            status: c.text(5)?,
            status_message: c.opt_json(6)?,
            updated_at: c.int(7)?,
            data: c.json(8)?,
            internal: c.opt_json(9)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddl_splits_into_its_statements() {
        let stmts: Vec<_> = ddl().collect();
        assert!(stmts[0].contains("CREATE TABLE schema_version ("), "{}", stmts[0]);
        assert_eq!(
            stmts.last().unwrap(),
            &"CREATE UNIQUE INDEX inbox_lane_message ON inbox (lane, message_id)"
        );
        assert_eq!(stmts.iter().filter(|s| s.contains("CREATE TABLE")).count(), 6);
    }

    #[test]
    fn selects_read_json_columns_as_text() {
        assert_eq!(ChatItemRow::SELECT, "id, lane, seq, message_id, json(payload)");
        assert_eq!(
            LaneRow::SELECT,
            "lane, version, json(state), json(agent_stack), lease_holder, lease_token, lease_until, waiter, drained_id"
        );
        assert_eq!(
            TaskRow::SELECT,
            "task_id, lane, seq, name, message_id, status, json(status_message), updated_at, json(data), json(internal)"
        );
    }

    #[test]
    fn rows_decode_with_nulls_and_json() {
        let row: Row = vec![
            "item-1".into(),
            "front-desk".into(),
            SqlValue::Int(3),
            SqlValue::Null,
            r#"{"message":{"id":"item-1","role":"USER","content":[{"text":"hi"}]}}"#.into(),
        ];
        let item = ChatItemRow::from_row(&row).unwrap();
        assert_eq!(item.message_id, None);
        assert_eq!(item.payload["role"], "user");
        assert_eq!(item.lane, "front-desk");

        let bad: Row =
            vec!["i".into(), "l".into(), "not an int".into(), SqlValue::Null, "{}".into()];
        assert!(
            matches!(ChatItemRow::from_row(&bad), Err(StorageError::Decode(m)) if m.starts_with("seq"))
        );
    }
}
