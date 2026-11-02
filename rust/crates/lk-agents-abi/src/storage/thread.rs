//! The typed thread records the storage port speaks: what A2A task decisions
//! (`lk_agents_core::a2a`) read and request, and `lk-agents-storage` reads and writes. The host
//! reads their twins in [`super`]; chat items are `livekit.agent` records and A2A messages the
//! `a2a` crate's.

use std::time::Duration;

use livekit_protocol::agent::ChatRole;
use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use serde_json::{Map, Value};

use super::{Ask, StorageError, TaskInternal, TaskState};

/// One history item of a lane.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ItemRow {
    pub id: String,
    /// The A2A `messageId` of the message this item came from, the idempotency key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub message_id: Option<String>,
    /// The `ChatContext.ChatItem`, as protobuf JSON.
    pub payload: ChatItem,
}

impl ItemRow {
    pub fn new(kind: chat_item::Item) -> Self {
        let id = match &kind {
            chat_item::Item::Message(m) => &m.id,
            chat_item::Item::FunctionCall(f) => &f.id,
            chat_item::Item::FunctionCallOutput(f) => &f.id,
            chat_item::Item::AgentHandoff(h) => &h.id,
            chat_item::Item::AgentConfigUpdate(c) => &c.id,
        }
        .clone();
        Self { id, message_id: None, payload: ChatItem { item: Some(kind) } }
    }

    pub fn kind(&self) -> Option<&chat_item::Item> {
        self.payload.item.as_ref()
    }

    pub fn kind_mut(&mut self) -> Option<&mut chat_item::Item> {
        self.payload.item.as_mut()
    }

    /// Unix seconds; `0` when unset.
    pub fn created_at(&self) -> f64 {
        let t = match self.kind() {
            Some(chat_item::Item::Message(m)) => m.created_at.as_ref(),
            Some(chat_item::Item::FunctionCall(f)) => f.created_at.as_ref(),
            Some(chat_item::Item::FunctionCallOutput(f)) => f.created_at.as_ref(),
            Some(chat_item::Item::AgentHandoff(h)) => h.created_at.as_ref(),
            Some(chat_item::Item::AgentConfigUpdate(c)) => c.created_at.as_ref(),
            None => None,
        };
        t.map_or(0.0, |t| t.seconds as f64 + f64::from(t.nanos) / 1e9)
    }

    /// Instructions are rendered per run, never stored, and `agent_config_update` is a
    /// runtime note to the model, not a chat item of the extension.
    pub fn stored(&self) -> bool {
        match self.kind() {
            Some(chat_item::Item::AgentConfigUpdate(_)) | None => false,
            Some(chat_item::Item::Message(m)) => {
                !matches!(m.role(), ChatRole::System | ChatRole::Developer)
            }
            _ => true,
        }
    }
}

/// One A2A task of a lane. The A2A `Task` is
/// `id = task_id`, `contextId = the thread id`, `status = (status, data.statusMessage, updated_at)`,
/// plus the fields of `data`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TaskRow {
    pub task_id: String,
    pub lane: String,
    /// Submission order; the lane runs its runnable tasks in it.
    pub seq: i64,
    /// An optional application task label.
    pub name: Option<String>,
    pub message_id: Option<String>,
    pub status: TaskState,
    /// Unix milliseconds.
    pub updated_at: i64,
    /// The A2A `Task` fields: `history`, `artifacts`, `metadata`.
    pub data: TaskData,
    /// The pending question and handler settlement used to resume this task.
    pub internal: Option<TaskInternal>,
}

impl TaskRow {
    pub fn terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

/// The `data` column: the A2A `Task` fields a row keeps, as v1 protobuf JSON; an absent field is
/// left out.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TaskData {
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub history: Vec<a2a::Message>,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub artifacts: Option<Vec<a2a::Artifact>>,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Map::is_empty"))]
    pub metadata: Map<String, Value>,
    /// `Task.status.message`.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub status_message: Option<a2a::Message>,
}

/// One message received while the lane was running, under the `inbox` policy.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InboxRow {
    pub id: i64,
    pub lane: String,
    /// The task created for the message; also its answer stream.
    pub task_id: String,
    pub message_id: String,
    pub message: a2a::Message,
    pub received_at: i64,
}

/// One task row a write upserts: every column but `lane`, `seq` and `updated_at`, which the
/// statement fills.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TaskWrite {
    pub task_id: String,
    pub status: TaskState,
    pub message_id: Option<String>,
    /// An optional application task label.
    pub name: Option<String>,
    pub data: TaskData,
    pub internal: Option<TaskInternal>,
}

impl TaskWrite {
    /// The row as stored, `seq` and `updated_at` unknown until written.
    pub fn row(&self, lane: &str) -> TaskRow {
        TaskRow {
            task_id: self.task_id.clone(),
            lane: lane.to_owned(),
            seq: 0,
            name: self.name.clone(),
            message_id: self.message_id.clone(),
            status: self.status,
            updated_at: 0,
            data: self.data.clone(),
            internal: self.internal.clone(),
        }
    }
}

impl From<TaskRow> for TaskWrite {
    fn from(r: TaskRow) -> Self {
        Self {
            task_id: r.task_id,
            status: r.status,
            message_id: r.message_id,
            name: r.name,
            data: r.data,
            internal: r.internal,
        }
    }
}

/// The `livekit/reason` codes of a failed task (server.md "Tasks on the wire").
pub const REASON_WORKER_LOST: &str = "worker-lost";
pub const REASON_LEASE_TIMEOUT: &str = "lease-timeout";
pub const REASON_THREAD_NOT_FOUND: &str = "thread-not-found";
pub const REASON_STORAGE_UNAVAILABLE: &str = "storage-unavailable";

/// The channel of the append-only run that writes a task's `submitted` row before the lease.
pub const SUBMIT_CHANNEL: &str = "submit";

/// What `run_ended` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum RunOutcome {
    Completed,
    InputRequired,
    Failed,
    Canceled,
    Rejected,
}

impl RunOutcome {
    /// The outcome a host names, as the journal spells it.
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "completed" => Self::Completed,
            "input-required" => Self::InputRequired,
            "failed" => Self::Failed,
            "canceled" => Self::Canceled,
            "rejected" => Self::Rejected,
            _ => return None,
        })
    }
}

/// What a write changes; everything left `None` keeps the existing row's value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskPatch {
    pub message_id: Option<String>,
    /// An optional application task label, the `name` column.
    pub name: Option<String>,
    /// Merged over the row's `data.metadata`; an empty map changes nothing.
    pub metadata: Option<Map<String, Value>>,
    pub ask: Option<Ask>,
    pub status_message: Option<a2a::Message>,
    pub history: Option<Vec<a2a::Message>>,
    pub artifacts: Option<Vec<a2a::Artifact>>,
    /// The turn answers the row `working`: no run holds it once committed.
    pub answered: bool,
}

/// The pending transition a task waits on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaitOptions {
    /// For a conversation tool's `InputRequired`, the pending call's tool.
    pub tool: Option<String>,
    /// For a conversation tool's `InputRequired`, the pending call.
    pub call_id: Option<String>,
    /// The tool the answer is handed to, when the `InputRequired` named one.
    pub answer_tool: Option<String>,
    /// The tool's parameter schema, added to the status message as a data part.
    pub schema: Option<Value>,
}

/// `input-required`'s patch: the question is the status message (`message_id`), `ask` the
/// pending transition, the tool's schema a data part when given.
pub fn wait_patch(
    task_id: &str,
    question: &str,
    opts: WaitOptions,
    message_id: String,
) -> TaskPatch {
    let ask = Ask {
        question: question.to_owned(),
        tool: opts.tool,
        call_id: opts.call_id,
        answer_tool: opts.answer_tool,
    };
    let mut parts = vec![a2a::Part::text(question)];
    if let Some(schema) = opts.schema {
        parts.push(a2a::Part::data(schema));
    }
    TaskPatch {
        ask: Some(ask),
        status_message: Some(agent_message(parts, task_id, message_id)),
        ..TaskPatch::default()
    }
}

/// A task write: `patch` over `base`, `data` and `internal` rebuilt from both. A terminal base is
/// immutable.
pub fn merge_write(
    task_id: &str,
    base: Option<&TaskWrite>,
    status: TaskState,
    patch: TaskPatch,
) -> Result<TaskWrite, StorageError> {
    if let Some(b) = base
        && b.status.is_terminal()
    {
        return Err(StorageError::Invalid(format!(
            "task {task_id} is {}: terminal rows are immutable",
            b.status.as_str()
        )));
    }
    let mut data = base.map(|b| b.data.clone()).unwrap_or_default();
    if let Some(metadata) = patch.metadata {
        data.metadata.extend(metadata);
    }
    if let Some(history) = patch.history {
        data.history = history;
    }
    if let Some(artifacts) = patch.artifacts {
        data.artifacts = Some(artifacts);
    }
    if let Some(message) = patch.status_message {
        data.status_message = Some(message);
    }

    let ask = if patch.ask.is_none() && (status.is_terminal() || status == TaskState::Working) {
        // a terminal or running row holds no pending question
        None
    } else {
        patch.ask.or_else(|| {
            base.and_then(|b| b.internal.as_ref().and_then(|internal| internal.ask.clone()))
        })
    };

    Ok(TaskWrite {
        task_id: task_id.to_owned(),
        status,
        message_id: patch.message_id.or_else(|| base.and_then(|b| b.message_id.clone())),
        name: patch.name.or_else(|| base.and_then(|b| b.name.clone())),
        data,
        internal: {
            let answered = patch.answered;
            (ask.is_some() || answered).then_some(TaskInternal { ask, answered })
        },
    })
}

/// An agent message `message_id` of `task_id`.
pub fn agent_message(parts: Vec<a2a::Part>, task_id: &str, message_id: String) -> a2a::Message {
    a2a::Message {
        message_id,
        task_id: Some(task_id.to_owned()),
        ..a2a::Message::new(a2a::Role::Agent, parts)
    }
}

/// Waiting for a lane another holder has: a pause from 100 ms, doubling up to 1 s, until a
/// deadline.
#[derive(Debug, Clone)]
pub struct LanePoll {
    deadline: Duration,
    pause: Duration,
}

impl LanePoll {
    pub fn new(now: Duration, timeout: Duration) -> Self {
        Self { deadline: now + timeout, pause: Duration::from_millis(100) }
    }

    /// When to try again after a miss at `now`; `None` once the deadline passed.
    pub fn next(&mut self, now: Duration) -> Option<Duration> {
        if now >= self.deadline {
            return None;
        }
        let at = now + self.pause;
        self.pause = (self.pause * 2).min(Duration::from_secs(1));
        Some(at)
    }
}

/// Unix seconds as a `Timestamp`, to the nearest nanosecond, so the float reads back unchanged.
pub fn timestamp(t: f64) -> pbjson_types::Timestamp {
    let seconds = t.floor();
    let nanos = ((t - seconds) * 1e9).round() as i32;
    if nanos >= 1_000_000_000 {
        pbjson_types::Timestamp { seconds: seconds as i64 + 1, nanos: 0 }
    } else {
        pbjson_types::Timestamp { seconds: seconds as i64, nanos }
    }
}

/// The host's twins (docs/internals/storage.md#thread-database): chat items and A2A messages
/// become their protobuf JSON, which cannot fail for these types.
#[cfg(feature = "serde")]
mod host {
    use serde::Serialize;
    use serde_json::Value;

    use super::super as host;
    use super::{InboxRow, ItemRow, StorageError, TaskData, TaskRow};

    fn json<T: Serialize>(v: &T) -> Value {
        serde_json::to_value(v).unwrap_or_default()
    }

    impl From<ItemRow> for host::ItemRow {
        fn from(row: ItemRow) -> Self {
            host::ItemRow { payload: json(&row.payload), id: row.id, message_id: row.message_id }
        }
    }

    impl TryFrom<host::ItemRow> for ItemRow {
        type Error = StorageError;

        /// A host item whose payload is not a `ChatItem` is an invalid argument.
        fn try_from(row: host::ItemRow) -> Result<Self, StorageError> {
            let payload = serde_json::from_value(row.payload)
                .map_err(|e| StorageError::Invalid(format!("item {}: {e}", row.id)))?;
            Ok(ItemRow { id: row.id, message_id: row.message_id, payload })
        }
    }

    impl From<TaskData> for host::TaskData {
        fn from(data: TaskData) -> Self {
            host::TaskData {
                history: data.history.iter().map(json).collect(),
                artifacts: data.artifacts.map(|a| a.iter().map(json).collect()),
                metadata: data.metadata,
                status_message: data.status_message.as_ref().map(json),
            }
        }
    }

    impl From<TaskRow> for host::TaskRecord {
        fn from(row: TaskRow) -> Self {
            host::TaskRecord {
                task_id: row.task_id,
                lane: row.lane,
                seq: row.seq,
                name: row.name,
                message_id: row.message_id,
                status: row.status,
                updated_at: row.updated_at,
                data: row.data.into(),
                internal: row.internal,
            }
        }
    }

    impl From<InboxRow> for host::InboxRow {
        fn from(row: InboxRow) -> Self {
            host::InboxRow {
                message: json(&row.message),
                id: row.id,
                lane: row.lane,
                task_id: row.task_id,
                message_id: row.message_id,
                received_at: row.received_at,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(id: &str) -> a2a::Message {
        a2a::Message { message_id: id.into(), ..a2a::Message::new(a2a::Role::User, vec![]) }
    }

    #[test]
    fn a_later_write_keeps_the_row_and_clears_the_ask_when_terminal_or_working() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::InputRequired,
            TaskPatch {
                message_id: Some("m1".into()),
                ask: Some(Ask { question: "when?".into(), ..Ask::default() }),
                history: Some(vec![message("m1")]),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let ask = |w: &TaskWrite| w.internal.as_ref().and_then(|internal| internal.ask.clone());
        assert_eq!(ask(&base).map(|a| a.question), Some("when?".into()));
        let msg = merge_write(
            "AST_1",
            Some(&base),
            TaskState::InputRequired,
            TaskPatch { history: Some(vec![message("m2"), message("m3")]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&msg), ask(&base), "input-required keeps the pending transition");
        assert_eq!(msg.message_id.as_deref(), Some("m1"));
        let done = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Completed,
            TaskPatch { artifacts: Some(vec![]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&done), None);
        assert_eq!(done.data.artifacts, Some(vec![]));
        assert_eq!(done.data.history, vec![message("m1")]);
        let working =
            merge_write("AST_1", Some(&base), TaskState::Working, TaskPatch::default()).unwrap();
        assert_eq!(ask(&working), None);
        let again = merge_write("AST_1", Some(&done), TaskState::Working, TaskPatch::default());
        assert!(matches!(again, Err(StorageError::Invalid(m)) if m.contains("is completed")));
    }

    #[test]
    fn metadata_merges_over_the_stored_metadata() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::Submitted,
            TaskPatch {
                name: Some("T".into()),
                metadata: Some(json!({"a": 1, "b": 1}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let next = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Canceled,
            TaskPatch {
                metadata: Some(json!({"b": 2, "c": 3}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&next.data.metadata).unwrap(),
            json!({"a": 1, "b": 2, "c": 3})
        );
        assert_eq!(next.name.as_deref(), Some("T"));
    }

    /// The float reads back unchanged, and a rounding to a whole second carries.
    #[test]
    fn a_timestamp_keeps_the_seconds_float() {
        let t = timestamp(1791760302.4035509);
        assert_eq!(t.seconds as f64 + f64::from(t.nanos) / 1e9, 1791760302.4035509);
        assert_eq!(timestamp(0.9999999999), pbjson_types::Timestamp { seconds: 1, nanos: 0 });
    }
}
