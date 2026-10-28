//! A thread database's rows as the host reads them (docs/internals/storage.md#thread-database).
//! The storage core keeps its typed `#[table]` rows and converts them into these; chat items and
//! A2A messages cross as their protobuf JSON. The turn records (`storage_turn`) still carry the
//! core's own models until they move here with the turn bridge (I-16).

use serde_json::{Map, Value};

/// A task's state: the storage core's `TaskState` as the host reads it.
#[crate::host(record(module = "storage_class", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskState {
    #[host(rename = "submitted")]
    Submitted,
    #[host(rename = "working")]
    Working,
    #[host(rename = "input-required")]
    InputRequired,
    #[host(rename = "completed")]
    Completed,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "canceled")]
    Canceled,
    #[host(rename = "rejected")]
    Rejected,
}

impl TaskState {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::Submitted => "submitted",
            TaskState::Working => "working",
            TaskState::InputRequired => "input-required",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Canceled => "canceled",
            TaskState::Rejected => "rejected",
        }
    }
}

/// What a task waiting on the person holds: the question and the call that takes the answer.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ask {
    pub question: String,
    /// The tool that asked.
    pub tool: Option<String>,
    /// The asking call, whose output the answer becomes.
    pub call_id: Option<String>,
    /// The tool that takes the answer before the model runs.
    pub answer_tool: Option<String>,
}

/// A created response stream: its id and when it expires (unix seconds).
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    pub stream_id: String,
    pub expires_at: i64,
}

/// One history item of a lane; `payload` is the `ChatContext.ChatItem`.
#[crate::host(record(module = "storage_class", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ItemRow {
    pub id: String,
    /// The A2A `messageId` of the message this item came from, the idempotency key.
    pub message_id: Option<String>,
    pub payload: Value,
}

/// The A2A `Task` fields a task keeps, as v1 protobuf JSON.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskData {
    pub history: Vec<Value>,
    pub artifacts: Option<Vec<Value>>,
    pub metadata: Map<String, Value>,
    /// `Task.status.message`.
    pub status_message: Option<Value>,
}

/// One A2A task of a lane: `id = task_id`, `contextId` the thread id, `status = (status,
/// data.status_message, updated_at)`, plus the fields of `data`.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRecord {
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
    pub data: TaskData,
    /// The question the task waits on the person to answer, while it does.
    pub internal: Option<Ask>,
}

/// One page of A2A `tasks/list`.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskPage {
    pub tasks: Vec<TaskRecord>,
    /// Empty on the last page.
    pub next_page_token: String,
    pub page_size: i64,
    /// Every page's tasks, in all.
    pub total_size: i64,
}

/// One lane: a registered agent's state, lease and inbox cursor.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct LaneRow {
    pub lane: String,
    /// The fencing token: a run loaded at `v` commits only `WHERE version = v`.
    pub version: i64,
    /// The application's typed state, one snapshot per run.
    pub state: Option<Value>,
    pub lease_holder: Option<String>,
    /// Unix seconds on agent-db's clock; `0` is released.
    pub lease_until: i64,
    /// A worker that failed the lease under `interrupt` and is polling for it.
    pub waiter: Option<String>,
    /// The id of the last inbox row a drain consumed.
    pub drained_id: i64,
}

/// One journal entry: every `AgentSessionEvent`, plus `run_started` and `run_ended`.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct EventRow {
    pub id: i64,
    pub ts_ms: i64,
    pub lane: String,
    pub run_id: String,
    pub r#type: String,
    pub payload: Value,
}

/// One message received while the lane was running, under the `inbox` policy.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct InboxRow {
    pub id: i64,
    pub lane: String,
    /// The task created for the message; also its answer stream.
    pub task_id: String,
    pub message_id: String,
    /// The A2A `Message`.
    pub message: Value,
    pub received_at: i64,
}
