use livekit_protocol::agent::ChatRole;
use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
/// The thread database's tables, as livekit-protocol defines them (version 1). Every `#[table]`
/// record below is checked against them when the crate compiles.
pub use livekit_protocol::agent::{THREAD_SCHEMA_V1 as THREAD_SCHEMA, THREAD_SCHEMA_VERSION};
use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::conn::SqlValue;
use crate::run::{RunOutcome, TaskState};
use crate::sql::{FromCell, json_cell, statement};

/// One history item of a lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
#[table("items")]
pub struct ItemRow {
    pub id: String,
    /// The A2A `messageId` of the message this item came from, the idempotency key.
    #[serde(default)]
    #[python(default = "None")]
    pub message_id: Option<String>,
    /// The `ChatContext.ChatItem`, as protobuf JSON.
    #[table(json)]
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
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
    pub(crate) fn stored(&self) -> bool {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
#[table("tasks")]
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
    #[table(json)]
    pub data: TaskData,
    /// What the framework keeps to resume the task: the question it waits on the person to
    /// answer, while it does.
    #[table(json)]
    pub internal: Option<Ask>,
}

impl TaskRow {
    pub fn terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

/// The `data` column: the A2A `Task` fields a row keeps, as v1 protobuf JSON; an absent field is
/// left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct TaskData {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[python(ty = "list[dict[str, Any]]", import = "typing:Any", default_factory = "list")]
    pub history: Vec<a2a::Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(ty = "list[dict[str, Any]]", import = "typing:Any", default = "None")]
    pub artifacts: Option<Vec<a2a::Artifact>>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub metadata: Map<String, Value>,
    /// `Task.status.message`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(ty = "dict[str, Any]", import = "typing:Any", default = "None")]
    pub status_message: Option<a2a::Message>,
}

/// What a task waiting on the person holds: the question and the call that takes the answer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct Ask {
    pub question: String,
    /// The tool that asked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub tool: Option<String>,
    /// The asking call, whose output the answer becomes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub call_id: Option<String>,
    /// The tool that takes the answer before the model runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[python(default = "None")]
    pub answer_tool: Option<String>,
}

/// One lane: a registered agent's state, lease and inbox cursor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
#[table("lanes")]
pub struct LaneRow {
    pub lane: String,
    /// The fencing token: a run loaded at `v` commits only `WHERE version = v`.
    pub version: i64,
    /// The application's typed state, one snapshot per run.
    #[serde(default)]
    #[python(default = "None")]
    pub state: Option<Value>,
    pub lease_holder: Option<String>,
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
#[table("events")]
pub struct EventRow {
    pub id: i64,
    pub ts_ms: i64,
    pub lane: String,
    pub run_id: String,
    #[host(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

impl EventRow {
    /// The run bracket this event is, if it is one.
    pub fn run_event(&self) -> Option<RunEvent> {
        let payload = self.payload.clone();
        match self.r#type.as_str() {
            RUN_STARTED_EVENT => serde_json::from_value(payload).ok().map(RunEvent::Started),
            RUN_ENDED_EVENT => serde_json::from_value(payload).ok().map(RunEvent::Ended),
            _ => None,
        }
    }
}

pub const RUN_STARTED_EVENT: &str = "run_started";
pub const RUN_ENDED_EVENT: &str = "run_ended";

pub enum RunEvent {
    Started(RunStarted),
    Ended(RunEnded),
}

/// `run_started`'s payload.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunStarted {
    #[serde(default)]
    pub task_id: Option<String>,
    /// The answer stream, the run's task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    /// `false` for a `concurrent` run, which holds no lease.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusive: Option<bool>,
}

/// `run_ended`'s payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnded {
    pub outcome: RunOutcome,
    #[serde(default)]
    pub reason: Option<String>,
}

/// One message received while the lane was running, under the `inbox` policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
#[table("inbox")]
pub struct InboxRow {
    pub id: i64,
    pub lane: String,
    /// The task created for the message; also its answer stream.
    pub task_id: String,
    pub message_id: String,
    #[table(json)]
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub message: a2a::Message,
    pub received_at: i64,
}

json_cell!(TaskData, Ask, ChatItem, a2a::Message);

impl FromCell for TaskState {
    fn from_cell(cell: &SqlValue) -> Result<Self, String> {
        cell.as_str().and_then(TaskState::parse).ok_or_else(|| "a task state".into())
    }
}

statement! {
    /// Creates the lane's row on its first use.
    InsertLane<'a> { lane: &'a str } => "INSERT OR IGNORE INTO lanes (lane, version) VALUES (:lane, 0)"
}

statement! {
    /// Takes the lane when nobody holds it, its lease lapsed, or this holder already has it,
    /// unless an older message is still waiting for it: waiters go in arrival order
    /// (docs/internals/storage.md#lanes). A waiting message stops counting once its row is older than `wait_ms`
    /// (its waiter died) or once it is queued in the inbox (drained as one batch). A run with no
    /// task row (a voice turn) is not ordered. Answers the lane's version when taken.
    TakeLease<'a> {
        lane: &'a str,
        holder: &'a str,
        token: &'a str,
        lease_secs: i64,
        task_id: &'a str,
        wait_ms: i64,
    } => r#"
UPDATE lanes
SET lease_holder = :holder, lease_token = :token, lease_until = unixepoch() + :lease_secs
WHERE lane = :lane
  AND (lease_until < unixepoch() OR lease_holder = :holder)
  AND NOT EXISTS (
    SELECT 1 FROM tasks older
    WHERE older.lane = :lane
      AND older.status = 'submitted'
      AND older.message_id IS NOT NULL
      AND older.updated_at > CAST(unixepoch('subsec') * 1000 AS INTEGER) - :wait_ms
      AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = :lane AND inbox.task_id = older.task_id)
      AND older.seq < (SELECT seq FROM tasks WHERE lane = :lane AND task_id = :task_id)
  )
RETURNING version
"#
}

statement! {
    /// Takes the lane in the inbox receive, only for the request that inserted this inbox row,
    /// so a retried message takes nothing. Answers the lane's version when taken.
    TakeLeaseForMessage<'a> {
        lane: &'a str,
        holder: &'a str,
        token: &'a str,
        lease_secs: i64,
        message_id: &'a str,
        task_id: &'a str,
    } => r#"
UPDATE lanes
SET lease_holder = :holder, lease_token = :token, lease_until = unixepoch() + :lease_secs
WHERE lane = :lane
  AND (lease_until < unixepoch() OR lease_holder = :holder)
  AND EXISTS (SELECT 1 FROM inbox WHERE lane = :lane AND message_id = :message_id AND task_id = :task_id)
RETURNING version
"#
}

statement! {
    /// Pushes the lease out while the token is still this one. A lapsed lease nobody took yet
    /// extends, so a holder stalled behind a slow database keeps its lane; one another holder
    /// took (a new token) or one released (no token) does not.
    ExtendLease<'a> { lane: &'a str, holder: &'a str, token: &'a str, lease_secs: i64 } => r#"
UPDATE lanes
SET lease_until = unixepoch() + :lease_secs
WHERE lane = :lane AND lease_holder = :holder AND lease_token = :token
"#
}

statement! {
    /// Frees the lane, if the lease is still this one.
    ReleaseLease<'a> { lane: &'a str, token: &'a str } => r#"
UPDATE lanes
SET lease_until = 0, lease_token = NULL
WHERE lane = :lane AND lease_token = :token
"#
}

statement! {
    /// Frees the lane only when no inbox row is waiting past `drained_id`, in the same
    /// statement, so a message inserted meanwhile keeps the lease held and its runner loops.
    ReleaseLeaseIfDrained<'a> { lane: &'a str, token: &'a str } => r#"
UPDATE lanes
SET lease_until = 0, lease_token = NULL
WHERE lane = :lane AND lease_token = :token
  AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = :lane AND inbox.id > lanes.drained_id)
"#
}

statement! {
    /// A waiter that took the lease is no longer waiting.
    ClearWaiter<'a> { lane: &'a str, holder: &'a str, token: &'a str } => r#"
UPDATE lanes
SET waiter = NULL
WHERE lane = :lane AND lease_token = :token AND waiter = :holder
"#
}

statement! {
    /// Records a holder waiting under `interrupt`, only while someone holds the lane; the
    /// holder yields at its next checkpoint.
    SetWaiter<'a> { lane: &'a str, holder: &'a str } => r#"
UPDATE lanes
SET waiter = :holder
WHERE lane = :lane AND lease_until > unixepoch()
"#
}

statement! {
    /// The holder waiting for the lane, if any.
    SelectWaiter<'a> { lane: &'a str } => "SELECT waiter FROM lanes WHERE lane = :lane"
}

statement! {
    /// A run's first event, only under this lease and never for a run that already ended, so a
    /// retried run writes nothing.
    InsertRunStarted<'a> {
        lane: &'a str,
        token: &'a str,
        ts_ms: i64,
        run_id: &'a str,
        payload: SqlValue,
    } => r#"
INSERT INTO events (ts_ms, lane, run_id, type, payload)
SELECT :ts_ms, :lane, :run_id, 'run_started', jsonb(:payload)
FROM lanes
WHERE lane = :lane AND lease_token = :token
  AND NOT EXISTS (SELECT 1 FROM events WHERE lane = :lane AND run_id = :run_id AND type = 'run_ended')
"#
}

statement! {
    /// A run's last event when it ends without a commit (failed, or already committed), only
    /// under this lease.
    InsertRunEnded<'a> {
        lane: &'a str,
        token: &'a str,
        ts_ms: i64,
        run_id: &'a str,
        payload: SqlValue,
    } => r#"
INSERT INTO events (ts_ms, lane, run_id, type, payload)
SELECT :ts_ms, :lane, :run_id, 'run_ended', jsonb(:payload)
FROM lanes
WHERE lane = :lane AND lease_token = :token
"#
}

statement! {
    /// Whether this run already committed.
    SelectCommittedRun<'a> { lane: &'a str, run_id: &'a str } => r#"
SELECT 1 FROM events WHERE lane = :lane AND run_id = :run_id AND type = 'run_ended' LIMIT 1
"#
}

statement! {
    /// A turn's `submitted` row, written before the lease is tried so a `tasks/get` before the
    /// commit finds it. A retried task id or message id inserts nothing; the commit's upsert
    /// later moves the row to its end state and keeps its `seq`.
    InsertSubmittedTask<'a> {
        lane: &'a str,
        task_id: &'a str,
        name: Option<&'a str>,
        message_id: Option<&'a str>,
        data: SqlValue,
        internal: SqlValue,
    } => r#"
INSERT OR IGNORE INTO tasks
  (task_id, lane, seq, name, message_id, status, data, internal, updated_at)
SELECT :task_id, :lane, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = :lane), 0) + 1,
  :name, :message_id, 'submitted', jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
"#
}

statement! {
    /// A turn's `submitted` row in the inbox receive, only when this request inserted the
    /// inbox row.
    InsertSubmittedTaskForMessage<'a> {
        lane: &'a str,
        task_id: &'a str,
        message_id: &'a str,
        data: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, lane, seq, name, message_id, status, data, internal, updated_at)
SELECT :task_id, :lane, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = :lane), 0) + 1,
  NULL, :message_id, 'submitted', jsonb(:data), NULL, CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM inbox
WHERE lane = :lane AND message_id = :message_id AND task_id = :task_id
"#
}

statement! {
    /// The pickup's first half: the status of a row still runnable (`working` means a dead
    /// holder had it).
    SelectPickupStatus<'a> { lane: &'a str, task_id: &'a str } => r#"
SELECT status FROM tasks
WHERE lane = :lane AND task_id = :task_id AND status IN ('submitted', 'working')
"#
}

statement! {
    /// The pickup's second half: the row moves to `working`.
    PickupTask<'a> { lane: &'a str, task_id: &'a str } => r#"
UPDATE tasks
SET status = 'working', updated_at = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE lane = :lane AND task_id = :task_id AND status IN ('submitted', 'working')
"#
}

statement! {
    /// The pickup inside the acquire: the row moves to `working` only under this lease.
    PickupTaskHeld<'a> { lane: &'a str, token: &'a str, task_id: &'a str } => r#"
UPDATE tasks
SET status = 'working', updated_at = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE lane = :lane AND task_id = :task_id AND status IN ('submitted', 'working')
  AND EXISTS (SELECT 1 FROM lanes WHERE lane = :lane AND lease_token = :token)
"#
}

statement! {
    /// A task's status.
    SelectTaskStatus<'a> { lane: &'a str, task_id: &'a str } => r#"
SELECT status FROM tasks WHERE lane = :lane AND task_id = :task_id
"#
}

statement! {
    /// Fails a `working` row whose holder died.
    FailOrphan<'a> { lane: &'a str, task_id: &'a str, status_message: SqlValue } => r#"
UPDATE tasks
SET status = 'failed', data = jsonb_set(data, '$.statusMessage', jsonb(:status_message)),
  updated_at = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE lane = :lane AND task_id = :task_id AND status = 'working'
"#
}

statement! {
    /// A message received while the lane runs; a retried message id inserts nothing.
    InsertInbox<'a> { lane: &'a str, task_id: &'a str, message_id: &'a str, message: SqlValue } => r#"
INSERT OR IGNORE INTO inbox (lane, task_id, message_id, message, received_at)
VALUES (:lane, :task_id, :message_id, jsonb(:message), unixepoch())
"#
}

statement! {
    /// The task a message id was received under.
    SelectInboxTask<'a> { lane: &'a str, message_id: &'a str } => r#"
SELECT task_id FROM inbox WHERE lane = :lane AND message_id = :message_id
"#
}

statement! {
    /// The schema probe: fails `no such table` on a database no SDK wrote yet.
    SelectSchemaVersion {} => "SELECT version FROM schema_version"
}

/// The error [`SelectSchemaVersion`] answers on a database no SDK wrote yet.
pub(crate) const NO_SCHEMA: &str = "no such table: schema_version";

statement! {
    /// `schema_version`'s one row, written with the DDL.
    InsertSchemaVersion { version: i64 } => "INSERT INTO schema_version (version) VALUES (:version)"
}

statement! {
    /// Inserts or replaces a history item at the end of the lane.
    InsertItem<'a> {
        lane: &'a str,
        version: i64,
        token: &'a str,
        id: &'a str,
        message_id: Option<&'a str>,
        payload: SqlValue,
    } => r#"
INSERT INTO items (id, lane, seq, message_id, payload)
SELECT :id, :lane, COALESCE((SELECT MAX(seq) FROM items WHERE lane = :lane), 0) + 1, :message_id, jsonb(:payload)
FROM lanes
WHERE lane = :lane AND version = :version AND lease_token = :token
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload
"#
}

statement! {
    /// [`InsertItem`] with no lease.
    InsertItemUnfenced<'a> {
        lane: &'a str,
        id: &'a str,
        message_id: Option<&'a str>,
        payload: SqlValue,
    } => r#"
INSERT INTO items (id, lane, seq, message_id, payload)
SELECT :id, :lane, COALESCE((SELECT MAX(seq) FROM items WHERE lane = :lane), 0) + 1, :message_id, jsonb(:payload)
FROM lanes
WHERE lane = :lane
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload
"#
}

statement! {
    /// Removes a history item.
    DeleteItem<'a> { lane: &'a str, version: i64, token: &'a str, id: &'a str } => r#"
DELETE FROM items
WHERE lane = :lane AND id = :id
  AND EXISTS (SELECT 1 FROM lanes WHERE lane = :lane AND version = :version AND lease_token = :token)
"#
}

statement! {
    /// [`DeleteItem`] with no lease.
    DeleteItemUnfenced<'a> { lane: &'a str, id: &'a str } => r#"
DELETE FROM items
WHERE lane = :lane AND id = :id
  AND EXISTS (SELECT 1 FROM lanes WHERE lane = :lane)
"#
}

statement! {
    /// Appends a journal event.
    InsertEvent<'a> {
        lane: &'a str,
        version: i64,
        token: &'a str,
        ts_ms: i64,
        run_id: &'a str,
        event_type: &'a str,
        payload: SqlValue,
    } => r#"
INSERT INTO events (ts_ms, lane, run_id, type, payload)
SELECT :ts_ms, :lane, :run_id, :event_type, jsonb(:payload)
FROM lanes
WHERE lane = :lane AND version = :version AND lease_token = :token
"#
}

statement! {
    /// [`InsertEvent`] with no lease.
    InsertEventUnfenced<'a> {
        lane: &'a str,
        ts_ms: i64,
        run_id: &'a str,
        event_type: &'a str,
        payload: SqlValue,
    } => r#"
INSERT INTO events (ts_ms, lane, run_id, type, payload)
SELECT :ts_ms, :lane, :run_id, :event_type, jsonb(:payload)
FROM lanes
WHERE lane = :lane
"#
}

statement! {
    /// Inserts a task row or updates every column but its `seq`.
    UpsertTask<'a> {
        lane: &'a str,
        version: i64,
        token: &'a str,
        task_id: &'a str,
        name: Option<&'a str>,
        message_id: Option<&'a str>,
        status: &'a str,
        data: SqlValue,
        internal: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, lane, seq, name, message_id, status, data, internal, updated_at)
SELECT :task_id, :lane, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = :lane), 0) + 1,
  :name, :message_id, :status, jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM lanes
WHERE lane = :lane AND version = :version AND lease_token = :token
ON CONFLICT(task_id) DO UPDATE SET
  status = excluded.status, name = excluded.name, message_id = excluded.message_id,
  data = excluded.data, internal = excluded.internal,
  updated_at = excluded.updated_at
"#
}

statement! {
    /// [`UpsertTask`] with no lease.
    UpsertTaskUnfenced<'a> {
        lane: &'a str,
        task_id: &'a str,
        name: Option<&'a str>,
        message_id: Option<&'a str>,
        status: &'a str,
        data: SqlValue,
        internal: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, lane, seq, name, message_id, status, data, internal, updated_at)
SELECT :task_id, :lane, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = :lane), 0) + 1,
  :name, :message_id, :status, jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM lanes
WHERE lane = :lane
ON CONFLICT(task_id) DO UPDATE SET
  status = excluded.status, name = excluded.name, message_id = excluded.message_id,
  data = excluded.data, internal = excluded.internal,
  updated_at = excluded.updated_at
"#
}

statement! {
    /// Ends a commit and releases the lease: bumps `version` and writes the snapshots that
    /// changed (NULL keeps the stored one). No row answered means the run outlived its lease.
    CommitAndRelease<'a> {
        lane: &'a str,
        version: i64,
        token: &'a str,
        state: SqlValue,
        drained_id: Option<i64>,
    } => r#"
UPDATE lanes
SET version = version + 1,
  state = COALESCE(jsonb(:state), state),
  drained_id = COALESCE(:drained_id, drained_id),
  lease_until = 0, lease_token = NULL
WHERE lane = :lane AND version = :version AND lease_token = :token
RETURNING version
"#
}

statement! {
    /// [`CommitAndRelease`] that keeps the lease, extended, for a runner that continues.
    CommitAndExtend<'a> {
        lane: &'a str,
        version: i64,
        token: &'a str,
        state: SqlValue,
        drained_id: Option<i64>,
        lease_secs: i64,
    } => r#"
UPDATE lanes
SET version = version + 1,
  state = COALESCE(jsonb(:state), state),
  drained_id = COALESCE(:drained_id, drained_id),
  lease_until = unixepoch() + :lease_secs
WHERE lane = :lane AND version = :version AND lease_token = :token
RETURNING version
"#
}

statement! {
    /// The lane's history in order.
    SelectItems<'a> { lane: &'a str } => r#"
SELECT id, message_id, json(payload) FROM items WHERE lane = :lane ORDER BY seq
"#
}

statement! {
    /// One page of the history: at most `limit` items after skipping `offset`.
    SelectItemsPage<'a> { lane: &'a str, limit: i64, offset: i64 } => r#"
SELECT id, message_id, json(payload) FROM items WHERE lane = :lane ORDER BY seq LIMIT :limit OFFSET :offset
"#
}

statement! {
    /// The lane's journal after the event `after_id`.
    SelectLaneEvents<'a> { lane: &'a str, after_id: i64 } => r#"
SELECT id, ts_ms, lane, run_id, type, json(payload) FROM events
WHERE lane = :lane AND id > :after_id ORDER BY id
"#
}

statement! {
    /// Every lane's journal after the event `after_id`, at most `limit` events.
    SelectEvents { after_id: i64, limit: i64 } => r#"
SELECT id, ts_ms, lane, run_id, type, json(payload) FROM events
WHERE id > :after_id ORDER BY id LIMIT :limit
"#
}

statement! {
    /// One lane's row.
    SelectLane<'a> { lane: &'a str } => r#"
SELECT lane, version, json(state), lease_holder, lease_until, waiter, drained_id
FROM lanes WHERE lane = :lane
"#
}

statement! {
    /// Every lane's row.
    SelectLanes {} => r#"
SELECT lane, version, json(state), lease_holder, lease_until, waiter, drained_id
FROM lanes ORDER BY lane
"#
}

statement! {
    /// The newest `limit` task rows.
    SelectTasks<'a> { lane: &'a str, limit: i64 } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks WHERE lane = :lane ORDER BY seq DESC LIMIT :limit
"#
}

statement! {
    /// The newest `limit` task rows of one status.
    SelectTasksByStatus<'a> { lane: &'a str, status: &'a str, limit: i64 } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks WHERE lane = :lane AND status = :status ORDER BY seq DESC LIMIT :limit
"#
}

statement! {
    /// One task row.
    SelectTask<'a> { lane: &'a str, task_id: &'a str } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks WHERE lane = :lane AND task_id = :task_id
"#
}

statement! {
    /// One task row, whichever lane it is on.
    SelectThreadTask<'a> { task_id: &'a str } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks WHERE task_id = :task_id
"#
}

statement! {
    /// One page of the thread's tasks, every lane's, in A2A `tasks/list` order: last updated
    /// first, then by task id. Only those of `status` when given, updated after `updated_after`,
    /// and past the previous page's last row, `(before_at, before_id)`.
    SelectThreadTasksPage<'a> {
        status: Option<&'a str>,
        updated_after: i64,
        before_at: i64,
        before_id: &'a str,
        limit: i64,
    } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks
WHERE (:status IS NULL OR status = :status)
  AND updated_at > :updated_after
  AND (updated_at < :before_at OR (updated_at = :before_at AND task_id < :before_id))
ORDER BY updated_at DESC, task_id DESC
LIMIT :limit
"#
}

statement! {
    /// How many tasks [`SelectThreadTasksPage`]'s filters match, every page.
    CountThreadTasks<'a> { status: Option<&'a str>, updated_after: i64 } => r#"
SELECT COUNT(*) FROM tasks WHERE (:status IS NULL OR status = :status) AND updated_at > :updated_after
"#
}

statement! {
    /// The task row an A2A message id created.
    SelectTaskForMessage<'a> { lane: &'a str, message_id: &'a str } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks WHERE lane = :lane AND message_id = :message_id
"#
}

statement! {
    /// Every inbox row in arrival order.
    SelectInbox<'a> { lane: &'a str } => r#"
SELECT id, lane, task_id, message_id, json(message), received_at FROM inbox
WHERE lane = :lane ORDER BY id
"#
}

statement! {
    /// The inbox rows not yet drained, past the lane's `drained_id`.
    SelectInboxPending<'a> { lane: &'a str } => r#"
SELECT id, lane, task_id, message_id, json(message), received_at FROM inbox
WHERE lane = :lane AND id > (SELECT drained_id FROM lanes WHERE lane = :lane) ORDER BY id
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageError;
    use crate::conn::Row;
    use crate::sql::Statement;

    #[test]
    fn a_row_decodes_and_a_bad_cell_names_its_column() {
        let mut lane: Row = vec![
            "front-desk".into(),
            SqlValue::Int(2),
            SqlValue::Null,
            SqlValue::Null,
            "not an int".into(),
            SqlValue::Null,
            SqlValue::Int(0),
        ];
        assert!(
            matches!(LaneRow::from_row(&lane), Err(StorageError::Decode(m)) if m.starts_with("lease_until"))
        );
        lane[4] = SqlValue::Int(0);
        assert_eq!(LaneRow::from_row(&lane).unwrap().lease_until, 0);
    }

    /// Instructions and config updates are rendered per run, never stored.
    #[test]
    fn only_conversation_items_are_stored() {
        use livekit_protocol::agent::{AgentConfigUpdate, ChatMessage};
        let message = |role: ChatRole| {
            ItemRow::new(chat_item::Item::Message(ChatMessage {
                role: role as i32,
                ..Default::default()
            }))
        };
        assert!(message(ChatRole::User).stored() && message(ChatRole::Assistant).stored());
        assert!(!message(ChatRole::System).stored() && !message(ChatRole::Developer).stored());
        assert!(
            !ItemRow::new(chat_item::Item::AgentConfigUpdate(AgentConfigUpdate::default()))
                .stored()
        );
    }

    /// An unfenced write is its fenced twin less the fence, and a commit that keeps the lease is
    /// the one that releases it but for the lease.
    #[test]
    fn twin_statements_differ_only_where_they_should() {
        let fence = " AND version = :version AND lease_token = :token";
        for (fenced, unfenced) in [
            (InsertItem::SQL, InsertItemUnfenced::SQL),
            (DeleteItem::SQL, DeleteItemUnfenced::SQL),
            (InsertEvent::SQL, InsertEventUnfenced::SQL),
            (UpsertTask::SQL, UpsertTaskUnfenced::SQL),
        ] {
            assert!(fenced.contains(fence), "{fenced}");
            assert_eq!(fenced.replace(fence, ""), unfenced);
        }
        assert_eq!(
            CommitAndRelease::SQL.replace(
                "lease_until = 0, lease_token = NULL",
                "lease_until = unixepoch() + :lease_secs"
            ),
            CommitAndExtend::SQL
        );
    }

    /// Each read selects its record's columns, in the record's order.
    #[test]
    fn reads_select_their_record() {
        let one_line = |sql: &str| sql.split_whitespace().collect::<Vec<_>>().join(" ");
        for (sql, columns) in [
            (SelectItems::SQL, ItemRow::SELECT),
            (SelectItemsPage::SQL, ItemRow::SELECT),
            (SelectLaneEvents::SQL, EventRow::SELECT),
            (SelectEvents::SQL, EventRow::SELECT),
            (SelectLane::SQL, LaneRow::SELECT),
            (SelectLanes::SQL, LaneRow::SELECT),
            (SelectTasks::SQL, TaskRow::SELECT),
            (SelectTasksByStatus::SQL, TaskRow::SELECT),
            (SelectTask::SQL, TaskRow::SELECT),
            (SelectTaskForMessage::SQL, TaskRow::SELECT),
            (SelectThreadTask::SQL, TaskRow::SELECT),
            (SelectThreadTasksPage::SQL, TaskRow::SELECT),
            (SelectInbox::SQL, InboxRow::SELECT),
            (SelectInboxPending::SQL, InboxRow::SELECT),
        ] {
            assert!(one_line(sql).starts_with(&format!("SELECT {columns} FROM")), "{sql}");
        }
    }
}
