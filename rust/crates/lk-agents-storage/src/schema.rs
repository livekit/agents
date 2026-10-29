use livekit_protocol::agent::chat_context::ChatItem;
/// The thread database's tables, as livekit-protocol defines them (version 1). Every `#[table]`
/// record below is checked against them when the crate compiles.
pub use livekit_protocol::agent::{THREAD_SCHEMA_V1 as THREAD_SCHEMA, THREAD_SCHEMA_VERSION};
use lk_agents_host_derive::Table;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::run::{RunOutcome, TaskState};
use crate::sql::SqlValue;
use crate::sql::{FromCell, json_cell, statement};

pub use lk_agents_abi::storage::thread::{InboxRow, ItemRow, TaskData, TaskRow};
pub use lk_agents_abi::storage::{Ask, EventRow, LaneRow, TaskInternal};

/// The `items` columns of [`ItemRow`].
#[derive(Table)]
#[table("items", remote = ItemRow)]
#[expect(dead_code, reason = "a column declaration")]
struct ItemColumns {
    id: String,
    message_id: Option<String>,
    #[table(json)]
    payload: ChatItem,
}

/// The `tasks` columns of [`TaskRow`].
#[derive(Table)]
#[table("tasks", remote = TaskRow)]
#[expect(dead_code, reason = "a column declaration")]
struct TaskColumns {
    task_id: String,
    lane: String,
    seq: i64,
    name: Option<String>,
    message_id: Option<String>,
    status: TaskState,
    updated_at: i64,
    #[table(json)]
    data: TaskData,
    #[table(json)]
    internal: Option<TaskInternal>,
}

/// The `inbox` columns of [`InboxRow`].
#[derive(Table)]
#[table("inbox", remote = InboxRow)]
#[expect(dead_code, reason = "a column declaration")]
struct InboxColumns {
    id: i64,
    lane: String,
    task_id: String,
    message_id: String,
    #[table(json)]
    message: a2a::Message,
    received_at: i64,
}

/// The `lanes` columns of [`LaneRow`].
#[derive(Table)]
#[table("lanes", remote = LaneRow)]
#[expect(dead_code, reason = "a column declaration")]
struct LaneColumns {
    lane: String,
    version: i64,
    state: Option<Value>,
    lease_holder: Option<String>,
    lease_until: i64,
    waiter: Option<String>,
    drained_id: i64,
}

/// The `events` columns of [`EventRow`].
#[derive(Table)]
#[table("events", remote = EventRow)]
#[expect(dead_code, reason = "a column declaration")]
struct EventColumns {
    id: i64,
    ts_ms: i64,
    lane: String,
    run_id: String,
    r#type: String,
    payload: Value,
}

pub const RUN_STARTED_EVENT: &str = "run_started";
pub const RUN_ENDED_EVENT: &str = "run_ended";

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

json_cell!(TaskData, TaskInternal, ChatItem, a2a::Message);

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
    /// The lane's fence as it stands: its version and lease token.
    SelectFence<'a> { lane: &'a str } => "SELECT version, lease_token FROM lanes WHERE lane = :lane"
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
        internal: SqlValue,
        lane: &'a str,
        task_id: &'a str,
        message_id: &'a str,
        data: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, lane, seq, name, message_id, status, data, internal, updated_at)
SELECT :task_id, :lane, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = :lane), 0) + 1,
  NULL, :message_id, 'submitted', jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
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
SET status = 'failed', data = jsonb_set(data, '$.status_message', jsonb(:status_message)),
  updated_at = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE lane = :lane AND task_id = :task_id AND status = 'working'
"#
}

statement! {
    /// `tasks/cancel`: a row not yet terminal goes `canceled`, `metadata` merged over its own,
    /// its pending question dropped. A terminal row is left as it is.
    CancelTask<'a> { lane: &'a str, task_id: &'a str, metadata: SqlValue } => r#"
UPDATE tasks
SET status = 'canceled', internal = CASE WHEN json_array_length(internal, '$.streams') > 0 THEN jsonb_remove(internal, '$.ask') END,
  data = jsonb_set(data, '$.metadata', jsonb(json_patch(COALESCE(data -> '$.metadata', '{}'), :metadata))),
  updated_at = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE lane = :lane AND task_id = :task_id AND status IN ('submitted', 'working', 'input-required')
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
    /// The catalog exists even before framework initialization.
    SelectSchemaExists {} => r#"
SELECT EXISTS (
    SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_version'
)
"#
}

statement! {
    /// The version of a schema the catalog probe found.
    SelectSchemaVersion {} => "SELECT version FROM schema_version"
}

statement! {
    /// `schema_version`'s one row, written with the DDL.
    InsertSchemaVersion { version: i64 } => "INSERT INTO schema_version (version) VALUES (:version)"
}

statement! {
    /// Inserts or replaces a history item at the end of the lane. An id another lane holds sets
    /// `lane` to NULL, which the NOT NULL column refuses: the whole commit fails, nothing moved.
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
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload,
  lane = CASE WHEN items.lane = excluded.lane THEN items.lane END
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
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload,
  lane = CASE WHEN items.lane = excluded.lane THEN items.lane END
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
    /// Inserts a task row or updates every column but its `seq`; a terminal row stays as it is,
    /// and a task id another lane holds fails the commit as [`InsertItem`] does.
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
  updated_at = excluded.updated_at,
  lane = CASE WHEN tasks.lane = excluded.lane THEN tasks.lane END
WHERE tasks.status NOT IN ('completed', 'failed', 'canceled', 'rejected')
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
  updated_at = excluded.updated_at,
  lane = CASE WHEN tasks.lane = excluded.lane THEN tasks.lane END
WHERE tasks.status NOT IN ('completed', 'failed', 'canceled', 'rejected')
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
    /// One page of the thread's items in insertion order (`pos`), after `after`, of one lane
    /// when given: each row its position and lane, then the item.
    SelectItemsAfter<'a> { lane: Option<&'a str>, after: i64, limit: i64 } => r#"
SELECT pos, lane, id, message_id, json(payload) FROM items
WHERE (:lane IS NULL OR lane = :lane) AND pos > :after
ORDER BY pos LIMIT :limit
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
    /// One page of the thread's tasks in A2A `tasks/list` order: last updated first, then by
    /// task id. Of `lane` and `status` when given, updated after `updated_after`, and past the
    /// previous page's last row, `(before_at, before_id)`.
    SelectThreadTasksPage<'a> {
        lane: Option<&'a str>,
        status: Option<&'a str>,
        updated_after: i64,
        before_at: i64,
        before_id: &'a str,
        limit: i64,
    } => r#"
SELECT task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks
WHERE (:lane IS NULL OR lane = :lane)
  AND (:status IS NULL OR status = :status)
  AND updated_at > :updated_after
  AND (updated_at < :before_at OR (updated_at = :before_at AND task_id < :before_id))
ORDER BY updated_at DESC, task_id DESC
LIMIT :limit
"#
}

statement! {
    /// One page of the thread's tasks in insertion order (`pos`), after `after`, with
    /// [`SelectThreadTasksPage`]'s filters: each row its position, then the task.
    SelectThreadTasksInserted<'a> {
        lane: Option<&'a str>,
        status: Option<&'a str>,
        updated_after: i64,
        after: i64,
        limit: i64,
    } => r#"
SELECT pos, task_id, lane, seq, name, message_id, status, updated_at,
  json(data), json(internal)
FROM tasks
WHERE (:lane IS NULL OR lane = :lane)
  AND (:status IS NULL OR status = :status)
  AND updated_at > :updated_after
  AND pos > :after
ORDER BY pos
LIMIT :limit
"#
}

statement! {
    /// How many tasks a task page's filters match, every page.
    CountThreadTasks<'a> { lane: Option<&'a str>, status: Option<&'a str>, updated_after: i64 } => r#"
SELECT COUNT(*) FROM tasks
WHERE (:lane IS NULL OR lane = :lane) AND (:status IS NULL OR status = :status)
  AND updated_at > :updated_after
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
    use crate::sql::Bind;
    use crate::sql::Row;
    use crate::sql::Table;
    use livekit_protocol::agent::ChatRole;
    use livekit_protocol::agent::chat_context::chat_item;

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
            matches!(LaneRow::from_row(&lane), Err(StorageError::Unavailable(m)) if m.starts_with("lease_until"))
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
