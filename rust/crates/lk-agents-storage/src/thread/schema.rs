use livekit_protocol::agent::chat_context::ChatItem;
/// The thread database's tables, as livekit-protocol defines them (version 1). Every `#[table]`
/// record below is checked against them when the crate compiles.
pub use livekit_protocol::agent::{THREAD_SCHEMA_V1 as THREAD_SCHEMA, THREAD_SCHEMA_VERSION};
use lk_agents_host_derive::Table;
use serde_json::Value;

use crate::sql::SqlValue;
use crate::sql::{FromCell, json_cell, statement};
use crate::thread::run::TaskState;

pub use lk_agents_abi::storage::thread::{InboxRow, ItemRow, TaskData, TaskRow};
pub use lk_agents_abi::storage::{Ask, EventRow, PartitionRow, TaskInternal};

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
    partition_key: String,
    pos: i64,
    name: Option<String>,
    message_id: Option<String>,
    status: TaskState,
    updated_at_ms: i64,
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
    partition_key: String,
    task_id: String,
    message_id: String,
    #[table(json)]
    message: a2a::Message,
    received_at_ms: i64,
}

/// The `partitions` columns of [`PartitionRow`].
#[derive(Table)]
#[table("partitions", remote = PartitionRow)]
#[expect(dead_code, reason = "a column declaration")]
struct PartitionColumns {
    partition_key: String,
    version: i64,
    state: Option<Value>,
    lease_holder: Option<String>,
    lease_expires_at_ms: i64,
    drained_id: i64,
}

/// The `events` columns of [`EventRow`].
#[derive(Table)]
#[table("events", remote = EventRow)]
#[expect(dead_code, reason = "a column declaration")]
struct EventColumns {
    id: i64,
    occurred_at_ms: i64,
    partition_key: String,
    run_id: String,
    r#type: String,
    payload: Value,
}

json_cell!(TaskData, TaskInternal, ChatItem, a2a::Message);

/// The framework statements a connection (a fault-injecting one, a simulated database) tells
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    /// Takes the partition's lease: an acquire or an inbox receive.
    LeaseTake,
    /// Commits a run: bumps the partition's version.
    Commit,
    /// Reads the partition's version and lease.
    FenceRead,
    /// Writes the partition row or the journal, nothing a run staged.
    Bookkeeping,
}

impl FromCell for TaskState {
    fn from_cell(cell: &SqlValue) -> Result<Self, crate::StorageError> {
        cell.as_str()
            .and_then(TaskState::from_wire)
            .ok_or_else(|| crate::StorageError::Invalid("a task state".into()))
    }
}

statement! {
    /// Creates the partition's row on its first use.
    InsertPartition<'a> { partition_key: &'a str } as Bookkeeping => "INSERT OR IGNORE INTO partitions (partition_key, version) VALUES (:partition_key, 0)"
}

statement! {
    /// Takes the partition when nobody holds it, its lease lapsed, or this holder already has it,
    /// unless an older message is still waiting for it: waiters go in arrival order. A waiting
    /// message stops counting once its row is older than `wait_ms`
    /// (its waiter died) or once it is queued in the inbox (drained as one batch). A run with no
    /// task row (a voice turn) is not ordered. Answers the partition's version when taken.
    TakeLease<'a> {
        partition_key: &'a str,
        holder: &'a str,
        token: &'a str,
        lease_ms: i64,
        task_id: &'a str,
        wait_ms: i64,
    } as LeaseTake => r#"
UPDATE partitions
SET lease_holder = :holder, lease_token = :token, lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :lease_ms
WHERE partition_key = :partition_key
  AND (lease_expires_at_ms < CAST(unixepoch('subsec') * 1000 AS INTEGER) OR lease_holder = :holder)
  AND NOT EXISTS (
    SELECT 1 FROM tasks older
    WHERE older.partition_key = :partition_key
      AND older.status = 'submitted'
      AND older.message_id IS NOT NULL
      AND older.updated_at_ms > CAST(unixepoch('subsec') * 1000 AS INTEGER) - :wait_ms
      AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.partition_key = :partition_key AND inbox.task_id = older.task_id)
      AND older.pos < (SELECT pos FROM tasks WHERE partition_key = :partition_key AND task_id = :task_id)
  )
RETURNING version
"#
}

statement! {
    /// Takes the partition in the inbox receive, only for the request that inserted this inbox row,
    /// so a retried message takes nothing. Answers the partition's version when taken.
    TakeLeaseForMessage<'a> {
        partition_key: &'a str,
        holder: &'a str,
        token: &'a str,
        lease_ms: i64,
        message_id: &'a str,
        task_id: &'a str,
    } as LeaseTake => r#"
UPDATE partitions
SET lease_holder = :holder, lease_token = :token, lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :lease_ms
WHERE partition_key = :partition_key
  AND (lease_expires_at_ms < CAST(unixepoch('subsec') * 1000 AS INTEGER) OR lease_holder = :holder)
  AND EXISTS (SELECT 1 FROM inbox WHERE partition_key = :partition_key AND message_id = :message_id AND task_id = :task_id)
RETURNING version
"#
}

statement! {
    /// Pushes the lease out while the token is still this one. A lapsed lease nobody took yet
    /// extends, so a holder stalled behind a slow database keeps its partition; one another holder
    /// took (a new token) or one released (no token) does not.
    ExtendLease<'a> { partition_key: &'a str, holder: &'a str, token: &'a str, lease_ms: i64 } => r#"
UPDATE partitions
SET lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :lease_ms
WHERE partition_key = :partition_key AND lease_holder = :holder AND lease_token = :token
"#
}

statement! {
    /// Frees the partition, if the lease is still this one.
    ReleaseLease<'a> { partition_key: &'a str, token: &'a str } => r#"
UPDATE partitions
SET lease_expires_at_ms = 0, lease_token = NULL
WHERE partition_key = :partition_key AND lease_token = :token
"#
}

statement! {
    /// Frees the partition only when no inbox row is waiting past `drained_id`, in the same
    /// statement, so a message inserted meanwhile keeps the lease held and its runner loops.
    ReleaseLeaseIfDrained<'a> { partition_key: &'a str, token: &'a str } => r#"
UPDATE partitions
SET lease_expires_at_ms = 0, lease_token = NULL
WHERE partition_key = :partition_key AND lease_token = :token
  AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.partition_key = :partition_key AND inbox.id > partitions.drained_id)
"#
}

statement! {
    /// A run's first event, only under this lease and never for a run that already ended, so a
    /// retried run writes nothing.
    InsertRunStarted<'a> {
        partition_key: &'a str,
        token: &'a str,
        occurred_at_ms: i64,
        run_id: &'a str,
        payload: SqlValue,
    } as Bookkeeping => r#"
INSERT INTO events (occurred_at_ms, partition_key, run_id, type, payload)
SELECT :occurred_at_ms, :partition_key, :run_id, 'run_started', jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND lease_token = :token
  AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')
"#
}

statement! {
    /// A run's last event when it ends without a commit (failed, canceled, or already
    /// committed), once: under this lease, or while its holder still has the partition under another
    /// token (its next turn re-entered before this cleanup applied). Another holder's partition is
    /// left alone: the run is then found interrupted.
    InsertRunEnded<'a> {
        partition_key: &'a str,
        token: &'a str,
        holder: &'a str,
        occurred_at_ms: i64,
        run_id: &'a str,
        payload: SqlValue,
    } as Bookkeeping => r#"
INSERT INTO events (occurred_at_ms, partition_key, run_id, type, payload)
SELECT :occurred_at_ms, :partition_key, :run_id, 'run_ended', jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND (lease_token = :token OR lease_holder = :holder)
  AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')
"#
}

statement! {
    /// The partition's lease as an acquire found it, and the database's clock.
    SelectLeaseBefore<'a> { partition_key: &'a str } => "SELECT lease_holder, lease_token, lease_expires_at_ms, CAST(unixepoch('subsec') * 1000 AS INTEGER) FROM partitions WHERE partition_key = :partition_key"
}

statement! {
    /// The partition's fence as it stands: its version, lease token and lease expiry.
    SelectFence<'a> { partition_key: &'a str } as FenceRead => "SELECT version, lease_token, lease_expires_at_ms FROM partitions WHERE partition_key = :partition_key"
}

statement! {
    /// Whether this run already committed.
    SelectCommittedRun<'a> { partition_key: &'a str, run_id: &'a str } => r#"
SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended' LIMIT 1
"#
}

statement! {
    /// A turn's `submitted` row, written before the lease is tried so a `tasks/get` before the
    /// commit finds it. A retried task id or message id inserts nothing; the commit's upsert
    /// later moves the row to its end state and keeps its `pos`.
    InsertSubmittedTask<'a> {
        endpoint: Option<&'a str>,
        partition_key: &'a str,
        task_id: &'a str,
        name: Option<&'a str>,
        message_id: Option<&'a str>,
        data: SqlValue,
        internal: SqlValue,
    } => r#"
INSERT OR IGNORE INTO tasks
  (task_id, partition_key, endpoint, name, message_id, status, data, internal, updated_at_ms)
SELECT :task_id, :partition_key, :endpoint,
  :name, :message_id, 'submitted', jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
"#
}

statement! {
    /// A turn's `submitted` row in the inbox receive, only when this request inserted the
    /// inbox row; a retry after a lost acknowledgment inserts nothing.
    InsertSubmittedTaskForMessage<'a> {
        endpoint: Option<&'a str>,
        internal: SqlValue,
        partition_key: &'a str,
        task_id: &'a str,
        message_id: &'a str,
        data: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, partition_key, endpoint, name, message_id, status, data, internal, updated_at_ms)
SELECT :task_id, :partition_key, :endpoint,
  NULL, :message_id, 'submitted', jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM inbox
WHERE partition_key = :partition_key AND message_id = :message_id AND task_id = :task_id
  AND NOT EXISTS (SELECT 1 FROM tasks WHERE partition_key = :partition_key AND message_id = :message_id)
"#
}

statement! {
    /// The pickup's first half: the status of a row still runnable (`working` means a dead
    /// holder had it; a row a turn answered `working` is no run's).
    SelectPickupStatus<'a> { partition_key: &'a str, task_id: &'a str } => r#"
SELECT status FROM tasks
WHERE partition_key = :partition_key AND task_id = :task_id AND status IN ('submitted', 'working')
  AND NOT COALESCE(json_extract(internal, '$.answered'), 0)
"#
}

statement! {
    /// The pickup's second half: the row moves to `working`.
    PickupTask<'a> { partition_key: &'a str, task_id: &'a str } => r#"
UPDATE tasks
SET status = 'working', updated_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE partition_key = :partition_key AND task_id = :task_id AND status IN ('submitted', 'working')
  AND NOT COALESCE(json_extract(internal, '$.answered'), 0)
"#
}

statement! {
    /// The pickup inside the acquire: the row moves to `working` only under this lease.
    PickupTaskHeld<'a> { partition_key: &'a str, token: &'a str, task_id: &'a str } => r#"
UPDATE tasks
SET status = 'working', updated_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE partition_key = :partition_key AND task_id = :task_id AND status IN ('submitted', 'working')
  AND NOT COALESCE(json_extract(internal, '$.answered'), 0)
  AND EXISTS (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND lease_token = :token)
"#
}

statement! {
    /// A task's status.
    SelectTaskStatus<'a> { partition_key: &'a str, task_id: &'a str } => r#"
SELECT status FROM tasks WHERE partition_key = :partition_key AND task_id = :task_id
"#
}

statement! {
    /// Fails a `working` row whose holder died.
    FailOrphan<'a> { partition_key: &'a str, task_id: &'a str, status_message: SqlValue } => r#"
UPDATE tasks
SET status = 'failed', data = jsonb_set(data, '$.status_message', jsonb(:status_message)),
  updated_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE partition_key = :partition_key AND task_id = :task_id AND status = 'working'
"#
}

statement! {
    /// `tasks/cancel`: a row not yet terminal goes `canceled`, `metadata` merged over its own,
    /// its pending question dropped. A terminal row is left as it is.
    CancelTask<'a> { partition_key: &'a str, task_id: &'a str, metadata: SqlValue } => r#"
UPDATE tasks
SET status = 'canceled', internal = NULL,
  data = jsonb_set(data, '$.metadata', jsonb(json_patch(COALESCE(data -> '$.metadata', '{}'), :metadata))),
  updated_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE partition_key = :partition_key AND task_id = :task_id AND status IN ('submitted', 'working', 'input-required')
"#
}

statement! {
    /// A message received while the partition runs; a retried message id inserts nothing.
    InsertInbox<'a> { partition_key: &'a str, task_id: &'a str, message_id: &'a str, message: SqlValue } => r#"
INSERT OR IGNORE INTO inbox (partition_key, task_id, message_id, message, received_at_ms)
SELECT :partition_key, :task_id, :message_id, jsonb(:message), CAST(unixepoch('subsec') * 1000 AS INTEGER)
WHERE NOT EXISTS (SELECT 1 FROM tasks WHERE partition_key = :partition_key AND message_id = :message_id)
"#
}

statement! {
    /// The task a message id was received under.
    SelectInboxTask<'a> { partition_key: &'a str, message_id: &'a str } => r#"
SELECT task_id FROM tasks WHERE partition_key = :partition_key AND message_id = :message_id
UNION ALL
SELECT task_id FROM inbox WHERE partition_key = :partition_key AND message_id = :message_id
LIMIT 1
"#
}

statement! {
    /// Consumed terminal requests are temporary; their task receipts remain for deduplication.
    PruneInbox<'a> { partition_key: &'a str, version: i64, token: &'a str, drained_id: Option<i64> } => r#"
DELETE FROM inbox
WHERE partition_key = :partition_key
  AND id <= COALESCE(:drained_id, (SELECT drained_id FROM partitions WHERE partition_key = :partition_key))
  AND EXISTS (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND version = :version AND lease_token = :token)
  AND EXISTS (SELECT 1 FROM tasks WHERE tasks.partition_key = inbox.partition_key AND tasks.task_id = inbox.task_id
    AND tasks.status IN ('completed', 'failed', 'canceled', 'rejected'))
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
    /// Inserts or replaces a history item at the end of the partition. An id another partition holds sets
    /// `partition_key` to NULL, which the NOT NULL column refuses: the whole commit fails, nothing moved.
    InsertItem<'a> {
        partition_key: &'a str,
        version: i64,
        token: &'a str,
        id: &'a str,
        message_id: Option<&'a str>,
        payload: SqlValue,
    } => r#"
INSERT INTO items (id, partition_key, seq, message_id, payload)
SELECT :id, :partition_key, COALESCE((SELECT MAX(seq) FROM items WHERE partition_key = :partition_key), 0) + 1, :message_id, jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload,
  partition_key = CASE WHEN items.partition_key = excluded.partition_key THEN items.partition_key END
"#
}

statement! {
    /// [`InsertItem`] with no lease, guarded on its run: a re-sent commit writes nothing twice.
    InsertItemUnfenced<'a> {
        partition_key: &'a str,
        run_id: &'a str,
        id: &'a str,
        message_id: Option<&'a str>,
        payload: SqlValue,
    } => r#"
INSERT INTO items (id, partition_key, seq, message_id, payload)
SELECT :id, :partition_key, COALESCE((SELECT MAX(seq) FROM items WHERE partition_key = :partition_key), 0) + 1, :message_id, jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')
ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload,
  partition_key = CASE WHEN items.partition_key = excluded.partition_key THEN items.partition_key END
"#
}

statement! {
    /// Removes a history item.
    DeleteItem<'a> { partition_key: &'a str, version: i64, token: &'a str, id: &'a str } => r#"
DELETE FROM items
WHERE partition_key = :partition_key AND id = :id
  AND EXISTS (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND version = :version AND lease_token = :token)
"#
}

statement! {
    /// [`DeleteItem`] with no lease, guarded on its run: a re-sent commit writes nothing twice.
    DeleteItemUnfenced<'a> { partition_key: &'a str, run_id: &'a str, id: &'a str } => r#"
DELETE FROM items
WHERE partition_key = :partition_key AND id = :id
  AND EXISTS (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended'))
"#
}

statement! {
    /// Appends a journal event.
    InsertEvent<'a> {
        partition_key: &'a str,
        version: i64,
        token: &'a str,
        occurred_at_ms: i64,
        run_id: &'a str,
        event_type: &'a str,
        payload: SqlValue,
    } as Bookkeeping => r#"
INSERT INTO events (occurred_at_ms, partition_key, run_id, type, payload)
SELECT :occurred_at_ms, :partition_key, :run_id, :event_type, jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
"#
}

statement! {
    /// [`InsertEvent`] with no lease, guarded on its run: a re-sent commit writes nothing twice.
    InsertEventUnfenced<'a> {
        partition_key: &'a str,
        occurred_at_ms: i64,
        run_id: &'a str,
        event_type: &'a str,
        payload: SqlValue,
    } as Bookkeeping => r#"
INSERT INTO events (occurred_at_ms, partition_key, run_id, type, payload)
SELECT :occurred_at_ms, :partition_key, :run_id, :event_type, jsonb(:payload)
FROM partitions
WHERE partition_key = :partition_key AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')
"#
}

statement! {
    /// Inserts a task row or updates every column but its `pos`; a terminal row stays as it is,
    /// and a task id another partition holds fails the commit as [`InsertItem`] does.
    UpsertTask<'a> {
        partition_key: &'a str,
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
  (task_id, partition_key, name, message_id, status, data, internal, updated_at_ms)
SELECT :task_id, :partition_key,
  :name, :message_id, :status, jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM partitions
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
ON CONFLICT(task_id) DO UPDATE SET
  status = excluded.status, name = excluded.name, message_id = excluded.message_id,
  data = excluded.data, internal = excluded.internal,
  updated_at_ms = excluded.updated_at_ms,
  partition_key = CASE WHEN tasks.partition_key = excluded.partition_key THEN tasks.partition_key END
WHERE tasks.status NOT IN ('completed', 'failed', 'canceled', 'rejected')
"#
}

statement! {
    /// [`UpsertTask`] with no lease, guarded on its run: a re-sent commit writes nothing twice.
    UpsertTaskUnfenced<'a> {
        partition_key: &'a str,
        run_id: &'a str,
        task_id: &'a str,
        name: Option<&'a str>,
        message_id: Option<&'a str>,
        status: &'a str,
        data: SqlValue,
        internal: SqlValue,
    } => r#"
INSERT INTO tasks
  (task_id, partition_key, name, message_id, status, data, internal, updated_at_ms)
SELECT :task_id, :partition_key,
  :name, :message_id, :status, jsonb(:data), jsonb(:internal), CAST(unixepoch('subsec') * 1000 AS INTEGER)
FROM partitions
WHERE partition_key = :partition_key AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')
ON CONFLICT(task_id) DO UPDATE SET
  status = excluded.status, name = excluded.name, message_id = excluded.message_id,
  data = excluded.data, internal = excluded.internal,
  updated_at_ms = excluded.updated_at_ms,
  partition_key = CASE WHEN tasks.partition_key = excluded.partition_key THEN tasks.partition_key END
WHERE tasks.status NOT IN ('completed', 'failed', 'canceled', 'rejected')
"#
}

statement! {
    /// Ends a commit and releases the lease: bumps `version` and writes the snapshots that
    /// changed (NULL keeps the stored one). No row answered means the run outlived its lease.
    CommitAndRelease<'a> {
        partition_key: &'a str,
        version: i64,
        token: &'a str,
        state: SqlValue,
        drained_id: Option<i64>,
    } as Commit => r#"
UPDATE partitions
SET version = version + 1,
  state = COALESCE(jsonb(:state), state),
  drained_id = COALESCE(:drained_id, drained_id),
  lease_expires_at_ms = 0, lease_token = NULL
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
RETURNING version
"#
}

statement! {
    /// Aborts the enclosing batch unless the lease still holds the partition at `version`: inserting
    /// a partition row without a version fails its NOT NULL constraint. What follows it in the batch,
    /// application SQL that cannot itself be fenced, then never applies after a takeover.
    AssertFence<'a> { partition_key: &'a str, version: i64, token: &'a str } as Bookkeeping => r#"
INSERT INTO partitions (partition_key, version)
SELECT :partition_key, NULL
WHERE NOT EXISTS (
  SELECT 1 FROM partitions WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
)
"#
}

statement! {
    /// [`CommitAndRelease`] that keeps the lease, extended, for a runner that continues.
    CommitAndExtend<'a> {
        partition_key: &'a str,
        version: i64,
        token: &'a str,
        state: SqlValue,
        drained_id: Option<i64>,
        lease_ms: i64,
    } as Commit => r#"
UPDATE partitions
SET version = version + 1,
  state = COALESCE(jsonb(:state), state),
  drained_id = COALESCE(:drained_id, drained_id),
  lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :lease_ms
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
RETURNING version
"#
}

statement! {
    /// A conversation checkpoint retains the active run's version and ownership.
    CheckpointState<'a> { partition_key: &'a str, version: i64, token: &'a str, state: SqlValue } as Commit => r#"
UPDATE partitions SET state = COALESCE(jsonb(:state), state)
WHERE partition_key = :partition_key AND version = :version AND lease_token = :token
RETURNING version
"#
}

statement! {
    /// The partition's history in order.
    SelectItems<'a> { partition_key: &'a str } => r#"
SELECT id, message_id, json(payload) FROM items WHERE partition_key = :partition_key ORDER BY seq
"#
}

statement! {
    /// One page of the thread's items in insertion order (`pos`), after `after`, of one partition
    /// when given: each row its position, partition and seq, then the item.
    SelectItemsAfter<'a> { partition_key: Option<&'a str>, after: i64, limit: i64 } => r#"
SELECT pos, partition_key, seq, id, message_id, json(payload) FROM items
WHERE :partition_key IS NULL AND pos > :after
UNION ALL
SELECT pos, partition_key, seq, id, message_id, json(payload) FROM items
WHERE :partition_key IS NOT NULL AND partition_key = :partition_key AND pos > :after
ORDER BY pos LIMIT :limit
"#
}

statement! {
    /// [`SelectItemsAfter`] newest first: the page before `before`.
    SelectItemsBefore<'a> { partition_key: Option<&'a str>, before: i64, limit: i64 } => r#"
SELECT pos, partition_key, seq, id, message_id, json(payload) FROM items
WHERE :partition_key IS NULL AND pos < :before
UNION ALL
SELECT pos, partition_key, seq, id, message_id, json(payload) FROM items
WHERE :partition_key IS NOT NULL AND partition_key = :partition_key AND pos < :before
ORDER BY pos DESC LIMIT :limit
"#
}

statement! {
    /// The partition's journal after the event `after_id`.
    SelectPartitionEvents<'a> { partition_key: &'a str, after_id: i64 } => r#"
SELECT id, occurred_at_ms, partition_key, run_id, type, json(payload) FROM events
WHERE partition_key = :partition_key AND id > :after_id ORDER BY id
"#
}

statement! {
    /// Every partition's journal after the event `after_id`, at most `limit` events.
    SelectEvents { after_id: i64, limit: i64 } => r#"
SELECT id, occurred_at_ms, partition_key, run_id, type, json(payload) FROM events
WHERE id > :after_id ORDER BY id LIMIT :limit
"#
}

statement! {
    /// One partition's row.
    SelectPartition<'a> { partition_key: &'a str } => r#"
SELECT partition_key, version, json(state), lease_holder, lease_expires_at_ms, drained_id
FROM partitions WHERE partition_key = :partition_key
"#
}

statement! {
    /// Every partition's row.
    SelectPartitions {} => r#"
SELECT partition_key, version, json(state), lease_holder, lease_expires_at_ms, drained_id
FROM partitions ORDER BY partition_key
"#
}

statement! {
    /// The newest `limit` task rows.
    SelectTasks<'a> { partition_key: &'a str, limit: i64 } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks WHERE partition_key = :partition_key ORDER BY pos DESC LIMIT :limit
"#
}

statement! {
    /// The newest `limit` task rows of one status.
    SelectTasksByStatus<'a> { partition_key: &'a str, status: &'a str, limit: i64 } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks WHERE partition_key = :partition_key AND status = :status ORDER BY pos DESC LIMIT :limit
"#
}

statement! {
    /// One task row.
    SelectTask<'a> { partition_key: &'a str, task_id: &'a str } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks WHERE partition_key = :partition_key AND task_id = :task_id
"#
}

statement! {
    /// One task row, whichever partition it is on.
    SelectThreadTask<'a> { task_id: &'a str } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks WHERE task_id = :task_id
"#
}

statement! {
    /// One page of the thread's tasks in A2A `tasks/list` order: last updated first, then by
    /// task id. Of `partition_key` and `status` when given, updated after `updated_after`, and past the
    /// previous page's last row, `(before_at, before_id)`.
    SelectThreadTasksPage<'a> {
        partition_key: Option<&'a str>,
        status: Option<&'a str>,
        updated_after: i64,
        before_at: i64,
        before_id: &'a str,
        limit: i64,
    } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks
WHERE (:partition_key IS NULL OR partition_key = :partition_key)
  AND (:status IS NULL OR status = :status)
  AND updated_at_ms > :updated_after
  AND (updated_at_ms < :before_at OR (updated_at_ms = :before_at AND task_id < :before_id))
ORDER BY updated_at_ms DESC, task_id DESC
LIMIT :limit
"#
}

statement! {
    /// One page of the thread's tasks in insertion order (`pos`), after `after`, with
    /// [`SelectThreadTasksPage`]'s filters: each row its position, then the task.
    SelectThreadTasksInserted<'a> {
        partition_key: Option<&'a str>,
        status: Option<&'a str>,
        updated_after: i64,
        after: i64,
        limit: i64,
    } => r#"
SELECT pos, task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks
WHERE (:partition_key IS NULL OR partition_key = :partition_key)
  AND (:status IS NULL OR status = :status)
  AND updated_at_ms > :updated_after
  AND pos > :after
ORDER BY pos
LIMIT :limit
"#
}

statement! {
    /// How many tasks a task page's filters match, every page.
    CountThreadTasks<'a> { partition_key: Option<&'a str>, status: Option<&'a str>, updated_after: i64 } => r#"
SELECT COUNT(*) FROM tasks
WHERE (:partition_key IS NULL OR partition_key = :partition_key) AND (:status IS NULL OR status = :status)
  AND updated_at_ms > :updated_after
"#
}

statement! {
    /// The task row an A2A message id created.
    SelectTaskForMessage<'a> { partition_key: &'a str, message_id: &'a str } => r#"
SELECT task_id, partition_key, pos, name, message_id, status, updated_at_ms,
  json(data), json(internal)
FROM tasks WHERE partition_key = :partition_key AND message_id = :message_id
"#
}

statement! {
    /// Every inbox row in arrival order.
    SelectInbox<'a> { partition_key: &'a str } => r#"
SELECT id, partition_key, task_id, message_id, json(message), received_at_ms FROM inbox
WHERE partition_key = :partition_key ORDER BY id
"#
}

statement! {
    /// The inbox rows not yet drained, past the partition's `drained_id`.
    SelectInboxPending<'a> { partition_key: &'a str } => r#"
SELECT id, partition_key, task_id, message_id, json(message), received_at_ms FROM inbox
WHERE partition_key = :partition_key AND id > (SELECT drained_id FROM partitions WHERE partition_key = :partition_key) ORDER BY id
"#
}

/// The framework's private tables, beside the platform's v1 thread tables: context scopes, and
/// application transactions applied by request id, so a resend after a lost acknowledgment
/// finds its own and writes nothing twice (a day of them is kept). Sent with the bootstrap and
/// every acquire, so a database an older SDK created gains them.
pub const FRAMEWORK_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS context_scopes (
        scope_id TEXT PRIMARY KEY,
        partition_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        fields JSONB NOT NULL,
        contracts JSONB NOT NULL,
        manifests JSONB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS applied_writes (
        id TEXT PRIMARY KEY,
        applied_at_ms INTEGER NOT NULL
    )";

statement! {
    /// Records an application transaction, first in its batch: a second apply fails the batch.
    RecordAppliedWrite<'a> { id: &'a str } => r#"
INSERT INTO applied_writes (id, applied_at_ms) VALUES (:id, CAST(unixepoch('subsec') * 1000 AS INTEGER))
"#
}

statement! {
    /// Forgets the application transactions applied more than a day ago.
    PruneAppliedWrites {} => "DELETE FROM applied_writes WHERE applied_at_ms < CAST(unixepoch('subsec') * 1000 AS INTEGER) - 86400000"
}

statement! {
    /// Whether an application transaction applied.
    SelectAppliedWrite<'a> { id: &'a str } => "SELECT 1 FROM applied_writes WHERE id = :id"
}

statement! {
    /// The partition's scopes.
    SelectScopes<'a> { partition_key: &'a str } => r#"
SELECT scope_id, revision, json(fields), json(contracts), json(manifests)
FROM context_scopes WHERE partition_key = :partition_key ORDER BY scope_id
"#
}

statement! {
    /// Writes a scope when its stored revision is `expected_revision` (a missing scope is at
    /// 0) and it is not another partition's; under the partition's lease when a fence is given.
    UpsertScope<'a> {
        scope_id: &'a str,
        partition_key: &'a str,
        expected_revision: i64,
        revision: i64,
        fields: SqlValue,
        contracts: SqlValue,
        manifests: SqlValue,
        version: Option<i64>,
        token: Option<&'a str>,
    } => r#"
INSERT INTO context_scopes (scope_id, partition_key, revision, fields, contracts, manifests)
SELECT :scope_id, :partition_key,
  CASE WHEN COALESCE((SELECT revision FROM context_scopes WHERE scope_id = :scope_id), 0)
            = :expected_revision
       THEN :revision ELSE NULL END,
  jsonb(:fields), jsonb(:contracts), jsonb(:manifests)
WHERE :version IS NULL OR EXISTS
  (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND version = :version AND lease_token = :token)
ON CONFLICT(scope_id) DO UPDATE SET
  revision = CASE WHEN context_scopes.partition_key = excluded.partition_key THEN excluded.revision ELSE NULL END,
  fields = excluded.fields, contracts = excluded.contracts, manifests = excluded.manifests
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
        let mut partition_key: Row = vec![
            "front-desk".into(),
            SqlValue::Int(2),
            SqlValue::Null,
            SqlValue::Null,
            "not an int".into(),
            SqlValue::Int(0),
        ];
        assert!(
            matches!(PartitionRow::from_row(&partition_key), Err(StorageError::Invalid(m)) if m.starts_with("lease_expires_at_ms"))
        );
        partition_key[4] = SqlValue::Int(0);
        assert_eq!(PartitionRow::from_row(&partition_key).unwrap().lease_expires_at_ms, 0);
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

    /// An unfenced write is its fenced twin guarded on its run instead of the fence, and a commit
    /// that keeps the lease is the one that releases it but for the lease.
    #[test]
    fn twin_statements_differ_only_where_they_should() {
        let fence = " AND version = :version AND lease_token = :token";
        let run_not_ended = " AND NOT EXISTS (SELECT 1 FROM events WHERE partition_key = :partition_key AND run_id = :run_id AND type = 'run_ended')";
        for (fenced, unfenced) in [
            (InsertItem::SQL, InsertItemUnfenced::SQL),
            (DeleteItem::SQL, DeleteItemUnfenced::SQL),
            (InsertEvent::SQL, InsertEventUnfenced::SQL),
            (UpsertTask::SQL, UpsertTaskUnfenced::SQL),
        ] {
            assert!(fenced.contains(fence), "{fenced}");
            assert_eq!(fenced.replace(fence, run_not_ended), unfenced);
        }
        assert_eq!(
            CommitAndRelease::SQL.replace(
                "lease_expires_at_ms = 0, lease_token = NULL",
                "lease_expires_at_ms = CAST(unixepoch('subsec') * 1000 AS INTEGER) + :lease_ms"
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
            (SelectPartitionEvents::SQL, EventRow::SELECT),
            (SelectEvents::SQL, EventRow::SELECT),
            (SelectPartition::SQL, PartitionRow::SELECT),
            (SelectPartitions::SQL, PartitionRow::SELECT),
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

statement! {
    /// Pending inbox entries with their originally admitted handler destination.
    SelectRoutedInbox<'a> { partition_key: &'a str } => r#"
SELECT i.id, i.partition_key, i.task_id, i.message_id, json(i.message), i.received_at_ms, t.endpoint
FROM inbox i JOIN tasks t ON t.partition_key = i.partition_key AND t.task_id = i.task_id
WHERE i.partition_key = :partition_key
  AND i.id > COALESCE((SELECT drained_id FROM partitions WHERE partition_key = :partition_key), 0)
ORDER BY i.id
"#
}

statement! {
    /// Immutable values referenced by compact routing snapshots, indexed by event_key.
    SelectSessionItems<'a> { partition_key: &'a str, keys: SqlValue } => r#"
SELECT event_key, json(payload) FROM events
WHERE partition_key = :partition_key AND type = 'session_item'
  AND event_key IN (SELECT value FROM json_each(:keys))
"#
}

statement! {
    /// The checkpoint inserts new immutable history revisions before publishing references.
    InsertSessionItem<'a> { partition_key: &'a str, version: i64, token: &'a str, key: &'a str, run_id: &'a str, occurred_at_ms: i64, value: SqlValue } => r#"
INSERT INTO events (partition_key, run_id, occurred_at_ms, type, event_key, payload)
SELECT :partition_key, :run_id, :occurred_at_ms, 'session_item', :key, jsonb(:value)
WHERE EXISTS (SELECT 1 FROM partitions WHERE partition_key = :partition_key AND version = :version AND lease_token = :token)
"#
}
