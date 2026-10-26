//! The SQL text of a thread database: the statements [`super::ThreadDb`] issues (plan 3.3.2,
//! 3.3.4 and 5.2) and the builders of the ones that vary with the fence or the lease outcome.

use crate::schema::{ChatItemRow, EventRow, InboxRow, LaneRow, TaskRow};

use super::LeaseAfter;

pub const INSERT_LANE: &str = "INSERT OR IGNORE INTO lanes (lane, version) VALUES (?, 0)";
/// Waiters take the lane in arrival order (docs/lanes.md): a run whose message has a `submitted`
/// row yields to an older one still waiting, as long as that row is fresh (a waiter that died
/// leaves its row behind; past the wait timeout it no longer counts). A run with no row of its
/// own, a voice turn, is not ordered, and rows queued in the inbox are drained as one batch and
/// order nothing.
pub const ACQUIRE: &str = "UPDATE lanes SET lease_holder = ?, lease_token = ?, lease_until = unixepoch() + ? \
     WHERE lane = ? AND (lease_until < unixepoch() OR lease_holder = ?) \
     AND NOT EXISTS (SELECT 1 FROM tasks older WHERE older.lane = lanes.lane AND older.status = 'submitted' \
     AND older.message_id IS NOT NULL AND older.updated_at > unixepoch() - ? \
     AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = older.lane AND inbox.task_id = older.task_id) \
     AND older.seq < (SELECT seq FROM tasks WHERE lane = lanes.lane AND task_id = ?)) RETURNING version";
/// After a successful acquire: the waiter that was this holder is served.
pub const CLEAR_WAITER: &str =
    "UPDATE lanes SET waiter = NULL WHERE lane = ? AND lease_token = ? AND waiter = ?";
pub const RUN_STARTED: &str = "INSERT INTO events (ts_ms, lane, run_id, type, payload) \
     SELECT ?, ?, ?, 'run_started', jsonb(?) FROM lanes WHERE lane = ? AND lease_token = ? \
     AND NOT EXISTS (SELECT 1 FROM events WHERE lane = lanes.lane AND run_id = ? AND type = 'run_ended')";
/// Plan 5.2: a `run_id` that already committed.
pub const COMMITTED_RUN: &str =
    "SELECT 1 FROM events WHERE lane = ? AND run_id = ? AND type = 'run_ended' LIMIT 1";
/// The token is the fence: a lapsed lease nobody took yet renews, so a holder stalled behind a
/// slow database keeps its lane; one another holder acquired, or a released one (the token is
/// cleared), does not.
pub const RENEW: &str = "UPDATE lanes SET lease_until = unixepoch() + ? \
     WHERE lane = ? AND lease_holder = ? AND lease_token = ? RETURNING version";
pub const RELEASE: &str =
    "UPDATE lanes SET lease_until = 0, lease_token = NULL WHERE lane = ? AND lease_token = ?";
pub const RUN_ENDED: &str = "INSERT INTO events (ts_ms, lane, run_id, type, payload) \
     SELECT ?, ?, ?, 'run_ended', jsonb(?) FROM lanes WHERE lane = ? AND lease_token = ?";
pub const SET_WAITER: &str =
    "UPDATE lanes SET waiter = ? WHERE lane = ? AND lease_until > unixepoch()";
pub const WAITER: &str = "SELECT waiter FROM lanes WHERE lane = ?";
/// The receive (plan 3.3.4) is one atomic `Batch`, the `BEGIN .. COMMIT` of the plan: inside an
/// interactive transaction agent-db answers a `RETURNING` with an `ExecResult` only, and a
/// `Query` runs on the read pool outside the transaction, so the lease's `RETURNING version` needs
/// the batch. The task insert and the lease attempt are conditioned on the inbox row this request
/// inserted (`lane, message_id, task_id`), so a retried `messageId` inserts nothing, takes nothing,
/// and reads back the task it belongs to.
pub const INSERT_INBOX: &str = "INSERT OR IGNORE INTO inbox (lane, task_id, message_id, message, received_at) VALUES (?, ?, ?, jsonb(?), unixepoch())";
/// `OR IGNORE`: a message that continues a waiting task has its row already.
pub const INSERT_TASK_SUBMITTED: &str = "INSERT OR IGNORE INTO tasks (task_id, lane, seq, name, message_id, status, status_message, \
     data, internal, updated_at) \
     SELECT ?, ?, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = ?), 0) + 1, ?, ?, 'submitted', NULL, \
     jsonb(?), jsonb(?), unixepoch() \
     FROM inbox WHERE lane = ? AND message_id = ? AND task_id = ?";
/// The early `submitted` row under the other policies (plan 3.3.4): a small unfenced insert
/// before the lease is tried, so a `tasks/get` before the commit finds the task. `OR IGNORE`
/// covers a retried `task_id` and a `messageId` already on a row of the lane; the commit's upsert
/// moves the row to its terminal state and keeps its `seq`.
pub const INSERT_SUBMITTED_TASK: &str = "INSERT OR IGNORE INTO tasks (task_id, lane, seq, name, message_id, status, status_message, \
     data, internal, updated_at) \
     SELECT ?, ?, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = ?), 0) + 1, ?, ?, 'submitted', NULL, \
     jsonb(?), jsonb(?), unixepoch()";
pub const ACQUIRE_FOR_MESSAGE: &str = "UPDATE lanes SET lease_holder = ?, lease_token = ?, lease_until = unixepoch() + ? \
     WHERE lane = ? AND (lease_until < unixepoch() OR lease_holder = ?) \
     AND EXISTS (SELECT 1 FROM inbox WHERE lane = lanes.lane AND message_id = ? AND task_id = ?) RETURNING version";
pub const INBOX_TASK: &str = "SELECT task_id FROM inbox WHERE lane = ? AND message_id = ?";
pub const COUNT_PENDING: &str = "SELECT COUNT(*) FROM inbox WHERE lane = ? AND id > (SELECT drained_id FROM lanes WHERE lane = ?)";
/// The drain check's release: only when nothing is pending above `drained_id`, in the same
/// statement, so a row inserted between the count and the release keeps the lease held.
pub const RELEASE_IF_DRAINED: &str = "UPDATE lanes SET lease_until = 0, lease_token = NULL WHERE lane = ? AND lease_token = ? \
     AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = lanes.lane AND inbox.id > lanes.drained_id)";
/// The pickup is these two in one batch: the status before, then the move to `working`. An
/// `input-required` row is picked up by the message that continues it.
pub const PICKUP_STATUS: &str = "SELECT status FROM tasks \
     WHERE lane = ? AND task_id = ? AND status IN ('submitted', 'working', 'input-required')";
pub const PICKUP_TASK: &str = "UPDATE tasks SET status = 'working', updated_at = unixepoch() \
     WHERE lane = ? AND task_id = ? AND status IN ('submitted', 'working', 'input-required')";
/// The pickup inside the acquire batch: the run's row goes `working` only under this lease.
pub const PICKUP_HELD: &str = "UPDATE tasks SET status = 'working', updated_at = unixepoch() \
     WHERE lane = ? AND task_id = ? AND status IN ('submitted', 'working', 'input-required') \
     AND EXISTS (SELECT 1 FROM lanes WHERE lane = tasks.lane AND lease_token = ?)";
pub const TASK_STATUS: &str = "SELECT status FROM tasks WHERE lane = ? AND task_id = ?";
pub const MARK_ORPHAN_FAILED: &str = "UPDATE tasks SET status = 'failed', status_message = jsonb(?), updated_at = unixepoch() \
     WHERE lane = ? AND task_id = ? AND status = 'working'";
/// The schema probe: one read, which fails `no such table` on a database no SDK wrote yet.
pub const SCHEMA_VERSION_READ: &str = "SELECT version FROM schema_version";
pub const NO_SCHEMA: &str = "no such table: schema_version";

/// The fence every commit write carries: the lane at the loaded version, under the lease token.
const FENCED: &str = "FROM lanes WHERE lane = ? AND version = ? AND lease_token = ?";
/// The `concurrent` policy's condition: the lane exists.
const UNFENCED: &str = "FROM lanes WHERE lane = ?";

pub fn insert_item(fenced: bool) -> String {
    format!(
        "INSERT INTO items (id, lane, seq, message_id, payload) \
         SELECT ?, ?, COALESCE((SELECT MAX(seq) FROM items WHERE lane = ?), 0) + 1, ?, jsonb(?) {} \
         ON CONFLICT(id) DO UPDATE SET message_id = excluded.message_id, payload = excluded.payload",
        cond(fenced)
    )
}

pub fn delete_item(fenced: bool) -> String {
    format!("DELETE FROM items WHERE lane = ? AND id = ? AND EXISTS (SELECT 1 {})", cond(fenced))
}

pub fn insert_event(fenced: bool) -> String {
    format!(
        "INSERT INTO events (ts_ms, lane, run_id, type, payload) SELECT ?, ?, ?, ?, jsonb(?) {}",
        cond(fenced)
    )
}

pub fn upsert_task(fenced: bool) -> String {
    format!(
        "INSERT INTO tasks (task_id, lane, seq, name, message_id, status, status_message, data, internal, updated_at) \
         SELECT ?, ?, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = ?), 0) + 1, ?, ?, ?, jsonb(?), jsonb(?), jsonb(?), \
         unixepoch() {} \
         ON CONFLICT(task_id) DO UPDATE SET status = excluded.status, name = excluded.name, \
         message_id = excluded.message_id, status_message = excluded.status_message, data = excluded.data, \
         internal = excluded.internal, updated_at = excluded.updated_at",
        cond(fenced)
    )
}

/// The lane update that ends a fenced commit: bump `version`, write the snapshots that changed,
/// and release (`lease_until = 0`) or renew.
pub fn update_lane(after: LeaseAfter) -> String {
    let lease = match after {
        LeaseAfter::Release => "0, lease_token = NULL",
        LeaseAfter::Renew => "unixepoch() + ?",
    };
    format!(
        "UPDATE lanes SET version = version + 1, state = COALESCE(jsonb(?), state), \
         agent_stack = COALESCE(jsonb(?), agent_stack), drained_id = COALESCE(?, drained_id), lease_until = {lease} \
         WHERE lane = ? AND version = ? AND lease_token = ? RETURNING version"
    )
}

fn cond(fenced: bool) -> &'static str {
    if fenced { FENCED } else { UNFENCED }
}

pub fn select_items() -> String {
    format!("SELECT {} FROM items WHERE lane = ? ORDER BY seq", ChatItemRow::SELECT)
}
pub fn select_lane_events() -> String {
    format!("SELECT {} FROM events WHERE lane = ? AND id > ? ORDER BY id", EventRow::SELECT)
}
pub fn select_events() -> String {
    format!("SELECT {} FROM events WHERE id > ? ORDER BY id LIMIT ?", EventRow::SELECT)
}
pub fn select_lane() -> String {
    format!("SELECT {} FROM lanes WHERE lane = ?", LaneRow::SELECT)
}
pub fn select_lanes() -> String {
    format!("SELECT {} FROM lanes ORDER BY lane", LaneRow::SELECT)
}
pub fn select_tasks() -> String {
    format!("SELECT {} FROM tasks WHERE lane = ? ORDER BY seq DESC LIMIT ?", TaskRow::SELECT)
}
pub fn select_task() -> String {
    format!("SELECT {} FROM tasks WHERE lane = ? AND task_id = ?", TaskRow::SELECT)
}
pub fn select_tasks_by_status() -> String {
    format!(
        "SELECT {} FROM tasks WHERE lane = ? AND status = ? ORDER BY seq DESC LIMIT ?",
        TaskRow::SELECT
    )
}
/// `working` rows: a crashed holder's, re-run once by the next holder (plan 3.3.2).
pub fn select_orphans() -> String {
    format!(
        "SELECT {} FROM tasks WHERE lane = ? AND status = 'working' ORDER BY seq",
        TaskRow::SELECT
    )
}
pub fn select_inbox() -> String {
    format!("SELECT {} FROM inbox WHERE lane = ? ORDER BY id", InboxRow::SELECT)
}

pub fn select_inbox_pending() -> String {
    format!(
        "SELECT {} FROM inbox WHERE lane = ? AND id > (SELECT drained_id FROM lanes WHERE lane = ?) ORDER BY id",
        InboxRow::SELECT
    )
}
