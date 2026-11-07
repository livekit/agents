//! A run's records and decisions: the task states and their transitions ([`merge_write`]), the
//! run's outcome and usage, and what a run commits ([`run_commit`]): the changed and deleted
//! items, the journal entries, the task rows, the snapshot that changed and `run_ended`.

use std::time::Duration;

use crate::thread::db::{Commit, Committed, EventWrite, Fenced, Resolution};
use crate::thread::lease::LanePoll;
use crate::thread::schema::*;
use crate::{LaneAccess, LeaseAfterCommit, StorageError, TaskPickup, TaskWrite, WhenHeld};
use livekit_protocol::agent::chat_context::ChatItem;
use lk_agents_abi::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use lk_agents_abi::storage::journal::{JournalEvent, RunEnded, RunStarted};
use lk_agents_abi::usage::Usage;
use serde_json::Value;

use crate::shortuuid;
use crate::thread::lease::Lease;
use lk_agents_net::clock::unix_now_ms;

use super::{Lane, db};
use std::collections::BTreeMap;

pub use crate::thread::schema::{ItemRow, LaneRow, TaskRow};

/// a waiter that has not acquired after 60 s fails `lease-timeout`.
pub use lk_agents_abi::a2a::options::LEASE_WAIT_TIMEOUT;

pub use lk_agents_abi::storage::TaskState;
pub use lk_agents_abi::storage::thread::{
    REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE, REASON_THREAD_NOT_FOUND, REASON_WORKER_LOST,
    RunOutcome, SUBMIT_CHANNEL, TaskPatch, WaitOptions, agent_message, merge_write, wait_patch,
};

/// Everything a run commits, as staged by its owner.
pub struct RunWrites<'a> {
    pub run_id: &'a str,
    pub task_id: &'a str,
    pub channel: Option<&'a str>,
    /// How the run took its lane; `Concurrent` writes append-only.
    pub access: LaneAccess,
    pub history: &'a [ItemRow],
    /// What was loaded, by id, in order.
    pub loaded: &'a [(String, ChatItem)],
    pub state: Option<&'a Value>,
    pub loaded_state: Option<&'a Value>,
    /// The journal entries the run recorded, in order.
    pub events: &'a [EventWrite],
    pub tasks: &'a [TaskWrite],
    pub usage: &'a Usage,
    pub outcome: RunOutcome,
    pub reason: Option<&'a str>,
    /// When the commit's own entries are stamped, Unix milliseconds.
    pub now_ms: i64,
    /// The last inbox row a drain consumed.
    pub drained_id: Option<i64>,
    /// The task the turn answers, read back with the commit.
    pub answer_task: Option<String>,
}

/// The run's one write: the changed and deleted items, their journal entries, the task rows,
/// usage, the snapshot when it changed, and `run_ended`. An append-only (`concurrent`) run writes
/// no state and may only add items: editing or deleting a loaded one is refused.
pub fn run_commit(w: RunWrites<'_>) -> Result<Commit, StorageError> {
    let (items, delete_items) = item_changes(w.history, w.loaded);
    let concurrent = w.access == LaneAccess::Concurrent;
    if concurrent {
        let edited = items.iter().find(|i| w.loaded.iter().any(|(id, _)| *id == i.id));
        if let Some(id) = edited.map(|i| &i.id).or(delete_items.first()) {
            return Err(StorageError::Invalid(format!(
                "a concurrent run only appends to the history; it edited or deleted item {id}"
            )));
        }
    }
    let mut events = Vec::new();
    if concurrent {
        let started = RunStarted {
            task_id: Some(w.task_id.to_owned()),
            stream_id: Some(w.task_id.to_owned()),
            channel: w.channel.map(str::to_owned),
            // the journal's spelling of a `concurrent` run
            exclusive: Some(false),
        };
        events.push(EventWrite::new(w.now_ms, JournalEvent::RunStarted(started)));
    }
    events.extend(items.iter().filter(|i| !w.loaded.iter().any(|(id, _)| *id == i.id)).map(|i| {
        EventWrite::new(
            w.now_ms,
            JournalEvent::ConversationItemAdded { item: Box::new(i.payload.clone()) },
        )
    }));
    events.extend(w.events.iter().cloned());
    if w.usage.requests != 0 || w.usage.tool_calls != 0 {
        events.push(EventWrite::new(w.now_ms, JournalEvent::Usage(w.usage.clone())));
    }
    Ok(Commit {
        run_id: w.run_id.to_owned(),
        delete_items,
        items,
        events,
        tasks: w.tasks.to_vec(),
        state: w.state.filter(|s| !concurrent && Some(*s) != w.loaded_state).cloned(),
        loaded_state: w.loaded_state.cloned(),
        context_scopes: Vec::new(),
        scope_writes: Vec::new(),
        app_writes: Vec::new(),
        drained_id: w.drained_id.filter(|_| !concurrent),
        run_ended: EventWrite::new(
            w.now_ms,
            JournalEvent::RunEnded(RunEnded {
                outcome: w.outcome,
                reason: w.reason.map(str::to_owned),
            }),
        ),
        read_task: w.answer_task,
    })
}

/// The stored items that differ from what was `loaded`, and the loaded ids no longer in the
/// history, in loaded order.
pub fn item_changes(
    history: &[ItemRow],
    loaded: &[(String, ChatItem)],
) -> (Vec<ItemRow>, Vec<String>) {
    let by_id: BTreeMap<&str, &ChatItem> = loaded.iter().map(|(id, i)| (id.as_str(), i)).collect();
    let stored: Vec<&ItemRow> = history.iter().filter(|i| i.stored()).collect();
    let changed = stored
        .iter()
        .filter(|i| by_id.get(i.id.as_str()) != Some(&&i.payload))
        .map(|i| (*i).clone())
        .collect();
    let kept: std::collections::BTreeSet<&str> = stored.iter().map(|i| i.id.as_str()).collect();
    let deleted =
        loaded.iter().filter(|(id, _)| !kept.contains(id.as_str())).map(|(id, _)| id.clone());
    (changed, deleted.collect())
}

pub use lk_agents_abi::a2a::metadata::{
    CHANNEL_KEY, EXTENSION_URI, INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, KIND_KEY,
    MERGED_INTO_KEY, REASON_KEY,
};

/// One run as the journal tells it: `run_started` and, when it ended, `run_ended`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub run_id: String,
    pub lane: String,
    pub task_id: Option<String>,
    pub channel: Option<String>,
    pub started_at_ms: i64,
    pub ended_at_ms: Option<i64>,
    pub outcome: Option<RunOutcome>,
    pub reason: Option<String>,
}

impl RunSummary {
    /// A `run_started` without a `run_ended`: the worker died mid-run.
    pub fn interrupted(&self) -> bool {
        self.ended_at_ms.is_none()
    }

    /// In the order the runs started; a `run_ended` whose run was not started is ignored, and a
    /// second `run_started` of a run (a holder re-entering its lease) replaces the first in place.
    /// A `run_started` or `run_ended` of the wrong shape is refused, never skipped.
    pub fn from_events(events: &[EventRow]) -> Result<Vec<RunSummary>, StorageError> {
        let mut runs: Vec<RunSummary> = Vec::new();
        for ev in events {
            if !matches!(ev.r#type.as_str(), "run_started" | "run_ended") {
                continue;
            }
            let event = JournalEvent::from_row(&ev.r#type, ev.payload.clone()).map_err(|e| {
                StorageError::Invalid(format!("journal entry {} ({}): {e}", ev.id, ev.r#type))
            })?;
            match event {
                JournalEvent::RunStarted(started) => {
                    let summary = RunSummary {
                        run_id: ev.run_id.clone(),
                        lane: ev.lane.clone(),
                        task_id: started.task_id,
                        channel: started.channel,
                        started_at_ms: ev.ts_ms,
                        ended_at_ms: None,
                        outcome: None,
                        reason: None,
                    };
                    match runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        Some(existing) => *existing = summary,
                        None => runs.push(summary),
                    }
                }
                JournalEvent::RunEnded(ended) => {
                    if let Some(run) = runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        run.ended_at_ms = Some(ev.ts_ms);
                        run.outcome = Some(ended.outcome);
                        run.reason = ended.reason;
                    }
                }
                _ => {}
            }
        }
        Ok(runs)
    }
}

/// How many rows a lookup by id or `messageId` scans.
pub const LOOKUP_LIMIT: i64 = 500;
/// How many `input-required` rows `pending_ask` scans.
pub const PENDING_ASK_LIMIT: i64 = 50;

#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Idempotency key: a `run_id` that already committed yields a closed run.
    pub run_id: String,
    /// The A2A task this run serves.
    pub task_id: String,
    pub access: LaneAccess,
    /// The request's channel, for the `run_started` event.
    pub channel: Option<String>,
    pub wait_timeout: Duration,
    /// What the acquire does with `task_id`'s row (`Run::picked_up`).
    pub pickup: TaskPickup,
}

impl RunOptions {
    pub fn new(run_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            task_id: task_id.into(),
            access: LaneAccess::Exclusive(WhenHeld::Wait),
            channel: None,
            wait_timeout: LEASE_WAIT_TIMEOUT,
            pickup: TaskPickup::Leave,
        }
    }
}

/// What this run appends to the lane's journal, stamped as recorded.
#[derive(Default)]
pub struct EventLog {
    new: Vec<EventWrite>,
}

impl EventLog {
    pub fn append(&mut self, event: JournalEvent) {
        self.new.push(EventWrite::new(unix_now_ms(), event));
    }
}

/// How long a commit whose acknowledgment was lost is resolved before the caller is answered
/// [`StorageError::Uncertain`]; the run then still holds the commit for [`Run::resolve_later`].
pub const RESOLVE_WAIT: Duration = Duration::from_secs(10);
/// How long a resolver keeps checking a commit nobody could resolve in the caller's wait.
pub const RESOLVE_LATER: Duration = Duration::from_secs(3600);

/// What [`Run::commit`] does besides writing the run.
#[derive(Debug, Clone, Default)]
pub struct CommitOptions {
    pub lease: LeaseAfterCommit,
    /// The last inbox row a drain consumed.
    pub drained_id: Option<i64>,
    /// The task the turn answers, read back with the commit.
    pub answer_task: Option<String>,
}

/// Everything a run reads and writes, committed together.
pub struct Run {
    pub(super) lane: Lane,
    /// The lease the run holds; `None` under the `concurrent` policy, which takes none.
    pub(super) lease: Option<Lease>,
    pub(super) access: LaneAccess,
    pub(super) holder: String,
    pub(super) run_id: String,
    pub(super) task_id: String,
    pub(super) channel: Option<String>,
    /// A `run_id` that already committed: the run writes nothing and answers nothing twice.
    pub(super) closed: bool,
    /// The whole history, loaded; append as the model and the tools speak, delete by removing.
    pub(super) history: Vec<ItemRow>,
    /// What was loaded, by id, in order.
    pub(super) loaded: Vec<(String, ChatItem)>,
    /// The state snapshot as loaded, `None` for a fresh lane; the caller can replace it
    /// explicitly, and the commit writes it when it differs from what was loaded.
    pub(super) state: Option<Value>,
    pub(super) context_scopes: Vec<ContextScope>,
    pub(super) scope_writes: Vec<ContextScopeWrite>,
    /// Application statements managed transactions staged, applied first in the commit.
    pub(super) app_writes: Vec<AppStatement>,
    pub(super) loaded_state: Option<Value>,
    pub(super) events: EventLog,
    pub(super) tasks: Tasks,
    /// Whether the acquire's `pickup` took the row to `working`.
    pub(super) picked_up: bool,
    pub(super) usage: Usage,
    pub(super) outcome: RunOutcome,
    pub(super) reason: Option<String>,
    pub(super) finished: bool,
    /// The commit [`Run::commit`] could not resolve in its wait, and whether it kept the lease.
    pub(super) unresolved: Option<(Commit, LeaseAfterCommit)>,
}

impl Run {
    pub fn holder(&self) -> &str {
        &self.holder
    }
    pub fn id(&self) -> &str {
        &self.run_id
    }
    pub fn closed(&self) -> bool {
        self.closed
    }
    pub fn picked_up(&self) -> bool {
        self.picked_up
    }
    pub fn history(&self) -> &[ItemRow] {
        &self.history
    }
    pub fn history_mut(&mut self) -> &mut Vec<ItemRow> {
        &mut self.history
    }
    pub fn state(&self) -> Option<&Value> {
        self.state.as_ref()
    }
    pub fn set_state(&mut self, state: Value) {
        self.state = Some(state);
    }
    pub fn context_scopes(&self) -> &[ContextScope] {
        &self.context_scopes
    }
    pub fn stage_scopes(&mut self, writes: Vec<ContextScopeWrite>) {
        self.scope_writes = writes;
    }
    pub fn stage_statements(&mut self, writes: Vec<AppStatement>) {
        self.app_writes = writes;
    }
    pub fn events(&mut self) -> &mut EventLog {
        &mut self.events
    }
    pub fn tasks(&mut self) -> &mut Tasks {
        &mut self.tasks
    }
    pub fn set_usage(&mut self, usage: Usage) {
        self.usage = usage;
    }
    pub fn end_as(&mut self, outcome: RunOutcome, reason: Option<String>) {
        self.outcome = outcome;
        self.reason = reason;
    }

    pub fn lease(&self) -> Option<&Lease> {
        self.lease.as_ref()
    }

    /// The version loaded: the commit's fence; `0` for a run that took no lease.
    pub fn version(&self) -> i64 {
        self.lease.as_ref().map_or(0, |l| l.fence().version)
    }

    /// The run's writes, as the core composes them.
    pub fn writes(&self, opts: CommitOptions) -> Result<Commit, StorageError> {
        let mut commit = run_commit(RunWrites {
            run_id: &self.run_id,
            task_id: &self.task_id,
            channel: self.channel.as_deref(),
            access: self.access,
            history: &self.history,
            loaded: &self.loaded,
            state: self.state.as_ref(),
            loaded_state: self.loaded_state.as_ref(),
            events: &self.events.new,
            tasks: self.tasks.writes(),
            usage: &self.usage,
            outcome: self.outcome,
            reason: self.reason.as_deref(),
            now_ms: unix_now_ms(),
            drained_id: opts.drained_id,
            answer_task: opts.answer_task,
        })?;
        commit.context_scopes = self.context_scopes.clone();
        commit.scope_writes = self.scope_writes.clone();
        commit.app_writes = self.app_writes.clone();
        Ok(commit)
    }

    fn finish(&mut self) -> Result<(), StorageError> {
        if self.finished {
            return Err(StorageError::Invalid(format!("run {} already ended", self.run_id)));
        }
        self.finished = true;
        if let Some(lease) = &self.lease {
            lease.stop();
        }
        Ok(())
    }

    /// Leaving normally: one fenced write of everything the run holds. A closed run releases the
    /// lease and writes nothing. `VersionConflict` means its token or version was replaced.
    pub async fn commit(&mut self, opts: CommitOptions) -> Result<Committed, StorageError> {
        self.finish()?;
        let version = self.version();
        if let Some(lease) = &self.lease
            && self.closed
        {
            lease.release(None).await?;
            return Ok(Committed { version, task: None });
        }
        if self.lease.is_none() {
            self.lane.thread.ensure_schema().await?;
        }
        let lease = opts.lease;
        let commit = self.writes(opts)?;
        match self.send(&commit, lease).await {
            Err(StorageError::Uncertain(why)) => self.resolve_lost_ack(commit, lease, why).await,
            Err(StorageError::Invalid(why)) if !commit.app_writes.is_empty() => {
                Err(self.refused_commit(&commit, why).await)
            }
            out => out,
        }
    }

    /// Resolves a commit whose acknowledgment was lost, keeping it for a later resolution
    /// while the answer is still uncertain.
    async fn resolve_lost_ack(
        &mut self,
        commit: Commit,
        lease: LeaseAfterCommit,
        why: String,
    ) -> Result<Committed, StorageError> {
        log::warn!(target: lk_agents_abi::LOG_TARGET, lane = self.lane.name(), runId = self.run_id.as_str(); "commit acknowledgment lost, resolving: {why}");
        let resolved = self.resolve(&commit, lease, RESOLVE_WAIT).await;
        if matches!(resolved, Err(StorageError::Uncertain(_))) {
            self.unresolved = Some((commit, lease));
        }
        resolved
    }

    /// A commit carrying application SQL the database refused: a lane that moved on aborted it
    /// at its fence (`VersionConflict`); otherwise an application statement was refused.
    async fn refused_commit(&self, commit: &Commit, why: String) -> StorageError {
        let fence = self.lease.as_ref().map(Lease::fence);
        let (conn, read_task) = (&*self.lane.thread.conn, commit.read_task.as_deref());
        match db::resolve_commit(conn, self.lane.name(), &self.run_id, fence, read_task).await {
            Ok(Resolution::Rejected) => StorageError::VersionConflict {
                lane: self.lane.name().to_owned(),
                expected: fence.map_or(0, |f| f.version),
            },
            _ => StorageError::Invalid(why),
        }
    }

    /// Sends the run's one write, fenced on its lease when it holds one.
    async fn send(
        &self,
        commit: &Commit,
        after: LeaseAfterCommit,
    ) -> Result<Committed, StorageError> {
        let fenced = self.lease.as_ref().map(|lease| Fenced {
            fence: lease.fence(),
            keep: (after == LeaseAfterCommit::Keep).then(|| self.lane.thread.timing.secs()),
        });
        db::commit(&*self.lane.thread.conn, self.lane.name(), commit, fenced).await
    }

    /// A commit whose acknowledgment was lost, resolved by the run's identity: committed (its
    /// record), not applied while the fence holds (sent again: the fence and the run id make it
    /// safe), or rejected (`VersionConflict`: the lane moved on without it). `Uncertain` once
    /// `wait` passes with no authoritative answer.
    async fn resolve(
        &self,
        commit: &Commit,
        lease: LeaseAfterCommit,
        wait: Duration,
    ) -> Result<Committed, StorageError> {
        let fence = self.lease.as_ref().map(Lease::fence);
        let epoch = tokio::time::Instant::now();
        let mut poll = LanePoll::new(epoch.elapsed(), wait);
        let mut last: String;
        loop {
            let (conn, read_task) = (&*self.lane.thread.conn, commit.read_task.as_deref());
            match db::resolve_commit(conn, self.lane.name(), &self.run_id, fence, read_task).await {
                Ok(Resolution::Committed(committed)) => return Ok(*committed),
                Ok(Resolution::Rejected) => {
                    return Err(StorageError::VersionConflict {
                        lane: self.lane.name().to_owned(),
                        expected: fence.map_or(0, |f| f.version),
                    });
                }
                Ok(Resolution::NotApplied) => match self.send(commit, lease).await {
                    Err(StorageError::Uncertain(why) | StorageError::Unavailable(why)) => {
                        last = why
                    }
                    out => return out,
                },
                Err(e) => last = e.to_string(),
            }
            let Some(at) = poll.next(epoch.elapsed()) else {
                return Err(StorageError::Uncertain(format!(
                    "run {} unresolved after {}s: {last}",
                    self.run_id,
                    wait.as_secs_f64()
                )));
            };
            tokio::time::sleep_until(epoch + at).await;
        }
    }

    /// Keeps resolving a commit [`Run::commit`] left uncertain, up to [`RESOLVE_LATER`]: the
    /// authoritative outcome, never a guess. `None` when nothing was left unresolved.
    pub async fn resolve_later(mut self) -> Option<Result<Committed, StorageError>> {
        let (commit, lease) = self.unresolved.take()?;
        Some(self.resolve(&commit, lease, RESOLVE_LATER).await)
    }

    /// Leaving by error: `run_ended` with the outcome and the reason, and the release; nothing
    /// else. A `run_ended` that could not be written is logged.
    pub async fn abandon(
        &mut self,
        outcome: RunOutcome,
        reason: impl Into<String>,
    ) -> Result<(), StorageError> {
        self.finish()?;
        let Some(lease) = &self.lease else { return Ok(()) };
        if self.closed {
            return lease.release(None).await;
        }
        let ended = EventWrite::new(
            unix_now_ms(),
            JournalEvent::RunEnded(RunEnded { outcome, reason: Some(reason.into()) }),
        );
        if let Err(e) = lease.release(Some((&self.run_id, &ended))).await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, lane = self.lane.name(), runId = self.run_id.as_str(); "run_ended could not be written after a failed run: {e}");
        }
        Ok(())
    }
}

/// The lane's task rows as this run writes them.
pub struct Tasks {
    lane: Lane,
    /// In write order.
    writes: Vec<TaskWrite>,
    /// The lane's newest rows as the acquire loaded them, newest first: what the run's lookups
    /// read instead of the database. Complete when shorter than [`LOOKUP_LIMIT`].
    loaded: Option<Vec<TaskRow>>,
}

impl Tasks {
    pub fn new(lane: Lane) -> Self {
        Self { lane, writes: Vec::new(), loaded: None }
    }

    /// Over the rows an acquire loaded, newest first.
    pub fn loaded(lane: Lane, loaded: Vec<TaskRow>) -> Self {
        Self { lane, writes: Vec::new(), loaded: Some(loaded) }
    }

    /// The loaded rows, when they are the whole lane.
    fn snapshot(&self) -> Option<&[TaskRow]> {
        self.loaded.as_deref().filter(|rows| rows.len() < LOOKUP_LIMIT as usize)
    }

    /// The lane's newest rows, `limit` of them, as of the acquire.
    pub async fn recent(&self, limit: i64) -> Result<Vec<TaskRow>, StorageError> {
        if let Some(rows) = self.loaded.as_deref() {
            return Ok(rows.iter().take(limit as usize).cloned().collect());
        }
        self.lane.tasks(None, limit).await
    }

    /// The `working` rows, oldest first: a dead holder's, as of the acquire.
    pub async fn working(&self) -> Result<Vec<TaskRow>, StorageError> {
        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .rev()
                .filter(|r| r.status == TaskState::Working)
                .cloned()
                .collect());
        }
        let working = self.lane.tasks(Some(TaskState::Working), LOOKUP_LIMIT).await?;
        Ok(working.into_iter().rev().collect())
    }

    /// The pending writes, in the order they were first made.
    pub fn writes(&self) -> &[TaskWrite] {
        &self.writes
    }

    /// The first loaded row that matches, else, when the loaded rows are not the whole lane, the
    /// stored one `read` finds.
    async fn find(
        &self,
        matches: impl Fn(&TaskRow) -> bool,
        read: impl std::future::Future<Output = Result<Option<TaskRow>, StorageError>>,
    ) -> Result<Option<TaskRow>, StorageError> {
        match self.loaded.as_deref().and_then(|rows| rows.iter().find(|r| matches(r))) {
            Some(r) => Ok(Some(r.clone())),
            None if self.snapshot().is_some() => Ok(None),
            None => read.await,
        }
    }

    /// The requested pending question, or the newest one when no task was named.
    pub async fn pending_ask(
        &self,
        task_id: Option<&str>,
    ) -> Result<Option<TaskRow>, StorageError> {
        if let Some(task_id) = task_id {
            return Ok(self
                .find(|row| row.task_id == task_id, self.lane.task(task_id))
                .await?
                .filter(|row| {
                    row.status == TaskState::InputRequired
                        && row.internal.as_ref().is_some_and(|internal| internal.ask.is_some())
                }));
        }

        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .find(|r| {
                    r.status == TaskState::InputRequired
                        && r.internal.as_ref().is_some_and(|i| i.ask.is_some())
                })
                .cloned());
        }
        Ok(self
            .lane
            .tasks(Some(TaskState::InputRequired), PENDING_ASK_LIMIT)
            .await?
            .into_iter()
            .find(|r| r.internal.as_ref().is_some_and(|i| i.ask.is_some())))
    }

    /// Insert or update one row; what the patch leaves `None` keeps the existing row's value. A
    /// stored row that is already terminal (a client canceled it mid-turn) stays as it is, as the
    /// commit would leave it; rewriting this run's own terminal write is refused.
    pub async fn write(
        &mut self,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<(), StorageError> {
        let base = match self.writes.iter().find(|w| w.task_id == task_id) {
            Some(w) => Some(w.clone()),
            None => match self.find(|r| r.task_id == task_id, self.lane.task(task_id)).await? {
                Some(row) if row.terminal() => return Ok(()),
                row => row.map(TaskWrite::from),
            },
        };
        let write = merge_write(task_id, base.as_ref(), status, patch)?;
        match self.writes.iter().position(|w| w.task_id == task_id) {
            Some(i) => self.writes[i] = write,
            None => self.writes.push(write),
        }
        Ok(())
    }

    /// `input-required`: the question is the status message, `ask` the pending transition.
    pub async fn wait(
        &mut self,
        task_id: &str,
        question: &str,
        opts: WaitOptions,
    ) -> Result<(), StorageError> {
        let patch = wait_patch(task_id, question, opts, shortuuid("msg_"));
        self.write(task_id, TaskState::InputRequired, patch).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_protocol::agent::chat_context::chat_item;
    use serde_json::json;

    #[test]
    fn item_changes_skip_instructions_and_unchanged_items_and_list_deletions() {
        use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
        use livekit_protocol::agent::{AgentConfigUpdate, ChatMessage, ChatRole, FunctionCall};
        let msg = |id: &str, role: ChatRole, text: &str| {
            ItemRow::new(chat_item::Item::Message(ChatMessage {
                id: id.into(),
                role: role as i32,
                content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
                ..Default::default()
            }))
        };
        let call = ItemRow::new(chat_item::Item::FunctionCall(FunctionCall {
            id: "i3".into(),
            name: "f".into(),
            ..Default::default()
        }));
        let loaded: Vec<(String, ChatItem)> =
            [msg("i1", ChatRole::User, "hi"), msg("i2", ChatRole::Assistant, "hello"), call]
                .into_iter()
                .map(|i| (i.id, i.payload))
                .collect();
        let mut history = vec![
            msg("sys", ChatRole::System, "be nice"),
            msg("dev", ChatRole::Developer, "x"),
            ItemRow::new(chat_item::Item::AgentConfigUpdate(AgentConfigUpdate {
                id: "cfg".into(),
                ..Default::default()
            })),
            msg("i1", ChatRole::User, "hi"),
            // edited
            msg("i2", ChatRole::Assistant, "hey"),
            // i3 removed; i4 new
            ItemRow { message_id: Some("m4".into()), ..msg("i4", ChatRole::User, "bye") },
        ];
        let (changed, deleted) = item_changes(&history, &loaded);
        assert_eq!(
            changed.iter().map(|w| (w.id.as_str(), w.message_id.as_deref())).collect::<Vec<_>>(),
            vec![("i2", None), ("i4", Some("m4"))]
        );
        assert_eq!(deleted, vec!["i3".to_owned()]);
        history.clear();
        let (changed, deleted) = item_changes(&history, &loaded);
        assert!(changed.is_empty());
        assert_eq!(deleted, vec!["i1", "i2", "i3"]);
    }

    #[test]
    fn run_summaries_follow_run_started_and_run_ended() {
        let ev = |id: i64, run_id: &str, ty: &str, payload: Value| EventRow {
            id,
            ts_ms: id * 10,
            lane: "front-desk".into(),
            run_id: run_id.into(),
            r#type: ty.into(),
            payload,
        };
        let events = vec![
            ev(1, "r1", "run_started", json!({"task_id": "AST_1", "channel": "a2a"})),
            ev(2, "r1", "run_ended", json!({"outcome": "completed", "reason": null})),
            ev(3, "r2", "run_started", json!({"task_id": "AST_2", "channel": null})),
            ev(4, "r9", "run_ended", json!({"outcome": "failed"})),
        ];
        let runs = RunSummary::from_events(&events).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].outcome, Some(RunOutcome::Completed));
        assert_eq!(runs[0].ended_at_ms, Some(20));
        assert!(!runs[0].interrupted());
        assert_eq!(runs[1].channel, None);
        assert!(runs[1].interrupted());
        // docs/internals/storage.md#journal: a malformed entry is refused, not skipped
        let corrupt = [ev(5, "r3", "run_ended", json!({"outcome": "maybe"}))];
        assert!(matches!(RunSummary::from_events(&corrupt), Err(StorageError::Invalid(_))));
    }
}
