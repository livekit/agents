pub use lk_agents_abi::telemetry::health::PartitionKind;
use std::time::Duration;

use crate::thread::db::{Commit, Committed, EventWrite, Fenced, Resolution};
use crate::thread::schema::*;
use crate::{LeaseAfterCommit, PartitionAccess, StorageError, TaskPickup, TaskWrite, WhenHeld};
use livekit_protocol::agent::chat_context::ChatItem;
use lk_agents_abi::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use lk_agents_abi::storage::journal::{JournalEvent, RunEnded, RunStarted};
use lk_agents_abi::usage::Usage;
use serde_json::Value;

use crate::shortuuid;
use crate::sql::Bind as _;
use crate::thread::lease::Lease;
use lk_agents_net::clock::unix_now_ms;

use super::{Partition, Resend, db};
use lk_agents_abi::storage::Unacknowledged;
use std::collections::BTreeMap;

pub use crate::thread::schema::{ItemRow, PartitionRow, TaskRow};

/// a waiter that has not acquired after 60 s fails `lease-timeout`.
pub use lk_agents_abi::a2a::options::LEASE_WAIT_TIMEOUT;

pub use lk_agents_abi::storage::TaskState;
pub use lk_agents_abi::storage::thread::{
    RunOutcome, SUBMIT_CHANNEL, TaskPatch, TaskReason, WaitOptions, agent_message, merge_write,
    wait_patch,
};

/// Everything a run commits, as staged by its owner.
pub struct RunWrites<'a> {
    pub run_id: &'a str,
    pub task_id: &'a str,
    pub channel: Option<&'a str>,
    /// How the run took its partition; `Concurrent` writes append-only.
    pub access: PartitionAccess,
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
    let concurrent = w.access == PartitionAccess::Concurrent;
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
        session_items: Vec::new(),
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
    pub partition_key: String,
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
                        partition_key: ev.partition_key.clone(),
                        task_id: started.task_id,
                        channel: started.channel,
                        started_at_ms: ev.occurred_at_ms,
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
                        run.ended_at_ms = Some(ev.occurred_at_ms);
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
    pub access: PartitionAccess,
    /// The request's channel, for the `run_started` event.
    pub channel: Option<String>,
    pub wait_timeout: Duration,
    /// What the acquire does with `task_id`'s row (`Run::picked_up`).
    pub pickup: TaskPickup,
    /// What the run is, for the partition health metrics.
    pub kind: PartitionKind,
}

impl RunOptions {
    pub fn new(run_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            task_id: task_id.into(),
            access: PartitionAccess::Exclusive(WhenHeld::Wait),
            channel: None,
            wait_timeout: LEASE_WAIT_TIMEOUT,
            pickup: TaskPickup::Leave,
            kind: PartitionKind::A2a,
        }
    }
}

/// What this run appends to the partition's journal, stamped as recorded.
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
    pub(super) session_items: crate::session_snapshot::SessionItems,
    pub(super) partition_key: Partition,
    /// The lease the run holds; `None` under the `concurrent` policy, which takes none.
    pub(super) lease: Option<Lease>,
    pub(super) access: PartitionAccess,
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
    /// The state snapshot as loaded, `None` for a fresh partition; the caller can replace it
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
    /// The commit [`Run::commit`] could not resolve in its wait, whether it kept the lease, and
    /// why its outcome is unknown.
    pub(super) unresolved: Option<(Commit, LeaseAfterCommit, Unacknowledged)>,
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
        if let Some(state) = &commit.state {
            let (packed, items) =
                self.session_items.clone().pack(state, &shortuuid("session_item_"))?;
            commit.state = Some(packed);
            commit.session_items = items;
        }
        commit.loaded_state =
            commit.loaded_state.as_ref().map(crate::session_snapshot::routing_budget);
        Ok(commit)
    }

    fn finish(&mut self) -> Result<(), StorageError> {
        if self.finished {
            return Err(StorageError::Invalid(format!("run {} already ended", self.run_id)));
        }
        self.finished = true;
        Ok(())
    }

    /// Persist conversation progress without committing application fields or releasing ownership.
    pub async fn checkpoint_session(
        &mut self,
        state: Value,
        history: Vec<ChatItem>,
    ) -> Result<(), StorageError> {
        if self.closed || self.finished {
            return Err(StorageError::Invalid("the run has ended".into()));
        }
        let lease = self.lease.as_ref().ok_or_else(|| {
            StorageError::Invalid("session checkpoints require an exclusive turn".into())
        })?;
        lease.check()?;
        let rows: Vec<_> = history
            .into_iter()
            .map(|payload| {
                payload.item.map(ItemRow::new).ok_or_else(|| {
                    StorageError::Invalid(
                        "an empty conversation item cannot be checkpointed".into(),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let mut commit = run_commit(RunWrites {
            run_id: &self.run_id,
            task_id: &self.task_id,
            channel: self.channel.as_deref(),
            access: self.access,
            history: &rows,
            loaded: &self.loaded,
            state: Some(&state),
            loaded_state: self.loaded_state.as_ref(),
            events: &[],
            tasks: &[],
            usage: &Usage::default(),
            outcome: self.outcome,
            reason: None,
            now_ms: unix_now_ms(),
            drained_id: None,
            answer_task: None,
        })?;
        commit.context_scopes = self.context_scopes.clone();
        let id = shortuuid("checkpoint_");
        let mut cache = self.session_items.clone();
        let (packed, items) = cache.pack(&state, &id)?;
        commit.state = Some(packed);
        commit.session_items = items;
        let conn = &*self.partition_key.thread.conn;
        let partition_key = self.partition_key.name();
        let fence = lease.fence();
        let write = || async {
            lease.check()?;
            db::session_checkpoint(conn, partition_key, fence, &id, &commit).await
        };
        let probe = || async {
            let rows = conn.query(SelectAppliedWrite { id: &id }.bind()).await?;
            Ok((!rows.is_empty()).then_some(()))
        };
        self.partition_key
            .thread
            .resend(crate::telemetry::StorageWrite::Commit)
            .probed(write, probe)
            .await?;
        self.loaded = rows
            .iter()
            .filter(|row| row.stored())
            .map(|row| (row.id.clone(), row.payload.clone()))
            .collect();
        self.history = rows;
        self.session_items = cache;
        self.loaded_state = Some(state.clone());
        self.state = Some(state);
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
            let released = lease.release(None).await;
            lease.stop();
            return released.map(|()| Committed { version, task: None });
        }
        if self.lease.is_none() {
            self.partition_key.thread.ensure_schema().await?;
        }
        let lease = opts.lease;
        let out = match self.writes(opts) {
            Ok(commit) => match self.send(&commit, lease).await {
                Err(StorageError::Uncertain(why)) => {
                    self.resolve_lost_ack(commit, lease, why).await
                }
                Err(StorageError::Invalid(why)) if !commit.app_writes.is_empty() => {
                    Err(self.refused_commit(&commit, why).await)
                }
                out => out,
            },
            Err(error) => Err(error),
        };
        self.settle(&out, lease);
        out
    }

    /// The lease renews until the commit's outcome is known; one left unresolved keeps it for
    /// [`Run::resolve_later`]. A commit that released it leaves its handle nothing to release.
    fn settle(&self, out: &Result<Committed, StorageError>, after: LeaseAfterCommit) {
        let Some(lease) = &self.lease else { return };
        if matches!(out, Err(StorageError::Uncertain(_))) {
            return;
        }
        lease.stop();
        if out.is_ok() && after == LeaseAfterCommit::Release {
            lease.mark_released();
        }
    }

    /// Resolves a commit whose acknowledgment was lost, keeping it for a later resolution
    /// while the answer is still uncertain.
    async fn resolve_lost_ack(
        &mut self,
        commit: Commit,
        lease: LeaseAfterCommit,
        why: Unacknowledged,
    ) -> Result<Committed, StorageError> {
        log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.name(), runId = self.run_id.as_str(); "commit acknowledgment lost, resolving: {why}");
        match self.resolve(&commit, lease, RESOLVE_WAIT, why).await {
            Err(StorageError::Uncertain(why)) => {
                self.unresolved = Some((commit, lease, why.clone()));
                Err(StorageError::Uncertain(why))
            }
            resolved => resolved,
        }
    }

    /// A commit carrying application SQL the database refused: a partition that moved on aborted it
    /// at its fence (`VersionConflict`); otherwise an application statement was refused.
    async fn refused_commit(&self, commit: &Commit, why: String) -> StorageError {
        let fence = self.lease.as_ref().map(Lease::fence);
        let (conn, read_task) = (&*self.partition_key.thread.conn, commit.read_task.as_deref());
        match db::resolve_commit(conn, self.partition_key.name(), &self.run_id, fence, read_task)
            .await
        {
            Ok(Resolution::Rejected) => StorageError::VersionConflict {
                partition_key: self.partition_key.name().to_owned(),
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
            keep: (after == LeaseAfterCommit::Keep)
                .then(|| self.partition_key.thread.timing.millis()),
        });
        db::commit(&*self.partition_key.thread.conn, self.partition_key.name(), commit, fenced)
            .await
    }

    /// A commit whose acknowledgment was lost, resolved by the run's identity: committed (its
    /// record), not applied while the fence holds (sent again: the fence and the run id make it
    /// safe), or rejected (`VersionConflict`: the partition moved on without it). `Uncertain` once
    /// `wait` passes with no authoritative answer.
    async fn resolve(
        &self,
        commit: &Commit,
        lease: LeaseAfterCommit,
        wait: Duration,
        lost: Unacknowledged,
    ) -> Result<Committed, StorageError> {
        let fence = self.lease.as_ref().map(Lease::fence);
        let (conn, read_task) = (&*self.partition_key.thread.conn, commit.read_task.as_deref());
        let (partition_key, run_id) = (self.partition_key.name(), self.run_id.as_str());
        let probe = || async move {
            match db::resolve_commit(conn, partition_key, run_id, fence, read_task).await {
                Ok(Resolution::Committed(committed)) => Ok(Some(*committed)),
                Ok(Resolution::Rejected) => Err(StorageError::VersionConflict {
                    partition_key: partition_key.to_owned(),
                    expected: fence.map_or(0, |f| f.version),
                }),
                Ok(Resolution::NotApplied) => Ok(None),
                // a probe that could not be read answers nothing
                Err(e) => Err(StorageError::Unavailable(format!("resolving: {e}"))),
            }
        };
        let thread = &self.partition_key.thread;
        let (clock, cx) = (thread.background.clock.clone(), &thread.cx);
        Resend::after_lost(clock, wait, cx, crate::telemetry::StorageWrite::Commit, lost)
            .probed(|| self.send(commit, lease), probe)
            .await
    }

    /// Keeps resolving a commit [`Run::commit`] left uncertain, up to [`RESOLVE_LATER`]: the
    /// authoritative outcome, never a guess. `None` when nothing was left unresolved.
    pub async fn resolve_later(mut self) -> Option<Result<Committed, StorageError>> {
        let (commit, lease, lost) = self.unresolved.take()?;
        let out = self.resolve(&commit, lease, RESOLVE_LATER, lost).await;
        self.settle(&out, lease);
        Some(out)
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
        let released = lease.release(Some((&self.run_id, &ended))).await;
        lease.stop();
        if let Err(e) = released {
            log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.name(), runId = self.run_id.as_str(); "run_ended could not be written after a failed run: {e}");
        }
        Ok(())
    }
}

/// A run dropped before it committed or was abandoned (its owner's future was dropped, or an
/// error ended it) still ends: `run_ended` `canceled` and the release, written in the background.
impl Drop for Run {
    fn drop(&mut self) {
        if self.finished || self.closed {
            return;
        }
        if let Some(lease) = self.lease.take() {
            lease.release_dropped(&self.run_id);
        }
    }
}

/// The partition's task rows as this run writes them.
pub struct Tasks {
    partition_key: Partition,
    /// In write order.
    writes: Vec<TaskWrite>,
    /// The partition's newest rows as the acquire loaded them, newest first: what the run's lookups
    /// read instead of the database. Complete when shorter than [`LOOKUP_LIMIT`].
    loaded: Option<Vec<TaskRow>>,
}

impl Tasks {
    pub fn new(partition_key: Partition) -> Self {
        Self { partition_key, writes: Vec::new(), loaded: None }
    }

    /// Over the rows an acquire loaded, newest first.
    pub fn loaded(partition_key: Partition, loaded: Vec<TaskRow>) -> Self {
        Self { partition_key, writes: Vec::new(), loaded: Some(loaded) }
    }

    /// The loaded rows, when they are the whole partition.
    fn snapshot(&self) -> Option<&[TaskRow]> {
        self.loaded.as_deref().filter(|rows| rows.len() < LOOKUP_LIMIT as usize)
    }

    /// The partition's newest rows, `limit` of them, as of the acquire.
    pub async fn recent(&self, limit: i64) -> Result<Vec<TaskRow>, StorageError> {
        if let Some(rows) = self.loaded.as_deref() {
            return Ok(rows.iter().take(limit as usize).cloned().collect());
        }
        self.partition_key.tasks(None, limit).await
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
        let working = self.partition_key.tasks(Some(TaskState::Working), LOOKUP_LIMIT).await?;
        Ok(working.into_iter().rev().collect())
    }

    /// The pending writes, in the order they were first made.
    pub fn writes(&self) -> &[TaskWrite] {
        &self.writes
    }

    /// The first loaded row that matches, else, when the loaded rows are not the whole partition, the
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
                .find(|row| row.task_id == task_id, self.partition_key.task(task_id))
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
            .partition_key
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
            None => {
                match self.find(|r| r.task_id == task_id, self.partition_key.task(task_id)).await? {
                    Some(row) if row.terminal() => return Ok(()),
                    row => row.map(TaskWrite::from),
                }
            }
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
#[path = "run_tests.rs"]
mod tests;
