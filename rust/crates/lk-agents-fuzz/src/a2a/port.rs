//! The fake storage port: one lane of a thread in memory, executing the A2A task machine's
//! operations with the storage contract's semantics and injecting every typed error the real
//! backends report. What it reproduces of `lk-agents-storage`:
//!
//! - deduplication by `messageId`; claims only from `submitted` or `working` (a pickup), failures
//!   of `working` rows only (`fail_orphan`), a cancel of any row not yet terminal;
//! - the lease: taken when free, lapsed or this holder's, in arrival order (an older `submitted`
//!   row not in the inbox, younger than the wait, goes first), renewed by a live holder, a waiter
//!   mark set under `interrupt` while the lane is held and cleared once that waiter takes it;
//! - an exclusive run reads the rows its acquire loaded, a concurrent one reads them live; a
//!   staged write over a stored terminal row keeps that row, and the commit never overwrites one;
//! - one fenced commit applied once by run id, history changes as item upserts and deletions, a
//!   concurrent run refused for editing loaded history; a lost acknowledgment resolved by the run
//!   id, its commit sent again while its token still holds the lane.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::a2a::options::LEASE_WAIT_TIMEOUT;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::{InboxRow, ItemRow, TaskRow, TaskWrite, merge_write};
use lk_agents_abi::storage::thread::{REASON_STORAGE_UNAVAILABLE, REASON_WORKER_LOST};
use lk_agents_core::a2a::failed_message;
use lk_agents_core::a2a::task::{Completion, HandlerEvent, Operation};

use super::handler::Handler;

pub const LANE: &str = "front-desk";

/// How often an operation fails, in percent, while faults are on.
#[derive(Debug, Clone, Copy, Default)]
pub struct Faults {
    pub unavailable: u8,
    pub lost_ack: u8,
    pub refused_commit: u8,
    pub lease_lost: u8,
    pub cancelled: u8,
}

/// The lane's lease: who holds it, under which token, until when.
#[derive(Debug, Clone)]
struct Held {
    worker: usize,
    token: u64,
    until: Duration,
}

/// One request's open run.
#[derive(Debug, Clone)]
pub struct OpenRun {
    pub run_id: String,
    pub exclusive: bool,
    /// The lease token the run was opened under.
    token: u64,
    /// The rows the acquire loaded, newest first: what an exclusive run's reads see.
    loaded_tasks: Option<Vec<TaskRow>>,
    /// The history as loaded, and as the turn changed it.
    loaded: Vec<ItemRow>,
    history: Vec<ItemRow>,
    writes: Vec<TaskWrite>,
    /// The messages of the turn whose handler ran.
    handled: Vec<String>,
    handler: Option<Handler>,
    /// The commit's `keep_lease` and `drained_id`, kept for its resend once uncertain.
    sent: (bool, Option<i64>),
}

/// What the port observed, for the rules.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    /// Committed runs per `messageId` whose turn they ran.
    pub executions: BTreeMap<String, BTreeSet<String>>,
    /// Commits applied whose acknowledgment was lost: the message's task, as the commit staged it.
    pub applied_unacknowledged: BTreeMap<String, Option<TaskState>>,
    /// A write that changed a terminal row.
    pub terminal_changes: Vec<String>,
    /// Every applied run's terminal task writes: `(run_id, exclusive, task_id, status)`.
    pub staged_terminal: Vec<(String, bool, String, TaskState)>,
    /// Writes storage refused for what the run staged, not for an injected fault:
    /// `(exclusive, why)`.
    pub refused: Vec<(bool, String)>,
    /// The most questions pending at once after a commit, and which.
    pub most_pending: Vec<String>,
    pub lost_acks: u64,
    pub injected: BTreeMap<&'static str, u64>,
}

#[derive(Debug)]
pub struct Port {
    pub tasks: BTreeMap<String, TaskRow>,
    items: Vec<ItemRow>,
    inbox: Vec<InboxRow>,
    drained: i64,
    lease: Option<Held>,
    tokens: u64,
    waiter: Option<usize>,
    committed_runs: BTreeSet<String>,
    seq: i64,
    pub ttl: Duration,
    pub ledger: Ledger,
}

/// A row a pickup can claim.
fn runnable(row: &TaskRow) -> bool {
    matches!(row.status, TaskState::Submitted | TaskState::Working)
}

fn pending_question(row: &TaskRow) -> bool {
    row.status == TaskState::InputRequired && row.internal.as_ref().is_some_and(|i| i.ask.is_some())
}

impl Port {
    pub fn new(ttl: Duration) -> Self {
        Self {
            tasks: BTreeMap::new(),
            items: Vec::new(),
            inbox: Vec::new(),
            drained: 0,
            lease: None,
            tokens: 0,
            waiter: None,
            committed_runs: BTreeSet::new(),
            seq: 0,
            ttl,
            ledger: Ledger::default(),
        }
    }

    pub fn pending_inbox(&self) -> usize {
        self.inbox.iter().filter(|r| r.id > self.drained).count()
    }

    /// The messages of inbox rows whose task no turn answered (not ended, not asking).
    pub fn unanswered_receipts(&self) -> Vec<&str> {
        self.inbox
            .iter()
            .filter(|r| {
                self.tasks
                    .get(&r.task_id)
                    .is_some_and(|t| !t.terminal() && t.status != TaskState::InputRequired)
            })
            .map(|r| r.message_id.as_str())
            .collect()
    }

    fn held_by_other(&self, worker: usize, now: Duration) -> bool {
        self.lease.as_ref().is_some_and(|h| h.worker != worker && h.until > now)
    }

    fn holds(&self, worker: usize, token: u64) -> bool {
        self.lease.as_ref().is_some_and(|h| h.worker == worker && h.token == token)
    }

    /// `TakeLease`: free, lapsed or this worker's, unless an older message still waits its turn.
    fn may_take(&self, worker: usize, task_id: &str, now: Duration) -> bool {
        if self.held_by_other(worker, now) {
            return false;
        }
        let Some(seq) = self.tasks.get(task_id).map(|r| r.seq) else { return true };
        let since = now.as_millis() as i64 - LEASE_WAIT_TIMEOUT.as_millis() as i64;
        !self.tasks.values().any(|older| {
            older.status == TaskState::Submitted
                && older.message_id.is_some()
                && older.updated_at > since
                && !self.inbox.iter().any(|i| i.task_id == older.task_id)
                && older.seq < seq
        })
    }

    fn take(&mut self, worker: usize, now: Duration) -> u64 {
        self.tokens += 1;
        self.lease = Some(Held { worker, token: self.tokens, until: now + self.ttl });
        if self.waiter == Some(worker) {
            self.waiter = None;
        }
        self.tokens
    }

    /// A live holder renews; a dead one lets the lease lapse.
    pub fn renew(&mut self, worker: usize, now: Duration) {
        if let Some(h) = &mut self.lease
            && h.worker == worker
        {
            h.until = now + self.ttl;
        }
    }

    fn release(&mut self, worker: usize) {
        if self.lease.as_ref().is_some_and(|h| h.worker == worker) {
            self.lease = None;
        }
    }

    fn insert(&mut self, write: &TaskWrite, now: Duration) {
        self.seq += 1;
        let mut row = write.row(LANE);
        row.seq = self.seq;
        row.updated_at = now.as_millis() as i64;
        self.tasks.insert(row.task_id.clone(), row);
    }

    fn set_status(&mut self, task_id: &str, status: TaskState, now: Duration) {
        if let Some(row) = self.tasks.get_mut(task_id) {
            if row.terminal() {
                self.ledger
                    .terminal_changes
                    .push(format!("{task_id} {:?} -> {status:?}", row.status));
                return;
            }
            row.status = status;
            row.updated_at = now.as_millis() as i64;
        }
    }

    /// `pickup`: a `submitted` or `working` row goes `working`; its status before.
    fn pickup(&mut self, task_id: &str, now: Duration) -> Option<TaskState> {
        let was = self.tasks.get(task_id).filter(|r| runnable(r)).map(|r| r.status);
        if was.is_some() {
            self.set_status(task_id, TaskState::Working, now);
        }
        was
    }

    /// `fail_orphan`: only a `working` row fails, its status message naming `reason`.
    fn fail_working(&mut self, task_id: &str, reason: &str, now: Duration) {
        if let Some(row) = self.tasks.get_mut(task_id)
            && row.status == TaskState::Working
        {
            row.status = TaskState::Failed;
            row.updated_at = now.as_millis() as i64;
            let id = format!("msg_fail_{}", row.seq);
            row.data.status_message = Some(failed_message(task_id, reason, id));
        }
    }

    /// A client's `tasks/cancel`.
    pub fn cancel(&mut self, task_id: &str, now: Duration) {
        if self.tasks.get(task_id).is_some_and(|r| !r.terminal()) {
            self.set_status(task_id, TaskState::Canceled, now);
        }
    }

    fn inject(&mut self, rng: &mut Rng, pct: u8, kind: &'static str) -> bool {
        let hit = pct > 0 && rng.u8(0..100) < pct;
        if hit {
            *self.ledger.injected.entry(kind).or_default() += 1;
        }
        hit
    }

    /// A row as `run` reads it: an exclusive run its acquire's rows, a concurrent one live.
    fn read(&self, run: &OpenRun, task_id: &str) -> Option<TaskRow> {
        match &run.loaded_tasks {
            Some(rows) => rows.iter().find(|r| r.task_id == task_id).cloned(),
            None => self.tasks.get(task_id).cloned(),
        }
    }

    /// Executes one operation for `worker`; `None` means the operation waits (a held lane under
    /// `wait`), to be retried at a later step.
    #[allow(clippy::too_many_arguments, reason = "the request's live resources and the faults")]
    pub fn execute(
        &mut self,
        worker: usize,
        run: &mut Option<OpenRun>,
        kept: &mut bool,
        operation: &Operation,
        waited: Duration,
        wait_timeout: Duration,
        now: Duration,
        faults: Option<Faults>,
        rng: &mut Rng,
    ) -> Option<Result<Completion, StorageError>> {
        let f = faults.unwrap_or_default();
        let reads = matches!(
            operation,
            Operation::FindReceipt(_)
                | Operation::FindTarget(_)
                | Operation::ReadTask(_)
                | Operation::ReadTasks(_)
                | Operation::ReadInbox
                | Operation::ReadPending(_)
                | Operation::Checkpoint(_)
        );
        if (reads || matches!(operation, Operation::Submit(_) | Operation::Receive { .. }))
            && self.inject(rng, f.unavailable, "Unavailable")
        {
            return Some(Err(StorageError::Unavailable("the database is unreachable".into())));
        }
        if let Operation::Open { .. } = operation {
            return self.open(worker, run, kept, operation, waited, wait_timeout, now);
        }
        if let Some(open) = run.as_mut()
            && let Some(result) = Self::turn(open, operation, f, rng, &mut self.ledger)
        {
            return Some(result);
        }
        Some(self.step(worker, run, kept, operation, now, f, rng))
    }

    #[allow(clippy::too_many_arguments, reason = "the request's live resources")]
    fn open(
        &mut self,
        worker: usize,
        run: &mut Option<OpenRun>,
        kept: &mut bool,
        operation: &Operation,
        waited: Duration,
        wait_timeout: Duration,
        now: Duration,
    ) -> Option<Result<Completion, StorageError>> {
        let Operation::Open { run_id, task_id, exclusive, wait, interrupt, pickup, resume, .. } =
            operation
        else {
            unreachable!("an open")
        };
        let history_end = self.items.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let items = self.items.clone();
        let open = |exclusive, token, loaded_tasks| OpenRun {
            run_id: run_id.clone(),
            exclusive,
            token,
            loaded_tasks,
            loaded: items.clone(),
            history: items.clone(),
            writes: Vec::new(),
            handled: Vec::new(),
            handler: None,
            sent: (false, None),
        };
        if !exclusive {
            // no lease, nothing claimed: the run goes only while its row is still `submitted`
            let picked_up = *pickup
                && self.tasks.get(task_id).is_some_and(|r| r.status == TaskState::Submitted);
            *run = Some(open(false, 0, None));
            return Some(Ok(Completion::Opened { closed: false, picked_up, history_end }));
        }
        if self.committed_runs.contains(run_id) {
            *run = Some(open(false, 0, None));
            return Some(Ok(Completion::Opened { closed: true, picked_up: false, history_end }));
        }
        let mine = *resume && *kept && self.lease.as_ref().is_some_and(|h| h.worker == worker);
        if !mine && !self.may_take(worker, task_id, now) {
            if *wait && waited < wait_timeout {
                if *interrupt && self.held_by_other(worker, now) {
                    self.waiter = Some(worker);
                }
                return None;
            }
            *self.ledger.injected.entry("Rejected").or_default() += 1;
            return Some(Err(StorageError::Rejected { running_task_id: None }));
        }
        let token =
            if mine { self.lease.as_ref().map_or(0, |h| h.token) } else { self.take(worker, now) };
        *kept = false;
        let picked_up = *pickup && self.pickup(task_id, now).is_some();
        let mut loaded: Vec<TaskRow> = self.tasks.values().cloned().collect();
        loaded.sort_by_key(|r| std::cmp::Reverse(r.seq));
        *run = Some(open(true, token, Some(loaded)));
        Some(Ok(Completion::Opened { closed: false, picked_up, history_end }))
    }

    /// The turn's operations on the open run: its reads, staging and the handler.
    fn turn(
        open: &mut OpenRun,
        operation: &Operation,
        f: Faults,
        rng: &mut Rng,
        ledger: &mut Ledger,
    ) -> Option<Result<Completion, StorageError>> {
        let mut inject = |pct: u8, kind: &'static str| {
            let hit = pct > 0 && rng.u8(0..100) < pct;
            if hit {
                *ledger.injected.entry(kind).or_default() += 1;
            }
            hit
        };
        Some(match operation {
            Operation::Handle(handle) => {
                open.handled = handle.batch.iter().map(|m| m.message_id.clone()).collect();
                open.handler = Some(Handler::new(handle, &open.run_id, rng));
                Ok(Completion::Done)
            }
            Operation::Next(verdict) => {
                if inject(f.lease_lost, "LeaseLost") {
                    return Some(Err(StorageError::LeaseLost("renewal failed".into())));
                }
                if inject(f.cancelled, "Cancelled") {
                    return Some(Err(StorageError::Cancelled("the host cancelled".into())));
                }
                if inject(f.unavailable, "HandlerUnavailable") {
                    return Some(Err(StorageError::Unavailable(
                        "the handler lost the database".into(),
                    )));
                }
                let handler = open.handler.as_mut().expect("a handler is open");
                let event = handler.next(*verdict);
                if let HandlerEvent::Output(out) = &event
                    && out.input_required.is_some()
                {
                    *ledger.injected.entry("Question").or_default() += 1;
                }
                Ok(Completion::Event(event))
            }
            _ => return None,
        })
    }

    #[allow(clippy::too_many_lines, reason = "one arm per storage operation")]
    #[allow(clippy::too_many_arguments, reason = "the request's live resources and the faults")]
    fn step(
        &mut self,
        worker: usize,
        run: &mut Option<OpenRun>,
        kept: &mut bool,
        operation: &Operation,
        now: Duration,
        f: Faults,
        rng: &mut Rng,
    ) -> Result<Completion, StorageError> {
        Ok(match operation {
            Operation::FindReceipt(_) if self.inject(rng, f.unavailable / 2, "NotFound") => {
                return Err(StorageError::NotFound("the thread was deleted".into()));
            }
            Operation::FindReceipt(message_id) => Completion::Task(
                self.tasks.values().find(|r| r.message_id.as_deref() == Some(message_id)).cloned(),
            ),
            Operation::FindTarget(task_id) | Operation::ReadTask(task_id) => {
                Completion::Task(self.tasks.get(task_id).cloned())
            }
            Operation::ReadTasks(task_ids) => Completion::Tasks(
                task_ids.iter().filter_map(|t| self.tasks.get(t).cloned()).collect(),
            ),
            Operation::Submit(write) => {
                let exists = self.tasks.contains_key(&write.task_id)
                    || self.tasks.values().any(|r| r.message_id == write.message_id);
                if !exists {
                    self.insert(write, now);
                }
                Completion::Submitted(!exists)
            }
            Operation::Receive { message, task } => {
                if let Some(existing) =
                    self.tasks.values().find(|r| r.message_id == task.message_id)
                {
                    return Ok(Completion::Received {
                        duplicate: Some(existing.task_id.clone()),
                        leased: false,
                    });
                }
                self.insert(task, now);
                let id = self.inbox.last().map_or(1, |r| r.id + 1);
                self.inbox.push(InboxRow {
                    id,
                    lane: LANE.into(),
                    task_id: task.task_id.clone(),
                    message_id: message.message_id.clone(),
                    message: message.clone(),
                    received_at: now.as_secs() as i64,
                });
                // `TakeLeaseForMessage`: no arrival order, the lane only free or lapsed
                let leased = !self.held_by_other(worker, now);
                if leased {
                    self.take(worker, now);
                    *kept = true;
                }
                Completion::Received { duplicate: None, leased }
            }
            Operation::Accepted(_) => Completion::Done,
            Operation::ReadInbox => Completion::Inbox(
                self.inbox.iter().filter(|r| r.id > self.drained).cloned().collect(),
            ),
            Operation::Pickup(task_id) => Completion::Picked(self.pickup(task_id, now)),
            Operation::FailTask { task_id, reason } => {
                if self.pickup(task_id, now).is_some() {
                    self.fail_working(task_id, reason, now);
                }
                Completion::Done
            }
            Operation::RecoverOrphans { skip } => {
                let open = run.as_ref().expect("a run is open");
                let working: Vec<String> = open
                    .loaded_tasks
                    .iter()
                    .flatten()
                    .filter(|r| r.status == TaskState::Working && !skip.contains(&r.task_id))
                    .map(|r| r.task_id.clone())
                    .collect();
                for task_id in working {
                    self.fail_working(&task_id, REASON_WORKER_LOST, now);
                }
                Completion::Done
            }
            Operation::ReadPending(named) => {
                let open = run.as_ref().expect("a run is open");
                Completion::Task(match named {
                    Some(task_id) => self.read(open, task_id).filter(pending_question),
                    None => match &open.loaded_tasks {
                        Some(rows) => rows.iter().find(|r| pending_question(r)).cloned(),
                        None => self
                            .tasks
                            .values()
                            .filter(|r| pending_question(r))
                            .max_by_key(|r| r.seq)
                            .cloned(),
                    },
                })
            }
            Operation::Stage(staged) => {
                let open = run.as_mut().expect("a run is open");
                for edit in staged.edits.iter().cloned() {
                    edit.apply(&mut open.history);
                }
                for t in &staged.transitions {
                    let base = match open.writes.iter().find(|w| w.task_id == t.task_id) {
                        Some(w) => Some(w.clone()),
                        None => match open
                            .loaded_tasks
                            .as_ref()
                            .and_then(|rows| rows.iter().find(|r| r.task_id == t.task_id))
                            .or_else(|| self.tasks.get(&t.task_id))
                        {
                            // a stored terminal row stays as it is
                            Some(row) if row.terminal() => continue,
                            row => row.cloned().map(TaskWrite::from),
                        },
                    };
                    let write = merge_write(&t.task_id, base.as_ref(), t.status, t.patch.clone())
                        .inspect_err(|e| {
                        self.ledger.refused.push((open.exclusive, format!("{}: {e}", open.run_id)));
                    })?;
                    match open.writes.iter().position(|w| w.task_id == t.task_id) {
                        Some(i) => open.writes[i] = write,
                        None => open.writes.push(write),
                    }
                }
                Completion::Done
            }
            Operation::Checkpoint(task_ids) => Completion::Checkpoint {
                other_waiting: self.waiter.is_some_and(|w| w != worker),
                canceled: task_ids
                    .iter()
                    .any(|t| self.tasks.get(t).is_some_and(|r| r.status == TaskState::Canceled)),
            },
            Operation::Commit { answer_task, keep_lease, drained_id } => {
                let open = run.take().expect("a run is open");
                if open.exclusive && self.inject(rng, f.lease_lost, "LeaseLapsed") {
                    // the lease lapsed while the turn ran and another holder took the lane
                    self.release(worker);
                }
                if open.exclusive && !self.holds(worker, open.token) {
                    *self.ledger.injected.entry("VersionConflict").or_default() += 1;
                    return Err(StorageError::VersionConflict { lane: LANE.into(), expected: 0 });
                }
                if !open.exclusive
                    && let Some(id) = Self::edited(&open)
                {
                    let why =
                        format!("a concurrent run only appends; {} edited item {id}", open.run_id);
                    self.ledger.refused.push((false, why.clone()));
                    return Err(StorageError::Invalid(why));
                }
                if self.inject(rng, f.refused_commit, "Refused") {
                    self.release(worker);
                    return Err(StorageError::Invalid("sqlite: disk I/O error".into()));
                }
                if self.inject(rng, f.lost_ack, "UncertainBefore") {
                    self.ledger.lost_acks += 1;
                    *run = Some(OpenRun { sent: (*keep_lease, *drained_id), ..open });
                    return Err(StorageError::Uncertain("lost before it applied".into()));
                }
                self.apply(worker, &open, *keep_lease, *drained_id, now);
                if self.inject(rng, f.lost_ack, "UncertainAfter") {
                    self.ledger.lost_acks += 1;
                    if let Some(message) = open.handled.last()
                        && let Some(row) =
                            self.tasks.values().find(|r| r.message_id.as_deref() == Some(message))
                    {
                        let staged = open.writes.iter().find(|w| w.task_id == row.task_id);
                        let staged = staged.map(|w| w.status);
                        self.ledger.applied_unacknowledged.insert(row.task_id.clone(), staged);
                    }
                    *run = Some(open);
                    return Err(StorageError::Uncertain("lost after it applied".into()));
                }
                *kept = *keep_lease;
                Completion::Committed {
                    task: answer_task.as_ref().and_then(|t| self.tasks.get(t).cloned()),
                    run_id: open.run_id,
                }
            }
            Operation::Abandon { .. } => {
                if let Some(open) = run.take()
                    && open.exclusive
                {
                    self.release(worker);
                }
                Completion::Done
            }
            Operation::ResolveLater { task_id } => {
                // the resolver finds the run by its id: applied, nothing to do; not applied while
                // its token still holds the lane, sent again; else rejected, and the message's
                // task fails storage-unavailable (`fail_orphan`: only while `working`)
                let open = run.take().expect("a run is open");
                if !self.committed_runs.contains(&open.run_id) {
                    if !open.exclusive || self.holds(worker, open.token) {
                        let (keep_lease, drained_id) = open.sent;
                        self.apply(worker, &open, keep_lease, drained_id, now);
                    } else {
                        self.fail_working(task_id, REASON_STORAGE_UNAVAILABLE, now);
                    }
                }
                self.release(worker);
                Completion::Done
            }
            Operation::CloseRun => {
                if let Some(open) = run.take() {
                    self.apply(worker, &open, false, None, now);
                }
                Completion::Done
            }
            Operation::ReleaseDrained => {
                let released = self.pending_inbox() == 0;
                if released {
                    self.release(worker);
                    *kept = false;
                }
                Completion::Released(released)
            }
            Operation::RunningTask => Completion::RunningTask(None),
            Operation::Open { .. } | Operation::Handle(_) | Operation::Next(_) => {
                unreachable!("executed elsewhere")
            }
        })
    }

    /// The id of a loaded item the run changed or removed.
    fn edited(open: &OpenRun) -> Option<String> {
        open.loaded.iter().find_map(|loaded| {
            match open.history.iter().find(|i| i.id == loaded.id) {
                Some(now) if now.payload == loaded.payload => None,
                _ => Some(loaded.id.clone()),
            }
        })
    }

    fn apply(
        &mut self,
        worker: usize,
        open: &OpenRun,
        keep_lease: bool,
        drained_id: Option<i64>,
        now: Duration,
    ) {
        if !self.committed_runs.insert(open.run_id.clone()) {
            return;
        }
        // the history: loaded items the run removed go, changed and new ones are upserted
        self.items.retain(|i| {
            !open.loaded.iter().any(|l| l.id == i.id) || open.history.iter().any(|h| h.id == i.id)
        });
        for item in &open.history {
            match self.items.iter_mut().find(|i| i.id == item.id) {
                Some(stored) => *stored = item.clone(),
                None => self.items.push(item.clone()),
            }
        }
        for write in &open.writes {
            if write.status.is_terminal() {
                self.ledger.staged_terminal.push((
                    open.run_id.clone(),
                    open.exclusive,
                    write.task_id.clone(),
                    write.status,
                ));
            }
            match self.tasks.get_mut(&write.task_id) {
                // `UpsertTask`: a terminal row stays as it is
                Some(row) if row.terminal() => {}
                Some(row) => {
                    let seq = row.seq;
                    *row = TaskRow { seq, updated_at: now.as_millis() as i64, ..write.row(LANE) };
                }
                None => self.insert(write, now),
            }
        }
        for message in &open.handled {
            self.ledger.executions.entry(message.clone()).or_default().insert(open.run_id.clone());
        }
        if let Some(id) = drained_id {
            self.drained = self.drained.max(id);
        }
        if open.exclusive {
            if keep_lease {
                self.renew(worker, now);
            } else {
                self.release(worker);
            }
        }
        let pending: Vec<String> = self
            .tasks
            .values()
            .filter(|r| pending_question(r))
            .map(|r| r.task_id.clone())
            .collect();
        if pending.len() > self.ledger.most_pending.len() {
            self.ledger.most_pending = pending;
        }
    }
}

/// A user message `message_id` on `channel`, continuing `task_id` when given.
pub fn wire(text: &str, message_id: &str, channel: &str, task_id: Option<&str>) -> a2a::Message {
    let mut message = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text(text)]);
    message.message_id = message_id.into();
    message.task_id = task_id.map(str::to_owned);
    message.metadata = Some(
        [(
            lk_agents_abi::a2a::metadata::CHANNEL_KEY.to_owned(),
            serde_json::Value::String(channel.into()),
        )]
        .into(),
    );
    message
}
