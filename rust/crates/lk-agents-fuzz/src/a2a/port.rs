//! The fake storage port: one lane of a thread in memory, executing the A2A task machine's
//! operations with the storage contract's semantics (deduplication by `messageId`, a fenced lease
//! that lapses, conditional claims, an inbox drained by one runner, commits that are applied once
//! by run id) and injecting every typed error the real backends report.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::{InboxRow, TaskRow, TaskWrite};
use lk_agents_core::a2a::task::{Completion, Operation};

pub const LANE: &str = "front-desk";

/// How often an operation fails, in percent, while faults are on.
#[derive(Debug, Clone, Copy)]
pub struct Faults {
    pub unavailable: u8,
    pub lost_ack: u8,
    pub refused_commit: u8,
    pub lease_lost: u8,
    pub cancelled: u8,
}

/// The lane's lease: who holds it, under which run, until when.
#[derive(Debug, Clone)]
struct Held {
    worker: usize,
    until: Duration,
}

/// One request's open run.
#[derive(Debug, Clone)]
pub struct OpenRun {
    pub run_id: String,
    pub exclusive: bool,
    /// The tasks the turn staged as completed, and the messages it answered.
    pub staged: Vec<(String, String)>,
}

/// What the port observed, for the rules.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    /// Committed runs per `messageId` they answered.
    pub executions: BTreeMap<String, BTreeSet<String>>,
    /// Commits applied whose acknowledgment was lost, by task.
    pub applied_unacknowledged: BTreeSet<String>,
    /// A write that changed a terminal row.
    pub terminal_changes: Vec<String>,
    pub lost_acks: u64,
    pub injected: BTreeMap<&'static str, u64>,
}

#[derive(Debug)]
pub struct Port {
    pub tasks: BTreeMap<String, TaskRow>,
    inbox: Vec<InboxRow>,
    drained: i64,
    lease: Option<Held>,
    committed_runs: BTreeSet<String>,
    seq: i64,
    pub ttl: Duration,
    pub ledger: Ledger,
}

impl Port {
    pub fn new(ttl: Duration) -> Self {
        Self {
            tasks: BTreeMap::new(),
            inbox: Vec::new(),
            drained: 0,
            lease: None,
            committed_runs: BTreeSet::new(),
            seq: 0,
            ttl,
            ledger: Ledger::default(),
        }
    }

    pub fn pending_inbox(&self) -> usize {
        self.inbox.iter().filter(|r| r.id > self.drained).count()
    }

    /// Inbox rows past the drain whose task has not ended.
    pub fn undrained(&self) -> usize {
        self.inbox
            .iter()
            .filter(|r| r.id > self.drained)
            .filter(|r| self.tasks.get(&r.task_id).is_some_and(|t| !t.terminal()))
            .count()
    }

    fn held_by_other(&self, worker: usize, now: Duration) -> bool {
        self.lease.as_ref().is_some_and(|h| h.worker != worker && h.until > now)
    }

    fn take(&mut self, worker: usize, now: Duration) {
        self.lease = Some(Held { worker, until: now + self.ttl });
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

    fn set_status(&mut self, task_id: &str, status: TaskState, now: Duration) -> bool {
        match self.tasks.get_mut(task_id) {
            Some(row) if row.terminal() => {
                if row.status != status {
                    self.ledger
                        .terminal_changes
                        .push(format!("{task_id} {:?} -> {status:?}", row.status));
                }
                false
            }
            Some(row) => {
                row.status = status;
                row.updated_at = now.as_millis() as i64;
                true
            }
            None => false,
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

    /// Executes one operation for `worker`; `Err(None)` means the operation waits (a held lane
    /// under `wait`), to be retried at a later step.
    #[allow(clippy::too_many_lines, reason = "one arm per operation of the port")]
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
        let f = faults.unwrap_or(Faults {
            unavailable: 0,
            lost_ack: 0,
            refused_commit: 0,
            lease_lost: 0,
            cancelled: 0,
        });
        let reads = matches!(
            operation,
            Operation::FindReceipt(_)
                | Operation::FindTarget(_)
                | Operation::ReadTask(_)
                | Operation::ReadInbox
        );
        if (reads || matches!(operation, Operation::Submit(_) | Operation::Receive { .. }))
            && self.inject(rng, f.unavailable, "Unavailable")
        {
            return Some(Err(StorageError::Unavailable("the database is unreachable".into())));
        }
        Some(Ok(match operation {
            Operation::FindReceipt(_) if self.inject(rng, f.unavailable / 2, "NotFound") => {
                return Some(Err(StorageError::NotFound("the thread was deleted".into())));
            }
            Operation::FindReceipt(message_id) => Completion::Task(
                self.tasks.values().find(|r| r.message_id.as_deref() == Some(message_id)).cloned(),
            ),
            Operation::FindTarget(task_id) => Completion::Task(self.tasks.get(task_id).cloned()),
            Operation::ReadTask(task_id) => Completion::Task(self.tasks.get(task_id).cloned()),
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
                    return Some(Ok(Completion::Received {
                        duplicate: Some(existing.task_id.clone()),
                        leased: false,
                    }));
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
                let leased = !self.held_by_other(worker, now);
                if leased {
                    self.take(worker, now);
                    *kept = true;
                }
                Completion::Received { duplicate: None, leased }
            }
            Operation::Accepted(_) => Completion::Done,
            Operation::Open { run_id, task_id, exclusive, wait, pickup, resume, .. } => {
                if !exclusive {
                    let picked_up = *pickup
                        && self
                            .tasks
                            .get(task_id)
                            .is_some_and(|r| r.status == TaskState::Submitted);
                    *run = Some(OpenRun {
                        run_id: run_id.clone(),
                        exclusive: false,
                        staged: Vec::new(),
                    });
                    return Some(Ok(Completion::Opened { closed: false, picked_up }));
                }
                if self.committed_runs.contains(run_id) {
                    *run = Some(OpenRun {
                        run_id: run_id.clone(),
                        exclusive: false,
                        staged: Vec::new(),
                    });
                    return Some(Ok(Completion::Opened { closed: true, picked_up: false }));
                }
                let mine = *resume && *kept;
                if !mine && self.held_by_other(worker, now) {
                    if *wait && waited < wait_timeout {
                        return None;
                    }
                    *self.ledger.injected.entry("Rejected").or_default() += 1;
                    return Some(Err(StorageError::Rejected { running_task_id: None }));
                }
                self.take(worker, now);
                *kept = false;
                let picked_up = *pickup
                    && self.tasks.get(task_id).is_some_and(|r| !r.terminal())
                    && self.set_status(task_id, TaskState::Working, now);
                *run =
                    Some(OpenRun { run_id: run_id.clone(), exclusive: true, staged: Vec::new() });
                Completion::Opened { closed: false, picked_up }
            }
            Operation::ReadInbox => Completion::Inbox(
                self.inbox.iter().filter(|r| r.id > self.drained).cloned().collect(),
            ),
            Operation::Pickup(task_id) => {
                let was = self.tasks.get(task_id).filter(|r| !r.terminal()).map(|r| r.status);
                if was.is_some() {
                    self.set_status(task_id, TaskState::Working, now);
                }
                Completion::Picked(was)
            }
            Operation::FailTask { task_id, .. } => {
                if self.tasks.get(task_id).is_some_and(|r| !r.terminal()) {
                    self.set_status(task_id, TaskState::Failed, now);
                }
                Completion::Done
            }
            Operation::RecoverOrphans { skip } => {
                let orphans: Vec<String> = self
                    .tasks
                    .values()
                    .filter(|r| r.status == TaskState::Working && !skip.contains(&r.task_id))
                    .map(|r| r.task_id.clone())
                    .collect();
                for task_id in orphans {
                    self.set_status(&task_id, TaskState::Failed, now);
                }
                Completion::Done
            }
            Operation::Run { batch, task_ids, .. } => {
                if self.inject(rng, f.lease_lost, "LeaseLost") {
                    return Some(Err(StorageError::LeaseLost("renewal failed".into())));
                }
                if self.inject(rng, f.cancelled, "Cancelled") {
                    return Some(Err(StorageError::Cancelled("the host cancelled".into())));
                }
                if self.inject(rng, f.unavailable, "Unavailable") {
                    return Some(Err(StorageError::Unavailable(
                        "the handler lost the database".into(),
                    )));
                }
                let open = run.as_mut().expect("a run is open");
                open.staged = task_ids
                    .iter()
                    .zip(batch)
                    .map(|(task_id, message)| (task_id.clone(), message.message_id.clone()))
                    .collect();
                let answer_task = task_ids.last().cloned().unwrap_or_default();
                Completion::Ran { answer_task, output: Some(serde_json::json!("done")) }
            }
            Operation::Commit { answer_task, keep_lease, drained_id } => {
                let open = run.take().expect("a run is open");
                if open.exclusive && self.inject(rng, f.lease_lost, "LeaseLapsed") {
                    // the lease lapsed while the turn ran: another holder may take the lane
                    self.release(worker);
                }
                if open.exclusive && !self.lease.as_ref().is_some_and(|h| h.worker == worker) {
                    *self.ledger.injected.entry("VersionConflict").or_default() += 1;
                    return Some(Err(StorageError::VersionConflict {
                        lane: LANE.into(),
                        expected: 0,
                    }));
                }
                if self.inject(rng, f.refused_commit, "Invalid") {
                    self.release(worker);
                    return Some(Err(StorageError::Invalid("sqlite: disk I/O error".into())));
                }
                let lost_before = self.inject(rng, f.lost_ack, "UncertainBefore");
                if lost_before {
                    *run = Some(open);
                    self.ledger.lost_acks += 1;
                    return Some(Err(StorageError::Uncertain("lost before it applied".into())));
                }
                self.apply(worker, &open, *keep_lease, *drained_id, now);
                if self.inject(rng, f.lost_ack, "UncertainAfter") {
                    self.ledger.lost_acks += 1;
                    if let Some(task) = answer_task {
                        self.ledger.applied_unacknowledged.insert(task.clone());
                    }
                    *run = Some(OpenRun { staged: Vec::new(), ..open });
                    return Some(Err(StorageError::Uncertain("lost after it applied".into())));
                }
                *kept = *keep_lease;
                Completion::Committed {
                    task: answer_task.as_ref().and_then(|t| self.tasks.get(t).cloned()),
                    run_id: open.run_id,
                }
            }
            Operation::Abandon { .. } => {
                if run.take().is_some() {
                    self.release(worker);
                }
                Completion::Done
            }
            Operation::ResolveLater { task_id } => {
                // the resolver finds the run by its id: applied, nothing to do; else the
                // authoritative answer is a rejection and the task fails storage-unavailable
                let open = run.take().expect("a run is open");
                if !self.committed_runs.contains(&open.run_id)
                    && self.tasks.get(task_id).is_some_and(|r| !r.terminal())
                {
                    self.set_status(task_id, TaskState::Failed, now);
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
        }))
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
        for (task_id, message_id) in &open.staged {
            // a task a client canceled mid-turn keeps its terminal state
            if self.tasks.get(task_id).is_some_and(|r| !r.terminal())
                && self.set_status(task_id, TaskState::Completed, now)
            {
                self.ledger
                    .executions
                    .entry(message_id.clone())
                    .or_default()
                    .insert(open.run_id.clone());
            }
        }
        if let Some(id) = drained_id {
            self.drained = self.drained.max(id);
        }
        if keep_lease {
            self.renew(worker, now);
        } else {
            self.release(worker);
        }
    }
}

/// A user message `message_id` on `channel`.
pub fn wire(text: &str, message_id: &str, channel: &str) -> a2a::Message {
    let mut message = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text(text)]);
    message.message_id = message_id.into();
    message.metadata = Some(
        [(
            lk_agents_abi::a2a::metadata::CHANNEL_KEY.to_owned(),
            serde_json::Value::String(channel.into()),
        )]
        .into(),
    );
    message
}
