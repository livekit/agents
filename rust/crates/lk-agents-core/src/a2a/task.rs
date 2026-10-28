//! Arrival and inbox orchestration. Storage owns atomic admission and fencing; this machine
//! decides which operation follows each outcome. The runtime retains runs, leases and streams.

#[cfg(test)]
mod tests;

use std::time::Duration;

use lk_agents_abi::a2a_protocol as a2a;

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::StreamInfo;
use lk_agents_abi::storage::TaskInternal;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::LanePoll;
use lk_agents_abi::storage::thread::TaskWrite;
use lk_agents_abi::storage::thread::{InboxRow, TaskData};
use lk_agents_abi::storage::thread::{REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE, TaskRow};

use crate::a2a::{Message, Policy, TurnOptions, TurnResult, resolve_debounce, resolve_policy};

/// One completion belongs to exactly one issued storage/handler operation.
#[derive(Debug)]
pub enum Event {
    Start,
    Wake,
    Completed { id: u64, result: Result<Completion, StorageError> },
}

#[derive(Debug)]
pub enum Effect {
    Execute { id: u64, operation: Operation },
    Finished(Result<TurnResult, StorageError>),
}

/// Operations at the storage run, durable inbox and handler boundaries.
#[derive(Debug)]
pub enum Operation {
    FindReceipt(String),
    FindTarget(String),
    Submit(TaskWrite),
    Receive {
        message: a2a::Message,
        task: TaskWrite,
    },
    Accepted(String),
    Open {
        run_id: String,
        task_id: String,
        exclusive: bool,
        wait: bool,
        interrupt: bool,
        pickup: bool,
        resume: bool,
    },
    ReadTask(String),
    ReadInbox,
    Pickup(String),
    FailOrphan {
        task_id: String,
        reason: &'static str,
    },
    RecoverOrphans {
        skip: Vec<String>,
    },
    CloseRun,
    Run {
        batch: Vec<Message>,
        task_ids: Vec<String>,
        interrupt: bool,
        drain: bool,
    },
    CommitDrain {
        through: i64,
    },
    ReleaseDrained,
    RunningTask,
}

#[derive(Debug)]
pub enum Completion {
    Task(Option<TaskRow>),
    Submitted(bool),
    Received { duplicate: Option<String>, leased: bool },
    Opened { closed: bool, picked_up: bool },
    Inbox(Vec<InboxRow>),
    Picked(Option<TaskState>),
    Turn(TurnResult),
    DrainCommitted { run_id: String },
    Released(bool),
    RunningTask(Option<String>),
    Done,
}

#[derive(Debug, Clone, Copy)]
enum Next {
    Open,
    Drain,
    Pickup,
}

#[derive(Debug)]
enum Phase {
    New,
    Receipt,
    Target,
    Submit,
    Receive,
    Accepted(Next),
    Debounce { at: Duration, next: Next },
    Open,
    RejectSubmit,
    DuplicateClose,
    Pickup,
    Recover,
    Run,
    Observe,
    Poll { at: Duration },
    Inbox,
    DrainOpen,
    DrainPickup,
    DrainOrphan,
    DrainRecover,
    DrainRun,
    DrainCommit,
    Release,
    RecoveryOpen,
    RecoveryOrphans,
    RecoveryClose,
    Close,
    Failed,
    Refused,
    Finished,
}

/// One durable arrival. IDs are supplied once; drain run IDs use a private seeded counter.
/// All live resources and the input queue belong to the executor.
#[derive(Debug)]
pub struct TaskMachine {
    thread_id: String,
    lane: String,
    message: Message,
    task_id: String,
    stream: Option<StreamInfo>,
    policy: Policy,
    debounce: Duration,
    wait_timeout: Duration,
    phase: Phase,
    sequence: u64,
    pending: Option<u64>,
    run_sequence: u64,
    observed: Option<TaskRow>,
    result: Option<TurnResult>,
    poll: Option<LanePoll>,
    leased: bool,
    rows: Vec<InboxRow>,
    index: usize,
    batch: Vec<Message>,
    task_ids: Vec<String>,
    draining: bool,
    last_run_id: Option<String>,
}

impl TaskMachine {
    pub fn new(
        thread_id: String,
        lane: String,
        message: Message,
        task_id: String,
        stream: Option<StreamInfo>,
        options: &TurnOptions,
    ) -> Self {
        let policy = resolve_policy(&options.policy, &message.channel);
        Self {
            thread_id,
            lane,
            debounce: resolve_debounce(&options.debounce, &message.channel, policy),
            message,
            task_id,
            stream,
            policy,
            wait_timeout: options.wait_timeout(),
            phase: Phase::New,
            sequence: 0,
            pending: None,
            run_sequence: 0,
            observed: None,
            result: None,
            poll: None,
            leased: false,
            rows: Vec::new(),
            index: 0,
            batch: Vec::new(),
            task_ids: Vec::new(),
            draining: false,
            last_run_id: None,
        }
    }

    pub fn wake_at(&self) -> Option<Duration> {
        match self.phase {
            Phase::Debounce { at, .. } | Phase::Poll { at } => Some(at),
            _ => None,
        }
    }

    pub fn handle(&mut self, event: Event, now: Duration) -> Vec<Effect> {
        let operation = match event {
            Event::Start if matches!(self.phase, Phase::New) => {
                self.phase = Phase::Receipt;
                Some(Operation::FindReceipt(self.message.message_id.clone()))
            }
            Event::Wake => match self.phase {
                Phase::Debounce { at, next } if now >= at => Some(self.next(next)),
                Phase::Poll { at } if now >= at => {
                    self.phase = Phase::Observe;
                    Some(Operation::ReadTask(self.task_id.clone()))
                }
                _ => None,
            },
            Event::Completed { id, result } if self.pending == Some(id) => {
                self.pending = None;
                match result {
                    Ok(result) => match self.complete(result, now) {
                        Ok(operation) => operation,
                        Err(error) => return self.finish(Err(error)),
                    },
                    Err(error) => return self.error(error, now),
                }
            }
            _ => None,
        };
        if matches!(self.phase, Phase::Finished) && self.result.is_some() {
            let result = self.result.take().expect("finished result");
            return self.finish(Ok(result));
        }
        operation.map_or_else(Vec::new, |operation| self.issue(operation))
    }

    fn issue(&mut self, operation: Operation) -> Vec<Effect> {
        self.sequence += 1;
        self.pending = Some(self.sequence);
        vec![Effect::Execute { id: self.sequence, operation }]
    }

    fn finish(&mut self, result: Result<TurnResult, StorageError>) -> Vec<Effect> {
        self.phase = Phase::Finished;
        self.pending = None;
        vec![Effect::Finished(result)]
    }

    fn answer(&mut self, row: TaskRow) -> Operation {
        self.observed = Some(row);
        self.phase = Phase::Close;
        Operation::CloseRun
    }

    fn submitted(&self) -> TaskWrite {
        TaskWrite {
            task_id: self.task_id.clone(),
            status: TaskState::Submitted,
            message_id: Some(self.message.message_id.clone()),
            name: None,
            data: TaskData { history: vec![self.message.to_a2a()], ..Default::default() },
            internal: self.stream.as_ref().map(|stream| TaskInternal {
                ask: None,
                streams: vec![stream.clone()],
                ..Default::default()
            }),
        }
    }

    fn next(&mut self, next: Next) -> Operation {
        match next {
            Next::Drain => {
                self.phase = Phase::Inbox;
                Operation::ReadInbox
            }
            Next::Pickup => {
                self.phase = Phase::Pickup;
                Operation::Pickup(self.task_id.clone())
            }
            Next::Open => {
                self.phase = Phase::Open;
                Operation::Open {
                    run_id: self.task_id.clone(),
                    task_id: self.task_id.clone(),
                    exclusive: self.policy != Policy::Concurrent,
                    wait: self.policy != Policy::Reject,
                    interrupt: self.policy == Policy::Interrupt,
                    pickup: self.policy != Policy::Reject,
                    resume: false,
                }
            }
        }
    }

    fn debounce(&mut self, next: Next, now: Duration) -> Option<Operation> {
        if self.debounce.is_zero() {
            Some(self.next(next))
        } else {
            self.phase = Phase::Debounce { at: now + self.debounce, next };
            None
        }
    }

    fn observe(&mut self) -> Operation {
        self.phase = Phase::Observe;
        Operation::ReadTask(self.task_id.clone())
    }

    fn poll(&mut self, now: Duration) {
        // A durable inbox receipt is never abandoned because a polling budget elapsed.
        let poll = self.poll.get_or_insert_with(|| LanePoll::new(now, self.wait_timeout));
        let at = match poll.next(now) {
            Some(at) => at,
            None => {
                *poll = LanePoll::new(now, self.wait_timeout);
                now + Duration::from_secs(1)
            }
        };
        self.phase = Phase::Poll { at };
    }

    fn drain_pickup(&mut self) -> Operation {
        if self.index < self.rows.len() {
            self.phase = Phase::DrainPickup;
            Operation::Pickup(self.rows[self.index].task_id.clone())
        } else if self.batch.is_empty() {
            self.phase = Phase::DrainCommit;
            Operation::CommitDrain { through: self.rows.last().expect("a drain has rows").id }
        } else {
            self.phase = Phase::DrainRecover;
            Operation::RecoverOrphans {
                skip: self.rows.iter().map(|r| r.task_id.clone()).collect(),
            }
        }
    }

    fn complete(
        &mut self,
        result: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, StorageError> {
        let operation = match (&self.phase, result) {
            (Phase::Receipt, Completion::Task(Some(row))) => {
                self.task_id = row.task_id.clone();
                if !matches!(row.status, TaskState::Submitted | TaskState::Working) {
                    self.answer(row)
                } else {
                    self.observe()
                }
            }
            (Phase::Receipt, Completion::Task(None)) => {
                if let Some(target) = &self.message.task_id {
                    self.phase = Phase::Target;
                    Operation::FindTarget(target.clone())
                } else {
                    self.arrive()
                }
            }
            (Phase::Target, Completion::Task(Some(row))) if row.lane == self.lane => {
                if row.status.is_terminal() {
                    return Err(StorageError::Invalid(
                        "a follow-up to a terminal task must start a new task".into(),
                    ));
                }
                self.arrive()
            }
            (Phase::Target, Completion::Task(_)) => {
                return Err(StorageError::NotFound("the addressed task is not in this lane".into()));
            }
            (Phase::RejectSubmit, Completion::Submitted(false)) => {
                self.phase = Phase::DuplicateClose;
                Operation::CloseRun
            }
            (Phase::DuplicateClose, Completion::Done)
            | (Phase::Submit, Completion::Submitted(false)) => {
                self.phase = Phase::Receipt;
                Operation::FindReceipt(self.message.message_id.clone())
            }
            (Phase::Submit, Completion::Submitted(true)) => {
                self.phase = Phase::Accepted(Next::Open);
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::RejectSubmit, Completion::Submitted(true)) => {
                self.phase = Phase::Accepted(Next::Pickup);
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::Receive, Completion::Received { duplicate: Some(task), .. }) => {
                self.task_id = task;
                self.observe()
            }
            (Phase::Receive, Completion::Received { duplicate: None, leased }) => {
                self.leased = leased;
                self.phase = Phase::Accepted(if leased { Next::Drain } else { Next::Open });
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::Accepted(next), Completion::Done) => {
                let next = *next;
                if self.policy == Policy::Batch && !self.leased {
                    self.observe()
                } else {
                    return Ok(self.debounce(next, now));
                }
            }
            (Phase::Open, Completion::Opened { closed, picked_up }) => {
                if closed {
                    self.phase = Phase::RecoveryClose;
                    Operation::CloseRun
                } else if self.policy == Policy::Reject {
                    self.phase = Phase::RejectSubmit;
                    Operation::Submit(self.submitted())
                } else if !picked_up {
                    self.phase = Phase::RecoveryClose;
                    Operation::CloseRun
                } else {
                    self.phase = Phase::Recover;
                    Operation::RecoverOrphans { skip: vec![self.task_id.clone()] }
                }
            }
            (Phase::Pickup, Completion::Picked(None)) => {
                self.phase = Phase::RecoveryClose;
                Operation::CloseRun
            }
            (Phase::Pickup, Completion::Picked(Some(_))) => {
                self.phase = Phase::Recover;
                Operation::RecoverOrphans { skip: vec![self.task_id.clone()] }
            }
            (Phase::Recover, Completion::Done) => {
                self.phase = Phase::Run;
                Operation::Run {
                    batch: vec![self.message.clone()],
                    task_ids: vec![self.task_id.clone()],
                    interrupt: self.policy == Policy::Interrupt,
                    drain: false,
                }
            }
            (Phase::Run, Completion::Turn(result)) => {
                self.result = Some(result);
                self.phase = Phase::Close;
                Operation::CloseRun
            }
            (Phase::Close, Completion::Done) => {
                if self.result.is_none() {
                    let row = self.observed.take().ok_or_else(|| {
                        StorageError::Invalid("A2A task disappeared after acceptance".into())
                    })?;
                    self.result = Some(crate::a2a::result_for(
                        row,
                        &self.thread_id,
                        self.last_run_id.as_deref(),
                    ));
                }
                self.phase = Phase::Finished;
                return Ok(None);
            }
            (Phase::Observe, Completion::Task(Some(row)))
                if !matches!(row.status, TaskState::Submitted | TaskState::Working) =>
            {
                self.answer(row)
            }
            (Phase::Observe, Completion::Task(row)) => {
                self.observed = row;
                if self.policy == Policy::Batch {
                    self.phase = Phase::Inbox;
                    Operation::ReadInbox
                } else if self.observed.as_ref().is_some_and(|r| r.status == TaskState::Working) {
                    self.phase = Phase::RecoveryOpen;
                    self.run_sequence += 1;
                    Operation::Open {
                        run_id: format!("{}_recover_{}", self.task_id, self.run_sequence),
                        task_id: self.task_id.clone(),
                        exclusive: true,
                        wait: false,
                        interrupt: false,
                        pickup: false,
                        resume: false,
                    }
                } else {
                    self.poll(now);
                    return Ok(None);
                }
            }
            (Phase::Inbox, Completion::Inbox(rows)) if rows.is_empty() => {
                if self.leased {
                    self.phase = Phase::Release;
                    Operation::ReleaseDrained
                } else {
                    self.poll(now);
                    return Ok(None);
                }
            }
            (Phase::Inbox, Completion::Inbox(rows)) => {
                self.run_sequence += 1;
                let task_id = rows.last().expect("nonempty inbox").task_id.clone();
                self.rows = rows;
                self.index = 0;
                self.batch.clear();
                self.task_ids.clear();
                self.phase = Phase::DrainOpen;
                self.draining = true;
                Operation::Open {
                    run_id: format!("{}_drain_{}", self.task_id, self.run_sequence),
                    task_id,
                    exclusive: true,
                    wait: false,
                    interrupt: false,
                    pickup: false,
                    resume: self.leased,
                }
            }
            (Phase::DrainOpen, Completion::Opened { closed: false, .. }) => {
                self.leased = true;
                self.drain_pickup()
            }
            (Phase::DrainOpen, Completion::Opened { closed: true, .. }) => {
                self.phase = Phase::RecoveryClose;
                Operation::CloseRun
            }
            (Phase::DrainPickup, Completion::Picked(was)) => {
                let row = &self.rows[self.index];
                match was {
                    Some(TaskState::Working) => {
                        self.phase = Phase::DrainOrphan;
                        return Ok(Some(Operation::FailOrphan {
                            task_id: row.task_id.clone(),
                            reason: lk_agents_abi::storage::thread::REASON_WORKER_LOST,
                        }));
                    }
                    Some(_) => {
                        // Receipt identity stays separate from the client's continuation target.
                        self.batch.push(Message::from_a2a(
                            &row.message,
                            row.received_at as f64,
                            None,
                        )?);
                        self.task_ids.push(row.task_id.clone());
                    }
                    None => {}
                }
                self.index += 1;
                self.drain_pickup()
            }
            (Phase::DrainOrphan, Completion::Done) => {
                self.index += 1;
                self.drain_pickup()
            }
            (Phase::DrainRecover, Completion::Done) => {
                self.phase = Phase::DrainRun;
                Operation::Run {
                    batch: self.batch.clone(),
                    task_ids: self.task_ids.clone(),
                    interrupt: false,
                    drain: true,
                }
            }
            (Phase::DrainRun, Completion::Done) => {
                self.phase = Phase::DrainCommit;
                Operation::CommitDrain { through: self.rows.last().expect("a drain has rows").id }
            }
            (Phase::DrainCommit, Completion::DrainCommitted { run_id }) => {
                self.last_run_id = Some(run_id);
                self.phase = Phase::Release;
                Operation::ReleaseDrained
            }
            (Phase::Release, Completion::Released(released)) => {
                self.leased = !released;
                if released {
                    self.draining = false;
                    self.observe()
                } else {
                    self.phase = Phase::Inbox;
                    Operation::ReadInbox
                }
            }
            (Phase::RecoveryOpen, Completion::Opened { .. }) => {
                self.phase = Phase::RecoveryOrphans;
                Operation::RecoverOrphans { skip: Vec::new() }
            }
            (Phase::RecoveryOrphans, Completion::Done) => {
                self.phase = Phase::RecoveryClose;
                Operation::CloseRun
            }
            (Phase::RecoveryClose, Completion::Done) => self.observe(),
            (Phase::Failed, Completion::Done) => self.observe(),
            (Phase::Refused, Completion::RunningTask(running_task_id)) => {
                return Err(StorageError::Rejected { running_task_id });
            }
            _ => return Err(StorageError::Invalid("unexpected A2A operation outcome".into())),
        };
        Ok(Some(operation))
    }

    fn arrive(&mut self) -> Operation {
        match self.policy {
            Policy::Batch => {
                self.phase = Phase::Receive;
                Operation::Receive { message: self.message.to_a2a(), task: self.submitted() }
            }
            Policy::Reject => self.next(Next::Open),
            _ => {
                self.phase = Phase::Submit;
                Operation::Submit(self.submitted())
            }
        }
    }

    fn error(&mut self, error: StorageError, now: Duration) -> Vec<Effect> {
        match (&self.phase, &error) {
            (Phase::Open, StorageError::Rejected { .. }) if self.policy == Policy::Reject => {
                self.phase = Phase::Refused;
                self.issue(Operation::RunningTask)
            }
            (Phase::DrainOpen | Phase::RecoveryOpen, StorageError::Rejected { .. }) => {
                self.poll(now);
                Vec::new()
            }
            (Phase::Open, StorageError::Rejected { .. }) => {
                self.phase = Phase::Failed;
                self.issue(Operation::FailOrphan {
                    task_id: self.task_id.clone(),
                    reason: REASON_LEASE_TIMEOUT,
                })
            }
            (Phase::Run | Phase::DrainRun, StorageError::Unavailable(_)) => {
                self.phase = Phase::Failed;
                self.issue(Operation::FailOrphan {
                    task_id: self.task_id.clone(),
                    reason: REASON_STORAGE_UNAVAILABLE,
                })
            }
            _ => self.finish(Err(error)),
        }
    }
}
