//! One A2A message's task, from arrival to its answer (docs/internals/storage.md#the-unit-of-work):
//! deduplication by `messageId`, the channel's arrival policy, the early `submitted` row, the
//! lane claim, the drain of a batch inbox, the turn ([`turn`]: the pending question, the handler's
//! checkpoints and output, the settlement), its commit and every typed storage failure. Storage
//! owns atomic admission and fencing; this machine decides which operation follows each outcome.
//! The runtime executes one [`Operation`] at a time and feeds its typed result back.

#[cfg(test)]
mod tests;
mod turn;

use std::time::Duration;

use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::StreamInfo;
use lk_agents_abi::storage::TaskInternal;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::{
    InboxRow, LanePoll, REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE, REASON_WORKER_LOST,
    RunOutcome, TaskData, TaskRow, TaskWrite,
};
use serde_json::Value;

use crate::a2a::{
    HandlerOutput, HistoryEdit, Message, Policy, TaskTransition, TurnOptions, TurnResult, Verdict,
    failed_message, python_reason, resolve_debounce, resolve_policy, result_for,
    unpersisted_failed,
};
use lk_agents_abi::storage::RunUsage;
use lk_agents_abi::storage::context::ContextScopeWrite;
use lk_agents_abi::storage::thread::ItemRow;
use turn::{Step, Turning};

/// One completion belongs to exactly one issued operation.
#[derive(Debug)]
pub enum Event {
    Start,
    Wake,
    Completed { id: u64, result: Result<Completion, StorageError> },
}

#[derive(Debug)]
pub enum Effect {
    Execute { id: u64, operation: Operation },
    Finished(Result<Answer, StorageError>),
}

/// The operations the runtime executes against storage, the response stream and the handler.
/// At most one run is open at a time: `Open` opens it, `Commit`, `Abandon`, `ResolveLater` and
/// `CloseRun` end it.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// The task an earlier send of this `messageId` created on the lane.
    FindReceipt(String),
    /// The task a continuation names, on any lane of the thread.
    FindTarget(String),
    /// The early `submitted` row, inserted unless its id or `messageId` exists.
    Submit(TaskWrite),
    /// The `batch` policy's inbox receipt with its `submitted` row; takes the lease when free.
    Receive {
        message: a2a::Message,
        task: TaskWrite,
    },
    /// The receipt is durable: the response stream may announce the task.
    Accepted(String),
    Open {
        run_id: String,
        task_id: String,
        channel: Option<String>,
        /// `false` is the `concurrent` policy: no lease, append-only.
        exclusive: bool,
        /// Wait for the lane up to the door's `wait_timeout`.
        wait: bool,
        interrupt: bool,
        /// Claim `task_id` (`submitted` to `working`) in the acquire.
        pickup: bool,
        /// Run under the lease the receive took.
        resume: bool,
    },
    ReadTask(String),
    ReadInbox,
    /// Claims a row under the open run's lease.
    Pickup(String),
    /// The open run's `working` rows, as of its acquire.
    ReadWorking,
    /// Ends each named `working` task `failed` with its status message, which names `reason`.
    FailOrphans {
        reason: &'static str,
        failed: Vec<(String, a2a::Message)>,
    },
    /// The pending question a turn answers, in the open run: the task `named` while it still
    /// waits, else the lane's newest one.
    ReadPending(Option<String>),
    /// Rows by id: the response streams a drained batch's receipts carry.
    ReadTasks(Vec<String>),
    /// Applies history edits, task rows, journal entries and, once the turn settled, how it
    /// ended to the open run; nothing is committed.
    Stage(Box<Stage>),
    /// Opens the handler's turn over the open run.
    Handle(Box<Handle>),
    /// Answers the handler's last checkpoint with the verdict, when it asked, then reads its
    /// next checkpoint or its output.
    Next(Option<Verdict>),
    /// What a checkpoint reads for each of the message's tasks (the answer, the receipt): the
    /// lane's waiter mark and the task's status.
    Checkpoint(Vec<String>),
    /// Commits the open run, reading `answer_task` back with the write.
    Commit {
        answer_task: Option<String>,
        keep_lease: bool,
        drained_id: Option<i64>,
    },
    /// Ends the open run without its writes: `run_ended` and the release.
    Abandon {
        outcome: RunOutcome,
        reason: String,
    },
    /// Hands the open run, whose commit is unresolved, to a background resolver that fails
    /// `task_id` `storage-unavailable` on an authoritative rejection.
    ResolveLater {
        task_id: String,
    },
    /// Commits the open run with nothing of its own, when one is open.
    CloseRun,
    /// Releases the receive's or the drain's kept lease when nothing is pending past it.
    ReleaseDrained,
    /// The lane's last runs, newest first: what names the task of the run holding the lane.
    ReadRuns(usize),
}

/// What a turn stages into its open run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stage {
    pub edits: Vec<HistoryEdit>,
    pub transitions: Vec<TaskTransition>,
    /// Journal entries: `(type, payload)`.
    pub events: Vec<(String, Value)>,
    /// How the run ends, once the turn settled.
    pub end: Option<RunEnd>,
}

/// A settled turn's end: the run's outcome and what the handler leaves for the commit.
#[derive(Debug, Clone, PartialEq)]
pub struct RunEnd {
    pub outcome: RunOutcome,
    pub reason: Option<String>,
    /// The state after the handler; `None` keeps the loaded one.
    pub state: Option<Value>,
    pub scope_writes: Vec<ContextScopeWrite>,
    pub usage: RunUsage,
}

/// What the handler is opened on, besides what the open run loaded.
#[derive(Debug, Clone, PartialEq)]
pub struct Handle {
    pub lane: String,
    pub context_scope_id: String,
    /// The batch as one user chat message.
    pub message: ItemRow,
    pub batch: Vec<Message>,
    /// The task that carries the reply.
    pub task_id: String,
    /// The tool the message is handed to before the model runs.
    pub answer_tool: Option<String>,
    /// Without the lane's lease (`concurrent`) context fields are not written.
    pub read_only: bool,
    /// How many of the lane's newest task rows the handler's context carries.
    pub recent_tasks: i64,
}

/// How many task rows the handler's context carries.
const TASKS_IN_CONTEXT: i64 = 50;

/// What the handler's stream says next.
#[derive(Debug, Clone, PartialEq)]
pub enum HandlerEvent {
    /// It asks whether to go on.
    Checkpoint,
    Output(Box<HandlerOutput>),
}

#[derive(Debug)]
pub enum Completion {
    Task(Option<TaskRow>),
    Tasks(Vec<TaskRow>),
    Submitted(bool),
    Received {
        duplicate: Option<String>,
        leased: bool,
    },
    /// `history_end`: the latest `created_at` the run loaded, Unix seconds.
    Opened {
        closed: bool,
        picked_up: bool,
        history_end: f64,
    },
    Inbox(Vec<InboxRow>),
    Picked(Option<TaskState>),
    Event(HandlerEvent),
    /// `holder` is this worker's lane holder; `rows` the waiter mark and task status read per
    /// task.
    Checkpoint {
        holder: String,
        rows: Vec<(Option<String>, Option<TaskState>)>,
    },
    Committed {
        task: Option<TaskRow>,
        run_id: String,
    },
    Released(bool),
    /// Newest first.
    Runs(Vec<LaneRun>),
    Done,
}

/// One of the lane's runs, as its journal tells it.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneRun {
    pub task_id: Option<String>,
    /// Started and never ended: the run holding the lane, or one whose worker died.
    pub interrupted: bool,
}

/// How many runs a refusal reads to name the running task.
const RUNS_FOR_REJECT: usize = 50;

/// How the message was answered: its task as stored, and the reply when this worker's turn
/// committed it.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub task_id: String,
    /// `None` when the row could not be read back.
    pub task: Option<TaskRow>,
    /// When the row is missing: the reason it failed, or `None` for a commit still resolving
    /// (the task stays `working`; never a failure nobody observed).
    pub reason: Option<&'static str>,
    /// The committed turn's reply and the task that carries it.
    pub output: Option<Value>,
    pub answered_by: Option<String>,
    pub run_id: Option<String>,
}

impl Answer {
    fn row(task_id: String, task: Option<TaskRow>, reason: Option<&'static str>) -> Self {
        Self { task_id, task, reason, output: None, answered_by: None, run_id: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Next {
    Open,
    Drain,
    Pickup,
}

/// What follows closing a run.
#[derive(Debug, Clone, Copy, PartialEq)]
enum After {
    Observe,
    Receipt,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    New,
    Receipt,
    Target,
    Submit,
    Receive,
    Accepted(Next),
    Debounce {
        at: Duration,
        next: Next,
    },
    Open,
    RejectSubmit,
    RejectAccepted,
    Closing(After),
    Pickup,
    Recover,
    Commit,
    Abandon {
        unavailable: bool,
    },
    Resolve,
    Answer {
        reason: Option<&'static str>,
    },
    Observe,
    Poll {
        at: Duration,
        stale: bool,
    },
    Inbox,
    DrainOpen,
    DrainPickup,
    DrainOrphan,
    DrainRecover,
    DrainCommit,
    DrainAbandon {
        unavailable: bool,
    },
    Release,
    RecoveryOpen {
        stale: bool,
    },
    RecoveryFail,
    RecoveryOrphans,
    Refused,
    /// Failing tasks ([`Failing`]) between two phases.
    Failing,
    Turn(Step),
    Finished,
}

/// One durable arrival. IDs are supplied once; recovery and drain run IDs derive from the task
/// id with a private counter. All live resources belong to the executor.
#[derive(Debug)]
pub struct TaskMachine {
    lane: String,
    message: Message,
    task_id: String,
    /// What the ids this machine mints are made from: its first task id, never reassigned.
    ids: String,
    minted: u64,
    /// Unix seconds at `now` zero.
    epoch: f64,
    stream: Option<StreamInfo>,
    policy: Policy,
    debounce: Duration,
    wait_timeout: Duration,
    phase: Phase,
    sequence: u64,
    pending: Option<u64>,
    /// The task being answered and the output of this worker's turn.
    answer: Option<(String, Option<Value>)>,
    error: Option<StorageError>,
    run_id: Option<String>,
    poll: Option<LanePoll>,
    /// The receive or the drain holds the lease between runs.
    leased: bool,
    rows: Vec<InboxRow>,
    index: usize,
    batch: Vec<Message>,
    task_ids: Vec<String>,
    /// The latest `created_at` the open run loaded.
    history_end: f64,
    turn: Option<Turning>,
    failing: Option<Failing>,
}

/// Tasks being failed: a named one, which a pickup first takes to `working` unless it already
/// ended, or a dead holder's `working` rows but `skip`. `then` takes the `Done` that ends it.
#[derive(Debug)]
struct Failing {
    reason: &'static str,
    then: Phase,
    target: FailTarget,
}

#[derive(Debug)]
enum FailTarget {
    Named(String),
    Orphans { skip: Vec<String> },
}

impl TaskMachine {
    /// `epoch` is the Unix time, in seconds, at `now` zero.
    pub fn new(
        lane: String,
        message: Message,
        task_id: String,
        stream: Option<StreamInfo>,
        options: &TurnOptions,
        epoch: f64,
    ) -> Self {
        let policy = resolve_policy(&options.policy, &message.channel);
        Self {
            lane,
            debounce: resolve_debounce(&options.debounce, &message.channel, policy),
            message,
            ids: task_id.trim_start_matches("AST_").to_owned(),
            minted: 0,
            epoch,
            task_id,
            stream,
            policy,
            wait_timeout: options.wait_timeout(),
            phase: Phase::New,
            sequence: 0,
            pending: None,
            answer: None,
            error: None,
            run_id: None,
            poll: None,
            leased: false,
            rows: Vec::new(),
            index: 0,
            batch: Vec::new(),
            task_ids: Vec::new(),
            history_end: 0.0,
            turn: None,
            failing: None,
        }
    }

    /// A fresh id of `kind` (`item`, `msg`, `art`, a run's).
    fn mint(&mut self, kind: &str) -> String {
        self.minted += 1;
        format!("{kind}_{}{}", self.ids, self.minted)
    }

    pub fn policy(&self) -> Policy {
        self.policy
    }

    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout
    }

    /// The caller's result for the machine's `answer`: a row it could not read is the task as
    /// this worker left it, failed with the machine's reason or still `working` while a commit
    /// resolves.
    pub fn result(&mut self, answer: Answer, thread_id: &str, now: Duration) -> TurnResult {
        let message_id = self.mint("msg");
        let now_ms = ((self.epoch + now.as_secs_f64()) * 1000.0) as i64;
        let task = answer.task.unwrap_or_else(|| match answer.reason {
            Some(reason) => unpersisted_failed(
                &answer.task_id,
                &self.lane,
                Some(&self.message),
                reason,
                now_ms,
                message_id,
            ),
            None => TaskWrite {
                task_id: answer.task_id.clone(),
                status: TaskState::Working,
                ..self.submitted()
            }
            .row(&self.lane),
        });
        let mut result = result_for(task, thread_id, answer.run_id.as_deref());
        if let Some(answered_by) = answer.answered_by {
            result.answered_by = answered_by;
            result.output = answer.output;
        }
        result
    }

    pub fn wake_at(&self) -> Option<Duration> {
        match self.phase {
            Phase::Debounce { at, .. } | Phase::Poll { at, .. } => Some(at),
            _ => None,
        }
    }

    pub fn handle(&mut self, event: Event, now: Duration) -> Vec<Effect> {
        let step = match event {
            Event::Start if self.phase == Phase::New => {
                self.phase = Phase::Receipt;
                Ok(Some(Operation::FindReceipt(self.message.message_id.clone())))
            }
            Event::Wake => match self.phase {
                Phase::Debounce { at, next } if now >= at => Ok(Some(self.next(next))),
                Phase::Poll { at, stale } if now >= at => {
                    Ok(Some(if stale { self.recovery_open(true) } else { self.observe() }))
                }
                _ => Ok(None),
            },
            Event::Completed { id, result } if self.pending == Some(id) => {
                self.pending = None;
                match result {
                    Ok(completion) => self.complete(completion, now),
                    Err(error) => self.fail(error, now),
                }
            }
            _ => Ok(None),
        };
        match step {
            Ok(Some(operation)) => self.issue(operation),
            Ok(None) => Vec::new(),
            Err(Finish::Answer(answer)) => {
                self.phase = Phase::Finished;
                vec![Effect::Finished(Ok(answer))]
            }
            Err(Finish::Error(error)) => {
                self.phase = Phase::Finished;
                vec![Effect::Finished(Err(error))]
            }
        }
    }

    fn issue(&mut self, operation: Operation) -> Vec<Effect> {
        self.sequence += 1;
        self.pending = Some(self.sequence);
        vec![Effect::Execute { id: self.sequence, operation }]
    }

    fn submitted(&self) -> TaskWrite {
        TaskWrite {
            task_id: self.task_id.clone(),
            status: TaskState::Submitted,
            message_id: Some(self.message.message_id.clone()),
            name: None,
            data: TaskData { history: vec![self.message.to_a2a()], ..Default::default() },
            internal: self
                .stream
                .as_ref()
                .map(|stream| TaskInternal { ask: None, streams: vec![stream.clone()] }),
        }
    }

    fn open(&mut self, run_id: String, task_id: String, channel: Option<String>) -> Operation {
        Operation::Open {
            run_id,
            task_id,
            channel,
            exclusive: true,
            wait: false,
            interrupt: false,
            pickup: false,
            resume: false,
        }
    }

    fn next(&mut self, next: Next) -> Operation {
        match next {
            Next::Drain => self.read_inbox(),
            Next::Pickup => {
                self.phase = Phase::Pickup;
                Operation::Pickup(self.task_id.clone())
            }
            Next::Open => {
                self.phase = Phase::Open;
                Operation::Open {
                    run_id: self.task_id.clone(),
                    task_id: self.task_id.clone(),
                    channel: Some(self.message.channel.clone()),
                    exclusive: self.policy != Policy::Concurrent,
                    wait: !matches!(self.policy, Policy::Reject | Policy::Concurrent),
                    interrupt: self.policy == Policy::Interrupt,
                    pickup: self.policy != Policy::Reject,
                    resume: false,
                }
            }
        }
    }

    fn after_accepted(&mut self, next: Next, now: Duration) -> Option<Operation> {
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

    fn read_inbox(&mut self) -> Operation {
        self.phase = Phase::Inbox;
        Operation::ReadInbox
    }

    /// Waits for the task. A `batch` receipt is never abandoned: the inbox drains it. A
    /// `concurrent` task has no lane to tell a live run from a dead one: once the wait ran out the
    /// request answers the task as it stands. Any other task still unanswered once the lane could
    /// have taken it is a dead waiter's (`stale`).
    fn poll(&mut self, now: Duration) -> Option<Operation> {
        let poll = self.poll.get_or_insert_with(|| LanePoll::new(now, self.wait_timeout));
        let (at, stale) = match poll.next(now) {
            Some(at) => (at, false),
            None => {
                *poll = LanePoll::new(now, self.wait_timeout);
                if self.policy == Policy::Concurrent {
                    return Some(self.answer_task(None));
                }
                (now + Duration::from_secs(1), self.policy != Policy::Batch)
            }
        };
        self.phase = Phase::Poll { at, stale };
        None
    }

    /// Takes the lane without waiting, to fail what a dead holder left: its `working` rows, and
    /// with `stale` this message's own `submitted` row.
    fn recovery_open(&mut self, stale: bool) -> Operation {
        self.phase = Phase::RecoveryOpen { stale };
        let run_id = self.mint("recover");
        self.open(run_id, self.task_id.clone(), None)
    }

    /// Answers with the task as it now stands; `reason` when it cannot be read.
    fn answer_task(&mut self, reason: Option<&'static str>) -> Operation {
        self.phase = Phase::Answer { reason };
        Operation::ReadTask(self.answering())
    }

    /// The task the answer is about: the one this worker's turn answered, or this message's.
    fn answering(&self) -> String {
        self.answer.as_ref().map_or_else(|| self.task_id.clone(), |a| a.0.clone())
    }

    fn close(&mut self, after: After) -> Operation {
        self.phase = Phase::Closing(after);
        Operation::CloseRun
    }

    fn drain_pickup(&mut self) -> Operation {
        if self.index < self.rows.len() {
            self.phase = Phase::DrainPickup;
            Operation::Pickup(self.rows[self.index].task_id.clone())
        } else if self.batch.is_empty() {
            self.drain_commit()
        } else {
            let skip = self.rows.iter().map(|r| r.task_id.clone()).collect();
            self.fail_orphans(skip, Phase::DrainRecover)
        }
    }

    /// Fails `task_id` `reason` unless it already ended; `then` follows.
    fn fail_task(&mut self, task_id: String, reason: &'static str, then: Phase) -> Operation {
        self.phase = Phase::Failing;
        let target = FailTarget::Named(task_id.clone());
        self.failing = Some(Failing { reason, then, target });
        Operation::Pickup(task_id)
    }

    /// Fails a dead holder's `working` rows `worker-lost`, but `skip`; `then` follows.
    fn fail_orphans(&mut self, skip: Vec<String>, then: Phase) -> Operation {
        self.phase = Phase::Failing;
        let target = FailTarget::Orphans { skip };
        self.failing = Some(Failing { reason: REASON_WORKER_LOST, then, target });
        Operation::ReadWorking
    }

    /// Fails each of `task_ids`, each with its status message.
    fn failed(&mut self, reason: &'static str, task_ids: Vec<String>) -> Operation {
        let failed = task_ids
            .into_iter()
            .map(|task_id| {
                let message = failed_message(&task_id, reason, self.mint("msg"));
                (task_id, message)
            })
            .collect();
        Operation::FailOrphans { reason, failed }
    }

    fn failing_step(
        &mut self,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let failing = self.failing.as_ref().expect("tasks are failing");
        let task_ids: Vec<String> = match (&failing.target, completion) {
            (FailTarget::Named(task_id), Completion::Picked(was)) => {
                was.map(|_| task_id.clone()).into_iter().collect()
            }
            (FailTarget::Orphans { skip }, Completion::Tasks(rows)) => {
                rows.into_iter().map(|r| r.task_id).filter(|t| !skip.contains(t)).collect()
            }
            (_, Completion::Done) => Vec::new(),
            (_, completion) => {
                return Err(Finish::Error(StorageError::Invalid(format!(
                    "unexpected A2A operation outcome {completion:?} while failing tasks"
                ))));
            }
        };
        if task_ids.is_empty() {
            self.phase = self.failing.take().expect("tasks are failing").then;
            return self.complete(Completion::Done, now);
        }
        let reason = failing.reason;
        Ok(Some(self.failed(reason, task_ids)))
    }

    fn drain_commit(&mut self) -> Operation {
        self.phase = Phase::DrainCommit;
        Operation::Commit {
            answer_task: None,
            keep_lease: true,
            drained_id: self.rows.last().map(|r| r.id),
        }
    }

    fn complete(
        &mut self,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        if let Completion::Opened { history_end, .. } = completion {
            self.history_end = history_end;
        }
        if let Phase::Turn(step) = self.phase {
            return self.turn_step(step, completion, now);
        }
        if self.phase == Phase::Failing {
            return self.failing_step(completion, now);
        }
        let operation = match (self.phase, completion) {
            (Phase::Receipt, Completion::Task(Some(row))) => {
                self.task_id = row.task_id.clone();
                if answered(&row) {
                    return Err(Finish::Answer(Answer::row(row.task_id.clone(), Some(row), None)));
                }
                self.observe()
            }
            (Phase::Receipt, Completion::Task(None)) => match &self.message.task_id {
                Some(target) => {
                    self.phase = Phase::Target;
                    Operation::FindTarget(target.clone())
                }
                None => self.arrive(),
            },
            (Phase::Target, Completion::Task(Some(row))) if row.lane == self.lane => {
                if row.terminal() {
                    return Err(Finish::Error(StorageError::Invalid(
                        "a follow-up to a terminal task must start a new task".into(),
                    )));
                }
                self.arrive()
            }
            (Phase::Target, Completion::Task(_)) => {
                return Err(Finish::Error(StorageError::NotFound(
                    "the addressed task is not in this lane".into(),
                )));
            }
            (Phase::Submit, Completion::Submitted(true)) => {
                self.phase = Phase::Accepted(Next::Open);
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::Submit, Completion::Submitted(false))
            | (Phase::Closing(After::Receipt), Completion::Done) => {
                self.phase = Phase::Receipt;
                Operation::FindReceipt(self.message.message_id.clone())
            }
            (Phase::Receive, Completion::Received { duplicate: Some(task_id), .. }) => {
                self.task_id = task_id;
                self.observe()
            }
            (Phase::Receive, Completion::Received { duplicate: None, leased }) => {
                self.leased = leased;
                self.phase = Phase::Accepted(if leased { Next::Drain } else { Next::Open });
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::Accepted(next), Completion::Done) => {
                if self.policy == Policy::Batch && !self.leased {
                    self.observe()
                } else {
                    return Ok(self.after_accepted(next, now));
                }
            }
            (Phase::Open, Completion::Opened { closed: true, .. }) => self.close(After::Observe),
            (Phase::Open, Completion::Opened { .. }) if self.policy == Policy::Reject => {
                // the row is written under the lease: a refused message leaves none
                self.phase = Phase::RejectSubmit;
                Operation::Submit(self.submitted())
            }
            (Phase::Open, Completion::Opened { picked_up: false, .. }) => {
                // a client canceled the task before its turn: nothing runs
                self.close(After::Observe)
            }
            (Phase::Open, Completion::Opened { picked_up: true, .. }) => self.recover(now),
            (Phase::RejectSubmit, Completion::Submitted(true)) => {
                self.phase = Phase::RejectAccepted;
                Operation::Accepted(self.task_id.clone())
            }
            (Phase::RejectSubmit, Completion::Submitted(false)) => {
                // a resend that raced its first send: answered by the first send's task
                self.close(After::Receipt)
            }
            (Phase::RejectAccepted, Completion::Done) => self.next(Next::Pickup),
            (Phase::Pickup, Completion::Picked(None)) => self.close(After::Observe),
            (Phase::Pickup, Completion::Picked(Some(_))) => self.recover(now),
            (Phase::Recover, Completion::Done) => self.single_turn(now),
            (Phase::Commit, Completion::Committed { task, run_id }) => {
                let (task_id, output) = self.answer.take().unwrap_or_default();
                return Err(Finish::Answer(Answer {
                    output,
                    answered_by: Some(task_id.clone()),
                    run_id: Some(run_id),
                    ..Answer::row(task_id, task, Some("not-written"))
                }));
            }
            (
                Phase::Abandon { unavailable: true } | Phase::DrainAbandon { unavailable: true },
                Completion::Done,
            ) => {
                self.answer = None;
                let then = Phase::Answer { reason: Some(REASON_STORAGE_UNAVAILABLE) };
                self.fail_task(self.task_id.clone(), REASON_STORAGE_UNAVAILABLE, then)
            }
            (
                Phase::Abandon { unavailable: false } | Phase::DrainAbandon { unavailable: false },
                Completion::Done,
            ) => {
                return Err(Finish::Error(self.error.take().unwrap_or_else(|| {
                    StorageError::Invalid("a run was abandoned without its cause".into())
                })));
            }
            (Phase::Resolve, Completion::Done) => self.answer_task(None),
            (Phase::Answer { reason }, Completion::Done) => self.answer_task(reason),
            (Phase::Answer { reason }, Completion::Task(task)) => {
                let answer = Answer::row(self.answering(), task, reason);
                return Err(Finish::Answer(Answer { run_id: self.run_id.take(), ..answer }));
            }
            (Phase::Observe, Completion::Task(Some(row))) if answered(&row) => {
                let answer = Answer::row(row.task_id.clone(), Some(row), Some("not-written"));
                return Err(Finish::Answer(Answer { run_id: self.run_id.take(), ..answer }));
            }
            (Phase::Observe, Completion::Task(row)) => {
                if self.policy == Policy::Batch {
                    self.read_inbox()
                } else if row.as_ref().is_some_and(|r| r.status == TaskState::Working)
                    && self.policy != Policy::Concurrent
                {
                    // the row's holder may have died: whoever takes the lane fails it
                    self.recovery_open(false)
                } else {
                    return Ok(self.poll(now));
                }
            }
            (Phase::Inbox, Completion::Inbox(rows)) if rows.is_empty() => {
                if self.leased {
                    self.phase = Phase::Release;
                    Operation::ReleaseDrained
                } else {
                    return Ok(self.poll(now));
                }
            }
            (Phase::Inbox, Completion::Inbox(rows)) => {
                let run_id = self.mint("drain");
                let last = rows.last().expect("nonempty inbox");
                let task_id = last.task_id.clone();
                let channel = Some(crate::a2a::channel_of(&last.message));
                self.rows = rows;
                self.index = 0;
                self.batch.clear();
                self.task_ids.clear();
                self.phase = Phase::DrainOpen;
                Operation::Open {
                    run_id,
                    task_id,
                    channel,
                    exclusive: true,
                    wait: false,
                    interrupt: false,
                    pickup: false,
                    resume: self.leased,
                }
            }
            (Phase::DrainOpen, Completion::Opened { .. }) => {
                self.leased = true;
                self.drain_pickup()
            }
            (Phase::DrainPickup, Completion::Picked(was)) => {
                let row = &self.rows[self.index];
                match was {
                    Some(TaskState::Working) => {
                        // a dead holder's: the pickup took it, it fails worker-lost
                        let task_id = row.task_id.clone();
                        self.phase = Phase::DrainOrphan;
                        return Ok(Some(self.failed(REASON_WORKER_LOST, vec![task_id])));
                    }
                    Some(_) => {
                        // the receipt's id stays apart from the client's continuation target
                        let message = Message::from_a2a(&row.message, row.received_at as f64, None)
                            .map_err(Finish::Error)?;
                        self.batch.push(message);
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
                self.start_turn(self.batch.clone(), self.task_ids.clone(), true, now)
            }
            (Phase::DrainCommit, Completion::Committed { run_id, .. }) => {
                self.run_id = Some(run_id);
                self.phase = Phase::Release;
                Operation::ReleaseDrained
            }
            (Phase::Release, Completion::Released(released)) => {
                self.leased = !released;
                if released { self.observe() } else { self.read_inbox() }
            }
            (Phase::RecoveryOpen { stale: true }, Completion::Opened { .. }) => {
                // still `submitted` with the lane free: its waiter died before the lane
                self.fail_task(self.task_id.clone(), REASON_WORKER_LOST, Phase::RecoveryFail)
            }
            (Phase::RecoveryOpen { stale: false }, Completion::Opened { .. })
            | (Phase::RecoveryFail, Completion::Done) => {
                self.fail_orphans(Vec::new(), Phase::RecoveryOrphans)
            }
            (Phase::RecoveryOrphans, Completion::Done) => self.close(After::Observe),
            (Phase::Closing(After::Observe), Completion::Done) => self.observe(),
            (Phase::Refused, Completion::Runs(runs)) => {
                let running_task_id =
                    runs.into_iter().find(|r| r.interrupted).and_then(|r| r.task_id);
                return Err(Finish::Error(StorageError::Rejected { running_task_id }));
            }
            (phase, completion) => {
                return Err(Finish::Error(StorageError::Invalid(format!(
                    "unexpected A2A operation outcome {completion:?} in {phase:?}"
                ))));
            }
        };
        Ok(Some(operation))
    }

    /// Under the lease, a dead holder's `working` rows fail before the turn; a `concurrent` run
    /// holds no lease and fails nothing.
    fn recover(&mut self, now: Duration) -> Operation {
        self.phase = Phase::Recover;
        if self.policy == Policy::Concurrent {
            return self.single_turn(now);
        }
        self.fail_orphans(vec![self.task_id.clone()], Phase::Recover)
    }

    /// This message's own turn.
    fn single_turn(&mut self, now: Duration) -> Operation {
        self.start_turn(vec![self.message.clone()], vec![self.task_id.clone()], false, now)
    }

    fn arrive(&mut self) -> Operation {
        match self.policy {
            Policy::Batch => {
                self.phase = Phase::Receive;
                Operation::Receive { message: self.message.to_a2a(), task: self.submitted() }
            }
            // the row is written under the lease: a refused message leaves none
            Policy::Reject => self.next(Next::Open),
            _ => {
                self.phase = Phase::Submit;
                Operation::Submit(self.submitted())
            }
        }
    }

    fn fail(&mut self, error: StorageError, now: Duration) -> Result<Option<Operation>, Finish> {
        if self.phase == Phase::Failing {
            // a failure while failing tasks is the failure of the phase that started it
            self.phase = self.failing.take().expect("tasks are failing").then;
        }
        let operation = match (self.phase, &error) {
            (Phase::Open, StorageError::Rejected { .. }) if self.policy == Policy::Reject => {
                self.phase = Phase::Refused;
                Operation::ReadRuns(RUNS_FOR_REJECT)
            }
            (Phase::Open, StorageError::Rejected { .. }) => {
                // the lane was not released within the wait
                let then = Phase::Answer { reason: Some(REASON_LEASE_TIMEOUT) };
                self.fail_task(self.task_id.clone(), REASON_LEASE_TIMEOUT, then)
            }
            (Phase::Observe | Phase::Inbox, StorageError::Unavailable(_)) => {
                // a durable task is watched through a transient failure, not given up on
                return Ok(self.poll(now));
            }
            (Phase::DrainOpen | Phase::RecoveryOpen { .. }, StorageError::Rejected { .. }) => {
                return Ok(self.poll(now));
            }
            (Phase::Turn(_), _) => {
                let unavailable =
                    matches!(error, StorageError::Unavailable(_) | StorageError::LeaseLost(_));
                let outcome = match error {
                    StorageError::Cancelled(_) => RunOutcome::Canceled,
                    _ => RunOutcome::Failed,
                };
                let reason = python_reason(&error);
                self.error = Some(error);
                let drain = self.turn.take().is_some_and(|t| t.drain);
                self.phase = if drain {
                    Phase::DrainAbandon { unavailable }
                } else {
                    Phase::Abandon { unavailable }
                };
                Operation::Abandon { outcome, reason }
            }
            (Phase::Answer { reason }, _) => {
                // the answer's row is unreadable: the task as this worker left it
                let answer = Answer::row(self.answering(), None, reason);
                return Err(Finish::Answer(Answer { run_id: self.run_id.take(), ..answer }));
            }
            (Phase::DrainCommit, StorageError::Uncertain(_)) => {
                self.phase = Phase::Resolve;
                Operation::ResolveLater { task_id: self.task_id.clone() }
            }
            (Phase::DrainCommit, _) => {
                let then = Phase::Answer { reason: Some(REASON_STORAGE_UNAVAILABLE) };
                self.fail_task(self.task_id.clone(), REASON_STORAGE_UNAVAILABLE, then)
            }
            (Phase::Commit, StorageError::Uncertain(_)) => {
                // a lost acknowledgment is not a failure: the message's task stays as it is while
                // the resolver finds the run (decision 39); once applied, it names its answer
                self.answer = None;
                self.phase = Phase::Resolve;
                Operation::ResolveLater { task_id: self.task_id.clone() }
            }
            (Phase::Commit, _) => {
                // a rejected save fails the message's own task; its staged changes are absent,
                // so a question it answered still waits
                self.answer = None;
                let then = Phase::Answer { reason: Some(REASON_STORAGE_UNAVAILABLE) };
                self.fail_task(self.task_id.clone(), REASON_STORAGE_UNAVAILABLE, then)
            }
            _ => return Err(Finish::Error(error)),
        };
        Ok(Some(operation))
    }
}

/// Whether a turn answered the row: it ended, or it asks a question.
fn answered(row: &TaskRow) -> bool {
    row.terminal() || row.status == TaskState::InputRequired
}

/// How a step ends the machine.
enum Finish {
    Answer(Answer),
    Error(StorageError),
}
