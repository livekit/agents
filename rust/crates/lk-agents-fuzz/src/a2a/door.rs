//! One request at the A2A text door, as a process: the steps of `lk_agents_runtime::a2a::Turn`
//! (`exclusive` for `interrupt` and `queue`, `concurrent`, `reject`; `batch` in [`super::batch`])
//! and the storage engine's run under them, one plan or core decision per step. Every decision is
//! a core's: the storage plans and decodings, `run_commit`, `merge_write`, `Renewal` and
//! `LanePoll`; the A2A core's `resolve_policy`, `resolve_debounce`, `answer_pending`,
//! `checkpoint_stop`, `settle` and `result_for`. The handler is a fake: it works for a drawn time,
//! reaches checkpoints, and ends with text, a raise or a question, yielding (or not) once a
//! checkpoint says stop.

use std::time::Duration;

use fastrand::Rng;
use livekit_protocol::agent::chat_context::ChatItem;
use lk_agents_core::a2a::{
    HandlerOutput, Message, PendingCall, Policy, Settle, Stop, TurnIds, TurnResult, answer_pending,
    batch_to_chat_message, channel_of, checkpoint_stop, failed_message, python_reason,
    resolve_debounce, resolve_policy, result_for, settle, unpersisted_failed,
};
use lk_agents_storage::sql::Table;
use lk_agents_storage::db::{self, Acquire, AcquireOutcome, Acquired, Commit, Decode, EventWrite, Fenced, Plan, Read};
use lk_agents_storage::TaskWrite;
use lk_agents_storage::lease::{LanePoll, Renewal, Renewed, lease_secs};
use lk_agents_storage::run::{
    ItemRow, LEASE_WAIT_TIMEOUT, LOOKUP_LIMIT, REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE,
    REASON_WORKER_LOST, RunOutcome, RunSummary, RunUsage, RunWrites, TaskPatch, TaskRow, TaskState,
    merge_write, run_commit,
};
use lk_agents_storage::schema::{RunEnded, RunStarted, TaskData};
use lk_agents_storage::{Fence, StorageError};

use super::batch::{After, Await};
use super::{LANE, Observed, Refusal, Response, Setup, StagedRun, THREAD};
use crate::storage::db::{CommitFault, SimDb};

/// The runs `reject` reads to name the one holding the lane (`RUNS_FOR_REJECT`).
const RUNS_FOR_REJECT: usize = 50;
/// The door's answer to a follow-up naming a terminal task: a follow-up is a new task.
pub const FOLLOW_UP_REFUSED: &str = "UnsupportedOperationError";

/// How the fake handler ends.
#[derive(Debug, Clone, Copy)]
enum Ending {
    Text,
    Raise,
    Ask,
}

/// The fake handler: when it ends, its next checkpoint, how it ends, whether it yields when a
/// checkpoint answers stop (a handler between awaits does not see it at once).
struct Handler {
    ends_at: i64,
    next_checkpoint: i64,
    ending: Ending,
    honors_yield: bool,
    yielded: bool,
}

/// A lease a run holds, and its renewal.
pub(super) struct Held {
    pub fence: Fence,
    renewal: Renewal,
}

/// An open run: the lease (none under `concurrent`), what it loaded and what the turn staged.
pub(super) struct Turn {
    pub(super) run_id: String,
    pub(super) held: Option<Held>,
    loaded: Vec<(String, ChatItem)>,
    history: Vec<ItemRow>,
    /// The lane's newest rows as the acquire loaded them, newest first; `None` for a run that
    /// took no lease, whose lookups read the database.
    tasks: Option<Vec<TaskRow>>,
    writes: Vec<TaskWrite>,
    pub(super) answer_task: String,
    batch: Vec<Message>,
    task_ids: Vec<String>,
    now_s: f64,
    handler: Handler,
    stop: Option<Stop>,
    pub(super) commit: Option<Commit>,
    pub(super) output: Option<serde_json::Value>,
    /// A drain's turn: the last inbox row it consumed, and what follows the drain.
    pub(super) drain: Option<(i64, After)>,
}

pub(super) enum Phase {
    Send,
    Acquire {
        poll: LanePoll,
        submit: bool,
        submitted: Option<bool>,
        marked: bool,
    },
    /// `reject`: the lane is taken; the row is written under the lease, then picked up.
    Admit(Box<Acquired>),
    Await(Await),
    /// `batch`: one drain run per pass over the pending inbox rows, under the runner's lease
    /// or a fresh one.
    Drain {
        lease: Option<Fence>,
        then: After,
    },
    /// A drain run committed keeping the lease: released unless a row arrived meanwhile.
    Release {
        fence: Fence,
        then: After,
    },
    Turn(Box<Turn>),
    Commit(Box<Turn>),
    Resolve {
        turn: Box<Turn>,
        poll: LanePoll,
    },
    /// Answered with the task as it stood; the commit keeps resolving in the background.
    Later {
        turn: Box<Turn>,
        poll: LanePoll,
    },
    Done,
}

pub struct Request {
    pub id: usize,
    /// The message as sent; chosen by the client at the first step unless this is a retry.
    pub message: Option<a2a::Message>,
    pub arrive_at: i64,
    pub alive: bool,
    /// The worker stalls (a paused process) until then.
    pub paused_until: i64,
    pub(super) phase: Phase,
    pub(super) ready_at: i64,
    pub(super) task_id: String,
    pub(super) holder: String,
    pub(super) tokens: u64,
    interrupt: bool,
    /// The last drain run this request made, named in its answer.
    pub(super) last_run_id: Option<String>,
}

pub(super) fn execute<D: Decode>(
    db: &mut SimDb,
    plan: Plan<D>,
    fault: Option<CommitFault>,
) -> Result<D::Output, StorageError> {
    let out = db.batch(&plan.statements, fault)?;
    plan.decoder.decode(&out)
}

pub(super) fn read<T>(db: &SimDb, read: Read<T>) -> Result<Vec<T>, StorageError> {
    read.decode(&db.query(&read.statement)?)
}

pub(super) fn task(db: &SimDb, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
    Ok(read(db, db::task(LANE, task_id))?.into_iter().next())
}

pub(super) fn ms(t: i64) -> Duration {
    Duration::from_millis(t as u64)
}

pub(super) fn failed(task_id: &str, reason: &str) -> a2a::Message {
    failed_message(task_id, reason, format!("msg_{task_id}_{reason}"))
}

fn has_ask(r: &TaskRow) -> bool {
    r.status == TaskState::InputRequired && r.internal.as_ref().is_some_and(|i| i.ask.is_some())
}

/// The row `matches` finds in the loaded rows, else, when they are not the whole lane, the
/// stored one (`Tasks::find`).
fn find(
    db: &SimDb,
    loaded: Option<&[TaskRow]>,
    matches: impl Fn(&TaskRow) -> bool,
    task_id: &str,
) -> Result<Option<TaskRow>, StorageError> {
    match loaded.and_then(|rows| rows.iter().find(|r| matches(r))) {
        Some(r) => Ok(Some(r.clone())),
        None if loaded.is_some_and(|rows| rows.len() < LOOKUP_LIMIT as usize) => Ok(None),
        None => task(db, task_id),
    }
}

/// The question the message answers: the task it names when that one waits, else the newest
/// waiting one (`Tasks::pending_ask`).
fn pending_ask(
    db: &SimDb,
    loaded: Option<&[TaskRow]>,
    named: Option<&str>,
) -> Result<Option<TaskRow>, StorageError> {
    if let Some(named) = named {
        return Ok(find(db, loaded, |r| r.task_id == named, named)?.filter(has_ask));
    }
    match loaded.filter(|rows| rows.len() < LOOKUP_LIMIT as usize) {
        Some(rows) => Ok(rows.iter().find(|r| has_ask(r)).cloned()),
        None => Ok(read(db, db::tasks(LANE, Some(TaskState::InputRequired), LOOKUP_LIMIT))?
            .into_iter()
            .find(has_ask)),
    }
}

impl Request {
    pub fn new(id: usize, message: Option<a2a::Message>, arrive_at: i64, worker: usize) -> Self {
        Request {
            id,
            message,
            arrive_at,
            alive: true,
            paused_until: 0,
            phase: Phase::Send,
            ready_at: arrive_at,
            task_id: format!("AST_r{id}"),
            holder: format!("w{worker}/run_r{id}"),
            tokens: 0,
            interrupt: false,
            last_run_id: None,
        }
    }

    pub fn done(&self) -> bool {
        !self.alive || matches!(self.phase, Phase::Done)
    }

    /// Whether the client still waits on this request's answer.
    pub fn sent(&self) -> bool {
        !matches!(self.phase, Phase::Send)
    }

    /// When the next step is due: the handler's next event or the lease renewal while a turn
    /// runs, never before a pause ends.
    pub fn wake(&self) -> i64 {
        let due = match &self.phase {
            Phase::Turn(t) => {
                let h = &t.handler;
                let renew =
                    t.held.as_ref().map_or(i64::MAX, |l| l.renewal.wake_at().as_millis() as i64);
                h.next_checkpoint.min(h.ends_at).min(renew)
            }
            _ => self.ready_at,
        };
        due.max(self.paused_until)
    }

    pub fn step(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
    ) {
        self.ready_at = now;
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        self.phase = match phase {
            Phase::Send => self.send(now, db, rng, t, seen),
            Phase::Acquire { poll, submit, submitted, marked } => {
                self.acquire(now, db, rng, t, seen, poll, submit, submitted, marked)
            }
            Phase::Admit(acquired) => self.admit(now, db, rng, t, seen, *acquired),
            Phase::Await(a) => self.await_task(now, db, seen, a),
            Phase::Drain { lease, then } => self.drain(now, db, rng, t, seen, lease, then),
            Phase::Release { fence, then } => self.release_drained(now, db, seen, fence, then),
            Phase::Turn(turn) => self.turn(now, db, rng, t, seen, turn),
            Phase::Commit(turn) => self.commit(now, db, rng, t, seen, turn),
            Phase::Resolve { turn, poll } => self.resolve(now, db, rng, t, seen, turn, poll),
            Phase::Later { turn, poll } => self.later(now, db, rng, t, turn, poll),
            Phase::Done => Phase::Done,
        };
    }

    pub(super) fn message(&self) -> &a2a::Message {
        self.message.as_ref().expect("the client chose the message before the first step")
    }

    pub(super) fn respond(
        &self,
        seen: &mut Observed,
        answer: Result<TurnResult, StorageError>,
        uncertain: bool,
    ) {
        let message_id = self.message().message_id.clone();
        seen.responses.push(Response { request: self.id, message_id, answer, uncertain });
    }

    /// Answers the task as it stands, or the failed one it could not write.
    pub(super) fn answer_row(
        &self,
        seen: &mut Observed,
        record: Option<TaskRow>,
        task_id: &str,
        reason: &str,
        now: i64,
        run_id: Option<&str>,
    ) -> Phase {
        let record = self.record_or_failed(record, task_id, reason, now);
        self.respond(seen, Ok(result_for(record, THREAD, run_id)), false);
        Phase::Done
    }

    /// The record this request has for `task_id`, or the failed one it could not write.
    pub(super) fn record_or_failed(
        &self,
        record: Option<TaskRow>,
        task_id: &str,
        reason: &str,
        now: i64,
    ) -> TaskRow {
        record.unwrap_or_else(|| {
            let msg = Message::from_a2a(self.message(), now as f64 / 1000.0, None).ok();
            unpersisted_failed(task_id, LANE, msg.as_ref(), reason, now, format!("msg_{task_id}"))
        })
    }

    pub(super) fn submitted_row(&self, now: i64) -> TaskWrite {
        let msg =
            Message::from_a2a(self.message(), now as f64 / 1000.0, None).expect("a messageId");
        TaskWrite {
            task_id: self.task_id.clone(),
            status: TaskState::Submitted,
            message_id: Some(msg.message_id.clone()),
            name: None,
            data: TaskData { history: vec![msg.to_a2a()], ..TaskData::default() },
            internal: None,
        }
    }

    /// The door refuses a follow-up to a terminal task; otherwise the channel's policy and
    /// debounce. Under the exclusive policies the `submitted` row is
    /// durable before the debounce when there is one, else it goes in the acquire batch.
    fn send(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
    ) -> Phase {
        if let Some(named) = self.message().task_id.clone() {
            match task(db, &named) {
                Err(e) => return self.fail(seen, e),
                Ok(Some(r)) if r.terminal() => {
                    seen.follow_ups_refused += 1;
                    let message = format!("{named} is {:?}: a follow-up is a new task", r.status);
                    let refused = StorageError::Server { code: FOLLOW_UP_REFUSED.into(), message };
                    return self.fail(seen, refused);
                }
                Ok(_) => {}
            }
        }
        let channel = channel_of(self.message());
        let policy = resolve_policy(&t.policy, &channel);
        self.interrupt = policy == Policy::Interrupt;
        let debounce = resolve_debounce(&t.debounce, &channel, policy);
        match policy {
            Policy::Batch => return self.receive(now, db, t, seen, debounce),
            Policy::Concurrent => return self.concurrent(now, db, rng, t, seen),
            Policy::Reject => return self.reject(now, db, t, seen),
            Policy::Interrupt | Policy::Queue => {}
        }
        let poll = LanePoll::new(ms(now), LEASE_WAIT_TIMEOUT);
        if debounce.is_zero() {
            return Phase::Acquire { poll, submit: true, submitted: None, marked: false };
        }
        match db.batch(&[db::submit(LANE, &self.submitted_row(now))], None) {
            Ok(out) => {
                self.ready_at = now + debounce.as_millis() as i64;
                let submitted = Some(out.results[0].rows_affected() > 0);
                let poll = LanePoll::new(ms(self.ready_at), LEASE_WAIT_TIMEOUT);
                Phase::Acquire { poll, submit: false, submitted, marked: false }
            }
            Err(e) => self.fail(seen, e),
        }
    }

    pub(super) fn fail(&self, seen: &mut Observed, e: StorageError) -> Phase {
        self.respond(seen, Err(e), false);
        Phase::Done
    }

    /// One acquire attempt under `holder` for `run_id`, picking up this request's row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_acquire(
        &mut self,
        now: i64,
        db: &mut SimDb,
        t: &Setup,
        run_id: &str,
        task_id: &str,
        holder: &str,
        submit: Option<&TaskWrite>,
        pickup: bool,
    ) -> Result<AcquireOutcome, StorageError> {
        self.tokens += 1;
        let token = format!("lease_r{}_{}", self.id, self.tokens);
        let channel = channel_of(self.message());
        let started = RunStarted {
            task_id: Some(task_id.to_owned()),
            stream_id: Some(task_id.to_owned()),
            channel: Some(channel),
            exclusive: None,
        };
        let req = Acquire {
            lane: LANE,
            holder,
            token: &token,
            lease_secs: lease_secs(t.lease_ttl),
            wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
            run_id,
            task_id,
            run_started: EventWrite::run_started(now, &started),
            submit,
            pickup,
            tasks_limit: LOOKUP_LIMIT,
        };
        execute(db, db::acquire(&req, Vec::new()), None)
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        mut poll: LanePoll,
        submit: bool,
        submitted: Option<bool>,
        mut marked: bool,
    ) -> Phase {
        let row = self.submitted_row(now);
        let (run_id, holder) = (self.task_id.clone(), self.holder.clone());
        let submit = submit.then_some(&row);
        let mut acquired =
            match self.try_acquire(now, db, t, &run_id, &run_id, &holder, submit, true) {
                Ok(AcquireOutcome::Acquired(a)) => a,
                Ok(AcquireOutcome::Held { submitted: s }) => {
                    let submitted = submitted.or(s);
                    if self.interrupt && !marked {
                        match db.batch(&[db::set_waiter(LANE, &self.holder)], None) {
                            Ok(out) => marked = out.results[0].rows_affected() > 0,
                            Err(e) => return self.fail(seen, e),
                        }
                    }
                    return match poll.next(ms(now)) {
                        Some(at) => {
                            self.ready_at = at.as_millis() as i64;
                            Phase::Acquire { poll, submit: false, submitted, marked }
                        }
                        None => self.lease_timeout(now, db, seen),
                    };
                }
                Err(e) => return self.fail(seen, e),
            };
        acquired.submitted = submitted.or(acquired.submitted);
        let fence = acquired.fence.clone();
        let message_id = self.message().message_id.clone();
        if acquired.submitted == Some(false)
            && let Some(existing) =
                acquired.tasks.iter().find(|r| r.message_id.as_deref() == Some(&message_id))
        {
            // a retried messageId: the run ends without a turn, then the task it created
            seen.deduplicated += 1;
            let existing = existing.clone();
            if let Err(e) = self.close_empty(now, db, &fence, acquired.already_committed) {
                return self.fail(seen, e);
            }
            if existing.terminal() {
                self.respond(seen, Ok(result_for(existing, THREAD, None)), false);
                return Phase::Done;
            }
            return Phase::Await(Await::new(existing.task_id, now, false, true));
        }
        if acquired.already_committed || !acquired.picked_up {
            // canceled before its turn: nothing runs
            return self.close_unanswered(now, db, seen, &fence, acquired.already_committed);
        }
        let task_id = self.task_id.clone();
        if let Err(e) = self.fail_orphans(db, &acquired.tasks, std::slice::from_ref(&task_id)) {
            return self.abandon(now, db, seen, &fence, &task_id, false, e);
        }
        let batch = match Message::from_a2a(self.message(), 0.0, None) {
            Ok(m) => vec![m],
            Err(e) => return self.abandon(now, db, seen, &fence, &task_id, false, e),
        };
        self.open_turn(
            now,
            db,
            rng,
            t,
            seen,
            Opened::held(task_id.clone(), *acquired, t, now),
            batch,
            vec![task_id],
        )
    }

    /// A run that answers nothing: closed, then the task as it stands.
    fn close_unanswered(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        fence: &Fence,
        already_committed: bool,
    ) -> Phase {
        if let Err(e) = self.close_empty(now, db, fence, already_committed) {
            return self.fail(seen, e);
        }
        match task(db, &self.task_id) {
            Ok(record) => self.answer_row(seen, record, &self.task_id, "not-written", now, None),
            Err(e) => self.fail(seen, e),
        }
    }

    /// A dead holder's `working` rows, found under this run's lease, fail `worker-lost`; `skip`
    /// is this run's own.
    pub(super) fn fail_orphans(
        &self,
        db: &mut SimDb,
        loaded: &[TaskRow],
        skip: &[String],
    ) -> Result<(), StorageError> {
        let working = match loaded.len() < LOOKUP_LIMIT as usize {
            true => {
                loaded.iter().rev().filter(|r| r.status == TaskState::Working).cloned().collect()
            }
            false => {
                let rows = read(db, db::tasks(LANE, Some(TaskState::Working), LOOKUP_LIMIT))?;
                rows.into_iter().rev().collect::<Vec<_>>()
            }
        };
        for row in working.iter().filter(|r| !skip.contains(&r.task_id)) {
            let fail =
                db::fail_orphan(LANE, &row.task_id, &failed(&row.task_id, REASON_WORKER_LOST));
            db.batch(&[fail], None)?;
        }
        Ok(())
    }

    /// The turn's start: what this turn adds goes after everything loaded; the pending
    /// question it answers settled first (`Turn::turn` up to the handler).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open_turn(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        opened: Opened,
        mut batch: Vec<Message>,
        task_ids: Vec<String>,
    ) -> Phase {
        let Opened { run_id, held, items, tasks, drain } = opened;
        let after = items.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let mut now_s = now as f64 / 1000.0;
        if after >= now_s {
            now_s = after + 2e-6;
        }
        let fence = held.as_ref().map(|h| h.fence.clone());
        let (open_run, in_drain) = (run_id.clone(), drain.is_some());
        let failed_open = |me: &mut Self, db: &mut SimDb, seen: &mut Observed, e| match &fence {
            Some(fence) => me.abandon(now, db, seen, fence, &open_run, in_drain, e),
            None => me.fail(seen, e),
        };
        // a received message keeps its arrival time; this request's own is stamped now
        if drain.is_none()
            && let Some(last) = batch.last_mut()
        {
            last.received_at = now_s;
        }
        let last = batch.last().expect("a turn has a message").clone();
        let received = task_ids.last().expect("a task per message").clone();
        let pending = match pending_ask(db, tasks.as_deref(), last.task_id.as_deref()) {
            Ok(p) => p,
            Err(e) => return failed_open(self, db, seen, e),
        };
        let mut history = items.clone();
        let answer = answer_pending(
            &mut history,
            pending.as_ref(),
            &last,
            &received,
            now_s,
            format!("item_out_{run_id}"),
        );
        if pending.as_ref().is_some_and(|p| last.task_id.as_ref() == Some(&p.task_id)) {
            seen.continuations += 1;
        }
        let ends_at = now + rng.i64(t.work_ms.0..t.work_ms.1);
        let mut turn = Turn {
            run_id,
            held,
            loaded: items.iter().map(|i| (i.id.clone(), i.payload.clone())).collect(),
            history,
            tasks,
            writes: Vec::new(),
            answer_task: answer.answer_task,
            batch,
            task_ids,
            now_s,
            handler: Handler {
                ends_at,
                next_checkpoint: now + rng.i64(20..400),
                ending: match rng.u8(0..10) {
                    0 => Ending::Raise,
                    1..4 => Ending::Ask,
                    _ => Ending::Text,
                },
                honors_yield: rng.u8(0..4) != 0,
                yielded: false,
            },
            stop: None,
            commit: None,
            output: None,
            drain,
        };
        for tr in answer.transitions {
            if let Err(e) = turn.write(db, &tr.task_id, tr.status, tr.patch) {
                return failed_open(self, db, seen, e);
            }
        }
        Phase::Turn(Box::new(turn))
    }

    /// The lane was not released in time: the early row ends failed `lease-timeout`.
    fn lease_timeout(&self, now: i64, db: &mut SimDb, seen: &mut Observed) -> Phase {
        seen.lease_timeouts += 1;
        let task_id = self.task_id.clone();
        let mut failed_row = || -> Result<Option<TaskRow>, StorageError> {
            if execute(db, db::pickup(LANE, &task_id), None)?.is_some() {
                db.batch(
                    &[db::fail_orphan(LANE, &task_id, &failed(&task_id, REASON_LEASE_TIMEOUT))],
                    None,
                )?;
            }
            task(db, &task_id)
        };
        match failed_row() {
            Ok(record) => self.answer_row(seen, record, &task_id, REASON_LEASE_TIMEOUT, now, None),
            Err(e) => self.fail(seen, e),
        }
    }

    /// Ends a run that ran no turn: released when it was already committed, else one empty
    /// fenced commit (its `run_ended`).
    pub(super) fn close_empty(
        &self,
        now: i64,
        db: &mut SimDb,
        fence: &Fence,
        already_committed: bool,
    ) -> Result<(), StorageError> {
        if already_committed {
            db.batch(&db::release(fence, None), None)?;
            return Ok(());
        }
        let commit = empty_commit(&self.task_id, now);
        execute(
            db,
            db::commit(LANE, &commit, Some(Fenced { fence, keep: None }), Vec::new())?,
            None,
        )?;
        Ok(())
    }

    /// Leaving by error: `run_ended` failed or canceled and the release. A drain's run (`drain`)
    /// answers the error (`close_run`); a turn answering its message fails its task
    /// `storage-unavailable` when the database was unavailable (`end_answered`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn abandon(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        fence: &Fence,
        run_id: &str,
        drain: bool,
        e: StorageError,
    ) -> Phase {
        let outcome = match e {
            StorageError::Cancelled(_) => RunOutcome::Canceled,
            _ => RunOutcome::Failed,
        };
        let ended =
            EventWrite::run_ended(now, &RunEnded { outcome, reason: Some(python_reason(&e)) });
        let _ = db.batch(&db::release(fence, Some((run_id, &ended))), None);
        if !drain && matches!(e, StorageError::Unavailable(_)) {
            return self.fail_unavailable(now, db, seen, e);
        }
        self.fail(seen, e)
    }

    /// The task fails `storage-unavailable` and is answered as it stands.
    fn fail_unavailable(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        e: StorageError,
    ) -> Phase {
        let task_id = &self.task_id;
        let fail = db::fail_orphan(LANE, task_id, &failed(task_id, REASON_STORAGE_UNAVAILABLE));
        if let Err(e) = db.batch(&[fail], None) {
            return self.fail(seen, e);
        }
        match task(db, task_id) {
            Ok(Some(record)) => self.answer_row(seen, Some(record), task_id, "", now, None),
            _ => self.fail(seen, e),
        }
    }

    /// `concurrent`: the row, then a run with no lease over the history, append-only. A resent
    /// task or message id awaits the task it created instead.
    fn concurrent(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
    ) -> Phase {
        let inserted = match db.batch(&[db::submit(LANE, &self.submitted_row(now))], None) {
            Ok(out) => out.results[0].rows_affected() > 0,
            Err(e) => return self.fail(seen, e),
        };
        if !inserted {
            seen.deduplicated += 1;
            let message_id = self.message().message_id.clone();
            let existing = match task(db, &self.task_id) {
                Ok(Some(r)) => Ok(Some(r)),
                Ok(None) => read(db, db::task_for_message(LANE, &message_id))
                    .map(|rows| rows.into_iter().next()),
                Err(e) => Err(e),
            };
            return match existing {
                Err(e) => self.fail(seen, e),
                Ok(Some(r)) if !r.terminal() => {
                    Phase::Await(Await::new(r.task_id, now, false, false))
                }
                Ok(r) => {
                    let id = r.as_ref().map_or(self.task_id.clone(), |r| r.task_id.clone());
                    self.answer_row(seen, r, &id, "not-written", now, None)
                }
            };
        }
        let items = match read(db, db::history(LANE)) {
            Ok(items) => items,
            Err(e) => return self.fail(seen, e),
        };
        seen.concurrent_runs += 1;
        let batch = match Message::from_a2a(self.message(), 0.0, None) {
            Ok(m) => vec![m],
            Err(e) => return self.fail(seen, e),
        };
        let opened =
            Opened { run_id: self.task_id.clone(), held: None, items, tasks: None, drain: None };
        let task_ids = vec![self.task_id.clone()];
        self.open_turn(now, db, rng, t, seen, opened, batch, task_ids)
    }

    /// `reject`: refused while another run holds the lane, naming it; else the lane is taken
    /// without waiting and the row written under the lease, so a refused message leaves none.
    fn reject(&mut self, now: i64, db: &mut SimDb, t: &Setup, seen: &mut Observed) -> Phase {
        let info = match read(db, db::lane_info(LANE)) {
            Ok(rows) => rows.into_iter().next(),
            Err(e) => return self.fail(seen, e),
        };
        if info.is_some_and(|i| i.lease_until > now / 1000) {
            return self.refuse(now, db, seen);
        }
        let (run_id, holder) = (self.task_id.clone(), self.holder.clone());
        match self.try_acquire(now, db, t, &run_id, &run_id, &holder, None, false) {
            Ok(AcquireOutcome::Acquired(a)) => {
                if a.already_committed {
                    return self.close_unanswered(now, db, seen, &a.fence.clone(), true);
                }
                let row = self.submitted_row(now);
                match db.batch(&[db::submit(LANE, &row)], None) {
                    Ok(_) => Phase::Admit(a),
                    Err(e) => self.abandon(now, db, seen, &a.fence.clone(), &run_id, false, e),
                }
            }
            // the lane was taken between the check and the open
            Ok(AcquireOutcome::Held { .. }) => self.refuse(now, db, seen),
            Err(e) => self.fail(seen, e),
        }
    }

    /// `RunRejected`, naming the run that holds the lane: the newest without its `run_ended`.
    fn refuse(&self, now: i64, db: &mut SimDb, seen: &mut Observed) -> Phase {
        let events = match read(db, db::lane_events(LANE, 0)) {
            Ok(events) => events,
            Err(e) => return self.fail(seen, e),
        };
        let mut runs = RunSummary::from_events(&events);
        let keep = runs.len().saturating_sub(RUNS_FOR_REJECT);
        runs.drain(..keep);
        let running_task_id =
            runs.into_iter().rev().find(RunSummary::interrupted).and_then(|r| r.task_id);
        // what the checker reads, outside the simulated network: the lane as it stands, held
        // until its lease's last second as the lease statements read it
        let held = db
            .inspect(&db::lane_info(LANE).statement)
            .iter()
            .filter_map(|r| lk_agents_storage::schema::LaneRow::from_row(r).ok())
            .any(|l| l.lease_until >= now / 1000);
        seen.refusals.push(Refusal { request: self.id, task_id: self.task_id.clone(), held });
        self.fail(seen, StorageError::RunRejected { running_task_id })
    }

    /// `reject` under its lease: the row picked up, the dead holders' rows failed, the turn.
    fn admit(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        acquired: Acquired,
    ) -> Phase {
        let fence = acquired.fence.clone();
        let task_id = self.task_id.clone();
        match execute(db, db::pickup(LANE, &task_id), None) {
            // canceled by a client before the turn started: nothing runs
            Ok(None) => return self.close_unanswered(now, db, seen, &fence, false),
            Ok(Some(_)) => {}
            Err(e) => return self.abandon(now, db, seen, &fence, &task_id, false, e),
        }
        if let Err(e) = self.fail_orphans(db, &acquired.tasks, std::slice::from_ref(&task_id)) {
            return self.abandon(now, db, seen, &fence, &task_id, false, e);
        }
        let batch = match Message::from_a2a(self.message(), 0.0, None) {
            Ok(m) => vec![m],
            Err(e) => return self.abandon(now, db, seen, &fence, &task_id, false, e),
        };
        self.open_turn(
            now,
            db,
            rng,
            t,
            seen,
            Opened::held(task_id.clone(), acquired, t, now),
            batch,
            vec![task_id],
        )
    }

    /// The handler's next event: a renewal, a checkpoint, or its end.
    fn turn(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        mut turn: Box<Turn>,
    ) -> Phase {
        if let Some(held) = &mut turn.held
            && held.renewal.wake_at() <= ms(now)
        {
            let extend = db::extend(&held.fence, lease_secs(t.lease_ttl));
            let outcome = db.batch(&[extend], None).map(|out| out.results[0].rows_affected() > 0);
            return match held.renewal.on_extend(ms(now), outcome.as_ref().map(|r| *r)) {
                Renewed::Extended | Renewed::Retry => Phase::Turn(turn),
                Renewed::Lost(why) => {
                    let fence = held.fence.clone();
                    self.abandon_turn(now, db, seen, &fence, &turn, StorageError::Unavailable(why))
                }
            };
        }
        let h = &mut turn.handler;
        if h.next_checkpoint < h.ends_at {
            if turn.stop.is_none() {
                match execute(db, db::checkpoint(LANE, &turn.answer_task), None) {
                    Ok((waiter, status)) => {
                        let holder = turn.held.as_ref().map_or("", |h| h.fence.holder.as_str());
                        turn.stop =
                            checkpoint_stop(self.interrupt, waiter.as_deref(), holder, status);
                    }
                    Err(e) => return self.abandon_open(now, db, seen, &turn, e),
                }
            }
            let h = &mut turn.handler;
            if turn.stop.is_some() && h.honors_yield {
                h.yielded = true;
                h.ends_at = now;
            } else {
                h.next_checkpoint = now + rng.i64(20..400);
            }
            return Phase::Turn(turn);
        }
        let out = turn.handler_output(self.id);
        let answer_canceled = if turn.stop.is_some() || out.interrupted {
            match task(db, &turn.answer_task) {
                Ok(row) => row.is_some_and(|r| r.status == TaskState::Canceled),
                Err(e) => return self.abandon_open(now, db, seen, &turn, e),
            }
        } else {
            false
        };
        match turn.stop {
            Some(Stop::Interrupted) => seen.interrupted += 1,
            Some(Stop::ClientCanceled) => seen.client_canceled += 1,
            None => {}
        }
        match self.stage(now, db, &mut turn, &out, answer_canceled, seen) {
            Ok(()) => Phase::Commit(turn),
            Err(e) => self.abandon_open(now, db, seen, &turn, e),
        }
    }

    /// An open turn's run ends by error: abandoned under its lease, or answered without one.
    fn abandon_open(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        turn: &Turn,
        e: StorageError,
    ) -> Phase {
        match &turn.held {
            Some(h) => self.abandon_turn(now, db, seen, &h.fence.clone(), turn, e),
            None => self.fail(seen, e),
        }
    }

    fn abandon_turn(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        fence: &Fence,
        turn: &Turn,
        e: StorageError,
    ) -> Phase {
        self.abandon(now, db, seen, fence, &turn.run_id, turn.drain.is_some(), e)
    }

    /// The handler's end through the A2A core's `settle`, then the run's one commit.
    fn stage(
        &self,
        now: i64,
        db: &SimDb,
        turn: &mut Turn,
        out: &HandlerOutput,
        answer_canceled: bool,
        seen: &mut Observed,
    ) -> Result<(), StorageError> {
        let user_message =
            batch_to_chat_message(&turn.batch, turn.now_s, format!("item_u_{}", turn.run_id));
        let task_ids = turn.task_ids.clone();
        let batch = turn.batch.clone();
        let settled = settle(
            &mut turn.history,
            Settle {
                batch: &batch,
                task_ids: &task_ids,
                answer_task: &turn.answer_task,
                user_message,
                out,
                stop: turn.stop,
                answer_canceled,
                now: turn.now_s,
                ids: TurnIds {
                    assistant_item: format!("item_a_{}", turn.run_id),
                    status_message: format!("msg_s_{}", turn.run_id),
                    artifact: format!("art_{}", turn.run_id),
                },
            },
        )?;
        for tr in settled.transitions {
            turn.write(db, &tr.task_id, tr.status, tr.patch)?;
        }
        let events: Vec<EventWrite> = settled
            .events
            .into_iter()
            .map(|(r#type, payload)| EventWrite { ts_ms: now, r#type, payload })
            .collect();
        let channel = channel_of(&turn.batch.last().expect("a message").raw);
        let commit = run_commit(RunWrites {
            run_id: &turn.run_id,
            task_id: turn.task_ids.last().expect("a task"),
            channel: Some(&channel),
            exclusive: turn.held.is_some(),
            history: &turn.history,
            loaded: &turn.loaded,
            state: None,
            loaded_state: None,
            events: &events,
            tasks: &turn.writes,
            usage: RunUsage::default(),
            outcome: settled.outcome,
            reason: settled.reason.as_deref(),
            now_ms: now,
            drained_id: turn.drain.as_ref().map(|d| d.0),
            answer_task: Some(turn.answer_task.clone()),
        })?;
        if turn.batch.len() > 1 {
            seen.batched += 1;
        }
        if turn.held.is_none() {
            let asks = turn.writes.iter().filter(|w| w.status == TaskState::InputRequired);
            seen.concurrent_asks.extend(asks.map(|w| w.task_id.clone()));
        }
        seen.staged.push(StagedRun {
            run_id: turn.run_id.clone(),
            messages: turn.batch.iter().map(|m| m.message_id.clone()).collect(),
            interrupt: self.interrupt && turn.drain.is_none(),
            leased: turn.held.is_some(),
            yielded_to_waiter: turn.stop == Some(Stop::Interrupted),
            writes: turn.writes.iter().map(|w| (w.task_id.clone(), w.status)).collect(),
            items: commit
                .items
                .iter()
                .filter(|i| !turn.loaded.iter().any(|(id, _)| *id == i.id))
                .map(|i| i.id.clone())
                .collect(),
        });
        turn.output = settled.output;
        turn.commit = Some(commit);
        Ok(())
    }
}

/// How a turn's run was opened: its id, its lease, what it loaded.
pub(super) struct Opened {
    pub run_id: String,
    pub held: Option<Held>,
    pub items: Vec<ItemRow>,
    pub tasks: Option<Vec<TaskRow>>,
    pub drain: Option<(i64, After)>,
}

impl Opened {
    /// Run `run_id` under the lease the acquire took, renewed from now.
    pub fn held(run_id: String, acquired: Acquired, t: &Setup, now: i64) -> Self {
        let renewal = Renewal::new(t.lease_ttl, t.renew_interval, ms(now));
        Opened {
            run_id,
            held: Some(Held { fence: acquired.fence, renewal }),
            items: acquired.items,
            tasks: Some(acquired.tasks),
            drain: None,
        }
    }
}

impl Turn {
    /// One staged row: `patch` over this run's earlier write or the stored row.
    fn write(
        &mut self,
        db: &SimDb,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<(), StorageError> {
        let base = match self.writes.iter().find(|w| w.task_id == task_id) {
            Some(w) => Some(w.clone()),
            None => find(db, self.tasks.as_deref(), |r| r.task_id == task_id, task_id)?
                .map(TaskWrite::from),
        };
        let write = merge_write(task_id, base.as_ref(), status, patch)?;
        match self.writes.iter().position(|w| w.task_id == task_id) {
            Some(i) => self.writes[i] = write,
            None => self.writes.push(write),
        }
        Ok(())
    }

    fn handler_output(&self, request: usize) -> HandlerOutput {
        let h = &self.handler;
        let mut out = HandlerOutput { interrupted: h.yielded, ..HandlerOutput::default() };
        if h.yielded {
            return out;
        }
        match h.ending {
            Ending::Text => out.output = Some(serde_json::json!(format!("answer r{request}"))),
            Ending::Raise => out.error = Some("RuntimeError: the tool failed".into()),
            Ending::Ask => {
                out.input_required = Some("which day?".into());
                out.pending_call = Some(PendingCall {
                    name: "ask_day".into(),
                    call_id: format!("call_r{request}"),
                    answer_tool: None,
                });
            }
        }
        out
    }
}

/// A run's commit with nothing but its `run_ended`.
pub(super) fn empty_commit(run_id: &str, now: i64) -> Commit {
    run_commit(RunWrites {
        run_id,
        task_id: "",
        channel: None,
        exclusive: true,
        history: &[],
        loaded: &[],
        state: None,
        loaded_state: None,
        events: &[],
        tasks: &[],
        usage: RunUsage::default(),
        outcome: RunOutcome::Completed,
        reason: None,
        now_ms: now,
        drained_id: None,
        answer_task: None,
    })
    .expect("an empty exclusive commit")
}
