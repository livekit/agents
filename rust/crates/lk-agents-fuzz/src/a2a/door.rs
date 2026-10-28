//! One request at the A2A text door, as a process: the exclusive policies' steps
//! (`lk_agents_runtime::a2a::Turn::exclusive` and the storage engine's run under it), one plan
//! or core decision per step. Every decision is a core's: the storage plans and decodings,
//! `run_commit`, `merge_write`, `Renewal` and `LanePoll`; the A2A core's `resolve_policy`,
//! `resolve_debounce`, `answer_pending`, `checkpoint_stop`, `settle` and `result_for`. The
//! handler is a fake: it works for a drawn time, reaches checkpoints, and ends with text, a
//! raise or a question, yielding (or not) once a checkpoint says stop.

use std::time::Duration;

use fastrand::Rng;
use livekit_protocol::agent::chat_context::ChatItem;
use lk_agents_a2a::{
    HandlerOutput, Message, PendingCall, Policy, PolicySpec, Settle, Stop, TurnIds, TurnResult,
    answer_pending, batch_to_chat_message, channel_of, checkpoint_stop, failed_message,
    resolve_debounce, resolve_policy, result_for, settle, unpersisted_failed,
};
use lk_agents_storage::db::{
    self, Acquire, AcquireOutcome, Commit, Decode, EventWrite, Fenced, Plan, Read, Resolution,
    TaskWrite,
};
use lk_agents_storage::lease::{LanePoll, Renewal, Renewed, lease_secs};
use lk_agents_storage::run::{
    ItemRow, LEASE_WAIT_TIMEOUT, LOOKUP_LIMIT, REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE,
    REASON_WORKER_LOST, RunOutcome, RunUsage, RunWrites, TaskPatch, TaskRow, TaskState,
    merge_write, run_commit,
};
use lk_agents_storage::schema::{RunEnded, RunStarted, TaskData};
use lk_agents_storage::{Fence, StorageError};

use super::{LANE, Observed, Response, StagedRun, THREAD, Timing};
use crate::storage::db::{CommitFault, SimDb};

/// The runtime's bound on resolving a lost commit acknowledgment before answering
/// (`RESOLVE_WAIT`), and on resolving it afterwards (`RESOLVE_LATER`).
const RESOLVE_WAIT: Duration = Duration::from_secs(10);
const RESOLVE_LATER: Duration = Duration::from_secs(3600);

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

/// An open run: the lease, what the acquire loaded and what the turn staged.
struct Turn {
    fence: Fence,
    renewal: Renewal,
    loaded: Vec<(String, ChatItem)>,
    history: Vec<ItemRow>,
    /// The lane's newest rows as the acquire loaded them, newest first.
    tasks: Vec<TaskRow>,
    writes: Vec<TaskWrite>,
    answer_task: String,
    batch: Vec<Message>,
    now_s: f64,
    handler: Handler,
    stop: Option<Stop>,
    commit: Option<Commit>,
    output: Option<serde_json::Value>,
}

enum Phase {
    Send,
    Acquire {
        poll: LanePoll,
        submit: bool,
        submitted: Option<bool>,
        marked: bool,
    },
    /// A retried `messageId`: the task it created, awaited, failed `worker-lost` once its
    /// holder's lease lapsed.
    Existing {
        task: String,
        poll: LanePoll,
        attempt: u32,
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
    phase: Phase,
    ready_at: i64,
    task_id: String,
    holder: String,
    tokens: u64,
    interrupt: bool,
}

fn execute<D: Decode>(
    db: &mut SimDb,
    plan: Plan<D>,
    fault: Option<CommitFault>,
) -> Result<D::Output, StorageError> {
    let out = db.batch(&plan.statements, fault)?;
    plan.decoder.decode(&out)
}

fn read<T>(db: &SimDb, read: Read<T>) -> Result<Vec<T>, StorageError> {
    read.decode(&db.query(&read.statement)?)
}

fn task(db: &SimDb, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
    Ok(read(db, db::task(LANE, task_id))?.into_iter().next())
}

fn ms(t: i64) -> Duration {
    Duration::from_millis(t as u64)
}

fn failed(task_id: &str, reason: &str) -> a2a::Message {
    failed_message(task_id, reason, format!("msg_{task_id}_{reason}"))
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
                h.next_checkpoint.min(h.ends_at).min(t.renewal.wake_at().as_millis() as i64)
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
        t: &Timing,
        seen: &mut Observed,
    ) {
        self.ready_at = now;
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        self.phase = match phase {
            Phase::Send => self.send(now, db, t, seen),
            Phase::Acquire { poll, submit, submitted, marked } => {
                self.acquire(now, db, rng, t, seen, poll, submit, submitted, marked)
            }
            Phase::Existing { task, poll, attempt } => {
                self.existing(now, db, seen, task, poll, attempt)
            }
            Phase::Turn(turn) => self.turn(now, db, rng, t, seen, turn),
            Phase::Commit(turn) => self.commit(now, db, rng, t, seen, turn),
            Phase::Resolve { turn, poll } => self.resolve(now, db, rng, t, seen, turn, poll),
            Phase::Later { turn, poll } => self.later(now, db, rng, t, turn, poll),
            Phase::Done => Phase::Done,
        };
    }

    fn message(&self) -> &a2a::Message {
        self.message.as_ref().expect("the client chose the message before the first step")
    }

    fn respond(
        &self,
        seen: &mut Observed,
        answer: Result<TurnResult, StorageError>,
        uncertain: bool,
    ) {
        let message_id = self.message().message_id.clone();
        seen.responses.push(Response { request: self.id, message_id, answer, uncertain });
    }

    /// The record this request has for `task_id`, or the failed one it could not write.
    fn record_or_failed(
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

    fn submitted_row(&self, now: i64) -> TaskWrite {
        let msg =
            Message::from_a2a(self.message(), now as f64 / 1000.0, None).expect("a messageId");
        TaskWrite {
            task_id: self.task_id.clone(),
            status: TaskState::Submitted,
            message_id: Some(msg.message_id.clone()),
            name: None,
            data: TaskData { history: vec![msg.to_a2a()], ..TaskData::default() },
            ask: None,
        }
    }

    /// The channel's policy and debounce; under a debounce the `submitted` row is durable before
    /// the wait, else it goes in the acquire batch.
    fn send(&mut self, now: i64, db: &mut SimDb, t: &Timing, seen: &mut Observed) -> Phase {
        let channel = channel_of(self.message());
        let policy = resolve_policy(&PolicySpec::Default, &channel);
        self.interrupt = policy == Policy::Interrupt;
        let debounce = resolve_debounce(&t.debounce, &channel, policy);
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

    fn fail(&self, seen: &mut Observed, e: StorageError) -> Phase {
        self.respond(seen, Err(e), false);
        Phase::Done
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
        mut poll: LanePoll,
        submit: bool,
        submitted: Option<bool>,
        mut marked: bool,
    ) -> Phase {
        self.tokens += 1;
        let token = format!("lease_r{}_{}", self.id, self.tokens);
        let row = self.submitted_row(now);
        let channel = channel_of(self.message());
        let started = RunStarted {
            task_id: Some(self.task_id.clone()),
            stream_id: Some(self.task_id.clone()),
            channel: Some(channel),
            exclusive: None,
        };
        let req = Acquire {
            lane: LANE,
            holder: &self.holder,
            token: &token,
            lease_secs: lease_secs(t.lease_ttl),
            wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
            run_id: &self.task_id,
            task_id: &self.task_id,
            run_started: EventWrite::run_started(now, &started),
            submit: submit.then_some(&row),
            pickup: true,
            tasks_limit: LOOKUP_LIMIT,
        };
        let mut acquired = match execute(db, db::acquire(&req, Vec::new()), None) {
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
            let poll = LanePoll::new(ms(now), LEASE_WAIT_TIMEOUT);
            return Phase::Existing { task: existing.task_id, poll, attempt: 0 };
        }
        if acquired.already_committed || !acquired.picked_up {
            // canceled before its turn: nothing runs
            if let Err(e) = self.close_empty(now, db, &fence, acquired.already_committed) {
                return self.fail(seen, e);
            }
            return match task(db, &self.task_id) {
                Ok(record) => {
                    let record =
                        self.record_or_failed(record, &self.task_id.clone(), "not-written", now);
                    self.respond(seen, Ok(result_for(record, THREAD, None)), false);
                    Phase::Done
                }
                Err(e) => self.fail(seen, e),
            };
        }
        // a dead holder's working rows fail worker-lost under this lease
        for row in acquired.tasks.iter().rev().filter(|r| r.status == TaskState::Working) {
            if row.task_id == self.task_id {
                continue;
            }
            let fail =
                db::fail_orphan(LANE, &row.task_id, &failed(&row.task_id, REASON_WORKER_LOST));
            if let Err(e) = db.batch(&[fail], None) {
                return self.abandon(now, db, seen, &fence, e);
            }
        }
        let after = acquired.items.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let mut now_s = now as f64 / 1000.0;
        if after >= now_s {
            now_s = after + 2e-6;
        }
        let last = match Message::from_a2a(self.message(), now_s, None) {
            Ok(m) => m,
            Err(e) => return self.abandon(now, db, seen, &fence, e),
        };
        let mut history = acquired.items.clone();
        let pending = acquired
            .tasks
            .iter()
            .find(|r| r.status == TaskState::InputRequired && r.internal.is_some())
            .cloned();
        let answer = answer_pending(
            &mut history,
            pending.as_ref(),
            &last,
            &self.task_id,
            now_s,
            format!("item_out_r{}", self.id),
        );
        if pending.as_ref().is_some_and(|p| last.task_id.as_ref() == Some(&p.task_id)) {
            seen.continuations += 1;
        }
        let renewal = Renewal::new(t.lease_ttl, t.renew_interval, ms(now));
        let ends_at = now + rng.i64(t.work_ms.0..t.work_ms.1);
        let mut turn = Turn {
            fence,
            renewal,
            loaded: acquired.items.iter().map(|i| (i.id.clone(), i.payload.clone())).collect(),
            history,
            tasks: acquired.tasks,
            writes: Vec::new(),
            answer_task: answer.answer_task,
            batch: vec![last],
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
        };
        for tr in answer.transitions {
            if let Err(e) = turn.write(&tr.task_id, tr.status, tr.patch) {
                return self.abandon(now, db, seen, &turn.fence, e);
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
            Ok(record) => {
                let record = self.record_or_failed(record, &task_id, REASON_LEASE_TIMEOUT, now);
                self.respond(seen, Ok(result_for(record, THREAD, None)), false);
                Phase::Done
            }
            Err(e) => self.fail(seen, e),
        }
    }

    /// Ends a run that ran no turn: released when it was already committed, else one empty
    /// fenced commit (its `run_ended`).
    fn close_empty(
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

    /// Leaving by error: `run_ended` failed and the release, then the error answered.
    fn abandon(
        &self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        fence: &Fence,
        e: StorageError,
    ) -> Phase {
        let ended = EventWrite::run_ended(
            now,
            &RunEnded { outcome: RunOutcome::Failed, reason: Some(e.to_string()) },
        );
        let _ = db.batch(&db::release(fence, Some((&self.task_id, &ended))), None);
        self.fail(seen, e)
    }

    /// The handler's next event: a renewal, a checkpoint, or its end.
    fn turn(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
        mut turn: Box<Turn>,
    ) -> Phase {
        if turn.renewal.wake_at() <= ms(now) {
            let extend = db::extend(&turn.fence, lease_secs(t.lease_ttl));
            let outcome = db.batch(&[extend], None).map(|out| out.results[0].rows_affected() > 0);
            return match turn.renewal.on_extend(ms(now), outcome.as_ref().map(|r| *r)) {
                Renewed::Extended | Renewed::Retry => Phase::Turn(turn),
                Renewed::Lost(why) => {
                    let fence = turn.fence.clone();
                    self.abandon(now, db, seen, &fence, StorageError::Unavailable(why))
                }
            };
        }
        let h = &mut turn.handler;
        if h.next_checkpoint < h.ends_at {
            if turn.stop.is_none() {
                match execute(db, db::checkpoint(LANE, &turn.answer_task), None) {
                    Ok((waiter, status)) => {
                        turn.stop = checkpoint_stop(
                            self.interrupt,
                            waiter.as_deref(),
                            &self.holder,
                            status,
                        );
                    }
                    Err(e) => {
                        let fence = turn.fence.clone();
                        return self.abandon(now, db, seen, &fence, e);
                    }
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
                Err(e) => {
                    let fence = turn.fence.clone();
                    return self.abandon(now, db, seen, &fence, e);
                }
            }
        } else {
            false
        };
        match turn.stop {
            Some(Stop::Interrupted) => seen.interrupted += 1,
            Some(Stop::ClientCanceled) => seen.client_canceled += 1,
            None => {}
        }
        match self.stage(now, &mut turn, &out, answer_canceled, seen) {
            Ok(()) => Phase::Commit(turn),
            Err(e) => {
                let fence = turn.fence.clone();
                self.abandon(now, db, seen, &fence, e)
            }
        }
    }

    /// The handler's end through the A2A core's `settle`, then the run's one commit.
    fn stage(
        &self,
        now: i64,
        turn: &mut Turn,
        out: &HandlerOutput,
        answer_canceled: bool,
        seen: &mut Observed,
    ) -> Result<(), StorageError> {
        let task_ids = [self.task_id.clone()];
        let user_message =
            batch_to_chat_message(&turn.batch, turn.now_s, format!("item_u_r{}", self.id));
        let settled = settle(
            &mut turn.history,
            Settle {
                batch: &turn.batch,
                task_ids: &task_ids,
                answer_task: &turn.answer_task,
                user_message,
                out,
                stop: turn.stop,
                answer_canceled,
                now: turn.now_s,
                ids: TurnIds {
                    assistant_item: format!("item_a_r{}", self.id),
                    status_message: format!("msg_s_r{}", self.id),
                    artifact: format!("art_r{}", self.id),
                },
            },
        )?;
        for tr in settled.transitions {
            turn.write(&tr.task_id, tr.status, tr.patch)?;
        }
        let events: Vec<EventWrite> = settled
            .events
            .into_iter()
            .map(|(r#type, payload)| EventWrite { ts_ms: now, r#type, payload })
            .collect();
        let channel = channel_of(self.message());
        let commit = run_commit(RunWrites {
            run_id: &self.task_id,
            task_id: &self.task_id,
            channel: Some(&channel),
            exclusive: true,
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
            drained_id: None,
            answer_task: Some(turn.answer_task.clone()),
        })?;
        seen.staged.push(StagedRun {
            run_id: self.task_id.clone(),
            message_id: self.message().message_id.clone(),
            interrupt: self.interrupt,
            yielded_to_waiter: turn.stop == Some(Stop::Interrupted),
            writes: turn.writes.iter().map(|w| (w.task_id.clone(), w.status)).collect(),
        });
        turn.output = settled.output;
        turn.commit = Some(commit);
        Ok(())
    }

    fn commit(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
        turn: Box<Turn>,
    ) -> Phase {
        match send_commit(db, rng, t, &turn) {
            Ok(committed) => self.answered(now, seen, &turn, committed.task),
            Err(StorageError::Uncertain(_)) => {
                seen.lost_acks += 1;
                Phase::Resolve { turn, poll: LanePoll::new(ms(now), RESOLVE_WAIT) }
            }
            Err(_) => self.rejected(now, db, seen, &turn),
        }
    }

    /// The task after the commit, and the reply.
    fn answered(&self, now: i64, seen: &mut Observed, turn: &Turn, task: Option<TaskRow>) -> Phase {
        let record = self.record_or_failed(task, &self.task_id, "not-written", now);
        let result = TurnResult {
            task: record,
            output: turn.output.clone(),
            thread_id: THREAD.to_owned(),
            answered_by: turn.answer_task.clone(),
            run_id: Some(self.task_id.clone()),
        };
        self.respond(seen, Ok(result), false);
        Phase::Done
    }

    /// A rejected save fails the answer `storage-unavailable`, its staged changes absent.
    fn rejected(&self, now: i64, db: &mut SimDb, seen: &mut Observed, turn: &Turn) -> Phase {
        seen.rejected += 1;
        let answer = turn.answer_task.clone();
        let fail = db::fail_orphan(LANE, &answer, &failed(&answer, REASON_STORAGE_UNAVAILABLE));
        if let Err(e) = db.batch(&[fail], None) {
            return self.fail(seen, e);
        }
        match task(db, &answer) {
            Ok(record) => {
                let record =
                    self.record_or_failed(record, &answer, REASON_STORAGE_UNAVAILABLE, now);
                self.respond(seen, Ok(result_for(record, THREAD, None)), false);
                Phase::Done
            }
            Err(e) => self.fail(seen, e),
        }
    }

    /// Decision 39: a lost acknowledgment is resolved by the run's identity; after the bounded
    /// wait the request answers the task as it stands and the resolution continues.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
        turn: Box<Turn>,
        mut poll: LanePoll,
    ) -> Phase {
        match resolve_once(db, rng, t, &turn, &self.task_id) {
            Some(Ok(committed)) => return self.answered(now, seen, &turn, committed.task),
            Some(Err(_)) => return self.rejected(now, db, seen, &turn),
            None => {}
        }
        if let Some(at) = poll.next(ms(now)) {
            self.ready_at = at.as_millis() as i64;
            return Phase::Resolve { turn, poll };
        }
        seen.unresolved += 1;
        let current = match task(db, &turn.answer_task) {
            Ok(Some(row)) => row,
            // unreadable: the row as this worker left it, picked up
            _ => TaskRow { status: TaskState::Working, ..self.submitted_row(now).row(LANE) },
        };
        self.respond(seen, Ok(result_for(current, THREAD, None)), true);
        Phase::Later { turn, poll: LanePoll::new(ms(now), RESOLVE_LATER) }
    }

    fn later(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        turn: Box<Turn>,
        mut poll: LanePoll,
    ) -> Phase {
        match resolve_once(db, rng, t, &turn, &self.task_id) {
            Some(Ok(_)) => Phase::Done,
            Some(Err(StorageError::VersionConflict { .. })) => {
                let answer = &turn.answer_task;
                let fail =
                    db::fail_orphan(LANE, answer, &failed(answer, REASON_STORAGE_UNAVAILABLE));
                let _ = db.batch(&[fail], None);
                Phase::Done
            }
            Some(Err(_)) => Phase::Done,
            None => match poll.next(ms(now)) {
                Some(at) => {
                    self.ready_at = at.as_millis() as i64;
                    Phase::Later { turn, poll }
                }
                None => Phase::Done,
            },
        }
    }

    /// `await_task` with `fail_if_free`: the task ends, or the lease frees and a `working` row
    /// (its holder died) fails `worker-lost` under a fresh lease.
    fn existing(
        &mut self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        task_id: String,
        mut poll: LanePoll,
        attempt: u32,
    ) -> Phase {
        let record = match task(db, &task_id) {
            Ok(r) => r,
            Err(e) => return self.fail(seen, e),
        };
        if let Some(r) = &record
            && r.terminal()
        {
            self.respond(seen, Ok(result_for(r.clone(), THREAD, None)), false);
            return Phase::Done;
        }
        if record.as_ref().is_some_and(|r| r.status == TaskState::Working) {
            match self.take_over(now, db, &task_id, attempt) {
                Err(e) => return self.fail(seen, e),
                Ok(true) => match task(db, &task_id) {
                    Ok(Some(r)) if r.terminal() => {
                        self.respond(seen, Ok(result_for(r, THREAD, None)), false);
                        return Phase::Done;
                    }
                    Ok(_) => {}
                    Err(e) => return self.fail(seen, e),
                },
                Ok(false) => {}
            }
        }
        match poll.next(ms(now)) {
            Some(at) => {
                self.ready_at = at.as_millis() as i64;
                Phase::Existing { task: task_id, poll, attempt: attempt + 1 }
            }
            None => {
                // not answered in time: the task as it stands
                let record = self.record_or_failed(record, &task_id, "not-written", now);
                self.respond(seen, Ok(result_for(record, THREAD, None)), false);
                Phase::Done
            }
        }
    }

    /// A run that takes a free lane without waiting, fails the dead holders' `working` rows and
    /// commits nothing else. `false` while another holder has the lane.
    fn take_over(
        &mut self,
        now: i64,
        db: &mut SimDb,
        task_id: &str,
        attempt: u32,
    ) -> Result<bool, StorageError> {
        self.tokens += 1;
        let token = format!("lease_r{}_{}", self.id, self.tokens);
        let run_id = format!("lost_r{}_{attempt}", self.id);
        let holder = format!("{}/lost", self.holder);
        let req = Acquire {
            lane: LANE,
            holder: &holder,
            token: &token,
            lease_secs: 30,
            wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
            run_id: &run_id,
            task_id,
            run_started: EventWrite::run_started(now, &RunStarted::default()),
            submit: None,
            pickup: false,
            tasks_limit: LOOKUP_LIMIT,
        };
        let acquired = match execute(db, db::acquire(&req, Vec::new()), None)? {
            AcquireOutcome::Acquired(a) => a,
            AcquireOutcome::Held { .. } => return Ok(false),
        };
        if !acquired.already_committed {
            for row in acquired.tasks.iter().rev().filter(|r| r.status == TaskState::Working) {
                let fail =
                    db::fail_orphan(LANE, &row.task_id, &failed(&row.task_id, REASON_WORKER_LOST));
                db.batch(&[fail], None)?;
            }
        }
        let fence = acquired.fence;
        if acquired.already_committed {
            db.batch(&db::release(&fence, None), None)?;
            return Ok(true);
        }
        let commit = empty_commit(&run_id, now);
        execute(
            db,
            db::commit(LANE, &commit, Some(Fenced { fence: &fence, keep: None }), Vec::new())?,
            None,
        )?;
        Ok(true)
    }
}

impl Turn {
    /// One staged row: `patch` over this run's earlier write or the loaded row.
    fn write(
        &mut self,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<(), StorageError> {
        let base = match self.writes.iter().find(|w| w.task_id == task_id) {
            Some(w) => Some(w.clone()),
            None => self.tasks.iter().find(|r| r.task_id == task_id).cloned().map(TaskWrite::from),
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

/// The run's fenced commit, its acknowledgment possibly lost before or after it applied, the
/// connection sometimes down for a while after.
fn send_commit(
    db: &mut SimDb,
    rng: &mut Rng,
    t: &Timing,
    turn: &Turn,
) -> Result<lk_agents_storage::Committed, StorageError> {
    let fault = match rng.u8(0..100) {
        n if n < t.lost_ack_pct => Some(CommitFault::LoseAckBefore),
        n if n < 2 * t.lost_ack_pct => Some(CommitFault::LoseAckAfter),
        _ => None,
    };
    let commit = turn.commit.as_ref().expect("a staged commit");
    let plan =
        db::commit(LANE, commit, Some(Fenced { fence: &turn.fence, keep: None }), Vec::new())?;
    let out = execute(db, plan, fault);
    if fault.is_some() && rng.u8(0..4) == 0 {
        // the dropped connection takes a while to come back, sometimes past the resolution wait
        db.cut_for(rng.i64(1000..20_000));
    }
    out
}

/// One resolution lookup by the run's identity: committed, rejected (`VersionConflict`), or
/// not applied (sent again); `None` while there is no authoritative answer.
fn resolve_once(
    db: &mut SimDb,
    rng: &mut Rng,
    t: &Timing,
    turn: &Turn,
    run_id: &str,
) -> Option<Result<lk_agents_storage::Committed, StorageError>> {
    let lookup = db::resolve_commit(LANE, run_id, Some(&turn.fence), Some(&turn.answer_task));
    match execute(db, lookup, None) {
        Ok(Resolution::Committed(c)) => Some(Ok(c)),
        Ok(Resolution::Rejected) => Some(Err(StorageError::VersionConflict {
            lane: LANE.to_owned(),
            expected: turn.fence.version,
        })),
        Ok(Resolution::NotApplied) => match send_commit(db, rng, t, turn) {
            Err(StorageError::Uncertain(_) | StorageError::Unavailable(_)) => None,
            out => Some(out),
        },
        Err(_) => None,
    }
}

/// A run's commit with nothing but its `run_ended`.
fn empty_commit(run_id: &str, now: i64) -> Commit {
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
