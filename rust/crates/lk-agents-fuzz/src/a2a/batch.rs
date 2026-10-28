//! The `batch` policy at the door (`Turn::batch`, `drain`, `await_task`), one plan per step: a
//! message goes into the lane's inbox in one atomic receive that also tries the lease. The
//! request that took it is the runner: it lets the burst land (the debounce), then drains, one
//! run per pass over the pending rows, keeping the lease past each pass's last row and releasing
//! it only once nothing arrived meanwhile. The others wait on their task and drain themselves
//! once the lane is free with rows pending (the runner died). A resent message awaits the task
//! the first receive created.

use std::time::Duration;

use lk_agents_core::a2a::{Message, result_for};
use lk_agents_storage::db::{
    self, AcquireOutcome, Acquired, EventWrite, Fenced, Receive, ReceiveInput,
};
use lk_agents_storage::lease::{LanePoll, lease_secs};
use lk_agents_storage::run::{
    LEASE_WAIT_TIMEOUT, LOOKUP_LIMIT, REASON_WORKER_LOST, RunOutcome, RunUsage, RunWrites,
    TaskState, run_commit,
};
use lk_agents_storage::schema::{InboxRow, RunStarted, TaskData};
use lk_agents_storage::{Fence, StorageError};

use super::door::{Opened, Phase, Request, empty_commit, execute, failed, ms, read, task};
use super::{LANE, Observed, Setup, THREAD};
use crate::storage::db::SimDb;

/// What follows a drain: the runner answers its own task; a waiting request goes on waiting.
pub(crate) enum After {
    Answer,
    Await(Await),
}

/// `await_task`: polls the task until it ends; meanwhile drains a free lane with rows pending
/// (`drain_if_free`) or fails the dead holder's `working` row (`fail_if_free`).
pub(crate) struct Await {
    task: String,
    poll: LanePoll,
    attempt: u32,
    drain_if_free: bool,
    fail_if_free: bool,
    /// This pass already drained: on to the orphan check.
    drained: bool,
}

impl Await {
    pub fn new(task: String, now: i64, drain_if_free: bool, fail_if_free: bool) -> Self {
        let poll = LanePoll::new(ms(now), LEASE_WAIT_TIMEOUT);
        Await { task, poll, attempt: 0, drain_if_free, fail_if_free, drained: false }
    }
}

impl Request {
    /// The receive: the inbox row, the `submitted` row and the lease attempt in one batch.
    pub(super) fn receive(
        &mut self,
        now: i64,
        db: &mut SimDb,
        t: &Setup,
        seen: &mut Observed,
        debounce: Duration,
    ) -> Phase {
        self.tokens += 1;
        let token = format!("lease_r{}_{}", self.id, self.tokens);
        let holder = format!("{}/{token}", self.holder);
        let msg = match Message::from_a2a(self.message(), now as f64 / 1000.0, None) {
            Ok(m) => m.to_a2a(),
            Err(e) => return self.fail(seen, e),
        };
        let input = ReceiveInput {
            task_id: &self.task_id,
            message: &msg,
            data: &TaskData { history: vec![msg.clone()], ..TaskData::default() },
            internal: None,
        };
        let plan = db::receive(LANE, &holder, &token, lease_secs(t.lease_ttl), input);
        match execute(db, plan, None) {
            Err(e) => self.fail(seen, e),
            Ok(Receive::Duplicate(existing)) => {
                seen.deduplicated += 1;
                let existing = existing.unwrap_or_else(|| self.task_id.clone());
                match task(db, &existing) {
                    Ok(Some(r)) if !r.terminal() => {
                        // the first receive may have landed with its answer lost: nobody
                        // drains the row unless this request does once the lease frees
                        Phase::Await(Await::new(r.task_id, now, true, true))
                    }
                    Ok(r) => self.answer_row(seen, r, &existing, "not-written", now, None),
                    Err(e) => self.fail(seen, e),
                }
            }
            Ok(Receive::Received(version)) => {
                seen.received.push((self.id, msg.message_id));
                match version {
                    Some(version) => {
                        let fence = Fence { lane: LANE.to_owned(), holder, token, version };
                        self.ready_at = now + debounce.as_millis() as i64;
                        Phase::Drain { lease: Some(fence), then: After::Answer }
                    }
                    None => Phase::Await(Await::new(self.task_id.clone(), now, true, false)),
                }
            }
        }
    }

    /// One pass of the drain: the pending rows, a run over them under the kept lease or a
    /// fresh one taken without waiting, each row picked up; a row a dead drain had fails
    /// `worker-lost`. Nothing pending: the lease released, and the drain is over.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn drain(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut fastrand::Rng,
        t: &Setup,
        seen: &mut Observed,
        lease: Option<Fence>,
        then: After,
    ) -> Phase {
        let rows = match read(db, db::inbox(LANE, true)) {
            Ok(rows) => rows,
            Err(e) => return self.fail(seen, e),
        };
        let Some(last) = rows.last() else {
            if let Some(fence) = &lease
                && let Err(e) = db.batch(&[db::release_if_drained(fence)], None)
            {
                return self.fail(seen, e);
            }
            return self.after_drain(now, db, seen, then);
        };
        let (last_id, last_task) = (last.id, last.task_id.clone());
        let run_id = format!("drain_r{}_{}", self.id, self.tokens + 1);
        let holder = match &lease {
            Some(fence) => fence.holder.clone(),
            None => format!("{}/{run_id}", self.holder),
        };
        let acquired = match self.try_acquire(now, db, t, &run_id, &last_task, &holder, None, false)
        {
            Ok(AcquireOutcome::Acquired(a)) => *a,
            // a runner holds the lane: a waiting request goes on waiting
            Ok(AcquireOutcome::Held { .. }) => {
                return match then {
                    After::Await(a) => self.after_drain(now, db, seen, After::Await(a)),
                    After::Answer => self.fail(seen, StorageError::LeaseHeld(LANE.to_owned())),
                };
            }
            Err(e) => return self.fail(seen, e),
        };
        let fence = acquired.fence.clone();
        let (batch, task_ids) = match pick_up(db, &rows) {
            Ok(picked) => picked,
            Err(e) => return self.abandon(now, db, seen, &fence, &run_id, true, e),
        };
        if batch.is_empty() {
            // every row was answered or failed: the pass commits only its progress
            let commit = drain_commit(&run_id, &last_task, last_id, now);
            let keep = Some(Fenced { fence: &fence, keep: Some(lease_secs(t.lease_ttl)) });
            let plan = match db::commit(LANE, &commit, keep, Vec::new()) {
                Ok(plan) => plan,
                Err(e) => return self.abandon(now, db, seen, &fence, &run_id, true, e),
            };
            return match execute(db, plan, None) {
                Ok(_) => Phase::Release { fence, then },
                Err(e) => self.fail(seen, e),
            };
        }
        let all: Vec<String> = rows.iter().map(|r| r.task_id.clone()).collect();
        if let Err(e) = self.fail_orphans(db, &acquired.tasks, &all) {
            return self.abandon(now, db, seen, &fence, &run_id, true, e);
        }
        let opened =
            Opened { drain: Some((last_id, then)), ..Opened::held(run_id, acquired, t, now) };
        self.open_turn(now, db, rng, t, seen, opened, batch, task_ids)
    }

    /// A drain pass committed keeping the lease: released when no row arrived meanwhile, else
    /// another pass.
    pub(super) fn release_drained(
        &mut self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        fence: Fence,
        then: After,
    ) -> Phase {
        match db.batch(&[db::release_if_drained(&fence)], None) {
            Ok(out) if out.results[0].rows_affected() > 0 => self.after_drain(now, db, seen, then),
            Ok(_) => Phase::Drain { lease: Some(fence), then },
            Err(e) => self.fail(seen, e),
        }
    }

    fn after_drain(&mut self, now: i64, db: &mut SimDb, seen: &mut Observed, then: After) -> Phase {
        match then {
            After::Answer => match task(db, &self.task_id) {
                Ok(record) => {
                    let (task_id, run_id) = (self.task_id.clone(), self.last_run_id.clone());
                    self.answer_row(seen, record, &task_id, "not-written", now, run_id.as_deref())
                }
                Err(e) => self.fail(seen, e),
            },
            After::Await(mut a) => {
                a.drained = true;
                Phase::Await(a)
            }
        }
    }

    /// One pass of `await_task`.
    pub(super) fn await_task(
        &mut self,
        now: i64,
        db: &mut SimDb,
        seen: &mut Observed,
        mut a: Await,
    ) -> Phase {
        let record = match task(db, &a.task) {
            Ok(r) => r,
            Err(e) => return self.fail(seen, e),
        };
        if let Some(r) = &record
            && r.terminal()
        {
            self.respond(seen, Ok(result_for(r.clone(), THREAD, None)), false);
            return Phase::Done;
        }
        if a.drain_if_free && !std::mem::take(&mut a.drained) {
            match read(db, db::inbox(LANE, true)) {
                Ok(rows) if !rows.is_empty() => {
                    return Phase::Drain { lease: None, then: After::Await(a) };
                }
                Ok(_) => {}
                Err(e) => return self.fail(seen, e),
            }
        }
        if a.fail_if_free && record.as_ref().is_some_and(|r| r.status == TaskState::Working) {
            match self.take_over(now, db, &a.task, a.attempt) {
                Err(e) => return self.fail(seen, e),
                Ok(true) => match task(db, &a.task) {
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
        match a.poll.next(ms(now)) {
            Some(at) => {
                self.ready_at = at.as_millis() as i64;
                a.attempt += 1;
                Phase::Await(a)
            }
            // not answered in time: the task as it stands
            None => self.answer_row(seen, record, &a.task.clone(), "not-written", now, None),
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
        let req = db::Acquire {
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
        let acquired: Acquired = match execute(db, db::acquire(&req, Vec::new()), None)? {
            AcquireOutcome::Acquired(a) => *a,
            AcquireOutcome::Held { .. } => return Ok(false),
        };
        let fence = acquired.fence.clone();
        if acquired.already_committed {
            db.batch(&db::release(&fence, None), None)?;
            return Ok(true);
        }
        self.fail_orphans(db, &acquired.tasks, &[])?;
        let commit = empty_commit(&run_id, now);
        execute(
            db,
            db::commit(LANE, &commit, Some(Fenced { fence: &fence, keep: None }), Vec::new())?,
            None,
        )?;
        Ok(true)
    }
}

/// Each pending row picked up into the batch; a row already terminal (an earlier drain answered
/// it) is skipped, one a dead drain had (`working`) fails `worker-lost`.
fn pick_up(db: &mut SimDb, rows: &[InboxRow]) -> Result<(Vec<Message>, Vec<String>), StorageError> {
    let mut batch = Vec::new();
    let mut task_ids = Vec::new();
    for row in rows {
        let Some(was) = execute(db, db::pickup(LANE, &row.task_id), None)? else { continue };
        if was == TaskState::Working {
            let fail =
                db::fail_orphan(LANE, &row.task_id, &failed(&row.task_id, REASON_WORKER_LOST));
            db.batch(&[fail], None)?;
            continue;
        }
        batch.push(Message::from_a2a(&row.message, row.received_at as f64, Some(&row.task_id))?);
        task_ids.push(row.task_id.clone());
    }
    Ok((batch, task_ids))
}

/// A drain pass that ran no turn: its `run_ended` and the inbox progress.
fn drain_commit(run_id: &str, task_id: &str, drained_id: i64, now: i64) -> db::Commit {
    run_commit(RunWrites {
        run_id,
        task_id,
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
        drained_id: Some(drained_id),
        answer_task: None,
    })
    .expect("an empty exclusive commit")
}
