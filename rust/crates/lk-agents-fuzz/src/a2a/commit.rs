//! A turn's run ends in one commit: fenced under its lease (kept past a drain's last inbox row),
//! or appended without one under `concurrent`. A lost acknowledgment is resolved by the run's
//! identity (decision 39): within the bounded wait the request is answered with the authoritative
//! outcome, after it with the task as it stands while the resolution goes on; a drain's run
//! answers the error instead (`close_run`). A rejected save fails the answer
//! `storage-unavailable`, its staged changes absent.

use std::time::Duration;

use fastrand::Rng;
use lk_agents_a2a::{TurnResult, result_for};
use lk_agents_storage::StorageError;
use lk_agents_storage::db::{self, Fenced, Plan, Resolution};
use lk_agents_storage::lease::{LanePoll, lease_secs};
use lk_agents_storage::run::{REASON_STORAGE_UNAVAILABLE, TaskRow, TaskState};

use super::door::{Phase, Request, Turn, execute, failed, ms, task};
use super::{LANE, Observed, Setup, THREAD};
use crate::storage::db::{CommitFault, SimDb};

/// The runtime's bound on resolving a lost commit acknowledgment before answering
/// (`RESOLVE_WAIT`), and on resolving it afterwards (`RESOLVE_LATER`).
const RESOLVE_WAIT: Duration = Duration::from_secs(10);
const RESOLVE_LATER: Duration = Duration::from_secs(3600);

impl Request {
    pub(super) fn commit(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        turn: Box<Turn>,
    ) -> Phase {
        match send_commit(db, rng, t, &turn) {
            Ok(committed) => self.committed(now, seen, turn, committed),
            Err(StorageError::Uncertain(_)) => {
                seen.lost_acks += 1;
                Phase::Resolve { turn, poll: LanePoll::new(ms(now), RESOLVE_WAIT) }
            }
            Err(e) if turn.drain.is_some() => self.fail(seen, e),
            Err(_) => self.rejected(now, db, seen, &turn),
        }
    }

    /// A drain's run goes on to the release; a turn answering its message answers it.
    fn committed(
        &mut self,
        now: i64,
        seen: &mut Observed,
        turn: Box<Turn>,
        committed: lk_agents_storage::Committed,
    ) -> Phase {
        match turn.drain {
            Some((_, then)) => {
                self.last_run_id = Some(turn.run_id.clone());
                let fence = turn.held.expect("a drain run holds the lease").fence;
                self.ready_at = now;
                Phase::Release { fence, then }
            }
            None => self.answered(now, seen, &turn, committed.task),
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
            run_id: Some(turn.run_id.clone()),
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
                self.answer_row(seen, record, &answer, REASON_STORAGE_UNAVAILABLE, now, None)
            }
            Err(e) => self.fail(seen, e),
        }
    }

    /// Decision 39: a lost acknowledgment is resolved by the run's identity; after the bounded
    /// wait the request answers the task as it stands and the resolution continues. A drain's
    /// run answers the error instead (`close_run`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        seen: &mut Observed,
        turn: Box<Turn>,
        mut poll: LanePoll,
    ) -> Phase {
        match resolve_once(db, rng, t, &turn) {
            Some(Ok(committed)) => return self.committed(now, seen, turn, committed),
            Some(Err(e)) if turn.drain.is_some() => return self.fail(seen, e),
            Some(Err(_)) => return self.rejected(now, db, seen, &turn),
            None => {}
        }
        if let Some(at) = poll.next(ms(now)) {
            self.ready_at = at.as_millis() as i64;
            return Phase::Resolve { turn, poll };
        }
        seen.unresolved += 1;
        if turn.drain.is_some() {
            let why = format!("run {} unresolved", turn.run_id);
            return self.fail(seen, StorageError::Uncertain(why));
        }
        let current = match task(db, &turn.answer_task) {
            Ok(Some(row)) => row,
            // unreadable: the row as this worker left it, picked up
            _ => TaskRow { status: TaskState::Working, ..self.submitted_row(now).row(LANE) },
        };
        self.respond(seen, Ok(result_for(current, THREAD, None)), true);
        Phase::Later { turn, poll: LanePoll::new(ms(now), RESOLVE_LATER) }
    }

    pub(super) fn later(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Setup,
        turn: Box<Turn>,
        mut poll: LanePoll,
    ) -> Phase {
        match resolve_once(db, rng, t, &turn) {
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
}

impl Turn {
    /// The commit's plan: fenced under the lease, keeping it past a drain's last row; an
    /// append without one.
    fn plan(&self, t: &Setup) -> Result<Plan<db::CommitDecode>, StorageError> {
        let commit = self.commit.as_ref().expect("a staged commit");
        let keep = self.drain.as_ref().map(|_| lease_secs(t.lease_ttl));
        let fenced = self.held.as_ref().map(|h| Fenced { fence: &h.fence, keep });
        db::commit(LANE, commit, fenced, Vec::new())
    }
}

/// The run's commit, its acknowledgment possibly lost before or after it applied, the
/// connection sometimes down for a while after.
fn send_commit(
    db: &mut SimDb,
    rng: &mut Rng,
    t: &Setup,
    turn: &Turn,
) -> Result<lk_agents_storage::Committed, StorageError> {
    let fault = match rng.u8(0..100) {
        n if n < t.lost_ack_pct => Some(CommitFault::LoseAckBefore),
        n if n < 2 * t.lost_ack_pct => Some(CommitFault::LoseAckAfter),
        _ => None,
    };
    let out = execute(db, turn.plan(t)?, fault);
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
    t: &Setup,
    turn: &Turn,
) -> Option<Result<lk_agents_storage::Committed, StorageError>> {
    let fence = turn.held.as_ref().map(|h| &h.fence);
    let lookup = db::resolve_commit(LANE, &turn.run_id, fence, Some(&turn.answer_task));
    match execute(db, lookup, None) {
        Ok(Resolution::Committed(c)) => Some(Ok(c)),
        Ok(Resolution::Rejected) => Some(Err(StorageError::VersionConflict {
            lane: LANE.to_owned(),
            expected: fence.map_or(0, |f| f.version),
        })),
        Ok(Resolution::NotApplied) => match send_commit(db, rng, t, turn) {
            Err(StorageError::Uncertain(_) | StorageError::Unavailable(_)) => None,
            out => Some(out),
        },
        Err(_) => None,
    }
}
