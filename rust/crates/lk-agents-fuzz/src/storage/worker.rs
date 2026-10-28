//! One simulated worker: the synchronous steps a worker's runtime takes for each message it
//! serves, every decision the storage core's (`lk_agents_storage::db` plans, `run_commit`,
//! `Renewal`, `LanePoll`, the commit `Resolution`), every statement executed on [`SimDb`]. The
//! steps mirror `lk_agents_runtime::storage` and its A2A turn, so a decision the cores get wrong
//! shows here as a broken invariant.

use std::time::Duration;

use fastrand::Rng;
use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use livekit_protocol::agent::{ChatMessage, ChatRole};
use lk_agents_storage::db::{self, Acquire, AcquireOutcome, Commit, Decode, EventWrite, Fenced, Plan, Resolution};
use lk_agents_storage::TaskWrite;
use lk_agents_storage::lease::{LanePoll, Renewal, Renewed, lease_secs};
use lk_agents_storage::run::{
    ItemRow, RunOutcome, RunUsage, RunWrites, TaskPatch, TaskRow, TaskState, merge_write,
    run_commit,
};
use lk_agents_storage::schema::{RunStarted, TaskData};
use lk_agents_storage::{Fence, StorageError};

use super::db::{CommitFault, SimDb};
use super::{LANE, Observed, Timing};

/// Where a worker is with its current message.
enum Phase {
    Submit,
    Acquire { poll: LanePoll },
    Work { until: i64, held: Held, commit: Box<Commit> },
    Commit { held: Held, commit: Box<Commit> },
    Resolve { held: Held, commit: Box<Commit> },
    Done,
}

/// The lane as this worker took it.
struct Held {
    fence: Fence,
    renewal: Renewal,
}

pub struct Worker {
    pub id: usize,
    holder: String,
    messages: Vec<usize>,
    current: usize,
    phase: Phase,
    /// When this worker's next step is due.
    ready_at: i64,
    pub alive: bool,
    /// The worker stalls (a paused process) until then: no step, no renewal.
    pub paused_until: i64,
    tokens: u64,
}

/// A message's task id.
pub fn task_id(m: usize) -> String {
    format!("AST_{m}")
}

/// What a dead holder's orphan or a rejected save is failed with.
fn status_message(task_id: &str) -> a2a::Message {
    a2a::Message {
        message_id: format!("msg_{task_id}"),
        task_id: Some(task_id.to_owned()),
        ..a2a::Message::new(a2a::Role::Agent, vec![a2a::Part::text("the task failed")])
    }
}

fn item(id: String, role: ChatRole) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id,
        role: role as i32,
        ..Default::default()
    }))
}

fn execute<D: Decode>(
    db: &mut SimDb,
    plan: Plan<D>,
    fault: Option<CommitFault>,
) -> Result<D::Output, StorageError> {
    let out = db.batch(&plan.statements, fault)?;
    plan.decoder.decode(&out)
}

impl Worker {
    pub fn new(id: usize, messages: Vec<usize>) -> Self {
        Self {
            id,
            holder: format!("w{id}"),
            messages,
            current: 0,
            phase: Phase::Submit,
            ready_at: 0,
            alive: true,
            paused_until: 0,
            tokens: 0,
        }
    }

    pub fn done(&self) -> bool {
        !self.alive || self.current >= self.messages.len()
    }

    /// When this worker's next step is due: its work's end or its lease's renewal, whichever
    /// comes first, never before a pause ends.
    pub fn wake(&self) -> i64 {
        let due = match &self.phase {
            Phase::Work { until, held, .. } => {
                (*until).min(held.renewal.wake_at().as_millis() as i64)
            }
            _ => self.ready_at,
        };
        due.max(self.paused_until)
    }

    /// One step at `now`.
    pub fn step(
        &mut self,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
    ) {
        let m = self.messages[self.current];
        self.ready_at = now;
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        self.phase = match phase {
            Phase::Submit => self.submit(m, now, db, seen),
            Phase::Acquire { poll } => self.acquire(m, poll, now, db, rng, t),
            Phase::Work { until, held, commit } => self.work(until, held, commit, now, db, t),
            Phase::Commit { held, commit } => self.commit(m, held, commit, db, rng, t, seen),
            Phase::Resolve { held, commit } => self.resolve(m, held, commit, db, seen),
            Phase::Done => Phase::Done,
        };
        if matches!(self.phase, Phase::Done) {
            self.current += 1;
            self.phase = Phase::Submit;
        }
    }

    /// The message's `submitted` row, durable before the worker waits for the lane.
    fn submit(&mut self, m: usize, now: i64, db: &mut SimDb, seen: &mut Observed) -> Phase {
        let row = TaskWrite {
            task_id: task_id(m),
            status: TaskState::Submitted,
            message_id: Some(format!("m{m}")),
            name: None,
            data: TaskData::default(),
            internal: None,
        };
        match db.batch(&[db::submit(LANE, &row)], None) {
            Ok(out) => {
                if out.results[0].rows_affected() > 0 {
                    seen.acknowledged.push(task_id(m));
                }
                let start = Duration::from_millis(now as u64);
                Phase::Acquire { poll: LanePoll::new(start, Duration::from_secs(60)) }
            }
            Err(_) => Phase::Submit,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn acquire(
        &mut self,
        m: usize,
        mut poll: LanePoll,
        now: i64,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
    ) -> Phase {
        self.tokens += 1;
        let token = format!("lease_{}_{}", self.id, self.tokens);
        let run_id = format!("run_{m}_{}", self.id);
        let task = task_id(m);
        let req = Acquire {
            lane: LANE,
            holder: &self.holder,
            token: &token,
            lease_secs: lease_secs(t.lease_ttl),
            wait_ms: 60_000,
            run_id: &run_id,
            task_id: &task,
            run_started: EventWrite::run_started(now, &RunStarted::default()),
            submit: None,
            pickup: true,
            tasks_limit: 500,
        };
        let acquired = match execute(db, db::acquire(&req, Vec::new()), None) {
            Ok(AcquireOutcome::Acquired(a)) => a,
            Ok(AcquireOutcome::Held { .. }) | Err(_) => {
                return match poll.next(Duration::from_millis(now as u64)) {
                    Some(at) => {
                        self.ready_at = at.as_millis() as i64;
                        Phase::Acquire { poll }
                    }
                    // lease-timeout: the row stays submitted for the sweeper to settle
                    None => Phase::Done,
                };
            }
        };
        let fence = acquired.fence.clone();
        if acquired.already_committed || !acquired.picked_up {
            let _ = db.batch(&db::release(&fence, None), None);
            return Phase::Done;
        }
        // a dead holder's working rows fail worker-lost under this lease, as a turn's do
        for row in acquired.tasks.iter().filter(|r| r.status == TaskState::Working) {
            if row.task_id != task {
                let fail = db::fail_orphan(LANE, &row.task_id, &status_message(&row.task_id));
                let _ = db.batch(&[fail], None);
            }
        }
        let commit = compose(m, &run_id, &acquired.items, &acquired.tasks, now);
        let renewal =
            Renewal::new(t.lease_ttl, t.renew_interval, Duration::from_millis(now as u64));
        let until = now + rng.i64(t.work_ms.0..t.work_ms.1);
        Phase::Work { until, held: Held { fence, renewal }, commit: Box::new(commit) }
    }

    /// Working under the lease: commit once the work ended (a paused owner resumes straight into
    /// its commit, its lease possibly gone), else renew when due.
    fn work(
        &mut self,
        until: i64,
        mut held: Held,
        commit: Box<Commit>,
        now: i64,
        db: &mut SimDb,
        t: &Timing,
    ) -> Phase {
        if now >= until {
            return Phase::Commit { held, commit };
        }
        let extend = db::extend(&held.fence, lease_secs(t.lease_ttl));
        let outcome = db.batch(&[extend], None).map(|out| out.results[0].rows_affected() > 0);
        let now = Duration::from_millis(now as u64);
        match held.renewal.on_extend(now, outcome.as_ref().map(|r| *r)) {
            Renewed::Extended | Renewed::Retry => Phase::Work { until, held, commit },
            Renewed::Lost(_) => {
                // observed lease loss cancels the run: nothing is committed
                let _ = db.batch(&db::release(&held.fence, None), None);
                Phase::Done
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        m: usize,
        held: Held,
        commit: Box<Commit>,
        db: &mut SimDb,
        rng: &mut Rng,
        t: &Timing,
        seen: &mut Observed,
    ) -> Phase {
        let fault = match rng.u8(0..100) {
            n if n < t.lost_ack_pct => Some(CommitFault::LoseAckBefore),
            n if n < 2 * t.lost_ack_pct => Some(CommitFault::LoseAckAfter),
            _ => None,
        };
        let plan =
            db::commit(LANE, &commit, Some(Fenced { fence: &held.fence, keep: None }), Vec::new())
                .expect("a commit plan");
        match execute(db, plan, fault) {
            Ok(done) => {
                seen.committed.push((commit.run_id.clone(), done.version));
                Phase::Done
            }
            Err(StorageError::Uncertain(_)) => Phase::Resolve { held, commit },
            // not applied: the same fenced commit again
            Err(StorageError::Unavailable(_)) => Phase::Commit { held, commit },
            Err(_) => self.rejected(m, db, seen),
        }
    }

    /// A lost acknowledgment, resolved by the run's identity before anything is reported.
    fn resolve(
        &mut self,
        m: usize,
        held: Held,
        commit: Box<Commit>,
        db: &mut SimDb,
        seen: &mut Observed,
    ) -> Phase {
        seen.resolved += 1;
        let lookup = db::resolve_commit(LANE, &commit.run_id, Some(&held.fence), None);
        match execute(db, lookup, None) {
            Ok(Resolution::Committed(done)) => {
                seen.committed.push((commit.run_id.clone(), done.version));
                Phase::Done
            }
            Ok(Resolution::NotApplied) => Phase::Commit { held, commit },
            Ok(Resolution::Rejected) => self.rejected(m, db, seen),
            Err(_) => Phase::Resolve { held, commit },
        }
    }

    /// A rejected save fails the original task `storage-unavailable`, its staged changes absent.
    fn rejected(&mut self, m: usize, db: &mut SimDb, seen: &mut Observed) -> Phase {
        seen.rejected.push(task_id(m));
        let fail = db::fail_orphan(LANE, &task_id(m), &status_message(&task_id(m)));
        let _ = db.batch(&[fail], None);
        Phase::Done
    }
}

/// The run's one commit: the user's item and the answer after what was loaded, the task
/// completed.
fn compose(m: usize, run_id: &str, loaded: &[ItemRow], tasks: &[TaskRow], now: i64) -> Commit {
    let mut history = loaded.to_vec();
    history.push(item(format!("u{m}"), ChatRole::User));
    history.push(item(format!("a{m}"), ChatRole::Assistant));
    let loaded_ids: Vec<(String, ChatItem)> =
        loaded.iter().map(|i| (i.id.clone(), i.payload.clone())).collect();
    let base = tasks.iter().find(|r| r.task_id == task_id(m)).cloned().map(TaskWrite::from);
    let done = merge_write(&task_id(m), base.as_ref(), TaskState::Completed, TaskPatch::default())
        .expect("a working row");
    run_commit(RunWrites {
        run_id,
        task_id: &task_id(m),
        channel: None,
        exclusive: true,
        history: &history,
        loaded: &loaded_ids,
        state: None,
        loaded_state: None,
        events: &[],
        tasks: std::slice::from_ref(&done),
        usage: RunUsage::default(),
        outcome: RunOutcome::Completed,
        reason: None,
        now_ms: now,
        drained_id: None,
        answer_task: None,
    })
    .expect("an exclusive commit")
}
