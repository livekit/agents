//! The storage simulation (decision 24, SCOPE §6): workers sharing one thread's lane, every
//! decision the storage core's, every statement run on a real SQLite in memory ([`db`]) on the
//! simulation's clock, with the faults AgentDB shows: a commit's acknowledgment lost before or
//! after it applied, windows with the database unreachable, a paused owner whose lease lapses
//! while another worker takes the lane, workers lost mid-run, and clients canceling tasks.
//!
//! The workers and the canceling client are processes of the shared scheduler ([`crate::sched`]):
//! each step it picks one of those that are due, in arrival order (the lowest id) or by BasicPOS.
//! A seed draws the setup, the faults and the sampled order; the same seed replays the same run.
//!
//! The invariants, checked over the whole run and the database it left:
//! * a stale owner never commits: every commit returned a version of its own, one per bump;
//! * every acknowledged submission is durable;
//! * a task reaches one terminal state and keeps it, and a task whose run committed is
//!   completed (or a client's cancel won);
//! * an uncertain commit resolves to the original run: one `run_ended` per run;
//! * the history is neither lost nor duplicated.

pub mod db;
pub mod worker;

use std::collections::BTreeMap;
use std::time::Duration;

use std::convert::Infallible;

use fastrand::Rng;
use lk_agents_storage::sql::Table;
use lk_agents_storage::db as plans;
use lk_agents_storage::run::{TaskRow, TaskState};
use lk_agents_storage::schema::ItemRow;
use serde_json::Map;

use crate::sched::{Sampling, Sched, Step, Tick};
use db::{Clock, SimDb};
use worker::{Worker, task_id};

pub const LANE: &str = "front-desk";

/// The timing a seed draws, calibrated to production defaults scaled down.
#[derive(Debug, Clone)]
pub struct Timing {
    pub lease_ttl: Duration,
    pub renew_interval: Duration,
    /// How long a run works between its acquire and its commit, in milliseconds.
    pub work_ms: (i64, i64),
    /// The share of commits whose acknowledgment is lost before it applied, and as many after.
    pub lost_ack_pct: u8,
}

/// What the run observed, for the invariants.
#[derive(Debug, Default)]
pub struct Observed {
    /// Tasks whose `submitted` row a worker was told is durable.
    pub acknowledged: Vec<String>,
    /// Each run a worker saw committed, with the version the commit returned.
    pub committed: Vec<(String, i64)>,
    /// Tasks whose save a worker saw rejected.
    pub rejected: Vec<String>,
    /// Commits whose acknowledgment was lost and that were resolved.
    pub resolved: u64,
    /// Each task's statuses as they changed, step by step.
    pub statuses: BTreeMap<String, Vec<TaskState>>,
}

/// A finished run: its steps and what broke.
#[derive(Debug)]
pub struct Outcome {
    pub seed: u64,
    pub steps: u64,
    pub violations: Vec<String>,
    /// What the run reached: commits, lost acknowledgments resolved, rejected saves (a stale
    /// owner fenced out, or a lane that moved on).
    pub committed: usize,
    pub resolved: u64,
    pub rejected: usize,
}

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = 0x7363_6865_64;

/// One seeded run with `workers` workers serving `messages` messages, the due processes (ids
/// `0..workers` are workers, `workers` the canceling client) stepped in the `sampling` order.
pub fn run(seed: u64, workers: usize, messages: usize, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let mut sched: Sched<Infallible, Infallible> =
        Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let timing = Timing {
        lease_ttl: Duration::from_secs(rng.u64(2..6)),
        renew_interval: Duration::from_millis(rng.u64(500..1500)),
        work_ms: (50, rng.i64(200..3000)),
        lost_ack_pct: rng.u8(0..15),
    };
    let clock: Clock = Clock::default();
    let mut db = SimDb::new(clock.clone());
    let mut seen = Observed::default();
    let mut pool: Vec<Worker> = (0..workers)
        .map(|w| Worker::new(w, (0..messages).filter(|m| m % workers == w).collect()))
        .collect();
    let canceling = workers as u64;
    let mut cancels = rng.usize(0..=messages.min(3));
    let mut cancel_at = rng.i64(0..5000);
    let mut steps = 0u64;
    let mut violations = Vec::new();
    loop {
        let now = clock.get();
        let mut due: Vec<u64> =
            pool.iter().filter(|w| !w.done() && w.wake() <= now).map(|w| w.id as u64).collect();
        if cancels > 0 && cancel_at <= now {
            due.push(canceling);
        }
        let next = pool.iter().filter(|w| !w.done()).map(Worker::wake);
        let next = next.chain((cancels > 0).then_some(cancel_at));
        let p = match sched.tick(now as u64, &due, next.map(|at| at as u64)) {
            Tick::Step(Step::Process(p)) => p,
            Tick::Step(Step::Item { .. }) => unreachable!("only processes"),
            Tick::Advance(at) => {
                clock.set(at as i64);
                continue;
            }
            Tick::Quiet => break,
        };
        steps += 1;
        if p == canceling {
            let task = task_id(rng.usize(0..messages));
            let cancel = plans::cancel(LANE, &task, &Map::new());
            let _ = db.batch(&cancel.statements, None);
            cancels -= 1;
            cancel_at = now + rng.i64(100..3000);
        } else {
            let w = &mut pool[p as usize];
            // a worker can stall (paused process) or die mid-run, as real ones do
            match rng.u16(0..1000) {
                0..3 => w.alive = false,
                3..10 => w.paused_until = now + rng.i64(0..3 * timing.lease_ttl.as_millis() as i64),
                10..14 => db.cut_for(rng.i64(100..3000)),
                _ => w.step(now, &mut db, &mut rng, &timing, &mut seen),
            }
        }
        // a database request takes a few milliseconds
        clock.set(clock.get() + rng.i64(1..15));
        observe(&db, &mut seen, &mut violations);
    }
    sweep(&mut db, &clock, &timing);
    observe(&db, &mut seen, &mut violations);
    check(&db, &seen, &mut violations);
    let (committed, resolved, rejected) =
        (seen.committed.len(), seen.resolved, seen.rejected.len());
    Outcome { seed, steps, violations, committed, resolved, rejected }
}

/// Every task's status now; a terminal status that changes is a violation at once.
fn observe(db: &SimDb, seen: &mut Observed, violations: &mut Vec<String>) {
    let rows = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    for row in rows.iter().map(|r| TaskRow::from_row(r).expect("a task row")) {
        let log = seen.statuses.entry(row.task_id.clone()).or_default();
        if log.last() == Some(&row.status) {
            continue;
        }
        if let Some(last) = log.last()
            && last.is_terminal()
        {
            violations.push(format!("{}: terminal {last:?} became {:?}", row.task_id, row.status));
        }
        log.push(row.status);
    }
}

/// After every lease lapsed, one last holder fails what dead workers left working, so every
/// task ends terminal.
fn sweep(db: &mut SimDb, clock: &Clock, timing: &Timing) {
    clock.set(clock.get() + 2 * timing.lease_ttl.as_millis() as i64 + 2000);
    let rows = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    for row in rows.iter().map(|r| TaskRow::from_row(r).expect("a task row")) {
        if row.terminal() {
            continue;
        }
        let pickup = plans::pickup(LANE, &row.task_id);
        let _ = db.batch(&pickup.statements, None);
        let failed = a2a::Message::new(a2a::Role::Agent, vec![a2a::Part::text("swept")]);
        let _ = db.batch(&[plans::fail_orphan(LANE, &row.task_id, &failed)], None);
    }
}

fn check(db: &SimDb, seen: &Observed, violations: &mut Vec<String>) {
    let tasks: BTreeMap<String, TaskRow> = db
        .inspect(&plans::tasks(LANE, None, 10_000).statement)
        .iter()
        .map(|r| TaskRow::from_row(r).expect("a task row"))
        .map(|t| (t.task_id.clone(), t))
        .collect();
    for task in &seen.acknowledged {
        if !tasks.contains_key(task) {
            violations.push(format!("{task}: acknowledged, then lost"));
        }
    }
    for task in tasks.values().filter(|t| !t.terminal()) {
        violations.push(format!("{}: ended {:?}, not terminal", task.task_id, task.status));
    }
    // one commit per version bump: a stale owner that committed would repeat a version (a
    // worker lost after its commit applied leaves a version nobody saw returned)
    let mut versions: Vec<i64> = seen.committed.iter().map(|(_, v)| *v).collect();
    versions.sort_unstable();
    let lane = db.inspect(&plans::lane_info(LANE).statement);
    let lane_version = lane.first().and_then(|r| r[1].as_i64()).unwrap_or(0);
    let mut distinct = versions.clone();
    distinct.dedup();
    if distinct.len() != versions.len() || versions.last().is_some_and(|v| *v > lane_version) {
        violations.push(format!("commits returned versions {versions:?}, lane at {lane_version}"));
    }
    let events = db.inspect(&plans::lane_events(LANE, 0).statement);
    let mut ended: BTreeMap<String, usize> = BTreeMap::new();
    for e in &events {
        let run_id = e[3].as_str().unwrap_or_default().to_owned();
        if e[4].as_str() == Some("run_ended") {
            *ended.entry(run_id).or_default() += 1;
        }
    }
    let items: Vec<String> = db
        .inspect(&plans::history(LANE).statement)
        .iter()
        .map(|r| ItemRow::from_row(r).expect("an item row").id)
        .collect();
    for (run_id, _) in &seen.committed {
        if ended.get(run_id) != Some(&1) {
            violations.push(format!("{run_id}: committed with {:?} run_ended", ended.get(run_id)));
        }
        let m = run_id.split('_').nth(1).unwrap_or_default();
        for id in [format!("u{m}"), format!("a{m}")] {
            if !items.contains(&id) {
                violations
                    .push(format!("{run_id}: committed, its item {id} is not in the history"));
            }
        }
        let task = tasks.get(&format!("AST_{m}")).map(|t| t.status);
        if !matches!(task, Some(TaskState::Completed | TaskState::Canceled)) {
            violations.push(format!("{run_id}: committed, its task is {task:?}"));
        }
    }
    for (run_id, n) in &ended {
        if *n > 1 {
            violations.push(format!("{run_id}: {n} run_ended"));
        }
    }
    let mut unique = items.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != items.len() {
        violations.push("the history holds an item twice".into());
    }
}
