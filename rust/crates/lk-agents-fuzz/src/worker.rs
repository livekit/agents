//! The worker's process pool (`lk_agents_worker::ipc::pool::PoolState`, decision 24, K-2): job
//! requests admitted, reserved on warmed processes, launched, cancelled and drained, over fake job
//! processes that spawn, initialize, crash, hang, exit after their job and are reaped as real ones
//! are -- spawns that fail, initializations that crash or never finish, a launch that finds its
//! process dead, exit watches and reaps that fail and are retried, closes that settle late.
//!
//! The driver mirrors the runtime's pool driver: it reads each request's reservation after every
//! step, launches a reserved job, forgets a settled request, and reports the cleanup complete
//! once the pool holds no process. Everything a fake reports reaches the shared scheduler once
//! due; virtual time moves only when nothing is pending.
//!
//! The rules, over the trace:
//! * `worker_admitted_requests_settle`: every admitted request completes, fails or is
//!   cancelled; a close fails what had not completed;
//! * `worker_one_disposition_per_assignment`: a process serves one request, a request completes
//!   once, and a job never runs twice at once (a duplicate is refused while it runs);
//! * `worker_retired_processes_stay_retired`: no stale event revives a process: once retiring
//!   it never serves again, and once released it is never named again;
//! * `worker_processes_are_released`: every process created is released or reported in the
//!   close's missing cleanup, and the close completes exactly once when nothing is left.

pub mod registration;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use fastrand::Rng;
use lk_agents_worker::ipc::pool::{
    PoolEffect, PoolInput, PoolJoinOutcome, PoolLimits, PoolState, ProcessId, ProcessPhase,
    ReservationId, ReservationState,
};

use crate::inference::Timeline;
use crate::sched::{Sampling, Sched, Step as Picked};

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = 0x7363_6865_64;

/// Who reports: one job process (its OS side), or the worker's driver and the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Proc(u64),
    Driver,
}

impl crate::sched::Source for Src {}

/// What the simulation delivers to the pool, recorded as sent.
#[derive(Debug, Clone, PartialEq)]
pub enum In {
    Start,
    Admit(u64, String),
    Created(u64),
    Ready(u64),
    SpawnFailed(u64),
    LaunchSucceeded(u64, u64),
    LaunchFailed(u64, u64),
    Exited(u64),
    ExitFailed(u64),
    Closed(u64),
    Joined(u64, bool),
    JoinFailed(u64),
    Cancel(u64),
    Forget(u64),
    Drain,
    Close,
    CleanupComplete,
    Wake,
}

impl In {
    fn input(&self) -> PoolInput {
        let (p, r) = (ProcessId, ReservationId);
        match self.clone() {
            In::Start => PoolInput::Start,
            In::Admit(req, job_id) => PoolInput::Admit { request: r(req), job_id },
            In::Created(id) => PoolInput::Created(p(id)),
            In::Ready(id) => PoolInput::Ready(p(id)),
            In::SpawnFailed(id) => PoolInput::SpawnFailed(p(id)),
            In::LaunchSucceeded(req, id) => {
                PoolInput::LaunchSucceeded { request: r(req), process: p(id) }
            }
            In::LaunchFailed(req, id) => {
                PoolInput::LaunchFailed { request: r(req), process: p(id) }
            }
            In::Exited(id) => PoolInput::Exited(p(id)),
            In::ExitFailed(id) => PoolInput::ExitFailed(p(id)),
            In::Closed(id) => PoolInput::Closed(p(id)),
            In::Joined(id, reaped) => PoolInput::Joined {
                process: p(id),
                outcome: if reaped {
                    PoolJoinOutcome::Reaped
                } else {
                    PoolJoinOutcome::NotLaunched
                },
            },
            In::JoinFailed(id) => PoolInput::JoinFailed(p(id)),
            In::Cancel(req) => PoolInput::Cancel(r(req)),
            In::Forget(req) => PoolInput::Forget(r(req)),
            In::Drain => PoolInput::Drain,
            In::Close => PoolInput::Close,
            In::CleanupComplete => PoolInput::CleanupComplete,
            In::Wake => PoolInput::Wake,
        }
    }
}

/// One delivered input, the effects it returned, and the pool as the driver then reads it: each
/// admitted request's reservation and each created process's phase.
#[derive(Debug)]
pub struct Step {
    pub at: u64,
    pub input: In,
    pub effects: Vec<PoolEffect>,
    pub reservations: BTreeMap<u64, Option<ReservationState>>,
    pub phases: BTreeMap<u64, ProcessPhase>,
}

#[derive(Debug, Default)]
pub struct Trace {
    pub steps: Vec<Step>,
    /// Each admitted request's job id.
    pub jobs: BTreeMap<u64, String>,
}

#[derive(Debug)]
pub struct Outcome {
    pub seed: u64,
    pub violations: Vec<String>,
    pub reached: BTreeSet<&'static str>,
    pub trace: Trace,
}

type Rule = (&'static str, fn(&Trace) -> Vec<String>);

pub const RULES: &[Rule] = &[
    ("worker_admitted_requests_settle", settle),
    ("worker_one_disposition_per_assignment", one_disposition),
    ("worker_retired_processes_stay_retired", no_revival),
    ("worker_processes_are_released", released),
];

pub fn check(trace: &Trace) -> Vec<String> {
    RULES
        .iter()
        .flat_map(|(name, rule)| rule(trace).into_iter().map(move |v| format!("{name}: {v}")))
        .collect()
}

/// One job process as the OS sees it.
#[derive(Default)]
struct Proc {
    spawned: bool,
    /// When it exited, once it has.
    exited: Option<u64>,
    /// It never finishes initializing.
    hangs: bool,
    /// The pool asked it to close.
    closing: bool,
    /// What waits on its exit: the exit watch, the reap, the close.
    on_exit: Vec<In>,
}

/// The job processes, and what they report.
struct Os {
    rng: Rng,
    procs: BTreeMap<u64, Proc>,
}

impl Os {
    fn create(&mut self, id: u64, now: u64, timeline: &mut Timeline<Src, In>) {
        let rng = &mut self.rng;
        let mut proc = Proc::default();
        let at = now + rng.u64(5..50);
        if rng.u8(0..100) < 3 {
            timeline.at(at, Src::Proc(id), In::SpawnFailed(id));
        } else {
            proc.spawned = true;
            timeline.at(at, Src::Proc(id), In::Created(id));
            match rng.u8(0..100) {
                // It crashes while it loads its plugins.
                0..4 => Self::exit_at(&mut proc, id, at + rng.u64(50..2_000), timeline),
                4..6 => proc.hangs = true,
                _ => timeline.at(at + rng.u64(100..3_000), Src::Proc(id), In::Ready(id)),
            }
        }
        self.procs.insert(id, proc);
    }

    /// The process exits at `at`: whatever waits on its exit is told.
    fn exit_at(proc: &mut Proc, id: u64, at: u64, timeline: &mut Timeline<Src, In>) {
        if proc.exited.is_some() {
            return;
        }
        proc.exited = Some(at);
        for waiting in proc.on_exit.drain(..) {
            timeline.at(at, Src::Proc(id), waiting);
        }
    }

    /// `then` once the process has exited (at once if it has, or never spawned).
    fn on_exit(&mut self, id: u64, now: u64, then: In, timeline: &mut Timeline<Src, In>) {
        let proc = self.procs.get_mut(&id).expect("a created process");
        match proc.exited {
            Some(at) => timeline.at(now.max(at) + self.rng.u64(0..5), Src::Proc(id), then),
            None if !proc.spawned => timeline.at(now + self.rng.u64(0..5), Src::Proc(id), then),
            None => proc.on_exit.push(then),
        }
    }

    fn effect(&mut self, effect: &PoolEffect, now: u64, timeline: &mut Timeline<Src, In>) {
        match *effect {
            PoolEffect::Create(ProcessId(id)) => self.create(id, now, timeline),
            PoolEffect::WatchExit(ProcessId(id)) if self.rng.u8(0..100) < 2 => {
                timeline.at(now + 10, Src::Proc(id), In::ExitFailed(id));
            }
            PoolEffect::WatchExit(ProcessId(id)) => self.on_exit(id, now, In::Exited(id), timeline),
            PoolEffect::Join(ProcessId(id)) if self.rng.u8(0..100) < 2 => {
                timeline.at(now + 10, Src::Proc(id), In::JoinFailed(id));
            }
            PoolEffect::Join(ProcessId(id)) => {
                let spawned = self.procs[&id].spawned;
                self.on_exit(id, now, In::Joined(id, spawned), timeline);
            }
            // The process is asked to shut down; it exits a moment later (a hung one is killed).
            PoolEffect::Close(ProcessId(id)) => {
                let mut proc = self.procs.remove(&id).expect("a created process");
                if !proc.closing && proc.spawned && proc.exited.is_none() {
                    proc.closing = true;
                    let at = now + self.rng.u64(0..800);
                    Self::exit_at(&mut proc, id, at, timeline);
                }
                self.procs.insert(id, proc);
                self.on_exit(id, now, In::Closed(id), timeline);
            }
            _ => {}
        }
    }

    /// A reserved job is handed to its process: it runs, then the process exits; a process that
    /// died first fails the launch.
    fn launch(&mut self, request: u64, id: u64, now: u64, timeline: &mut Timeline<Src, In>) {
        let at = now + self.rng.u64(1..20);
        let mut proc = self.procs.remove(&id).expect("a created process");
        if proc.exited.is_some_and(|exit| exit <= at) || self.rng.u8(0..100) < 3 {
            timeline.at(at, Src::Proc(id), In::LaunchFailed(request, id));
        } else {
            timeline.at(at, Src::Proc(id), In::LaunchSucceeded(request, id));
            let ends = at + self.rng.u64(1_000..20_000);
            Self::exit_at(&mut proc, id, ends, timeline);
        }
        self.procs.insert(id, proc);
    }
}

pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let limits = PoolLimits {
        idle_processes: rng.usize(0..4),
        concurrent_initializations: rng.usize(1..4),
        pending_jobs: rng.usize(1..12),
        cleanup_timeout: Duration::from_secs(rng.u64(1..10)),
    };
    let mut pool = PoolState::new(limits);
    let mut sched: Sched<Src, In> = Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let mut timeline = Timeline::default();
    let mut os = Os { rng: Rng::with_seed(rng.u64(..)), procs: BTreeMap::new() };
    let mut trace = Trace::default();

    timeline.at(0, Src::Driver, In::Start);
    let jobs = rng.u64(3..25);
    let mut job_ids: Vec<String> = Vec::new();
    for request in 0..jobs {
        let at = rng.u64(0..60_000);
        // The server dispatches the same job twice now and then.
        let job = if !job_ids.is_empty() && rng.u8(0..100) < 10 {
            job_ids[rng.usize(0..job_ids.len())].clone()
        } else {
            format!("AJ_{request}")
        };
        job_ids.push(job.clone());
        timeline.at(at, Src::Driver, In::Admit(request, job));
        if rng.u8(0..100) < 8 {
            timeline.at(at + rng.u64(0..5_000), Src::Driver, In::Cancel(request));
        }
    }
    if rng.u8(0..100) < 20 {
        timeline.at(rng.u64(10_000..60_000), Src::Driver, In::Drain);
    }
    timeline.at(rng.u64(30_000..90_000), Src::Driver, In::Close);

    let mut now = 0u64;
    let mut launched: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut forgotten: BTreeSet<u64> = BTreeSet::new();
    let mut created: BTreeSet<u64> = BTreeSet::new();
    let mut closed = false;
    // The driver reports its cleanup complete once, when the closed pool holds no process.
    let mut cleanup_reported = false;
    for _ in 0..200_000 {
        timeline.release(now, &mut sched);
        let due = pool.wake_at().is_some_and(|at| at <= Duration::from_millis(now));
        let input = if due {
            In::Wake
        } else {
            match sched.next(&[]) {
                Some(Picked::Item { item, .. }) => item,
                Some(Picked::Process(_)) => unreachable!("no processes"),
                None => {
                    let wake = pool.wake_at().map(|d| d.as_nanos().div_ceil(1_000_000) as u64);
                    match timeline.next_due().into_iter().chain(wake).min() {
                        Some(at) => now = at.max(now + 1),
                        None => break,
                    }
                    continue;
                }
            }
        };
        if let In::Admit(request, job) = &input {
            trace.jobs.insert(*request, job.clone());
        }
        closed |= input == In::Close;
        let effects = pool.handle(input.input(), Duration::from_millis(now));
        for effect in &effects {
            if let PoolEffect::Create(ProcessId(id)) = effect {
                created.insert(*id);
            }
            os.effect(effect, now, &mut timeline);
        }
        // The driver reads the pool: launch what is reserved, forget what settled.
        let mut reservations = BTreeMap::new();
        for &request in trace.jobs.keys() {
            let state = pool.reservation(ReservationId(request));
            match state {
                Some(ReservationState::Reserved(ProcessId(id)))
                    if launched.insert((request, id)) =>
                {
                    os.launch(request, id, now, &mut timeline);
                }
                Some(ReservationState::Completed | ReservationState::Failed(_))
                    if forgotten.insert(request) =>
                {
                    timeline.at(now, Src::Driver, In::Forget(request));
                }
                _ => {}
            }
            reservations.insert(request, state);
        }
        if closed && !pool.has_processes() && !cleanup_reported {
            cleanup_reported = true;
            timeline.at(now, Src::Driver, In::CleanupComplete);
        }
        let phases = created.iter().map(|&id| (id, pool.phase(ProcessId(id)))).collect();
        trace.steps.push(Step { at: now, input, effects, reservations, phases });
    }
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    for step in &trace.steps {
        reached.insert(match &step.input {
            In::SpawnFailed(_) => "SpawnFailed",
            In::LaunchFailed(..) => "LaunchFailed",
            In::LaunchSucceeded(..) => "Launched",
            In::ExitFailed(_) => "ExitFailed",
            In::JoinFailed(_) => "JoinFailed",
            In::Cancel(_) => "Cancel",
            In::Drain => "Drain",
            _ => "Step",
        });
        for effect in &step.effects {
            match effect {
                PoolEffect::Refused { .. } => reached.insert("Refused"),
                PoolEffect::CleanupIncomplete(_) => reached.insert("CleanupIncomplete"),
                PoolEffect::CloseComplete => reached.insert("CloseComplete"),
                _ => false,
            };
        }
        for state in step.reservations.values().flatten() {
            if matches!(state, ReservationState::Failed(_)) {
                reached.insert("RequestFailed");
            }
        }
    }
    reached
}

// ----------------------------------------------------------------------------------------------
// The rules
// ----------------------------------------------------------------------------------------------

fn admitted(step: &Step) -> Option<u64> {
    match &step.input {
        In::Admit(request, _)
            if !step.effects.iter().any(|e| matches!(e, PoolEffect::Refused { .. })) =>
        {
            Some(*request)
        }
        _ => None,
    }
}

fn settle(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut open: BTreeMap<u64, Option<ReservationState>> = BTreeMap::new();
    let mut cancelled = BTreeSet::new();
    for step in &trace.steps {
        if let Some(request) = admitted(step) {
            open.insert(request, None);
        }
        if let In::Cancel(request) = step.input {
            cancelled.insert(request);
        }
        for (request, state) in &step.reservations {
            if let Some(seen) = open.get_mut(request) {
                *seen = state.or(*seen);
            }
        }
        if step.input == In::Close {
            for (request, state) in &step.reservations {
                if open.contains_key(request)
                    && !cancelled.contains(request)
                    && matches!(
                        state,
                        Some(ReservationState::Waiting | ReservationState::Reserved(_))
                    )
                {
                    violations.push(format!("request {request} still {state:?} after the close"));
                }
            }
        }
    }
    for (request, state) in open {
        let settled =
            matches!(state, Some(ReservationState::Completed | ReservationState::Failed(_)));
        if !settled && !cancelled.contains(&request) {
            violations.push(format!("request {request} ended {state:?}"));
        }
    }
    violations
}

fn one_disposition(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    // Each process's request, each request's completion, each running job's process.
    let mut served: BTreeMap<u64, u64> = BTreeMap::new();
    let mut completed: BTreeMap<u64, u64> = BTreeMap::new();
    let mut running: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for step in &trace.steps {
        // A job runs until its process retires.
        running.retain(|_, (_, process)| {
            !matches!(
                step.phases.get(process),
                Some(ProcessPhase::Retiring | ProcessPhase::Cleaned) | None
            )
        });
        if let In::Admit(request, job) = &step.input
            && let Some((other, _)) = running.get(job.as_str())
            && admitted(step).is_some()
        {
            violations.push(format!(
                "job {job} admitted as request {request} while request {other} runs it"
            ));
        }
        for (&request, state) in &step.reservations {
            match state {
                Some(ReservationState::Reserved(ProcessId(p))) => {
                    if let Some(other) = served.insert(*p, request)
                        && other != request
                    {
                        violations.push(format!(
                            "process {p} reserved for request {request} after request {other}"
                        ));
                    }
                }
                Some(ReservationState::Completed) if !completed.contains_key(&request) => {
                    let process = served
                        .iter()
                        .find(|(_, r)| **r == request)
                        .map(|(p, _)| *p)
                        .unwrap_or(u64::MAX);
                    completed.insert(request, process);
                    let job = trace.jobs[&request].as_str();
                    if let Some((other, _)) = running.insert(job, (request, process)) {
                        violations.push(format!(
                            "job {job} runs as requests {other} and {request} at once"
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    violations
}

fn no_revival(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut retired: BTreeSet<u64> = BTreeSet::new();
    let mut released: BTreeSet<u64> = BTreeSet::new();
    for step in &trace.steps {
        for effect in &step.effects {
            let named = match effect {
                PoolEffect::Create(ProcessId(p))
                | PoolEffect::Join(ProcessId(p))
                | PoolEffect::WatchExit(ProcessId(p)) => *p,
                PoolEffect::Close(ProcessId(p)) => *p,
                PoolEffect::Released { process: ProcessId(p), .. } => {
                    if !released.insert(*p) {
                        violations.push(format!("process {p} released twice"));
                    }
                    continue;
                }
                _ => continue,
            };
            if released.contains(&named) {
                violations.push(format!(
                    "released process {named} named again at {} ms: {effect:?}",
                    step.at
                ));
            }
        }
        for (&p, phase) in &step.phases {
            match phase {
                ProcessPhase::Retiring | ProcessPhase::Cleaned => {
                    retired.insert(p);
                }
                phase if retired.contains(&p) => {
                    violations
                        .push(format!("retired process {p} is {phase:?} again at {} ms", step.at));
                }
                _ => {}
            }
        }
    }
    violations
}

fn released(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut created = BTreeSet::new();
    let mut released = BTreeSet::new();
    let mut reported = BTreeSet::new();
    let mut completes = 0;
    let mut closed = false;
    for step in &trace.steps {
        closed |= step.input == In::Close;
        for effect in &step.effects {
            match effect {
                PoolEffect::Create(ProcessId(p)) => {
                    created.insert(*p);
                }
                PoolEffect::Released { process: ProcessId(p), .. } => {
                    released.insert(*p);
                }
                PoolEffect::CleanupIncomplete(observation) => {
                    reported.extend(observation.processes.iter().map(|p| p.0));
                }
                PoolEffect::CloseComplete => {
                    completes += 1;
                    if let Some(p) = created.iter().find(|p| !released.contains(p)) {
                        violations.push(format!(
                            "the close completed at {} ms with process {p} unreleased",
                            step.at
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    for p in created.iter().filter(|p| !released.contains(p) && !reported.contains(p)) {
        violations.push(format!("process {p} neither released nor reported"));
    }
    if closed && completes != 1 && created.iter().all(|p| released.contains(p)) {
        violations.push(format!("the close completed {completes} times"));
    }
    violations
}

#[cfg(test)]
mod tests;
