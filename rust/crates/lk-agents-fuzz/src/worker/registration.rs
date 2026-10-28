//! The worker's registration and job assignment (`lk_agents_worker::state::WorkerState`, K-2):
//! a connection that registers, drops and reconnects under a new generation; a fake LiveKit
//! server that dispatches jobs while the worker is registered, answers an acceptance with an
//! assignment (late, never, or twice), sends assignments for jobs the worker does not hold and
//! terminations; the host's `job_request_fnc` that accepts, rejects, fails or never answers; job
//! launches that fail, report a status before the launch completes, end on their own or on a
//! termination; load samples, some over the threshold; a drain with or without a deadline; a
//! rollover once the worker's job history fills.
//!
//! The driver mirrors the runtime's (`lk_agents_runtime::worker`): a server message carries the
//! generation current when it is delivered, a launch the generation of the effect that asked for
//! it and may land after a reconnect; the host's pending decisions are aborted when the
//! connection drops; what the worker sends while offline is dropped; one load sample is
//! outstanding at a time.
//!
//! The rules, over the trace:
//! * `worker_registers_first`: on each connection `register` goes out first, and no status
//!   update or ping before the server answered it;
//! * `worker_availability_answered_once`: each job offered on a connection is answered once,
//!   accepted or rejected, unless the connection dropped first; none offered during a drain is
//!   accepted;
//! * `worker_one_disposition_per_assignment`: an assignment launches its job once over the whole
//!   run, or is answered failed when the worker does not hold it; a job is never launched twice;
//! * `worker_launched_jobs_are_kept`: a launched job stays active until its final status, across
//!   reconnects, the next registration reports it for migration, and only an active job is
//!   terminated;
//! * `worker_accepted_requests_settle`: every accepted request is launched or dropped within
//!   the assignment timeout, and nothing stays reserved at the end;
//! * `worker_drain_ends_once`: a drain ends once, finished only when no job is pending or active.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use fastrand::Rng;
use livekit_protocol::worker_message::Message as Out;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::HostError;
use lk_agents_worker::protocol::{
    AvailabilityRequest, Job, JobAssignment, JobStatus, JobTermination, RegisterWorkerResponse,
    ServerMessage, server_message,
};
use lk_agents_worker::{
    ASSIGNMENT_TIMEOUT, AcceptArguments, EnvOption, JobDecision, Slot, WorkerConfig, WorkerState,
    WorkerStateEffect, WorkerStateInput,
};

use crate::inference::Timeline;
use crate::sched::{Sampling, Sched, Step as Picked, Tick};

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = u64::from_be_bytes(*b"\0\0\0sched");

/// Who reports: the server over the socket, the host, or the worker's driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Server,
    Host,
    Driver,
}

impl crate::sched::Source for Src {}

/// What the simulation delivers, recorded as sent.
#[derive(Debug, Clone, PartialEq)]
pub enum In {
    /// The socket connected: a new generation.
    Connect,
    /// What the server sends on connection `on` (`0`: whichever is open); it dies with its
    /// socket.
    Registered {
        on: u64,
    },
    Offer {
        on: u64,
        job: String,
        resuming: bool,
    },
    Assign {
        on: u64,
        job: String,
    },
    Terminate {
        on: u64,
        job: String,
    },
    /// The transport wrote an acceptance on connection `on`.
    Sent {
        on: u64,
        job: String,
    },
    Disconnect,
    Decision {
        generation: u64,
        slot: Slot,
        accept: Option<bool>,
    },
    Launched {
        generation: u64,
        slot: Slot,
        ok: bool,
    },
    Status {
        job: String,
        status: JobStatus,
    },
    Load(f32),
    Drain(Option<Duration>),
    Wake,
}

/// The worker as it stood after a step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Seen {
    pub generation: u64,
    /// A socket is open: the driver drops what the worker sends while offline.
    pub connected: bool,
    pub registered: bool,
    pub reserved: usize,
    pub active: BTreeSet<String>,
}

/// One delivered input, the effects it returned and the worker after it, at `at` ms.
#[derive(Debug, Clone)]
pub struct Step {
    pub at: u64,
    pub input: In,
    pub effects: Vec<WorkerStateEffect>,
    pub after: Seen,
}

#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub steps: Vec<Step>,
}

/// A finished run: what broke, and what it reached.
#[derive(Debug)]
pub struct Outcome {
    pub seed: u64,
    pub violations: Vec<String>,
    pub reached: BTreeSet<&'static str>,
    pub trace: Trace,
}

type Rule = (&'static str, fn(&Trace) -> Vec<String>);

pub const RULES: &[Rule] = &[
    ("worker_registers_first", registers_first),
    ("worker_availability_answered_once", answered_once),
    ("worker_one_disposition_per_assignment", one_disposition),
    ("worker_launched_jobs_are_kept", jobs_kept),
    ("worker_accepted_requests_settle", accepted_settle),
    ("worker_drain_ends_once", drain_once),
];

/// Every rule over `trace`, each violation prefixed with its rule.
pub fn check(trace: &Trace) -> Vec<String> {
    RULES
        .iter()
        .flat_map(|(name, rule)| rule(trace).into_iter().map(move |v| format!("{name}: {v}")))
        .collect()
}

fn ts(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

fn job(id: &str) -> Job {
    Job { id: id.into(), dispatch_id: format!("d-{id}"), ..Default::default() }
}

fn server(message: server_message::Message) -> ServerMessage {
    ServerMessage { message: Some(message) }
}

/// The fakes' side of the run: what they hold and how they answer.
struct World {
    rng: Rng,
    generation: u64,
    connected: bool,
    registered: bool,
    /// Jobs offered so far; a new offer names the next.
    offered: u32,
    load_pending: bool,
    /// Jobs the host is running, and whether each has ended.
    running: BTreeMap<String, bool>,
    /// The server's dispatching stops then; the run ends once all has settled.
    dispatch_until: u64,
}

impl World {
    /// The server's next dispatch on this connection, 0.3-4 s out.
    fn next_offer(&mut self, now: u64, timeline: &mut Timeline<Src, In>) {
        let at = now + self.rng.u64(300..4_000);
        if at < self.dispatch_until {
            self.offered += 1;
            let job = format!("J{}", self.offered);
            let resuming = self.rng.u8(0..10) == 0;
            timeline.at(at, Src::Server, In::Offer { on: self.generation, job, resuming });
        }
    }

    /// The fakes answer what the worker asked for.
    fn react(&mut self, effects: &[WorkerStateEffect], now: u64, timeline: &mut Timeline<Src, In>) {
        let g = self.generation;
        for effect in effects {
            match effect {
                WorkerStateEffect::Send(message) => match &message.message {
                    Some(Out::Register(_)) if self.rng.u8(0..30) != 0 => {
                        let at = now + self.rng.u64(20..300);
                        timeline.at(at, Src::Server, In::Registered { on: g });
                    }
                    // the server never answers; the socket times out
                    Some(Out::Register(_)) => {
                        timeline.at(now + 10_000, Src::Server, In::Disconnect);
                    }
                    _ => {}
                },
                WorkerStateEffect::SendAvailability { job_id, .. } => {
                    let job = job_id.clone();
                    let sent = In::Sent { on: g, job: job.clone() };
                    timeline.at(now + self.rng.u64(0..20), Src::Driver, sent);
                    match self.rng.u8(0..20) {
                        // another worker won it: no assignment
                        0 | 1 => {}
                        2 => {
                            for _ in 0..2 {
                                let at = now + self.rng.u64(20..600);
                                timeline.at(
                                    at,
                                    Src::Server,
                                    In::Assign { on: g, job: job.clone() },
                                );
                            }
                        }
                        // late, past the assignment timeout
                        3 => {
                            let at = now + self.rng.u64(7_000..9_000);
                            timeline.at(at, Src::Server, In::Assign { on: g, job });
                        }
                        _ => {
                            let at = now + self.rng.u64(20..600);
                            timeline.at(at, Src::Server, In::Assign { on: g, job });
                        }
                    }
                }
                WorkerStateEffect::AskUser { slot, .. } => {
                    let accept = match self.rng.u8(0..20) {
                        0 => Some(false),
                        1 => None,
                        _ => Some(true),
                    };
                    let decision = In::Decision { generation: g, slot: *slot, accept };
                    let at = now
                        + if self.rng.u8(0..10) == 0 {
                            self.rng.u64(0..3_000)
                        } else {
                            self.rng.u64(0..200)
                        };
                    timeline.at(at, Src::Host, decision);
                }
                WorkerStateEffect::LaunchJob { slot, info } => {
                    let ok = self.rng.u8(0..20) != 0;
                    let launched_at = now + self.rng.u64(10..1_000);
                    timeline.at(
                        launched_at,
                        Src::Host,
                        In::Launched { generation: g, slot: *slot, ok },
                    );
                    if ok {
                        let id = info.job.id.clone();
                        self.running.insert(id.clone(), false);
                        // the job process may report before the launch call returns
                        let running_at = now + self.rng.u64(5..1_200);
                        timeline.at(
                            running_at,
                            Src::Host,
                            In::Status { job: id.clone(), status: JobStatus::JsRunning },
                        );
                        let status = if self.rng.u8(0..5) == 0 {
                            JobStatus::JsFailed
                        } else {
                            JobStatus::JsSuccess
                        };
                        let end = launched_at.max(running_at) + self.rng.u64(1_000..60_000);
                        timeline.at(end, Src::Host, In::Status { job: id.clone(), status });
                        // the room closes under it now and then: the server terminates the job
                        if self.rng.u8(0..8) == 0 {
                            let at = launched_at + self.rng.u64(100..end - launched_at);
                            timeline.at(at, Src::Server, In::Terminate { on: 0, job: id });
                        }
                    }
                }
                WorkerStateEffect::TerminateJob(id) if self.running.get(id) == Some(&false) => {
                    let at = now + self.rng.u64(10..500);
                    timeline.at(
                        at,
                        Src::Host,
                        In::Status { job: id.clone(), status: JobStatus::JsSuccess },
                    );
                }
                WorkerStateEffect::SampleLoad if !self.load_pending => {
                    self.load_pending = true;
                    let load = if self.rng.u8(0..8) == 0 {
                        0.7 + self.rng.f32() * 0.3
                    } else {
                        self.rng.f32() * 0.6
                    };
                    timeline.at(now + self.rng.u64(1..20), Src::Host, In::Load(load));
                }
                WorkerStateEffect::RolloverConnection => {
                    timeline.at(now + self.rng.u64(0..50), Src::Driver, In::Disconnect);
                }
                _ => {}
            }
        }
    }
}

/// One seeded run, the pending reports picked in the `sampling` order.
pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let mut sched: Sched<Src, In> = Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let mut timeline: Timeline<Src, In> = Timeline::default();
    let cfg = WorkerConfig {
        agent_name: "front-desk".into(),
        deployment: "prod".into(),
        ws_url: "wss://project.livekit.cloud".into(),
        load_threshold: EnvOption::Fixed(0.7),
        num_idle_processes: EnvOption::Fixed(rng.u32(0..4)),
        ..WorkerConfig::default()
    };
    // a small history bound now and then, so the rollover is reached
    let capacity = if rng.u8(0..4) == 0 { rng.usize(4..12) } else { 4096 };
    let mut worker = WorkerState::with_assignment_capacity(cfg, capacity);
    let dispatch_until = rng.u64(5_000..60_000);
    let mut world = World {
        rng: Rng::with_seed(rng.u64(..)),
        generation: 0,
        connected: false,
        registered: false,
        offered: 0,
        load_pending: false,
        running: BTreeMap::new(),
        dispatch_until,
    };
    timeline.at(0, Src::Driver, In::Connect);
    for _ in 0..rng.usize(0..4) {
        timeline.at(rng.u64(500..dispatch_until), Src::Server, In::Disconnect);
    }
    for _ in 0..rng.usize(0..3) {
        let job = format!("X{}", rng.u32(0..1000));
        let input =
            if rng.bool() { In::Assign { on: 0, job } } else { In::Terminate { on: 0, job } };
        timeline.at(rng.u64(500..dispatch_until), Src::Server, input);
    }
    if rng.u8(0..3) == 0 {
        let timeout = match rng.u8(0..3) {
            0 => None,
            _ => Some(Duration::from_millis(rng.u64(1_000..30_000))),
        };
        timeline.at(rng.u64(1_000..dispatch_until + 5_000), Src::Driver, In::Drain(timeout));
    }

    let mut trace = Trace::default();
    let effects = worker.start(ts(0));
    world.react(&effects, 0, &mut timeline);
    let mut now = 0;
    // quiet once nothing is reported and nothing else can arrive: only the periodic timers
    let end = dispatch_until + 90_000;
    while now <= end {
        timeline.release(now, &mut sched);
        let input = if worker.next_deadline().is_some_and(|at| at <= ts(now)) {
            In::Wake
        } else {
            let wake = worker.next_deadline().map(|t| t.as_nanos().div_ceil(1_000_000));
            match sched.tick(now, &[], timeline.next_due().into_iter().chain(wake)) {
                Tick::Step(Picked::Item { item, .. }) => item,
                Tick::Step(Picked::Process(_)) => unreachable!("no processes"),
                Tick::Advance(at) => {
                    now = at;
                    continue;
                }
                Tick::Quiet => break,
            }
        };
        if !world.delivers(&input) {
            continue;
        }
        let effects = deliver(&mut worker, &mut world, &input, now, &mut timeline);
        world.react(&effects, now, &mut timeline);
        let after = Seen {
            generation: world.generation,
            connected: world.connected,
            registered: worker.registered(),
            reserved: worker.reserved_slots(),
            active: worker.active_jobs().iter().map(|j| j.job.id.clone()).collect(),
        };
        trace.steps.push(Step { at: now, input, effects, after });
    }
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

impl World {
    /// Whether `input` still reaches the worker: what the server or the transport sent on a
    /// connection dies with it, and the server dispatches only to a registered worker.
    fn delivers(&self, input: &In) -> bool {
        let open = |on: &u64| self.connected && (*on == 0 || *on == self.generation);
        match input {
            In::Registered { on } | In::Sent { on, .. } => open(on),
            In::Offer { on, .. } | In::Assign { on, .. } | In::Terminate { on, .. } => {
                open(on) && self.registered
            }
            In::Disconnect => self.connected,
            // the driver aborts the host's pending decisions when the connection drops
            In::Decision { generation, .. } => *generation == self.generation,
            _ => true,
        }
    }
}

/// One input through the worker, as the runtime's driver feeds it.
fn deliver(
    worker: &mut WorkerState,
    world: &mut World,
    input: &In,
    now: u64,
    timeline: &mut Timeline<Src, In>,
) -> Vec<WorkerStateEffect> {
    let g = world.generation;
    let at = ts(now);
    let ev = match input.clone() {
        In::Wake => return worker.tick(at),
        In::Drain(timeout) => return worker.drain(at, timeout),
        In::Connect => {
            world.connected = true;
            world.generation += 1;
            WorkerStateInput::Connecting { generation: world.generation }
        }
        In::Registered { .. } => {
            world.registered = true;
            world.next_offer(now, timeline);
            let response =
                RegisterWorkerResponse { worker_id: format!("W_{g}"), ..Default::default() };
            WorkerStateInput::Registered { generation: g, response }
        }
        In::Offer { on, job: id, resuming } => {
            world.next_offer(now, timeline);
            // the server re-dispatches an offer now and then
            if world.rng.u8(0..20) == 0 {
                let again = In::Offer { on, job: id.clone(), resuming };
                timeline.at(now + world.rng.u64(0..500), Src::Server, again);
            }
            let req = AvailabilityRequest { job: Some(job(&id)), resuming };
            let message = server(server_message::Message::Availability(req));
            WorkerStateInput::Server { generation: g, message }
        }
        In::Assign { job: id, .. } => {
            let a =
                JobAssignment { job: Some(job(&id)), token: "tok".into(), ..Default::default() };
            WorkerStateInput::Server {
                generation: g,
                message: server(server_message::Message::Assignment(a)),
            }
        }
        In::Terminate { job: id, .. } => {
            let t = JobTermination { job_id: id };
            WorkerStateInput::Server {
                generation: g,
                message: server(server_message::Message::Termination(t)),
            }
        }
        In::Disconnect => {
            (world.connected, world.registered) = (false, false);
            // the driver reconnects after its retry delay
            timeline.at(now + world.rng.u64(100..2_000), Src::Driver, In::Connect);
            WorkerStateInput::Disconnected { generation: g }
        }
        In::Decision { generation, slot, accept } => {
            let decision = match accept {
                Some(true) => JobDecision::Accept(AcceptArguments::default()),
                Some(false) => JobDecision::Reject { terminate: false },
                None => JobDecision::Failed(HostError {
                    kind: "RuntimeError".into(),
                    message: "job_request_fnc raised".into(),
                    host_ref: None,
                }),
            };
            WorkerStateInput::RequestDecision { generation, slot, decision }
        }
        In::Sent { job, .. } => WorkerStateInput::AvailabilitySent { generation: g, job_id: job },
        In::Launched { generation, slot, ok } => {
            let result = match ok {
                true => Ok(String::new()),
                false => Err(HostError {
                    kind: "RuntimeError".into(),
                    message: "no process".into(),
                    host_ref: None,
                }),
            };
            WorkerStateInput::JobLaunched { generation, slot, result }
        }
        In::Status { job, status } => {
            if matches!(status, JobStatus::JsSuccess | JobStatus::JsFailed) {
                world.running.insert(job.clone(), true);
            }
            WorkerStateInput::JobStatus { job_id: job, status }
        }
        In::Load(load) => {
            world.load_pending = false;
            WorkerStateInput::LoadSampled(load)
        }
    };
    worker.handle(at, ev)
}

/// The messages a step sent.
fn protocol_messages(step: &Step) -> impl Iterator<Item = &Out> {
    step.effects.iter().filter_map(|e| match e {
        WorkerStateEffect::Send(m) => m.message.as_ref(),
        _ => None,
    })
}

/// Acceptance effects and protocol rejections share the same reply accounting.
fn availability(effect: &WorkerStateEffect) -> Option<(&String, bool)> {
    match effect {
        WorkerStateEffect::SendAvailability { job_id, .. } => Some((job_id, true)),
        WorkerStateEffect::Send(m) => match &m.message {
            Some(Out::Availability(a)) => Some((&a.job_id, a.available)),
            _ => None,
        },
        _ => None,
    }
}

/// What a trace reached, for the campaign's reach report.
pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    for step in &trace.steps {
        if matches!(step.input, In::Connect) && step.after.generation > 1 {
            reached.insert("WorkerReconnected");
        }
        for e in &step.effects {
            reached.insert(match e {
                WorkerStateEffect::LaunchJob { .. } => "WorkerJobLaunched",
                WorkerStateEffect::TerminateJob(_) => "WorkerJobTerminated",
                WorkerStateEffect::RolloverConnection => "WorkerRollover",
                WorkerStateEffect::DrainFinished => "WorkerDrainFinished",
                WorkerStateEffect::DrainTimedOut => "WorkerDrainTimedOut",
                WorkerStateEffect::CancelRequest { .. } => "WorkerRequestCancelled",
                _ => continue,
            });
        }
        for m in protocol_messages(step) {
            match m {
                Out::MigrateJob(_) => {
                    reached.insert("WorkerJobsMigrated");
                }
                Out::Availability(a) if !a.available => {
                    reached.insert("WorkerOfferRejected");
                }
                Out::UpdateJob(u) if u.status() == JobStatus::JsFailed => {
                    reached.insert("WorkerJobFailed");
                }
                _ => {}
            }
        }
        if let In::Launched { generation, .. } = step.input
            && generation < step.after.generation
        {
            reached.insert("WorkerLaunchedAfterReconnect");
        }
    }
    reached
}

fn registers_first(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut first_sent = true;
    let mut registered = false;
    for step in &trace.steps {
        if matches!(step.input, In::Connect) {
            (first_sent, registered) = (false, false);
        }
        if matches!(step.input, In::Disconnect) {
            first_sent = true;
        }
        if !step.after.connected {
            continue;
        }
        for effect in &step.effects {
            let message = match effect {
                WorkerStateEffect::Send(m) => m.message.as_ref(),
                WorkerStateEffect::SendAvailability { job_id, .. } => {
                    if !first_sent {
                        out.push(format!(
                            "at {} ms {job_id} was accepted before register",
                            step.at
                        ));
                    }
                    first_sent = true;
                    continue;
                }
                _ => continue,
            };
            let Some(m) = message else { continue };
            if !first_sent && !matches!(m, Out::Register(_)) {
                out.push(format!("at {} ms {m:?} went out before register", step.at));
            }
            first_sent = true;
            let registered = registered || step.after.registered;
            if !registered && matches!(m, Out::UpdateWorker(_) | Out::Ping(_)) {
                out.push(format!("at {} ms a status update before the registration", step.at));
            }
        }
        registered = step.after.registered;
    }
    out
}

fn answered_once(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    // per connection: offers and answers by job
    let mut offered: BTreeMap<String, u32> = BTreeMap::new();
    let mut answers: BTreeMap<String, u32> = BTreeMap::new();
    let mut draining = false;
    // a request in flight when the drain began may still be accepted, as in the reference
    let mut offered_draining: BTreeSet<String> = BTreeSet::new();
    let close = |offered: &mut BTreeMap<String, u32>,
                 answers: &mut BTreeMap<String, u32>,
                 out: &mut Vec<String>,
                 at| {
        for (job, n) in answers.iter() {
            let asked = offered.get(job).copied().unwrap_or(0);
            if *n > asked {
                out.push(format!("by {at} ms {job} was answered {n} times, offered {asked}"));
            }
        }
        offered.clear();
        answers.clear();
    };
    for step in &trace.steps {
        if matches!(step.input, In::Connect) {
            close(&mut offered, &mut answers, &mut out, step.at);
        }
        if matches!(step.input, In::Drain(_)) {
            draining = true;
        }
        if let In::Offer { job, .. } = &step.input {
            *offered.entry(job.clone()).or_default() += 1;
            if draining {
                offered_draining.insert(job.clone());
            }
        }
        if !step.after.connected {
            continue;
        }
        for (job_id, accepted) in step.effects.iter().filter_map(availability) {
            *answers.entry(job_id.clone()).or_default() += 1;
            if accepted && offered_draining.contains(job_id) {
                out.push(format!("at {} ms {} was accepted while draining", step.at, job_id));
            }
        }
    }
    close(&mut offered, &mut answers, &mut out, u64::MAX);
    // every offer of the last connection that lived on was answered
    if let Some(last) = trace.steps.iter().rposition(|s| matches!(s.input, In::Connect)) {
        let tail = &trace.steps[last..];
        if !tail.iter().any(|s| matches!(s.input, In::Disconnect)) {
            let offered: BTreeSet<&String> = tail
                .iter()
                .filter_map(|s| match &s.input {
                    In::Offer { job, .. } => Some(job),
                    _ => None,
                })
                .collect();
            let answered: BTreeSet<&String> = tail
                .iter()
                .flat_map(|step| step.effects.iter().filter_map(availability))
                .map(|(job_id, _)| job_id)
                .collect();
            for job in offered.difference(&answered) {
                out.push(format!("{job} was offered and never answered"));
            }
        }
    }
    out
}

fn one_disposition(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut launched: BTreeSet<String> = BTreeSet::new();
    for step in &trace.steps {
        for e in &step.effects {
            if let WorkerStateEffect::LaunchJob { info, .. } = e
                && !launched.insert(info.job.id.clone())
            {
                out.push(format!("at {} ms {} launched again", step.at, info.job.id));
            }
        }
        if let In::Assign { job, .. } = &step.input {
            let launches = step
                .effects
                .iter()
                .filter(|e| matches!(e, WorkerStateEffect::LaunchJob { .. }))
                .count();
            let updates = protocol_messages(step)
                .filter(|m| matches!(m, Out::UpdateJob(u) if u.job_id == *job))
                .count();
            if launches + updates > 1 {
                out.push(format!(
                    "at {} ms the assignment of {job} was answered {} times",
                    step.at,
                    launches + updates
                ));
            }
        }
    }
    out
}

fn jobs_kept(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut active: BTreeSet<String> = BTreeSet::new();
    let mut ended: BTreeSet<String> = BTreeSet::new();
    for step in &trace.steps {
        if let In::Status { job, status: JobStatus::JsSuccess | JobStatus::JsFailed } = &step.input
        {
            ended.insert(job.clone());
        }
        for e in &step.effects {
            if let WorkerStateEffect::TerminateJob(id) = e
                && !active.contains(id)
            {
                out.push(format!("at {} ms terminated {id}, which is not active", step.at));
            }
        }
        for m in protocol_messages(step) {
            if let Out::MigrateJob(mj) = m {
                let named: BTreeSet<String> = mj.job_ids.iter().cloned().collect();
                if matches!(step.input, In::Registered { .. }) && named != active {
                    out.push(format!(
                        "at {} ms migration named {named:?}, active {active:?}",
                        step.at
                    ));
                }
            }
        }
        for lost in active.difference(&step.after.active) {
            if !ended.contains(lost) {
                out.push(format!(
                    "at {} ms {lost} left the active jobs before its final status",
                    step.at
                ));
            }
        }
        active = step.after.active.clone();
    }
    out
}

fn accepted_settle(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    // what may hold a reservation: a request the host is deciding, a launch in progress, an
    // acceptance still inside its assignment timeout
    let mut asked: BTreeSet<Slot> = BTreeSet::new();
    let mut launching: BTreeSet<Slot> = BTreeSet::new();
    let mut accepted: BTreeMap<String, u64> = BTreeMap::new();
    let grace = ASSIGNMENT_TIMEOUT.as_millis() as u64 + 1_100;
    for step in &trace.steps {
        match &step.input {
            In::Disconnect => {
                asked.clear();
                accepted.clear();
            }
            In::Decision { slot, .. } => {
                asked.remove(slot);
            }
            In::Launched { slot, .. } => {
                launching.remove(slot);
            }
            _ => {}
        }
        for e in &step.effects {
            match e {
                WorkerStateEffect::AskUser { slot, .. } => {
                    asked.insert(*slot);
                }
                WorkerStateEffect::CancelRequest { slot } => {
                    asked.remove(slot);
                }
                WorkerStateEffect::LaunchJob { slot, info } => {
                    launching.insert(*slot);
                    accepted.remove(&info.job.id);
                }
                _ => {}
            }
        }
        for (job_id, available) in step.effects.iter().filter_map(availability) {
            if available {
                accepted.insert(job_id.clone(), step.at);
            }
        }
        accepted.retain(|_, since| step.at <= *since + grace);
        let bound = asked.len() + launching.len() + accepted.len();
        if step.after.reserved > bound {
            out.push(format!(
                "at {} ms {} requests reserved, {bound} can be: an acceptance outlived its \
                 assignment timeout",
                step.at, step.after.reserved
            ));
        }
    }
    if let Some(last) = trace.steps.last()
        && last.after.reserved != 0
    {
        out.push(format!("{} requests stayed reserved at the end", last.after.reserved));
    }
    out
}

fn drain_once(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    let mut ends = 0;
    for step in &trace.steps {
        for e in &step.effects {
            match e {
                WorkerStateEffect::DrainFinished => {
                    ends += 1;
                    if step.after.reserved > 0 || !step.after.active.is_empty() {
                        out.push(format!(
                            "at {} ms drained with {} pending and {:?} active",
                            step.at, step.after.reserved, step.after.active
                        ));
                    }
                }
                WorkerStateEffect::DrainTimedOut => ends += 1,
                _ => {}
            }
        }
    }
    let drained = trace.steps.iter().any(|s| matches!(s.input, In::Drain(_)));
    if ends > 1 || (!drained && ends > 0) {
        out.push(format!("the drain ended {ends} times"));
    }
    // nothing left to wait for: the drain must have ended
    let idle =
        trace.steps.last().is_some_and(|s| s.after.reserved == 0 && s.after.active.is_empty());
    if drained && ends == 0 && idle {
        out.push("the drain never ended with nothing left to wait for".into());
    }
    out
}

#[cfg(test)]
mod tests;
