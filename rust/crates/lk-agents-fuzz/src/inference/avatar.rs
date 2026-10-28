//! One avatar instance's provider sessions (`lk_agents_inference::avatar::lifecycle`) against a
//! fake gateway, with the program that creates, adopts, abandons and closes them.
//!
//! The gateway keeps one session per create operation (its idempotency key is the request id):
//! a retried create replays the session the first attempt made, from its idempotency record in
//! 20-200 ms. Each exchange connects in 30-400 ms, or is refused, or goes silent until the network
//! drops it one to two minutes later; a create answers in
//! 0.5-4 s, a terminate in 50-500 ms, or is
//! rejected; and an exchange may apply on the gateway and lose its answer (the connection resets
//! after the session was made or ended), which is the case idempotency exists for. Some sessions
//! come back without the ids a terminate needs. The program creates once (a second caller
//! sometimes races it), may stop waiting, may not be there to receive the session, closes at a
//! random moment and closes again after a failed terminate.
//!
//! The rules, over the trace and the gateway's ledger:
//! * `avatar_sessions_are_never_leaked`: every session the gateway still runs at the end is one
//!   the instance reported it could not end (unterminable, or held after a failed terminate);
//! * `avatar_creates_one_session`: a create racing one in flight or an adopted session is
//!   refused, and at most one session is adopted;
//! * `avatar_caller_has_one_outcome`: an accepted create's caller gets one outcome unless it
//!   stopped waiting first, and a close during creation answers it `Closed`;
//! * `avatar_retirements_have_their_cause`: a session ends unheld only for a cause that happened
//!   (the caller gone, a refused delivery, a close, a replay) and only if the gateway made it;
//! * `avatar_attempts_are_released`: every launched attempt is released or reported missing, and
//!   the machine settles;
//! * `avatar_retirement_is_bounded`: a create left without an answer (its caller gone, a close, a
//!   give-up after the gateway may have received it) is reported unresolved at most once, never
//!   before `RETIREMENT_TIMEOUT` since it was left.

use std::collections::{BTreeMap, BTreeSet};

use fastrand::Rng;
use lk_agents_abi::resources::ResourceId;
use lk_agents_inference::attempts::{ATTEMPT_CLEANUP_TIMEOUT, AttemptFailure, RequestPolicy};
use lk_agents_inference::avatar::lifecycle::{
    AvatarRequestId, AvatarSessionEffect as Fx, AvatarSessionEvent as Ev, AvatarSessionLifecycle,
    CreateFailure, ProviderSession, RETIREMENT_TIMEOUT, Retirement,
};

use super::{Timeline, dur, ms, ts};
use crate::sched::{Sampling, Sched, Step as Picked};

const AVATAR: ResourceId = ResourceId(1);

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = 0x7363_6865_64;

/// Who reports an event: one exchange's transport, the creating caller, a second caller, or the
/// program closing the instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Attempt(AvatarRequestId, u32),
    Caller,
    Racer,
    Program,
}

impl crate::sched::Source for Src {}

#[derive(Debug, Clone)]
pub struct Step {
    pub at: u64,
    pub event: Ev,
    pub effects: Vec<Fx>,
}

#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub steps: Vec<Step>,
    /// Every session the gateway made, by the create request that made it.
    pub made: BTreeMap<AvatarRequestId, ProviderSession>,
    /// The sessions the gateway still runs at the end.
    pub running: Vec<ProviderSession>,
    pub settled: bool,
    /// The program set a read limit: without one a silent gateway is waited on until the
    /// retirement's deadline.
    pub read_limit: bool,
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
    ("avatar_sessions_are_never_leaked", never_leaked),
    ("avatar_creates_one_session", one_session),
    ("avatar_caller_has_one_outcome", one_outcome),
    ("avatar_retirements_have_their_cause", retirements),
    ("avatar_attempts_are_released", released),
    ("avatar_retirement_is_bounded", retirement_bounded),
];

/// The subject kind a finding of `trace` is keyed by (`known_findings.tsv`): a run whose program
/// set no read limit waits on a silent gateway as long as it takes, and is reported apart.
pub fn subject(trace: &Trace) -> &'static str {
    if trace.read_limit { "AvatarSession" } else { "AvatarSession/no-read-limit" }
}

/// The violations of `trace` outside the open findings.
pub fn unknown(trace: &Trace, violations: &[String]) -> Vec<String> {
    let subject = subject(trace);
    violations
        .iter()
        .filter(|v| {
            let rule = v.split(':').next().unwrap_or_default();
            !crate::check::KNOWN_FINDINGS.iter().any(|k| k.contract == rule && k.subject == subject)
        })
        .cloned()
        .collect()
}

pub fn check(trace: &Trace) -> Vec<String> {
    RULES
        .iter()
        .flat_map(|(name, rule)| rule(trace).into_iter().map(move |v| format!("{name}: {v}")))
        .collect()
}

/// The fake gateway: one session per create operation, ended by terminate. A request it
/// received is applied whatever becomes of the client's exchange; one stopped before it was sent
/// never reaches it.
struct Gateway {
    rng: Rng,
    /// Each create operation's session, decided by its first attempt.
    sessions: BTreeMap<AvatarRequestId, ProviderSession>,
    made: BTreeMap<AvatarRequestId, ProviderSession>,
    running: BTreeMap<String, ProviderSession>,
    /// What the gateway will do once each received request is processed: (attempt, sent at,
    /// applied at, the exchange).
    applying: Vec<((AvatarRequestId, u32), u64, u64, Exchange)>,
    lost: BTreeSet<(AvatarRequestId, u32)>,
    closing: BTreeSet<(AvatarRequestId, u32)>,
}

/// What an exchange does once its request reaches the gateway.
#[derive(Clone)]
enum Exchange {
    Create,
    Terminate(ProviderSession),
}

impl Gateway {
    fn launch(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        exchange: Exchange,
        now: u64,
        timeline: &mut Timeline<Src, Ev>,
    ) {
        let src = Src::Attempt(request, attempt);
        let replay = matches!(exchange, Exchange::Create) && self.received(request);
        let rng = &mut self.rng;
        let mut at = now + rng.u64(30..400);
        let fate = rng.u16(0..1000);
        // Silence is not forever: an unanswered SYN gives up in about two minutes and a load
        // balancer drops an idle connection in one or two.
        let gives_up = |rng: &mut Rng, at: u64, timeline: &mut Timeline<Src, Ev>| {
            let at = at + rng.u64(60_000..130_000);
            timeline.at(
                at,
                src,
                Ev::Failed {
                    request,
                    attempt,
                    failure: AttemptFailure::Connection("timed out".into()),
                },
            );
            Some(at)
        };
        let ends = match fate {
            0..30 => gives_up(rng, at, timeline),
            30..80 => {
                timeline.at(
                    at,
                    src,
                    Ev::Failed {
                        request,
                        attempt,
                        failure: AttemptFailure::Connection("refused".into()),
                    },
                );
                Some(at)
            }
            _ => {
                timeline.at(at, src, Ev::Connected { request, attempt });
                at += rng.u64(1..20);
                timeline.at(at, src, Ev::RequestWritten { request, attempt });
                let sent = at;
                at += match exchange {
                    // A replay of a create the gateway already made answers from its idempotency
                    // record, without a second provider round trip.
                    Exchange::Create if replay => rng.u64(20..200),
                    Exchange::Create => rng.u64(500..4_000),
                    Exchange::Terminate(_) => rng.u64(50..500),
                };
                match fate {
                    80..160 => {
                        // A replay of a create the gateway made can fail in front of it (rate
                        // limits, overload), never with a final answer that it was not made.
                        let status = [429, 500, 503, 400][rng.usize(0..if replay { 3 } else { 4 })];
                        let failure = AttemptFailure::Rejected {
                            status,
                            message: format!("status {status}"),
                            request_id: None,
                        };
                        timeline.at(at, src, Ev::Failed { request, attempt, failure });
                        Some(at)
                    }
                    // Received, then silence: it never applies.
                    160..180 => gives_up(rng, at, timeline),
                    _ => {
                        let event = match &exchange {
                            Exchange::Create => {
                                let n = request.0;
                                let unterminable = rng.u8(0..100) < 5;
                                let session = self.sessions.entry(request).or_insert_with(|| {
                                    ProviderSession {
                                        session_id: Some(format!("s{n}")),
                                        provider_session_id: Some(format!("p{n}")),
                                        terminate_token: (!unterminable).then(|| format!("t{n}")),
                                        sample_rate: 24_000,
                                    }
                                });
                                Ev::Created { request, attempt, session: session.clone() }
                            }
                            Exchange::Terminate(_) => Ev::Terminated { request, attempt },
                        };
                        // It applies on the gateway; on some the answer is lost on the way back.
                        let answer = if self.rng.u8(0..100) < 8 {
                            Ev::Failed {
                                request,
                                attempt,
                                failure: AttemptFailure::Connection("reset".into()),
                            }
                        } else {
                            event
                        };
                        self.applying.push(((request, attempt), sent, at, exchange));
                        timeline.at(at, src, answer);
                        Some(at)
                    }
                }
            }
        };
        if self.rng.u8(0..100) < 2 {
            self.lost.insert((request, attempt));
        } else if let Some(at) = ends {
            self.closing.insert((request, attempt));
            timeline.at(at + self.rng.u64(0..30), src, Ev::AttemptClosed { request, attempt });
        }
    }

    /// The gateway received a create of this operation already (made, or still being made).
    fn received(&self, request: AvatarRequestId) -> bool {
        self.made.contains_key(&request) || self.applying.iter().any(|((r, _), ..)| *r == request)
    }

    /// The gateway processes what it received by `now` (all of it, with `None`).
    fn apply(&mut self, now: Option<u64>) {
        let due: Vec<_> = self
            .applying
            .extract_if(.., |(_, _, at, _)| now.is_none_or(|now| *at <= now))
            .collect();
        for ((request, _), _, _, exchange) in due {
            match exchange {
                // One session per operation, ever: a later copy of a create it already made
                // (a replay it received twice) makes nothing, even once that session ended.
                Exchange::Create if self.made.contains_key(&request) => {}
                Exchange::Create => {
                    let session = self.sessions[&request].clone();
                    self.made.insert(request, session.clone());
                    if let Some(id) = &session.provider_session_id {
                        self.running.insert(id.clone(), session);
                    }
                }
                Exchange::Terminate(session) => {
                    if let Some(id) = &session.provider_session_id {
                        self.running.remove(id);
                    }
                }
            }
        }
    }

    fn stop(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        now: u64,
        timeline: &mut Timeline<Src, Ev>,
    ) {
        // A request not sent yet never reaches the gateway; one it received still applies.
        self.applying.retain(|(key, sent, ..)| *key != (request, attempt) || *sent <= now);
        let src = Src::Attempt(request, attempt);
        let pending_close = timeline.forget_where(&src, |e| matches!(e, Ev::AttemptClosed { .. }));
        if self.lost.contains(&(request, attempt))
            || (self.closing.contains(&(request, attempt)) && !pending_close)
        {
            return;
        }
        self.closing.insert((request, attempt));
        timeline.at(now + self.rng.u64(0..30), src, Ev::AttemptClosed { request, attempt });
    }
}

/// The program around the instance.
struct Program {
    rng: Rng,
    /// The caller is still waiting for its create.
    waiting: bool,
    /// A delivered session finds its caller gone this often, in percent.
    refuses: u8,
}

fn policy(rng: &mut Rng) -> RequestPolicy {
    RequestPolicy {
        connect_timeout: Some(dur(10_000)),
        write_timeout: Some(dur(10_000)),
        read_timeout: rng.bool().then(|| dur([2_000, 10_000][rng.usize(0..2)])),
        total_timeout: (rng.u8(0..10) < 3).then(|| dur(rng.u64(3..20) * 1_000)),
        max_retry: rng.u32(0..=3),
        retry_interval: dur([500, 2_000][rng.usize(0..2)]),
    }
}

pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let policy = policy(&mut rng);
    let mut machine = AvatarSessionLifecycle::new(AVATAR, policy);
    let mut sched: Sched<Src, Ev> = Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let mut timeline = Timeline::default();
    let mut gateway = Gateway {
        rng: Rng::with_seed(rng.u64(..)),
        sessions: BTreeMap::new(),
        made: BTreeMap::new(),
        running: BTreeMap::new(),
        applying: Vec::new(),
        lost: BTreeSet::new(),
        closing: BTreeSet::new(),
    };
    let mut program =
        Program { rng: Rng::with_seed(rng.u64(..)), waiting: true, refuses: rng.u8(0..20) };
    timeline.at(0, Src::Caller, Ev::Create);
    if rng.u8(0..100) < 20 {
        timeline.at(rng.u64(0..3_000), Src::Caller, Ev::Cancel);
    }
    if rng.u8(0..100) < 15 {
        timeline.at(rng.u64(0..6_000), Src::Racer, Ev::Create);
    }
    // The call ends: early on some, after a conversation on most.
    let close_at = if rng.u8(0..100) < 30 { rng.u64(0..6_000) } else { rng.u64(6_000..60_000) };
    timeline.at(close_at, Src::Program, Ev::Close);

    let mut trace = Trace { read_limit: policy.read_timeout.is_some(), ..Trace::default() };
    let mut now = 0;
    for _ in 0..100_000 {
        gateway.apply(Some(now));
        timeline.release(now, &mut sched);
        let event = if machine.wake_at().is_some_and(|at| at <= ts(now)) {
            Ev::Wake
        } else {
            match sched.next(&[]) {
                Some(Picked::Item { item, .. }) => item,
                Some(Picked::Process(_)) => unreachable!("no processes"),
                None => {
                    match timeline.next_due().into_iter().chain(machine.wake_at().map(ms)).min() {
                        Some(at) => now = at.max(now + 1),
                        None => break,
                    }
                    continue;
                }
            }
        };
        if matches!(event, Ev::Cancel) {
            program.waiting = false;
        }
        let effects = machine.handle(event.clone(), ts(now));
        for effect in &effects {
            match effect {
                Fx::AvatarSessionCreate { request, attempt, .. } => {
                    gateway.launch(*request, *attempt, Exchange::Create, now, &mut timeline);
                }
                Fx::AvatarSessionTerminate { request, attempt, session, .. } => {
                    gateway.launch(
                        *request,
                        *attempt,
                        Exchange::Terminate(session.clone()),
                        now,
                        &mut timeline,
                    );
                }
                Fx::AvatarRequestCancel { request, attempt } => {
                    gateway.stop(*request, *attempt, now, &mut timeline)
                }
                // The caller takes it, unless it is gone by the time it arrives.
                Fx::CreateFinished(Ok(_)) => {
                    let gone = !program.waiting || program.rng.u8(0..100) < program.refuses;
                    let at = now + program.rng.u64(0..5);
                    timeline.at(
                        at,
                        Src::Caller,
                        if gone { Ev::DeliveryRefused } else { Ev::Adopted },
                    );
                }
                // The program's `aclose()` raised; it closes again a moment later, or not.
                Fx::TerminateFailed { .. } if program.rng.u8(0..100) < 70 => {
                    timeline.at(now + program.rng.u64(500..3_000), Src::Program, Ev::Close);
                }
                _ => {}
            }
        }
        trace.steps.push(Step { at: now, event, effects });
    }
    trace.settled = machine.is_settled();
    gateway.apply(None);
    trace.made = gateway.made;
    trace.running = gateway.running.into_values().collect();
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    for effect in trace.steps.iter().flat_map(|s| &s.effects) {
        reached.insert(match effect {
            Fx::CreateRefused => "CreateRefused",
            Fx::AttemptFailed { .. } => "Retried",
            Fx::CreateFinished(Ok(_)) => "Delivered",
            Fx::CreateFinished(Err(CreateFailure::Closed)) => "ClosedWhileCreating",
            Fx::CreateFinished(Err(CreateFailure::Api(_))) => "CreateFailed",
            Fx::Retiring { reason: Retirement::CallerGone, .. } => "RetiredCallerGone",
            Fx::Retiring { reason: Retirement::Refused, .. } => "RetiredRefused",
            Fx::Retiring { reason: Retirement::Closed, .. } => "RetiredClosed",
            Fx::Retiring { reason: Retirement::Late, .. } => "RetiredLate",
            Fx::Unterminable(_) => "Unterminable",
            Fx::SessionTerminated(_) => "Terminated",
            Fx::TerminateFailed { .. } => "TerminateFailed",
            Fx::CleanupIncomplete { .. } => "CleanupIncomplete",
            Fx::RetirementIncomplete { .. } => "RetirementIncomplete",
            _ => continue,
        });
    }
    reached
}

// ----------------------------------------------------------------------------------------------
// The rules
// ----------------------------------------------------------------------------------------------

fn effects(trace: &Trace) -> impl Iterator<Item = (&Step, &Fx)> {
    trace.steps.iter().flat_map(|s| s.effects.iter().map(move |e| (s, e)))
}

fn never_leaked(trace: &Trace) -> Vec<String> {
    // What the instance said it could not end, and still holds.
    let mut reported: Vec<&ProviderSession> = Vec::new();
    for (_, effect) in effects(trace) {
        match effect {
            Fx::Unterminable(session) | Fx::TerminateFailed { session, .. } => {
                reported.push(session)
            }
            Fx::SessionTerminated(session) => reported.retain(|s| *s != session),
            _ => {}
        }
    }
    trace
        .running
        .iter()
        .filter(|s| !reported.contains(s))
        .map(|s| format!("{:?} still runs on the gateway, never reported", s.provider_session_id))
        .collect()
}

fn one_session(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    // Whether a create is in flight or a session was adopted, as the program sees it.
    let (mut in_flight, mut adopted) = (false, 0);
    for step in &trace.steps {
        let refused = step.effects.contains(&Fx::CreateRefused);
        match step.event {
            Ev::Create if (in_flight || adopted > 0) && !refused => {
                violations.push(format!(
                    "a create at {} ms accepted while one was in flight or adopted",
                    step.at
                ));
            }
            Ev::Create if !refused => in_flight = true,
            // The caller stopped waiting: the machine may refuse or take the next create.
            Ev::Cancel => in_flight = false,
            Ev::Adopted => adopted += 1,
            _ => {}
        }
        if step.effects.iter().any(|e| matches!(e, Fx::CreateFinished(_))) {
            in_flight = false;
        }
    }
    if adopted > 1 {
        violations.push(format!("{adopted} sessions adopted"));
    }
    violations
}

fn one_outcome(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    // The accepted create's caller, while it is owed an outcome: whether it stopped waiting.
    let mut owed: Option<bool> = None;
    for step in &trace.steps {
        match step.event {
            Ev::Create if !step.effects.contains(&Fx::CreateRefused) => owed = Some(false),
            Ev::Cancel if owed.is_some() => owed = Some(true),
            _ => {}
        }
        for effect in &step.effects {
            let Fx::CreateFinished(outcome) = effect else { continue };
            match owed.take() {
                Some(false)
                    if matches!(step.event, Ev::Close)
                        && !matches!(outcome, Err(CreateFailure::Closed)) =>
                {
                    violations.push(format!(
                        "a close at {} ms answered the creation {outcome:?}",
                        step.at
                    ));
                }
                Some(false) => {}
                Some(true) => violations.push(format!(
                    "an outcome at {} ms for a caller that stopped waiting",
                    step.at
                )),
                None => violations.push(format!("an outcome at {} ms nobody was owed", step.at)),
            }
        }
        if matches!(step.event, Ev::Close) {
            if owed == Some(false) {
                violations.push(format!(
                    "a close at {} ms left the creation's caller unanswered",
                    step.at
                ));
            }
            owed = None;
        }
    }
    if owed == Some(false) {
        violations.push("the creation's caller never got an outcome".into());
    }
    violations
}

fn retirements(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let (mut cancelled, mut refused, mut closed) = (false, false, false);
    let made: Vec<&ProviderSession> = trace.made.values().collect();
    for step in &trace.steps {
        match step.event {
            Ev::Cancel => cancelled = true,
            Ev::DeliveryRefused => refused = true,
            Ev::Close => closed = true,
            _ => {}
        }
        for effect in &step.effects {
            let session = match effect {
                Fx::Retiring { session, reason } => {
                    let caused = match reason {
                        Retirement::CallerGone => cancelled,
                        Retirement::Refused => refused,
                        Retirement::Closed => closed,
                        Retirement::Late => true,
                    };
                    if !caused {
                        violations.push(format!(
                            "{reason:?} retirement at {} ms without its cause",
                            step.at
                        ));
                    }
                    session
                }
                Fx::AvatarSessionTerminate { session, .. }
                | Fx::Unterminable(session)
                | Fx::SessionTerminated(session) => session,
                _ => continue,
            };
            if !made.contains(&session) {
                violations.push(format!(
                    "{:?} ended at {} ms, the gateway never made it",
                    session.provider_session_id, step.at
                ));
            }
        }
    }
    violations
}

fn released(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut ended: BTreeMap<(AvatarRequestId, u32), Option<u64>> = BTreeMap::new();
    let mut done: BTreeSet<(AvatarRequestId, u32)> = BTreeSet::new();
    for step in &trace.steps {
        match &step.event {
            Ev::Created { request, attempt, .. }
            | Ev::Terminated { request, attempt }
            | Ev::Failed { request, attempt, .. } => {
                if let Some(end) = ended.get_mut(&(*request, *attempt)) {
                    end.get_or_insert(step.at);
                }
            }
            Ev::AttemptClosed { request, attempt } => {
                done.insert((*request, *attempt));
            }
            _ => {}
        }
        for effect in &step.effects {
            match effect {
                Fx::AvatarSessionCreate { request, attempt, .. }
                | Fx::AvatarSessionTerminate { request, attempt, .. } => {
                    ended.insert((*request, *attempt), None);
                }
                Fx::AvatarRequestCancel { request, attempt } => {
                    if let Some(end) = ended.get_mut(&(*request, *attempt)) {
                        end.get_or_insert(step.at);
                    }
                }
                Fx::CleanupIncomplete { request, attempt } => {
                    let since = ended.get(&(*request, *attempt)).copied().flatten();
                    if since
                        .is_none_or(|s| step.at < s + ATTEMPT_CLEANUP_TIMEOUT.as_millis() as u64)
                    {
                        violations.push(format!(
                            "{request:?}/{attempt} reported missing at {} ms, ended {since:?}",
                            step.at
                        ));
                    }
                    done.insert((*request, *attempt));
                }
                _ => {}
            }
        }
    }
    for key in ended.keys().filter(|k| !done.contains(k)) {
        violations.push(format!("{key:?} neither released nor reported"));
    }
    if !trace.settled {
        violations.push("the machine did not settle".into());
    }
    violations
}

fn retirement_bounded(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let bound = RETIREMENT_TIMEOUT.as_millis() as u64;
    // Each create request, and when it was first left without its caller's answer.
    let mut left: BTreeMap<AvatarRequestId, Option<u64>> = BTreeMap::new();
    let mut reported: BTreeSet<AvatarRequestId> = BTreeSet::new();
    for step in &trace.steps {
        let leaves = matches!(step.event, Ev::Cancel | Ev::Close)
            || step
                .effects
                .iter()
                .any(|e| matches!(e, Fx::CreateFinished(Err(CreateFailure::Api(_)))));
        if leaves {
            for since in left.values_mut() {
                since.get_or_insert(step.at);
            }
        }
        for effect in &step.effects {
            match effect {
                Fx::AvatarSessionCreate { request, .. } => {
                    left.entry(*request).or_insert(None);
                }
                Fx::RetirementIncomplete { request } => {
                    if !reported.insert(*request) {
                        violations.push(format!("{request:?} reported unresolved twice"));
                    }
                    match left.get(request).copied().flatten() {
                        Some(since) if step.at >= since + bound => {}
                        since => violations.push(format!(
                            "{request:?} reported unresolved at {} ms, left {since:?}",
                            step.at
                        )),
                    }
                }
                _ => {}
            }
        }
    }
    violations
}

#[cfg(test)]
mod tests;
