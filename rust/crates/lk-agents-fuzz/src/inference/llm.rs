//! One chat completion request (`lk_agents_inference::llm::lifecycle::LlmRequest`) against a fake
//! gateway, with a consumer that waits for chunks, takes its time over them and may give up.
//!
//! The gateway, per attempt: a connection in 30-400 ms that may hang or be refused; a rejection
//! (429, 5xx, 4xx); a response whose first chunk and gaps are the recorded gpt-4.1-mini ones
//! (`corpus/calibration.json`), which may stall, drop, turn malformed or end without its
//! terminator; and an attempt release reported 0-30 ms after the exchange ends or is stopped, or,
//! rarely, never.
//!
//! The rules, over the trace:
//! * `llm_request_has_one_outcome`: one outcome unless the consumer cancelled first, none after
//!   that, nothing delivered after it;
//! * `llm_output_is_never_replayed`: no attempt launches after visible output was delivered;
//! * `llm_retries_are_bounded_and_backed_off`: at most `max_retry` retries, each launched exactly
//!   its backoff after the failure (100 ms, then `retry_interval`);
//! * `llm_deadlines_follow_the_policy`: a connect, write, read or total limit ends the attempt or
//!   the request at its deadline, never before it and never later; a read limit runs only while
//!   the consumer waits;
//! * `llm_only_the_current_attempt_delivers`: a stale attempt's chunk is never delivered;
//! * `llm_attempts_are_released`: every launched attempt is stopped or ends on its own, and its
//!   release is observed or reported missing; the machine settles.

use std::collections::{BTreeMap, BTreeSet};

use fastrand::Rng;
use lk_agents_abi::errors::TimeoutPhase;
use lk_agents_abi::resources::ResourceId;
use lk_agents_inference::llm::lifecycle::{
    ATTEMPT_CLEANUP_TIMEOUT, AttemptFailure, LlmRequest, LlmRequestEffect, LlmRequestEvent,
    LlmRequestId, RequestPolicy,
};
use lk_agents_inference::llm::parse::WireChunk;

use super::{Timeline, dur, ms, ts};
use crate::corpus;
use crate::sched::{Sampling, Sched, Step as Picked};

const ID: LlmRequestId = LlmRequestId(1);
const LLM: ResourceId = ResourceId(1);
const FIRST_CHUNK: [&str; 3] = ["llm", "openai/gpt-4.1-mini", "first_chunk"];
const CHUNK_GAP: [&str; 3] = ["llm", "openai/gpt-4.1-mini", "chunk_gap"];

/// The first retry's backoff (`APIConnectOptions`: the first retry waits 0.1 s).
const FIRST_RETRY_MS: u64 = 100;

/// A consumer that has no outcome by then gives up: the caller's own patience.
const PATIENCE_MS: u64 = 120_000;

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = 0x7363_6865_64;

/// Who reports an event: one attempt's transport, or the consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Attempt(u32),
    Consumer,
}

impl crate::sched::Source for Src {}

/// One delivered event (`None`: the request's start) and the effects it returned, at `at` ms.
#[derive(Debug, Clone)]
pub struct Step {
    pub at: u64,
    pub event: Option<LlmRequestEvent>,
    pub effects: Vec<LlmRequestEffect>,
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub policy: RequestPolicy,
    pub steps: Vec<Step>,
    /// The machine settled once nothing was left to report.
    pub settled: bool,
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
    ("llm_request_has_one_outcome", one_outcome),
    ("llm_output_is_never_replayed", never_replayed),
    ("llm_retries_are_bounded_and_backed_off", retries_backed_off),
    ("llm_deadlines_follow_the_policy", deadlines),
    ("llm_only_the_current_attempt_delivers", current_only),
    ("llm_attempts_are_released", released),
];

/// Every rule over `trace`, each violation prefixed with its rule.
pub fn check(trace: &Trace) -> Vec<String> {
    RULES
        .iter()
        .flat_map(|(name, rule)| rule(trace).into_iter().map(move |v| format!("{name}: {v}")))
        .collect()
}

/// The limits a program sets: the defaults (10 s each phase, 3 retries, 2 s apart), tighter or
/// disabled ones, and a total limit on some.
fn draw_policy(rng: &mut Rng) -> RequestPolicy {
    let mut limit = || match rng.u8(0..10) {
        0 => None,
        1..=3 => Some(dur([1_000, 2_000, 5_000][rng.usize(0..3)])),
        _ => Some(dur(10_000)),
    };
    let (connect, write, read) = (limit(), limit(), limit());
    RequestPolicy {
        connect_timeout: connect,
        write_timeout: write,
        read_timeout: read,
        total_timeout: (rng.u8(0..10) < 3).then(|| dur(rng.u64(2..30) * 1_000)),
        max_retry: rng.u32(0..=4),
        retry_interval: dur([500, 1_000, 2_000][rng.usize(0..3)]),
    }
}

fn text(content: &str) -> WireChunk {
    serde_json::from_value(
        serde_json::json!({"id": "c", "choices": [{"delta": {"content": content}}]}),
    )
    .expect("a chunk")
}

fn usage() -> WireChunk {
    serde_json::from_value(serde_json::json!({
        "id": "c", "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    }))
    .expect("a usage chunk")
}

/// The fake gateway: what each attempt will report, planned when it launches.
struct Gateway {
    rng: Rng,
    /// Attempts whose release is planned (or already reported).
    closing: BTreeSet<u32>,
    /// Attempts whose transport never reports its release.
    lost: BTreeSet<u32>,
}

impl Gateway {
    fn launch(&mut self, attempt: u32, now: u64, timeline: &mut Timeline<Src, LlmRequestEvent>) {
        let src = Src::Attempt(attempt);
        let (reports, ends_at) = self.plan(attempt, now);
        for (at, event) in reports {
            timeline.at(at, src, event);
        }
        if self.rng.u8(0..100) < 2 {
            self.lost.insert(attempt);
        } else if let Some(at) = ends_at {
            self.closing.insert(attempt);
            timeline.at(at + self.rng.u64(0..30), src, LlmRequestEvent::AttemptClosed { attempt });
        }
    }

    /// What one attempt reports, and when its exchange ends on its own (`None`: it hangs).
    fn plan(&mut self, attempt: u32, now: u64) -> (Vec<(u64, LlmRequestEvent)>, Option<u64>) {
        use LlmRequestEvent as E;
        let rng = &mut self.rng;
        let mut reports = Vec::new();
        let mut at = now + rng.u64(30..400);
        let fate = rng.u16(0..1000);
        match fate {
            // The connection hangs: nothing, until a limit or the consumer stops it.
            0..30 => return (reports, None),
            30..80 => {
                let failure = AttemptFailure::Connection("connection refused".into());
                reports.push((at, E::Failed { attempt, failure }));
                return (reports, Some(at));
            }
            _ => {}
        }
        reports.push((at, E::Connected { attempt }));
        at += rng.u64(1..30);
        reports.push((at, E::RequestWritten { attempt }));
        match fate {
            80..180 => {
                let status = [429, 500, 502, 503, 400, 401][rng.usize(0..6)];
                at += rng.u64(50..400);
                let message = format!("status {status}");
                let failure = AttemptFailure::Rejected { status, message, request_id: None };
                reports.push((at, E::Failed { attempt, failure }));
                return (reports, Some(at));
            }
            // Accepted, then silence.
            180..210 => return (reports, None),
            _ => {}
        }
        at += corpus::draw_ms(rng, &FIRST_CHUNK);
        let chunks = rng.u32(3..40);
        let broken = rng.u8(0..100);
        let breaks_at = rng.u32(0..chunks);
        for k in 0..chunks {
            if k == breaks_at {
                let failure = match broken {
                    0..8 => Some(AttemptFailure::Connection("connection reset".into())),
                    // The body ends without `[DONE]`: only the release says so.
                    8..11 => return (reports, Some(at)),
                    // The stream stalls.
                    11..14 => return (reports, None),
                    14..16 => Some(AttemptFailure::Malformed("expected value at line 1".into())),
                    _ => None,
                };
                if let Some(failure) = failure {
                    reports.push((at, E::Failed { attempt, failure }));
                    return (reports, Some(at));
                }
            }
            if rng.u8(0..10) == 0 {
                reports.push((at, E::Progress { attempt }));
            }
            reports.push((at, E::Chunk { attempt, chunk: text(&format!("w{attempt}.{k} ")) }));
            at += corpus::draw_ms(rng, &CHUNK_GAP);
        }
        if rng.bool() {
            reports.push((at, E::Chunk { attempt, chunk: usage() }));
        }
        reports.push((at, E::Ended { attempt }));
        (reports, Some(at))
    }

    /// The transport is stopped: what it had not reported yet is gone, and it releases the
    /// attempt (unless it never does).
    fn stop(&mut self, attempt: u32, now: u64, timeline: &mut Timeline<Src, LlmRequestEvent>) {
        let src = Src::Attempt(attempt);
        let pending_close =
            timeline.forget_where(&src, |e| matches!(e, LlmRequestEvent::AttemptClosed { .. }));
        if self.lost.contains(&attempt) || (self.closing.contains(&attempt) && !pending_close) {
            return;
        }
        self.closing.insert(attempt);
        timeline.at(now + self.rng.u64(0..30), src, LlmRequestEvent::AttemptClosed { attempt });
    }
}

/// The consumer: waits for the next chunk, takes its time over one, may give up.
struct Consumer {
    rng: Rng,
    waiting: bool,
    done: bool,
}

/// One seeded run, the pending reports picked in the `sampling` order.
pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let policy = draw_policy(&mut rng);
    let mut sched: Sched<Src, LlmRequestEvent> =
        Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let mut timeline = Timeline::default();
    let mut gateway = Gateway {
        rng: Rng::with_seed(rng.u64(..)),
        closing: BTreeSet::new(),
        lost: BTreeSet::new(),
    };
    let mut consumer = Consumer { rng: Rng::with_seed(rng.u64(..)), waiting: true, done: false };
    let cancel_at = if rng.u8(0..100) < 15 { rng.u64(0..20_000) } else { PATIENCE_MS };
    timeline.at(0, Src::Consumer, LlmRequestEvent::DemandStarted);
    timeline.at(cancel_at, Src::Consumer, LlmRequestEvent::Cancel);

    let (mut request, effects) = LlmRequest::start(ID, LLM, policy, ("<think>", "</think>"), ts(0));
    let mut trace = Trace { policy, steps: Vec::new(), settled: false };
    let mut now = 0;
    react(&effects, now, &mut gateway, &mut consumer, &mut timeline);
    trace.steps.push(Step { at: now, event: None, effects });
    for _ in 0..100_000 {
        timeline.release(now, &mut sched);
        let event = if request.wake_at().is_some_and(|at| at <= ts(now)) {
            LlmRequestEvent::Wake
        } else {
            match sched.next(&[]) {
                Some(Picked::Item { item, .. }) => item,
                Some(Picked::Process(_)) => unreachable!("no processes"),
                None => {
                    let next =
                        timeline.next_due().into_iter().chain(request.wake_at().map(ms)).min();
                    match next {
                        Some(at) => now = at.max(now + 1),
                        None => break,
                    }
                    continue;
                }
            }
        };
        if consumer.done
            && matches!(event, LlmRequestEvent::Cancel | LlmRequestEvent::DemandStarted)
        {
            continue;
        }
        let effects = request.handle(event.clone(), ts(now));
        react(&effects, now, &mut gateway, &mut consumer, &mut timeline);
        trace.steps.push(Step { at: now, event: Some(event), effects });
    }
    trace.settled = request.is_settled();
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

fn react(
    effects: &[LlmRequestEffect],
    now: u64,
    gateway: &mut Gateway,
    consumer: &mut Consumer,
    timeline: &mut Timeline<Src, LlmRequestEvent>,
) {
    for effect in effects {
        match effect {
            LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                gateway.launch(*attempt, now, timeline)
            }
            LlmRequestEffect::LlmRequestCancel { attempt, .. } => {
                gateway.stop(*attempt, now, timeline)
            }
            // It got a chunk: it stops waiting while it handles it (backpressure), then waits again.
            LlmRequestEffect::Deliver(_) if consumer.waiting && !consumer.done => {
                consumer.waiting = false;
                timeline.at(now, Src::Consumer, LlmRequestEvent::DemandEnded);
                let busy = if consumer.rng.u8(0..20) == 0 {
                    consumer.rng.u64(1_000..15_000)
                } else {
                    consumer.rng.u64(0..50)
                };
                timeline.at(now + busy, Src::Consumer, LlmRequestEvent::DemandStarted);
                consumer.waiting = true;
            }
            LlmRequestEffect::Finished(_) => consumer.done = true,
            _ => {}
        }
    }
}

/// What a trace reached, for the campaign's reach report.
pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    let mut current = None;
    for step in &trace.steps {
        if let Some(LlmRequestEvent::Chunk { attempt, .. }) = &step.event
            && current != Some(*attempt)
        {
            reached.insert("StaleChunk");
        }
        if matches!(step.event, Some(LlmRequestEvent::Cancel)) {
            reached.insert("Cancelled");
        }
        for effect in &step.effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => current = Some(*attempt),
                LlmRequestEffect::AttemptFailed { error, .. } => {
                    reached.insert("Retried");
                    if let Some(phase) = error.phase {
                        reached.insert(timeout_label(phase));
                    }
                }
                LlmRequestEffect::Finished(Ok(())) => {
                    reached.insert("Succeeded");
                }
                LlmRequestEffect::Finished(Err(error)) => {
                    reached.insert(if error.retryable { "GaveUpRetryable" } else { "GaveUpFinal" });
                    if let Some(phase) = error.phase {
                        reached.insert(timeout_label(phase));
                    }
                }
                LlmRequestEffect::CleanupIncomplete { .. } => {
                    reached.insert("CleanupIncomplete");
                }
                _ => {}
            }
        }
    }
    reached
}

fn timeout_label(phase: TimeoutPhase) -> &'static str {
    match phase {
        TimeoutPhase::Connect => "ConnectTimeout",
        TimeoutPhase::Write => "WriteTimeout",
        TimeoutPhase::Read => "ReadTimeout",
        TimeoutPhase::Total => "TotalTimeout",
        _ => "OtherTimeout",
    }
}

// ----------------------------------------------------------------------------------------------
// The rules
// ----------------------------------------------------------------------------------------------

fn effects(trace: &Trace) -> impl Iterator<Item = (&Step, &LlmRequestEffect)> {
    trace.steps.iter().flat_map(|s| s.effects.iter().map(move |e| (s, e)))
}

fn one_outcome(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let (mut outcomes, mut cancelled) = (0, false);
    for step in &trace.steps {
        if matches!(step.event, Some(LlmRequestEvent::Cancel)) && outcomes == 0 {
            cancelled = true;
        }
        for effect in &step.effects {
            match effect {
                LlmRequestEffect::Finished(_) if cancelled => {
                    violations
                        .push(format!("an outcome at {} ms after the consumer cancelled", step.at));
                }
                LlmRequestEffect::Finished(_) => outcomes += 1,
                LlmRequestEffect::Deliver(_) if outcomes > 0 || cancelled => {
                    violations.push(format!(
                        "a chunk delivered at {} ms after the request ended",
                        step.at
                    ));
                }
                _ => {}
            }
        }
    }
    if outcomes > 1 {
        violations.push(format!("{outcomes} outcomes"));
    }
    if outcomes == 0 && !cancelled {
        violations.push("the request never ended".into());
    }
    violations
}

fn never_replayed(trace: &Trace) -> Vec<String> {
    let mut delivered_at = None;
    let mut violations = Vec::new();
    for (step, effect) in effects(trace) {
        match effect {
            LlmRequestEffect::Deliver(chunk) if chunk.has_response() => {
                delivered_at.get_or_insert(step.at);
            }
            LlmRequestEffect::LlmRequestStart { attempt, .. } if let Some(at) = delivered_at => {
                violations.push(format!(
                    "attempt {attempt} launched at {} ms, output was delivered at {at} ms",
                    step.at
                ));
            }
            _ => {}
        }
    }
    violations
}

fn retries_backed_off(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut launches = 0u32;
    let mut failed: Option<(u32, u64)> = None;
    for (step, effect) in effects(trace) {
        match effect {
            LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                if *attempt != launches {
                    violations.push(format!("attempt {attempt} launched as launch {launches}"));
                }
                if *attempt > 0 {
                    let backoff = if *attempt == 1 {
                        FIRST_RETRY_MS
                    } else {
                        trace.policy.retry_interval.as_millis() as u64
                    };
                    match failed {
                        Some((f, at)) if f + 1 == *attempt && step.at == at + backoff => {}
                        Some((f, at)) => violations.push(format!(
                            "attempt {attempt} launched at {} ms; attempt {f} failed at {at} ms, backoff {backoff} ms",
                            step.at
                        )),
                        None => violations.push(format!("attempt {attempt} launched with no failure before it")),
                    }
                }
                launches += 1;
            }
            LlmRequestEffect::AttemptFailed { attempt, .. } => failed = Some((*attempt, step.at)),
            _ => {}
        }
    }
    if launches > trace.policy.max_retry + 1 {
        violations.push(format!("{launches} attempts, max_retry {}", trace.policy.max_retry));
    }
    violations
}

/// Where the current attempt is, as the policy's limits read it.
#[derive(Clone, Copy)]
enum Stage {
    Connecting(u64),
    Writing(u64),
    Reading(u64),
}

/// The deadline the current attempt's stage sets, with its phase.
fn stage_deadline(
    policy: &RequestPolicy,
    stage: Stage,
    demand: Option<u64>,
) -> Option<(u64, TimeoutPhase)> {
    let limit = |l: Option<std::time::Duration>| l.map(|d| d.as_millis() as u64);
    match stage {
        Stage::Connecting(since) => {
            limit(policy.connect_timeout).map(|l| (since + l, TimeoutPhase::Connect))
        }
        Stage::Writing(since) => {
            limit(policy.write_timeout).map(|l| (since + l, TimeoutPhase::Write))
        }
        Stage::Reading(last) => {
            let since = demand?.max(last);
            limit(policy.read_timeout).map(|l| (since + l, TimeoutPhase::Read))
        }
    }
}

fn deadlines(trace: &Trace) -> Vec<String> {
    let policy = &trace.policy;
    let total = policy.total_timeout.map(|d| d.as_millis() as u64);
    let mut violations = Vec::new();
    let mut current: Option<(u32, Stage)> = None;
    let mut demand: Option<u64> = None;
    let mut decided = false;
    for step in &trace.steps {
        // What should have fired before this step.
        if !decided
            && let Some(total) = total
            && step.at > total
        {
            violations
                .push(format!("still undecided at {} ms past the {total} ms total limit", step.at));
            decided = true;
        }
        if let Some((attempt, stage)) = current
            && let Some((due, phase)) = stage_deadline(policy, stage, demand)
            && step.at > due
        {
            violations.push(format!(
                "attempt {attempt} ran to {} ms past its {phase:?} deadline {due} ms",
                step.at
            ));
            current = None;
        }
        let ends = step.effects.iter().any(|e| {
            matches!(
                e,
                LlmRequestEffect::Finished(_)
                    | LlmRequestEffect::AttemptFailed { .. }
                    | LlmRequestEffect::LlmRequestCancel { .. }
            )
        });
        // A timeout this step reports must be due.
        for effect in &step.effects {
            let error = match effect {
                LlmRequestEffect::AttemptFailed { error, .. }
                | LlmRequestEffect::Finished(Err(error)) => error,
                _ => continue,
            };
            let Some(phase) = error.phase else { continue };
            let due = match phase {
                TimeoutPhase::Total => total,
                _ => current
                    .and_then(|(_, stage)| stage_deadline(policy, stage, demand))
                    .filter(|(_, p)| *p == phase)
                    .map(|(d, _)| d),
            };
            match due {
                Some(due) if step.at >= due => {}
                _ => violations.push(format!(
                    "a {phase:?} timeout at {} ms that was not due ({due:?})",
                    step.at
                )),
            }
        }
        match &step.event {
            Some(LlmRequestEvent::DemandStarted) => demand = Some(step.at),
            Some(LlmRequestEvent::DemandEnded) => demand = None,
            Some(LlmRequestEvent::Cancel) => decided = true,
            Some(event) => {
                if let Some((attempt, stage)) = &mut current {
                    *stage = match (event, *stage) {
                        (LlmRequestEvent::Connected { attempt: a }, Stage::Connecting(_))
                            if a == attempt =>
                        {
                            Stage::Writing(step.at)
                        }
                        (
                            LlmRequestEvent::RequestWritten { attempt: a },
                            Stage::Connecting(_) | Stage::Writing(_),
                        ) if a == attempt => Stage::Reading(step.at),
                        (
                            LlmRequestEvent::Progress { attempt: a }
                            | LlmRequestEvent::Chunk { attempt: a, .. },
                            _,
                        ) if a == attempt => Stage::Reading(step.at),
                        (_, stage) => stage,
                    };
                }
            }
            None => {}
        }
        if ends || matches!(step.event, Some(LlmRequestEvent::Cancel)) {
            current = None;
        }
        for effect in &step.effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                    current = Some((*attempt, Stage::Connecting(step.at)))
                }
                LlmRequestEffect::Finished(_) => decided = true,
                _ => {}
            }
        }
    }
    violations
}

fn current_only(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    let mut current = None;
    let mut live = false;
    for step in &trace.steps {
        let delivers = step.effects.iter().any(|e| matches!(e, LlmRequestEffect::Deliver(_)));
        match &step.event {
            Some(LlmRequestEvent::Chunk { attempt, .. })
                if delivers && (current != Some(*attempt) || !live) =>
            {
                violations.push(format!(
                    "attempt {attempt}'s chunk delivered at {} ms; current {current:?}",
                    step.at
                ));
            }
            Some(LlmRequestEvent::Chunk { .. }) => {}
            _ if delivers => {
                violations.push(format!("a delivery at {} ms not caused by a chunk", step.at))
            }
            _ => {}
        }
        for effect in &step.effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                    (current, live) = (Some(*attempt), true)
                }
                LlmRequestEffect::AttemptFailed { .. }
                | LlmRequestEffect::Finished(_)
                | LlmRequestEffect::LlmRequestCancel { .. } => live = false,
                _ => {}
            }
        }
    }
    violations
}

fn released(trace: &Trace) -> Vec<String> {
    let mut violations = Vec::new();
    // Per launched attempt: when it ended (its own terminal, or a stop), and how it was released.
    let mut ended: BTreeMap<u32, Option<u64>> = BTreeMap::new();
    let mut closed: BTreeSet<u32> = BTreeSet::new();
    let mut incomplete: BTreeSet<u32> = BTreeSet::new();
    let mut current = None;
    for step in &trace.steps {
        match &step.event {
            Some(LlmRequestEvent::Ended { attempt } | LlmRequestEvent::Failed { attempt, .. }) => {
                if let Some(end) = ended.get_mut(attempt) {
                    end.get_or_insert(step.at);
                }
            }
            Some(LlmRequestEvent::AttemptClosed { attempt }) => {
                closed.insert(*attempt);
                if let Some(end) = ended.get_mut(attempt) {
                    end.get_or_insert(step.at);
                }
            }
            _ => {}
        }
        let decided = step.effects.iter().any(|e| {
            matches!(e, LlmRequestEffect::Finished(_) | LlmRequestEffect::AttemptFailed { .. })
        }) || matches!(step.event, Some(LlmRequestEvent::Cancel));
        for effect in &step.effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                    ended.insert(*attempt, None);
                    current = Some(*attempt);
                }
                LlmRequestEffect::LlmRequestCancel { attempt, .. } => {
                    match ended.get_mut(attempt) {
                        Some(end) => {
                            end.get_or_insert(step.at);
                        }
                        None => violations.push(format!(
                            "attempt {attempt} stopped at {} ms, never launched",
                            step.at
                        )),
                    }
                }
                LlmRequestEffect::CleanupIncomplete { attempt, .. } => {
                    let since = ended.get(attempt).copied().flatten();
                    let deadline = ATTEMPT_CLEANUP_TIMEOUT.as_millis() as u64;
                    if since.is_none_or(|since| step.at < since + deadline) {
                        violations.push(format!("attempt {attempt}'s release reported missing at {} ms, ended {since:?}", step.at));
                    }
                    if closed.contains(attempt) {
                        violations.push(format!(
                            "attempt {attempt}'s release reported missing after it was released"
                        ));
                    }
                    incomplete.insert(*attempt);
                }
                _ => {}
            }
        }
        // The attempt the request stopped waiting on is stopped, unless it already ended.
        if decided
            && let Some(attempt) = current.take()
            && ended.get(&attempt).is_some_and(Option::is_none)
        {
            violations.push(format!(
                "attempt {attempt} left running when the request moved on at {} ms",
                step.at
            ));
        }
    }
    for (attempt, end) in &ended {
        if !closed.contains(attempt) && !incomplete.contains(attempt) {
            violations
                .push(format!("attempt {attempt} (ended {end:?}) neither released nor reported"));
        }
    }
    if !trace.settled {
        violations.push("the machine did not settle".into());
    }
    violations
}

#[cfg(test)]
mod tests;
