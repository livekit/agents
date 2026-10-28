//! The A2A simulation (K-4, decisions 28, 82): clients sending messages to one agent's text door,
//! every decision `lk_agents_core::a2a::task::TaskMachine`'s, every storage operation executed by
//! the fake storage port ([`port`]) that injects each typed error (`Unavailable`, `Uncertain`
//! before and after the commit applied, `Rejected` from a held lane, `LeaseLost`,
//! `VersionConflict`, `NotFound`, a refused statement, a cancelled handler).
//!
//! Requests arrive out of order on three channels under door options the seed draws (the
//! default map, one policy for every channel, or a map mixing them), workers die mid-request and
//! their clients resend, other clients resend at once or cancel tasks. Faults stop after a
//! window; the run then goes on until every live request is answered.
//!
//! The rules ([`rules`]) hold over the whole run: a message is executed once; every request is
//! answered unless its worker died; an error answer is one the client can act on; a terminal task
//! never changes; a commit that applied is never answered as a failure (decision 39); every
//! answer to one message names its task; every received inbox message is drained.

pub mod port;
pub mod rules;

use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::storage::StorageError;
use lk_agents_core::a2a::task::{Answer, Effect, Event, Operation, TaskMachine};
use lk_agents_core::a2a::{DebounceSpec, Message, Policy, PolicySpec, TurnOptions};

use port::{Faults, OpenRun, Port};

/// How long the clients keep sending and the faults keep firing.
const FAULT_WINDOW: Duration = Duration::from_secs(30);
/// Past this, a request still unanswered is a liveness failure.
const HORIZON: Duration = Duration::from_secs(900);
const CHANNELS: [&str; 3] = ["a2a", "delegation", "sms"];
const POLICIES: [Policy; 5] =
    [Policy::Batch, Policy::Interrupt, Policy::Queue, Policy::Concurrent, Policy::Reject];

/// One HTTP request for a message, on its own worker.
#[derive(Debug)]
pub struct Request {
    pub message_id: String,
    channel: &'static str,
    pub policy: Policy,
    start: Duration,
    machine: TaskMachine,
    run: Option<OpenRun>,
    kept: bool,
    pending: Option<(u64, Operation, Duration)>,
    started: bool,
    /// The worker died: it stops and lets its lease lapse.
    pub killed: bool,
    pub answer: Option<Result<Answer, StorageError>>,
}

/// A finished run.
#[derive(Debug)]
pub struct Record {
    pub requests: Vec<Request>,
    pub port: Port,
    pub steps: u64,
    pub now: Duration,
}

fn options(rng: &mut Rng) -> TurnOptions {
    let policy = match rng.u8(0..3) {
        0 => PolicySpec::Default,
        1 => PolicySpec::All(POLICIES[rng.usize(0..POLICIES.len())]),
        _ => PolicySpec::ByChannel(
            CHANNELS
                .iter()
                .map(|c| ((*c).to_owned(), POLICIES[rng.usize(0..POLICIES.len())]))
                .collect(),
        ),
    };
    let debounce = if rng.bool() {
        DebounceSpec::All(Duration::from_millis(rng.u64(0..1500)))
    } else {
        DebounceSpec::Default
    };
    TurnOptions { policy, debounce, wait_timeout: Some(Duration::from_secs(rng.u64(2..8))) }
}

fn request(
    message_id: &str,
    channel: &'static str,
    start: Duration,
    opts: &TurnOptions,
    rng: &mut Rng,
) -> Request {
    let wire = port::wire("hello", message_id, channel);
    let message = Message::from_a2a(&wire, start.as_secs_f64(), None).expect("a valid message");
    let task_id = format!("AST_{:08x}", rng.u32(..));
    let machine = TaskMachine::new(port::LANE.into(), message, task_id, None, opts);
    Request {
        message_id: message_id.into(),
        channel,
        policy: machine.policy(),
        start,
        machine,
        run: None,
        kept: false,
        pending: None,
        started: false,
        killed: false,
        answer: None,
    }
}

/// One seeded run of `messages` messages.
pub fn run(seed: u64, messages: usize) -> Record {
    let mut rng = Rng::with_seed(seed);
    let opts = options(&mut rng);
    let faults = Faults {
        unavailable: rng.u8(0..6),
        lost_ack: rng.u8(0..12),
        refused_commit: rng.u8(0..4),
        lease_lost: rng.u8(0..4),
        cancelled: rng.u8(0..4),
    };
    let kill_pct = rng.u8(0..8);
    let mut port = Port::new(Duration::from_secs(rng.u64(2..6)));
    let mut requests: Vec<Request> = Vec::new();
    for i in 0..messages {
        let channel = CHANNELS[rng.usize(0..CHANNELS.len())];
        let start = Duration::from_millis(rng.u64(0..FAULT_WINDOW.as_millis() as u64 / 2));
        let id = format!("m{i}");
        requests.push(request(&id, channel, start, &opts, &mut rng));
        if rng.u8(0..4) == 0 {
            // a resend at once, before the first answer
            let again = start + Duration::from_millis(rng.u64(0..800));
            requests.push(request(&id, channel, again, &opts, &mut rng));
        }
    }
    let mut now = Duration::ZERO;
    let mut steps = 0u64;
    while now < HORIZON {
        let faulty = now < FAULT_WINDOW;
        let ready: Vec<usize> = (0..requests.len())
            .filter(|&i| {
                let r = &requests[i];
                !r.killed
                    && r.answer.is_none()
                    && (!r.started && r.start <= now
                        || r.pending.is_some()
                        || r.machine.wake_at().is_some_and(|at| at <= now))
            })
            .collect();
        if requests.iter().all(|r| r.killed || r.answer.is_some()) {
            break;
        }
        // a live holder renews its lease; a dead one lets it lapse
        for (i, r) in requests.iter().enumerate() {
            if !r.killed && r.answer.is_none() {
                port.renew(i, now);
            }
        }
        if faulty && rng.u16(0..400) == 0 {
            let ids: Vec<String> = port.tasks.keys().cloned().collect();
            if !ids.is_empty() {
                port.cancel(&ids[rng.usize(0..ids.len())], now);
            }
        }
        let Some(&i) = (!ready.is_empty()).then(|| &ready[rng.usize(0..ready.len())]) else {
            now += Duration::from_millis(50);
            continue;
        };
        steps += 1;
        let r = &mut requests[i];
        let effects = if !r.started {
            r.started = true;
            r.machine.handle(Event::Start, now)
        } else if let Some((id, operation, since)) = r.pending.take() {
            if faulty && matches!(operation, Operation::Run { .. }) && rng.u8(0..100) < kill_pct {
                r.killed = true;
                Vec::new()
            } else {
                let wait_timeout = r.machine.wait_timeout();
                match port.execute(
                    i,
                    &mut r.run,
                    &mut r.kept,
                    &operation,
                    now.saturating_sub(since),
                    wait_timeout,
                    now,
                    faulty.then_some(faults),
                    &mut rng,
                ) {
                    Some(result) => r.machine.handle(Event::Completed { id, result }, now),
                    None => {
                        r.pending = Some((id, operation, since));
                        now += Duration::from_millis(10);
                        continue;
                    }
                }
            }
        } else {
            r.machine.handle(Event::Wake, now)
        };
        for effect in effects {
            match effect {
                Effect::Execute { id, operation } => r.pending = Some((id, operation, now)),
                Effect::Finished(answer) => r.answer = Some(answer),
            }
        }
        // a client whose request failed or whose worker died sends the message again
        let resend = {
            let r = &requests[i];
            (r.killed || matches!(r.answer, Some(Err(_)))) && now < FAULT_WINDOW * 2
        };
        if resend && requests.iter().filter(|o| o.message_id == requests[i].message_id).count() < 4
        {
            let id = requests[i].message_id.clone();
            let channel = requests[i].channel;
            let again = now + Duration::from_millis(rng.u64(100..2000));
            requests.push(request(&id, channel, again, &opts, &mut rng));
        }
        now += Duration::from_millis(rng.u64(0..20));
    }
    Record { requests, port, steps, now }
}
