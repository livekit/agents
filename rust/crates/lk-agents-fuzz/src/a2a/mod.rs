//! The A2A simulation (K-4, decision 24): clients sending messages to one agent's text door
//! over its thread, every decision the A2A and storage cores', every statement run on the
//! storage simulation's SQLite ([`crate::storage::db`]) on the simulated clock.
//!
//! Requests ([`door`]) arrive out of order on three channels under door options the seed draws:
//! the default map (`a2a` interrupts, `delegation` queues, `sms` batches), or maps that put the
//! `batch` ([`batch`]), `concurrent` and `reject` policies on the same lane as the exclusive ones.
//! A client resends a message whose answer it lost (its worker died, or the door answered an
//! error) and sometimes resends one at once; a client answers a pending question by continuing its
//! task, which the door refuses once that task ended; another cancels tasks at random, racing
//! their turns; a third pages through `ListTasks` ([`list`]) while the tasks change. The faults
//! are AgentDB's: commit acknowledgments lost before or after they applied, unreachable windows,
//! paused workers whose lease lapses, workers lost mid-turn.
//!
//! The rules ([`rules`]) hold over the whole run: a message is executed once and every answer
//! to it names the same task; every request is answered once, unless its worker died, with an
//! outcome or an error the client can act on, never a framework-internal one; a terminal
//! task never changes; a cancel answers the row as it is; a committed run's rows are what it
//! staged (a lost acknowledgment is resolved by the run's identity, never a failure it did not
//! observe, decision 39) and its history items stay, once; only `interrupt` yields to a newer
//! message; one question is pending at a time; a received inbox message is drained; a refusal
//! happens only while another run holds the lane and leaves no row; a listing returns each task
//! that did not change once and refuses a page token made under other filters.
//!
//! Outside the simulation: the SSE stream of a `SubscribeToTask` and its resume from a cursor
//! (the current `Task` first, then the frames after the offset) are runtime code
//! (`lk_agents_runtime::a2a` and its response streams) with no core decision to step.

pub mod batch;
mod commit;
pub mod door;
pub mod list;
pub mod rules;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::time::Duration;

use fastrand::Rng;
use lk_agents_a2a::{
    DELEGATION_CHANNEL, DebounceSpec, Policy, PolicySpec, channel_of, resolve_policy,
};
use lk_agents_storage::StorageError;
use lk_agents_storage::db::{self as plans, Decode};
use lk_agents_storage::run::{CHANNEL_KEY, REASON_WORKER_LOST, TaskRow, TaskState};
use lk_agents_storage::schema::{EventRow, ItemRow};
use serde_json::{Map, Value};

use crate::sched::{Sampling, Sched, Step};
use crate::storage::db::{Clock, SimDb};
use door::Request;
use list::Lister;

pub const LANE: &str = "front-desk";
pub const THREAD: &str = "AT_sim";
/// An inbox channel: `batch` by default.
const SMS_CHANNEL: &str = "sms";

/// What a seed draws: the door's options, and production timings scaled down.
#[derive(Debug, Clone)]
pub struct Setup {
    pub policy: PolicySpec,
    pub lease_ttl: Duration,
    pub renew_interval: Duration,
    /// How long a handler works, in milliseconds.
    pub work_ms: (i64, i64),
    /// The share of commits whose acknowledgment is lost before it applied, and as many after.
    pub lost_ack_pct: u8,
    pub debounce: DebounceSpec,
}

/// One answer a request gave its client.
#[derive(Debug, Clone)]
pub struct Response {
    pub request: usize,
    pub message_id: String,
    pub answer: Result<lk_agents_a2a::TurnResult, StorageError>,
    /// Answered after the bounded wait on an unresolved commit (decision 39).
    pub uncertain: bool,
}

/// A run that staged its commit: each row it writes, with its status.
#[derive(Debug, Clone)]
pub struct StagedRun {
    pub run_id: String,
    /// The messages of its turn: one, or a drained batch.
    pub messages: Vec<String>,
    pub interrupt: bool,
    /// It held the lease; a `concurrent` run holds none.
    pub leased: bool,
    /// The turn yielded to a newer message waiting for the lane.
    pub yielded_to_waiter: bool,
    pub writes: Vec<(String, TaskState)>,
    /// The history items it adds.
    pub items: Vec<String>,
}

/// A `reject` request refused: its task, and whether another run held the lane when it was
/// refused, as the database stood.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub request: usize,
    pub task_id: String,
    pub held: bool,
}

/// A client's `tasks/cancel`: the row before and what the cancel answered.
#[derive(Debug, Clone)]
pub struct Cancel {
    pub task_id: String,
    pub before: TaskState,
    pub after: Option<TaskState>,
}

/// What the run observed, for the rules.
#[derive(Debug, Default)]
pub struct Observed {
    pub responses: Vec<Response>,
    /// Requests whose worker died before answering.
    pub lost: Vec<usize>,
    pub staged: Vec<StagedRun>,
    pub cancels: Vec<Cancel>,
    pub listings: Vec<list::Listing>,
    pub refusals: Vec<Refusal>,
    /// Messages a `batch` request put in the inbox, by request.
    pub received: Vec<(usize, String)>,
    /// Each task's statuses as they changed, step by step.
    pub statuses: BTreeMap<String, Vec<TaskState>>,
    /// The most input-required rows holding a question at one time, but those asked by
    /// `concurrent` runs, which may overlap.
    pub most_pending: usize,
    /// The tasks pending at once when the most were.
    pub most_pending_tasks: Vec<String>,
    /// Tasks a `concurrent` run left waiting on a question.
    pub concurrent_asks: BTreeSet<String>,
    pub deduplicated: u64,
    pub continuations: u64,
    pub interrupted: u64,
    pub client_canceled: u64,
    pub lost_acks: u64,
    pub unresolved: u64,
    pub rejected: u64,
    pub lease_timeouts: u64,
    /// Drain turns over more than one message.
    pub batched: u64,
    pub concurrent_runs: u64,
    pub follow_ups_refused: u64,
}

/// Everything a finished run leaves for the rules: what was observed and the database after.
#[derive(Debug)]
pub struct Record {
    pub seed: u64,
    pub steps: u64,
    pub policy: PolicySpec,
    /// Requests sent, by id: the message each carried.
    pub requests: Vec<(usize, String)>,
    /// The messages sent continuing a task (`taskId` set): answers to a pending question.
    pub continued: BTreeSet<String>,
    /// Each message's policy, by its channel.
    pub policies: BTreeMap<String, Policy>,
    pub seen: Observed,
    /// The tasks once every request was done, before the sweep failed what dead workers left.
    pub at_rest: Vec<TaskRow>,
    pub tasks: Vec<TaskRow>,
    pub items: Vec<ItemRow>,
    pub events: Vec<EventRow>,
    /// Violations found while running (a terminal row that changed).
    pub online: Vec<String>,
}

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = u64::from_be_bytes(*b"\0\0\0sched");
const CANCELER: u64 = 1 << 20;
const LISTER: u64 = CANCELER + 1;
const WORKERS: usize = 3;
/// The simulated clock starts at a real time (2026-01-01, in milliseconds): the database's
/// `unixepoch()` is never 0, which a released lease's `lease_until` would equal.
const EPOCH: i64 = 1_767_225_600_000;

/// The client side of the run: what it sends next and what it learned from the answers.
struct Clients {
    /// The question the client last saw pending: it answers by continuing that task.
    pending: Option<String>,
    cancels: usize,
    cancel_at: i64,
}

/// One seeded run with `messages` messages, the due processes stepped in `sampling` order.
pub fn run(seed: u64, messages: usize, sampling: Sampling) -> Record {
    let mut rng = Rng::with_seed(seed);
    let mut sched: Sched<Infallible, Infallible> =
        Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM), &[]);
    let timing = Setup {
        policy: draw_policy(&mut rng),
        lease_ttl: Duration::from_secs(rng.u64(2..6)),
        renew_interval: Duration::from_millis(rng.u64(500..1500)),
        work_ms: (50, rng.i64(200..3000)),
        lost_ack_pct: rng.u8(0..15),
        debounce: match rng.u8(0..3) {
            0 => DebounceSpec::All(Duration::from_millis(rng.u64(100..800))),
            _ => DebounceSpec::Default,
        },
    };
    let clock = Clock::default();
    clock.set(EPOCH);
    let mut db = SimDb::new(clock.clone());
    let mut seen = Observed::default();
    let mut requests: Vec<Request> = Vec::new();
    for id in 0..messages {
        requests.push(Request::new(id, None, EPOCH + rng.i64(0..8000), rng.usize(0..WORKERS)));
    }
    let mut clients =
        Clients { pending: None, cancels: rng.usize(0..=3), cancel_at: EPOCH + rng.i64(0..6000) };
    let mut lister = Lister::new(EPOCH + rng.i64(500..6000));
    let mut answered = 0;
    let mut steps = 0u64;
    let mut online = Vec::new();
    loop {
        let now = clock.get();
        let mut due: Vec<u64> =
            requests.iter().filter(|r| !r.done() && r.wake() <= now).map(|r| r.id as u64).collect();
        if clients.cancels > 0 && clients.cancel_at <= now {
            due.push(CANCELER);
        }
        if !lister.done() && lister.wake() <= now {
            due.push(LISTER);
        }
        if due.is_empty() {
            let next = requests.iter().filter(|r| !r.done()).map(Request::wake);
            let next = next
                .chain((clients.cancels > 0).then_some(clients.cancel_at))
                .chain((!lister.done()).then(|| lister.wake()))
                .min();
            match next {
                Some(at) => clock.set(at.max(now + 1)),
                None => break,
            }
            continue;
        }
        steps += 1;
        let Some(Step::Process(p)) = sched.next(&due) else { unreachable!("a due process") };
        match p {
            CANCELER => {
                cancel(&mut db, &mut rng, &mut seen);
                clients.cancels -= 1;
                clients.cancel_at = now + rng.i64(100..3000);
            }
            LISTER => lister.step(now, &mut db, &mut rng, &mut seen),
            _ => {
                let r = &mut requests[p as usize];
                if r.message.is_none() {
                    r.message = Some(compose(r.id, &mut rng, clients.pending.as_deref()));
                }
                // a worker can stall (paused process) or die mid-turn, as real ones do
                match rng.u16(0..1000) {
                    0..3 if r.sent() => {
                        r.alive = false;
                        seen.lost.push(r.id);
                    }
                    3..10 => {
                        r.paused_until = now + rng.i64(0..3 * timing.lease_ttl.as_millis() as i64)
                    }
                    // most outages are a reconnect; some outlast the commit resolution's wait
                    10..14 => db.cut_for(match rng.u8(0..8) {
                        0 => rng.i64(8_000..20_000),
                        _ => rng.i64(100..3000),
                    }),
                    _ => r.step(now, &mut db, &mut rng, &timing, &mut seen),
                }
                // a client resends a message once at once, and again when its answer was lost
                let r = &requests[p as usize];
                let resend = (r.sent() && r.id < messages && rng.u8(0..40) == 0)
                    || (!r.alive && rng.u8(0..5) != 0);
                if resend && requests.len() < 4 * messages {
                    let at = now + rng.i64(10..3000);
                    let retry =
                        Request::new(requests.len(), r.message.clone(), at, rng.usize(0..WORKERS));
                    requests.push(retry);
                }
            }
        }
        for response in &seen.responses[answered..] {
            if let Ok(result) = &response.answer
                && result.task.status == TaskState::InputRequired
            {
                clients.pending = Some(result.task.task_id.clone());
            }
            // a follow-up the door refused is not sent again: the client starts a new task
            let refused = matches!(&response.answer,
                Err(StorageError::Server { code, .. }) if code == door::FOLLOW_UP_REFUSED);
            if refused {
                clients.pending = None;
            }
            if response.answer.is_err()
                && !refused
                && rng.u8(0..3) != 0
                && requests.len() < 4 * messages
            {
                let message = requests[response.request].message.clone();
                let at = now + rng.i64(100..3000);
                requests.push(Request::new(requests.len(), message, at, rng.usize(0..WORKERS)));
            }
        }
        answered = seen.responses.len();
        // a database request takes a few milliseconds
        clock.set(clock.get() + rng.i64(1..15));
        observe(&db, &mut seen, &mut online);
    }
    let at_rest = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    sweep(&mut db, &clock);
    observe(&db, &mut seen, &mut online);
    let tasks = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    let items = db.inspect(&plans::history(LANE).statement);
    let events = db.inspect(&plans::lane_events(LANE, 0).statement);
    Record {
        seed,
        steps,
        policy: timing.policy.clone(),
        requests: requests
            .iter()
            .filter_map(|r| Some((r.id, r.message.as_ref()?.message_id.clone())))
            .collect(),
        policies: requests
            .iter()
            .filter_map(|r| r.message.as_ref())
            .map(|m| (m.message_id.clone(), resolve_policy(&timing.policy, &channel_of(m))))
            .collect(),
        continued: requests
            .iter()
            .filter_map(|r| r.message.as_ref())
            .filter(|m| m.task_id.is_some())
            .map(|m| m.message_id.clone())
            .collect(),
        seen,
        at_rest: at_rest.iter().map(|r| TaskRow::from_row(r).expect("a task row")).collect(),
        tasks: tasks.iter().map(|r| TaskRow::from_row(r).expect("a task row")).collect(),
        items: items.iter().map(|r| ItemRow::from_row(r).expect("an item row")).collect(),
        events: events.iter().map(|r| EventRow::from_row(r).expect("an event row")).collect(),
        online,
    }
}

/// The door's options: the default map, or maps that put `batch`, `concurrent` and `reject` on
/// the lane beside the exclusive policies.
fn draw_policy(rng: &mut Rng) -> PolicySpec {
    let map = |pairs: &[(&str, Policy)]| {
        PolicySpec::ByChannel(pairs.iter().map(|(c, p)| ((*c).to_owned(), *p)).collect())
    };
    match rng.u8(0..6) {
        0 | 1 => PolicySpec::Default,
        2 => PolicySpec::All(Policy::Concurrent),
        3 => PolicySpec::All(Policy::Reject),
        4 => map(&[("a2a", Policy::Concurrent), (DELEGATION_CHANNEL, Policy::Batch)]),
        _ => map(&[("a2a", Policy::Reject), ("sms", Policy::Queue)]),
    }
}

/// The message request `id` sends: on the `a2a`, `delegation` or `sms` channel, continuing the
/// pending question's task when the client saw one.
fn compose(id: usize, rng: &mut Rng, pending: Option<&str>) -> a2a::Message {
    let mut message =
        a2a::Message::new(a2a::Role::User, vec![a2a::Part::text(format!("message {id}"))]);
    message.message_id = format!("m{id}");
    let channel = match rng.u8(0..3) {
        0 => None,
        1 => Some(DELEGATION_CHANNEL),
        _ => Some(SMS_CHANNEL),
    };
    if let Some(channel) = channel {
        let channel = Value::String(channel.to_owned());
        message.metadata = Some([(CHANNEL_KEY.to_owned(), channel)].into_iter().collect());
    }
    if rng.u8(0..3) != 0 {
        message.task_id = pending.map(str::to_owned);
    }
    message
}

/// A client's `tasks/cancel` of a task it knows, racing whatever runs it.
fn cancel(db: &mut SimDb, rng: &mut Rng, seen: &mut Observed) {
    let rows = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    if rows.is_empty() {
        return;
    }
    let row = TaskRow::from_row(&rows[rng.usize(0..rows.len())]).expect("a task row");
    let plan = plans::cancel(LANE, &row.task_id, &Map::new());
    let Ok(out) = db.batch(&plan.statements, None) else { return };
    let after = plan.decoder.decode(&out).expect("a cancel answer");
    seen.cancels.push(Cancel {
        task_id: row.task_id,
        before: row.status,
        after: after.map(|r| r.status),
    });
}

/// Every task's status now; a terminal status that changes is a violation at once.
fn observe(db: &SimDb, seen: &mut Observed, online: &mut Vec<String>) {
    let rows = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    let mut pending = Vec::new();
    for row in rows.iter().map(|r| TaskRow::from_row(r).expect("a task row")) {
        if row.status == TaskState::InputRequired
            && row.internal.as_ref().is_some_and(|i| i.ask.is_some())
            && !seen.concurrent_asks.contains(&row.task_id)
        {
            pending.push(row.task_id.clone());
        }
        let log = seen.statuses.entry(row.task_id.clone()).or_default();
        if log.last() == Some(&row.status) {
            continue;
        }
        if let Some(last) = log.last()
            && last.is_terminal()
        {
            online.push(format!("{}: terminal {last:?} became {:?}", row.task_id, row.status));
        }
        log.push(row.status);
    }
    if pending.len() > seen.most_pending {
        seen.most_pending = pending.len();
        seen.most_pending_tasks = pending;
    }
}

/// After every lease lapsed, one last holder fails what dead workers left running, so every
/// task ends terminal or waiting on its question.
fn sweep(db: &mut SimDb, clock: &Clock) {
    clock.set(clock.get() + 120_000);
    let rows = db.inspect(&plans::tasks(LANE, None, 10_000).statement);
    for row in rows.iter().map(|r| TaskRow::from_row(r).expect("a task row")) {
        if row.terminal() || row.status == TaskState::InputRequired {
            continue;
        }
        let pickup = plans::pickup(LANE, &row.task_id);
        let _ = db.batch(&pickup.statements, None);
        let failed =
            lk_agents_a2a::failed_message(&row.task_id, REASON_WORKER_LOST, "swept".into());
        let _ = db.batch(&[plans::fail_orphan(LANE, &row.task_id, &failed)], None);
    }
}
