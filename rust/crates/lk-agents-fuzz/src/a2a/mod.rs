//! The A2A simulation (K-4): clients sending messages to one agent's text door, each request run
//! by the runtime's own executor (`lk_agents_runtime::a2a::Turn`, every decision
//! `lk_agents_core::a2a::task::TaskMachine`'s) over lk-agents-storage, on an in-memory database
//! that reads the run's virtual clock (`testing::MemoryConnection`) behind storage's
//! fault-injecting connection (`testing::Faulty`): a commit's acknowledgment lost before or after
//! it applied, a commit refused (a disk I/O error, or too large), the database unreachable for a
//! while. The handler, at its host port, loses the lane, is cancelled or loses the database now
//! and then.
//!
//! Requests arrive out of order on three channels under door options the seed draws (the
//! default map, one policy for every channel, or a map mixing them), workers die mid-turn (their
//! lease lapses) and their clients resend, other clients resend at once or cancel tasks, and a
//! client asked a question answers it, continuing the waiting task or not. The handler's turn is
//! several steps (the pending question, its checkpoints, its output, the settlement), each
//! interleaved with other requests by the database's latency. Faults stop after a window; the
//! run then goes on until every live request is answered.
//!
//! The rules ([`rules`]) hold over the whole run: a message is executed once and answered by its
//! own task or the one it was merged into; every request is answered unless its worker died; an
//! error answer is one the client can act on; a terminal task never changes; a commit that
//! applied is never answered as a failure; every answer to one message names its task; every
//! received inbox message is drained; at most one question waits on the lane; a commit leaves
//! every terminal row it wrote as it wrote it, unless a client canceled it, and none `working`.

mod database;
mod handler;
pub mod rules;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::Clock;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::thread::TaskRow;
use lk_agents_abi::storage::{StorageError, TaskState};
use lk_agents_core::a2a::{
    DebounceSpec, Policy, PolicySpec, TurnOptions, TurnResult, resolve_policy,
};
use lk_agents_runtime::a2a::Turn;
use lk_agents_runtime::net::clock::SimClock;
use lk_agents_storage::engine::{LeaseTiming, SchemaCell};
use lk_agents_storage::testing::CommitFault;
use lk_agents_storage::{SqliteConnection, Thread};

pub use database::Ledger;
use database::{Database, World};
use handler::{Handlers, Host, MARKER};

pub const LANE: &str = "front-desk";
const THREAD: &str = "AT_dst";

/// How long the clients keep sending and the faults keep firing.
const FAULT_WINDOW: Duration = Duration::from_secs(30);
/// Past this, a request still unanswered is a liveness failure.
const HORIZON: Duration = Duration::from_secs(900);
/// How often the clients and the faults act.
const TICK: Duration = Duration::from_millis(10);
const CHANNELS: [&str; 3] = ["a2a", "delegation", "sms"];
const POLICIES: [Policy; 5] =
    [Policy::Batch, Policy::Interrupt, Policy::Queue, Policy::Concurrent, Policy::Reject];

/// How often each fault fires, in percent, while faults are on.
#[derive(Debug, Clone, Copy, Default)]
pub struct Faults {
    pub unavailable: u8,
    pub lost_ack: u8,
    pub refused_commit: u8,
    pub lease_lost: u8,
    pub cancelled: u8,
}

/// One HTTP request for a message, on its own worker.
#[derive(Debug)]
pub struct Request {
    pub message_id: String,
    /// The task this message continues, by its `taskId`.
    pub continues: Option<String>,
    channel: &'static str,
    pub policy: Policy,
    start: Duration,
    task_id: String,
    started: bool,
    /// The worker died: it stops and lets its lease lapse.
    pub killed: bool,
    pub answer: Option<Result<TurnResult, StorageError>>,
}

/// A finished run.
#[derive(Debug)]
pub struct Record {
    pub requests: Vec<Request>,
    /// Every task row as the run left it.
    pub tasks: BTreeMap<String, TaskRow>,
    /// The messages of inbox rows whose task no turn answered (not ended, not asking, not
    /// answered `Working` by its handler).
    pub unanswered_receipts: Vec<String>,
    pub ledger: Ledger,
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
    continues: Option<String>,
    channel: &'static str,
    start: Duration,
    opts: &TurnOptions,
    rng: &mut Rng,
) -> Request {
    Request {
        message_id: message_id.into(),
        continues,
        channel,
        policy: resolve_policy(&opts.policy, channel),
        start,
        task_id: format!("AST_{:08x}", rng.u32(..)),
        started: false,
        killed: false,
        answer: None,
    }
}

/// A user message `message_id` on `channel`, continuing `task_id` when given.
fn wire(message_id: &str, channel: &str, task_id: Option<&str>) -> a2a::Message {
    let mut message = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("hello")]);
    message.message_id = message_id.into();
    message.task_id = task_id.map(str::to_owned);
    message.metadata = Some(
        [(
            lk_agents_abi::a2a::metadata::CHANNEL_KEY.to_owned(),
            serde_json::Value::String(channel.into()),
        )]
        .into(),
    );
    message
}

/// One seeded run of `messages` messages, on the DST's deterministic scheduler.
pub fn run(seed: u64, messages: usize) -> Record {
    crate::drive::simulate_with(seed.wrapping_add(1), simulate(seed, messages))
}

async fn simulate(seed: u64, messages: usize) -> Record {
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
    let ttl = Duration::from_secs(rng.u64(2..6));
    let timing = LeaseTiming::new(ttl, ttl / 3).expect("a lease timing");
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let epoch = clock.now();
    let world = World::new(rng.u64(..), clock.clone(), FAULT_WINDOW);
    let database = Database::new(world.clone()).expect("an in-memory database");
    let handlers = Handlers::new(world.clone(), faults);
    let schema = SchemaCell::default();
    let open = |holder: String, conn: Arc<dyn SqliteConnection>| {
        let schema = schema.clone();
        async move { Thread::new(conn, THREAD, timing, schema, &holder).await }
    };
    let client = open("client".into(), database.clone()).await.expect("a thread");

    let mut requests: Vec<Request> = Vec::new();
    for i in 0..messages {
        let channel = CHANNELS[rng.usize(0..CHANNELS.len())];
        let start = Duration::from_millis(rng.u64(0..FAULT_WINDOW.as_millis() as u64 / 2));
        let id = format!("m{i}");
        requests.push(request(&id, None, channel, start, &opts, &mut rng));
        if rng.u8(0..4) == 0 {
            // a resend at once, before the first answer
            let again = start + Duration::from_millis(rng.u64(0..800));
            requests.push(request(&id, None, channel, again, &opts, &mut rng));
        }
    }
    let mut workers: Vec<Option<tokio::task::JoinHandle<_>>> = Vec::new();
    let mut now = Duration::ZERO;
    while now < HORIZON {
        let faulty = now < FAULT_WINDOW;
        workers.resize_with(requests.len(), || None);
        let mut opened_failed = Vec::new();
        for (i, r) in requests.iter_mut().enumerate().filter(|(_, r)| !r.started && r.start <= now)
        {
            r.started = true;
            // Opening the thread is the request's first read: it fails as the database does.
            let thread = match open(format!("worker{i}"), database.clone()).await {
                Ok(thread) => thread,
                Err(e) => {
                    r.answer = Some(Err(e));
                    opened_failed.push(i);
                    continue;
                }
            };
            let host = Arc::new(Host { request: i, handlers: handlers.clone() });
            let turn = Turn::new(thread, LANE, host, clock.clone(), opts.clone());
            let message = wire(&r.message_id, r.channel, r.continues.as_deref());
            let task_id = r.task_id.clone();
            workers[i] =
                Some(tokio::spawn(async move { turn.run_message(message, Some(task_id)).await }));
        }
        // a worker dies mid-turn: its lease lapses
        let running: Vec<usize> = handlers.open.lock().unwrap().iter().copied().collect();
        if faulty && !running.is_empty() && rng.u16(0..1000) < u16::from(kill_pct) * 3 {
            let i = running[rng.usize(0..running.len())];
            if let Some(worker) = workers[i].take() {
                worker.abort();
                requests[i].killed = true;
                handlers.open.lock().unwrap().remove(&i);
                follow_up(&mut requests, i, now, &opts, &mut rng);
            }
        }
        let mut ended = opened_failed;
        for (i, worker) in workers.iter_mut().enumerate() {
            if worker.as_ref().is_some_and(tokio::task::JoinHandle::is_finished) {
                let answer = worker.take().expect("finished").await.expect("a worker that ran");
                requests[i].answer = Some(answer);
                ended.push(i);
            }
        }
        for i in ended {
            follow_up(&mut requests, i, now, &opts, &mut rng);
        }
        if faulty {
            if inject(&database, &world, &faults, &mut rng) {
                database.lapse_lease().await;
            }
            if rng.u16(0..400) == 0 {
                let started: Vec<String> =
                    requests.iter().filter(|r| r.started).map(|r| r.task_id.clone()).collect();
                if !started.is_empty() {
                    let task_id = &started[rng.usize(0..started.len())];
                    let _ = client.lane(LANE).cancel_task(task_id, Default::default()).await;
                }
            }
        }
        if requests.iter().all(|r| r.killed || r.answer.is_some()) {
            break;
        }
        clock.wait_until(clock.now().saturating_add(TICK)).await;
        now = clock.now().saturating_duration_since(epoch);
    }
    for worker in workers.into_iter().flatten() {
        worker.abort();
    }
    finish(requests, &database, &world, &handlers, &open, now).await
}

/// After request `i` ended: a client whose request failed or whose worker died sends the message
/// again; a client asked a question answers it, naming the waiting task or not.
fn follow_up(
    requests: &mut Vec<Request>,
    i: usize,
    now: Duration,
    opts: &TurnOptions,
    rng: &mut Rng,
) {
    let r = &requests[i];
    let resend = (r.killed || matches!(r.answer, Some(Err(_)))) && now < FAULT_WINDOW * 2;
    if resend && requests.iter().filter(|o| o.message_id == r.message_id).count() < 4 {
        let (id, continues, channel) = (r.message_id.clone(), r.continues.clone(), r.channel);
        let again = now + Duration::from_millis(rng.u64(100..2000));
        requests.push(request(&id, continues, channel, again, opts, rng));
    }
    let r = &requests[i];
    let asked = match &r.answer {
        Some(Ok(answer)) if answer.task.status == TaskState::InputRequired => {
            Some(answer.task.task_id.clone())
        }
        _ => None,
    };
    let reply = format!("{}_reply", r.message_id);
    if let Some(asked) = asked
        && now < FAULT_WINDOW * 2
        && rng.u8(0..10) < 8
        && !requests.iter().any(|o| o.message_id == reply)
    {
        let channel = r.channel;
        let continues = rng.bool().then_some(asked);
        let at = now + Duration::from_millis(rng.u64(50..1500));
        requests.push(request(&reply, continues, channel, at, opts, rng));
    }
}

/// Storage's faults, armed at random while faults are on: the next commit's acknowledgment lost
/// before or after it applies, or the commit refused; the database unreachable for a while.
/// True when the lane's lease lapses now as well.
fn inject(database: &Database, world: &World, f: &Faults, rng: &mut Rng) -> bool {
    let mut fire = |pct: u8, kind: &'static str| {
        let hit = pct > 0 && rng.u16(0..1000) < u16::from(pct);
        if hit {
            *world.ledger.lock().unwrap().injected.entry(kind).or_default() += 1;
        }
        hit
    };
    for (pct, fault, kind) in [
        (f.lost_ack, CommitFault::LoseAckBefore, "UncertainBefore"),
        (f.lost_ack, CommitFault::LoseAckAfter, "UncertainAfter"),
        (f.refused_commit, CommitFault::Refuse, "Refused"),
        (f.refused_commit, CommitFault::RefuseTooLarge, "RefusedTooLarge"),
    ] {
        if fire(pct, kind) {
            database.faults.fail_next_commit(fault);
        }
    }
    let unreachable = fire(f.unavailable, "Unavailable");
    let lapsed = fire(f.lease_lost, "LeaseLapsed");
    if unreachable {
        database.faults.unreachable_for(Duration::from_millis(rng.u64(50..1500)));
    }
    lapsed
}

/// What the run left: every task row, the receipts no turn answered, and which invocations'
/// commits applied, by the markers their outputs left in the history: what they executed, and
/// the tasks they answered `Working`.
async fn finish<F, Fut>(
    requests: Vec<Request>,
    database: &Database,
    world: &World,
    handlers: &Handlers,
    open: &F,
    now: Duration,
) -> Record
where
    F: Fn(String, Arc<dyn SqliteConnection>) -> Fut,
    Fut: std::future::Future<Output = Result<Thread, StorageError>>,
{
    let reader = open("reader".into(), database.memory()).await.expect("a thread").lane(LANE);
    let text = |row: &Vec<lk_agents_storage::sql::SqlValue>| {
        row[0].as_str().unwrap_or_default().to_owned()
    };
    let mut tasks = BTreeMap::new();
    for row in database.select("SELECT task_id FROM tasks").await {
        let task_id = text(&row);
        if let Ok(Some(task)) = reader.task(&task_id).await {
            tasks.insert(task_id, task);
        }
    }
    let mut ledger = world.ledger.lock().unwrap().clone();
    let invocations = handlers.invocations.lock().unwrap().clone();
    let working = handlers.working.lock().unwrap().clone();
    for row in database.select("SELECT id FROM items").await {
        let Some(n) = text(&row).strip_prefix(MARKER).and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        for message in &invocations[n] {
            ledger.executions.entry(message.clone()).or_default().insert(n);
        }
        ledger.answered_working.extend(working.get(&n).cloned());
    }
    let unanswered_receipts = database
        .select(
            "SELECT i.message_id, t.task_id FROM inbox i JOIN tasks t ON t.task_id = i.task_id \
             WHERE t.status IN ('submitted', 'working')",
        )
        .await
        .into_iter()
        .filter(|row| !ledger.answered_working.contains(row[1].as_str().unwrap_or_default()))
        .map(|row| text(&row))
        .collect();
    Record { requests, tasks, unanswered_receipts, ledger, now }
}
