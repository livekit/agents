use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, UNIX_EPOCH};

use parking_lot::Mutex;

use super::Registry;
use crate::drive::StorageSetup;
use crate::record::{StorageOperation, StorageStep};
use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as _,
};
use lk_agents_abi::storage::{StorageError, Unacknowledged};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{ChatItem, ChatMessage, Clock, Role, StreamInput, Ts};
use lk_agents_runtime::storage::backends::sqlite::{FileConnection, SqliteStorage, Workers};
use lk_agents_runtime::storage::sql::{BatchResult, Row, SqlValue, Statement, StatementKind};
use lk_agents_runtime::storage::testing::{Dying, TempDir, Wrapped, lessee};
use lk_agents_runtime::storage::thread::LeaseTiming;
use lk_agents_runtime::storage::thread::lease::Holder;
use lk_agents_runtime::storage::voice::VoiceStore;
use lk_agents_runtime::storage::{
    CommitOptions, ItemRow, PartitionAccess, RunOptions, Storage as _, Thread, ThreadDbConnection,
    WhenHeld,
};

// The voice partition runs on storage's own code (`VoiceStore`, `Partition`, `Lease`, renewal, fenced
// commit) over the production SQLite backend in a directory of the run's own: its workers are
// tasks on the run's scheduler, its `unixepoch()` the run's clock. Each thread connection is
// wrapped: it answers late, loses answers, refuses and goes down, and observes which lease holds
// the partition. A worker can also die with requests queued, at the scheduler. `layer` adds SQLite
// refusals, clock jumps, crashes and another worker on the partition.

mod layer;
pub(crate) mod text;
pub use layer::storage_faults;

const PARTITION_KEY: &str = "voice";

/// Database workers that died in this process's runs: the survey's evidence the fault is reached.
static WORKER_DEATHS: AtomicU64 = AtomicU64::new(0);

pub fn worker_deaths() -> u64 {
    WORKER_DEATHS.load(Ordering::Relaxed)
}

/// Batches a worker death left `Uncertain`, and those a resend of the same batch then answered.
static DEATHS_UNCERTAIN: AtomicU64 = AtomicU64::new(0);
static DEATHS_RESENT: AtomicU64 = AtomicU64::new(0);

/// `(uncertain, resolved by a resend)` over this process's runs.
pub fn death_resends() -> (u64, u64) {
    (DEATHS_UNCERTAIN.load(Ordering::Relaxed), DEATHS_RESENT.load(Ordering::Relaxed))
}

/// The call's partition: the production store, the recording port the session is handed, and the
/// database under both.
pub struct FakeStore {
    voice: Arc<VoiceStore>,
    pub port: Arc<dyn VoiceStorage>,
    database: Arc<FakeDatabase>,
    _dir: TempDir,
}

impl FakeStore {
    /// `VoiceStore::bound` over a fresh partition, as `Namespace::voice_store` builds it.
    pub async fn open(
        setup: StorageSetup,
        seed: u64,
        registry: Arc<Registry>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let dir = TempDir::new("lk-agents-fuzz-");
        let dying = Arc::new(Dying::new(registry.world.clone()));
        let skew = Arc::new(layer::Skewed::new(clock));
        let clock: Arc<dyn Clock> = skew.clone();
        let workers = Workers { spawner: dying.clone(), clock: clock.clone() };
        let storage = SqliteStorage::open(&dir.0, LeaseTiming::default(), workers);
        let storage = Arc::new(storage.await.expect("a storage directory"));
        let thread = storage.create_thread(Default::default()).await.expect("a thread");
        let observer = storage.connect(&thread.thread_id).await.expect("an observer");
        let database = Arc::new(FakeDatabase::new(
            setup,
            seed,
            registry.clone(),
            clock,
            (storage.clone(), thread.thread_id.clone(), observer),
            (dying, skew),
        ));
        let calls = database.clone();
        let wrap = move |inner: FileConnection| {
            Arc::new(FakeConnection { inner, db: calls.clone() }) as Arc<dyn ThreadDbConnection>
        };
        let backend = Arc::new(Wrapped { storage, wrap });
        let voice =
            VoiceStore::bound(backend, thread.thread_id, PARTITION_KEY.into(), None, Cx::NONE)
                .await
                .expect("a fresh partition_key loads");
        // A bound call (`store.bind(app)`): its store carries the call's session state from the
        // start, so each turn's start writes the partition too.
        let initial =
            serde_json::json!({"version": 1, "kind": "voice", "agent": "", "history_ids": []});
        lk_agents_abi::storage::context::ContextActivation::stage_session(
            &voice.activation(),
            initial,
        )
        .expect("an empty voice session state stages");
        let voice = Arc::new(voice);
        database.state.lock().armed = true;
        let port = Arc::new(RecordedStore { store: voice.clone(), registry });
        Self { voice, port, database, _dir: dir }
    }

    /// The host's `aclose()` once the session ended, then what it left in the database.
    pub async fn close(&self) {
        let _ = self.voice.aclose().await;
        self.database.settle().await;
        WORKER_DEATHS.fetch_add(self.database.dying.deaths(), Ordering::Relaxed);
        self.database.observer.close().await;
        self.database.storage.close().await;
    }

    /// An acquire's lease take applied and its answer still owed: the instant a dropped caller
    /// leaves the partition taken with no `Lease` to release it.
    pub fn take_unanswered(&self) -> bool {
        self.database.state.lock().taking > 0
    }
}

/// The thread's database as every connection to it meets it: seeded latency and faults, and what
/// was observed of its leases.
struct FakeDatabase {
    storage: Arc<SqliteStorage>,
    thread_id: String,
    /// A connection of the observer's own, never faulted.
    observer: FileConnection,
    dying: Arc<Dying>,
    /// The database's view of time: the run's clock, its wall skewed by `layer`'s jumps.
    clock: Arc<dyn Clock>,
    skew: Arc<layer::Skewed>,
    registry: Arc<Registry>,
    setup: StorageSetup,
    state: Mutex<State>,
}

struct State {
    rng: fastrand::Rng,
    /// Faults and latency start once the store is open.
    armed: bool,
    down_until: Ts,
    /// A fault that may cost ownership was injected: an outage or a write never applied.
    faulted: bool,
    /// When each lease's owner last sent a request carrying its token, or took it.
    touched: BTreeMap<String, Ts>,
    takes: u32,
    /// Every this many lease takes, a text turn moved the partition first.
    text_turn_every: u32,
    /// Lease takes applied whose answers are still owed.
    taking: u32,
    /// `layer`'s own draws.
    layer: fastrand::Rng,
    /// Another worker's connection is on the partition, and how many came.
    intruding: bool,
    intruders: u32,
    /// The call's lease tokens another worker overwrote.
    stolen: Vec<String>,
    /// Batches a worker death left `Uncertain`, until a resend of one is answered.
    uncertain: Vec<Vec<Statement>>,
}

/// What one request meets: its latency on either side of the apply, and how it fails.
struct Plan {
    before: Duration,
    after: Duration,
    refused: Option<StorageError>,
    answer_lost: bool,
    /// The database's worker dies with this write queued.
    worker_dies: bool,
    /// `after` is the deliberate slow answer, recorded as it starts.
    slowed: bool,
}

impl FakeDatabase {
    fn new(
        setup: StorageSetup,
        seed: u64,
        registry: Arc<Registry>,
        clock: Arc<dyn Clock>,
        (storage, thread_id, observer): (Arc<SqliteStorage>, String, FileConnection),
        (dying, skew): (Arc<Dying>, Arc<layer::Skewed>),
    ) -> Self {
        // Mixed: neighbouring seeds' streams stay correlated at one position for a while.
        let mut rng =
            fastrand::Rng::with_seed((seed ^ 0x6c61_6e65).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let text_turn_every = rng.u32(2..=6);
        let state = State {
            rng,
            armed: false,
            down_until: Ts::ZERO,
            faulted: false,
            touched: BTreeMap::new(),
            takes: 0,
            text_turn_every,
            taking: 0,
            layer: fastrand::Rng::with_seed(seed ^ 0x006c_6179_6572),
            intruding: false,
            intruders: 0,
            stolen: Vec::new(),
            uncertain: Vec::new(),
        };
        let state = Mutex::new(state);
        Self { storage, thread_id, observer, dying, clock, skew, registry, setup, state }
    }

    /// Stress frequencies, not measured rates.
    fn plan(&self, write: bool, take: bool, carried: Option<&str>) -> Plan {
        let now = self.clock.now();
        let mut state = self.state.lock();
        if let Some(token) = carried {
            state.touched.insert(token.to_owned(), now);
        }
        let mut plan = Plan {
            before: Duration::ZERO,
            after: Duration::ZERO,
            refused: None,
            answer_lost: false,
            worker_dies: false,
            slowed: false,
        };
        if !state.armed {
            return plan;
        }
        let State { rng, down_until, .. } = &mut *state;
        let ms = Duration::from_millis;
        (plan.before, plan.after) = if rng.u32(0..32) == 0 {
            (ms(rng.u64(0..=50)), ms(rng.u64(250..=1500)))
        } else {
            (ms(rng.u64(0..=2)), ms(rng.u64(1..=5)))
        };
        if take && let Some(answer) = self.setup.take_answer_ms {
            plan.after = ms(answer);
        }
        if self.setup.faults {
            // Answered after an interruption's watchdog gave up on the speech waiting for it.
            if take && rng.u32(0..4) == 0 {
                plan.after = ms(rng.u64(5_500..=8_000));
                plan.slowed = true;
            }
            if now >= *down_until && rng.u32(0..128) == 0 {
                // Mostly a blip; a quarter outlast the lease's TTL.
                let down = if rng.u32(0..4) == 0 {
                    rng.u64(10_500..=15_000)
                } else {
                    rng.u64(100..=3_000)
                };
                *down_until = now.saturating_add(ms(down));
            }
            match rng.u32(0..128) {
                0 if write => plan.answer_lost = true,
                2 if write => plan.worker_dies = true,
                1 if write => {
                    plan.refused =
                        Some(StorageError::Uncertain(Unacknowledged::sent("lost before")))
                }
                _ => {}
            }
        }
        let scheduled = self
            .setup
            .outage
            .is_some_and(|(from, len)| (from..from + len).contains(&(now.as_nanos() / 1_000_000)));
        if scheduled || now < state.down_until {
            plan.refused = Some(StorageError::Unavailable("the database is unreachable".into()));
        }
        state.faulted |= plan.refused.is_some();
        plan
    }

    /// Notes a batch a worker death left `Uncertain`, and a later identical batch answered.
    fn count_resend(
        &self,
        died: bool,
        sent: Vec<Statement>,
        result: &Result<BatchResult, StorageError>,
    ) {
        let mut state = self.state.lock();
        match result {
            Err(StorageError::Uncertain(_)) if died => {
                DEATHS_UNCERTAIN.fetch_add(1, Ordering::Relaxed);
                state.uncertain.push(sent);
            }
            Ok(_) => {
                if let Some(at) = state.uncertain.iter().position(|batch| *batch == sent) {
                    state.uncertain.swap_remove(at);
                    DEATHS_RESENT.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(_) => {}
        }
    }

    async fn wait(&self, latency: Duration) {
        if latency.is_zero() {
            return;
        }
        let until = self.clock.now().saturating_add(latency);
        self.registry.storage_due(until);
        self.clock.wait_until(until).await;
    }

    /// The lease holding the partition now, as `(token, until)` in database milliseconds.
    async fn lessee(&self) -> Option<(String, i64)> {
        lessee(&self.observer, PARTITION_KEY).await
    }

    /// Another worker's handle on the same database, beside the call's.
    async fn thread(&self) -> Option<Thread> {
        self.storage.open(&self.thread_id).await.ok()
    }

    fn unix_millis(&self) -> i64 {
        self.clock.wall().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64
    }

    /// After a lease take applied: the taker's lease is touched, or the partition was held, and by an
    /// owner that last spoke this long ago.
    async fn observe_take(&self, takes: &[Statement]) {
        let Some((token, until)) = self.lessee().await else { return };
        let now = self.clock.now();
        let taken = SqlValue::Text(token.clone());
        let ours = takes.iter().any(|s| s.params().contains(&taken));
        let mut state = self.state.lock();
        if ours {
            state.touched.insert(token, now);
        } else if until >= self.unix_millis() {
            let spoke = state.touched.get(&token).copied().unwrap_or(Ts::ZERO);
            let idle_ms = now.saturating_duration_since(spoke).as_millis() as u64;
            drop(state);
            self.registry.record_storage(StorageStep::PartitionHeld { idle_ms });
        }
    }

    /// A text turn between voice turns, by another worker over the same database: it takes the
    /// partition only if free, edits the history and commits.
    async fn text_turn(&self, turn: u32, what: u32) {
        let Some(thread) = self.thread().await else { return };
        let mut options = RunOptions::new(format!("text_{turn}"), "");
        options.access = PartitionAccess::Exclusive(WhenHeld::Reject);
        if let Ok(mut run) =
            thread.partition(PARTITION_KEY).open(options, &Holder::new("text")).await
        {
            let rows = run.history().to_vec();
            let mut items: Vec<ChatItem> =
                rows.iter().filter_map(|row| decode_chat_item(row.payload.clone()).ok()).collect();
            edit(&mut items, turn, what);
            let message_id =
                |id: &str| rows.iter().find(|row| row.id == id).and_then(|r| r.message_id.clone());
            *run.history_mut() = items
                .iter()
                .map(|item| ItemRow {
                    id: item.id().to_owned(),
                    message_id: message_id(item.id()),
                    payload: encode_chat_item(item),
                })
                .collect();
            let _ = run.commit(CommitOptions::default()).await;
        }
        thread.close().await;
    }

    /// Once the store closed: runs started and never ended, and a lease still held.
    async fn settle(&self) {
        let Some(thread) = self.thread().await else { return };
        let partition_key = thread.partition(PARTITION_KEY);
        let runs = partition_key.runs(usize::MAX).await.unwrap_or_default();
        let unended = runs.iter().filter(|run| run.interrupted()).count() as u32;
        // Another worker's lease is its own to release.
        let held = self.lessee().await.is_some_and(|(token, until)| {
            until >= self.unix_millis() && !token.starts_with(layer::INTRUDER)
        });
        let faulted = self.state.lock().faulted;
        self.registry.record_storage(StorageStep::PartitionSettled { unended, held, faulted });
        // The thread keeps its rows in the order they were first written.
        let items = partition_key.history().await.unwrap_or_default();
        let items = items.into_iter().map(|row| row.id).collect();
        self.registry.record_storage(StorageStep::Saved { items });
    }
}

/// One worker's connection to the database, as [`FakeDatabase`] faults it.
struct FakeConnection {
    inner: FileConnection,
    db: Arc<FakeDatabase>,
}

/// Counts a lease take whose answer is still owed, until it is answered or dropped.
struct Taking<'a>(&'a Mutex<State>);

impl Drop for Taking<'_> {
    fn drop(&mut self) {
        self.0.lock().taking -= 1;
    }
}

#[async_trait::async_trait]
impl ThreadDbConnection for FakeConnection {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        let db = &self.db;
        let plan = db.plan(false, false, None);
        let layer = db.layer(false, false);
        db.wait(plan.before).await;
        if let Some(error) = plan.refused.or(layer.refused) {
            return Err(error);
        }
        let rows = self.inner.query(statement).await;
        db.wait(plan.after).await;
        rows
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        let db = &self.db;
        let takes: Vec<Statement> = statements
            .iter()
            .filter(|s| s.kind() == Some(StatementKind::LeaseTake))
            .cloned()
            .collect();
        if !takes.is_empty() {
            let turn = {
                let mut state = db.state.lock();
                state.takes += 1;
                let turn = state.takes;
                (state.armed && turn.is_multiple_of(state.text_turn_every))
                    .then(|| (turn, state.rng.u32(0..4)))
            };
            if let Some((turn, what)) = turn {
                db.text_turn(turn, what).await;
            }
        }
        // A request carrying the holding lease's token is its owner speaking: a renewal, a
        // commit or a release.
        let carried = db.lessee().await.map(|(token, _)| token).filter(|token| {
            let token = SqlValue::Text(token.clone());
            statements.iter().any(|s| s.params().contains(&token))
        });
        let plan = db.plan(true, !takes.is_empty(), carried.as_deref());
        let layer = db.layer(true, !takes.is_empty());
        if let Some(steal) = layer.intrude {
            db.registry.world.spawn_detached(db.clone().intrude(steal));
        }
        db.wait(plan.before).await;
        if let Some(error) = plan.refused.or(layer.refused) {
            return Err(error);
        }
        if plan.worker_dies {
            db.dying.die_next_poll();
        }
        let sent = statements.clone();
        let result = if db.carries_stolen(&statements) {
            db.fenced(self.inner.batch(statements)).await
        } else {
            self.inner.batch(statements).await
        };
        db.count_resend(plan.worker_dies, sent, &result);
        if layer.crash && result.is_ok() {
            return Err(db.crash());
        }
        let _taking = if takes.is_empty() {
            None
        } else {
            db.observe_take(&takes).await;
            db.state.lock().taking += 1;
            Some(Taking(&db.state))
        };
        if plan.slowed {
            let ms = plan.after.as_millis() as u64;
            db.registry.record_storage(StorageStep::Slowed { ms });
        }
        db.wait(plan.after).await;
        if plan.answer_lost && result.is_ok() {
            return Err(StorageError::Uncertain(Unacknowledged::sent("the answer was lost")));
        }
        result
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

/// The session's storage port: the production store, every request and answer recorded.
struct RecordedStore {
    store: Arc<VoiceStore>,
    registry: Arc<Registry>,
}

impl VoiceStorage for RecordedStore {
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn StreamInput<VoiceStorageEvent>>,
    ) -> Result<(), StorageError> {
        let requested = match &command {
            VoiceStorageCommand::Settle { speech, .. } => Some((*speech, StorageOperation::Settle)),
            VoiceStorageCommand::Abort { speech } => Some((*speech, StorageOperation::Abandon)),
            VoiceStorageCommand::ObserveLoss { .. } => None,
        };
        if let Some((speech, operation)) = requested {
            self.registry.record_storage(StorageStep::Requested { speech, operation });
        }
        let events = Arc::new(RecordedEvents { events, registry: self.registry.clone() });
        self.store.clone().dispatch(command, events)
    }

    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        let operation = StorageOperation::Acquire;
        self.registry.record_storage(StorageStep::Requested { speech, operation });
        let pending = self.registry.storage_pending();
        Box::pin(async move {
            let _pending = pending;
            let result = self.store.acquire(speech, local).await;
            self.registry.record_storage(StorageStep::Answered {
                speech,
                operation,
                result: result.as_ref().map(|_| ()).map_err(Clone::clone),
                delivered: true,
            });
            result
        })
    }

    fn checkpoint(
        &self,
        sequence: u64,
        agent: String,
        history: Vec<ChatItem>,
    ) -> StorageWork<'_, ()> {
        self.store.checkpoint(sequence, agent, history)
    }

    fn lease_lost(&self) -> bool {
        self.store.lease_lost()
    }
}

struct RecordedEvents {
    events: Arc<dyn StreamInput<VoiceStorageEvent>>,
    registry: Arc<Registry>,
}

impl StreamInput<VoiceStorageEvent> for RecordedEvents {
    fn push(&self, event: VoiceStorageEvent) -> Result<(), lk_agents_abi::errors::ProviderError> {
        let (speech, operation, result) = match &event {
            VoiceStorageEvent::Settled { speech, result, .. } => {
                (*speech, StorageOperation::Settle, result.clone())
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                (*speech, StorageOperation::Abandon, result.clone())
            }
            VoiceStorageEvent::LeaseLost { speech } => {
                (*speech, StorageOperation::ObserveLoss, Ok(()))
            }
        };
        let pushed = self.events.push(event);
        let delivered = pushed.is_ok();
        self.registry.record_storage(StorageStep::Answered {
            speech,
            operation,
            result,
            delivered,
        });
        pushed
    }

    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.events.flush()
    }

    fn end_input(&self) {
        self.events.end_input()
    }
}

/// A text turn's edit: an agent answer rewritten (`what` 0), the two oldest plain messages
/// compacted into a summary (1), or an exchange appended. Tool items are never rewritten.
fn edit(partition_key: &mut Vec<ChatItem>, turn: u32, what: u32) {
    let plain = |item: &ChatItem| {
        item.as_message().is_some_and(|m| matches!(m.role, Role::User | Role::Assistant))
    };
    let said =
        |id: String, role, text: &str, at| ChatItem::Message(ChatMessage::text(id, role, text, at));
    match what {
        0 => {
            let answer = partition_key
                .iter_mut()
                .find(|item| item.as_message().is_some_and(|m| m.role == Role::Assistant));
            if let Some(answer) = answer {
                *answer = said(
                    answer.id().to_owned(),
                    Role::Assistant,
                    "Sure, edited.",
                    answer.created_at(),
                );
                return;
            }
        }
        1 => {
            let oldest: Vec<usize> = partition_key
                .iter()
                .enumerate()
                .filter(|(_, item)| plain(item))
                .map(|(i, _)| i)
                .take(2)
                .collect();
            if let [first, second] = oldest[..] {
                let at = partition_key[first].created_at();
                partition_key.remove(second);
                partition_key[first] = said(
                    format!("partition_summary_{turn}"),
                    Role::Assistant,
                    "Summary of the start.",
                    at,
                );
                return;
            }
        }
        _ => {}
    }
    let at = partition_key.iter().map(ChatItem::created_at).max().unwrap_or(UNIX_EPOCH);
    partition_key.push(said(
        format!("partition_text_{turn}_user"),
        Role::User,
        "can you also text me the address?",
        at,
    ));
    partition_key.push(said(
        format!("partition_text_{turn}_agent"),
        Role::Assistant,
        "Sure, I've sent it.",
        at,
    ));
}
