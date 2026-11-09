use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use super::Registry;
use crate::drive::StorageSetup;
use crate::record::{StorageOperation, StorageStep};
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as _,
};
use lk_agents_abi::{ChatItem, ChatMessage, Clock, Role, StreamInput, Ts};
use lk_agents_runtime::storage::sql::{BatchResult, Row, SqlValue, Statement};
use lk_agents_runtime::storage::testing::{MemoryConnection, MemoryStorage};
use lk_agents_runtime::storage::thread::lease::Holder;
use lk_agents_runtime::storage::thread::{LeaseTiming, SchemaCell};
use lk_agents_runtime::storage::voice::VoiceStore;
use lk_agents_runtime::storage::{
    CommitOptions, ItemRow, LaneAccess, RunOptions, Thread, ThreadDbConnection, WhenHeld,
};

// The voice lane runs on storage's own code (`VoiceStore`, `Lane`, `Lease`, renewal, fenced
// commit) over its in-memory SQLite, on the run's clock. Only the connection is faked: it answers
// late, loses answers, refuses and goes down, and observes which lease holds the lane.

const THREAD: &str = "AT_fuzz";
const LANE: &str = "voice";

/// What an acquire's lease take looks like on the wire: storage's `TakeLease` statements.
const TAKE: &str = "SET waiter = CASE WHEN lease_holder";

/// The call's lane: the production store, the recording port the session is handed, and the
/// database under both.
pub struct FakeStore {
    voice: Arc<VoiceStore>,
    pub port: Arc<dyn VoiceStorage>,
    database: Arc<FakeDatabase>,
}

impl FakeStore {
    /// `VoiceStore::bound` over a fresh lane, as `Storage._core_voice_store` builds it.
    pub async fn open(
        setup: StorageSetup,
        seed: u64,
        registry: Arc<Registry>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let database = Arc::new(FakeDatabase::new(setup, seed, registry.clone(), clock));
        let backend =
            Arc::new(MemoryStorage { connection: database.clone(), lease: LeaseTiming::default() });
        let voice = VoiceStore::bound(backend, THREAD.into(), LANE.into(), None)
            .await
            .expect("a fresh lane loads");
        let voice = Arc::new(voice);
        database.state.lock().unwrap().armed = true;
        let port = Arc::new(RecordedStore { store: voice.clone(), registry });
        Self { voice, port, database }
    }

    /// The host's `aclose()` once the session ended, then what it left in the database.
    pub async fn close(&self) {
        let _ = self.voice.aclose().await;
        self.database.settle().await;
    }

    /// An acquire's lease take applied and its answer still owed: the instant a dropped caller
    /// leaves the lane taken with no `Lease` to release it.
    pub fn take_unanswered(&self) -> bool {
        self.database.state.lock().unwrap().taking > 0
    }
}

/// Storage's in-memory database behind seeded latency and faults.
struct FakeDatabase {
    inner: MemoryConnection,
    clock: Arc<dyn Clock>,
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
    /// Every this many lease takes, a text turn moved the lane first.
    text_turn_every: u32,
    /// Lease takes applied whose answers are still owed.
    taking: u32,
}

/// What one request meets: its latency on either side of the apply, and how it fails.
struct Plan {
    before: Duration,
    after: Duration,
    refused: Option<StorageError>,
    answer_lost: bool,
}

impl FakeDatabase {
    fn new(setup: StorageSetup, seed: u64, registry: Arc<Registry>, clock: Arc<dyn Clock>) -> Self {
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
        };
        let inner = MemoryConnection::new(clock.clone()).expect("an in-memory database");
        Self { inner, clock, registry, setup, state: Mutex::new(state) }
    }

    /// Stress frequencies, not measured rates.
    fn plan(&self, write: bool, take: bool, carried: Option<&str>) -> Plan {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        if let Some(token) = carried {
            state.touched.insert(token.to_owned(), now);
        }
        let mut plan = Plan {
            before: Duration::ZERO,
            after: Duration::ZERO,
            refused: None,
            answer_lost: false,
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
                1 if write => plan.refused = Some(StorageError::Uncertain("lost before".into())),
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

    async fn wait(&self, latency: Duration) {
        if latency.is_zero() {
            return;
        }
        let until = self.clock.now().saturating_add(latency);
        self.registry.storage_due(until);
        self.clock.wait_until(until).await;
    }

    /// The lease holding the lane now, as `(token, until)` in the database's seconds.
    async fn lessee(&self) -> Option<(String, i64)> {
        let read = Statement::raw("SELECT lease_token, lease_until FROM lanes WHERE lane = ?1")
            .with(vec![SqlValue::Text(LANE.into())]);
        let rows = self.inner.query(read).await.ok()?;
        match rows.first()?.as_slice() {
            [SqlValue::Text(token), SqlValue::Int(until)] => Some((token.clone(), *until)),
            _ => None,
        }
    }

    fn unix_secs(&self) -> i64 {
        self.clock.wall().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
    }

    /// After a lease take applied: the taker's lease is touched, or the lane was held, and by an
    /// owner that last spoke this long ago.
    async fn observe_take(&self, takes: &[Statement]) {
        let Some((token, until)) = self.lessee().await else { return };
        let now = self.clock.now();
        let taken = SqlValue::Text(token.clone());
        let ours = takes.iter().any(|s| s.params().contains(&taken));
        let mut state = self.state.lock().unwrap();
        if ours {
            state.touched.insert(token, now);
        } else if until >= self.unix_secs() {
            let spoke = state.touched.get(&token).copied().unwrap_or(Ts::ZERO);
            let idle_ms = now.saturating_duration_since(spoke).as_millis() as u64;
            drop(state);
            self.registry.record_storage(StorageStep::LaneHeld { idle_ms });
        }
    }

    /// A text turn between voice turns, by another worker over the same database: it takes the
    /// lane only if free, edits the history and commits.
    async fn text_turn(&self, turn: u32, what: u32) {
        let conn: Arc<dyn ThreadDbConnection> = Arc::new(self.inner.clone());
        let lease = LeaseTiming::default();
        let Ok(thread) = Thread::new(conn, THREAD, lease, SchemaCell::default(), "text").await
        else {
            return;
        };
        let mut options = RunOptions::new(format!("text_{turn}"), "");
        options.access = LaneAccess::Exclusive(WhenHeld::Reject);
        if let Ok(mut run) = thread.lane(LANE).open(options, &Holder::new("text")).await {
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
        let conn: Arc<dyn ThreadDbConnection> = Arc::new(self.inner.clone());
        let lease = LeaseTiming::default();
        let Ok(thread) = Thread::new(conn, THREAD, lease, SchemaCell::default(), "audit").await
        else {
            return;
        };
        let lane = thread.lane(LANE);
        let runs = lane.runs(usize::MAX).await.unwrap_or_default();
        let unended = runs.iter().filter(|run| run.interrupted()).count() as u32;
        let held = self.lessee().await.is_some_and(|(_, until)| until >= self.unix_secs());
        let faulted = self.state.lock().unwrap().faulted;
        self.registry.record_storage(StorageStep::LaneSettled { unended, held, faulted });
    }
}

/// Counts a lease take whose answer is still owed, until it is answered or dropped.
struct Taking<'a>(&'a Mutex<State>);

impl Drop for Taking<'_> {
    fn drop(&mut self) {
        self.0.lock().unwrap().taking -= 1;
    }
}

#[async_trait::async_trait]
impl ThreadDbConnection for FakeDatabase {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        let plan = self.plan(false, false, None);
        self.wait(plan.before).await;
        if let Some(error) = plan.refused {
            return Err(error);
        }
        let rows = self.inner.query(statement).await;
        self.wait(plan.after).await;
        rows
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        let takes: Vec<Statement> =
            statements.iter().filter(|s| s.sql().contains(TAKE)).cloned().collect();
        if !takes.is_empty() {
            let turn = {
                let mut state = self.state.lock().unwrap();
                state.takes += 1;
                let turn = state.takes;
                (state.armed && turn.is_multiple_of(state.text_turn_every))
                    .then(|| (turn, state.rng.u32(0..4)))
            };
            if let Some((turn, what)) = turn {
                self.text_turn(turn, what).await;
            }
        }
        // A request carrying the holding lease's token is its owner speaking: a renewal, a
        // commit or a release.
        let carried = self.lessee().await.map(|(token, _)| token).filter(|token| {
            let token = SqlValue::Text(token.clone());
            statements.iter().any(|s| s.params().contains(&token))
        });
        let plan = self.plan(true, !takes.is_empty(), carried.as_deref());
        self.wait(plan.before).await;
        if let Some(error) = plan.refused {
            return Err(error);
        }
        let result = self.inner.batch(statements).await;
        let _taking = if takes.is_empty() {
            None
        } else {
            self.observe_take(&takes).await;
            self.state.lock().unwrap().taking += 1;
            Some(Taking(&self.state))
        };
        self.wait(plan.after).await;
        if plan.answer_lost && result.is_ok() {
            return Err(StorageError::Uncertain("the answer was lost".into()));
        }
        result
    }

    async fn close(&self) {}
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
fn edit(lane: &mut Vec<ChatItem>, turn: u32, what: u32) {
    let plain = |item: &ChatItem| {
        item.as_message().is_some_and(|m| matches!(m.role, Role::User | Role::Assistant))
    };
    let said =
        |id: String, role, text: &str, at| ChatItem::Message(ChatMessage::text(id, role, text, at));
    match what {
        0 => {
            let answer = lane
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
            let oldest: Vec<usize> = lane
                .iter()
                .enumerate()
                .filter(|(_, item)| plain(item))
                .map(|(i, _)| i)
                .take(2)
                .collect();
            if let [first, second] = oldest[..] {
                let at = lane[first].created_at();
                lane.remove(second);
                lane[first] = said(
                    format!("lane_summary_{turn}"),
                    Role::Assistant,
                    "Summary of the start.",
                    at,
                );
                return;
            }
        }
        _ => {}
    }
    let at = lane.iter().map(ChatItem::created_at).max().unwrap_or(UNIX_EPOCH);
    lane.push(said(
        format!("lane_text_{turn}_user"),
        Role::User,
        "can you also text me the address?",
        at,
    ));
    lane.push(said(format!("lane_text_{turn}_agent"), Role::Assistant, "Sure, I've sent it.", at));
}
