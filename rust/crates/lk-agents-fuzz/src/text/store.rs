use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, UNIX_EPOCH};

use parking_lot::Mutex;

use lk_agents_abi::storage::{StorageError, Unacknowledged};
use lk_agents_abi::{Clock, Ts};
use lk_agents_runtime::storage::backends::sqlite::{FileConnection, SqliteStorage, Workers};
use lk_agents_runtime::storage::sql::{BatchResult, Row, Statement};
use lk_agents_runtime::storage::testing::{Dying, TempDir, Wrapped, lessee};
use lk_agents_runtime::storage::thread::LeaseTiming;
use lk_agents_runtime::storage::{Storage, Thread, ThreadDbConnection};

// The production SQLite backend in a directory of the run's own, its workers tasks on the run's
// runtime and its `unixepoch()` the run's clock, as the voice lane runs it. Each turn's
// connection is wrapped: it answers late, loses writes and answers, goes down, and kills its
// database worker with requests queued.

/// One thread's database for a run.
pub struct Store {
    storage: Arc<SqliteStorage>,
    backend: Arc<dyn Storage>,
    pub thread_id: String,
    /// A connection of the observer's own, never faulted.
    observer: FileConnection,
    faults: Arc<Faults>,
    clock: Arc<dyn Clock>,
    _dir: TempDir,
}

struct Faults {
    rng: Mutex<fastrand::Rng>,
    enabled: AtomicBool,
    down_until: Mutex<Ts>,
    faulted: AtomicBool,
    dying: Arc<Dying>,
    clock: Arc<dyn Clock>,
}

impl Store {
    pub async fn open(seed: u64, faults: bool, clock: Arc<dyn Clock>) -> Self {
        let dir = TempDir::new("lk-agents-fuzz-text-");
        let spawner = Arc::new(lk_agents_runtime::net::task::TokioSpawner::current());
        let dying = Arc::new(Dying::new(spawner));
        let workers = Workers { spawner: dying.clone(), clock: clock.clone() };
        let storage = SqliteStorage::open(&dir.0, LeaseTiming::default(), workers);
        let storage = Arc::new(storage.await.expect("a storage directory"));
        let thread = storage.create_thread(Default::default()).await.expect("a thread");
        let observer = storage.connect(&thread.thread_id).await.expect("an observer");
        let rng =
            fastrand::Rng::with_seed((seed ^ 0x7465_7874).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let faults = Arc::new(Faults {
            rng: Mutex::new(rng),
            enabled: AtomicBool::new(faults),
            down_until: Mutex::new(Ts::ZERO),
            faulted: AtomicBool::new(false),
            dying,
            clock: clock.clone(),
        });
        let wrapped = faults.clone();
        let wrap = move |inner: FileConnection| {
            Arc::new(FaultyConnection { inner, faults: wrapped.clone() })
                as Arc<dyn ThreadDbConnection>
        };
        let backend: Arc<dyn Storage> = Arc::new(Wrapped { storage: storage.clone(), wrap });
        let thread_id = thread.thread_id;
        Self { storage, backend, thread_id, observer, faults, clock, _dir: dir }
    }

    /// A worker's handle on the thread, through the faulted connection.
    pub async fn thread(&self) -> Result<Thread, StorageError> {
        self.backend.open(&self.thread_id).await
    }

    /// The observer's handle, never faulted.
    pub async fn clean_thread(&self) -> Thread {
        self.storage.open(&self.thread_id).await.expect("the observer opens the thread")
    }

    /// The database answers every request from now on.
    pub fn heal(&self) {
        self.faults.enabled.store(false, Ordering::Release);
        *self.faults.down_until.lock() = Ts::ZERO;
    }

    pub fn worker_deaths(&self) -> u64 {
        self.faults.dying.deaths()
    }

    pub fn faulted(&self) -> bool {
        self.faults.faulted.load(Ordering::Acquire)
    }

    /// Whether a lease holds `lane` now, by the database's clock.
    pub async fn held(&self, lane: &str) -> bool {
        let now = self.clock.wall().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        lessee(&self.observer, lane).await.is_some_and(|(_, until)| until >= now as i64)
    }

    pub async fn close(&self) {
        self.observer.close().await;
        self.storage.close().await;
    }
}

/// What one request meets.
#[derive(Default)]
struct Plan {
    before: Duration,
    after: Duration,
    refused: Option<StorageError>,
    answer_lost: bool,
    worker_dies: bool,
}

impl Faults {
    /// Stress frequencies, not measured rates.
    fn plan(&self, write: bool) -> Plan {
        let mut plan = Plan::default();
        if !self.enabled.load(Ordering::Acquire) {
            return plan;
        }
        let now = self.clock.now();
        let mut rng = self.rng.lock();
        let ms = Duration::from_millis;
        (plan.before, plan.after) = if rng.u32(0..32) == 0 {
            (ms(rng.u64(0..=50)), ms(rng.u64(250..=1500)))
        } else {
            (ms(rng.u64(0..=2)), ms(rng.u64(1..=5)))
        };
        let mut down_until = self.down_until.lock();
        if now >= *down_until && rng.u32(0..160) == 0 {
            // Mostly a blip; a quarter outlast the lease's TTL.
            let down =
                if rng.u32(0..4) == 0 { rng.u64(10_500..=15_000) } else { rng.u64(100..=3_000) };
            *down_until = now.saturating_add(ms(down));
        }
        match rng.u32(0..96) {
            0 if write => plan.answer_lost = true,
            1 if write => {
                plan.refused = Some(StorageError::Uncertain(Unacknowledged::sent("lost before")))
            }
            2 if write => plan.worker_dies = true,
            _ => {}
        }
        if now < *down_until {
            plan.refused = Some(StorageError::Unavailable("the database is unreachable".into()));
        }
        if plan.refused.is_some() || plan.answer_lost || plan.worker_dies {
            self.faulted.store(true, Ordering::Release);
        }
        plan
    }

    async fn wait(&self, latency: Duration) {
        if !latency.is_zero() {
            self.clock.wait_until(self.clock.now().saturating_add(latency)).await;
        }
    }
}

struct FaultyConnection {
    inner: FileConnection,
    faults: Arc<Faults>,
}

#[async_trait::async_trait]
impl ThreadDbConnection for FaultyConnection {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        let plan = self.faults.plan(false);
        self.faults.wait(plan.before).await;
        if let Some(error) = plan.refused {
            return Err(error);
        }
        let rows = self.inner.query(statement).await;
        self.faults.wait(plan.after).await;
        rows
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        let plan = self.faults.plan(true);
        self.faults.wait(plan.before).await;
        if let Some(error) = plan.refused {
            return Err(error);
        }
        if plan.worker_dies {
            self.faults.dying.die_next_poll();
        }
        let result = self.inner.batch(statements).await;
        self.faults.wait(plan.after).await;
        if plan.answer_lost && result.is_ok() {
            return Err(StorageError::Uncertain(Unacknowledged::sent("the answer was lost")));
        }
        result
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}
