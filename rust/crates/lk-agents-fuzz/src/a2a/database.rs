//! The thread's database as the simulation runs it: lk-agents-storage's in-memory connection on
//! the run's virtual clock, behind storage's fault-injecting one, every request a little late so
//! the workers' operations interleave. Between batches it observes what the rules read: a
//! terminal row that changed, the rows a commit applied while its acknowledgment was lost, the
//! terminal rows each commit wrote, the most questions pending at once.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::Clock;
use lk_agents_abi::storage::{StorageError, TaskState};
use lk_agents_storage::SqliteConnection;
use lk_agents_storage::sql::{BatchResult, Row, SqlValue, Statement};
use lk_agents_storage::testing::{Faulty, MemoryConnection};

/// What the run observed, for the rules.
#[derive(Debug, Default, Clone)]
pub struct Ledger {
    /// Committed handler invocations per `messageId` whose turn they ran.
    pub executions: BTreeMap<String, BTreeSet<usize>>,
    /// Rows a commit applied whose acknowledgment was lost, as it left them.
    pub applied_unacknowledged: BTreeMap<String, TaskState>,
    /// A write that changed a terminal row.
    pub terminal_changes: Vec<String>,
    /// Every applied commit's terminal rows: `(commit, task_id, status)`.
    pub staged_terminal: Vec<(u64, String, TaskState)>,
    /// The most questions pending at once, and which.
    pub most_pending: Vec<String>,
    pub lost_acks: u64,
    pub injected: BTreeMap<&'static str, u64>,
}

/// What the simulation's parts share: its randomness, its clock, its faults and what it observed.
pub struct World {
    rng: Mutex<Rng>,
    pub clock: Arc<dyn Clock>,
    /// Faults fire until then.
    faulty_until: lk_agents_abi::Ts,
    pub ledger: Mutex<Ledger>,
}

impl World {
    pub fn new(seed: u64, clock: Arc<dyn Clock>, faulty_for: Duration) -> Arc<Self> {
        let faulty_until = clock.now().saturating_add(faulty_for);
        Arc::new(World {
            rng: Mutex::new(Rng::with_seed(seed)),
            clock,
            faulty_until,
            ledger: Mutex::default(),
        })
    }

    pub fn ms(&self, range: std::ops::Range<u64>) -> Duration {
        Duration::from_millis(self.rng.lock().unwrap().u64(range))
    }

    pub fn draw<T>(&self, draw: impl FnOnce(&mut Rng) -> T) -> T {
        draw(&mut self.rng.lock().unwrap())
    }

    /// One fault of `pct` percent, counted under `kind`, while faults are on.
    pub fn inject(&self, pct: u8, kind: &'static str) -> bool {
        let hit = pct > 0
            && self.clock.now() < self.faulty_until
            && self.rng.lock().unwrap().u8(0..100) < pct;
        if hit {
            *self.ledger.lock().unwrap().injected.entry(kind).or_default() += 1;
        }
        hit
    }

    pub fn wall_secs(&self) -> f64 {
        let wall = self.clock.wall().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        wall.as_secs_f64()
    }
}

pub struct Database {
    memory: MemoryConnection,
    pub faults: Arc<Faulty>,
    world: Arc<World>,
    commits: Mutex<u64>,
}

/// A task row as the observer reads it: its status, and whether it holds a pending question.
type Rows = BTreeMap<String, (TaskState, bool)>;

const ROWS: &str = "SELECT task_id, status, json_extract(internal, '$.ask') IS NOT NULL FROM tasks";

impl Database {
    pub fn new(world: Arc<World>) -> Result<Arc<Self>, StorageError> {
        let memory = MemoryConnection::new(world.clock.clone())?;
        let faults = Arc::new(Faulty::new(Box::new(memory.clone())));
        Ok(Arc::new(Database { memory, faults, world, commits: Mutex::new(0) }))
    }

    /// The database without faults: what the run left, read once it is over.
    pub fn memory(&self) -> Arc<dyn SqliteConnection> {
        Arc::new(self.memory.clone())
    }

    pub async fn select(&self, sql: &'static str) -> Vec<Row> {
        self.memory.query(Statement::raw(sql)).await.unwrap_or_default()
    }

    /// The lane's lease lapses as a paused holder's does, its renewal not yet landed: a waiter
    /// may take the lane, and the holder's commit then meets the fence.
    pub async fn lapse_lease(&self) {
        let lapse =
            Statement::raw("UPDATE lanes SET lease_until = 0 WHERE lease_token IS NOT NULL");
        let _ = self.memory.batch(vec![lapse]).await;
    }

    async fn rows(&self) -> Rows {
        let text = |v: &SqlValue| v.as_str().unwrap_or_default().to_owned();
        self.select(ROWS)
            .await
            .iter()
            .filter_map(|r| {
                let status = TaskState::parse(r[1].as_str()?)?;
                Some((text(&r[0]), (status, r[2].as_i64() == Some(1))))
            })
            .collect()
    }

    fn observe(
        &self,
        before: &Rows,
        after: &Rows,
        commit: bool,
        result: &Result<BatchResult, StorageError>,
    ) {
        let mut ledger = self.world.ledger.lock().unwrap();
        let mut changed = Vec::new();
        for (task_id, (status, _)) in after {
            let was = before.get(task_id).map(|(s, _)| *s);
            if was == Some(*status) {
                continue;
            }
            if let Some(was) = was.filter(|s| s.is_terminal()) {
                ledger.terminal_changes.push(format!("{task_id} {was:?} -> {status:?}"));
            }
            changed.push((task_id.clone(), *status));
        }
        if commit {
            let mut commits = self.commits.lock().unwrap();
            *commits += 1;
            match result {
                Ok(_) => {
                    let terminal = changed.iter().filter(|(_, s)| s.is_terminal());
                    let staged = terminal.map(|(id, s)| (*commits, id.clone(), *s));
                    ledger.staged_terminal.extend(staged.collect::<Vec<_>>());
                }
                Err(StorageError::Uncertain(_)) => {
                    ledger.lost_acks += 1;
                    ledger.applied_unacknowledged.extend(changed);
                }
                Err(_) => {}
            }
        }
        let pending: Vec<String> = after
            .iter()
            .filter(|(_, (s, ask))| *s == TaskState::InputRequired && *ask)
            .map(|(id, _)| id.clone())
            .collect();
        if pending.len() > ledger.most_pending.len() {
            ledger.most_pending = pending;
        }
    }
}

#[async_trait::async_trait]
impl SqliteConnection for Database {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        tokio::time::sleep(self.world.ms(0..15)).await;
        self.faults.query(statement).await
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        tokio::time::sleep(self.world.ms(0..15)).await;
        // The commit's fenced lane update, as storage's fault connection recognizes it.
        let commit = statements.iter().any(|s| s.sql.contains("version = version + 1"));
        let before = self.rows().await;
        let result = self.faults.batch(statements).await;
        let after = self.rows().await;
        self.observe(&before, &after, commit, &result);
        result
    }

    async fn close(&self) {}
}
