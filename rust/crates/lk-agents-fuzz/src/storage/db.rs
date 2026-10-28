//! The simulated thread database: a real SQLite in memory, running the storage core's own
//! statements, on the simulation's clock (`unixepoch()` reads it), with the faults a real AgentDB
//! connection shows: a commit's acknowledgment lost before or after it applied, and windows in
//! which the database cannot be reached.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use lk_agents_storage::StorageError;
use lk_agents_storage::sql::{BatchResult, Row, Statement};
use lk_agents_storage::{db, sqlite};
use rusqlite::functions::FunctionFlags;

/// How the commit batch a worker sends fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitFault {
    /// The socket dropped with the batch in flight; it never reached the database.
    LoseAckBefore,
    /// The batch committed and its answer was lost.
    LoseAckAfter,
}

/// The simulation's clock in milliseconds, shared with the database's `unixepoch()` (a SQLite
/// function must be `Send`).
#[derive(Clone, Default)]
pub struct Clock(Arc<AtomicI64>);

impl Clock {
    pub fn get(&self) -> i64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn set(&self, ms: i64) {
        self.0.store(ms, Ordering::Relaxed);
    }
}

pub struct SimDb {
    conn: rusqlite::Connection,
    clock: Clock,
    /// No request reaches the database until then (a reconnect window).
    unreachable_until: Cell<i64>,
}

impl SimDb {
    /// A database with the thread schema, its clock the simulation's.
    pub fn new(clock: Clock) -> Self {
        let mut conn = rusqlite::Connection::open_in_memory().expect("an in-memory database");
        let now = clock.clone();
        // the statements read the database's clock: in the simulation it is the simulated one
        conn.create_scalar_function("unixepoch", -1, FunctionFlags::SQLITE_UTF8, move |ctx| {
            let ms = now.get();
            Ok(match ctx.len() {
                0 => rusqlite::types::Value::Integer(ms / 1000),
                _ => rusqlite::types::Value::Real(ms as f64 / 1000.0),
            })
        })
        .expect("the clock function");
        sqlite::batch(&mut conn, &db::bootstrap()).expect("the thread schema");
        Self { conn, clock, unreachable_until: Cell::new(0) }
    }

    /// Requests fail `Unavailable` for `ms` from now.
    pub fn cut_for(&self, ms: i64) {
        self.unreachable_until.set(self.clock.get() + ms);
    }

    fn reachable(&self) -> Result<(), StorageError> {
        match self.clock.get() < self.unreachable_until.get() {
            true => Err(StorageError::Unavailable("the database is unreachable".into())),
            false => Ok(()),
        }
    }

    pub fn query(&self, s: &Statement) -> Result<Vec<Row>, StorageError> {
        self.reachable()?;
        sqlite::query(&self.conn, s)
    }

    pub fn batch(
        &mut self,
        statements: &[Statement],
        fault: Option<CommitFault>,
    ) -> Result<BatchResult, StorageError> {
        self.reachable()?;
        match fault {
            None => sqlite::batch(&mut self.conn, statements),
            Some(CommitFault::LoseAckBefore) => Err(StorageError::Uncertain("lost before".into())),
            Some(CommitFault::LoseAckAfter) => {
                sqlite::batch(&mut self.conn, statements)?;
                Err(StorageError::Uncertain("lost after".into()))
            }
        }
    }

    /// A read the checker makes, outside the simulated network: never faulted.
    pub fn inspect(&self, s: &Statement) -> Vec<Row> {
        sqlite::query(&self.conn, s).expect("an inspection read")
    }
}
