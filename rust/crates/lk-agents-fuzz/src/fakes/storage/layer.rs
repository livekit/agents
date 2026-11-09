use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use lk_agents_abi::storage::{StorageError, Unacknowledged};
use lk_agents_abi::{Clock, Ts};
use lk_agents_runtime::storage::ThreadDbConnection as _;
use lk_agents_runtime::storage::sql::{SqlValue, Statement};
use lk_agents_runtime::storage::thread::LeaseTiming;

use super::{FakeDatabase, LANE};
use crate::record::{StorageFault, StorageStep};

// Faults of the storage layer itself, on top of latency, lost answers and outages: SQLite
// refusing a statement as `sqlite_error` maps it, the wall clock `unixepoch()` reads jumping,
// the database's worker crashing with a write applied (the next request reopens the file and
// recovers its WAL), and another worker's connection holding the free lane or stealing it.
// Stress frequencies, not measured rates, drawn from a stream of their own so the older faults
// keep their draws.

/// Lease tokens another worker's connection writes, so its leases are told apart.
pub(super) const INTRUDER: &str = "intruder_";

/// Per fault kind, in this process: SQLite refusals, clock jumps, crashes, holds, steals.
static INJECTED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// `[sqlite, clock jump, crash, held, stolen]` storage faults injected in this process.
pub fn storage_faults() -> [u64; 5] {
    INJECTED.each_ref().map(|n| n.load(Ordering::Relaxed))
}

/// The run's clock with its wall reading skewed: what the database's `unixepoch()` reads.
pub(super) struct Skewed {
    pub(super) inner: Arc<dyn Clock>,
    skew_ms: AtomicI64,
}

impl Skewed {
    pub(super) fn new(inner: Arc<dyn Clock>) -> Self {
        Skewed { inner, skew_ms: AtomicI64::new(0) }
    }
}

impl Clock for Skewed {
    fn now(&self) -> Ts {
        self.inner.now()
    }
    fn wait_until(&self, at: Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        self.inner.wait_until(at)
    }
    fn wall(&self) -> SystemTime {
        let skew = self.skew_ms.load(Ordering::Relaxed);
        let wall = self.inner.wall();
        let by = Duration::from_millis(skew.unsigned_abs());
        if skew >= 0 { wall + by } else { wall - by }
    }
}

/// What one request meets from this layer.
#[derive(Default)]
pub(super) struct Layer {
    pub(super) refused: Option<StorageError>,
    /// The worker crashes once the write applied, its answer lost.
    pub(super) crash: bool,
    /// Another worker's connection comes for the lane: `Some(true)` steals it.
    pub(super) intrude: Option<bool>,
}

/// The SQLite failures `sqlite_error` maps to `Unavailable`, by name and message.
const SQLITE: [(&str, &str); 4] = [
    ("busy", "database is locked"),
    ("locked", "database table is locked"),
    ("ioerr", "disk I/O error"),
    ("full", "database or disk is full"),
];

impl FakeDatabase {
    /// This layer's draw for one request.
    pub(super) fn layer(&self, write: bool, take: bool) -> Layer {
        let mut layer = Layer::default();
        let mut guard = self.state.lock();
        if !guard.armed || !self.setup.faults {
            return layer;
        }
        let mut faults = Vec::new();
        let state = &mut *guard;
        let rng = &mut state.layer;
        if rng.u32(0..96) == 0 {
            let (name, message) = SQLITE[rng.usize(0..SQLITE.len())];
            layer.refused = Some(StorageError::Unavailable(format!("sqlite: {message}")));
            faults.push(StorageFault::Sqlite(name));
            // As an outage's refusal, it may cost the lease its renewal.
            state.faulted = true;
        }
        if rng.u32(0..256) == 0 {
            // Forward past the lease's TTL, or back a little.
            let ms = if rng.bool() { rng.i64(11_000..=15_000) } else { -rng.i64(1_000..=3_000) };
            self.skew.skew_ms.fetch_add(ms, Ordering::Relaxed);
            faults.push(StorageFault::ClockJump(ms));
            state.faulted |= ms > 0;
        }
        if write && layer.refused.is_none() && rng.u32(0..192) == 0 {
            layer.crash = true;
            faults.push(StorageFault::Crash);
            state.faulted = true;
        }
        if take && !state.intruding && state.layer.u32(0..16) == 0 {
            layer.intrude = Some(state.layer.u32(0..3) == 0);
            state.intruding = true;
        }
        drop(guard);
        for fault in faults {
            self.record_fault(fault);
        }
        layer
    }

    pub(super) fn record_fault(&self, fault: StorageFault) {
        let kind = match fault {
            StorageFault::Sqlite(_) => 0,
            StorageFault::ClockJump(_) => 1,
            StorageFault::Crash => 2,
            StorageFault::Held => 3,
            StorageFault::Stolen => 4,
        };
        INJECTED[kind].fetch_add(1, Ordering::Relaxed);
        self.registry.record_storage(StorageStep::Fault { fault });
    }

    /// The crash's answer: the worker stopped mid-batch, and dies at its next poll.
    pub(super) fn crash(&self) -> StorageError {
        self.dying.die_next_poll();
        StorageError::Uncertain(Unacknowledged::sent("the sqlite worker stopped mid-batch"))
    }

    /// Another worker's connection takes the lane, free (`steal` false) or not, holds it for a
    /// while renewing as a live owner does (one to four seconds), then releases it.
    pub(super) async fn intrude(self: Arc<Self>, steal: bool) {
        let (hold_ms, serial) = {
            let mut state = self.state.lock();
            state.intruders += 1;
            (state.layer.u64(1_000..=4_000), state.intruders)
        };
        let token = format!("{INTRUDER}{serial}");
        let ttl = LeaseTiming::default().ttl.as_secs();
        let params = || vec![SqlValue::Text(token.clone()), SqlValue::Text(LANE.into())];
        let free = if steal { "" } else { " AND lease_until < unixepoch()" };
        let take = format!(
            "UPDATE lanes SET lease_holder = 'intruder', lease_token = ?1, \
             lease_until = unixepoch() + {ttl} WHERE lane = ?2{free}"
        );
        let before = self.lessee().await;
        let took = self.observer.batch(vec![Statement::raw(take).with(params())]).await;
        if !took.is_ok_and(|r| r.at(0).is_ok_and(|s| s.rows_affected() > 0)) {
            self.state.lock().intruding = false;
            return;
        }
        let now = self.clock.now();
        {
            let mut state = self.state.lock();
            state.touched.insert(token.clone(), now);
            if steal
                && let Some((stolen, until)) = before
                && until >= self.unix_secs()
            {
                state.stolen.push(stolen);
                state.faulted = true;
            }
        }
        self.record_fault(if steal { StorageFault::Stolen } else { StorageFault::Held });
        let renew = format!(
            "UPDATE lanes SET lease_until = unixepoch() + {ttl} WHERE lane = ?2 AND lease_token = ?1"
        );
        let renewal = LeaseTiming::default().renew_interval.as_millis() as u64;
        let mut left = hold_ms;
        while left > 0 {
            let step = left.min(renewal);
            self.wait(Duration::from_millis(step)).await;
            left -= step;
            let _ = self.observer.batch(vec![Statement::raw(renew.clone()).with(params())]).await;
            self.state.lock().touched.insert(token.clone(), self.clock.now());
        }
        let release = "UPDATE lanes SET lease_until = 0, lease_token = NULL WHERE lane = ?2 AND lease_token = ?1";
        let _ = self.observer.batch(vec![Statement::raw(release).with(params())]).await;
        self.state.lock().intruding = false;
    }

    /// The lane's version while another worker holds it, or `None`.
    async fn intruded_version(&self) -> Option<i64> {
        let select = Statement::raw("SELECT version, lease_token FROM lanes WHERE lane = ?1")
            .with(vec![SqlValue::Text(LANE.into())]);
        let rows = self.observer.query(select).await.ok()?;
        match rows.first()?.as_slice() {
            [SqlValue::Int(version), SqlValue::Text(token)] if token.starts_with(INTRUDER) => {
                Some(*version)
            }
            _ => None,
        }
    }

    /// Whether `statements` carry a lease token another worker stole.
    pub(super) fn carries_stolen(&self, statements: &[Statement]) -> bool {
        let state = self.state.lock();
        state.stolen.iter().any(|token| {
            let token = SqlValue::Text(token.clone());
            statements.iter().any(|s| s.params().contains(&token))
        })
    }

    /// Around a write under a stolen token: whether it moved the lane while the thief held it.
    pub(super) async fn fenced<T>(&self, write: impl std::future::Future<Output = T>) -> T {
        let before = self.intruded_version().await;
        let result = write.await;
        if let (Some(before), Some(after)) = (before, self.intruded_version().await) {
            let applied = before != after;
            self.registry.record_storage(StorageStep::StaleWrite { applied });
        }
        result
    }
}
