use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_abi::storage::journal::{JournalEvent, RunEnded};
use lk_agents_abi::storage::thread::AbandonReason;
use lk_agents_abi::{Clock as _, HOST_TEARDOWN_TIMEOUT};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::clock::RealClock;
use lk_agents_net::clock::unix_now_ms;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::sql::{Bind, SqlValue, Statement};
use crate::thread::db::EventWrite;
use crate::thread::schema::{
    ExtendLease, InsertRunEnded, ReleaseLease, ReleaseLeaseIfDrained, WithdrawWaiter,
};
use crate::thread::{LeaseTiming, Thread};
use crate::{Fence, RunOutcome, StorageError, shortuuid};

pub use lk_agents_abi::storage::thread::LanePoll;

// The renewal task holds only the database and the fence, never the lease, so dropping the last
// `Lease` handle stops it.

/// An owner's lease identity, `<label>/<id>`. Every acquire of one owner carries it, so the
/// owner's next acquire re-enters a lease a canceled acquire or a lost cleanup left behind.
/// Minted once per owner, which acquires one lane at a time: two owners never share one. The
/// releases its dropped leases leave run on its set, which the owner's close awaits.
#[derive(Clone)]
pub struct Holder {
    id: Arc<str>,
    releases: Arc<TaskSet>,
}

impl Holder {
    pub fn new(label: &str) -> Self {
        let id = format!("{label}/{}", shortuuid("own_")).into();
        Self { id, releases: Arc::default() }
    }

    pub fn as_str(&self) -> &str {
        &self.id
    }

    /// Waits, at most [`HOST_TEARDOWN_TIMEOUT`], for the releases dropped leases left; `false`
    /// when one is still in flight.
    pub async fn released(&self) -> bool {
        let clock = RealClock::new();
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.releases.wait_empty(&clock, Some(deadline)).await
    }
}

/// The release by token, after the `run_ended` of `run_id` when given; both are fenced on the
/// token, so a lease since taken again is left alone.
fn release(lane: &str, token: &str, ended: Option<(&str, &EventWrite)>) -> Vec<Statement> {
    let mut batch = Vec::new();
    if let Some((run_id, e)) = ended {
        let payload = SqlValue::json(&e.payload);
        batch.push(InsertRunEnded { lane, token, ts_ms: e.ts_ms, run_id, payload }.bind());
    }
    batch.push(ReleaseLease { lane, token }.bind());
    batch
}

/// [`release`] from a drop, on the spawner captured while the lease was taken (the caller may
/// already be gone, or off the runtime), counted by its owner.
fn release_detached(spawner: &TokioSpawner, owner: &Holder, db: Thread, batch: Vec<Statement>) {
    owner.releases.spawn(spawner, "lease release", async move {
        if let Err(error) = db.conn().batch(batch).await {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "lease cleanup failed: {error}");
        }
    });
}

/// A canceled acquire must withdraw its mark even when its future is dropped.
pub(crate) struct Waiter {
    db: Thread,
    lane: String,
    holder: String,
    spawner: TokioSpawner,
}

impl Waiter {
    pub(crate) fn new(db: Thread, lane: &str, holder: &str) -> Self {
        Self {
            db,
            lane: lane.to_owned(),
            holder: holder.to_owned(),
            spawner: TokioSpawner::current(),
        }
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        let (db, lane, holder) = (self.db.clone(), self.lane.clone(), self.holder.clone());
        // The acquire's caller may already be gone. This conditional cleanup ends on its own;
        // an owner change also clears the mark if the worker or database is unavailable.
        self.spawner.spawn_detached(async move {
            if let Err(error) = db.exec(WithdrawWaiter { lane: &lane, holder: &holder }).await {
                log::debug!(target: lk_agents_abi::LOG_TARGET, lane = lane.as_str(); "waiter cleanup failed: {error}");
            }
        });
    }
}

/// A lease take in flight, under the token it mints: the only way to a [`Lease`]. Dropped
/// before [`PendingLease::into_lease`] while a request may have taken the lane (an abandoned
/// future, an error, a lost acknowledgment), it releases by token and ends the run `canceled`.
pub(crate) struct PendingLease {
    db: Thread,
    lane: String,
    token: String,
    holder: Holder,
    run_id: Option<String>,
    armed: bool,
    spawner: TokioSpawner,
}

impl PendingLease {
    pub(crate) fn new(db: Thread, lane: &str, holder: &Holder, run_id: Option<&str>) -> Self {
        let (lane, token, holder) = (lane.to_owned(), shortuuid("lease_"), holder.clone());
        let run_id = run_id.map(str::to_owned);
        Self { db, lane, token, holder, run_id, armed: false, spawner: TokioSpawner::current() }
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Before each request that may take the lane.
    pub(crate) fn arm(&mut self) {
        self.armed = true;
    }

    /// The database answered that the lane is not held under this token.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    /// The lane taken at `version` under this token.
    pub(crate) fn into_lease(mut self, version: i64, at: Instant) -> Lease {
        self.armed = false;
        let (lane, token, holder) = (self.lane.clone(), self.token.clone(), self.holder.clone());
        let fence = Fence { lane, holder: holder.as_str().into(), token, version };
        Lease::new(self.db.clone(), fence, holder, at)
    }
}

impl Drop for PendingLease {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let reason = Some(AbandonReason::AcquireCanceled.as_str().to_owned());
        let event = JournalEvent::RunEnded(RunEnded { outcome: RunOutcome::Canceled, reason });
        let ended = EventWrite::new(unix_now_ms(), event);
        let ended = self.run_id.as_deref().map(|run_id| (run_id, &ended));
        let batch = release(&self.lane, &self.token, ended);
        release_detached(&self.spawner, &self.holder, self.db.clone(), batch);
    }
}

/// A lease an acquire or a receive took: the lane at `version` under `token`. Cloned by the run
/// and its checkpoint; [`Lease::keep_alive`] extends it while the run works.
#[derive(Clone)]
pub struct Lease {
    inner: Arc<Held>,
}

struct Held {
    db: Thread,
    fence: Fence,
    ownership: Arc<Ownership>,
    /// Owned here: dropping the last handle cancels it.
    renewal: Mutex<Option<OwnedTask>>,
    /// Released, or handed back by a commit: dropping the last handle writes nothing.
    released: AtomicBool,
    owner: Holder,
    spawner: TokioSpawner,
}

impl Drop for Held {
    fn drop(&mut self) {
        if !self.released.load(Ordering::Acquire) {
            let batch = release(&self.fence.lane, &self.fence.token, None);
            release_detached(&self.spawner, &self.owner, self.db.clone(), batch);
        }
    }
}

/// Shared with renewal without keeping the lease (and its owned task) alive.
struct Ownership {
    lane: String,
    deadline: Mutex<Instant>,
    lost: watch::Sender<Option<String>>,
}

impl Ownership {
    fn check_at(&self, deadline: Instant) -> Result<(), StorageError> {
        if Instant::now() >= deadline {
            self.lose("the ownership deadline elapsed".into());
        }
        match self.lost.borrow().clone() {
            Some(reason) => Err(StorageError::LeaseLost(reason)),
            None => Ok(()),
        }
    }

    fn check(&self) -> Result<(), StorageError> {
        self.check_at(*self.deadline.lock())
    }

    fn renewed(&self, started: Instant, ttl: std::time::Duration) -> Result<(), StorageError> {
        let mut deadline = self.deadline.lock();
        self.check_at(*deadline)?;
        // Start at the request, not its acknowledgment: database latency consumes our budget.
        *deadline = started + ttl;
        Ok(())
    }

    fn lose(&self, reason: String) {
        let reason = format!("lease on lane {}: {reason}", self.lane);
        if self.lost.send_if_modified(|lost| {
            if lost.is_some() {
                return false;
            }
            *lost = Some(reason.clone());
            true
        }) {
            log::warn!(target: lk_agents_abi::LOG_TARGET, lane = self.lane.as_str(); "{reason}");
        }
    }
}

impl Lease {
    fn new(db: Thread, fence: Fence, owner: Holder, acquired_at: Instant) -> Self {
        let (lost, _) = watch::channel(None);
        let ownership = Arc::new(Ownership {
            lane: fence.lane.clone(),
            deadline: Mutex::new(acquired_at + db.timing.ttl),
            lost,
        });
        let (renewal, released) = (Mutex::new(None), AtomicBool::new(false));
        let spawner = TokioSpawner::current();
        let held = Held { db, fence, ownership, renewal, released, owner, spawner };
        Self { inner: Arc::new(held) }
    }

    pub fn fence(&self) -> &Fence {
        &self.inner.fence
    }

    pub fn holder(&self) -> &str {
        &self.inner.fence.holder
    }

    pub fn owner(&self) -> &Holder {
        &self.inner.owner
    }

    /// Renews on start-to-start deadlines. Each request is bounded by the interval and the
    /// remaining ownership budget. A failed or late acknowledgment never extends that budget;
    /// once ownership is lost this handle cannot revive itself.
    pub fn keep_alive(&self) {
        let mut renewal = self.inner.renewal.lock();
        if renewal.is_some() || self.check().is_err() {
            return;
        }
        let work = LeaseRenewal {
            db: self.inner.db.clone(),
            fence: self.inner.fence.clone(),
            ownership: self.inner.ownership.clone(),
        };
        *renewal = Some(TokioSpawner::current().spawn(work.run()));
    }

    /// Ends the renewal; the lease itself is released by the commit or a release.
    pub fn stop(&self) {
        self.inner.renewal.lock().take();
    }

    /// A write already released the lease (a commit that did not keep it).
    pub(crate) fn released(&self) {
        self.inner.released.store(true, Ordering::Release);
    }

    pub fn is_lost(&self) -> bool {
        self.check().is_err()
    }

    /// Check before admitting another handler step, including after a process pause.
    pub fn check(&self) -> Result<(), StorageError> {
        self.inner.ownership.check()
    }

    /// Resolves with the error that cancels the run once the lease is lost; pending while it is
    /// held, so race it against the run.
    pub async fn lost(&self) -> StorageError {
        let ownership = &self.inner.ownership;
        let mut rx = ownership.lost.subscribe();
        loop {
            if let Err(error) = ownership.check() {
                return error;
            }
            let deadline = *ownership.deadline.lock();
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => {},
                changed = rx.changed() => {
                    if changed.is_err() { return StorageError::LeaseLost("lease observer closed".into()); }
                }
            }
        }
    }

    /// Releases without writing anything else but, when given, the `run_ended` event of `run_id`.
    pub async fn release(&self, ended: Option<(&str, &EventWrite)>) -> Result<(), StorageError> {
        let Fence { lane, token, .. } = &self.inner.fence;
        self.inner.db.conn().batch(release(lane, token, ended)).await?;
        self.released();
        Ok(())
    }

    /// Releases when no inbox row is pending past `drained_id`; `true` when released. A row
    /// pending keeps the lease, extended; a lease another holder took is `LeaseLost`.
    pub async fn release_if_drained(&self) -> Result<bool, StorageError> {
        let (db, fence) = (&self.inner.db, &self.inner.fence);
        if db.exec(ReleaseLeaseIfDrained { lane: &fence.lane, token: &fence.token }).await? > 0 {
            self.released();
            return Ok(true);
        }
        if db.exec(extend(fence, db.timing.secs())).await? == 0 {
            return Err(StorageError::LeaseLost(format!("lease on lane {} was taken", fence.lane)));
        }
        Ok(false)
    }
}

/// Pushes the lease out while its token is still this one.
/// The lease's renewal task; it holds the ownership, not the lease handle.
struct LeaseRenewal {
    db: Thread,
    fence: Fence,
    ownership: Arc<Ownership>,
}

impl LeaseRenewal {
    async fn run(self) {
        let LeaseTiming { ttl, renew_interval: interval } = self.db.timing;
        let mut next = *self.ownership.deadline.lock() - ttl + interval;
        loop {
            let deadline = *self.ownership.deadline.lock();
            tokio::time::sleep_until(next.min(deadline)).await;
            if self.ownership.check().is_err() {
                return;
            }
            let started = Instant::now();
            next = started + interval;
            if self.renew_once(started, next.min(deadline), ttl).await.is_break() {
                return;
            }
        }
    }

    /// One extend, bounded by `until`; breaks once ownership is lost.
    async fn renew_once(
        &self,
        started: Instant,
        until: Instant,
        ttl: std::time::Duration,
    ) -> ControlFlow<()> {
        let extend = self.db.exec(extend(&self.fence, self.db.timing.secs()));
        let outcome = tokio::time::timeout_at(until, extend)
            .await
            .unwrap_or_else(|_| Err(StorageError::Unavailable("the extend timed out".into())));
        if self.ownership.check().is_err() {
            return ControlFlow::Break(());
        }
        match outcome {
            Ok(rows) if rows > 0 => {
                if self.ownership.renewed(started, ttl).is_err() {
                    return ControlFlow::Break(());
                }
            }
            Ok(_) => {
                self.ownership.lose("the lease was taken or released".into());
                return ControlFlow::Break(());
            }
            Err(error) if error.retryable() => {
                log::debug!(target: lk_agents_abi::LOG_TARGET, lane = self.fence.lane.as_str(); "lease extend failed, retrying: {error}");
            }
            Err(error) => {
                self.ownership.lose(format!("lease renewal failed: {error}"));
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

fn extend(fence: &Fence, lease_secs: i64) -> ExtendLease<'_> {
    ExtendLease { lane: &fence.lane, holder: &fence.holder, token: &fence.token, lease_secs }
}

#[cfg(test)]
#[path = "lease_tests.rs"]
mod tests;
