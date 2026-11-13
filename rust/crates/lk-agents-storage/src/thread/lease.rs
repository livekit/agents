use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_abi::storage::journal::{JournalEvent, RunEnded};
use lk_agents_abi::storage::thread::AbandonReason;
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, Ts};
use lk_agents_abi::{OwnedTask, Spawner, SpawnerExt};
use lk_agents_net::clock::unix_now_ms;
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;

use crate::sql::{Bind, SqlValue, Statement};
use crate::telemetry::{self, StorageWrite, TeardownStage};
use crate::thread::db::EventWrite;
use crate::thread::schema::{
    ExtendLease, InsertRunEnded, ReleaseLease, ReleaseLeaseIfDrained, SelectFence,
};
use crate::thread::{LeaseTiming, Resend, Thread};
use crate::{Fence, RunOutcome, StorageError, shortuuid};

// The renewal task holds only the database and the fence, never the lease, so dropping the last
// `Lease` handle stops it.

/// An owner's lease identity, `<label>/<id>`. Every acquire of one owner carries it, so the
/// owner's next acquire re-enters a lease a canceled acquire or a lost cleanup left behind.
/// Minted once per owner, which acquires one partition at a time: two owners never share one. The
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

    /// Runs `cleanup` on `spawner`, in the set [`Holder::wait_released`] awaits, its result
    /// published on `producer`: a cancel or a panic ends the signal instead.
    pub(crate) fn spawn_settling_cleanup<T: Send + Sync + 'static>(
        &self,
        spawner: &dyn Spawner,
        name: &str,
        producer: Producer<Option<T>>,
        cleanup: impl std::future::Future<Output = T> + Send + 'static,
    ) {
        self.releases.spawn_settling(spawner, name, producer, cleanup);
    }

    /// Waits, at most [`HOST_TEARDOWN_TIMEOUT`] on `clock`, for the releases dropped leases
    /// left; `false` when one is still in flight.
    pub async fn wait_released(&self, clock: &dyn Clock) -> bool {
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.releases.wait_empty(clock, Some(deadline)).await
    }
}

impl std::fmt::Debug for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.id)
    }
}

impl PartialEq for Holder {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Holder {}

/// The release by token, after the `run_ended` of `run_id` when given. The release is fenced on
/// the token, so a lease since taken again is left alone; the `run_ended` also lands while
/// `holder` re-entered the partition under another token, once.
fn release(
    partition_key: &str,
    token: &str,
    holder: &str,
    ended: Option<(&str, &EventWrite)>,
) -> Vec<Statement> {
    let mut batch = Vec::new();
    if let Some((run_id, e)) = ended {
        let payload = SqlValue::json(&e.payload);
        let occurred_at_ms = e.occurred_at_ms;
        batch.push(
            InsertRunEnded { partition_key, token, holder, occurred_at_ms, run_id, payload }.bind(),
        );
    }
    batch.push(ReleaseLease { partition_key, token }.bind());
    batch
}

/// [`release`] from a drop, on the thread's storage executor (the caller may already be gone,
/// or off the runtime), counted by its owner. One still running past [`HOST_TEARDOWN_TIMEOUT`]
/// is a teardown overrun.
fn release_detached(owner: &Holder, db: Thread, partition_key: &str, batch: Vec<Statement>) {
    let partition_key = partition_key.to_owned();
    let open = db.keep_open("lease release");
    let background = db.background.clone();
    owner.releases.spawn(&*background.spawner, "lease release", async move {
        let _open = open;
        let release = db.resend(StorageWrite::Release).send(|| db.conn().batch(batch.clone()));
        let mut release = std::pin::pin!(release);
        let bounded = lk_agents_abi::timeout(db.clock(), HOST_TEARDOWN_TIMEOUT, &mut release);
        let out = match bounded.await {
            Ok(out) => out,
            Err(_) => {
                let stage = TeardownStage::LeaseRelease;
                telemetry::teardown_overrun(&db.cx, db.clock(), stage, &partition_key);
                release.await
            }
        };
        if let Err(error) = out {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "lease cleanup failed: {error}");
        }
    });
}

/// A lease take in flight, under the token it mints: the only way to a [`Lease`]. Dropped
/// while a request may have taken the partition (an abandoned future, an error, a lost
/// acknowledgment), it releases by token and ends the run `canceled`.
pub(crate) struct PendingLease {
    db: Thread,
    partition_key: String,
    token: String,
    holder: Holder,
    run_id: Option<String>,
    /// A request may have taken the partition under this token.
    armed: bool,
    resend: Resend,
}

impl PendingLease {
    /// A lost acknowledgment is sent again under the same token for up to `wait`.
    pub(crate) fn new(
        db: Thread,
        partition_key: &str,
        holder: &Holder,
        run_id: Option<&str>,
        wait: std::time::Duration,
        write: StorageWrite,
    ) -> Self {
        let (partition_key, token, holder) =
            (partition_key.to_owned(), shortuuid("lease_"), holder.clone());
        let run_id = run_id.map(str::to_owned);
        let resend = db.resend_within(wait, write);
        Self { db, partition_key, token, holder, run_id, armed: false, resend }
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Whether an earlier request's answer was lost: it may have taken the partition.
    pub(crate) fn lost(&self) -> bool {
        self.resend.lost()
    }

    /// One request that may take the partition under this token, `version` reading from its answer
    /// the version it took the partition at. `Continue` after a lost acknowledgment, to send it again;
    /// `Break` once the answer is known, with the lease when it took the partition.
    pub(crate) async fn attempt<T>(
        &mut self,
        request: impl Future<Output = Result<T, StorageError>>,
        version: impl FnOnce(&T) -> Option<i64>,
    ) -> ControlFlow<Result<(T, Option<Lease>), StorageError>> {
        let at = self.db.clock().now();
        self.armed = true;
        let out = request.await;
        let answer = match self.resend.after(out).await {
            ControlFlow::Continue(()) => return ControlFlow::Continue(()),
            ControlFlow::Break(Ok(answer)) => answer,
            ControlFlow::Break(Err(error)) => return ControlFlow::Break(Err(error)),
        };
        let Some(version) = version(&answer) else {
            // after a lost answer the partition may still be held under this token
            self.armed = self.resend.lost();
            return ControlFlow::Break(Ok((answer, None)));
        };
        self.armed = false;
        let fence = Fence {
            partition_key: self.partition_key.clone(),
            holder: self.holder.clone(),
            token: self.token.clone(),
            version,
        };
        ControlFlow::Break(Ok((answer, Some(Lease::new(self.db.clone(), fence, at)))))
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
        let batch = release(&self.partition_key, &self.token, self.holder.as_str(), ended);
        release_detached(&self.holder, self.db.clone(), &self.partition_key, batch);
    }
}

/// A lease an acquire or a receive took: the partition at `version` under `token`. Cloned by the run
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
}

impl Drop for Held {
    fn drop(&mut self) {
        if !self.released.load(Ordering::Acquire) {
            let (fence, db) = (&self.fence, self.db.clone());
            let batch = release(&fence.partition_key, &fence.token, fence.holder.as_str(), None);
            release_detached(&fence.holder, db, &fence.partition_key, batch);
        }
    }
}

/// Shared with renewal without keeping the lease (and its owned task) alive.
struct Ownership {
    partition_key: String,
    /// What the deadline is read on.
    clock: Arc<dyn Clock>,
    deadline: Mutex<Ts>,
    /// Why ownership was lost, set once. Held with the ownership, which every waiter borrows.
    lost: Producer<Option<String>>,
}

impl Ownership {
    fn check_at(&self, deadline: Ts) -> Result<(), StorageError> {
        if self.clock.now() >= deadline {
            self.lose("the ownership deadline elapsed".into());
        }
        match self.lost.get() {
            Some(reason) => Err(StorageError::LeaseLost(reason)),
            None => Ok(()),
        }
    }

    fn check(&self) -> Result<(), StorageError> {
        self.check_at(*self.deadline.lock())
    }

    fn renewed(&self, started: Ts, ttl: std::time::Duration) -> Result<(), StorageError> {
        let mut deadline = self.deadline.lock();
        self.check_at(*deadline)?;
        // Start at the request, not its acknowledgment: database latency consumes our budget.
        *deadline = started.saturating_add(ttl);
        Ok(())
    }

    fn lose(&self, reason: String) {
        let reason = format!("lease on partition_key {}: {reason}", self.partition_key);
        let first = self.lost.modify(|lost| {
            let first = lost.is_none();
            if first {
                *lost = Some(reason.clone());
            }
            first
        });
        if first {
            log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(); "{reason}");
        }
    }
}

impl Lease {
    fn new(db: Thread, fence: Fence, acquired_at: Ts) -> Self {
        let (lost, _) = signal::signal(None);
        let ownership = Arc::new(Ownership {
            partition_key: fence.partition_key.clone(),
            clock: db.background.clock.clone(),
            deadline: Mutex::new(acquired_at.saturating_add(db.timing.ttl)),
            lost,
        });
        let (renewal, released) = (Mutex::new(None), AtomicBool::new(false));
        let held = Held { db, fence, ownership, renewal, released };
        Self { inner: Arc::new(held) }
    }

    pub fn fence(&self) -> &Fence {
        &self.inner.fence
    }

    pub fn owner(&self) -> &Holder {
        &self.inner.fence.holder
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
        *renewal = Some(self.inner.db.background.spawner.spawn(work.run()));
    }

    /// Ends the renewal; the lease itself is released by the commit or a release.
    pub fn stop(&self) {
        self.inner.renewal.lock().take();
    }

    /// A write already released the lease (a commit that did not keep it).
    pub(crate) fn mark_released(&self) {
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
        let mut lost = ownership.lost.cell();
        loop {
            if let Err(error) = ownership.check() {
                return error;
            }
            let deadline = *ownership.deadline.lock();
            // `self` holds the producer for the whole wait.
            tokio::select! {
                biased;
                _ = ownership.clock.wait_until(deadline) => {},
                _ = lost.changed() => {}
            }
        }
    }

    /// Releases without writing anything else but, when given, the `run_ended` event of `run_id`.
    /// Both are keyed by the token, so a lost acknowledgment is sent again.
    pub async fn release(&self, ended: Option<(&str, &EventWrite)>) -> Result<(), StorageError> {
        let Fence { partition_key, token, holder, .. } = &self.inner.fence;
        let (db, conn) = (&self.inner.db, self.inner.db.conn());
        let batch = || conn.batch(release(partition_key, token, holder.as_str(), ended));
        db.resend(StorageWrite::Release).send(batch).await?;
        self.mark_released();
        Ok(())
    }

    /// The release of a run dropped before it ended, in the background: `run_ended` `canceled`
    /// for `run_id`, then the lease, both fenced on its token.
    pub(crate) fn release_dropped(&self, run_id: &str) {
        let reason = Some(AbandonReason::RunDropped.as_str().to_owned());
        let event = JournalEvent::RunEnded(RunEnded { outcome: RunOutcome::Canceled, reason });
        let ended = EventWrite::new(unix_now_ms(), event);
        let Fence { partition_key, token, holder, .. } = &self.inner.fence;
        let batch = release(partition_key, token, holder.as_str(), Some((run_id, &ended)));
        release_detached(holder, self.inner.db.clone(), partition_key, batch);
        self.stop();
        self.mark_released();
    }

    /// Releases when no inbox row is pending past `drained_id`; `true` when released. A row
    /// pending keeps the lease, extended; a lease another holder took is `LeaseLost`. After a
    /// lost acknowledgment a token no longer the partition's is this release's.
    pub async fn release_if_drained(&self) -> Result<bool, StorageError> {
        let (db, fence) = (&self.inner.db, &self.inner.fence);
        let (partition_key, token) = (&*fence.partition_key, &*fence.token);
        let released = db.resend(StorageWrite::Release).probed(
            || async move { Ok(db.exec(ReleaseLeaseIfDrained { partition_key, token }).await? > 0) },
            // a lost release left the partition under another token, or none
            || async move {
                let tokens =
                    db.read(SelectFence { partition_key }, |row| Ok(row[1].as_str().map(str::to_owned)));
                let held = tokens.await?.into_iter().any(|t| t.as_deref() == Some(token));
                Ok((!held).then_some(true))
            },
        );
        if released.await? {
            self.mark_released();
            return Ok(true);
        }
        if db.exec_idempotent(StorageWrite::Extend, || extend(fence, db.timing.millis())).await?
            == 0
        {
            return Err(StorageError::LeaseLost(format!(
                "lease on partition_key {} was taken",
                fence.partition_key
            )));
        }
        Ok(false)
    }
}

/// The lease's renewal task; it holds the ownership, not the lease handle.
struct LeaseRenewal {
    db: Thread,
    fence: Fence,
    ownership: Arc<Ownership>,
}

impl LeaseRenewal {
    async fn run(self) {
        let LeaseTiming { ttl, renew_interval: interval } = self.db.timing;
        let clock = self.db.clock();
        let mut next = self.ownership.deadline.lock().saturating_sub(ttl).saturating_add(interval);
        loop {
            let deadline = *self.ownership.deadline.lock();
            clock.wait_until(next.min(deadline)).await;
            if self.ownership.check().is_err() {
                return;
            }
            let started = clock.now();
            next = started.saturating_add(interval);
            if self.renew_once(started, next.min(deadline), ttl).await.is_break() {
                return;
            }
        }
    }

    /// One extend, bounded by `until`; breaks once ownership is lost. Keyed by the token, an
    /// extend whose answer was lost is sent again until the ownership deadline.
    async fn renew_once(
        &self,
        started: Ts,
        until: Ts,
        ttl: std::time::Duration,
    ) -> ControlFlow<()> {
        let wait = self.ownership.deadline.lock().saturating_duration_since(started);
        let extend = || self.db.exec(extend(&self.fence, self.db.timing.millis()));
        let resend = self.db.resend_within(wait, StorageWrite::Extend);
        let outcome = lk_agents_abi::before(self.db.clock(), Some(until), resend.send(extend))
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
                log::debug!(target: lk_agents_abi::LOG_TARGET, partition_key = self.fence.partition_key.as_str(); "lease extend failed, retrying: {error}");
            }
            Err(error) => {
                self.ownership.lose(format!("lease renewal failed: {error}"));
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

fn extend(fence: &Fence, lease_ms: i64) -> ExtendLease<'_> {
    ExtendLease {
        partition_key: &fence.partition_key,
        holder: fence.holder.as_str(),
        token: &fence.token,
        lease_ms,
    }
}

#[cfg(test)]
#[path = "lease_tests.rs"]
mod tests;
