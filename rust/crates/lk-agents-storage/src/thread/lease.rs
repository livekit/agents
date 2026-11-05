//! A held lease: its fence, the renewal task that keeps it, and its observed loss. The renewal
//! task holds only the database and the fence, never the lease itself, so dropping the last
//! [`Lease`] handle stops it. A waiter paces its acquire attempts with [`LanePoll`].

use std::ops::ControlFlow;
use std::sync::Arc;

use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::sql::{Bind, SqlValue};
use crate::thread::db::EventWrite;
use crate::thread::schema::{
    ExtendLease, InsertRunEnded, ReleaseLease, ReleaseLeaseIfDrained, WithdrawWaiter,
};
use crate::thread::{LeaseTiming, Thread};
use crate::{Fence, StorageError};

pub use lk_agents_abi::storage::thread::LanePoll;

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
                log::debug!(target: "livekit.agents", lane = lane.as_str(); "waiter cleanup failed: {error}");
            }
        });
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
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(); "{reason}");
        }
    }
}

impl Lease {
    pub(crate) fn new(db: Thread, fence: Fence, acquired_at: Instant) -> Self {
        let (lost, _) = watch::channel(None);
        let ownership = Arc::new(Ownership {
            lane: fence.lane.clone(),
            deadline: Mutex::new(acquired_at + db.timing.ttl),
            lost,
        });
        Self { inner: Arc::new(Held { db, fence, ownership, renewal: Mutex::new(None) }) }
    }

    pub fn fence(&self) -> &Fence {
        &self.inner.fence
    }

    pub fn holder(&self) -> &str {
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
        *renewal = Some(TokioSpawner::current().spawn(work.run()));
    }

    /// Ends the renewal; the lease itself is released by the commit or a release.
    pub fn stop(&self) {
        self.inner.renewal.lock().take();
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
        let mut batch = Vec::new();
        if let Some((run_id, e)) = ended {
            let payload = SqlValue::json(&e.payload);
            batch.push(InsertRunEnded { lane, token, ts_ms: e.ts_ms, run_id, payload }.bind());
        }
        batch.push(ReleaseLease { lane, token }.bind());
        self.inner.db.conn().batch(batch).await?;
        Ok(())
    }

    /// Releases when no inbox row is pending past `drained_id`; `true` when released. A row
    /// pending keeps the lease, extended; a lease another holder took is `LeaseLost`.
    pub async fn release_if_drained(&self) -> Result<bool, StorageError> {
        let (db, fence) = (&self.inner.db, &self.inner.fence);
        if db.exec(ReleaseLeaseIfDrained { lane: &fence.lane, token: &fence.token }).await? > 0 {
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
                log::debug!(target: "livekit.agents", lane = self.fence.lane.as_str(); "lease extend failed, retrying: {error}");
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
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn the_poll_backs_off_to_a_second_and_stops_at_its_deadline() {
        let t0 = Duration::from_secs(100);
        let ms = Duration::from_millis;
        let mut p = LanePoll::new(t0, ms(2000));
        assert_eq!(p.next(t0), Some(t0 + ms(100)));
        assert_eq!(p.next(t0 + ms(100)), Some(t0 + ms(300)));
        assert_eq!(p.next(t0 + ms(300)), Some(t0 + ms(700)));
        assert_eq!(p.next(t0 + ms(700)), Some(t0 + ms(1500)));
        assert_eq!(p.next(t0 + ms(1500)), Some(t0 + ms(2500)));
        assert_eq!(p.next(t0 + ms(2500)), None);
        assert_eq!(LeaseTiming::new(ms(1500), ms(1)).unwrap().secs(), 2);
    }
}
