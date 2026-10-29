//! A held lease: its fence, the renewal task that keeps it, and its observed loss. The renewal
//! task holds only the database and the fence, never the lease itself, so dropping the last
//! [`Lease`] handle stops it. A waiter paces its acquire attempts with [`LanePoll`].

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::sync::watch;

use crate::db::EventWrite;
use crate::engine::{LeaseTiming, Thread};
use crate::schema::{ExtendLease, InsertRunEnded, ReleaseLease, ReleaseLeaseIfDrained};
use crate::sql::{Bind, SqlValue};
use crate::{Fence, StorageError};

pub use lk_agents_abi::storage::thread::LanePoll;

/// A lease an acquire or a receive took: the lane at `version` under `token`. Cloned by the run
/// and its checkpoint; [`Lease::keep_alive`] extends it while the run works.
#[derive(Clone)]
pub struct Lease {
    inner: Arc<Held>,
}

struct Held {
    db: Thread,
    fence: Fence,
    /// Why the lease was lost, once it was.
    lost: Arc<watch::Sender<Option<String>>>,
    /// Owned here: dropping the last handle cancels it.
    renewal: Mutex<Option<OwnedTask>>,
}

impl Lease {
    pub(crate) fn new(db: Thread, fence: Fence) -> Self {
        let (lost, _) = watch::channel(None);
        Self {
            inner: Arc::new(Held { db, fence, lost: Arc::new(lost), renewal: Mutex::new(None) }),
        }
    }

    pub fn fence(&self) -> &Fence {
        &self.inner.fence
    }

    /// The version the lease loaded: the commit's fence.
    pub fn version(&self) -> i64 {
        self.inner.fence.version
    }

    pub fn holder(&self) -> &str {
        &self.inner.fence.holder
    }

    /// Extends the lease every `renew_interval`, each extend bounded by the interval, until
    /// [`Lease::stop`] or the last handle drops. An extend that changes no row marks the lease
    /// lost, and so do failures once they outlast `ttl - renew_interval`: the lease may have
    /// lapsed by then.
    pub fn keep_alive(&self) {
        let (db, fence, lost) =
            (self.inner.db.clone(), self.inner.fence.clone(), self.inner.lost.clone());
        let work = async move {
            let LeaseTiming { ttl, renew_interval: interval } = db.timing;
            let mut extended_at = tokio::time::Instant::now();
            loop {
                tokio::time::sleep(interval).await;
                let extend = db.exec(extend(&fence, db.timing.secs()));
                let outcome = tokio::time::timeout(interval, extend).await.unwrap_or_else(|_| {
                    Err(StorageError::Unavailable("the extend timed out".into()))
                });
                let now = tokio::time::Instant::now();
                let reason = match outcome {
                    Ok(rows) if rows > 0 => {
                        extended_at = now;
                        continue;
                    }
                    Ok(_) => "the lease was taken or released".to_owned(),
                    Err(e) if e.retryable() && now - extended_at < ttl.saturating_sub(interval) => {
                        log::debug!(target: "livekit.agents", lane = fence.lane.as_str(); "lease extend failed, retrying: {e}");
                        continue;
                    }
                    Err(e) => format!("lease renewal failed: {e}"),
                };
                let reason = format!("lease on lane {}: {reason}", fence.lane);
                log::warn!(target: "livekit.agents", lane = fence.lane.as_str(); "{reason}");
                lost.send_replace(Some(reason));
                return;
            }
        };
        let task = TokioSpawner::current().spawn(work);
        *self.inner.renewal.lock().expect("renewal") = Some(task);
    }

    /// Ends the renewal; the lease itself is released by the commit or a release.
    pub fn stop(&self) {
        self.inner.renewal.lock().expect("renewal").take();
    }

    pub fn is_lost(&self) -> bool {
        self.inner.lost.borrow().is_some()
    }

    /// Resolves with the error that cancels the run once the lease is lost; pending while it is
    /// held, so race it against the run.
    pub async fn lost(&self) -> StorageError {
        let mut rx = self.inner.lost.subscribe();
        let reason = rx.wait_for(Option::is_some).await.map(|r| r.clone().unwrap_or_default());
        match reason {
            Ok(reason) => StorageError::LeaseLost(reason),
            Err(_) => std::future::pending().await,
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
fn extend(fence: &Fence, lease_secs: i64) -> ExtendLease<'_> {
    ExtendLease { lane: &fence.lane, holder: &fence.holder, token: &fence.token, lease_secs }
}

/// The default lease TTL, measured on the database's clock.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(10);
/// The default interval at which an open run renews.
pub const DEFAULT_RENEW_INTERVAL: Duration = Duration::from_secs(3);

/// A lease TTL as the whole seconds the database counts, rounded up.
pub fn lease_secs(ttl: Duration) -> i64 {
    (ttl.as_secs() + u64::from(ttl.subsec_nanos() > 0)) as i64
}

#[cfg(test)]
mod tests {
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
        assert_eq!(lease_secs(Duration::from_millis(1500)), 2);
    }
}
