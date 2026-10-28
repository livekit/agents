//! A held lease: its fence, the renewal task that keeps it, and its observed loss. The renewal
//! task holds only the database and the fence, never the lease itself, so dropping the last
//! [`Lease`] handle stops it.

use std::sync::{Arc, Mutex};

use lk_agents_storage::db::{self, EventWrite};
use lk_agents_storage::lease::{Renewal, Renewed};
use lk_agents_storage::{Fence, StorageError};
use tokio::sync::watch;

use super::engine::ThreadDb;
use crate::task::OwnedTask;

/// A lease an acquire or a receive took: the lane at `version` under `token`. Cloned by the run
/// and its checkpoint; [`Lease::keep_alive`] extends it while the run works.
#[derive(Clone)]
pub struct Lease {
    inner: Arc<Held>,
}

struct Held {
    db: ThreadDb,
    fence: Fence,
    /// Why the lease was lost, once it was.
    lost: Arc<watch::Sender<Option<String>>>,
    /// Owned here: dropping the last handle cancels it.
    renewal: Mutex<Option<OwnedTask>>,
}

impl Lease {
    pub(crate) fn new(db: ThreadDb, fence: Fence) -> Self {
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

    /// Extends the lease on the core's schedule until [`Lease::stop`] or the last handle drops.
    /// An extend that changes no row, or failures that outlast the lease, mark it lost.
    pub fn keep_alive(&self) {
        let (db, fence, lost) =
            (self.inner.db.clone(), self.inner.fence.clone(), self.inner.lost.clone());
        let work = async move {
            let timing = db.timing;
            let epoch = tokio::time::Instant::now();
            let mut renewal = Renewal::new(timing.ttl, timing.renew_interval, epoch.elapsed());
            loop {
                tokio::time::sleep_until(epoch + renewal.wake_at()).await;
                let extend = db.exec(db::extend(&fence, timing.secs()));
                let outcome = match tokio::time::timeout(renewal.attempt_timeout(), extend).await {
                    Ok(Ok(rows)) => Ok(rows > 0),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(StorageError::Unavailable("the extend timed out".into())),
                };
                match renewal.on_extend(epoch.elapsed(), outcome.as_ref().map(|r| *r)) {
                    Renewed::Extended => {}
                    Renewed::Retry => {
                        log::debug!(target: "livekit.agents", lane = fence.lane.as_str(); "lease extend failed, retrying: {:?}", outcome.err());
                    }
                    Renewed::Lost(reason) => {
                        let reason = format!("lease on lane {}: {reason}", fence.lane);
                        log::warn!(target: "livekit.agents", lane = fence.lane.as_str(); "{reason}");
                        lost.send_replace(Some(reason));
                        return;
                    }
                }
            }
        };
        let task = OwnedTask::spawn(&tokio::runtime::Handle::current(), work, |_| async {});
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
            Ok(reason) => StorageError::Unavailable(reason),
            Err(_) => std::future::pending().await,
        }
    }

    /// Releases without writing anything else but, when given, the `run_ended` event of `run_id`.
    pub async fn release(&self, ended: Option<(&str, &EventWrite)>) -> Result<(), StorageError> {
        self.inner.db.conn().batch(db::release(&self.inner.fence, ended)).await?;
        Ok(())
    }

    /// Releases when no inbox row is pending past `drained_id`; `true` when released.
    pub async fn release_if_drained(&self) -> Result<bool, StorageError> {
        Ok(self.inner.db.exec(db::release_if_drained(&self.inner.fence)).await? > 0)
    }
}
