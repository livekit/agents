//! A held lease: its fence, the renewal task that keeps it, and its observed loss, with the
//! timing decisions on supplied time: when a held lease is renewed and when a failing renewal
//! means the lease is lost ([`Renewal`]), and how a waiter paces its acquire attempts
//! ([`LanePoll`]). The renewal task holds only the database and the fence, never the lease
//! itself, so dropping the last [`Lease`] handle stops it.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_net::task::OwnedTask;
use tokio::sync::watch;

use crate::db::{self, EventWrite};
use crate::engine::ThreadDb;
use crate::{Fence, StorageError};

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

/// The default lease TTL, measured on the database's clock.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(10);
/// The default interval at which an open run renews.
pub const DEFAULT_RENEW_INTERVAL: Duration = Duration::from_secs(3);

/// A lease TTL as the whole seconds the database counts, rounded up.
pub fn lease_secs(ttl: Duration) -> i64 {
    (ttl.as_secs() + u64::from(ttl.subsec_nanos() > 0)) as i64
}

/// What an extend attempt means for the lease.
#[derive(Debug, Clone, PartialEq)]
pub enum Renewed {
    /// The lease runs another TTL from now.
    Extended,
    /// A retryable failure while the lease cannot have lapsed yet: try again at the next wake.
    Retry,
    /// The lease is gone: another holder took it, it was released, or renewal kept failing
    /// until it could have lapsed. Stop admission and cancel the run.
    Lost(String),
}

/// The renewal schedule of one held lease: an extend every `interval`, each bounded by
/// `interval`, and the lease counted lost once failures outlast `ttl - interval`.
#[derive(Debug, Clone)]
pub struct Renewal {
    ttl: Duration,
    interval: Duration,
    extended_at: Duration,
    next: Duration,
}

impl Renewal {
    /// A lease taken at `acquired_at`.
    pub fn new(ttl: Duration, interval: Duration, acquired_at: Duration) -> Self {
        Self { ttl, interval, extended_at: acquired_at, next: acquired_at + interval }
    }

    /// When the next extend is sent.
    pub fn wake_at(&self) -> Duration {
        self.next
    }

    /// How long one extend may take before it counts as failed.
    pub fn attempt_timeout(&self) -> Duration {
        self.interval
    }

    /// The extend sent at the wake answered at `now`: `Ok(true)` when it changed the lease row,
    /// `Ok(false)` when the token is no longer the lane's, `Err` when it failed or timed out.
    pub fn on_extend(&mut self, now: Duration, outcome: Result<bool, &StorageError>) -> Renewed {
        self.next = now + self.interval;
        match outcome {
            Ok(true) => {
                self.extended_at = now;
                Renewed::Extended
            }
            Ok(false) => Renewed::Lost("the lease was taken or released".into()),
            Err(e)
                if e.retryable()
                    && now.saturating_sub(self.extended_at)
                        < self.ttl.saturating_sub(self.interval) =>
            {
                Renewed::Retry
            }
            Err(e) => Renewed::Lost(format!("lease renewal failed: {e}")),
        }
    }
}

/// Waiting for a lane another holder has: a pause from 100 ms, doubling up to 1 s, until a
/// deadline.
#[derive(Debug, Clone)]
pub struct LanePoll {
    deadline: Duration,
    pause: Duration,
}

impl LanePoll {
    pub fn new(now: Duration, timeout: Duration) -> Self {
        Self { deadline: now + timeout, pause: Duration::from_millis(100) }
    }

    /// When to try again after a miss at `now`; `None` once the deadline passed.
    pub fn next(&mut self, now: Duration) -> Option<Duration> {
        if now >= self.deadline {
            return None;
        }
        let at = now + self.pause;
        self.pause = (self.pause * 2).min(Duration::from_secs(1));
        Some(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unavailable() -> StorageError {
        StorageError::Unavailable("the node is draining".into())
    }

    #[test]
    fn failing_extends_are_retried_until_the_lease_could_have_lapsed() {
        let t0 = Duration::from_secs(100);
        let s = Duration::from_secs;
        let mut r = Renewal::new(s(10), s(3), t0);
        assert_eq!(r.wake_at(), t0 + s(3));
        assert_eq!(r.on_extend(t0 + s(3), Ok(true)), Renewed::Extended);
        assert_eq!(r.wake_at(), t0 + s(6));
        assert_eq!(r.on_extend(t0 + s(6), Err(&unavailable())), Renewed::Retry);
        assert_eq!(r.on_extend(t0 + s(9), Err(&unavailable())), Renewed::Retry);
        // 7 s since the last extend: one more interval and the lease may have lapsed
        assert!(matches!(r.on_extend(t0 + s(10), Err(&unavailable())), Renewed::Lost(_)));
        let mut r = Renewal::new(s(10), s(3), t0);
        assert!(matches!(r.on_extend(t0 + s(3), Ok(false)), Renewed::Lost(_)));
        let mut r = Renewal::new(s(10), s(3), t0);
        let fatal = StorageError::Unauthorized("key".into());
        assert!(matches!(r.on_extend(t0 + s(3), Err(&fatal)), Renewed::Lost(_)));
    }

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
