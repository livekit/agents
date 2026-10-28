//! Lease timing decisions on supplied time: when a held lease is renewed and when a failing
//! renewal means the lease is lost ([`Renewal`]), and how a waiter paces its acquire attempts
//! ([`LanePoll`]). The runtime owns the timers and executes the extends. Times are monotonic
//! readings as a [`Duration`] since the runtime's epoch.

use std::time::Duration;

use crate::StorageError;

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
