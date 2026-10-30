//! The process's clocks: Tokio time for scheduling, the OS clock (or a synthetic one) for wall
//! stamps. The `Clock` contract and the virtual clock the DST moves stay in `lk_agents_abi`.

use parking_lot::Mutex;
use std::time::{Duration, SystemTime};

use lk_agents_abi::{Clock, TICK_EPSILON, Ts, simulated_epoch, unix_secs};

/// `time.time()`, for code with no clock of its own.
pub fn unix_now() -> f64 {
    unix_secs(SystemTime::now())
}

/// `int(time.time() * 1000)`: whole Unix milliseconds, for code with no clock of its own.
pub fn unix_now_ms() -> i64 {
    (unix_now() * 1000.0) as i64
}

/// Tokio's clock for scheduling, a synthetic one for wall stamps, so a recorded case replays
/// byte-exactly.
pub struct SimClock {
    real: RealClock,
    origin: SystemTime,
}

impl SimClock {
    pub fn new() -> Self {
        SimClock { real: RealClock::new(), origin: simulated_epoch() }
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SimClock {
    fn now(&self) -> Ts {
        self.real.now()
    }

    fn wait_until(&self, at: Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move { self.real.wait_until(at).await })
    }

    fn wall(&self) -> SystemTime {
        self.real.nudged(self.origin + Duration::from_nanos(self.now().as_nanos()))
    }
}

/// Anchors `Ts::ZERO` at construction. Scheduling follows Tokio time; wall stamps use the OS clock.
pub struct RealClock {
    origin: tokio::time::Instant,
    last_wall: Mutex<SystemTime>,
}

impl RealClock {
    pub fn new() -> Self {
        RealClock {
            origin: tokio::time::Instant::now(),
            last_wall: Mutex::new(SystemTime::UNIX_EPOCH),
        }
    }
}

impl Default for RealClock {
    fn default() -> Self {
        Self::new()
    }
}

impl RealClock {
    /// The candidate reading is passed in so tests can present a repeat or a backwards step.
    fn nudged(&self, candidate: SystemTime) -> SystemTime {
        let mut last = self.last_wall.lock();
        *last = if candidate > *last { candidate } else { *last + TICK_EPSILON };
        *last
    }
}

impl Clock for RealClock {
    fn now(&self) -> Ts {
        Ts::from_nanos(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }

    fn wall(&self) -> SystemTime {
        // Real time can repeat (coarse clocks) or go backwards (NTP): nudge it too.
        self.nudged(SystemTime::now())
    }

    fn wait_until(&self, at: Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            let now = self.now();
            if at > now {
                tokio::time::sleep(at.saturating_duration_since(now)).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{Elapsed, timeout};
    use std::future::Future;
    use std::task::Poll;

    #[tokio::test(start_paused = true)]
    async fn runtime_clock_deadlines_follow_paused_tokio_time() {
        let clock = RealClock::new();
        let deadline = Ts::ZERO.saturating_add(Duration::from_secs(5));
        tokio::time::advance(Duration::from_secs(4)).await;
        assert_eq!(clock.now(), Ts::ZERO.saturating_add(Duration::from_secs(4)));
        let wait = clock.wait_until(deadline);
        tokio::pin!(wait);
        assert!(std::future::poll_fn(|cx| Poll::Ready(wait.as_mut().poll(cx))).await.is_pending());
        wait.await;
        assert_eq!(clock.now(), deadline);
        assert_eq!(
            timeout(&clock, Duration::from_secs(2), std::future::pending::<()>()).await,
            Err(Elapsed)
        );
        assert_eq!(clock.now(), Ts::ZERO.saturating_add(Duration::from_secs(7)));
    }

    // Catches: `RealClock::wall` reporting the nudge origin instead of the real time.
    #[test]
    fn the_real_clock_wall_reports_real_time_and_never_repeats() {
        let c = RealClock::new();
        let reads: Vec<_> = (0..8).map(|_| c.wall()).collect();
        assert!(reads.windows(2).all(|w| w[1] > w[0]), "{reads:?}");
        // 2024-01-01, i.e. safely in the past but far past the UNIX epoch the nudge starts from.
        let sane = SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_067_200);
        assert!(reads[0] > sane, "wall must be the OS clock, not the nudge origin");
        // Not after a later read: equal on a coarse clock (macOS counts microseconds).
        assert!(reads[0] <= SystemTime::now(), "and it must not be in the future");
    }

    // Catches: the repeat/backwards branch of the nudge, which a live `wall()` read never hits.
    #[test]
    fn the_real_clock_nudges_past_a_repeated_or_backwards_os_reading() {
        let c = RealClock::new();
        let first = c.wall();

        // A coarse clock handing back the same reading twice.
        assert_eq!(c.nudged(first), first + TICK_EPSILON, "an equal reading must still advance");

        // An NTP step backwards: previous value plus one tick.
        assert_eq!(
            c.nudged(first - Duration::from_secs(1)),
            first + 2 * TICK_EPSILON,
            "a backwards step must not drag `created_at` backwards"
        );

        // ...and a genuinely later reading is taken as is, rather than nudged one tick at a time.
        let later = first + Duration::from_secs(10);
        assert_eq!(c.nudged(later), later);
    }
}
