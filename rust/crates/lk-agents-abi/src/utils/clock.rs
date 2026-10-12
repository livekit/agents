use std::future::Future;
use std::pin::pin;
use std::sync::Mutex;
use std::task::Poll;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use tokio::sync::Notify;

/// Nanoseconds since the session started. Not `Instant`, so tests can virtualise time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ts(u64);

impl Ts {
    pub const ZERO: Ts = Ts(0);

    pub const fn from_nanos(n: u64) -> Self {
        Ts(n)
    }
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Saturating: a past deadline is normal and must not panic on a host value.
    pub fn saturating_add(self, d: Duration) -> Ts {
        Ts(self.0.saturating_add(d.as_nanos().min(u64::MAX as u128) as u64))
    }

    pub fn saturating_sub(self, d: Duration) -> Ts {
        Ts(self.0.saturating_sub(d.as_nanos().min(u64::MAX as u128) as u64))
    }

    /// Saturating at zero rather than panicking on reordering.
    pub fn saturating_duration_since(self, earlier: Ts) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

/// The core's only source of time: `now` drives scheduling, `wall` stamps chat items.
#[async_trait]
pub trait Clock: Send + Sync {
    /// Monotonic. Repeated calls at the same instant return the same value.
    fn now(&self) -> Ts;

    /// The only thing that sleeps; every timeout in the workspace is [`timeout`] over this.
    async fn wait_until(&self, at: Ts);

    /// For `ChatItem::created_at` only. Strictly increasing even within one instant: `ChatContext`
    /// orders by `created_at` (the reference nudges 1 µs, `tests/virtual_time.py:57`).
    fn wall(&self) -> SystemTime;
}

/// 2024-01-01T00:00:00Z.
pub fn simulated_epoch() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_067_200)
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

#[async_trait]
impl Clock for SimClock {
    fn now(&self) -> Ts {
        self.real.now()
    }

    async fn wait_until(&self, at: Ts) {
        self.real.wait_until(at).await
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
        let mut last = self.last_wall.lock().unwrap();
        *last = if candidate > *last { candidate } else { *last + TICK_EPSILON };
        *last
    }
}

#[async_trait]
impl Clock for RealClock {
    fn now(&self) -> Ts {
        Ts(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }

    fn wall(&self) -> SystemTime {
        // Real time can repeat (coarse clocks) or go backwards (NTP): nudge it too.
        self.nudged(SystemTime::now())
    }

    async fn wait_until(&self, at: Ts) {
        let now = self.now();
        if at > now {
            tokio::time::sleep(at.saturating_duration_since(now)).await;
        }
    }
}

/// `_TICK_EPSILON` in `tests/virtual_time.py`.
const TICK_EPSILON: Duration = Duration::from_micros(1);

pub struct VirtualClock {
    state: Mutex<Virtual>,
    /// Woken by every `advance`.
    wake: Notify,
}

struct Virtual {
    now: Ts,
    wall_origin: SystemTime,
    last_wall: SystemTime,
}

impl VirtualClock {
    pub fn new() -> Self {
        let origin = simulated_epoch();
        VirtualClock {
            state: Mutex::new(Virtual { now: Ts::ZERO, wall_origin: origin, last_wall: origin }),
            wake: Notify::new(),
        }
    }

    /// Going backwards is a test bug, not a runtime condition, so it panics.
    pub fn advance(&self, d: Duration) {
        {
            let mut s = self.state.lock().unwrap();
            s.now = s.now.saturating_add(d);
        }
        self.wake.notify_waiters();
    }

    /// Panics on a backwards jump.
    pub fn advance_to(&self, at: Ts) {
        {
            let mut s = self.state.lock().unwrap();
            assert!(at >= s.now, "VirtualClock cannot go backwards: {:?} -> {:?}", s.now, at);
            s.now = at;
        }
        self.wake.notify_waiters();
    }
}

impl Default for VirtualClock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Clock for VirtualClock {
    fn now(&self) -> Ts {
        self.state.lock().unwrap().now
    }

    async fn wait_until(&self, at: Ts) {
        loop {
            // Armed before the read, so an `advance` in between is not missed.
            let parked = self.wake.notified();
            let mut parked = pin!(parked);
            parked.as_mut().enable();
            if self.now() >= at {
                return;
            }
            parked.await;
        }
    }

    fn wall(&self) -> SystemTime {
        let mut s = self.state.lock().unwrap();
        let tracking = s.wall_origin + Duration::from_nanos(s.now.as_nanos());
        s.last_wall = if tracking > s.last_wall { tracking } else { s.last_wall + TICK_EPSILON };
        s.last_wall
    }
}

/// A value, not an error: each site decides what a timeout means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elapsed;

/// A ready result beats an elapsed deadline.
pub async fn timeout<F: Future>(
    clock: &dyn Clock,
    limit: Duration,
    fut: F,
) -> Result<F::Output, Elapsed> {
    let deadline = clock.now().saturating_add(limit);
    let mut fut = pin!(fut);
    let mut wait = pin!(clock.wait_until(deadline));
    std::future::poll_fn(|cx| {
        if let Poll::Ready(v) = fut.as_mut().poll(cx) {
            return Poll::Ready(Ok(v));
        }
        if wait.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(Elapsed));
        }
        Poll::Pending
    })
    .await
}

/// Read once when a session starts: every stamp that crosses is arithmetic against this one pair,
/// so events never disagree about when `Ts::ZERO` was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallAnchor {
    pub wall: f64,
    pub at: Ts,
}

impl WallAnchor {
    pub fn of(clock: &dyn Clock) -> Self {
        let wall = clock
            .wall()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_default();
        WallAnchor { wall, at: clock.now() }
    }

    /// Saturating at the anchor; [`seconds`](Self::seconds) is the signed form.
    pub fn wall(&self, ts: Ts) -> f64 {
        self.wall + ts.saturating_duration_since(self.at).as_secs_f64()
    }

    /// Negative before the anchor (the turn machine can mint one while the room connects).
    pub fn seconds(&self, ts: Ts) -> f64 {
        if ts >= self.at {
            self.wall + ts.saturating_duration_since(self.at).as_secs_f64()
        } else {
            self.wall - self.at.saturating_duration_since(ts).as_secs_f64()
        }
    }

    /// Saturating at zero (right for a playback position); to place an item use `ts_checked`.
    pub fn ts(&self, wall: f64) -> Ts {
        self.ts_checked(wall).unwrap_or(Ts::ZERO)
    }

    /// `None` for a stamp before `Ts::ZERO` or a NaN. The realtime bridge's `turn_started_at`.
    pub fn ts_checked(&self, wall: f64) -> Option<Ts> {
        if !wall.is_finite() {
            return None;
        }
        let delta = wall - self.wall;
        if delta >= 0.0 {
            // Only fails on an absurd magnitude; saturate to stay total.
            let d = Duration::try_from_secs_f64(delta).unwrap_or(Duration::MAX);
            return Some(self.at.saturating_add(d));
        }
        let back = Duration::try_from_secs_f64(-delta).unwrap_or(Duration::MAX);
        let since_zero = Duration::from_nanos(self.at.as_nanos());
        (back <= since_zero).then(|| self.at.saturating_sub(back))
    }
}

#[cfg(test)]
mod wall_anchor_tests {
    use super::{Ts, WallAnchor};

    fn anchor() -> WallAnchor {
        // Arbitrary numbers; the difference is what is under test.
        WallAnchor { wall: 1_787_000_000.0, at: Ts::from_nanos(10_000_000_000) }
    }

    // Catches: shipping the raw `Ts` as a wall clock (a 1970 timestamp), an unsigned underflow
    // before the anchor, the inverse not round-tripping, or a stamp before `Ts::ZERO` placed at
    // `Ts::ZERO` (`ts` clamps, `ts_checked` declines, NaN too).
    #[test]
    fn the_anchor_maps_both_ways_around_the_wall_reading() {
        let a = anchor();
        for (ns, wall) in [(12_500_000_000, 1_787_000_002.5), (7_000_000_000, 1_786_999_997.0)] {
            assert_eq!(a.seconds(Ts::from_nanos(ns)), wall);
            assert_eq!(a.ts_checked(wall), Some(Ts::from_nanos(ns)));
            assert_eq!(a.ts(wall), Ts::from_nanos(ns));
        }
        assert_eq!(a.ts_checked(1_786_999_989.0), None);
        assert_eq!(a.ts_checked(1_786_999_990.0), Some(Ts::ZERO), "the epoch itself is placeable");
        assert_eq!(a.ts_checked(f64::NAN), None);
        assert_eq!(a.ts(1_786_999_989.0), Ts::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a virtual-clock wait that never returns, or returns before time moved.
    #[tokio::test]
    async fn a_virtual_timeout_fires_only_when_time_moves() {
        let c = std::sync::Arc::new(VirtualClock::new());
        let waiter = {
            let c = c.clone();
            tokio::spawn(async move {
                timeout(&*c, Duration::from_secs(1), std::future::pending::<()>()).await
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "nothing moved time, the wait must be parked");
        c.advance(Duration::from_millis(999));
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "999 ms is not 1 s");
        c.advance(Duration::from_millis(1));
        assert_eq!(waiter.await.unwrap(), Err(Elapsed));
    }

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

    #[tokio::test]
    async fn a_ready_future_wins_over_the_deadline() {
        let c = VirtualClock::new();
        assert_eq!(timeout(&c, Duration::ZERO, async { 7 }).await, Ok(7));
    }

    // Catches: two chat items created in the same tick get equal `created_at`.
    #[test]
    fn wall_is_strictly_increasing_within_one_instant() {
        let c = VirtualClock::new();
        let reads: Vec<_> = (0..5).map(|_| c.wall()).collect();
        assert!(reads.windows(2).all(|w| w[1] > w[0]), "{reads:?}");
        assert_eq!(c.now(), Ts::ZERO, "wall reads must not move the scheduling clock");
    }

    // Catches: `wall` drifting from `now` because the nudge accumulated instead of re-syncing.
    #[test]
    fn wall_resyncs_to_the_clock_after_it_advances() {
        let c = VirtualClock::new();
        let origin = c.wall(); // burns one nudge
        c.advance(Duration::from_secs(1));
        let after = c.wall();
        assert_eq!(after.duration_since(origin).unwrap(), Duration::from_secs(1) - TICK_EPSILON);
    }

    // Catches: a deadline computed in the past panicking on a host-supplied value, or the
    // saturating ops degenerating into "always zero".
    #[test]
    fn ts_arithmetic_saturates_instead_of_panicking() {
        assert_eq!(Ts::from_nanos(500).saturating_sub(Duration::from_nanos(200)), Ts::from_nanos(300));
        assert_eq!(
            Ts::from_nanos(500).saturating_duration_since(Ts::from_nanos(200)),
            Duration::from_nanos(300)
        );
        assert_eq!(Ts::ZERO.saturating_sub(Duration::from_secs(1)), Ts::ZERO);
        assert_eq!(Ts::ZERO.saturating_duration_since(Ts::from_nanos(99)), Duration::ZERO);
        assert_eq!(
            Ts::from_nanos(u64::MAX).saturating_add(Duration::from_secs(1)).as_nanos(),
            u64::MAX
        );
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
        assert!(
            reads[0].duration_since(SystemTime::now()).is_err(),
            "and it must not be in the future"
        );
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

    #[test]
    #[should_panic(expected = "cannot go backwards")]
    fn virtual_clock_rejects_backwards_jumps() {
        let c = VirtualClock::new();
        c.advance(Duration::from_secs(5));
        c.advance_to(Ts::from_nanos(1));
    }
}
