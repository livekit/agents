use std::future::Future;
use std::pin::pin;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::Mutex;
use std::task::{Poll, Waker};
use std::time::{Duration, SystemTime};

/// `t` as `time.time()` reads it: seconds since the Unix epoch, negative before it.
pub fn unix_secs(t: SystemTime) -> f64 {
    match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64(),
        Err(before) => -before.duration().as_secs_f64(),
    }
}

/// The inverse of [`unix_secs`]; `None` for a value no wall clock returns (NaN, infinities, out of
/// range), never a panic.
pub fn wall_from_secs(secs: f64) -> Option<SystemTime> {
    let magnitude = Duration::try_from_secs_f64(secs.abs()).ok()?;
    if secs < 0.0 {
        SystemTime::UNIX_EPOCH.checked_sub(magnitude)
    } else {
        SystemTime::UNIX_EPOCH.checked_add(magnitude)
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// `time.time()`, for code with no clock of its own.
pub fn unix_now() -> f64 {
    unix_secs(SystemTime::now())
}

/// `int(time.time() * 1000)`: whole Unix milliseconds, for code with no clock of its own.
pub fn unix_now_ms() -> i64 {
    (unix_now() * 1000.0) as i64
}

/// Seconds from a host value as a `Duration`: zero for NaN or a non-positive value, at most 1e9 s
/// (+inf included), so a deadline built from it never overflows.
pub fn secs(s: f64) -> Duration {
    if s.is_nan() || s <= 0.0 { Duration::ZERO } else { Duration::from_secs_f64(s.min(1e9)) }
}

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

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
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
pub trait Clock: Send + Sync {
    /// Monotonic. Repeated calls at the same instant return the same value.
    fn now(&self) -> Ts;

    /// The only thing that sleeps; every timeout in the workspace is [`timeout`] over this.
    fn wait_until(&self, at: Ts) -> crate::BoxFuture<'_, ()>;

    /// For `ChatItem::created_at` only. Strictly increasing even within one instant: `ChatContext`
    /// orders by `created_at`, so equal readings are nudged by [`TICK_EPSILON`].
    fn wall(&self) -> SystemTime;
}

/// 2024-01-01T00:00:00Z.
pub fn simulated_epoch() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_067_200)
}

/// The nudge that keeps wall stamps strictly increasing.
pub const TICK_EPSILON: Duration = Duration::from_micros(1);

pub struct VirtualClock {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<Virtual>,
}

struct Virtual {
    now: Ts,
    wall_origin: SystemTime,
    last_wall: SystemTime,
    /// The waits parked on a later instant, woken by every `advance`.
    parked: Vec<Waker>,
}

impl Default for Virtual {
    fn default() -> Self {
        let origin = simulated_epoch();
        Virtual { now: Ts::ZERO, wall_origin: origin, last_wall: origin, parked: Vec::new() }
    }
}

impl VirtualClock {
    pub fn new() -> Self {
        VirtualClock { state: Default::default() }
    }

    /// Going backwards is a test bug, not a runtime condition, so it panics.
    pub fn advance(&self, d: Duration) {
        let s = self.state.lock().unwrap();
        let at = s.now.saturating_add(d);
        Self::move_to(s, at);
    }

    /// Panics on a backwards jump.
    pub fn advance_to(&self, at: Ts) {
        let s = self.state.lock().unwrap();
        assert!(at >= s.now, "VirtualClock cannot go backwards: {:?} -> {:?}", s.now, at);
        Self::move_to(s, at);
    }

    /// Wakes every parked wait outside the lock; each re-reads the time it waits for.
    fn move_to(mut s: std::sync::MutexGuard<'_, Virtual>, at: Ts) {
        s.now = at;
        let parked = std::mem::take(&mut s.parked);
        drop(s);
        parked.into_iter().for_each(Waker::wake);
    }
}

impl Default for VirtualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for VirtualClock {
    fn now(&self) -> Ts {
        self.state.lock().unwrap().now
    }

    fn wait_until(&self, at: Ts) -> crate::BoxFuture<'_, ()> {
        Box::pin(std::future::poll_fn(move |cx| {
            let mut s = self.state.lock().unwrap();
            if s.now >= at {
                return Poll::Ready(());
            }
            // Once per task: a select loop polling the same wait again does not grow the list.
            if !s.parked.iter().any(|w| w.will_wake(cx.waker())) {
                s.parked.push(cx.waker().clone());
            }
            Poll::Pending
        }))
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
    before(clock, Some(clock.now().saturating_add(limit)), fut).await
}

/// `fut` unless `deadline` passes first; `None` never elapses. A ready result beats an elapsed
/// deadline.
pub async fn before<F: Future>(
    clock: &dyn Clock,
    deadline: Option<Ts>,
    fut: F,
) -> Result<F::Output, Elapsed> {
    let mut fut = pin!(fut);
    let mut wait = pin!(async {
        match deadline {
            Some(at) => clock.wait_until(at).await,
            None => std::future::pending().await,
        }
    });
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
        let wall = unix_secs(clock.wall());
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
    #[test]
    fn a_virtual_timeout_fires_only_when_time_moves() {
        let c = VirtualClock::new();
        let mut waiter = pin!(timeout(&c, Duration::from_secs(1), std::future::pending::<()>()));
        struct Woken(std::sync::OnceLock<()>);
        impl std::task::Wake for Woken {
            fn wake(self: std::sync::Arc<Self>) {
                let _ = self.0.set(());
            }
        }
        let woken = std::sync::Arc::new(Woken(std::sync::OnceLock::new()));
        let waker = Waker::from(woken.clone());
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(waiter.as_mut().poll(&mut cx).is_pending(), "nothing moved time");
        c.advance(Duration::from_millis(999));
        assert!(waiter.as_mut().poll(&mut cx).is_pending(), "999 ms is not 1 s");
        c.advance(Duration::from_millis(1));
        assert!(woken.0.get().is_some(), "moving time wakes the parked wait");
        assert_eq!(waiter.as_mut().poll(&mut cx), Poll::Ready(Err(Elapsed)));
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
        assert_eq!(
            Ts::from_nanos(500).saturating_sub(Duration::from_nanos(200)),
            Ts::from_nanos(300)
        );
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
}

#[cfg(test)]
mod wall_tests {
    use super::{unix_secs, wall_from_secs};

    // Catches: converting `created_at` through anything coarser than f64 (the 1e-6 `_summarize`
    // nudge must survive).
    #[test]
    fn a_wall_clock_stamp_survives_the_round_trip_including_the_summarize_nudge() {
        let now = 1_767_225_600.123_456_7_f64; // a real `time.time()` magnitude
        assert_eq!(unix_secs(wall_from_secs(now).unwrap()), now);

        let nudged = now + 1e-6;
        assert!(nudged > now, "the f64 itself must be distinguishable");
        assert!(
            wall_from_secs(nudged).unwrap() > wall_from_secs(now).unwrap(),
            "a 1us nudge collapsed on the way into SystemTime"
        );
        assert_eq!(unix_secs(wall_from_secs(nudged).unwrap()), nudged);
    }

    // Below ~4.5e6 s an f64 is finer than `SystemTime`'s nanosecond, so such stamps quantize; no
    // real `time.time()` is there.
    // Catches: "fixing" that by rounding or a coarser representation.
    #[test]
    fn sub_nanosecond_stamps_quantize_and_real_wall_clocks_do_not() {
        // Below the boundary: an f64 with more precision than a nanosecond loses the excess.
        let tiny = 0.745_466_620_754_925_9_f64;
        assert_ne!(unix_secs(wall_from_secs(tiny).unwrap()), tiny, "the boundary moved");
        assert_eq!(unix_secs(wall_from_secs(tiny).unwrap()), 0.745_466_621);

        // Above it -- every value `time.time()` can return -- the round trip is exact.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..100_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let now = 1.7e9 + (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0e8;
            assert_eq!(unix_secs(wall_from_secs(now).unwrap()), now, "{now} did not survive");
        }
    }

    // Catches: `Duration::from_secs_f64` panicking on a host-supplied nan/inf/negative.
    #[test]
    fn a_nonsensical_created_at_is_none_not_a_panic() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e30] {
            assert!(wall_from_secs(bad).is_none(), "{bad} should not be accepted");
        }
        // ...but a pre-epoch stamp is representable, and round-trips.
        assert_eq!(unix_secs(wall_from_secs(-1.5).unwrap()), -1.5);
    }
}
