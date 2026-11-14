use std::time::Duration;

use lk_agents_abi::Ts;

/// When a stream's frames may go: no sooner than `min_interval` after the frame before, and, with
/// `real_time`, no faster than the durations they last. Shared streams reserve their slots in
/// turn.
#[derive(Debug, Default)]
pub(super) struct Pacer {
    min_interval: Option<Duration>,
    real_time: bool,
    /// The earliest the next frame may go.
    next_at: Option<Ts>,
    /// When the frames sent since the last gap began, and how long they last.
    played: Option<(Ts, Duration)>,
}

impl Pacer {
    pub(super) fn new(min_interval: Option<Duration>, real_time: bool) -> Self {
        Pacer { min_interval, real_time, ..Pacer::default() }
    }

    /// Reserves the slot of a frame lasting `duration`: when it may go.
    pub(super) fn reserve(&mut self, now: Ts, duration: Duration) -> Ts {
        let mut at = self.next_at.map_or(now, |next| next.max(now));
        if self.real_time && !duration.is_zero() {
            let (origin, sent) = match self.played {
                // the frames fell behind real time: the pace restarts from here
                Some((origin, sent)) if origin.saturating_add(sent) >= at => (origin, sent),
                _ => (at, Duration::ZERO),
            };
            at = origin.saturating_add(sent);
            self.played = Some((origin, sent + duration));
        }
        if let Some(interval) = self.min_interval {
            self.next_at = Some(at.saturating_add(interval));
        }
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Ts {
        Ts::from_nanos(n * 1_000_000)
    }

    // Catches: a rate limit counted per stream instead of across the streams sharing it, or a
    // frame sent before the interval since the one before.
    #[test]
    fn frames_are_spaced_by_the_minimum_interval_whoever_sends_them() {
        let mut pacer = Pacer::new(Some(Duration::from_millis(60)), false);
        assert_eq!(pacer.reserve(ms(0), Duration::ZERO), ms(0));
        assert_eq!(pacer.reserve(ms(10), Duration::ZERO), ms(60));
        assert_eq!(pacer.reserve(ms(10), Duration::ZERO), ms(120));
        assert_eq!(pacer.reserve(ms(500), Duration::ZERO), ms(500));
    }

    // Catches: frames sent faster than they play, or a gap in the input made up by a burst.
    #[test]
    fn real_time_frames_go_no_faster_than_they_last_and_restart_after_a_gap() {
        let mut pacer = Pacer::new(None, true);
        let frame = Duration::from_millis(20);
        assert_eq!(pacer.reserve(ms(0), frame), ms(0));
        assert_eq!(pacer.reserve(ms(1), frame), ms(20));
        assert_eq!(pacer.reserve(ms(2), frame), ms(40));
        assert_eq!(pacer.reserve(ms(1000), frame), ms(1000));
        assert_eq!(pacer.reserve(ms(1000), frame), ms(1020));
    }
}
