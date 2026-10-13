use std::collections::HashMap;
use std::hash::Hash;

use crate::Ts;

/// One closed enum per machine, so timers stay bounded.
pub trait TimerId: Copy + Eq + Hash + std::fmt::Debug {}
impl<T: Copy + Eq + Hash + std::fmt::Debug> TimerId for T {}

/// `arm` replaces (a stale handle can never fire), `cancel` is idempotent. A linear scan, not a
/// heap: ~12 live timers, and replace-on-arm would need tombstones.
#[derive(Debug)]
pub struct TimerSet<T> {
    armed: HashMap<T, Ts>,
}

impl<T: TimerId> Default for TimerSet<T> {
    fn default() -> Self {
        TimerSet { armed: HashMap::new() }
    }
}

impl<T: TimerId> TimerSet<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the deadline it replaced (for tests).
    pub fn arm(&mut self, timer: T, at: Ts) -> Option<Ts> {
        self.armed.insert(timer, at)
    }

    pub fn cancel(&mut self, timer: T) -> Option<Ts> {
        self.armed.remove(&timer)
    }

    pub fn is_armed(&self, timer: T) -> bool {
        self.armed.contains_key(&timer)
    }

    pub fn len(&self) -> usize {
        self.armed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.armed.is_empty()
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        self.armed.values().copied().min()
    }

    /// Call until `None`. Ties break by the timer's own ordering, so a shared deadline is
    /// deterministic.
    pub fn pop_due(&mut self, now: Ts) -> Option<T>
    where
        T: Ord,
    {
        let key = self
            .armed
            .iter()
            .filter(|&(_, &at)| at <= now)
            .min_by(|(ka, a), (kb, b)| a.cmp(b).then_with(|| ka.cmp(kb)))
            .map(|(k, _)| *k)?;
        self.armed.remove(&key);
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Ordered as declared, which is what `pop_due`'s tie-break uses.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum T {
        EotPredictionTimeout,
        EndpointingDelay,
        UserAway,
    }

    fn ts(ms: u64) -> Ts {
        Ts::ZERO.saturating_add(Duration::from_millis(ms))
    }

    // Catches: a stale timer firing after the caller rescheduled it.
    #[test]
    fn arm_replaces_rather_than_accumulating() {
        let mut t = TimerSet::new();
        t.arm(T::EndpointingDelay, ts(100));
        let replaced = t.arm(T::EndpointingDelay, ts(500));

        assert_eq!(replaced, Some(ts(100)), "arm must report the deadline it moved");
        assert_eq!(t.len(), 1, "one timer, moved -- not two");
        assert_eq!(t.pop_due(ts(200)), None, "the replaced deadline must not fire");
        assert_eq!(t.pop_due(ts(500)), Some(T::EndpointingDelay));
    }

    // Catches: cleanup paths that cancel unconditionally blowing up on an unarmed timer.
    #[test]
    fn cancel_is_idempotent() {
        let mut t = TimerSet::<T>::new();
        assert_eq!(t.cancel(T::EndpointingDelay), None);
        t.arm(T::EndpointingDelay, ts(10));
        assert_eq!(t.cancel(T::EndpointingDelay), Some(ts(10)));
        assert_eq!(t.cancel(T::EndpointingDelay), None);
        assert!(t.is_empty());
    }


    // Catches: `is_armed` and `is_empty` answering a constant.
    #[test]
    fn is_armed_and_is_empty_track_the_set_rather_than_answering_a_constant() {
        let mut t = TimerSet::new();
        assert!(!t.is_armed(T::EndpointingDelay));
        assert!(t.is_empty());

        t.arm(T::EndpointingDelay, ts(100));
        assert!(t.is_armed(T::EndpointingDelay));
        assert!(!t.is_empty());
        assert!(!t.is_armed(T::EotPredictionTimeout), "a different timer is a different key");

        // Both ways out of "armed" have to clear it.
        assert_eq!(t.pop_due(ts(100)), Some(T::EndpointingDelay));
        assert!(!t.is_armed(T::EndpointingDelay), "firing disarms");
        assert!(t.is_empty());

        t.arm(T::EndpointingDelay, ts(100));
        t.cancel(T::EndpointingDelay);
        assert!(!t.is_armed(T::EndpointingDelay), "and so does cancelling");
    }

    // Catches: nondeterministic order when two timers share a deadline.
    #[test]
    fn ties_pop_in_a_stable_order() {
        let order = |()| {
            let mut t = TimerSet::new();
            t.arm(T::EndpointingDelay, ts(50));
            t.arm(T::EotPredictionTimeout, ts(50));
            t.arm(T::UserAway, ts(50));
            std::iter::from_fn(|| t.pop_due(ts(50))).collect::<Vec<_>>()
        };
        let first = order(());
        assert_eq!(first.len(), 3);
        for _ in 0..20 {
            assert_eq!(order(()), first, "pop order must not vary between runs");
        }
    }

    // Catches: `pop_due` returning an unfired timer, or `next_deadline` reporting a cancelled one.
    #[test]
    fn nothing_fires_early_and_cancelled_timers_vanish() {
        let mut t = TimerSet::new();
        t.arm(T::EndpointingDelay, ts(100));
        t.arm(T::EotPredictionTimeout, ts(300));

        assert_eq!(t.pop_due(ts(99)), None);
        assert_eq!(t.next_deadline(), Some(ts(100)));

        t.cancel(T::EndpointingDelay);
        assert_eq!(t.next_deadline(), Some(ts(300)));
        assert_eq!(t.pop_due(ts(299)), None);
        assert_eq!(t.pop_due(ts(300)), Some(T::EotPredictionTimeout));
        assert_eq!(t.next_deadline(), None);
    }
}
