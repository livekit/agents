//! A level-triggered cell whose value only moves forward.

use tokio::sync::watch;

/// Level-triggered, so a late observer never hangs. Values that legitimately go backwards
/// (`authorized`, `allow_interruptions`) must not use this.
pub struct Monotone<T: Clone> {
    tx: watch::Sender<T>,
}

impl<T: Clone> Monotone<T> {
    pub fn new(initial: T) -> Self {
        Monotone { tx: watch::channel(initial).0 }
    }

    pub fn get(&self) -> T {
        self.tx.borrow().clone()
    }

    /// Checks the current value first; cancel-safe (usable as a `select!` branch).
    pub async fn wait_for(&self, pred: impl Fn(&T) -> bool) -> T {
        let mut rx = self.tx.subscribe();
        // `wait_for` only errors when every sender is gone, and `&self` *is* the sender.
        let v = rx.wait_for(pred).await.expect("Monotone owns its sender for the whole call");
        v.clone()
    }

    /// Debug-panics if `v` is strictly less than the current value. Incomparable values pass.
    pub fn set(&self, v: T)
    where
        T: PartialOrd,
    {
        self.tx.send_modify(|cur| {
            debug_assert!(
                v.partial_cmp(cur) != Some(std::cmp::Ordering::Less),
                "Monotone<{}>::set would regress; only level-triggered watch cells may go \
                 backwards (§2.4: `authorized`, `allow_interruptions`)",
                std::any::type_name::<T>()
            );
            *cur = v;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Step {
        Pending,
        Scheduled,
        Done,
    }

    // Catches: routing a value that goes backwards (`_clear_authorization()`,
    // `speech_handle.py:303`) through `Monotone`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "would regress")]
    fn a_regressing_set_is_a_debug_panic() {
        let m = Monotone::new(Step::Pending);
        m.set(Step::Done);
        m.set(Step::Scheduled);
    }
}
