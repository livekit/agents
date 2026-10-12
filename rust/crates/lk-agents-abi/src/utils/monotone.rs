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
    use std::sync::Arc;
    use std::time::Duration;

    /// Real-time guard, so these tests fail rather than hang.
    const GUARD: Duration = Duration::from_secs(2);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Step {
        Pending,
        Scheduled,
        Done,
    }

    // Catches: an observer subscribing after the publish waiting forever (`oneshot`/`Notify`).
    #[tokio::test]
    async fn wait_for_sees_a_value_published_before_it_was_called() {
        let m = Monotone::new(Step::Pending);
        m.set(Step::Done);

        let got = tokio::time::timeout(GUARD, m.wait_for(|s| *s >= Step::Scheduled))
            .await
            .expect("a late observer must not block on an already-satisfied state");
        assert_eq!(got, Step::Done);
    }

    // Catches: waking only the first waiter (`SpeechHandle` is awaited from several places).
    #[tokio::test]
    async fn every_observer_is_woken_not_just_the_first() {
        let m = Arc::new(Monotone::new(Step::Pending));
        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let m = Arc::clone(&m);
                tokio::spawn(async move { m.wait_for(|s| *s == Step::Done).await })
            })
            .collect();

        // Let all four register first, so this is a genuine wake.
        tokio::task::yield_now().await;
        m.set(Step::Done);

        for w in waiters {
            let got = tokio::time::timeout(GUARD, w).await.expect("an observer was never woken");
            assert_eq!(got.unwrap(), Step::Done);
        }
    }

    // Catches: losing a transition when a `select!` branch is dropped and rebuilt.
    #[tokio::test]
    async fn a_dropped_and_retried_wait_for_still_observes_the_transition() {
        let m = Monotone::new(Step::Pending);

        // Poll once and abandon the future, the way `select!` does on every iteration.
        {
            let mut pending = Box::pin(m.wait_for(|s| *s == Step::Done));
            assert!(
                tokio::time::timeout(Duration::from_millis(5), &mut pending).await.is_err(),
                "the predicate is not satisfied yet"
            );
        }
        m.set(Step::Done);

        let got = tokio::time::timeout(GUARD, m.wait_for(|s| *s == Step::Done))
            .await
            .expect("the transition was lost when the first observer was dropped");
        assert_eq!(got, Step::Done);
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
