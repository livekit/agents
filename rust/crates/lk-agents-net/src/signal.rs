//! The one way to wait on state another party produces. A [`Producer`] is the only writer and is
//! not `Clone`; dropping it, however its owner ends (a return, a cancel, a panic, a runtime
//! drop), ends the signal, and every [`Cell`] waiting on it resolves: `Ok` when the final value
//! satisfies the wait, else `Err(Ended)`. So a waiter that is never woken cannot be written.
//!
//! Hold the producer where the progress happens: in the task that makes it (moved into its
//! future, see [`TaskSet::spawn_settling`](crate::task::TaskSet::spawn_settling)) or in the
//! table of the loop that owns the state, never in the waiter's own object.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the one module built on tokio's watch channel"
)]

use std::future::Future;
use std::panic::Location;

use tokio::sync::watch;

/// The producer is gone and the awaited state can no longer come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the producer ended before the awaited state came")]
pub struct Ended;

/// A signal holding `init`: its only writer and a reader to clone for every waiter.
#[track_caller]
pub fn signal<T: Send + Sync>(init: T) -> (Producer<T>, Cell<T>) {
    let (tx, rx) = watch::channel(init);
    let origin = Location::caller();
    (Producer { tx, origin }, Cell { rx, origin })
}

/// A value set once: [`Cell::wait`] resolves with it, or `Err(Ended)` if the producer dropped
/// first.
#[track_caller]
pub fn once<T: Send + Sync>() -> (Producer<Option<T>>, Cell<Option<T>>) {
    signal(None)
}

/// The only writer of a signal. Every write wakes every waiter; dropping it ends the signal.
pub struct Producer<T> {
    tx: watch::Sender<T>,
    origin: &'static Location<'static>,
}

impl<T> Producer<T> {
    pub fn set(&self, value: T) {
        self.tx.send_replace(value);
    }

    /// Changes the value in place; always wakes the waiters, who re-check their condition.
    pub fn modify<R>(&self, change: impl FnOnce(&mut T) -> R) -> R {
        let mut out = None;
        self.tx.send_modify(|value| out = Some(change(value)));
        out.expect("send_modify runs its closure")
    }

    pub fn with<R>(&self, read: impl FnOnce(&T) -> R) -> R {
        read(&self.tx.borrow())
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.tx.borrow().clone()
    }

    /// Another reader.
    pub fn cell(&self) -> Cell<T> {
        Cell { rx: self.tx.subscribe(), origin: self.origin }
    }

    /// Sets the terminal value and ends the signal.
    pub fn finish(self, terminal: T) {
        self.set(terminal);
    }
}

impl<T: PartialOrd> Producer<T> {
    /// Sets a value that only moves forward; debug-panics if `value` is strictly less than the
    /// current one (incomparable values pass).
    pub fn advance(&self, value: T) {
        self.tx.send_modify(|current| {
            debug_assert!(
                value.partial_cmp(current) != Some(std::cmp::Ordering::Less),
                "Producer<{}>::advance would regress",
                std::any::type_name::<T>()
            );
            *current = value;
        });
    }
}

impl<T> Producer<Option<T>> {
    /// Sets the value once and ends the signal.
    pub fn resolve(self, value: T) {
        self.finish(Some(value));
    }
}

impl<T> std::fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Producer").field("origin", &self.origin).finish_non_exhaustive()
    }
}

/// A reader of a signal, for any number of waiters.
pub struct Cell<T> {
    rx: watch::Receiver<T>,
    origin: &'static Location<'static>,
}

impl<T> Clone for Cell<T> {
    fn clone(&self) -> Self {
        Cell { rx: self.rx.clone(), origin: self.origin }
    }
}

impl<T> std::fmt::Debug for Cell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cell").field("origin", &self.origin).finish_non_exhaustive()
    }
}

impl<T> Cell<T> {
    /// A cell whose producer is already gone: it holds `value` forever.
    #[track_caller]
    pub fn ended_with(value: T) -> Self
    where
        T: Send + Sync,
    {
        signal(value).1
    }

    pub fn with<R>(&self, read: impl FnOnce(&T) -> R) -> R {
        read(&self.rx.borrow())
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.rx.borrow().clone()
    }

    /// The producer is gone: the value is final.
    pub fn ended(&self) -> bool {
        self.rx.has_changed().is_err()
    }

    /// What `map` takes from the first value it accepts, checking the current value first;
    /// `Err(Ended)` once the producer is gone and the final value is refused. Cancel-safe.
    #[track_caller]
    pub fn wait_map<'a, R>(
        &'a self,
        mut map: impl FnMut(&T) -> Option<R> + Send + 'a,
    ) -> impl Future<Output = Result<R, Ended>> + Send + 'a
    where
        T: Send + Sync,
        R: Send,
    {
        let mut rx = self.rx.clone();
        let parked = park::Parked::new(self.origin, Location::caller());
        async move {
            let _parked = parked;
            let mut found = None;
            rx.wait_for(|value| {
                found = map(value);
                found.is_some()
            })
            .await
            .map_err(|_| Ended)?;
            Ok(found.expect("the accepted value was mapped"))
        }
    }

    /// The first value `ready` accepts; see [`wait_map`](Self::wait_map).
    #[track_caller]
    pub fn wait_for<'a>(
        &'a self,
        ready: impl Fn(&T) -> bool + Send + 'a,
    ) -> impl Future<Output = Result<T, Ended>> + Send + 'a
    where
        T: Clone + Send + Sync,
    {
        self.wait_map(move |value| ready(value).then(|| value.clone()))
    }

    /// The next write after the last one this cell saw; `Err(Ended)` once the producer is gone.
    /// Cancel-safe.
    pub async fn changed(&mut self) -> Result<(), Ended>
    where
        T: Send + Sync,
    {
        self.rx.changed().await.map_err(|_| Ended)
    }
}

impl<T: Clone + Send + Sync> Cell<Option<T>> {
    /// The value once set; `Err(Ended)` if the producer dropped without setting it.
    #[track_caller]
    pub fn wait(&self) -> impl Future<Output = Result<T, Ended>> + Send + '_ {
        self.wait_map(Option::clone)
    }
}

/// State several parties change under a lock, with waiters that re-check it after every change.
/// A waiter borrows it, so it cannot end under them: give the state a terminal value (a close)
/// that every wait accepts.
pub struct Shared<S> {
    state: parking_lot::Mutex<S>,
    changed: Producer<()>,
}

impl<S> Shared<S> {
    #[track_caller]
    pub fn new(state: S) -> Self {
        Shared { state: parking_lot::Mutex::new(state), changed: signal(()).0 }
    }

    /// Reads (or changes without waking) the state.
    pub fn with<R>(&self, read: impl FnOnce(&mut S) -> R) -> R {
        read(&mut self.state.lock())
    }

    /// Changes the state and wakes every waiter.
    pub fn modify<R>(&self, change: impl FnOnce(&mut S) -> R) -> R {
        let out = change(&mut self.state.lock());
        self.changed.modify(|()| ());
        out
    }

    /// What `take` finds, checking now and after every change; cancel-safe. A take may change
    /// the state (pop an item), so finding something wakes the other waiters.
    pub async fn wait_until<R>(&self, mut take: impl FnMut(&mut S) -> Option<R>) -> R {
        // Subscribed before the check, so a change after it wakes the wait.
        let mut changed = self.changed.cell();
        loop {
            let found = take(&mut self.state.lock());
            if let Some(found) = found {
                self.changed.modify(|()| ());
                return found;
            }
            // `self` holds the producer for the whole wait.
            let _ = changed.changed().await;
        }
    }
}

/// Under the `dst` feature, the waiters parked on a signal, labelled by where their producer and
/// their wait were created, so a simulation can assert none is parked once everything is quiet.
#[cfg(feature = "dst")]
pub mod park {
    use std::collections::BTreeMap;
    use std::panic::Location;
    use std::sync::Arc;

    use parking_lot::Mutex;

    /// A waiter parked on a signal.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Waiter {
        /// Where the signal (its producer) was created.
        pub producer: &'static Location<'static>,
        /// Where the wait was started.
        pub waiter: &'static Location<'static>,
    }

    #[derive(Default)]
    struct Registry {
        next: u64,
        parked: BTreeMap<u64, Waiter>,
    }

    thread_local! {
        static REGISTRY: Arc<Mutex<Registry>> = Arc::default();
    }

    /// The waiters registered on this thread (a simulation's executor) that are still pending.
    pub fn parked() -> Vec<Waiter> {
        REGISTRY.with(|registry| registry.lock().parked.values().copied().collect())
    }

    /// Registered while its wait is alive; a wait that resolved or was dropped leaves.
    pub(super) struct Parked {
        registry: Arc<Mutex<Registry>>,
        id: u64,
    }

    impl Parked {
        pub(super) fn new(
            producer: &'static Location<'static>,
            waiter: &'static Location<'static>,
        ) -> Self {
            let registry = REGISTRY.with(Arc::clone);
            let id = {
                let mut r = registry.lock();
                r.next += 1;
                let id = r.next;
                r.parked.insert(id, Waiter { producer, waiter });
                id
            };
            Parked { registry, id }
        }
    }

    impl Drop for Parked {
        fn drop(&mut self) {
            self.registry.lock().parked.remove(&self.id);
        }
    }
}

#[cfg(not(feature = "dst"))]
mod park {
    use std::panic::Location;

    pub(super) struct Parked;

    impl Parked {
        pub(super) fn new(_: &'static Location<'static>, _: &'static Location<'static>) -> Self {
            Parked
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn poll<F: Future>(fut: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        fut.poll(&mut Context::from_waker(Waker::noop()))
    }

    // Catches: a waiter left parked when its producer is dropped without setting the value.
    #[test]
    fn dropping_the_producer_resolves_every_waiter_within_one_poll() {
        let (producer, cell) = signal(0u32);
        let other = cell.clone();
        let mut a = pin!(cell.wait_for(|v| *v == 2));
        let mut b = pin!(other.wait_for(|v| *v >= 1));
        assert!(poll(a.as_mut()).is_pending() && poll(b.as_mut()).is_pending());
        producer.set(1);
        drop(producer);
        assert_eq!(poll(a.as_mut()), Poll::Ready(Err(Ended)));
        assert_eq!(poll(b.as_mut()), Poll::Ready(Ok(1)), "the final value still counts");
        assert!(cell.ended());
    }

    #[test]
    fn a_once_resolves_or_ends() {
        let (producer, cell) = once::<&str>();
        let mut wait = pin!(cell.wait());
        assert!(poll(wait.as_mut()).is_pending());
        producer.resolve("done");
        assert_eq!(poll(wait.as_mut()), Poll::Ready(Ok("done")));
        let (producer, cell) = once::<&str>();
        drop(producer);
        assert_eq!(poll(pin!(cell.wait())), Poll::Ready(Err(Ended)));
    }

    // Catches: a producer moved into a task that panics staying alive and its waiters hanging.
    #[tokio::test]
    async fn a_panic_in_a_settling_task_resolves_its_waiters() {
        use crate::task::{TaskSet, TokioSpawner};
        let tasks = std::sync::Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        let (producer, cell) = once::<u32>();
        tasks.spawn_settling(&spawner, "panics", producer, async { panic!("boom") });
        let waited = tokio::time::timeout(std::time::Duration::from_secs(5), cell.wait()).await;
        assert_eq!(waited, Ok(Err(Ended)));
        let (producer, cell) = once::<u32>();
        tasks.spawn_settling(&spawner, "settles", producer, async { 7 });
        assert_eq!(cell.wait().await, Ok(7));
    }

    // Catches: a value that goes backwards routed through `advance`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "would regress")]
    fn a_regressing_advance_is_a_debug_panic() {
        let (producer, _cell) = signal(2u8);
        producer.advance(1);
    }

    #[cfg(feature = "dst")]
    #[test]
    fn a_parked_waiter_is_listed_until_it_resolves() {
        let (producer, cell) = signal(false);
        let mut wait = Box::pin(cell.wait_for(|v| *v));
        assert!(poll(wait.as_mut()).is_pending());
        assert_eq!(park::parked().len(), 1);
        producer.set(true);
        assert_eq!(poll(wait.as_mut()), Poll::Ready(Ok(true)));
        drop(wait);
        assert!(park::parked().is_empty());
    }
}
