//! Closeable resources that cannot leak: an [`Owned`] value's close always runs, explicitly or,
//! once dropped, as a bounded cleanup its owner's [`TaskSet`] joins.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use lk_agents_abi::{BoxFuture, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, StreamOutput, Ts};

use crate::task::TaskSet;

/// Where work that outlives its caller runs: the owner's task set, which joins it, on its
/// executor and clock. The one way to start such work.
#[derive(Clone)]
pub struct Background {
    pub tasks: Arc<TaskSet>,
    pub spawner: Arc<dyn Spawner>,
    pub clock: Arc<dyn Clock>,
}

impl Background {
    pub fn new(tasks: Arc<TaskSet>, spawner: Arc<dyn Spawner>, clock: Arc<dyn Clock>) -> Self {
        Background { tasks, spawner, clock }
    }

    /// The default bound of a close starting now.
    pub fn deadline(&self) -> Ts {
        self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)
    }

    /// Work the owner's set joins and cancels at its close; refused once the set closed.
    #[track_caller]
    pub fn spawn(&self, name: &str, work: impl Future<Output = ()> + Send + 'static) {
        let _ = self.tasks.try_spawn(self.spawner.as_ref(), name, work);
    }

    /// Runs `close`, bounded by its own deadline, to its end: admitted after the set closed and
    /// never cancelled by it.
    #[track_caller]
    pub fn cleanup<T: Send + 'static>(
        &self,
        name: &str,
        close: impl Future<Output = T> + Send + 'static,
    ) {
        self.tasks.cleanup(self.spawner.as_ref(), name, async move {
            drop(close.await);
        });
    }
}

/// An [`Owned`] value's admitted close: awaited, it resolves to its outcome; dropped while
/// pending, the same operation, with its original deadline, finishes as a cleanup of the owner's
/// set.
#[must_use = "a dropped Closing finishes in the background"]
pub struct Closing {
    future: Option<BoxFuture<'static, Closed>>,
    background: Background,
}

impl Future for Closing {
    type Output = Closed;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Closed> {
        let future = self.future.as_mut().expect("a Closing is polled until it resolves, once");
        let result = std::task::ready!(future.as_mut().poll(cx));
        self.future = None;
        Poll::Ready(result)
    }
}

impl Drop for Closing {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            self.background.cleanup("abandoned close", future);
        }
    }
}

/// How an [`Owned`] value is closed, bounded by the deadline.
pub type Closer<S> = Box<dyn FnOnce(S, Ts) -> BoxFuture<'static, Closed> + Send>;

/// A closeable resource and its owner's [`Background`]: `close` runs it explicitly, and dropping
/// it unclosed (a cancelled caller, an early return) runs the bounded close in the background.
/// Hold a resource in one of these as soon as it exists, before the next await.
pub struct Owned<S: Send + 'static> {
    held: Option<(S, Closer<S>)>,
    background: Background,
}

impl<S: ?Sized + Send + 'static> Owned<Box<S>> {
    /// A provider or host stream, closed by its own `close`.
    pub fn stream<T>(background: &Background, stream: Box<S>) -> Self
    where
        S: StreamOutput<T>,
    {
        Owned::new(background, stream, Box::new(|stream, deadline| stream.close(deadline)))
    }
}

impl<S: Send + 'static> Owned<S> {
    pub fn new(background: &Background, value: S, closer: Closer<S>) -> Self {
        Owned { held: Some((value, closer)), background: background.clone() }
    }

    /// Closes it by `deadline`; dropping the returned future leaves the close running.
    pub fn close_by(mut self, deadline: Ts) -> Closing {
        let (value, closer) = self.held.take().expect("an owned value is closed once");
        Closing { future: Some(closer(value, deadline)), background: self.background.clone() }
    }

    /// Closes it within the default bound.
    pub fn close(self) -> Closing {
        let deadline = self.background.deadline();
        self.close_by(deadline)
    }

    /// The value, now the caller's to close.
    pub fn into_inner(mut self) -> S {
        self.held.take().expect("an owned value is present until closed").0
    }
}

impl<S: Send + 'static> std::ops::Deref for Owned<S> {
    type Target = S;

    fn deref(&self) -> &S {
        &self.held.as_ref().expect("an owned value is present until closed").0
    }
}

impl<S: Send + 'static> std::ops::DerefMut for Owned<S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.held.as_mut().expect("an owned value is present until closed").0
    }
}

impl<S: Send + 'static> Drop for Owned<S> {
    fn drop(&mut self) {
        if let Some((value, closer)) = self.held.take() {
            let deadline = self.background.deadline();
            self.background.cleanup("dropped resource close", closer(value, deadline));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TokioSpawner;
    use parking_lot::Mutex;

    struct Probe(Arc<Mutex<Vec<&'static str>>>);

    impl StreamOutput<()> for Probe {
        fn next(
            &mut self,
        ) -> BoxFuture<'_, Result<Option<()>, lk_agents_abi::errors::ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async move {
                tokio::task::yield_now().await;
                self.0.lock().push("closed");
                Closed::Clean
            })
        }
    }

    fn background(tasks: &Arc<TaskSet>) -> Background {
        Background::new(
            Arc::clone(tasks),
            Arc::new(TokioSpawner::current()),
            Arc::new(crate::clock::RealClock::new()),
        )
    }

    // Catches: a stream dropped by a cancelled reader never reaching its close, or a close the
    // owner's set does not join.
    #[tokio::test]
    async fn a_dropped_owned_stream_is_closed_and_joined() {
        let (tasks, log) = (Arc::<TaskSet>::default(), Arc::new(Mutex::new(Vec::new())));
        let stream = Owned::stream(&background(&tasks), Box::new(Probe(log.clone())));
        let mut reader = Box::pin(async move {
            let mut stream = stream;
            let _ = stream.next().await;
        });
        let waker = std::task::Waker::noop();
        assert!(reader.as_mut().poll(&mut Context::from_waker(waker)).is_pending());
        drop(reader);
        assert!(!tasks.is_empty(), "the close is the set's until it ends");
        tasks.close();
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
        assert_eq!(*log.lock(), ["closed"], "closed once, even though the set was closed");
    }

    // Catches: an explicit close cut off when its awaiting caller is cancelled mid-close.
    #[tokio::test]
    async fn an_abandoned_explicit_close_still_finishes() {
        let (tasks, log) = (Arc::<TaskSet>::default(), Arc::new(Mutex::new(Vec::new())));
        let stream = Owned::stream(&background(&tasks), Box::new(Probe(log.clone())));
        let mut closing = stream.close();
        let waker = std::task::Waker::noop();
        assert!(Pin::new(&mut closing).poll(&mut Context::from_waker(waker)).is_pending());
        drop(closing);
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
        assert_eq!(*log.lock(), ["closed"]);
    }
}
