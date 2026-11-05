//! Shared access to a native stream: one read at a time, retained pending reads, and one owned
//! cleanup outcome. A cancelled waiter never loses the source or repeats an in-flight read.

use futures_util::{FutureExt, future::Shared};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamOutput, Ts};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::watch;

#[derive(Debug, PartialEq)]
pub enum Pulled<T> {
    Item(T),
    End,
    Cancelled,
}

pub use lk_agents_abi::errors::PullError;

type Pending<T> = Shared<BoxFuture<'static, Arc<Mutex<Read<T>>>>>;
type Cleanup = Shared<BoxFuture<'static, Closed>>;

struct Read<T> {
    source: Option<Box<dyn StreamOutput<T>>>,
    item: Option<Result<Option<T>, ProviderError>>,
}

enum State<T> {
    Ready(Box<dyn StreamOutput<T>>),
    Reading(Pending<T>),
    Closing(Cleanup),
}

struct Inner<T> {
    state: Mutex<Option<State<T>>>,
    reading: AtomicBool,
    closing: watch::Sender<bool>,
    clock: Arc<dyn Clock>,
}

/// The binding's shared access to one stream. This is lifecycle machinery, not another stream
/// contract; language adapters use it for read admission, cancellation and close settlement.
pub struct SharedStream<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for SharedStream<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

struct Reading<'a>(&'a AtomicBool);
impl Drop for Reading<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<T: Send + 'static> SharedStream<T> {
    pub fn new(source: Box<dyn StreamOutput<T>>, clock: Arc<dyn Clock>) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(Some(State::Ready(source))),
                reading: AtomicBool::new(false),
                closing: watch::channel(false).0,
                clock,
            }),
        }
    }

    pub fn is_closed(&self) -> bool {
        *self.inner.closing.borrow()
    }

    pub async fn pull(&self) -> Result<Pulled<T>, PullError> {
        if self.is_closed() {
            return Ok(Pulled::End);
        }
        self.inner
            .reading
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| PullError::Overlap)?;
        let _reading = Reading(&self.inner.reading);
        let Some(pending) = self.inner.begin_read() else {
            return Ok(Pulled::End);
        };
        // No state lock is held while polling the source, which may enter a language runtime.
        let completed = pending.await;
        self.inner.finish_read(&completed)
    }

    pub async fn aclose(&self) -> Closed {
        self.close(self.inner.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await
    }

    /// The first close owns the deadline. Every caller waits for the same cleanup outcome;
    /// abandoning a waiter leaves the pending work available to the next waiter.
    pub async fn close(&self, deadline: Ts) -> Closed {
        self.inner.closing.send_replace(true);
        let cleanup = self.inner.begin_close(deadline);
        cleanup.await
    }
}

impl<T: Send + 'static> Inner<T> {
    /// Joins the read in flight or starts one; `None` once the stream is closing.
    fn begin_read(&self) -> Option<Pending<T>> {
        let mut state = self.state.lock();
        let pending = match state.take().expect("a stream always owns its state") {
            State::Ready(source) => read_next(source, self.closing.subscribe()),
            State::Reading(pending) => pending,
            State::Closing(cleanup) => {
                *state = Some(State::Closing(cleanup));
                return None;
            }
        };
        *state = Some(State::Reading(pending.clone()));
        Some(pending)
    }

    /// Hands the source back for the next read, unless a close won while it was read.
    fn finish_read(&self, completed: &Mutex<Read<T>>) -> Result<Pulled<T>, PullError> {
        let (item, previous) = {
            let mut state = self.state.lock();
            if *self.closing.borrow() {
                return Ok(Pulled::Cancelled);
            }
            let mut completed = completed.lock();
            let source = completed.source.take().expect("one reader owns the completed source");
            let item = completed.item.take();
            (item, state.replace(State::Ready(source)))
        };
        // A completed future may retain host-backed values. Drop it after the state guards.
        drop(previous);
        match item {
            Some(Ok(Some(item))) => Ok(Pulled::Item(item)),
            Some(Ok(None)) => {
                self.closing.send_replace(true);
                Ok(Pulled::End)
            }
            Some(Err(error)) => {
                self.closing.send_replace(true);
                Err(error.into())
            }
            None => Ok(Pulled::Cancelled),
        }
    }

    /// Installs the one cleanup every closer awaits; the first close owns its deadline.
    fn begin_close(&self, deadline: Ts) -> Cleanup {
        let mut state = self.state.lock();
        let cleanup = match state.take().expect("a stream always owns its state") {
            State::Closing(cleanup) => cleanup,
            previous => close_source(previous, self.clock.clone(), deadline).boxed().shared(),
        };
        *state = Some(State::Closing(cleanup.clone()));
        cleanup
    }
}

/// One shared read of the source's next item; a close wins over an item ready at once.
fn read_next<T: Send + 'static>(
    mut source: Box<dyn StreamOutput<T>>,
    mut closing: watch::Receiver<bool>,
) -> Pending<T> {
    async move {
        let item = tokio::select! {
            biased;
            _ = closing.wait_for(|closed| *closed) => None,
            item = source.next() => Some(item),
        };
        Arc::new(Mutex::new(Read { source: Some(source), item }))
    }
    .boxed()
    .shared()
}

/// Closes the source once any read in flight hands it back, within the deadline.
async fn close_source<T>(previous: State<T>, clock: Arc<dyn Clock>, deadline: Ts) -> Closed {
    let mut outcome = Closed::Clean;
    let bounded = Closed::bounded(&*clock, deadline, async {
        let source = match previous {
            State::Ready(source) => Some(source),
            State::Reading(pending) => pending.await.lock().source.take(),
            State::Closing(_) => unreachable!("cleanup already installed"),
        };
        if let Some(source) = source {
            outcome = source.close(deadline).await;
        }
    })
    .await;
    outcome.and(bounded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Yields its items, then waits forever; counts closes.
    struct Scripted {
        items: Vec<u32>,
        closes: Arc<AtomicUsize>,
    }

    impl StreamOutput<u32> for Scripted {
        fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<u32>, ProviderError>> {
            Box::pin(async move {
                if self.items.is_empty() {
                    std::future::pending::<()>().await;
                }
                Ok(Some(self.items.remove(0)))
            })
        }

        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move {
                self.closes.fetch_add(1, Ordering::SeqCst);
                Closed::Clean
            })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<u32> for Scripted {}

    fn source(items: Vec<u32>) -> (SharedStream<u32>, Arc<AtomicUsize>) {
        let closes = Arc::new(AtomicUsize::new(0));
        (
            SharedStream::new(
                Box::new(Scripted { items, closes: closes.clone() }),
                Arc::new(crate::clock::RealClock::new()),
            ),
            closes,
        )
    }

    // Catches: an overlapping pull queued behind the first (or taking its item) instead of
    // refused at once.
    #[tokio::test]
    async fn an_overlapping_pull_is_refused_while_the_first_waits() {
        let (source, _) = source(vec![]);
        let blocked = tokio::spawn({
            let source = source.clone();
            async move { source.pull().await }
        });
        tokio::task::yield_now().await;
        assert!(matches!(source.pull().await, Err(PullError::Overlap)));
        blocked.abort();
        let _ = blocked.await;
        let (source, _) = self::source(vec![1]);
        assert!(matches!(source.pull().await, Ok(Pulled::Item(1))));
    }

    // Catches: a close that waits behind the blocked pull it must end, the blocked pull reported
    // as a clean end, or the source closed twice.
    #[tokio::test]
    async fn close_settles_a_blocked_pull_and_closes_the_source_once() {
        let (source, closes) = source(vec![]);
        let blocked = tokio::spawn({
            let source = source.clone();
            async move { source.pull().await }
        });
        tokio::task::yield_now().await;
        assert_eq!(source.aclose().await, Closed::Clean);
        assert!(matches!(blocked.await.unwrap(), Ok(Pulled::Cancelled)));
        assert_eq!(source.aclose().await, Closed::Clean);
        assert!(matches!(source.pull().await, Ok(Pulled::End)));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }
    struct Closing {
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl StreamOutput<()> for Closing {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<()>, ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async move {
                self.entered.notify_one();
                self.release.notified().await;
                Closed::Failed
            })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<()> for Closing {}

    #[tokio::test]
    async fn concurrent_closes_wait_for_and_preserve_the_same_failure() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let source = SharedStream::new(
            Box::new(Closing { entered: entered.clone(), release: release.clone() }),
            Arc::new(crate::clock::RealClock::new()),
        );
        let first = tokio::spawn({
            let source = source.clone();
            async move { source.aclose().await }
        });
        entered.notified().await;
        let second = tokio::spawn({
            let source = source.clone();
            async move { source.aclose().await }
        });
        tokio::task::yield_now().await;
        assert!(!second.is_finished(), "close must wait for cleanup already in progress");
        release.notify_one();
        assert_eq!(first.await.unwrap(), Closed::Failed);
        assert_eq!(second.await.unwrap(), Closed::Failed);
        assert_eq!(source.aclose().await, Closed::Failed);
    }

    #[tokio::test]
    async fn abandoned_close_can_resume_and_a_wedged_cleanup_has_a_shared_deadline() {
        let clock = Arc::new(lk_agents_abi::utils::clock::VirtualClock::new());
        let entered = Arc::new(tokio::sync::Notify::new());
        let source = SharedStream::new(
            Box::new(Closing {
                entered: entered.clone(),
                release: Arc::new(tokio::sync::Notify::new()),
            }),
            clock.clone(),
        );
        let first = tokio::spawn({
            let source = source.clone();
            async move { source.aclose().await }
        });
        entered.notified().await;
        first.abort();
        let _ = first.await;
        clock.advance(HOST_TEARDOWN_TIMEOUT);
        assert_eq!(source.aclose().await, Closed::TimedOut);
        assert_eq!(source.aclose().await, Closed::TimedOut);
    }
    struct Gated {
        reads: Arc<AtomicUsize>,
        entered: Arc<tokio::sync::Notify>,
        released: watch::Receiver<bool>,
    }
    impl StreamOutput<u32> for Gated {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<u32>, ProviderError>> {
            Box::pin(async move {
                self.reads.fetch_add(1, Ordering::SeqCst);
                self.entered.notify_one();
                let _ = self.released.wait_for(|released| *released).await;
                Ok(Some(42))
            })
        }
        fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async { Closed::Clean })
        }
    }

    #[tokio::test]
    async fn abandoning_a_read_resumes_the_same_pending_pull() {
        let reads = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        let (release, released) = watch::channel(false);
        let source = SharedStream::new(
            Box::new(Gated { reads: reads.clone(), entered: entered.clone(), released }),
            Arc::new(crate::clock::RealClock::new()),
        );
        let first = tokio::spawn({
            let source = source.clone();
            async move { source.pull().await }
        });
        entered.notified().await;
        first.abort();
        let _ = first.await;
        release.send_replace(true);
        assert_eq!(source.pull().await.unwrap(), Pulled::Item(42));
        assert_eq!(reads.load(Ordering::SeqCst), 1, "a cancelled waiter must not repeat next()");
        assert_eq!(source.aclose().await, Closed::Clean);
    }
}
