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
        async move { source.try_pull().unwrap().await }
    });
    tokio::task::yield_now().await;
    assert!(source.try_pull().is_none());
    blocked.abort();
    let _ = blocked.await;
    let (source, _) = self::source(vec![1]);
    assert!(matches!(source.try_pull().unwrap().await, Ok(Pulled::Item(1))));
}

// Catches: a close that waits behind the blocked pull it must end, the blocked pull reported
// as a clean end, or the source closed twice.
#[tokio::test]
async fn close_settles_a_blocked_pull_and_closes_the_source_once() {
    let (source, closes) = source(vec![]);
    let blocked = tokio::spawn({
        let source = source.clone();
        async move { source.try_pull().unwrap().await }
    });
    tokio::task::yield_now().await;
    assert_eq!(source.aclose().await, Closed::Clean);
    assert!(matches!(blocked.await.unwrap(), Ok(Pulled::Cancelled)));
    assert_eq!(source.aclose().await, Closed::Clean);
    assert!(matches!(source.try_pull().unwrap().await, Ok(Pulled::End)));
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
    released: CancellationToken,
}
impl StreamOutput<u32> for Gated {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<u32>, ProviderError>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.entered.notify_one();
            self.released.cancelled().await;
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
    let released = CancellationToken::new();
    let source = SharedStream::new(
        Box::new(Gated {
            reads: reads.clone(),
            entered: entered.clone(),
            released: released.clone(),
        }),
        Arc::new(crate::clock::RealClock::new()),
    );
    let first = tokio::spawn({
        let source = source.clone();
        async move { source.try_pull().unwrap().await }
    });
    entered.notified().await;
    first.abort();
    let _ = first.await;
    released.cancel();
    assert_eq!(source.try_pull().unwrap().await.unwrap(), Pulled::Item(42));
    assert_eq!(reads.load(Ordering::SeqCst), 1, "a cancelled waiter must not repeat next()");
    assert_eq!(source.aclose().await, Closed::Clean);
}
