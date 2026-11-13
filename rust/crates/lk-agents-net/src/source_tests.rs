#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
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

/// Its read blocks its poll for a while, then waits forever: a close lands mid-poll.
struct SlowPoll {
    polling: Arc<tokio::sync::Notify>,
}

impl StreamOutput<u32> for SlowPoll {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<u32>, ProviderError>> {
        let polling = self.polling.clone();
        Box::pin(std::future::poll_fn(move |_| {
            polling.notify_one();
            std::thread::sleep(std::time::Duration::from_millis(50));
            std::task::Poll::Pending
        }))
    }

    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<u32> for SlowPoll {}

// Catches: a close whose cancel woke the read while a reader that was going away polled it, lost
// by the shared read (`futures::Shared` drops a wake that lands mid-poll), so the close waited out the
// whole teardown deadline and left the source open: a `tts.stream()` interrupted as it started.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_close_landing_while_the_read_is_polled_is_not_lost() {
    let polling = Arc::new(tokio::sync::Notify::new());
    let source = SharedStream::new(
        Box::new(SlowPoll { polling: polling.clone() }),
        Arc::new(crate::clock::RealClock::new()),
    );
    let reader = tokio::spawn({
        let source = source.clone();
        async move { source.try_pull().unwrap().await }
    });
    polling.notified().await;
    // the reader is cancelled mid-poll (a host cancelled its pull): only the close is left
    reader.abort();
    // a timer would wake the close by chance: it must not need one
    let started = std::time::Instant::now();
    let closed = tokio::time::timeout(std::time::Duration::from_secs(3), source.aclose()).await;
    let took = started.elapsed();
    assert_eq!(closed, Ok(Closed::Clean));
    assert!(
        took < std::time::Duration::from_millis(500),
        "the close waited {took:?} on a read that never saw it"
    );
    assert!(reader.await.unwrap_err().is_cancelled());
}

/// Yields its items, then fails with a 401.
struct Failing(Vec<u32>);

impl StreamOutput<u32> for Failing {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<u32>, ProviderError>> {
        Box::pin(async move {
            match self.0.is_empty() {
                true => Err(refused()),
                false => Ok(Some(self.0.remove(0))),
            }
        })
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<u32> for Failing {}

fn refused() -> ProviderError {
    lk_agents_abi::errors::ApiError::status("refused", 401, None).into()
}

// Catches: input after a provider's failure answered with a generic close (or accepted) instead
// of that failure, an end of input masking a later failure, or a clean end taken for a failure.
#[tokio::test]
async fn the_input_answers_with_the_failure_the_output_read() {
    let gate = InputGate::default();
    let mut output = super::Gated::new(Failing(vec![1]), gate.clone());
    assert_eq!(gate.admit(), Ok(()));
    gate.end();
    let ended = gate.admit().expect_err("an ended input refuses");
    assert_eq!(output.next().await, Ok(Some(1)));
    assert_eq!(output.next().await, Err(refused()));
    assert_eq!(gate.admit(), Err(refused()), "the failure, not {ended}");
    assert_eq!(gate.refused(), refused());
    Box::new(output).close(Ts::ZERO).await;
    assert_eq!(gate.admit(), Err(refused()), "a close keeps the failure");

    let shared =
        SharedStream::new(Box::new(Failing(vec![])), Arc::new(crate::clock::RealClock::new()));
    assert_eq!(shared.input_failure(), None);
    assert_eq!(shared.try_pull().unwrap().await, Err(refused()));
    assert_eq!((shared.admit_input(), shared.input_failure()), (Err(refused()), Some(refused())));

    let (shared, _) = source(vec![]);
    shared.aclose().await;
    assert_eq!(shared.admit_input(), Err(ended));
    assert_eq!(shared.input_failure(), None, "a close is no failure");
}
