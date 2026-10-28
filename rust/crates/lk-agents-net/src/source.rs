//! A native source as a host consumes it: one pull at a time (an overlapping pull is refused
//! before it touches the source), and a close that stops admission, settles the pull in flight
//! without waiting behind it, then closes the source under its deadline.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_abi::{Closed, StreamSource, Ts};
use tokio::sync::{Mutex, Notify};

#[derive(Debug, PartialEq)]
pub enum Pulled<T> {
    Item(T),
    End,
    /// The source was closed while this pull waited.
    Cancelled,
}

pub use lk_agents_abi::errors::PullError;

struct Inner<T> {
    source: Mutex<Option<Box<dyn StreamSource<T>>>>,
    closed: AtomicBool,
    closing: Notify,
}

/// Cheap to clone: every clone is the same source.
pub struct NativeSource<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for NativeSource<T> {
    fn clone(&self) -> Self {
        NativeSource { inner: self.inner.clone() }
    }
}

impl<T: Send + 'static> NativeSource<T> {
    pub fn new(source: Box<dyn StreamSource<T>>) -> Self {
        NativeSource {
            inner: Arc::new(Inner {
                source: Mutex::new(Some(source)),
                closed: AtomicBool::new(false),
                closing: Notify::new(),
            }),
        }
    }

    pub async fn pull(&self) -> Result<Pulled<T>, PullError> {
        let mut slot = self.inner.source.try_lock().map_err(|_| PullError::Overlap)?;
        let closing = self.inner.closing.notified();
        if self.inner.closed.load(Ordering::SeqCst) {
            return Ok(Pulled::End);
        }
        let Some(source) = slot.as_mut() else { return Ok(Pulled::End) };
        tokio::select! {
            biased;
            () = closing => Ok(Pulled::Cancelled),
            pulled = source.next() => Ok(match pulled? {
                Some(item) => Pulled::Item(item),
                None => Pulled::End,
            }),
        }
    }

    /// Idempotent: a second close finds nothing left to release.
    pub async fn close(&self, deadline: Ts) -> Closed {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.closing.notify_waiters();
        let source = self.inner.source.lock().await.take();
        match source {
            Some(source) => source.close(deadline).await,
            None => Closed::Clean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::ProviderError;
    use std::sync::atomic::AtomicUsize;

    /// Yields its items, then waits forever; counts closes.
    struct Scripted {
        items: Vec<u32>,
        closes: Arc<AtomicUsize>,
    }

    impl StreamSource<u32> for Scripted {
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

    fn source(items: Vec<u32>) -> (NativeSource<u32>, Arc<AtomicUsize>) {
        let closes = Arc::new(AtomicUsize::new(0));
        (NativeSource::new(Box::new(Scripted { items, closes: closes.clone() })), closes)
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
        assert_eq!(source.close(Ts::from_nanos(0)).await, Closed::Clean);
        assert!(matches!(blocked.await.unwrap(), Ok(Pulled::Cancelled)));
        assert_eq!(source.close(Ts::from_nanos(0)).await, Closed::Clean);
        assert!(matches!(source.pull().await, Ok(Pulled::End)));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }
}
