//! The duplex split. Never fuse the halves: dropping a Rust future does not cancel the host's
//! `__anext__`, which stays parked in a `tee` generator and poisons the stream (`tts/tts.py:543`).

use std::future::Future;
use std::time::Duration;

use async_trait::async_trait;
use crate::utils::clock::{Clock, Elapsed, Ts, timeout};
use crate::errors::ProviderError;

/// `&self` and `Sync`, so the pump holds an `Arc` while the drain runs elsewhere.
pub trait StreamSink<T>: Send + Sync {
    /// `Ok(())` means "handed over", not "accepted": a failure inside the loop surfaces on the
    /// [`StreamSource`].
    fn push(&self, item: T) -> Result<(), ProviderError>;

    /// End of the current segment (`_FlushSentinel`), not a close.
    fn flush(&self) -> Result<(), ProviderError>;

    /// `flush()` plus `close()`. A gone loop is reported by the source.
    fn end_input(&self);
}

/// `next` takes `&mut self`, so there is exactly one drain.
#[async_trait]
pub trait StreamSource<T>: Send {
    /// `Ok(None)` is a clean end. Kept apart: `__anext__` re-raises the task's exception on
    /// exhaustion.
    async fn next(&mut self) -> Result<Option<T>, ProviderError>;

    /// Whether a `next()` future may be dropped mid-poll. `false` for a host stream (interruption
    /// waits for the chunk).
    fn cancel_safe(&self) -> bool {
        false
    }

    /// The only teardown: dropping does not cancel the host task. Bounded by `deadline` on the
    /// session clock.
    async fn close(self: Box<Self>, deadline: Ts) -> Closed;

    /// A synthesis stream's `_started_time` (`tts/tts.py`): the `perf_counter()` at which text was
    /// first handed to the provider, once it has been -- even by an attempt that then failed
    /// without audio. `None` when unknown, and for every other kind of stream.
    fn started_time(&self) -> Option<f64> {
        None
    }
}

/// The reference's `INTERRUPTION_TIMEOUT`. Real teardowns take under a millisecond.
pub const HOST_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    Clean,
    /// Abandoned at the deadline; a host generator is still running.
    TimedOut,
}

impl Closed {
    /// On timeout the future is dropped, leaking one host task rather than hanging the pump. A past
    /// deadline still gives it one poll.
    pub async fn bounded<F: Future<Output = ()>>(clock: &dyn Clock, deadline: Ts, teardown: F) -> Closed {
        let limit = deadline.saturating_duration_since(clock.now());
        match timeout(clock, limit, teardown).await {
            Ok(()) => Closed::Clean,
            Err(Elapsed) => Closed::TimedOut,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::clock::VirtualClock;
    use std::sync::Arc;

    /// A host-shaped source whose teardown is a future the host controls.
    struct Host {
        clock: Arc<VirtualClock>,
        wedged: bool,
    }

    #[async_trait]
    impl StreamSource<()> for Host {
        async fn next(&mut self) -> Result<Option<()>, ProviderError> {
            Ok(None)
        }
        async fn close(self: Box<Self>, deadline: Ts) -> Closed {
            let wedged = self.wedged;
            Closed::bounded(&*self.clock, deadline, async move {
                if wedged {
                    std::future::pending::<()>().await;
                }
            })
            .await
        }
    }

    // Catches: a wedged host teardown hanging the pump, or a bound on tokio's clock, not the session's.
    #[tokio::test]
    async fn a_close_that_never_resolves_is_timed_out_at_the_deadline_on_the_clock() {
        let clock = Arc::new(VirtualClock::new());
        let source: Box<dyn StreamSource<()>> = Box::new(Host { clock: Arc::clone(&clock), wedged: true });
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);

        let closing = tokio::spawn(source.close(deadline));
        tokio::task::yield_now().await;
        assert!(!closing.is_finished(), "nothing moved time, the close must still be waiting");
        clock.advance(HOST_TEARDOWN_TIMEOUT - Duration::from_millis(1));
        tokio::task::yield_now().await;
        assert!(!closing.is_finished(), "one millisecond before the deadline is not the deadline");
        clock.advance(Duration::from_millis(1));
        assert_eq!(closing.await.unwrap(), Closed::TimedOut);
    }

    // Catches: `bounded` reporting `TimedOut` for a teardown that answered.
    #[tokio::test]
    async fn a_teardown_that_answers_is_clean() {
        let clock = Arc::new(VirtualClock::new());
        let source: Box<dyn StreamSource<()>> = Box::new(Host { clock: Arc::clone(&clock), wedged: false });
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        assert_eq!(source.close(deadline).await, Closed::Clean);

        // A deadline already behind the clock still gives a ready teardown its one poll.
        let source: Box<dyn StreamSource<()>> = Box::new(Host { clock: Arc::clone(&clock), wedged: false });
        clock.advance(Duration::from_secs(1));
        assert_eq!(source.close(Ts::ZERO).await, Closed::Clean);
    }
}
