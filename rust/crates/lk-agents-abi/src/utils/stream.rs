//! The duplex split. Never fuse the halves: dropping a Rust future does not cancel the host's
//! `__anext__`, which stays parked in a `tee` generator and poisons the stream (`tts/tts.py:543`).

/// The input and output halves of one bidirectional stream.
pub type StreamPair<I, O> = (Box<dyn StreamSink<I>>, Box<dyn StreamSource<O>>);

use std::future::Future;
use std::time::Duration;

use crate::errors::ProviderError;
use crate::utils::clock::{Clock, Elapsed, Ts, timeout};

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
pub trait StreamSource<T>: Send {
    /// `Ok(None)` is a clean end. Kept apart: `__anext__` re-raises the task's exception on
    /// exhaustion.
    fn next(&mut self) -> crate::BoxFuture<'_, Result<Option<T>, ProviderError>>;

    /// Whether a `next()` future may be dropped mid-poll. `false` for a host stream (interruption
    /// waits for the chunk).
    fn cancel_safe(&self) -> bool {
        false
    }

    /// The only teardown: dropping does not cancel the host task. Bounded by `deadline` on the
    /// session clock.
    fn close(self: Box<Self>, deadline: Ts) -> crate::BoxFuture<'static, Closed>;

    /// A synthesis stream's `_started_time` (`tts/tts.py`): the `perf_counter()` at which text was
    /// first handed to the provider, once it has been -- even by an attempt that then failed
    /// without audio. `None` when unknown, and for every other kind of stream.
    fn started_time(&self) -> Option<f64> {
        None
    }

    /// The records of the provider calls this stream makes itself (a `StreamAdapter`'s one
    /// synthesis per sentence), finished since the last read; `in_flight` also ends the call
    /// still running, as cancelled. `None` for a stream that is one call, which its reader meters.
    fn take_call_metrics(
        &mut self,
        _in_flight: bool,
    ) -> Option<Vec<crate::metrics::base::AgentMetrics>> {
        None
    }

    /// [`take_call_metrics`](Self::take_call_metrics) for another task, readable while this
    /// stream's reader waits on it (a node's default `tts_node` pumps it on its own task).
    fn call_ledger(&self) -> Option<CallLedger> {
        None
    }

    /// Hands a stream that retries its failed attempts itself (a host STT's) who else hears of
    /// each retried failure, as it happens; any other stream ignores it.
    fn report_retries(&mut self, _report: RetryReport) {}
}

/// A boxed sink is the sink it holds.
impl<T, S: StreamSink<T> + ?Sized> StreamSink<T> for Box<S> {
    fn push(&self, item: T) -> Result<(), ProviderError> {
        S::push(self, item)
    }

    fn flush(&self) -> Result<(), ProviderError> {
        S::flush(self)
    }

    fn end_input(&self) {
        S::end_input(self)
    }
}

/// A boxed source is the source it holds, every capability included.
impl<T, S: StreamSource<T> + ?Sized> StreamSource<T> for Box<S> {
    fn next(&mut self) -> crate::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        S::next(self)
    }

    fn cancel_safe(&self) -> bool {
        S::cancel_safe(self)
    }

    fn close(self: Box<Self>, deadline: Ts) -> crate::BoxFuture<'static, Closed> {
        S::close(*self, deadline)
    }

    fn started_time(&self) -> Option<f64> {
        S::started_time(self)
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<crate::metrics::base::AgentMetrics>> {
        S::take_call_metrics(self, in_flight)
    }

    fn call_ledger(&self) -> Option<CallLedger> {
        S::call_ledger(self)
    }

    fn report_retries(&mut self, report: RetryReport) {
        S::report_retries(self, report)
    }
}

/// Told of each failed attempt a stream retries.
pub type RetryReport =
    std::sync::Arc<dyn Fn(crate::errors::ProviderError, u32, std::time::Duration) + Send + Sync>;

/// A stream's provider calls, read by another task: `take(in_flight)` is
/// [`StreamSource::take_call_metrics`].
pub type CallLedger =
    std::sync::Arc<dyn Fn(bool) -> Vec<crate::metrics::base::AgentMetrics> + Send + Sync>;

/// How long any cleanup may take to settle (a stream's close, a request attempt, a run's
/// cancelled work, the host loop's close) before it is reported incomplete; the reference's
/// `INTERRUPTION_TIMEOUT`. Real teardowns take under a millisecond.
pub const HOST_TEARDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    Clean,
    /// The teardown ran and failed (a host `aclose()` raised); the failure is logged where it
    /// happened. Never reported as `Clean`.
    Failed,
    /// Abandoned at the deadline; a host generator is still running.
    TimedOut,
}

impl Closed {
    /// Two teardowns as one: the worse outcome, a timeout over a failure over a clean close.
    pub fn and(self, other: Closed) -> Closed {
        match (self, other) {
            (Closed::TimedOut, _) | (_, Closed::TimedOut) => Closed::TimedOut,
            (Closed::Failed, _) | (_, Closed::Failed) => Closed::Failed,
            (Closed::Clean, Closed::Clean) => Closed::Clean,
        }
    }

    /// On timeout the future is dropped, leaking one host task rather than hanging the pump. A past
    /// deadline still gives it one poll.
    pub async fn bounded<F: Future<Output = ()>>(
        clock: &dyn Clock,
        deadline: Ts,
        teardown: F,
    ) -> Closed {
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

    impl StreamSource<()> for Host {
        fn next(&mut self) -> crate::BoxFuture<'_, Result<Option<()>, ProviderError>> {
            Box::pin(async move { Ok(None) })
        }
        fn close(self: Box<Self>, deadline: Ts) -> crate::BoxFuture<'static, Closed> {
            Box::pin(async move {
                let wedged = self.wedged;
                Closed::bounded(&*self.clock, deadline, async move {
                    if wedged {
                        std::future::pending::<()>().await;
                    }
                })
                .await
            })
        }
    }

    fn poll<F: Future + ?Sized>(f: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        f.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    // Catches: a wedged host teardown hanging the pump, or a bound on tokio's clock, not the session's.
    #[test]
    fn a_close_that_never_resolves_is_timed_out_at_the_deadline_on_the_clock() {
        let clock = Arc::new(VirtualClock::new());
        let source: Box<dyn StreamSource<()>> =
            Box::new(Host { clock: Arc::clone(&clock), wedged: true });
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);

        let mut closing = source.close(deadline);
        assert!(poll(closing.as_mut()).is_pending(), "nothing moved time, the close must wait");
        clock.advance(HOST_TEARDOWN_TIMEOUT - Duration::from_millis(1));
        assert!(poll(closing.as_mut()).is_pending(), "one millisecond before is not the deadline");
        clock.advance(Duration::from_millis(1));
        assert_eq!(poll(closing.as_mut()), std::task::Poll::Ready(Closed::TimedOut));
    }

    // Catches: `bounded` reporting `TimedOut` for a teardown that answered.
    #[test]
    fn a_teardown_that_answers_is_clean() {
        let clock = Arc::new(VirtualClock::new());
        let source: Box<dyn StreamSource<()>> =
            Box::new(Host { clock: Arc::clone(&clock), wedged: false });
        let deadline = clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        assert_eq!(poll(source.close(deadline).as_mut()), std::task::Poll::Ready(Closed::Clean));

        // A deadline already behind the clock still gives a ready teardown its one poll.
        let source: Box<dyn StreamSource<()>> =
            Box::new(Host { clock: Arc::clone(&clock), wedged: false });
        clock.advance(Duration::from_secs(1));
        assert_eq!(poll(source.close(Ts::ZERO).as_mut()), std::task::Poll::Ready(Closed::Clean));
    }
}
