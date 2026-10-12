//! The drain half of anything the chain fans out: an `mpsc` plus the task that fills it.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Clock, Closed, StreamSource, Ts};
use tokio::sync::{mpsc, oneshot, watch};
use lk_agents_abi::AbortHandle;

use crate::fallback::chain::{Attempt, Backend, FallbackChain, Recovery};

/// The task closes its member on stop rather than being aborted.
pub(crate) struct ChainTask {
    pub(crate) stop: watch::Sender<bool>,
    pub(crate) done: Option<oneshot::Receiver<()>>,
    pub(crate) clock: Arc<dyn Clock>,
}

impl ChainTask {
    pub(crate) fn pair(clock: Arc<dyn Clock>) -> (watch::Receiver<bool>, oneshot::Sender<()>, ChainTask) {
        let (stop_tx, stop_rx) = watch::channel(false);
        let (done_tx, done_rx) = oneshot::channel();
        (stop_rx, done_tx, ChainTask { stop: stop_tx, done: Some(done_rx), clock })
    }
}

/// `source.next()` or the consumer's stop, whichever comes first. A stop reads as the stream's
/// clean end, so the attempt loop breaks `Served` and closes its member on the way out.
pub(crate) async fn next_or_stop<T, S: StreamSource<T> + ?Sized>(
    source: &mut S,
    stop: &mut watch::Receiver<bool>,
) -> Result<Option<T>, ProviderError> {
    if *stop.borrow() {
        return Ok(None);
    }
    tokio::select! {
        biased;
        changed = stop.changed() => {
            let _ = changed;
            Ok(None)
        }
        next = source.next() => next,
    }
}

/// The first item marks the entry serving; `is_output` says which items a retry would duplicate.
/// A consumer that dropped the stream ends the attempt `Served`, not as a provider failure.
pub(crate) async fn forward<T, B: Backend, S: StreamSource<T> + ?Sized>(
    chain: &FallbackChain<B>,
    index: usize,
    source: &mut S,
    stop: &mut watch::Receiver<bool>,
    tx: &mut mpsc::Sender<Result<T, ProviderError>>,
    is_output: fn(&T) -> bool,
) -> Attempt<()> {
    let (mut serving, mut produced) = (false, false);
    loop {
        match next_or_stop(source, stop).await {
            Ok(None) => return Attempt::Served(()),
            Ok(Some(item)) => {
                if !serving {
                    serving = true;
                    chain.mark_serving(index);
                }
                produced |= is_output(&item);
                if tx.send(Ok(item)).await.is_err() {
                    return Attempt::Served(());
                }
            }
            Err(error) => return Attempt::Failed { error, produced_output: produced },
        }
    }
}

pub(crate) fn teardown_deadline(clock: &dyn Clock) -> Ts {
    clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
}

/// Resolves once `stop` is true. A chain dropped without closing never stops its probes.
pub(crate) async fn stopped(stop: &mut watch::Receiver<bool>) {
    if stop.wait_for(|s| *s).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// `source.next()`, or `None` once the chain closes (the probe then closes its stream).
pub(crate) async fn probe_next<T, S: StreamSource<T> + ?Sized>(
    source: &mut S,
    stop: &mut watch::Receiver<bool>,
) -> Option<Result<Option<T>, ProviderError>> {
    tokio::select! {
        biased;
        () = stopped(stop) => None,
        next = source.next() => Some(next),
    }
}

/// A probe's stream read to its end and discarded: a clean end is recovery, an error is not, and
/// a close of the chain is neither.
pub(crate) async fn drain_probe<T, S: StreamSource<T> + ?Sized>(
    mut source: Box<S>,
    clock: &dyn Clock,
    mut stop: watch::Receiver<bool>,
) -> Result<Recovery, ProviderError> {
    let verdict = loop {
        match probe_next(source.as_mut(), &mut stop).await {
            None => break Ok(Recovery::NotYet),
            Some(Ok(None)) => break Ok(Recovery::Recovered),
            Some(Ok(Some(_))) => {}
            Some(Err(e)) => break Err(e),
        }
    };
    source.close(teardown_deadline(clock)).await;
    verdict
}

pub(crate) struct ChannelSource<T> {
    pub(crate) rx: mpsc::Receiver<Result<T, ProviderError>>,
    pub(crate) task: Option<AbortHandle>,
    pub(crate) chain: ChainTask,
}

#[async_trait]
impl<T: Send> StreamSource<T> for ChannelSource<T> {
    /// An `mpsc` drain: dropping a `next()` mid-poll loses nothing, and `close` aborts the filler.
    fn cancel_safe(&self) -> bool {
        true
    }

    /// `Ok(None)` is the chain task finishing; `Err` is what it reported.
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        match self.rx.recv().await {
            None => Ok(None),
            Some(Ok(v)) => Ok(Some(v)),
            Some(Err(e)) => Err(e),
        }
    }

    /// Stop the chain task and wait for it under the deadline; the receiver closes first so a
    /// pending `send` fails rather than blocks. An unanswered task is aborted and `TimedOut`.
    async fn close(mut self: Box<Self>, deadline: Ts) -> Closed {
        let _ = self.chain.stop.send(true);
        self.rx.close();
        let done = self.chain.done.take();
        let closed = Closed::bounded(self.chain.clock.as_ref(), deadline, async move {
            if let Some(done) = done {
                let _ = done.await;
            }
        })
        .await;
        if let Some(h) = self.task.take() {
            // Finished tasks ignore the abort; a wedged one is cut here.
            h.abort();
        }
        closed
    }
}
