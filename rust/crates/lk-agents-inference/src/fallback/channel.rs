//! The drain half of anything the chain fans out: an `mpsc` plus the task that fills it.

use std::sync::Arc;

use crate::fallback::chain::StreamTask;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamOutput, Ts};
use tokio::sync::{mpsc, oneshot, watch};

use crate::fallback::chain::{Attempt, Backend, ChainCall, FallbackChain, Recovery};
use lk_agents_abi::BoxFuture;

/// The task closes its member on stop rather than being aborted.
pub(crate) struct ChainTask {
    pub(crate) stop: watch::Sender<bool>,
    pub(crate) done: Option<oneshot::Receiver<()>>,
    pub(crate) clock: Arc<dyn Clock>,
}

impl ChainTask {
    pub(crate) fn pair(
        clock: Arc<dyn Clock>,
    ) -> (watch::Receiver<bool>, oneshot::Sender<()>, ChainTask) {
        let (stop_tx, stop_rx) = watch::channel(false);
        let (done_tx, done_rx) = oneshot::channel();
        (stop_rx, done_tx, ChainTask { stop: stop_tx, done: Some(done_rx), clock })
    }
}

/// `source.next()` or the consumer's stop, whichever comes first. A stop reads as the stream's
/// clean end, so the attempt loop breaks `Served` and closes its member on the way out.
pub(crate) async fn next_or_stop<T, S: StreamOutput<T> + ?Sized>(
    source: &mut S,
    stop: &mut watch::Receiver<bool>,
) -> Result<Option<T>, ProviderError> {
    if *stop.borrow() {
        return Err(ApiError::status("fallback cancelled", 499, None, None).into());
    }
    tokio::select! {
        biased;
        changed = stop.changed() => {
            let _ = changed;
            Err(ApiError::status("fallback cancelled", 499, None, None).into())
        }
        next = source.next() => next,
    }
}

/// The first item marks the entry serving; `is_output` says which items a retry would duplicate.
/// A consumer that dropped the stream ends the attempt `Served`, not as a provider failure.
pub(crate) async fn forward<T, B: Backend, S: StreamOutput<T> + ?Sized>(
    chain: &FallbackChain<B>,
    index: usize,
    source: &mut S,
    stop: &mut watch::Receiver<bool>,
    tx: &mut mpsc::Sender<Result<T, ProviderError>>,
    is_output: fn(&T) -> bool,
) -> Attempt<()> {
    let (mut serving, mut produced) = (false, false);
    loop {
        let item = match next_or_stop(source, stop).await {
            Ok(Some(item)) => item,
            Ok(None) => return Attempt::Served(()),
            Err(error) => return Attempt::Failed { error, produced_output: produced },
        };
        if !serving {
            serving = true;
            chain.mark_serving(index);
        }
        let visible = is_output(&item);
        if !send_or_stop(tx, Ok(item), stop).await {
            return Attempt::Failed {
                error: ApiError::status("fallback cancelled", 499, None, None).into(),
                produced_output: produced,
            };
        }
        produced |= visible;
    }
}

/// Items a streamed request holds ahead of its reader. Bounded, so a slow reader backpressures
/// the provider; the text leg may not drop (§1.2d).
const STREAM_BUFFER: usize = 32;

/// How an entry serves one attempt of a streamed request: it forwards the opened session into
/// `tx` until the session ends, fails or the consumer stops.
pub(crate) trait ServeAttempt<B: Backend, T>: Send + Sync + 'static {
    fn serve<'a>(
        self: Arc<Self>,
        chain: Arc<FallbackChain<B>>,
        index: usize,
        session: B::Session,
        tx: &'a mut mpsc::Sender<Result<T, ProviderError>>,
        stop: watch::Receiver<bool>,
    ) -> BoxFuture<'a, Attempt<()>>;
}

impl<B: Backend> FallbackChain<B> {
    /// Runs `req` on a stream task feeding the returned source. `settled` sees the chain's
    /// verdict before a failure reaches the consumer; `keep` is dropped last, when the task ends
    /// or is aborted.
    pub(crate) fn serve_stream<T: Send + 'static>(
        self: &Arc<Self>,
        req: B::Req,
        call: ChainCall,
        attempt: Arc<impl ServeAttempt<B, T>>,
        settled: impl FnOnce(&Self, Result<(), &ProviderError>) + Send + 'static,
        keep: impl Send + 'static,
    ) -> ChannelSource<T> {
        let (mut tx, rx) = mpsc::channel(STREAM_BUFFER);
        let (stop_rx, done_tx, chain_task) = ChainTask::pair(self.clock());
        let chain = Arc::clone(self);
        let task = self.spawn_stream(async move {
            let _keep = keep;
            let serving = Arc::clone(&chain);
            let result = chain
                .run_cancellable(
                    &req,
                    call,
                    &mut tx,
                    stop_rx,
                    move |index, _, session, tx, stop| {
                        Arc::clone(&attempt).serve(Arc::clone(&serving), index, session, tx, stop)
                    },
                )
                .await;
            settled(&chain, result.as_ref().map(|_| ()));
            if let Err(error) = result {
                let _ = tx.send(Err(error)).await;
            }
            let _ = done_tx.send(());
        });
        ChannelSource { rx, task: Some(task), chain: chain_task }
    }
}

/// Delivery waits belong to the consumer and remain cancellable while the channel is full.
pub(crate) async fn send_or_stop<T>(
    tx: &mpsc::Sender<T>,
    item: T,
    stop: &mut watch::Receiver<bool>,
) -> bool {
    use futures_util::future::{Either, select};
    if *stop.borrow() {
        return false;
    }
    match select(Box::pin(tx.send(item)), Box::pin(stop.changed())).await {
        Either::Left((result, _)) => result.is_ok(),
        Either::Right(_) => false,
    }
}

/// Resolves once `stop` is true. A chain dropped without closing never stops its probes.
pub(crate) async fn stopped(stop: &mut watch::Receiver<bool>) {
    if stop.wait_for(|s| *s).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// `source.next()`, or `None` once the chain closes (the probe then closes its stream).
pub(crate) async fn probe_next<T, S: StreamOutput<T> + ?Sized>(
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
pub(crate) async fn drain_probe<T, S: StreamOutput<T> + ?Sized>(
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
    source.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    verdict
}

pub(crate) struct ChannelSource<T> {
    pub(crate) rx: mpsc::Receiver<Result<T, ProviderError>>,
    pub(crate) task: Option<StreamTask>,
    pub(crate) chain: ChainTask,
}

impl<T> Drop for ChannelSource<T> {
    fn drop(&mut self) {
        let _ = self.chain.stop.send(true);
    }
}

impl<T: Send + 'static> StreamOutput<T> for ChannelSource<T> {
    /// An `mpsc` drain: dropping a `next()` mid-poll loses nothing, and `close` aborts the filler.
    fn cancel_safe(&self) -> bool {
        true
    }

    /// `Ok(None)` is the chain task finishing; `Err` is what it reported.
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            match self.rx.recv().await {
                None => Ok(None),
                Some(Ok(v)) => Ok(Some(v)),
                Some(Err(e)) => Err(e),
            }
        })
    }

    /// Stop the chain task and wait for it under the deadline; the receiver closes first so a
    /// pending `send` fails rather than blocks. An unanswered task is aborted and `TimedOut`.
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            let _ = self.chain.stop.send(true);
            self.rx.close();
            let done = self.chain.done.take();
            let closed = Closed::bounded(self.chain.clock.as_ref(), deadline, async move {
                if let Some(done) = done {
                    let _ = done.await;
                }
            })
            .await;
            if let Some(task) = self.task.take() {
                // Finished tasks ignore the cancel; a wedged one is cut here.
                task.cancel();
            }
            closed
        })
    }
}

impl<T: Send + 'static> lk_agents_abi::utils::stream::ProviderStream<T> for ChannelSource<T> {}
