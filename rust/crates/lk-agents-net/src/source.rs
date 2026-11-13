use futures_util::FutureExt;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{CallLedger, RetryReport};
use lk_agents_abi::{
    BoxFuture, Clock, Closed, HOST_TEARDOWN_TIMEOUT, InputState, ProviderStream, StreamOutput, Ts,
};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio_util::sync::CancellationToken;

// A cancelled waiter never loses the source or repeats an in-flight read.

/// One duplex stream's [`InputState`], shared by its halves: the output side writes how the
/// stream ended, the provider's own failure included, and every input call answers from it first.
#[derive(Clone)]
pub struct InputGate(Arc<crate::signal::Shared<InputState>>);

impl Default for InputGate {
    fn default() -> Self {
        InputGate(Arc::new(crate::signal::Shared::new(InputState::Open)))
    }
}

impl std::fmt::Debug for InputGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.with(|state| f.debug_tuple("InputGate").field(state).finish())
    }
}

impl InputGate {
    /// `Ok` while the input is open, else the provider's failure or the end.
    pub fn admit(&self) -> Result<(), ProviderError> {
        self.0.with(|state| state.admit())
    }

    /// `then` runs only while the input is open, under the gate: nothing it admits lands after
    /// an end or a failure.
    pub fn admit_with<R>(
        &self,
        then: impl FnOnce() -> Result<R, ProviderError>,
    ) -> Result<R, ProviderError> {
        self.0.with(|state| {
            state.admit()?;
            then()
        })
    }

    /// The input refused for a reason of its own (its queue is gone): it has ended, and the
    /// provider's failure, if any, is the answer.
    pub fn refused(&self) -> ProviderError {
        self.0.modify(|state| {
            state.end();
            state.admit().expect_err("an ended input refuses")
        })
    }

    pub fn end(&self) {
        self.0.modify(InputState::end);
    }

    pub fn fail(&self, error: ProviderError) {
        self.0.modify(|state| state.fail(error));
    }

    pub fn failure(&self) -> Option<ProviderError> {
        self.0.with(|state| state.failure().cloned())
    }

    /// The provider's failure once recorded; cancel-safe. A stream that ends cleanly never
    /// resolves it, so it is raced against the stream's own read.
    pub async fn failed(&self) -> ProviderError {
        self.0.wait_until(|state| state.failure().cloned()).await
    }
}

/// An output that records in its stream's [`InputGate`] how it ended: its failure as it reads
/// it, its end, its close or its drop.
pub struct Gated<S> {
    source: S,
    gate: EndOnDrop,
}

struct EndOnDrop(InputGate);

impl Drop for EndOnDrop {
    fn drop(&mut self) {
        self.0.end();
    }
}

impl<S> Gated<S> {
    pub fn new(source: S, gate: InputGate) -> Self {
        Gated { source, gate: EndOnDrop(gate) }
    }
}

impl<T: Send + 'static, S: StreamOutput<T>> StreamOutput<T> for Gated<S> {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            let next = self.source.next().await;
            match &next {
                Ok(Some(_)) => {}
                Ok(None) => self.gate.0.end(),
                Err(error) => self.gate.0.fail(error.clone()),
            }
            next
        })
    }

    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        let Gated { source, gate } = *self;
        drop(gate);
        Box::new(source).close(deadline)
    }
}

impl<T: Send + 'static, S: ProviderStream<T>> ProviderStream<T> for Gated<S> {
    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.source.take_call_metrics(in_flight)
    }

    fn call_ledger(&self) -> Option<CallLedger> {
        self.source.call_ledger()
    }

    fn report_retries(&mut self, report: RetryReport) {
        self.source.report_retries(report)
    }
}

#[derive(Debug, PartialEq)]
pub enum Pulled<T> {
    Item(T),
    End,
    Cancelled,
}

type Pending<T> = Arc<Relay<Arc<Mutex<Read<T>>>>>;
type Cleanup = Arc<Relay<Closed>>;

/// Work several waiters may need finished, driven by one of them at a time: the driver awaits it
/// in its own task, which its executor polls again after every wake, one that lands mid-poll
/// included. A driver that goes away hands the work back and another waiter takes it; the rest
/// wait on the slot. (`futures::Shared` drops a wake that lands while another waiter polls, which
/// left a close waiting on a read that had already been told to stop.)
struct Relay<O> {
    slot: crate::signal::Shared<Slot<O>>,
}

enum Slot<O> {
    Idle(BoxFuture<'static, O>),
    Driven,
    Done(O),
}

enum Turn<O> {
    Drive(BoxFuture<'static, O>),
    Finished(O),
}

impl<O: Clone + Send + 'static> Relay<O> {
    fn new(work: BoxFuture<'static, O>) -> Arc<Self> {
        Arc::new(Relay { slot: crate::signal::Shared::new(Slot::Idle(work)) })
    }

    /// The work's outcome, driven here whenever no other waiter is driving it.
    async fn run(&self) -> O {
        let turn = self
            .slot
            .wait_until(|slot| match std::mem::replace(slot, Slot::Driven) {
                Slot::Idle(work) => Some(Turn::Drive(work)),
                Slot::Done(out) => {
                    *slot = Slot::Done(out.clone());
                    Some(Turn::Finished(out))
                }
                Slot::Driven => None,
            })
            .await;
        let work = match turn {
            Turn::Finished(out) => return out,
            Turn::Drive(work) => work,
        };
        let mut held = HandBack { slot: &self.slot, work: Some(work) };
        let out = held.work.as_mut().expect("held until it finishes").await;
        held.work = None;
        self.slot.modify(|slot| *slot = Slot::Done(out.clone()));
        out
    }
}

/// The work a driver holds: back in the slot, for the next waiter, if the driver goes away.
struct HandBack<'a, O> {
    slot: &'a crate::signal::Shared<Slot<O>>,
    work: Option<BoxFuture<'static, O>>,
}

impl<O> Drop for HandBack<'_, O> {
    fn drop(&mut self) {
        if let Some(work) = self.work.take() {
            self.slot.modify(|slot| *slot = Slot::Idle(work));
        }
    }
}

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
    closing: CancellationToken,
    /// Its input's answer: the failure read, or the end or close.
    input: InputGate,
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
                closing: CancellationToken::new(),
                input: InputGate::default(),
                clock,
            }),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closing.is_cancelled()
    }

    /// Whether the stream's input still takes anything: once the output failed, its failure.
    pub fn admit_input(&self) -> Result<(), ProviderError> {
        self.inner.input.admit()
    }

    /// The output's failure, once read: an end of input after it answers with it too.
    pub fn input_failure(&self) -> Option<ProviderError> {
        self.inner.input.failure()
    }

    /// Admits one read at a time: `None` while another pull of this stream is in flight, which
    /// the language adapter raises as its overlap error.
    pub fn try_pull(&self) -> Option<impl Future<Output = Result<Pulled<T>, ProviderError>> + '_> {
        let closed = self.is_closed();
        let reading = if closed {
            None
        } else {
            let reading = &self.inner.reading;
            reading.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).ok()?;
            Some(Reading(reading))
        };
        Some(async move {
            let _reading = reading;
            if closed {
                return Ok(Pulled::End);
            }
            let Some(pending) = self.inner.begin_read() else {
                return Ok(Pulled::End);
            };
            // No state lock is held while polling the source, which may enter a language runtime.
            let completed = pending.run().await;
            self.inner.finish_read(&completed)
        })
    }

    pub async fn aclose(&self) -> Closed {
        self.close(self.inner.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await
    }

    /// The first close owns the deadline. Every caller waits for the same cleanup outcome;
    /// abandoning a waiter leaves the pending work available to the next waiter.
    pub async fn close(&self, deadline: Ts) -> Closed {
        self.inner.input.end();
        self.inner.closing.cancel();
        let cleanup = self.inner.begin_close(deadline);
        cleanup.run().await
    }
}

impl<T: Send + 'static> Inner<T> {
    /// Joins the read in flight or starts one; `None` once the stream is closing.
    fn begin_read(&self) -> Option<Pending<T>> {
        let mut state = self.state.lock();
        let pending = match state.take().expect("a stream always owns its state") {
            State::Ready(source) => read_next(source, self.closing.clone()),
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
    fn finish_read(&self, completed: &Mutex<Read<T>>) -> Result<Pulled<T>, ProviderError> {
        let (item, previous) = {
            let mut state = self.state.lock();
            if self.closing.is_cancelled() {
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
                self.input.end();
                self.closing.cancel();
                Ok(Pulled::End)
            }
            Some(Err(error)) => {
                self.input.fail(error.clone());
                self.closing.cancel();
                Err(error)
            }
            None => Ok(Pulled::Cancelled),
        }
    }

    /// Installs the one cleanup every closer awaits; the first close owns its deadline.
    fn begin_close(&self, deadline: Ts) -> Cleanup {
        let mut state = self.state.lock();
        let cleanup = match state.take().expect("a stream always owns its state") {
            State::Closing(cleanup) => cleanup,
            previous => Relay::new(close_source(previous, self.clock.clone(), deadline).boxed()),
        };
        *state = Some(State::Closing(cleanup.clone()));
        cleanup
    }
}

/// One shared read of the source's next item; a close wins over an item ready at once.
fn read_next<T: Send + 'static>(
    mut source: Box<dyn StreamOutput<T>>,
    closing: CancellationToken,
) -> Pending<T> {
    Relay::new(
        async move {
            let item = tokio::select! {
                biased;
                () = closing.cancelled() => None,
                item = source.next() => Some(item),
            };
            Arc::new(Mutex::new(Read { source: Some(source), item }))
        }
        .boxed(),
    )
}

/// Closes the source once any read in flight hands it back, within the deadline.
async fn close_source<T: Send + 'static>(
    previous: State<T>,
    clock: Arc<dyn Clock>,
    deadline: Ts,
) -> Closed {
    let mut outcome = Closed::Clean;
    let bounded = Closed::bounded(&*clock, deadline, async {
        let source = match previous {
            State::Ready(source) => Some(source),
            State::Reading(pending) => pending.run().await.lock().source.take(),
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
#[path = "source_tests.rs"]
mod tests;
