use futures_util::{FutureExt, future::Shared};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamOutput, Ts};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio_util::sync::CancellationToken;

// A cancelled waiter never loses the source or repeats an in-flight read.

#[derive(Debug, PartialEq)]
pub enum Pulled<T> {
    Item(T),
    End,
    Cancelled,
}

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
    closing: CancellationToken,
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
                clock,
            }),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closing.is_cancelled()
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
            let completed = pending.await;
            self.inner.finish_read(&completed)
        })
    }

    pub async fn aclose(&self) -> Closed {
        self.close(self.inner.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await
    }

    /// The first close owns the deadline. Every caller waits for the same cleanup outcome;
    /// abandoning a waiter leaves the pending work available to the next waiter.
    pub async fn close(&self, deadline: Ts) -> Closed {
        self.inner.closing.cancel();
        let cleanup = self.inner.begin_close(deadline);
        cleanup.await
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
                self.closing.cancel();
                Ok(Pulled::End)
            }
            Some(Err(error)) => {
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
            previous => close_source(previous, self.clock.clone(), deadline).boxed().shared(),
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
    async move {
        let item = tokio::select! {
            biased;
            () = closing.cancelled() => None,
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
#[path = "source_tests.rs"]
mod tests;
