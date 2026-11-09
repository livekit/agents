/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_host_derive::host(interface(
        host = "collections.abc.AsyncIterable",
        attrs(aiter(name = "__aiter__"))
    ))]
    pub struct AsyncIterable;

    #[lk_agents_host_derive::host(interface(
        host = "collections.abc.AsyncIterator",
        attrs(anext(name = "__anext__"))
    ))]
    pub struct AsyncIterator;

    #[lk_agents_host_derive::host(interface(
        protocol = "supports aclose()",
        attrs(aclose),
        protocols(matches(required(aclose)))
    ))]
    pub struct AsyncClose;

    #[lk_agents_host_derive::host(interface(
        protocol = "framework host stream input",
        attrs(
            push(name = "_core_push"),
            flush(name = "_core_flush"),
            end_input(name = "_core_end_input")
        )
    ))]
    pub struct StreamInput;
}

use crate::generated::accessors as access;
use crate::py_calls::Started;
use crate::py_calls::captured_pair;
use crate::py_calls::start_in;
use crate::py_errors::captured;
use lk_agents_abi::Closed;
use lk_agents_abi::StreamInput;
use lk_agents_abi::StreamOutput;
use lk_agents_abi::errors::{ProviderError, PullError};
use lk_agents_abi::utils::clock::Clock;
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::utils::stream::Entry;
use lk_agents_runtime::net::source::Pulled;
use lk_agents_runtime::net::source::SharedStream;
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Reads one host item as `T`.
pub type Convert<T> = fn(&Bound<'_, PyAny>) -> PyResult<T>;

/// Writes one input item as the host's.
pub type ToHost<I> = fn(Python<'_>, I) -> PyResult<Bound<'_, PyAny>>;

/// Input entries a host stream holds before its drain hands them over: past this, a push is
/// refused rather than queued.
pub const SINK_QUEUE: usize = 256;

pub struct HostStream<T> {
    host: Arc<HostIterator>,
    convert: Convert<T>,
    clock: Arc<dyn Clock>,
    /// The pull whose caller stopped awaiting it: the next one resumes it.
    pending: Option<Started>,
    ended: bool,
    input_stopped: Arc<AtomicBool>,
    failure: tokio::sync::watch::Sender<Option<ProviderError>>,
}

/// A host async iterator, the loop it runs on and its own context: a copy taken once, at open.
/// Shared, so a drop can hand it on without the GIL.
struct HostIterator {
    iterator: Py<PyAny>,
    event_loop: Py<PyAny>,
    ctx: Py<PyAny>,
}

impl<T> HostStream<T> {
    /// The stream of `iterable` (its `__aiter__()`, taken inside the stream's context), opened in
    /// `context` itself, the context its every pull and its close run in: a fresh capture, or the
    /// context the call that returned it ran in.
    pub fn open(
        py: Python<'_>,
        iterable: &Bound<'_, PyAny>,
        context: &HostRef,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<Self> {
        let (event_loop, ctx) = captured_pair(py, context)?;
        Self::open_in(iterable, &event_loop, &ctx, convert, clock)
    }

    /// An iterable returned by a call whose context must also own every pull and its close.
    pub(crate) fn open_in(
        iterable: &Bound<'_, PyAny>,
        event_loop: &Bound<'_, PyAny>,
        ctx: &Bound<'_, PyAny>,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<Self> {
        let iterator =
            access::Context::run(ctx)?.call1((access::AsyncIterable::aiter(iterable)?,))?;
        Ok(HostStream {
            host: Arc::new(HostIterator {
                iterator: iterator.unbind(),
                event_loop: event_loop.clone().unbind(),
                ctx: ctx.clone().unbind(),
            }),
            convert,
            clock,
            pending: None,
            ended: false,
            input_stopped: Arc::new(AtomicBool::new(false)),
            failure: tokio::sync::watch::channel(None).0,
        })
    }

    /// The two halves of the duplex stream `obj`: input through its `_core_push`, `_core_flush`
    /// and `_core_end_input`, output by iterating it, both in one copy of `context`.
    pub fn duplex<I>(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        context: &HostRef,
        to_host: ToHost<I>,
        convert: Convert<T>,
        clock: Arc<dyn Clock>,
    ) -> PyResult<lk_agents_abi::DuplexStream<HostSink<I>, Self>> {
        let source = Self::open(py, obj, context, convert, clock)?;
        let sink = HostSink {
            shared: Arc::new(SinkShared {
                target: obj.clone().unbind(),
                event_loop: source.host.event_loop.clone_ref(py),
                ctx: source.host.ctx.clone_ref(py),
                to_host,
                stopped: source.input_stopped.clone(),
                failure: source.failure.clone(),
                queue: Mutex::new(SinkQueue {
                    entries: VecDeque::new(),
                    scheduled: false,
                    ended: false,
                }),
            }),
        };
        Ok(lk_agents_abi::DuplexStream::new(sink, source))
    }

    fn start(&self, close: bool) -> Result<Started, ProviderError> {
        Python::attach(|py| {
            let host = &self.host;
            let iterator = host.iterator.clone_ref(py);
            start_in(py, host.event_loop.bind(py), host.ctx.bind(py), move |py| {
                let callable = if close {
                    access::AsyncClose::aclose(iterator.bind(py))?
                } else {
                    access::AsyncIterator::anext(iterator.bind(py))?
                };
                callable.call0()
            })
        })
        .map_err(captured)
    }
}

impl<T: Send + 'static> StreamOutput<T> for HostStream<T> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            if self.ended {
                return Ok(None);
            }
            let mut failure = self.failure.subscribe();
            if let Some(error) = failure.borrow().clone() {
                self.ended = true;
                self.input_stopped.store(true, Ordering::Release);
                return Err(error);
            }
            let pull = match &mut self.pending {
                Some(pull) => pull,
                None => self.pending.insert(self.start(false)?),
            };
            let pulled = tokio::select! {
                biased;
                failed = failure.wait_for(|error| error.is_some()) => {
                    let error = failed.expect("the stream owns its failure sender").clone().expect("waited for failure");
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    return Err(error);
                },
                pulled = pull.settled() => pulled,
            };
            self.pending = None;
            let item = Python::attach(|py| match pulled {
                Ok(item) => (self.convert)(item.bind(py)).map(Some),
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) => {
                    Ok(None)
                }
                Err(e) => Err(e),
            });
            match item {
                Ok(Some(item)) => Ok(Some(item)),
                Ok(None) => {
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    Ok(None)
                }
                Err(e) => {
                    self.ended = true;
                    self.input_stopped.store(true, Ordering::Release);
                    Err(captured(e))
                }
            }
        })
    }

    /// A dropped pull is kept and resumed, never repeated.
    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        // Closed here, however this close ends: the drop backstop stays off.
        self.ended = true;
        Box::pin(async move {
            self.input_stopped.store(true, Ordering::Release);
            let clock = self.clock.clone();
            let pending = self.pending.take();
            let has_aclose =
                Python::attach(|py| access::AsyncClose::matches(self.host.iterator.bind(py)));
            let mut failed = false;
            let teardown = async {
                if let Some(mut pull) = pending {
                    pull.cancel();
                    let _ = pull.settled().await;
                }
                match has_aclose {
                    Ok(true) => {}
                    Ok(false) => return,
                    Err(error) => {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, "looking up host stream cleanup failed: {error}");
                        failed = true;
                        return;
                    }
                }
                let closed = match self.start(true) {
                    Ok(mut closing) => closing.settled().await.map_err(captured),
                    Err(e) => Err(e),
                };
                if let Err(e) = closed {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "closing a host stream failed: {e}");
                    failed = true;
                }
            };
            let closed = Closed::bounded(&*clock, deadline, teardown).await;
            if failed { closed.and(Closed::Failed) } else { closed }
        })
    }
}

impl<T: Send + 'static> lk_agents_abi::utils::stream::ProviderStream<T> for HostStream<T> {}

/// The backstop of a stream dropped unclosed and unfinished (an owner that leaked it): its
/// pending pull is cancelled and its generator's `aclose()` queued on its loop, so the host task
/// never runs on unowned.
impl<T> Drop for HostStream<T> {
    fn drop(&mut self) {
        self.input_stopped.store(true, Ordering::Release);
        if self.ended {
            return;
        }
        let host = Arc::clone(&self.host);
        close_later(move |py| {
            let HostIterator { iterator, event_loop, ctx } = &*host;
            Ok((iterator.clone_ref(py), event_loop.clone_ref(py), ctx.clone_ref(py)))
        });
    }
}

/// A host iterator, and the loop and the context it is closed in.
type Located = (Py<PyAny>, Py<PyAny>, Py<PyAny>);

/// `aclose()` of the host iterator nobody closed that `located` names, queued as a task of its
/// loop, never awaited. Scheduled from the blocking I/O pool, so a drop holding a lock never
/// waits for the GIL. Lost only when the interpreter or the process runtime is gone.
fn close_later(located: impl FnOnce(Python<'_>) -> PyResult<Located> + Send + 'static) {
    let schedule = move || {
        let scheduled = Python::try_attach(|py| {
            let (iterator, event_loop, ctx) = located(py)?;
            schedule_aclose(py, iterator, event_loop, ctx)
        });
        if let Some(Err(error)) = scheduled {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "closing an abandoned host stream failed: {error}");
        }
    };
    on_process(async move {
        let pool = lk_agents_runtime::net::blocking_io::blocking_io();
        let _ = pool.run((), schedule).await;
    });
}

/// Queues `iterator.aclose()` as a task of `event_loop` in `ctx`, when it has one.
fn schedule_aclose(
    py: Python<'_>,
    iterator: Py<PyAny>,
    event_loop: Py<PyAny>,
    ctx: Py<PyAny>,
) -> PyResult<()> {
    if !access::AsyncClose::matches(iterator.bind(py))? {
        return Ok(());
    }
    let (loop_ref, task_ctx) = (event_loop.clone_ref(py), ctx.clone_ref(py));
    let start = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
        let py = args.py();
        let closing = access::AsyncClose::aclose(iterator.bind(py))?.call0()?;
        let coroutine = crate::py_calls::awaiting(py)?.call1((closing,))?;
        access::EventLoop::create_task(loop_ref.bind(py), &coroutine, task_ctx.bind(py))?;
        PyResult::Ok(())
    })?;
    access::EventLoop::call_soon_threadsafe(event_loop.bind(py))?
        .call1((access::Context::run(ctx.bind(py))?, start))?;
    Ok(())
}

/// Runs `work` on the process runtime, off the dropping thread.
fn on_process(work: impl std::future::Future<Output = ()> + Send + 'static) {
    match crate::py_calls::process_handle() {
        Ok(handle) => drop(handle.spawn(work)),
        Err(_) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "an abandoned host stream could not be closed: no process runtime")
        }
    }
}

/// The entry of a call that returns a host stream: a caller dropped before it took the stream
/// (a cancelled open) leaves the call to settle, and the stream it returned is closed in the
/// context it ran in, never discarded with its generator running.
pub struct StreamCall;

/// A started call and the `(loop, context)` pair it runs in, until its caller takes its value.
struct UnclaimedStream(Option<(Started, Py<PyAny>)>);

impl UnclaimedStream {
    /// The call's value, now the caller's.
    async fn settle(mut self) -> PyResult<crate::py_calls::Called> {
        let Some((started, _)) = &mut self.0 else { unreachable!("claimed once") };
        let value = started.settled().await;
        let ran_in = self.0.take().map(|(_, ran_in)| ran_in);
        Ok(crate::py_calls::Called { value: value?, ran_in })
    }
}

impl Drop for UnclaimedStream {
    fn drop(&mut self) {
        let Some((mut started, ran_in)) = self.0.take() else { return };
        started.cancel();
        on_process(async move {
            if let Ok(value) = started.settled().await {
                close_later(move |py| {
                    let (event_loop, ctx) = ran_in.bind(py).extract()?;
                    Ok((value, event_loop, ctx))
                });
            }
        });
    }
}

impl lk_agents_abi::invoke::HostEntry for StreamCall {
    type Call = crate::py_calls::HostCall;
    type Value = crate::py_calls::Called;
    type Error = PyErr;

    fn enter(
        &self,
        context: &HostRef,
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
        call: crate::py_calls::HostCall,
    ) -> lk_agents_abi::BoxFuture<'static, PyResult<crate::py_calls::Called>> {
        let context = context.clone();
        Box::pin(async move {
            UnclaimedStream(Some(crate::py_calls::start_call(&context, span, call)?)).settle().await
        })
    }

    /// A cancellable open settles under its own cancellation: nothing is dropped unsettled.
    fn enter_cancellable(
        &self,
        context: &HostRef,
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
        call: crate::py_calls::HostCall,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> lk_agents_abi::BoxFuture<'static, PyResult<crate::py_calls::Called>> {
        crate::py_calls::Entry.enter_cancellable(context, span, call, cancel)
    }

    fn enter_now(&self, call: crate::py_calls::HostCall) -> PyResult<crate::py_calls::Called> {
        crate::py_calls::Entry.enter_now(call)
    }

    fn refused(&self, label: &'static str, refusal: lk_agents_abi::invoke::Refusal) -> PyErr {
        crate::py_calls::Entry.refused(label, refusal)
    }
}

/// The input half of a duplex host stream.
pub struct HostSink<I> {
    shared: Arc<SinkShared<I>>,
}

struct SinkShared<I> {
    target: Py<PyAny>,
    event_loop: Py<PyAny>,
    ctx: Py<PyAny>,
    to_host: ToHost<I>,
    stopped: Arc<AtomicBool>,
    failure: tokio::sync::watch::Sender<Option<ProviderError>>,
    queue: Mutex<SinkQueue<I>>,
}

struct SinkQueue<I> {
    entries: VecDeque<Entry<I>>,
    /// A drain is scheduled on the loop and has not emptied the queue yet.
    scheduled: bool,
    ended: bool,
}

impl<I: Send + 'static> HostSink<I> {
    fn admit(&self, entry: Entry<I>) -> Result<(), ProviderError> {
        let schedule = {
            let mut queue = self.shared.queue.lock();
            if let Some(failed) = self.shared.failure.borrow().clone() {
                return Err(failed);
            }
            if self.shared.stopped.load(Ordering::Acquire) || queue.ended {
                return Err(lk_agents_abi::errors::ApiError::new("stream input is closed").into());
            }
            let ending = matches!(entry, Entry::EndInput);
            if !ending && queue.entries.len() >= SINK_QUEUE {
                return Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(format!(
                    "the host stream is not taking its input: {SINK_QUEUE} entries are waiting"
                ))));
            }
            queue.ended |= ending;
            queue.entries.push_back(entry);
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py)) {
            let e = captured(e);
            self.shared.failure.send_replace(Some(e.clone()));
            return Err(e);
        }
        Ok(())
    }

    fn schedule(&self, py: Python<'_>) -> PyResult<()> {
        let shared = self.shared.clone();
        let drain = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
            drain(args.py(), &shared);
            PyResult::Ok(())
        })?;
        let s = &self.shared;
        access::EventLoop::call_soon_threadsafe(s.event_loop.bind(py))?
            .call1((access::Context::run(s.ctx.bind(py))?, drain))?;
        Ok(())
    }
}

/// On the loop, inside the stream's context: everything queued, in order, until the queue is
/// empty; the first failure stops delivery and is kept for the next push.
fn drain<I>(py: Python<'_>, shared: &SinkShared<I>) {
    loop {
        if shared.stopped.load(Ordering::Acquire) {
            let _dropped = {
                let mut queue = shared.queue.lock();
                queue.scheduled = false;
                std::mem::take(&mut queue.entries)
            };
            return;
        }
        let batch: Vec<Entry<I>> = {
            let mut queue = shared.queue.lock();
            if queue.entries.is_empty() {
                queue.scheduled = false;
                return;
            }
            queue.entries.drain(..).collect()
        };
        for entry in batch {
            let target = shared.target.bind(py);
            let delivered = match entry {
                Entry::Item(item) => (shared.to_host)(py, item)
                    .and_then(|v| access::StreamInput::push(target)?.call1((v,))),
                Entry::Flush => access::StreamInput::flush(target).and_then(|call| call.call0()),
                Entry::EndInput => {
                    access::StreamInput::end_input(target).and_then(|call| call.call0())
                }
            };
            if let Err(e) = delivered {
                // Capturing reads the exception's attributes: Python code, run before the lock.
                let failed = crate::py_errors::capture(py, &e);
                // Dropped after the guard: an item's host objects may run Python finalizers.
                shared.failure.send_replace(Some(failed));
                let _dropped = {
                    let mut queue = shared.queue.lock();
                    queue.scheduled = false;
                    std::mem::take(&mut queue.entries)
                };
                return;
            }
        }
    }
}

impl<I: Send + Sync + 'static> StreamInput<I> for HostSink<I> {
    fn push(&self, item: I) -> Result<(), ProviderError> {
        self.admit(Entry::Item(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.admit(Entry::Flush)
    }

    /// A refused end is the failure the source reports.
    fn end_input(&self) {
        let _ = self.admit(Entry::EndInput);
    }

    /// Awaits the host's handling, an awaitable it returns included: an input that awaits room
    /// holds the pump, so the backlog stays with the pump's queue.
    fn send(
        &self,
        entry: Entry<I>,
    ) -> lk_agents_abi::BoxFuture<'static, Result<(), ProviderError>> {
        let shared = self.shared.clone();
        Box::pin(async move {
            if let Some(failed) = shared.failure.borrow().clone() {
                return Err(failed);
            }
            if shared.stopped.load(Ordering::Acquire) {
                return Err(lk_agents_abi::errors::ApiError::new("stream input is closed").into());
            }
            let started = Python::attach(|py| {
                let (target, to_host) = (shared.target.clone_ref(py), shared.to_host);
                start_in(py, shared.event_loop.bind(py), shared.ctx.bind(py), move |py| {
                    call_input(py, target.bind(py), to_host, entry)
                })
            });
            let sent = match started {
                Ok(mut started) => started.settled().await.map(drop),
                Err(e) => Err(e),
            };
            sent.map_err(|e| {
                let e = captured(e);
                shared.failure.send_replace(Some(e.clone()));
                e
            })
        })
    }
}

/// Calls the host input's method for `entry`, returning what it returned (maybe awaitable).
fn call_input<'py, I>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    to_host: ToHost<I>,
    entry: Entry<I>,
) -> PyResult<Bound<'py, PyAny>> {
    match entry {
        Entry::Item(item) => access::StreamInput::push(target)?.call1((to_host(py, item)?,)),
        Entry::Flush => access::StreamInput::flush(target)?.call0(),
        Entry::EndInput => access::StreamInput::end_input(target)?.call0(),
    }
}

#[cfg(test)]
#[path = "py_streams_tests.rs"]
pub(crate) mod tests;

pub fn recv_native<'py, T: Send + 'static>(
    source: &SharedStream<T>,
    py: Python<'py>,
    to_py: fn(Python<'_>, T) -> PyResult<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    crate::py_calls::future_into_py(py, async move {
        let Some(pull) = source.try_pull() else {
            return Err(crate::py_errors::HostException::to_py_err(&PullError::Overlap));
        };
        match pull.await {
            Ok(Pulled::Item(item)) => Python::attach(|py| to_py(py, item)).map(Some),
            Ok(Pulled::End | Pulled::Cancelled) => Ok(None),
            Err(error) => Err(crate::py_errors::HostException::to_py_err(&error)),
        }
    })
}

pub fn close_native<'py, T: Send + 'static>(
    source: &SharedStream<T>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let source = source.clone();
    crate::py_calls::future_into_py(py, async move {
        if source.aclose().await == lk_agents_abi::Closed::TimedOut {
            log::error!(target: lk_agents_abi::LOG_TARGET, "a native stream did not release its resources in time");
        }
        Ok(())
    })
}
