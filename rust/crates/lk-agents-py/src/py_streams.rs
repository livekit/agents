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

    /// Keeps a close nobody awaits until it finishes; the loop's teardown awaits it.
    #[lk_agents_host_derive::host(interface(
        module = "livekit.agents.utils.aio",
        attrs(_keep_closing)
    ))]
    pub struct AioModule;
}

use crate::generated::accessors as access;
use crate::py_calls::Child;
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
use lk_agents_runtime::net::source::InputGate;
use lk_agents_runtime::net::source::Pulled;
use lk_agents_runtime::net::source::SharedStream;
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

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
    input: Arc<InputStop>,
}

/// Whether a duplex stream still takes input, and the host task of the input entry in flight:
/// stopping cancels it, so an input waiting on output nobody reads never outlives the stream.
struct InputStop {
    event_loop: Arc<Py<PyAny>>,
    in_flight: Arc<Child>,
    /// What every input call answers once the stream ended: either half's failure, which ends
    /// the output too, or the end.
    gate: InputGate,
}

impl InputStop {
    /// From any thread, a dropping one included: never waits for the GIL.
    fn stop(&self) {
        self.gate.end();
        self.in_flight.cancel_later(&self.event_loop);
    }

    fn stopped(&self) -> bool {
        self.in_flight.cancelled()
    }
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
            input: Arc::new(InputStop {
                event_loop: Arc::new(event_loop.clone().unbind()),
                in_flight: Arc::default(),
                gate: InputGate::default(),
            }),
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
                stop: source.input.clone(),
                queue: Mutex::new(SinkQueue { entries: VecDeque::new(), scheduled: false }),
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
            let gate = self.input.gate.clone();
            if let Some(error) = gate.failure() {
                self.ended = true;
                self.input.stop();
                return Err(error);
            }
            let pull = match &mut self.pending {
                Some(pull) => pull,
                None => self.pending.insert(self.start(false)?),
            };
            let pulled = tokio::select! {
                biased;
                error = gate.failed() => {
                    self.ended = true;
                    self.input.stop();
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
                    self.input.stop();
                    Ok(None)
                }
                Err(e) => {
                    let error = captured(e);
                    gate.fail(error.clone());
                    self.ended = true;
                    self.input.stop();
                    Err(error)
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
            self.input.stop();
            let clock = self.clock.clone();
            let pending = self.pending.take();
            let has_aclose =
                Python::attach(|py| access::AsyncClose::matches(self.host.iterator.bind(py)));
            let mut failed = false;
            let teardown = async {
                if let Some(mut pull) = pending {
                    pull.cancel();
                    // what the cancelled pull ends with is no reader's: the stream is closing
                    if let Err(error) = pull.settled().await {
                        log::debug!(target: lk_agents_abi::LOG_TARGET, "a pending pull ended while closing: {error}");
                    }
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
        self.input.stop();
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
/// loop that the loop's teardown awaits. Scheduled from the blocking I/O pool, so a drop holding
/// a lock never waits for the GIL. Lost only when the interpreter or the process runtime is gone.
fn close_later(located: impl FnOnce(Python<'_>) -> PyResult<Located> + Send + 'static) {
    crate::py_calls::off_thread("closing an abandoned host stream", move || {
        let scheduled = Python::try_attach(|py| {
            let (iterator, event_loop, ctx) = located(py)?;
            schedule_aclose(py, iterator, event_loop, ctx)
        });
        if let Some(Err(error)) = scheduled {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "closing an abandoned host stream failed: {error}");
        }
    });
}

/// Queues `iterator.aclose()` as a task of `event_loop` in `ctx`, when it has one, held until it
/// finishes (asyncio holds a task only weakly) and awaited by the loop's teardown
/// (`utils.aio.close_loop`).
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
        let task =
            access::EventLoop::create_task(loop_ref.bind(py), &coroutine, task_ctx.bind(py))?;
        access::AioModule::_keep_closing(&access::AioModule::import(py)?)?.call1((task,))?;
        PyResult::Ok(())
    })?;
    access::EventLoop::call_soon_threadsafe(event_loop.bind(py))?
        .call1((access::Context::run(ctx.bind(py))?, start))?;
    Ok(())
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
        started.cancel_later();
        let background = match lk_agents_runtime::process::env().background() {
            Ok(background) => background,
            Err(error) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "an unclaimed host stream could not be closed: {error}");
                return;
            }
        };
        let clock = Arc::clone(&background.clock);
        // A cleanup of the process's tasks: its own bound, then the stream's close is queued.
        background.cleanup("closing an unclaimed host stream", async move {
            let limit = lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
            match lk_agents_abi::timeout(&*clock, limit, started.settled()).await {
                Ok(Ok(value)) => close_later(move |py| {
                    let (event_loop, ctx) = ran_in.bind(py).extract()?;
                    Ok((value, event_loop, ctx))
                }),
                Ok(Err(_)) => {}
                Err(_) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "an unclaimed host stream's call did not settle in time; it is not closed")
                }
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
    stop: Arc<InputStop>,
    queue: Mutex<SinkQueue<I>>,
}

struct SinkQueue<I> {
    entries: VecDeque<Entry<I>>,
    /// A drain is scheduled on the loop and has not emptied the queue yet.
    scheduled: bool,
}

impl<I: Send + 'static> HostSink<I> {
    fn admit(&self, entry: Entry<I>) -> Result<(), ProviderError> {
        let schedule = {
            let mut queue = self.shared.queue.lock();
            let gate = &self.shared.stop.gate;
            gate.admit()?;
            let ending = matches!(entry, Entry::EndInput);
            if !ending && queue.entries.len() >= SINK_QUEUE {
                return Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    lk_agents_abi::errors::ErrorKind::Server,
                    format!(
                        "the host stream is not taking its input: {SINK_QUEUE} entries are waiting"
                    ),
                )));
            }
            if ending {
                gate.end();
            }
            queue.entries.push_back(entry);
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py)) {
            let e = captured(e);
            self.shared.stop.gate.fail(e.clone());
            return Err(e);
        }
        Ok(())
    }

    fn schedule(&self, py: Python<'_>) -> PyResult<()> {
        let (shared, lost) = (self.shared.clone(), self.shared.clone());
        let release = move || {
            let closed = lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Connection,
                "the host event loop closed",
            );
            lost.stop.gate.fail(closed.into());
            // Dropped after the guard: an item's host objects may run Python finalizers.
            let _dropped = {
                let mut queue = lost.queue.lock();
                queue.scheduled = false;
                std::mem::take(&mut queue.entries)
            };
        };
        let s = &self.shared;
        crate::py_calls::call_soon_or_lost(
            py,
            s.event_loop.bind(py),
            Some(s.ctx.bind(py)),
            move |py| drain(py, &shared),
            release,
        )
    }
}

/// On the loop, inside the stream's context: everything queued, in order, until the queue is
/// empty; the first failure stops delivery and is kept for the next push. An entry whose handling
/// is awaitable runs as a task of that context, and the drain resumes once it settled: the queue
/// stays scheduled meanwhile, so pushes queue behind it and the loop never blocks on it.
fn drain<I: Send + 'static>(py: Python<'_>, shared: &Arc<SinkShared<I>>) {
    loop {
        let entry = {
            let mut queue = shared.queue.lock();
            let entry = if shared.stop.stopped() { None } else { queue.entries.pop_front() };
            queue.scheduled = entry.is_some();
            entry
        };
        let Some(entry) = entry else { return discard(shared) };
        let handled = call_input(py, shared.target.bind(py), shared.to_host, entry)
            .and_then(|value| crate::py_calls::as_coroutine(&value));
        match handled {
            Ok(None) => {}
            Ok(Some(coroutine)) => {
                if let Err(e) = await_then_drain(py, shared, &coroutine) {
                    failed(py, shared, &e);
                }
                return;
            }
            Err(e) => return failed(py, shared, &e),
        }
    }
}

/// Runs `coroutine` as a task of the stream's context, published so a stop cancels it, then
/// resumes the drain in that same context.
fn await_then_drain<I: Send + 'static>(
    py: Python<'_>,
    shared: &Arc<SinkShared<I>>,
    coroutine: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let task =
        access::EventLoop::create_task(shared.event_loop.bind(py), coroutine, shared.ctx.bind(py))?;
    let resumed = Arc::clone(shared);
    let done = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
        let py = args.py();
        resumed.stop.in_flight.settled(py);
        match access::Task::result(&args.get_item(0)?)?.call0() {
            Ok(_) => drain(py, &resumed),
            // Cancelled by a stop: what is queued is discarded with the input. Any other
            // cancellation (its loop closing) fails the input, so the queue is never left
            // scheduled with nothing to drain it.
            Err(e)
                if resumed.stop.stopped()
                    && e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py) =>
            {
                discard(&resumed)
            }
            Err(e) => failed(py, &resumed, &e),
        }
        PyResult::Ok(())
    })?;
    let context = pyo3::types::PyDict::new(py);
    context.set_item("context", shared.ctx.bind(py))?;
    access::Task::add_done_callback(&task)?.call((done,), Some(&context))?;
    shared.stop.in_flight.publish(py, &task)
}

/// The input failed: kept for the next push, and what is queued is discarded.
fn failed<I>(py: Python<'_>, shared: &SinkShared<I>, error: &PyErr) {
    // Capturing reads the exception's attributes: Python code, run before the lock.
    shared.stop.gate.fail(crate::py_errors::capture(py, error));
    discard(shared);
}

/// Unschedules the queue once stopped or failed, dropping what it holds.
fn discard<I>(shared: &SinkShared<I>) {
    if !shared.stop.stopped() && shared.stop.gate.failure().is_none() {
        return;
    }
    // Dropped after the guard: an item's host objects may run Python finalizers.
    let _dropped = {
        let mut queue = shared.queue.lock();
        queue.scheduled = false;
        std::mem::take(&mut queue.entries)
    };
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
            shared.stop.gate.admit()?;
            if matches!(entry, Entry::EndInput) {
                shared.stop.gate.end();
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
                shared.stop.gate.fail(e.clone());
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
