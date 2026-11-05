//! Calls and futures crossing between Rust and the Python interpreter.

/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_abi::host(interface(module = "asyncio", attrs(get_running_loop)))]
    pub struct AsyncioModule;

    #[lk_agents_abi::host(interface(module = "contextvars", attrs(copy_context)))]
    pub struct ContextVarsModule;

    #[lk_agents_abi::host(interface(
        module = "inspect",
        attrs(isawaitable, iscoroutine, getattr_static)
    ))]
    pub struct InspectModule;

    #[lk_agents_abi::host(interface(host = "contextvars.Context", attrs(copy, run)))]
    pub struct Context;

    #[lk_agents_abi::host(interface(
        host = "asyncio.AbstractEventLoop",
        attrs(call_soon_threadsafe),
        methods(create_task(args(coro, context(kw))))
    ))]
    pub struct EventLoop;

    #[lk_agents_abi::host(interface(
        host = "asyncio.Task",
        attrs(cancel, result, add_done_callback)
    ))]
    pub struct Task;

    #[lk_agents_abi::host(interface(module = "lk_agents_host", attrs(awaiting)))]
    pub struct AwaitableAdapterModule;

    #[lk_agents_abi::host(interface(module = "traceback", attrs(format_exception)))]
    pub struct TracebackModule;

    /// The host's current OpenTelemetry span, read at an entry and activated for a call.
    #[lk_agents_abi::host(interface(
        module = "livekit.agents.telemetry",
        attrs(_activate, _current)
    ))]
    pub struct TelemetryModule;
}

use crate::generated::accessors as access;
use crate::py_objects::resolve;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::HostRef;
use lk_agents_abi::invoke::HostEntry;
use lk_agents_abi::invoke::HostNotices;
use lk_agents_abi::invoke::Refusal;
use lk_agents_abi::telemetry;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyCFunction;
use pyo3::types::PyTuple;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tokio::sync::oneshot;

/// Whether this thread holds the GIL. `PyGILState_Check` (CPython 3.4+) is outside the limited
/// API pyo3 exposes under abi3, but every interpreter the wheel supports exports it.
pub fn holds_gil() -> bool {
    unsafe extern "C" {
        fn PyGILState_Check() -> std::ffi::c_int;
    }
    // SAFETY: callable from any thread, initialized interpreter or not.
    unsafe { PyGILState_Check() == 1 }
}

/// The dispatch a generated proxy's call takes, in `context`, requiring no scope.
#[cfg(test)]
pub(super) fn call_in<F>(context: &HostRef, call: F) -> BoxFuture<'static, PyResult<Py<PyAny>>>
where
    F: for<'py> FnOnce(Python<'py>) -> PyResult<Invocation<'py>> + Send + 'static,
{
    let called = lk_agents_runtime::host::call(
        &Entry,
        "test",
        context,
        &[],
        &lk_agents_abi::telemetry::Cx::NONE,
        Box::new(call),
    );
    Box::pin(async move { called.await.map(|c| c.value) })
}

/// A host call started on its loop: resolves to its value, or the value its awaitable resolved to.
/// Dropping it before it settles cancels the host task; the task still settles on its loop.
pub(crate) struct Started {
    receiver: oneshot::Receiver<PyResult<Py<PyAny>>>,
    in_flight: InFlight,
}

impl Started {
    pub(crate) async fn settled(&mut self) -> PyResult<Py<PyAny>> {
        let result = (&mut self.receiver).await.unwrap_or_else(|_| {
            Err(pyo3::exceptions::PyRuntimeError::new_err("the host event loop dropped the call"))
        });
        self.in_flight.settled.store(true, Ordering::SeqCst);
        result
    }

    /// Asks the host task to cancel; it settles with its own outcome.
    pub(crate) fn cancel(&self) {
        self.in_flight.cancel();
    }
}

/// The host task of a call in flight, cancelled when the call is dropped before it settles.
struct InFlight {
    event_loop: Py<PyAny>,
    state: Arc<Mutex<InFlightState>>,
    settled: Arc<AtomicBool>,
}

#[derive(Default)]
struct InFlightState {
    task: Option<Py<PyAny>>,
    cancelled: bool,
}

impl InFlight {
    fn cancel(&self) {
        if self.settled.load(Ordering::SeqCst) {
            return;
        }
        // Startup holds the GIL before locking state; release state before acquiring the GIL.
        let task = {
            let mut state = self.state.lock();
            state.cancelled = true;
            state.task.take()
        };
        if let Some(task) = task {
            Python::attach(|py| {
                if let Ok(cancel) = access::Task::cancel(task.bind(py)) {
                    let _ = access::EventLoop::call_soon_threadsafe(self.event_loop.bind(py))
                        .and_then(|call| call.call1((cancel,)));
                }
            });
        }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// `awaiting(awaitable)`: any awaitable as the coroutine `loop.create_task` takes.
pub(crate) fn awaiting(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static AWAITING: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    AWAITING
        .get_or_try_init(py, || {
            let code = c"async def awaiting(awaitable):\n    return await awaitable\n";
            let module = PyModule::from_code(py, code, c"lk_agents_host.py", c"lk_agents_host")?;
            Ok(access::AwaitableAdapterModule::awaiting(&module)?.unbind())
        })
        .map(|f| f.bind(py))
}

/// Starts `call` on `event_loop` inside `ctx`, the context object itself (never a copy): the call
/// runs in it, and an awaitable it returns runs as a task of that same context, so what the call
/// sets before and after its awaits stays in `ctx`. Callers serialize their entries into one `ctx`.
pub(crate) fn start_in<F>(
    py: Python<'_>,
    event_loop: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
    call: F,
) -> PyResult<Started>
where
    F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
{
    let (sender, receiver) = oneshot::channel::<PyResult<Py<PyAny>>>();
    let state = Arc::new(Mutex::new(InFlightState::default()));
    let settled = Arc::new(AtomicBool::new(false));
    let deliver = {
        let sender = Arc::new(Mutex::new(Some(sender)));
        let settled = settled.clone();
        move |result: PyResult<Py<PyAny>>| {
            settled.store(true, Ordering::SeqCst);
            if let Some(sender) = sender.lock().take() {
                let _ = sender.send(result);
            }
        }
    };
    let task_slot = state.clone();
    let call = Mutex::new(Some(call));
    let task_ctx = ctx.clone().unbind();
    let loop_ref = event_loop.clone().unbind();
    // Runs on the loop thread, inside `ctx`: the call, then its awaitable as a task of `ctx`.
    let start = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
        let py = args.py();
        let Some(call) = call.lock().take() else {
            return Ok(());
        };
        if task_slot.lock().cancelled {
            deliver(Err(pyo3::exceptions::asyncio::CancelledError::new_err(())));
            return Ok(());
        }
        let value = match call(py) {
            Ok(value) => value,
            Err(e) => {
                deliver(Err(e));
                return Ok(());
            }
        };
        let inspect = access::InspectModule::import(py)?;
        if !access::InspectModule::isawaitable(&inspect)?.call1((&value,))?.is_truthy()? {
            deliver(Ok(value.unbind()));
            return Ok(());
        }
        let coroutine =
            if access::InspectModule::iscoroutine(&inspect)?.call1((&value,))?.is_truthy()? {
                value
            } else {
                awaiting(py)?.call1((value,))?
            };
        let future =
            access::EventLoop::create_task(loop_ref.bind(py), &coroutine, task_ctx.bind(py))?;
        let deliver = Mutex::new(Some(deliver.clone()));
        let done = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
            let finished = args.get_item(0)?;
            let result = access::Task::result(&finished)?.call0().map(Bound::unbind);
            if let Some(deliver) = deliver.lock().take() {
                deliver(result);
            }
            Ok(())
        })?;
        access::Task::add_done_callback(&future)?.call1((done,))?;
        let cancel_now = {
            let mut state = task_slot.lock();
            if !state.cancelled {
                state.task = Some(future.clone().unbind());
            }
            state.cancelled
        };
        if cancel_now {
            access::Task::cancel(&future)?.call0()?;
        }
        Ok(())
    })?;
    access::EventLoop::call_soon_threadsafe(event_loop)?
        .call1((access::Context::run(ctx)?, start))?;
    Ok(Started {
        receiver,
        in_flight: InFlight { event_loop: event_loop.clone().unbind(), state, settled },
    })
}

/// The running loop and a copy of the current context, retained: what a host object's proxy
/// runs its calls in when they carry no context of their own. Refused outside a running loop.
pub fn capture_context(py: Python<'_>) -> PyResult<HostRef> {
    let event_loop = access::AsyncioModule::get_running_loop(&(py.import("asyncio")?))?.call0()?;
    let ctx = access::ContextVarsModule::copy_context(&(py.import("contextvars")?))?.call0()?;
    let pair = pyo3::types::PyTuple::new(py, [event_loop, ctx])?;
    Ok(crate::py_objects::retain(pair.into_any().unbind()))
}

/// The `(loop, contextvars.Context)` pair `context` retained at its operation's entry.
pub(crate) fn captured_pair<'py>(
    py: Python<'py>,
    context: &HostRef,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let captured = resolve(py, context)?;
    captured.bind(py).extract()
}

/// A host call: builds the callable, its positional arguments and its keyword arguments, inside
/// the entered context.
pub type HostCall = Box<dyn for<'py> FnOnce(Python<'py>) -> PyResult<Invocation<'py>> + Send>;

/// What a [`HostCall`] calls: the callable, its positional and its keyword arguments.
pub type Invocation<'py> =
    (Bound<'py, PyAny>, Bound<'py, PyTuple>, Option<Bound<'py, pyo3::types::PyDict>>);

/// A copy of `ctx` with `span` the current OpenTelemetry span: what a host call runs in, and the
/// stream it returns is pulled in.
fn copy_in_span<'py>(
    py: Python<'py>,
    ctx: &Bound<'py, PyAny>,
    span: Option<telemetry::record::SpanContext>,
) -> PyResult<Bound<'py, PyAny>> {
    let ctx = access::Context::copy(ctx)?.call0()?;
    if let Some(span) = span {
        let activate = telemetry_hook(py, &ACTIVATE, access::TelemetryModule::_activate)?;
        let ids = (span.trace_id, span.span_id, span.sampled);
        access::Context::run(&ctx)?.call1((activate, ids.0, ids.1, ids.2))?;
    }
    Ok(ctx)
}

/// What an entered host call returned, and the `(loop, contextvars.Context)` it ran in.
pub struct Called {
    pub value: Py<PyAny>,
    ran_in: Option<Py<PyAny>>,
}

impl Called {
    /// The context a stream or proxy the call returned runs in: the very context the call ran
    /// in, as Python runs a call and its `async for` in one context, so what the call set before
    /// returning is visible to its generator. A call made in place ran in no copy of its own,
    /// and no span: a copy of its captured context.
    pub fn returned_context(&self, py: Python<'_>, captured: &HostRef) -> PyResult<HostRef> {
        let pair = match &self.ran_in {
            Some(pair) => pair.clone_ref(py),
            None => {
                let (event_loop, ctx) = captured_pair(py, captured)?;
                let ctx = copy_in_span(py, &ctx, None)?;
                pyo3::types::PyTuple::new(py, [event_loop, ctx])?.into_any().unbind()
            }
        };
        Ok(crate::py_objects::retain(pair))
    }
}

static ACTIVATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static CURRENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `livekit.agents.telemetry`'s `_activate` or `_current`, resolved once.
fn telemetry_hook<'py>(
    py: Python<'py>,
    slot: &'static PyOnceLock<Py<PyAny>>,
    get: fn(&Bound<'py, pyo3::types::PyModule>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<&'py Bound<'py, PyAny>> {
    slot.get_or_try_init(py, || Ok(get(&access::TelemetryModule::import(py)?)?.unbind()))
        .map(|f| f.bind(py))
}

/// An exception's formatted traceback, for the span that records it; `None` for one never raised.
pub(crate) fn traceback(value: &Bound<'_, PyAny>) -> Option<String> {
    if access::BaseException::traceback(value).ok()?.is_none() {
        return None;
    }
    let module = access::TracebackModule::import(value.py()).ok()?;
    let lines = access::TracebackModule::format_exception(&module).ok()?.call1((value,)).ok()?;
    Some(lines.extract::<Vec<String>>().ok()?.concat())
}

/// The current OpenTelemetry span of the calling context, when valid.
fn current_span(py: Python<'_>) -> Option<telemetry::record::SpanContext> {
    let current = telemetry_hook(py, &CURRENT, access::TelemetryModule::_current).ok()?;
    let current = current.call0().ok().filter(|c| !c.is_none())?;
    crate::generated::records::telemetry::span_context_from_py(&current).ok()?.valid()
}

/// Starts `call` in a copy of `context` with `span` current; the pair it runs in is returned with
/// its value.
fn start_call(
    context: &HostRef,
    span: Option<telemetry::record::SpanContext>,
    call: HostCall,
) -> PyResult<(Started, Py<PyAny>)> {
    Python::attach(|py| {
        let (event_loop, ctx) = captured_pair(py, context)?;
        let ctx = copy_in_span(py, &ctx, span)?;
        let started = start_in(py, &event_loop, &ctx, move |py| {
            let (callable, arguments, keywords) = call(py)?;
            callable.call(arguments, keywords.as_ref())
        })?;
        let pair = pyo3::types::PyTuple::new(py, [event_loop, ctx])?;
        Ok((started, pair.into_any().unbind()))
    })
}

/// The Python entry of the runtime's `HostInvoker`.
pub struct Entry;

impl HostEntry for Entry {
    type Call = HostCall;
    type Value = Called;
    type Error = PyErr;

    /// Runs `call` in a copy of `context`, once polled: resolves to the call's value, or the
    /// value its awaitable resolved to.
    fn enter(
        &self,
        context: &HostRef,
        span: Option<telemetry::record::SpanContext>,
        call: HostCall,
    ) -> BoxFuture<'static, PyResult<Called>> {
        let context = context.clone();
        Box::pin(async move {
            let (mut started, ran_in) = start_call(&context, span, call)?;
            let value = started.settled().await?;
            Ok(Called { value, ran_in: Some(ran_in) })
        })
    }

    fn enter_cancellable(
        &self,
        context: &HostRef,
        span: Option<telemetry::record::SpanContext>,
        call: HostCall,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'static, PyResult<Called>> {
        let context = context.clone();
        Box::pin(async move {
            if cancel.is_cancelled() {
                return Err(pyo3::exceptions::asyncio::CancelledError::new_err(()));
            }
            let (mut started, ran_in) = start_call(&context, span, call)?;
            let value = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    started.cancel();
                    started.settled().await
                }
                result = started.settled() => result,
            }?;
            Ok(Called { value, ran_in: Some(ran_in) })
        })
    }

    fn enter_now(&self, call: HostCall) -> PyResult<Called> {
        Python::attach(|py| {
            let (callable, arguments, keywords) = call(py)?;
            let value = callable.call(arguments, keywords.as_ref()).map(Bound::unbind)?;
            Ok(Called { value, ran_in: None })
        })
    }

    fn refused(&self, label: &'static str, refusal: Refusal) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{label}: {refusal}"))
    }
}

/// The [`CallContext`](lk_agents_abi::invoke::CallContext) of a `_core` call, captured at its
/// entry: the running loop and a copy of the caller's context, none outside a running loop.
pub fn entered() -> lk_agents_abi::invoke::CallContext {
    let host_context = Python::attach(|py| capture_context(py).ok());
    lk_agents_abi::invoke::CallContext { host_context, ..Default::default() }
}

/// The caller's span at a `_core` call's entry, recording to the process tracer.
pub fn entry_cx() -> telemetry::Cx {
    let caller = Python::attach(current_span);
    lk_agents_runtime::telemetry::tracer().root().continuing(caller)
}

/// Notifications waiting for the host loop past which a new lossy one is dropped (and logged).
pub const NOTIFY_QUEUE: usize = 256;

/// The `notify` calls of one proxy: delivered in call order, each in a copy of its own context,
/// by one drain scheduled at a time on the loop of the notice that found the queue idle. Never a
/// task per call, which could reorder them (a detector's audio frames).
#[derive(Clone, Default)]
pub struct Notifier(Arc<Notices>);

#[derive(Default)]
struct Notices {
    queue: Mutex<NoticeQueue>,
}

#[derive(Default)]
struct NoticeQueue {
    entries: VecDeque<(HostRef, &'static str, HostCall)>,
    /// A drain is scheduled on the loop and has not emptied the queue yet.
    scheduled: bool,
}

impl Notifier {
    fn queue(&self, context: HostRef, label: &'static str, lossy: bool, call: HostCall) {
        let schedule = {
            let mut queue = self.0.queue.lock();
            if lossy && queue.entries.len() >= NOTIFY_QUEUE {
                drop(queue);
                log::warn!(target: "livekit.agents", "{label} dropped: {NOTIFY_QUEUE} notifications are waiting for the host loop");
                return;
            }
            queue.entries.push_back((context.clone(), label, call));
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py, &context)) {
            // Dropped after the guard: a dropped call's host objects may run Python finalizers.
            let dropped = {
                let mut queue = self.0.queue.lock();
                queue.scheduled = false;
                std::mem::take(&mut queue.entries)
            };
            drop(dropped);
            log::warn!(target: "livekit.agents", "{label} failed: {e}");
        }
    }

    fn schedule(&self, py: Python<'_>, context: &HostRef) -> PyResult<()> {
        let (event_loop, _) = captured_pair(py, context)?;
        let notices = self.0.clone();
        let drain = PyCFunction::new_closure(py, None, None, move |args, _| {
            deliver(args.py(), &notices);
            PyResult::Ok(())
        })?;
        access::EventLoop::call_soon_threadsafe(&event_loop)?.call1((drain,))?;
        Ok(())
    }
}

/// A control notification (a flush, a clear, a session event) is never dropped: lost to a slow
/// loop it is a wrong outcome, and the queue only grows while the host loop stalls. A lossy one
/// (a stream's frame) is dropped, and logged, once [`NOTIFY_QUEUE`] wait for the host loop.
impl HostNotices for Notifier {
    type Call = HostCall;

    fn notify(&self, context: HostRef, label: &'static str, lossy: bool, call: HostCall) {
        self.queue(context, label, lossy, call);
    }
}

/// On the loop: every queued notification, in order, until the queue is empty. An awaitable a
/// notification returns runs as a task of its context copy, not awaited.
fn deliver(py: Python<'_>, notices: &Notices) {
    loop {
        let batch: Vec<(HostRef, &'static str, HostCall)> = {
            let mut queue = notices.queue.lock();
            if queue.entries.is_empty() {
                queue.scheduled = false;
                return;
            }
            queue.entries.drain(..).collect()
        };
        for (context, label, call) in batch {
            let delivered = (|| -> PyResult<()> {
                let (event_loop, ctx) = captured_pair(py, &context)?;
                let ctx = access::Context::copy(&ctx)?.call0()?;
                let (callable, args, keywords) = call(py)?;
                let mut run = vec![callable];
                run.extend(args.iter());
                let value =
                    access::Context::run(&ctx)?.call(PyTuple::new(py, run)?, keywords.as_ref())?;
                if access::InspectModule::isawaitable(&(access::InspectModule::import(py)?))?
                    .call1((&value,))?
                    .is_truthy()?
                {
                    access::EventLoop::create_task(
                        &event_loop,
                        &awaiting(py)?.call1((value,))?,
                        &ctx,
                    )?;
                }
                Ok(())
            })();
            if let Err(e) = delivered {
                log::warn!(target: "livekit.agents", "{label} failed: {e}");
            }
        }
    }
}

/// A Rust future as the awaitable a `_core` coroutine method returns, polled on the process
/// runtime.
pub fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    process_handle()?;
    boxed_future_into_py(py, Box::pin(fut))
}

// Share PyO3's scheduling and cancellation machinery across the generated futures with the
// same output type. Keep result conversion inside PyO3's existing completion path.
#[inline(never)]
fn boxed_future_into_py<'bound, T>(
    py: Python<'bound>,
    fut: BoxFuture<'static, PyResult<T>>,
) -> PyResult<Bound<'bound, PyAny>>
where
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    pyo3_async_runtimes::generic::future_into_py::<ProcessRuntime, _, _>(py, fut)
}

/// The runtime's fork guard (`lk_agents_runtime::process::check_pid`) as a `RuntimeError`.
pub fn check_pid() -> PyResult<()> {
    lk_agents_runtime::process::check_pid()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// The process runtime's handle; a forked child's refusal raises.
pub fn process_handle() -> PyResult<tokio::runtime::Handle> {
    lk_agents_runtime::process::handle()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// The process runtime as `pyo3_async_runtimes`' runtime port.
struct ProcessRuntime;

tokio::task_local! {
    /// The loop and context a `_core` coroutine's future resolves on.
    static LOCALS: pyo3_async_runtimes::TaskLocals;
}

impl pyo3_async_runtimes::generic::Runtime for ProcessRuntime {
    type JoinError = tokio::task::JoinError;
    type JoinHandle = tokio::task::JoinHandle<()>;

    fn spawn<F>(fut: F) -> Self::JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        // `future_into_py` checked this process may run it.
        process_handle().expect("the process runtime").spawn(fut)
    }

    /// On the blocking I/O pool, never Tokio's blocking pool; a panic fails the handle.
    fn spawn_blocking<F>(f: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        process_handle().expect("the process runtime").spawn(async move {
            let pool = lk_agents_runtime::net::blocking_io::blocking_io();
            if let Err(panicked) = pool.run((), f).await {
                std::panic::resume_unwind(Box::new(panicked));
            }
        })
    }
}

impl pyo3_async_runtimes::generic::ContextExt for ProcessRuntime {
    fn scope<F, R>(
        locals: pyo3_async_runtimes::TaskLocals,
        fut: F,
    ) -> std::pin::Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        Box::pin(LOCALS.scope(locals, fut))
    }

    fn get_task_locals() -> Option<pyo3_async_runtimes::TaskLocals> {
        LOCALS.try_with(Clone::clone).ok()
    }
}

/// `callable(*args, **kwargs)` in place, under the GIL the caller holds: a framework model, a view
/// over a host object, an enum member, a host function. What generated conversions build host
/// values with, and every binding helper's one call into host code.
pub fn call<'py, A>(
    callable: &Bound<'py, PyAny>,
    args: A,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>>
where
    A: pyo3::call::PyCallArgs<'py>,
{
    callable.call(args, kwargs)
}

/// `obj.name()` in place: a host object's framework accessor, read once when its proxy is taken.
pub fn read<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.call_method0(name)
}

#[cfg(test)]
mod tests {
    //! Interpreter call, context, cancellation and future-bridging regressions.

    use super::*;

    mod invocation {
        //! Against a live interpreter and a running loop: the context a call sees, what it leaves
        //! behind, failures and cancellation.
        use lk_agents_abi::{HostRef, invoke::HostNotices};

        use super::*;
        use pyo3::types::PyTuple;

        fn globals(py: Python<'_>) -> Bound<'_, PyAny> {
            crate::py_test_support::fixture(py, "invocation")
        }

        fn captured(globals: &Bound<'_, PyAny>, value: &str) -> HostRef {
            let pair = globals.getattr("capture").unwrap().call1((value,)).unwrap();
            globals.setattr("pair", &pair).unwrap();
            crate::py_objects::retain(pair.unbind())
        }

        fn callable(
            globals: Py<PyAny>,
            name: &'static str,
            args: Vec<&'static str>,
        ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<crate::py_calls::Invocation<'py>>
        + Send
        + 'static {
            move |py| {
                let f = globals.bind(py).getattr(name)?;
                Ok((f, PyTuple::new(py, args)?, None))
            }
        }

        // Catches: a sync callback run outside the captured context, off the loop thread, or one whose
        // context change leaks into the captured context or the caller's.
        #[tokio::test]
        async fn a_sync_call_runs_in_a_copy_of_the_captured_context_on_the_loop() {
            Python::initialize();
            let (globals, context) = Python::attach(|py| {
                let g = globals(py);
                let c = captured(&g, "captured");
                (g.unbind(), c)
            });
            let g = Python::attach(|py| globals.clone_ref(py));
            let value = call_in(&context, callable(g, "sync_call", vec!["sync"])).await.unwrap();
            let again = Python::attach(|py| globals.clone_ref(py));
            call_in(&context, callable(again, "sync_call", vec!["second"])).await.unwrap();
            Python::attach(|py| {
                assert_eq!(value.extract::<String>(py).unwrap(), "sync");
                let seen = globals.bind(py).getattr("seen").unwrap();
                let first: (String, bool) = seen.get_item("sync").unwrap().extract().unwrap();
                let second: (String, bool) = seen.get_item("second").unwrap().extract().unwrap();
                assert_eq!(first, ("captured".into(), true));
                assert_eq!(second, ("captured".into(), true), "the first call's change leaked");
                // The captured context is as the capture left it (the calls ran in copies).
                let var = globals.bind(py).getattr("var").unwrap();
                let ctx = globals.bind(py).getattr("pair").unwrap().get_item(1).unwrap();
                assert_eq!(ctx.get_item(var).unwrap().extract::<String>().unwrap(), "captured");
            });
        }

        // Catches: an awaitable run in a fresh context, so a change made before its first await is
        // lost after it (one context per resumption).
        #[tokio::test]
        async fn an_async_call_keeps_its_context_across_awaits() {
            Python::initialize();
            let (globals, context) = Python::attach(|py| {
                let g = globals(py);
                let c = captured(&g, "captured");
                (g.unbind(), c)
            });
            let g = Python::attach(|py| globals.clone_ref(py));
            call_in(&context, callable(g, "async_call", vec!["async"])).await.unwrap();
            Python::attach(|py| {
                let seen = globals.bind(py).getattr("seen").unwrap();
                assert_eq!(
                    seen.get_item("async").unwrap().extract::<String>().unwrap(),
                    "before await"
                );
            });
        }

        // Catches: a host exception lost or replaced, or a dropped call leaving its host task running.
        #[tokio::test]
        async fn failures_propagate_and_a_dropped_call_cancels_its_task() {
            Python::initialize();
            let (globals, context) = Python::attach(|py| {
                let g = globals(py);
                let c = captured(&g, "captured");
                (g.unbind(), c)
            });
            let g = Python::attach(|py| globals.clone_ref(py));
            let error = call_in(&context, callable(g, "failing", vec![])).await.unwrap_err();
            Python::attach(|py| {
                assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py))
            });

            let g = Python::attach(|py| globals.clone_ref(py));
            let call = call_in(&context, callable(g, "blocking", vec![]));
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(50), call).await.is_err()
            );
            for _ in 0..100 {
                let cancelled = Python::attach(|py| {
                    let blocked = globals.bind(py).getattr("blocked").unwrap();
                    blocked.get_item("cancelled").is_ok()
                });
                if cancelled {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!("the dropped call's host task was not cancelled");
        }

        // Catches: notifications reordered on their way to the host (a task per call), or one run
        // off the loop thread or outside a copy of its context.
        #[tokio::test]
        async fn notifications_reach_the_host_in_call_order_in_their_context() {
            Python::initialize();
            let (globals, context) = Python::attach(|py| {
                let g = globals(py);
                crate::py_test_support::call(
                    py,
                    "notifications_reach_the_host_in_call_order_in_their_context",
                    (&g,),
                )
                .unwrap();
                let c = captured(&g, "captured");
                (g.unbind(), c)
            });
            let notifier = Notifier::default();
            for i in 0..200 {
                let g = Python::attach(|py| globals.clone_ref(py));
                notifier.notify(
                    context.clone(),
                    "note",
                    false,
                    Box::new(move |py| {
                        let f = g.bind(py).getattr("note")?;
                        Ok((f, PyTuple::new(py, [i])?, None))
                    }),
                );
            }
            for _ in 0..200 {
                let order: Vec<(i64, String, bool)> = Python::attach(|py| {
                    globals.bind(py).getattr("order").unwrap().extract().unwrap()
                });
                if order.len() == 200 {
                    let expected: Vec<(i64, String, bool)> =
                        (0..200).map(|i| (i, "captured".to_owned(), true)).collect();
                    assert_eq!(order, expected);
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!("the notifications did not all reach the host");
        }

        // Catches: a control notification (a flush, a clear, a session event) dropped because the
        // host loop fell behind; only a lossy one (a stream's frame) may be.
        #[tokio::test]
        async fn a_notification_behind_a_stalled_loop_is_delivered_and_a_lossy_one_dropped() {
            Python::initialize();
            let (globals, context) = Python::attach(|py| {
                let g = globals(py);
                crate::py_test_support::call(
                    py,
                    "a_notification_behind_a_stalled_loop_is_delivered_and_a_lossy_one_dropped",
                    (&g,),
                )
                .unwrap();
                let c = captured(&g, "captured");
                (g.unbind(), c)
            });
            let notifier = Notifier::default();
            fn call(
                g: Py<PyAny>,
                name: &'static str,
                i: i64,
            ) -> impl for<'py> FnOnce(
                Python<'py>,
            )
                -> PyResult<crate::py_calls::Invocation<'py>>
            + Send
            + 'static {
                move |py| Ok((g.bind(py).getattr(name)?, PyTuple::new(py, [i])?, None))
            }
            let send = |name: &'static str, i: i64, lossy: bool| {
                let g = Python::attach(|py| globals.clone_ref(py));
                notifier.notify(context.clone(), "note", lossy, Box::new(call(g, name, i)))
            };
            send("stall", 0, false);
            let total = NOTIFY_QUEUE as i64 * 2;
            for i in 0..total {
                send("note", i, false);
            }
            for i in 0..total {
                send("note", total + i, true);
            }
            // The lossy ones arrived while every control one still waited: all dropped.
            let seen = || -> Vec<i64> {
                Python::attach(|py| globals.bind(py).getattr("seen").unwrap().extract().unwrap())
            };
            for _ in 0..300 {
                if seen().len() >= total as usize {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    assert_eq!(
                        seen(),
                        (0..total).collect::<Vec<_>>(),
                        "every control notification, in order"
                    );
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!("the control notifications did not all reach the host");
        }
    }

    mod cancellation {
        //! Cancellation before the host loop starts a call and while it publishes the host task.

        use parking_lot::Mutex;
        use std::sync::Arc;
        use std::task::{Context, Poll, Waker};

        use lk_agents_abi::invoke::HostEntry;
        use pyo3::prelude::*;
        use pyo3::types::{PyCFunction, PyTuple};

        struct ManualLoop(Py<PyAny>);

        impl ManualLoop {
            fn new() -> Self {
                Python::initialize();
                Python::attach(|py| {
                    let globals = crate::py_test_support::fixture(py, "manual_loop");
                    Self(globals.unbind())
                })
            }

            fn tick(&self) {
                Python::attach(|py| {
                    let globals = self.0.bind(py);
                    crate::py_test_support::call(py, "tick_loop", (globals,)).unwrap();
                });
            }

            fn entered(&self) -> bool {
                Python::attach(|py| {
                    self.0.bind(py).getattr("entered").unwrap().is_truthy().unwrap()
                })
            }
        }

        impl Drop for ManualLoop {
            fn drop(&mut self) {
                Python::attach(|py| {
                    if let Ok(loop_) = self.0.bind(py).getattr("loop") {
                        let _ = loop_.call_method0("close");
                    }
                });
            }
        }

        #[test]
        fn cancelled_queued_call_never_enters_the_callback_when_the_loop_resumes() {
            let host = ManualLoop::new();
            let (context, globals) = Python::attach(|py| {
                let globals = host.0.bind(py);
                let pair = PyTuple::new(
                    py,
                    [globals.getattr("loop").unwrap(), globals.getattr("context").unwrap()],
                )
                .unwrap();
                (crate::py_objects::retain(pair.into_any().unbind()), host.0.clone_ref(py))
            });
            let mut call = super::Entry.enter(
                &context,
                None,
                Box::new(move |py| {
                    Ok((globals.bind(py).getattr("callback")?, PyTuple::empty(py), None))
                }),
            );
            assert!(matches!(
                call.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            drop(call);
            host.tick();
            assert!(!host.entered(), "a cancelled queued call entered application code");
        }

        #[test]
        fn cancellation_during_task_creation_is_applied_when_the_task_is_published() {
            let host = ManualLoop::new();
            let started = Arc::new(Mutex::new(None::<super::Started>));
            Python::attach(|py| {
                let globals = host.0.bind(py);
                let cancel = started.clone();
                let hook = PyCFunction::new_closure(py, None, None, move |_, _| -> PyResult<()> {
                    cancel.lock().as_ref().unwrap().cancel();
                    Ok(())
                })
                .unwrap();
                globals.setattr("cancel_during_creation", hook).unwrap();
                crate::py_test_support::call(
                    py,
                    "cancellation_during_task_creation_is_applied_when_the_task_is_published",
                    (globals,),
                )
                .unwrap();
                let callback = globals.getattr("callback").unwrap().unbind();
                let call = super::start_in(
                    py,
                    &globals.getattr("loop").unwrap(),
                    &globals.getattr("context").unwrap(),
                    move |py| callback.bind(py).call0(),
                )
                .unwrap();
                *started.lock() = Some(call);
            });
            host.tick();
            host.tick();
            host.tick();
            assert!(!host.entered(), "cancellation during task publication was lost");
        }
    }

    mod parentage {
        //! Standard OTel child instrumentation through the real HostInvoker context boundary.

        use pyo3::prelude::*;
        use pyo3::types::PyTuple;

        use super::call_in;

        struct HostLoop(Py<PyAny>);

        impl Drop for HostLoop {
            fn drop(&mut self) {
                Python::attach(|py| {
                    if let Ok(close) = self.0.bind(py).getattr("close") {
                        let _ = close.call0();
                    }
                });
            }
        }

        fn callback(
            globals: Py<PyAny>,
            mode: &'static str,
        ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<crate::py_calls::Invocation<'py>>
        + Send
        + 'static {
            move |py| Ok((globals.bind(py).getattr("callback")?, PyTuple::new(py, [mode])?, None))
        }

        #[tokio::test]
        async fn invocation_trace_parentage_is_isolated_and_restored_after_failure_and_cancellation()
         {
            Python::initialize();
            let (host, first, second) = Python::attach(|py| {
                let globals = crate::py_test_support::fixture(py, "parentage");
                let capture = globals.getattr("capture").unwrap();
                let first = capture.call1(("one", 101u64, 11u64, true)).unwrap();
                let second = capture.call1(("two", 202u64, 22u64, true)).unwrap();
                (
                    HostLoop(globals.unbind()),
                    crate::py_objects::retain(first.unbind()),
                    crate::py_objects::retain(second.unbind()),
                )
            });
            fn owned(host: &HostLoop) -> Py<PyAny> {
                Python::attach(|py| host.0.clone_ref(py))
            }
            let (one, two) = tokio::join!(
                call_in(&first, callback(owned(&host), "success")),
                call_in(&second, callback(owned(&host), "success")),
            );
            Python::attach(|py| {
                assert_eq!(one.unwrap().extract::<(u64, bool)>(py).unwrap(), (101, true));
                assert_eq!(two.unwrap().extract::<(u64, bool)>(py).unwrap(), (202, true));
            });
            let failure = call_in(&first, callback(owned(&host), "failure")).await.unwrap_err();
            Python::attach(|py| {
                assert!(failure.is_instance_of::<pyo3::exceptions::PyValueError>(py))
            });
            let cancel = call_in(&second, callback(owned(&host), "cancel"));
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), cancel).await.is_err()
            );
            let settled = Python::attach(|py| {
                host.0
                    .bind(py)
                    .getattr("cancelled")
                    .unwrap()
                    .call_method1("wait", (2,))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            });
            assert!(settled, "HostInvoker must settle the cancelled callback");
            let unrecorded = Python::attach(|py| {
                let pair = host
                    .0
                    .bind(py)
                    .getattr("capture")
                    .unwrap()
                    .call1(("unrecorded", 303u64, 33u64, false))
                    .unwrap();
                crate::py_objects::retain(pair.unbind())
            });
            let result = call_in(&unrecorded, callback(owned(&host), "success")).await.unwrap();
            Python::attach(|py| {
                assert_eq!(result.extract::<(u64, bool)>(py).unwrap(), (303, false))
            });
            Python::attach(|py| {
                host.0.bind(py).getattr("verify").unwrap().call0().unwrap();
                for captured in [&first, &second] {
                    let pair = crate::py_objects::resolve(py, captured).unwrap();
                    let context = pair.bind(py).get_item(1).unwrap();
                    let var = host.0.bind(py).getattr("job").unwrap();
                    let value: String =
                        context.call_method1("get", (var,)).unwrap().extract().unwrap();
                    assert!(
                        value == "one" || value == "two",
                        "callback changed the captured context"
                    );
                }
            });
        }
    }

    mod futures {
        //! The shared Rust-to-asyncio bridge: results, task context, cancellation, and runtime cost.

        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        use pyo3::prelude::*;
        use pyo3::types::{PyCFunction, PyTuple};
        use pyo3_async_runtimes::generic::ContextExt;

        fn globals() -> Py<PyAny> {
            Python::initialize();
            Python::attach(|py| crate::py_test_support::fixture(py, "bridge_context").unbind())
        }

        struct FailingConversion;

        impl<'py> IntoPyObject<'py> for FailingConversion {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            fn into_pyobject(self, _: Python<'py>) -> PyResult<Self::Output> {
                Err(pyo3::exceptions::PyValueError::new_err("conversion failure"))
            }
        }

        #[test]
        fn async_bridge_preserves_results_errors_and_task_context() {
            let globals = globals();
            Python::attach(|py| {
                let globals = globals.bind(py);
                let job = globals.getattr("job").unwrap().unbind();
                let context = PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args: &Bound<'_, PyTuple>, _| {
                        let job = job.clone_ref(args.py());
                        super::future_into_py(args.py(), async move {
                            tokio::task::yield_now().await;
                            let locals = super::ProcessRuntime::get_task_locals().unwrap();
                            Python::attach(|py| {
                                locals
                                    .context(py)
                                    .call_method1("get", (job.bind(py),))?
                                    .extract::<String>()
                            })
                        })
                        .map(Bound::unbind)
                    },
                )
                .unwrap();
                globals.setattr("read_context", context).unwrap();
                let value =
                    PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                        let present = args.get_item(0)?.extract::<bool>()?;
                        super::future_into_py(
                            args.py(),
                            async move { Ok(present.then_some(42_u64)) },
                        )
                        .map(Bound::unbind)
                    })
                    .unwrap();
                globals.setattr("value", value).unwrap();
                let failure =
                    PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                        super::future_into_py(args.py(), async {
                            Err::<u64, _>(pyo3::exceptions::PyValueError::new_err("bridge failure"))
                        })
                        .map(Bound::unbind)
                    })
                    .unwrap();
                globals.setattr("failure", failure).unwrap();
                let conversion =
                    PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                        super::future_into_py(args.py(), async { Ok(FailingConversion) })
                            .map(Bound::unbind)
                    })
                    .unwrap();
                globals.setattr("conversion_failure", conversion).unwrap();
                crate::py_test_support::call(py, "verify_async_bridge", (globals,)).unwrap();
            });
        }

        struct Dropped(Arc<AtomicBool>);

        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        #[test]
        fn cancelling_the_python_future_drops_the_pending_rust_future() {
            let globals = globals();
            let entered = Arc::new(AtomicBool::new(false));
            let dropped = Arc::new(AtomicBool::new(false));
            Python::attach(|py| {
                let globals = globals.bind(py);
                let started = entered.clone();
                let finished = dropped.clone();
                let pending = PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args: &Bound<'_, PyTuple>, _| {
                        let entered = entered.clone();
                        let dropped = dropped.clone();
                        super::future_into_py(args.py(), async move {
                            let _dropped = Dropped(dropped);
                            entered.store(true, Ordering::SeqCst);
                            std::future::pending::<PyResult<u64>>().await
                        })
                        .map(Bound::unbind)
                    },
                )
                .unwrap();
                globals.setattr("pending", pending).unwrap();
                for (name, flag) in [("started", started), ("dropped", finished)] {
                    let read =
                        PyCFunction::new_closure(py, None, None, move |_, _| -> PyResult<bool> {
                            Ok(flag.load(Ordering::SeqCst))
                        })
                        .unwrap();
                    globals.setattr(name, read).unwrap();
                }
                crate::py_test_support::call(
                    py,
                    "cancelling_the_python_future_drops_the_pending_rust_future",
                    (globals,),
                )
                .unwrap();
            });
        }
    }
}
