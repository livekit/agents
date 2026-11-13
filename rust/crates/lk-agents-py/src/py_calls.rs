/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_host_derive::host(interface(module = "asyncio", attrs(get_running_loop)))]
    pub struct AsyncioModule;

    #[lk_agents_host_derive::host(interface(module = "contextvars", attrs(copy_context)))]
    pub struct ContextVarsModule;

    #[lk_agents_host_derive::host(interface(
        module = "inspect",
        attrs(isawaitable, iscoroutine, getattr_static)
    ))]
    pub struct InspectModule;

    #[lk_agents_host_derive::host(interface(host = "contextvars.Context", attrs(copy, run)))]
    pub struct Context;

    #[lk_agents_host_derive::host(interface(
        host = "asyncio.AbstractEventLoop",
        attrs(call_soon_threadsafe),
        methods(create_task(args(coro, context(kw))))
    ))]
    pub struct EventLoop;

    #[lk_agents_host_derive::host(interface(
        host = "asyncio.Task",
        attrs(cancel, cancelled, result, add_done_callback)
    ))]
    pub struct Task;

    #[lk_agents_host_derive::host(interface(module = "lk_agents_host", attrs(awaiting)))]
    pub struct AwaitableAdapterModule;

    #[lk_agents_host_derive::host(interface(module = "traceback", attrs(format_exception)))]
    pub struct TracebackModule;

    /// The host's current OpenTelemetry span, read at an entry and activated for a call.
    #[lk_agents_host_derive::host(interface(
        module = "livekit.agents.telemetry",
        attrs(_activate, _current)
    ))]
    pub struct TelemetryModule;

    /// The job the calling context runs in, read at an entry.
    #[lk_agents_host_derive::host(interface(module = "livekit.agents.job", attrs(_current_run)))]
    pub struct JobModule;
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

    /// [`cancel`](Self::cancel), off the calling thread: for a drop, which may hold a lock.
    pub(crate) fn cancel_later(&self) {
        self.in_flight.cancel_later();
    }
}

/// The host task of a call in flight, cancelled when the call is dropped before it settles.
struct InFlight {
    event_loop: Arc<Py<PyAny>>,
    child: Arc<Child>,
    settled: Arc<AtomicBool>,
}

/// A lock taken only while attached, so the GIL always comes first: the inverse order, which
/// deadlocks against the loop thread, cannot be written.
#[derive(Default)]
struct GilMutex<T>(Mutex<T>);

impl<T> GilMutex<T> {
    fn lock<'py>(&'py self, _: Python<'py>) -> parking_lot::MutexGuard<'py, T> {
        self.0.lock()
    }
}

/// A started call's host task, shared by the call, the loop callback that publishes the task and
/// the `_core` coroutine that started it.
#[derive(Default)]
pub(crate) struct Child {
    /// Strong: asyncio holds a pending task only weakly, and one awaiting what only it reaches
    /// would be collected mid-call. A loop's teardown cancels it (`utils.aio.close_loop`).
    task: GilMutex<Option<Py<PyAny>>>,
    /// Set without the GIL, at once: a call not started yet never starts.
    cancelled: AtomicBool,
}

impl Child {
    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// On the loop thread: `task` is the call's, cancelled at once if the call already was.
    pub(crate) fn publish(&self, py: Python<'_>, task: &Bound<'_, PyAny>) -> PyResult<()> {
        // Under the lock a cancel takes the task by: published first, or seen cancelled here.
        let cancel_now = {
            let mut slot = self.task.lock(py);
            let cancelled = self.cancelled();
            if !cancelled {
                *slot = Some(task.clone().unbind());
            }
            cancelled
        };
        if cancel_now {
            access::Task::cancel(task)?.call0()?;
        }
        Ok(())
    }

    /// On the loop thread: the published task settled, so a later cancel has nothing to reach.
    pub(crate) fn settled(&self, py: Python<'_>) {
        let task = self.task.lock(py).take();
        drop(task);
    }

    /// Cancelled from any thread: no task starts after this, and the published one is cancelled
    /// on its loop. Off the calling thread, which may hold a lock a GIL holder waits on.
    pub(crate) fn cancel_later(self: &Arc<Self>, event_loop: &Arc<Py<PyAny>>) {
        if self.cancelled.swap(true, Ordering::SeqCst) {
            return;
        }
        let (event_loop, child) = (event_loop.clone(), self.clone());
        off_thread("cancelling a host call", move || cancel_task(&event_loop, &child));
    }

    /// On the loop thread: cancelled now, its task once published.
    fn cancel_on_loop(&self, py: Python<'_>) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some(task) = self.task.lock(py).take() {
            let _ = access::Task::cancel(task.bind(py)).and_then(|cancel| cancel.call0());
        }
    }
}

tokio::task_local! {
    /// The host calls the `_core` coroutine being polled started.
    static STARTED: Arc<Children>;
}

/// The host calls a `_core` coroutine started: cancelled on the loop thread the moment its Python
/// caller cancels it, so none outlives the caller's resumption.
#[derive(Default)]
struct Children(Mutex<Vec<std::sync::Weak<Child>>>);

impl Children {
    fn add(&self, child: &Arc<Child>) {
        let mut children = self.0.lock();
        children.retain(|child| child.strong_count() > 0);
        children.push(Arc::downgrade(child));
    }

    fn cancel_on_loop(&self, py: Python<'_>) {
        let children = std::mem::take(&mut *self.0.lock());
        for child in children.iter().filter_map(std::sync::Weak::upgrade) {
            child.cancel_on_loop(py);
        }
    }
}

impl InFlight {
    /// On the calling thread: what a caller awaiting the call's settlement asks for.
    fn cancel(&self) {
        if !self.settled.load(Ordering::SeqCst) {
            self.child.cancelled.store(true, Ordering::SeqCst);
            cancel_task(&self.event_loop, &self.child);
        }
    }

    /// Off the calling thread, which may hold a lock a GIL holder waits on: what a drop asks for.
    fn cancel_later(&self) {
        if !self.settled.load(Ordering::SeqCst) {
            self.child.cancel_later(&self.event_loop);
        }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.cancel_later();
    }
}

/// Cancels the call's host task, once published and while it is alive.
fn cancel_task(event_loop: &Py<PyAny>, child: &Child) {
    // A finalizing interpreter runs no more tasks: nothing is left to cancel.
    let _ = Python::try_attach(|py| {
        let Some(task) = child.task.lock(py).take() else { return };
        if let Ok(cancel) = access::Task::cancel(task.bind(py)) {
            let _ = access::EventLoop::call_soon_threadsafe(event_loop.bind(py))
                .and_then(|call| call.call1((cancel,)));
        }
    });
}

/// Runs `work` on the blocking I/O pool, off the calling thread: what a drop does under the GIL,
/// so a drop holding a lock never waits for it. A cleanup of the process's tasks, which join it.
/// Lost only when the process runtime is gone.
pub(crate) fn off_thread(what: &'static str, work: impl FnOnce() + Send + 'static) {
    match lk_agents_runtime::process::env().background() {
        Ok(background) => background.cleanup(what, async move {
            if let Err(lost) =
                lk_agents_runtime::net::blocking_io::blocking_io().run((), work).await
            {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "{what}: {lost}");
            }
        }),
        Err(_) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "{what} skipped: no process runtime")
        }
    }
}

/// Schedules `drain` on `event_loop`, inside `ctx` when given. A loop that drops it unrun (it
/// closed, or refused it) runs `lost` instead, so a queue never waits on a drain that will not
/// come.
pub(crate) fn call_soon_or_lost(
    py: Python<'_>,
    event_loop: &Bound<'_, PyAny>,
    ctx: Option<&Bound<'_, PyAny>>,
    drain: impl FnOnce(Python<'_>) + Send + 'static,
    lost: impl FnOnce() + Send + 'static,
) -> PyResult<()> {
    let pending = Arc::new(Mutex::new(Unrun { drain: Some(drain), lost: Some(lost) }));
    let queued = pending.clone();
    let scheduled = (|| {
        let callback = PyCFunction::new_closure(py, None, None, move |args, _| {
            let drain = queued.lock().drain.take();
            if let Some(drain) = drain {
                drain(args.py());
            }
            PyResult::Ok(())
        })?;
        let call_soon = access::EventLoop::call_soon_threadsafe(event_loop)?;
        match ctx {
            Some(ctx) => call_soon.call1((access::Context::run(ctx)?, callback)),
            None => call_soon.call1((callback,)),
        }
    })();
    if let Err(e) = scheduled {
        // Now, not when the error's traceback lets the callback go.
        let lost = {
            let mut pending = pending.lock();
            pending.drain = None;
            pending.lost.take()
        };
        if let Some(lost) = lost {
            lost();
        }
        return Err(e);
    }
    Ok(())
}

/// A scheduled drain, and what runs if it is dropped before it ran.
struct Unrun<D, L: FnOnce()> {
    drain: Option<D>,
    lost: Option<L>,
}

impl<D, L: FnOnce()> Drop for Unrun<D, L> {
    fn drop(&mut self) {
        if self.drain.is_some()
            && let Some(lost) = self.lost.take()
        {
            lost();
        }
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

/// `value` as the coroutine `loop.create_task` takes, when it is awaitable; `None` for a value.
pub(crate) fn as_coroutine<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = value.py();
    let inspect = access::InspectModule::import(py)?;
    if !access::InspectModule::isawaitable(&inspect)?.call1((value,))?.is_truthy()? {
        return Ok(None);
    }
    if access::InspectModule::iscoroutine(&inspect)?.call1((value,))?.is_truthy()? {
        return Ok(Some(value.clone()));
    }
    awaiting(py)?.call1((value,)).map(Some)
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
    let child = Arc::new(Child::default());
    let _ = STARTED.try_with(|started| started.add(&child));
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
    let task_slot = child.clone();
    let call = Mutex::new(Some(call));
    let task_ctx = ctx.clone().unbind();
    let loop_ref = event_loop.clone().unbind();
    // Runs on the loop thread, inside `ctx`: the call, then its awaitable as a task of `ctx`.
    let start = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
        let py = args.py();
        let Some(call) = call.lock().take() else {
            return Ok(());
        };
        if task_slot.cancelled.load(Ordering::SeqCst) {
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
        let Some(coroutine) = as_coroutine(&value)? else {
            deliver(Ok(value.unbind()));
            return Ok(());
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
        task_slot.publish(py, &future)
    })?;
    access::EventLoop::call_soon_threadsafe(event_loop)?
        .call1((access::Context::run(ctx)?, start))?;
    Ok(Started {
        receiver,
        in_flight: InFlight { event_loop: Arc::new(event_loop.clone().unbind()), child, settled },
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

/// What a proxy taken from the host calls in: the context of the host call whose returned value
/// is being read ([`returning`]), else the running loop's context, else (no running loop) none,
/// so each call takes the context of the operation driving it.
pub fn proxy_context(py: Python<'_>) -> Option<HostRef> {
    RETURNING.with_borrow(Clone::clone).or_else(|| capture_context(py).ok())
}

thread_local! {
    static RETURNING: std::cell::RefCell<Option<HostRef>> = const { std::cell::RefCell::new(None) };
}

/// `read` a host call's returned value with `context` the one its proxies are taken in: a
/// capability inside a returned record calls in the context its call ran in, as a returned
/// capability does.
pub fn returning<T>(context: Option<HostRef>, read: impl FnOnce() -> T) -> T {
    struct Restore(Option<HostRef>);
    impl Drop for Restore {
        fn drop(&mut self) {
            RETURNING.set(self.0.take());
        }
    }
    let _restore = Restore(RETURNING.replace(context));
    read()
}

/// The context a proxy's call runs in: the one it captured, else its driving operation's.
pub fn calling_context(captured: &Option<HostRef>) -> Option<HostRef> {
    captured.clone().or_else(lk_agents_runtime::net::task::host_context)
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
        let span = crate::generated::records::telemetry::span_context_to_py(py, span)?;
        access::Context::run(&ctx)?.call1((activate, span))?;
    }
    Ok(ctx)
}

/// What an entered host call returned, and the `(loop, contextvars.Context)` it ran in.
pub struct Called {
    pub value: Py<PyAny>,
    pub(crate) ran_in: Option<Py<PyAny>>,
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
pub(crate) fn start_call(
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

/// The caller's span and job at a `_core` call's entry, recording to the process tracer: the
/// requests the call makes carry that job's identity.
pub fn entry_cx() -> PyResult<telemetry::Cx> {
    let (caller, job) = Python::attach(|py| Ok::<_, PyErr>((current_span(py), current_job(py)?)))?;
    Ok(lk_agents_runtime::telemetry::tracer().root().continuing(caller).with_job(job))
}

/// The job of the calling context: its `JobContext`'s run, once that run started its trace.
fn current_job(py: Python<'_>) -> PyResult<Option<telemetry::record::JobKey>> {
    static CURRENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let current = CURRENT.get_or_try_init(py, || {
        Ok::<_, PyErr>(access::JobModule::_current_run(&access::JobModule::import(py)?)?.unbind())
    })?;
    let run = current.bind(py).call0()?;
    if run.is_none() {
        return Ok(None);
    }
    Ok(run.cast::<crate::generated::worker::JobRun>()?.get().0.job())
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
                log::warn!(target: lk_agents_abi::LOG_TARGET, "{label} dropped: {NOTIFY_QUEUE} notifications are waiting for the host loop");
                return;
            }
            queue.entries.push_back((context.clone(), label, call));
            !std::mem::replace(&mut queue.scheduled, true)
        };
        if schedule && let Err(e) = Python::attach(|py| self.schedule(py, &context)) {
            // The drain was never queued: dropped, it already released the queue.
            log::warn!(target: lk_agents_abi::LOG_TARGET, "{label} failed: {e}");
        }
    }

    fn schedule(&self, py: Python<'_>, context: &HostRef) -> PyResult<()> {
        let (notices, lost) = (self.0.clone(), self.0.clone());
        let release = move || {
            // Dropped after the guard: a dropped call's host objects may run Python finalizers.
            let dropped = {
                let mut queue = lost.queue.lock();
                queue.scheduled = false;
                std::mem::take(&mut queue.entries)
            };
            if !dropped.is_empty() {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "{} notifications dropped: the host loop closed", dropped.len());
            }
        };
        let event_loop = match captured_pair(py, context) {
            Ok((event_loop, _)) => event_loop,
            Err(e) => {
                release();
                return Err(e);
            }
        };
        call_soon_or_lost(py, &event_loop, None, move |py| deliver(py, &notices), release)
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
                log::warn!(target: lk_agents_abi::LOG_TARGET, "{label} failed: {e}");
            }
        }
    }
}

/// A Rust future as the awaitable a `_core` coroutine method returns, polled on the process
/// runtime. The future, and every task it spawns, is driven by this entry: the loop and context
/// it resolves on are the host context a proxy taken with no running loop calls in.
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
    use pyo3_async_runtimes::generic;
    let locals = generic::get_current_locals::<ProcessRuntime>(py)?;
    let pair = PyTuple::new(py, [locals.event_loop(py), locals.context(py)])?;
    let entered = crate::py_objects::retain(pair.into_any().unbind());
    let started = Arc::new(Children::default());
    let fut =
        lk_agents_runtime::net::task::in_host_context(entered, STARTED.scope(started.clone(), fut));
    let future = generic::future_into_py_with_locals::<ProcessRuntime, _, _>(py, locals, fut)?;
    // Added before the caller awaits it, so it runs before the caller resumes.
    let on_done = PyCFunction::new_closure(py, None, None, move |args, _| {
        let future = args.get_item(0)?;
        if access::Task::cancelled(&future)?.call0()?.is_truthy()? {
            started.cancel_on_loop(args.py());
        }
        PyResult::Ok(())
    })?;
    access::Task::add_done_callback(&future)?.call1((on_done,))?;
    Ok(future)
}

/// The runtime's fork guard (`lk_agents_runtime::process::check_pid`) as a `RuntimeError`.
pub fn check_pid() -> PyResult<()> {
    lk_agents_runtime::process::check_pid()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// The process runtime's handle; a forked child's refusal raises.
fn process_handle() -> PyResult<tokio::runtime::Handle> {
    lk_agents_runtime::process::handle()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// The process runtime as `pyo3_async_runtimes`' runtime port.
struct ProcessRuntime;

tokio::task_local! {
    /// The loop and context a `_core` coroutine's future resolves on.
    static LOCALS: pyo3_async_runtimes::TaskLocals;
}

/// The host's executor: each `_core` coroutine's future is a task owned by its Python awaitable,
/// which `pyo3_async_runtimes` cancels with it.
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the executor primitive pyo3_async_runtimes drives the host's awaitables on"
)]
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
#[path = "py_calls_tests.rs"]
mod tests;
