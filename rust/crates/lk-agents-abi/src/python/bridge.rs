//! A host object plus its event loop, as the generated `#[host(proxy)]` impls and the host-stream
//! halves drive it.

use pyo3::prelude::*;

use crate::errors::ProviderError;

use super::host_errors::capture;

pub trait HostBridge: Send + Sync + 'static {
    fn object(&self) -> &Py<PyAny>;
    fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals;

    fn await_call(
        &self,
        f: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    ) -> impl std::future::Future<Output = PyResult<Py<PyAny>>> + Send {
        let future = Python::attach(|py| -> PyResult<_> {
            let coro = f(self.object().bind(py))?;
            pyo3_async_runtimes::into_future_with_locals(self.task_locals(), coro)
        });
        async move { future?.await }
    }

    /// The only teardown of a host stream (dropping does not cancel it).
    fn aclose(&self) -> impl std::future::Future<Output = ()> + Send {
        let close = self.await_call(|o| o.call_method0("aclose"));
        async move {
            let _ = close.await;
        }
    }

    /// `StopAsyncIteration` is `Ok(None)`, a clean end. So is `CancelledError`: a host stream
    /// cancelled under the core (its task cancelled at close) ended, it did not fail; the
    /// reference never reports its own cancellation as a provider error.
    fn anext(&self) -> impl std::future::Future<Output = PyResult<Option<Py<PyAny>>>> + Send {
        let next = self.await_call(|o| o.call_method0("__anext__"));
        async move {
            match next.await {
                Ok(item) => Ok(Some(item)),
                Err(err) => Python::attach(|py| {
                    let ended = err.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py)
                        || err.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py);
                    if ended { Ok(None) } else { Err(err) }
                }),
            }
        }
    }
}

/// The binding's async runtime, submitted by `lk-agents-py`: the in-place expansions await on it
/// and take their ambient clock and spawner from it.
pub struct HostRuntime {
    /// A future resolving to a host value as the awaitable a `_core` coroutine method returns.
    pub future_into_py: for<'py> fn(
        Python<'py>,
        crate::BoxFuture<'static, PyResult<Py<PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>>,
    /// The running loop's clock when it is on virtual time.
    pub loop_clock: fn(Python<'_>) -> Option<std::sync::Arc<dyn crate::Clock>>,
    /// The running loop's executor when it is on virtual time, else the process runtime's.
    pub spawner: fn() -> std::sync::Arc<dyn crate::Spawner>,
}

inventory::collect!(HostRuntime);

fn linked() -> Option<&'static HostRuntime> {
    inventory::iter::<HostRuntime>.into_iter().next()
}

fn host_runtime() -> &'static HostRuntime {
    linked().expect("no host runtime is linked (lk-agents-py)")
}

/// A Rust future resolving to a host value as the awaitable a `_core` coroutine method returns, on
/// [`HostRuntime`].
pub fn future_into_py(
    py: Python<'_>,
    fut: impl std::future::Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
) -> PyResult<Bound<'_, PyAny>> {
    (host_runtime().future_into_py)(py, Box::pin(fut))
}

/// A parameter marked `#[host(ambient)]`: the runtime's own clock and spawner, never the host's.
pub trait Ambient {
    fn ambient() -> Self;
}

/// The scoped clock ([`HostClock`]), else the running loop's when it is on virtual time, else the
/// wall clock.
impl Ambient for std::sync::Arc<dyn crate::Clock> {
    fn ambient() -> Self {
        CLOCK_SCOPE
            .with(|s| s.borrow().last().cloned())
            .or_else(|| crate::utils::clock::LoopClock::ambient().0)
            .unwrap_or_else(|| std::sync::Arc::new(crate::RealClock::new()))
    }
}

thread_local! {
    static CLOCK_SCOPE: std::cell::RefCell<Vec<std::sync::Arc<dyn crate::Clock>>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub use crate::utils::clock::HostClock;

impl Ambient for crate::utils::clock::LoopClock {
    fn ambient() -> Self {
        let loop_clock = linked().map(|runtime| runtime.loop_clock);
        crate::utils::clock::LoopClock(loop_clock.and_then(|f| Python::try_attach(f).flatten()))
    }
}

/// Popped on drop.
pub struct ClockScope(bool);

impl ClockScope {
    /// `clock` is the ambient one until the scope drops: a core object the host builds meanwhile
    /// (inside a proxy's call) runs on the proxy's clock.
    pub fn push(clock: std::sync::Arc<dyn crate::Clock>) -> ClockScope {
        CLOCK_SCOPE.with(|s| s.borrow_mut().push(clock));
        ClockScope(true)
    }
}

impl Drop for ClockScope {
    fn drop(&mut self) {
        if self.0 {
            CLOCK_SCOPE.with(|s| s.borrow_mut().pop());
        }
    }
}

/// Autoref probe: `(&&ClockOf(v)).scope()` opens `v`'s clock when it is a [`HostClock`].
pub struct ClockOf<'a, T: ?Sized>(pub &'a T);

pub trait ClockDeclared {
    fn scope(&self) -> ClockScope;
}

impl<T: ?Sized + HostClock> ClockDeclared for &ClockOf<'_, T> {
    fn scope(&self) -> ClockScope {
        let clock = self.0.host_clock();
        CLOCK_SCOPE.with(|s| s.borrow_mut().push(clock));
        ClockScope(true)
    }
}

pub trait ClockNone {
    fn scope(&self) -> ClockScope;
}

impl<T: ?Sized> ClockNone for ClockOf<'_, T> {
    fn scope(&self) -> ClockScope {
        ClockScope(false)
    }
}

/// [`HostRuntime::spawner`]: one runtime, so a chain's tasks never split from its awaited futures.
impl Ambient for std::sync::Arc<dyn crate::Spawner> {
    fn ambient() -> Self {
        (host_runtime().spawner)()
    }
}

// --------------------------------------------------------------------- the host stream bridge

/// Task-locals are captured once, at construction: a provider's stream opens on a tokio worker
/// with no running loop to capture, and without them the Python stream looks empty.
pub struct HostHandle {
    obj: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
}

impl HostHandle {
    pub fn from_parts(obj: Py<PyAny>, locals: pyo3_async_runtimes::TaskLocals) -> Self {
        Self { obj, locals }
    }

    /// Captures the loop that is running *now*.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        Ok(HostHandle {
            obj,
            locals: pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?,
        })
    }

    pub fn child(&self, obj: Py<PyAny>) -> Self {
        HostHandle { obj, locals: self.locals.clone() }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        self.child(self.obj.clone_ref(py))
    }

    pub fn bind<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.obj.bind(py)
    }

    pub fn locals(&self) -> &pyo3_async_runtimes::TaskLocals {
        &self.locals
    }
}

impl HostBridge for HostHandle {
    fn object(&self) -> &Py<PyAny> {
        &self.obj
    }
    fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals {
        &self.locals
    }
}

/// Awaited coroutines are **cancelled** when the Rust future is dropped: `into_future_with_locals`
/// leaves the task running, so an abandoned handler would keep `session.run` going.
pub struct CancellingHandle {
    obj: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
}

impl CancellingHandle {
    /// Captures the loop that is running *now*.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        Ok(CancellingHandle {
            obj,
            locals: pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?,
        })
    }

    /// Another object on the same loop and context (one this object's call returned).
    pub fn child(&self, obj: Py<PyAny>) -> Self {
        CancellingHandle { obj, locals: self.locals.clone() }
    }

    /// `obj` on a loop and context the caller captured.
    pub fn from_parts(obj: Py<PyAny>, locals: pyo3_async_runtimes::TaskLocals) -> Self {
        CancellingHandle { obj, locals }
    }
}

/// The same object and loop, its awaits cancelled on drop.
impl From<HostHandle> for CancellingHandle {
    fn from(handle: HostHandle) -> Self {
        CancellingHandle { obj: handle.obj, locals: handle.locals }
    }
}

impl HostBridge for CancellingHandle {
    fn object(&self) -> &Py<PyAny> {
        &self.obj
    }

    fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals {
        &self.locals
    }

    fn await_call(
        &self,
        f: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    ) -> impl std::future::Future<Output = PyResult<Py<PyAny>>> + Send {
        let submitted = Python::attach(|py| -> PyResult<_> {
            cancelling::submit(&self.locals, f(self.obj.bind(py))?)
        });
        async move { submitted?.await }
    }
}

mod cancelling {
    use std::future::Future;
    use std::pin::Pin;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};

    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use tokio::sync::oneshot;

    #[derive(Default)]
    struct Slot {
        task: Option<Py<PyAny>>,
        cancelled: bool,
    }

    type Outcome = PyResult<Py<PyAny>>;

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// In the captured context: the job's context variables are visible to it.
    pub(super) fn submit(
        locals: &pyo3_async_runtimes::TaskLocals,
        awaitable: Bound<'_, PyAny>,
    ) -> PyResult<Submitted> {
        let py = awaitable.py();
        let slot = Arc::new(Mutex::new(Slot::default()));
        let (tx, rx) = oneshot::channel();
        let starter =
            Starter { awaitable: Some(awaitable.unbind()), tx: Some(tx), slot: Arc::clone(&slot) };
        let kwargs = PyDict::new(py);
        kwargs.set_item("context", locals.context(py))?;
        let event_loop = locals.event_loop(py);
        event_loop.call_method("call_soon_threadsafe", (Py::new(py, starter)?,), Some(&kwargs))?;
        Ok(Submitted { rx, slot, event_loop: event_loop.unbind(), done: false })
    }

    /// Runs on the loop: `ensure_future`, remember the task, report its outcome.
    #[pyclass]
    struct Starter {
        awaitable: Option<Py<PyAny>>,
        tx: Option<oneshot::Sender<Outcome>>,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        slot: Arc<Mutex<Slot>>,
    }

    #[pymethods]
    impl Starter {
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        fn __call__(&mut self, py: Python<'_>) -> PyResult<()> {
            let (Some(awaitable), Some(tx)) = (self.awaitable.take(), self.tx.take()) else {
                return Ok(());
            };
            if self.slot.lock().expect("slot").cancelled {
                // nobody is waiting any more: never start it
                let _ = awaitable.bind(py).call_method0("close");
                return Ok(());
            }
            let task = match py.import("asyncio")?.call_method1("ensure_future", (awaitable,)) {
                Ok(task) => task,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return Ok(());
                }
            };
            task.call_method1("add_done_callback", (Completer { tx: Mutex::new(Some(tx)) },))?;
            let cancel_now = {
                let mut slot = self.slot.lock().expect("slot");
                if !slot.cancelled {
                    slot.task = Some(task.clone().unbind());
                }
                slot.cancelled
            };
            if cancel_now {
                task.call_method0("cancel")?;
            }
            Ok(())
        }
    }

    #[pyclass(frozen)]
    struct Completer {
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        tx: Mutex<Option<oneshot::Sender<Outcome>>>,
    }

    #[pymethods]
    impl Completer {
        fn __call__(&self, task: &Bound<'_, PyAny>) -> PyResult<()> {
            let outcome = if task.call_method0("cancelled")?.is_truthy()? {
                Err(pyo3::exceptions::asyncio::CancelledError::new_err(()))
            } else {
                task.call_method0("result").map(Bound::unbind)
            };
            if let Some(tx) = self.tx.lock().expect("completer").take() {
                let _ = tx.send(outcome);
            }
            Ok(())
        }
    }

    pub(super) struct Submitted {
        rx: oneshot::Receiver<Outcome>,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        slot: Arc<Mutex<Slot>>,
        event_loop: Py<PyAny>,
        done: bool,
    }

    impl Future for Submitted {
        type Output = Outcome;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Outcome> {
            match Pin::new(&mut self.rx).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(outcome) => {
                    self.done = true;
                    Poll::Ready(outcome.unwrap_or_else(|_| {
                        Err(pyo3::exceptions::asyncio::CancelledError::new_err(()))
                    }))
                }
            }
        }
    }

    impl Drop for Submitted {
        fn drop(&mut self) {
            if self.done {
                return;
            }
            // Released before the GIL is taken: the starter holds the GIL while it locks the slot.
            let task = {
                let mut slot = self.slot.lock().expect("slot");
                slot.cancelled = true;
                slot.task.take()
            };
            if let Some(task) = task {
                let event_loop = &self.event_loop;
                Python::try_attach(|py| {
                    if let Ok(cancel) = task.bind(py).getattr("cancel") {
                        // a closed loop has nothing left to cancel
                        let _ = event_loop.bind(py).call_method1("call_soon_threadsafe", (cancel,));
                    }
                });
            }
        }
    }
}

// --------------------------------------------------------------------- proxies (`#[host(proxy)]`)

/// Implemented beside each error type a proxied trait returns.
pub trait FromHostErr {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self;
}

impl FromHostErr for ProviderError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        capture(py, err)
    }
}

/// Keeps the original exception, so re-raising it raises what the host raised.
impl FromHostErr for crate::errors::HostError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        super::host_errors::capture_host(py, err)
    }
}

pub fn host_err<E: FromHostErr>(err: PyErr) -> E {
    Python::attach(|py| E::from_host(py, &err))
}

/// Implemented for `dyn Trait` by the expansion, so a wrapper in any crate turns a host argument
/// into `Arc<dyn Trait>` without naming the proxy.
pub trait HostProxy {
    /// A registered native class ([`HostNative`]) as itself, anything else behind the proxy.
    fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<std::sync::Arc<Self>>;
}

/// A `_core` class whose value the core uses as it is rather than proxying; submitted
/// (`inventory`) by the owning crate and tried first by [`HostProxy::from_host`].
pub struct HostNative {
    /// A fn: a `TypeId` is not const-constructible.
    pub trait_id: fn() -> std::any::TypeId,
    pub read: fn(&Bound<'_, PyAny>) -> Option<Box<dyn std::any::Any>>,
}

inventory::collect!(HostNative);

pub fn native<T: ?Sized + 'static>(obj: &Bound<'_, PyAny>) -> Option<std::sync::Arc<T>> {
    let id = std::any::TypeId::of::<T>();
    for n in inventory::iter::<HostNative> {
        if (n.trait_id)() != id {
            continue;
        }
        if let Some(b) = (n.read)(obj)
            && let Ok(arc) = b.downcast::<std::sync::Arc<T>>()
        {
            return Some(*arc);
        }
    }
    None
}

/// The generated proxy of a declared `#[host(proxy)]` trait, submitted (`inventory`) by the
/// registration shell that generates it: the in-place exports take such a trait through
/// [`declared_proxy`], since the proxy runs on the runtime this crate cannot name.
pub struct DeclaredProxy {
    /// A fn: a `TypeId` is not const-constructible.
    trait_id: fn() -> std::any::TypeId,
    /// `Box<Arc<dyn Trait>>`.
    from_host: fn(&Bound<'_, PyAny>) -> PyResult<Box<dyn std::any::Any>>,
}

impl DeclaredProxy {
    pub const fn of<T: ?Sized + 'static>(
        from_host: fn(&Bound<'_, PyAny>) -> PyResult<Box<dyn std::any::Any>>,
    ) -> Self {
        DeclaredProxy { trait_id: std::any::TypeId::of::<T>, from_host }
    }
}

inventory::collect!(DeclaredProxy);

/// The registration shell submits its [`DeclaredProxy`]s through this crate's `inventory`.
pub use inventory;

/// `obj` as the declared trait `T`, through its generated proxy.
pub fn declared_proxy<T: ?Sized + 'static>(obj: &Bound<'_, PyAny>) -> PyResult<std::sync::Arc<T>> {
    let id = std::any::TypeId::of::<T>();
    let entry = inventory::iter::<DeclaredProxy>
        .into_iter()
        .find(|p| (p.trait_id)() == id)
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "no proxy of `{}` is registered (the registration shell generates it)",
                std::any::type_name::<T>()
            ))
        })?;
    let proxy = (entry.from_host)(obj)?;
    proxy.downcast::<std::sync::Arc<T>>().map(|p| *p).map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "the registered proxy of `{}` is another trait's",
            std::any::type_name::<T>()
        ))
    })
}

// --------------------------------------------------------------------- a host coroutine from any thread

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// From any thread, in the captured context. The guard cancels the Python task when dropped.
///
/// The context is entered through a copy: a `Context` is entered by one thread at a time, and the
/// loop thread enters the captured one for each callback scheduled in it, so entering it here
/// races those ("cannot enter context: ... is already entered").
pub fn schedule_coroutine(
    py: Python<'_>,
    locals: &pyo3_async_runtimes::TaskLocals,
    coro: Bound<'_, PyAny>,
) -> PyResult<(CancelPythonTask, tokio::sync::oneshot::Receiver<PyResult<Py<PyAny>>>)> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let asyncio = py.import("asyncio")?;
    let event_loop = locals.event_loop(py);
    // A loop that stopped runs nothing more: the coroutine is closed here, not left to the
    // collector's "coroutine ... was never awaited" (the worker's periodic calls at shutdown).
    let scheduled = if event_loop.call_method0("is_running")?.is_truthy()? {
        locals
            .context(py)
            .call_method0("copy")?
            .call_method1("run", (asyncio.getattr("run_coroutine_threadsafe")?, &coro, event_loop))
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err("the event loop is not running"))
    };
    let future = match scheduled {
        Ok(future) => future,
        Err(e) => {
            let _ = coro.call_method0("close");
            return Err(e);
        }
    };
    let sender = std::sync::Mutex::new(Some(sender));
    let callback = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let result = args.get_item(0)?.call_method0("result").map(Bound::unbind);
        if let Some(sender) = sender.lock().unwrap().take() {
            let _ = sender.send(result);
        }
        Ok::<_, PyErr>(())
    })?;
    future.call_method1("add_done_callback", (callback,))?;
    Ok((CancelPythonTask(future.unbind()), receiver))
}

/// `concurrent.futures.Future.cancel` is thread-safe and propagates to the asyncio task.
pub struct CancelPythonTask(Py<PyAny>);

impl Drop for CancelPythonTask {
    fn drop(&mut self) {
        Python::attach(|py| {
            let _ = self.0.bind(py).call_method0("cancel");
        });
    }
}
