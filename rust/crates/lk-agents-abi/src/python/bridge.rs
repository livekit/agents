//! A host object plus its event loop, as the generated `#[host(proxy)]` impls and the host-stream
//! halves drive it.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::errors::ProviderError;

use super::host_errors::{capture, captured};

pub trait HostBridge: Send + Sync + 'static {
    fn object(&self) -> &Py<PyAny>;
    fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals;

    /// This object when it has the `_core_*` surface (`probe`), else a duck-typed view applying the
    /// host class's own `_core_*` functions (`livekit.agents._core_view.duck`); `host` is its path.
    fn core_surface(self, host: &str, probe: &str) -> PyResult<Self>
    where
        Self: Sized,
    {
        let _ = (host, probe);
        Ok(self)
    }

    /// `obj.a.b` (`""` is `obj`): a pure read touches no `asyncio` object, so any thread will do.
    fn read<T>(&self, path: &str, fold: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> PyResult<T> {
        Python::attach(|py| {
            let mut v = self.object().bind(py).clone();
            for part in path.split('.').filter(|p| !p.is_empty()) {
                v = v.getattr(part)?;
            }
            fold(&v)
        })
    }

    /// `Ok` means handed over, not run (see `HostHandle::call_in_loop`). Arguments are converted
    /// off the loop.
    fn call_soon(
        &self,
        method: &str,
        args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let method = self.object().bind(py).getattr(method)?;
            let args = args(py)?;
            let call: Vec<Bound<'_, PyAny>> = std::iter::once(method).chain(args.iter()).collect();
            let call = pyo3::types::PyTuple::new(py, call)?;
            self.task_locals().event_loop(py).call_method1("call_soon_threadsafe", call)?;
            Ok(())
        })
    }

    /// No GIL is held across the await.
    fn await_method(
        &self,
        method: &str,
        args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
    ) -> impl std::future::Future<Output = PyResult<Py<PyAny>>> + Send {
        self.await_call(move |o| o.call_method1(method, args(o.py())?))
    }

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

    /// On the loop thread: base-class stream constructors need a running loop.
    fn open(
        &self,
        f: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
    ) -> impl std::future::Future<Output = Result<HostHandle, crate::errors::ProviderError>> + Send
    {
        let host = Python::attach(|py| HostHandle::from_parts(self.object().clone_ref(py), self.task_locals().clone()));
        async move {
            let obj = Python::attach(|py| host.bind(py).clone().unbind());
            let stream = host.run_on_loop(move |py| Ok(f(obj.bind(py))?.unbind())).await?;
            Ok(host.child(stream))
        }
    }

    /// The only teardown of a host stream (dropping does not cancel it).
    fn aclose(&self) -> impl std::future::Future<Output = ()> + Send {
        let close = self.await_call(|o| o.call_method0("aclose"));
        async move {
            let _ = close.await;
        }
    }

    /// `StopAsyncIteration` is `Ok(None)`, a clean end.
    fn anext(&self) -> impl std::future::Future<Output = PyResult<Option<Py<PyAny>>>> + Send {
        let next = self.await_call(|o| o.call_method0("__anext__"));
        async move {
            match next.await {
                Ok(item) => Ok(Some(item)),
                Err(err) => Python::attach(|py| {
                    if err.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) { Ok(None) } else { Err(err) }
                }),
            }
        }
    }
}

/// A parameter marked `#[host(ambient)]`: the runtime's own clock and spawner, never the host's.
pub trait Ambient {
    fn ambient() -> Self;
}

/// The scoped clock ([`HostClock`]), else the running loop's when it is on virtual time
/// ([`LoopRuntime`](super::LoopRuntime)), else the wall clock.
impl Ambient for std::sync::Arc<dyn crate::Clock> {
    fn ambient() -> Self {
        CLOCK_SCOPE.with(|s| s.borrow().last().cloned()).unwrap_or_else(|| {
            match Python::try_attach(super::LoopRuntime::current).flatten() {
                Some(rt) => rt.clock(),
                None => std::sync::Arc::new(crate::RealClock::new()),
            }
        })
    }
}

thread_local! {
    static CLOCK_SCOPE: std::cell::RefCell<Vec<std::sync::Arc<dyn crate::Clock>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A value that says which clock the host objects handed with it are on (a session, a fake room).
/// A wrapper scopes it over the conversions after it, so a proxy built next uses that clock.
pub trait HostClock {
    fn host_clock(&self) -> std::sync::Arc<dyn crate::Clock>;
}

impl<T: HostClock + ?Sized> HostClock for std::sync::Arc<T> {
    fn host_clock(&self) -> std::sync::Arc<dyn crate::Clock> {
        (**self).host_clock()
    }
}

/// Popped on drop.
pub struct ClockScope(bool);

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

/// The running loop's executor when it is on virtual time ([`LoopRuntime`](super::LoopRuntime)),
/// else `pyo3_async_runtimes::tokio`'s runtime: a second runtime would split a chain's tasks from
/// its awaited futures.
impl Ambient for std::sync::Arc<dyn crate::Spawner> {
    fn ambient() -> Self {
        match Python::try_attach(super::LoopRuntime::current).flatten() {
            Some(rt) => rt.spawner(),
            None => std::sync::Arc::new(crate::TokioSpawner::new(pyo3_async_runtimes::tokio::get_runtime().handle().clone())),
        }
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
        Ok(HostHandle { obj, locals: pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)? })
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

    /// A direct call from a tokio worker fails (`push_text` creates a task), and `loop.call_soon`
    /// from a foreign thread leaves the item unwoken for ~1 s. `Ok(())` means handed to the loop.
    pub fn call_in_loop(&self, py: Python<'_>, name: &str, arg: Option<Bound<'_, PyAny>>) -> Result<(), ProviderError> {
        let arg = arg.map(Bound::unbind);
        self.call_soon(name, move |py| PyTuple::new(py, arg.map(|a| a.into_bound(py)))).map_err(|e| capture(py, &e))
    }

    /// Never call it from the loop thread itself: the callback cannot run while the caller blocks.
    pub async fn run_on_loop<T, F>(&self, f: F) -> Result<T, ProviderError>
    where
        F: FnOnce(Python<'_>) -> PyResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<T, ProviderError>>();
        // `FnOnce` in a `Fn` callback: taking the contents makes a second call a no-op.
        let once = std::sync::Mutex::new(Some((f, tx)));
        Python::attach(|py| -> PyResult<()> {
            let callback = pyo3::types::PyCFunction::new_closure(
                py,
                Some(c"lk_host_run_on_loop"),
                None,
                move |args, _kwargs| -> PyResult<()> {
                    if let Some((f, tx)) = once.lock().expect("run_on_loop poisoned").take() {
                        let py = args.py();
                        // Captured on the loop thread while the exception is live, so the chain can
                        // re-raise the original.
                        let _ = tx.send(f(py).map_err(|e| capture(py, &e)));
                    }
                    Ok(())
                },
            )?;
            self.locals.event_loop(py).call_method1("call_soon_threadsafe", (callback,))?;
            Ok(())
        })
        .map_err(captured)?;
        // No GIL is held across this await, so the loop thread can run the callback.
        rx.await.unwrap_or_else(|_| {
            Err(crate::errors::ApiError::new("the host event loop shut down before it could run the call").into())
        })
    }
}

impl HostBridge for HostHandle {
    fn object(&self) -> &Py<PyAny> {
        &self.obj
    }
    fn task_locals(&self) -> &pyo3_async_runtimes::TaskLocals {
        &self.locals
    }

    fn core_surface(self, host: &str, probe: &str) -> PyResult<Self> {
        Python::attach(|py| {
            let obj = self.obj.bind(py);
            if obj.hasattr(probe)? {
                return Ok(self.clone_ref(py));
            }
            let (module, class) = host.rsplit_once('.').unwrap_or(("builtins", host));
            let class = py.import(module)?.getattr(class)?;
            let view = py.import("livekit.agents._core_view")?.getattr("duck")?.call1((obj, class))?;
            Ok(self.child(view.unbind()))
        })
    }
}

/// The drain half of a host stream. The converter is a `fn` pointer, which keeps this `Send`.
pub struct HostSource<T> {
    handle: HostHandle,
    convert: fn(&Bound<'_, PyAny>) -> PyResult<T>,
    /// `close`'s `deadline` is on this clock; any other would make the bound real seconds in a
    /// virtual-clock test.
    clock: std::sync::Arc<dyn crate::Clock>,
}

impl<T> HostSource<T> {
    pub fn new(
        handle: HostHandle,
        convert: fn(&Bound<'_, PyAny>) -> PyResult<T>,
        clock: std::sync::Arc<dyn crate::Clock>,
    ) -> Self {
        HostSource { handle, convert, clock }
    }
}

#[async_trait::async_trait]
impl<T: Send> crate::StreamSource<T> for HostSource<T> {
    /// `StopAsyncIteration` is not captured (a clean end must leave no side-table entry); every
    /// other exception is, so the original can be re-raised.
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        let item = self.handle.anext().await.map_err(captured)?;
        let convert = self.convert;
        Python::attach(|py| item.map(|o| convert(o.bind(py))).transpose().map_err(|e| capture(py, &e)))
    }

    /// Past `deadline` the coroutine is abandoned, not cancelled.
    async fn close(self: Box<Self>, deadline: crate::Ts) -> crate::Closed {
        crate::Closed::bounded(&*self.clock, deadline, self.handle.aclose()).await
    }

    /// `SynthesizeStream._started_time`, which is `0` until `_mark_started()`; a stream without the
    /// attribute (not a synthesis stream) has none.
    fn started_time(&self) -> Option<f64> {
        let read = self.handle.read("_started_time", |v| v.extract::<f64>());
        read.ok().filter(|t| *t != 0.0)
    }
}

/// `&self` and `Sync`: shared while the source is drained elsewhere, as the reference's two tasks
/// share one stream (`tts/tts.py:543`). Never drops; the loss policy is the driver's.
pub struct HostSink<T> {
    handle: HostHandle,
    to_py: fn(Python<'_>, &T) -> PyResult<Py<PyAny>>,
}

impl<T> HostSink<T> {
    pub fn new(handle: HostHandle, to_py: fn(Python<'_>, &T) -> PyResult<Py<PyAny>>) -> Self {
        HostSink { handle, to_py }
    }
}

impl<T: Send + Sync> crate::StreamSink<T> for HostSink<T> {
    fn push(&self, item: T) -> Result<(), ProviderError> {
        Python::attach(|py| {
            // Converted here, not in the loop callback, to keep marshalling off the host's
            // critical path at frame rate.
            let arg = (self.to_py)(py, &item).map_err(|e| capture(py, &e))?;
            self.handle.call_in_loop(py, "_core_push", Some(arg.into_bound(py)))
        })
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Python::attach(|py| self.handle.call_in_loop(py, "flush", None))
    }

    /// A failure means the loop is gone, which the source reports as end of stream.
    fn end_input(&self) {
        let _ = self.handle.call_soon("end_input", |py| Ok(PyTuple::empty(py)));
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
        Ok(CancellingHandle { obj, locals: pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)? })
    }

    /// Another object on the same loop and context (one this object's call returned).
    pub fn child(&self, obj: Py<PyAny>) -> Self {
        CancellingHandle { obj, locals: self.locals.clone() }
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
        let submitted = Python::attach(|py| -> PyResult<_> { cancelling::submit(&self.locals, f(self.obj.bind(py))?) });
        async move { submitted?.await }
    }
}

mod cancelling {
    use std::future::Future;
    use std::pin::Pin;
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

    /// In the captured context: the job's context variables are visible to it.
    pub(super) fn submit(locals: &pyo3_async_runtimes::TaskLocals, awaitable: Bound<'_, PyAny>) -> PyResult<Submitted> {
        let py = awaitable.py();
        let slot = Arc::new(Mutex::new(Slot::default()));
        let (tx, rx) = oneshot::channel();
        let starter = Starter { awaitable: Some(awaitable.unbind()), tx: Some(tx), slot: Arc::clone(&slot) };
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
        slot: Arc<Mutex<Slot>>,
    }

    #[pymethods]
    impl Starter {
        fn __call__(&mut self, py: Python<'_>) -> PyResult<()> {
            let (Some(awaitable), Some(tx)) = (self.awaitable.take(), self.tx.take()) else { return Ok(()) };
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
                    Poll::Ready(outcome.unwrap_or_else(|_| Err(pyo3::exceptions::asyncio::CancelledError::new_err(()))))
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
    /// `clock` `None` is the ambient clock.
    fn from_host(obj: &Bound<'_, PyAny>, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<std::sync::Arc<Self>>;

    /// The view a running proxy hands back for another trait, on `handle`'s loop.
    fn view(handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<Box<Self>>;
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

/// A returned host stream or trait view. The expansion cannot see through a type alias
/// (`SttSource`), so it asks the type ([`ProxyValue`]).
pub trait ProxyReturn: Sized {
    fn from_proxy(obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<Self>;
}

fn clock_or_ambient(clock: Option<std::sync::Arc<dyn crate::Clock>>) -> std::sync::Arc<dyn crate::Clock> {
    clock.unwrap_or_else(<std::sync::Arc<dyn crate::Clock> as Ambient>::ambient)
}

impl<T> ProxyReturn for Box<dyn crate::StreamSource<T>>
where
    T: Send + 'static,
    T: for<'a, 'py> FromPyObject<'a, 'py>,
    for<'a, 'py> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    fn from_proxy(_obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<Self> {
        Ok(Box::new(HostSource::new(handle, |i| i.extract::<T>().map_err(Into::into), clock_or_ambient(clock))))
    }
}

impl<I, T> ProxyReturn for (Box<dyn crate::StreamSink<I>>, Box<dyn crate::StreamSource<T>>)
where
    I: Send + Sync + 'static,
    for<'x, 'py> &'x I: IntoPyObject<'py>,
    T: Send + 'static,
    T: for<'a, 'py> FromPyObject<'a, 'py>,
    for<'a, 'py> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    fn from_proxy(obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<Self> {
        let sink = HostSink::new(handle.clone_ref(obj.py()), |py, v: &I| -> PyResult<Py<PyAny>> {
            pyo3::IntoPyObjectExt::into_py_any(v, py)
        });
        let source = <Box<dyn crate::StreamSource<T>> as ProxyReturn>::from_proxy(obj, handle, clock)?;
        Ok((Box::new(sink), source))
    }
}

impl<Tr: ?Sized + HostProxy> ProxyReturn for std::sync::Arc<Tr> {
    fn from_proxy(_obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<Self> {
        Tr::view(handle, clock).map(std::sync::Arc::from)
    }
}

/// Autoref dispatch: `(&&&ProxyValue::<T>::new()).read(obj, handle, clock)` -- a [`ProxyReturn`],
/// a class (a clone of its value), else its own `FromPyObject`.
#[doc(hidden)]
pub struct ProxyValue<T>(std::marker::PhantomData<T>);

impl<T> ProxyValue<T> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ProxyValue(std::marker::PhantomData)
    }
}

#[doc(hidden)]
pub trait ProxyValueReturn<T> {
    fn read(&self, obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T>;
}

impl<T: ProxyReturn> ProxyValueReturn<T> for &&ProxyValue<T> {
    fn read(&self, obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T> {
        T::from_proxy(obj, handle, clock)
    }
}

#[doc(hidden)]
pub trait ProxyValuePlain<T> {
    fn read(&self, obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T>;
}

#[doc(hidden)]
pub trait ProxyValueClass<T> {
    fn read(&self, obj: &Bound<'_, PyAny>, handle: HostHandle, clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T>;
}

impl<T: pyo3::PyClass + Clone> ProxyValueClass<T> for &ProxyValue<T> {
    fn read(&self, obj: &Bound<'_, PyAny>, _handle: HostHandle, _clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T> {
        Ok(obj.extract::<pyo3::PyRef<'_, T>>()?.clone())
    }
}

impl<T> ProxyValuePlain<T> for ProxyValue<T>
where
    T: for<'a, 'py> FromPyObject<'a, 'py>,
    for<'a, 'py> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    fn read(&self, obj: &Bound<'_, PyAny>, _handle: HostHandle, _clock: Option<std::sync::Arc<dyn crate::Clock>>) -> PyResult<T> {
        obj.extract::<T>().map_err(Into::into)
    }
}

// --------------------------------------------------------------------- a clock handed to a host provider

/// Submitted (`inventory`) by the crate owning `_core.SessionClock`, so a crate below can bind one.
pub struct SessionClockClass(pub for<'py> fn(Python<'py>, std::sync::Arc<dyn crate::Clock>) -> PyResult<Bound<'py, PyAny>>);

inventory::collect!(SessionClockClass);

/// Duck-typed: a plugin with its own sockets declares no `_bind_session_clock`. An error from a
/// declared hook propagates.
pub fn bind_host_provider(py: Python<'_>, obj: &Bound<'_, PyAny>, clock: &std::sync::Arc<dyn crate::Clock>) -> PyResult<()> {
    if !obj.hasattr("_bind_session_clock")? {
        return Ok(());
    }
    let class = inventory::iter::<SessionClockClass>
        .into_iter()
        .next()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no `_core.SessionClock` class is registered"))?;
    let handle = (class.0)(py, std::sync::Arc::clone(clock))?;
    obj.call_method1("_bind_session_clock", (handle,))?;
    Ok(())
}

// --------------------------------------------------------------------- a host coroutine from any thread

/// From any thread, in the captured context. The guard cancels the Python task when dropped.
pub fn schedule_coroutine(
    py: Python<'_>,
    locals: &pyo3_async_runtimes::TaskLocals,
    coro: Bound<'_, PyAny>,
) -> PyResult<(CancelPythonTask, tokio::sync::oneshot::Receiver<PyResult<Py<PyAny>>>)> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let asyncio = py.import("asyncio")?;
    let future = locals.context(py).call_method1(
        "run",
        (asyncio.getattr("run_coroutine_threadsafe")?, coro, locals.event_loop(py)),
    )?;
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
