//! A core that runs **on** a virtual-time asyncio loop.
//!
//! The reference's `virtual_time` tests (`tests/virtual_time.py`) run on a loop whose clock jumps
//! to the next timer whenever its `_ready` queue is empty. Work on tokio threads is invisible to
//! it, so a `wait_for(session.run(..), 60)` would jump 60 s while the core still works, and core
//! timers on the tokio clock would never see the jumps at all.
//!
//! On such a loop the core therefore runs on the loop thread: a small executor per loop
//! ([`LoopRuntime`]) whose every wake queues a pump callback in `_ready` (so the loop never counts
//! as idle while the core has work), and a clock whose `now` is `loop.time()` and whose waits are
//! `loop.call_at`. The ambient [`Clock`](crate::Clock) and [`Spawner`](crate::Spawner) of a binding
//! called on that loop are this runtime's ([`LoopRuntime::current`]); anywhere else nothing
//! changes.
//!
//! `LK_AGENTS_VIRTUAL_TIME` = `on` | `off` | `auto` (the default) picks the loops this applies to;
//! `auto` is a loop whose clock is more than a second off `time.monotonic()`.
//!
//! The GIL is held while the core runs, which is fine for tests. Nothing here blocks: a
//! synchronous wait on the core from the loop thread would deadlock, since the core runs there.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::ThreadId;
use std::time::{Duration, SystemTime};

use pyo3::prelude::*;
use pyo3::types::PyCFunction;

use crate::utils::spawner::{Abort, AbortHandle, BoxFuture, Spawner};
use crate::{Clock, Ts};

/// The env knob; see the module docs.
pub const VIRTUAL_TIME_ENV: &str = "LK_AGENTS_VIRTUAL_TIME";

/// A pump runs at most this many tasks before yielding to the loop's other callbacks.
const PUMP_BUDGET: usize = 256;

/// `_TICK_EPSILON` in `tests/virtual_time.py`: wall stamps stay strictly increasing.
const TICK_EPSILON: Duration = Duration::from_micros(1);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// --------------------------------------------------------------------- the float rule

/// `loop.time()` as a `Ts` past `origin`, rounded to the nanosecond, never negative.
fn loop_reading(origin: f64, time: f64) -> u64 {
    let ns = ((time - origin) * 1e9).round();
    if ns.is_finite() && ns > 0.0 { ns.min(u64::MAX as f64) as u64 } else { 0 }
}

/// The deadline `at` as a `loop.call_at` time.
fn loop_when(origin: f64, at: Ts) -> f64 {
    origin + at.as_nanos() as f64 / 1e9
}

/// Monotone, and never below a deadline that fired: `origin + at` read back through `loop.time()`
/// can round below `at`, and a driver that sees `now < at` after its timer fired re-arms forever.
#[derive(Default)]
struct HighWater(AtomicU64);

impl HighWater {
    fn raise(&self, ns: u64) -> u64 {
        self.0.fetch_max(ns, Ordering::AcqRel).max(ns)
    }

    fn get(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

// --------------------------------------------------------------------- the registry

struct Entry {
    event_loop: Py<PyAny>,
    /// `None`: a loop this does not apply to (remembered so it is asked once).
    runtime: Option<Arc<LoopRuntime>>,
}

/// Strong references, so a loop's id cannot be reused while it is listed; closed loops are swept
/// (and their tasks dropped) when a new loop registers.
static REGISTRY: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

thread_local! {
    /// The runtime whose pump is running on this thread.
    static CURRENT: RefCell<Option<Arc<LoopRuntime>>> = const { RefCell::new(None) };
}

struct CurrentGuard(Option<Arc<LoopRuntime>>);

impl CurrentGuard {
    fn enter(rt: &Arc<LoopRuntime>) -> Self {
        CurrentGuard(CURRENT.with(|c| c.borrow_mut().replace(Arc::clone(rt))))
    }
}

impl Drop for CurrentGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    On,
    Off,
    Auto,
}

fn mode() -> Mode {
    match std::env::var(VIRTUAL_TIME_ENV).map(|v| v.trim().to_ascii_lowercase()) {
        Ok(v) if matches!(v.as_str(), "on" | "1" | "true" | "yes") => Mode::On,
        Ok(v) if matches!(v.as_str(), "off" | "0" | "false" | "no") => Mode::Off,
        _ => Mode::Auto,
    }
}

/// A loop whose clock is more than a second away from `time.monotonic()` is not on real time.
fn applies_to(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> bool {
    match mode() {
        Mode::On => true,
        Mode::Off => false,
        Mode::Auto => (|| -> PyResult<bool> {
            let loop_time: f64 = event_loop.call_method0("time")?.extract()?;
            let monotonic: f64 = py.import("time")?.call_method0("monotonic")?.extract()?;
            Ok((loop_time - monotonic).abs() > 1.0)
        })()
        .unwrap_or(false),
    }
}

/// Drops every listed loop that is closed, and cancels its runtime's tasks. No lock is held
/// while Python runs.
fn sweep(py: Python<'_>) {
    let listed: Vec<Py<PyAny>> = lock(&REGISTRY).iter().map(|e| e.event_loop.clone_ref(py)).collect();
    let closed: Vec<usize> = listed
        .iter()
        .filter(|l| l.bind(py).call_method0("is_closed").and_then(|c| c.is_truthy()).unwrap_or(true))
        .map(|l| l.as_ptr() as usize)
        .collect();
    if closed.is_empty() {
        return;
    }
    let mut removed = Vec::new();
    lock(&REGISTRY).retain_mut(|e| {
        let gone = closed.contains(&(e.event_loop.as_ptr() as usize));
        if gone {
            removed.push((e.event_loop.clone_ref(py), e.runtime.take()));
        }
        !gone
    });
    for (_loop, runtime) in removed {
        if let Some(rt) = runtime {
            rt.shutdown();
        }
    }
}

fn lookup(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> Option<Arc<LoopRuntime>> {
    if let Some(e) = lock(&REGISTRY).iter().find(|e| e.event_loop.as_ptr() == event_loop.as_ptr()) {
        return e.runtime.clone();
    }
    sweep(py);
    let runtime = if applies_to(py, event_loop) {
        match LoopRuntime::new(py, event_loop) {
            Ok(rt) => Some(rt),
            Err(e) => {
                log::warn!("virtual time: could not attach to the running loop: {e}");
                None
            }
        }
    } else {
        None
    };
    lock(&REGISTRY).push(Entry { event_loop: event_loop.clone().unbind(), runtime: runtime.clone() });
    runtime
}

// --------------------------------------------------------------------- the runtime

/// Loop-thread work handed over from another thread.
type LoopOp = Box<dyn FnOnce(Python<'_>) + Send>;

/// One asyncio loop's executor and clock. See the module docs.
pub struct LoopRuntime {
    event_loop: Py<PyAny>,
    thread: ThreadId,
    /// `loop.time()` at `Ts::ZERO`.
    origin: f64,
    /// `time.time()` at `Ts::ZERO`, for a wall read off the loop thread.
    wall_origin: f64,
    high_water: HighWater,
    last_wall: Mutex<SystemTime>,
    queue: Mutex<VecDeque<Arc<Task>>>,
    /// Loop-thread work handed over from another thread (arming or cancelling a timer).
    ops: Mutex<Vec<LoopOp>>,
    /// Every live task, so a closed loop can drop the parked ones too.
    tasks: Mutex<HashMap<u64, Arc<Task>>>,
    next_id: AtomicU64,
    pump_pending: AtomicBool,
    closed: AtomicBool,
    pump: OnceLock<Py<PyAny>>,
}

impl LoopRuntime {
    /// The runtime of the loop running on this thread, when virtual time applies to it.
    pub fn current(py: Python<'_>) -> Option<Arc<LoopRuntime>> {
        if let Some(rt) = CURRENT.with(|c| c.borrow().clone()) {
            return Some(rt);
        }
        let running = py.import("asyncio").ok()?.call_method0("_get_running_loop").ok()?;
        if running.is_none() {
            return None;
        }
        lookup(py, &running)
    }

    fn new(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<Arc<Self>> {
        let origin: f64 = event_loop.call_method0("time")?.extract()?;
        let wall_origin: f64 = py.import("time")?.call_method0("time")?.extract()?;
        let rt = Arc::new(LoopRuntime {
            event_loop: event_loop.clone().unbind(),
            thread: std::thread::current().id(),
            origin,
            wall_origin,
            high_water: HighWater::default(),
            last_wall: Mutex::new(SystemTime::UNIX_EPOCH),
            queue: Mutex::new(VecDeque::new()),
            ops: Mutex::new(Vec::new()),
            tasks: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            pump_pending: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            pump: OnceLock::new(),
        });
        let weak = Arc::downgrade(&rt);
        let pump = PyCFunction::new_closure(py, Some(c"lk_loop_pump"), None, move |args, _kwargs| -> PyResult<()> {
            if let Some(rt) = weak.upgrade() {
                rt.run_pump(args.py());
            }
            Ok(())
        })?;
        let _ = rt.pump.set(pump.into_any().unbind());
        Ok(rt)
    }

    pub fn clock(self: &Arc<Self>) -> Arc<dyn Clock> {
        Arc::new(LoopClock(Arc::clone(self)))
    }

    pub fn spawner(self: &Arc<Self>) -> Arc<dyn Spawner> {
        Arc::new(LoopSpawner(Arc::clone(self)))
    }

    fn on_loop_thread(&self) -> bool {
        std::thread::current().id() == self.thread
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn spawn_task(self: &Arc<Self>, fut: BoxFuture<'static, ()>) -> Arc<Task> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let task = Arc::new(Task {
            id,
            fut: Mutex::new(Some(fut)),
            runtime: Arc::downgrade(self),
            queued: AtomicBool::new(false),
            aborted: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        });
        if self.is_closed() {
            task.finish_with(None);
            return task;
        }
        lock(&self.tasks).insert(id, Arc::clone(&task));
        // Never run inline: the first poll is a pump, like `loop.create_task`'s first step.
        task.schedule();
        task
    }

    /// Hands `op` to the loop thread (run at the next pump).
    fn submit(self: &Arc<Self>, op: impl FnOnce(Python<'_>) + Send + 'static) {
        if self.is_closed() {
            return;
        }
        lock(&self.ops).push(Box::new(op));
        self.request_pump();
    }

    /// Queues one pump in the loop's `_ready`. From another thread the GIL is taken on a blocking
    /// thread, never inside the waker (the waker's caller may hold a lock the loop thread wants).
    fn request_pump(self: &Arc<Self>) {
        if self.pump_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.on_loop_thread() {
            Python::attach(|py| self.call_soon_pump(py, false));
        } else {
            let me = Arc::clone(self);
            pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
                Python::attach(|py| me.call_soon_pump(py, true));
            });
        }
    }

    fn call_soon_pump(&self, py: Python<'_>, threadsafe: bool) {
        let Some(pump) = self.pump.get() else { return };
        let method = if threadsafe { "call_soon_threadsafe" } else { "call_soon" };
        if self.event_loop.bind(py).call_method1(method, (pump.clone_ref(py),)).is_err() {
            // The loop is closed: nothing will ever run here again.
            self.pump_pending.store(false, Ordering::Release);
            self.shutdown();
        }
    }

    fn run_pump(self: &Arc<Self>, py: Python<'_>) {
        self.pump_pending.store(false, Ordering::Release);
        if self.is_closed() {
            return;
        }
        let _current = CurrentGuard::enter(self);
        // So network I/O and `tokio::spawn` inside the core still reach the global runtime.
        let _tokio = pyo3_async_runtimes::tokio::get_runtime().enter();
        self.observe(py);
        let ops = std::mem::take(&mut *lock(&self.ops));
        for op in ops {
            op(py);
        }
        // Only what was queued when the pump began: a task woken now runs in the next pump, after
        // the callbacks this one queued (asyncio's own fairness).
        let batch = lock(&self.queue).len().min(PUMP_BUDGET);
        for _ in 0..batch {
            let Some(task) = lock(&self.queue).pop_front() else { break };
            task.run();
        }
        if !lock(&self.queue).is_empty() || !lock(&self.ops).is_empty() {
            self.request_pump();
        }
    }

    /// Drops every task (queued or parked) of a loop that is gone.
    fn shutdown(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        lock(&self.queue).clear();
        lock(&self.ops).clear();
        let tasks: Vec<Arc<Task>> = lock(&self.tasks).drain().map(|(_, t)| t).collect();
        for task in tasks {
            let fut = lock(&task.fut).take();
            task.finished.store(true, Ordering::Release);
            drop(fut);
        }
    }

    // ----------------------------------------------------------------- the clock

    /// Reads `loop.time()` (loop thread only) into the high-water mark.
    fn observe(&self, py: Python<'_>) -> u64 {
        match self.event_loop.bind(py).call_method0("time").and_then(|t| t.extract::<f64>()) {
            Ok(t) => self.high_water.raise(loop_reading(self.origin, t)),
            Err(_) => self.high_water.get(),
        }
    }

    fn now(&self) -> Ts {
        if self.on_loop_thread() {
            return Ts::from_nanos(Python::attach(|py| self.observe(py)));
        }
        // Off the loop thread: the last reading, without the GIL.
        Ts::from_nanos(self.high_water.get())
    }

    fn wall(&self) -> SystemTime {
        // The host's `time.time()` on the loop thread: a virtual-time test patches it.
        let secs = self
            .on_loop_thread()
            .then(|| {
                Python::attach(|py| -> PyResult<f64> { py.import("time")?.call_method0("time")?.extract() }).ok()
            })
            .flatten()
            .unwrap_or_else(|| self.wall_origin + self.now().as_nanos() as f64 / 1e9);
        let candidate = SystemTime::UNIX_EPOCH + Duration::try_from_secs_f64(secs.max(0.0)).unwrap_or_default();
        let mut last = lock(&self.last_wall);
        *last = if candidate > *last { candidate } else { *last + TICK_EPSILON };
        *last
    }

    /// Arms `loop.call_at` for `at` on the loop thread.
    fn arm(self: &Arc<Self>, py: Python<'_>, at: Ts, timer: &Arc<Mutex<Timer>>) {
        if lock(timer).dropped || self.is_closed() {
            return;
        }
        let fired = {
            let timer = Arc::clone(timer);
            let rt = Arc::downgrade(self);
            PyCFunction::new_closure(py, Some(c"lk_loop_timer"), None, move |_args, _kwargs| -> PyResult<()> {
                if let Some(rt) = rt.upgrade() {
                    rt.high_water.raise(at.as_nanos());
                }
                let waker = {
                    let mut t = lock(&timer);
                    t.fired = true;
                    t.handle = None;
                    t.waker.take()
                };
                if let Some(w) = waker {
                    w.wake();
                }
                Ok(())
            })
        };
        let Ok(fired) = fired else { return };
        // A closed loop refuses: the wait never ends, and the task is dropped with the loop.
        let Ok(handle) = self.event_loop.bind(py).call_method1("call_at", (loop_when(self.origin, at), fired)) else {
            return;
        };
        let cancel_now = {
            let mut t = lock(timer);
            if t.dropped {
                true
            } else {
                t.handle = Some(handle.clone().unbind());
                false
            }
        };
        if cancel_now {
            let _ = handle.call_method0("cancel");
        }
    }

    // ----------------------------------------------------------------- futures handed to Python

    /// `future_into_py` for this loop: the asyncio future is resolved directly on the loop thread,
    /// and cancelling it aborts the task.
    pub fn future_into_py<'py, F, T>(self: &Arc<Self>, py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
    where
        F: Future<Output = PyResult<T>> + Send + 'static,
        T: for<'a> IntoPyObject<'a> + Send + 'static,
    {
        let event_loop = self.event_loop.bind(py);
        let locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone()).copy_context(py)?;
        let py_fut = event_loop.call_method0("create_future")?;
        let target = py_fut.clone().unbind();
        let task = self.spawn_task(Box::pin(async move {
            let result = pyo3_async_runtimes::tokio::scope(locals, CatchUnwind(Box::pin(fut))).await;
            Python::attach(|py| {
                let target = target.bind(py);
                if target.call_method0("done").and_then(|d| d.is_truthy()).unwrap_or(true) {
                    return;
                }
                let outcome = result.and_then(|v| pyo3::IntoPyObjectExt::into_py_any(v, py));
                let _ = match outcome {
                    Ok(v) => target.call_method1("set_result", (v,)),
                    Err(e) => target.call_method1("set_exception", (e.into_value(py),)),
                };
            });
        }));
        let on_done = PyCFunction::new_closure(py, Some(c"lk_loop_future_done"), None, move |args, _kwargs| -> PyResult<()> {
            if args.get_item(0)?.call_method0("cancelled")?.is_truthy()? {
                task.abort();
            }
            Ok(())
        })?;
        py_fut.call_method1("add_done_callback", (on_done,))?;
        Ok(py_fut)
    }
}

/// A panic becomes the `RuntimeError` the awaiting host sees, rather than a future that never
/// resolves.
struct CatchUnwind<F>(Pin<Box<F>>);

impl<F, T> Future for CatchUnwind<F>
where
    F: Future<Output = PyResult<T>>,
{
    type Output = PyResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<PyResult<T>> {
        match std::panic::catch_unwind(AssertUnwindSafe(|| self.0.as_mut().poll(cx))) {
            Ok(p) => p,
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown error".to_owned());
                Poll::Ready(Err(pyo3::exceptions::PyRuntimeError::new_err(format!("rust future panicked: {msg}"))))
            }
        }
    }
}

/// `future_into_py` that runs on the loop when virtual time applies to it
/// ([`LoopRuntime::current`]), else `pyo3_async_runtimes::tokio::future_into_py`.
pub fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    match LoopRuntime::current(py) {
        Some(rt) => rt.future_into_py(py, fut),
        None => pyo3_async_runtimes::tokio::future_into_py(py, fut),
    }
}

// --------------------------------------------------------------------- tasks

struct Task {
    id: u64,
    /// `None` while polled (so an abort or a wake from inside the poll never locks it) and once done.
    fut: Mutex<Option<BoxFuture<'static, ()>>>,
    runtime: Weak<LoopRuntime>,
    queued: AtomicBool,
    aborted: AtomicBool,
    finished: AtomicBool,
}

impl Task {
    fn schedule(self: &Arc<Self>) {
        if self.finished.load(Ordering::Acquire) || self.queued.swap(true, Ordering::AcqRel) {
            return;
        }
        match self.runtime.upgrade() {
            Some(rt) if !rt.is_closed() => {
                lock(&rt.queue).push_back(Arc::clone(self));
                rt.request_pump();
            }
            // The loop is gone: nothing can poll this again.
            rt => self.finish_with(rt.as_ref()),
        }
    }

    /// On the loop thread, from a pump.
    fn run(self: &Arc<Self>) {
        self.queued.store(false, Ordering::Release);
        let Some(mut fut) = lock(&self.fut).take() else { return };
        let rt = self.runtime.upgrade();
        if self.aborted.load(Ordering::Acquire) {
            drop(fut);
            return self.finish_with(rt.as_ref());
        }
        let waker = Waker::from(Arc::clone(self));
        let mut cx = Context::from_waker(&waker);
        match std::panic::catch_unwind(AssertUnwindSafe(|| fut.as_mut().poll(&mut cx))) {
            Ok(Poll::Pending) => {
                let closed = rt.as_ref().is_none_or(|rt| rt.is_closed());
                if self.aborted.load(Ordering::Acquire) || closed {
                    drop(fut);
                    self.finish_with(rt.as_ref());
                } else {
                    *lock(&self.fut) = Some(fut);
                }
            }
            Ok(Poll::Ready(())) => {
                drop(fut);
                self.finish_with(rt.as_ref());
            }
            Err(_) => {
                log::error!("a task on the virtual-time loop panicked");
                drop(fut);
                self.finish_with(rt.as_ref());
            }
        }
    }

    fn finish_with(&self, rt: Option<&Arc<LoopRuntime>>) {
        self.finished.store(true, Ordering::Release);
        let fut = lock(&self.fut).take();
        drop(fut);
        if let Some(rt) = rt {
            lock(&rt.tasks).remove(&self.id);
        }
    }
}

impl Wake for Task {
    fn wake(self: Arc<Self>) {
        self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

impl Abort for Arc<Task> {
    /// Dropped at its next pump, on the loop thread (never inside the caller's own poll).
    fn abort(&self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        self.aborted.store(true, Ordering::Release);
        self.schedule();
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

// --------------------------------------------------------------------- the ambient pair

struct LoopSpawner(Arc<LoopRuntime>);

impl Spawner for LoopSpawner {
    fn spawn(&self, f: BoxFuture<'static, ()>) -> AbortHandle {
        AbortHandle::new(self.0.spawn_task(f))
    }
}

struct LoopClock(Arc<LoopRuntime>);

struct Timer {
    fired: bool,
    dropped: bool,
    waker: Option<Waker>,
    handle: Option<Py<PyAny>>,
}

/// `wait_until` on the loop clock: one `loop.call_at`, cancelled on drop.
struct LoopWait {
    rt: Arc<LoopRuntime>,
    at: Ts,
    timer: Option<Arc<Mutex<Timer>>>,
}

impl Future for LoopWait {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.rt.now() >= self.at {
            return Poll::Ready(());
        }
        if let Some(timer) = &self.timer {
            let mut t = lock(timer);
            if t.fired {
                return Poll::Ready(());
            }
            if !t.waker.as_ref().is_some_and(|w| w.will_wake(cx.waker())) {
                t.waker = Some(cx.waker().clone());
            }
            return Poll::Pending;
        }
        let timer = Arc::new(Mutex::new(Timer { fired: false, dropped: false, waker: Some(cx.waker().clone()), handle: None }));
        self.timer = Some(Arc::clone(&timer));
        let (rt, at) = (Arc::clone(&self.rt), self.at);
        if rt.on_loop_thread() {
            Python::attach(|py| rt.arm(py, at, &timer));
        } else {
            let armed = Arc::clone(&rt);
            rt.submit(move |py| armed.arm(py, at, &timer));
        }
        Poll::Pending
    }
}

impl Drop for LoopWait {
    fn drop(&mut self) {
        let Some(timer) = self.timer.take() else { return };
        let handle = {
            let mut t = lock(&timer);
            t.dropped = true;
            t.waker = None;
            t.handle.take()
        };
        let Some(handle) = handle else { return };
        if self.rt.on_loop_thread() {
            Python::attach(|py| {
                let _ = handle.bind(py).call_method0("cancel");
            });
        } else {
            self.rt.submit(move |py| {
                let _ = handle.bind(py).call_method0("cancel");
            });
        }
    }
}

#[async_trait::async_trait]
impl Clock for LoopClock {
    fn now(&self) -> Ts {
        self.0.now()
    }

    async fn wait_until(&self, at: Ts) {
        LoopWait { rt: Arc::clone(&self.0), at, timer: None }.await
    }

    fn wall(&self) -> SystemTime {
        self.0.wall()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a reading that rounds a fired deadline back below itself (the driver then sees
    // `now < at` after its timer fired and re-arms it forever), and the high-water mark failing to
    // repair it. The origins are realistic `loop.time()` values, where the f64 spacing is coarse.
    #[test]
    fn a_fired_deadline_is_never_read_back_below_itself() {
        let mut below = 0;
        for origin in [0.0, 12_345.678_901, 987_654.321_012_345, 1.0e7 + 0.1] {
            let hw = HighWater::default();
            for ms in 1..2_000u64 {
                let at = Ts::from_nanos(ms * 1_000_000 + ms % 7);
                let read = loop_reading(origin, loop_when(origin, at));
                if read < at.as_nanos() {
                    below += 1;
                }
                // What the timer callback does, then what `now` reports.
                hw.raise(at.as_nanos());
                assert!(hw.raise(read) >= at.as_nanos(), "origin {origin}, at {at:?}: read {read}");
            }
        }
        assert!(below > 0, "the float rule is untested: no reading rounded below its deadline");
    }

    // Catches: `now` going backwards when the loop's clock (or a stale off-thread reading) does.
    #[test]
    fn readings_are_monotone() {
        let hw = HighWater::default();
        let origin = 5_000.25;
        let mut last = 0;
        for t in [5_000.25, 5_001.0, 5_000.5, 4_000.0, 5_002.0, 5_001.999] {
            let now = hw.raise(loop_reading(origin, t));
            assert!(now >= last, "{now} < {last} at loop time {t}");
            last = now;
        }
        assert_eq!(last, 1_750_000_000, "the latest forward reading is kept");
        assert_eq!(loop_reading(origin, 1.0), 0, "before the origin reads as zero, not a wrap");
        assert_eq!(loop_reading(origin, f64::NAN), 0);
    }
}
