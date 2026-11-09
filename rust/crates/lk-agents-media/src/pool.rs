#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "media's worker pool is the one threaded module of the sans-I/O cores"
)]

use std::collections::VecDeque;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// An idle thread waits this long for its next job before it exits.
const IDLE_KEEPALIVE: Duration = Duration::from_secs(10);

/// A kind of job with its own ceiling inside a pool's total.
pub trait Class: Copy + Debug + Send + Sync + 'static {
    fn index(self) -> usize;
}

/// A pool with one class: only the total bounds it.
impl Class for () {
    fn index(self) -> usize {
        0
    }
}

/// The media pool's classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobClass {
    Decoder,
    Recording,
    Image,
    Formatting,
    EotModel,
    VadModel,
}

impl Class for JobClass {
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolError<C: Class> {
    #[error("worker limits must be positive and each class must fit the total ceiling")]
    InvalidLimits,
    #[error("worker capacity exhausted for {0:?}")]
    Exhausted(C),
}

pub type MediaPool = Pool<JobClass>;
pub type MediaPoolError = PoolError<JobClass>;

/// Per-class ceilings bound long-lived resources independently of short jobs.
#[derive(Debug, Clone, Copy)]
pub struct MediaPoolLimits {
    pub total: usize,
    pub decoders: usize,
    pub recordings: usize,
    pub images: usize,
    pub formatting: usize,
    pub eot_models: usize,
    pub vad_models: usize,
}

impl Default for MediaPoolLimits {
    fn default() -> Self {
        Self {
            total: 64,
            decoders: 32,
            recordings: 16,
            images: 4,
            formatting: 8,
            eot_models: 1,
            vad_models: 32,
        }
    }
}

impl MediaPool {
    pub fn new(limits: MediaPoolLimits) -> Result<Self, MediaPoolError> {
        let classes = [
            limits.decoders,
            limits.recordings,
            limits.images,
            limits.formatting,
            limits.eot_models,
            limits.vad_models,
        ];
        Pool::with_limits("lk-media-worker", limits.total, &classes)
    }
}

/// The process's media pool, started on first use.
pub fn media_pool() -> MediaPool {
    static POOL: PerProcess<MediaPool> =
        PerProcess::new(|| MediaPool::new(MediaPoolLimits::default()).expect("default limits"));
    POOL.get()
}

/// A process-wide value made on first use in each process: a forked child, whose copy has no
/// threads and may hold state (a lock, a permit) its parent's threads owned, makes its own
/// without touching the parent's, which is leaked. Reading it takes no lock, so a lock held at
/// the fork cannot block the child.
pub struct PerProcess<T> {
    current: AtomicPtr<(u32, T)>,
    make: fn() -> T,
}

impl<T: Clone> PerProcess<T> {
    pub const fn new(make: fn() -> T) -> Self {
        Self { current: AtomicPtr::new(std::ptr::null_mut()), make }
    }

    pub fn get(&self) -> T {
        let pid = std::process::id();
        loop {
            let current = self.current.load(Ordering::Acquire);
            // SAFETY: a published value is leaked, never freed.
            if let Some((owner, value)) = unsafe { current.as_ref() }
                && *owner == pid
            {
                return value.clone();
            }
            let made = Box::into_raw(Box::new((pid, (self.make)())));
            match self.current.compare_exchange(current, made, Ordering::AcqRel, Ordering::Acquire)
            {
                // SAFETY: just published and never freed.
                Ok(_) => return unsafe { (*made).1.clone() },
                // Another thread of this process published first: ours was never shared.
                // SAFETY: `made` came from Box::into_raw above and nothing else holds it.
                Err(_) => drop(unsafe { Box::from_raw(made) }),
            }
        }
    }
}

type Task = Box<dyn FnOnce() + Send>;

struct State {
    used: Box<[usize]>,
    total: usize,
    tasks: VecDeque<Task>,
    idle: usize,
    capacity_waiters: Vec<Waker>,
}

struct Shared {
    /// The process the pool's threads run in: a forked child's inherited permits and jobs leave
    /// the parent's state alone.
    pid: u32,
    thread_name: &'static str,
    total: usize,
    limits: Box<[usize]>,
    state: Mutex<State>,
    work: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn fits(&self, state: &State, class: usize) -> bool {
        state.used[class] < self.limits[class] && state.total < self.total
    }

    /// A forked child's copy of its parent's pool: no thread serves it and a parent's thread may
    /// have held its lock at the fork, so the child never locks it.
    fn inherited(&self) -> bool {
        self.pid != std::process::id()
    }

    fn release(&self, class: usize) {
        if self.inherited() {
            return;
        }
        let waiters = {
            let mut state = self.lock();
            state.used[class] -= 1;
            state.total -= 1;
            std::mem::take(&mut state.capacity_waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
    }

    fn run(self: &Arc<Self>, task: Task) {
        let spawn = {
            let mut state = self.lock();
            state.tasks.push_back(task);
            state.tasks.len() > state.idle
        };
        if !spawn {
            self.work.notify_one();
            return;
        }
        // One thread per queued job beyond the idle ones, so no job waits behind a long-lived
        // one; every job holds a permit, so threads never outnumber the total ceiling.
        let shared = self.clone();
        let spawned =
            std::thread::Builder::new().name(self.thread_name.into()).spawn(move || work(shared));
        if spawned.is_err() {
            // No thread will run it: the job settles lost instead of waiting on one.
            let task = self.lock().tasks.pop_back();
            drop(task);
        }
    }
}

/// A bounded pool of threads. Clones share the same class and total capacity.
pub struct Pool<C> {
    shared: Arc<Shared>,
    class: PhantomData<fn(C)>,
}

impl<C> Clone for Pool<C> {
    fn clone(&self) -> Self {
        Self { shared: self.shared.clone(), class: PhantomData }
    }
}

/// A class reservation. Jobs retain it until they settle; dropping it releases the slot and
/// wakes capacity waiters.
pub struct Permit<C: Class> {
    shared: Arc<Shared>,
    class: C,
}

impl<C: Class> Drop for Permit<C> {
    fn drop(&mut self) {
        self.shared.release(self.class.index());
    }
}

pub use lk_agents_abi::errors::JobLost;

struct Slot<T> {
    result: Option<Result<T, JobLost>>,
    waker: Option<Waker>,
}

/// One job's result, written once by its [`Settle`].
struct JobCell<T> {
    /// The process whose pool queued the job: a forked child never sees it settle.
    pid: u32,
    slot: Mutex<Slot<T>>,
    ready: Condvar,
}

impl<T> JobCell<T> {
    /// The slot, or none in a forked child, whose copy of the job will never settle.
    fn lock(&self) -> Option<MutexGuard<'_, Slot<T>>> {
        (self.pid == std::process::id())
            .then(|| self.slot.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// A job's producer side, owned by its queued closure: it releases the job's capacity, then
/// settles the job with its result or, dropped without one (never run, unwound), with
/// [`JobLost`].
struct Settle<T, C: Class> {
    cell: Arc<JobCell<T>>,
    permit: Option<Permit<C>>,
}

impl<T, C: Class> Settle<T, C> {
    fn finish(mut self, result: T) {
        self.settle(Ok(result));
    }

    fn settle(&mut self, result: Result<T, JobLost>) {
        // The permit is present until the first settlement.
        let Some(permit) = self.permit.take() else { return };
        drop(permit);
        let Some(mut slot) = self.cell.lock() else { return };
        slot.result = Some(result);
        let waker = slot.waker.take();
        drop(slot);
        self.cell.ready.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T, C: Class> Drop for Settle<T, C> {
    fn drop(&mut self) {
        self.settle(Err(JobLost));
    }
}

/// The completion of one admitted job. Awaiting it observes the result; dropping it leaves the
/// job running with its capacity.
pub struct Job<T> {
    cell: Arc<JobCell<T>>,
}

impl<T> Future for Job<T> {
    type Output = Result<T, JobLost>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(mut slot) = self.cell.lock() else { return Poll::Ready(Err(JobLost)) };
        match slot.result.take() {
            Some(result) => Poll::Ready(result),
            None => {
                slot.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl<T> Job<T> {
    /// Observe settlement from another worker or a synchronous caller.
    /// Async callers must await the receipt instead of blocking their executor.
    pub fn wait(self) -> Result<T, JobLost> {
        let Some(mut slot) = self.cell.lock() else { return Err(JobLost) };
        loop {
            if let Some(result) = slot.result.take() {
                return result;
            }
            slot = self.cell.ready.wait(slot).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// Resolves once a class could be admitted; it reserves nothing.
pub struct CapacityAvailable {
    shared: Arc<Shared>,
    class: usize,
}

impl Future for CapacityAvailable {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // A forked child admits without counting; see `Pool::try_admit`.
        if self.shared.inherited() {
            return Poll::Ready(());
        }
        let mut state = self.shared.lock();
        if self.shared.fits(&state, self.class) {
            return Poll::Ready(());
        }
        if !state.capacity_waiters.iter().any(|waiter| waiter.will_wake(cx.waker())) {
            state.capacity_waiters.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl<C: Class> Pool<C> {
    /// A pool of at most `total` jobs, each class at most its entry of `limits` (by
    /// [`Class::index`]), its threads named `thread_name`.
    pub fn with_limits(
        thread_name: &'static str,
        total: usize,
        limits: &[usize],
    ) -> Result<Self, PoolError<C>> {
        if total == 0 || limits.iter().any(|&limit| limit == 0 || limit > total) {
            return Err(PoolError::InvalidLimits);
        }
        let state = State {
            used: vec![0; limits.len()].into(),
            total: 0,
            tasks: VecDeque::new(),
            idle: 0,
            capacity_waiters: Vec::new(),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                pid: std::process::id(),
                thread_name,
                total,
                limits: limits.into(),
                state: Mutex::new(state),
                work: Condvar::new(),
            }),
            class: PhantomData,
        })
    }

    /// Refuse overload synchronously without retaining a job or an admission waiter.
    pub fn try_admit(&self, class: C) -> Result<Permit<C>, PoolError<C>> {
        // A forked child admits without counting: the job it spawns settles lost.
        if self.shared.inherited() {
            return Ok(Permit { shared: self.shared.clone(), class });
        }
        let mut state = self.shared.lock();
        if !self.shared.fits(&state, class.index()) {
            return Err(PoolError::Exhausted(class));
        }
        state.used[class.index()] += 1;
        state.total += 1;
        Ok(Permit { shared: self.shared.clone(), class })
    }

    /// Wait after refused admission without retaining work or reserving capacity.
    pub fn wait_for_capacity(&self, class: C) -> CapacityAvailable {
        CapacityAvailable { shared: self.shared.clone(), class: class.index() }
    }

    /// Synchronous ingress refuses overload before retaining work. There is no admission queue.
    pub fn try_spawn<T: Send + 'static>(
        &self,
        class: C,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<Job<T>, PoolError<C>> {
        Ok(self.spawn_admitted(self.try_admit(class)?, work))
    }

    /// Wait for capacity, then run `work` and await its result.
    pub async fn run<T: Send + 'static>(
        &self,
        class: C,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, JobLost> {
        loop {
            match self.try_admit(class) {
                Ok(permit) => return self.spawn_admitted(permit, work).await,
                Err(_) => self.wait_for_capacity(class).await,
            }
        }
    }

    /// [`Pool::run`] for fallible work: a panic is the work's own failure.
    pub async fn try_run<T: Send + 'static, E: From<JobLost> + Send + 'static>(
        &self,
        class: C,
        work: impl FnOnce() -> Result<T, E> + Send + 'static,
    ) -> Result<T, E> {
        self.run(class, work).await?
    }

    /// Run work under a reservation taken earlier, so callers can move owned state after
    /// admission.
    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: Permit<C>,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Job<T> {
        let cell = Arc::new(JobCell {
            pid: self.shared.pid,
            slot: Mutex::new(Slot { result: None, waker: None }),
            ready: Condvar::new(),
        });
        if self.shared.inherited() {
            return Job { cell };
        }
        let settle = Settle { cell: cell.clone(), permit: Some(permit) };
        self.shared.run(Box::new(move || {
            // A panic drops `settle` unfinished, which settles the job lost.
            if let Ok(result) = std::panic::catch_unwind(AssertUnwindSafe(work)) {
                settle.finish(result);
            }
        }));
        Job { cell }
    }
}

fn work(shared: Arc<Shared>) {
    let mut state = shared.lock();
    loop {
        if let Some(task) = state.tasks.pop_front() {
            drop(state);
            task();
            state = shared.lock();
            continue;
        }
        state.idle += 1;
        let (next, timeout) = shared
            .work
            .wait_timeout_while(state, IDLE_KEEPALIVE, |state| state.tasks.is_empty())
            .unwrap_or_else(|p| p.into_inner());
        state = next;
        state.idle -= 1;
        if timeout.timed_out() && state.tasks.is_empty() {
            return;
        }
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
