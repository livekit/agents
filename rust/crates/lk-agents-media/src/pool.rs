//! Admitted worker threads: a bounded pool of plain threads, no async runtime. Callers that must
//! not block their loop hand work here and await its [`Job`]. Admission refuses overload
//! synchronously; a running job keeps its capacity until it settles, so a caller dropping its
//! receipt never frees capacity early.
//!
//! A process has two pools: the media pool ([`media_pool`]: codec, recording, image and model
//! work) and the blocking I/O pool (`lk_agents_net::blocking_io`: name resolution, files and
//! process sampling). Tokio's blocking pool is not used.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
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
/// threads and may hold state its parent's threads owned, makes its own. The parent's copy is
/// abandoned, never dropped.
pub struct PerProcess<T> {
    slot: Mutex<Option<(u32, T)>>,
    make: fn() -> T,
}

impl<T: Clone> PerProcess<T> {
    pub const fn new(make: fn() -> T) -> Self {
        Self { slot: Mutex::new(None), make }
    }

    pub fn get(&self) -> T {
        let pid = std::process::id();
        let mut slot = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        match &*slot {
            Some((owner, value)) if *owner == pid => value.clone(),
            _ => {
                std::mem::forget(slot.take());
                let value = (self.make)();
                *slot = Some((pid, value.clone()));
                value
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

    fn release(&self, class: usize) {
        let waiters = {
            let mut state = self.lock();
            state.used[class] -= 1;
            state.total -= 1;
            std::mem::take(&mut state.capacity_waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
    }

    fn run(self: &Arc<Self>, task: Task) {
        let mut state = self.lock();
        state.tasks.push_back(task);
        if state.tasks.len() <= state.idle {
            self.work.notify_one();
        } else {
            // One thread per queued job beyond the idle ones, so no job waits behind a long-lived
            // one; every job holds a permit, so threads never outnumber the total ceiling.
            let shared = self.clone();
            std::thread::Builder::new()
                .name(self.thread_name.into())
                .spawn(move || work(shared))
                .expect("spawning a pool thread");
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

/// The job panicked; its capacity was released when it unwound.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker job panicked")]
pub struct JobPanicked;

struct Slot<T> {
    result: Option<Result<T, JobPanicked>>,
    waker: Option<Waker>,
}

/// The completion of one admitted job. Awaiting it observes the result; dropping it leaves the
/// job running with its capacity.
pub struct Job<T> {
    slot: Arc<(Mutex<Slot<T>>, Condvar)>,
}

impl<T> Future for Job<T> {
    type Output = Result<T, JobPanicked>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut slot = self.slot.0.lock().unwrap_or_else(|p| p.into_inner());
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
    pub fn wait(self) -> Result<T, JobPanicked> {
        let (slot, ready) = &*self.slot;
        let mut slot = slot.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(result) = slot.result.take() {
                return result;
            }
            slot = ready.wait(slot).unwrap_or_else(|p| p.into_inner());
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
        let mut state = self.shared.lock();
        if self.shared.fits(&state, self.class) {
            return Poll::Ready(());
        }
        state.capacity_waiters.push(cx.waker().clone());
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
    ) -> Result<T, JobPanicked> {
        loop {
            match self.try_admit(class) {
                Ok(permit) => return self.spawn_admitted(permit, work).await,
                Err(_) => self.wait_for_capacity(class).await,
            }
        }
    }

    /// Run work under a reservation taken earlier, so callers can move owned state after
    /// admission.
    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: Permit<C>,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Job<T> {
        let slot = Arc::new((Mutex::new(Slot { result: None, waker: None }), Condvar::new()));
        let settled = slot.clone();
        self.shared.run(Box::new(move || {
            let result = std::panic::catch_unwind(AssertUnwindSafe(work)).map_err(|_| JobPanicked);
            // Capacity is released by settlement, before the caller observes the result.
            drop(permit);
            let waker = {
                let mut slot = settled.0.lock().unwrap_or_else(|p| p.into_inner());
                slot.result = Some(result);
                slot.waker.take()
            };
            settled.1.notify_all();
            if let Some(waker) = waker {
                waker.wake();
            }
        }));
        Job { slot }
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
mod tests;
