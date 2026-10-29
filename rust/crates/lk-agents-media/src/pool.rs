//! Admitted native workers for codec, recording, image and model work: a bounded pool
//! of plain threads, no async runtime. Callers that must not block their loop hand work here and
//! await its [`MediaJob`]. Admission refuses overload synchronously; a running job keeps its
//! capacity until it settles, so a caller dropping its receipt never frees capacity early.

use std::collections::VecDeque;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

pub use lk_agents_abi::media::pool::{JobClass, MediaExecutor, MediaPoolError};

/// An idle thread waits this long for its next job before it exits.
const IDLE_KEEPALIVE: Duration = Duration::from_secs(10);

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

impl MediaPoolLimits {
    fn of(&self, class: JobClass) -> usize {
        match class {
            JobClass::Decoder => self.decoders,
            JobClass::Recording => self.recordings,
            JobClass::Image => self.images,
            JobClass::Formatting => self.formatting,
            JobClass::EotModel => self.eot_models,
            JobClass::VadModel => self.vad_models,
        }
    }
}

const CLASSES: usize = 6;

fn index(class: JobClass) -> usize {
    class as usize
}

type Job = Box<dyn FnOnce() + Send>;

struct State {
    used: [usize; CLASSES],
    total: usize,
    jobs: VecDeque<Job>,
    idle: usize,
    capacity_waiters: Vec<Waker>,
}

struct Shared {
    limits: MediaPoolLimits,
    state: Mutex<State>,
    work: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn fits(&self, state: &State, class: JobClass) -> bool {
        state.used[index(class)] < self.limits.of(class) && state.total < self.limits.total
    }
}

/// The pool's threads and admission state.
struct Pool(Arc<Shared>);

impl MediaExecutor for Pool {
    fn admit(&self, class: JobClass) -> Result<(), MediaPoolError> {
        let mut state = self.0.lock();
        if !self.0.fits(&state, class) {
            return Err(MediaPoolError::Exhausted(class));
        }
        state.used[index(class)] += 1;
        state.total += 1;
        Ok(())
    }

    fn release(&self, class: JobClass) {
        let waiters = {
            let mut state = self.0.lock();
            state.used[index(class)] -= 1;
            state.total -= 1;
            std::mem::take(&mut state.capacity_waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
    }

    fn poll_capacity(&self, class: JobClass, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.0.lock();
        if self.0.fits(&state, class) {
            return Poll::Ready(());
        }
        state.capacity_waiters.push(cx.waker().clone());
        Poll::Pending
    }

    fn run(&self, job: Job) {
        let mut state = self.0.lock();
        state.jobs.push_back(job);
        if state.jobs.len() <= state.idle {
            self.0.work.notify_one();
        } else {
            // One thread per queued job beyond the idle ones, so no job waits behind a long-lived
            // one; every job holds a permit, so threads never outnumber the total ceiling.
            let shared = self.0.clone();
            std::thread::Builder::new()
                .name("lk-media-worker".into())
                .spawn(move || run(shared))
                .expect("spawning a media worker thread");
        }
    }
}

/// The process's admission pool. Clones share the same class and total capacity.
#[derive(Clone)]
pub struct MediaPool {
    pool: Arc<dyn MediaExecutor>,
}

/// The typed face of a lent pool.
impl From<Arc<dyn MediaExecutor>> for MediaPool {
    fn from(pool: Arc<dyn MediaExecutor>) -> Self {
        Self { pool }
    }
}

/// A class reservation. Native jobs retain it until they settle; dropping it releases the slot
/// and wakes capacity waiters.
pub struct MediaPermit {
    pool: Arc<dyn MediaExecutor>,
    class: JobClass,
}

impl Drop for MediaPermit {
    fn drop(&mut self) {
        self.pool.release(self.class);
    }
}

/// The job panicked; its capacity was released when it unwound.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("media worker job panicked")]
pub struct JobPanicked;

struct Slot<T> {
    result: Option<Result<T, JobPanicked>>,
    waker: Option<Waker>,
}

/// The completion of one admitted job. Awaiting it observes the result; dropping it leaves the
/// job running with its capacity.
pub struct MediaJob<T> {
    slot: Arc<(Mutex<Slot<T>>, Condvar)>,
}

impl<T> Future for MediaJob<T> {
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

impl<T> MediaJob<T> {
    /// Observe settlement from another native worker or a synchronous caller.
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

/// Resolves once `class` could be admitted; it reserves nothing.
pub struct CapacityAvailable {
    pool: Arc<dyn MediaExecutor>,
    class: JobClass,
}

impl Future for CapacityAvailable {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.pool.poll_capacity(self.class, cx)
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
        if limits.total == 0 || classes.into_iter().any(|limit| limit == 0 || limit > limits.total)
        {
            return Err(MediaPoolError::InvalidLimits);
        }
        Ok(Self {
            pool: Arc::new(Pool(Arc::new(Shared {
                limits,
                state: Mutex::new(State {
                    used: [0; CLASSES],
                    total: 0,
                    jobs: VecDeque::new(),
                    idle: 0,
                    capacity_waiters: Vec::new(),
                }),
                work: Condvar::new(),
            }))),
        })
    }

    /// The pool as the runtime lends it.
    pub fn executor(&self) -> Arc<dyn MediaExecutor> {
        self.pool.clone()
    }

    /// Refuse overload synchronously without retaining a job or an admission waiter.
    pub fn try_admit(&self, class: JobClass) -> Result<MediaPermit, MediaPoolError> {
        self.pool.admit(class)?;
        Ok(MediaPermit { pool: self.pool.clone(), class })
    }

    /// Wait after refused admission without retaining work or reserving capacity.
    pub fn wait_for_capacity(&self, class: JobClass) -> CapacityAvailable {
        CapacityAvailable { pool: self.pool.clone(), class }
    }

    /// Synchronous ingress refuses overload before retaining work. There is no admission queue.
    pub fn try_spawn<T: Send + 'static>(
        &self,
        class: JobClass,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<MediaJob<T>, MediaPoolError> {
        Ok(self.spawn_admitted(self.try_admit(class)?, work))
    }

    /// Run work under a reservation taken earlier, so callers can move owned state after
    /// admission.
    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: MediaPermit,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> MediaJob<T> {
        let slot = Arc::new((Mutex::new(Slot { result: None, waker: None }), Condvar::new()));
        let settled = slot.clone();
        let job: Job = Box::new(move || {
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
        });
        self.pool.run(job);
        MediaJob { slot }
    }
}

fn run(shared: Arc<Shared>) {
    let mut state = shared.lock();
    loop {
        if let Some(job) = state.jobs.pop_front() {
            drop(state);
            job();
            state = shared.lock();
            continue;
        }
        state.idle += 1;
        let (next, timeout) = shared
            .work
            .wait_timeout_while(state, IDLE_KEEPALIVE, |state| state.jobs.is_empty())
            .unwrap_or_else(|p| p.into_inner());
        state = next;
        state.idle -= 1;
        if timeout.timed_out() && state.jobs.is_empty() {
            return;
        }
    }
}

#[cfg(test)]
mod tests;
