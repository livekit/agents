//! Admitted native workers for codec, recording, image and model work (decision 84): a bounded pool
//! of plain threads, no async runtime. Callers that must not block their loop hand work here and
//! await its [`MediaWorker`]. Admission refuses overload synchronously; a running job keeps its
//! capacity until it settles, so a caller dropping its receipt never frees capacity early.

use std::collections::VecDeque;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

pub use lk_agents_abi::media::workers::{
    MediaPool, WorkerClass as MediaWorkerClass, WorkerError as MediaWorkerError,
};

/// An idle thread waits this long for its next job before it exits.
const IDLE_KEEPALIVE: Duration = Duration::from_secs(10);

/// Per-class ceilings bound long-lived resources independently of short jobs.
#[derive(Debug, Clone, Copy)]
pub struct MediaWorkerLimits {
    pub total: usize,
    pub decoders: usize,
    pub recordings: usize,
    pub images: usize,
    pub formatting: usize,
    pub eot_models: usize,
    pub vad_models: usize,
}

impl Default for MediaWorkerLimits {
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

impl MediaWorkerLimits {
    fn of(&self, class: MediaWorkerClass) -> usize {
        match class {
            MediaWorkerClass::Decoder => self.decoders,
            MediaWorkerClass::Recording => self.recordings,
            MediaWorkerClass::Image => self.images,
            MediaWorkerClass::Formatting => self.formatting,
            MediaWorkerClass::EotModel => self.eot_models,
            MediaWorkerClass::VadModel => self.vad_models,
        }
    }
}

const CLASSES: usize = 6;

fn index(class: MediaWorkerClass) -> usize {
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
    limits: MediaWorkerLimits,
    state: Mutex<State>,
    work: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn fits(&self, state: &State, class: MediaWorkerClass) -> bool {
        state.used[index(class)] < self.limits.of(class) && state.total < self.limits.total
    }
}

/// The pool's threads and admission state.
struct Pool(Arc<Shared>);

impl MediaPool for Pool {
    fn admit(&self, class: MediaWorkerClass) -> Result<(), MediaWorkerError> {
        let mut state = self.0.lock();
        if !self.0.fits(&state, class) {
            return Err(MediaWorkerError::Exhausted(class));
        }
        state.used[index(class)] += 1;
        state.total += 1;
        Ok(())
    }

    fn release(&self, class: MediaWorkerClass) {
        let waiters = {
            let mut state = self.0.lock();
            state.used[index(class)] -= 1;
            state.total -= 1;
            std::mem::take(&mut state.capacity_waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
    }

    fn poll_capacity(&self, class: MediaWorkerClass, cx: &mut Context<'_>) -> Poll<()> {
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
pub struct MediaWorkers {
    pool: Arc<dyn MediaPool>,
}

/// The typed face of a lent pool.
impl From<Arc<dyn MediaPool>> for MediaWorkers {
    fn from(pool: Arc<dyn MediaPool>) -> Self {
        Self { pool }
    }
}

/// A class reservation. Native jobs retain it until they settle; dropping it releases the slot
/// and wakes capacity waiters.
pub struct MediaWorkerPermit {
    pool: Arc<dyn MediaPool>,
    class: MediaWorkerClass,
}

impl Drop for MediaWorkerPermit {
    fn drop(&mut self) {
        self.pool.release(self.class);
    }
}

/// The job panicked; its capacity was released when it unwound.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("media worker job panicked")]
pub struct WorkerPanicked;

struct Slot<T> {
    result: Option<Result<T, WorkerPanicked>>,
    waker: Option<Waker>,
}

/// The completion of one admitted job. Awaiting it observes the result; dropping it leaves the
/// job running with its capacity.
pub struct MediaWorker<T> {
    slot: Arc<(Mutex<Slot<T>>, Condvar)>,
}

impl<T> Future for MediaWorker<T> {
    type Output = Result<T, WorkerPanicked>;

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

impl<T> MediaWorker<T> {
    /// Observe settlement from another native worker or a synchronous caller.
    /// Async callers must await the receipt instead of blocking their executor.
    pub fn wait(self) -> Result<T, WorkerPanicked> {
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
    pool: Arc<dyn MediaPool>,
    class: MediaWorkerClass,
}

impl Future for CapacityAvailable {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.pool.poll_capacity(self.class, cx)
    }
}

impl MediaWorkers {
    pub fn new(limits: MediaWorkerLimits) -> Result<Self, MediaWorkerError> {
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
            return Err(MediaWorkerError::InvalidLimits);
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
    pub fn pool(&self) -> Arc<dyn MediaPool> {
        self.pool.clone()
    }

    /// Refuse overload synchronously without retaining a job or an admission waiter.
    pub fn try_admit(
        &self,
        class: MediaWorkerClass,
    ) -> Result<MediaWorkerPermit, MediaWorkerError> {
        self.pool.admit(class)?;
        Ok(MediaWorkerPermit { pool: self.pool.clone(), class })
    }

    /// Wait after refused admission without retaining work or reserving capacity.
    pub fn wait_for_capacity(&self, class: MediaWorkerClass) -> CapacityAvailable {
        CapacityAvailable { pool: self.pool.clone(), class }
    }

    /// Synchronous ingress refuses overload before retaining work. There is no admission queue.
    pub fn try_spawn<T: Send + 'static>(
        &self,
        class: MediaWorkerClass,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<MediaWorker<T>, MediaWorkerError> {
        Ok(self.spawn_admitted(self.try_admit(class)?, work))
    }

    /// Run work under a reservation taken earlier, so callers can move owned state after
    /// admission.
    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: MediaWorkerPermit,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> MediaWorker<T> {
        let slot = Arc::new((Mutex::new(Slot { result: None, waker: None }), Condvar::new()));
        let settled = slot.clone();
        let job: Job = Box::new(move || {
            let result =
                std::panic::catch_unwind(AssertUnwindSafe(work)).map_err(|_| WorkerPanicked);
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
        MediaWorker { slot }
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
