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
    WorkerClass as MediaWorkerClass, WorkerError as MediaWorkerError,
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
    pub uploads: usize,
    pub worker_io: usize,
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
            uploads: 2,
            worker_io: 4,
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
            MediaWorkerClass::Upload => self.uploads,
            MediaWorkerClass::WorkerIo => self.worker_io,
        }
    }
}

const CLASSES: usize = 8;

fn index(class: MediaWorkerClass) -> usize {
    class as usize
}

type Job = Box<dyn FnOnce() + Send>;

struct State {
    used: [usize; CLASSES],
    total: usize,
    /// Payload classes together leave `limits.total - payload` slots for worker I/O sampling.
    payload: usize,
    jobs: VecDeque<Job>,
    idle: usize,
    capacity_waiters: Vec<Waker>,
}

struct Shared {
    limits: MediaWorkerLimits,
    payload_limit: usize,
    state: Mutex<State>,
    work: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn fits(&self, state: &State, class: MediaWorkerClass) -> bool {
        state.used[index(class)] < self.limits.of(class)
            && state.total < self.limits.total
            && (class == MediaWorkerClass::WorkerIo || state.payload < self.payload_limit)
    }
}

/// The process's admission pool. Clones share the same class and total capacity.
#[derive(Clone)]
pub struct MediaWorkers {
    shared: Arc<Shared>,
}

/// A class reservation. Native jobs retain it until they settle; dropping it releases the slot
/// and wakes capacity waiters.
pub struct MediaWorkerPermit {
    shared: Arc<Shared>,
    class: MediaWorkerClass,
}

impl Drop for MediaWorkerPermit {
    fn drop(&mut self) {
        let waiters = {
            let mut state = self.shared.lock();
            state.used[index(self.class)] -= 1;
            state.total -= 1;
            if self.class != MediaWorkerClass::WorkerIo {
                state.payload -= 1;
            }
            std::mem::take(&mut state.capacity_waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
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
    shared: Arc<Shared>,
    class: MediaWorkerClass,
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

impl MediaWorkers {
    pub fn new(limits: MediaWorkerLimits) -> Result<Self, MediaWorkerError> {
        let classes = [
            limits.decoders,
            limits.recordings,
            limits.images,
            limits.formatting,
            limits.eot_models,
            limits.vad_models,
            limits.uploads,
            limits.worker_io,
        ];
        if limits.total == 0 || classes.into_iter().any(|limit| limit == 0 || limit > limits.total)
        {
            return Err(MediaWorkerError::InvalidLimits);
        }
        // Keep health sampling serviceable even when long-lived payload classes fill their caps.
        let reserved = limits.worker_io.min(limits.total - 1);
        Ok(Self {
            shared: Arc::new(Shared {
                limits,
                payload_limit: limits.total - reserved,
                state: Mutex::new(State {
                    used: [0; CLASSES],
                    total: 0,
                    payload: 0,
                    jobs: VecDeque::new(),
                    idle: 0,
                    capacity_waiters: Vec::new(),
                }),
                work: Condvar::new(),
            }),
        })
    }

    /// Refuse overload synchronously without retaining a job or an admission waiter.
    pub fn try_admit(
        &self,
        class: MediaWorkerClass,
    ) -> Result<MediaWorkerPermit, MediaWorkerError> {
        let mut state = self.shared.lock();
        if !self.shared.fits(&state, class) {
            return Err(MediaWorkerError::Exhausted(class));
        }
        state.used[index(class)] += 1;
        state.total += 1;
        if class != MediaWorkerClass::WorkerIo {
            state.payload += 1;
        }
        Ok(MediaWorkerPermit { shared: self.shared.clone(), class })
    }

    /// Wait after refused admission without retaining work or reserving capacity.
    pub fn wait_for_capacity(&self, class: MediaWorkerClass) -> CapacityAvailable {
        CapacityAvailable { shared: self.shared.clone(), class }
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
        let mut state = self.shared.lock();
        state.jobs.push_back(job);
        if state.jobs.len() <= state.idle {
            self.shared.work.notify_one();
        } else {
            // One thread per queued job beyond the idle ones, so no job waits behind a long-lived
            // one; every job holds a permit, so threads never outnumber the total ceiling.
            let shared = self.shared.clone();
            std::thread::Builder::new()
                .name("lk-media-worker".into())
                .spawn(move || run(shared))
                .expect("spawning a media worker thread");
        }
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
