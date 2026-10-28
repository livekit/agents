//! Bounded blocking work for worker discovery, setup and health sampling.
//! Capacity is independent of media payloads and stays reserved until work is observed complete.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use lk_agents_abi::utils::workers::{WorkerClass, WorkerError};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

#[derive(Clone)]
pub struct WorkerIo {
    runtime: tokio::runtime::Handle,
    capacity: Arc<Semaphore>,
    released: Arc<Notify>,
}

/// Admission acquired before retaining work or spawning a task.
pub struct WorkerIoPermit {
    permit: Option<OwnedSemaphorePermit>,
    released: Arc<Notify>,
}

impl Drop for WorkerIoPermit {
    fn drop(&mut self) {
        self.permit.take();
        self.released.notify_waiters();
    }
}

/// Dropping the receipt transfers its join obligation to an owned cleanup task.
pub struct WorkerIoTask<T: Send + 'static> {
    join: Option<JoinHandle<T>>,
    runtime: tokio::runtime::Handle,
    permit: Option<Arc<WorkerIoPermit>>,
}

impl<T: Send + 'static> Future for WorkerIoTask<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let result = Pin::new(this.join.as_mut().expect("worker polled after completion")).poll(cx);
        if result.is_ready() {
            this.join.take();
            this.permit.take();
        }
        result
    }
}

impl<T: Send + 'static> Drop for WorkerIoTask<T> {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let permit = self.permit.take();
            self.runtime.spawn(async move {
                let _permit = permit;
                if let Err(error) = join.await {
                    log::error!(target: "livekit.agents", "worker I/O failed during cleanup: {error}");
                }
            });
        }
    }
}

impl WorkerIo {
    pub fn new(runtime: tokio::runtime::Handle, capacity: usize) -> Result<Self, WorkerError> {
        if capacity == 0 || capacity > Semaphore::MAX_PERMITS {
            return Err(WorkerError::InvalidLimits);
        }
        Ok(Self {
            runtime,
            capacity: Arc::new(Semaphore::new(capacity)),
            released: Arc::new(Notify::new()),
        })
    }

    pub fn runtime_handle(&self) -> tokio::runtime::Handle {
        self.runtime.clone()
    }

    pub fn try_admit(&self) -> Result<WorkerIoPermit, WorkerError> {
        let permit = self
            .capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| WorkerError::Exhausted(WorkerClass::WorkerIo))?;
        Ok(WorkerIoPermit { permit: Some(permit), released: self.released.clone() })
    }

    /// Wait without reserving capacity or retaining a blocking job.
    pub async fn wait_for_capacity(&self) {
        loop {
            let changed = self.released.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.capacity.available_permits() > 0 {
                return;
            }
            changed.await;
        }
    }

    pub fn try_spawn<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<WorkerIoTask<T>, WorkerError> {
        Ok(self.spawn_admitted(self.try_admit()?, work))
    }

    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: WorkerIoPermit,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> WorkerIoTask<T> {
        let permit = Arc::new(permit);
        let held = permit.clone();
        let join = self.runtime.spawn_blocking(move || {
            let _permit = held;
            work()
        });
        WorkerIoTask { join: Some(join), runtime: self.runtime.clone(), permit: Some(permit) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn capacity_wait_observes_release_without_losing_a_notification() {
        let workers = WorkerIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let permit = workers.try_admit().unwrap();
        let available = workers.wait_for_capacity();
        tokio::pin!(available);
        assert!(futures_util::poll!(available.as_mut()).is_pending());
        drop(permit);
        tokio::time::timeout(Duration::from_secs(1), available).await.unwrap();
        let _admitted = workers.try_admit().unwrap();
    }

    #[tokio::test]
    async fn cancelled_receipt_keeps_running_work_admitted_until_cleanup_joins() {
        let workers = WorkerIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = workers
            .try_spawn(move || {
                let _ = started.send(());
                wait.recv().unwrap();
            })
            .unwrap();
        ready.await.unwrap();
        let caller = tokio::spawn(task);
        caller.abort();
        let _ = caller.await;
        assert!(matches!(workers.try_admit(), Err(WorkerError::Exhausted(WorkerClass::WorkerIo))));
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), workers.wait_for_capacity()).await.unwrap();
        assert_eq!(workers.try_spawn(|| 42).unwrap().await.unwrap(), 42);
    }

    #[tokio::test]
    async fn refused_work_never_runs() {
        let workers = WorkerIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let _permit = workers.try_admit().unwrap();
        assert!(matches!(
            workers.try_spawn(|| panic!("work was admitted while capacity was exhausted")),
            Err(WorkerError::Exhausted(WorkerClass::WorkerIo))
        ));
    }
}
