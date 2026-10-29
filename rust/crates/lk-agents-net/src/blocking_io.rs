//! Bounded blocking work for worker discovery, setup and health sampling.
//! Capacity is independent of media payloads and stays reserved until work is observed complete.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use lk_agents_abi::SpawnerExt;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use crate::task::TokioSpawner;

/// Why blocking I/O was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BlockingIoError {
    #[error("blocking I/O capacity must be positive")]
    InvalidCapacity,
    #[error("blocking I/O capacity exhausted")]
    Exhausted,
}

#[derive(Clone)]
pub struct BlockingIo {
    runtime: tokio::runtime::Handle,
    capacity: Arc<Semaphore>,
    released: Arc<Notify>,
}

/// Admission acquired before retaining work or spawning a task.
pub struct BlockingIoPermit {
    permit: Option<OwnedSemaphorePermit>,
    released: Arc<Notify>,
}

impl Drop for BlockingIoPermit {
    fn drop(&mut self) {
        self.permit.take();
        self.released.notify_waiters();
    }
}

/// Dropping the receipt transfers its join obligation to an owned cleanup task.
pub struct BlockingIoTask<T: Send + 'static> {
    join: Option<JoinHandle<T>>,
    runtime: tokio::runtime::Handle,
    permit: Option<Arc<BlockingIoPermit>>,
}

impl<T: Send + 'static> Future for BlockingIoTask<T> {
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

impl<T: Send + 'static> Drop for BlockingIoTask<T> {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let permit = self.permit.take();
            self.runtime.spawn(async move {
                let _permit = permit;
                if let Err(error) = join.await {
                    log::error!(target: "livekit.agents", "blocking I/O failed during cleanup: {error}");
                }
            });
        }
    }
}

impl BlockingIo {
    pub fn new(runtime: tokio::runtime::Handle, capacity: usize) -> Result<Self, BlockingIoError> {
        if capacity == 0 || capacity > Semaphore::MAX_PERMITS {
            return Err(BlockingIoError::InvalidCapacity);
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

    /// Runs `work` on the injected runtime, wherever the caller polls: tasks it spawns land there
    /// too. Dropping the returned future cancels the work.
    pub async fn run<T: Send + 'static>(
        &self,
        work: impl Future<Output = T> + Send + 'static,
    ) -> T {
        match TokioSpawner::new(self.runtime.clone()).spawn(work).await {
            Ok(value) => value,
            Err(error) => std::panic::resume_unwind(Box::new(error.to_string())),
        }
    }

    pub fn try_admit(&self) -> Result<BlockingIoPermit, BlockingIoError> {
        let permit =
            self.capacity.clone().try_acquire_owned().map_err(|_| BlockingIoError::Exhausted)?;
        Ok(BlockingIoPermit { permit: Some(permit), released: self.released.clone() })
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
    ) -> Result<BlockingIoTask<T>, BlockingIoError> {
        Ok(self.spawn_admitted(self.try_admit()?, work))
    }

    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: BlockingIoPermit,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> BlockingIoTask<T> {
        let permit = Arc::new(permit);
        let held = permit.clone();
        let join = self.runtime.spawn_blocking(move || {
            let _permit = held;
            work()
        });
        BlockingIoTask { join: Some(join), runtime: self.runtime.clone(), permit: Some(permit) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn capacity_wait_observes_release_without_losing_a_notification() {
        let blocking_io = BlockingIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let permit = blocking_io.try_admit().unwrap();
        let available = blocking_io.wait_for_capacity();
        tokio::pin!(available);
        assert!(futures_util::poll!(available.as_mut()).is_pending());
        drop(permit);
        tokio::time::timeout(Duration::from_secs(1), available).await.unwrap();
        let _admitted = blocking_io.try_admit().unwrap();
    }

    #[tokio::test]
    async fn cancelled_receipt_keeps_running_work_admitted_until_cleanup_joins() {
        let blocking_io = BlockingIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = blocking_io
            .try_spawn(move || {
                let _ = started.send(());
                wait.recv().unwrap();
            })
            .unwrap();
        ready.await.unwrap();
        let caller = tokio::spawn(task);
        caller.abort();
        let _ = caller.await;
        assert!(matches!(blocking_io.try_admit(), Err(BlockingIoError::Exhausted)));
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), blocking_io.wait_for_capacity())
            .await
            .unwrap();
        assert_eq!(blocking_io.try_spawn(|| 42).unwrap().await.unwrap(), 42);
    }

    #[tokio::test]
    async fn refused_work_never_runs() {
        let blocking_io = BlockingIo::new(tokio::runtime::Handle::current(), 1).unwrap();
        let _permit = blocking_io.try_admit().unwrap();
        assert!(matches!(
            blocking_io.try_spawn(|| panic!("work was admitted while capacity was exhausted")),
            Err(BlockingIoError::Exhausted)
        ));
    }
}
