//! Admission shared by blocking media resources and image preparation. A permit stays with the
//! native work, including after its async caller is cancelled; dropping a join never frees capacity.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaWorkerClass {
    Decoder,
    Recording,
    Image,
}

/// Per-class ceilings bound long-lived resources independently of short jobs.
#[derive(Debug, Clone, Copy)]
pub struct MediaWorkerLimits {
    pub total: usize,
    pub decoders: usize,
    pub recordings: usize,
    pub images: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MediaWorkerError {
    #[error("media worker limits must be positive and each class must fit the total ceiling")]
    InvalidLimits,
    #[error("media worker capacity exhausted for {0:?}")]
    Exhausted(MediaWorkerClass),
}

#[derive(Clone)]
pub struct MediaWorkers {
    handle: tokio::runtime::Handle,
    total: Arc<Semaphore>,
    decoders: Arc<Semaphore>,
    recordings: Arc<Semaphore>,
    images: Arc<Semaphore>,
}

/// An awaited completion receipt. Dropping the awaiter transfers the join and admission to the
/// runtime supervisor, so cancellation never detaches native work or frees its capacity early.
pub struct MediaWorker<T: Send + 'static> {
    join: Option<JoinHandle<T>>,
    runtime: tokio::runtime::Handle,
    permits: Option<Arc<(OwnedSemaphorePermit, OwnedSemaphorePermit)>>,
}

impl<T: Send + 'static> Future for MediaWorker<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let result = Pin::new(this.join.as_mut().expect("worker polled after completion")).poll(cx);
        if result.is_ready() {
            this.join.take();
            this.permits.take();
        }
        result
    }
}

impl<T: Send + 'static> Drop for MediaWorker<T> {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let permits = self.permits.take();
            self.runtime.spawn(async move {
                let _permits = permits;
                if let Err(err) = join.await {
                    log::error!(target: "livekit.agents", "media worker failed during cleanup: {err}");
                }
            });
        }
    }
}

impl MediaWorkers {
    pub(crate) fn runtime_handle(&self) -> tokio::runtime::Handle {
        self.handle.clone()
    }

    pub fn new(
        handle: tokio::runtime::Handle,
        limits: MediaWorkerLimits,
    ) -> Result<Self, MediaWorkerError> {
        if limits.total == 0
            || [limits.decoders, limits.recordings, limits.images]
                .into_iter()
                .any(|limit| limit == 0 || limit > limits.total)
            || limits.total > Semaphore::MAX_PERMITS
        {
            return Err(MediaWorkerError::InvalidLimits);
        }
        Ok(Self {
            handle,
            total: Arc::new(Semaphore::new(limits.total)),
            decoders: Arc::new(Semaphore::new(limits.decoders)),
            recordings: Arc::new(Semaphore::new(limits.recordings)),
            images: Arc::new(Semaphore::new(limits.images)),
        })
    }

    /// Synchronous ingress refuses overload before retaining work. There is no admission queue.
    pub fn try_spawn<T: Send + 'static>(
        &self,
        class: MediaWorkerClass,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<MediaWorker<T>, MediaWorkerError> {
        let capacity = match class {
            MediaWorkerClass::Decoder => &self.decoders,
            MediaWorkerClass::Recording => &self.recordings,
            MediaWorkerClass::Image => &self.images,
        };
        let class_permit =
            capacity.clone().try_acquire_owned().map_err(|_| MediaWorkerError::Exhausted(class))?;
        let total_permit = self
            .total
            .clone()
            .try_acquire_owned()
            .map_err(|_| MediaWorkerError::Exhausted(class))?;
        let permits = Arc::new((class_permit, total_permit));
        let held = permits.clone();
        let join = self.handle.spawn_blocking(move || {
            let _permits = held;
            work()
        });
        Ok(MediaWorker { join: Some(join), runtime: self.handle.clone(), permits: Some(permits) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workers(total: usize) -> MediaWorkers {
        MediaWorkers::new(
            tokio::runtime::Handle::current(),
            MediaWorkerLimits { total, decoders: 1, recordings: 1, images: 1 },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn cancelled_caller_does_not_release_a_running_worker() {
        let workers = workers(1);
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = workers
            .try_spawn(MediaWorkerClass::Decoder, move || {
                let _ = started.send(());
                wait.recv().unwrap();
            })
            .unwrap();
        ready.await.unwrap();
        let waiter = tokio::spawn(task);
        waiter.abort();
        let _ = waiter.await;
        assert!(matches!(
            workers.try_spawn(MediaWorkerClass::Recording, || ()),
            Err(MediaWorkerError::Exhausted(MediaWorkerClass::Recording))
        ));
        release.send(()).unwrap();
        // Capacity is released by observed settlement, not by dropping the caller.
        loop {
            match workers.try_spawn(MediaWorkerClass::Recording, || ()) {
                Ok(task) => {
                    task.await.unwrap();
                    break;
                }
                Err(MediaWorkerError::Exhausted(_)) => tokio::task::yield_now().await,
                Err(err) => panic!("{err}"),
            }
        }
    }

    #[tokio::test]
    async fn observing_completion_releases_capacity_while_the_receipt_remains_live() {
        let workers = workers(1);
        let mut receipt = workers.try_spawn(MediaWorkerClass::Decoder, || ()).unwrap();
        (&mut receipt).await.unwrap();
        workers.try_spawn(MediaWorkerClass::Recording, || ()).unwrap().await.unwrap();
        drop(receipt);
    }

    #[tokio::test]
    async fn idle_decoders_leave_other_class_capacity_available() {
        let workers = workers(2);
        let (release, wait) = std::sync::mpsc::channel();
        let decoder = workers.try_spawn(MediaWorkerClass::Decoder, move || wait.recv()).unwrap();
        assert!(matches!(
            workers.try_spawn(MediaWorkerClass::Decoder, || ()),
            Err(MediaWorkerError::Exhausted(MediaWorkerClass::Decoder))
        ));
        workers.try_spawn(MediaWorkerClass::Recording, || ()).unwrap().await.unwrap();
        release.send(()).unwrap();
        decoder.await.unwrap().unwrap();
    }
}
