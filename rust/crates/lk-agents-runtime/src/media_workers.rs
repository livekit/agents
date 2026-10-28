//! Admission shared by native decoders, recording, image preparation, EOT models and upload.
//! Permits stay with native work after caller cancellation; dropping a join never frees capacity.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

pub use lk_agents_abi::utils::workers::{
    WorkerClass as MediaWorkerClass, WorkerError as MediaWorkerError,
};

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

#[derive(Clone)]
pub struct MediaWorkers {
    handle: tokio::runtime::Handle,
    total: Arc<Semaphore>,
    payload: Arc<Semaphore>,
    decoders: Arc<Semaphore>,
    recordings: Arc<Semaphore>,
    images: Arc<Semaphore>,
    formatting: Arc<Semaphore>,
    eot_models: Arc<Semaphore>,
    vad_models: Arc<Semaphore>,
    uploads: Arc<Semaphore>,
    worker_io: Arc<Semaphore>,
    released: Arc<tokio::sync::Notify>,
}

/// A class reservation for an async I/O resource or a native job. Native jobs must retain this
/// until actual settlement; `try_spawn` also retains it when the receipt is abandoned.
pub struct MediaWorkerPermit {
    _class: Option<OwnedSemaphorePermit>,
    _total: Option<OwnedSemaphorePermit>,
    _payload: Option<OwnedSemaphorePermit>,
    // Fields drop in declaration order: notify only after all reservations are released.
    _released: AdmissionReleased,
}

struct AdmissionReleased {
    changed: Arc<tokio::sync::Notify>,
    armed: bool,
}

impl Drop for AdmissionReleased {
    fn drop(&mut self) {
        if self.armed {
            self.changed.notify_waiters();
        }
    }
}

/// An awaited completion receipt. Dropping the awaiter transfers the join and admission to the
/// runtime supervisor, so cancellation never detaches native work or frees its capacity early.
pub struct MediaWorker<T: Send + 'static> {
    join: Option<JoinHandle<T>>,
    runtime: tokio::runtime::Handle,
    permits: Option<Arc<MediaWorkerPermit>>,
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
            || [
                limits.decoders,
                limits.recordings,
                limits.images,
                limits.formatting,
                limits.eot_models,
                limits.vad_models,
                limits.uploads,
                limits.worker_io,
            ]
            .into_iter()
            .any(|limit| limit == 0 || limit > limits.total)
            || limits.total > Semaphore::MAX_PERMITS
        {
            return Err(MediaWorkerError::InvalidLimits);
        }
        // Keep health sampling serviceable even when long-lived payload classes fill their caps.
        let reserved = limits.worker_io.min(limits.total - 1);
        Ok(Self {
            handle,
            total: Arc::new(Semaphore::new(limits.total)),
            payload: Arc::new(Semaphore::new(limits.total - reserved)),
            decoders: Arc::new(Semaphore::new(limits.decoders)),
            recordings: Arc::new(Semaphore::new(limits.recordings)),
            images: Arc::new(Semaphore::new(limits.images)),
            formatting: Arc::new(Semaphore::new(limits.formatting)),
            eot_models: Arc::new(Semaphore::new(limits.eot_models)),
            vad_models: Arc::new(Semaphore::new(limits.vad_models)),
            uploads: Arc::new(Semaphore::new(limits.uploads)),
            worker_io: Arc::new(Semaphore::new(limits.worker_io)),
            released: Arc::new(tokio::sync::Notify::new()),
        })
    }

    /// Refuse overload synchronously without retaining a job or an admission waiter.
    pub fn try_admit(
        &self,
        class: MediaWorkerClass,
    ) -> Result<MediaWorkerPermit, MediaWorkerError> {
        let mut permit = MediaWorkerPermit {
            _class: None,
            _total: None,
            _payload: None,
            _released: AdmissionReleased { changed: self.released.clone(), armed: false },
        };
        for (slot, capacity) in [
            (&mut permit._payload, (class != MediaWorkerClass::WorkerIo).then_some(&self.payload)),
            (&mut permit._class, Some(self.class_capacity(class))),
            (&mut permit._total, Some(&self.total)),
        ] {
            if let Some(capacity) = capacity {
                *slot = Some(
                    capacity
                        .clone()
                        .try_acquire_owned()
                        .map_err(|_| MediaWorkerError::Exhausted(class))?,
                );
                permit._released.armed = true;
            }
        }
        Ok(permit)
    }

    fn class_capacity(&self, class: MediaWorkerClass) -> &Arc<Semaphore> {
        match class {
            MediaWorkerClass::Decoder => &self.decoders,
            MediaWorkerClass::Recording => &self.recordings,
            MediaWorkerClass::Image => &self.images,
            MediaWorkerClass::Formatting => &self.formatting,
            MediaWorkerClass::EotModel => &self.eot_models,
            MediaWorkerClass::VadModel => &self.vad_models,
            MediaWorkerClass::Upload => &self.uploads,
            MediaWorkerClass::WorkerIo => &self.worker_io,
        }
    }

    /// Wait after refused admission without retaining work or reserving capacity. A notification
    /// is enabled before checking every ceiling, so concurrent release cannot strand the caller.
    pub async fn wait_for_capacity(&self, class: MediaWorkerClass) {
        loop {
            let changed = self.released.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.class_capacity(class).available_permits() > 0
                && self.total.available_permits() > 0
                && (class == MediaWorkerClass::WorkerIo || self.payload.available_permits() > 0)
            {
                return;
            }
            changed.await;
        }
    }

    /// Synchronous ingress refuses overload before retaining work. There is no admission queue.
    pub fn try_spawn<T: Send + 'static>(
        &self,
        class: MediaWorkerClass,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<MediaWorker<T>, MediaWorkerError> {
        Ok(self.spawn_admitted(self.try_admit(class)?, work))
    }

    /// Launch work after reserving capacity, allowing callers to move owned state after admission.
    pub fn spawn_admitted<T: Send + 'static>(
        &self,
        permit: MediaWorkerPermit,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> MediaWorker<T> {
        let permits = Arc::new(permit);
        let held = permits.clone();
        let join = self.handle.spawn_blocking(move || {
            let _permits = held;
            work()
        });
        MediaWorker { join: Some(join), runtime: self.handle.clone(), permits: Some(permits) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workers(total: usize) -> MediaWorkers {
        MediaWorkers::new(
            tokio::runtime::Handle::current(),
            MediaWorkerLimits {
                total,
                decoders: 1,
                recordings: 1,
                images: 1,
                formatting: 1,
                eot_models: 1,
                vad_models: 1,
                uploads: 1,
                worker_io: 1,
            },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn capacity_notification_observes_released_reservations() {
        let workers = workers(3);
        let permit = workers.try_admit(MediaWorkerClass::Recording).unwrap();
        assert!(workers.try_admit(MediaWorkerClass::Recording).is_err());
        let available = workers.wait_for_capacity(MediaWorkerClass::Recording);
        tokio::pin!(available);
        assert!(futures_util::poll!(available.as_mut()).is_pending());
        drop(permit);
        available.await;
        let _admitted = workers.try_admit(MediaWorkerClass::Recording).unwrap();
    }

    #[tokio::test]
    async fn failed_partial_admission_notifies_after_releasing_its_reservations() {
        let workers = workers(3);
        let _recording = workers.try_admit(MediaWorkerClass::Recording).unwrap();
        let changed = workers.released.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        assert!(workers.try_admit(MediaWorkerClass::Recording).is_err());
        assert!(futures_util::poll!(changed.as_mut()).is_ready());
        let _decoder = workers.try_admit(MediaWorkerClass::Decoder).unwrap();
    }

    #[tokio::test]
    async fn refusing_admission_without_a_reservation_does_not_notify_capacity() {
        let workers = workers(1);
        let permit = workers.try_admit(MediaWorkerClass::Recording).unwrap();
        let changed = workers.released.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        assert!(workers.try_admit(MediaWorkerClass::Decoder).is_err());
        assert!(futures_util::poll!(changed.as_mut()).is_pending());
        drop(permit);
        changed.await;
    }

    #[tokio::test]
    async fn native_payload_saturation_reserves_health_sampling_capacity() {
        let workers = workers(4);
        let _decoder = workers.try_admit(MediaWorkerClass::Decoder).unwrap();
        let _recording = workers.try_admit(MediaWorkerClass::Recording).unwrap();
        let _image = workers.try_admit(MediaWorkerClass::Image).unwrap();
        assert!(matches!(
            workers.try_admit(MediaWorkerClass::EotModel),
            Err(MediaWorkerError::Exhausted(MediaWorkerClass::EotModel))
        ));
        let _health = workers.try_admit(MediaWorkerClass::WorkerIo).unwrap();
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
    async fn models_and_uploads_share_the_payload_ceiling() {
        let workers = workers(3);
        let (release_decoder, decoder_wait) = std::sync::mpsc::channel();
        let (release_model, model_wait) = std::sync::mpsc::channel();
        let decoder =
            workers.try_spawn(MediaWorkerClass::Decoder, move || decoder_wait.recv()).unwrap();
        let model =
            workers.try_spawn(MediaWorkerClass::EotModel, move || model_wait.recv()).unwrap();
        assert!(matches!(
            workers.try_spawn(MediaWorkerClass::Upload, || ()),
            Err(MediaWorkerError::Exhausted(MediaWorkerClass::Upload))
        ));
        release_decoder.send(()).unwrap();
        decoder.await.unwrap().unwrap();
        workers.try_spawn(MediaWorkerClass::Upload, || ()).unwrap().await.unwrap();
        assert!(matches!(
            workers.try_spawn(MediaWorkerClass::EotModel, || ()),
            Err(MediaWorkerError::Exhausted(MediaWorkerClass::EotModel))
        ));
        release_model.send(()).unwrap();
        model.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn idle_decoders_leave_other_class_capacity_available() {
        let workers = workers(3);
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

lk_agents_abi::inventory::submit! {
    lk_agents_abi::utils::workers::WorkerService {
        try_spawn: |class, job| {
            let workers = crate::process::workers()
                .map_err(|error| MediaWorkerError::Unavailable(error.to_string()))?;
            let receipt = workers.try_spawn(class, job)?;
            Ok(Box::pin(async move { receipt.await.map_err(|error| error.to_string()) }))
        },
    }
}
