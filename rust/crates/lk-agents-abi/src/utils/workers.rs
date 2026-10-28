//! Link-time runtime service for admitted native work. Implementations own admission and settlement;
//! callers never wait for a slot or release a running job's capacity by cancelling its receipt.

use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerClass {
    Decoder,
    Recording,
    Image,
    EotModel,
    Upload,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerError {
    #[error("worker limits must be positive and each class must fit the total ceiling")]
    InvalidLimits,
    #[error("worker capacity exhausted for {0:?}")]
    Exhausted(WorkerClass),
    #[error("native worker service unavailable: {0}")]
    Unavailable(String),
}

pub type WorkerJoin = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;
pub type NativeJob = Box<dyn FnOnce() + Send + 'static>;

/// Linked by the runtime; registration starts no threads and allocates no process state.
pub struct WorkerService {
    pub try_spawn: fn(WorkerClass, NativeJob) -> Result<WorkerJoin, WorkerError>,
}

inventory::collect!(WorkerService);

pub fn try_spawn(class: WorkerClass, job: NativeJob) -> Result<WorkerJoin, WorkerError> {
    let mut services = inventory::iter::<WorkerService>.into_iter();
    let service =
        services.next().ok_or_else(|| WorkerError::Unavailable("no runtime linked".into()))?;
    if services.next().is_some() {
        return Err(WorkerError::Unavailable("multiple runtimes linked".into()));
    }
    (service.try_spawn)(class, job)
}
