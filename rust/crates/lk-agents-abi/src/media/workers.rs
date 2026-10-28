//! Native worker admission contracts shared by the injected pool and host services.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerClass {
    Decoder,
    Recording,
    Image,
    Formatting,
    EotModel,
    VadModel,
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
