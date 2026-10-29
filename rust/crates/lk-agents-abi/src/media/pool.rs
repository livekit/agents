//! Native worker admission contracts shared by the injected pool and host services.

use std::task::{Context, Poll};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobClass {
    Decoder,
    Recording,
    Image,
    Formatting,
    EotModel,
    VadModel,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MediaPoolError {
    #[error("worker limits must be positive and each class must fit the total ceiling")]
    InvalidLimits,
    #[error("worker capacity exhausted for {0:?}")]
    Exhausted(JobClass),
    #[error("native worker service unavailable: {0}")]
    Unavailable(String),
}

/// The process's admitted native worker pool as the runtime lends it;
/// `lk_agents_media::pool::MediaPool` is its typed face.
pub trait MediaExecutor: Send + Sync {
    /// Reserve a slot of `class`, or refuse at once.
    fn admit(&self, class: JobClass) -> Result<(), MediaPoolError>;
    /// Release a slot [`admit`](Self::admit) reserved, waking capacity waiters.
    fn release(&self, class: JobClass);
    /// Ready once `class` could be admitted, else woken by the next release; reserves nothing.
    fn poll_capacity(&self, class: JobClass, cx: &mut Context<'_>) -> Poll<()>;
    /// Run an admitted job on a worker thread.
    fn run(&self, job: Box<dyn FnOnce() + Send>);
}
