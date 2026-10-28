//! Failures shared by native pull sources and their generated host-facing resources.

use super::ProviderError;

#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum PullError {
    /// `StreamStateError`: another pull is in flight.
    #[error("another pull of this stream is still in flight")]
    #[host(kind = "livekit.agents._exceptions.StreamStateError")]
    Overlap,
    #[error(transparent)]
    #[host(delegate)]
    Failed(#[from] ProviderError),
}
