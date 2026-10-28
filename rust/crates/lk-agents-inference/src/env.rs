//! What the hosting process lends the model clients beyond `lk_agents_net::env` (decision 88):
//! the media worker pool and image preparation.

use std::sync::Arc;

use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_abi::ImageContent;
use lk_agents_media::workers::MediaWorkers;

/// [`lk_agents_net::env::Env`] plus the media workers and image preparation the model clients use.
pub trait Env: lk_agents_net::env::Env {
    /// The process's admitted CPU workers for model, codec and image work.
    fn workers() -> Result<MediaWorkers, String>;
    /// How a realtime model prepares the images it sends.
    fn images() -> Arc<dyn ImageResolver + Send + Sync>;
}

/// [`Env::workers`] handed to a client that acquires the pool where it uses it: after a fork the
/// process refuses, where a pool held from before would wait on threads the child lacks.
pub type WorkerPool = fn() -> Result<MediaWorkers, String>;

pub use lk_agents_net::env::Standalone;

/// A Rust caller with no host: one worker pool for the process, and no image preparation.
impl Env for Standalone {
    fn workers() -> Result<MediaWorkers, String> {
        static WORKERS: std::sync::OnceLock<Result<MediaWorkers, String>> =
            std::sync::OnceLock::new();
        WORKERS
            .get_or_init(|| {
                MediaWorkers::new(Default::default()).map_err(|error| error.to_string())
            })
            .clone()
    }

    fn images() -> Arc<dyn ImageResolver + Send + Sync> {
        Arc::new(NoImagePreparation)
    }
}

/// Images need the host process's preparation service.
struct NoImagePreparation;

impl ImageResolver for NoImagePreparation {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        Err(format!("image {}: no image preparation in this process", image.id))
    }
}
