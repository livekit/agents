//! What the hosting process lends inference's host-constructed classes (decision 88): the shared
//! process environment plus the media worker pool, image preparation and the voice core's
//! assistant-markup projection.

use std::sync::Arc;

use lk_agents_abi::ImageContent;
use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_media::workers::MediaWorkers;
pub use lk_agents_net::env::Standalone;

/// Access the pool at use time so a process fork is checked before dispatching work.
pub type WorkerPool = fn() -> Result<MediaWorkers, String>;

pub trait Env: lk_agents_net::env::Env {
    /// The process's media worker pool.
    fn workers() -> Result<MediaWorkers, String>;
    /// How a request's images are prepared for a provider.
    fn images() -> Arc<dyn ImageResolver + Send + Sync>;
    /// An assistant message as a judge reads it, without expressive markup.
    fn strip_assistant_markup(text: &str) -> String;
}

/// A Rust caller with no host: its own worker pool, no image preparation, text kept as is.
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

    fn strip_assistant_markup(text: &str) -> String {
        text.to_owned()
    }
}

struct NoImagePreparation;

impl ImageResolver for NoImagePreparation {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        Err(format!("image {}: no image preparation in this process", image.id))
    }
}
