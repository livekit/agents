//! Model services injected through the shared process environment.

use std::sync::Arc;

use lk_agents_abi::ImageContent;
use lk_agents_abi::llm::provider_format::{ImageResolver, SerializedImage};
use lk_agents_media::workers::MediaWorkers;

/// Access the pool at use time so a process fork is checked before dispatching work.
pub type WorkerPool = fn() -> Result<MediaWorkers, String>;

pub struct Services {
    pub workers: WorkerPool,
    pub images: Arc<dyn ImageResolver + Send + Sync>,
    pub strip_assistant_markup: fn(&str) -> String,
}

pub type Standalone = lk_agents_net::env::Standalone<Services>;

impl Default for Services {
    fn default() -> Self {
        Self {
            workers: standalone_workers,
            images: Arc::new(NoImagePreparation),
            strip_assistant_markup: str::to_owned,
        }
    }
}

fn standalone_workers() -> Result<MediaWorkers, String> {
    static WORKERS: std::sync::OnceLock<Result<MediaWorkers, String>> = std::sync::OnceLock::new();
    WORKERS
        .get_or_init(|| MediaWorkers::new(Default::default()).map_err(|error| error.to_string()))
        .clone()
}

struct NoImagePreparation;

impl ImageResolver for NoImagePreparation {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        Err(format!("image {}: no image preparation in this process", image.id))
    }
}
