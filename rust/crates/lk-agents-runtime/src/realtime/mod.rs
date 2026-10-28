//! Realtime WebSocket execution, framing, media preparation and host integration.

mod encode;
mod event_retention;
pub mod fallback;
#[cfg(test)]
mod machine_tests;
#[cfg(feature = "media-images")]
mod model;
pub(crate) mod queue;
pub mod session;
#[cfg(test)]
mod testfix;
pub mod wire;
mod writer;

pub use lk_agents_abi::llm::realtime::{AudioDelta, Event, GenerationCreated};
pub use lk_agents_inference::realtime::{
    Dialect, INFERENCE, OPENAI, OptionsPatch, SessionOptions, XAI,
};
pub use session::{Auth, Connect, NativeRealtime, SessionHandle};
pub use wire::{Azure, RandomIds, SeqIds};

#[cfg(feature = "media-images")]
pub mod images;
