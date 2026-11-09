mod encode;
#[cfg(test)]
#[path = "openai_tests.rs"]
mod tests;
mod model;
pub(crate) mod queue;
pub mod session;
#[cfg(test)]
mod testfix;
pub mod wire;
mod writer;

pub use super::{Dialect, INFERENCE, OPENAI, OptionsPatch, SessionOptions, VOICE_LIVE, XAI};
pub use lk_agents_abi::llm::realtime::{AudioDelta, Event, GenerationCreated};
pub use session::{Auth, Connect, OpenAIRealtime, OpenAIRealtimeSession};
pub use wire::{Azure, RandomIds, SeqIds};
