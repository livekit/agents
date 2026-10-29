//! The realtime model over its WebSocket: session execution, framing and the declared classes.

mod encode;
#[cfg(test)]
mod machine_tests;
mod model;
pub(crate) mod queue;
pub mod session;
#[cfg(test)]
mod testfix;
pub mod wire;
mod writer;

pub use super::{Dialect, INFERENCE, OPENAI, OptionsPatch, SessionOptions, XAI};
pub use lk_agents_abi::llm::realtime::{AudioDelta, Event, GenerationCreated};
pub use session::{Auth, Connect, NativeRealtime, SessionHandle};
pub use wire::{Azure, RandomIds, SeqIds};
