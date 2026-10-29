//! Synchronous realtime conversation, response and connection policy.

pub mod connection;
pub mod dialect;
pub mod host;
pub mod machine;
pub mod native;
pub mod protocol;
pub mod remote;
#[cfg(test)]
mod testfix;

pub use connection::{ConnectionEffect, RealtimeConnection};
pub use dialect::{Dialect, INFERENCE, OPENAI, TurnDetection, UnknownDialect, XAI};
pub use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError, TextDelta, TokenUsage,
};
pub use machine::{
    ContextRetirement, OptionsPatch, Outgoing, RealtimeMachine, SessionOptions, Step,
};
pub use protocol::Incoming;
pub use remote::{RemoteChatContext, RemoteError};
