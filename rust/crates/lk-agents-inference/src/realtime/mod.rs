//! Synchronous realtime conversation, response and connection policy.

pub mod dialect;
pub mod host;
pub mod machine;
pub mod openai;
pub mod protocol;
pub mod remote;
#[cfg(test)]
mod testfix;

pub use dialect::{
    Dialect, INFERENCE, OPENAI, Protocol, TurnDetection, UnknownDialect, VOICE_LIVE, XAI,
};
pub use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError, TextDelta,
};
pub use machine::connection::RealtimeConnection;
pub use machine::{
    ContextRetirement, OptionsPatch, Outgoing, RealtimeMachine, ReplyKind, SessionOptions, Step,
};
pub use protocol::Incoming;
pub use remote::{RemoteChatContext, RemoteError};
