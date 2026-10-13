//! `NativeRealtime`: the OpenAI Realtime wire protocol, and xAI over the same wire module via a
//! [`Dialect`]. The Python plugin classes remain, so native and host-backed coexist.
//!
//! [`wire`] (JSON in/out), [`dialect`] (xAI's deltas), [`remote`] (the server-conversation mirror),
//! [`machine`] (the state machine) and [`session`] (the driver). `generate_reply`'s future becomes
//! [`lk_agents_abi::RealtimeCommand::GenerateReply`] correlated with [`Event::GenerationCreated`] by `req_id`. No Azure
//! and no resampling (`push_audio` takes 24 kHz mono).

pub mod dialect;
pub mod machine;
pub mod remote;
pub mod session;
pub mod wire;

#[cfg(test)]
mod testfix;

pub use dialect::{Dialect, OPENAI, TurnDetection, UnknownDialect, XAI};
pub use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError, TextDelta, TokenUsage,
};
pub use machine::{RealtimeMachine, SessionOptions, Step};
pub use remote::{RemoteChatContext, RemoteError};
pub use session::{NativeRealtime, SessionHandle};
pub use wire::{IdGen, Incoming, RandomIds, SeqIds};
