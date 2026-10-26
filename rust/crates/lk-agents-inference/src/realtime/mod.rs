//! `NativeRealtime`: the OpenAI Realtime wire protocol, and xAI over the same wire module via a
//! [`Dialect`]. The Python plugin classes remain, so native and host-backed coexist.
//!
//! [`wire`] (JSON in/out), [`dialect`] (xAI's deltas), [`remote`] (the server-conversation mirror),
//! [`machine`] (the state machine) and [`session`] (the driver). `generate_reply`'s future becomes
//! [`lk_agents_abi::RealtimeCommand::GenerateReply`] correlated with [`Event::GenerationCreated`] by `req_id`. Azure
//! and LiveKit Inference are [`session::Connect`]s; input audio is resampled to 24 kHz mono.

pub mod dialect;
pub mod machine;
#[cfg(feature = "python")]
mod python;
pub mod remote;
pub mod session;
pub mod wire;

#[cfg(test)]
mod testfix;

pub use dialect::{Dialect, INFERENCE, OPENAI, TurnDetection, UnknownDialect, XAI};
pub use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeCapabilities, RealtimeError, RealtimeModelError, TextDelta, TokenUsage,
};
pub use machine::{OptionsPatch, RealtimeMachine, SessionOptions, Step};
pub use remote::{RemoteChatContext, RemoteError};
pub use session::{Auth, Connect, NativeRealtime, SessionHandle};
pub use wire::{Azure, IdGen, Incoming, RandomIds, SeqIds, WireShape};
