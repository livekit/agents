//! Provider fallback: one generic [`FallbackChain`] for LLM/STT/TTS, and the realtime adapter's
//! decisions ([`realtime_policy`]). A fallback adapter implements the same traits as its entries.
//! Realtime differs: failure is a pushed event, recovery is a cooldown, a healthy primary is not
//! preempted, and availability is per session. The `metrics_collected` tee lives in
//! `lk-agents-py`, since these traits carry no event emitter.

mod chain;
mod channel;
mod llm;
pub mod policy;
pub mod realtime;
/// The realtime adapter's decisions, for a host that owns the model objects.
pub mod realtime_policy;
mod stt;
mod tts;

pub use chain::{
    Attempt, AvailabilityChanged, Backend, ChainObserver, ChainOutcome, FallbackChain,
    IgnoreEvents, NoBackends, Recovery, RecoverySlot,
};
pub use llm::{FallbackLlm, LlmBackend};
pub use policy::{
    AfterOutput, ChainPolicy, bound_views, caller_default, conn_options_for_attempt,
    conn_options_for_recovery, resampling_plan,
};
pub use realtime::{CapabilityMismatch, merge_capabilities_of};
pub use realtime_policy::{AvailabilityChange, ErrorPlan, RealtimePolicy};
pub use stt::{
    FallbackStt, InputTee, SttBackend, SttChainError, SttReq, SttSession, merge_capabilities as merge_stt_capabilities,
};
pub use tts::{
    FallbackTts, Token, TokenLog, TtsBackend, TtsChainError, TtsReq, merge_capabilities as merge_tts_capabilities,
};

#[cfg(test)]
mod adapter_tests;
#[cfg(test)]
mod tests;
