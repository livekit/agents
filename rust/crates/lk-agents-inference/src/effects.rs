//! Provider lifecycle effect contracts, grouped by the machine that produces them.
//!
//! These are independent protocols, not variants of a common provider effect enum. Their
//! executors own sockets, timers, delivery and cleanup; machines receive outcomes and time.

use crate::avatar::machine::{AvatarRequestId, CreateFailure, ProviderSession, Retirement};
use crate::interruption::{OverlappingSpeechEvent, RequestIdentity};
use crate::llm::machine::LlmRequestId;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{LlmChunk, SpeechEvent, SynthesizedAudio, TranscriptChunk, Ts};

/// Recognition stream work from [`SttStream`](crate::stt::machine::SttStream), performed by
/// the private STT executor (`stt/executor.rs`).
#[derive(Debug, Clone, PartialEq)]
pub enum SttStreamEffect {
    OpenSttStream { stream_id: u64, stt: ResourceId },
    SttFinalize { stream_id: u64 },
    Deliver(SpeechEvent),
    EndOfSpeech,
    Finished(Result<(), ApiError>),
}

/// Synthesis stream work from [`TtsSynthesis`](crate::tts::machine::TtsSynthesis), performed
/// by the private TTS executor (`tts/pump.rs`).
pub enum TtsSynthesisEffect {
    TtsSynthesisStart,
    TtsTextSend(String),
    TtsFinalFlush,
    TtsAlignment { segment_id: String, tokens: Vec<TranscriptChunk> },
    TtsAudio(SynthesizedAudio),
    Finished(Result<(), ApiError>),
    TtsCancel,
}

/// LLM request work from [`LlmRequest`](crate::llm::machine::LlmRequest), performed by
/// [`LlmChunks`](crate::llm::gateway::LlmChunks).
#[derive(Debug, Clone, PartialEq)]
pub enum LlmRequestEffect {
    LlmRequestStart {
        request_id: LlmRequestId,
        attempt: u32,
        llm: ResourceId,
    },
    LlmRequestCancel {
        request_id: LlmRequestId,
        attempt: u32,
    },
    Deliver(LlmChunk),
    /// A failed attempt that will be retried.
    AttemptFailed {
        attempt: u32,
        error: ApiError,
    },
    /// The request's single outcome. Cancellation has none: the consumer asked for it.
    Finished(Result<(), ApiError>),
    /// An attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete {
        request_id: LlmRequestId,
        attempt: u32,
    },
}

/// Fallback request work from [`ChainRequest`](crate::fallback::machine::ChainRequest),
/// performed by the fallback chain and realtime-session adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackChainEffect {
    /// Open entry `index`: its request, stream, synthesis or session.
    Start {
        attempt: u64,
        index: usize,
        backend: ResourceId,
    },
    Cancel {
        attempt: u64,
    },
    WakeAt(Ts),
    Finished(Result<(), ProviderError>),
}

/// Connection/reconnection work from
/// [`RealtimeConnection`](crate::realtime::machine::connection::RealtimeConnection), performed
/// by the OpenAI realtime session adapter.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeConnectionEffect {
    Connect { generation: u64 },
    Close { generation: u64 },
    Retry { error: ApiError },
    Failed { error: ApiError },
    CleanupIncomplete { generation: u64 },
}

/// Provider-session acquisition/retirement work from
/// [`AvatarSessionLifecycle`](crate::avatar::machine::AvatarSessionLifecycle), performed by
/// [`GatewayAvatar`](crate::avatar::gateway::GatewayAvatar).
#[derive(Debug, Clone, PartialEq)]
pub enum AvatarSessionEffect {
    /// The caller is refused: a creation is in flight, a session was adopted or the instance
    /// is closed.
    CreateRefused,
    AvatarSessionCreate {
        request: AvatarRequestId,
        attempt: u32,
        avatar: ResourceId,
    },
    AvatarSessionTerminate {
        request: AvatarRequestId,
        attempt: u32,
        avatar: ResourceId,
        session: ProviderSession,
    },
    AvatarRequestCancel {
        request: AvatarRequestId,
        attempt: u32,
    },
    /// A failed attempt that will be retried.
    AttemptFailed {
        request: AvatarRequestId,
        attempt: u32,
        error: ApiError,
    },
    /// The create caller's single outcome. A caller that stopped waiting has none.
    CreateFinished(Result<ProviderSession, CreateFailure>),
    Retiring {
        session: ProviderSession,
        reason: Retirement,
    },
    /// A session that cannot be terminated (no provider id or token) is released to its idle
    /// timeout.
    Unterminable(ProviderSession),
    SessionTerminated(ProviderSession),
    /// The instance still holds `session`; the next close retries.
    TerminateFailed {
        session: ProviderSession,
        error: ApiError,
    },
    /// An attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete {
        request: AvatarRequestId,
        attempt: u32,
    },
    /// A retiring create was still unanswered at its deadline: a session it may have made bills
    /// until the provider's idle timeout; a later answer is still terminated.
    RetirementIncomplete {
        request: AvatarRequestId,
    },
}

/// Adaptive-interruption work from
/// [`InterruptionMachine`](crate::interruption::machine::InterruptionMachine), performed by
/// the interruption transport (`interruption/transport.rs`).
#[derive(Debug, Clone, PartialEq)]
pub enum InterruptionEffect {
    InterruptionInference { identity: RequestIdentity, samples: Vec<i16> },
    Overlap { request: Option<ReqId>, event: Box<OverlappingSpeechEvent> },
}
