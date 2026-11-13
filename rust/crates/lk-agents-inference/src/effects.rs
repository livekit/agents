use crate::interruption::{OverlappingSpeechEvent, RequestIdentity};
use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::turn::detector::ReqId;

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
    Connect {
        generation: u64,
    },
    Close {
        generation: u64,
    },
    /// A connect attempt failed and is retried after `backoff`, which the session announces.
    Retry {
        error: ApiError,
        backoff: lk_agents_core::request::Backoff,
    },
    /// The connected session dropped and is reconnected now.
    Reconnect {
        error: ApiError,
    },
    Failed {
        error: ApiError,
    },
    CleanupIncomplete {
        generation: u64,
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
