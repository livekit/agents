//! The `Native*` inference-gateway clients, detectors and fallback adapters, over the shared
//! transport of `lk-agents-net`.

/// docs/CONTRACTS.md §4.7's provider traits, re-exported so a call site names one crate.
pub use lk_agents_abi as abi;

// Conversation judges: model calls run concurrently on one conversation.
pub mod evals;

pub mod extra;
pub mod headers;
pub mod ids;
pub mod images;
pub mod provider;
// The two model-backed detectors: same credentials and transport as the providers below.
pub mod eot;
pub mod interruption;

// A full-duplex speech model run as the session's realtime model.
pub mod duplex;

// A fallback adapter is a provider assembled from other providers, over any `Arc<dyn Llm>`.
pub mod fallback;

// AMD has no socket of its own: it classifies a greeting with this crate's LLM and STT.
pub mod amd;

// The avatar sessions the gateway provisions: one create and one terminate over HTTPS.
pub mod avatar;
pub mod codecs;

pub mod llm;
// `_core`'s inference lookups, declared in ABI.
pub mod local_models;
pub mod models;
pub mod stt;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod tts;
pub mod vad;

pub use ids::{RequestKind, create_inference_request_id, shortuuid};
pub use llm::models::{drop_unsupported_params, min_reasoning_effort};

/// This crate, under the name every other crate calls it by, so `#[derive(Host)]` paths resolve
/// here too.
extern crate self as lk_agents_inference;

// Two network clients over this crate's transport.
pub mod realtime;
