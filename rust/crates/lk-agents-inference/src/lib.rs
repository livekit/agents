//! The `Native*` inference-gateway clients, detectors and fallback adapters. The socket appears
//! only in [`transport`] and [`native`]; every other module is pure over bytes, JSON and state.

/// docs/CONTRACTS.md §4.7's provider traits, re-exported so a call site names one crate.
pub use lk_agents_abi as abi;

pub mod creds;
pub mod extra;
pub mod headers;
pub mod ids;
// The two model-backed detectors: same credentials and transport as the providers below.
pub mod eot;
pub mod interruption;

// A fallback adapter is a provider assembled from other providers, over any `Arc<dyn Llm>`.
pub mod fallback;

// AMD has no socket of its own: it classifies a greeting with this crate's LLM and STT.
pub mod amd;

pub mod llm;
pub mod stt;
pub mod testkit;
pub mod token;
pub mod transport;
pub mod tts;
pub mod vad;

pub mod native;

#[cfg(feature = "python")]
pub(crate) mod python;

// The `_utils` surface, flat, because the binding re-exports it under the same names (imported by
// `job.py:515`, `evals/evaluation.py:118` and `openai/responses/llm.py:16`).
pub use headers::{
    DEFAULT_INFERENCE_URL, HEADER_AGENT_ID, HEADER_INFERENCE_PRIORITY, HEADER_INFERENCE_PROVIDER,
    HEADER_JOB_ID, HEADER_ROOM_ID, HEADER_SESSION_ID, HEADER_USER_AGENT, HEADER_WORKER_TOKEN,
    JobIdentity, QUOTA_HEADER_FIELDS, STAGING_INFERENCE_URL, default_inference_url,
    default_inference_url_from_env, extract_quota_usage, inference_headers,
};
pub use ids::{RequestKind, create_inference_request_id, shortuuid};
pub use llm::models::{drop_unsupported_params, min_reasoning_effort};
pub use token::{TokenError, create_access_token};

/// This crate, under the name every other crate calls it by, so `#[derive(Host)]` paths resolve
/// here too.
extern crate self as lk_agents_inference;

// Two network clients over this crate's transport.
pub mod mcp;
pub mod realtime;
