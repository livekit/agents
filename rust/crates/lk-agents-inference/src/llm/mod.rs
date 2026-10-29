//! `inference/llm.py` -- the gateway LLM, HTTPS and OpenAI-compatible (§1.4): model tables
//! ([`models`]), kwargs composition ([`request`]), streaming decode ([`parse`]), the request
//! exchange and the request's lifecycle ([`lifecycle`]), and the client executing it ([`gateway`]).

pub mod algorithms;
pub mod gateway;
pub mod lifecycle;
pub mod models;
pub mod parse;
pub mod request;
pub mod thinking;

/// `_LLMOptions`.
#[derive(Debug, Clone)]
pub struct LlmOptions {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    /// `X-LiveKit-Inference-Provider`. `None` lets the gateway route.
    pub provider: Option<String>,
    /// `X-LiveKit-Inference-Priority`.
    pub inference_class: Option<String>,
    pub extra_kwargs: serde_json::Map<String, serde_json::Value>,
}
