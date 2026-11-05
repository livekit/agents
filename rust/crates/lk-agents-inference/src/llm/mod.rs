//! `inference/llm.py` -- the gateway LLM, HTTPS and OpenAI-compatible (§1.4): model tables
//! ([`models`]), kwargs composition ([`request`]), streaming decode ([`parse`]), the request
//! exchange and the request's lifecycle ([`machine`]), and the client executing it ([`gateway`]).

pub mod algorithms;
pub mod gateway;
pub mod machine;
pub mod models;
pub mod parse;
pub mod request;
pub mod thinking;

/// `_LLMOptions`.
#[derive(Debug, Clone)]
pub struct LlmOptions {
    pub model: String,
    pub base_url: String,
    pub auth: Auth,
    /// `X-LiveKit-Inference-Provider`. `None` lets the gateway route.
    pub provider: Option<String>,
    /// `X-LiveKit-Inference-Priority`.
    pub inference_class: Option<String>,
    /// Tool schemas in OpenAI's strict form; some OpenAI-compatible providers refuse it.
    pub strict_tools: bool,
    /// How the request body is sent; JSON by default.
    pub body: BodyEncoding,
    /// The host class it reports as (`llm.LLM.label`).
    pub label: String,
    pub extra_kwargs: serde_json::Map<String, serde_json::Value>,
}

/// A request body re-encoded for a provider that accepts it (Cerebras: msgpack, gzip), which
/// shortens the upload of a long conversation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BodyEncoding {
    /// `application/vnd.msgpack` instead of JSON.
    pub msgpack: bool,
    /// `Content-Encoding: gzip`.
    pub gzip: bool,
}

/// Who a request authenticates as.
#[derive(Debug, Clone)]
pub enum Auth {
    /// The LiveKit Inference gateway: a token signed with the project's key and secret; requests
    /// carry the job's identification and the provider pin.
    Gateway { api_key: String, api_secret: String },
    /// A provider's own API key, sent as the bearer straight to `base_url`; no LiveKit header.
    ProviderKey(String),
}

impl Auth {
    /// The `Authorization` bearer.
    pub fn bearer(&self) -> Result<String, lk_agents_abi::errors::ApiError> {
        match self {
            Auth::Gateway { api_key, api_secret } => {
                lk_agents_net::token::inference_bearer(api_key, api_secret)
            }
            Auth::ProviderKey(key) => Ok(key.clone()),
        }
    }
}
