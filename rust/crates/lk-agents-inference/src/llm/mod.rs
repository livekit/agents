pub mod algorithms;
pub mod gateway;
pub mod models;
pub mod parse;
pub mod request;
pub mod thinking;

/// Open a completion whose single stream owns all retry decisions. A fallback chain's terminal
/// errors are final, so its own loop is never replayed by this request.
pub async fn chat(
    llm: std::sync::Arc<dyn lk_agents_abi::Llm>,
    cx: &crate::telemetry::Cx,
    req: &lk_agents_abi::ChatRequest,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
    observer: Option<crate::retry::Observer>,
) -> Result<lk_agents_abi::LlmStream, lk_agents_abi::errors::ProviderError> {
    let options = req.conn_options;
    options.validate().map_err(|error| {
        lk_agents_abi::errors::ApiError::new(error.to_string()).with_retryable(false)
    })?;
    let mut attempt = req.clone();
    attempt.conn_options.max_retry = 0;
    let provider = llm.with_conn_options(&attempt.conn_options).unwrap_or_else(|| llm.clone());
    let cx = cx.clone();
    let open = std::sync::Arc::new(move || {
        let (provider, attempt, cx) = (provider.clone(), attempt.clone(), cx.clone());
        Box::pin(async move { provider.chat(&cx, &attempt).await })
            as lk_agents_abi::BoxFuture<'static, _>
    });
    let report: crate::retry::Observer = std::sync::Arc::new(move |error, retry, attempt| {
        llm.failed(error.clone(), matches!(retry, lk_agents_core::request::Retry::At(_)));
        if let Some(observer) = &observer {
            observer(error, retry, attempt);
        }
    });
    Ok(crate::retry::stream(
        &options,
        clock,
        open,
        lk_agents_abi::LlmChunk::has_response,
        report,
        tokio_util::sync::CancellationToken::new(),
    )
    .await)
}

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
