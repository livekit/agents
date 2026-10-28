//! `inference/llm.py` -- the gateway LLM, HTTPS and OpenAI-compatible (§1.4): model tables
//! ([`models`]), kwargs composition ([`request`]), streaming decode ([`parse`]), the request
//! exchange ([`build_request`]) and the request's lifecycle ([`lifecycle`]). The runtime executes
//! the exchange (`lk-agents-runtime`'s gateway LLM).

pub mod lifecycle;
pub mod models;
pub mod parse;
pub mod request;

use crate::abi::ToolSpec;
use crate::headers::{Headers, identification_headers};
use crate::transport::HttpRequest;
use lk_agents_abi::ChatContext;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use request::ChatArgs;

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

/// The `chat.completions.create` exchange for one request: the body from the conversation, the
/// tools and the composed kwargs; the headers from `extra_headers`, the job's identification, the
/// provider pin and `bearer`, the gateway token the runtime signed.
pub fn build_request(
    opts: &LlmOptions,
    identification: Option<&Headers>,
    images: &(dyn ImageResolver + Send + Sync),
    chat_ctx: &ChatContext,
    tools: &[ToolSpec],
    args: ChatArgs,
    bearer: &str,
) -> Result<HttpRequest, ApiError> {
    let mut extra = request::compose_extra(&args, &opts.extra_kwargs);
    // Before `tool_choice` is popped (as in `LLMStream.__init__`): the model must see the
    // tool list as passed to `chat()`.
    models::drop_unsupported_params(&opts.model, &mut extra, !tools.is_empty());
    let extra_headers = extra.remove("extra_headers");

    let formatted = lk_agents_abi::llm::provider_format::openai::to_chat_ctx(chat_ctx, images)
        .map_err(|e| ApiError::new(format!("failed to format chat context: {e}")))?;
    let schemas = request::tool_schemas(tools)?;
    let body = request::build_body(&opts.model, formatted.messages, &schemas, extra);

    // `extra_headers` (the composed kwarg, so the instance's replaces the call's), overlaid
    // with the job's identification, the provider pin and the bearer.
    let mut headers: Headers = match extra_headers {
        Some(serde_json::Value::Object(map)) => map
            .into_iter()
            .map(|(k, v)| (k, v.as_str().map_or_else(|| v.to_string(), str::to_owned)))
            .collect(),
        _ => Headers::new(),
    };
    let class = args.inference_class.as_deref().or(opts.inference_class.as_deref());
    for (name, value) in identification_headers(identification, class, opts.provider.as_deref()).into_vec() {
        headers.set(name, value);
    }
    headers.set("Authorization", format!("Bearer {bearer}"));
    headers.set("Content-Type", "application/json");

    Ok(HttpRequest {
        url: request::completions_url(&opts.base_url),
        headers: headers.into_vec(),
        body: serde_json::to_vec(&serde_json::Value::Object(body))
            .map_err(|e| ApiError::new(e.to_string()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::NoImages;

    fn opts() -> LlmOptions {
        LlmOptions {
            model: "openai/gpt-4o".into(),
            provider: None,
            base_url: "https://gw/v1".into(),
            api_key: "k".into(),
            api_secret: "s".into(),
            inference_class: None,
            extra_kwargs: Default::default(),
        }
    }

    fn header<'a>(request: &'a HttpRequest, name: &str) -> Option<&'a str> {
        request.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    // Catches: job headers sent when there is no job, the bearer missing, or a wrong path.
    #[test]
    fn with_no_job_the_request_carries_no_livekit_job_headers() {
        let request = build_request(&opts(), None, &NoImages, &ChatContext::default(), &[], ChatArgs::default(), "t")
            .unwrap();
        let job: Vec<&str> = request
            .headers
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| ["X-LiveKit-Room-ID", "X-LiveKit-Job-ID", "X-LiveKit-Agent-ID"].contains(k))
            .collect();
        assert!(job.is_empty(), "got {job:?}");
        assert_eq!(header(&request, "Authorization"), Some("Bearer t"));
        assert!(request.url.ends_with("/v1/chat/completions"), "{}", request.url);
    }

    // Catches: the host's identification headers (the job's `X-LiveKit-*` identity, read where
    // `chat()` runs) not reaching the request, or the native priority/User-Agent leaking in beside
    // them. The pinned provider and the bearer are still ours.
    #[test]
    fn the_hosts_identification_headers_ride_on_the_request() {
        let opts = LlmOptions { provider: Some("openai".into()), inference_class: Some("low".into()), ..opts() };
        let ident: Headers = [
            ("User-Agent", "LiveKit Agents/1 (python 3)"),
            ("X-LiveKit-Room-ID", "RM_1"),
            ("X-LiveKit-Job-ID", "AJ_1"),
            ("X-LiveKit-Agent-ID", "PA_1"),
            ("X-LiveKit-Session-ID", "inference_abc"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let request =
            build_request(&opts, Some(&ident), &NoImages, &ChatContext::default(), &[], ChatArgs::default(), "t")
                .unwrap();
        assert_eq!(header(&request, "X-LiveKit-Room-ID"), Some("RM_1"));
        assert_eq!(header(&request, "X-LiveKit-Job-ID"), Some("AJ_1"));
        assert_eq!(header(&request, "X-LiveKit-Agent-ID"), Some("PA_1"));
        assert_eq!(header(&request, "X-LiveKit-Session-ID"), Some("inference_abc"));
        assert_eq!(header(&request, "User-Agent"), Some("LiveKit Agents/1 (python 3)"));
        assert_eq!(header(&request, "X-LiveKit-Inference-Priority"), None, "the host's map already decided it");
        assert_eq!(header(&request, "X-LiveKit-Inference-Provider"), Some("openai"));
    }
}
