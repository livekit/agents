//! Gateway request composition over the shared agent core's formatting algorithms.

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ChatContext, ToolSpec};
use lk_agents_inference::headers::{Headers, identification_headers};
use lk_agents_inference::llm::request::ChatArgs;
use lk_agents_inference::llm::{LlmOptions, models, request};
use lk_agents_net::transport::HttpRequest;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

/// `ToolContext.parse_function_tools("openai", strict=True)`; the gateway always uses strict.
pub fn tool_schemas(tools: &[ToolSpec]) -> Result<Vec<Value>, ApiError> {
    tools
        .iter()
        // The inference gateway's chat-completions endpoint supports client functions only.
        .filter(|tool| tool.provider_tool.is_none())
        .map(|tool| {
            lk_agents_core::llm::schema::build_strict_openai_schema(
                &tool.name,
                Some(&tool.description),
                &tool.json_schema,
            )
            .map(Value::Object)
            .map_err(|e| ApiError::new(format!("tool {:?} has an unusable schema: {e}", tool.name)))
        })
        .collect()
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

    let formatted = lk_agents_core::llm::provider_format::openai::to_chat_ctx(chat_ctx, images)
        .map_err(|e| ApiError::new(format!("failed to format chat context: {e}")))?;
    let schemas = tool_schemas(tools)?;
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
    for (name, value) in
        identification_headers(identification, class, opts.provider.as_deref()).into_vec()
    {
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
    use lk_agents_inference::testkit::NoImages;

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
        let request = build_request(
            &opts(),
            None,
            &NoImages,
            &ChatContext::default(),
            &[],
            ChatArgs::default(),
            "t",
        )
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
        let opts = LlmOptions {
            provider: Some("openai".into()),
            inference_class: Some("low".into()),
            ..opts()
        };
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
        let request = build_request(
            &opts,
            Some(&ident),
            &NoImages,
            &ChatContext::default(),
            &[],
            ChatArgs::default(),
            "t",
        )
        .unwrap();
        assert_eq!(header(&request, "X-LiveKit-Room-ID"), Some("RM_1"));
        assert_eq!(header(&request, "X-LiveKit-Job-ID"), Some("AJ_1"));
        assert_eq!(header(&request, "X-LiveKit-Agent-ID"), Some("PA_1"));
        assert_eq!(header(&request, "X-LiveKit-Session-ID"), Some("inference_abc"));
        assert_eq!(header(&request, "User-Agent"), Some("LiveKit Agents/1 (python 3)"));
        assert_eq!(
            header(&request, "X-LiveKit-Inference-Priority"),
            None,
            "the host's map already decided it"
        );
        assert_eq!(header(&request, "X-LiveKit-Inference-Provider"), Some("openai"));
    }

    // Catches: a tool schema reaching the gateway un-strict, or a bad schema failing silently.
    #[test]
    fn gateway_chat_completions_ignores_all_server_side_provider_declarations() {
        for family in [None, Some("openai"), Some("xai"), Some("gemini"), Some("mistral")] {
            let declaration = ToolSpec {
                name: "provider_id".into(),
                description: String::new(),
                json_schema: Value::Null,
                flags: Default::default(),
                provider_tool: Some(lk_agents_abi::llm::llm::ProviderToolDeclaration {
                    provider: family.map(str::to_owned),
                    payload: json!({"type": "web_search"}),
                }),
            };
            assert!(tool_schemas(&[declaration]).unwrap().is_empty());
        }
    }

    #[test]
    fn tool_schemas_are_built_in_the_strict_openai_form() {
        let tool = lk_agents_abi::ToolSpec {
            provider_tool: None,
            name: "lookup".into(),
            description: "look it up".into(),
            json_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            flags: Default::default(),
        };
        let built = tool_schemas(std::slice::from_ref(&tool)).expect("a usable schema");
        assert_eq!(built.len(), 1);
        assert_eq!(built[0]["type"], json!("function"));
        assert_eq!(built[0]["function"]["name"], json!("lookup"));
        assert_eq!(built[0]["function"]["strict"], json!(true));
        assert_eq!(built[0]["function"]["description"], json!("look it up"));
        assert_eq!(built[0]["function"]["parameters"]["additionalProperties"], json!(false));
        assert_eq!(built[0]["function"]["parameters"]["required"], json!(["q"]));
        assert!(tool_schemas(&[]).unwrap().is_empty());
    }
}
