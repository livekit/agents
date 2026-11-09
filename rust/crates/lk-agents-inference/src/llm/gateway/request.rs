//! The `chat.completions.create` request, over the shared agent core's formatting algorithms:
//! to the LiveKit Inference gateway, or straight to a provider with its own key.

use crate::headers::{Headers, gateway_headers};
use crate::llm::request::ChatArgs;
use crate::llm::{Auth, BodyEncoding, LlmOptions, models, request};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ChatContext, ToolSpec};
use lk_agents_net::transport::HttpRequest;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

/// `ToolContext.parse_function_tools("openai", strict=strict)`. Chat completions take client
/// functions only: a provider's server-side tools belong to its own APIs.
pub fn tool_schemas(tools: &[ToolSpec], strict: bool) -> Result<Vec<Value>, ApiError> {
    tools
        .iter()
        .filter(|tool| tool.provider_tool.is_none())
        .map(|tool| {
            if !strict {
                return Ok(Value::Object(lk_agents_core::llm::schema::build_legacy_openai_schema(
                    &tool.name,
                    Some(&tool.description),
                    tool.json_schema.clone(),
                    false,
                )));
            }
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
/// tools and the composed kwargs (`extra_body` merged into it, `extra_query` onto the URL); the
/// headers from `extra_headers`, then, on the gateway, the job's identification and the provider
/// pin, and `bearer`.
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
    let extra_body = extra.remove("extra_body");
    let extra_query = extra.remove("extra_query");

    let formatted = lk_agents_core::llm::provider_format::openai::to_chat_ctx(chat_ctx, images)
        .map_err(|e| ApiError::new(format!("failed to format chat context: {e}")))?;
    let schemas = tool_schemas(tools, opts.strict_tools)?;
    let messages = formatted
        .messages
        .into_iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::new(format!("failed to serialize chat context: {e}")))?;
    let mut body = request::build_body(&opts.model, messages, &schemas, extra);
    if let Some(Value::Object(fields)) = extra_body {
        body.extend(fields);
    }

    // `extra_headers` (the composed kwarg, so the instance's replaces the call's), overlaid
    // with the gateway's identification and provider pin, and the bearer.
    let mut headers: Headers = match extra_headers {
        Some(Value::Object(map)) => map.into_iter().map(|(k, v)| (k, text(v))).collect(),
        _ => Headers::new(),
    };
    if let Auth::Gateway { .. } = opts.auth {
        let class = args.inference_class.as_deref().or(opts.inference_class.as_deref());
        for (name, value) in
            gateway_headers(identification, class, opts.provider.as_deref(), bearer)
        {
            headers.insert(name, value);
        }
        // The call's own class overrides the one the identification was read with.
        if let Some(class) = args.inference_class.as_deref().filter(|class| !class.is_empty()) {
            headers.insert(crate::headers::HEADER_INFERENCE_PRIORITY, class);
        }
        headers.remove(crate::headers::HEADER_SESSION_ID);
    }
    headers.insert("Authorization", format!("Bearer {bearer}"));
    let body = encode_body(&Value::Object(body), opts.body, &mut headers)?;

    let mut url = url::Url::parse(&request::completions_url(&opts.base_url))
        .map_err(|e| ApiError::new(format!("invalid base_url {:?}: {e}", opts.base_url)))?;
    if let Some(Value::Object(query)) = extra_query {
        url.query_pairs_mut().extend_pairs(query.into_iter().map(|(k, v)| (k, text(v))));
    }
    Ok(HttpRequest { url: url.into(), headers, body })
}

/// The body's bytes as `encoding` asks, with the headers that name it.
fn encode_body(
    body: &Value,
    encoding: BodyEncoding,
    headers: &mut Headers,
) -> Result<Vec<u8>, ApiError> {
    let invalid = |e: String| ApiError::new(format!("cannot encode the request: {e}"));
    let mut bytes = if encoding.msgpack {
        headers.insert("Content-Type", "application/vnd.msgpack");
        rmp_serde::to_vec_named(body).map_err(|e| invalid(e.to_string()))?
    } else {
        headers.insert("Content-Type", "application/json");
        serde_json::to_vec(body).map_err(|e| invalid(e.to_string()))?
    };
    if encoding.gzip {
        bytes = lk_agents_net::http::gzip(&bytes);
        headers.insert("Content-Encoding", "gzip");
    }
    Ok(bytes)
}

/// A header or query value: a string as itself, anything else as its JSON.
fn text(value: Value) -> String {
    value.as_str().map_or_else(|| value.to_string(), str::to_owned)
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
