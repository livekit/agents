//! Building the `chat.completions.create` body: `LLM.chat()`'s kwargs composition and
//! `LLMStream._run()`'s final assembly, as two pure functions over `serde_json::Map`.

use crate::abi::{ToolChoice, ToolSpec};
use crate::headers::{HEADER_INFERENCE_PROVIDER, Headers};
use lk_agents_abi::errors::ApiError;
use serde_json::{Map, Value, json};

/// The OpenAI wire form of a [`ToolChoice`]; the named form is rebuilt, dropping extra keys.
pub fn tool_choice_to_json(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        // `Forbidden` is the literal `"none"` (not `NOT_GIVEN`).
        ToolChoice::Forbidden => json!("none"),
        ToolChoice::Named(name) => json!({"type": "function", "function": {"name": name}}),
    }
}

/// `ToolContext.parse_function_tools("openai", strict=True)`; the gateway always uses strict.
pub fn tool_schemas(tools: &[ToolSpec]) -> Result<Vec<Value>, ApiError> {
    tools
        .iter()
        .map(|tool| {
            lk_agents_abi::llm::schema::build_strict_openai_schema(
                &tool.name,
                Some(&tool.description),
                &tool.json_schema,
            )
            .map_err(|e| ApiError::new(format!("tool {:?} has an unusable schema: {e}", tool.name)))
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
pub struct ChatArgs {
    /// Merged first, so the lowest precedence.
    pub extra_kwargs: Option<Map<String, Value>>,
    pub parallel_tool_calls: Option<bool>,
    pub tool_choice: Option<ToolChoice>,
    /// Already converted by `llm_utils.to_openai_response_format`; for direct callers.
    pub response_format: Option<Value>,
    /// Overrides the instance's.
    pub inference_class: Option<String>,
}

/// `LLM.chat()`'s `extra` composition. The last step is `extra.update(self._opts.extra_kwargs)`:
/// the instance kwargs override everything passed to this call.
pub fn compose_extra(args: &ChatArgs, opts_extra: &Map<String, Value>) -> Map<String, Value> {
    let mut extra = args.extra_kwargs.clone().unwrap_or_default();

    // `parallel_tool_calls if is_given(...) else self._opts.extra_kwargs.get(..., NOT_GIVEN)`.
    let parallel = args
        .parallel_tool_calls
        .or_else(|| opts_extra.get("parallel_tool_calls").and_then(Value::as_bool));
    if let Some(parallel) = parallel {
        extra.insert("parallel_tool_calls".into(), json!(parallel));
    }

    // Only the per-call tool choice is read: the reference's normalisation of an instance-level
    // one is overwritten by the raw value below, so it reaches the wire verbatim.
    if let Some(choice) = args.tool_choice.clone() {
        extra.insert("tool_choice".into(), tool_choice_to_json(&choice));
    }

    if let Some(fmt) = args.response_format.clone() {
        extra.insert("response_format".into(), fmt);
    }

    for (k, v) in opts_extra {
        extra.insert(k.clone(), v.clone());
    }
    extra
}

/// `LLMStream._run()`'s body assembly, after [`super::models::drop_unsupported_params`] has run
/// over `extra`. `messages` and `tools` come from other crates; this only places them.
pub fn build_body(
    model: &str,
    messages: Vec<Value>,
    tools: &[Value],
    mut extra: Map<String, Value>,
) -> Map<String, Value> {
    // `if not self._tools:` -- an empty tool list pops tool_choice (else a 400).
    normalize_tool_choice(&mut extra, !tools.is_empty());

    let mut body = Map::new();
    body.insert("messages".into(), Value::Array(messages));
    // `tools=tool_schemas or openai.omit` -- an empty list is omitted, not sent as `[]`.
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools.to_vec()));
    }
    body.insert("model".into(), json!(model));
    body.insert("stream_options".into(), json!({"include_usage": true}));
    body.insert("stream".into(), json!(true));
    for (k, v) in extra {
        // `extra_headers` is a client-level kwarg, not a body field.
        if k == "extra_headers" {
            continue;
        }
        body.insert(k, v);
    }
    body
}

/// `extra_headers`: the inference headers plus, when the instance names one, the provider pin.
pub fn request_headers(mut base: Headers, provider: Option<&str>) -> Headers {
    // `if self._provider:` -- an empty provider string pins nothing.
    if let Some(provider) = provider
        && !provider.is_empty()
    {
        base.set(HEADER_INFERENCE_PROVIDER, provider);
    }
    base
}

/// `f"{base_url}/chat/completions"`, as `openai.AsyncClient(base_url=...)` resolves it.
pub fn completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect()
    }

    // Catches: "fixing" the precedence so per-call arguments win.
    #[test]
    fn the_instance_kwargs_override_the_per_call_ones() {
        let args = ChatArgs {
            extra_kwargs: Some(opts(&[("temperature", json!(0.9))])),
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        };
        let extra = compose_extra(
            &args,
            &opts(&[("temperature", json!(0.1)), ("tool_choice", json!("none"))]),
        );
        assert_eq!(extra["temperature"], json!(0.1));
        assert_eq!(extra["tool_choice"], json!("none"));
    }

    // Catches: passing the caller's tool_choice dict through unmodified.
    #[test]
    fn a_named_tool_choice_is_rebuilt_to_the_two_key_form() {
        let extra = compose_extra(
            &ChatArgs {
                tool_choice: Some(ToolChoice::Named("lookup".into())),
                ..Default::default()
            },
            &Map::new(),
        );
        assert_eq!(
            extra["tool_choice"],
            json!({"type": "function", "function": {"name": "lookup"}})
        );
    }

    // Catches: normalising, rejecting or dropping an instance-level `tool_choice` (it goes verbatim).
    #[test]
    fn an_instance_level_tool_choice_reaches_the_wire_verbatim() {
        for raw in [json!("Auto"), json!("required"), json!({"function": {"name": "f"}, "x": 1})] {
            let extra = compose_extra(&ChatArgs::default(), &opts(&[("tool_choice", raw.clone())]));
            assert_eq!(extra["tool_choice"], raw, "not normalised, not dropped");
        }
    }

    // Catches: a tool schema reaching the gateway un-strict, or a bad schema failing silently.
    #[test]
    fn tool_schemas_are_built_in_the_strict_openai_form() {
        let tool = crate::abi::ToolSpec {
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

    // Catches: R2b on `if not self._tools`.
    #[test]
    fn an_empty_tool_list_drops_both_the_tools_key_and_the_tool_choice() {
        let extra = opts(&[("tool_choice", json!("auto")), ("temperature", json!(0.2))]);
        let body = build_body("openai/gpt-4o", vec![], &[], extra);
        assert!(!body.contains_key("tool_choice"));
        assert!(!body.contains_key("tools"), "`tools=[] or omit` omits");
        assert_eq!(body["temperature"], json!(0.2));
    }

    // Catches: `extra_headers` reaching the request body.
    #[test]
    fn extra_headers_never_reaches_the_body() {
        let extra = opts(&[("extra_headers", json!({"X-LiveKit-Job-ID": "AJ_1"}))]);
        let body = build_body("m", vec![], &[json!({"type": "function"})], extra);
        assert!(!body.contains_key("extra_headers"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
    }

    // Catches: R2b on `if self._provider`.
    #[test]
    fn an_empty_provider_pins_no_header() {
        assert_eq!(request_headers(Headers::new(), Some("")).get(HEADER_INFERENCE_PROVIDER), None);
        assert_eq!(
            request_headers(Headers::new(), Some("openai")).get(HEADER_INFERENCE_PROVIDER),
            Some("openai")
        );
    }

    // Catches: double slashes with a trailing-slash base URL.
    #[test]
    fn the_completions_url_tolerates_a_trailing_slash() {
        assert_eq!(
            completions_url("https://agent-gateway.livekit.cloud/v1"),
            "https://agent-gateway.livekit.cloud/v1/chat/completions"
        );
        assert_eq!(
            completions_url("https://agent-gateway.livekit.cloud/v1/"),
            "https://agent-gateway.livekit.cloud/v1/chat/completions"
        );
    }
}

/// Whether a tool selection is valid for the available tools.
pub fn retain_tool_choice(has_tools: bool) -> bool { has_tools }

pub fn normalize_tool_choice(extra: &mut Map<String, Value>, has_tools: bool) {
    if !retain_tool_choice(has_tools) { extra.remove("tool_choice"); }
}
