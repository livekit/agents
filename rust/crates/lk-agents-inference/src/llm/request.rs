use crate::abi::ToolChoice;
use crate::headers::{HEADER_INFERENCE_PROVIDER, Headers};
use lk_agents_core::llm::provider_format::openai::Message;
use serde::Serialize;
use serde_json::{Map, Value};

/// `tool_choice` in the chat completions form: a mode, or one named function.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum WireToolChoice {
    Mode(ToolChoiceMode),
    Function(NamedFunction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    Auto,
    Required,
    /// [`ToolChoice::Forbidden`]: the literal `"none"`, not an omitted field.
    None,
}

/// `{"type": "function", "function": {"name": ...}}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "function")]
pub struct NamedFunction {
    pub function: FunctionName,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionName {
    pub name: String,
}

impl From<&ToolChoice> for WireToolChoice {
    fn from(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Mode(ToolChoiceMode::Auto),
            ToolChoice::Required => Self::Mode(ToolChoiceMode::Required),
            ToolChoice::Forbidden => Self::Mode(ToolChoiceMode::None),
            ToolChoice::Named(name) => {
                Self::Function(NamedFunction { function: FunctionName { name: name.clone() } })
            }
        }
    }
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

/// The `chat.completions.create` body: the fields the framework sets, then the caller's other
/// kwargs as given.
#[derive(Debug, Clone, Serialize)]
pub struct ChatBody {
    pub model: String,
    pub messages: Vec<Message>,
    /// `tools=tool_schemas or openai.omit`: an empty list is omitted, not sent as `[]`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Map<String, Value>>,
    /// Always `true`: the response is read as a stream.
    pub stream: bool,
    pub stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Any other parameter, passed through. Never holds a key the fields above own.
    #[serde(flatten)]
    pub kwargs: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct StreamOptions {
    /// The final chunk carries the request's usage.
    pub include_usage: bool,
}

/// The body keys the framework always sets; a kwarg never overrides them.
const OWNED_KEYS: &[&str] = &["model", "messages", "tools", "stream", "stream_options"];

/// `LLM.chat()`'s kwargs: the call's `extra_kwargs`, overridden by the instance's.
pub fn compose_extra(args: &ChatArgs, opts_extra: &Map<String, Value>) -> Map<String, Value> {
    let mut extra = args.extra_kwargs.clone().unwrap_or_default();
    extra.extend(opts_extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    extra
}

/// `LLMStream._run()`'s body assembly, after [`super::models::drop_unsupported_params`] has run
/// over `kwargs`. The call's explicit `parallel_tool_calls`, `tool_choice` and `response_format`
/// (the framework's per-turn decisions, such as forbidding tools) override the kwargs'.
pub fn build_body(
    model: &str,
    messages: Vec<Message>,
    tools: Vec<Map<String, Value>>,
    args: &ChatArgs,
    mut kwargs: Map<String, Value>,
) -> ChatBody {
    kwargs.retain(|key, _| !OWNED_KEYS.contains(&key.as_str()));
    let mut owned = |key: &str, given: bool| {
        if given {
            kwargs.remove(key);
        }
    };
    owned("tool_choice", args.tool_choice.is_some());
    owned("parallel_tool_calls", args.parallel_tool_calls.is_some());
    owned("response_format", args.response_format.is_some());
    let mut tool_choice = args.tool_choice.as_ref().map(WireToolChoice::from);
    // `if not self._tools:` -- an empty tool list drops tool_choice (else a 400).
    if tools.is_empty() {
        tool_choice = None;
        kwargs.remove("tool_choice");
    }
    ChatBody {
        model: model.to_owned(),
        messages,
        tools,
        stream: true,
        stream_options: StreamOptions { include_usage: true },
        tool_choice,
        parallel_tool_calls: args.parallel_tool_calls,
        response_format: args.response_format.clone(),
        kwargs,
    }
}

/// `extra_headers`: the inference headers plus, when the instance names one, the provider pin.
pub fn request_headers(mut base: Headers, provider: Option<&str>) -> Headers {
    // `if self._provider:` -- an empty provider string pins nothing.
    if let Some(provider) = provider
        && !provider.is_empty()
    {
        base.insert(HEADER_INFERENCE_PROVIDER, provider);
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
    use serde_json::json;

    fn opts(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect()
    }

    fn tool() -> Vec<Map<String, Value>> {
        vec![opts(&[("type", json!("function"))])]
    }

    fn body(args: &ChatArgs, tools: Vec<Map<String, Value>>, kwargs: Map<String, Value>) -> Value {
        serde_json::to_value(build_body("m", vec![], tools, args, kwargs)).unwrap()
    }

    // Catches: an instance `tool_choice` overriding the framework's per-call one (forbidding
    // tools while draining), or a call's `extra_kwargs` overriding the instance's.
    #[test]
    fn explicit_per_call_choices_win_and_the_instance_kwargs_beat_the_calls() {
        let args = ChatArgs {
            extra_kwargs: Some(opts(&[("temperature", json!(0.9))])),
            tool_choice: Some(ToolChoice::Forbidden),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let extra = compose_extra(
            &args,
            &opts(&[
                ("temperature", json!(0.1)),
                ("tool_choice", json!("required")),
                ("parallel_tool_calls", json!(true)),
            ]),
        );
        let body = body(&args, tool(), extra);
        assert_eq!(body["temperature"], json!(0.1));
        assert_eq!(body["tool_choice"], json!("none"));
        assert_eq!(body["parallel_tool_calls"], json!(false));
    }

    // Catches: passing the caller's tool_choice dict through unmodified.
    #[test]
    fn a_named_tool_choice_is_rebuilt_to_the_two_key_form() {
        let args = ChatArgs {
            tool_choice: Some(ToolChoice::Named("lookup".into())),
            ..Default::default()
        };
        assert_eq!(
            body(&args, tool(), Map::new())["tool_choice"],
            json!({"type": "function", "function": {"name": "lookup"}})
        );
    }

    // Catches: normalising, rejecting or dropping an instance-level `tool_choice` (it goes verbatim).
    #[test]
    fn an_instance_level_tool_choice_reaches_the_wire_verbatim() {
        for raw in [json!("Auto"), json!("required"), json!({"function": {"name": "f"}, "x": 1})] {
            let extra = compose_extra(&ChatArgs::default(), &opts(&[("tool_choice", raw.clone())]));
            let body = body(&ChatArgs::default(), tool(), extra);
            assert_eq!(body["tool_choice"], raw, "not normalised, not dropped");
        }
    }

    // Catches: R2b on `if not self._tools`.
    #[test]
    fn an_empty_tool_list_drops_both_the_tools_key_and_the_tool_choice() {
        let extra = opts(&[("tool_choice", json!("auto")), ("temperature", json!(0.2))]);
        let args = ChatArgs { tool_choice: Some(ToolChoice::Required), ..Default::default() };
        let body = body(&args, vec![], extra);
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("tools").is_none(), "`tools=[] or omit` omits");
        assert_eq!(body["temperature"], json!(0.2));
    }

    // Catches: a kwarg overriding a field the framework owns, which would send the key twice.
    #[test]
    fn a_kwarg_never_overrides_the_fields_the_framework_sets() {
        let extra = opts(&[("model", json!("other")), ("stream", json!(false))]);
        let body = body(&ChatArgs::default(), tool(), extra);
        assert_eq!(body["model"], json!("m"));
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
}
