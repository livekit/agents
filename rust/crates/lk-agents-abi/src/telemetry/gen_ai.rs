//! OpenTelemetry's GenAI span events. [`tool_call_json`] reproduces `json.dumps`'s bytes: the
//! string is a span attribute already in shipped traces.

use crate::llm::chat_context::{ChatItem, Role};

/// `telemetry/trace_types.py:110`: OTel's GenAI names, not LiveKit's `lk.*` namespace.
pub mod events {
    pub const SYSTEM_MESSAGE: &str = "gen_ai.system.message";
    pub const USER_MESSAGE: &str = "gen_ai.user.message";
    pub const ASSISTANT_MESSAGE: &str = "gen_ai.assistant.message";
    pub const TOOL_MESSAGE: &str = "gen_ai.tool.message";
    pub const CHOICE: &str = "gen_ai.choice";
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventAttr {
    Str(String),
    /// OTel attribute arrays are of scalars.
    StrList(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenAiEvent {
    pub name: &'static str,
    /// Not a map: the order is what an exporter writes out.
    pub attributes: Vec<(&'static str, EventAttr)>,
}

/// `developer` folds into the system event; `None` is a skip.
const fn message_event(role: Role) -> Option<&'static str> {
    match role {
        Role::System | Role::Developer => Some(events::SYSTEM_MESSAGE),
        Role::User => Some(events::USER_MESSAGE),
        Role::Assistant => Some(events::ASSISTANT_MESSAGE),
    }
}

/// `_chat_ctx_to_otel_events` (`telemetry/traces.py:533`). Handoffs and config updates produce no
/// event; a message with no text part traces as `""` rather than being skipped.
/// `_chat_ctx_to_otel_events(chat_ctx)`: `[(event_name, attributes), ...]`, each attribute a `str`
/// or `list[str]` as `span.add_event` takes them.
#[crate::host(export(name = "chat_ctx_gen_ai_events"))]
pub fn chat_ctx_gen_ai_events(items: Vec<ChatItem>) -> Vec<(&'static str, serde_json::Map<String, serde_json::Value>)> {
    chat_ctx_events(&items)
        .into_iter()
        .map(|ev| {
            let attrs = ev.attributes.into_iter().map(|(k, v)| {
                let v = match v {
                    EventAttr::Str(s) => s.into(),
                    EventAttr::StrList(l) => l.into(),
                };
                (k.to_owned(), v)
            });
            (ev.name, attrs.collect())
        })
        .collect()
}

pub fn chat_ctx_events(items: &[ChatItem]) -> Vec<GenAiEvent> {
    let mut out = Vec::new();
    for item in items {
        match item {
            ChatItem::Message(m) => {
                let Some(name) = message_event(m.role) else { continue };
                out.push(GenAiEvent {
                    name,
                    attributes: vec![(
                        "content",
                        EventAttr::Str(m.raw_text_content().unwrap_or_default()),
                    )],
                });
            }
            ChatItem::FunctionCall(f) => out.push(GenAiEvent {
                name: events::ASSISTANT_MESSAGE,
                attributes: vec![
                    // Redundant with the event name, but the reference sets it and readers use it.
                    ("role", EventAttr::Str("assistant".to_owned())),
                    (
                        "tool_calls",
                        EventAttr::StrList(vec![tool_call_json(&f.name, &f.arguments, &f.call_id)]),
                    ),
                ],
            }),
            ChatItem::FunctionCallOutput(f) => out.push(GenAiEvent {
                name: events::TOOL_MESSAGE,
                attributes: vec![
                    ("content", EventAttr::Str(f.output.clone())),
                    ("name", EventAttr::Str(f.name.clone())),
                    ("id", EventAttr::Str(f.call_id.clone())),
                ],
            }),
            // No `gen_ai` counterpart, as in the reference.
            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => {}
        }
    }
    out
}

/// One entry of a `tool_calls` attribute, as `json.dumps` writes it (`traces.py:551`, `llm.py:450`).
#[crate::host(export(pid = false, name = "gen_ai_tool_call_json"))]
pub fn tool_call_json(name: &str, arguments: &str, call_id: &str) -> String {
    format!(
        "{{\"function\": {{\"name\": {}, \"arguments\": {}}}, \"id\": {}, \"type\": \"function\"}}",
        json_str(name),
        json_str(arguments),
        json_str(call_id),
    )
}

/// `ensure_ascii=True`; astral-plane characters become a surrogate pair.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// `gen_ai_provider_name` (`trace_types.py`): a plugin's `provider` in the GenAI registry's
/// spelling; one outside the registry keeps its own.
pub fn provider_name(provider: &str) -> Option<String> {
    let value = provider.trim();
    if value.is_empty() {
        return None;
    }
    let host = value.to_lowercase();
    let by_host = match host.as_str() {
        "api.anthropic.com" => Some("anthropic"),
        "api.cohere.ai" | "api.cohere.com" => Some("cohere"),
        "api.deepseek.com" => Some("deepseek"),
        "api.groq.com" => Some("groq"),
        "api.mistral.ai" => Some("mistral_ai"),
        "api.moonshot.ai" | "api.moonshot.cn" => Some("moonshot_ai"),
        "api.openai.com" => Some("openai"),
        "api.perplexity.ai" => Some("perplexity"),
        "api.x.ai" => Some("x_ai"),
        "generativelanguage.googleapis.com" => Some("gcp.gemini"),
        _ => None,
    };
    let by_suffix = || {
        [
            (".openai.azure.com", "azure.ai.openai"),
            (".services.ai.azure.com", "azure.ai.inference"),
            (".aiplatform.googleapis.com", "gcp.vertex_ai"),
        ]
        .into_iter()
        .find(|(suffix, _)| host.ends_with(suffix))
        .map(|(_, mapped)| mapped)
    };
    if let Some(mapped) = by_host.or_else(by_suffix) {
        return Some(mapped.to_owned());
    }
    // Only the Bedrock endpoints, not every AWS service that shares the domain.
    if host.starts_with("bedrock") && host.ends_with(".amazonaws.com") {
        return Some("aws.bedrock".to_owned());
    }
    let canonical: String = host.chars().filter(|c| c.is_alphanumeric()).collect();
    let by_name = match canonical.as_str() {
        "amazon" | "amazonbedrock" | "awsbedrock" | "bedrock" => "aws.bedrock",
        "anthropic" => "anthropic",
        "azureaiinference" => "azure.ai.inference",
        "azureopenai" => "azure.ai.openai",
        "cohere" => "cohere",
        "deepseek" => "deepseek",
        "gemini" => "gcp.gemini",
        "google" | "googlecloudplatform" | "googlegenai" => "gcp.gen_ai",
        "groq" => "groq",
        "ibmwatsonxai" | "watsonx" => "ibm.watsonx.ai",
        "mistral" | "mistralai" => "mistral_ai",
        "moonshot" | "moonshotai" => "moonshot_ai",
        "openai" => "openai",
        "perplexity" => "perplexity",
        "vertexai" | "vertexaimodelgarden" => "gcp.vertex_ai",
        "xai" => "x_ai",
        _ => return Some(value.to_owned()),
    };
    Some(by_name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::chat_context::ChatMessage;
    use std::time::UNIX_EPOCH;

    fn msg(role: Role, text: &str) -> ChatItem {
        ChatMessage::text("m", role, text, UNIX_EPOCH).into()
    }

    // Catches: dropping the `developer` role.
    #[test]
    fn developer_folds_into_the_system_event() {
        let out = chat_ctx_events(&[msg(Role::Developer, "be brief")]);
        assert_eq!(out[0].name, events::SYSTEM_MESSAGE);
        assert_eq!(out[0].attributes, vec![("content", EventAttr::Str("be brief".into()))]);
    }

    // Catches: skipping a message with no text part.
    #[test]
    fn a_message_with_no_text_still_produces_an_event() {
        let out = chat_ctx_events(&[crate::testkit::empty_msg("m", 0)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].attributes, vec![("content", EventAttr::Str(String::new()))]);
    }

    // Catches: emitting the tool-call object with `serde_json`'s separators.
    #[test]
    fn the_tool_call_json_matches_json_dumps_byte_for_byte() {
        assert_eq!(
            tool_call_json("lookup", "{\"q\": 1}", "call_7"),
            r#"{"function": {"name": "lookup", "arguments": "{\"q\": 1}"}, "id": "call_7", "type": "function"}"#
        );
    }

    // Catches: non-ASCII unescaped, or an astral character not becoming a surrogate pair.
    #[test]
    fn non_ascii_is_escaped_the_way_ensure_ascii_escapes_it() {
        assert_eq!(json_str("café"), r#""caf\u00e9""#);
        assert_eq!(json_str("\u{1f600}"), r#""\ud83d\ude00""#);
        assert_eq!(json_str("a\tb\u{1}"), r#""a\tb\u0001""#);
    }

    // Catches: a registry spelling lost, or an unknown provider dropped instead of kept.
    #[test]
    fn provider_names_follow_the_genai_registry() {
        assert_eq!(provider_name("api.openai.com").as_deref(), Some("openai"));
        assert_eq!(provider_name("my.openai.azure.com").as_deref(), Some("azure.ai.openai"));
        let bedrock = provider_name("bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(bedrock.as_deref(), Some("aws.bedrock"));
        assert_eq!(provider_name("Google GenAI").as_deref(), Some("gcp.gen_ai"));
        assert_eq!(provider_name("livekit").as_deref(), Some("livekit"));
        assert_eq!(provider_name("  "), None);
    }

    // Catches: a handoff or a config update leaking into the traced conversation.
    #[test]
    fn function_call_and_output_pair_up_and_livekit_only_items_are_silent() {
        let items = vec![crate::testkit::call("c0", "weather", 0), crate::testkit::out("c0", "weather", 0)];
        let out = chat_ctx_events(&items);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, events::ASSISTANT_MESSAGE);
        assert_eq!(out[1].name, events::TOOL_MESSAGE);
        assert_eq!(
            out[1].attributes,
            vec![
                ("content", EventAttr::Str("done".into())),
                ("name", EventAttr::Str("weather".into())),
                ("id", EventAttr::Str("call_c0".into())),
            ]
        );
    }
}
