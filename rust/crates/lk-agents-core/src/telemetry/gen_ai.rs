//! OpenTelemetry GenAI semantic conventions the framework's spans follow: provider names, and the
//! message capture shapes (`gen_ai.system_instructions`, `gen_ai.input.messages`,
//! `gen_ai.output.messages`, `gen_ai.tool.definitions`).

use lk_agents_abi::llm::chat_context::{ChatContent, ImageSource, Role};
use lk_agents_abi::{ChatItem, ToolSpec};
use serde_json::{Value, json};

use crate::request::INSTRUCTIONS_MESSAGE_ID;

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

/// How a response ended: `error` when interrupted, `tool_call` with calls, else `stop`.
pub fn finish_reason(calls: bool, interrupted: bool) -> &'static str {
    match (interrupted, calls) {
        (true, _) => "error",
        (false, true) => "tool_call",
        (false, false) => "stop",
    }
}

fn is_system(item: &ChatItem) -> bool {
    matches!(item, ChatItem::Message(m) if matches!(m.role, Role::System | Role::Developer))
}

/// The agent's instructions message, else a leading system message, and the rest in order.
fn split(items: &[ChatItem]) -> (Option<&ChatItem>, Vec<&ChatItem>) {
    let at = items
        .iter()
        .position(|item| item.id() == INSTRUCTIONS_MESSAGE_ID && is_system(item))
        .or_else(|| items.first().filter(|item| is_system(item)).map(|_| 0));
    let rest = items.iter().enumerate().filter(|(i, _)| Some(*i) != at).map(|(_, it)| it);
    (at.map(|i| &items[i]), rest.collect())
}

fn text(content: &str) -> Value {
    json!({"type": "text", "content": content})
}

/// A string as JSON when it parses, else the string.
fn maybe_json(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn tool_call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"type": "tool_call", "id": id, "name": name, "arguments": maybe_json(arguments)})
}

/// Message parts; media payloads are never recorded, only their kind and URL or transcript.
fn parts(item: &ChatItem) -> Vec<Value> {
    match item {
        ChatItem::Message(m) => m
            .content
            .iter()
            .map(|c| match c {
                ChatContent::Text(t) => text(t),
                ChatContent::Image(image) => {
                    let uri = match &image.image {
                        ImageSource::Url(url) if !url.starts_with("data:") => url.clone(),
                        _ => String::new(),
                    };
                    json!({"type": "uri", "modality": "image", "mime_type": image.mime_type, "uri": uri})
                }
                ChatContent::Audio(audio) => {
                    let mut part = json!({"type": "blob", "modality": "audio", "content": ""});
                    if let Some(transcript) = audio.transcript.as_ref().filter(|t| !t.is_empty()) {
                        part["transcript"] = Value::String(transcript.clone());
                    }
                    part
                }
            })
            .collect(),
        ChatItem::FunctionCall(f) => vec![tool_call(&f.call_id, &f.name, &f.arguments)],
        ChatItem::FunctionCallOutput(o) => {
            let mut part =
                json!({"type": "tool_call_response", "id": o.call_id, "response": maybe_json(&o.output)});
            if !o.name.is_empty() {
                part["name"] = Value::String(o.name.clone());
            }
            vec![part]
        }
        ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => Vec::new(),
    }
}

fn role(item: &ChatItem) -> Option<&'static str> {
    Some(match item {
        ChatItem::Message(m) => match m.role {
            Role::System | Role::Developer => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        ChatItem::FunctionCall(_) => "assistant",
        ChatItem::FunctionCallOutput(_) => "tool",
        ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => return None,
    })
}

/// `gen_ai.system_instructions`: the agent's instructions, reported apart from the history.
pub fn system_instructions(items: &[ChatItem]) -> Vec<Value> {
    match split(items).0 {
        Some(ChatItem::Message(m)) => m.raw_text_content().map(|t| text(&t)).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// `gen_ai.input.messages`: the history as sent; consecutive tool calls of one assistant turn
/// are one message, items with no conversational part are skipped.
pub fn input_messages(items: &[ChatItem]) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    let mut last_role = None;
    for item in split(items).1 {
        let (Some(role), parts) = (role(item), parts(item)) else { continue };
        if parts.is_empty() {
            continue;
        }
        let merged = last_role == Some("assistant")
            && role == "assistant"
            && matches!(item, ChatItem::FunctionCall(_));
        match messages.last_mut() {
            Some(last) if merged => {
                if let Some(list) = last["parts"].as_array_mut() {
                    list.extend(parts);
                }
            }
            _ => {
                messages.push(json!({"role": role, "parts": parts}));
                last_role = Some(role);
            }
        }
    }
    messages
}

/// `gen_ai.output.messages`: the response text and calls as one assistant message.
pub fn output_messages(
    text_out: &str,
    calls: &[(String, String, String)],
    finish_reason: &str,
) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    if !text_out.is_empty() {
        parts.push(text(text_out));
    }
    parts.extend(calls.iter().map(|(id, name, arguments)| tool_call(id, name, arguments)));
    if parts.is_empty() {
        return Vec::new();
    }
    vec![json!({"role": "assistant", "parts": parts, "finish_reason": finish_reason})]
}

/// `lk.pii.chat_ctx`: the context as `ChatContext.to_dict` writes it, without media, timestamps
/// or metrics.
pub fn chat_ctx(items: &[ChatItem]) -> Value {
    let opts = lk_agents_abi::ToDictOptions {
        exclude_metrics: true,
        ..lk_agents_abi::ToDictOptions::default()
    };
    crate::llm::chat_dict::to_dict(items, &opts, &lk_agents_abi::ToDictHost::default())
        .map_or(Value::Null, Value::Object)
}

/// `gen_ai.tool.definitions`: name and description; schemas are left out for their size.
pub fn tool_definitions(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| match &tool.provider_tool {
            Some(_) => json!({"type": tool.name, "name": tool.name}),
            None if tool.description.is_empty() => json!({"type": "function", "name": tool.name}),
            None => {
                json!({"type": "function", "name": tool.name, "description": tool.description})
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{ChatMessage, FunctionCall};
    use std::time::SystemTime;

    // Catches: the instructions left in the history, consecutive calls split into messages, or
    // a handoff reported as a message.
    #[test]
    fn messages_follow_the_genai_shapes() {
        let at = SystemTime::UNIX_EPOCH;
        let call = |id: &str| {
            ChatItem::FunctionCall(FunctionCall {
                id: id.into(),
                call_id: id.into(),
                arguments: "{\"a\":1}".into(),
                name: "f".into(),
                created_at: at,
                extra: Default::default(),
                group_id: None,
            })
        };
        let items = vec![
            ChatMessage::text(INSTRUCTIONS_MESSAGE_ID, Role::System, "be brief", at).into(),
            ChatMessage::text("u", Role::User, "hi", at).into(),
            call("c1"),
            call("c2"),
        ];
        assert_eq!(
            system_instructions(&items),
            vec![json!({"type": "text", "content": "be brief"})]
        );
        let input = input_messages(&items);
        assert_eq!(input.len(), 2, "{input:?}");
        assert_eq!(input[1]["parts"].as_array().map(Vec::len), Some(2));
        assert_eq!(input[1]["parts"][0]["arguments"], json!({"a": 1}));
        let out = output_messages("ok", &[], finish_reason(false, false));
        assert_eq!(out[0]["finish_reason"], "stop");
        assert!(output_messages("", &[], "stop").is_empty());
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
}
