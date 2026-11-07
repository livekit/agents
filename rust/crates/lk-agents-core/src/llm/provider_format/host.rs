//! `to_provider_format` and `to_fnc_ctx` as the host calls them. Images arrive pre-serialised, so
//! each is encoded once; dropped items are logged with the reference's warning, not raised.

use std::collections::BTreeMap;

use lk_agents_abi::errors::Invalid;
use serde::Serialize;
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

use super::{
    DropReason, Formatted, PreResolved, SerializedImage, ToolSchema, anthropic, aws, google,
    mistralai, openai,
};
use lk_agents_abi::{ChatContext, ChatItem};

/// `ChatContext.to_provider_format(format, ...)` -> `(messages, data)`. `data` is `None` for OpenAI
/// and a dict for the dialects that hoist something out of band.
pub fn to_provider_format(
    format: &str,
    items: Vec<ChatItem>,
    images: Option<BTreeMap<String, SerializedImage>>,
    inject_dummy_user_message: bool,
    inject_trailing_user_message: bool,
    thought_signatures: Option<BTreeMap<String, Vec<u8>>>,
) -> Result<(Vec<Value>, Value), Invalid> {
    let ctx = ChatContext::new(items);
    let images = PreResolved::from_pairs(images.unwrap_or_default());
    let e = |e: super::FormatError| Invalid(e.to_string());
    let formatted = match format {
        "openai" => finish(openai::to_chat_ctx(&ctx, &images).map_err(e)?),
        "openai.responses" => finish(openai::to_responses_chat_ctx(&ctx, &images).map_err(e)?),
        "anthropic" => {
            let opts =
                anthropic::Options { inject_dummy_user_message, inject_trailing_user_message };
            finish(anthropic::to_chat_ctx(&ctx, &images, opts).map_err(e)?)
        }
        "aws" => finish(aws::to_chat_ctx(&ctx, &images, inject_dummy_user_message).map_err(e)?),
        "google" => {
            let f = google::to_chat_ctx(
                &ctx,
                &images,
                inject_dummy_user_message,
                thought_signatures.as_ref(),
            );
            finish(f.map_err(e)?)
        }
        "mistralai" => {
            let f = mistralai::to_conversations_ctx(&ctx, &images, inject_dummy_user_message);
            finish(f.map_err(e)?)
        }
        _ => {
            return Err(Invalid(format!(
                "unknown provider format {format:?}: expected one of 'openai', 'openai.responses', \
                 'anthropic', 'aws', 'google', 'mistralai'"
            )));
        }
    };
    formatted.map_err(|error| Invalid(error.to_string()))
}

/// `convert_mid_conversation_instructions(chat_ctx, role=, template=)` as `(index, rewritten)`: the
/// input item each output item comes from, and the message replacing it when it was rewritten, so
/// the host keeps its own objects for the rest.
pub fn convert_mid_conversation_instructions(
    items: Vec<ChatItem>,
    role: lk_agents_abi::Role,
    template: &str,
) -> Vec<(usize, Option<ChatItem>)> {
    let ctx = ChatContext::new(items);
    let index =
        |found: &dyn Fn(&ChatItem) -> bool| ctx.items.iter().position(found).unwrap_or_default();
    super::convert_mid_conversation_instructions(&ctx, role, template)
        .into_iter()
        .map(|item| match item {
            std::borrow::Cow::Borrowed(kept) => (index(&|i| std::ptr::eq(i, kept)), None),
            std::borrow::Cow::Owned(new) => (index(&|i| i.id() == new.id()), Some(new)),
        })
        .collect()
}

fn finish<M: Serialize, D: Serialize>(
    f: Formatted<M, D>,
) -> serde_json::Result<(Vec<Value>, Value)> {
    for d in &f.dropped {
        let message = match d.reason {
            DropReason::OutputWithoutCall => {
                "function output missing the corresponding function call, ignoring"
            }
            DropReason::CallWithoutOutput => {
                "function call missing the corresponding function output, ignoring"
            }
        };
        log::warn!(target: lk_agents_abi::LOG_TARGET, call_id = d.call_id.as_str(), function_name = d.tool_name.as_str(); "{message}");
    }
    let data = serde_json::to_value(f.data)?;
    Ok((serialize_list(&f.messages)?, data))
}

fn serialize_list<T: Serialize>(items: &[T]) -> serde_json::Result<Vec<Value>> {
    items.iter().map(serde_json::to_value).collect()
}

/// `to_fnc_ctx(tools)` for a dialect; each tool is the host's `[kind, schema, name]` (`name` only for
/// `raw`). `tool_behavior` and `use_parameters_json_schema` are Gemini-only (Gemini Live rejects
/// `parameters_json_schema`, googleapis/python-genai#1147).
pub fn to_fnc_ctx(
    format: &str,
    tools: Vec<Value>,
    tool_behavior: Option<&str>,
    use_parameters_json_schema: bool,
) -> Result<Vec<Value>, Invalid> {
    let specs = tools.into_iter().map(tool_schema).collect::<Result<Vec<_>, _>>()?;
    let e = |e: super::FormatError| Invalid(e.to_string());
    // Mistral has no rewrapper, and an unknown format gets the same message.
    let formatted = match format {
        "openai" => serialize_list(&openai::to_fnc_ctx(&specs).map_err(e)?),
        // Not a flag: the Responses API spreads raw schemas flat and includes `ProviderTool`s.
        "openai.responses" => serialize_list(&openai::to_responses_fnc_ctx(&specs).map_err(e)?),
        "anthropic" => serialize_list(&anthropic::to_fnc_ctx(&specs).map_err(e)?),
        "aws" => serialize_list(&aws::to_fnc_ctx(&specs).map_err(e)?),
        "google" => {
            // `UNSPECIFIED` is spelled by omitting `tool_behavior`.
            if let Some(b) = tool_behavior.filter(|b| {
                lk_agents_abi::llm::provider_format::vocabulary::GeminiToolBehavior::from_wire(b)
                    .is_none()
            }) {
                return Err(Invalid(format!(
                    "unknown tool_behavior {b:?}: expected 'BLOCKING' or 'NON_BLOCKING'"
                )));
            }
            serialize_list(
                &google::to_fnc_ctx(&specs, tool_behavior, use_parameters_json_schema)
                    .map_err(e)?,
            )
        }
        _ => {
            return Err(Invalid(format!(
                "{format:?} has no tool-schema rewrapper: Mistral takes the OpenAI shape verbatim"
            )));
        }
    };
    formatted.map_err(|error| Invalid(error.to_string()))
}

fn tool_schema(tool: Value) -> Result<ToolSchema, Invalid> {
    let Value::Array(mut parts) = tool else {
        return Err(Invalid(format!("a tool schema is [kind, schema, name], got {tool}")));
    };
    parts.resize(3, Value::Null);
    let (name, payload) = (parts.pop().unwrap_or_default(), parts.pop().unwrap_or_default());
    let kind = parts.pop().and_then(|k| k.as_str().map(str::to_owned)).unwrap_or_default();
    Ok(match kind.as_str() {
        "raw" => ToolSchema::Raw {
            name: name
                .as_str()
                .ok_or_else(|| Invalid("a raw tool schema carries its name".into()))?
                .to_owned(),
            raw_schema: payload,
        },
        "legacy" => ToolSchema::Legacy(payload),
        "strict" => ToolSchema::Strict(payload),
        "provider" => ToolSchema::Provider(payload),
        _ => {
            return Err(Invalid(format!(
                "unknown tool schema kind {kind:?}: expected 'raw', 'legacy', 'strict' or 'provider'"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a dropped instruction shifting the indices the host maps back to its own items.
    #[test]
    fn converted_items_point_back_at_the_items_they_come_from() {
        use super::super::testkit::msg;
        use lk_agents_abi::Role;
        let items = vec![
            msg("s0", Role::System, &["p"]),
            msg("s1", Role::System, &[""]),
            msg("u", Role::User, &["hi"]),
            msg("s2", Role::Developer, &["mid"]),
        ];
        let plan = convert_mid_conversation_instructions(items, Role::User, "<{content}>");
        let indices: Vec<usize> = plan.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, [0, 2, 3]);
        assert!(plan[0].1.is_none() && plan[1].1.is_none());
        let rewritten = plan[2].1.as_ref().and_then(ChatItem::as_message).unwrap();
        assert_eq!(
            (rewritten.role, rewritten.raw_text_content().as_deref()),
            (Role::User, Some("<mid>"))
        );
    }

    // Catches: an unknown tool kind or `tool_behavior` defaulting instead of refusing.
    #[test]
    fn unknown_words_are_refused() {
        assert!(to_fnc_ctx("openai", vec![json!(["mcp", {}])], None, false).is_err());
        for refused in ["UNSPECIFIED", "blocking"] {
            assert!(to_fnc_ctx("google", vec![], Some(refused), false).is_err());
        }
        assert!(to_fnc_ctx("google", vec![], Some("NON_BLOCKING"), false).is_ok());
    }
}
