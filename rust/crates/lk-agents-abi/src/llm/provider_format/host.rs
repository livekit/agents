//! `to_provider_format` and `to_fnc_ctx` as the host calls them. Images arrive pre-serialised, so
//! each is encoded once; dropped items are logged with the reference's warning, not raised.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::{
    DropReason, Formatted, PreResolved, SerializedImage, ToolSchema, anthropic, aws, google, mistralai, openai,
};
use crate::{ChatContext, ChatItem};

/// `ChatContext.to_provider_format(format, ...)` -> `(messages, data)`. `data` is `None` for OpenAI
/// and a dict for the dialects that hoist something out of band.
#[crate::host(export(
    kw_only = "images",
    defaults(inject_dummy_user_message = true, inject_trailing_user_message = false)
))]
pub fn to_provider_format(
    format: &str,
    items: Vec<ChatItem>,
    images: Option<HashMap<String, SerializedImage>>,
    inject_dummy_user_message: bool,
    inject_trailing_user_message: bool,
    thought_signatures: Option<HashMap<String, Vec<u8>>>,
) -> Result<(Vec<Value>, Value), String> {
    let ctx = ChatContext::new(items);
    let images = PreResolved::from_pairs(images.unwrap_or_default());
    let e = |e: super::FormatError| e.to_string();
    let hoisted = |system_messages: &[String]| json!({ "system_messages": system_messages });
    Ok(match format {
        "openai" => finish(openai::to_chat_ctx(&ctx, &images).map_err(e)?, |_| Value::Null),
        "openai.responses" => finish(openai::to_responses_chat_ctx(&ctx, &images).map_err(e)?, |_| Value::Null),
        "anthropic" => {
            let opts = anthropic::Options { inject_dummy_user_message, inject_trailing_user_message };
            finish(anthropic::to_chat_ctx(&ctx, &images, opts).map_err(e)?, |d| hoisted(&d.system_messages))
        }
        "aws" => {
            finish(aws::to_chat_ctx(&ctx, &images, inject_dummy_user_message).map_err(e)?, |d| hoisted(&d.system_messages))
        }
        "google" => {
            let f = google::to_chat_ctx(&ctx, &images, inject_dummy_user_message, thought_signatures.as_ref());
            finish(f.map_err(e)?, |d| hoisted(&d.system_messages))
        }
        "mistralai" => {
            let f = mistralai::to_conversations_ctx(&ctx, &images, inject_dummy_user_message);
            finish(f.map_err(e)?, |d| json!({ "instructions": d.instructions }))
        }
        _ => {
            return Err(format!(
                "unknown provider format {format:?}: expected one of 'openai', 'openai.responses', \
                 'anthropic', 'aws', 'google', 'mistralai'"
            ));
        }
    })
}

/// `convert_mid_conversation_instructions(chat_ctx, role=, template=)` as `(index, rewritten)`: the
/// input item each output item comes from, and the message replacing it when it was rewritten, so
/// the host keeps its own objects for the rest.
#[crate::host(export(kw_only = "role"))]
pub fn convert_mid_conversation_instructions(
    items: Vec<ChatItem>,
    role: crate::Role,
    template: &str,
) -> Vec<(usize, Option<ChatItem>)> {
    let ctx = ChatContext::new(items);
    let index = |found: &dyn Fn(&ChatItem) -> bool| ctx.items.iter().position(found).unwrap_or_default();
    super::convert_mid_conversation_instructions(&ctx, role, template)
        .into_iter()
        .map(|item| match item {
            std::borrow::Cow::Borrowed(kept) => (index(&|i| std::ptr::eq(i, kept)), None),
            std::borrow::Cow::Owned(new) => (index(&|i| i.id() == new.id()), Some(new)),
        })
        .collect()
}

fn finish<D>(f: Formatted<D>, data: impl FnOnce(&D) -> Value) -> (Vec<Value>, Value) {
    for d in &f.dropped {
        let message = match d.reason {
            DropReason::OutputWithoutCall => "function output missing the corresponding function call, ignoring",
            DropReason::CallWithoutOutput => "function call missing the corresponding function output, ignoring",
        };
        log::warn!(target: "livekit.agents", call_id = d.call_id.as_str(), function_name = d.tool_name.as_str(); "{message}");
    }
    let data = data(&f.data);
    (f.messages, data)
}

/// `to_fnc_ctx(tools)` for a dialect; each tool is the host's `[kind, schema, name]` (`name` only for
/// `raw`). `tool_behavior` and `use_parameters_json_schema` are Gemini-only (Gemini Live rejects
/// `parameters_json_schema`, googleapis/python-genai#1147).
#[crate::host(export(kw_only = "tool_behavior", defaults(use_parameters_json_schema = false)))]
pub fn to_fnc_ctx(
    format: &str,
    tools: Vec<Value>,
    tool_behavior: Option<&str>,
    use_parameters_json_schema: bool,
) -> Result<Vec<Value>, String> {
    let specs = tools.into_iter().map(tool_schema).collect::<Result<Vec<_>, _>>()?;
    let e = |e: super::FormatError| e.to_string();
    // Mistral has no rewrapper, and an unknown format gets the same message.
    Ok(match format {
        "openai" => openai::to_fnc_ctx(&specs),
        // Not a flag: the Responses API spreads raw schemas flat and includes `ProviderTool`s.
        "openai.responses" => openai::to_responses_fnc_ctx(&specs),
        "anthropic" => anthropic::to_fnc_ctx(&specs).map_err(e)?,
        "aws" => aws::to_fnc_ctx(&specs).map_err(e)?,
        "google" => {
            // `UNSPECIFIED` is spelled by omitting `tool_behavior`.
            if let Some(b) = tool_behavior.filter(|b| !matches!(*b, "BLOCKING" | "NON_BLOCKING")) {
                return Err(format!("unknown tool_behavior {b:?}: expected 'BLOCKING' or 'NON_BLOCKING'"));
            }
            google::to_fnc_ctx(&specs, tool_behavior, use_parameters_json_schema).map_err(e)?
        }
        _ => {
            return Err(format!("{format:?} has no tool-schema rewrapper: Mistral takes the OpenAI shape verbatim"));
        }
    })
}

fn tool_schema(tool: Value) -> Result<ToolSchema, String> {
    let Value::Array(mut parts) = tool else { return Err(format!("a tool schema is [kind, schema, name], got {tool}")) };
    parts.resize(3, Value::Null);
    let (name, payload) = (parts.pop().unwrap_or_default(), parts.pop().unwrap_or_default());
    let kind = parts.pop().and_then(|k| k.as_str().map(str::to_owned)).unwrap_or_default();
    Ok(match kind.as_str() {
        "raw" => ToolSchema::Raw {
            name: name.as_str().ok_or("a raw tool schema carries its name")?.to_owned(),
            raw_schema: payload,
        },
        "legacy" => ToolSchema::Legacy(payload),
        "strict" => ToolSchema::Strict(payload),
        "provider" => ToolSchema::Provider(payload),
        _ => {
            return Err(format!(
                "unknown tool schema kind {kind:?}: expected 'raw', 'legacy', 'strict' or 'provider'"
            ));
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
        use crate::Role;
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
        assert_eq!((rewritten.role, rewritten.raw_text_content().as_deref()), (Role::User, Some("<mid>")));
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
