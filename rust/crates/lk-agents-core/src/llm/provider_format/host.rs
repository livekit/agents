//! `to_provider_format` and `to_fnc_ctx` as the host calls them. Images arrive pre-serialised, so
//! each is encoded once; dropped items are logged with the reference's warning, not raised.

use std::collections::BTreeMap;

use lk_agents_abi::errors::Invalid;
use lk_agents_abi::llm::provider_format::ProviderFormat;
use lk_agents_abi::llm::provider_format::host::ConvertedItem;
use lk_agents_abi::llm::provider_format::vocabulary::GeminiToolBehavior;
use serde::Serialize;
use serde_json::Value;

use super::{
    DropReason, Formatted, PreResolved, SerializedImage, ToolSchema, anthropic, aws, google,
    mistralai, openai,
};
use lk_agents_abi::{ChatContext, ChatItem};

/// `ChatContext.to_provider_format(format, ...)` -> `(messages, data)`. `data` is `None` for OpenAI
/// and a dict for the dialects that hoist something out of band.
pub fn to_provider_format(
    format: ProviderFormat,
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
        ProviderFormat::OpenAi => finish(openai::to_chat_ctx(&ctx, &images).map_err(e)?),
        ProviderFormat::OpenAiResponses => {
            finish(openai::to_responses_chat_ctx(&ctx, &images).map_err(e)?)
        }
        ProviderFormat::Anthropic => {
            let opts =
                anthropic::Options { inject_dummy_user_message, inject_trailing_user_message };
            finish(anthropic::to_chat_ctx(&ctx, &images, opts).map_err(e)?)
        }
        ProviderFormat::Aws => {
            finish(aws::to_chat_ctx(&ctx, &images, inject_dummy_user_message).map_err(e)?)
        }
        ProviderFormat::Google => {
            let f = google::to_chat_ctx(
                &ctx,
                &images,
                inject_dummy_user_message,
                thought_signatures.as_ref(),
            );
            finish(f.map_err(e)?)
        }
        ProviderFormat::MistralAi => {
            let f = mistralai::to_conversations_ctx(&ctx, &images, inject_dummy_user_message);
            finish(f.map_err(e)?)
        }
    };
    formatted.map_err(|error| Invalid(error.to_string()))
}

/// `convert_mid_conversation_instructions(chat_ctx, role=, template=)`: the input item each output
/// item comes from, and the message replacing it when it was rewritten, so the host keeps its own
/// objects for the rest.
pub fn convert_mid_conversation_instructions(
    items: Vec<ChatItem>,
    role: lk_agents_abi::Role,
    template: &str,
) -> Vec<ConvertedItem> {
    let ctx = ChatContext::new(items);
    let index =
        |found: &dyn Fn(&ChatItem) -> bool| ctx.items.iter().position(found).unwrap_or_default();
    super::convert_mid_conversation_instructions(&ctx, role, template)
        .into_iter()
        .map(|item| match item {
            std::borrow::Cow::Borrowed(kept) => {
                ConvertedItem { index: index(&|i| std::ptr::eq(i, kept)), rewritten: None }
            }
            std::borrow::Cow::Owned(new) => {
                ConvertedItem { index: index(&|i| i.id() == new.id()), rewritten: Some(new) }
            }
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

/// `to_fnc_ctx(tools)` for a dialect. `tool_behavior` and `use_parameters_json_schema` are
/// Gemini-only (Gemini Live rejects `parameters_json_schema`, googleapis/python-genai#1147).
pub fn to_fnc_ctx(
    format: ProviderFormat,
    specs: Vec<ToolSchema>,
    tool_behavior: Option<GeminiToolBehavior>,
    use_parameters_json_schema: bool,
) -> Result<Vec<Value>, Invalid> {
    let e = |e: super::FormatError| Invalid(e.to_string());
    let formatted = match format {
        ProviderFormat::OpenAi => serialize_list(&openai::to_fnc_ctx(&specs).map_err(e)?),
        // Not a flag: the Responses API spreads raw schemas flat and includes `ProviderTool`s.
        ProviderFormat::OpenAiResponses => {
            serialize_list(&openai::to_responses_fnc_ctx(&specs).map_err(e)?)
        }
        ProviderFormat::Anthropic => serialize_list(&anthropic::to_fnc_ctx(&specs).map_err(e)?),
        ProviderFormat::Aws => serialize_list(&aws::to_fnc_ctx(&specs).map_err(e)?),
        ProviderFormat::Google => serialize_list(
            &google::to_fnc_ctx(&specs, tool_behavior, use_parameters_json_schema).map_err(e)?,
        ),
        ProviderFormat::MistralAi => {
            return Err(Invalid(
                "\"mistralai\" has no tool-schema rewrapper: Mistral takes the OpenAI shape verbatim"
                    .into(),
            ));
        }
    };
    formatted.map_err(|error| Invalid(error.to_string()))
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
        let indices: Vec<usize> = plan.iter().map(|c| c.index).collect();
        assert_eq!(indices, [0, 2, 3]);
        assert!(plan[0].rewritten.is_none() && plan[1].rewritten.is_none());
        let rewritten = plan[2].rewritten.as_ref().and_then(ChatItem::as_message).unwrap();
        assert_eq!(
            (rewritten.role, rewritten.raw_text_content().as_deref()),
            (Role::User, Some("<mid>"))
        );
    }
}
