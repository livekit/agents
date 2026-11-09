use super::group::{
    DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE, convert_mid_conversation_instructions, group_tool_calls,
};
use super::openai::{image_url, split_content};
use super::wire::Content;
use super::{FormatError, Formatted, ImageResolver, SerializedImage};
use lk_agents_abi::llm::provider_format::vocabulary::{
    AssistantRole, MistralEntryObject as EntryObject, UserRole,
};
use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Role};
use serde::Serialize;
use std::borrow::Cow;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MistralFormatData {
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entry {
    pub object: EntryObject,
    #[serde(flatten)]
    pub data: EntryData,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum EntryData {
    #[serde(rename = "message.input")]
    MessageInput { role: UserRole, content: Content<Chunk> },
    #[serde(rename = "message.output")]
    MessageOutput { role: AssistantRole, content: Content<Chunk> },
    #[serde(rename = "function.call")]
    FunctionCall { tool_call_id: String, name: String, arguments: String },
    #[serde(rename = "function.result")]
    FunctionResult { tool_call_id: String, result: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Chunk {
    Text { text: String },
    ImageUrl { image_url: String },
}

pub fn to_conversations_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    inject_dummy_user_message: bool,
) -> Result<Formatted<Entry, MistralFormatData>, FormatError> {
    let converted = convert_mid_conversation_instructions(
        chat_ctx,
        Role::User,
        DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
    );
    let grouping = group_tool_calls(converted.iter().map(Cow::as_ref))?;
    let mut entries = Vec::new();
    let mut instructions = None;
    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }
        if let Some(message) = group.message {
            if matches!(message.role, Role::System | Role::Developer) {
                let text_parts: Vec<&str> = message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ChatContent::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                instructions = (!text_parts.is_empty()).then(|| text_parts.join("\n"));
                continue;
            }
            if let Some(data) = to_entry(message, images)? {
                entries.push(Entry { object: EntryObject::Entry, data });
            }
        }
        entries.extend(group.tool_calls.iter().map(|call| Entry {
            object: EntryObject::Entry,
            data: EntryData::FunctionCall {
                tool_call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        }));
        entries.extend(group.tool_outputs.iter().map(|output| Entry {
            object: EntryObject::Entry,
            data: EntryData::FunctionResult {
                tool_call_id: output.call_id.clone(),
                result: output.output.clone(),
            },
        }));
    }
    if inject_dummy_user_message
        && entries.last().is_some_and(|entry| matches!(entry.data, EntryData::MessageOutput { .. }))
    {
        entries.push(Entry {
            object: EntryObject::Entry,
            data: EntryData::MessageInput {
                role: UserRole::User,
                content: Content::Text("(empty)".into()),
            },
        });
    }
    Ok(Formatted {
        messages: entries,
        data: MistralFormatData { instructions },
        dropped: grouping.dropped,
    })
}

fn to_entry(
    message: &ChatMessage,
    images: &dyn ImageResolver,
) -> Result<Option<EntryData>, FormatError> {
    if matches!(message.role, Role::System | Role::Developer) {
        return Ok(None);
    }
    let content = split_content(message, images, to_image_content, |text| Chunk::Text { text })?;
    Ok(Some(if message.role == Role::User {
        EntryData::MessageInput { role: UserRole::User, content }
    } else {
        EntryData::MessageOutput { role: AssistantRole::Assistant, content }
    }))
}

fn to_image_content(image: &SerializedImage) -> Chunk {
    Chunk::ImageUrl { image_url: image_url(image) }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::ChatItem;

    use super::*;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, msg, msg_with, no_images, out, text_content, text_of, url_images,
    };

    fn run(
        items: Vec<ChatItem>,
        images: &dyn ImageResolver,
    ) -> Formatted<Entry, MistralFormatData> {
        to_conversations_ctx(&ctx(items), images, false).unwrap()
    }

    // Catches: a later instruction replacing the preamble instead of keeping its place.
    #[test]
    fn the_preamble_is_the_instructions_and_later_ones_stay_in_place() {
        let items = vec![
            msg("s0", Role::System, &["first"]),
            msg("u", Role::User, &["hi"]),
            msg("s1", Role::Developer, &["second"]),
        ];
        let f = run(items, &no_images());
        assert_eq!(f.data.instructions.as_deref(), Some("first"));
        assert_eq!(
            text_of(&f.messages),
            r#"[{"object":"entry","type":"message.input","role":"user","content":"hi"},{"object":"entry","type":"message.input","role":"user","content":"<instructions>\nsecond\n</instructions>"}]"#
        );
        // A textless later instruction is dropped and cannot erase the preamble.
        let items = vec![
            msg("s0", Role::System, &["preamble"]),
            msg_with("s1", Role::System, vec![image("i")]),
        ];
        assert_eq!(run(items, &url_images(&["i"])).data.instructions.as_deref(), Some("preamble"));
    }

    // Catches: a request ending on the assistant, which the Conversations API refuses.
    #[test]
    fn an_assistant_last_entry_gets_an_empty_user_turn() {
        let items = vec![msg("u", Role::User, &["hi"]), msg("a", Role::Assistant, &["hello"])];
        let f = to_conversations_ctx(&ctx(items.clone()), &no_images(), true).unwrap();
        assert_eq!(
            serde_json::to_value(f.messages.last().unwrap()).unwrap(),
            serde_json::json!({"object": "entry", "type": "message.input", "role": "user", "content": "(empty)"})
        );
        assert_eq!(run(items, &no_images()).messages.len(), 2);
    }

    // Catches: an image-only preamble not clearing the instructions.
    #[test]
    fn an_instruction_message_with_no_text_parts_sets_no_instructions() {
        let items = vec![msg_with("s0", Role::System, vec![image("i")])];
        assert_eq!(run(items, &url_images(&["i"])).data.instructions, None);

        // ...but an empty-string part is still a part, and joins.
        let items = vec![msg_with("s", Role::System, vec![text_content("a"), text_content("")])];
        assert_eq!(run(items, &no_images()).data.instructions.as_deref(), Some("a\n"));
    }

    // Catches: tagging entries by `role` instead of by `type`.
    #[test]
    fn entries_are_tagged_by_type_with_a_distinct_type_per_direction() {
        let items = vec![
            msg("u", Role::User, &["hi"]),
            msg("a", Role::Assistant, &["hello"]),
            call("c", "c0", "f", "{}"),
            out("o", "c0", "f", "ok"),
        ];
        assert_eq!(
            text_of(&run(items, &no_images()).messages),
            r#"[{"object":"entry","type":"message.input","role":"user","content":"hi"},{"object":"entry","type":"message.output","role":"assistant","content":"hello"},{"object":"entry","type":"function.call","tool_call_id":"c0","name":"f","arguments":"{}"},{"object":"entry","type":"function.result","tool_call_id":"c0","result":"ok"}]"#
        );
    }

    // Catches: parsing a call's arguments instead of passing the raw string.
    #[test]
    fn call_arguments_stay_a_raw_string_and_malformed_ones_do_not_raise() {
        let items = vec![call("c", "c0", "f", "not json"), out("o", "c0", "f", "ok")];
        assert!(text_of(&run(items, &no_images()).messages).contains(r#""arguments":"not json""#));
    }

    // Catches: emitting OpenAI's nested `image_url: {url, detail}` object.
    #[test]
    fn images_flatten_to_a_bare_image_url_string() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            text_of(&run(vec![m], &url_images(&["i"])).messages),
            r#"[{"object":"entry","type":"message.input","role":"user","content":[{"type":"image_url","image_url":"https://example.test/i.png"}]}]"#
        );
    }
}
