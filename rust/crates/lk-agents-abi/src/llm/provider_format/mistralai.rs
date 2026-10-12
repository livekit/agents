//! `_provider_format/mistralai.py`: entries tagged by `type`, no coalescing, and the last system
//! message wins `instructions`.

use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Role};
use serde_json::Value;

use crate::llm::provider_format::group::group_tool_calls;
use crate::llm::provider_format::image::{ImageResolver, SerializedImage, data_uri};
use crate::llm::provider_format::openai::split_content;
use crate::llm::provider_format::{FormatError, Formatted};

/// `mistralai.py:12`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MistralFormatData {
    pub instructions: Option<String>,
}

/// `mistralai.py:17`.
pub fn to_conversations_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
) -> Result<Formatted<MistralFormatData>, FormatError> {
    let grouping = group_tool_calls(&chat_ctx.items)?;
    let mut entries: Vec<Value> = Vec::new();
    let mut instructions: Option<String> = None;

    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }

        if let Some(message) = group.message {
            if message.role == Role::System || message.role == Role::Developer {
                // Joins every non-media part; no text parts resets `instructions` to `None`.
                let text_parts: Vec<&str> = message
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ChatContent::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect();
                instructions =
                    if text_parts.is_empty() { None } else { Some(text_parts.join("\n")) };
                // Skips the rest of the group, as the reference (unreachable: system messages get
                // their own group).
                continue;
            }

            if let Some(entry) = to_entry(message, images)? {
                entries.push(entry);
            }
        }

        for call in &group.tool_calls {
            entries.push(serde_json::json!({
                "type": "function.call",
                "tool_call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }));
        }

        for output in &group.tool_outputs {
            entries.push(serde_json::json!({
                "type": "function.result",
                "tool_call_id": output.call_id,
                "result": output.output,
            }));
        }
    }

    Ok(Formatted {
        messages: entries,
        data: MistralFormatData { instructions },
        dropped: grouping.dropped,
    })
}

/// `mistralai.py:71`: `None` for a role other than user or assistant.
fn to_entry(
    message: &ChatMessage,
    images: &dyn ImageResolver,
) -> Result<Option<Value>, FormatError> {
    let (entry_type, role) = match message.role {
        Role::User => ("message.input", "user"),
        Role::Assistant => ("message.output", "assistant"),
        Role::System | Role::Developer => return Ok(None),
    };

    // OpenAI's content walk and `{"type": "text"}` part shape.
    let (list_content, text_content) = split_content(message, images, to_image_content, "text")?;
    let content = match list_content {
        None => Value::String(text_content),
        Some(list) => Value::Array(list),
    };

    Ok(Some(serde_json::json!({ "type": entry_type, "role": role, "content": content })))
}

/// `mistralai.py:108`: `image_url` as a string, no `detail`.
fn to_image_content(img: &SerializedImage) -> Value {
    let url = match (&img.external_url, &img.data_bytes) {
        (Some(url), _) => url.clone(),
        (None, bytes) => data_uri(img.mime_type.as_deref(), bytes.as_deref().unwrap_or(&[])),
    };
    serde_json::json!({ "type": "image_url", "image_url": url })
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::ChatItem;

    use super::*;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, msg, msg_with, no_images, out, text_content, url_images,
    };

    fn run(items: Vec<ChatItem>, images: &dyn ImageResolver) -> Formatted<MistralFormatData> {
        to_conversations_ctx(&ctx(items), images).unwrap()
    }

    fn text_of(f: &Formatted<MistralFormatData>) -> String {
        serde_json::to_string(&f.messages).unwrap()
    }

    // Catches: concatenating system messages instead of last-one-wins.
    #[test]
    fn the_last_instruction_message_wins_rather_than_being_concatenated() {
        let items = vec![
            msg("s0", Role::System, &["first"]),
            msg("u", Role::User, &["hi"]),
            msg("s1", Role::Developer, &["second"]),
        ];
        assert_eq!(run(items, &no_images()).data.instructions.as_deref(), Some("second"));
    }

    // Catches: an image-only system message not erasing an earlier preamble.
    #[test]
    fn an_instruction_message_with_no_text_parts_clears_the_instructions() {
        let items = vec![
            msg("s0", Role::System, &["preamble"]),
            msg_with("s1", Role::System, vec![image("i")]),
        ];
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
            text_of(&run(items, &no_images())),
            r#"[{"type":"message.input","role":"user","content":"hi"},{"type":"message.output","role":"assistant","content":"hello"},{"type":"function.call","tool_call_id":"c0","name":"f","arguments":"{}"},{"type":"function.result","tool_call_id":"c0","result":"ok"}]"#
        );
    }

    // Catches: parsing a call's arguments instead of passing the raw string.
    #[test]
    fn call_arguments_stay_a_raw_string_and_malformed_ones_do_not_raise() {
        let items = vec![call("c", "c0", "f", "not json"), out("o", "c0", "f", "ok")];
        assert!(text_of(&run(items, &no_images())).contains(r#""arguments":"not json""#));
    }

    // Catches: emitting OpenAI's nested `image_url: {url, detail}` object.
    #[test]
    fn images_flatten_to_a_bare_image_url_string() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            text_of(&run(vec![m], &url_images(&["i"]))),
            r#"[{"type":"message.input","role":"user","content":[{"type":"image_url","image_url":"https://example.test/i.png"}]}]"#
        );
    }
}
