//! `_provider_format/aws.py`: Bedrock `Converse`. Anthropic's structure with different leaves, and
//! no URL image source at all.

use lk_agents_abi::ChatContext;
use serde_json::{Map, Value};

use crate::llm::provider_format::image::{ImageResolver, SerializedImage, bytes_value};
use crate::llm::provider_format::turns::{Dialect, Turns, merge_turns};
use crate::llm::provider_format::{FormatError, Formatted, ToolSchema, field, is_truthy, parse_arguments};

const PROVIDER: &str = "AWS Bedrock";

/// `aws.py:13`. Bedrock names formats, not MIME types.
const AWS_IMAGE_FORMATS: [(&str, &str); 4] =
    [("image/jpeg", "jpeg"), ("image/png", "png"), ("image/gif", "gif"), ("image/webp", "webp")];

/// `aws.py:21`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BedrockFormatData {
    pub system_messages: Vec<String>,
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    inject_dummy_user_message: bool,
) -> Result<Formatted<BedrockFormatData>, FormatError> {
    let dialect = Dialect {
        assistant: "assistant",
        output: "user",
        key: "content",
        text: |text| serde_json::json!({ "text": text }),
        image: build_image,
        call: &|call| {
            Ok(serde_json::json!({
                "toolUse": {
                    "toolUseId": call.call_id,
                    "name": call.name,
                    "input": parse_arguments(&call.arguments)?,
                }
            }))
        },
        // `status` is always `"success"`: the reference ignores `is_error` here. Faithful; a fix
        // must change both sides. (Its `isinstance(output, dict)` branch is dead.)
        result: |output| {
            serde_json::json!({
                "toolResult": {
                    "toolUseId": output.call_id,
                    "content": [{ "text": output.output }],
                    "status": "success",
                }
            })
        },
    };
    let Turns { turns: mut messages, system_messages, dropped, .. } =
        merge_turns(chat_ctx, images, &dialect)?;

    if inject_dummy_user_message
        && messages.first().and_then(|m| m.get("role")?.as_str()) != Some("user")
    {
        messages
            .insert(0, serde_json::json!({ "role": "user", "content": [{ "text": "(empty)" }] }));
    }

    Ok(Formatted { messages, data: BedrockFormatData { system_messages }, dropped })
}

/// `aws.py:96`.
fn build_image(img: &SerializedImage) -> Result<Value, FormatError> {
    if img.external_url.is_some() {
        return Err(FormatError::ExternalUrlUnsupported { provider: PROVIDER });
    }
    Ok(serde_json::json!({
        "image": {
            "format": image_format(img.mime_type.as_deref())?,
            "source": { "bytes": bytes_value(img.data_bytes.as_deref().unwrap_or(&[])) },
        }
    }))
}

/// `aws.py:113`. A missing MIME type means jpeg (the encoder's only output).
fn image_format(mime_type: Option<&str>) -> Result<&'static str, FormatError> {
    let Some(mime_type) = mime_type.filter(|m| !m.is_empty()) else { return Ok("jpeg") };
    AWS_IMAGE_FORMATS
        .iter()
        .find(|(mime, _)| *mime == mime_type)
        .map(|(_, format)| *format)
        .ok_or_else(|| FormatError::UnsupportedImageMime {
            provider: PROVIDER,
            mime_type: mime_type.to_string(),
        })
}

/// `aws.py:123`: legacy schemas only (no strict mode).
pub fn to_fnc_ctx(tools: &[ToolSchema]) -> Result<Vec<Value>, FormatError> {
    let mut specs = Vec::new();
    for tool in tools {
        let mut spec = Map::new();
        match tool {
            ToolSchema::Legacy(schema) => {
                spec.insert("name".into(), field(schema, "name")?.clone());
                // A falsy description means no key (`_strip_nones`).
                if let Some(d) = schema.get("description").filter(|d| is_truthy(d)) {
                    spec.insert("description".into(), d.clone());
                }
                let parameters = field(schema, "parameters")?;
                let parameters = if is_truthy(parameters) {
                    parameters.clone()
                } else {
                    Value::Object(Map::new())
                };
                spec.insert("inputSchema".into(), serde_json::json!({ "json": parameters }));
            }
            ToolSchema::Raw { name, raw_schema } => {
                spec.insert("name".into(), Value::String(name.clone()));
                // `.get("description", "")`: an empty string is sent, unlike the legacy arm above.
                spec.insert(
                    "description".into(),
                    raw_schema.get("description").cloned().unwrap_or_else(|| "".into()),
                );
                spec.insert(
                    "inputSchema".into(),
                    serde_json::json!({
                        "json": raw_schema
                            .get("parameters")
                            .cloned()
                            .unwrap_or_else(|| Value::Object(Map::new()))
                    }),
                );
            }
            // `raise ValueError("Invalid function tool")` (`aws.py:152`).
            other => {
                return Err(FormatError::MalformedToolSchema {
                    detail: format!("{PROVIDER} builds legacy schemas only, got {other:?}"),
                });
            }
        }
        specs.push(serde_json::json!({ "toolSpec": spec }));
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::{ChatItem, InferenceDetail, Role};

    use super::*;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, url_images,
    };

    fn text_of(items: Vec<ChatItem>, images: &dyn ImageResolver) -> String {
        serde_json::to_string(&to_chat_ctx(&ctx(items), images, true).unwrap().messages).unwrap()
    }

    // Catches: mapping `is_error` onto `status: "error"`, which the reference never does.
    #[test]
    fn a_failed_tool_result_is_still_reported_to_bedrock_as_success() {
        let mut o = out("o", "c0", "f", "boom");
        if let ChatItem::FunctionCallOutput(x) = &mut o {
            x.is_error = true;
        }
        let items = vec![call("c", "c0", "f", "{}"), o];
        assert!(text_of(items, &no_images()).contains(r#""status":"success""#));
    }

    // Catches: falling back to base64 or an empty payload for a URL image instead of raising.
    #[test]
    fn a_url_image_is_a_hard_error_not_a_silent_drop() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            to_chat_ctx(&ctx(vec![m]), &url_images(&["i"]), true).err(),
            Some(FormatError::ExternalUrlUnsupported { provider: "AWS Bedrock" })
        );
    }

    // Catches: defaulting an unknown MIME type to jpeg, or erroring on a missing one.
    #[test]
    fn a_missing_mime_type_defaults_to_jpeg_but_an_unsupported_one_raises() {
        assert_eq!(image_format(None).unwrap(), "jpeg");
        assert_eq!(image_format(Some("image/webp")).unwrap(), "webp");
        assert_eq!(image_format(Some("image/gif")).unwrap(), "gif");
        assert_eq!(
            image_format(Some("image/svg+xml")).unwrap_err(),
            FormatError::UnsupportedImageMime {
                provider: "AWS Bedrock",
                mime_type: "image/svg+xml".into(),
            }
        );
    }

    // Catches: base64-ing the image payload (Bedrock takes raw bytes).
    #[test]
    fn image_bytes_travel_tagged_rather_than_base64_inline() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            text_of(vec![m], &inline_images(&["i"])),
            r#"[{"role":"user","content":[{"image":{"format":"jpeg","source":{"bytes":{"$lk_bytes_b64":"/9j/"}}}}]}]"#
        );
    }

    // Catches: reading Anthropic's key names or result shape into this dialect.
    #[test]
    fn tool_use_and_tool_result_use_bedrocks_camel_case_shape() {
        let items = vec![call("c", "c0", "f", r#"{"a":1}"#), out("o", "c0", "f", "ok")];
        assert_eq!(
            text_of(items, &no_images()),
            r#"[{"role":"user","content":[{"text":"(empty)"}]},{"role":"assistant","content":[{"toolUse":{"toolUseId":"c0","name":"f","input":{"a":1}}}]},{"role":"user","content":[{"toolResult":{"toolUseId":"c0","content":[{"text":"ok"}],"status":"success"}}]}]"#
        );
    }

    // Catches: prepending the dummy turn when not asked, or Anthropic's text block shape.
    #[test]
    fn the_dummy_turn_uses_bedrocks_bare_text_block() {
        assert_eq!(
            text_of(vec![msg("a", Role::Assistant, &["hi"])], &no_images()),
            r#"[{"role":"user","content":[{"text":"(empty)"}]},{"role":"assistant","content":[{"text":"hi"}]}]"#
        );
        let f = to_chat_ctx(&ctx(vec![msg("a", Role::Assistant, &["hi"])]), &no_images(), false);
        assert_eq!(f.unwrap().messages.len(), 1);
    }

    // Catches: sending `description: ""` for a function tool (the raw arm does send it).
    #[test]
    fn a_falsy_description_is_omitted_for_function_tools_and_kept_for_raw_ones() {
        let legacy = ToolSchema::Legacy(serde_json::json!({
            "name": "t", "description": "", "parameters": {},
        }));
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[legacy]).unwrap()).unwrap(),
            r#"[{"toolSpec":{"name":"t","inputSchema":{"json":{}}}}]"#
        );

        let raw = ToolSchema::Raw { name: "t".into(), raw_schema: serde_json::json!({}) };
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[raw]).unwrap()).unwrap(),
            r#"[{"toolSpec":{"name":"t","description":"","inputSchema":{"json":{}}}}]"#
        );
    }

    // Catches: accepting a strict schema here.
    #[test]
    fn a_strict_schema_is_rejected_because_bedrock_has_no_strict_mode() {
        let strict = ToolSchema::Strict(serde_json::json!({ "function": { "name": "t" } }));
        assert!(matches!(to_fnc_ctx(&[strict]), Err(FormatError::MalformedToolSchema { .. })));
    }

    // Catches: an empty `parameters` reaching Bedrock as `null` or a missing key.
    #[test]
    fn a_falsy_parameters_schema_becomes_an_empty_object_not_null() {
        for falsy in [serde_json::json!(null), serde_json::json!({}), serde_json::json!(false)] {
            let legacy = ToolSchema::Legacy(
                serde_json::json!({ "name": "t", "description": "d", "parameters": falsy }),
            );
            assert!(
                serde_json::to_string(&to_fnc_ctx(&[legacy]).unwrap())
                    .unwrap()
                    .contains(r#""inputSchema":{"json":{}}"#)
            );
        }
    }

    // Catches: `inference_detail` leaking into the Bedrock payload.
    #[test]
    fn bedrock_images_carry_no_detail_field() {
        let img = SerializedImage::inline(b"x".to_vec(), "image/png", InferenceDetail::High);
        assert_eq!(
            serde_json::to_string(&build_image(&img).unwrap()).unwrap(),
            r#"{"image":{"format":"png","source":{"bytes":{"$lk_bytes_b64":"eA=="}}}}"#
        );
    }

    // Catches: hoisting `developer` into `system_messages`, or hoisting an empty system message.
    #[test]
    fn the_system_prompt_is_hoisted_only_for_a_non_empty_system_role_message() {
        let f =
            to_chat_ctx(&ctx(vec![msg("s", Role::System, &["p"])]), &no_images(), true).unwrap();
        assert_eq!(f.data.system_messages, ["p"]);

        let f =
            to_chat_ctx(&ctx(vec![msg("d", Role::Developer, &["p"])]), &no_images(), true).unwrap();
        assert!(f.data.system_messages.is_empty());
        assert!(serde_json::to_string(&f.messages).unwrap().contains(r#"{"text":"p"}"#));

        // Hoisted by neither branch: it becomes an empty user turn and disappears.
        let f =
            to_chat_ctx(&ctx(vec![msg("s", Role::System, &[""])]), &no_images(), false).unwrap();
        assert!(f.data.system_messages.is_empty());
        assert!(f.messages.is_empty());
    }

    // Catches: reading OpenAI's "concatenate every text part" rule into this dialect.
    #[test]
    fn empty_text_parts_are_dropped_rather_than_emitted_as_empty_blocks() {
        assert_eq!(
            text_of(vec![msg("m", Role::User, &["a", "", "b"])], &no_images()),
            r#"[{"role":"user","content":[{"text":"a"},{"text":"b"}]}]"#
        );
    }
}
