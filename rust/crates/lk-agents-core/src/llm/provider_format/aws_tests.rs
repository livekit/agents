use lk_agents_abi::llm::provider_format::ProviderFormat;
use lk_agents_abi::{ChatItem, InferenceDetail, Role};
use serde_json::json;

use super::*;
use crate::llm::provider_format::host;
use crate::llm::provider_format::testkit::{
    call, ctx, image, inline_images, msg, msg_with, no_images, out, text_content, text_of,
    url_images, wire,
};

fn run(items: Vec<ChatItem>, images: &dyn ImageResolver) -> Formatted<Message, BedrockFormatData> {
    to_chat_ctx(&ctx(items), images, true).unwrap()
}

// Catches: ignoring the failure and reporting success to the model.
#[test]
fn a_failed_tool_result_is_reported_to_bedrock_as_error() {
    let mut o = out("o", "c0", "f", "boom");
    if let ChatItem::FunctionCallOutput(x) = &mut o {
        x.is_error = true;
    }
    let items = vec![call("c", "c0", "f", "{}"), o];
    assert!(text_of(&run(items, &no_images()).messages).contains(r#""status":"error""#));
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
    assert_eq!(image_format(None).unwrap(), ImageFormat::Jpeg);
    assert_eq!(image_format(Some("image/webp")).unwrap(), ImageFormat::Webp);
    assert_eq!(image_format(Some("image/gif")).unwrap(), ImageFormat::Gif);
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
        text_of(&run(vec![m], &inline_images(&["i"])).messages),
        r#"[{"role":"user","content":[{"image":{"format":"jpeg","source":{"bytes":{"$lk_bytes_b64":"/9j/"}}}}]}]"#
    );
}

// Catches: reading Anthropic's key names or result shape into this dialect.
#[test]
fn tool_use_and_tool_result_use_bedrocks_camel_case_shape() {
    let items = vec![call("c", "c0", "f", r#"{"a":1}"#), out("o", "c0", "f", "ok")];
    assert_eq!(
        text_of(&run(items, &no_images()).messages),
        r#"[{"role":"user","content":[{"text":"(empty)"}]},{"role":"assistant","content":[{"toolUse":{"toolUseId":"c0","name":"f","input":{"a":1}}}]},{"role":"user","content":[{"toolResult":{"toolUseId":"c0","content":[{"text":"ok"}],"status":"success"}}]}]"#
    );
}

// Catches: prepending the dummy turn when not asked, or Anthropic's text block shape.
#[test]
fn the_dummy_turn_uses_bedrocks_bare_text_block() {
    assert_eq!(
        text_of(&run(vec![msg("a", Role::Assistant, &["hi"])], &no_images()).messages),
        r#"[{"role":"user","content":[{"text":"(empty)"}]},{"role":"assistant","content":[{"text":"hi"}]}]"#
    );
    let f = to_chat_ctx(&ctx(vec![msg("a", Role::Assistant, &["hi"])]), &no_images(), false);
    assert_eq!(f.unwrap().messages.len(), 1);
}

// Catches: sending an empty description, which Bedrock rejects.
#[test]
fn an_empty_description_is_omitted_for_function_and_raw_tools() {
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
        r#"[{"toolSpec":{"name":"t","inputSchema":{"json":{}}}}]"#
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
    let f = to_chat_ctx(&ctx(vec![msg("s", Role::System, &["p"])]), &no_images(), true).unwrap();
    assert_eq!(f.data.system_messages, ["p"]);

    let f = to_chat_ctx(&ctx(vec![msg("d", Role::Developer, &["p"])]), &no_images(), true).unwrap();
    assert!(f.data.system_messages.is_empty());
    assert!(serde_json::to_string(&f.messages).unwrap().contains(r#"{"text":"p"}"#));

    // Hoisted by neither branch: it becomes an empty user turn and disappears.
    let f = to_chat_ctx(&ctx(vec![msg("s", Role::System, &[""])]), &no_images(), false).unwrap();
    assert!(f.data.system_messages.is_empty());
    assert!(f.messages.is_empty());
}

// Catches: reading OpenAI's "concatenate every text part" rule into this dialect.
#[test]
fn empty_text_parts_are_dropped_rather_than_emitted_as_empty_blocks() {
    assert_eq!(
        text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images()).messages),
        r#"[{"role":"user","content":[{"text":"a"},{"text":"b"}]}]"#
    );
}

#[test]
fn bedrock_raw_tools_omit_missing_and_empty_descriptions() {
    for schema in [json!({}), json!({"description": ""})] {
        let tools = host::to_fnc_ctx(
            ProviderFormat::Aws,
            vec![ToolSchema::Raw { name: "lookup".into(), raw_schema: schema }],
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            tools,
            [json!({"toolSpec": {
                "name": "lookup", "inputSchema": {"json": {}}
            }})]
        );
    }
}

#[test]
fn bedrock_failed_tool_results_have_error_status() {
    let mut output = out("o", "c0", "f", "permission denied");
    if let ChatItem::FunctionCallOutput(output) = &mut output {
        output.is_error = true;
    }
    let messages = wire(ProviderFormat::Aws, vec![call("c", "c0", "f", "{}"), output]);
    assert_eq!(messages[1]["content"][0]["toolResult"]["status"], "error");
}

#[test]
fn bedrock_images_are_only_emitted_for_user_messages() {
    let messages = wire(
        ProviderFormat::Aws,
        vec![msg_with("a", Role::Assistant, vec![image("i"), text_content("caption")])],
    );
    assert_eq!(messages[0]["content"], json!([{"text": "caption"}]));
}
