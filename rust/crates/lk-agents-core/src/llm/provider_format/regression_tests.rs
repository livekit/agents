use super::host::{to_fnc_ctx, to_provider_format};
use super::testkit::{call, image, msg, msg_with, out, text_content};
use lk_agents_abi::{ChatItem, Role};
use serde_json::{Value, json};

fn wire(format: &str, items: Vec<ChatItem>) -> Vec<Value> {
    to_provider_format(format, items, None, false, false, None).unwrap().0
}

#[test]
fn bedrock_raw_tools_omit_missing_and_empty_descriptions() {
    for schema in [json!({}), json!({"description": ""})] {
        let tools = to_fnc_ctx("aws", vec![json!(["raw", schema, "lookup"])], None, false).unwrap();
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
    let messages = wire("aws", vec![call("c", "c0", "f", "{}"), output]);
    assert_eq!(messages[1]["content"][0]["toolResult"]["status"], "error");
}

#[test]
fn anthropic_unknown_image_mime_omits_media_type() {
    for mime in [None, Some("image/unknown".into())] {
        let mut images = super::testkit::inline_images(&["i"]).0;
        images.get_mut("i").unwrap().mime_type = mime;
        let (messages, _) = to_provider_format(
            "anthropic",
            vec![msg_with("u", Role::User, vec![image("i")])],
            Some(images),
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            messages[0]["content"][0],
            json!({
                "type": "image", "source": {"type": "base64", "data": "/9j/"}
            })
        );
    }
}

#[test]
fn anthropic_array_results_are_blocks_only_when_their_schema_validates() {
    for output in ["[1,2]", r#"[{"type":"image"}]"#, r#"[{"type":"tool_use"}]"#] {
        let messages =
            wire("anthropic", vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", output)]);
        assert_eq!(messages[1]["content"][0]["content"], output);
    }
    let output = r#"[{"type":"text","text":"found"}]"#;
    let messages = wire("anthropic", vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", output)]);
    assert_eq!(messages[1]["content"][0]["content"], json!([{"type":"text","text":"found"}]));
}

#[test]
fn openai_images_are_only_emitted_for_user_messages() {
    for format in ["openai", "openai.responses"] {
        for role in [Role::System, Role::Developer, Role::Assistant] {
            // Invalid-role images must not even ask the image resolver to encode them.
            let messages =
                wire(format, vec![msg_with("m", role, vec![image("i"), text_content("caption")])]);
            assert_eq!(messages[0]["content"], "caption");
        }
    }
}

#[test]
fn bedrock_images_are_only_emitted_for_user_messages() {
    let messages = wire(
        "aws",
        vec![msg_with("a", Role::Assistant, vec![image("i"), text_content("caption")])],
    );
    assert_eq!(messages[0]["content"], json!([{"text": "caption"}]));
}

#[test]
fn openai_encrypted_reasoning_round_trips_in_vendor_content() {
    let mut message = msg("a", Role::Assistant, &["answer"]);
    let reasoning =
        json!({"reasoning": [{"type":"reasoning", "encrypted_content":"opaque-ciphertext"}]});
    message.as_message_mut().unwrap().extra.insert("openai".into(), reasoning.clone());
    let messages = wire("openai", vec![message]);
    assert_eq!(messages[0]["extra_content"]["openai"], reasoning);
}

#[test]
fn responses_strict_function_tools_use_the_flat_schema() {
    let tools = to_fnc_ctx(
        "openai.responses",
        vec![json!(["strict", {
            "type": "function", "function": {
                "name": "lookup", "parameters": {"type":"object"}, "strict": true
            }
        }])],
        None,
        false,
    )
    .unwrap();
    assert_eq!(
        tools,
        [
            json!({"type":"function", "name":"lookup", "parameters":{"type":"object"}, "strict":true})
        ]
    );
}

#[test]
fn openai_missing_image_mime_uses_a_valid_jpeg_data_uri() {
    let mut images = super::testkit::inline_images(&["i"]).0;
    images.get_mut("i").unwrap().mime_type = None;
    let (messages, _) = to_provider_format(
        "openai",
        vec![msg_with("u", Role::User, vec![image("i")])],
        Some(images),
        false,
        false,
        None,
    )
    .unwrap();
    assert_eq!(messages[0]["content"][0]["image_url"]["url"], "data:image/jpeg;base64,/9j/");
}

#[test]
fn responses_unknown_phase_is_not_sent_to_the_server() {
    let mut message = msg("a", Role::Assistant, &["answer"]);
    message.as_message_mut().unwrap().extra.insert("openai".into(), json!({"phase":"invalid"}));
    let messages = wire("openai.responses", vec![message]);
    assert_eq!(messages, [json!({"role":"assistant", "content":"answer"})]);
}

#[test]
fn gemini_optional_image_and_function_fields_are_omitted() {
    let image_data = super::SerializedImage {
        mime_type: None,
        data_bytes: None,
        external_url: None,
        inference_detail: lk_agents_abi::InferenceDetail::Auto,
    };
    let (messages, _) = to_provider_format(
        "google",
        vec![msg_with("u", Role::User, vec![image("i")])],
        Some([("i".into(), image_data)].into()),
        false,
        false,
        None,
    )
    .unwrap();
    assert_eq!(messages[0]["parts"], json!([{"inline_data":{}}]));
    let tools = to_fnc_ctx(
        "google",
        vec![json!(["legacy", {
            "name":"lookup", "description":null, "parameters":{}
        }])],
        None,
        true,
    )
    .unwrap();
    assert_eq!(tools, [json!({"name":"lookup"})]);
}
