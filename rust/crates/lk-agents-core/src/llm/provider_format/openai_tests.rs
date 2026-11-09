use lk_agents_abi::llm::provider_format::ProviderFormat;
use serde_json::json;

use super::*;
use crate::llm::provider_format::testkit::{
    call, ctx, image, inline_images, msg, msg_with, no_images, out, text_of, url_images, wire,
};
use crate::llm::provider_format::{ToolSchema, host};

fn chat(items: Vec<lk_agents_abi::ChatItem>, images: &dyn ImageResolver) -> Vec<Message> {
    to_chat_ctx(&ctx(items), images).unwrap().messages
}

fn responses(items: Vec<lk_agents_abi::ChatItem>, images: &dyn ImageResolver) -> Vec<InputItem> {
    to_responses_chat_ctx(&ctx(items), images).unwrap().messages
}

// Catches: `_filter_extra` copying every key present, or emitting `extra_content` for `{}`.
#[test]
fn extra_content_carries_only_truthy_vendor_keys_in_the_constants_order() {
    let mut m = msg("m", Role::User, &["hi"]);
    let e = &mut m.as_message_mut().unwrap().extra;
    e.insert("xai".into(), serde_json::json!({ "x": 1 }));
    e.insert("is_summary".into(), Value::Bool(true));
    e.insert("google".into(), serde_json::json!({ "g": 1 }));
    e.insert("livekit".into(), serde_json::json!({}));
    e.insert("anthropic".into(), serde_json::json!({ "a": 1 }));
    assert_eq!(
        text_of(&chat(vec![m], &no_images())),
        r#"[{"role":"user","content":"hi","extra_content":{"google":{"g":1},"xai":{"x":1}}}]"#
    );

    // ...and nothing truthy means no key, not an empty object.
    let mut m = msg("m", Role::User, &["hi"]);
    m.as_message_mut().unwrap().extra.insert("google".into(), serde_json::json!(0));
    assert_eq!(text_of(&chat(vec![m], &no_images())), r#"[{"role":"user","content":"hi"}]"#);
}

// Catches: emitting a one-element content list for a text-only message.
#[test]
fn text_only_content_collapses_to_a_string_and_mixed_content_does_not() {
    assert_eq!(
        text_of(&chat(vec![msg("m", Role::User, &["a", "b"])], &no_images())),
        r#"[{"role":"user","content":"a\nb"}]"#
    );
    let m = msg_with("m", Role::User, vec![image("img_1")]);
    assert_eq!(
        text_of(&chat(vec![m], &url_images(&["img_1"]))),
        r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/img_1.png","detail":"high"}}]}]"#
    );
}

// Catches: appending the text where it appeared instead of at the end, or when empty.
#[test]
fn accumulated_text_is_appended_after_every_image_and_omitted_when_empty() {
    use crate::llm::provider_format::testkit::text_content;
    let m =
        msg_with("m", Role::User, vec![text_content("before"), image("i"), text_content("after")]);
    assert_eq!(
        text_of(&chat(vec![m], &url_images(&["i"]))),
        r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/i.png","detail":"high"}},{"type":"text","text":"before\nafter"}]}]"#
    );
    let m = msg_with("m", Role::User, vec![image("i")]);
    assert!(!text_of(&chat(vec![m], &url_images(&["i"]))).contains(r#""type":"text""#));
}

// Catches: joining with the separator after each part, or skipping empty parts (Anthropic's rule).
#[test]
fn an_empty_trailing_text_part_still_contributes_its_newline() {
    assert_eq!(
        text_of(&chat(vec![msg("m", Role::User, &["a", ""])], &no_images())),
        r#"[{"role":"user","content":"a\n"}]"#
    );
    assert_eq!(
        text_of(&chat(vec![msg("m", Role::User, &["", "a"])], &no_images())),
        r#"[{"role":"user","content":"a"}]"#,
        "a leading empty part contributes nothing, because the separator comes first"
    );
}

// Catches: dropping a message-less tool-call group's placeholder.
#[test]
fn a_message_less_group_still_emits_a_bare_assistant_message_to_hang_calls_off() {
    let items = vec![call("c", "c0", "lookup", "{\"q\":1}"), out("o", "c0", "lookup", "ok")];
    assert_eq!(
        text_of(&chat(items, &no_images())),
        r#"[{"role":"assistant","tool_calls":[{"id":"c0","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]},{"role":"tool","tool_call_id":"c0","content":"ok"}]"#
    );
}

// Catches: emitting `tool_calls` before `content`/`extra_content`.
#[test]
fn tool_calls_are_appended_after_content_and_extra_content() {
    let mut m = msg("t", Role::Assistant, &["thinking"]);
    m.as_message_mut().unwrap().extra.insert("xai".into(), serde_json::json!(1));
    let items = vec![m, call("t/0", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
    let out_text = text_of(&chat(items, &no_images()));
    let content_at = out_text.find(r#""content""#).unwrap();
    let extra_at = out_text.find(r#""extra_content""#).unwrap();
    let calls_at = out_text.find(r#""tool_calls""#).unwrap();
    assert!(content_at < extra_at && extra_at < calls_at, "{out_text}");
}

// Catches: filtering a tool call's extra with a different rule than a message's.
#[test]
fn a_tool_calls_extra_content_is_filtered_the_same_way() {
    let mut c = call("c", "c0", "f", "{}");
    if let lk_agents_abi::ChatItem::FunctionCall(f) = &mut c {
        f.extra.insert("google".into(), serde_json::json!({ "sig": "x" }));
        f.extra.insert("is_summary".into(), Value::Bool(true));
    }
    let items = vec![c, out("o", "c0", "f", "ok")];
    assert!(
        text_of(&chat(items, &no_images())).contains(r#""extra_content":{"google":{"sig":"x"}}}]"#)
    );
}

// Catches: base64-ing an image that already has an external URL, or dropping `detail`.
#[test]
fn an_inline_image_becomes_a_data_uri_and_a_url_image_is_passed_through() {
    let m = msg_with("m", Role::User, vec![image("i")]);
    assert_eq!(
        text_of(&chat(vec![m.clone()], &inline_images(&["i"]))),
        r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/","detail":"auto"}}]}]"#
    );
    assert!(text_of(&chat(vec![m], &url_images(&["i"]))).contains("https://example.test/i.png"));
}

// Catches: emitting the Chat Completions shapes from the Responses converter.
#[test]
fn the_responses_dialect_emits_top_level_calls_and_no_extra_content() {
    let mut m = msg("t", Role::Assistant, &["thinking"]);
    m.as_message_mut().unwrap().extra.insert("google".into(), serde_json::json!(1));
    let items = vec![m, call("t/0", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
    assert_eq!(
        text_of(&responses(items, &no_images())),
        r#"[{"role":"assistant","content":"thinking"},{"type":"function_call","call_id":"c0","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c0","output":"ok"}]"#
    );
}

// Catches: emitting a placeholder assistant item for a message-less group in Responses.
#[test]
fn the_responses_dialect_emits_no_placeholder_for_a_message_less_group() {
    let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
    assert!(!text_of(&responses(items, &no_images())).contains(r#""role""#));
}

// Catches: echoing `phase` back on a user item, or on a null phase.
#[test]
fn phase_is_echoed_only_for_an_assistant_item_and_only_when_present() {
    let with_phase = |role, phase: Value| {
        let mut m = msg("m", role, &["x"]);
        m.as_message_mut()
            .unwrap()
            .extra
            .insert("openai".into(), serde_json::json!({ "phase": phase }));
        text_of(&responses(vec![m], &no_images()))
    };
    assert!(with_phase(Role::Assistant, "commentary".into()).contains(r#""phase":"commentary""#));
    assert!(!with_phase(Role::User, "commentary".into()).contains("phase"));
    assert!(!with_phase(Role::Assistant, Value::Null).contains("phase"));
    // The key lands after `content`, which is where the reference's assignment puts it.
    let out_text = with_phase(Role::Assistant, "final_answer".into());
    assert!(out_text.find(r#""content""#) < out_text.find(r#""phase""#), "{out_text}");
}

// Pins a deliberate deviation: a non-dict `extra["openai"]` raises in the reference; here it is
// "no phase" and the request is still formatted.
#[test]
fn a_non_object_openai_extra_yields_no_phase_rather_than_the_references_attributeerror() {
    for garbage in [serde_json::json!("commentary"), serde_json::json!(7), serde_json::json!([])] {
        let mut m = msg("m", Role::Assistant, &["x"]);
        m.as_message_mut().unwrap().extra.insert("openai".into(), garbage);
        let text = text_of(&responses(vec![m], &no_images()));
        assert_eq!(text, r#"[{"role":"assistant","content":"x"}]"#);
    }
}

// Catches: nesting a raw schema under `function` in Responses, or not tagging it.
#[test]
fn the_two_fnc_ctx_dialects_wrap_a_raw_schema_differently() {
    let raw = ToolSchema::Raw {
        name: "t".into(),
        raw_schema: serde_json::json!({ "name": "t", "parameters": {} }),
    };
    assert_eq!(
        serde_json::to_string(&to_fnc_ctx(std::slice::from_ref(&raw)).unwrap()).unwrap(),
        r#"[{"type":"function","function":{"name":"t","parameters":{}}}]"#
    );
    assert_eq!(
        serde_json::to_string(&to_responses_fnc_ctx(std::slice::from_ref(&raw)).unwrap()).unwrap(),
        r#"[{"type":"function","name":"t","parameters":{},"strict":null}]"#
    );
}

// Catches: dropping provider tools from Responses, or letting them into Chat Completions.
#[test]
fn provider_tools_reach_the_responses_dialect_only() {
    let tools = [ToolSchema::Provider(serde_json::json!({ "type": "web_search" }))];
    assert!(to_fnc_ctx(&tools).unwrap().is_empty());
    assert_eq!(
        serde_json::to_value(to_responses_fnc_ctx(&tools).unwrap()).unwrap(),
        serde_json::json!([{ "type": "web_search" }])
    );
}

// Catches: reusing the Chat Completions image shape in Responses (flat `image_url` string).
#[test]
fn the_responses_dialect_flattens_the_image_url_and_renames_the_part_type() {
    let m = msg_with("m", Role::User, vec![image("i")]);
    assert_eq!(
        text_of(&responses(vec![m.clone()], &url_images(&["i"]))),
        r#"[{"role":"user","content":[{"type":"input_image","image_url":"https://example.test/i.png","detail":"high"}]}]"#
    );
    assert_eq!(
        text_of(&responses(vec![m], &inline_images(&["i"]))),
        r#"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/jpeg;base64,/9j/","detail":"auto"}]}]"#
    );
}

// Catches: emitting `{"type":"text"}` for a Responses text part.
#[test]
fn the_responses_dialect_names_its_trailing_text_part_input_text() {
    use crate::llm::provider_format::testkit::text_content;
    let m = msg_with("m", Role::User, vec![image("i"), text_content("caption")]);
    assert!(
        text_of(&responses(vec![m], &url_images(&["i"])))
            .contains(r#"{"type":"input_text","text":"caption"}"#)
    );
}

#[test]
fn openai_images_are_only_emitted_for_user_messages() {
    use crate::llm::provider_format::testkit::text_content;
    for format in [ProviderFormat::OpenAi, ProviderFormat::OpenAiResponses] {
        for role in [Role::System, Role::Developer, Role::Assistant] {
            // Invalid-role images must not even ask the image resolver to encode them.
            let messages =
                wire(format, vec![msg_with("m", role, vec![image("i"), text_content("caption")])]);
            assert_eq!(messages[0]["content"], "caption");
        }
    }
}

#[test]
fn openai_encrypted_reasoning_round_trips_in_vendor_content() {
    let mut message = msg("a", Role::Assistant, &["answer"]);
    let reasoning =
        json!({"reasoning": [{"type":"reasoning", "encrypted_content":"opaque-ciphertext"}]});
    message.as_message_mut().unwrap().extra.insert("openai".into(), reasoning.clone());
    let messages = wire(ProviderFormat::OpenAi, vec![message]);
    assert_eq!(messages[0]["extra_content"]["openai"], reasoning);
}

#[test]
fn responses_strict_function_tools_use_the_flat_schema() {
    let tools = host::to_fnc_ctx(
        ProviderFormat::OpenAiResponses,
        vec![ToolSchema::Strict(json!({
            "type": "function", "function": {
                "name": "lookup", "parameters": {"type":"object"}, "strict": true
            }
        }))],
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
    let mut images = inline_images(&["i"]).0;
    images.get_mut("i").unwrap().mime_type = None;
    let (messages, _) = host::to_provider_format(
        ProviderFormat::OpenAi,
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
    let messages = wire(ProviderFormat::OpenAiResponses, vec![message]);
    assert_eq!(messages, [json!({"role":"assistant", "content":"answer"})]);
}
