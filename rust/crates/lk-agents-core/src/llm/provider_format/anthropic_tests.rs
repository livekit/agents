use lk_agents_abi::{ChatItem, Role};

use super::*;
use crate::llm::provider_format::ToolSchema;
use crate::llm::provider_format::testkit::{
    call, ctx, image, inline_images, msg, msg_with, no_images, out, text_content, text_of,
    url_images,
};

fn run(
    items: Vec<ChatItem>,
    images: &dyn ImageResolver,
) -> Formatted<Message, AnthropicFormatData> {
    to_chat_ctx(&ctx(items), images, Options::default()).unwrap()
}

// Catches: emitting one turn per item instead of coalescing.
#[test]
fn consecutive_items_of_one_effective_role_coalesce_into_a_single_turn() {
    let items = vec![
        msg("u", Role::User, &["question"]),
        msg("t", Role::Assistant, &["thinking"]),
        call("t/0", "c0", "f", "{}"),
        out("o", "c0", "f", "ok"),
        msg("u2", Role::User, &["more"]),
    ];
    assert_eq!(
        text_of(&run(items, &no_images()).messages),
        r#"[{"role":"user","content":[{"type":"text","text":"question"}]},{"role":"assistant","content":[{"type":"text","text":"thinking"},{"type":"tool_use","id":"c0","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c0","content":"ok","is_error":false},{"type":"text","text":"more"}]}]"#
    );
}

// Catches: hoisting `developer` into `system_messages` alongside `system`.
#[test]
fn only_the_system_role_is_hoisted_out_of_band_not_developer() {
    let f = run(vec![msg("d", Role::Developer, &["dev preamble"])], &no_images());
    assert!(f.data.system_messages.is_empty());
    assert_eq!(
        text_of(&f.messages),
        r#"[{"role":"user","content":[{"type":"text","text":"dev preamble"}]}]"#
    );

    let f = run(vec![msg("s", Role::System, &["sys preamble"])], &no_images());
    assert_eq!(f.data.system_messages, ["sys preamble"]);
}

// Catches: reading OpenAI's text rule into this dialect.
#[test]
fn empty_text_parts_are_dropped_rather_than_concatenated() {
    assert_eq!(
        text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images()).messages),
        r#"[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]"#
    );
}

// Catches: stringifying valid blocks, or interpreting arbitrary JSON as content blocks.
#[test]
fn only_a_json_list_tool_result_is_unwrapped_into_content_blocks() {
    let unwrap = |output: &str| {
        let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", output)];
        text_of(&run(items, &no_images()).messages)
    };
    assert!(
        unwrap(r#"[{"type":"text","text":"ok"}]"#)
            .contains(r#""content":[{"type":"text","text":"ok"}]"#)
    );
    assert!(unwrap(r#"{"k":1}"#).contains(r#""content":"{\"k\":1}""#));
    assert!(unwrap("42").contains(r#""content":"42""#));
    assert!(unwrap("not json").contains(r#""content":"not json""#));
}

// Catches: dropping `is_error`.
#[test]
fn a_failed_tool_result_carries_is_error() {
    let mut o = out("o", "c0", "f", "boom");
    if let ChatItem::FunctionCallOutput(x) = &mut o {
        x.is_error = true;
    }
    let items = vec![call("c", "c0", "f", "{}"), o];
    assert!(text_of(&run(items, &no_images()).messages).contains(r#""is_error":true"#));
}

// Catches: appending the "(empty)" turn, or prepending it when the conversation opens on user.
#[test]
fn the_dummy_user_turn_is_prepended_and_only_when_needed() {
    let f = run(vec![msg("a", Role::Assistant, &["hi"])], &no_images());
    assert_eq!(
        text_of(&f.messages),
        r#"[{"role":"user","content":[{"type":"text","text":"(empty)"}]},{"role":"assistant","content":[{"type":"text","text":"hi"}]}]"#
    );
    assert_eq!(
        text_of(&run(vec![msg("u", Role::User, &["hi"])], &no_images()).messages),
        r#"[{"role":"user","content":[{"type":"text","text":"hi"}]}]"#
    );
    // An empty conversation still gets one -- `not messages` is the first disjunct.
    assert_eq!(run(vec![], &no_images()).messages.len(), 1);
    // ...unless the caller turned it off.
    let opts = Options { inject_dummy_user_message: false, ..Options::default() };
    assert!(to_chat_ctx(&ctx(vec![]), &no_images(), opts).unwrap().messages.is_empty());
}

// Catches: enabling the trailing turn unconditionally, or appending it after a user turn.
#[test]
fn the_trailing_user_turn_is_off_by_default_and_only_follows_an_assistant_turn() {
    let items = vec![msg("u", Role::User, &["q"]), msg("a", Role::Assistant, &["partial"])];
    assert!(
        !text_of(&run(items.clone(), &no_images()).messages)
            .contains(r#"{"type":"text","text":"."}"#)
    );

    let opts = Options { inject_dummy_user_message: true, inject_trailing_user_message: true };
    let f = to_chat_ctx(&ctx(items), &no_images(), opts).unwrap();
    assert!(
        text_of(&f.messages)
            .ends_with(r#"{"role":"user","content":[{"type":"text","text":"."}]}]"#)
    );

    let f = to_chat_ctx(&ctx(vec![msg("u", Role::User, &["q"])]), &no_images(), opts).unwrap();
    assert_eq!(f.messages.len(), 1, "a trailing user turn needs no help");
}

// Catches: emitting OpenAI's `data:` URI shape.
#[test]
fn images_use_anthropics_source_object_not_a_data_uri() {
    let m = msg_with("m", Role::User, vec![image("i")]);
    assert!(text_of(&run(vec![m.clone()], &inline_images(&["i"])).messages).contains(
        r#"{"type":"image","source":{"type":"base64","data":"/9j/","media_type":"image/jpeg"}}"#
    ));
    assert!(text_of(&run(vec![m], &url_images(&["i"])).messages).contains(
        r#"{"type":"image","source":{"type":"url","url":"https://example.test/i.png"}}"#
    ));
}

// Catches: emitting an empty `content: []` turn for an item with no blocks.
#[test]
fn a_turn_that_contributes_no_blocks_is_not_emitted() {
    let items = vec![
        msg("u", Role::User, &["q"]),
        msg_with("a", Role::Assistant, vec![text_content("")]),
        msg("u2", Role::User, &["more"]),
    ];
    assert_eq!(
        text_of(&run(items, &no_images()).messages),
        r#"[{"role":"user","content":[{"type":"text","text":"q"}]},{"role":"user","content":[{"type":"text","text":"more"}]}]"#
    );
}

// Catches: unpacking a strict schema as if flat, or forgetting `strict: true`.
#[test]
fn strict_and_legacy_tool_schemas_unpack_from_different_levels() {
    let strict = ToolSchema::Strict(serde_json::json!({
        "type": "function",
        "function": { "name": "t", "description": null, "parameters": { "type": "object" } },
    }));
    assert_eq!(
        serde_json::to_string(&to_fnc_ctx(&[strict]).unwrap()).unwrap(),
        r#"[{"name":"t","description":"","input_schema":{"type":"object"},"strict":true}]"#
    );

    let legacy = ToolSchema::Legacy(serde_json::json!({
        "name": "t", "description": "does a thing", "parameters": { "type": "object" },
    }));
    assert_eq!(
        serde_json::to_string(&to_fnc_ctx(&[legacy]).unwrap()).unwrap(),
        r#"[{"name":"t","description":"does a thing","input_schema":{"type":"object"}}]"#
    );
}
