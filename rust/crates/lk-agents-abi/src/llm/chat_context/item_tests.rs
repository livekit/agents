//! The chat items on their own: text content, instruction and tool-name classification, stamps.

use crate::{Clock, VirtualClock};

use super::*;
use crate::testkit::{t, text};

fn assistant(content: Vec<ChatContent>) -> ChatMessage {
    ChatMessage::new("item_1", Role::Assistant, content, t(1))
}

// Catches: joining the text parts with anything but a newline, or `Some("")` for a message
// with no text parts (diffs would report it updated each time).
#[test]
fn raw_text_content_joins_parts_with_a_newline() {
    let m = assistant(vec![text("one"), text("two"), text("three")]);
    assert_eq!(m.raw_text_content().as_deref(), Some("one\ntwo\nthree"));
    assert_eq!(assistant(vec![]).raw_text_content(), None);
    assert_eq!(assistant(vec![text("")]).raw_text_content().as_deref(), Some(""));
}


// Catches: the #4845 bug class -- the instruction test written as `role == System`.
#[test]
fn is_instruction_covers_developer_as_well_as_system() {
    for role in [Role::System, Role::Developer] {
        assert!(ChatItem::from(ChatMessage::text("i", role, "x", t(0))).is_instruction());
    }
    for role in [Role::User, Role::Assistant] {
        assert!(!ChatItem::from(ChatMessage::text("i", role, "x", t(0))).is_instruction());
    }
}

// Catches: giving a handoff or config-update item a `tool_name`.
#[test]
fn only_function_calls_and_outputs_have_a_tool_name() {
    assert_eq!(crate::testkit::call("c", "lookup", 0).tool_name(), Some("lookup"));
    assert_eq!(crate::testkit::out("o", "lookup", 0).tool_name(), Some("lookup"));
    assert_eq!(crate::testkit::msg("m", Role::User, "hi", 0).tool_name(), None);
    assert_eq!(crate::testkit::handoff("h", 0).tool_name(), None);
    assert_eq!(crate::testkit::config_update("u", 0).tool_name(), None);
}

// Catches: stamping `created_at` from the monotonic clock, or a `wall()` that repeats.
#[test]
fn two_items_minted_in_one_tick_get_distinct_ordered_stamps() {
    let clock = VirtualClock::new();
    let first = ChatMessage::text("item_a", Role::User, "hi", clock.wall());
    let second = ChatMessage::text("item_b", Role::Assistant, "hello", clock.wall());
    assert!(
        second.created_at > first.created_at,
        "{:?} !> {:?}",
        second.created_at,
        first.created_at
    );
}
