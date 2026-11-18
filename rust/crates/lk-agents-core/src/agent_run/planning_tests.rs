use super::planning::{INSTRUCTIONS_MESSAGE_ID, RequestSnapshot, plan_request};
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use std::time::{Duration, UNIX_EPOCH};

/// A request's instructions and one-turn extras apply to that request only.
/// https://github.com/livekit/agents/pull/1822 https://github.com/livekit/agents/pull/1905
/// https://github.com/livekit/agents/pull/1921 https://github.com/livekit/agents/pull/2058
/// https://github.com/livekit/agents/pull/2677
#[test]
fn instruction_replacement_is_isolated_and_turn_extras_do_not_persist() {
    let at = UNIX_EPOCH + Duration::from_secs(3);
    let history: Vec<ChatItem> = vec![
        ChatMessage::text(INSTRUCTIONS_MESSAGE_ID, Role::System, "stale", UNIX_EPOCH).into(),
        ChatMessage::text("user", Role::User, "hello", at).into(),
    ];
    let saved = history.clone();
    let texts = |instructions, turn_instructions| {
        plan_request(RequestSnapshot {
            history: &history,
            instructions,
            tools: &[],
            on_enter: false,
            turn_instructions,
            created_at: at + Duration::from_secs(1),
        })
        .chat_ctx
        .items
        .iter()
        .filter_map(|i| i.as_message().and_then(ChatMessage::raw_text_content))
        .collect::<Vec<_>>()
    };

    let first = texts("first", Some("one turn only"));
    let next = texts("replacement", None);

    assert_eq!(first, ["first", "hello", "one turn only"]);
    assert_eq!(next, ["replacement", "hello"]);
    assert_eq!(history, saved);
}
