//! Real speech projection in the prompts built by inference's conversation judges.

use std::time::SystemTime;

use lk_agents_abi::{ChatItem, ChatMessage, Role};
use lk_agents_inference::evals::prompts;

#[test]
fn judge_prompts_strip_only_assistant_markup_in_conversation_and_reference() {
    let message = |id: &str, role, text: &str| -> ChatItem {
        ChatMessage::text(id, role, text, SystemTime::UNIX_EPOCH).into()
    };
    let marked = "<expr type=\"expression\" label=\"happy\"/>Hello.";
    let conversation =
        vec![message("assistant", Role::Assistant, marked), message("user", Role::User, marked)];
    let reference = vec![
        message("system", Role::System, "hidden instructions"),
        message("reference", Role::Assistant, marked),
    ];
    let prompt =
        prompts::built_in("conciseness").unwrap().prompt(&conversation, Some(&reference));
    assert!(prompt.contains("Conversation:\nassistant: Hello.\nuser: <expr"));
    assert!(prompt.contains("Reference:\nassistant: Hello."));
    assert!(!prompt.contains("hidden instructions"));
}
