use super::*;
use crate::agent::{Source, Visibility};
use crate::agent_session::tests::driver;

/// Recognition queued while a save is running must include the foreign edits that save
/// discovers. Keeping its earlier full snapshot would write those edits back to old values.
#[test]
fn a_waiting_checkpoint_uses_the_reconciled_conversation() {
    let mut driver = driver();
    driver.session.storage.enabled = true;
    driver.checkpoint_conversation();
    let agent = driver.activity.agent_name().to_owned();
    let message = |id, role, text| ChatMessage::text(id, role, text, std::time::UNIX_EPOCH);
    let original = ChatItem::Message(message("text", Role::Assistant, "Original."));
    let _ = driver.session.chat_ctx.commit(original, Source::HostCtx, Visibility::Both, &agent);
    driver.checkpoint_conversation();
    let first = driver
        .take_outbox()
        .into_iter()
        .find_map(|effect| match effect {
            VoiceEffect::SaveConversation(progress) => Some(progress),
            _ => None,
        })
        .expect("the first save starts");

    let human = ChatItem::Message(message("voice", Role::User, "Please continue."));
    let _ = driver.session.chat_ctx.commit(human, Source::HostCtx, Visibility::Both, &agent);
    driver.checkpoint_conversation();
    assert!(driver.take_outbox().is_empty(), "the successor waits for the running save");

    let updated = ChatItem::Message(message("text", Role::Assistant, "Edited by text."));
    driver.conversation_saved(
        Ts::ZERO,
        first.sequence,
        first.history.items,
        Ok(VoiceHistory {
            items: vec![updated.clone()],
            stored: vec![updated.clone()],
            revision: (1, 1),
        }),
    );
    let next = driver
        .take_outbox()
        .into_iter()
        .find_map(|effect| match effect {
            VoiceEffect::SaveConversation(progress) => Some(progress),
            _ => None,
        })
        .expect("the waiting recognition starts saving");
    assert_eq!(next.history.items.iter().find(|item| item.id() == "text"), Some(&updated));
    assert!(next.history.items.iter().any(|item| item.id() == "voice"));
}
