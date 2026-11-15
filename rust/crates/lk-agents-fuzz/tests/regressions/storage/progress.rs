//! A bound voice conversation saves pending recognition independently of agent speech.

use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
use lk_agents_fuzz::record::{Kind, StorageStep};

/// Interim updates are durable before any reply starts; accepting the turn replaces its draft
/// under the same item identity, even when no reply is requested.
#[test]
fn recognition_is_checkpointed_before_a_reply_and_a_silent_turn_is_saved() {
    let run = simulate(async {
        let setup = Setup {
            stt: true,
            turn: TurnMode::Manual,
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("book".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("book a table".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(0),
                stt_flush: Millis(0),
                skip_reply: true,
            },
            Input::Advance { ms: Millis(2_000) },
            Input::Close { drain: false },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        run
    });
    assert_eq!(run.record.resources(Kind::Chat).count(), 0, "the user turn requested no reply");
    let storage = run.record.host_view().storage();
    let drafts: Vec<_> = storage
        .iter()
        .filter_map(|(_, step)| match step {
            StorageStep::ConversationCheckpoint { pending_user_turns, .. } => {
                pending_user_turns.first()
            }
            _ => None,
        })
        .collect();
    assert!(
        drafts.iter().any(|message| message.raw_text_content().as_deref() == Some("book")),
        "the first interim was durable"
    );
    let draft = drafts
        .iter()
        .find(|message| message.raw_text_content().as_deref() == Some("book a table"))
        .expect("the later interim replaced it durably");
    assert!(
        drafts.iter().all(|message| message.id == draft.id),
        "interim revisions retained one identity"
    );
    let saved = storage
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .expect("the closed store was read back");
    let users: Vec<_> = saved
        .iter()
        .filter_map(ChatItem::as_message)
        .filter(|message| message.role == Role::User)
        .collect();
    assert_eq!(users.len(), 1, "the draft was not duplicated: {saved:?}");
    assert_eq!(users[0].id, draft.id);
    assert_eq!(users[0].raw_text_content().as_deref(), Some("book a table"));
}
