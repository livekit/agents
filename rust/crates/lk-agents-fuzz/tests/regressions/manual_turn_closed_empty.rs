//! A close commits a manual user turn even when it holds no words.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, TurnMode};
use lk_agents_voice::events::SessionEvent;

// Catches: the close's trailing commit skipped when the turn held no words. The reference's close
// always runs `_commit_user_turn`, and in manual turn detection `_run_eou_detection` proceeds with
// an empty transcript, so the history ends with the (empty) user turn, as in `parity.py
// update_turn_detection`.
#[test]
fn a_close_commits_an_empty_manual_turn() {
    let setup = Setup { stt: true, turn: TurnMode::Manual, ..Setup::default() };
    let inputs = [
        Input::Advance { ms: Millis(100) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(5_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let users: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::User => Some(m.text_content()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(users.len(), 1, "{users:?}");
}
