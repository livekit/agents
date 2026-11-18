//! A real storage read completes before a host accepts new local history.

use std::time::UNIX_EPOCH;

use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_abi::voice::speech_handle::SpeechHandle as _;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use lk_agents_fuzz::drive::{Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Ref, SayInput};
use lk_agents_fuzz::record::StorageStep;
use lk_agents_voice::events::SessionEvent;

fn item(id: &str, role: Role, text: &str) -> ChatItem {
    ChatMessage::text(id, role, text, UNIX_EPOCH).into()
}

#[test]
fn a_foreign_read_rebases_a_disjoint_local_edit_before_its_following_save() {
    history_receipt(false);
}

#[test]
fn a_foreign_read_refuses_a_same_row_local_collision_without_losing_either_intent() {
    history_receipt(true);
}

fn history_receipt(conflict: bool) {
    let (run, local, foreign, retained) = simulate(async {
        let setup =
            Setup { tts: false, storage: Some(StorageSetup::default()), ..Setup::default() };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let original = item("case_item_0", Role::User, "original human words");
        let other = item("foreign", Role::Assistant, "original remote words");
        run.foreign_history(vec![original.clone(), other.clone()]).await.unwrap();
        run.apply(&Input::SetAudioOutput { attach: false }).await;
        run.apply(&Input::Say {
            text: AgentText("Ready.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        })
        .await;
        let first = run.speech_handle(Ref::new(0)).expect("the host created the first speech");
        run.until("the initial speech commits and releases its partition", first.wait_reported())
            .await;

        let local = item("case_item_0", Role::User, "explicit local intent");
        let foreign = if conflict {
            item("case_item_0", Role::User, "foreign edit")
        } else {
            item("foreign", Role::Assistant, "foreign edit")
        };
        let mut current = vec![original, other];
        *current.iter_mut().find(|item| item.id() == foreign.id()).unwrap() = foreign.clone();
        current.push(item("foreign-addition", Role::User, "new external input"));
        run.foreign_history(current).await.unwrap();
        let (read, release) = run.hold_next_history_read();
        run.apply(&Input::Say {
            text: AgentText("Continuing.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        })
        .await;
        let second = run.speech_handle(Ref::new(1)).expect("the host created the second speech");
        run.until("the owned snapshot query completed before the local edit", read.wait())
            .await
            .unwrap();
        run.apply(&Input::UpdateChatCtx {
            items: vec![lk_agents_fuzz::input::Role::User],
            text: AgentText("explicit local intent".into()),
        })
        .await;
        release.resolve(());
        run.until(
            "the second speech settles its real receipt or typed conflict",
            second.wait_reported(),
        )
        .await;
        let retained = run.registry.session().unwrap().agent_chat_ctx();
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::AwaitExit {}).await;
        (run, local, foreign, retained)
    });
    let storage = run.record.host_view().storage();
    let saved = storage
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .expect("the observer read the real database after close");
    assert!(saved.contains(&foreign), "an unread newer foreign payload remains authoritative");
    assert!(saved.iter().any(|item| item.id() == "foreign-addition"));
    if conflict {
        assert!(!saved.contains(&local), "a conflicting local edit receives no successful save");
        assert!(retained.contains(&local), "the native owner retains the explicit local intent");
        assert!(run.record.host_view().events().iter().any(|(_, event)| matches!(event,
            SessionEvent::Diagnostic(diagnostic) if diagnostic.message == lk_agents_voice::events::diagnostics::HISTORY_CONFLICT
        )));
    } else {
        assert!(saved.contains(&local), "the disjoint local edit is rebased and durably saved");
        assert!(run.record.host_view().events().iter().any(|(_, event)| matches!(event,
            SessionEvent::Diagnostic(diagnostic) if diagnostic.message == lk_agents_voice::events::diagnostics::HISTORY_REBASED
        )));
    }
    let (findings, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(findings.is_empty(), "the unchanged payload and lease rules apply: {findings:#?}");
}
