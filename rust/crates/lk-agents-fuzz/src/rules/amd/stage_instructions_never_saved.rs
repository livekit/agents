use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::amd::STAGE,
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording, StorageStep, amd::Answer},
};

// Spec §14 invariant 8: stage
// instructions reach every request of the reply they were decided for, its tool follow-ups
// included (spec §10.2), and never the conversation the session records or saves.

pub struct StageInstructionsNeverSaved;

impl Rule for StageInstructionsNeverSaved {
    const CONTRACT: Contract = Contract {
        id: "stage_instructions_never_saved",
        guarantee: "A staged reply's requests carry AMD's stage instructions, and no recorded or \
                    saved item ever does.",
        examples: &[staged_and_unsaved],
        counterexamples: &[staged_but_lost, staged_and_saved],
        probe: Some(a_screener),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let mut staged_any = false;
        for turn in amd.gated_turns() {
            let Some(Answer::Decide { at, verdict, .. }) = amd.deciding(&turn) else { continue };
            if !matches!(verdict, AmdVerdict::Stage | AmdVerdict::Silence | AmdVerdict::Ivr) {
                continue;
            }
            staged_any = true;
            let requests = amd.requests_answering(&turn);
            for request in requests.iter().filter(|q| !amd.preemptive(q) && q.opened > at) {
                check.expect_with(
                    Subject::Resource(Kind::Chat, request.chat),
                    request.carries(STAGE),
                    || format!("request {} of turn {}'s staged reply lost its stage", request.chat, turn.turn),
                );
            }
        }
        if !staged_any {
            return;
        }
        for (at, event) in r.host_view().events() {
            let SessionEvent::ConversationItemAdded { item, .. } = event else { continue };
            let text = item.as_message().and_then(|m| m.raw_text_content()).unwrap_or_default();
            check.expect_with(Subject::Session, !text.contains(STAGE), || {
                format!("the conversation recorded AMD's stage instructions at {} ms", at.at_ms)
            });
        }
        for (at, step) in r.host_view().storage() {
            let StorageStep::Saved { items } = step else { continue };
            let saved = items.iter().any(|item| {
                item.as_message().and_then(|m| m.raw_text_content()).is_some_and(|t| t.contains(STAGE))
            });
            check.expect_with(Subject::Session, !saved, || {
                format!("the thread saved AMD's stage instructions at {} ms", at.at_ms)
            });
        }
    }
}

fn staged() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "you have reached the screener")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Stage })
}

fn staged_and_unsaved() -> Recording {
    staged().request(10, 1, "you have reached the screener", &[STAGE], &[]).done()
}

fn staged_but_lost() -> Recording {
    staged().request(10, 1, "you have reached the screener", &[], &[]).done()
}

fn staged_and_saved() -> Recording {
    staged()
        .request(10, 1, "you have reached the screener", &[STAGE], &[])
        .session(SessionEvent::ConversationItemAdded {
            item: Box::new(lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
                "stage",
                lk_agents_abi::Role::System,
                STAGE,
                std::time::UNIX_EPOCH,
            ))),
            agent: "agent".into(),
            source: lk_agents_voice::agent::Source::HostCtx,
            visibility: lk_agents_voice::agent::Visibility::Both,
        })
        .done()
}

/// A screener's turn is answered under stage instructions.
fn a_screener() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("you have reached the screener", AmdVerdict::Stage)
}
