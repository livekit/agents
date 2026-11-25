use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording},
};

// A turn committed under the gate replies only after an allowed
// decision for it, from `ReplyDecided` or `Detach`. A denied turn yields no reply, an allowed one
// at most one. It also owns spec §14 invariants 3 (`wait` never replies) and 13 (a
// customer-handled stage never replies, including turns `Detach` settles): the scripted
// answerer denies both. Judged on what answered the turn: model requests made for its words and
// audio at the sink, never the session's own account of its gate.

pub struct ReplyFollowsDecision;

impl Rule for ReplyFollowsDecision {
    const CONTRACT: Contract = Contract {
        id: "reply_follows_decision",
        guarantee: "A turn committed under the AMD gate is answered, and heard, only after an \
                    allowing decision for it, at most once; a denied turn is never answered.",
        examples: &[allowed_then_answered, denied_and_silent, decided_before_the_host_heard],
        counterexamples: &[answered_before_its_decision, denied_but_heard, answered_twice],
        probe: Some(an_allowed_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for turn in amd.gated_turns() {
            let allowed = amd.allowed_at(&turn);
            let requests = amd.requests_answering(&turn);
            for request in requests.iter().filter(|q| !amd.preemptive(q)) {
                check.expect_with(
                    Subject::Resource(Kind::Chat, request.chat),
                    allowed.is_some_and(|a| a.at() < request.opened),
                    || {
                        format!(
                            "turn {} ({:?}) was answered by request {} at {} ms, but no decision \
                             allowed it before then ({allowed:?})",
                            turn.turn, turn.words, request.chat, request.opened.at_ms
                        )
                    },
                );
            }
            let speeches: BTreeSet<u64> = requests.iter().filter_map(|q| q.speech).collect();
            let heard: Vec<(u64, _)> =
                speeches.iter().filter_map(|s| amd.first_audio(*s).map(|at| (*s, at))).collect();
            for (speech, at) in &heard {
                check.expect_with(
                    Subject::Speech(*speech),
                    allowed.is_some_and(|a| a.at() < *at),
                    || {
                        format!(
                            "speech {speech} answering turn {} was heard at {} ms with no \
                             allowing decision before it ({allowed:?})",
                            turn.turn, at.at_ms
                        )
                    },
                );
            }
            let answers = amd.answers_for(&turn);
            if !answers.is_empty() && allowed.is_none() {
                check.expect_with(Subject::Session, heard.is_empty() && requests.iter().all(|q| amd.preemptive(q)), || {
                    format!("turn {} was denied ({answers:?}) and was answered", turn.turn)
                });
            }
            if !heard.is_empty() {
                check.expect_with(Subject::Session, heard.len() <= 1, || {
                    format!("turn {} was answered aloud by {} speeches", turn.turn, heard.len())
                });
            }
        }
    }
}

fn attached() -> Authored {
    Authored::new().sent(Input::AmdAttach {}).user_turn(Some(1), 1, "hello who is this")
}

fn decide(verdict: AmdVerdict) -> Input {
    Input::AmdDecide { turn: Ref::new(0), verdict }
}

fn allowed_then_answered() -> Recording {
    attached()
        .sent(decide(AmdVerdict::Allow))
        .request(10, 1, "hello who is this", &[], &[])
        .pushed(10)
        .done()
}

/// AMD answers in the gap between the gate's commit and the host hearing it: the session
/// reports the commit to AMD in its own transition, so the decision still follows the commit.
fn decided_before_the_host_heard() -> Recording {
    use lk_agents_voice::amd::{AmdObservation, AmdRunId};
    use lk_agents_voice::audio_recognition::TurnId;
    use lk_agents_voice::events::{SessionEvent, UserTurnCompletedEvent};
    use lk_agents_voice::trace::{GateStep, Step};
    let words = "hello who is this";
    Authored::new()
        .sent(Input::AmdAttach {})
        .journal(Step::Gate(GateStep::Committed { run: AmdRunId(1), turn: TurnId(1) }))
        .opens(Kind::AmdTurn, crate::record::Detail::AmdTurn { run: 1, turn: 1 })
        .amd(
            1,
            AmdObservation::TurnCommitted {
                turn: TurnId(1),
                transcript: words.into(),
                eot_delay: None,
                skip_reply: false,
                bound: true,
                speaking: false,
                agent: "agent".into(),
                at: lk_agents_abi::Ts::ZERO,
            },
        )
        .sent(decide(AmdVerdict::Allow))
        .journal(Step::Turn(lk_agents_voice::trace::TurnStep::EndOfTurn {
            transcript: words.into(),
            verdict: lk_agents_voice::audio_recognition::UserTurnCommitVerdict::Reply,
            committed: true,
        }))
        .session(SessionEvent::UserTurnCompleted(UserTurnCompletedEvent {
            turn: TurnId(1),
            new_transcript: words.into(),
            transcript_confidence: 1.0,
        }))
        .session(SessionEvent::ConversationItemAdded {
            item: Box::new(lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
                "user_1",
                lk_agents_abi::Role::User,
                words,
                std::time::UNIX_EPOCH,
            ))),
            agent: "agent".into(),
            source: lk_agents_voice::agent::Source::UserTurn { turn: TurnId(1) },
            visibility: lk_agents_voice::agent::Visibility::Both,
        })
        .request(10, 1, words, &[], &[])
        .pushed(10)
        .done()
}

fn denied_and_silent() -> Recording {
    attached().sent(decide(AmdVerdict::Wait)).later(5_000).done()
}

fn answered_before_its_decision() -> Recording {
    attached()
        .request(10, 1, "hello who is this", &[], &[])
        .sent(decide(AmdVerdict::Allow))
        .pushed(10)
        .done()
}

fn denied_but_heard() -> Recording {
    attached()
        .sent(decide(AmdVerdict::Deny))
        .request(10, 1, "hello who is this", &[], &[])
        .pushed(10)
        .done()
}

fn answered_twice() -> Recording {
    attached()
        .sent(decide(AmdVerdict::Allow))
        .request(10, 1, "hello who is this", &[], &[])
        .request(11, 1, "hello who is this", &[], &[])
        .pushed(10)
        .pushed(11)
        .done()
}

/// A turn AMD allows is answered after its decision.
fn an_allowed_turn() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("hello who is this", AmdVerdict::Allow)
}
