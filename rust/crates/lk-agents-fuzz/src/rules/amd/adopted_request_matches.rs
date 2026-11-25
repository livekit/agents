use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use crate::fakes::amd::SEND_DTMF_TOOL;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::amd::STAGE,
    input::{AmdVerdict, Input, Ref},
    record::{Recording, amd::Answer},
};

// An allowed reply adopts the
// preemptive generation only when the decision adds no instructions and no DTMF, so the request
// that was heard equals the one a new generation would have sent. A preemptive generation never
// carries AMD's DTMF tool.

pub struct AdoptedRequestMatches;

impl Rule for AdoptedRequestMatches {
    const CONTRACT: Contract = Contract {
        id: "adopted_request_matches",
        guarantee: "A preemptive generation is heard as a gated turn's reply only when the turn's \
                    decision adds no stage instructions and no DTMF tool, and it never carries \
                    either.",
        examples: &[plain_allow_adopts],
        counterexamples: &[staged_allow_adopts],
        probe: Some(a_direct_human),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        // Once an agent declares its own tool with that id, the tool is in every request after.
        let declared = amd.agent_declared_at(SEND_DTMF_TOOL);
        for turn in amd.gated_turns() {
            for request in amd.requests_answering(&turn).into_iter().filter(|q| amd.preemptive(q)) {
                let amds_dtmf = declared.is_none_or(|at| request.opened < at);
                check.expect_with(
                    Subject::Resource(crate::record::Kind::Chat, request.chat),
                    !request.carries(STAGE)
                        && !(amds_dtmf && request.tools.iter().any(|t| t == SEND_DTMF_TOOL)),
                    || format!("preemptive request {} carries AMD's stage or DTMF", request.chat),
                );
                let Some(heard) = request.speech.and_then(|s| amd.first_audio(s)) else {
                    continue;
                };
                // The decision applied, not the first one sent: a detach replaces a decision
                // stored while the turn was in its hook.
                let allowed = amd.deciding(&turn).filter(Answer::allows);
                let plain = matches!(
                    allowed,
                    Some(Answer::Decide { verdict: AmdVerdict::Allow, .. } | Answer::Detach { .. })
                );
                check.expect_with(
                    Subject::Resource(crate::record::Kind::Chat, request.chat),
                    plain && allowed.is_some_and(|a| a.at() < heard),
                    || {
                        format!(
                            "preemptive request {} was heard for turn {} although its decision \
                             was {allowed:?}",
                            request.chat, turn.turn
                        )
                    },
                );
            }
        }
    }
}

fn guessed() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .speech_created(10, SpeechKind::Preemptive)
        .request(10, 9, "hello this is Sam", &[], &[])
        .user_turn(Some(1), 1, "hello this is Sam")
}

fn plain_allow_adopts() -> Recording {
    guessed()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .pushed(10)
        .done()
}

fn staged_allow_adopts() -> Recording {
    guessed()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Stage })
        .pushed(10)
        .done()
}

/// A direct human answer adopts the preemptive generation made for its words.
fn a_direct_human() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("hello this is Sam", AmdVerdict::Allow)
}
