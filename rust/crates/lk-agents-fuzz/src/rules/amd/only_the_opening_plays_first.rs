use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::trace::{GateStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdOpening, AmdVerdict, Input, Millis, Ref},
    record::{Observed, Recording},
};

// Spec §14 invariant 15, the AMD proposal "Activation and the opening": while AMD is
// attached and before any decision allows a reply, nothing but the opening is heard; and an
// opening is never heard once the participant started speaking after activation, before it
// played (spec §10.6). Speeches already live when AMD attached are left as they were.

pub struct OnlyTheOpeningPlaysFirst;

impl Rule for OnlyTheOpeningPlaysFirst {
    const CONTRACT: Contract = Contract {
        id: "only_the_opening_plays_first",
        guarantee: "Under AMD, nothing but the opening is heard before a decision allows a \
                    reply, and the opening is not heard once the participant spoke first.",
        examples: &[opening_then_allowed_reply],
        counterexamples: &[reply_before_any_decision, opening_after_the_participant_spoke],
        probe: Some(an_opening),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for attachment in amd.attachments() {
            let allowed = amd
                .answers()
                .into_iter()
                .filter(|a| a.allows() && amd.attached_at(a.at()).is_some_and(|x| x.run == attachment.run))
                .map(|a| a.at())
                .next()
                .unwrap_or(Observed::END);
            let end = allowed.min(attachment.until());
            // The opening is the speech the session let bypass the hold (the proposal's owner of
            // invariant 15: "DST sink rule with `OpeningBypass`"). The session applies
            // `Activated` after the driver has moved on, so no input window can pick it out.
            let opening = attachment.activated.and_then(|(activated, opening)| {
                opening?;
                r.journal().iter().find_map(|(at, step)| match step {
                    Step::Gate(GateStep::OpeningBypass { speech })
                        if *at > activated && *at < attachment.until() =>
                    {
                        Some((activated, speech.0))
                    }
                    _ => None,
                })
            });
            // A reply to an earlier run's turn was allowed by that run's completion.
            let earlier: std::collections::BTreeSet<u64> = amd
                .gated_turns()
                .iter()
                .filter(|t| t.run != attachment.run)
                .flat_map(|t| amd.requests_answering(t))
                .filter_map(|q| q.speech)
                .collect();
            for speech in r.host_view().speeches() {
                if speech.created_at().is_none_or(|at| at < attachment.attached)
                    || earlier.contains(&speech.id)
                {
                    continue;
                }
                let Some(heard) = amd.first_audio(speech.id) else { continue };
                if heard < attachment.attached || heard > end {
                    continue;
                }
                check.expect_with(
                    Subject::Speech(speech.id),
                    opening.is_some_and(|(_, o)| o == speech.id),
                    || {
                        format!(
                            "speech {} was heard at {} ms under run {} before any decision \
                             allowed a reply, and it is not the opening",
                            speech.id, heard.at_ms, attachment.run
                        )
                    },
                );
            }
            let Some((activated, opening)) = opening else { continue };
            let heard = amd.first_audio(opening);
            let spoke_first = r.applied().iter().any(|a| {
                a.accepted
                    && a.at > activated
                    && heard.is_none_or(|h| a.at < h)
                    && matches!(a.input, Input::VadStart { .. })
            });
            if spoke_first {
                check.expect_with(Subject::Speech(opening), heard.is_none(), || {
                    format!("the opening {opening} was heard after the participant spoke first")
                });
            }
        }
    }
}

fn activated() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::AmdActivate { opening: Some(AmdOpening::Text) })
        .speech_created(5, SpeechKind::Say)
        .journal(Step::Gate(GateStep::OpeningBypass {
            speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId(5),
        }))
        .sent(Input::Advance { ms: Millis(1) })
}

fn opening_then_allowed_reply() -> Recording {
    activated()
        .pushed(5)
        .user_turn(Some(1), 1, "hello")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello", &[], &[])
        .pushed(10)
        .done()
}

fn reply_before_any_decision() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::AmdActivate { opening: None })
        .user_turn(Some(1), 1, "hello")
        .speech_created(10, SpeechKind::TurnReply)
        .request(10, 1, "hello", &[], &[])
        .pushed(10)
        .done()
}

fn opening_after_the_participant_spoke() -> Recording {
    activated().sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).pushed(5).done()
}

/// The opening plays at activation, before any turn.
fn an_opening() -> crate::drive::Case {
    use crate::fakes::amd::probe::*;
    let mut inputs = attached(Some(AmdOpening::Text));
    inputs.push(advance(3_000));
    call(inputs)
}
