use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::trace::{GateStep, SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdOpening, AmdVerdict, Input, Millis, Ref},
    record::{Observed, Recording},
};

// Spec §14 invariant 15: while AMD is attached and before a decision that allows a reply is
// applied, nothing but the opening is heard; and an opening is never heard once the participant
// started speaking after activation, before it played (spec §10.6): before the session's
// `speaking` edge for it. A run lets at most one speech
// bypass its hold, and that speech is a `say` or a `generate_reply`. A speech already authorized
// when the run attached is left as it was.

pub struct OnlyTheOpeningPlaysFirst;

impl Rule for OnlyTheOpeningPlaysFirst {
    const CONTRACT: Contract = Contract {
        id: "only_the_opening_plays_first",
        guarantee: "Under AMD, nothing but the opening is heard before an allowing decision is \
                    applied, a run lets at most one say or generate_reply bypass its hold, and \
                    the opening is not heard once the participant spoke first.",
        examples: &[opening_then_allowed_reply, speech_after_the_opening_started_playing],
        counterexamples: &[
            reply_before_any_decision,
            opening_after_the_participant_spoke,
            reply_heard_before_its_decision_applied,
            two_bypasses,
        ],
        probe: Some(an_opening),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for attachment in amd.attachments() {
            // The earliest allow applied to a turn the run had bound: a decision sent while its
            // turn's hook still runs, or for a turn committed before activation, frees nothing.
            let allowed = amd
                .gated_turns()
                .into_iter()
                .filter(|t| t.run == attachment.run && amd.bound(t))
                .filter_map(|t| amd.applied(&t))
                .filter(|(answer, _)| answer.allows())
                .map(|(_, at)| at)
                .min()
                .unwrap_or(Observed::END);
            let end = allowed.min(attachment.until());
            let within = |at: &Observed| *at > attachment.attached && *at < attachment.until();
            // The session's own order: what it authorized before it attached the run was already
            // playing.
            let attached_step = r.journal().iter().find_map(|(at, step)| match step {
                Step::Gate(GateStep::Attached { run }) if run.0 == attachment.run => Some(*at),
                _ => None,
            });
            let authorized_before = |speech: u64| {
                attached_step.is_some_and(|attached| {
                    r.journal().iter().any(|(at, step)| {
                        *at < attached
                            && matches!(step, Step::Speech(SpeechStep::Authorized { speech: s })
                                if s.0 == speech)
                    })
                })
            };
            let bypasses: Vec<u64> = r
                .journal()
                .iter()
                .filter(|(at, _)| within(at))
                .filter_map(|(_, step)| match step {
                    Step::Gate(GateStep::OpeningBypass { speech }) => Some(speech.0),
                    _ => None,
                })
                .collect();
            check.expect_with(Subject::Session, bypasses.len() <= 1, || {
                format!("run {} let {} speeches bypass its hold", attachment.run, bypasses.len())
            });
            for speech in &bypasses {
                let kind = r.host_view().speech(*speech).kind();
                check.expect_with(
                    Subject::Speech(*speech),
                    matches!(kind, Some(SpeechKind::Say | SpeechKind::GenerateReply)),
                    || format!("the opening {speech} of run {} is a {kind:?}", attachment.run),
                );
            }
            // The opening is the speech the session let bypass the hold. The session applies
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
                if earlier.contains(&speech.id) || authorized_before(speech.id) {
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
                            "speech {} was heard at {} ms under run {} before any allowing \
                             decision was applied, and it is not the opening",
                            speech.id, heard.at_ms, attachment.run
                        )
                    },
                );
            }
            let Some((activated, opening)) = opening else { continue };
            let heard = amd.first_audio(opening);
            // The opening plays from the session's `speaking` edge for it, as Python sets
            // `_opening_played` on `agent_state_changed` to `speaking` (`detector.py:799`); speech
            // after it is an ordinary interruption, even before its first audio reaches the sink
            // (`test_participant_speech_after_the_opening_starts_is_a_normal_interruption`).
            let played = r.journal().iter().find_map(|(at, step)| match step {
                Step::Speech(SpeechStep::AgentStartedSpeaking { speech }) if speech.0 == opening => {
                    Some(*at)
                }
                _ => None,
            });
            // Only while the run is attached: after its completion the opening still plays, and
            // speech interrupts it as any other (spec §10.6).
            let spoke_first = r.applied().iter().any(|a| {
                a.accepted
                    && a.at > activated
                    && a.at < attachment.until()
                    && played.is_none_or(|p| a.at < p)
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

/// A decision sent while the turn's customer hook still runs is not applied yet: a reply heard
/// before the hook returns played through the hold.
fn reply_heard_before_its_decision_applied() -> Recording {
    use lk_agents_voice::agent_session::{HookId, HookKind};
    use lk_agents_voice::audio_recognition::TurnId;
    use lk_agents_voice::trace::HandoffStep;
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::AmdActivate { opening: None })
        .sent(Input::Advance { ms: Millis(1) })
        .user_turn(Some(1), 1, "hello")
        .journal(Step::Handoff(HandoffStep::HookIssued {
            hook: HookId(3),
            kind: HookKind::OnUserTurnCompleted,
            agent: None,
            speech: None,
            turn: Some(TurnId(1)),
        }))
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .speech_created(10, SpeechKind::TurnReply)
        .request(10, 1, "hello", &[], &[])
        .pushed(10)
        .done()
}

/// Two speeches bypass one run's hold.
fn two_bypasses() -> Recording {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    activated()
        .speech_created(6, SpeechKind::Say)
        .journal(Step::Gate(GateStep::OpeningBypass { speech: SpeechId(6) }))
        .done()
}

/// The session started speaking the opening before the participant spoke; its first audio
/// reached the sink after. Speech then interrupts it as any other, so it may be heard.
fn speech_after_the_opening_started_playing() -> Recording {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    activated()
        .journal(Step::Speech(SpeechStep::AgentStartedSpeaking { speech: SpeechId(5) }))
        .sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) })
        .pushed(5)
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
