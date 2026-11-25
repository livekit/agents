use std::collections::BTreeMap;

use lk_agents_voice::amd::AmdObservation;
use lk_agents_voice::audio_recognition::TurnId;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Recording, amd::Answer},
};

// The session reports `AgentBusy` only when it changes, and `SilenceHold` as a pair: it enters
// the silence hold for a turn once, and leaves it once, and only for a turn whose applied
// decision asked for silence. AMD's idle and inference-timeout rules count from these reports, so
// a repeated or unpaired one moves AMD's clocks.

pub struct ObservationsPair;

impl Rule for ObservationsPair {
    const CONTRACT: Contract = Contract {
        id: "observations_pair",
        guarantee: "Each AMD run hears AgentBusy only when it changes, and SilenceHold for a turn \
                    as one entry then one exit, only for a turn whose applied decision asked \
                    for silence.",
        examples: &[held_then_released],
        counterexamples: &[busy_twice, held_twice, released_unheld, held_without_silence],
        probe: Some(a_voicemail_greeting),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let turns = amd.gated_turns();
        for attachment in amd.attachments() {
            let run = attachment.run;
            let mut busy: Option<bool> = None;
            let mut held: BTreeMap<TurnId, bool> = BTreeMap::new();
            for (at, observation) in amd.observed(run) {
                match observation {
                    AmdObservation::AgentBusy { busy: now, .. } => {
                        check.expect_with(Subject::Session, busy != Some(*now), || {
                            format!("run {run} heard AgentBusy {now} twice in a row at {} ms", at.at_ms)
                        });
                        busy = Some(*now);
                    }
                    AmdObservation::SilenceHold { turn, held: now, .. } => {
                        let was = held.insert(*turn, *now).unwrap_or(false);
                        check.expect_with(Subject::Session, was != *now, || {
                            format!(
                                "run {run} heard SilenceHold {{ turn: {}, held: {now} }} at {} ms \
                                 with the hold already {was}",
                                turn.0, at.at_ms
                            )
                        });
                        if !*now {
                            continue;
                        }
                        let silence = turns
                            .iter()
                            .find(|t| t.run == run && t.turn == turn.0)
                            .and_then(|t| amd.applied(t))
                            .is_some_and(|(answer, _)| {
                                matches!(
                                    answer,
                                    Answer::Decide {
                                        verdict: AmdVerdict::Silence | AmdVerdict::Ivr,
                                        ..
                                    }
                                )
                            });
                        check.expect_with(Subject::Session, silence, || {
                            format!(
                                "run {run} entered the silence hold for turn {} at {} ms with no \
                                 applied decision asking for silence",
                                turn.0, at.at_ms
                            )
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

fn busy(busy: bool) -> AmdObservation {
    AmdObservation::AgentBusy { busy, at: lk_agents_abi::Ts::ZERO }
}

fn hold(held: bool) -> AmdObservation {
    AmdObservation::SilenceHold { turn: TurnId(1), held, at: lk_agents_abi::Ts::ZERO }
}

fn decided(verdict: AmdVerdict) -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .amd(1, busy(false))
        .user_turn(Some(1), 1, "please leave a message")
        .amd(1, busy(true))
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict })
}

fn held_then_released() -> Recording {
    decided(AmdVerdict::Silence).amd(1, hold(true)).later(1_500).amd(1, hold(false)).done()
}

fn busy_twice() -> Recording {
    decided(AmdVerdict::Allow).amd(1, busy(true)).done()
}

fn held_twice() -> Recording {
    decided(AmdVerdict::Silence).amd(1, hold(true)).amd(1, hold(true)).done()
}

fn released_unheld() -> Recording {
    decided(AmdVerdict::Silence).amd(1, hold(false)).done()
}

fn held_without_silence() -> Recording {
    decided(AmdVerdict::Stage).amd(1, hold(true)).done()
}

/// A voicemail greeting's reply enters the silence hold and leaves it.
fn a_voicemail_greeting() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("please leave a message", AmdVerdict::Silence)
}
