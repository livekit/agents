//! The campaign: which inputs to send, and which runs are worth doing.
//!
//! Every policy decision about exploration lives here, and nothing else does. `input.rs` knows
//! what an input is and which ones are legal right now; it deliberately does not know which of
//! them is interesting, because that judgement changes as the search gets smarter and the table
//! of inputs should not change with it. Running a case and judging it is `drive.rs`, so a smarter
//! search cannot break a replayed one.
//!
//! Two files, two subjects. [`setup`] draws what a session is built with, from arms that learn
//! from seed to seed which switches keep finding new ground. [`conversation`] is the search: a
//! person on a phone talking turn by turn to the agent, the developer calling the API, and the
//! models, outputs and room behaving and failing as real ones do. One mode since 2026-10-22: a
//! uniform sampler over the whole table ran beside it, and what only it reached (drains, model
//! failures, room events, output resets, camera frames) became behaviours of the conversation's
//! actors, and its "anything at any time" became the labelled glitch.
//!
//! Every change to any of them is measured on the survey's yardstick -- variants reached,
//! distinct edges, distinct findings, at 64 seeds on three windows and at 256 -- and kept only
//! if it wins; `tests/survey.rs` says how.

mod conversation;
mod setup;
mod uniform;

pub use conversation::{Dials, explore, explore_at, randomness};
pub use setup::{Drawn, SetupArms, sample_setup};

#[cfg(test)]
mod tests {
    use fastrand::Rng;

    use super::*;
    use crate::{
        drive::{Run, Setup},
        input::Input,
        record::Kind,
    };

    /// A session with a microphone and all three observers of the person on it.
    fn listening() -> Setup {
        Setup { stt: true, vad: true, eot: true, ..Setup::default() }
    }

    /// A conversation is a conversation: the person speaks, the detector and the recogniser both
    /// close the speech, and the driver decides the turn. A uniform sampler reaches this in one
    /// run out of ten; the search must reach it every time it has the legs to.
    #[test]
    fn a_conversation_reaches_a_decided_turn() {
        let outcome = explore_at(listening(), 3, 150, Dials::conversation());
        let names = |name: &str| outcome.case.inputs.iter().filter(|i| i.name() == name).count();
        assert!(
            names("VadEnd") >= 1 && names("SttFinal") >= 1,
            "no speech was closed by both observers"
        );
        assert!(
            outcome.reached.iter().any(|v| v.starts_with("Decided::")),
            "a full turn produced no decision: {:?}",
            outcome.reached
        );
    }

    /// The generator's own invariant, which no driver rule can hold it to: every final the
    /// recogniser reports has microphone audio behind it, forwarded to that recogniser's stream
    /// after the utterance opened and before the final. A generator that broke this would be
    /// blaming the driver for transcripts of nothing.
    #[test]
    fn conversation_cases_are_coherent() {
        use crate::record::{Applied, Tag};
        for seed in 0..8u64 {
            let setup = Setup { stt: true, ..sample_setup(&mut Rng::with_seed(seed)) };
            let case = explore_at(setup, seed, 150, Dials::conversation()).case;
            let run = crate::drive::simulate_with(case.schedule, async {
                let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
                for input in &case.inputs {
                    run.apply(input).await;
                }
                run
            });
            let record = &run.record;
            for final_ in record.applied().iter().filter(|a| a.accepted) {
                let Input::SttFinal { utterance, .. } = &final_.input else { continue };
                let opened = record.resource(Kind::Utterance, utterance.nth).map(|r| r.opened_by());
                let Some(Some(Applied { at: start, input: Input::SttStart { stt }, .. })) = opened
                else {
                    panic!("seed {seed}: utterance {} was not opened by an SttStart", utterance.nth)
                };
                let heard = record.forwarded().iter().any(|(at, kind, nth, tags)| {
                    *kind == Kind::Stt
                        && *nth == stt.nth
                        && start.seq < at.seq
                        && at.seq < final_.at.seq
                        && tags.iter().any(|t| matches!(t, Tag::Mic { .. }))
                });
                assert!(
                    heard,
                    "seed {seed}: SttFinal at seq {} for utterance {} on stt#{} has no microphone \
                     audio forwarded since its SttStart at seq {}",
                    final_.at.seq, utterance.nth, stt.nth, start.seq
                );
            }
        }
    }
}
