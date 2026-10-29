//! The campaign: which inputs to send, and which runs are worth doing.
//!
//! Every policy decision about exploration lives here, and nothing else does. `input.rs` knows
//! what an input is and which ones are legal right now; it deliberately does not know which of
//! them is interesting, because that judgement changes as the search gets smarter and the table
//! of inputs should not change with it. Running a case and judging it is `drive.rs`, so a smarter
//! search cannot change how a case is judged.
//!
//! Two files, two subjects. [`setup`] draws what a session is built with, from the seed alone.
//! [`conversation`] is the search: a person on a phone talking turn by turn to the agent, the
//! developer calling the API, and the models, outputs and room behaving and failing as real ones
//! do. One mode since 2026-10-22: a uniform sampler over the whole table ran beside it, and what
//! only it reached (drains, model failures, room events, output resets, camera frames) became
//! behaviours of the conversation's actors, and its "anything at any time" became the labelled
//! glitch. A seed is the whole of a run: its setup, its conversation and its schedule are all
//! drawn from it, so the same seed on the same code is the same run.
//!
//! Every change to any of them is measured on the survey's yardstick -- variants reached,
//! distinct edges, distinct findings, at 64 seeds on three windows and at 256 -- and kept only
//! if it wins; `tests/survey.rs` says how.

mod conversation;
mod setup;
mod uniform;

pub use conversation::{
    Dials, STT_MODELS, SttModel, explore, explore_at, explore_with, randomness,
};
pub use setup::sample_setup;

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
    /// run out of ten; the search must reach it in most short runs, and every run in which both
    /// observers closed a speech must decide it. Over several seeds, so a change to what the
    /// generator draws does not turn one unlucky seed into a failure.
    #[test]
    fn a_conversation_reaches_a_decided_turn() {
        let mut closed = 0;
        for seed in 0..12u64 {
            let outcome = explore_at(listening(), seed, 200, Dials::conversation());
            let names =
                |name: &str| outcome.case.inputs.iter().filter(|i| i.name() == name).count();
            // A final with words: Deepgram's empty finals over noise close no turn.
            let worded =
                outcome.case.inputs.iter().any(
                    |i| matches!(i, Input::SttFinal { text, .. } if !text.0.trim().is_empty()),
                );
            if names("VadEnd") == 0 || !worded {
                continue;
            }
            closed += 1;
            assert!(
                outcome.reached.iter().any(|v| v.starts_with("Decided::")),
                "seed {seed}: a full turn produced no decision: {:?}",
                outcome.reached
            );
        }
        assert!(closed >= 8, "only {closed} of 12 runs had a speech closed by both observers");
    }

    /// The generator's own invariant, which no driver rule can hold it to: every final the
    /// recogniser reports has microphone audio behind it, forwarded to that recogniser's stream
    /// in the seconds before the final. A real recogniser opens its utterance with its first
    /// words (as measured), so the audio comes before the start of speech, not after it. A
    /// generator that broke this would be blaming the driver for transcripts of nothing.
    /// How long before its final a recogniser's words may have been spoken: the slowest final
    /// the person's providers send (a late one, 3 s) and a second final after it.
    const HEARD_WITHIN_MS: u64 = 5_000;

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
                        && at.seq < final_.at.seq
                        && at.at_ms + HEARD_WITHIN_MS >= final_.at.at_ms
                        && tags.iter().any(|t| matches!(t, Tag::Mic { .. }))
                });
                assert!(
                    heard,
                    "seed {seed}: SttFinal at seq {} for utterance {} on stt#{} (opened at seq {}) \
                     has no microphone audio forwarded in the {HEARD_WITHIN_MS} ms before it",
                    final_.at.seq, utterance.nth, stt.nth, start.seq
                );
            }
        }
    }
}
