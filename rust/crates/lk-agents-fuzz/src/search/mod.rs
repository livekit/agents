mod conversation;
mod setup;
mod uniform;

pub use conversation::{
    Dials, STT_MODELS, SttModel, explore, explore_at, explore_with, randomness,
};
pub use setup::sample_setup;

// Exploration policy lives only here: `input.rs` knows what is legal, `drive/` how a case is
// judged. A seed draws a run's setup, conversation and schedule, so seed plus code is the run.
// A search change is kept only if it wins on the survey's yardstick (`src/survey_tests.rs`).

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

    /// The person speaks, the detector and the recogniser both close the speech, and the driver
    /// decides the turn. The search must reach this in about half of short runs, and every run in
    /// which both observers closed a speech must decide it.
    #[test]
    fn a_conversation_reaches_a_decided_turn() {
        let mut closed = 0;
        for seed in 0..48u64 {
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
        assert!(closed >= 20, "only {closed} of 48 runs had a speech closed by both observers");
    }

    /// How long before its final a recogniser's words may have been spoken: the slowest final
    /// the person's providers send (a late one, 3 s) and a second final after it.
    const HEARD_WITHIN_MS: u64 = 5_000;

    /// The generator's own invariant: every final the recogniser reports has microphone audio
    /// forwarded to its stream in the seconds before it; otherwise the generator would blame the
    /// driver for transcripts of nothing.
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
