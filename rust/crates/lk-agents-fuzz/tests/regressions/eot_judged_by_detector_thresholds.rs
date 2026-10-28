//! A prediction is judged against the detector's thresholds, not the mini model's stock table.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::fakes::EOT_THRESHOLD;
use lk_agents_fuzz::input::{Confidence, Input, Millis, Ref, UserText};
use lk_agents_voice::trace::{Step, TurnStep};
use lk_agents_voice::{events::SessionEvent, trace::Step::Emitted};

// Catches: the session judging every prediction against `ThresholdOptions::new(V1Mini, None,
// None)` (en 0.36) whatever the detector held: `unlikely_threshold=0.001` was ignored, and a
// question the mini model scored 0.34 waited the full `max_delay` on 7-10% of turns where the
// reference replied at once (agents-bench, 2026-10-27, "Is there a vegetarian option on the
// menu?").
#[test]
fn a_prediction_is_judged_by_the_detectors_threshold() {
    let setup = Setup { vad: true, stt: true, eot: true, ..Setup::default() };
    let vad = Ref::new(0);
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad, speech_ms: Millis(300) },
        Input::Advance { ms: Millis(1_500) },
        // 224 ms of silence: the detector is asked, its request registered a step later.
        Input::VadInference { vad, probability: Confidence(5), speaking: false, duration: Millis(224) },
        Input::Advance { ms: Millis(10) },
        // Under the detector's 0.5, over the stock table's 0.36.
        Input::EotAnswer { prediction: Ref::new(0), probability: Confidence(45), backchannel: None },
        Input::VadEnd { vad, silence_ms: Millis(250) },
        Input::SttStart { stt: Ref::new(0) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("is there a vegetarian option on the menu".into()),
            speaker: None,
        },
        Input::Advance { ms: Millis(1_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let thresholds: Vec<f64> = journal
        .iter()
        .filter_map(|(_, step)| match step {
            Emitted(SessionEvent::EotPrediction(ev)) => Some(ev.threshold),
            _ => None,
        })
        .collect();
    assert_eq!(thresholds, [EOT_THRESHOLD], "judged once, by the detector's threshold");
    // 0.45 < 0.5: the countdown runs `max_delay`, so no turn ends within the second after.
    let ended = journal.iter().any(|(_, step)| matches!(step, Step::Turn(TurnStep::EndOfTurn { .. })));
    assert!(!ended, "an unlikely end of turn waited min_delay, as the stock table would have it");
}
