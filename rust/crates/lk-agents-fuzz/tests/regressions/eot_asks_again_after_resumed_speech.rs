//! A prediction answered during a pause the user talks through is dropped: the next pause asks.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Confidence, Input, Millis, Ref, UserText};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step::Emitted;

// Catches: speech resuming in the segment cancelled only a pending prediction and kept an answered
// one, so no pause after it asked again and the turn was judged by a mid-sentence answer. The
// reference drops `_turn_detector_prediction_fut` whatever its state (`_on_vad_event`).
#[test]
fn the_pause_after_resumed_speech_asks_again() {
    let setup = Setup { vad: true, stt: true, eot: true, ..Setup::default() };
    let vad = Ref::new(0);
    let silence = Input::VadInference {
        vad,
        probability: Confidence(5),
        speaking: false,
        duration: Millis(224),
    };
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad, speech_ms: Millis(300) },
        Input::Advance { ms: Millis(500) },
        // A pause mid-sentence: asked, and answered "likely".
        silence.clone(),
        Input::Advance { ms: Millis(10) },
        Input::EotAnswer { prediction: Ref::new(0), probability: Confidence(90), backchannel: None },
        Input::Advance { ms: Millis(10) },
        // The user goes on.
        Input::VadInference { vad, probability: Confidence(90), speaking: true, duration: Millis(32) },
        Input::Advance { ms: Millis(800) },
        // The real end: asked again, answered "unlikely".
        silence,
        Input::Advance { ms: Millis(10) },
        Input::EotAnswer { prediction: Ref::new(1), probability: Confidence(5), backchannel: None },
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
    let judged: Vec<f64> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Emitted(SessionEvent::EotPrediction(ev)) => Some(ev.probability),
            _ => None,
        })
        .collect();
    assert_eq!(judged, [0.05], "judged by the last pause's answer, not the mid-sentence one");
}
