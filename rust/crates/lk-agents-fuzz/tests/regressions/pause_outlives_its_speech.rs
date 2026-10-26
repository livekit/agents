//! A pause taken for a speech that then ends does not keep the room's speaker paused.
use lk_agents_fuzz::drive::{Case, RoomLegs, Setup, run};
use lk_agents_fuzz::input::{AgentText, Confidence, Input, Millis, Ref, SayInput};

// Catches: the person speaking over a reply still thinking paused the room's audio output, the
// reply was then interrupted and the queued say took the floor; the pause was released only for
// a paused speech that was still current when it ended, so the speaker stayed paused and every
// later speech waited on it until the close (`speech_settles`, survey seed 4919). The host's fake
// output only records a pause, so only the room's own leg showed it.
#[test]
fn the_next_speech_plays_after_a_paused_one_is_interrupted() {
    let inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Say {
            text: AgentText("Second.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(10) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(800),
        },
        Input::Advance { ms: Millis(50) },
        Input::InterruptSpeech { speech: Ref::new(0), force: true },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(100) },
    ];
    let setup =
        Setup { vad: true, room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
