//! A final transcript ends the pause it took: the paused speech is interrupted.
use lk_agents_fuzz::drive::{Case, RoomLegs, Setup, run};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput, UserText};

fn final_(text: &str) -> Input {
    Input::SttFinal { utterance: Ref::new(0), text: UserText(text.into()), speaker: None }
}

// Catches: a final transcript paused the speaking agent while a drain held scheduling, so the
// user turn was dropped without ending the pause and no false-interruption timer resumed it: the
// speech stayed paused until the close (`speech_settles`, survey seed 2068). The reference's
// `on_final_transcript` always ends the pause with `_cancel_speech_pause`, interrupting it.
#[test]
fn a_final_transcript_interrupts_the_speech_it_paused() {
    let setup = Setup {
        stt: true,
        pausable_output: true,
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("A long answer.".into()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        },
    ];
    // Four seconds of audio, fed as it plays: past the echo canceller's warm-up.
    for _ in 0..8 {
        inputs.push(Input::FeedAudio { stream: Ref::new(0), chunks: Count(50) });
        inputs.push(Input::Advance { ms: Millis(450) });
    }
    inputs.extend([
        Input::Drain { wait: false },
        Input::SttStart { stt: Ref::new(0) },
        final_("no no"),
        Input::Advance { ms: Millis(100) },
        final_("not that one, the other one"),
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ]);
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
