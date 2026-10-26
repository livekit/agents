//! A pause taken on the room's speaker never outlives the speech it was taken for.
use lk_agents_fuzz::drive::{Case, RoomLegs, Setup, run};
use lk_agents_fuzz::input::{AgentText, Confidence, Count, Input, Millis, Ref, SayInput};

fn say(input: SayInput) -> Input {
    Input::Say { text: AgentText("Second.".into()), input, allow: None, to_ctx: true }
}

/// The person starts talking over a reply still thinking, which pauses the speaker.
fn talk_over() -> Vec<Input> {
    vec![
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(800),
        },
        Input::Advance { ms: Millis(50) },
    ]
}

/// Every speech settles: none waits on a speaker left paused.
fn settles(inputs: Vec<Input>) {
    let setup = Setup {
        vad: true,
        sync_transcription: true,
        pausable_output: true,
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}

// Catches: the paused reply interrupted and a queued say taking the floor; the pause was released
// only for a paused speech still current when it ended, and the floor leaves a speech at its last
// step, so the speaker stayed paused and every later speech waited until the close
// (`speech_settles`, survey seed 4919). The host's fake output only records a pause, so only the
// room's own leg showed it.
#[test]
fn the_queued_speech_plays_after_a_paused_one_is_interrupted() {
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        say(SayInput::Words),
        Input::Advance { ms: Millis(10) },
    ];
    inputs.extend(talk_over());
    inputs.extend([
        Input::InterruptSpeech { speech: Ref::new(0), force: true },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(100) },
    ]);
    settles(inputs);
}

// Catches: the same pause outliving its interrupted speech with nothing queued, so the next say,
// seconds later, played into a paused speaker (survey seed 4080).
#[test]
fn a_later_speech_plays_after_a_paused_one_is_interrupted() {
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(10) },
    ];
    inputs.extend(talk_over());
    inputs.extend([
        Input::InterruptSpeech { speech: Ref::new(0), force: true },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(3000) },
        say(SayInput::Words),
        Input::Advance { ms: Millis(100) },
    ]);
    settles(inputs);
}

// Catches: the host attaching another audio output while a speech was paused: every later gate
// went to the new output, the speech kept the room's speaker it started on, and nothing resumed
// it, so it played nothing until the close (survey seed 2042).
#[test]
fn a_paused_speech_plays_on_after_the_output_is_replaced() {
    let mut inputs = vec![Input::Advance { ms: Millis(100) }, say(SayInput::Audio)];
    inputs.extend(talk_over());
    inputs.extend([
        Input::SetAudioOutput { attach: true },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(100) },
        Input::FeedAudio { stream: Ref::new(0), chunks: Count(2) },
        Input::FeedEnd { stream: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
    ]);
    settles(inputs);
}
