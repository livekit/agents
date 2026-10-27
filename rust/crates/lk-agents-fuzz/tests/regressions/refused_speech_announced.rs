//! A speech handle refused before it reached the queue is still announced and ended.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
use lk_agents_voice::events::SpeechOutcome;

/// Each speech the host holds a handle for: whether it was announced, and what `SpeechDone`s it
/// heard.
fn announced(setup: Setup, inputs: &[Input]) -> Vec<(bool, Vec<SpeechOutcome>)> {
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
        .host_view()
        .speeches()
        .iter()
        .filter(|s| s.is_known())
        .map(|s| {
            let done = s.done_announcements().into_iter().map(|(_, outcome)| outcome).collect();
            (s.created_at().is_some(), done)
        })
        .collect()
}

fn say(input: SayInput) -> Input {
    Input::Say { text: AgentText("Hello.".into()), input, allow: None, to_ctx: true }
}

fn realtime() -> Setup {
    // No voice: with one, the TTS speaks what the model cannot.
    Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        tts: false,
        ..Setup::default()
    }
}

// Catches: `say(words)` with no TTS answered with a failed handle the host stream never heard of
// (survey seed 0, speech 3).
#[test]
fn say_without_tts_is_announced() {
    let setup = Setup { tts: false, ..Setup::default() };
    let inputs = [say(SayInput::Words), Input::Advance { ms: Millis(20) }];
    assert_eq!(announced(setup, &inputs), [(true, vec![SpeechOutcome::Failed])]);
}

// Catches: `say(stream)` refused by a realtime model answered with a failed handle the host
// stream never heard of (survey seed 1, speech 4).
#[test]
fn say_stream_on_realtime_is_announced() {
    let inputs = [say(SayInput::TextStream), Input::Advance { ms: Millis(20) }];
    assert_eq!(announced(realtime(), &inputs), [(true, vec![SpeechOutcome::Failed])]);
}

// Catches: a realtime `generate_reply()` still waiting on the server when it is interrupted,
// answered with an interrupted handle the host stream never heard of (survey seed 14, speech 8).
#[test]
fn interrupted_realtime_request_is_announced() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(20) },
    ];
    assert_eq!(announced(realtime(), &inputs), [(true, vec![SpeechOutcome::Interrupted])]);
}
