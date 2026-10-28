//! A playout whose finish the output never reports settles once the output played all of it.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: an output that played the speech's audio and lost its "playback finished" report (a
// transport drops it) held the speech waiting on its playout until the session closed, and every
// speech queued behind it (DST seeds 229, 803 with the lost-finish fault).
#[test]
fn a_lost_finish_releases_the_speech_after_its_audio() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::Advance { ms: Millis(30_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech still waits on a finish the output lost");
}

// Catches: a sink that reset mid-speech, dropping what it held, then lost the finish of the
// speech's later audio held the speech until the close: the dropped audio never counted as
// played, so the margin never armed (DST seed 10521 with the lost-finish fault).
#[test]
fn a_lost_finish_after_a_reset_releases_the_speech() {
    let text = |t: &str| AgentText(t.into());
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: text("Sure, I can") },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::SinkReset { output: Ref::new(0) },
        Input::LlmText { chat: Ref::new(0), text: text(" help with that.") },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutFinishLost { segment: Ref::new(1) },
        Input::Advance { ms: Millis(30_000) },
    ];
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech still waits on a finish the output lost");
}
