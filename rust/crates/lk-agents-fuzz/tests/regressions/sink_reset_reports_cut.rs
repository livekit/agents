//! An output that throws its playback count away reports what it owed as cut first.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: the fake output resetting its count without reporting the segment it owed, which no
// sink does (the reference's `DataStreamAudioOutput` reports it finished, then resets). The
// wrapper above it kept counting the segment, so the speech waiting on it, or the next one,
// waited until the close (`speech_settles`, survey seed 755; `synthesized_speech_is_played`).
#[test]
fn a_reset_releases_the_speech_waiting_on_playout() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::SinkReset { output: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
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
    assert!(speech.is_done(), "the speech still waits on a playout the output forgot");
}
