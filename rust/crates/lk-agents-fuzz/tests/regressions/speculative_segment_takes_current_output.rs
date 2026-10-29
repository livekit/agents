//! A segment synthesised before its authorization takes the outputs current at its activation: an
//! audio output detached meanwhile never plays it, and its words show without that audio.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
use lk_agents_fuzz::record::Sink;

#[test]
fn a_segment_synthesised_before_a_detach_plays_nothing_on_the_detached_output() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::PauseAuthorization {},
            Input::Say {
                text: AgentText("Words synthesised ahead.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(100) },
            Input::ResumeAuthorization {},
            Input::Advance { ms: Millis(5000) },
        ] {
            run.apply(&input).await;
        }
        let sink = run.record.sink();
        assert!(
            !sink.iter().any(|(_, call)| matches!(call, Sink::Audio { .. })),
            "the speech activated after its output was detached: no audio output plays it"
        );
        let shown: String = sink
            .iter()
            .filter_map(|(_, call)| match call {
                Sink::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, "Words synthesised ahead.");
    });
}
