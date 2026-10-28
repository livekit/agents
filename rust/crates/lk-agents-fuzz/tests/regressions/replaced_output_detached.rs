//! A selected replacement leaves an admitted transcript on its original output until it ends.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
use lk_agents_fuzz::record::{Kind, Sink};

#[test]
fn a_replaced_synchronised_text_leg_stays_attached_until_its_segment_settles() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Say {
                text: AgentText("First segment, still speaking.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
            Input::SetTranscriptionOutput { attach: true },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        assert!(
            run.record.sink().iter().any(|(_, call)| matches!(call, Sink::Audio { output: 0, .. })),
            "the first segment was admitted before the replacement"
        );
        assert!(
            run.record.resource(Kind::Output, 1).unwrap().closed_at().is_none(),
            "the admitted old output must remain live after selection changes"
        );
        for input in [
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::Say {
                text: AgentText("Next segment.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(50) },
        ] {
            run.apply(&input).await;
        }
        let text_for = |id| {
            run.record
                .sink()
                .iter()
                .filter_map(|(_, call)| match call {
                    Sink::Text { output, text } if *output == id => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        };
        assert_eq!(text_for(1), "First segment, still speaking.");
        assert_eq!(text_for(2), "Next segment.");
        assert!(
            run.record.resource(Kind::Output, 1).unwrap().closed_at().is_some(),
            "the old binding was released after the segment settled"
        );
    });
}
