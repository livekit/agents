//! A disabled transcription leg keeps the segment already admitted to it until that segment ends.

#[test]
fn disabling_selected_text_keeps_its_admitted_segment_until_it_finishes() {
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
    use lk_agents_fuzz::record::Sink;
    use lk_agents_voice::agent_session::IoKind;
    let say = |text: &str| Input::Say {
        text: AgentText(text.into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            say("First segment, still speaking."),
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
            Input::SetIoEnabled { leg: IoKind::Transcription, enabled: false },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            say("Next segment."),
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        let shown: String = run
            .record
            .sink()
            .iter()
            .filter_map(|(_, call)| match call {
                Sink::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, "First segment, still speaking.");
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
