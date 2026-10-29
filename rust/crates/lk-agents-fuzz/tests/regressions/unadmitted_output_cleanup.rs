//! An interruption may clear only the destination physically admitted for that speech.
use lk_agents_fuzz::search;

#[test]
fn interrupting_a_captured_output_does_not_clear_it_before_physical_admission() {
    let seed = 149;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let outcome = search::explore(setup, seed, 3_000);
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
    assert!(outcome.case.inputs.iter().any(|input| matches!(
        input,
        lk_agents_fuzz::input::Input::SetAudioOutput { attach: true }
    )));
    assert!(
        outcome
            .case
            .inputs
            .iter()
            .any(|input| matches!(input, lk_agents_fuzz::input::Input::Interrupt { .. }))
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}

#[test]
fn disabling_selected_text_keeps_its_admitted_segment_until_it_finishes() {
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
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
    });
}
