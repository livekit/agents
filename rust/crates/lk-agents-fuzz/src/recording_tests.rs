mod a_paused_speaker {
    //! A speaker that pauses mid-segment reports the rest of it as a run that starts later, at the
    //! next offset (the room output's ledger reports each run so): the recording holds that run
    //! where it played, not right after the first.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

    // Catches: a placement that continued its segment's offsets appended right after the previous
    // one whatever its start, so the run after a pause landed early and the recording was silent
    // where it played: by 1.2 s (survey seeds 153 and 209), or by 20 ms, inside the input clock's
    // resync tolerance (survey seed 202).
    #[test]
    fn the_run_after_a_pause_is_recorded_where_it_played() {
        // (what plays before the pause, the pause's end): both reported runs played.
        for (first, resumes) in [(500, 1_700), (56, 76)] {
            let inputs = [
                Input::Recorder { attach: true },
                Input::Advance { ms: Millis(500) },
                Input::Say {
                    text: AgentText("I'm sorry, I did not catch that at all.".into()),
                    input: SayInput::Words,
                    allow: None,
                    to_ctx: true,
                },
                Input::Advance { ms: Millis(100) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(first) },
                // The speaker pauses, then plays on from where it stopped.
                Input::Advance { ms: Millis(resumes) },
                Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(1_000) },
                Input::Advance { ms: Millis(5_000) },
            ];
            let run = simulate(async {
                let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
                for input in &inputs {
                    run.apply(input).await;
                }
                run
            });
            let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
            let unheard: Vec<_> =
                failures.iter().filter(|f| f.contract == "recording_is_what_played").collect();
            assert!(unheard.is_empty(), "paused at {first} ms until {resumes} ms: {unheard:?}");
        }
    }
}

mod a_recorder_attached_mid_segment {
    //! A recorder attached while a segment plays records the rest of it from the attach on: the
    //! person hears it. What played before the attach is not the recording's.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

    // Catches: a recorder attached mid-segment never given the segment's audio, captured before
    // it existed, so the recording was silent while the person heard the rest (survey seed 241).
    #[test]
    fn the_rest_of_the_playing_segment_is_recorded() {
        let inputs = [
            Input::Say {
                text: AgentText("I'm sorry, I did not catch that at all.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(300) },
            Input::Advance { ms: Millis(300) },
            Input::Recorder { attach: true },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(500) },
            Input::Advance { ms: Millis(500) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
        let unheard: Vec<_> =
            failures.iter().filter(|f| f.contract == "recording_is_what_played").collect();
        assert!(unheard.is_empty(), "{unheard:?}");
        let played = run.record.sink().iter().any(|(_, sink)| {
            matches!(
                sink,
                lk_agents_fuzz::record::Sink::Recorder(
                    lk_agents_fuzz::record::RecorderCall::PlaybackProgressed
                )
            )
        });
        assert!(played, "the recorder was told the rest of the segment played");
    }
}
