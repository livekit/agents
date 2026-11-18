mod the_room_speaker_is_the_reference {
    use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate, simulate_with};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};
    use lk_agents_fuzz::record::{RecorderCall, Sink};

    fn assert_matches(run: &Run) {
        assert!(run.record.driver_finished(), "the real session closed");
        assert!(
            !run.record
                .sink()
                .iter()
                .any(|(_, s)| { matches!(s, Sink::Recorder(RecorderCall::Failed)) }),
            "the recording completed successfully"
        );
        assert!(run.record.sink().iter().any(|(_, s)| {
            matches!(s, Sink::Recorder(RecorderCall::SpeakerChecked { samples }) if *samples > 0)
        }), "compare independently observed speaker PCM, including the final tail");
        let (failures, expectations) = lk_agents_fuzz::check::check_all(&run.record);
        let failures: Vec<_> =
            failures.iter().filter(|f| f.contract == "recording_is_what_played").collect();
        assert!(expectations.get("recording_is_what_played").is_some_and(|n| *n > 0));
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn recording_matches_the_speaker_through_interruptions_and_close() {
        for (elapsed, interrupt) in
            [(2_000, false), (165, true), (395, true), (165, false), (395, false)]
        {
            let run = simulate(async {
                let setup =
                    Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
                let mut run = Run::start(&setup, 0, Default::default()).await;
                for input in [
                    Input::Recorder { attach: true },
                    Input::Say {
                        text: AgentText(
                            "A reply with audio still buffered when interrupted".into(),
                        ),
                        input: SayInput::Audio,
                        allow: None,
                        to_ctx: true,
                    },
                    Input::FeedAudio { stream: Ref::new(0), chunks: Count(100) },
                    Input::FeedEnd { stream: Ref::new(0) },
                    Input::Advance { ms: Millis(elapsed) },
                ] {
                    run.apply(&input).await;
                }
                if interrupt {
                    run.apply(&Input::Interrupt { force: true, wait: false }).await;
                    run.apply(&Input::Advance { ms: Millis(3_000) }).await;
                }
                run.apply(&Input::Close { drain: false }).await;
                run.apply(&Input::AwaitExit {}).await;
                run
            });
            assert!(
                run.record.sink().iter().any(|(_, s)| {
                    matches!(s, Sink::Recorder(RecorderCall::PlaybackProgressed))
                }),
                "exercise real room playback: {elapsed}, interrupt={interrupt}"
            );
            assert_matches(&run);
        }
    }

    #[test]
    fn attaching_between_progress_reports_preserves_the_audible_audio() {
        for (schedule, attach_at) in [(0, 5), (1, 15), (2, 175), (3, 199), (4, 221)] {
            let run = simulate_with(schedule, async {
                let setup =
                    Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
                let mut run = Run::start(&setup, schedule, Default::default()).await;
                for input in [
                    Input::Say {
                        text: AgentText("The recorder joins a reply already playing".into()),
                        input: SayInput::Audio,
                        allow: None,
                        to_ctx: true,
                    },
                    Input::FeedAudio { stream: Ref::new(0), chunks: Count(100) },
                    Input::FeedEnd { stream: Ref::new(0) },
                    Input::Advance { ms: Millis(attach_at) },
                    Input::Recorder { attach: true },
                    Input::Advance { ms: Millis(3_000) },
                    Input::Close { drain: false },
                    Input::AwaitExit {},
                ] {
                    run.apply(&input).await;
                }
                run
            });
            assert_matches(&run);
        }
    }

    #[test]
    fn a_starved_audio_stream_keeps_real_silence_before_it_resumes() {
        let run = simulate(async {
            let setup =
                Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Recorder { attach: true },
                Input::Say {
                    text: AgentText("An answer that stalls waiting for audio".into()),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(20) },
                Input::Advance { ms: Millis(2_000) },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(30) },
                Input::FeedEnd { stream: Ref::new(0) },
                Input::Advance { ms: Millis(3_000) },
                Input::Close { drain: false },
                Input::AwaitExit {},
            ] {
                run.apply(&input).await;
            }
            run
        });
        assert_matches(&run);
    }

    // Regression for https://github.com/livekit/agents/pull/6921: no microphone frames
    // arrive until well after the greeting, and the next reply keeps its real position.
    #[test]
    fn a_late_microphone_does_not_collapse_twenty_seconds_of_silence() {
        use lk_agents_fuzz::record::Tag;
        let run = simulate(async {
            let setup =
                Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Recorder { attach: true },
                Input::Say {
                    text: AgentText("Hello, can you hear me?".into()),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(20) },
                Input::FeedEnd { stream: Ref::new(0) },
                Input::Advance { ms: Millis(20_000) },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(5) },
                Input::Advance { ms: Millis(100) },
                Input::Say {
                    text: AgentText("There you are.".into()),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedAudio { stream: Ref::new(1), chunks: Count(20) },
                Input::FeedEnd { stream: Ref::new(1) },
                Input::Advance { ms: Millis(3_000) },
                Input::Close { drain: false },
                Input::AwaitExit {},
            ] {
                run.apply(&input).await;
            }
            run
        });
        assert_matches(&run);
        assert!(
            run.record
                .sink()
                .iter()
                .any(|(_, sink)| { matches!(sink, Sink::Recorder(RecorderCall::InputFrame)) }),
            "the late microphone actually reached the recording"
        );
        assert!(
            run.room_heard_audio().iter().any(|(at, frame)| {
                let sample = i16::from_le_bytes(frame.data[..2].try_into().unwrap());
                matches!(Tag::decode(sample), Tag::Fed { stream: 1, .. })
                    && at.saturating_duration_since(lk_agents_abi::Ts::ZERO).as_secs() >= 20
            }),
            "the second reply played after the real silence"
        );
    }

    #[test]
    fn clock_corrections_keep_transcripts_and_traces_on_the_recording_timeline() {
        use lk_agents_abi::telemetry::decl::SpanDecl as _;
        use lk_agents_abi::telemetry::record::Record;
        use lk_agents_abi::telemetry::spans::agent_turn::AgentSpeaking;
        use lk_agents_abi::{ChatItem, Role};
        use lk_agents_fuzz::record::Tag;
        use lk_agents_voice::events::SessionEvent;
        use std::time::Duration;

        for (forward, attach_late) in [(true, false), (false, false), (true, true), (false, true)] {
            let (run, report) = simulate(async {
                let setup =
                    Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
                let mut run = Run::start(&setup, 0, Default::default()).await;
                if !attach_late {
                    run.apply(&Input::Recorder { attach: true }).await;
                }
                for stream in 0..2 {
                    if stream == 1 {
                        run.shift_wall_clock(forward, Duration::from_secs(10));
                        if attach_late {
                            run.apply(&Input::Recorder { attach: true }).await;
                        }
                    }
                    for input in [
                        Input::Say {
                            text: AgentText(format!("Reply {stream}")),
                            input: SayInput::Audio,
                            allow: None,
                            to_ctx: true,
                        },
                        Input::FeedAudio { stream: Ref::new(stream), chunks: Count(10) },
                        Input::FeedEnd { stream: Ref::new(stream) },
                        Input::Advance { ms: Millis(500) },
                    ] {
                        run.apply(&input).await;
                    }
                }
                run.apply(&Input::Advance { ms: Millis(3_000) }).await;
                run.apply(&Input::Close { drain: false }).await;
                run.apply(&Input::AwaitExit {}).await;
                let report = run.session_report().await;
                (run, report)
            });
            assert_matches(&run);
            let spoken_at = run
                .room_heard_audio()
                .iter()
                .find_map(|(at, frame)| {
                    let sample = i16::from_le_bytes(frame.data[..2].try_into().unwrap());
                    matches!(Tag::decode(sample), Tag::Fed { stream: 1, .. }).then_some(*at)
                })
                .expect("the reply after the clock correction played");
            let expected = lk_agents_abi::unix_secs(lk_agents_abi::simulated_epoch())
                + spoken_at.saturating_duration_since(lk_agents_abi::Ts::ZERO).as_secs_f64();
            let (origin, written) = run
                .record
                .sink()
                .iter()
                .find_map(|(_, sink)| match sink {
                    Sink::Recorder(RecorderCall::Origin {
                        unix_micros,
                        written_through_unix_micros,
                    }) => Some((
                        *unix_micros as f64 / 1e6,
                        written_through_unix_micros.unwrap() as f64 / 1e6,
                    )),
                    _ => None,
                })
                .expect("the finalized recording's bounds");
            let epoch = lk_agents_abi::unix_secs(lk_agents_abi::simulated_epoch());
            let first_recorded = if attach_late { expected } else { epoch };
            assert!(
                (origin - first_recorded).abs() < 0.001,
                "a late recorder must not acquire a shifted wall origin"
            );
            assert!(
                written <= epoch + run.record.now().at_ms as f64 / 1000.0 + 0.001,
                "a clock correction must not flush silence ahead of the real conversation"
            );
            let started = report
                .chat_history
                .iter()
                .filter_map(|item| match item {
                    ChatItem::Message(message) if message.role == Role::Assistant => Some(message),
                    _ => None,
                })
                .nth(1)
                .expect("both replies were kept")
                .metrics
                .started_speaking_at
                .unwrap();
            let span_start = run
                .record
                .host_view()
                .events()
                .iter()
                .rev()
                .find_map(|(_, event)| {
                    let SessionEvent::Telemetry(records) = event else { return None };
                    records.iter().find_map(|record| match record {
                        Record::Span(span) if span.name == AgentSpeaking::NAME => {
                            Some(span.start_unix_nanos as f64 / 1e9)
                        }
                        _ => None,
                    })
                })
                .expect("the speaking trace was exported");
            for (label, at) in [("transcript", started), ("trace", span_start)] {
                assert!(
                    (at - expected).abs() < 0.001,
                    "{label}: forward={forward}, skew={} seconds",
                    at - expected
                );
            }
        }
    }

    #[test]
    fn cloud_upload_origin_aligns_transcript_and_trace_with_the_recorded_audio() {
        use lk_agents_abi::telemetry::decl::SpanDecl as _;
        use lk_agents_abi::telemetry::record::Record;
        use lk_agents_abi::telemetry::report::{CloudAgent, RecordingOptions};
        use lk_agents_abi::telemetry::spans::agent_turn::AgentSpeaking;
        use lk_agents_abi::{ChatItem, Role};
        use lk_agents_voice::events::SessionEvent;

        let (run, mut report) = simulate(async {
            let setup =
                Setup { room_legs: Some(RoomLegs { pre_connect: None }), ..Setup::default() };
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Recorder { attach: true },
                // Session creation is deliberately earlier than the file's first PCM.
                Input::Advance { ms: Millis(400) },
                Input::Say {
                    text: AgentText("The transcript and waveform share one clock".into()),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(100) },
                Input::FeedEnd { stream: Ref::new(0) },
                Input::Advance { ms: Millis(2_000) },
                Input::Close { drain: false },
                Input::AwaitExit {},
            ] {
                run.apply(&input).await;
            }
            let report = run.session_report().await;
            (run, report)
        });
        assert_matches(&run);
        let origin = run
            .record
            .sink()
            .iter()
            .find_map(|(_, s)| match s {
                Sink::Recorder(RecorderCall::Origin { unix_micros, .. }) => {
                    Some(*unix_micros as f64 / 1e6)
                }
                _ => None,
            })
            .expect("the finalized file's own sample-zero timestamp");
        // The owner fills these from its recorder; exercise the real upload's conversion.
        report.audio_recording_started_at = Some(origin);
        report.audio_recording_path = Some("audio.ogg".into());
        report.started_at = Some(lk_agents_abi::unix_secs(lk_agents_abi::simulated_epoch()));
        let upload = lk_agents_core::telemetry::report::upload(
            &report,
            &RecordingOptions::default(),
            false,
            &CloudAgent::default(),
        )
        .unwrap();
        let header_ms = upload.header.start_time_millis as f64;
        assert!(
            header_ms - report.started_at.unwrap() * 1000.0 >= 399.0,
            "audio is anchored at sample zero, independently of session creation"
        );

        let heard = run.room_heard_audio();
        let (first, _) = heard.first().expect("the speaker played PCM");
        let expected = lk_agents_abi::unix_secs(lk_agents_abi::simulated_epoch())
            + first.saturating_duration_since(lk_agents_abi::Ts::ZERO).as_secs_f64();
        let message = report
            .chat_history
            .iter()
            .find_map(|item| match item {
                ChatItem::Message(message) if message.role == Role::Assistant => Some(message),
                _ => None,
            })
            .expect("the spoken message is in the final transcript");
        let started = message.metrics.started_speaking_at.expect("the transcript's spoken start");
        let span_start = run
            .record
            .host_view()
            .events()
            .iter()
            .find_map(|(_, event)| {
                let SessionEvent::Telemetry(records) = event else { return None };
                records.iter().find_map(|record| match record {
                    Record::Span(span) if span.name == AgentSpeaking::NAME => {
                        Some(span.start_unix_nanos as f64 / 1e9)
                    }
                    _ => None,
                })
            })
            .expect("the exported agent-speaking span");
        // The web maps all of these to the waveform by subtracting the uploaded origin.
        for at in [started, span_start] {
            let waveform_ms = at * 1000.0 - header_ms;
            let speaker_ms = expected * 1000.0 - header_ms;
            assert!(
                (waveform_ms - speaker_ms).abs() < 1.0,
                "exported {waveform_ms} ms, speaker {speaker_ms} ms"
            );
        }
    }
}

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

mod a_recorder_attached_while_a_segment_is_held {
    //! A segment that started on an output and is held there before it reports (the output
    //! replaced, then back) has played nothing: a recorder attached meanwhile records all of it,
    //! where the speaker reports it played.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

    // Catches: survey seed 591. The recorder attached seconds after the held segment started was
    // told it had played since then and given none of it, so the progress the speaker reported
    // from offset 0 placed silence where the person heard the audio.
    #[test]
    fn the_held_segment_is_recorded_where_it_plays() {
        let inputs = [
            Input::Say {
                text: AgentText("I'm sorry, I did not catch that at all.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(3_000) },
            Input::SetAudioOutput { attach: true },
            Input::Recorder { attach: true },
            Input::Advance { ms: Millis(1_000) },
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
        let progressed = run.record.sink().iter().any(|(_, sink)| {
            matches!(
                sink,
                lk_agents_fuzz::record::Sink::Recorder(
                    lk_agents_fuzz::record::RecorderCall::PlaybackProgressed
                )
            )
        });
        assert!(progressed, "the speaker reported the held segment playing");
    }
}

mod a_recorder_attached_while_a_segment_waits_for_audio {
    //! A segment whose speaker played all the audio it took, still open for the rest (a realtime
    //! model or an audio stream that has not sent it yet), goes on playing after a recorder
    //! attaches: the recording holds what follows, where the speaker reports it played.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};
    use lk_agents_fuzz::record::{RecorderCall, Sink};

    // Catches: survey seeds 123 and 776. The attach skipped the segment as played to its end, so
    // the recorder never joined it: the speaker's later reports, at the segment's own offsets,
    // placed nothing, and the recording was silent where the person heard the rest.
    #[test]
    fn the_rest_of_the_segment_is_recorded() {
        let stream = Ref::new(0);
        let segment = Ref::new(0);
        let inputs = [
            Input::Say {
                text: AgentText("Let me check that for you.".into()),
                input: SayInput::Audio,
                allow: None,
                to_ctx: true,
            },
            Input::FeedAudio { stream, chunks: Count(20) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment },
            // All it took played; the segment waits for the rest.
            Input::PlayoutProgressed { segment, ms: Millis(60_000) },
            Input::Advance { ms: Millis(3_000) },
            Input::Recorder { attach: true },
            Input::Advance { ms: Millis(100) },
            Input::FeedAudio { stream, chunks: Count(40) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutProgressed { segment, ms: Millis(60_000) },
            Input::Advance { ms: Millis(3_000) },
            Input::FeedEnd { stream },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutFinished { segment },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let told = run
            .record
            .sink()
            .iter()
            .any(|(_, s)| *s == Sink::Recorder(RecorderCall::PlaybackProgressed));
        assert!(told, "the recorder was told the rest played");
        let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
        let unheard: Vec<_> =
            failures.iter().filter(|f| f.contract == "recording_is_what_played").collect();
        assert!(unheard.is_empty(), "{unheard:?}");
    }
}
