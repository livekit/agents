//! Realtime turns regression tests.

mod realtime_generation_without_message {
    //! A realtime generation that ends before any message still ends its speech.
    use lk_agents_fuzz::drive::{NodeSetup, Outcome, RealtimeSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
    use lk_agents_voice::events::SessionEvent;
    use lk_agents_voice::trace::Step;

    // Catches: the reply's binding opened the agent's `transcription_node` at activation, and a
    // generation the socket's drop cancelled before its first message never flushed it, so the step
    // waited on the node's drain until the session closed (`speech_settles`, survey seed 6784).
    #[test]
    fn a_generation_cancelled_before_its_first_message_ends_the_speech() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(0),
                ..RealtimeSetup::default()
            }),
            nodes: Some(NodeSetup {
                llm: false,
                tts: false,
                stt: false,
                llm_text: None,
                transcription: true,
            }),
            ..Setup::default()
        };
        let rt = |event| Input::RealtimeEvent { socket: Ref::new(0), event };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                // The server has opened the generation and sent no message yet.
                Input::Advance { ms: Millis(300) },
                rt(RtEvent::Disconnected),
                Input::Advance { ms: Millis(800) },
                rt(RtEvent::Reconnected),
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            let journal = run.record.journal();
            for (at, step) in journal.iter() {
                eprintln!("J {} {:?}", at.at_ms, step);
            }
            let done = journal
                .iter()
                .any(|(_, step)| matches!(step, Step::Emitted(SessionEvent::SpeechDone { .. })));
            assert!(done, "the speech whose generation was cancelled is done before the close");
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod realtime_onset_reply_heard {
    //! The reply to a turn the realtime server heard over a playing reply is heard.
    use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};

    // Catches: c6881abb9's silence, as the person on the call met it (`speech_is_heard`). The
    // server's onset over a reply still playing (its generation long done) paused the reply instead
    // of interrupting it; the person's turn then interrupted the paused reply, and that interrupt's
    // cancel reached the server after it had created its reply to the turn: the new response was
    // cancelled before a word of it played, and the agent said nothing.
    #[test]
    fn the_reply_to_a_turn_over_a_playing_reply_is_heard() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
            pausable_output: true,
            ..Setup::default()
        };
        let inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            // Past the echo canceller's warm-up; the person talks over the reply, then stops.
            Input::Advance { ms: Millis(4_000) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStarted },
            Input::Advance { ms: Millis(1_000) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStopped },
            Input::Advance { ms: Millis(20_000) },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"speech_is_heard"), "{:#?}", outcome.failures);
        assert!(outcome.expectations.get("speech_is_heard").is_some_and(|n| *n > 0));
    }
}

mod realtime_outputs_before_next_reply {
    //! A realtime tool's output reaches the server with its commit, before anything the commit lets
    //! go asks the server for a reply.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{RealtimeStep, Step, ToolStep};

    // Catches: the outputs pushed to the server only on the step's later wake, after the commit had
    // already let a waiting reply go: a background tool's deferred reply, held while another
    // speech's tools ran, was asked for (`response.create`) before that speech's tool output reached
    // the server, and read the call without its result (`realtime_reply_reads_tool_output`, campaign
    // seed 1978). The outputs now go with the commit, ahead of everything it releases.
    #[test]
    fn the_output_goes_with_the_commit() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(10_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("shipped".into()),
            },
            Input::Advance { ms: Millis(10_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        });
        let journal = run.record.journal();
        let position = |pick: &dyn Fn(&Step) -> bool| journal.iter().position(|(_, s)| pick(s));
        let pushed = position(&|s| {
            matches!(
                s,
                Step::Realtime(RealtimeStep::Sent {
                    kind: lk_agents_abi::RealtimeCommandKind::PushItem,
                    ..
                })
            )
        });
        let finished = position(&|s| matches!(s, Step::Tool(ToolStep::Finished { .. })));
        let (Some(pushed), Some(finished)) = (pushed, finished) else {
            panic!("the tool's output pushed ({pushed:?}) and committed ({finished:?})")
        };
        assert!(pushed < finished, "pushed at step {pushed}, after the commit at step {finished}");
    }
}

mod realtime_reply_answered_closing {
    //! A realtime reply the server answers while the session closes is cancelled, not left generating.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};

    // Catches: a turn committed while the session was closing asked the server for a reply, and the
    // answer was discarded (nothing plays it) without telling the server to stop: still generating,
    // it refused the next reply asked meanwhile (`conversation_already_has_active_response`,
    // `realtime_server_accepts_requests`, survey seed 8349). The discarded answer is now cancelled.
    #[test]
    fn a_reply_answered_while_closing_frees_the_server() {
        let setup = Setup {
            hook_bound_ms: Some(2_500),
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(0),
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let inputs = [
            // The agent's `on_enter` is still running, so the close takes its bound.
            Input::Close { drain: false },
            Input::Advance { ms: Millis(100) },
            Input::CommitTurn { skip_reply: false },
            Input::Advance { ms: Millis(500) },
            Input::CommitTurn { skip_reply: false },
            Input::Advance { ms: Millis(2_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let (failures, _) = check_all(&run.record);
        let refused: Vec<_> = failures
            .iter()
            .filter(|f| f.contract == "realtime_server_accepts_requests")
            .map(|f| f.detail.as_str())
            .collect();
        assert!(refused.is_empty(), "{refused:?}");
    }

    // Catches: a reply the server opened on its own (its idle timeout) while the session closed was
    // discarded and left generating, as only a reply the client had asked for was cancelled: the
    // turn committed next asked over it and was refused (`conversation_already_has_active_response`,
    // `realtime_server_accepts_requests`, survey seed 104301).
    #[test]
    fn a_server_reply_opened_while_closing_frees_the_server() {
        let setup = Setup {
            hook_bound_ms: Some(2_500),
            realtime: Some(RealtimeSetup {
                turn_detection: true,
                tool_calls: Some(0),
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let inputs = [
            Input::Close { drain: false },
            Input::Advance { ms: Millis(100) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
            Input::Advance { ms: Millis(100) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(5_000),
                stt_flush: Millis(120),
                skip_reply: false,
            },
            Input::Advance { ms: Millis(2_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let accepted = run.record.applied().iter().filter(|a| a.accepted).count();
        assert_eq!(accepted, inputs.len(), "every input reached the session");
        let (failures, _) = check_all(&run.record);
        let refused: Vec<_> = failures
            .iter()
            .filter(|f| f.contract == "realtime_server_accepts_requests")
            .map(|f| f.detail.as_str())
            .collect();
        assert!(refused.is_empty(), "{refused:?}");
    }
}

mod realtime_reply_asked_when_authorized {
    //! A realtime reply is announced when asked for, and asks the server once its speech holds the
    //! floor.
    use lk_agents_abi::RealtimeCommandKind;
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
    use lk_agents_voice::trace::{RealtimeStep, SpeechStep, Step};

    // Catches: the request going to the server at once and the speech created only when the server
    // answered, so `speech_created` came after the model call and a reply queued behind a playing one
    // made the server generate over it. The reference schedules the handle at `generate_reply()` and
    // its `_realtime_reply_task` calls `rt_session.generate_reply()` once the speech is authorized.
    #[test]
    fn a_queued_realtime_reply_asks_the_server_once_authorized() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let reply = || Input::GenerateReply { user_input: None, instructions: None, allow: None };
        let inputs = [
            reply(),
            Input::Advance { ms: Millis(20) },
            reply(),
            Input::Advance { ms: Millis(20) },
        ];
        let done = [
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(2_000) },
        ];
        let count = |run: &Run, what: fn(&Step) -> bool| {
            run.record.journal().iter().filter(|(_, step)| what(step)).count()
        };
        let asked = |step: &Step| {
            matches!(
                step,
                Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::GenerateReply, .. })
            )
        };
        let created = |step: &Step| matches!(step, Step::Speech(SpeechStep::ReplyCreated { .. }));
        let (queued, played) = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            let queued = (count(&run, created), count(&run, asked));
            for input in &done {
                run.apply(input).await;
            }
            (queued, count(&run, asked))
        });
        assert_eq!(queued, (2, 1), "both replies announced, only the playing one asked");
        assert_eq!(played, 2, "the second is asked once the first is done");
    }
}

mod realtime_reply_ends_after_its_playout {
    //! A realtime reply whose audio finished playing before the server ended its generation is done,
    //! and binds the output only for the message it sent.
    use lk_agents_fuzz::drive::{NodeSetup, RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::fakes::RealtimeDelays;
    use lk_agents_fuzz::input::{Input, Millis, Ref};
    use lk_agents_fuzz::record::Sink;
    use lk_agents_voice::events::SessionEvent;

    // Catches: the reply held running until the session closed when the server's `response.done`
    // came after the speaker reported its last message played: a binding that carried no message
    // never flushed its transcription node, and its drain never ended (5574d1ccc, bc0b5de0c).
    #[test]
    fn the_reply_is_done_once_the_server_ends_it() {
        let setup = Setup {
            nodes: Some(NodeSetup {
                llm: false,
                tts: false,
                stt: false,
                llm_text: None,
                transcription: true,
            }),
            // A server whose reply calls no tool and ends 30 s after its last audio.
            realtime: Some(RealtimeSetup {
                tool_calls: Some(0),
                delays: RealtimeDelays { done: Some(30_000), ..RealtimeDelays::default() },
                ..RealtimeSetup::default()
            }),
            sync_transcription: true,
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            // The reply's last message is in the sink; the server has not yet said it is done.
            Input::Advance { ms: Millis(20_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(20_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let done = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter(|(_, ev)| matches!(ev, SessionEvent::SpeechDone { .. }))
            .count();
        assert_eq!(done, 1);
    }

    // Catches: after the speaker reported the reply's one message played, the generation bound the
    // output again for a next message before the server said whether one would come. None came, and
    // that binding's transcript chain ended a segment the server never sent: the transcription output
    // was flushed twice for one message (the reference binds the output once per message).
    #[test]
    fn the_transcript_ends_once_per_message() {
        let setup = Setup {
            nodes: Some(NodeSetup {
                llm: false,
                tts: false,
                stt: false,
                llm_text: None,
                transcription: true,
            }),
            realtime: Some(RealtimeSetup {
                tool_calls: Some(0),
                delays: RealtimeDelays { done: Some(30_000), ..RealtimeDelays::default() },
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(40_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let done = run
            .record
            .host_view()
            .events()
            .into_iter()
            .any(|(_, ev)| matches!(ev, SessionEvent::SpeechDone { .. }));
        assert!(done, "the scenario: the server ended the reply");
        let flushes = run
            .record
            .sink()
            .iter()
            .filter(|(_, sink)| matches!(sink, Sink::TextFlush { .. }))
            .count();
        assert_eq!(flushes, 1, "one message, one transcript segment");
    }
}

mod realtime_reply_queued {
    //! A reply asked while the realtime server generates one of its own waits for it, and is heard.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RtEvent, ToolAnswer};
    use lk_agents_fuzz::record::{Breach, Recording, Tag};

    fn realtime(setup: Setup, inputs: &[Input]) -> Recording {
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in inputs {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        });
        run.record
    }

    fn refused(r: &Recording) -> Vec<&Breach> {
        let breaches = r.inference_view().realtime_breaches();
        breaches.iter().map(|(_, _, b)| b).filter(|b| matches!(b, Breach::Refused { .. })).collect()
    }

    /// Whether the n-th realtime generation's audio reached the speaker.
    fn heard(r: &Recording, n: u32) -> bool {
        r.host_view().audio_pushed().iter().any(|p| {
            p.tags
                .iter()
                .any(|t| matches!(t, Tag::Generated { generation, .. } if *generation == n))
        })
    }

    // Catches: a host `generate_reply` whose `response.create` crossed a reply the server started on
    // its own: refused (`conversation_already_has_active_response`), and the host's reply was lost.
    // It waits for the server's reply and goes after it, one at a time, as speeches are scheduled.
    #[test]
    fn a_reply_crossing_the_servers_own_waits_and_is_heard() {
        // Only a server with its own turn detection opens a reply unasked (its idle timeout).
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let r = realtime(
            setup,
            &[
                // The server opens a reply of its own as the host asks for one.
                Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(3_000) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::PlayoutFinished { segment: Ref::new(0) },
                Input::Advance { ms: Millis(10_000) },
            ],
        );
        let failed: Vec<_> = r
            .host_view()
            .speeches()
            .into_iter()
            .filter_map(|s| s.error().map(str::to_owned))
            .collect();
        assert!(failed.is_empty(), "{failed:?}");
        assert!(heard(&r, 1), "the host's reply, asked again after the server's, is heard");
    }

    // Catches: the tools' follow-up asked over a reply the server was still generating (one the host
    // asked for while the tool ran): refused (`conversation_already_has_active_response`), the tools'
    // result never spoken. It waits for that reply to be done, which is not cut.
    #[test]
    fn a_tool_follow_up_waits_for_the_reply_generating() {
        let setup = Setup {
            tools: vec!["lookup_order".into()],
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            ..Setup::default()
        };
        let r = realtime(
            setup,
            &[
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(5_000) },
                // The reply's words play out; the tool still runs, and the host's reply goes.
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::PlayoutFinished { segment: Ref::new(0) },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(600) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::Value,
                    text: AgentText("shipped".into()),
                },
                Input::Advance { ms: Millis(5_000) },
            ],
        );
        assert!(refused(&r).is_empty(), "{:?}", refused(&r));
        // The host's reply (the second generation) played on; the follow-up came after it.
        let cut = r.journal().iter().any(|(_, s)| {
            matches!(
                s,
                lk_agents_voice::trace::Step::Realtime(
                    lk_agents_voice::trace::RealtimeStep::Sent {
                        kind: lk_agents_abi::RealtimeCommandKind::Interrupt,
                        ..
                    }
                )
            )
        });
        assert!(!cut && heard(&r, 1), "the host's reply is not cut");
    }
}

mod realtime_server_onset_interrupts {
    //! The realtime server hearing the user start speaking interrupts the reply it is playing.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};
    use lk_agents_voice::trace::{GateOp, InterruptCause, InterruptStep, Step};

    // Catches: the server's `input_speech_started` taking the guarded interruption by audio activity,
    // which pauses a pausable sink instead of interrupting. On a real call the reply then sat paused
    // after its generation had ended, no false-interruption timer ended the pause, and the next
    // server turn interrupted it then, so its cancel landed on that new response and the agent went
    // silent. The reference's `_on_input_speech_started` calls `interrupt(source="audio_activity")`.
    #[test]
    fn a_server_speech_onset_interrupts_the_playing_reply() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
            pausable_output: true,
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            // Past the echo canceller's warm-up, which holds interruptions back.
            Input::Advance { ms: Millis(4_000) },
            // The server's VAD hears only the audio it was sent.
            Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
            Input::Advance { ms: Millis(100) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStarted },
            Input::Advance { ms: Millis(20) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let interrupted = journal.iter().any(|(_, step)| {
            matches!(
                step,
                Step::Interrupt(InterruptStep::Interrupted {
                    cause: InterruptCause::ByAudioActivity,
                    ..
                })
            )
        });
        let paused = journal.iter().any(|(_, step)| {
            matches!(step, Step::Interrupt(InterruptStep::PlayoutGate { op: GateOp::Pause, .. }))
        });
        assert!(interrupted, "the reply is interrupted");
        assert!(!paused, "not paused");
    }
}

mod realtime_transcript_forwarded {
    //! A realtime generation's words reach the transcription output.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AGENT_TURNS, Input, Millis, Ref, RtEvent};
    use lk_agents_fuzz::record::Sink;

    // Catches: the realtime step recording the server's text as the transcript but never forwarding
    // it, so a transcription output showed nothing of anything a realtime model said
    // (`text_conservation`, survey seed 35). `_realtime_generation_task` forwards each message's text
    // to `text_output` through `forward_generation`.
    #[test]
    fn a_realtime_reply_is_shown() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let shown: String = run
            .record
            .sink()
            .iter()
            .filter_map(|(_, s)| match s {
                Sink::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(AGENT_TURNS.contains(&shown.as_str()) && !shown.is_empty(), "{shown:?}");
        let flushed = run.record.sink().iter().any(|(_, s)| matches!(s, Sink::TextFlush { .. }));
        assert!(flushed, "the message's segment was closed");
    }
}

mod realtime_voiced_text_retries_told {
    //! A text-only realtime reply spoken by a voice that does not stream tells each retry of a
    //! sentence, as a pipeline reply does.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis};
    use lk_agents_voice::events::{ErrorSource, SessionEvent};

    // Catches: the realtime reply opened its TTS stream without the adapter's retry report, so a
    // sentence retried while the voice was down was not the session's recoverable error and its wait
    // was not journalled: the reply failed four seconds after the run looked quiet
    // (`speech_settles`, survey seed 3409).
    #[test]
    fn each_retry_of_a_sentence_is_a_recoverable_error() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                text_only: true,
                tool_calls: Some(0),
                ..RealtimeSetup::default()
            }),
            tts: true,
            chunked_tts: true,
            ..Setup::default()
        };
        let inputs = [
            Input::TtsOutage { down: true },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(30_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let events = run.record.host_view().events();
        let retried = events
            .iter()
            .filter(|(_, ev)| {
                matches!(ev, SessionEvent::Error(e) if e.source == ErrorSource::Tts && e.recoverable)
            })
            .count();
        let failed = events.iter().any(|(_, ev)| {
            matches!(ev, SessionEvent::Error(e) if e.source == ErrorSource::Tts && !e.recoverable)
        });
        assert!(failed, "the scenario: the voice stays down, the reply's synthesis fails");
        assert_eq!(retried, 3, "each retry of the sentence is told");
    }
}
