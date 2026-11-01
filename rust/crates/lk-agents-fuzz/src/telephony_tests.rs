//! Telephony regression tests.

mod amd_awaits_the_callee {
    //! AMD's clock starts when the callee answers in the room: a ringing SIP call is not timed, and a
    //! callee whose audio never comes, or who hangs up first, is `participant_missing`.
    use lk_agents_abi::voice::amd::{AmdCategory, AmdReason};
    use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, RoomSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AmdInput, Input, Millis, RoomIn};
    use lk_agents_voice::events::SessionEvent;

    /// AMD's verdict after `inputs`, if any.
    fn verdict(answer: AmdAnswer, inputs: &[Input]) -> Option<(AmdCategory, AmdReason)> {
        let setup = Setup {
            amd: Some(AmdSetup { answer, ..AmdSetup::default() }),
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in inputs {
                run.apply(input).await;
            }
            run
        });
        run.record.host_view().events().into_iter().find_map(|(_, ev)| match ev {
            SessionEvent::AmdResult { verdict } => Some((verdict.category, verdict.reason)),
            _ => None,
        })
    }

    fn advance(ms: u64) -> Input {
        Input::Advance { ms: Millis(ms) }
    }

    const NO_SPEECH: Option<(AmdCategory, AmdReason)> =
        Some((AmdCategory::Uncertain, AmdReason::NoSpeechTimeout));
    const MISSING: Option<(AmdCategory, AmdReason)> =
        Some((AmdCategory::Uncertain, AmdReason::ParticipantMissing));

    // Catches: listening before the callee's audio is there, or the no-speech timer not armed once
    // it is: their track opens listening at once.
    #[test]
    fn the_callees_audio_starts_listening() {
        assert_eq!(verdict(AmdAnswer::Track, &[advance(9_900)]), None);
        assert_eq!(verdict(AmdAnswer::Track, &[advance(10_100)]), NO_SPEECH);
    }

    // Catches: a wait for the track with no bound (`_TRACK_PUBLICATION_TIMEOUT`, 5 s), or the
    // callee joining in time not heard.
    #[test]
    fn a_callee_whose_audio_never_comes_is_missing() {
        assert_eq!(verdict(AmdAnswer::Missing, &[advance(4_900)]), None);
        assert_eq!(verdict(AmdAnswer::Missing, &[advance(5_100)]), MISSING);
        let joined = [advance(1_000), Input::Room { event: RoomIn::Linked }, advance(10_100)];
        assert_eq!(verdict(AmdAnswer::Missing, &joined), NO_SPEECH);
    }

    // Catches: a SIP call timed while it rings (its track is there before the callee answers), or
    // its answer not opening listening.
    #[test]
    fn a_ringing_sip_call_is_not_timed() {
        assert_eq!(verdict(AmdAnswer::Sip, &[advance(30_000)]), None);
        let answered = [advance(30_000), Input::Amd { input: AmdInput::Answer }, advance(10_100)];
        assert_eq!(verdict(AmdAnswer::Sip, &answered), NO_SPEECH);
    }

    // Catches: a SIP wait that outlives the call it waits on.
    #[test]
    fn a_sip_call_that_ends_unanswered_is_missing() {
        let hung_up = [advance(1_000), Input::Room { event: RoomIn::Unlinked }, advance(20)];
        assert_eq!(verdict(AmdAnswer::Sip, &hung_up), MISSING);
    }
}

mod amd_consumes_turns {
    //! Once AMD has called the line a machine, the turns it hears are its own, not the agent's.
    use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AmdInput, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    /// One spoken turn on utterance `nth`: onset, a final, the end of speech.
    fn turn(nth: u32, text: &str) -> Vec<Input> {
        vec![
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(nth),
                text: UserText(text.into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(nth) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(20) },
        ]
    }

    /// Whether AMD's verdict was a machine, and the speeches the session created after it.
    fn after_verdict(interrupt_on_machine: bool, close: bool) -> (bool, usize) {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: Vec::new(),
            amd: Some(AmdSetup {
                interrupt_on_machine,
                answer: AmdAnswer::Track,
                ..AmdSetup::default()
            }),
            ..Setup::default()
        };
        let mut inputs = turn(0, "hi you have reached the voicemail of theo");
        // The classifier's request, answered as a model answers it.
        inputs.extend([
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "save_prediction".into(),
                args: ToolArgs(r#"{"label":"machine-vm"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(3000) },
            Input::Amd { input: AmdInput::Execute },
        ]);
        if close {
            inputs.push(Input::Amd { input: AmdInput::Close });
        }
        inputs.extend(turn(1, "please leave a message after the tone"));
        inputs.push(Input::Advance { ms: Millis(3000) });
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let events = run.record.host_view().events();
        let verdict = events
            .iter()
            .position(|(_, ev)| matches!(ev, SessionEvent::AmdResult { .. }))
            .expect("AMD gave a verdict");
        let machine = matches!(events[verdict].1, SessionEvent::AmdResult { verdict } if verdict.is_machine());
        let created = events[verdict..]
            .iter()
            .filter(|(_, ev)| matches!(ev, SessionEvent::SpeechCreated { .. }))
            .count();
        (machine, created)
    }

    // Catches: a turn heard after a machine verdict answered by the agent. `AMD._on_end_of_turn`
    // has the activity skip the reply (`info.skip_reply`) while the detector holds a machine verdict
    // under `interrupt_on_machine`: the developer is leaving a voicemail, and an auto reply to the
    // greeting's tail would talk over it.
    #[test]
    fn a_machine_verdict_skips_the_replies() {
        assert_eq!(after_verdict(true, false), (true, 0));
    }

    // The skip is the detector's: without `interrupt_on_machine`, or once it is closed, turns are
    // answered again.
    #[test]
    fn replies_resume_without_the_detector() {
        assert_eq!(after_verdict(false, false), (true, 1));
        assert_eq!(after_verdict(true, true), (true, 1));
    }
}

mod amd_releasing_turn_no_reply {
    //! The turn whose end releases AMD's machine verdict gets no reply, not one created and then cut.
    use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AmdInput, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    // Catches: the session decided the turn before AMD heard its end, so the turn that released a
    // machine verdict under `interrupt_on_machine` still created a reply, which AMD's interrupt then
    // cut. The reference's `on_end_of_turn` asks `amd._on_end_of_turn(info)` first, which forwards the
    // end of turn to the classifier and, the verdict now out, skips the reply.
    #[test]
    fn the_releasing_turn_gets_no_reply() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: Vec::new(),
            amd: Some(AmdSetup {
                interrupt_on_machine: true,
                answer: AmdAnswer::Track,
                ..AmdSetup::default()
            }),
            ..Setup::default()
        };
        let inputs = [
            // A long greeting: a machine's silence threshold applies.
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(3000) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("hi you have reached the voicemail of theo".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(20) },
            // The classifier answers while the greeting's turn is still open.
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "save_prediction".into(),
                args: ToolArgs(r#"{"label":"machine-vm"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
            // Past the machine silence threshold: the verdict now waits on the end of turn alone.
            Input::Advance { ms: Millis(2000) },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(3000) },
            Input::Amd { input: AmdInput::Execute },
            Input::Advance { ms: Millis(1000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let events = run.record.host_view().events();
        let machine = events.iter().any(
            |(_, ev)| matches!(ev, SessionEvent::AmdResult { verdict } if verdict.is_machine()),
        );
        let created = events
            .iter()
            .filter(|(_, ev)| matches!(ev, SessionEvent::SpeechCreated { .. }))
            .count();
        assert_eq!((machine, created), (true, 0));
    }
}

mod amd_tap_opened_late {
    //! AMD's own recognition stream, landing after the detector stopped reading, is closed.
    use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Case, Setup, run};
    use lk_agents_fuzz::input::{Input, Millis};

    fn closes_its_stream(answer: AmdAnswer, before_close: Millis) {
        let setup = Setup {
            connect_ms: Some(15_000),
            amd: Some(AmdSetup { own_stt: true, answer, ..AmdSetup::default() }),
            ..Setup::default()
        };
        let inputs = vec![Input::Advance { ms: before_close }, Input::Close { drain: false }];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
    }

    // Catches: the stream `_run_stt` opened for AMD landing after the session closed (the detector's
    // loop gone), dropped with its message and never closed (`provider_streams_closed`, survey seed
    // 25): a host stream is not cancelled by a drop.
    #[test]
    fn a_tap_opened_after_the_close_is_closed() {
        closes_its_stream(AmdAnswer::Track, Millis(20));
    }

    // The same with the verdict already out (`participant_missing`, the callee's track never came)
    // when the stream lands.
    #[test]
    fn a_tap_opened_after_the_verdict_is_closed() {
        closes_its_stream(AmdAnswer::Missing, Millis(6_000));
    }
}

mod amd_tap_queued_at_close {
    //! AMD's own recognition stream, opened while the detector was closing, is closed.
    use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Case, Setup, run};
    use lk_agents_fuzz::input::{AmdInput, Input, Millis};

    // Catches: the stream `_run_stt` opened for AMD queued on the detector's channel as `AMD.aclose()`
    // ended its loop: dropped with the channel, never handled and never closed
    // (`provider_streams_closed`, survey seed 7022): a host stream is not cancelled by a drop.
    #[test]
    fn a_tap_queued_as_the_detector_closes_is_closed() {
        let setup = Setup {
            amd: Some(AmdSetup { own_stt: true, answer: AmdAnswer::Track, ..AmdSetup::default() }),
            ..Setup::default()
        };
        let inputs = vec![
            Input::Amd { input: AmdInput::Close },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: false },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
    }
}

mod dtmf_close {
    //! A close ends a DTMF collection as it begins, before its drain.
    use lk_agents_fuzz::drive::{Case, RoomSetup, Setup, run};
    use lk_agents_fuzz::input::{
        AgentText, Count, DtmfInput, DtmfKey, Input, Millis, RoomIn, SayInput,
    };

    fn dtmf(input: DtmfInput, key: DtmfKey) -> Input {
        Input::Dtmf { input, key, digits: Count(0), confirm: false, timeout: Millis(120) }
    }

    // Catches: the collection kept listening until the close's hard edge, so a key pressed while the
    // close drained completed it with a digit pressed after the session began closing (survey seed
    // 2112). `GetDtmfTask.on_exit`, which the close runs first, stops listening.
    #[test]
    fn a_key_pressed_while_closing_is_not_collected() {
        let inputs = vec![
            Input::Room { event: RoomIn::Linked },
            Input::Advance { ms: Millis(20) },
            dtmf(DtmfInput::Start, DtmfKey::Pound),
            // A speech the close's drain waits on, so the session is still up when the key comes.
            Input::Say {
                text: AgentText("Your balance is due on the fourteenth of next month.".into()),
                input: SayInput::Words,
                allow: Some(false),
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: true },
            dtmf(DtmfInput::Digit, DtmfKey::Two),
            Input::Advance { ms: Millis(1_000) },
        ];
        // The caller presses the key in the room.
        let room = Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false });
        let outcome =
            run(Case { setup: Setup { room, ..Setup::default() }, inputs, ..Case::default() });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }
}

mod dtmf_read_back_realtime {
    //! A DTMF read-back on a realtime model is answered on its socket.
    use lk_agents_fuzz::drive::{Case, RealtimeSetup, RoomSetup, Setup, run};
    use lk_agents_fuzz::input::{Count, DtmfInput, DtmfKey, Input, Millis, RoomIn};

    // Catches: the confirm arm's read-back built a pipeline reply on a realtime session, which has no
    // LLM, so it failed with "the session has no llm" though nothing failed (`speech_error_is_injected`,
    // survey seeds 320, 621, 731). `GetDtmfTask` calls `session.generate_reply()`, which a realtime
    // model answers on its own socket.
    #[test]
    fn a_read_back_on_a_realtime_model_is_spoken() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            // The caller presses the key in the room.
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            ..Setup::default()
        };
        // One digit, read back for confirmation.
        let dtmf = |input, key| Input::Dtmf {
            input,
            key,
            digits: Count(0),
            confirm: true,
            timeout: Millis(120),
        };
        let inputs = vec![
            Input::Room { event: RoomIn::Linked },
            Input::Advance { ms: Millis(20) },
            dtmf(DtmfInput::Start, DtmfKey::Pound),
            Input::Advance { ms: Millis(20) },
            dtmf(DtmfInput::Digit, DtmfKey::Two),
            Input::Advance { ms: Millis(1_000) },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        assert!(outcome.ok(), "{:#?}", outcome.failures);
        let spoke = outcome.reached.iter().any(|v| v == "Realtime::GenerationStatus");
        assert!(spoke, "no realtime generation answered the read-back");
    }
}
