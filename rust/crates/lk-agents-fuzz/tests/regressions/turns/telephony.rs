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
    // 2112).
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
    // survey seeds 320, 621, 731).
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

mod dtmf_debounce_follows_user_state {
    use lk_agents_fuzz::drive::{RoomSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Count, DtmfInput, DtmfKey, Input, Millis, RoomIn};
    use lk_agents_fuzz::record::{Answer, Recording};

    /// The user speaking holds back a pending DTMF collection until the agent listens again.
    /// https://github.com/livekit/agents/pull/5900
    #[test]
    fn user_speech_postpones_dtmf_collection_until_listening_resumes() {
        let dtmf = |input| Input::Dtmf {
            input,
            key: DtmfKey::Two,
            digits: Count(0),
            confirm: false,
            timeout: Millis(300),
        };
        let setup = Setup {
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            ..Setup::default()
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Room { event: RoomIn::Linked },
                Input::Advance { ms: Millis(100) },
                dtmf(DtmfInput::Start),
                Input::Advance { ms: Millis(10) },
                dtmf(DtmfInput::Digit),
                Input::Advance { ms: Millis(100) },
                Input::ClaimUserTurn {},
                Input::Advance { ms: Millis(1_000) },
            ] {
                run.apply(&input).await;
            }
            let answered = |record: &Recording| {
                record
                    .host_view()
                    .api_calls()
                    .iter()
                    .filter(|call| matches!(call.returned(), Some((_, Answer::Dtmf(_)))))
                    .count()
            };
            assert_eq!(
                answered(&run.record),
                0,
                "a user speaking during debounce must postpone collection"
            );
            run.apply(&Input::ReleaseUserTurn {}).await;
            run.apply(&Input::Advance { ms: Millis(400) }).await;
            assert_eq!(answered(&run.record), 1, "listening must rearm the debounce");
        });
    }
}
