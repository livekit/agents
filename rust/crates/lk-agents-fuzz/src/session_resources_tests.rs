//! Lifecycle regression tests.

mod live_options {
    //! A live option change reaches the next reply.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{
        AgentText, Count, Input, Knob, Millis, Ref, ToolAnswer, ToolArgs, TurnMode,
    };

    /// One tool step under `max_tool_steps = 0`, with the limit raised to `raise` first when given.
    fn tool_step(raise: Option<u32>) -> Case {
        let mut inputs = Vec::new();
        if let Some(steps) = raise {
            inputs.push(Input::Configure {
                knob: Knob::MaxToolSteps,
                mode: TurnMode::Default,
                ms: Millis(0),
                on: false,
                words: Count(steps),
            });
        }
        inputs.extend([
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("ok".into()),
            },
            Input::Advance { ms: Millis(200) },
        ]);
        Case {
            setup: Setup { max_tool_steps: Some(0), ..Setup::default() },
            inputs,
            ..Case::default()
        }
    }

    fn limit_reached(case: Case) -> bool {
        run(case).reached.iter().any(|v| v == "Core::PortFailure")
    }

    // Catches: replies reading the options as of the session's start, so `update_options(
    // max_tool_steps=)` never reaches a reply.
    #[test]
    fn a_raised_max_tool_steps_reaches_the_next_reply() {
        assert!(
            limit_reached(tool_step(None)),
            "the step limit of 0 was never hit: the probe is blind"
        );
        assert!(!limit_reached(tool_step(Some(3))), "the raised limit did not reach the reply");
    }
}

mod path_needs_both_switches {
    //! A leg's path is open only while its enable and its attachment both are.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_voice::agent_session::IoKind;

    // Catches: the harness reading a leg's two independent switches as one, so enabling the
    // transcription leg after detaching its output read as the path reopening. A speech spoken with
    // no text output then looked like it lost its words (`text_conservation`, and the same misread of
    // the audio input behind most `forwarded_while_enabled` findings).
    #[test]
    fn enabling_a_detached_leg_does_not_reopen_it() {
        let inputs = [
            Input::SetTranscriptionOutput { attach: false },
            Input::SetIoEnabled { leg: IoKind::Transcription, enabled: true },
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::Say {
                text: AgentText("Hello.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let r = &run.record;
        assert!(!r.inputs_view().transcription_output().on_at(r.now()));
        let speeches = r.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.completed_normally());
        let (failures, _) = check_all(r);
        let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"text_conservation"), "{failed:?}");
    }
}

mod pause_outlives_its_speech {
    //! A pause taken on the room's speaker never outlives the speech it was taken for.
    use lk_agents_fuzz::drive::{Case, RoomLegs, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Confidence, Count, Input, Millis, Ref, SayInput};

    fn say(input: SayInput) -> Input {
        Input::Say { text: AgentText("Second.".into()), input, allow: None, to_ctx: true }
    }

    /// The person starts talking over a reply still thinking, which pauses the speaker.
    fn talk_over() -> Vec<Input> {
        vec![
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(90),
                speaking: true,
                duration: Millis(800),
            },
            Input::Advance { ms: Millis(50) },
        ]
    }

    /// Every speech settles: none waits on a speaker left paused.
    fn settles(inputs: Vec<Input>) {
        let setup = Setup {
            vad: true,
            sync_transcription: true,
            pausable_output: true,
            room_legs: Some(RoomLegs { pre_connect: None }),
            ..Setup::default()
        };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"speech_settles"), "{failed:?}");
        assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
    }

    // Catches: the paused reply interrupted and a queued say taking the floor; the pause was released
    // only for a paused speech still current when it ended, and the floor leaves a speech at its last
    // step, so the speaker stayed paused and every later speech waited until the close
    // (`speech_settles`, survey seed 4919). The host's fake output only records a pause, so only the
    // room's own leg showed it.
    #[test]
    fn the_queued_speech_plays_after_a_paused_one_is_interrupted() {
        let mut inputs = vec![
            Input::Advance { ms: Millis(100) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            say(SayInput::Words),
            Input::Advance { ms: Millis(10) },
        ];
        inputs.extend(talk_over());
        inputs.extend([
            Input::InterruptSpeech { speech: Ref::new(0), force: true },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(100) },
        ]);
        settles(inputs);
    }

    // Catches: the same pause outliving its interrupted speech with nothing queued, so the next say,
    // seconds later, played into a paused speaker (survey seed 4080).
    #[test]
    fn a_later_speech_plays_after_a_paused_one_is_interrupted() {
        let mut inputs = vec![
            Input::Advance { ms: Millis(100) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(10) },
        ];
        inputs.extend(talk_over());
        inputs.extend([
            Input::InterruptSpeech { speech: Ref::new(0), force: true },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3000) },
            say(SayInput::Words),
            Input::Advance { ms: Millis(100) },
        ]);
        settles(inputs);
    }

    // Catches: the host attaching another audio output while a speech was paused: every later gate
    // went to the new output, the speech kept the room's speaker it started on, and nothing resumed
    // it, so it played nothing until the close (survey seed 2042).
    #[test]
    fn a_paused_speech_plays_on_after_the_output_is_replaced() {
        let mut inputs = vec![Input::Advance { ms: Millis(100) }, say(SayInput::Audio)];
        inputs.extend(talk_over());
        inputs.extend([
            Input::SetAudioOutput { attach: true },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(100) },
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(2) },
            Input::FeedEnd { stream: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ]);
        settles(inputs);
    }
}

mod paused_sink_released_when_done {
    //! A sink paused for a speech that ends while the false interruption resolves is released.
    use lk_agents_fuzz::drive::{Case, Policy, RoomLegs, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, TurnMode};

    // Catches: the false-interruption timer took the pause of a speech already done (its task had
    // marked the handle done, its step not yet handled) and resumed nothing, so the speech's own
    // release found no pause left and the room's audio output stayed paused: every later speech
    // played into it until the close (survey seed 31909 under the task bridge, speech_settles).
    #[test]
    fn the_next_speech_plays() {
        let setup = Setup {
            vad: true,
            turn: TurnMode::Manual,
            room_legs: Some(RoomLegs { pre_connect: None }),
            ..Setup::default()
        };
        let say = |input| Input::Say {
            text: AgentText("One moment.".into()),
            input,
            allow: None,
            to_ctx: true,
        };
        let inputs = vec![
            Input::Advance { ms: Millis(100) },
            say(SayInput::Audio),
            Input::Advance { ms: Millis(20) },
            // The person starts while the speech has no audio yet: the sink is paused.
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::Advance { ms: Millis(400) },
            // In one instant: the speech ends with nothing played, and the person stops.
            Input::FeedEnd { stream: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(100) },
            say(SayInput::Words),
            Input::Advance { ms: Millis(3_000) },
        ];
        for policy in [Policy::BasicPos, Policy::TaskBridge] {
            let stuck: Vec<u64> = (0..64)
                .filter(|&schedule| {
                    let case = Case {
                        setup: setup.clone(),
                        inputs: inputs.clone(),
                        schedule,
                        random: Vec::new(),
                        policy,
                    };
                    run(case).failures.iter().any(|f| f.contract == "speech_settles")
                })
                .collect();
            assert!(stuck.is_empty(), "{policy:?}: schedules leaving a speech unplayed: {stuck:?}");
        }
    }
}

mod remote_session_room {
    //! The remote session's room transport (`RoomTransport`), which no DST seed carries: the console
    //! seeds run the protocol over a socket.
    use lk_agents_abi::voice::rtc::{ConnectionState, StreamInfo};
    use lk_agents_fuzz::fakes::room::FakeRtc;
    use lk_agents_runtime::rtc::events::{RoomEventBus, byte_streams};
    use lk_agents_runtime::voice::remote_session::room::RoomTransport;
    use lk_agents_runtime::voice::remote_session::{
        SessionTransport, TOPIC_SESSION_MESSAGES, proto,
    };
    use prost::Message as _;
    use proto::agent_session_message::Message;
    use std::sync::Arc;

    fn ping(id: &str) -> proto::AgentSessionMessage {
        proto::AgentSessionMessage {
            message: Some(Message::Request(proto::SessionRequest {
                request_id: id.to_owned(),
                request: Some(proto::session_request::Request::Ping(
                    proto::session_request::Ping {},
                )),
            })),
        }
    }

    fn request_id(message: proto::AgentSessionMessage) -> String {
        match message.message {
            Some(Message::Request(request)) => request.request_id,
            _ => panic!("not a request"),
        }
    }

    // Catches: messages read out of order or lost to a malformed one, a sent message not on the
    // topic as one closed stream, a send on a disconnected room not raised, or a close leaving
    // the topic registered or a receive waiting.
    #[tokio::test]
    async fn the_room_carries_one_message_per_stream() {
        let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
        let room = rtc.room("call");
        let transport = Arc::new(RoomTransport::new(room.clone(), clock));
        transport.start().await.unwrap();
        transport.start().await.expect("idempotent");
        let info = || StreamInfo { topic: TOPIC_SESSION_MESSAGES.into(), ..StreamInfo::default() };
        let first = ping("req_1").encode_to_vec();
        let (head, tail) = first.split_at(3);
        room.send_byte_stream("cli", info(), [head.to_vec(), tail.to_vec()]).unwrap();
        room.send_byte_stream("cli", info(), [vec![0xff, 0xff]]).unwrap();
        room.send_byte_stream("cli", info(), [ping("req_2").encode_to_vec()]).unwrap();
        assert_eq!(request_id(transport.recv().await.unwrap().unwrap()), "req_1");
        assert_eq!(request_id(transport.recv().await.unwrap().unwrap()), "req_2");

        transport.send_message(&ping("req_3")).await.unwrap();
        let sent = room.local_participant().byte_streams();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].opts.topic, TOPIC_SESSION_MESSAGES);
        assert!(sent[0].opts.name.starts_with("AS_") && sent[0].closed);
        let decoded =
            proto::AgentSessionMessage::decode(sent[0].chunks.concat().as_slice()).unwrap();
        assert_eq!(request_id(decoded), "req_3");

        room.set_connection_state(ConnectionState::Disconnected);
        assert!(transport.send_message(&ping("req_4")).await.is_err());

        let waiting = {
            let transport = Arc::clone(&transport);
            tokio::spawn(async move { transport.recv().await })
        };
        tokio::task::yield_now().await;
        transport.close().await;
        assert!(waiting.await.unwrap().unwrap().is_none());
        byte_streams(room.as_ref(), TOPIC_SESSION_MESSAGES).expect("the topic is free again");
    }
}

mod replaced_inputs {
    //! A recogniser, a detector or a microphone the session replaced is closed, so nothing a case says
    //! into it can look like it was heard.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Models};
    use lk_agents_fuzz::record::Kind;

    // Catches: the fakes ignoring `end_input` and the harness keeping a replaced microphone open. A
    // real stream ends once its input does; the fake never did, so the conversation kept speaking
    // into the first recogniser, whose audio had moved to the new one, and every final it held back
    // for want of audio left the caller unheard (soak seeds 0, 1, 3 and 6 heard 0 to 4 turns of 40).
    #[test]
    fn replaced_inputs_close() {
        let inputs = [
            Input::UpdateModels { models: Models::All },
            Input::SetAudioInput { attach: true },
            Input::Advance { ms: Millis(20) },
        ];
        let run = simulate(async {
            let mut run = Run::start(
                &Setup { stt: true, vad: true, ..Setup::default() },
                0,
                Default::default(),
            )
            .await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        for kind in [Kind::Stt, Kind::Vad, Kind::Mic] {
            let live = run.record.live().of(kind);
            assert_eq!(live.len(), 1, "one {kind:?} live, the newest: {live:?}");
        }
    }
}

mod replaced_output_detached {
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
                run.record
                    .sink()
                    .iter()
                    .any(|(_, call)| matches!(call, Sink::Audio { output: 0, .. })),
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
}

mod replaced_output_mid_step {
    //! A replaced output still gets what the reference sends it: the rest of a step already running,
    //! and a synchroniser's closing flushes; the swap itself interrupts nothing.
    use lk_agents_abi::voice::events::AgentState;
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Case, Run, Setup, run, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
    use lk_agents_voice::events::{SessionEvent, SpeechOutcome};
    use lk_agents_voice::trace::{InterruptCause, InterruptStep, PlayoutStep, Step};

    fn clean(setup: Setup, inputs: Vec<Input>) {
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"output_replacement_is_clean"), "{failed:?}");
        assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
    }

    // A physically admitted old text destination gets its segment's closing flush while retained.
    // The replacement must not force that work onto the new destination or retire it before delivery.
    #[test]
    fn a_synchroniser_flushes_its_old_output_as_it_closes() {
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        let inputs = vec![
            Input::Say {
                text: AgentText("Hello there, thanks for calling the front desk.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
            Input::SetTranscriptionOutput { attach: true },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: true },
        ];
        clean(setup, inputs);
    }

    // Catches: a sink swap force-interrupting the speech playing into the old output. Replacement takes
    // effect at a segment boundary and does not interrupt speech (voice-session.md "Output handover
    // policy"): the playing segment finishes on its original output.
    #[test]
    fn a_swap_lets_the_playing_speech_finish() {
        let inputs = [
            Input::Say {
                text: AgentText("Hello there, thanks for calling the front desk.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
            Input::SetAudioOutput { attach: true },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let swapped = journal
            .iter()
            .any(|(_, s)| matches!(s, Step::Playout(PlayoutStep::OutputReplaced { .. })));
        assert!(swapped, "the output was replaced");
        let speaking = journal.iter().any(|(_, s)| {
            matches!(
                s,
                Step::Emitted(SessionEvent::AgentStateChanged { new: AgentState::Speaking, .. })
            )
        });
        assert!(speaking, "the agent was speaking when the output was replaced");
        let cut = journal.iter().any(|(_, s)| {
            matches!(
                s,
                Step::Interrupt(InterruptStep::Interrupted {
                    cause: InterruptCause::OutputReplaced,
                    ..
                })
            )
        });
        assert!(!cut, "the swap interrupted the playing speech");
        let completed = journal.iter().any(|(_, s)| {
            matches!(
                s,
                Step::Emitted(SessionEvent::SpeechDone { outcome: SpeechOutcome::Completed, .. })
            )
        });
        assert!(completed, "the speech completed");
        let (failures, _) = check_all(&run.record);
        let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(failed.is_empty(), "{failed:?}");
    }
}

mod room_close_ends_the_session {
    //! The room closing ends the session, however the room IO's own close races it.
    use lk_agents_abi::voice::events::CloseReason;
    use lk_agents_fuzz::drive::{Outcome, RoomLegs, RoomSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RoomIn};
    use lk_agents_voice::trace::{CloseStep, RoomEventKind, RoomStep, Step};

    // Catches: the room IO's `Close` stopped its forwarder before the forwarder passed the same
    // `Close` (and the link's end) to the session, so the session never closed and a later reply
    // played into the closed room output, cut with no cause (`speech_is_heard`, campaign seed 19655).
    #[test]
    fn the_session_closes_with_its_room() {
        let setup = Setup {
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            room_legs: Some(RoomLegs { pre_connect: None }),
            ..Setup::default()
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Room { event: RoomIn::Linked },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("One moment please.".into()) },
                Input::Advance { ms: Millis(400) },
                Input::Room { event: RoomIn::Closed },
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
            let closed = journal
                .iter()
                .position(|s| matches!(s, Step::Room(RoomStep::Event(RoomEventKind::Close))))
                .expect("the session hears the room close");
            assert!(
                journal[closed..].iter().any(|s| matches!(
                    s,
                    Step::Close(CloseStep::Requested { reason: CloseReason::JobShutdown, .. })
                )),
                "the room's close ends the session"
            );
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod server_generation_ends {
    //! A generation the realtime server opens on its own is a turn the case ends like any other.
    use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};

    // Catches: the fake server opening a second generation that it never counted as an open turn, so
    // nothing ever ended it and its speech waited on the socket until the close (`speech_settles`,
    // survey seed 179).
    #[test]
    fn a_server_generation_settles() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let inputs = vec![
            Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
            Input::Advance { ms: Millis(20) },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"speech_settles"), "{failed:?}");
        assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
    }
}

mod session_close_ends_the_job {
    //! A session that closes on its own error ends the job it is the primary session of.
    use lk_agents_abi::voice::events::CloseReason;
    use lk_agents_fuzz::drive::{Outcome, RealtimeSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
    use lk_agents_fuzz::record::Host;

    // Catches: nothing told the job its session closed, so after a fatal model error the process
    // lingered with a dead session until it was killed (`lk agent console`).
    #[test]
    fn a_session_closed_by_an_error_ends_the_job() {
        let setup = Setup { realtime: Some(RealtimeSetup::default()), ..Setup::default() };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Advance { ms: Millis(1_000) },
                Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::FatalError },
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            let ended: Vec<&String> = run
                .record
                .host()
                .iter()
                .filter_map(|(_, host)| match host {
                    Host::JobEnded(reason) => Some(reason),
                    _ => None,
                })
                .collect();
            assert_eq!(ended, [&CloseReason::Error.to_string()], "the job ends once, on the error");
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod sink_reset_reports_cut {
    //! An output that throws its playback count away reports what it owed as cut first.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
    use lk_agents_voice::events::SessionEvent;

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

    // Catches: a speech whose open segment the output cut and reset under it, then played the rest,
    // committed as complete: the last segment's finish was taken for the whole speech, though the
    // caller never heard its first words (survey seeds 24667, 6113, `shown_transcript_is_conserved`).
    #[test]
    fn a_speech_cut_by_a_reset_is_not_complete() {
        let text = |t: &str| AgentText(t.into());
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Sure, I can") },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::SinkReset { output: Ref::new(0) },
            Input::LlmText { chat: Ref::new(0), text: text(" help with that.") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::PlayoutFinished { segment: Ref::new(1) },
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
        let said: Vec<bool> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::Assistant => Some(m.interrupted),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(said, [true], "the speech's history entry claims it all played");
    }
}

mod swap_ends_the_servers_turn {
    //! A realtime session swapped for a fresh one takes the person's turn along, at the finish too.
    use lk_agents_fuzz::drive::{Outcome, RealtimeSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};

    // Catches: the finish ended a person the realtime server had heard start, but read only a dropped
    // connection as taking that turn along; after a fatal error the fallback swapped in a fresh
    // session that never heard them, refused the stop every round, and the run hit its budget
    // (`session_settles`, survey seeds 21875, 26055, 31823; 05ed23451).
    #[test]
    fn a_swap_after_the_server_heard_the_person_lets_the_run_go_quiet() {
        let setup = Setup {
            realtime: Some(RealtimeSetup { fallback: true, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let server = |event| Input::RealtimeEvent { socket: Ref::new(0), event };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Advance { ms: Millis(3_000) },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
                Input::Advance { ms: Millis(200) },
                server(RtEvent::SpeechStarted),
                Input::Advance { ms: Millis(200) },
                // The server's last word on the person is their start: then it fails, and is swapped.
                server(RtEvent::FatalError),
                Input::Advance { ms: Millis(1_000) },
            ] {
                run.apply(&input).await;
            }
            assert!(
                run.record.applied().iter().all(|a| a.accepted),
                "the scenario: heard, then swapped"
            );
            let outcome = finish(run, Outcome::default()).await;
            assert!(!outcome.stopped_early, "{:?}", outcome.ending);
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod unadmitted_output_cleanup {
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
}

#[test]
fn recorder_attachment_after_the_loop_exits_retires_its_registration() {
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{Input, Millis};

    simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in [
            Input::Recorder { attach: true },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(30_000) },
            Input::Recorder { attach: true },
            Input::Recorder { attach: true },
        ] {
            run.apply(&input).await;
        }
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
