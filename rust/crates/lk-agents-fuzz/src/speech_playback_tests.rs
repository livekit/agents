//! Speech regression tests.

mod console_reply_plays_out {
    //! A console reply settles on the CLI's playback receipt.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

    // Catches: the console refusing the CLI's unkeyed `AudioPlaybackFinished` (0ef7f361b): every
    // spoken reply waited out its playback deadline and failed "playback report timed out", which the
    // rules excused as a sink that lost its receipt. The CLI acks each flush it plays.
    #[test]
    fn a_console_say_finishes_on_the_cli_receipt() {
        let setup = Setup { console: true, ..Setup::default() };
        let say = |text: &str| Input::Say {
            text: AgentText(text.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        };
        let inputs = vec![
            Input::Advance { ms: Millis(100) },
            say("Hello from the console."),
            Input::Advance { ms: Millis(5_000) },
            say("And a second reply."),
            Input::Advance { ms: Millis(5_000) },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    }
}

mod cut_node_words_open_no_segment {
    //! A speech cut before its voice, whose words a `transcription_node` was still carrying, leaves
    //! the synchroniser no segment of its own.
    use lk_agents_fuzz::drive::{
        ConnSetup, NodeSetup, Outcome, Policy, Run, Setup, TtsAlignment, finish, simulate_with,
    };
    use lk_agents_fuzz::input::{
        AgentText, Confidence, Count, Input, IvrInput, Millis, Ref, Speaker, ToolArgs, TurnMode,
        UserText,
    };

    // Catches: a synthesis lost to a TTS outage forwarded its words into the `transcription_node`
    // after the person cut the speech, then ended its audio leg as an unvoiced segment before the
    // node had settled; the cut cancelled the node, so the synchroniser took that audio end with no
    // words, and the empty segment took the next speech's words and showed them whole before their
    // audio (`transcript_follows_audio`, campaign seed 24984; its case minimised under its schedule
    // and policy).
    #[test]
    fn the_next_speech_words_wait_for_their_audio() {
        let setup = Setup {
            stt: true,
            vad: true,
            eot: true,
            interruption: true,
            turn: TurnMode::Stt,
            nodes: Some(NodeSetup {
                llm: false,
                tts: false,
                stt: false,
                llm_text: None,
                transcription: true,
            }),
            user_turn_max_ms: Some(1000),
            tts: true,
            video: true,
            sync_transcription: true,
            conn: Some(ConnSetup { stt: 2, llm: 1, tts: 3, interval_ms: 120 }),
            tts_alignment: TtsAlignment::Words,
            ..Setup::default()
        };
        simulate_with(24984, async {
            let mut run = Run::start(&setup, 24984, Policy::BasicPos).await;
            for input in [
                Input::TtsOutage { down: true },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(17),
                    speaking: false,
                    duration: Millis(100),
                },
                Input::GenerateReply {
                    user_input: Some(UserText("yeah hi um I'm calling about my order".into())),
                    instructions: None,
                    allow: None,
                },
                Input::Advance { ms: Millis(14901) },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(100) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(10),
                    speaking: false,
                    duration: Millis(200),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(5) },
                Input::Advance { ms: Millis(58) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(5),
                    speaking: false,
                    duration: Millis(258),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(100) },
                Input::VideoFrames { video: Ref::new(0), frames: Count(2) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(88),
                    speaking: true,
                    duration: Millis(100),
                },
                Input::VadStart { vad: Ref::new(0), speech_ms: Millis(94) },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(100) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(52),
                    speaking: true,
                    duration: Millis(200),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::LlmText { chat: Ref::new(0), text: AgentText(": *the ".into()) },
                Input::Advance { ms: Millis(9) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("office is".into()) },
                Input::Advance { ms: Millis(5) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("ose".into()) },
                Input::Advance { ms: Millis(8) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("d* on Monday.\n- T".into()) },
                Input::VideoFrames { video: Ref::new(0), frames: Count(2) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(64),
                    speaking: true,
                    duration: Millis(300),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(53) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("uesday 9-5\n".into()) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("- We".into()) },
                Input::Advance { ms: Millis(5) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("dnes".into()) },
                Input::Advance { ms: Millis(5) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("day 9".into()) },
                Input::Advance { ms: Millis(37) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(67),
                    speaking: true,
                    duration: Millis(400),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(22) },
                Input::LlmText { chat: Ref::new(0), text: AgentText("-1 ☕".into()) },
                Input::Advance { ms: Millis(10) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "caller".into(),
                    args: ToolArgs("{\"label\":\"human<|im_end|>".into()),
                },
                Input::Advance { ms: Millis(573) },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(60),
                    speaking: true,
                    duration: Millis(500),
                },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(10) },
                Input::Advance { ms: Millis(100) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(80),
                    speaking: true,
                    duration: Millis(600),
                },
                Input::SttStart { stt: Ref::new(0) },
                Input::SttPreflight {
                    utterance: Ref::new(0),
                    text: UserText("can you".into()),
                    speaker: Some(Speaker(0)),
                },
                Input::Handoff {
                    agent: "lookup_order".into(),
                    instructions: AgentText("".into()),
                    tools: vec!["caller".into(), "agent-2".into()],
                    turn: None,
                },
                Input::Handoff {
                    agent: "sip_1".into(),
                    instructions: AgentText(
                        "Yes... no, wait -- actually, let me check that again.".into(),
                    ),
                    tools: vec!["caller".into(), "transfer".into(), "transfer".into()],
                    turn: None,
                },
                Input::TtsOutage { down: false },
                Input::Advance { ms: Millis(100) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::SttStart { stt: Ref::new(2) },
                Input::Ivr {
                    input: IvrInput::Start,
                    silence: Millis(0),
                    text: UserText("yeah hi um I'm calling about my order".into()),
                },
                Input::Handoff {
                    agent: "billing_agent".into(),
                    instructions: AgentText("Let me pull up your account -- one moment.".into()),
                    tools: vec!["billing_agent".into(), "agent-2".into(), "caller".into()],
                    turn: None,
                },
                Input::GenerateReply {
                    user_input: None,
                    instructions: Some(AgentText("".into())),
                    allow: None,
                },
                Input::Advance { ms: Millis(0) },
                Input::SttStart { stt: Ref::new(3) },
                Input::SttFinal {
                    utterance: Ref::new(10),
                    text: UserText("okay thanks bye".into()),
                    speaker: Some(Speaker(0)),
                },
                Input::SttEnd { utterance: Ref::new(10) },
                Input::Advance { ms: Millis(100) },
                Input::Advance { ms: Millis(100) },
                Input::Advance { ms: Millis(100) },
                Input::LlmText { chat: Ref::new(4), text: AgentText("ld. ".into()) },
            ] {
                run.apply(&input).await;
            }
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod cut_reply_keeps_heard_words {
    //! A reply cut mid-playout is committed as the words heard when the output reports the cut late.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
    use lk_agents_voice::events::SessionEvent;

    const TEXT: &str = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve.";

    // Catches: a cut speech reading the synchroniser's words only if the output's report of the cut
    // landed within a few scheduler hops. A host audio source reports from the host's loop, later
    // than that, so the reply was committed whole: on real calls every interrupted reply's history
    // held words the caller never heard (the room's transcript, paced, stopped where the audio did).
    // The reference awaits the output's playout event for the synchronised transcript.
    #[test]
    fn a_cut_reply_is_committed_as_heard() {
        let setup = Setup { sync_transcription: true, lagging_output: true, ..Setup::default() };
        let inputs = [
            Input::Say {
                text: AgentText(TEXT.into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(2_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let said: Vec<String> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::Assistant => m.text_content(),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        // The fake's audio is one short chunk and no progress was reported: nothing was heard.
        assert!(
            said.iter().all(|t| TEXT.starts_with(t.trim_end()) && t.len() < TEXT.len()),
            "{said:?}"
        );
    }
}

mod host_interrupt_lands_on_every_end {
    //! A host `interrupt()` lands on the speech it was asked of however that speech ends, and one
    //! asked of a speech that never ran lands nowhere: a done handle reads the core's verdict.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

    // Catches: an `interrupt()` asked of a `say()` handle before its speech existed was held by the
    // handle and sent once the loop created the speech, which the loop had created done (`say()` with
    // no TTS is refused): the host read the speech interrupted forever while the core had journalled
    // no interruption (`interrupted_handle_is_journalled`, survey seed 453 once the DST held a
    // call's handle from its return).
    #[test]
    fn an_interrupt_of_a_speech_created_done_lands_nowhere() {
        let inputs = [
            Input::Advance { ms: Millis(100) },
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::InterruptSpeech { speech: Ref::new(0), force: true },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run =
                Run::start(&Setup { tts: false, ..Setup::default() }, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        assert!(run.record.applied()[2].accepted, "the scenario: the uncreated handle interrupted");
        let speeches = run.record.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.is_done() && speech.error().is_some(), "the scenario: the say was refused");
        assert!(!speech.is_interrupted(), "{speech:?}");
        let (failures, _) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:#?}");
    }
}

mod lost_voice_ends_its_audio_once {
    //! A voice lost after its audio played ends the segment's audio leg once.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
    use lk_agents_voice::events::{ErrorSource, SessionEvent};
    use lk_agents_voice::trace::{PlayoutStep, Step};

    // Catches: a synthesis failing after its audio played reported its segment as heard with no audio,
    // so the segment ended its audio leg a second time as an unvoiced one; the synchroniser paired the
    // next speech's words with that empty audio and showed them all before any of their audio played
    // (`transcript_follows_audio`, campaign seed 20655; a7e9506e9).
    #[test]
    fn the_next_speech_words_follow_its_own_audio() {
        let text = |t: &str| AgentText(t.into());
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Let me check that.") },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                // The voice fails after its audio reached the speaker, which then plays it all.
                Input::TtsFail { tts: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(5_000) },
                Input::PlayoutFinished { segment: Ref::new(0) },
                Input::Advance { ms: Millis(500) },
                Input::Say {
                    text: text("Anything else?"),
                    input: SayInput::Words,
                    allow: None,
                    to_ctx: true,
                },
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            let at = |step: fn(&Step) -> bool| {
                run.record.journal().iter().position(|(_, s)| step(s)).expect("the scenario")
            };
            let played = at(|s| {
                matches!(s, Step::Playout(PlayoutStep::PlaybackFinished { interrupted: false, .. }))
            });
            let lost = at(
                |s| matches!(s, Step::Emitted(SessionEvent::Error(e)) if e.source == ErrorSource::Tts),
            );
            assert!(played < lost, "the scenario: the voice is lost after its audio played whole");
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod lost_voice_words_shown_whole {
    //! A reply whose voice is lost before any audio shows its words whole, not paced by the next
    //! reply's audio.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, TtsAlignment, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_fuzz::record::Sink;
    use lk_agents_voice::trace::{GenerationStep, Step};

    // Catches: with the voice down from the start, the segment's binding was activated only as its
    // queued words were forwarded, after the segment had decided against ending its audio leg (no
    // transcript chain open yet); the synchroniser kept the words until the next reply's audio paced
    // them (`aligned_transcript_follows_tts`, survey seed 27391).
    #[test]
    fn a_reply_whose_voice_was_down_shows_its_words_before_the_next_reply() {
        let text = |t: &str| AgentText(t.into());
        let setup = Setup {
            sync_transcription: true,
            tts_alignment: TtsAlignment::Words,
            aligned_transcript: Some(true),
            ..Setup::default()
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::TtsOutage { down: true },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("I'm sorry, I didn't catch that.") },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(10_000) },
                Input::TtsOutage { down: false },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText {
                    chat: Ref::new(1),
                    text: text("Done, your refund is on its way."),
                },
                Input::LlmEnd { chat: Ref::new(1) },
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            // The first reply's words reached the text output before the next reply's synthesis.
            let next = run.record.journal().iter().find_map(|(at, s)| match s {
                Step::Generation(GenerationStep::TtsRequested { .. }) => Some(*at),
                _ => None,
            });
            let next = next.expect("the second reply was synthesised");
            let shown = run.record.sink().iter().any(|(at, call)| {
                *at < next && matches!(call, Sink::Text { text, .. } if text.contains("catch"))
            });
            assert!(shown, "the lost voice's words waited for the next reply's audio");
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod playout_finish_lost {
    //! A playout whose finish the output never reports settles once the output played all of it.
    use lk_agents_fuzz::drive::{Case, Run, Setup, run, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

    // Catches: an output that played the speech's audio and lost its "playback finished" report (a
    // transport drops it) held the speech waiting on its playout until the session closed, and every
    // speech queued behind it (DST seeds 229, 803 with the lost-finish fault).
    #[test]
    fn a_lost_finish_releases_the_speech_after_its_audio() {
        let inputs = [
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::Advance { ms: Millis(30_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let host = run.record.host_view();
        let speeches = host.speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.is_done(), "the speech still waits on a finish the output lost");
    }

    // Catches: a sink that reset mid-speech, dropping what it held, then lost the finish of the
    // speech's later audio held the speech until the close: the dropped audio never counted as
    // played, so the margin never armed (DST seed 10521 with the lost-finish fault).
    #[test]
    fn a_lost_finish_after_a_reset_releases_the_speech() {
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
            Input::PlayoutFinishLost { segment: Ref::new(1) },
            Input::Advance { ms: Millis(30_000) },
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
        assert!(speech.is_done(), "the speech still waits on a finish the output lost");
    }

    // Catches: a speech cut after its output played all of its audio and lost the finish left that
    // segment outstanding on the output's count, so every later playout waited for one finish more
    // than it was owed: the next speech's real finish settled the old segment and the speech waited
    // out the lost-finish margin (DST seeds 7836, 6883 with finishes lost after a reset).
    #[test]
    fn a_cut_after_a_lost_finish_keeps_the_next_playout_in_step() {
        let say = |text: &str| Input::Say {
            text: AgentText(text.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        };
        let inputs = [
            say("Hello there."),
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinishLost { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(3_000) },
            say("Goodbye."),
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::PlayoutProgressed { segment: Ref::new(1), ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(200) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let host = run.record.host_view();
        let speeches = host.speeches();
        let [_, next] = &speeches[..] else { panic!("two speeches, got {}", speeches.len()) };
        assert!(next.is_done(), "the next speech waits for a finish its output already reported");
    }

    // Catches: the lost-finish margin started at the playout's next periodic look, up to a margin after
    // the report that completed what the output owed, so a lost finish held the speech for twice the
    // margin and the DST's release phase, waiting only on a journalled due time, never saw it go quiet
    // (DST seed 6883 with finishes lost after a reset).
    #[test]
    fn a_lost_finish_is_due_a_margin_after_the_last_report() {
        let inputs = [
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(1_500) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::PlayoutFinishLost { segment: Ref::new(0) },
            Input::Advance { ms: Millis(2_100) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let host = run.record.host_view();
        let speeches = host.speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.is_done(), "the lost finish was not due a margin after the last report");
    }

    // Catches: the speaker's own finish arriving after the playout's lost-finish margin had settled
    // the segment for it was refused by the output fake, which had already let the segment go: the
    // segment stayed open past the close and read as a leak (`nothing_left_live`, campaign seed
    // 11256, its output replaced). A late report of a finish the product settled still ends the
    // segment.
    #[test]
    fn a_finish_reported_after_the_margin_settles_the_segment() {
        let inputs = vec![
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::Advance { ms: Millis(5_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            // The output replaced: no clear at the close settles what it still holds.
            Input::SetAudioOutput { attach: true },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: true },
            Input::AwaitExit {},
        ];
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"nothing_left_live"), "{failed:?}");
    }
}

mod refused_speech_announced {
    //! A speech handle refused before it reached the queue is still announced and ended.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_voice::events::SpeechOutcome;

    /// Each speech the host holds a handle for: whether it was announced, and what `SpeechDone`s it
    /// heard.
    fn announced(setup: Setup, inputs: &[Input]) -> Vec<(bool, Vec<SpeechOutcome>)> {
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in inputs {
                run.apply(input).await;
            }
            run
        });
        run.record
            .host_view()
            .speeches()
            .iter()
            .filter(|s| s.is_known())
            .map(|s| {
                let done = s.done_announcements().into_iter().map(|(_, outcome)| outcome).collect();
                (s.created_at().is_some(), done)
            })
            .collect()
    }

    fn say(input: SayInput) -> Input {
        Input::Say { text: AgentText("Hello.".into()), input, allow: None, to_ctx: true }
    }

    fn realtime() -> Setup {
        // No voice: with one, the TTS speaks what the model cannot.
        Setup {
            realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
            tts: false,
            ..Setup::default()
        }
    }

    // Catches: `say(words)` with no TTS answered with a failed handle the host stream never heard of
    // (survey seed 0, speech 3).
    #[test]
    fn say_without_tts_is_announced() {
        let setup = Setup { tts: false, ..Setup::default() };
        let inputs = [say(SayInput::Words), Input::Advance { ms: Millis(20) }];
        assert_eq!(announced(setup, &inputs), [(true, vec![SpeechOutcome::Failed])]);
    }

    // Catches: `say(stream)` refused by a realtime model answered with a failed handle the host
    // stream never heard of (survey seed 1, speech 4).
    #[test]
    fn say_stream_on_realtime_is_announced() {
        let inputs = [say(SayInput::TextStream), Input::Advance { ms: Millis(20) }];
        assert_eq!(announced(realtime(), &inputs), [(true, vec![SpeechOutcome::Failed])]);
    }

    // Catches: a realtime `generate_reply()` still waiting on the server when it is interrupted,
    // answered with an interrupted handle the host stream never heard of (survey seed 14, speech 8).
    #[test]
    fn interrupted_realtime_request_is_announced() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(20) },
        ];
        assert_eq!(announced(realtime(), &inputs), [(true, vec![SpeechOutcome::Interrupted])]);
    }
}

mod reply_behind_tool_step {
    //! A reply made while a speech with tools plays asks its model once that speech's step is on the
    //! record, not once the whole speech is over.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs};
    use lk_agents_voice::trace::{GenerationStep, Step};

    // Catches: a reply created while a speech played waited for that speech to finish before planning
    // its request. A speech whose tools answered as its step ended goes on to a follow-up step, which
    // needs the floor; the new reply had taken the floor and waited for the speech's end, so both
    // waited until the replan's bound (the interruption timeout and a second): six seconds of dead air
    // (`speech_settles`, survey seed 14338). The reply is planned once the step is recorded.
    #[test]
    fn a_reply_behind_a_tool_step_asks_its_model_at_once() {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Sure, one moment.".into()) },
            // A tool the agent does not have: its error output asks the model for a follow-up step.
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "missing".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            // Asked while the first speech plays and its tool runs.
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(200) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(7_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let ended = run
            .record
            .applied()
            .iter()
            .find(|a| matches!(a.input, Input::PlayoutFinished { .. }))
            .map(|a| a.at.at_ms)
            .expect("the step ended");
        // The requests made after the step ended: the follow-up's and the second reply's.
        let asked: Vec<(u64, u64)> = run
            .record
            .journal()
            .iter()
            .filter_map(|(at, s)| match s {
                Step::Generation(GenerationStep::ChatRequested { speech, .. })
                    if at.at_ms >= ended =>
                {
                    Some((speech.0, at.at_ms - ended))
                }
                _ => None,
            })
            .collect();
        assert!(!asked.is_empty(), "nothing was asked after the step ended");
        assert!(asked.iter().all(|(_, after)| *after < 1_000), "asked late: {asked:?}");
    }
}

mod reply_during_playout {
    //! A reply asked for while the previous one is still playing makes its own model request.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_fuzz::record::Kind;

    // Catches: since c43d27219, `generate_reply` while the previous reply still spoke interrupted it
    // and never issued its own LLM request, its handle never settling (coordinator report, real
    // room). A reply asked for during a long playout must reach its model.
    #[test]
    fn a_reply_asked_during_playout_reaches_its_model() {
        let setup = Setup::default();
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmText {
                chat: Ref::new(0),
                text: AgentText("A long first answer that keeps playing for a while.".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::GenerateReply {
                user_input: None,
                instructions: Some(AgentText("say something else".into())),
                allow: None,
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
        let chats = run.record.resources(Kind::Chat).count();
        assert_eq!(chats, 2, "the second reply never asked its model");
    }
}

mod shown_words_follow_their_playback {
    //! A shown word is the speech's whose playback released it, and counts against that playback.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::trace::{PlayoutStep, Step};

    // Catches: after a sink reset the rest of a speech's audio plays as a segment of its own, whose
    // reports the speech's playout dropped as another key's: its played position stayed where the
    // reset left it, and the words that playback released were journalled against it, ahead of the
    // audio (`transcript_follows_audio`, campaign seed 13704; d5f0e2ff9).
    #[test]
    fn the_rest_after_a_sink_reset_is_the_speechs_playback() {
        let text = |t: &str| AgentText(t.into());
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Your order ships on Tuesday") },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::Advance { ms: Millis(500) },
                Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(500) },
                // The sink resets mid-speech: the rest of its audio plays as a segment of its own.
                Input::SinkReset { output: Ref::new(0) },
                Input::LlmText {
                    chat: Ref::new(0),
                    text: text(" and arrives by Friday afternoon."),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutStarted { segment: Ref::new(1) },
                Input::Advance { ms: Millis(1_000) },
                Input::PlayoutProgressed { segment: Ref::new(1), ms: Millis(5_000) },
                Input::PlayoutFinished { segment: Ref::new(1) },
                Input::Advance { ms: Millis(1_000) },
            ] {
                run.apply(&input).await;
            }
            let journal: Vec<&Step> = run.record.journal().iter().map(|(_, s)| s).collect();
            let reset = journal
                .iter()
                .position(|s| matches!(s, Step::Playout(PlayoutStep::SinkReset { .. })))
                .expect("the scenario: a sink reset mid-speech");
            assert!(
                journal[reset..]
                    .iter()
                    .any(|s| matches!(s, Step::Playout(PlayoutStep::TranscriptShown { .. }))),
                "the rest's playback released none of the speech's words"
            );
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod skip_reply_turn_no_hook {
    //! A turn committed with `skip_reply` keeps its words without `on_user_turn_completed`.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
    use lk_agents_voice::agent_session::HookKind;
    use lk_agents_voice::trace::{HandoffStep, Step};

    // Catches: a turn that asks for no reply (`commit_user_turn(skip_reply=True)`, or one AMD
    // consumed) was parked on the developer's `on_user_turn_completed` like any other, its item
    // committed only once the hook answered and stamped with its delay. The reference commits the
    // item and returns before the hook (found by `parity.py`'s payload comparison: the AMD scenario's
    // user item carried an `on_user_turn_completed_delay` the reference's has not).
    #[test]
    fn a_skip_reply_turn_runs_no_hook() {
        let setup = Setup {
            stt: true,
            turn: TurnMode::Manual,
            hook_bound_ms: Some(1_000),
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::SttStart { stt: Ref::new(0) },
                Input::SttFinal {
                    utterance: Ref::new(0),
                    text: UserText("yeah hi".into()),
                    speaker: None,
                },
                Input::SttEnd { utterance: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::CommitUserTurn {
                    transcript_timeout: Millis(100),
                    stt_flush: Millis(0),
                    skip_reply: true,
                },
                Input::Advance { ms: Millis(2_000) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let hooks = run
            .record
            .journal()
            .iter()
            .filter(|(_, s)| {
                matches!(
                    s,
                    Step::Handoff(HandoffStep::HookIssued {
                        kind: HookKind::OnUserTurnCompleted,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(hooks, 0, "on_user_turn_completed ran for a turn that asks for no reply");
        let users = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter(|(_, ev)| {
                matches!(ev, lk_agents_voice::events::SessionEvent::ConversationItemAdded { item, .. }
                    if matches!(&**item, ChatItem::Message(m) if m.role == Role::User))
            })
            .count();
        assert_eq!(users, 1, "the turn's words reach the record");
    }
}

mod speculative_segment_takes_current_output {
    //! A segment synthesised before its authorization takes the outputs current at its activation: an
    //! audio output detached meanwhile never plays it, and its words show without that audio.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
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
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}
