//! Input regression tests.

mod duplex_late_transcript {
    //! A duplex model's words that arrive after the sound they describe still enter the conversation.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::fakes::RealtimeDelays;
    use lk_agents_fuzz::input::{Input, Millis, Ref};
    use lk_agents_voice::events::SessionEvent;

    // Catches: the last words of a duplex model's utterance, which its text stream delivers after the
    // utterance's audio ended, being lost or failing the session: the segmenter attributes them to
    // the completed speech (`TranscriptAttribution`) and the driver commits them as an item of their
    // own, with no error.
    #[test]
    fn a_late_transcript_is_committed_after_its_speech() {
        // The model's utterance plays from 0.7 s to 2.1 s; its last words' text comes 1 s later.
        let delays = RealtimeDelays { late_words: Some(1_000), ..RealtimeDelays::default() };
        let setup = Setup {
            realtime: Some(RealtimeSetup { duplex: true, delays, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(1_500) },
            Input::Advance { ms: Millis(1_500) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let events = run.record.host_view().events();
        let errors: Vec<_> =
            events.iter().filter(|(_, ev)| matches!(ev, SessionEvent::Error(_))).collect();
        assert!(errors.is_empty(), "{errors:?}");
        let said = events
            .iter()
            .filter(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => {
                    matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::Assistant)
                }
                _ => false,
            })
            .count();
        assert_eq!(said, 2, "the utterance, and its late words as an item of their own");
    }
}

mod eot_asks_again_after_resumed_speech {
    //! A prediction answered during a pause the user talks through is dropped: the next pause asks.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Confidence, Input, Millis, Ref, UserText};
    use lk_agents_voice::events::SessionEvent;
    use lk_agents_voice::trace::Step::Emitted;

    // Catches: speech resuming in the segment cancelled only a pending prediction and kept an answered
    // one, so no pause after it asked again and the turn was judged by a mid-sentence answer. The
    // reference drops `_turn_detector_prediction_fut` whatever its state (`_on_vad_event`).
    #[test]
    fn the_pause_after_resumed_speech_asks_again() {
        let setup = Setup { vad: true, stt: true, eot: true, ..Setup::default() };
        let vad = Ref::new(0);
        let silence = Input::VadInference {
            vad,
            probability: Confidence(5),
            speaking: false,
            duration: Millis(224),
        };
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad, speech_ms: Millis(300) },
            Input::Advance { ms: Millis(500) },
            // A pause mid-sentence: asked, and answered "likely".
            silence.clone(),
            Input::Advance { ms: Millis(10) },
            Input::EotAnswer {
                prediction: Ref::new(0),
                probability: Confidence(90),
                backchannel: None,
            },
            Input::Advance { ms: Millis(10) },
            // The user goes on.
            Input::VadInference {
                vad,
                probability: Confidence(90),
                speaking: true,
                duration: Millis(32),
            },
            Input::Advance { ms: Millis(800) },
            // The real end: asked again, answered "unlikely".
            silence,
            Input::Advance { ms: Millis(10) },
            Input::EotAnswer {
                prediction: Ref::new(1),
                probability: Confidence(5),
                backchannel: None,
            },
            Input::VadEnd { vad, silence_ms: Millis(250) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("is there a vegetarian option on the menu".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let judged: Vec<f64> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Emitted(SessionEvent::EotPrediction(ev)) => Some(ev.probability),
                _ => None,
            })
            .collect();
        assert_eq!(judged, [0.05], "judged by the last pause's answer, not the mid-sentence one");
    }
}

mod eot_judged_by_detector_thresholds {
    //! A prediction is judged against the detector's thresholds, not the mini model's stock table.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::fakes::EOT_THRESHOLD;
    use lk_agents_fuzz::input::{Confidence, Input, Millis, Ref, UserText};
    use lk_agents_voice::trace::{Step, TurnStep};
    use lk_agents_voice::{events::SessionEvent, trace::Step::Emitted};

    // Catches: the session judging every prediction against `ThresholdOptions::new(V1Mini, None,
    // None)` (en 0.36) whatever the detector held: `unlikely_threshold=0.001` was ignored, and a
    // question the mini model scored 0.34 waited the full `max_delay` on 7-10% of turns where the
    // reference replied at once (agents-bench, 2026-10-27, "Is there a vegetarian option on the
    // menu?").
    #[test]
    fn a_prediction_is_judged_by_the_detectors_threshold() {
        let setup = Setup { vad: true, stt: true, eot: true, ..Setup::default() };
        let vad = Ref::new(0);
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad, speech_ms: Millis(300) },
            Input::Advance { ms: Millis(1_500) },
            // 224 ms of silence: the detector is asked, its request registered a step later.
            Input::VadInference {
                vad,
                probability: Confidence(5),
                speaking: false,
                duration: Millis(224),
            },
            Input::Advance { ms: Millis(10) },
            // Under the detector's 0.5, over the stock table's 0.36.
            Input::EotAnswer {
                prediction: Ref::new(0),
                probability: Confidence(45),
                backchannel: None,
            },
            Input::VadEnd { vad, silence_ms: Millis(250) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("is there a vegetarian option on the menu".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let thresholds: Vec<f64> = journal
            .iter()
            .filter_map(|(_, step)| match step {
                Emitted(SessionEvent::EotPrediction(ev)) => Some(ev.threshold),
                _ => None,
            })
            .collect();
        assert_eq!(thresholds, [EOT_THRESHOLD], "judged once, by the detector's threshold");
        // 0.45 < 0.5: the countdown runs `max_delay`, so no turn ends within the second after.
        let ended =
            journal.iter().any(|(_, step)| matches!(step, Step::Turn(TurnStep::EndOfTurn { .. })));
        assert!(
            !ended,
            "an unlikely end of turn waited min_delay, as the stock table would have it"
        );
    }
}

mod late_mic_released {
    //! A microphone attached as the session closes is let go with it.
    use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
    use lk_agents_fuzz::input::{Input, Millis};

    // Catches: a microphone attached in the step before the close left in the resource table. The
    // core registers it and queues its source switch; the loop stopped before the runtime took the
    // switch, and the runtime's finish applied it to the forwarder alone, so the id it held was never
    // released (survey seed 2303, every_resource_released).
    #[test]
    fn a_mic_attached_at_the_close_is_released() {
        let inputs = vec![
            Input::Advance { ms: Millis(20) },
            Input::SetAudioInput { attach: true },
            Input::Close { drain: false },
        ];
        let leaked: Vec<u64> = (0..16)
            .filter(|&schedule| {
                let case = Case {
                    setup: Setup { stt: true, vad: true, ..Setup::default() },
                    inputs: inputs.clone(),
                    schedule,
                    random: Vec::new(),
                    policy: Policy::BasicPos,
                };
                run(case).failures.iter().any(|f| f.contract == "every_resource_released")
            })
            .collect();
        assert!(leaked.is_empty(), "schedules leaving the microphone in the table: {leaked:?}");
    }
}

mod manual_turn_closed_empty {
    //! A close commits a manual user turn even when it holds no words.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, TurnMode};
    use lk_agents_voice::events::SessionEvent;

    // Catches: the close's trailing commit skipped when the turn held no words. The reference's close
    // always runs `_commit_user_turn`, and in manual turn detection `_run_eou_detection` proceeds with
    // an empty transcript, so the history ends with the (empty) user turn, as in `parity.py
    // update_turn_detection`.
    #[test]
    fn a_close_commits_an_empty_manual_turn() {
        let setup = Setup { stt: true, turn: TurnMode::Manual, ..Setup::default() };
        let inputs = [
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let users: Vec<_> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == Role::User => Some(m.text_content()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(users.len(), 1, "{users:?}");
    }
}

mod mic_gate_reopens {
    //! The microphone gate the agent's speech closed reopens before the next frame.
    use lk_agents_fuzz::drive::{Case, Policy, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};

    // Catches: the gate reopening through the owner's queue, behind frames already queued, so frames
    // pushed as it reopened were silenced (`forwarded_while_enabled`, campaign seed 27243; a45633b37).
    // The gate's effects settle in their batch, before another frame is forwarded. The race is the
    // scheduler's, so 32 schedules of each queue policy run.
    #[test]
    fn the_microphone_is_heard_once_the_gate_reopens() {
        let mic = |chunks| Input::MicFrames { mic: Ref::new(0), chunks: Count(chunks) };
        let inputs = vec![
            Input::Say {
                text: AgentText("Your balance is due on Friday, and the next one in March.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(20) },
            // The agent starts speaking: the recogniser's gate closes until the echo canceller warms.
            Input::PlayoutStarted { segment: Ref::new(0) },
            mic(10),
            Input::Advance { ms: Millis(100) },
            mic(10),
            Input::Advance { ms: Millis(2_900) },
            // The gate reopens, and the person's frames arrive in the same instant.
            mic(10),
            Input::Advance { ms: Millis(100) },
            mic(10),
            Input::Advance { ms: Millis(100) },
        ];
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            let silenced: Vec<u64> = (0..32)
                .filter(|&schedule| {
                    let case = Case {
                        setup: Setup { stt: true, interruption: true, ..Setup::default() },
                        inputs: inputs.clone(),
                        schedule,
                        random: Vec::new(),
                        policy,
                    };
                    run(case).failures.iter().any(|f| f.contract == "forwarded_while_enabled")
                })
                .collect();
            assert!(
                silenced.is_empty(),
                "{policy:?}: schedules silencing the microphone: {silenced:?}"
            );
        }
    }
}

mod recogniser_failed_speaking {
    //! A recogniser that goes away mid-utterance ends the speech it began.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    fn advance() -> Input {
        Input::Advance { ms: Millis(20) }
    }

    /// The person starts speaking on the recogniser, `gone` takes the recogniser away, and a reply's
    /// tool then waits for idle (`ctx.foreground()`): the reply settles only if the person stopped.
    fn heard_out(gone: Vec<Input>) {
        let mut inputs = vec![Input::SttStart { stt: Ref::new(0) }, advance()];
        inputs.extend(gone);
        inputs.extend([
            advance(),
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            advance(),
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            advance(),
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AfterIdle,
                text: AgentText(String::new()),
            },
            advance(),
        ]);
        let setup = Setup { stt: true, ..Setup::default() };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"speech_settles"), "{failed:?}");
        assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
    }

    // Catches: the recogniser failing while the person spoke, so no end of speech ever came and the
    // user was speaking for the rest of the call: never idle, a tool's `foreground()` waited on it,
    // and the reply waiting on that tool ran until the close (`speech_settles`, survey seed 3036).
    #[test]
    fn a_failed_recogniser_does_not_leave_the_person_speaking() {
        heard_out(vec![Input::SttFail { stt: Ref::new(0) }]);
    }
}

mod stt_end_resumes_pause {
    //! Under STT turn detection, the STT's end of speech arms the resume of a speech an interim paused.
    use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Count, Input, Millis, Ref, SayInput, TurnMode, UserText,
    };
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: only a VAD's end of speech armed the false-interruption timer, so with an STT and no
    // VAD an interim that paused the agent left it paused for good (and the session's close waited on
    // it). The reference's STT `END_OF_SPEECH` is `on_end_of_speech(None)` in "stt" mode, which starts
    // `_start_false_interruption_timer`; found by `scripts/parity.py false_interruption`.
    #[test]
    fn an_stt_end_of_speech_resumes_the_paused_speech() {
        let setup = Setup {
            stt: true,
            pausable_output: true,
            turn: TurnMode::Stt,
            room_legs: Some(RoomLegs { pre_connect: None }),
            ..Setup::default()
        };
        let mut inputs = vec![
            Input::Advance { ms: Millis(100) },
            Input::Say {
                text: AgentText("A long answer.".into()),
                input: SayInput::Audio,
                allow: None,
                to_ctx: true,
            },
        ];
        // Four seconds of audio, fed as it plays: past the echo canceller's warm-up.
        for _ in 0..8 {
            inputs.push(Input::FeedAudio { stream: Ref::new(0), chunks: Count(50) });
            inputs.push(Input::Advance { ms: Millis(450) });
        }
        inputs.extend([
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("uh".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ]);
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let resumed: Vec<bool> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Emitted(SessionEvent::AgentFalseInterruption { resumed }) => Some(*resumed),
                _ => None,
            })
            .collect();
        assert_eq!(resumed, [true], "one false interruption, resumed");
    }
}

mod text_before_attach {
    //! Words reach a transcription output only once its binding attached it.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_fuzz::record::Sink;

    // Catches: an authorized synthesised segment forwarding its words straight to the transcription
    // output it captured, ahead of the activation that attaches it. With the voice down until the
    // output is replaced, the words written as the synthesis gives up reach an output that was never
    // attached (`output_replacement_is_clean`, survey seed 1560 and 17 more in the 32768 window). The
    // segment activates after the replacement, so all its words go to the replacement.
    #[test]
    fn a_failed_voice_writes_no_words_to_an_unattached_replaced_output() {
        let text = |t: &str| AgentText(t.into());
        simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in [
                Input::TtsOutage { down: true },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
                Input::Advance { ms: Millis(20) },
                Input::SetTranscriptionOutput { attach: true },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::Advance { ms: Millis(10_000) },
            ] {
                run.apply(&input).await;
            }
            // Output 1 is the replaced transcription output, output 2 its replacement.
            let text: Vec<_> = run
                .record
                .sink()
                .iter()
                .filter_map(|(_, call)| match call {
                    Sink::Text { output, text } => Some((*output, text.as_str())),
                    _ => None,
                })
                .collect();
            assert_eq!(text, [(2, "Hello there."), (2, " How are you?")]);
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod transcript_path_reopened {
    //! A transcript path reopened mid-speech forwards the later segments only, in order.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_voice::trace::{PlayoutStep, Step};

    // Catches: `text_conservation` requiring the forwarded words to be a prefix of the speech's text
    // even when the transcription output was detached for its first segments. The sink is re-read per
    // segment, so a reattach part-way shows what comes after it and skips what came before (survey
    // seed 5066: step 1 spoken unshown, the tool reply's step shown).
    #[test]
    fn a_reattached_path_shows_what_follows() {
        let inputs = [
            Input::SetTranscriptionOutput { attach: false },
            Input::SetAudioOutput { attach: false },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("First part.".into()) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::SetTranscriptionOutput { attach: true },
            Input::Advance { ms: Millis(20) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("ok".into()),
            },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("Second part.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(100) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let forwarded: String = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Playout(PlayoutStep::TranscriptForwarded { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(forwarded, "Second part.", "only the step after the reattach was shown");
        let (failures, _) = check_all(&run.record);
        let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"text_conservation"), "{failed:?}");
    }
}

mod unset_detection_stt_only {
    //! With an STT, no VAD and no turn detection asked for, turns end on finals and the STT's speech
    //! events leave the user state alone; under a realtime model the STT's words stay out of history.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, UserText};
    use lk_agents_voice::agent::Source;
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    fn utterance(text: &str) -> [Input; 4] {
        [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal { utterance: Ref::new(0), text: UserText(text.into()), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(3_000) },
        ]
    }

    /// (user state changes, user items committed from a turn) over the run.
    fn observed(setup: &Setup) -> (usize, usize) {
        let run = simulate(async {
            let mut run = Run::start(setup, 0, Default::default()).await;
            for input in &utterance("hello there") {
                run.apply(input).await;
            }
            run
        });
        let (mut states, mut users) = (0, 0);
        for (_, step) in run.record.journal().iter() {
            match step {
                Step::Emitted(SessionEvent::UserStateChanged { .. }) => states += 1,
                Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                    let user = matches!(&**item, ChatItem::Message(m) if m.role == Role::User);
                    if user && matches!(source, Source::UserTurn { .. }) {
                        users += 1;
                    }
                }
                _ => {}
            }
        }
        (states, users)
    }

    // Catches: the unset mode resolved to `stt`, so an STT-only session moved the user state on the
    // STT's start and end of speech, which the reference's VAD-based path (`_turn_detection_mode =
    // None`) never reads; found by `scripts/parity.py user_turn_default_detection`.
    #[test]
    fn an_stt_only_turn_ends_on_its_final() {
        let setup = Setup { stt: true, ..Setup::default() };
        assert_eq!(observed(&setup), (0, 1), "no user state change, one user item");
    }

    // Catches: a realtime turn pushed the STT's words to the server as an item, beside the audio it
    // had heard, and kept them as the user item. The reference ignores the STT transcription for a
    // realtime model and answers from the committed audio; found by
    // `scripts/parity.py realtime_stt_default_detection`.
    #[test]
    fn a_realtime_turn_keeps_no_stt_words() {
        let realtime = Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() });
        let setup = Setup { stt: true, realtime, ..Setup::default() };
        assert_eq!(observed(&setup), (0, 0), "no user state change, no user item");
    }
}
