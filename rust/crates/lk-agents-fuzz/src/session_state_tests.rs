//! Lifecycle regression tests.

mod close_ends_its_tasks {
    //! A closing session's tasks end within its cleanup bound: its close cancels nothing still running.
    use lk_agents_fuzz::drive::{Case, Outcome, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Models, Ref, SayInput};

    // Catches: an interrupted `say` whose caller's text stream stays open went on reading it, and its
    // voice, its synthesis failed, went on waiting for the words, so the speech's task outlived the
    // close's release and was cancelled at the cleanup bound (`every_task_finishes`, 82 of the first
    // 1,024 survey seeds once the DST ran the runtime's real close, e.g. seed 15). `_tts_task`
    // cancels its forwarding at the interruption.
    #[test]
    fn an_interrupted_say_stops_reading_its_stream() {
        let outcome = closed(
            Setup { tts: true, ..Setup::default() },
            vec![
                Input::TtsOutage { down: true },
                Input::Say {
                    text: AgentText(String::new()),
                    input: SayInput::TextStream,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedText {
                    stream: Ref::new(0),
                    text: AgentText("One moment please.".into()),
                },
                Input::Advance { ms: Millis(2_000) },
            ],
        );
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }

    // Catches: a provider stream a handoff was still opening when the session closed kept its open
    // running, which nothing would take any more, past the close's cleanup bound (survey seed 313).
    #[test]
    fn a_close_abandons_a_stream_still_opening() {
        let outcome = closed(
            Setup { stt: true, connect_ms: Some(30_000), ..Setup::default() },
            vec![Input::UpdateModels { models: Models::All }, Input::Advance { ms: Millis(20) }],
        );
        assert!(outcome.ok(), "{:#?}", outcome.failures);
    }

    /// `inputs`, then the close, waited out before anything still pending is released.
    fn closed(setup: Setup, mut inputs: Vec<Input>) -> Outcome {
        inputs.extend([
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20_000) },
            Input::AwaitExit {},
        ]);
        run(Case { setup, inputs, ..Case::default() })
    }
}

mod console_cut_position {
    //! Elapsed time cannot establish that the console played queued audio.
    use std::sync::Arc;
    use std::time::Duration;

    use lk_agents_abi::voice::console::ConsolePort;
    use lk_agents_abi::voice::io::AudioOutputExt;
    use lk_agents_abi::{AudioFrame, Clock};
    use lk_agents_fuzz::fakes::FakeCli;
    use lk_agents_runtime::net::clock::SimClock;
    use lk_agents_runtime::net::task::TokioSpawner;

    // No transport has delivered this PCM to the CLI. A clock advance alone must not commit it as
    // heard; a late unkeyed acknowledgement must not borrow the next operation's identity either.
    #[tokio::test(start_paused = true)]
    async fn a_cut_console_segment_reports_only_confirmed_audio() {
        let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
        let (console, _cli) = FakeCli::connect(Arc::new(TokioSpawner::current()), clock, None);
        let output = console.clone().as_audio_output();
        let second = AudioFrame::new(vec![0; 96_000], 48_000, 1, 48_000);
        output.capture_frame(&second).await.unwrap();
        let first = output.tracker().current_segment().unwrap();
        tokio::time::advance(Duration::from_millis(300)).await;
        output.cut();
        let cut = output.wait_for_playout().await;
        assert!(cut.interrupted);
        assert_eq!(cut.playback_position, Duration::ZERO);
        assert!(!output.tracker().has_pending_segment(first));

        output.capture_frame(&second).await.unwrap();
        let next = output.tracker().current_segment().unwrap();
        assert_ne!(first, next);
        console.as_console().playout_finished();
        assert!(output.tracker().has_pending_segment(next));
        output.cut();
        let cut = output.wait_for_playout().await;
        assert!(cut.interrupted);
        assert_eq!(cut.playback_position, Duration::ZERO);
    }
}

mod cut_unvoiced_segment_stays_closed {
    //! A speech cut after its segment completed unvoiced does not end that segment's audio again.
    use lk_agents_fuzz::drive::{NodeSetup, Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};
    use lk_agents_voice::trace::{InterruptStep, PlayoutStep, SinkCall, Step};

    // Catches: a segment whose voice ended with no audio was cut by its speech's interruption (the
    // cut closed its words), then ended its audio leg as an unvoiced one anyway; the synchroniser
    // opened a segment for that audio end and showed the next speech's words whole before any of
    // their audio played (`transcript_follows_audio`, campaign seed 19354; 242c57d69).
    #[test]
    fn the_next_speech_words_wait_for_their_audio() {
        let text = |t: &str| AgentText(t.into());
        // A `tts_node` and no TTS: a reply's words are shown with no audio.
        let setup = Setup {
            tts: false,
            sync_transcription: true,
            nodes: Some(NodeSetup {
                llm: false,
                tts: true,
                stt: false,
                llm_text: None,
                transcription: true,
            }),
            ..Setup::default()
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(0), text: text("Okay. I've moved you to Friday.") },
                Input::Advance { ms: Millis(500) },
                Input::Interrupt { force: true, wait: false },
                Input::Advance { ms: Millis(500) },
                Input::Say {
                    text: text("Anything else?"),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::Advance { ms: Millis(20) },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(20) },
                Input::FeedEnd { stream: Ref::new(0) },
                Input::Advance { ms: Millis(5_000) },
            ] {
                run.apply(&input).await;
            }
            let at = |step: fn(&Step) -> bool| {
                run.record.journal().iter().position(|(_, s)| step(s)).expect("the scenario")
            };
            let forwarded =
                at(|s| matches!(s, Step::Playout(PlayoutStep::TranscriptForwarded { .. })));
            let cut = at(|s| matches!(s, Step::Interrupt(InterruptStep::Interrupted { .. })));
            let pushed = at(|s| {
                matches!(
                    s,
                    Step::Playout(PlayoutStep::SinkCalled { call: SinkCall::Push { .. }, .. })
                )
            });
            assert!(
                forwarded < cut && cut < pushed,
                "the scenario: words with no audio cut, then audio"
            );
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod determinism {
    //! The simulation must be reproducible: the same case twice must leave the same recording.
    //!
    //! This is a guard, not a demonstration. Tokio randomises the branch order of an unbiased
    //! `select!`, so concurrent generations used to race differently on every run and a finding landed
    //! on a different speech each time. `drive::simulate_with` pins that RNG and gives each case its own
    //! runtime. Anything that reintroduces real nondeterminism -- a `HashMap` iterated for ordering,
    //! wall-clock time read as data, a fresh `select!` outside a seeded runtime -- fails here first,
    //! with a diff, instead of showing up months later as a seed that no longer reproduces its run.
    use fastrand::Rng;
    use lk_agents_fuzz::{
        drive::{Run, simulate_with},
        record::Recording,
        search,
    };

    /// Wall-clock stamps the product puts on its own data, and the one hash that covers them.
    ///
    /// Everything a rule can read, flattened. Two runs of one case must agree on all of it.
    fn fingerprint(r: &Recording) -> String {
        let mut out = String::new();
        for (at, step) in r.journal() {
            out.push_str(&format!("J {} {:?}\n", at.seq, step));
        }
        for o in r.opened() {
            out.push_str(&format!(
                "O {} {:?}#{} {:?} {:?}\n",
                o.at.seq, o.kind, o.nth, o.id, o.detail
            ));
        }
        for (at, host) in r.host() {
            out.push_str(&format!("H {} {:?}\n", at.seq, host));
        }
        for (at, nth, answer) in r.results() {
            out.push_str(&format!("R {at:?} {nth} {answer:?}\n"));
        }
        for (at, sink) in r.sink() {
            out.push_str(&format!("S {} {:?}\n", at.seq, sink));
        }
        for (id, handle) in r.handles() {
            out.push_str(&format!("P {id} {handle:?}\n"));
        }
        out
    }

    fn rerun(case: &lk_agents_fuzz::drive::Case) -> String {
        simulate_with(case.schedule, async {
            let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
            for input in &case.inputs {
                run.apply(input).await;
            }
            format!("{}D {}\n", fingerprint(&run.record), run.dispatch_order())
        })
    }

    /// A case the search explored, ending included, leaves the same recording twice over, down to
    /// the last journal step.
    #[test]
    fn an_explored_case_reruns_identically() {
        let mut bad = 0;
        for seed in 0..6 {
            let case =
                search::explore(search::sample_setup(&mut Rng::with_seed(seed)), seed, 600).case;
            let (first, second) = (rerun(&case), rerun(&case));
            if first != second {
                bad += 1;
                let at = first
                    .lines()
                    .zip(second.lines())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| first.lines().count().min(second.lines().count()));
                let line = |text: &str| -> String {
                    text.lines().nth(at).unwrap_or("<end>").chars().collect()
                };
                println!(
                    "NOT REPRODUCIBLE seed {seed} line {at}:\n  {}\n  {}",
                    line(&first),
                    line(&second)
                );
            }
            // A fingerprint that is nearly empty would make this test pass without checking anything.
            assert!(
                first.lines().count() > 50,
                "seed {seed}: recording too small to be a real check"
            );
        }
        assert_eq!(bad, 0, "{bad} explored cases are not reproducible");
    }

    /// The same seed explores the same case, so a campaign is repeatable end to end. Setup included:
    /// the survey samples one per seed, and a case is both halves.
    #[test]
    fn the_same_seed_explores_the_same_case() {
        let setup = |seed| search::sample_setup(&mut Rng::with_seed(seed));
        assert_eq!(setup(7), setup(7), "one seed sampled two different setups");
        let first = search::explore(setup(7), 7, 300);
        let second = search::explore(setup(7), 7, 300);
        assert_eq!(first.case.inputs, second.case.inputs, "one seed sampled two different cases");
        assert_eq!(
            first.failures.iter().map(|f| (&f.contract, &f.subject)).collect::<Vec<_>>(),
            second.failures.iter().map(|f| (&f.contract, &f.subject)).collect::<Vec<_>>(),
            "one seed produced two different verdicts"
        );
    }
}

mod failed_step_releases_floor {
    //! A step whose model failed gives the floor back before waiting for its tools.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Catches: a step whose model request failed after dispatching a tool kept its speech current
    // while it waited for the tool, so a tool asking `ctx.wait_for_idle()` waited on its own speech
    // until the close (`speech_settles`, survey seed 4514). The reference marks the generation done
    // before awaiting the tools (`_pipeline_reply_task_impl`, `_mark_generation_done`).
    #[test]
    fn a_tool_waiting_for_idle_outlives_its_failed_step() {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmFail { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::AfterIdle,
                    text: AgentText(String::new()),
                },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            assert!(run.record.applied().iter().all(|a| a.accepted), "every input landed");
            run
        });
        let speeches = run.record.host_view().speeches();
        let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
        assert!(speech.is_done(), "the speech waits on a tool that waits on it");
    }
}

mod final_ends_pause {
    //! A final transcript ends the pause it took: the paused speech is interrupted.
    use lk_agents_fuzz::drive::{Case, RoomLegs, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput, UserText};

    fn final_(text: &str) -> Input {
        Input::SttFinal { utterance: Ref::new(0), text: UserText(text.into()), speaker: None }
    }

    // Catches: a final transcript paused the speaking agent while a drain held scheduling, so the
    // user turn was dropped without ending the pause and no false-interruption timer resumed it: the
    // speech stayed paused until the close (`speech_settles`, survey seed 2068). The reference's
    // `on_final_transcript` always ends the pause with `_cancel_speech_pause`, interrupting it.
    #[test]
    fn a_final_transcript_interrupts_the_speech_it_paused() {
        let setup = Setup {
            stt: true,
            pausable_output: true,
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
            Input::Drain { wait: false },
            Input::SttStart { stt: Ref::new(0) },
            final_("no no"),
            Input::Advance { ms: Millis(100) },
            final_("not that one, the other one"),
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ]);
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"speech_settles"), "{failed:?}");
        assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
    }
}

mod final_over_paused_no_preemptive {
    //! A final that cuts a paused speech asks for no preemptive generation: the speech was live.
    use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Confidence, Count, Input, Millis, Ref, SayInput, UserText,
    };
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the final's interruption of the paused speech landed before its preemptive generation
    // was decided, so a preemptive reply was planned (an extra LLM request) for a turn the agent was
    // still talking over. The reference's `on_final_transcript` cuts the pause in a task, after
    // `on_preemptive_generation` refused for the live speech; found by
    // `scripts/parity.py vad_interruption`.
    #[test]
    fn a_final_over_a_paused_speech_plans_no_preemptive_reply() {
        let setup = Setup {
            vad: true,
            stt: true,
            preemptive: true,
            pausable_output: true,
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
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(90),
                speaking: true,
                duration: Millis(800),
            },
            Input::Advance { ms: Millis(50) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("wait stop".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(50) },
        ]);
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let created = run
            .record
            .journal()
            .iter()
            .filter(|(_, step)| matches!(step, Step::Emitted(SessionEvent::SpeechCreated { .. })))
            .count();
        assert_eq!(created, 1, "only the say: the user is still speaking, no reply is planned yet");
    }
}

mod handoff_refused_releases {
    //! The providers a refused handoff hands over are released.
    use lk_agents_fuzz::drive::{Case, NodeSetup, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input};

    // Catches: `update_agent` registering the frame's providers before asking for its answer, so a
    // handoff refused because the session had ended left them in the table for good
    // (`every_resource_released`, survey seed 66). The host gives every leg on every handoff
    // (`_frame`), so each late handoff leaked three.
    #[test]
    fn a_handoff_refused_after_the_end_releases_its_providers() {
        let setup = Setup {
            stt: true,
            nodes: Some(NodeSetup {
                llm: true,
                tts: true,
                stt: true,
                llm_text: None,
                transcription: false,
            }),
            ..Setup::default()
        };
        let inputs = vec![
            Input::Close { drain: false },
            Input::AwaitExit {},
            Input::Handoff {
                agent: "late".into(),
                instructions: AgentText("Hi.".into()),
                tools: Vec::new(),
                turn: None,
            },
        ];
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"every_resource_released"), "{failed:?}");
    }
}

mod held_turn_idle {
    //! A turn held for the agent a handoff brings in does not hold idle off.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };

    // Catches: a user turn decided while a handoff drained was held for the arriving agent and held
    // idle off, while the drain waited on a speech whose tool waited on idle: nothing moved until the
    // close (`speech_settles`, survey seeds 1923, 3526, 4514). The reference has no held turn
    // (`on_end_of_turn` skips it while scheduling is paused), so it never blocks `wait_for_idle`.
    #[test]
    fn a_tool_waiting_for_idle_lets_a_drain_holding_a_turn_finish() {
        let setup = Setup {
            tools: vec!["lookup".into()],
            stt: true,
            turn: TurnMode::Manual,
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::Handoff {
                    agent: "next".into(),
                    instructions: AgentText("You are next.".into()),
                    tools: vec![],
                    turn: None,
                },
                Input::Advance { ms: Millis(20) },
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
                    skip_reply: false,
                },
                // Past the endpointing delay: the turn is decided, and held for the arriving agent.
                Input::Advance { ms: Millis(1_000) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::AfterIdle,
                    text: AgentText(String::new()),
                },
                Input::Advance { ms: Millis(200) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let speeches = run.record.host_view().speeches();
        let first = speeches.first().expect("the reply");
        assert!(first.is_done(), "the speech waits on a tool that waits on a held turn");
        assert_eq!(speeches.len(), 2, "the held turn reached the arriving agent and was answered");
    }
}

mod hold_at_close {
    //! A floor hold the host never released ends at the close's hard edge.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref};
    use lk_agents_voice::trace::{CloseStep, SpeechStep, Step};

    // Catches: a `wait_for_idle_and_hold()` granted before the session closes outliving it. Past the
    // edge `ReleaseIdleHold` is dropped and nothing else released the hold (soak seed 6). The late
    // release must stay harmless.
    #[test]
    fn a_hold_ends_with_the_session() {
        let inputs = [
            Input::WaitForIdleAndHold {},
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(6_000) },
            Input::ReleaseIdleHold { hold: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let granted = journal
            .iter()
            .find_map(|(_, s)| match s {
                Step::Speech(SpeechStep::IdleAnswered { hold: Some(hold), .. }) => Some(*hold),
                _ => None,
            })
            .expect("the hold was granted");
        let closing = journal
            .iter()
            .position(|(_, s)| matches!(s, Step::Close(CloseStep::Requested { .. })))
            .expect("the close was requested");
        let released: Vec<_> = journal
            .iter()
            .enumerate()
            .filter_map(|(i, (_, s))| match s {
                Step::Speech(SpeechStep::IdleHoldReleased { hold }) => Some((i > closing, *hold)),
                _ => None,
            })
            .collect();
        assert_eq!(released, [(true, granted)], "released once, by the close");
    }
}

mod idle_after_untranscribed_turn {
    //! A user turn that ended with no words to decide on does not hold `wait_for_idle()` off.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode};
    use lk_agents_fuzz::record::Answer;

    // Catches: the recogniser's `Ended` phase (speech heard, no countdown started) read as a turn
    // being decided. With no transcript coming, idle was never reached, and a tool waiting for idle
    // inside its own speech waited forever while the speech waited on the tool (survey seed 1678,
    // `session_settles`). The reference's `wait_for_idle` waits only on `_speaking` and a running
    // `_end_of_turn_task`.
    #[test]
    fn idle_is_reached_after_a_turn_with_no_words() {
        let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
        let inputs = [
            Input::Advance { ms: Millis(20) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::Advance { ms: Millis(300) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(500) },
            Input::WaitForIdle { hold: None },
            Input::Advance { ms: Millis(100) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let answers: Vec<_> = run
            .record
            .host_view()
            .api_calls()
            .iter()
            .filter(|c| matches!(c.input(), Input::WaitForIdle { .. }))
            .map(|c| c.returned().map(|(_, a)| a.clone()))
            .collect();
        assert_eq!(answers, [Some(Answer::IdleOk)]);
    }
}

mod idle_while_closing {
    //! Idle reached during a close's drain grants no floor hold: the waiter is refused as closing.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::trace::{CloseStep, SpeechStep, Step};

    // Catches: `settle_waiters` admitting idle waiters while the session closes. The reference's
    // `wait_for_idle` raises `ActivityClosedError` once closing; here the drain's last speech ending
    // granted a `wait_for_idle_and_hold()` its hold, which nothing could release past the edge, so it
    // outlived the session (soak seed 0).
    #[test]
    fn idle_during_a_close_grants_no_hold() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
            Input::WaitForIdleAndHold {},
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: true },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
        ];
        let setup = Setup { hook_bound_ms: Some(1_000), ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let closing = journal
            .iter()
            .position(|(_, s)| matches!(s, Step::Close(CloseStep::Requested { .. })))
            .expect("the close was requested");
        let answers: Vec<_> = journal[closing..]
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Speech(SpeechStep::IdleAnswered { hold, closing }) => Some((*hold, *closing)),
                _ => None,
            })
            .collect();
        assert_eq!(answers, [(None, true)], "the waiter is refused, not granted");
    }
}

mod lane_lost_call_goes_on {
    //! A lost voice-lane lease never ends the call: what the agent says after it is heard.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, UserText};
    use lk_agents_voice::trace::{SpeechStep, Step};

    // Catches: a lost lease cancelled the speech under it and every later one at its creation, each
    // announced `Completed` with nothing said, and a speech whose commit was answered after the loss
    // never finished (`speech_is_heard`, `no_premature_completion`, survey seeds 12, 341, 615;
    // 62855980b). voice-session.md: storage failures are logged and never end the call.
    #[test]
    fn the_agent_is_heard_after_its_lane_is_lost() {
        // The store loses the lease with its answer to the first turn.
        let setup = Setup { storage_lost_at: Some(1), ..Setup::default() };
        let reply = |chat: u32, said: &str| {
            [
                Input::GenerateReply {
                    user_input: Some(UserText("hello?".into())),
                    instructions: None,
                    allow: None,
                },
                Input::Advance { ms: Millis(20) },
                Input::LlmText { chat: Ref::new(chat), text: AgentText(said.into()) },
                Input::LlmEnd { chat: Ref::new(chat) },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutStarted { segment: Ref::new(chat) },
                Input::PlayoutProgressed { segment: Ref::new(chat), ms: Millis(5_000) },
                Input::PlayoutFinished { segment: Ref::new(chat) },
                Input::Advance { ms: Millis(1_000) },
            ]
        };
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in reply(0, "Hi, how can I help?").into_iter().chain(reply(1, "Sure.")) {
                run.apply(&input).await;
            }
            let journal: Vec<&Step> = run.record.journal().iter().map(|(_, s)| s).collect();
            let lost = journal
                .iter()
                .position(|s| matches!(s, Step::Speech(SpeechStep::LaneLost { .. })))
                .expect("the scenario: the lane's lease is lost mid-call");
            assert!(
                journal[lost..]
                    .iter()
                    .any(|s| matches!(s, Step::Speech(SpeechStep::AgentStartedSpeaking { .. }))),
                "the agent speaks after the loss"
            );
            let outcome = finish(run, Outcome::default()).await;
            assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
        });
    }
}

mod late_leg_open_closed {
    //! A recognition stream that finishes opening after the session ended is closed.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{Input, Millis, Models};

    // Catches: a reopened leg whose connection landed after the loop stopped was dropped with its
    // event, never reaching the provider's `close` (`provider_streams_closed`, survey seed 1876).
    #[test]
    fn a_stream_opened_after_the_close_is_closed() {
        let inputs = vec![
            Input::UpdateModels { models: Models::Stt },
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: false },
        ];
        let setup = Setup { stt: true, connect_ms: Some(15_000), ..Setup::default() };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"provider_streams_closed"), "{failed:?}");
    }
}

mod late_verdict_pause_resumes {
    //! A speech paused by a detector verdict that came after the person stopped resumes.
    use lk_agents_fuzz::drive::{Case, Outcome, RealtimeSetup, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{Count, Input, Millis, Ref};

    // Catches: the adaptive detector's verdict landing after the person's end of speech paused the
    // playing speech, and nothing ever armed its resume: the end of speech that arms it had already
    // come. With no reply to the turn to interrupt it (scheduling paused by a drain), the agent sat
    // silent mid-sentence for the rest of the call (`pause_resolves`, survey seed 273). The
    // reference's transcript paths arm the resume for a pause decided after the end of speech; every
    // pause decided with the person quiet now does.
    #[test]
    fn a_late_verdicts_pause_resumes() {
        // A realtime model with the session's own detector, recogniser and adaptive interruption.
        let setup = Setup {
            realtime: Some(RealtimeSetup::default()),
            stt: true,
            vad: true,
            interruption: true,
            pausable_output: true,
            ..Setup::default()
        };
        let outcome = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            // Warm up echo cancellation on a completed, observed first reply. The tested reply
            // then gets its overlap before its own missing-completion deadline can expire.
            run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
                .await;
            crate::playout::apply_with_playout(&mut run, &Input::Advance { ms: Millis(10_000) })
                .await;
            let next_segment = run.record.live().opened(lk_agents_fuzz::record::Kind::Segment);
            let inputs = vec![
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(2_500) },
                Input::PlayoutStarted { segment: Ref::new(next_segment) },
                Input::Advance { ms: Millis(500) },
                Input::Drain { wait: false },
                Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
                Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
                Input::Advance { ms: Millis(300) },
                Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
                Input::Advance { ms: Millis(100) },
                // The detector rules after the person has stopped.
                Input::OverlapVerdict {
                    port: Ref::new(0),
                    is_interruption: true,
                    agent_ended: false,
                },
                Input::Advance { ms: Millis(10_000) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            finish(
                run,
                Outcome { case: Case { setup, inputs, ..Case::default() }, ..Outcome::default() },
            )
            .await
        });
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"pause_resolves"), "{:#?}", outcome.failures);
        assert!(outcome.expectations.get("pause_resolves").is_some_and(|n| *n > 0), "no pause");
    }
}

mod left_at_close {
    //! What a closing session cut short ends with it: a tool body it answered for, a detached tool,
    //! a hook it issued.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::record::{Host, Kind};
    use lk_agents_voice::agent_session::HookKind;
    use lk_agents_voice::events::{SessionEvent, ToolEndStatus, ToolLifecycle};
    use lk_agents_voice::trace::{HandoffStep, HookEnd, Step};

    fn run(setup: &Setup, inputs: &[Input]) -> Run {
        simulate(async {
            let mut run = Run::start(setup, 0, Default::default()).await;
            for input in inputs {
                run.apply(input).await;
            }
            run
        })
    }

    // Catches: a blocking tool the close's bound answered with "session closing" left running at the
    // host. The reference's close cancels its tool tasks (`_ToolExecutor.aclose`); here the body
    // outlived the session and took a late answer (survey seed 189).
    #[test]
    fn a_tool_the_close_answered_is_cancelled() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            },
        ];
        let run = run(&Setup::default(), &inputs);
        assert!(run.record.driver_finished(), "the session closed");
        let answered = run.record.applied().last().expect("the late answer was applied");
        assert!(!answered.accepted, "the body was cancelled with the session");
        assert!(!run.record.live().has(Kind::Tool, 0), "nothing of the tool is left");
    }

    // Catches: a tool that detached (`ctx.update()`) still running at the host after the session
    // closed: the reference's close cancels it (`_ToolExecutor.aclose`) and tells its end, cancelled,
    // before the session's `close`.
    #[test]
    fn a_detached_tool_is_cancelled_before_the_close() {
        let inputs = [
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
                answer: ToolAnswer::Detached,
                text: AgentText("Looking it up.".into()),
            },
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(20_000) },
        ];
        let run = run(&Setup::default(), &inputs);
        assert!(run.record.driver_finished(), "the session closed");
        assert!(!run.record.live().has(Kind::Tool, 0), "the body was cancelled");
        let host = run.record.host();
        let ended: Vec<usize> = (0..host.len())
            .filter(|&i| {
                matches!(
                    &host[i].1,
                    Host::Session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended { .. }))
                )
            })
            .collect();
        let [ended] = ended[..] else { panic!("one end told, got {}", ended.len()) };
        let Host::Session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
            id,
            call_id,
            status,
            ..
        })) = &host[ended].1
        else {
            unreachable!()
        };
        assert_eq!(*status, ToolEndStatus::Cancelled);
        assert_eq!(*id, format!("{call_id}_final"), "the end of a call that detached");
        let close = host
            .iter()
            .position(|(_, e)| matches!(e, Host::Session(SessionEvent::Close { .. })))
            .expect("the session's close");
        assert!(ended < close, "the end is told before the session's close");
    }

    // Catches: a cut step's `function_tools_executed` issued past the edge, after the edge abandoned
    // every hook, still waited on when the loop stopped (survey seeds 15, 335).
    #[test]
    fn a_hook_issued_past_the_edge_ends_with_the_session() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "unknown".into(),
                args: ToolArgs("{}".into()),
            },
            Input::Advance { ms: Millis(20) },
            Input::Close { drain: true },
            Input::Advance { ms: Millis(40_000) },
        ];
        let setup = Setup { hook_bound_ms: Some(10_000), ..Setup::default() };
        let run = run(&setup, &inputs);
        assert!(run.record.driver_finished(), "the session closed");
        let journal = run.record.journal();
        let issued = journal
            .iter()
            .find_map(|(_, s)| match s {
                Step::Handoff(HandoffStep::HookIssued {
                    hook,
                    kind: HookKind::FunctionToolsExecuted,
                    ..
                }) => Some(*hook),
                _ => None,
            })
            .expect("the cut step asked its handlers");
        let ended: Vec<_> = journal
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Handoff(HandoffStep::HookFinished { hook, how, .. }) if *hook == issued => {
                    Some(*how)
                }
                _ => None,
            })
            .collect();
        assert_eq!(ended, [HookEnd::Abandoned], "abandoned once, by the close");
    }
}
