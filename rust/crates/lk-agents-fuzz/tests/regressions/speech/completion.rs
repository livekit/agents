mod speech_announced_before_user_item {
    //! `generate_reply(user_input=)` announces its speech, user-initiated, before the user item.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, UserText};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the user item committed before `speech_created`, which a listener pairing the two saw
    // in the wrong order.
    #[test]
    fn the_speech_is_announced_before_the_user_item() {
        let setup = Setup::default();
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let reply = Input::GenerateReply {
                user_input: Some(UserText("hello there".into())),
                instructions: None,
                allow: None,
            };
            for input in [reply, Input::Advance { ms: Millis(20) }] {
                run.apply(&input).await;
            }
            run
        });
        let seen: Vec<&'static str> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Emitted(SessionEvent::SpeechCreated { user_initiated, .. }) => {
                    assert!(user_initiated, "a generate_reply speech is user-initiated");
                    Some("speech_created")
                }
                Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                    ChatItem::Message(m) if m.role == Role::User => Some("user_item"),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(seen, ["speech_created", "user_item"]);
    }
}

mod reply_cut_before_its_request {
    //! A reply interrupted before its step runs makes no model request, and says so.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref};
    use lk_agents_voice::trace::{GenerationStep, Step};

    // Catches: the step journalling `ChatRequested` and reporting its request started before
    // polling the open an interrupt then dropped, so a request that never reached the model was on
    // the record and the next reply's failure was charged to the cut one (survey seed 304).
    #[test]
    fn a_reply_cut_before_its_request_requests_nothing() {
        let setup = Setup::default();
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::InterruptSpeech { speech: Ref::new(0), force: true },
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(20) },
                Input::LlmFail { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
            ] {
                run.apply(&input).await;
            }
            run
        });
        let next = run.record.host_view().speeches()[1].id;
        let requested: Vec<u64> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Generation(GenerationStep::ChatRequested { speech, .. }) => Some(speech.0),
                _ => None,
            })
            .collect();
        assert_eq!(requested, [next], "only the live reply asked its model");
        assert_eq!(run.record.inference_view().request_for(0), Some(next));
        let (failures, checked) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(checked.get("provider_failure_is_accounted"), Some(&1));
    }
}

mod step_cut_text_conserved {
    //! A host's unforced `interrupt()` during playout ends the whole speech: the cut step loses its
    //! words, the tool's answer is committed, and no follow-up is asked for.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Catches: `text_conservation` holding a speech that completed to every word its steps were
    // given, though the host's unforced `interrupt()` stops only the playing step (its words are
    // never forwarded) and the tool's follow-up then completes the speech (survey seed 2750).
    #[test]
    fn a_cut_step_does_not_owe_its_words() {
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
                Input::Advance { ms: Millis(20) },
                Input::InterruptSpeech { speech: Ref::new(0), force: false },
                // Words of the cut step: never forwarded.
                Input::LlmText {
                    chat: Ref::new(0),
                    text: AgentText("Your balance is due on the fourteenth.".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::Value,
                    text: AgentText("done".into()),
                },
                Input::Advance { ms: Millis(100) },
                Input::LlmEnd { chat: Ref::new(1) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let speeches = run.record.host_view().speeches();
        assert!(speeches[0].is_interrupted(), "the host's interrupt ends the speech");
        assert!(
            run.record.internal_view().speech(speeches[0].id).interrupted_at().is_some(),
            "its first step was cut"
        );
        let requests = run.record.inference_view().requests_for(speeches[0].id);
        assert_eq!(requests.len(), 1, "no follow-up after the interrupt: {requests:?}");
        let (failures, _) = check_all(&run.record);
        let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
        assert!(!failed.contains(&"text_conservation"), "{failures:?}");
    }
}

mod unheard_reply_not_committed {
    //! A reply interrupted before any of its audio was heard leaves no words behind.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::events::SessionEvent;

    // Catches: a segment whose synthesis ended interrupted (never opened, or cut while its open was
    // being retried) taking the lost-voice fallback, which forwards the segment's words as text and
    // counts them heard, so the reply was committed whole though not a word of it played.
    #[test]
    fn an_unheard_reply_is_not_committed() {
        let said = assistant_words(&[
            // Every synthesis fails at the open, so the segment sits in the retry until it is cut.
            Input::TtsOutage { down: true },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText {
                chat: Ref::new(0),
                text: AgentText("Your order shipped. Would you like a receipt?".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(5000) },
        ]);
        assert!(said.iter().all(|t| t.is_empty()), "committed unheard words: {said:?}");
    }

    fn assistant_words(inputs: &[Input]) -> Vec<String> {
        let run = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
            for input in inputs {
                run.apply(input).await;
            }
            run
        });
        run.record
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
            .collect()
    }
}

mod uninterruptible_drops_turn {
    //! A user turn ending while an uninterruptible speech plays is dropped: no user item, no reply.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, UserText};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the turn queued a reply behind the speech, which then answered words the agent had
    // talked over.
    #[test]
    fn a_turn_over_an_uninterruptible_speech_is_dropped() {
        let setup = Setup { stt: true, ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::Say {
                    text: AgentText("A long answer that cannot be cut.".into()),
                    input: SayInput::Words,
                    allow: Some(false),
                    to_ctx: true,
                },
                Input::Advance { ms: Millis(20) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::Advance { ms: Millis(200) },
                Input::SttStart { stt: Ref::new(0) },
                Input::SttFinal {
                    utterance: Ref::new(0),
                    text: UserText("stop please".into()),
                    speaker: None,
                },
                Input::SttEnd { utterance: Ref::new(0) },
                Input::Advance { ms: Millis(3_000) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let mut speeches = 0;
        let mut user_items = 0;
        for (_, step) in run.record.journal().iter() {
            match step {
                Step::Emitted(SessionEvent::SpeechCreated { .. }) => speeches += 1,
                Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => {
                    if matches!(&**item, ChatItem::Message(m) if m.role == Role::User) {
                        user_items += 1;
                    }
                }
                _ => {}
            }
        }
        assert_eq!((speeches, user_items), (1, 0), "only the say(), and no user item");
    }
}

mod unvoiced_synthesis_no_item {
    //! A speech whose voice failed before any of its audio played commits no assistant item.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_core::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the failed synthesis's words committed as an interrupted assistant message the caller
    // never heard.
    #[test]
    fn an_unvoiced_speech_commits_nothing() {
        let setup = Setup { sync_transcription: true, ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let inputs = [
                Input::TtsOutage { down: true },
                Input::Say {
                    text: AgentText("This will not be heard.".into()),
                    input: SayInput::Words,
                    allow: None,
                    to_ctx: true,
                },
                Input::Advance { ms: Millis(30_000) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let replies: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                    ChatItem::Message(m) if m.role == Role::Assistant => m.text_content(),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert!(replies.is_empty(), "{replies:?}");
    }
}

mod voice_task {
    //! `VoiceTask.run()` from a tool: the task's agent takes the floor, its tool's validated
    //! `Completed` returns to the parent's tool, and the parent is back without its hooks; a nested
    //! task, a cancelled await and a close mid-task end their tasks the same way.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::fakes::completion_value;
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::record::{Answer, Detail, Kind, Recording};
    use lk_agents_voice::agent_session::machine::activity::voice_task::VoiceTaskEnded;
    use lk_agents_voice::trace::{HandoffStep, Step};

    fn call(chat: u32) -> [Input; 3] {
        [
            Input::LlmToolCall {
                chat: Ref::new(chat),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(chat) },
            Input::Advance { ms: Millis(50) },
        ]
    }

    fn say(chat: u32) -> [Input; 3] {
        [
            Input::LlmText { chat: Ref::new(chat), text: AgentText("What is your email?".into()) },
            Input::LlmEnd { chat: Ref::new(chat) },
            Input::Advance { ms: Millis(50) },
        ]
    }

    fn answer(tool: u32, answer: ToolAnswer) -> [Input; 2] {
        [
            Input::ToolAnswered { tool: Ref::new(tool), answer, text: AgentText("done".into()) },
            Input::Advance { ms: Millis(50) },
        ]
    }

    fn task(tool: u32, agent: &str) -> [Input; 2] {
        [
            Input::VoiceTask { tool: Ref::new(tool), agent: agent.into(), reply: true },
            Input::Advance { ms: Millis(50) },
        ]
    }

    fn play(inputs: Vec<Input>) -> Recording {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let run = simulate(async move {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run.apply(&Input::Close { drain: false }).await;
            run.apply(&Input::Advance { ms: Millis(2_000) }).await;
            run
        });
        let (failures, _) = check_all(&run.record);
        let failed: Vec<_> =
            failures.iter().map(|f| format!("{}: {}", f.contract, f.detail)).collect();
        assert!(failed.is_empty(), "{failed:#?}");
        run.record
    }

    fn ends(record: &Recording) -> Vec<(Result<(), VoiceTaskEnded>, String)> {
        record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Handoff(HandoffStep::TaskEnded { how, restored, .. }) => {
                    Some((how.clone(), restored.clone()))
                }
                _ => None,
            })
            .collect()
    }

    fn task_answers(record: &Recording) -> Vec<Answer> {
        record
            .host_view()
            .api_calls()
            .iter()
            .filter(|c| matches!(c.input(), Input::VoiceTask { .. }))
            .filter_map(|c| c.returned().map(|(_, a)| a.clone()))
            .collect()
    }

    fn tool_call_id(record: &Recording, nth: u32) -> String {
        record.host_view().tools_run()[nth as usize].call_id().to_owned()
    }

    // Catches: a drain refused because a task holds the floor answered as if the session were
    // closing.
    #[test]
    fn a_drain_under_a_running_task_is_refused_as_task_running() {
        use lk_agents_voice::agent_session::DrainRefused;
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0));
        inputs.extend(task(0, "email"));
        inputs.extend([Input::Drain { wait: true }, Input::Advance { ms: Millis(50) }]);
        let record = play(inputs);
        let drains: Vec<Answer> = record
            .host_view()
            .api_calls()
            .iter()
            .filter(|c| matches!(c.input(), Input::Drain { .. }))
            .filter_map(|c| c.returned().map(|(_, a)| a.clone()))
            .collect();
        assert_eq!(drains, vec![Answer::Drain(Err(DrainRefused::TaskRunning))]);
    }

    // Catches: a completion that does not reach the awaiting tool, a parent left suspended, a
    // task's agent replying after it completed, or the parent's `on_enter` run on the restore.
    #[test]
    fn a_completed_task_returns_its_value_and_restores_the_parent() {
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0)); // the parent's tool 0
        inputs.extend(task(0, "email"));
        inputs.extend(call(1)); // the task's opening reply calls its tool 1
        inputs.extend(answer(1, ToolAnswer::Completed));
        inputs.extend(answer(0, ToolAnswer::Value)); // the parent's tool returns
        inputs.extend(say(2)); // and the parent replies to it
        let record = play(inputs);
        assert_eq!(ends(&record), vec![(Ok(()), "agent".to_owned())]);
        let value = completion_value(&tool_call_id(&record, 1));
        assert_eq!(task_answers(&record), vec![Answer::VoiceTask(Ok(value))]);
        // The request after the completion is the parent's reply to its own tool, on its speech.
        let speech_of = |kind, nth| match record.resource(kind, nth).map(|r| r.detail().clone()) {
            Some(Detail::Tool { speech, .. }) | Some(Detail::Chat { speech: Some(speech), .. }) => {
                speech
            }
            other => panic!("{kind:?} {nth}: {other:?}"),
        };
        assert_eq!(speech_of(Kind::Chat, 2), speech_of(Kind::Tool, 0));
    }

    // Catches: a nested task restoring the root instead of the task that started it, or ending the
    // outer task with the inner one.
    #[test]
    fn a_nested_task_hands_back_to_the_task_that_started_it() {
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0));
        inputs.extend(task(0, "email"));
        inputs.extend(call(1)); // the outer task's tool 1
        inputs.extend(task(1, "spell"));
        inputs.extend(call(2)); // the inner task's tool 2
        inputs.extend(answer(2, ToolAnswer::Completed));
        inputs.extend(answer(1, ToolAnswer::Completed));
        let record = play(inputs);
        assert_eq!(ends(&record), vec![(Ok(()), "email".to_owned()), (Ok(()), "agent".to_owned())]);
    }

    // Catches: a task outliving its awaiter: the parent's tool body ended (its await dropped) and
    // the task kept the floor.
    #[test]
    fn a_dropped_await_ends_the_task() {
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0));
        inputs.extend(task(0, "email"));
        inputs.extend(say(1));
        inputs.extend(answer(0, ToolAnswer::Error));
        let record = play(inputs);
        assert_eq!(ends(&record), vec![(Err(VoiceTaskEnded::Cancelled), "agent".to_owned())]);
        assert_eq!(task_answers(&record), vec![Answer::Abandoned]);
    }

    // Catches: a close mid-task leaving the awaiting tool waiting, or exiting the task's agent
    // instead of the parent.
    #[test]
    fn the_close_ends_a_running_task() {
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0));
        inputs.extend(task(0, "email"));
        inputs.extend(say(1));
        let record = play(inputs);
        assert_eq!(ends(&record), vec![(Err(VoiceTaskEnded::Closing), "agent".to_owned())]);
        assert_eq!(task_answers(&record), vec![Answer::VoiceTask(Err(VoiceTaskEnded::Closing))]);
    }

    // Catches: a task started outside a tool of the agent holding the floor, or by a speech the
    // task already holds the floor from.
    #[test]
    fn a_task_from_the_suspended_parent_is_refused() {
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        inputs.push(Input::Advance { ms: Millis(20) });
        inputs.extend(call(0));
        inputs.extend(task(0, "email"));
        inputs.extend(task(0, "spell")); // the parent's tool again: it is no longer driving
        let record = play(inputs);
        assert_eq!(
            task_answers(&record),
            vec![
                Answer::VoiceTask(Err(VoiceTaskEnded::Closing)),
                Answer::VoiceTask(Err(VoiceTaskEnded::NotInTool)),
            ]
        );
    }
}

mod cut_by_a_late_final {
    //! A speech made while the person is still being transcribed is cut by their words when the
    //! recogniser hands them over, here as its dropped connection's last final: they spoke, and
    //! the session hears their turn instead of talking over it.
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, UserText};

    // Catches: survey seeds 1722 and 1868. `speech_is_heard` counted the person's words only from
    // the case's own transcript inputs, so a speech cut by the final a dropped connection delivered
    // read as cut with the person silent.
    #[test]
    fn the_cut_is_the_persons() {
        let inputs = vec![
            Input::Advance { ms: Millis(1_000) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("my number is".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(300) },
            Input::Say {
                text: AgentText("Let me check that again.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::SttDrop { stt: Ref::new(0) },
            Input::Advance { ms: Millis(16_000) },
        ];
        let setup = Setup { stt: true, ..Setup::default() };
        let outcome = run(Case { setup, inputs, ..Case::default() });
        let failed: Vec<_> =
            outcome.failures.iter().filter(|f| f.contract == "speech_is_heard").collect();
        assert!(failed.is_empty(), "{failed:?}");
        assert!(outcome.expectations.get("speech_is_heard").is_some_and(|n| *n > 0));
    }
}

mod paced_voice_settles {
    //! A paced voice's later sentences go on its own clock: the run lets that clock run.
    use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

    // Catches (seed 1766): a paced voice holding its last sentences until the long audio it had
    // already sent played down, its segment still open, so the case's finish report was refused;
    // the release rounds then stepped the clock a millisecond at a time and never reached the
    // pacer's wake, and the run never went quiet.
    #[test]
    fn a_paced_reply_longer_than_its_release_rounds_completes() {
        let sentence = "You can ask for a copy of your data at any time, and you can ask us to \
                        delete it, except where the law requires us to keep it.";
        let reply = format!("Sure. {}", [sentence; 6].join(" "));
        let outcome = simulate(async {
            let setup = Setup { tts: true, tts_pacing: true, ..Setup::default() };
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmText { chat: Ref::new(0), text: AgentText(reply) },
                Input::LlmEnd { chat: Ref::new(0) },
            ] {
                run.apply(&input).await;
            }
            finish(run, Outcome::default()).await
        });
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    }
}

mod cleanup_during_natural_completion {
    use lk_agents_abi::errors::ProviderError;
    use lk_agents_abi::{
        ApiConnectOptions, BoxFuture, Closed, ProviderStream, StreamOutput, Ts, VirtualClock,
    };
    use lk_agents_fuzz::drive::simulate;
    use lk_agents_runtime::inference::retry::{Open, stream};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context, Waker};
    use tokio_util::sync::CancellationToken;
    struct Ending {
        started: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
        release: lk_agents_runtime::net::signal::Cell<bool>,
        panics: bool,
    }

    impl StreamOutput<u32> for Ending {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<u32>, ProviderError>> {
            Box::pin(async { Ok(None) })
        }

        fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async move {
                self.started.store(true, Ordering::SeqCst);
                self.release.wait_map(|released| released.then_some(())).await.unwrap();
                self.closed.store(true, Ordering::SeqCst);
                assert!(!self.panics, "injected fault: close panicked");
                Closed::Clean
            })
        }
    }

    impl ProviderStream<u32> for Ending {}

    #[test]
    fn closing_during_natural_completion_finishes_the_cleanup_already_started() {
        simulate(async {
            let started = Arc::new(AtomicBool::new(false));
            let closed = Arc::new(AtomicBool::new(false));
            let (release, ready) = lk_agents_runtime::net::signal::signal(false);
            let (opening, finished) = (started.clone(), closed.clone());
            let open: Open<u32> = Arc::new(move || {
                let source = Ending {
                    started: opening.clone(),
                    closed: finished.clone(),
                    release: ready.clone(),
                    panics: false,
                };
                Box::pin(async move { Ok(Box::new(source) as Box<dyn ProviderStream<u32>>) })
            });
            let mut stream = stream(
                "ending".into(),
                &ApiConnectOptions::default(),
                Arc::new(VirtualClock::new()),
                open,
                |_| true,
                Arc::new(|_, _, _| {}),
                CancellationToken::new(),
            )
            .await;
            let mut reading = stream.next();
            assert!(reading.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            assert!(started.load(Ordering::SeqCst));
            drop(reading);
            release.set(true);
            assert_eq!(stream.close(Ts::from_nanos(u64::MAX)).await, Closed::Clean);
            assert!(closed.load(Ordering::SeqCst), "the in-flight cleanup was joined");
        });
    }

    // Catches: seed 145; a close that panicked during a pull was polled again at teardown.
    #[test]
    fn a_panicked_close_is_never_resumed_at_teardown() {
        simulate(async {
            let (release, ready) = lk_agents_runtime::net::signal::signal(true);
            let open: Open<u32> = Arc::new(move || {
                let source = Ending {
                    started: Arc::new(AtomicBool::new(false)),
                    closed: Arc::new(AtomicBool::new(false)),
                    release: ready.clone(),
                    panics: true,
                };
                Box::pin(async move { Ok(Box::new(source) as Box<dyn ProviderStream<u32>>) })
            });
            let mut stream = stream(
                "panicking close".into(),
                &ApiConnectOptions::default(),
                Arc::new(VirtualClock::new()),
                open,
                |_| true,
                Arc::new(|_, _, _| {}),
                CancellationToken::new(),
            )
            .await;
            let mut reading = stream.next();
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reading.as_mut().poll(&mut Context::from_waker(Waker::noop()))
            }));
            assert!(panic.is_err(), "the original provider panic reaches its caller");
            drop(reading);
            assert_eq!(stream.close(Ts::from_nanos(u64::MAX)).await, Closed::Clean);
            drop(release);
        });
    }
}
