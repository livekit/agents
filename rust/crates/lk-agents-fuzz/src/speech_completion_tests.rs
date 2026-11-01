//! Speech regression tests.

mod speech_announced_before_user_item {
    //! `generate_reply(user_input=)` announces its speech, user-initiated, before the user item.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, UserText};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the user item committed before `speech_created`, which a listener pairing the two saw
    // in the wrong order. The reference emits `speech_created(user_initiated=True)` in
    // `_generate_reply` and adds the user message once the reply task sees its speech scheduled
    // (`_pipeline_reply_task_impl`); found by `scripts/parity.py generate_reply`.
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

mod step_cut_text_conserved {
    //! A host's unforced `interrupt()` during playout ends the whole speech: the cut step loses its
    //! words, the tool's answer is committed, and no follow-up is asked for.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Catches: `text_conservation` holding a speech that completed to every word its steps were
    // given, though the host's unforced `interrupt()` stops only the playing step (its words are
    // never forwarded) and the tool's follow-up then completes the speech (survey seed 2750). The
    // completion clause now skips a speech the driver journalled interrupting. The step-scoped cut is
    // gone since: as `_interrupt_fut` is one-shot, the speech ends interrupted and nothing follows.
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
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
    use lk_agents_voice::events::SessionEvent;

    // Catches: a segment whose synthesis ended interrupted (never opened, or cut while its open was
    // being retried) taking the lost-voice fallback, which forwards the segment's words as text and
    // counts them heard. The reply was committed whole, `interrupted` set, though not a word of it
    // played: on a real call the model then read its own unheard question as asked and answered past
    // the caller. The reference commits only the synchronised transcript of what played: nothing.
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
    // talked over. The reference's `_user_turn_completed_impl` returns with "skipping reply to user
    // input, current speech generation cannot be interrupted" and keeps no user item; found by
    // `scripts/parity.py say_uninterruptible`.
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
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
    use lk_agents_voice::{events::SessionEvent, trace::Step};

    // Catches: the failed synthesis's words committed as an interrupted assistant message the caller
    // never heard. The reference's `forward_generation` leaves a segment whose TTS failed with none of
    // its audio captured "skipped", so `forwarded_text` is empty and no message is added; found by
    // `scripts/parity.py tts_failure`.
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
    use lk_agents_voice::agent_activity::voice_task::{TaskEnd, VoiceTaskEnded};
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

    fn ends(record: &Recording) -> Vec<(TaskEnd, String)> {
        record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Handoff(HandoffStep::TaskEnded { how, restored, .. }) => {
                    Some((*how, restored.clone()))
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
        assert_eq!(ends(&record), vec![(TaskEnd::Completed, "agent".to_owned())]);
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
        assert_eq!(
            ends(&record),
            vec![
                (TaskEnd::Completed, "email".to_owned()),
                (TaskEnd::Completed, "agent".to_owned())
            ]
        );
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
        assert_eq!(ends(&record), vec![(TaskEnd::Cancelled, "agent".to_owned())]);
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
        assert_eq!(ends(&record), vec![(TaskEnd::Closing, "agent".to_owned())]);
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
