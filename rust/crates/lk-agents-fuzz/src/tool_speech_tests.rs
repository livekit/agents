mod tool_awaits_own_speech {
    //! A tool awaiting its own speech is refused (`RuntimeError`) instead of waiting forever.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Catches: `await ctx.speech_handle` inside a blocking tool waited for a speech that waited for
    // that tool: the call hung silently.
    #[test]
    fn a_tool_awaiting_its_own_speech_is_refused() {
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
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(20) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::AwaitOwnSpeech,
                    text: AgentText(String::new()),
                },
                Input::Advance { ms: Millis(200) },
            ];
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let ended =
            run.record.host_view().tools_run().iter().all(|t| !t.ended_announcements().is_empty());
        assert!(ended, "the tool's wait on its own speech was refused and the tool returned");
        let speeches = run.record.host_view().speeches();
        assert!(speeches[0].is_done(), "the speech no longer waits on a tool waiting on it");
    }
}

mod tool_reply_renders_its_template {
    //! A background tool's deferred reply is told what its `tool_handling` templates render.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::fakes::{TEMPLATED_REPLY, TEMPLATED_TOOL};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{Step, ToolStep};

    // Catches: the deferred reply scheduled with the framework's default instructions
    // whatever the agent's or the tool's toolset's `reply_at_tail_template` /
    // `reply_maybe_covered_template` said.
    #[test]
    fn a_deferred_reply_is_told_its_tools_template() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec![TEMPLATED_TOOL.into()],
            max_tool_steps: Some(1),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            // The tool answers with a progress update and keeps running in the background.
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            Input::Advance { ms: Millis(3_000) },
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(2) },
            Input::PlayoutFinished { segment: Ref::new(2) },
            Input::Advance { ms: Millis(100) },
            // News from the background, once the agent is idle.
            Input::ToolUpdate {
                tool: Ref::new(0),
                progress: Some(AgentText("It shipped.".into())),
                end: ToolAnswer::Return,
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
        let told: Vec<String> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Tool(ToolStep::ReplyScheduled { instructions, .. }) => {
                    Some(instructions.clone())
                }
                _ => None,
            })
            .collect();
        assert!(!told.is_empty(), "the update's reply was scheduled");
        assert!(told.iter().all(|t| t.starts_with(TEMPLATED_REPLY)), "told {told:?}");
    }
}

mod tool_reply_scheduled_at_once {
    //! A background tool's next deferred reply is scheduled in the cycle that makes it.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{SpeechStep, Step, ToolStep};

    // Catches: a detached tool's second update, landing while the reply to its first was playing,
    // delivered by the cycle's waiter pass once that reply ended, after the cycle's scheduling pass
    // had run: the new reply sat queued with the floor free until some unrelated wake scheduled it
    // (`speech_is_heard`, survey seed 70).
    #[test]
    fn a_second_tool_update_reply_is_authorized_at_once() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            max_tool_steps: Some(1),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            // The tool answers with a progress update and keeps running in the background.
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            // The model speaks the progress and calls the tool again, which returns nothing.
            Input::Advance { ms: Millis(3_000) },
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(1_000) },
            // The first news, with the agent idle: its reply plays.
            Input::ToolUpdate {
                tool: Ref::new(0),
                progress: Some(AgentText("Found it.".into())),
                end: ToolAnswer::Return,
            },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(2) },
            // The second news lands while that reply plays, and waits for it to end.
            Input::ToolUpdate {
                tool: Ref::new(0),
                progress: Some(AgentText("It shipped.".into())),
                end: ToolAnswer::Return,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutFinished { segment: Ref::new(2) },
            Input::Advance { ms: Millis(20_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let replies: Vec<_> = journal
            .iter()
            .filter_map(|(at, s)| match s {
                Step::Tool(ToolStep::ReplyScheduled { speech, .. }) => Some((*at, *speech)),
                _ => None,
            })
            .collect();
        let [_, (at, speech)] = replies[..] else { panic!("two deferred replies: {replies:?}") };
        let authorized = journal.iter().find_map(|(a, s)| match s {
            Step::Speech(SpeechStep::Authorized { speech: sp }) if *sp == speech => Some(*a),
            _ => None,
        });
        assert!(
            authorized.is_some_and(|a| a.at_ms == at.at_ms),
            "made at {} ms, authorized at {:?}",
            at.at_ms,
            authorized.map(|a| a.at_ms)
        );
    }
}

mod tool_started_then_interrupted {
    //! A tool call reported started reaches the runner even when its speech is interrupted at once.
    use lk_agents_fuzz::drive::{Policy, Run, Setup, simulate_with};
    use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};

    // Catches: the runner task of a scheduled speech's call racing adoption against interruption.
    // The start was already journalled hosted and `ToolExecutionUpdated::Started` emitted, but an
    // interruption landing before the task's first poll dropped the body unrun (survey seed 170).
    // The race is the scheduler's, so 32 schedules of each queue policy run.
    #[test]
    fn a_started_call_runs_though_its_speech_is_interrupted() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(20) },
        ];
        let mut started = 0;
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..32 {
                let run = simulate_with(schedule, async {
                    let mut run = Run::start(&Setup::default(), schedule, policy).await;
                    for input in &inputs {
                        run.apply(input).await;
                    }
                    run
                });
                let r = &run.record;
                for (_, call_id, hosted) in r.internal_view().tool_calls_started() {
                    started += 1;
                    assert!(
                        !hosted || r.host_view().tool_ran(call_id),
                        "{policy:?} schedule {schedule}: the runner never saw {call_id}"
                    );
                }
            }
        }
        assert!(started > 0, "the scenario: a call started before the interruption");
    }
}

mod tool_update_before_detach_heard {
    //! A background tool's update that lands before its step's join heard it detach is kept.
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_voice::events::{SessionEvent, ToolLifecycle};
    use lk_agents_voice::trace::Step;

    // Catches: `ctx.update()` reaches the driver directly while the detach is heard at the join, in
    // the model's order: with a blocking call ahead of it, the second call's update arrived first and
    // was refused (`UpdateRefused`), never told nor replied to.
    #[test]
    fn an_update_ahead_of_the_join_is_told() {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let call = |args: &str| Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs(args.into()),
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call("{}"),
            call(r#"{"seconds":3}"#),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            // The second call detaches and updates while the first still runs.
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            Input::Advance { ms: Millis(10) },
            Input::ToolUpdate {
                tool: Ref::new(1),
                progress: Some(AgentText("Found it.".into())),
                end: ToolAnswer::Return,
            },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("12C".into()),
            },
            Input::Advance { ms: Millis(3_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let told: Vec<String> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id,
                    ..
                })) => Some(id.clone()),
                _ => None,
            })
            .collect();
        assert!(told.iter().any(|id| id.ends_with("_update_1")), "updates told: {told:?}");
    }
}

mod tool_update_as_first_report {
    //! A body's first report from the host: progress detaches it, an end answers its call.
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    fn reported(reports: &[Input]) -> Run {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let mut inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ];
        for report in reports {
            inputs.push(report.clone());
            inputs.push(Input::Advance { ms: Millis(100) });
        }
        inputs.push(Input::Advance { ms: Millis(3_000) });
        simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        })
    }

    fn update(progress: Option<&str>, end: ToolAnswer) -> Input {
        Input::ToolUpdate {
            tool: Ref::new(0),
            progress: progress.map(|p| AgentText(p.into())),
            end,
        }
    }

    fn answer(answer: ToolAnswer, text: &str) -> Input {
        Input::ToolAnswered { tool: Ref::new(0), answer, text: AgentText(text.into()) }
    }

    fn failed(run: &Run) -> Vec<String> {
        check_all(&run.record).0.into_iter().map(|f| f.contract.to_string()).collect()
    }

    // Catches: the rules took a body's first `ctx.update()` for a later update owed a
    // `<call_id>_update_<n>` (seed 126 TaskBridge, tool_updates_are_told) and its answer for the
    // call's output, though the product tells the update as the detach and records the answer as
    // the `<call_id>_final` call.
    #[test]
    fn a_first_update_is_the_detach() {
        let run = reported(&[
            update(Some("Checking."), ToolAnswer::Return),
            answer(ToolAnswer::Value, "12C"),
        ]);
        let failed = failed(&run);
        let misread = ["tool_updates_are_told", "tool_output_follows_outcome"];
        assert!(!failed.iter().any(|c| misread.contains(&c.as_str())), "{failed:?}");
    }

    // Catches: the rule held a detached body's own answer to its final, though the host's
    // update ending the body had answered it first and the product refused the later answer
    // (survey seed 6581 TaskBridge, tool_output_follows_outcome).
    #[test]
    fn a_detached_bodys_end_update_is_its_final() {
        for schedule in 0..8 {
            let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
            let inputs = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                update(Some("Checking."), ToolAnswer::AfterIdle),
                Input::Advance { ms: Millis(100) },
                update(None, ToolAnswer::AskVerbatim),
                answer(ToolAnswer::Ask, "Tuesday at 9 works."),
                Input::Advance { ms: Millis(3_000) },
            ];
            let run = simulate(async {
                let policy = lk_agents_fuzz::drive::Policy::TaskBridge;
                let mut run = Run::start(&setup, schedule, policy).await;
                for input in &inputs {
                    run.apply(input).await;
                }
                run
            });
            let failed = failed(&run);
            assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
        }
    }

    // Catches: a host update ending the body and the body's own answer, both in before the
    // runtime's body task next ran: the task took the answer and dropped the update the host had
    // sent first (survey seed 4271 TaskBridge, tool_output_follows_outcome).
    #[test]
    fn an_end_with_the_answer_in_one_instant_answers_the_call() {
        for schedule in 0..8 {
            let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
            let inputs = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                update(None, ToolAnswer::Recorded),
                answer(ToolAnswer::Value, "Your balance is $1,204.55."),
                Input::Advance { ms: Millis(3_000) },
            ];
            let run = simulate(async {
                let policy = lk_agents_fuzz::drive::Policy::TaskBridge;
                let mut run = Run::start(&setup, schedule, policy).await;
                for input in &inputs {
                    run.apply(input).await;
                }
                run
            });
            let failed = failed(&run);
            assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
        }
    }

    // Catches: the rule held the body's own answer to the call's output, though a host update
    // ending the body first answered the call and the driver refused the later answer (seed 211
    // Arrival, tool_output_follows_outcome).
    #[test]
    fn an_end_before_the_answer_answers_the_call() {
        let run = reported(&[
            update(None, ToolAnswer::Ask),
            answer(ToolAnswer::Verbatim, "Tuesday at 9 works."),
        ]);
        let failed = failed(&run);
        assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
    }
}

mod tool_answered_after_its_speech_failed {
    //! A call whose speech failed while its host handler ran asks for no reply: nothing is left to
    //! give one.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::check::check_all;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_voice::events::SessionEvent;
    use lk_agents_voice::trace::Step;

    // Catches: the rule held the reply flag of a call whose model request failed after it: the
    // speech failed while `function_tools_executed` ran, and the driver asks no reply of a step
    // over before its outputs are recorded (seed 1018 TaskBridge, tool_output_follows_outcome).
    #[test]
    fn the_output_asks_for_no_reply() {
        let setup =
            Setup { tools: vec!["lookup".into()], hook_bound_ms: Some(2_500), ..Setup::default() };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmFail { chat: Ref::new(0) },
            Input::Advance { ms: Millis(3_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("12C".into()),
            },
            Input::Advance { ms: Millis(3_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let recorded: Vec<bool> = run
            .record
            .journal()
            .iter()
            .filter_map(|(_, s)| match s {
                Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                    ChatItem::FunctionCallOutput(o) => Some(o.reply_required),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(recorded, [false], "the output, recorded once, asks for no reply");
        let failed: Vec<String> =
            check_all(&run.record).0.into_iter().map(|f| f.contract.to_string()).collect();
        assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
    }
}
