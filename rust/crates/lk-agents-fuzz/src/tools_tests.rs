mod input_required_tool_takes_the_answer {
    //! `InputRequired(tool=)`: the person's next turn is a call of that tool, with their words.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: the claim never made on a call: the person's answer went to the model to interpret,
    // and the tool that asked was called with the model's arguments.
    #[test]
    fn the_claimed_tool_takes_the_persons_words() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let call = |chat| Input::LlmToolCall {
            chat: Ref::new(chat),
            name: "lookup".into(),
            args: ToolArgs(r#"{"label":"guess"}"#.into()),
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call(0),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AskVerbatim,
                text: AgentText("Which day?".into()),
            },
            Input::Advance { ms: Millis(3_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Monday".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3_000) },
            // The turn's reply: the model calls the claimed tool with arguments of its own.
            call(1),
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Value,
                text: AgentText("Booked.".into()),
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
        let calls: Vec<String> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCall(c) => Some(c.arguments.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(calls, [r#"{"label":"guess"}"#, r#"{"label":"Monday"}"#], "calls: {calls:?}");
    }
}

mod ordinary_completed_replies {
    //! An ordinary tool's `Completed(value)` is its call's output, and the agent replies to it.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::record::Kind;
    use lk_agents_voice::events::SessionEvent;

    // Catches: a `Completed` from a tool outside any VoiceTask taken as a task's completion, its
    // output asking for no reply and the step ending there: the agent went silent. The fake host
    // hands a value with every completion, so the core alone decides.
    #[test]
    fn an_ordinary_completion_is_replied_to() {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs(r#"{"label":"order"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Completed,
                text: AgentText("booked".into()),
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
        let outputs: Vec<(String, bool)> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some((o.output.clone(), o.reply_required)),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(outputs, [("booked".to_owned(), true)], "outputs: {outputs:?}");
        assert_eq!(run.record.resources(Kind::Chat).count(), 2, "the follow-up was never asked");
    }
}

mod stored_answer_reaches_the_read_back {
    //! On a stored lane, the person's answer to a read-back is recorded as that call's output.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    /// The fixture tool the runs declare with `needs_approval=True`.
    const APPROVAL_TOOL: &str = "transfer";

    // Catches: the preemptive reply's rendering of the answer, given back by the lane's store, taken
    // as the lane's edit, so the stored output was the model's rendering, never the person's words
    // (`tool_output_follows_outcome`, survey seed 21687).
    #[test]
    fn the_answer_is_the_read_backs_output() {
        // A store that keeps its lease for the whole call.
        let setup = Setup {
            tools: vec![APPROVAL_TOOL.into()],
            stt: true,
            vad: true,
            preemptive: true,
            turn: TurnMode::Stt,
            storage_lost_at: Some(1_000),
            ..Setup::default()
        };
        let answer = || UserText("yes please".into());
        let inputs = [
            Input::GenerateReply {
                user_input: Some(UserText("transfer me".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: APPROVAL_TOOL.into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
            // The person answers; the reply to it is planned from their preflight words.
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(600) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
            Input::SttPreflight { utterance: Ref::new(0), text: answer(), speaker: None },
            Input::Advance { ms: Millis(50) },
            Input::SttFinal { utterance: Ref::new(0), text: answer(), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let outputs: Vec<String> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let asked = outputs.iter().any(|o| o.starts_with("waiting for the person's answer"));
        assert!(asked, "the scenario: the call waits on its read-back: {outputs:?}");
        assert_eq!(outputs.last().map(String::as_str), Some("yes please"), "{outputs:?}");
    }
}

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

mod cut_question {
    //! A question its speech was cut before putting is not the one the person's words answer.
    use lk_agents_fuzz::search;

    // Catches: a step whose speech was cut before its tools settled kept its question selected,
    // so the person's next words answered a question never put instead of the one asked (survey
    // seed 16796 BasicPos, tool_output_follows_outcome).
    #[test]
    fn the_next_words_answer_the_question_asked() {
        let seed = 16796;
        let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let outcome = search::explore(setup, seed, 3000);
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.to_string()).collect();
        assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
    }
}

mod questions_in_recorded_order {
    //! Questions wait in the order the session recorded them.
    use lk_agents_fuzz::search;

    // Catches: an output stamped from its run's anchor, before an output the session recorded
    // ahead of it, so the history put the later question first and the person's words answered
    // it (campaign seed 5264 Arrival, tool_output_follows_outcome).
    #[test]
    fn the_next_words_answer_the_question_recorded_first() {
        let seed = 5264;
        let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let outcome = search::explore(setup, seed, 3000);
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.to_string()).collect();
        assert!(!failed.iter().any(|c| c == "tool_output_follows_outcome"), "{failed:?}");
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

mod canonical_terminal_tool_result {
    //! One canonical question wins in model order, regardless of body completion order.
    use std::collections::BTreeMap;

    use lk_agents_abi::agent::run::RunOutcome;
    use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_fuzz::record::{Detail, Kind};
    use lk_agents_voice::events::SessionEvent;

    const QUESTION: &str = "Which day?";
    const UNSELECTED_QUESTION: &str = "Which city?";
    const ANSWER: &str = "Monday";

    #[test]
    fn first_terminal_question_controls_presentation_result_and_answer_routing() {
        scenario([0, 1]);
    }

    #[test]
    fn reverse_tool_completion_keeps_the_same_question_and_canonical_history_order() {
        scenario([1, 0]);
    }

    fn scenario(completion_order: [u32; 2]) {
        let setup = Setup {
            tts: false,
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into(), "reserve".into()],
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let opening = [
                // Exercise text-only presentation without requiring physical playback receipts.
                Input::SetAudioOutput { attach: false },
                Input::Advance { ms: Millis(20) },
                Input::Run { user_input: Some(UserText("Please help me book.".into())) },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs(r#"{"label":"day"}"#.into()),
                },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "reserve".into(),
                    args: ToolArgs(r#"{"label":"city"}"#.into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &opening {
                run.apply(input).await;
            }
            for tool in completion_order {
                let question = if tool == 0 { QUESTION } else { UNSELECTED_QUESTION };
                run.apply(&Input::ToolAnswered {
                    tool: Ref::new(tool),
                    answer: ToolAnswer::AskVerbatim,
                    text: AgentText(question.into()),
                })
                .await;
                // Make body completion order observable instead of delivering both in one tick.
                run.apply(&Input::Advance { ms: Millis(20) }).await;
            }
            run.apply(&Input::Advance { ms: Millis(5_000) }).await;

            let events = run.record.host_view().events();
            let settled: Vec<_> = events
                .iter()
                .filter_map(|(_, event)| match event {
                    SessionEvent::RunSettled { result: Ok(settled), .. } => Some(&settled.result),
                    _ => None,
                })
                .collect();
            let requirements: Vec<_> =
                settled.iter().flat_map(|result| result.input_requirements.iter()).collect();
            assert_eq!(
                requirements
                    .iter()
                    .map(|requirement| requirement.call.name.as_str())
                    .collect::<Vec<_>>(),
                ["lookup", "reserve"]
            );
            assert_eq!(
                requirements
                    .iter()
                    .map(|requirement| requirement.question.as_str())
                    .collect::<Vec<_>>(),
                [QUESTION, UNSELECTED_QUESTION]
            );
            let results: Vec<_> = settled.iter().map(|result| result.outcome.clone()).collect();
            assert_eq!(
                results,
                [RunOutcome::InputRequired(lk_agents_abi::agent::run::RunQuestion {
                    question: QUESTION.into(),
                    verbatim: true,
                    answer_tool: Some("lookup".into()),
                })],
                "the selected shared-run outcome must be the caller's result; events: {events:?}"
            );

            let spoken: Vec<_> = events
                .iter()
                .filter_map(|(_, event)| match event {
                    SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                        ChatItem::Message(message) if message.role == Role::Assistant => {
                            message.raw_text_content()
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            assert_eq!(spoken, [QUESTION], "only the selected question is presented");

            let history: Vec<_> = events
                .iter()
                .filter_map(|(_, event)| match event {
                    SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                        ChatItem::FunctionCall(call) => Some(("call", call.name.clone())),
                        ChatItem::FunctionCallOutput(output) => {
                            Some(("output", output.name.clone()))
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            assert_eq!(
                history,
                [
                    ("call", "lookup".into()),
                    ("call", "reserve".into()),
                    ("output", "lookup".into()),
                    ("output", "reserve".into()),
                ],
                "voice records the shared calls then outputs in model order"
            );

            let history_ids: Vec<String> = events
                .iter()
                .filter_map(|(_, event)| match event {
                    SessionEvent::ConversationItemAdded { item, .. }
                        if matches!(
                            item.as_ref(),
                            ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_)
                        ) =>
                    {
                        Some(item.id().to_owned())
                    }
                    _ => None,
                })
                .collect();

            let answer_turn = [
                Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
                Input::SttStart { stt: Ref::new(0) },
                Input::Advance { ms: Millis(300) },
                Input::SttFinal {
                    utterance: Ref::new(0),
                    text: UserText(ANSWER.into()),
                    speaker: None,
                },
                Input::SttEnd { utterance: Ref::new(0) },
                Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
                Input::Advance { ms: Millis(3_000) },
                Input::LlmToolCall {
                    chat: Ref::new(1),
                    name: "lookup".into(),
                    args: ToolArgs(r#"{"label":"model guess"}"#.into()),
                },
                Input::LlmEnd { chat: Ref::new(1) },
                Input::Advance { ms: Millis(100) },
                Input::ToolAnswered {
                    tool: Ref::new(2),
                    answer: ToolAnswer::Return,
                    text: AgentText(String::new()),
                },
                Input::Advance { ms: Millis(1_000) },
            ];
            for input in &answer_turn {
                run.apply(input).await;
            }
            let chats: Vec<_> = run.record.resources(Kind::Chat).collect();
            let Detail::Chat { ctx_ids, .. } = chats[1].detail() else {
                panic!("the next model request must expose its input history")
            };
            let carried: Vec<_> =
                ctx_ids.iter().filter(|id| history_ids.contains(id)).cloned().collect();
            assert_eq!(
                carried, history_ids,
                "the actual next request must preserve canonical stored history order"
            );
            run
        });

        let mut outputs = BTreeMap::new();
        let mut lookup_arguments = Vec::new();
        for (_, event) in run.record.host_view().events() {
            if let SessionEvent::ConversationItemAdded { item, .. } = event {
                match item.as_ref() {
                    ChatItem::FunctionCall(call) if call.name == "lookup" => {
                        lookup_arguments.push(call.arguments.clone());
                    }
                    ChatItem::FunctionCallOutput(output) if output.name == "reserve" => {
                        outputs.insert("reserve", output.output.clone());
                    }
                    ChatItem::FunctionCallOutput(output) if output.name == "lookup" => {
                        outputs.entry("lookup").or_insert_with(|| output.output.clone());
                        if output.output == ANSWER {
                            outputs.insert("lookup", output.output.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(
            lookup_arguments,
            [r#"{"label":"day"}"#, r#"{"label":"Monday"}"#],
            "the answer reaches the selected tool, not the second question's tool"
        );
        assert_eq!(outputs.get("lookup").map(String::as_str), Some(ANSWER));
        assert_eq!(
            outputs.get("reserve").map(String::as_str),
            Some(format!("{WAITING_PREFIX}{UNSELECTED_QUESTION}").as_str())
        );
        assert!(
            outputs["reserve"].starts_with(WAITING_PREFIX),
            "an unselected request must not be synthetically answered: {outputs:?}"
        );
    }
}

mod detached_duplicate_admission {
    //! Session-owned background bodies remain visible to the shared duplicate policy.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::record::Kind;
    use lk_agents_voice::events::SessionEvent;

    // The named fixture declares on_duplicate="reject", like the approval fixture above.
    const DUPLICATE_REJECT_TOOL: &str = "exclusive_lookup";

    fn call(chat: u32) -> Input {
        Input::LlmToolCall {
            chat: Ref::new(chat),
            name: DUPLICATE_REJECT_TOOL.into(),
            args: ToolArgs(r#"{"label":"same"}"#.into()),
        }
    }

    #[test]
    fn detached_body_blocks_a_later_step_until_its_final_completion() {
        let setup =
            Setup { tts: false, tools: vec![DUPLICATE_REJECT_TOOL.into()], ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            let first = [
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                call(0),
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                Input::ToolAnswered {
                    tool: Ref::new(0),
                    answer: ToolAnswer::Detached,
                    text: AgentText("Checking.".into()),
                },
                Input::Advance { ms: Millis(100) },
                // The first update released the model step, but the actual body remains live.
                call(1),
                Input::LlmEnd { chat: Ref::new(1) },
                Input::Advance { ms: Millis(100) },
            ];
            for input in &first {
                run.apply(input).await;
            }
            assert_eq!(
                run.record.host_view().tools_run().len(),
                1,
                "the rejected duplicate must not run a second host body"
            );
            let rejected = run.record.host_view().events().iter().any(|(_, event)| match event {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(output) => {
                        output.is_error && output.output.contains("already running")
                    }
                    _ => false,
                },
                _ => false,
            });
            assert!(rejected, "the shared policy must record the duplicate refusal");

            // Finish the model's follow-up to the duplicate refusal, then the background body.
            run.apply(&Input::LlmEnd { chat: Ref::new(2) }).await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            run.apply(&Input::ToolUpdate {
                tool: Ref::new(0),
                progress: None,
                end: ToolAnswer::Return,
            })
            .await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            let next_chat = run.record.resources(Kind::Chat).count() as u32;
            run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
                .await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            run.apply(&call(next_chat)).await;
            run.apply(&Input::LlmEnd { chat: Ref::new(next_chat) }).await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            assert_eq!(
                run.record.host_view().tools_run().len(),
                2,
                "final completion removes the live execution fact and permits the next body"
            );
            run.apply(&Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            })
            .await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            run
        });
        assert_eq!(run.record.host_view().tools_run().len(), 2);
    }
}

mod host_body_close_ownership {
    use lk_agents_fuzz::drive::{Case, Setup, run};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

    // Seed 2 / Arrival: a terminal update releases the core's call while the host body remains
    // alive; the body still owns close cleanup and must settle before the cleanup bound.
    fn case(refuses_cancel: bool) -> Case {
        Case {
            setup: Setup {
                tools: vec!["lookup".into()],
                tools_refuse_cancel: refuses_cancel,
                ..Setup::default()
            },
            inputs: vec![
                Input::GenerateReply { user_input: None, instructions: None, allow: None },
                Input::Advance { ms: Millis(100) },
                Input::LlmToolCall {
                    chat: Ref::new(0),
                    name: "lookup".into(),
                    args: ToolArgs("{}".into()),
                },
                Input::LlmEnd { chat: Ref::new(0) },
                Input::Advance { ms: Millis(100) },
                Input::ToolUpdate { tool: Ref::new(0), progress: None, end: ToolAnswer::Rejected },
                Input::Advance { ms: Millis(100) },
                Input::Close { drain: false },
                Input::AwaitExit {},
            ],
            ..Case::default()
        }
    }

    #[test]
    fn semantic_terminal_update_does_not_orphan_the_actual_host_body() {
        let outcome = run(case(false));
        assert!(
            outcome.reached.iter().any(|variant| variant == "Tool::CallEnded"),
            "the semantic result never settled the call: {:?}",
            outcome.reached
        );
        assert!(outcome.ok(), "{:?}", outcome.failures);
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
    }
    // Catches: the body answering `Detached` after its terminal update had already ended the
    // call. The core refused the detached body's report, and every later one, yet the harness
    // kept the tool running for a final update that could never land: the release phase sent it
    // forever and the run never went quiet (survey seed 210 / Arrival).
    #[test]
    fn a_body_detaching_after_its_terminal_update_is_over() {
        let mut case = case(false);
        case.inputs.truncate(5);
        case.inputs.extend([
            Input::ToolUpdate { tool: Ref::new(0), progress: None, end: ToolAnswer::Recorded },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            Input::Advance { ms: Millis(100) },
        ]);
        let outcome = run(case);
        assert!(outcome.ok(), "{:?}", outcome.failures);
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
    }

    // Catches: a host body refusing the close's cancellation held the call that awaited it past
    // the close's cleanup bound as well, though only the body owns its cleanup: the call's task
    // was cancelled there (survey seed 5901 TaskBridge, every_task_finishes).
    #[test]
    fn a_body_refusing_the_close_alone_runs_to_the_bound() {
        use lk_agents_fuzz::{drive::Policy, search};
        let seed = 5901;
        let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
        let dials = search::Dials::of(seed);
        let outcome = search::explore_with(setup, seed, 3000, dials, Policy::TaskBridge);
        let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.to_string()).collect();
        assert!(!failed.iter().any(|c| c == "every_task_finishes"), "{failed:?}");
    }

    #[test]
    fn a_host_refusing_close_cancellation_still_obeys_the_existing_cleanup_bound() {
        let outcome = run(case(true));
        assert!(
            outcome
                .expectations
                .get("close_completes_at_its_bound")
                .is_some_and(|count| *count > 0),
            "the refusing host did not exercise bounded close"
        );
        assert!(outcome.ok(), "{:?}", outcome.failures);
        assert!(!outcome.stopped_early, "{:?}", outcome.ending);
    }
}
