//! Tools regression tests.

mod input_required_tool_takes_the_answer {
    //! `InputRequired(tool=)`: the person's next turn is a call of that tool, with their words.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: the claim never made on a call (`_claim_next_turn`, `_answer_arguments` absent): the
    // person's answer went to the model to interpret, and the tool that asked was called with the
    // model's arguments (finding 7 of the tools review).
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
    // output asking for no reply and the step ending there: the agent went silent (the host's
    // `task` flag shadowed, finding 1 of the tools review). The fake host hands a value with every
    // completion, so the core alone decides.
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

    // Catches: the preemptive reply to the person's answer planned its request with the answer already in the
    // waiting output, rendered for the model ("not run yet: the person answered ..."). The lane's
    // store gave that rendering back for the row it had not stored yet, and the session's record took
    // it as the lane's edit: the output stopped waiting before the answer was recorded, so the stored
    // output was the model's rendering, never the person's words (`tool_output_follows_outcome`,
    // survey seed 21687).
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
    // that tool: the call hung silently. The reference raises (`SpeechHandle.wait_for_playout`'s
    // circular-wait guard); the port's comments claimed the shim had it, and nothing did.
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
    // (the user-away timer, 15 s later in survey seed 70), the agent silent with the tool's news in
    // hand (`speech_is_heard`). What the waiters start is now scheduled in the same cycle.
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
    // interruption landing before the task's first poll dropped the body unrun (survey seed 170;
    // 9f070e85f). The race is the scheduler's, so 32 schedules of each queue policy run.
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
    // was refused (`UpdateRefused`), never told nor replied to (finding 3 of the tools review).
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

mod two_questions_one_answer {
    //! Two `InputRequired` calls in one step: the person's next message answers both.
    use std::collections::BTreeMap;

    use lk_agents_abi::ChatItem;
    use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: the record answering only the last question asked (`pending_input`) while the request
    // answered every waiting output: the first call's output stayed waiting on the record forever.
    #[test]
    fn the_answer_is_every_waiting_calls_output() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let call = |label: &str| Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs(format!(r#"{{"label":"{label}"}}"#)),
        };
        let ask = |tool, text: &str| Input::ToolAnswered {
            tool: Ref::new(tool),
            answer: ToolAnswer::AskVerbatim,
            text: AgentText(text.into()),
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call("day"),
            call("time"),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            ask(0, "Which day?"),
            ask(1, "Which time?"),
            Input::Advance { ms: Millis(5_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Monday at noon".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let mut outputs = BTreeMap::new();
        for (_, ev) in run.record.host_view().events() {
            if let SessionEvent::ConversationItemAdded { item, .. } = ev
                && let ChatItem::FunctionCallOutput(o) = item.as_ref()
            {
                outputs.insert(o.call_id.clone(), o.output.clone());
            }
        }
        assert_eq!(outputs.len(), 2, "outputs: {outputs:?}");
        assert!(
            outputs.values().all(|o| o == "Monday at noon"),
            "an output still waits for the answer: {outputs:?}"
        );
        assert!(outputs.values().all(|o| !o.starts_with(WAITING_PREFIX)));
    }
}
