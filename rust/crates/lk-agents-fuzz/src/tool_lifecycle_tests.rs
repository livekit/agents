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

mod interrupted_during_tools_frees_the_floor {
    //! Interrupting a reply while its tool runs gives the conversation back at once: the
    //! interrupted speech is done and the next reply answered without waiting on the tool, whose
    //! late result still commits, once, with no follow-up reply.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Policy, Run, Setup, StorageSetup, simulate_with};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, UserText};
    use lk_agents_fuzz::record::{Kind, StorageOperation, StorageStep};
    use lk_agents_voice::events::SessionEvent;
    use lk_agents_voice::trace::{SpeechStep, Step};

    /// The reply calls `lookup`, which hangs; the person interrupts at 300 ms and asks again.
    fn interrupted_during_lookup() -> Vec<Input> {
        vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::Interrupt { force: false, wait: false },
            Input::Advance { ms: Millis(50) },
            Input::GenerateReply {
                user_input: Some(UserText("are you there?".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(500) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("Yes.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(1_000) },
        ]
    }

    fn run(late: &[Input]) -> Run {
        run_inputs(&interrupted_during_lookup(), late, 0, Policy::default())
    }

    fn run_inputs(inputs: &[Input], late: &[Input], schedule: u64, policy: Policy) -> Run {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        run_setup(&setup, inputs, late, schedule, policy)
    }

    /// A call with a thread: every turn holds the voice partition until it commits.
    fn run_stored(late: &[Input], schedule: u64, policy: Policy) -> Run {
        let setup = Setup {
            tools: vec!["lookup".into()],
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        run_setup(&setup, &interrupted_during_lookup(), late, schedule, policy)
    }

    fn run_setup(
        setup: &Setup,
        inputs: &[Input],
        late: &[Input],
        schedule: u64,
        policy: Policy,
    ) -> Run {
        let setup = setup.clone();
        simulate_with(schedule, async move {
            let mut run = Run::start(&setup, schedule, policy).await;
            // A speaker that reports its playout, so a reply's audio ends as played.
            for input in inputs.iter().chain(late) {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        })
    }

    fn outputs(run: &Run) -> Vec<String> {
        let events = run.record.host_view().events();
        let added = events.into_iter().filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => Some(item),
            _ => None,
        });
        added
            .filter_map(|item| match item.as_ref() {
                ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect()
    }

    /// When each speech was done, in speech order.
    fn done_at(run: &Run) -> Vec<u64> {
        let order = run.record.speech_order();
        let mut done: Vec<_> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(at, ev)| match ev {
                SessionEvent::SpeechDone { speech, .. } => Some((order[&speech.0], at.at_ms)),
                _ => None,
            })
            .collect();
        done.sort();
        done.into_iter().map(|(_, at)| at).collect()
    }

    // Catches: the interrupted reply's speech held until its tool ended or the 5s interruption
    // watchdog fired, so `interrupt()` and anything awaiting the speech stalled for 5s.
    #[test]
    fn the_interrupted_speech_is_done_while_its_tool_still_runs() {
        let late = [
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("found".into()),
            },
            Input::Advance { ms: Millis(6_000) },
        ];
        let run = run(&late);
        let done = done_at(&run);
        assert!(done.len() == 2 && done[0] < 1_000, "speeches done at {done:?} ms; cut at 300 ms");
        assert_eq!(outputs(&run), ["found"], "the late result commits once");
        assert_eq!(
            run.record.live().opened(Kind::Chat),
            2,
            "no follow-up for the interrupted reply"
        );
    }

    // Catches: a cut landing while the step's end was still on its way, which reached the tools
    // already cut and was held to the watchdog. Every cut point in the step's first 50 ms, under
    // each queue policy: the race is the executor's.
    #[test]
    fn a_cut_at_any_point_after_the_call_is_done_at_once() {
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..16 {
                for after in [0, 1, 2, 5, 10, 20, 50] {
                    let mut inputs = interrupted_during_lookup();
                    inputs[4] = Input::Advance { ms: Millis(after) };
                    let late = [Input::Advance { ms: Millis(6_000) }];
                    let run = run_inputs(&inputs, &late, schedule, policy);
                    let done = done_at(&run);
                    let cut = 100 + after;
                    assert!(
                        done.first().is_some_and(|at| *at < cut + 1_000),
                        "{policy:?} schedule {schedule}, cut {after} ms after the call: \
                         speeches done at {done:?} ms"
                    );
                }
            }
        }
    }

    // Catches: a tool that never answers outliving the reply that gave the floor back: the
    // watchdog still ends it, with one output.
    #[test]
    fn the_watchdog_still_ends_a_hung_tool() {
        let run = run(&[Input::Advance { ms: Millis(6_000) }]);
        assert!(done_at(&run).first().is_some_and(|at| *at < 1_000), "{:?}", done_at(&run));
        assert_eq!(outputs(&run).len(), 1, "outputs: {:?}", outputs(&run));
        assert_eq!(run.record.live().opened(Kind::Chat), 2);
    }
    /// Each turn's storage requests and answers, in order: `(speech, operation, answered)`.
    fn turns(run: &Run) -> Vec<(u64, StorageOperation, bool)> {
        let ended = |op| matches!(op, StorageOperation::Settle | StorageOperation::Abandon);
        run.record
            .host_view()
            .storage()
            .iter()
            .filter_map(|(_, step)| match step {
                StorageStep::Requested { speech, operation } if ended(*operation) => {
                    Some((*speech, *operation, false))
                }
                StorageStep::Answered { speech, operation, .. } if ended(*operation) => {
                    Some((*speech, *operation, true))
                }
                _ => None,
            })
            .collect()
    }

    /// The case's scenario happened: the reply's call started before the person cut it. A seed
    /// whose storage answered late may end the reply before its model ever called the tool.
    fn tool_started(run: &Run) -> bool {
        !run.record.internal_view().tool_calls_started().is_empty()
    }

    fn storage_failures(run: &Run) -> Vec<String> {
        let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
        failures
            .iter()
            .filter(|f| f.contract.starts_with("storage_"))
            .map(|f| format!("{f:?}"))
            .collect()
    }

    /// Each of the two turns ends once, and each end asked for is answered.
    fn assert_each_turn_saved_once(run: &Run) {
        let turns = turns(run);
        let mut asked: Vec<u64> = turns.iter().filter(|t| !t.2).map(|t| t.0).collect();
        let mut answered: Vec<u64> = turns.iter().filter(|t| t.2).map(|t| t.0).collect();
        assert_eq!(asked.len(), 2, "each turn ends once: {turns:?}");
        asked.sort_unstable();
        answered.sort_unstable();
        assert_eq!(answered, asked, "every end answered: {turns:?}");
    }

    // Catches: on a call with a thread, the interrupted reply held the person until its tool ended,
    // or, once freed, its turn never saved or never released its hold on the partition.
    #[test]
    fn on_a_thread_the_floor_is_free_and_each_turn_is_saved() {
        let mut exercised = 0;
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..8 {
                let late = [
                    Input::ToolAnswered {
                        tool: Ref::new(0),
                        answer: ToolAnswer::Value,
                        text: AgentText("found".into()),
                    },
                    Input::Advance { ms: Millis(6_000) },
                ];
                let run = run_stored(&late, schedule, policy);
                if !tool_started(&run) {
                    continue;
                }
                exercised += 1;
                let done = done_at(&run);
                // Cut at 300 ms, the tool answering at 1850 ms: the first is done at once, and the
                // second when it has played, not held for the first turn's commit.
                assert!(
                    done.len() == 2 && done[0] < 1_000 && done[1] < 4_000,
                    "{policy:?} {schedule}: speeches done at {done:?} ms; cut at 300 ms"
                );
                assert_eq!(outputs(&run), ["found"], "{policy:?} {schedule}");
                assert!(
                    turns(&run).iter().all(|t| t.1 == StorageOperation::Settle),
                    "{policy:?} {schedule}: {:?}",
                    turns(&run)
                );
                assert_each_turn_saved_once(&run);
                assert_eq!(storage_failures(&run), Vec::<String>::new(), "{policy:?} {schedule}");
            }
        }
        assert!(exercised >= 12, "the scenario ran in only {exercised} of 24 runs");
    }

    // Catches: an interrupted turn whose tool hangs to the watchdog blocking the next turn's commit
    // forever, or ending twice.
    #[test]
    fn on_a_thread_a_turn_ended_by_the_watchdog_lets_the_next_commit() {
        let mut exercised = 0;
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..8 {
                let late = [Input::Advance { ms: Millis(40_000) }];
                let run = run_stored(&late, schedule, policy);
                if !tool_started(&run) {
                    continue;
                }
                exercised += 1;
                let done = done_at(&run);
                // The watchdog ends the first turn at 5300 ms; neither speech waits for it.
                assert!(
                    done.len() == 2 && done[0] < 1_000 && done[1] < 4_000,
                    "{policy:?} {schedule}: speeches done at {done:?} ms"
                );
                assert_eq!(outputs(&run).len(), 1, "{policy:?} {schedule}: {:?}", outputs(&run));
                assert_each_turn_saved_once(&run);
                let order = run.record.speech_order();
                let second = turns(&run).into_iter().filter(|t| !t.2).max_by_key(|t| order[&t.0]);
                assert!(
                    second.is_some_and(|t| t.1 == StorageOperation::Settle),
                    "{policy:?} {schedule}: the answered turn commits: {:?}",
                    turns(&run)
                );
                assert_eq!(storage_failures(&run), Vec::<String>::new(), "{policy:?} {schedule}");
            }
        }
        assert!(exercised >= 12, "the scenario ran in only {exercised} of 24 runs");
    }
    fn failures_of(run: &Run, contract: &str) -> Vec<String> {
        let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
        failures.iter().filter(|f| f.contract == contract).map(|f| format!("{f:?}")).collect()
    }

    fn closed() -> [Input; 2] {
        [Input::Close { drain: false }, Input::AwaitExit {}]
    }

    // Catches: the session's opening handoff saved after the person's first words, or inside the
    // second turn: a turn's start wrote the model's view, which leaves the handoff out, as the
    // partition's history.
    #[test]
    fn on_a_thread_the_handoff_is_saved_where_it_happened() {
        let inputs = [
            Input::GenerateReply {
                user_input: Some(UserText("book it".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(200) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("found".into()),
            },
            Input::Advance { ms: Millis(300) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("Booked.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(3_000) },
            Input::GenerateReply {
                user_input: Some(UserText("are you there?".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(300) },
            Input::LlmText { chat: Ref::new(2), text: AgentText("Yes.".into()) },
            Input::LlmEnd { chat: Ref::new(2) },
            Input::Advance { ms: Millis(3_000) },
        ];
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..8 {
                let setup = Setup {
                    tools: vec!["lookup".into()],
                    storage: Some(StorageSetup::default()),
                    ..Setup::default()
                };
                let run = run_setup(&setup, &inputs, &closed(), schedule, policy);
                let order = failures_of(&run, "storage_saves_the_conversation_in_order");
                assert_eq!(order, Vec::<String>::new(), "{policy:?} {schedule}");
            }
        }
    }

    // Catches (seed 4, BasicPos): a turn cut while its tool ran, after some of its words were
    // heard. A checkpoint taken once it was cut dropped the call it had already written (the cut
    // reply's run was no longer read) and wrote the heard words in its place, so the thread kept
    // the words before the call the session holds ahead of them.
    #[test]
    fn on_a_thread_a_cut_turn_keeps_its_call_ahead_of_its_words() {
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmText {
                chat: Ref::new(0),
                text: AgentText("Let me look that up for you, one moment please.".into()),
            },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_500) },
            Input::Interrupt { force: true, wait: false },
            Input::Advance { ms: Millis(1_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("found".into()),
            },
            Input::Advance { ms: Millis(6_000) },
        ];
        let mut exercised = 0;
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..8 {
                let setup = Setup {
                    tools: vec!["lookup".into()],
                    storage: Some(StorageSetup::default()),
                    ..Setup::default()
                };
                let run = run_setup(&setup, &inputs, &closed(), schedule, policy);
                let cut = run.record.journal().iter().any(|(_, step)| {
                    matches!(step, Step::Speech(SpeechStep::ReleasedToTools { .. }))
                });
                if !cut {
                    continue;
                }
                exercised += 1;
                let order = failures_of(&run, "storage_saves_the_conversation_in_order");
                assert_eq!(order, Vec::<String>::new(), "{policy:?} {schedule}");
            }
        }
        assert!(exercised >= 12, "the scenario ran in only {exercised} of 24 runs");
    }

    // Catches: the cut turn's late output saved after the next turn the person took meanwhile,
    // or kept at the end of the session's history instead of right after its call.
    #[test]
    fn on_a_thread_a_late_output_is_saved_with_its_turn() {
        let mut exercised = 0;
        for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
            for schedule in 0..8 {
                let mut late = vec![
                    Input::ToolAnswered {
                        tool: Ref::new(0),
                        answer: ToolAnswer::Value,
                        text: AgentText("found".into()),
                    },
                    Input::Advance { ms: Millis(6_000) },
                ];
                late.extend(closed());
                let run = run_stored(&late, schedule, policy);
                if !tool_started(&run) {
                    continue;
                }
                exercised += 1;
                let order = failures_of(&run, "storage_saves_the_conversation_in_order");
                assert_eq!(order, Vec::<String>::new(), "{policy:?} {schedule}");
            }
        }
        assert!(exercised >= 12, "the scenario ran in only {exercised} of 24 runs");
    }
}
