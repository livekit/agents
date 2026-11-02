//! `AgentRun` driven directly, event by event, with no executor: each test plays the runtime's
//! part and asserts the effects and the settled result.
//!
//! Shared drivers and approval/checkpoint/coordinator contracts live here. Larger flow and
//! lifecycle suites use the sibling `agent_run_flow_tests.rs` and `agent_run_lifecycle_tests.rs`.

mod approval {
    //! The approval gate across invocations: a history-only continuation, as `agent.run()` sees it.

    use lk_agents_abi::llm::tool_executor::Approval;

    use super::*;
    use crate::approval;
    use lk_agents_abi::agent::ReadBack;

    fn cancel_tool() -> FunctionToolInfo {
        let mut t = tool(
            "cancel_appointment",
            "Cancel the patient's appointment on a date.\n\nArgs:\n    date: the date",
            json!({
                "type": "object",
                "properties": {"date": {"type": "string"}, "notify": {"type": "boolean", "default": true}},
                "required": ["date"],
            }),
        );
        t.needs_approval = Some(ReadBack::Default);
        t
    }

    /// One invocation on `history`: the model calls the tool with `arguments`; the host validates
    /// them as `validated`. Returns the driver once the run settled or the body was launched.
    fn turn(history: Vec<ChatItem>, message: &str, arguments: Value, validated: Value) -> Driver {
        let mut d = Driver::new(text_agent(vec![cancel_tool()]), message, history);
        d.reply("", &[("cancel_appointment", arguments)]);
        let (call, _) = *d.prepares().last().unwrap();
        d.prepared(call, validated);
        d
    }

    fn question(d: &Driver) -> String {
        match &d.result().outcome {
            RunOutcome::InputRequired { question, verbatim: true, .. } => question.clone(),
            other => panic!("expected a verbatim question, got {other:?}"),
        }
    }

    // Catches (c7 defect 1): a history-only continuation unable to grant (the waiting
    // output never replaced by the answer); approval decided on raw model JSON instead of the
    // validated arguments (a defaulted parameter spelled differently must not ask again); an
    // answer granting twice; the
    // history losing the person's words or the model not told the call has not run.
    #[test]
    fn approval_binds_the_validated_arguments_and_each_answer_is_consumed_once() {
        let validated = json!({"date": "Monday", "notify": true});
        let first = turn(
            vec![],
            "cancel monday",
            json!({"date": "Monday", "notify": true}),
            validated.clone(),
        );
        assert!(first.executes().is_empty(), "the body ran before approval");
        assert_eq!(
            question(&first),
            "Please confirm: Cancel the patient's appointment on a date (date='Monday', notify=True). Yes or no?"
        );
        let history = first.result().all_messages.clone();
        let ChatItem::FunctionCall(asked) = &history[1] else { panic!() };
        assert_eq!(approval::mark(asked), Some(Approval::Asked));

        // The person answers; the model repeats the call with strict mode's `null` for the default.
        let mut second = Driver::new(text_agent(vec![cancel_tool()]), "yes", history);
        let (_, ctx, _) = second.request();
        let outputs: Vec<_> = ctx
            .items
            .iter()
            .filter_map(|i| match i {
                ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs, [approval::answered_text("cancel_appointment", "yes")]);
        second.reply("", &[("cancel_appointment", json!({"date": "Monday", "notify": null}))]);
        let (call, arguments) = second.prepares()[0].clone();
        assert_eq!(
            arguments["notify"],
            json!(true),
            "the schema default replaces the null sentinel"
        );
        second.prepared(call, validated.clone());
        let (_, _, granted) = second.executes().pop().expect("the approved repeat runs");
        assert_eq!(approval::mark(&granted), Some(Approval::Granted));
        second.run_tools(|_| value("cancelled"));
        second.reply("Done, it is cancelled.", &[]);
        let history = second.result().all_messages.clone();
        let stored: Vec<_> =
            transcript(&history).into_iter().filter(|(k, _)| k == "output").collect();
        assert_eq!(stored, pairs(&[("output", "yes"), ("output", "cancelled")]));

        // A later identical call asks again: the answer was consumed.
        let third = turn(history, "cancel monday again", json!({"date": "Monday"}), validated);
        assert!(third.executes().is_empty());
        assert!(question(&third).starts_with("Please confirm:"));
    }

    // Catches: approval surviving a change the application's validator made to the arguments (the
    // identity is the validated value, not the model's JSON).
    #[test]
    fn changed_validated_arguments_need_a_fresh_approval() {
        let first = turn(
            vec![],
            "cancel monday",
            json!({"date": "Monday"}),
            json!({"date": "2026-11-02", "notify": true}),
        );
        let history = first.result().all_messages.clone();
        let mut second = Driver::new(text_agent(vec![cancel_tool()]), "yes", history);
        second.reply("", &[("cancel_appointment", json!({"date": "Monday"}))]);
        let (call, _) = second.prepares()[0].clone();
        // Validated a week later: a different date to approve.
        second.prepared(call, json!({"date": "2026-11-09", "notify": true}));
        assert!(second.executes().is_empty());
        assert!(question(&second).contains("date='2026-11-09'"));
    }

    // Catches: two identical repeats in one step both running on one answer (the grant is decided
    // in model order, whichever preparation finishes first).
    #[test]
    fn one_answer_grants_one_repeat_within_a_step() {
        let validated = json!({"date": "Monday", "notify": true});
        let first = turn(vec![], "cancel", json!({"date": "Monday"}), validated.clone());
        let mut d = Driver::new(
            text_agent(vec![cancel_tool()]),
            "yes",
            first.result().all_messages.clone(),
        );
        d.reply(
            "",
            &[
                ("cancel_appointment", json!({"date": "Monday"})),
                ("cancel_appointment", json!({"date": "Monday"})),
            ],
        );
        let calls = d.prepares();
        d.prepared(calls[1].0, validated.clone());
        assert!(d.executes().is_empty(), "the second waits for the first's decision");
        d.prepared(calls[0].0, validated);
        let executed: Vec<CallId> = d.executes().iter().map(|(c, ..)| *c).collect();
        assert_eq!(executed, [calls[0].0]);
        d.run_tools(|_| value("cancelled"));
        assert!(matches!(&d.result().outcome, RunOutcome::InputRequired { verbatim: true, .. }));
    }

    // Catches: a read-back function's missing-detail question marked as a read-back (a later answer
    // would then grant), or its prepared arguments leaked.
    #[test]
    fn a_read_back_function_may_ask_for_a_missing_detail_first() {
        let mut t = cancel_tool();
        t.needs_approval = Some(ReadBack::Function);
        let mut d = Driver::new(text_agent(vec![t]), "cancel", vec![]);
        d.reply("", &[("cancel_appointment", json!({"date": "Monday"}))]);
        let (call, _) = d.prepares()[0].clone();
        d.prepared(call, json!({"date": "Monday", "notify": true}));
        let prepared = d.effects.iter().find_map(|e| match e {
            Effect::ReadBackRender { prepared, .. } => Some(*prepared),
            _ => None,
        });
        let answer =
            ReadBackAnswer::InputRequired { question: "Which email?".into(), verbatim: false };
        d.send(Event::ReadBackRendered { call, result: Ok(answer) });
        assert!(d.effects.contains(&Effect::ToolArgumentsRelease { prepared: prepared.unwrap() }));
        d.reply("Which email should the confirmation go to?", &[]);
        let ChatItem::FunctionCall(c) = &d.result().all_messages[1] else { panic!() };
        assert_eq!(approval::mark(c), None);
    }

    // Framework validation feedback must not spend the person's approval as a new human reply.
    #[test]
    fn approval_survives_output_validation_feedback_in_the_answering_invocation() {
        let validated = json!({"date": "Monday", "notify": true});
        let first = turn(vec![], "cancel", json!({"date": "Monday"}), validated.clone());
        let mut definition = text_agent(vec![cancel_tool()]);
        definition.output = OutputSpec::Text { validated: true };
        let mut second = Driver::new(definition, "yes", first.result().all_messages.clone());
        second.reply("Done", &[]);
        let validation = second
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::OutputValidate { validation, .. } => Some(*validation),
                _ => None,
            })
            .unwrap();
        second.send(Event::OutputValidated {
            validation,
            verdict: OutputVerdict::Retry("Run the requested tool first".into()),
        });
        second.reply("", &[("cancel_appointment", json!({"date":"Monday"}))]);
        let (call, _) = second.prepares()[0].clone();
        second.prepared(call, validated);
        assert_eq!(second.executes().len(), 1, "validation feedback does not answer a read-back");
    }

    // Catches: an A2A continuation never granting the repeat, because the door wrote the person's
    // answer over the waiting output before the run started and the run counted only the answers it
    // wrote itself; or because stored history read the JSON identity back as the arguments object.
    #[test]
    fn a_read_back_the_transport_answered_grants_the_repeat() {
        let validated = json!({"date": "Monday", "notify": true});
        let first = turn(vec![], "cancel", json!({"date": "Monday"}), validated.clone());
        let mut history = first.result().all_messages.clone();
        for item in &mut history {
            match item {
                ChatItem::FunctionCallOutput(o) => o.output = "yes".into(),
                ChatItem::FunctionCall(c) => {
                    let stored = c.extra[approval::APPROVAL_IDENTITY_KEY].as_str().unwrap();
                    let read_back = serde_json::from_str(stored).unwrap();
                    c.extra.insert(approval::APPROVAL_IDENTITY_KEY.into(), read_back);
                }
                _ => {}
            }
        }
        let mut second = Driver::new(text_agent(vec![cancel_tool()]), "yes", history);
        second.reply("", &[("cancel_appointment", json!({"date": "Monday"}))]);
        let (call, _) = second.prepares()[0].clone();
        second.prepared(call, validated);
        assert_eq!(second.executes().len(), 1, "the approved repeat runs");
    }
}
mod checkpoints {
    //! Managed checkpoints settle before model admission and accepted answer delivery.

    use super::*;

    fn managed(tools: Vec<FunctionToolInfo>) -> Driver {
        let input = RunInput {
            invocation_id: "managed".into(),
            opening: Opening::Message(Box::new(user("message", "Find a room"))),
            history: Vec::new(),
            answer_tool: None,
            delegated: false,
            mode: crate::tool_outcome::Mode::Text,
            llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
            wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
            cleanup_timeout: Duration::from_secs(5),
        };
        let mut driver = Driver {
            run: AgentRun::new(text_agent(tools), input).unwrap().with_checkpoints(),
            now: Ts::ZERO,
            effects: Vec::new(),
            settled: Vec::new(),
        };
        driver.send(Event::Start);
        driver
    }

    fn checkpoint(driver: &Driver) -> CheckpointId {
        driver
            .effects
            .iter()
            .rev()
            .find_map(|effect| match effect {
                Effect::HostCheckpoint { checkpoint, .. } => Some(*checkpoint),
                _ => None,
            })
            .expect("a managed checkpoint")
    }

    fn verdict(driver: &mut Driver, yielded: bool) -> Vec<Effect> {
        driver.send(Event::HostCheckpointDone {
            checkpoint: checkpoint(driver),
            result: Ok(yielded),
            tools: None,
        })
    }

    #[test]
    fn a_waiter_after_a_tool_preserves_its_result_without_starting_the_next_model() {
        let mut driver = managed(vec![string_tool("lookup", "room")]);
        assert_eq!(driver.requests(), 0);
        let first = checkpoint(&driver);
        assert!(
            driver
                .send(Event::HostCheckpointDone {
                    checkpoint: CheckpointId(first.0 + 1),
                    result: Ok(false),
                    tools: None,
                })
                .is_empty(),
            "a stale verdict cannot launch a request"
        );
        verdict(&mut driver, false);
        driver.reply("", &[("lookup", json!({"room":"suite"}))]);
        driver.run_tools(|_| value("suite found"));
        assert_eq!(driver.requests(), 1, "the next model still waits for admission");
        let effects = verdict(&mut driver, true);
        let result = driver.result();
        assert!(result.interrupted);
        assert_eq!(result.usage.requests, 1);
        assert_eq!(result.usage.tool_calls, 1);
        assert_eq!(result.outcome, RunOutcome::NoOutput);
        assert!(result.new_messages.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output) if output.output == "suite found"
        )));
        assert!(!effects.iter().any(|effect| matches!(
            effect,
            Effect::LlmRequestStart { .. } | Effect::Emit(StreamEvent::TextDelta { .. })
        )));
    }

    #[test]
    fn the_accepted_result_waits_for_its_delivery_verdict() {
        let mut driver = managed(Vec::new());
        verdict(&mut driver, false);
        driver.reply("the answer", &[]);
        assert!(driver.finished().is_none());
        verdict(&mut driver, true);
        assert!(driver.result().interrupted);
        assert!(driver.result().new_messages.iter().any(|item| matches!(item,
            ChatItem::Message(message) if message.raw_text_content().as_deref() == Some("the answer")
        )));

        let mut continued = managed(Vec::new());
        verdict(&mut continued, false);
        continued.reply("the accepted answer", &[]);
        verdict(&mut continued, false);
        assert!(!continued.result().interrupted);
        assert_eq!(
            continued.result().outcome,
            RunOutcome::Completed { output: Output::Text("the accepted answer".into()) }
        );
    }

    #[test]
    fn cancellation_waits_for_a_pending_checkpoint_callback_to_settle() {
        let mut driver = managed(Vec::new());
        let pending = checkpoint(&driver);
        let effects = driver.send(Event::Cancel);
        assert!(effects.contains(&Effect::HostCheckpointCancel { checkpoint: pending }));
        assert!(driver.finished().is_none());
        verdict(&mut driver, false);
        assert!(matches!(driver.finished(), Some(Err(RunError::Cancelled))));
        assert_eq!(driver.requests(), 0);
        assert!(
            driver
                .send(Event::HostCheckpointDone {
                    checkpoint: pending,
                    result: Ok(false),
                    tools: None
                })
                .is_empty()
        );
    }

    #[test]
    fn discovered_tools_replace_the_previous_steps_tools_before_request_and_execution() {
        let mut driver = managed(vec![string_tool("search", "query")]);
        verdict(&mut driver, false);
        driver.reply("", &[("search", json!({"query":"room"}))]);
        driver.run_tools(|_| value("found book"));
        let pending = checkpoint(&driver);
        driver.send(Event::HostCheckpointDone {
            checkpoint: pending,
            result: Ok(false),
            tools: Some(vec![string_tool("book", "room")]),
        });
        let (_, _, offered) = driver.request();
        assert_eq!(offered.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["book"]);
        driver.reply("", &[("book", json!({"room":"suite"}))]);
        driver.run_tools(|_| value("booked"));
        verdict(&mut driver, false);
        driver.reply("Reserved.", &[]);
        verdict(&mut driver, false);
        assert!(driver.result().new_messages.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output) if output.output == "booked" && !output.is_error
        )));
    }

    #[test]
    fn an_invalid_discovery_fails_before_the_model_request() {
        let mut driver = managed(Vec::new());
        let tool = string_tool("book", "room");
        driver.send(Event::HostCheckpointDone {
            checkpoint: checkpoint(&driver),
            result: Ok(false),
            tools: Some(vec![tool.clone(), tool]),
        });
        assert_eq!(driver.requests(), 0);
        assert!(
            matches!(driver.finished(), Some(Err(RunError::Host(error))) if error.contains("book"))
        );
    }
}
mod coordinated {
    //! A coordinated invocation (VoiceSession): the coordinator plans every request and decides each
    //! settled step's continuation.

    use super::*;

    fn coordinated(agent: AgentDefinition) -> Driver {
        Driver::with_input(agent, coordinated_input())
    }

    fn coordinated_input() -> RunInput {
        RunInput {
            invocation_id: "reply_1".into(),
            opening: Opening::Coordinated { approval_answers: Default::default() },
            history: vec![user("msg_1", "Book me Monday").into()],
            answer_tool: None,
            delegated: false,
            mode: crate::tool_outcome::Mode::Voice,
            llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
            wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
            cleanup_timeout: Duration::from_secs(5),
        }
    }

    struct Plan {
        request: RequestId,
        message_id: String,
        pending: Vec<ChatItem>,
        tools_forbidden: bool,
    }

    impl Driver {
        fn plans(&self) -> Vec<Plan> {
            (self.effects.iter())
                .filter_map(|e| match e {
                    Effect::PlanRequest {
                        request, message_id, pending, tools_forbidden, ..
                    } => Some(Plan {
                        request: *request,
                        message_id: message_id.clone(),
                        pending: pending.clone(),
                        tools_forbidden: *tools_forbidden,
                    }),
                    _ => None,
                })
                .collect()
        }

        fn respond(&mut self, request: RequestId, text: &str, calls: &[(&str, Value)]) {
            let calls = (calls.iter().enumerate())
                .map(|(i, (name, arguments))| FunctionToolCall {
                    name: (*name).into(),
                    arguments: arguments.to_string(),
                    call_id: format!("call_{}_{i}", request.0),
                    extra: None,
                })
                .collect();
            let text = (!text.is_empty()).then_some(text);
            self.send(Event::LlmChunk { request, chunk: chunk(text, calls, None) });
            self.send(Event::LlmEnded { request });
        }

        fn step_settled(&self) -> (StepId, Vec<ChatItem>, bool) {
            (self.effects.iter().rev())
                .find_map(|e| match e {
                    Effect::HandleToolResults { step, items, continues, .. } => {
                        Some((*step, items.clone(), *continues))
                    }
                    _ => None,
                })
                .expect("a settled step")
        }
    }

    // Catches: a coordinated run planning its own request (rendering instructions, pushing a
    // message), the next step requested before the coordinator's verdict, the step's items missing
    // from the next plan, or a forbidding verdict not reaching the next request.
    #[test]
    fn the_coordinator_plans_each_step_and_decides_its_continuation() {
        let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
        assert!(d.effects.iter().all(|e| !matches!(e, Effect::LlmRequestStart { .. })));
        let [first] = &d.plans()[..] else { panic!("one plan: {:?}", d.effects) };
        assert!(first.pending.is_empty());
        assert!(!first.tools_forbidden);
        let (request, message_id) = (first.request, first.message_id.clone());

        d.respond(request, "Let me look.", &[("find_slot", json!({"day": "Monday"}))]);
        d.run_tools(|_| value("9:00 is free"));
        let (step, items, continues) = d.step_settled();
        assert!(continues);
        assert_eq!(
            transcript(&items),
            pairs(&[("call", r#"find_slot({"day":"Monday"})"#), ("output", "9:00 is free")])
        );
        assert_eq!(d.plans().len(), 1, "no step before the verdict");

        d.send(Event::ToolResultsHandled {
            step,
            continuation: AgentRunContinuation::Continue { tools_forbidden: true },
            reply_required: Default::default(),
        });
        let plans = d.plans();
        let second = &plans[1];
        assert!(second.tools_forbidden);
        assert_eq!(second.pending[0].id(), message_id);
        assert_eq!(
            transcript(&second.pending),
            pairs(&[
                ("assistant", "Let me look."),
                ("call", r#"find_slot({"day":"Monday"})"#),
                ("output", "9:00 is free"),
            ])
        );

        d.respond(second.request, "Monday 9:00 is free.", &[]);
        assert_eq!(
            d.result().outcome,
            RunOutcome::Completed { output: Output::Text("Monday 9:00 is free.".into()) }
        );
    }

    // Catches: an interrupted reply requesting another step or settling as a plain completion, or
    // an ending verdict leaving the run open.
    #[test]
    fn an_interrupted_or_ended_step_ends_the_run_without_another_request() {
        for (verdict, interrupted) in
            [(AgentRunContinuation::Interrupted, true), (AgentRunContinuation::Stop, false)]
        {
            let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
            let request = d.plans()[0].request;
            d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
            d.run_tools(|_| value("9:00 is free"));
            let (step, ..) = d.step_settled();
            d.send(Event::ToolResultsHandled {
                step,
                continuation: verdict,
                reply_required: Default::default(),
            });
            assert_eq!(d.plans().len(), 1);
            let result = d.result();
            assert_eq!(result.outcome, RunOutcome::NoOutput);
            assert_eq!(result.interrupted, interrupted);
        }
    }

    // Catches: a coordinator's interruption discarding an output the run already accepted.
    #[test]
    fn an_accepted_output_survives_an_interrupting_verdict() {
        let mut d = coordinated(agent(
            vec![],
            OutputSpec::Structured { schema: json!({"type": "integer"}) },
        ));
        let request = d.plans()[0].request;
        d.respond(request, "", &[("final_result", json!({"result": 4}))]);
        let validation = (d.effects.iter())
            .find_map(|e| match e {
                Effect::OutputValidate { validation, .. } => Some(*validation),
                _ => None,
            })
            .expect("the candidate is validated");
        d.send(Event::OutputValidated {
            validation,
            verdict: OutputVerdict::Accepted { text: None },
        });
        let (step, _, continues) = d.step_settled();
        assert!(!continues);
        d.send(Event::ToolResultsHandled {
            step,
            continuation: AgentRunContinuation::Interrupted,
            reply_required: Default::default(),
        });
        assert_eq!(
            d.result().outcome,
            RunOutcome::Completed { output: Output::Validated(validation) }
        );
    }

    #[test]
    fn terminal_tool_selection_and_canonical_items_follow_model_order_not_completion_order() {
        for reversed in [false, true] {
            let tools = vec![string_tool("flight", "day"), string_tool("hotel", "city")];
            let mut d = coordinated(text_agent(tools));
            let request = d.plans()[0].request;
            d.respond(
                request,
                "",
                &[("flight", json!({"day": "?"})), ("hotel", json!({"city": "?"}))],
            );
            for (call, arguments) in d.prepares() {
                d.prepared(call, Value::Object(arguments));
            }
            let mut calls = d.executes();
            let selected = calls[0].2.call_id.clone();
            if reversed {
                calls.reverse();
            }
            for (call, _, item) in calls {
                d.send(Event::ToolSettled {
                    call,
                    outcome: ToolOutcome::InputRequired {
                        question: format!("{}?", item.name),
                        verbatim: true,
                        tool: Some(item.name),
                    },
                });
            }
            let (step, items, continues) = d.step_settled();
            assert!(!continues);
            let selected_effect = d.effects.iter().rev().find_map(|effect| match effect {
                Effect::HandleToolResults { terminal_call, .. } => terminal_call.as_deref(),
                _ => None,
            });
            assert_eq!(selected_effect, Some(selected.as_str()));
            assert!(matches!(
                items.as_slice(),
                [
                    ChatItem::FunctionCall(_),
                    ChatItem::FunctionCall(_),
                    ChatItem::FunctionCallOutput(_),
                    ChatItem::FunctionCallOutput(_)
                ]
            ));
            let ChatItem::FunctionCallOutput(second) = &items[3] else { unreachable!() };
            assert!(!second.reply_required, "an unselected question is not another pending reply");
            d.send(Event::ToolResultsHandled {
                step,
                continuation: AgentRunContinuation::Continue { tools_forbidden: false },
                reply_required: Default::default(),
            });
            assert_eq!(
                d.result().outcome,
                RunOutcome::InputRequired {
                    question: "flight?".into(),
                    verbatim: true,
                    answer_tool: Some("flight".into()),
                }
            );
            assert_eq!(d.result().pending_call.as_ref().unwrap().call_id, selected);
        }
    }

    #[test]
    fn retry_budget_failure_has_no_selected_terminal_delivery_and_cannot_be_reopened() {
        let mut definition =
            text_agent(vec![string_tool("retry", "day"), string_tool("ask", "city")]);
        definition.retries.tools = 0;
        let mut d = coordinated(definition);
        let request = d.plans()[0].request;
        d.respond(
            request,
            "",
            &[("retry", json!({"day": "Monday"})), ("ask", json!({"city": "?"}))],
        );
        d.run_tools(|name| {
            if name == "retry" {
                ToolOutcome::ModelRetry("try again".into())
            } else {
                ToolOutcome::InputRequired {
                    question: "city?".into(),
                    verbatim: true,
                    tool: Some("ask".into()),
                }
            }
        });
        let (step, items, _) = d.step_settled();
        assert!(
            d.effects.iter().any(|effect| matches!(
                effect,
                Effect::HandleToolResults { terminal_call: None, .. }
            ))
        );
        let edits = items
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCallOutput(output) => Some((output.call_id.clone(), true)),
                _ => None,
            })
            .collect();
        d.send(Event::ToolResultsHandled {
            step,
            continuation: AgentRunContinuation::Continue { tools_forbidden: false },
            reply_required: edits,
        });
        assert!(
            matches!(d.finished(), Some(Err(RunError::ToolRetriesExceeded { tool, .. })) if tool == "retry")
        );
        assert_eq!(d.plans().len(), 1);
    }

    #[test]
    fn coordinator_reply_edits_update_canonical_history_and_can_end_an_ordinary_step() {
        let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
        let request = d.plans()[0].request;
        d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
        d.run_tools(|_| value("9:00 is free"));
        let (step, items, _) = d.step_settled();
        let ChatItem::FunctionCallOutput(output) = &items[1] else { panic!() };
        d.send(Event::ToolResultsHandled {
            step: StepId(step.0 + 100),
            continuation: AgentRunContinuation::Stop,
            reply_required: [(output.call_id.clone(), false)].into(),
        });
        assert!(d.finished().is_none(), "a stale coordinator response cannot settle this step");
        d.send(Event::ToolResultsHandled {
            step,
            continuation: AgentRunContinuation::Continue { tools_forbidden: false },
            reply_required: [(output.call_id.clone(), false), ("unknown_call".into(), true)].into(),
        });
        assert_eq!(d.result().outcome, RunOutcome::NoOutput);
        let edited = d
            .result()
            .all_messages
            .iter()
            .find_map(|item| match item {
                ChatItem::FunctionCallOutput(item) if item.id == output.id => Some(item),
                _ => None,
            })
            .unwrap();
        assert!(!edited.reply_required);
        assert_eq!(d.plans().len(), 1, "a suppressed ordinary reply must not request another step");
    }

    #[test]
    fn a_host_can_request_an_ordinary_reply_without_reviving_an_explicit_stop() {
        for explicit_stop in [false, true] {
            let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
            let request = d.plans()[0].request;
            d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
            d.run_tools(|_| {
                if explicit_stop {
                    ToolOutcome::StopResponse
                } else {
                    ToolOutcome::Value {
                        text: "done".into(),
                        reply_required: false,
                        verbatim: false,
                        more: None,
                    }
                }
            });
            let (step, items, continues) = d.step_settled();
            assert!(!continues);
            let ChatItem::FunctionCallOutput(output) = &items[1] else { panic!() };
            d.send(Event::ToolResultsHandled {
                step,
                continuation: AgentRunContinuation::Continue { tools_forbidden: true },
                reply_required: [(output.call_id.clone(), true)].into(),
            });
            if explicit_stop {
                assert_eq!(d.result().outcome, RunOutcome::NoOutput);
                assert_eq!(d.plans().len(), 1);
            } else {
                let plans = d.plans();
                assert_eq!(plans.len(), 2);
                assert!(plans[1].tools_forbidden);
                assert!(plans[1].pending.iter().any(|item| matches!(item, ChatItem::FunctionCallOutput(item) if item.id == output.id && item.reply_required)));
            }
        }
    }

    // Catches: a coordinated run refusing to start without a model of its own (a realtime session's
    // coordinator streams from its server), or a call's first update leaving its step unsettled.
    #[test]
    fn a_call_detaching_with_its_first_update_settles_its_step() {
        let mut input = coordinated_input();
        input.llm = None;
        let mut d = Driver::with_input(text_agent(vec![string_tool("find_slot", "day")]), input);
        let request = d.plans()[0].request;
        d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
        d.run_tools(|_| ToolOutcome::Update {
            message: "Still looking.".into(),
            raw: "looking".into(),
        });
        let (_, items, continues) = d.step_settled();
        assert!(continues);
        assert_eq!(transcript(&items)[1], ("output".to_owned(), "Still looking.".to_owned()));
    }

    // Catches: a coordinator unable to find a call it fed by the model's id, or a call that settled
    // with the step (a validated `final_result`) unreadable before the verdict.
    #[test]
    fn a_settled_steps_calls_stay_readable_until_the_verdict() {
        let mut d = coordinated(agent(
            vec![],
            OutputSpec::Structured { schema: json!({"type": "integer"}) },
        ));
        let request = d.plans()[0].request;
        d.respond(request, "", &[("final_result", json!({"result": 4}))]);
        let call = d.run.step_call_id(&format!("call_{}_0", request.0)).expect("the call");
        assert!(matches!(d.run.step_call(call), Some((_, None))), "validating");
        let validation = (d.effects.iter())
            .find_map(|e| match e {
                Effect::OutputValidate { validation, .. } => Some(*validation),
                _ => None,
            })
            .expect("the candidate is validated");
        d.send(Event::OutputValidated {
            validation,
            verdict: OutputVerdict::Accepted { text: None },
        });
        d.step_settled();
        let (item, settled) = d.run.step_call(call).expect("readable until the verdict");
        assert_eq!(item.name, "final_result");
        assert!(settled.is_some());
    }

    // Catches: a coordinator's call recorded under an id of the run's own, so the coordinator's
    // record and the run's next plan hold the call twice.
    #[test]
    fn a_call_the_coordinator_made_keeps_its_item() {
        let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
        let request = d.plans()[0].request;
        let call = lk_agents_abi::FunctionCall {
            id: "voice_item_7".into(),
            call_id: "call_a".into(),
            arguments: json!({"day": "Monday"}).to_string(),
            name: "find_slot".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        d.send(Event::CallMade { request, call });
        d.send(Event::LlmEnded { request });
        d.run_tools(|_| value("9:00 is free"));
        let (_, items, _) = d.step_settled();
        assert_eq!(items[0].id(), "voice_item_7");
        assert_eq!(items[1].id(), "voice_item_7_out");
    }

    fn approval_tool() -> FunctionToolInfo {
        let mut t = string_tool("transfer", "to");
        t.needs_approval = Some(lk_agents_abi::agent::ReadBack::Default);
        t
    }

    /// A `transfer` call to `to` whose read-back was put to the person.
    fn asked(call_id: &str, to: &str) -> ChatItem {
        let identity = lk_agents_abi::agent::ArgumentIdentity::Json(json!({"to": to}));
        let mut call = lk_agents_abi::FunctionCall {
            id: format!("item_{call_id}"),
            call_id: call_id.into(),
            arguments: json!({"to": to}).to_string(),
            name: "transfer".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        let canonical = crate::approval::canonical(&identity);
        crate::approval::set_mark(
            &mut call,
            lk_agents_abi::llm::tool_executor::Approval::Asked,
            &canonical,
        );
        ChatItem::FunctionCall(call)
    }

    /// The model repeats `transfer(to)` on the first step; whether its body runs.
    fn repeat_runs(d: &mut Driver, to: &str) -> bool {
        let request = d.plans()[0].request;
        d.respond(request, "", &[("transfer", json!({"to": to}))]);
        let (call, arguments) = d.prepares()[0].clone();
        d.prepared(call, Value::Object(arguments));
        !d.executes().is_empty()
    }

    // Catches: a coordinated step granting only with the answers it opened with, though the person
    // answered before it called (the session commits the answer after the reply exists).
    #[test]
    fn a_step_that_has_not_called_reads_the_latest_answers() {
        let mut input = coordinated_input();
        input.history.push(asked("asked_1", "billing"));
        let mut d = Driver::with_input(text_agent(vec![approval_tool()]), input);
        d.send(Event::Answered { approval_answers: ["asked_1".to_owned()].into() });
        assert!(repeat_runs(&mut d, "billing"));
    }

    // Catches: a read-back another reply put after this run opened ignored, so an answer to the
    // earlier question grants a call though the person was asked something else last.
    #[test]
    fn a_read_back_recorded_by_another_reply_is_the_latest_question() {
        let mut input = coordinated_input();
        input.history.push(asked("asked_1", "billing"));
        input.opening = Opening::Coordinated { approval_answers: ["asked_1".to_owned()].into() };
        let mut d = Driver::with_input(text_agent(vec![approval_tool()]), input);
        d.send(Event::Recorded { item: Box::new(asked("asked_2", "sales")) });
        assert!(!repeat_runs(&mut d, "billing"));
    }

    // Catches: an answer the person moved on from still granting a later call of a step that had
    // started calling before they spoke again.
    #[test]
    fn speaking_again_revokes_the_answers_a_calling_step_read() {
        let mut input = coordinated_input();
        input.history.push(asked("asked_1", "billing"));
        input.opening = Opening::Coordinated { approval_answers: ["asked_1".to_owned()].into() };
        let mut d = Driver::with_input(
            text_agent(vec![approval_tool(), string_tool("lookup", "q")]),
            input,
        );
        let request = d.plans()[0].request;
        d.send(Event::LlmChunk {
            request,
            chunk: chunk(
                None,
                vec![FunctionToolCall {
                    name: "lookup".into(),
                    arguments: json!({"q": "x"}).to_string(),
                    call_id: "c0".into(),
                    extra: None,
                }],
                None,
            ),
        });
        d.send(Event::Answered { approval_answers: Default::default() });
        d.send(Event::LlmChunk {
            request,
            chunk: chunk(
                None,
                vec![FunctionToolCall {
                    name: "transfer".into(),
                    arguments: json!({"to": "billing"}).to_string(),
                    call_id: "c1".into(),
                    extra: None,
                }],
                None,
            ),
        });
        for (call, arguments) in d.prepares() {
            d.prepared(call, Value::Object(arguments));
        }
        assert!(d.executes().iter().all(|(_, _, c)| c.name != "transfer"));
    }
}
#[path = "agent_run_flow_tests.rs"]
mod flow;
#[path = "agent_run_lifecycle_tests.rs"]
mod lifecycle;

use std::time::Duration;

use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{
    ChatContext, ChatItem, ChatMessage, ChoiceDelta, FunctionToolCall, LlmChunk, ProviderId, Role,
    ToolSpec, Ts, WallAnchor,
};
use serde_json::{Map, Value, json};

use super::*;
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, Retries};

pub(super) const LLM: ResourceId = ResourceId(7);

pub(super) fn tool(name: &str, description: &str, schema: Value) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: name.into(),
            description: description.into(),
            json_schema: schema,
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: None,
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    }
}

pub(super) fn string_tool(name: &str, parameter: &str) -> FunctionToolInfo {
    tool(
        name,
        &format!("Use {name}."),
        json!({"type": "object", "properties": {parameter: {"type": "string"}}, "required": [parameter]}),
    )
}

pub(super) fn agent(tools: Vec<FunctionToolInfo>, output: OutputSpec) -> AgentDefinition {
    AgentDefinition {
        name: "scheduler".into(),
        instructions: "Book appointments.".into(),
        dynamic_instructions: false,
        tools,
        output,
        retries: Retries::default(),
    }
}

pub(super) fn text_agent(tools: Vec<FunctionToolInfo>) -> AgentDefinition {
    agent(tools, OutputSpec::Text { validated: false })
}

pub(super) fn user(id: &str, text: &str) -> ChatMessage {
    ChatMessage::text(id, Role::User, text, std::time::UNIX_EPOCH)
}

/// The runtime's side of one invocation: it delivers events at a supplied instant and keeps every
/// effect the run returned.
pub(super) struct Driver {
    pub(super) run: AgentRun,
    pub(super) now: Ts,
    pub(super) effects: Vec<Effect>,
    /// Calls whose body the driver settled.
    settled: Vec<CallId>,
}

impl Driver {
    pub(super) fn new(agent: AgentDefinition, message: &str, history: Vec<ChatItem>) -> Driver {
        Driver::with_input(
            agent,
            RunInput {
                invocation_id: "run_1".into(),
                opening: Opening::Message(Box::new(user("msg_1", message))),
                history,
                answer_tool: None,
                delegated: false,
                mode: crate::tool_outcome::Mode::Text,
                llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
                wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
                cleanup_timeout: Duration::from_secs(5),
            },
        )
    }

    pub(super) fn with_input(agent: AgentDefinition, input: RunInput) -> Driver {
        let mut driver = Driver {
            run: AgentRun::new(agent, input).unwrap(),
            now: Ts::ZERO,
            effects: Vec::new(),
            settled: Vec::new(),
        };
        driver.send(Event::Start);
        driver
    }

    /// Delivers one event; the effects it caused.
    pub(super) fn send(&mut self, event: Event) -> Vec<Effect> {
        if let Event::ToolSettled { call, .. } = &event {
            self.settled.push(*call);
        }
        let fx = self.run.handle(event, self.now);
        self.effects.extend(fx.iter().cloned());
        fx
    }

    pub(super) fn advance(&mut self, by: Duration) {
        self.now = self.now.saturating_add(by);
    }

    /// The last model request: its id, history and tools.
    pub(super) fn request(&self) -> (RequestId, ChatContext, Vec<ToolSpec>) {
        self.effects
            .iter()
            .rev()
            .find_map(|e| match e {
                Effect::LlmRequestStart { request, llm, chat_ctx, tools } => {
                    assert_eq!(*llm, LLM);
                    Some((*request, chat_ctx.clone(), tools.clone()))
                }
                _ => None,
            })
            .expect("a model request")
    }

    pub(super) fn requests(&self) -> usize {
        self.effects.iter().filter(|e| matches!(e, Effect::LlmRequestStart { .. })).count()
    }

    /// The model answers the last request with text and calls `(name, arguments)`.
    pub(super) fn reply(&mut self, text: &str, calls: &[(&str, Value)]) -> Vec<Effect> {
        let (request, _, _) = self.request();
        let n = self.requests();
        if !text.is_empty() {
            self.send(Event::LlmChunk { request, chunk: chunk(Some(text), vec![], None) });
        }
        let calls = calls
            .iter()
            .enumerate()
            .map(|(i, (name, arguments))| FunctionToolCall {
                name: (*name).into(),
                arguments: arguments.to_string(),
                call_id: format!("call_{n}_{i}"),
                extra: None,
            })
            .collect();
        let usage = TokenUsage { input_tokens: 10, output_tokens: 3, ..Default::default() };
        self.send(Event::LlmChunk { request, chunk: chunk(None, calls, Some(usage)) });
        self.send(Event::LlmEnded { request })
    }

    /// Every `ToolPrepare` so far: the call and its arguments.
    pub(super) fn prepares(&self) -> Vec<(CallId, Map<String, Value>)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                Effect::ToolPrepare { call, arguments, .. } => Some((*call, arguments.clone())),
                _ => None,
            })
            .collect()
    }

    pub(super) fn executes(&self) -> Vec<(CallId, ResourceId, lk_agents_abi::FunctionCall)> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                Effect::ToolExecute { call, prepared, function_call, .. } => {
                    Some((*call, *prepared, function_call.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// The host prepares the call with `identity` as its validated arguments.
    pub(super) fn prepared(&mut self, call: CallId, identity: Value) -> Vec<Effect> {
        let prepared = ResourceId(1000 + call.0);
        self.send(Event::ToolPrepared {
            call,
            prepared,
            identity: ArgumentIdentity::Json(identity),
        })
    }

    /// Prepares every pending call with its arguments as sent, then settles each body with
    /// `outcome`.
    pub(super) fn run_tools(&mut self, outcome: impl Fn(&str) -> ToolOutcome) {
        for (call, arguments) in self.prepares() {
            if self.executes().iter().all(|(c, ..)| *c != call) {
                self.prepared(call, Value::Object(arguments));
            }
        }
        for (call, _, function_call) in self.executes() {
            if !self.settled(call) {
                self.send(Event::ToolSettled { call, outcome: outcome(&function_call.name) });
            }
        }
    }

    fn settled(&self, call: CallId) -> bool {
        self.settled.contains(&call)
    }

    pub(super) fn finished(&self) -> Option<&Result<RunResult, RunError>> {
        self.effects.iter().find_map(|e| match e {
            Effect::Finished(r) => Some(r),
            _ => None,
        })
    }

    pub(super) fn result(&self) -> &RunResult {
        match self.finished() {
            Some(Ok(result)) => result,
            other => panic!("expected a result, got {other:?}"),
        }
    }
}

pub(super) fn chunk(
    text: Option<&str>,
    tool_calls: Vec<FunctionToolCall>,
    usage: Option<TokenUsage>,
) -> LlmChunk {
    LlmChunk {
        id: "chunk".into(),
        delta: Some(ChoiceDelta {
            role: Some(Role::Assistant),
            content: text.map(Into::into),
            tool_calls,
            extra: None,
        }),
        usage: usage.map(Box::new),
        flush: false,
        node_text: false,
    }
}

pub(super) fn value(text: &str) -> ToolOutcome {
    ToolOutcome::Value { text: text.into(), reply_required: true, verbatim: false, more: None }
}

/// The history as `(kind, text)` pairs: messages by role, calls by name, outputs by text.
pub(super) fn transcript(items: &[ChatItem]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|item| match item {
            ChatItem::Message(m) => {
                (format!("{:?}", m.role).to_lowercase(), m.raw_text_content().unwrap_or_default())
            }
            ChatItem::FunctionCall(c) => ("call".into(), format!("{}({})", c.name, c.arguments)),
            ChatItem::FunctionCallOutput(o) => ("output".into(), o.output.clone()),
            other => ("other".into(), other.id().to_owned()),
        })
        .collect()
}

pub(super) fn pairs(rows: &[(&str, &str)]) -> Vec<(String, String)> {
    rows.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}
