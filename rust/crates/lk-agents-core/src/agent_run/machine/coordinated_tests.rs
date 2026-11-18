use super::super::tests::*;
use super::super::*;
use lk_agents_abi::FunctionToolCall;
use lk_agents_abi::agent::FunctionToolInfo;
use serde_json::json;

fn coordinated(agent: AgentDefinition) -> Driver {
    Driver::with_input(agent, coordinated_input())
}

fn coordinated_input() -> RunInput {
    RunInput {
        invocation_id: "reply_1".into(),
        mode: crate::tool_outcome::Mode::Voice,
        ..input(
            Opening::Coordinated { approval_answers: Default::default() },
            vec![user("msg_1", "Book me Monday").into()],
        )
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
                AgentRunEffect::PlanRequest {
                    request,
                    message_id,
                    pending,
                    tools_forbidden,
                    ..
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
        self.send(AgentRunEvent::LlmChunk { request, chunk: chunk(text, calls, None) });
        self.send(AgentRunEvent::LlmEnded { request });
    }

    fn step_settled(&self) -> (StepId, Vec<ChatItem>, bool) {
        (self.effects.iter().rev())
            .find_map(|e| match e {
                AgentRunEffect::HandleToolResults { step, items, continues, .. } => {
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
    assert!(d.effects.iter().all(|e| !matches!(e, AgentRunEffect::LlmRequestStart { .. })));
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

    d.send(AgentRunEvent::ToolResultsHandled {
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
        d.send(AgentRunEvent::ToolResultsHandled {
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
    let mut d =
        coordinated(agent(vec![], OutputSpec::Structured { schema: json!({"type": "integer"}) }));
    let request = d.plans()[0].request;
    d.respond(request, "", &[("final_result", json!({"result": 4}))]);
    let validation = d.validation();
    d.send(AgentRunEvent::OutputValidated {
        validation,
        verdict: OutputVerdict::Accepted { text: None },
    });
    let (step, _, continues) = d.step_settled();
    assert!(!continues);
    d.send(AgentRunEvent::ToolResultsHandled {
        step,
        continuation: AgentRunContinuation::Interrupted,
        reply_required: Default::default(),
    });
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Validated(validation) });
}

#[test]
fn terminal_tool_selection_and_canonical_items_follow_model_order_not_completion_order() {
    for reversed in [false, true] {
        let tools = vec![string_tool("flight", "day"), string_tool("hotel", "city")];
        let mut d = coordinated(text_agent(tools));
        let request = d.plans()[0].request;
        d.respond(request, "", &[("flight", json!({"day": "?"})), ("hotel", json!({"city": "?"}))]);
        for (call, arguments) in d.prepares() {
            d.prepared(call, Value::Object(arguments));
        }
        let mut calls = d.executes();
        let selected = calls[0].2.call_id.clone();
        if reversed {
            calls.reverse();
        }
        for (call, _, item) in calls {
            d.send(AgentRunEvent::ToolSettled {
                call,
                report: (ToolOutcome::InputRequired {
                    question: format!("{}?", item.name),
                    verbatim: true,
                })
                .into(),
            });
        }
        let (step, items, continues) = d.step_settled();
        assert!(!continues);
        let selected_effect = d.effects.iter().rev().find_map(|effect| match effect {
            AgentRunEffect::HandleToolResults { terminal_call, .. } => terminal_call.as_deref(),
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
        assert!(!second.reply_required, "waiting on a human does not request a model reply");
        assert_eq!(second.output, "waiting for the person's answer to: hotel?");
        let requirements = d
            .effects
            .iter()
            .rev()
            .find_map(|effect| match effect {
                AgentRunEffect::HandleToolResults { input_requirements, .. } => {
                    Some(input_requirements)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(requirements.len(), 2);
        assert_eq!(requirements[0].call.call_id, selected);
        assert_eq!(requirements[1].question, "hotel?");
        d.send(AgentRunEvent::ToolResultsHandled {
            step,
            continuation: AgentRunContinuation::Continue { tools_forbidden: false },
            reply_required: Default::default(),
        });
        assert_eq!(
            d.result().outcome,
            RunOutcome::InputRequired(RunQuestion { question: "flight?".into(), verbatim: true })
        );
        assert_eq!(d.result().pending_call.as_ref().unwrap().call_id, selected);
        assert_eq!(d.result().input_requirements.len(), 2);
    }
}

// Catches: a failed batch selecting a question to answer, a failed run reopened by the
// coordinator's verdict, or the failure dropping a sibling's retained question.
#[test]
fn retry_budget_failure_has_no_selected_terminal_delivery_and_cannot_be_reopened() {
    let mut definition = text_agent(vec![string_tool("retry", "day"), string_tool("ask", "city")]);
    definition.retries.tools = 0;
    let mut d = coordinated(definition);
    let request = d.plans()[0].request;
    d.respond(request, "", &[("retry", json!({"day": "Monday"})), ("ask", json!({"city": "?"}))]);
    d.run_tools(|name| {
        if name == "retry" {
            ToolOutcome::ModelRetry("try again".into())
        } else {
            ToolOutcome::InputRequired { question: "city?".into(), verbatim: true }
        }
    });
    let (step, items, _) = d.step_settled();
    assert!(d.effects.iter().any(|effect| matches!(
        effect,
        AgentRunEffect::HandleToolResults { terminal_call: None, .. }
    )));
    let unrelated =
        Driver::new(text_agent(vec![string_tool("ask", "city")]), "unrelated", items.clone());
    let (_, planned, _) = unrelated.request();
    assert!(
        planned.items.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output)
                if output.output == "waiting for the person's answer to: city?"
        )),
        "a failed batch's question stays pending as model context"
    );
    let edits = items
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) => Some((output.call_id.clone(), true)),
            _ => None,
        })
        .collect();
    d.send(AgentRunEvent::ToolResultsHandled {
        step,
        continuation: AgentRunContinuation::Continue { tools_forbidden: false },
        reply_required: edits,
    });
    assert!(
        matches!(d.finished(), Some(Err(RunError::ToolRetriesExceeded { tool, .. })) if tool == "retry")
    );
    assert_eq!(d.plans().len(), 1);
    let requirements = d
        .effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::Finished { input_requirements, .. } => Some(input_requirements),
            _ => None,
        })
        .unwrap();
    assert_eq!(requirements.len(), 1);
    assert_eq!(requirements[0].question, "city?");
    assert_eq!(requirements[0].call.name, "ask");
}

#[test]
fn coordinator_reply_edits_update_canonical_history_and_can_end_an_ordinary_step() {
    let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
    let request = d.plans()[0].request;
    d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
    d.run_tools(|_| value("9:00 is free"));
    let (step, items, _) = d.step_settled();
    let ChatItem::FunctionCallOutput(output) = &items[1] else { panic!() };
    d.send(AgentRunEvent::ToolResultsHandled {
        step: StepId(step.0 + 100),
        continuation: AgentRunContinuation::Stop,
        reply_required: [(output.call_id.clone(), false)].into(),
    });
    assert!(d.finished().is_none(), "a stale coordinator response cannot settle this step");
    d.send(AgentRunEvent::ToolResultsHandled {
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
        d.send(AgentRunEvent::ToolResultsHandled {
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
        reply_required: true,
    });
    let (_, items, continues) = d.step_settled();
    assert!(continues);
    assert_eq!(transcript(&items)[1], ("output".to_owned(), "Still looking.".to_owned()));
}

// Catches: a coordinator unable to find a call it fed by the model's id, or a call that settled
// with the step (a validated `final_result`) unreadable before the verdict.
#[test]
fn a_settled_steps_calls_stay_readable_until_the_verdict() {
    let mut d =
        coordinated(agent(vec![], OutputSpec::Structured { schema: json!({"type": "integer"}) }));
    let request = d.plans()[0].request;
    d.respond(request, "", &[("final_result", json!({"result": 4}))]);
    let call = d.run.step_call_id(&format!("call_{}_0", request.0)).expect("the call");
    assert!(matches!(d.run.step_call(call), Some((_, None))), "validating");
    let validation = d.validation();
    d.send(AgentRunEvent::OutputValidated {
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
    d.send(AgentRunEvent::CallMade { request, call });
    d.send(AgentRunEvent::LlmEnded { request });
    d.run_tools(|_| value("9:00 is free"));
    let (_, items, _) = d.step_settled();
    assert_eq!(items[0].id(), "voice_item_7");
    assert_eq!(items[1].id(), "voice_item_7_out");
}

// Catches: an output stamped from the run's clock before the call the coordinator stamped
// finer, so the history held the output ahead of its call and the person's answer went to the
// other reply's question (DST tool_output_follows_outcome, VOICE_SEED=5264 VOICE_POLICY=Arrival).
#[test]
fn an_output_is_never_stamped_before_its_call() {
    let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
    let request = d.plans()[0].request;
    // The run's clock reads the instant as 1_700_000_000.0; the coordinator stamped the call later
    // in that same instant.
    let made = std::time::UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 13_000);
    let call = lk_agents_abi::FunctionCall {
        id: "voice_item_7".into(),
        call_id: "call_a".into(),
        arguments: json!({"day": "Monday"}).to_string(),
        name: "find_slot".into(),
        created_at: made,
        extra: Default::default(),
        group_id: None,
    };
    d.send(AgentRunEvent::CallMade { request, call });
    d.send(AgentRunEvent::LlmEnded { request });
    d.run_tools(|_| value("9:00 is free"));
    let (_, items, _) = d.step_settled();
    assert!(items[1].created_at() >= made, "{:?} before {made:?}", items[1].created_at());
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
    d.send(AgentRunEvent::Answered { approval_answers: ["item_asked_1".to_owned()].into() });
    assert!(repeat_runs(&mut d, "billing"));
}

#[test]
fn a_newer_question_does_not_revoke_an_explicitly_matched_readback_or_grant_it_twice() {
    let mut input = coordinated_input();
    input.history.push(asked("asked_1", "billing"));
    input.opening = Opening::Coordinated { approval_answers: ["item_asked_1".to_owned()].into() };
    let mut d = Driver::with_input(text_agent(vec![approval_tool()]), input);
    d.send(AgentRunEvent::Recorded { item: Box::new(asked("asked_2", "sales")) });
    assert!(repeat_runs(&mut d, "billing"));
    let executed = d.executes();
    assert_eq!(executed.len(), 1);
    assert_eq!(
        executed[0].2.extra[crate::approval::APPROVAL_SOURCE_KEY].as_str(),
        Some("item_asked_1"),
        "the grant names the matched older occurrence, not the newest question"
    );
    d.run_tools(|_| value("transferred"));
    let (step, _, _) = d.step_settled();
    d.send(AgentRunEvent::ToolResultsHandled {
        step,
        continuation: AgentRunContinuation::Continue { tools_forbidden: false },
        reply_required: Default::default(),
    });
    let request = d.plans().last().unwrap().request;
    d.respond(request, "", &[("transfer", json!({"to":"billing"}))]);
    let (call, arguments) = d.prepares().last().unwrap().clone();
    d.prepared(call, Value::Object(arguments));
    assert_eq!(d.executes().len(), 1, "the same admitted occurrence grants only one repeat");
    let (_, items, _) = d.step_settled();
    assert!(items.iter().any(|item| matches!(item,
        ChatItem::FunctionCall(call)
            if crate::approval::mark(call) == Some(lk_agents_abi::llm::tool_executor::Approval::Asked)
                && !call.extra.contains_key(crate::approval::APPROVAL_SOURCE_KEY)
    )));
}

// Catches: an answer the person moved on from still granting a later call of a step that had
// started calling before they spoke again.
#[test]
fn speaking_again_revokes_the_answers_a_calling_step_read() {
    let mut input = coordinated_input();
    input.history.push(asked("asked_1", "billing"));
    input.opening = Opening::Coordinated { approval_answers: ["item_asked_1".to_owned()].into() };
    let mut d =
        Driver::with_input(text_agent(vec![approval_tool(), string_tool("lookup", "q")]), input);
    let request = d.plans()[0].request;
    d.send(AgentRunEvent::LlmChunk {
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
    d.send(AgentRunEvent::Answered { approval_answers: Default::default() });
    d.send(AgentRunEvent::LlmChunk {
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
