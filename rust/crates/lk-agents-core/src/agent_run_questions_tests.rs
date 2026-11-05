//! Questions and approval across invocations: what the run asks, what it retains when several
//! calls wait on the person, and which answer grants an approved repeat.

use super::*;
use lk_agents_abi::llm::tool_executor::Approval;

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

// Catches: a history-only continuation unable to grant (the waiting output never replaced by the
// answer); approval decided on raw model JSON instead of the validated arguments (a defaulted
// parameter spelled differently must not ask again); an answer granting twice; the history losing
// the person's words or the model not told the call has not run.
#[test]
fn approval_binds_the_validated_arguments_and_each_answer_is_consumed_once() {
    let validated = json!({"date": "Monday", "notify": true});
    let first =
        turn(vec![], "cancel monday", json!({"date": "Monday", "notify": true}), validated.clone());
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
    assert_eq!(arguments["notify"], json!(true), "the schema default replaces the null sentinel");
    second.prepared(call, validated.clone());
    let (_, _, granted) = second.executes().pop().expect("the approved repeat runs");
    assert_eq!(approval::mark(&granted), Some(Approval::Granted));
    second.run_tools(|_| value("cancelled"));
    second.reply("Done, it is cancelled.", &[]);
    let history = second.result().all_messages.clone();
    let stored: Vec<_> = transcript(&history).into_iter().filter(|(k, _)| k == "output").collect();
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
    let mut d =
        Driver::new(text_agent(vec![cancel_tool()]), "yes", first.result().all_messages.clone());
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
        AgentRunEffect::ReadBackRender { prepared, .. } => Some(*prepared),
        _ => None,
    });
    let answer = ReadBackAnswer::InputRequired { question: "Which email?".into(), verbatim: false };
    d.send(AgentRunEvent::ReadBackRendered { call, result: Ok(answer) });
    assert!(
        d.effects.contains(&AgentRunEffect::ToolArgumentsRelease { prepared: prepared.unwrap() })
    );
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
    second.send(AgentRunEvent::OutputValidated {
        validation: second.validation(),
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

#[test]
fn standalone_result_retains_all_correlated_requests_after_core_is_dropped() {
    let mut driver = Driver::new(
        text_agent(vec![string_tool("flight", "day"), string_tool("hotel", "city")]),
        "book both",
        Vec::new(),
    );
    driver.reply("", &[("flight", json!({"day":"?"})), ("hotel", json!({"city":"?"}))]);
    driver.run_tools(|name| ToolOutcome::InputRequired {
        question: format!("{name}?"),
        verbatim: true,
        tool: Some(format!("answer_{name}")),
    });
    let result = driver.result().clone();
    drop(driver);
    assert_eq!(result.input_requirements.len(), 2);
    for (requirement, name) in result.input_requirements.iter().zip(["flight", "hotel"]) {
        assert_eq!(requirement.call.name, name);
        assert_eq!(requirement.question, format!("{name}?"));
        assert_eq!(requirement.call.answer_tool, Some(format!("answer_{name}")));
        assert!(requirement.verbatim);
        let output = result
            .all_messages
            .iter()
            .find_map(|item| match item {
                ChatItem::FunctionCallOutput(output)
                    if output.call_id == requirement.call.call_id =>
                {
                    Some(output)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(output.output, format!("waiting for the person's answer to: {name}?"));
    }
    assert!(
        matches!(result.outcome, RunOutcome::InputRequired { ref question, .. } if question == "flight?")
    );
    assert_eq!(result.pending_call.as_ref(), Some(&result.input_requirements[0].call));
}

#[test]
fn input_metadata_survives_a_different_terminal_outcome_in_the_same_batch() {
    let mut driver = Driver::new(
        text_agent(vec![string_tool("stop", "reason"), string_tool("hotel", "city")]),
        "do both",
        Vec::new(),
    );
    driver.reply("", &[("stop", json!({"reason":"done"})), ("hotel", json!({"city":"?"}))]);
    driver.run_tools(|name| {
        if name == "stop" {
            ToolOutcome::StopResponse
        } else {
            ToolOutcome::InputRequired {
                question: "Which city?".into(),
                verbatim: false,
                tool: None,
            }
        }
    });
    let result = driver.result();
    assert!(matches!(result.outcome, RunOutcome::NoOutput));
    assert!(result.pending_call.is_none());
    assert_eq!(result.input_requirements.len(), 1);
    assert_eq!(result.input_requirements[0].question, "Which city?");
    assert_eq!(result.input_requirements[0].call.name, "hotel");
}

#[test]
fn history_only_continuation_answers_the_presented_call_without_answering_other_requests() {
    let tools = vec![string_tool("flight", "day"), string_tool("hotel", "city")];
    let mut first = Driver::new(text_agent(tools.clone()), "book both", Vec::new());
    first.reply("", &[("flight", json!({"day":"?"})), ("hotel", json!({"city":"?"}))]);
    first.run_tools(|name| ToolOutcome::InputRequired {
        question: format!("{name}?"),
        verbatim: true,
        tool: None,
    });
    let second = Driver::new(text_agent(tools), "Monday", first.result().all_messages.clone());
    let (_, history, _) = second.request();
    let outputs: Vec<_> = history
        .items
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) => {
                Some((output.name.as_str(), output.output.as_str()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs,
        [("flight", "Monday"), ("hotel", "waiting for the person's answer to: hotel?"),]
    );
    let mut saved = history.items;
    assert!(
        crate::request::answer_pending_input(&mut saved, "unrelated new message").is_empty(),
        "an unpresented retained question is not implicitly selected on the next turn"
    );
}

fn approval_tools() -> Vec<lk_agents_abi::agent::FunctionToolInfo> {
    ["delete_first", "delete_second"]
        .into_iter()
        .map(|name| {
            let mut tool = string_tool(name, "target");
            tool.needs_approval = Some(lk_agents_abi::agent::ReadBack::Default);
            tool
        })
        .collect()
}

fn repeat_approval(history: Vec<ChatItem>, name: &str, target: &str) -> Driver {
    let mut driver = Driver::new(text_agent(approval_tools()), "yes", history);
    let (request, _, _) = driver.request();
    driver.send(AgentRunEvent::LlmChunk {
        request,
        chunk: chunk(
            None,
            vec![FunctionToolCall {
                name: name.into(),
                arguments: json!({"target": target}).to_string(),
                call_id: "fresh_repeat".into(),
                extra: None,
            }],
            None,
        ),
    });
    driver.send(AgentRunEvent::LlmEnded { request });
    for (call, arguments) in driver.prepares() {
        driver.prepared(call, Value::Object(arguments));
    }
    driver
}

#[test]
fn answering_one_approval_never_grants_a_retained_siblings_repeat() {
    for transport_answered in [false, true] {
        let mut first = Driver::new(text_agent(approval_tools()), "consider both", Vec::new());
        first.reply(
            "",
            &[
                ("delete_first", json!({"target":"one"})),
                ("delete_second", json!({"target":"two"})),
            ],
        );
        first.run_tools(|_| panic!("an unapproved body must not run"));
        let result = first.result();
        assert_eq!(result.input_requirements.len(), 2);
        let selected = &result.input_requirements[0].call.call_id;
        let mut history = result.all_messages.clone();
        if transport_answered {
            for item in &mut history {
                if let ChatItem::FunctionCallOutput(output) = item
                    && &output.call_id == selected
                {
                    output.output = "yes".into();
                }
            }
        }
        let repeated = repeat_approval(history.clone(), "delete_first", "one");
        assert_eq!(repeated.executes().len(), 1, "the selected read-back was answered");
        let repeated = repeat_approval(history, "delete_second", "two");
        assert!(repeated.executes().is_empty(), "the retained sibling was never answered");
        assert_eq!(repeated.result().input_requirements[0].call.name, "delete_second");
    }
}

#[test]
fn an_unselected_read_back_cannot_grant_even_if_a_transport_rewrites_its_output() {
    let mut first = Driver::new(text_agent(approval_tools()), "consider it", Vec::new());
    first.reply("", &[("delete_first", json!({"target":"one"}))]);
    first.run_tools(|_| panic!("an unapproved body must not run"));
    let mut history = first.result().all_messages.clone();
    for item in &mut history {
        match item {
            ChatItem::FunctionCall(call) => {
                call.extra.insert(crate::request::INPUT_REQUESTED_KEY.into(), json!(false));
            }
            ChatItem::FunctionCallOutput(output) => output.output = "unrelated answer".into(),
            _ => {}
        }
    }
    let repeated = repeat_approval(history, "delete_first", "one");
    assert!(repeated.executes().is_empty());
}

fn a_reported_question_with_a_pending_sibling(
    streaming: bool,
) -> (Driver, RequestId, CallId, String) {
    let mut driver = Driver::new(
        text_agent(vec![string_tool("hotel", "city"), string_tool("lookup", "job")]),
        "do both",
        Vec::new(),
    );
    let (request, ..) = driver.request();
    if streaming {
        let calls = ["hotel", "lookup"]
            .into_iter()
            .enumerate()
            .map(|(i, name)| FunctionToolCall {
                name: name.into(),
                arguments: "{}".into(),
                call_id: format!("call_{i}"),
                extra: None,
            })
            .collect();
        driver.send(AgentRunEvent::LlmChunk { request, chunk: chunk(None, calls, None) });
    } else {
        driver.reply("", &[("hotel", json!({"city":"?"})), ("lookup", json!({"job":"go"}))]);
    }
    let calls = driver.prepares();
    for (call, _) in &calls {
        driver.prepared(*call, json!({}));
    }
    let question_call = driver.executes()[0].2.call_id.clone();
    driver.send(AgentRunEvent::ToolSettled {
        call: calls[0].0,
        outcome: ToolOutcome::InputRequired {
            question: "Which city?".into(),
            verbatim: true,
            tool: Some("answer_hotel".into()),
        },
    });
    assert!(driver.finished().is_none(), "the sibling still owns work");
    (driver, request, calls[1].0, question_call)
}

fn assert_reported_question(effects: &[AgentRunEffect], call_id: &str) {
    let (result, requirements) = effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::Finished { result, input_requirements } => {
                Some((result, input_requirements))
            }
            _ => None,
        })
        .expect("the aborted run settles after cleanup");
    assert!(result.is_err());
    assert_eq!(requirements.len(), 1);
    let requirement = &requirements[0];
    assert_eq!(requirement.invocation_id, "run_1");
    assert_eq!(requirement.call.call_id, call_id);
    assert_eq!(requirement.call.name, "hotel");
    assert_eq!(requirement.call.answer_tool.as_deref(), Some("answer_hotel"));
    assert_eq!(requirement.question, "Which city?");
    assert!(requirement.verbatim);
}

#[test]
fn cancellation_retains_a_reported_question_before_its_sibling_settles() {
    let (mut driver, _, sibling, call_id) = a_reported_question_with_a_pending_sibling(false);
    driver.send(AgentRunEvent::Cancel);
    driver.send(AgentRunEvent::Cancel); // A repeated cancellation must not duplicate the question.
    let effects =
        driver.send(AgentRunEvent::ToolSettled { call: sibling, outcome: ToolOutcome::Cancelled });
    assert_reported_question(&effects, &call_id);
}

#[test]
fn model_failure_retains_a_reported_question_before_the_stream_ends() {
    let (mut driver, request, sibling, call_id) = a_reported_question_with_a_pending_sibling(true);
    driver.send(AgentRunEvent::LlmFailed { request, error: "provider failed".into() });
    let effects =
        driver.send(AgentRunEvent::ToolSettled { call: sibling, outcome: ToolOutcome::Cancelled });
    assert_reported_question(&effects, &call_id);
}
