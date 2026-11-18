use super::super::tests::*;
use super::super::*;
use lk_agents_abi::FunctionToolCall;
use lk_agents_abi::agent::FunctionToolInfo;
use lk_agents_abi::llm::tool_executor::Approval;
use serde_json::json;

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

/// A native conversation owner matched this message to the exact pending call.
fn admitted_answer(definition: AgentDefinition, first: &RunResult, words: &str) -> Driver {
    let mut next =
        input(Opening::Message(Box::new(user("owner_answer", words))), first.all_messages.clone());
    next.id_seed = first.all_messages.len() as u64;
    let answered = BTreeSet::from([first.pending_call.as_ref().unwrap().id.clone()]);
    Driver::start(AgentRun::new(definition, next).unwrap().with_input_answers(answered))
}

fn question(d: &Driver) -> String {
    match &d.result().outcome {
        RunOutcome::InputRequired(ask) if ask.verbatim => ask.question.clone(),
        other => panic!("expected a verbatim question, got {other:?}"),
    }
}

// Approval follows an owner's matched call ID and validated arguments. A model JSON spelling
// change cannot ask again, and a single matched response cannot grant multiple repeats.
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
    let mut second = admitted_answer(text_agent(vec![cancel_tool()]), first.result(), "yes");
    let (_, ctx, _) = second.request();
    let outputs: Vec<_> = ctx
        .items
        .iter()
        .filter_map(|i| match i {
            ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, [format!("waiting for the person's answer to: {}", question(&first))]);
    assert!(ctx.items.iter().any(|item| matches!(item,
        ChatItem::Message(message)
            if message.role == Role::User && message.raw_text_content().as_deref() == Some("yes")
    )));
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
    assert_eq!(
        stored,
        vec![
            ("output".into(), format!("waiting for the person's answer to: {}", question(&first))),
            ("output".into(), "cancelled".into()),
        ]
    );
    assert!(transcript(&history).contains(&("user".into(), "yes".into())));

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
    let mut second = admitted_answer(text_agent(vec![cancel_tool()]), first.result(), "yes");
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
    let mut d = admitted_answer(text_agent(vec![cancel_tool()]), first.result(), "yes");
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
    // The first call's result is reported before the second's read-back.
    d.reply("Monday is cancelled.", &[]);
    assert!(matches!(&d.result().outcome, RunOutcome::InputRequired(ask) if ask.verbatim));
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
    let mut second = admitted_answer(definition, first.result(), "yes");
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

// Recorded text alone is not response admission. A restored JSON argument identity still
// grants the same validated repeat when its owner explicitly admits the response origin.
#[test]
fn a_recorded_response_requires_owner_admission_before_granting_a_readback_repeat() {
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
    for admitted in [false, true] {
        let run = AgentRun::new(
            text_agent(vec![cancel_tool()]),
            input(Opening::Message(Box::new(user("answer", "yes"))), history.clone()),
        )
        .unwrap();
        let run = if admitted {
            run.with_input_answers(BTreeSet::from([first
                .result()
                .pending_call
                .as_ref()
                .unwrap()
                .id
                .clone()]))
        } else {
            run
        };
        let mut second = Driver::start(run);
        second.reply("", &[("cancel_appointment", json!({"date": "Monday"}))]);
        let (call, _) = second.prepares()[0].clone();
        second.prepared(call, validated.clone());
        assert_eq!(
            second.executes().len(),
            usize::from(admitted),
            "only an admitted response can grant the validated repeat"
        );
    }
}

#[test]
fn an_unbound_pending_history_cannot_authorize_approval_from_a_new_message() {
    let validated = json!({"date":"Monday", "notify":true});
    let first = turn(vec![], "cancel monday", json!({"date":"Monday"}), validated.clone());
    let mut history = first.result().all_messages.clone();
    // A copied old selection marker remains ordinary data, even when a new message says yes.
    for item in &mut history {
        if let ChatItem::FunctionCall(call) = item {
            call.extra.insert("lk.input_requested".into(), json!(true));
        }
    }
    let saved = history.clone();
    let repeated = turn(history, "yes", json!({"date":"Monday"}), validated);
    assert!(repeated.executes().is_empty(), "unbound history never supplies response authority");
    assert!(question(&repeated).starts_with("Please confirm:"));
    assert_eq!(&repeated.result().all_messages[..saved.len()], saved.as_slice());
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
    });
    let result = driver.result().clone();
    drop(driver);
    assert_eq!(result.input_requirements.len(), 2);
    for (requirement, name) in result.input_requirements.iter().zip(["flight", "hotel"]) {
        assert_eq!(requirement.call.name, name);
        assert_eq!(requirement.question, format!("{name}?"));
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
        matches!(result.outcome, RunOutcome::InputRequired(ref ask) if ask.question == "flight?")
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
            ToolOutcome::InputRequired { question: "Which city?".into(), verbatim: false }
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
fn unbound_history_keeps_every_pending_request_as_context_without_answering_one() {
    let tools = vec![string_tool("flight", "day"), string_tool("hotel", "city")];
    let mut first = Driver::new(text_agent(tools.clone()), "book both", Vec::new());
    first.reply("", &[("flight", json!({"day":"?"})), ("hotel", json!({"city":"?"}))]);
    first.run_tools(|name| ToolOutcome::InputRequired {
        question: format!("{name}?"),
        verbatim: true,
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
        [
            ("flight", "waiting for the person's answer to: flight?"),
            ("hotel", "waiting for the person's answer to: hotel?"),
        ]
    );
    assert_eq!(
        second.run.progress_history()[..first.result().all_messages.len()],
        first.result().all_messages,
        "copied history does not supply native response admission"
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

fn repeat_approval(
    history: Vec<ChatItem>,
    name: &str,
    target: &str,
    admitted: Option<BTreeSet<String>>,
) -> Driver {
    let run = AgentRun::new(
        text_agent(approval_tools()),
        input(Opening::Message(Box::new(user("answer", "yes"))), history),
    )
    .unwrap();
    let run = match admitted {
        Some(answered) => run.with_input_answers(answered),
        None => run,
    };
    let mut driver = Driver::start(run);
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
    for owner_admitted in [false, true] {
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
        let selected = &result.input_requirements[0].call.id;
        let history = result.all_messages.clone();
        let admitted = owner_admitted.then(|| BTreeSet::from([selected.clone()]));
        let repeated = repeat_approval(history.clone(), "delete_first", "one", admitted.clone());
        assert_eq!(
            repeated.executes().len(),
            usize::from(owner_admitted),
            "only the owner-matched read-back was answered"
        );
        let repeated = repeat_approval(history, "delete_second", "two", admitted);
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
        if let ChatItem::FunctionCallOutput(output) = item {
            output.output = "unrelated answer".into();
        }
    }
    let repeated = repeat_approval(history, "delete_first", "one", None);
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
        report: lk_agents_abi::llm::tool_executor::ToolReport {
            outcome: ToolOutcome::InputRequired { question: "Which city?".into(), verbatim: true },
            input_request: Some(Box::new(remote_input())),
            transition: None,
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
    assert_eq!(requirement.question, "Which city?");
    assert!(requirement.verbatim);
    assert_eq!(requirement.input_request, Some(remote_input()));
}

fn remote_input() -> lk_agents_abi::session::InputRequestRef {
    lk_agents_abi::session::InputRequestRef::RemoteTask {
        remote_id: "hotel".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "trip".into(),
            task_id: "booking".into(),
        },
        request_id: "booking/input_0".into(),
    }
}

#[test]
fn cancellation_retains_a_reported_question_before_its_sibling_settles() {
    let (mut driver, _, sibling, call_id) = a_reported_question_with_a_pending_sibling(false);
    driver.send(AgentRunEvent::Cancel);
    driver.send(AgentRunEvent::Cancel); // A repeated cancellation must not duplicate the question.
    let effects = driver.send(AgentRunEvent::ToolSettled {
        call: sibling,
        report: (ToolOutcome::Cancelled).into(),
    });
    assert_reported_question(&effects, &call_id);
}

#[test]
fn model_failure_retains_a_reported_question_before_the_stream_ends() {
    let (mut driver, request, sibling, call_id) = a_reported_question_with_a_pending_sibling(true);
    let error = lk_agents_abi::errors::ApiError::new(
        lk_agents_abi::errors::ErrorKind::Server,
        "provider failed",
    )
    .into();
    driver.send(AgentRunEvent::LlmFailed { request, error });
    let effects = driver.send(AgentRunEvent::ToolSettled {
        call: sibling,
        report: (ToolOutcome::Cancelled).into(),
    });
    assert_reported_question(&effects, &call_id);
}

#[test]
fn a_grouped_prompt_records_only_its_selected_sources_and_never_streams_unvalidated_text() {
    for indices in [json!([0, 1]), json!([1]), json!([0, 0]), json!([2])] {
        let mut driver = Driver::new(
            text_agent(vec![string_tool("first", "name"), string_tool("last", "name")]),
            "Check me in",
            Vec::new(),
        );
        driver.reply("", &[("first", json!({"name":"?"})), ("last", json!({"name":"?"}))]);
        driver.run_tools(|name| ToolOutcome::InputRequired {
            question: format!("What is your {name} name?"),
            verbatim: false,
        });
        let effects = driver.reply(
            "unvalidated text",
            &[(
                "present_input_requests",
                json!({
                    "requests":indices, "text":"Your name?"
                }),
            )],
        );
        assert!(!effects.iter().any(|effect| matches!(effect,
            AgentRunEffect::Emit(StreamEvent::TextDelta { delta, .. })
                if delta == "unvalidated text")));
        if indices == json!([0, 1]) || indices == json!([1]) {
            let result = driver.result();
            let selected: Vec<_> = result
                .input_requirements
                .iter()
                .enumerate()
                .filter_map(|(index, input)| input.presentation_message_id.as_ref().map(|_| index))
                .collect();
            assert_eq!(json!(selected), indices);
            assert!(
                result
                    .input_requirements
                    .iter()
                    .filter_map(|input| { input.presentation_message_id.as_ref() })
                    .all(|id| result.new_messages.iter().any(|item| item.id() == *id))
            );
        } else {
            assert!(driver.finished().unwrap().is_err());
            assert!(
                driver
                    .run
                    .input_requirements
                    .iter()
                    .all(|input| { input.presentation_message_id.is_none() })
            );
        }
    }
}
