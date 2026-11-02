//! Lossless reporting at the shared run boundary, independently of question presentation.

use super::tests::{Driver, string_tool, text_agent};
use lk_agents_abi::ChatItem;
use lk_agents_abi::agent::run::RunOutcome;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use serde_json::json;

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

#[test]
fn retry_failure_envelope_retains_a_siblings_input_requirement_without_becoming_success() {
    let mut definition = text_agent(vec![string_tool("retry", "day"), string_tool("ask", "city")]);
    definition.retries.tools = 0;
    let mut driver = Driver::new(definition, "book", Vec::new());
    driver.reply("", &[("retry", json!({"day":"Monday"})), ("ask", json!({"city":"?"}))]);
    driver.run_tools(|name| {
        if name == "retry" {
            ToolOutcome::ModelRetry("try again".into())
        } else {
            ToolOutcome::InputRequired {
                question: "Which city?".into(),
                verbatim: true,
                tool: Some("answer_city".into()),
            }
        }
    });
    let failure = driver
        .effects
        .iter()
        .find_map(|effect| match effect {
            crate::agent_run::AgentRunEffect::Finished { result, input_requirements } => {
                Some((result.as_ref(), input_requirements))
            }
            _ => None,
        })
        .unwrap();
    assert!(
        matches!(failure.0, Err(lk_agents_abi::agent::run::RunError::ToolRetriesExceeded { tool, .. }) if tool == "retry")
    );
    assert_eq!(failure.1.len(), 1);
    assert_eq!(failure.1[0].question, "Which city?");
    assert_eq!(failure.1[0].call.name, "ask");
    assert_eq!(failure.1[0].call.answer_tool.as_deref(), Some("answer_city"));
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
    use super::AgentRunEvent;
    use super::tests::chunk;
    use lk_agents_abi::FunctionToolCall;

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
        driver.prepared(call, serde_json::Value::Object(arguments));
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

#[test]
fn a_single_legacy_read_back_still_accepts_direct_and_transport_answers() {
    for transport_answered in [false, true] {
        let mut first = Driver::new(text_agent(approval_tools()), "consider it", Vec::new());
        first.reply("", &[("delete_first", json!({"target":"one"}))]);
        first.run_tools(|_| panic!("an unapproved body must not run"));
        let mut history = first.result().all_messages.clone();
        for item in &mut history {
            match item {
                ChatItem::FunctionCall(call) => {
                    call.extra.remove(crate::request::INPUT_REQUESTED_KEY);
                }
                ChatItem::FunctionCallOutput(output) if transport_answered => {
                    output.output = "yes".into();
                }
                _ => {}
            }
        }
        let repeated = repeat_approval(history, "delete_first", "one");
        assert_eq!(repeated.executes().len(), 1, "legacy single-call correlation is unambiguous");
    }
}

fn a_reported_question_with_a_pending_sibling(
    streaming: bool,
) -> (Driver, super::RequestId, super::CallId, String) {
    use super::{AgentRunEvent, tests::chunk};
    use lk_agents_abi::FunctionToolCall;

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

fn assert_reported_question(effects: &[super::AgentRunEffect], call_id: &str) {
    let (result, requirements) = effects
        .iter()
        .find_map(|effect| match effect {
            super::AgentRunEffect::Finished { result, input_requirements } => {
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
    use super::AgentRunEvent;

    let (mut driver, _, sibling, call_id) = a_reported_question_with_a_pending_sibling(false);
    driver.send(AgentRunEvent::Cancel);
    driver.send(AgentRunEvent::Cancel); // Repeated cancellation must not duplicate the metadata.
    let effects =
        driver.send(AgentRunEvent::ToolSettled { call: sibling, outcome: ToolOutcome::Cancelled });
    assert_reported_question(&effects, &call_id);
}

#[test]
fn model_failure_retains_a_reported_question_before_the_stream_ends() {
    use super::AgentRunEvent;

    let (mut driver, request, sibling, call_id) = a_reported_question_with_a_pending_sibling(true);
    driver.send(AgentRunEvent::LlmFailed { request, error: "provider failed".into() });
    let effects =
        driver.send(AgentRunEvent::ToolSettled { call: sibling, outcome: ToolOutcome::Cancelled });
    assert_reported_question(&effects, &call_id);
}
