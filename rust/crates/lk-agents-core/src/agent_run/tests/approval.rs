//! The approval gate across invocations: a history-only continuation, as `agent.run()` sees it.

use lk_agents_abi::llm::tool_executor::Approval;

use super::*;
use lk_agents_abi::agent::ReadBack;
use crate::approval;

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

// Catches (c7 defect 1, decision 15): a history-only continuation unable to grant (the waiting
// output never replaced by the answer); approval decided on raw model JSON instead of the
// validated arguments (a defaulted parameter spelled differently must not ask again); an
// answer granting twice; the
// history losing the person's words or the model not told the call has not run.
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
        Effect::ReadBackRender { prepared, .. } => Some(*prepared),
        _ => None,
    });
    let answer = ReadBackAnswer::InputRequired { question: "Which email?".into(), verbatim: false };
    d.send(Event::ReadBackRendered { call, result: Ok(answer) });
    assert!(d.effects.contains(&Effect::ToolArgumentsRelease { prepared: prepared.unwrap() }));
    d.reply("Which email should the confirmation go to?", &[]);
    let ChatItem::FunctionCall(c) = &d.result().all_messages[1] else { panic!() };
    assert_eq!(approval::mark(c), None);
}
