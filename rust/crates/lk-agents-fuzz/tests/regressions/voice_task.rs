//! `VoiceTask.run()` from a tool: the task's agent takes the floor, its tool's validated
//! `Completed` returns to the parent's tool, and the parent is back without its hooks; a nested
//! task, a cancelled await and a close mid-task end their tasks the same way.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::fakes::completion_value;
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_fuzz::record::{Answer, Detail, Kind, Recording};
use lk_agents_voice::agent_activity::voice_task::{TaskEnd, VoiceTaskEnded};
use lk_agents_voice::trace::{HandoffStep, Step};

fn call(chat: u32) -> [Input; 3] {
    [
        Input::LlmToolCall {
            chat: Ref::new(chat),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(chat) },
        Input::Advance { ms: Millis(50) },
    ]
}

fn say(chat: u32) -> [Input; 3] {
    [
        Input::LlmText { chat: Ref::new(chat), text: AgentText("What is your email?".into()) },
        Input::LlmEnd { chat: Ref::new(chat) },
        Input::Advance { ms: Millis(50) },
    ]
}

fn answer(tool: u32, answer: ToolAnswer) -> [Input; 2] {
    [
        Input::ToolAnswered { tool: Ref::new(tool), answer, text: AgentText("done".into()) },
        Input::Advance { ms: Millis(50) },
    ]
}

fn task(tool: u32, agent: &str) -> [Input; 2] {
    [
        Input::VoiceTask { tool: Ref::new(tool), agent: agent.into(), reply: true },
        Input::Advance { ms: Millis(50) },
    ]
}

fn play(inputs: Vec<Input>) -> Recording {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
    let run = simulate(async move {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::Advance { ms: Millis(2_000) }).await;
        run
    });
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| format!("{}: {}", f.contract, f.detail)).collect();
    assert!(failed.is_empty(), "{failed:#?}");
    run.record
}

fn ends(record: &Recording) -> Vec<(TaskEnd, String)> {
    record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Handoff(HandoffStep::TaskEnded { how, restored, .. }) => {
                Some((*how, restored.clone()))
            }
            _ => None,
        })
        .collect()
}

fn task_answers(record: &Recording) -> Vec<Answer> {
    record
        .host_view()
        .api_calls()
        .iter()
        .filter(|c| matches!(c.input(), Input::VoiceTask { .. }))
        .filter_map(|c| c.returned().map(|(_, a)| a.clone()))
        .collect()
}

fn tool_call_id(record: &Recording, nth: u32) -> String {
    record.host_view().tools_run()[nth as usize].call_id().to_owned()
}

// Catches: a completion that does not reach the awaiting tool, a parent left suspended, a
// task's agent replying after it completed, or the parent's `on_enter` run on the restore.
#[test]
fn a_completed_task_returns_its_value_and_restores_the_parent() {
    let mut inputs =
        vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    inputs.push(Input::Advance { ms: Millis(20) });
    inputs.extend(call(0)); // the parent's tool 0
    inputs.extend(task(0, "email"));
    inputs.extend(call(1)); // the task's opening reply calls its tool 1
    inputs.extend(answer(1, ToolAnswer::Completed));
    inputs.extend(answer(0, ToolAnswer::Value)); // the parent's tool returns
    inputs.extend(say(2)); // and the parent replies to it
    let record = play(inputs);
    assert_eq!(ends(&record), vec![(TaskEnd::Completed, "agent".to_owned())]);
    let value = completion_value(&tool_call_id(&record, 1));
    assert_eq!(task_answers(&record), vec![Answer::VoiceTask(Ok(value))]);
    // The request after the completion is the parent's reply to its own tool, on its speech.
    let speech_of = |kind, nth| match record.resource(kind, nth).map(|r| r.detail().clone()) {
        Some(Detail::Tool { speech, .. }) | Some(Detail::Chat { speech: Some(speech), .. }) => {
            speech
        }
        other => panic!("{kind:?} {nth}: {other:?}"),
    };
    assert_eq!(speech_of(Kind::Chat, 2), speech_of(Kind::Tool, 0));
}

// Catches: a nested task restoring the root instead of the task that started it, or ending the
// outer task with the inner one.
#[test]
fn a_nested_task_hands_back_to_the_task_that_started_it() {
    let mut inputs =
        vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    inputs.push(Input::Advance { ms: Millis(20) });
    inputs.extend(call(0));
    inputs.extend(task(0, "email"));
    inputs.extend(call(1)); // the outer task's tool 1
    inputs.extend(task(1, "spell"));
    inputs.extend(call(2)); // the inner task's tool 2
    inputs.extend(answer(2, ToolAnswer::Completed));
    inputs.extend(answer(1, ToolAnswer::Completed));
    let record = play(inputs);
    assert_eq!(
        ends(&record),
        vec![(TaskEnd::Completed, "email".to_owned()), (TaskEnd::Completed, "agent".to_owned())]
    );
}

// Catches: a task outliving its awaiter: the parent's tool body ended (its await dropped) and
// the task kept the floor.
#[test]
fn a_dropped_await_ends_the_task() {
    let mut inputs =
        vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    inputs.push(Input::Advance { ms: Millis(20) });
    inputs.extend(call(0));
    inputs.extend(task(0, "email"));
    inputs.extend(say(1));
    inputs.extend(answer(0, ToolAnswer::Error));
    let record = play(inputs);
    assert_eq!(ends(&record), vec![(TaskEnd::Cancelled, "agent".to_owned())]);
    assert_eq!(task_answers(&record), vec![Answer::Abandoned]);
}

// Catches: a close mid-task leaving the awaiting tool waiting, or exiting the task's agent
// instead of the parent.
#[test]
fn the_close_ends_a_running_task() {
    let mut inputs =
        vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    inputs.push(Input::Advance { ms: Millis(20) });
    inputs.extend(call(0));
    inputs.extend(task(0, "email"));
    inputs.extend(say(1));
    let record = play(inputs);
    assert_eq!(ends(&record), vec![(TaskEnd::Closing, "agent".to_owned())]);
    assert_eq!(task_answers(&record), vec![Answer::VoiceTask(Err(VoiceTaskEnded::Closing))]);
}

// Catches: a task started outside a tool of the agent holding the floor, or by a speech the
// task already holds the floor from.
#[test]
fn a_task_from_the_suspended_parent_is_refused() {
    let mut inputs =
        vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
    inputs.push(Input::Advance { ms: Millis(20) });
    inputs.extend(call(0));
    inputs.extend(task(0, "email"));
    inputs.extend(task(0, "spell")); // the parent's tool again: it is no longer driving
    let record = play(inputs);
    assert_eq!(
        task_answers(&record),
        vec![
            Answer::VoiceTask(Err(VoiceTaskEnded::Closing)),
            Answer::VoiceTask(Err(VoiceTaskEnded::NotInTool)),
        ]
    );
}
