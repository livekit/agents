//! `VoiceTask.run()` returns the value one of the task's tools completed it with, as validated.
//!
//! The host's task runner validates a `Completed` against the agent's `output_type` and hands
//! over the value; the core never makes one up. So every value `run()` returned is the one a
//! completed call reported, a call answered `Completed` before the value came back, each such
//! value is returned at most once, and a value comes back exactly when a task ended completed:
//! a task ended any other way returns none.
use lk_agents_voice::agent_activity::voice_task::TaskEnd;
use lk_agents_voice::trace::{HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::completion_value,
    input::{AgentText, Input, Ref, ToolAnswer},
    record::{Answer, Recording},
};

pub struct VoiceTaskReturnsItsValue;

impl Rule for VoiceTaskReturnsItsValue {
    const CONTRACT: Contract = Contract {
        id: "voice_task_returns_its_value",
        guarantee: "VoiceTask.run() returns only a value a task tool completed with, once, and \
                    exactly as many values as tasks ended completed.",
        examples: &[returned],
        counterexamples: &[made_up, returned_twice, completed_without_value],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Each value a completed call reports, with when it was answered.
        let completions: Vec<(u64, crate::record::Observed)> = r
            .applied()
            .iter()
            .filter_map(|a| match &a.input {
                Input::ToolAnswered { tool, answer: ToolAnswer::Completed, .. } if a.accepted => {
                    let call = r.resource(crate::record::Kind::Tool, tool.nth)?;
                    match call.detail() {
                        crate::record::Detail::Tool { call_id, .. } => {
                            Some((completion_value(call_id), a.at))
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect();
        let mut returned = Vec::new();
        let mut answered = 0;
        for call in r.host_view().api_calls() {
            if !matches!(call.input(), Input::VoiceTask { .. }) {
                continue;
            }
            match call.returned() {
                Some((at, Answer::VoiceTask(Ok(value)))) => {
                    let reported = completions.iter().any(|(v, when)| v == value && *when < at);
                    check.expect(
                        Subject::Session,
                        reported,
                        format!(
                            "run() returned {value} at seq {}, which no tool completed",
                            at.seq
                        ),
                    );
                    check.expect(
                        Subject::Session,
                        !returned.contains(value),
                        format!("run() returned {value} a second time at seq {}", at.seq),
                    );
                    returned.push(*value);
                }
                Some((_, Answer::VoiceTask(Err(_)) | Answer::Abandoned)) => {}
                Some(_) | None => continue,
            }
            answered += 1;
        }
        // Counted only once every task's call was answered.
        let tasks = r.internal_view().steps().iter().filter(|(_, s)| {
            matches!(
                s,
                Step::Handoff(HandoffStep::TaskEntered { .. } | HandoffStep::TaskRefused { .. })
            )
        });
        if tasks.count() != answered {
            return;
        }
        let completed = r
            .internal_view()
            .steps()
            .iter()
            .filter(|(_, s)| {
                matches!(s, Step::Handoff(HandoffStep::TaskEnded { how: TaskEnd::Completed, .. }))
            })
            .count();
        let abandoned = r
            .host_view()
            .api_calls()
            .iter()
            .filter(|c| matches!(c.returned(), Some((_, Answer::Abandoned))))
            .count();
        // An abandoned await may have missed its value: at most that many fewer returned.
        check.expect(
            Subject::Session,
            returned.len() <= completed && completed <= returned.len() + abandoned,
            format!("{completed} tasks ended completed, {} values returned", returned.len()),
        );
    }
}

fn task_input() -> Input {
    Input::VoiceTask { tool: Ref::new(0), agent: "email".into(), reply: true }
}

fn complete() -> Input {
    Input::ToolAnswered {
        tool: Ref::new(1),
        answer: ToolAnswer::Completed,
        text: AgentText("theo@example.com".into()),
    }
}

fn entered() -> Step {
    Step::Handoff(HandoffStep::TaskEntered {
        task: lk_agents_voice::agent_activity::voice_task::VoiceTaskId(1),
        parent: "desk".into(),
        agent: "email".into(),
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId(1),
    })
}

fn ended(how: TaskEnd) -> Step {
    Step::Handoff(HandoffStep::TaskEnded {
        task: lk_agents_voice::agent_activity::voice_task::VoiceTaskId(1),
        how,
        restored: "desk".into(),
    })
}

/// The task's call completed, the task ended completed, `run()` returned that call's value.
fn returned() -> Recording {
    Authored::new()
        .tool_call("call_parent", 1)
        .api_call(task_input())
        .journal(entered())
        .tool_call("call_task", 2)
        .sent(complete())
        .journal(ended(TaskEnd::Completed))
        .result(0, Answer::VoiceTask(Ok(completion_value("call_task"))))
        .done()
}

fn made_up() -> Recording {
    Authored::new()
        .tool_call("call_parent", 1)
        .api_call(task_input())
        .journal(entered())
        .tool_call("call_task", 2)
        .sent(complete())
        .journal(ended(TaskEnd::Completed))
        .result(0, Answer::VoiceTask(Ok(7)))
        .done()
}

fn returned_twice() -> Recording {
    Authored::new()
        .tool_call("call_parent", 1)
        .api_call(task_input())
        .journal(entered())
        .api_call(task_input())
        .journal(entered())
        .tool_call("call_task", 2)
        .sent(complete())
        .journal(ended(TaskEnd::Completed))
        .result(0, Answer::VoiceTask(Ok(completion_value("call_task"))))
        .result(1, Answer::VoiceTask(Ok(completion_value("call_task"))))
        .done()
}

/// The task ended completed, but `run()` got no value.
fn completed_without_value() -> Recording {
    use lk_agents_voice::agent_activity::voice_task::VoiceTaskEnded;
    Authored::new()
        .tool_call("call_parent", 1)
        .api_call(task_input())
        .journal(entered())
        .tool_call("call_task", 2)
        .sent(complete())
        .journal(ended(TaskEnd::Completed))
        .result(0, Answer::VoiceTask(Err(VoiceTaskEnded::Cancelled)))
        .done()
}
