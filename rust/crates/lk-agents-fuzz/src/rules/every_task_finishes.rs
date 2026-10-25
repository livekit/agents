//! Every task the session spawned finishes, without panicking, once the session has closed.
//!
//! The run joins each task once the close was asked, stepping the session until it exits. A task
//! still pending when no other work is left anywhere is waiting on something only it, or nothing,
//! can end: it would hold whatever it owns for the life of the process. The runner stops waiting
//! on it and records it as stuck. A task that panicked died with whatever it was doing half done.
//! Either way the report names the line that spawned it.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref},
    record::{Recording, TaskEnd},
};

pub struct EveryTaskFinishes;

impl Rule for EveryTaskFinishes {
    const CONTRACT: Contract = Contract {
        id: "every_task_finishes",
        guarantee: "Every task the session spawned finishes, without panicking, once the session \
                    has closed.",
        examples: &[all_finished],
        counterexamples: &[one_stuck, one_panicked],
        probe: Some(a_reply),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for task in r.tasks() {
            let during = r
                .applied()
                .iter()
                .rposition(|applied| applied.at <= task.spawned)
                .map_or("the session's start".into(), |nth| {
                    format!("input {nth} ({})", r.applied()[nth].input.name())
                });
            let ended = match &task.end {
                TaskEnd::Finished => None,
                TaskEnd::Stuck => Some("never finished".to_owned()),
                TaskEnd::Panicked(message) => Some(format!("panicked: {message}")),
            };
            check.expect_with(Subject::Session, ended.is_none(), || {
                format!(
                    "a task spawned at {} during {during}, at {} ms, {}",
                    task.site,
                    task.spawned.at_ms,
                    ended.unwrap_or_default()
                )
            });
        }
    }
}

/// Every task the run joined finished.
fn all_finished() -> Recording {
    Authored::new()
        .task(TaskEnd::Finished)
        .sent(Input::Close { drain: false })
        .task(TaskEnd::Finished)
        .done()
}

/// One task was still pending when nothing else could run.
fn one_stuck() -> Recording {
    Authored::new()
        .task(TaskEnd::Finished)
        .sent(Input::Close { drain: false })
        .task(TaskEnd::Stuck)
        .done()
}

/// One task panicked on its way out.
fn one_panicked() -> Recording {
    Authored::new()
        .sent(Input::Close { drain: false })
        .task(TaskEnd::Panicked("boom".into()))
        .done()
}

/// A reply runs to its end, so its tasks are spawned and joined.
fn a_reply() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ],
        ..Case::default()
    }
}
