//! Every task the session spawned finishes, without panicking, once the session has closed.
//!
//! The session's own close drains its tasks against the cleanup bound and cancels what is still
//! running there: the drain's stragglers are named as it reports them. The run then joins every
//! other task once the session exited. A task still pending when no other work is left anywhere
//! is waiting on something only it, or nothing, can end: it would hold whatever it owns for the
//! life of the process. The runner stops waiting on it and records it as stuck, naming the line
//! that spawned it. A task that panicked died with whatever it was doing half done.
//!
//! One straggler is owed to the host, not the session: a tool call whose body refused its
//! cancellation runs on until the close's bound cancels it. Each refusal excuses one late
//! `host tool body`; `close_completes_at_its_bound` judges that the close still ended on time.
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
        examples: &[all_finished, a_refused_cancel_runs_to_the_bound],
        counterexamples: &[one_stuck, one_panicked, one_late, more_late_than_refused],
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
        let mut refused = r.cancels_refused().len();
        for (at, name) in r.late_tasks() {
            if name == HOST_TOOL_BODY && refused > 0 {
                refused -= 1;
                continue;
            }
            check.expect_with(Subject::Session, false, || {
                format!(
                    "the session's task {name:?} was still running at its close's cleanup bound, \
                     at {} ms, and was cancelled",
                    at.at_ms
                )
            });
        }
    }
}

/// The runtime task that actually owns the host future (`generation::tools::run_body`).
/// One refused cancellation permits one late body, never an unrelated task or a second body.
const HOST_TOOL_BODY: &str = "host tool body";

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

/// The close's drain cancelled a task still running at the cleanup bound.
fn one_late() -> Recording {
    Authored::new()
        .task(TaskEnd::Finished)
        .sent(Input::Close { drain: false })
        .late_task("audio reader")
        .done()
}

/// A tool body refused its cancellation: the close cancelled its call at the bound.
fn a_refused_cancel_runs_to_the_bound() -> Recording {
    Authored::new()
        .task(TaskEnd::Finished)
        .sent(Input::Close { drain: false })
        .cancel_refused("call_1")
        .late_task(HOST_TOOL_BODY)
        .done()
}

/// One refusal excuses one late tool call, not two.
fn more_late_than_refused() -> Recording {
    Authored::new()
        .sent(Input::Close { drain: false })
        .cancel_refused("call_1")
        .late_task(HOST_TOOL_BODY)
        .late_task(HOST_TOOL_BODY)
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
