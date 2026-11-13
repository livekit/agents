use lk_agents_voice::{events::SessionEvent, trace::Step};

use crate::{
    authored::Authored,
    fakes::hazard::INJECTED,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref},
    record::{Fault, Observed, Recording, Site, TaskEnd},
};

// A task still pending when nothing else can run waits on something that never comes and would
// hold what it owns for the life of the process. Each tool body that refused its cancellation
// excuses one late `host tool body`: that straggler is the host's, not the session's. A panic the
// fakes injected is the fault's; past an injected hang the close may cancel what still waits on
// it, within the cleanup bound past the stop.

pub struct EveryTaskFinishes;

impl Rule for EveryTaskFinishes {
    const CONTRACT: Contract = Contract {
        id: "every_task_finishes",
        guarantee: "Every task the session spawned finishes, without panicking, once the session \
                    has closed.",
        examples: &[all_finished, a_refused_cancel_runs_to_the_bound, injected, hung_at_the_bound],
        counterexamples: &[
            one_stuck,
            one_panicked,
            one_late,
            more_late_than_refused,
            hung_past_the_bound,
        ],
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
                TaskEnd::Panicked(message) if message.starts_with(INJECTED) => None,
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
        let stopped = r.journal().iter().find_map(|(at, step)| {
            matches!(step, Step::Emitted(SessionEvent::Close { .. })).then_some(at.at_ms)
        });
        let bound = lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_millis() as u64;
        for (at, name) in r.late_tasks() {
            if name == HOST_TOOL_BODY && refused > 0 {
                refused -= 1;
                continue;
            }
            let at_the_bound = stopped.is_some_and(|stopped| at.at_ms <= stopped + bound);
            check.expect_with(Subject::Session, r.hung() && at_the_bound, || {
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

/// A task that unwound from an injected panic.
fn injected() -> Recording {
    Authored::new().task(TaskEnd::Panicked(format!("{INJECTED}: SttNext"))).done()
}

/// A provider hung, and the close cancelled what waited on it at the cleanup bound.
fn hung_at_the_bound() -> Recording {
    hung(4_990)
}

/// The same, cancelled past the bound.
fn hung_past_the_bound() -> Recording {
    hung(6_000)
}

fn hung(late_ms: u64) -> Recording {
    let mut r = Authored::new()
        .sent(Input::Close { drain: false })
        .journal(Step::Emitted(SessionEvent::Close {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            error: None,
        }))
        .later(late_ms)
        .late_task("tts forward")
        .done();
    r.record_fault(Observed::at(0, 0), Site::TtsNext, Fault::Hang);
    r
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
