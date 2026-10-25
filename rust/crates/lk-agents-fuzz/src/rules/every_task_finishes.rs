//! Every task the session spawned finishes once the session has closed.
//!
//! The run joins each task after the driver exits. A task still pending when no other work is
//! left anywhere is waiting on something only it, or nothing, can end: it would hold whatever it
//! owns for the life of the process. The runner stops waiting on it and records it as stuck.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref},
    record::Recording,
};

pub struct EveryTaskFinishes;

impl Rule for EveryTaskFinishes {
    const CONTRACT: Contract = Contract {
        id: "every_task_finishes",
        guarantee: "Every task the session spawned finishes once the session has closed.",
        examples: &[all_finished],
        counterexamples: &[one_stuck],
        probe: Some(a_reply),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (spawned, finished) in r.tasks() {
            // Which input was being applied is the closest the harness can name the spawner.
            let during = r
                .applied()
                .iter()
                .rposition(|applied| applied.at <= *spawned)
                .map_or("the session's start".into(), |nth| {
                    format!("input {nth} ({})", r.applied()[nth].input.name())
                });
            check.expect(
                Subject::Session,
                *finished,
                format!("a task spawned during {during}, at {} ms, never finished", spawned.at_ms),
            );
        }
    }
}

/// Every task the run joined finished.
fn all_finished() -> Recording {
    Authored::new().task(true).sent(Input::Close { drain: false }).task(true).done()
}

/// One task was still pending when nothing else could run.
fn one_stuck() -> Recording {
    Authored::new().task(true).sent(Input::Close { drain: false }).task(false).done()
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
