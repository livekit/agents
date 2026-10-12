//! A tool call the driver journals as hosted is the one it hands to the tool runner.
//!
//! `hosted` is the dispatch decision itself: `CallStarted { hosted: true }` means the driver
//! resolved the name to a runnable body and handed it over, `hosted: false` means it answered
//! the model in place ("Unknown function"), which a toolset may legitimately do for a name it
//! never advertised. So the two halves are checked in opposite directions: a hosted start must
//! reach the runner, and an unhosted one must not.
use lk_agents_voice::trace::{Step, ToolStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct ToolCallReachesRunner;

impl Rule for ToolCallReachesRunner {
    const CONTRACT: Contract = Contract {
        id: "tool_call_reaches_runner",
        guarantee: "A tool call journalled as hosted reaches the runner; one journalled unhosted does not.",
        examples: &[hosted_and_run, unhosted_and_not_run],
        counterexamples: &[hosted_not_run, unhosted_but_run],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
        for (at, call_id, hosted) in internal.tool_calls_started() {
            // The driver gets a turn between journalling the start and handing the call over.
            if !inputs.driver_ran_after(at) {
                continue;
            }
            let ran = host.tool_ran(call_id);
            if hosted {
                // The driver said it handed this call to a runner, so a body must have run.
                check.expect(
                    Subject::Session,
                    ran,
                    format!("journal hosted tool {call_id}, the runner never saw it"),
                );
            } else {
                // The driver said it answered this call itself, so no body may have run.
                check.expect(
                    Subject::Session,
                    !ran,
                    format!(
                        "journal answered tool {call_id} in place, but the runner ran a body for it"
                    ),
                );
            }
        }
    }
}

fn started(call_id: &str, hosted: bool) -> Step {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Step::Tool(ToolStep::CallStarted {
        speech: SpeechId(1),
        call_id: call_id.into(),
        name: "lookup".into(),
        hosted,
    })
}

/// The driver handed the call over and the runner received its body.
fn hosted_and_run() -> Recording {
    Authored::new().journal(started("c1", true)).tool_call("c1", 1).then_act().done()
}

/// The driver answered an unadvertised name itself, and no body ran.
fn unhosted_and_not_run() -> Recording {
    Authored::new().journal(started("c9", false)).then_act().done()
}

/// The driver said it handed the call over and nothing ran.
fn hosted_not_run() -> Recording {
    Authored::new().journal(started("c9", true)).then_act().done()
}

/// The driver said it answered in place, yet a body ran for that call.
fn unhosted_but_run() -> Recording {
    Authored::new().journal(started("c1", false)).tool_call("c1", 1).then_act().done()
}
