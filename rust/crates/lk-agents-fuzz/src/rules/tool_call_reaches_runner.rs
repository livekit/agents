use lk_agents_voice::trace::{Step, ToolStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

// `hosted: false` is a call answered in place (a name the request never offered). Admission
// happens inside the call's task, so a task that never ran looks like a read-back; the difference
// is that a hosted call always ends before its step finishes, and one that started and never
// ended was dropped by its task.

pub struct ToolCallReachesRunner;

impl Rule for ToolCallReachesRunner {
    const CONTRACT: Contract = Contract {
        id: "tool_call_reaches_runner",
        guarantee: "A tool call's body runs exactly when the tool machine admitted the hosted call, and the hosted call ends before its step finishes; an unhosted call never runs.",
        examples: &[admitted_and_run, hosted_unadmitted_not_run, unhosted_and_not_run],
        counterexamples: &[admitted_not_run, run_unadmitted, unhosted_but_run, started_never_ended],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
        let admitted = internal.tool_calls_admitted();
        for (at, call_id) in &admitted {
            // The call task gets a turn between the admission and the body.
            if !inputs.driver_ran_after(*at) {
                continue;
            }
            check.expect(
                Subject::Session,
                host.tool_ran(call_id),
                format!("tool {call_id} was admitted, the runner never ran it"),
            );
        }
        for (_, call_id, hosted) in internal.tool_calls_started() {
            let ran = host.tool_ran(call_id);
            let was_admitted = admitted.iter().any(|(_, c)| *c == call_id);
            check.expect(
                Subject::Session,
                !ran || (hosted && was_admitted),
                if hosted {
                    format!("tool {call_id} ran without the tool machine admitting it")
                } else {
                    format!(
                        "journal answered tool {call_id} in place, but the runner ran a body for it"
                    )
                },
            );
        }
        let mut open: Vec<(u64, &str)> = Vec::new();
        for (_, step) in internal.steps() {
            match step {
                Step::Tool(ToolStep::CallStarted { speech, call_id, hosted: true, .. }) => {
                    open.push((speech.0, call_id));
                }
                Step::Tool(ToolStep::CallEnded { call_id, .. }) => {
                    open.retain(|(_, c)| c != call_id);
                }
                Step::Tool(ToolStep::Finished { speech, .. }) => {
                    for (_, call_id) in open.extract_if(.., |(s, _)| *s == speech.0) {
                        check.expect(
                            Subject::Session,
                            false,
                            format!("tool {call_id} started, its step finished, it never ended"),
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

fn started(call_id: &str, hosted: bool) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    Step::Tool(ToolStep::CallStarted {
        speech: SpeechId(1),
        call_id: call_id.into(),
        name: "lookup".into(),
        hosted,
    })
}

fn admitted(call_id: &str) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    Step::Tool(ToolStep::CallAdmitted { speech: SpeechId(1), call_id: call_id.into() })
}

/// Handed over, admitted, and the runner received its body.
fn admitted_and_run() -> Recording {
    Authored::new()
        .journal(started("c1", true))
        .journal(admitted("c1"))
        .tool_call("c1", 1)
        .then_act()
        .done()
}

fn ended(call_id: &str) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    use lk_agents_voice::events::ToolEndStatus;
    Step::Tool(ToolStep::CallEnded {
        speech: SpeechId(1),
        call_id: call_id.into(),
        end: Some(ToolEndStatus::Done),
    })
}

fn step_finished() -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    Step::Tool(ToolStep::Finished { speech: SpeechId(1), committed: 2 })
}

/// Handed over, never admitted (a read-back asked instead): no body, the call answered.
fn hosted_unadmitted_not_run() -> Recording {
    Authored::new()
        .journal(started("c9", true))
        .journal(ended("c9"))
        .journal(step_finished())
        .then_act()
        .done()
}

/// The driver answered an unadvertised name itself, and no body ran.
fn unhosted_and_not_run() -> Recording {
    Authored::new().journal(started("c9", false)).then_act().done()
}

/// Admitted and nothing ran.
fn admitted_not_run() -> Recording {
    Authored::new().journal(started("c9", true)).journal(admitted("c9")).then_act().done()
}

/// A body ran that the machine never admitted (an unapproved call ran).
fn run_unadmitted() -> Recording {
    Authored::new().journal(started("c1", true)).tool_call("c1", 1).then_act().done()
}

/// The driver said it answered in place, yet a body ran for that call.
fn unhosted_but_run() -> Recording {
    Authored::new().journal(started("c1", false)).tool_call("c1", 1).then_act().done()
}

/// Handed over and announced, then its task dropped it: the step finished with the call unended.
fn started_never_ended() -> Recording {
    Authored::new().journal(started("c9", true)).journal(step_finished()).then_act().done()
}
