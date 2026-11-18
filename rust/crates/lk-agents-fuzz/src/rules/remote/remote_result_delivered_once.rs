use lk_agents_voice::{
    events::{SessionEvent, ToolLifecycle},
    trace::Step,
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// A task's result belongs to the call that started it: one body returns it, and the session
// resolves that call once. A later call that forwarded a reply returns only its handoff.

pub struct RemoteResultDeliveredOnce;

impl Rule for RemoteResultDeliveredOnce {
    const CONTRACT: Contract = Contract {
        id: "remote_result_delivered_once",
        guarantee: "A remote task's result is returned by at most one tool body and resolves its \
                    call at most once.",
        examples: &[delivered_once],
        counterexamples: &[returned_twice, resolved_twice],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (_, step) in r.remote() {
            let RemoteStep::Finished { task, result } = step else { continue };
            let returned: Vec<_> = r
                .remote()
                .iter()
                .filter_map(|(_, step)| match step {
                    RemoteStep::Returned { call, outcome: Some(outcome) }
                        if outcome.contains(result.as_str()) =>
                    {
                        Some(call.as_str())
                    }
                    _ => None,
                })
                .collect();
            check.expect_with(Subject::Session, returned.len() <= 1, || {
                format!("{task}'s result was returned by {returned:?}")
            });
            let resolved = r
                .journal()
                .iter()
                .filter(|(_, step)| {
                    matches!(step, Step::Emitted(SessionEvent::ToolExecutionUpdated(
                        ToolLifecycle::Ended { message: Some(message), .. }))
                        if message.contains(result.as_str()))
                })
                .count();
            check.expect_with(Subject::Session, resolved <= 1, || {
                format!("{task}'s result resolved a call {resolved} times")
            });
        }
    }
}

fn finished() -> RemoteStep {
    RemoteStep::Finished { task: "task-1".into(), result: "task-1 is booked".into() }
}

fn returned(call: &str) -> RemoteStep {
    RemoteStep::Returned { call: call.into(), outcome: Some("Value(\"task-1 is booked\")".into()) }
}

fn resolved() -> Step {
    Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
        id: "item_a_final".into(),
        call_id: "a".into(),
        message: Some("task-1 is booked".into()),
        status: lk_agents_voice::events::ToolEndStatus::Done,
    }))
}

fn delivered_once() -> Recording {
    Authored::new().remote(finished()).remote(returned("item_a")).journal(resolved()).done()
}

fn returned_twice() -> Recording {
    Authored::new().remote(finished()).remote(returned("item_a")).remote(returned("item_b")).done()
}

fn resolved_twice() -> Recording {
    Authored::new().remote(finished()).journal(resolved()).journal(resolved()).done()
}
