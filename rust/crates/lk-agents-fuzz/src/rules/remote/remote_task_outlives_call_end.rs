use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// Ending a call, cancelling a tool body or closing the session stops the local wait only;
// cancelling remote business work is an explicit control, which no voice input issues.

pub struct RemoteTaskOutlivesCallEnd;

impl Rule for RemoteTaskOutlivesCallEnd {
    const CONTRACT: Contract = Contract {
        id: "remote_task_outlives_call_end",
        guarantee: "No remote task is cancelled unless someone explicitly asked: a cancelled \
                    tool body or an ended call leaves it running.",
        examples: &[left_running],
        counterexamples: &[cancelled_at_close],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let cancelled: BTreeSet<&str> = r
            .remote()
            .iter()
            .filter_map(|(_, step)| match step {
                RemoteStep::CancelReceived { task } => Some(task.as_str()),
                _ => None,
            })
            .collect();
        let mut tasks = BTreeSet::new();
        for (_, step) in r.remote() {
            if let RemoteStep::Received { task, .. } = step
                && tasks.insert(task.as_str())
            {
                check.expect_with(Subject::Session, !cancelled.contains(task.as_str()), || {
                    format!("{task} was cancelled at the peer, though nobody asked")
                });
            }
        }
    }
}

fn started() -> RemoteStep {
    RemoteStep::Received {
        task: "task-1".into(),
        message: "m1".into(),
        answering: None,
        text: "Book a table".into(),
    }
}

fn left_running() -> Recording {
    Authored::new()
        .remote(started())
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .done()
}

fn cancelled_at_close() -> Recording {
    Authored::new()
        .remote(started())
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .remote(RemoteStep::CancelReceived { task: "task-1".into() })
        .done()
}
