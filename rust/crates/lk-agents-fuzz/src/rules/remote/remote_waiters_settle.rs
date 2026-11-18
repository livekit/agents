use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// Once the session exited, nothing of a delegation may still wait: each body returned or was
// cancelled, its delegation finished its local cleanup, and every response stream it opened at
// the peer was closed.

pub struct RemoteWaitersSettle;

impl Rule for RemoteWaitersSettle {
    const CONTRACT: Contract = Contract {
        id: "remote_waiters_settle",
        guarantee: "When the session has exited, every remote tool body has returned and settled, \
                    and every peer response stream it opened is closed.",
        examples: &[all_settled],
        counterexamples: &[never_settled, stream_left_open],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if !r.driver_finished() {
            return;
        }
        let ended = |want: fn(&RemoteStep) -> Option<&str>| -> BTreeSet<&str> {
            r.remote().iter().filter_map(|(_, step)| want(step)).collect()
        };
        let returned = ended(|s| match s {
            RemoteStep::Returned { call, .. } => Some(call),
            _ => None,
        });
        let settled = ended(|s| match s {
            RemoteStep::BodySettled { call } => Some(call),
            _ => None,
        });
        let mut closed = BTreeSet::new();
        for (_, step) in r.remote() {
            if let RemoteStep::StreamClosed { stream } = step {
                closed.insert(*stream);
            }
        }
        for (_, step) in r.remote() {
            match step {
                RemoteStep::BodyStarted { call, .. } => {
                    let call = call.as_str();
                    check.expect_with(Subject::Session, returned.contains(call), || {
                        format!("the remote body of {call} never returned")
                    });
                    check.expect_with(Subject::Session, settled.contains(call), || {
                        format!("the delegation of {call} never finished its cleanup")
                    });
                }
                RemoteStep::StreamOpened { stream } => {
                    check.expect_with(Subject::Session, closed.contains(stream), || {
                        format!("peer response stream {stream} was never closed")
                    });
                }
                _ => {}
            }
        }
    }
}

fn started() -> Authored {
    Authored::new()
        .remote(RemoteStep::BodyStarted { call: "item_a".into(), call_id: "a".into() })
        .remote(RemoteStep::StreamOpened { stream: 0 })
}

fn all_settled() -> Recording {
    started()
        .remote(RemoteStep::StreamClosed { stream: 0 })
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .remote(RemoteStep::BodySettled { call: "item_a".into() })
        .driver_finished()
        .done()
}

fn never_settled() -> Recording {
    started()
        .remote(RemoteStep::StreamClosed { stream: 0 })
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .driver_finished()
        .done()
}

fn stream_left_open() -> Recording {
    started()
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .remote(RemoteStep::BodySettled { call: "item_a".into() })
        .driver_finished()
        .done()
}
