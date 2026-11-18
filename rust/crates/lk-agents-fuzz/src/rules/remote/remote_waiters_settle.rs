use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, RemoteStep},
};

// Once the session exited, nothing of a delegation may still wait: each body returned or was
// cancelled, its delegation finished its local cleanup within the host teardown bound of the
// session's last step, and every response stream it opened at the peer was closed by then.

pub struct RemoteWaitersSettle;

impl Rule for RemoteWaitersSettle {
    const CONTRACT: Contract = Contract {
        id: "remote_waiters_settle",
        guarantee: "When the session has exited, every remote tool body has returned and settled, \
                    and every peer response stream it opened is closed, within the host \
                    teardown bound.",
        examples: &[all_settled],
        counterexamples: &[never_settled, stream_left_open, settled_long_after_exit],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if !r.driver_finished() {
            return;
        }
        // The session's last step, and the bound a host gives a delegation's cleanup after it.
        let exited = r.journal().last().map_or(0, |(at, _)| at.at_ms);
        let deadline = exited + lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_millis() as u64;
        let ended = |want: fn(&RemoteStep) -> Option<&str>| -> BTreeSet<&str> {
            r.remote()
                .iter()
                .filter(|(at, _)| at.at_ms <= deadline)
                .filter_map(|(_, step)| want(step))
                .collect()
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
        for (at, step) in r.remote() {
            if let RemoteStep::StreamClosed { stream } = step
                && at.at_ms <= deadline
            {
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
                        format!("the delegation of {call} had not finished its cleanup {deadline} ms in")
                    });
                }
                RemoteStep::StreamOpened { stream } => {
                    check.expect_with(Subject::Session, closed.contains(stream), || {
                        format!("peer response stream {stream} was still open {deadline} ms in")
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

fn settled_long_after_exit() -> Recording {
    started()
        .remote(RemoteStep::Returned { call: "item_a".into(), outcome: None })
        .driver_finished()
        .later(60_000)
        .remote(RemoteStep::StreamClosed { stream: 0 })
        .remote(RemoteStep::BodySettled { call: "item_a".into() })
        .done()
}
