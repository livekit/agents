use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Phase, Recording},
};

// Once the driver finished and the run closed, nothing waits on a runtime signal any more: a
// waiter still parked is a wait whose producer will never set it (`lk_agents_net::signal::park`).

pub struct NoWaiterAtQuiescence;

impl Rule for NoWaiterAtQuiescence {
    const CONTRACT: Contract = Contract {
        id: "no_waiter_at_quiescence",
        guarantee: "Once the driver finished and the run closed and went quiet, no waiter is \
                    parked on a runtime signal.",
        examples: &[none_parked],
        counterexamples: &[one_parked],
        probe: Some(super::every_wait_resolves::a_probed_say),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if !r.driver_finished() || r.phase() < Phase::Closed || r.livelock().is_some() {
            return;
        }
        check.expect_with(Subject::Session, r.parked().is_empty(), || {
            format!("{} waiters still parked: {}", r.parked().len(), r.parked().join("; "))
        });
    }
}

fn none_parked() -> Recording {
    Authored::new().driver_finished().done()
}

fn one_parked() -> Recording {
    let mut r = Authored::new().driver_finished().done();
    r.set_parked(vec!["voice/session/mod.rs:10 (signal made at voice/session/mod.rs:5)".into()]);
    r
}
