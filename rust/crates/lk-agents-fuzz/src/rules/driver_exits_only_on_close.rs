//! The driver loop ends only because a close was begun.
//!
//! Usually that is the host's `Close`, but the session closes itself too: past
//! `max_unrecoverable_errors` consecutive unrecoverable errors on one leg it begins closing with
//! `CloseReason::Error`. Either way the driver journals `CloseStep::Requested` before it winds
//! down, so that step is what this rule looks for. A loop that ends with neither is a loop that
//! died, and so is one that panicked, close or not.
use lk_agents_voice::trace::{CloseStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    record::Recording,
};

pub struct DriverExitsOnlyOnClose;

impl Rule for DriverExitsOnlyOnClose {
    const CONTRACT: Contract = Contract {
        id: "driver_exits_only_on_close",
        guarantee: "The driver task finishes only after a close was begun, by the host or by the session's own error policy, and never by panicking.",
        examples: &[exited_after_close, exited_after_self_close],
        counterexamples: &[exited_unasked, panicked_after_close],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // A finished driver task with no close behind it is a loop that died on its own.
        check.expect(
            Subject::Session,
            !r.driver_finished() || r.close_begun().is_some(),
            "driver exited with no close begun",
        );
        if let Some(panic) = r.driver_panic() {
            check.expect(Subject::Session, false, format!("driver panicked: {panic}"));
        }
    }
}

/// The host asked to close and the driver finished.
fn exited_after_close() -> Recording {
    Authored::new().sent(Input::Close { drain: false }).driver_finished().done()
}

/// The session began closing on its own error policy, then the driver finished.
fn exited_after_self_close() -> Recording {
    use lk_agents_abi::voice::events::CloseReason;
    Authored::new()
        .journal(Step::Close(CloseStep::Requested { reason: CloseReason::Error, drain: false }))
        .driver_finished()
        .done()
}

/// The driver task is finished and no close was ever begun.
fn exited_unasked() -> Recording {
    Authored::new().driver_finished().done()
}

/// The host asked to close and the driver panicked on the way down.
fn panicked_after_close() -> Recording {
    Authored::new().sent(Input::Close { drain: false }).driver_panicked("boom").done()
}
