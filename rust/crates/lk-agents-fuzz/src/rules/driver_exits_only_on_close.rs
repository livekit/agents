use lk_agents_voice::trace::{CloseStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    fakes::hazard::INJECTED,
    record::Recording,
};

// The session also closes itself past `max_unrecoverable_errors` on one leg. Either way the driver
// journals `CloseStep::Requested` first, so that step is the evidence of a close. A panic the
// fakes injected (`fakes::hazard`) is the fault's, not the driver's: `every_wait_resolves` judges
// what it leaves behind.

pub struct DriverExitsOnlyOnClose;

impl Rule for DriverExitsOnlyOnClose {
    const CONTRACT: Contract = Contract {
        id: "driver_exits_only_on_close",
        guarantee: "The driver task finishes only after a close was begun, by the host or by the session's own error policy, and never by panicking.",
        examples: &[exited_after_close, exited_after_self_close, injected_panic],
        counterexamples: &[exited_unasked, panicked_after_close],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let injected = r.driver_panic().is_some_and(|p| p.starts_with(INJECTED));
        // A finished driver task with no close behind it is a loop that died on its own.
        check.expect(
            Subject::Session,
            injected || !r.driver_finished() || r.close_begun().is_some(),
            "driver exited with no close begun",
        );
        if let Some(panic) = r.driver_panic()
            && !injected
        {
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

/// A provider's injected panic unwound out of the driver.
fn injected_panic() -> Recording {
    Authored::new().driver_panicked(&format!("{INJECTED}: ChatNext")).done()
}

/// The host asked to close and the driver panicked on the way down.
fn panicked_after_close() -> Recording {
    Authored::new().sent(Input::Close { drain: false }).driver_panicked("boom").done()
}
